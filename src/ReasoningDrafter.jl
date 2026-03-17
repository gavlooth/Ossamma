module ReasoningDrafterMod

"""
Reasoning Drafter: Speculative Decoding Module for AR Verifiers

A lightweight drafter optimized for reasoning-heavy token prediction.
Each block interleaves two structured modules around a GLU mixing core:

    RuleConditionedWavePDE → GLU(LinAttn ⊙ sigmoid(WavePDE)) → AlgebraicCircuit

- **RuleConditionedWavePDE**: VQ-quantizes the reasoning "situation", retrieves
  a rule vector, and modulates WavePDE dynamics (wave speed + damping per mode).
  Rules don't directly transform hidden states — they change HOW the wave
  equation propagates information. The physics does the reasoning.
- **GLU core**: LinearAttention provides content mixing; a separate (unmodulated)
  WavePDE provides frequency-selective gating. No FFN — the GLU's multiplicative
  nonlinearity + the surrounding structured modules provide sufficient
  per-position transformation.
- **AlgebraicCircuit**: Evaluates a bank of decomposable sum-product networks
  as a differentiable consistency checker before the output head.

Design choices for drafting:
- No SWAttention (local window is redundant at speculation-length sequences)
- No SwiGLU FFN (GLU + RuleConditionedWavePDE + Circuit = 3 nonlinearities)
- Bidirectional WavePDE (OK for AR inference since drafter re-forwards full sequence)
- 2-3 blocks total (~1M params at dim=256)
"""

using Lux
using Random
using NNlib

using ..Swamma: LuxLayer, RMSNorm, to_device_like
using ..LinearAttention: LinearAttentionLayer
using ..WavePDE: WavePDELayer
using ..RuleConditionedWavePDEMod: RuleConditionedWavePDE
using ..CircuitLayerMod: AlgebraicCircuitLayer

# ============================================================================
# Configuration
# ============================================================================

Base.@kwdef struct ReasoningDrafterConfig
    vocab_size::Int = 32000
    max_sequence_length::Int = 64
    embedding_dimension::Int = 256
    number_of_heads::Int = 4
    number_of_layers::Int = 2
    time_dimension::Int = 64         # for LinearAttention compatibility

    # WavePDE (GLU gate path)
    min_frequency::Float32 = 0.1f0
    max_frequency::Float32 = 10.0f0
    default_time_step::Float32 = 0.1f0

    # RuleConditionedWavePDE (reasoning situation → modulated wave dynamics)
    rc_code_dim::Int = 64
    rc_codebook_size::Int = 512
    rc_integration_steps::Int = 8

    # AlgebraicCircuit
    circuit_num_leaves::Int = 16
    circuit_product_arity::Int = 2
    circuit_num_sums::Int = 8
    circuit_num_circuits::Int = 4
end

# ============================================================================
# Block
# ============================================================================

struct ReasoningDrafterBlock{N,RC,GP,LA,WG,CN,GN,CL,ON} <: LuxLayer
    embedding_dimension::Int

    Norm::N
    RuleWave::RC
    GluProjection::GP
    LinAttn::LA
    WaveGate::WG
    ContentNorm::CN
    GateNorm::GN
    Circuit::CL
    OutputNorm::ON
end

function ReasoningDrafterBlock(config::ReasoningDrafterConfig)
    dim = config.embedding_dimension
    return ReasoningDrafterBlock(
        dim,
        RMSNorm(dim),
        RuleConditionedWavePDE(dim;
            code_dim = config.rc_code_dim,
            codebook_size = config.rc_codebook_size,
            default_time_step = config.default_time_step,
            integration_steps = config.rc_integration_steps,
        ),
        Lux.Dense(dim => 2 * dim),
        LinearAttentionLayer(dim, config.max_sequence_length,
            config.number_of_heads, config.time_dimension),
        WavePDELayer(dim, dim, dim,
            config.min_frequency, config.max_frequency, config.default_time_step),
        RMSNorm(dim),
        RMSNorm(dim),
        AlgebraicCircuitLayer(dim;
            num_leaves = config.circuit_num_leaves,
            product_arity = config.circuit_product_arity,
            num_sums = config.circuit_num_sums,
            num_circuits = config.circuit_num_circuits,
        ),
        Lux.LayerNorm((dim,)),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, block::ReasoningDrafterBlock)
    return (
        Norm = Lux.initialparameters(rng, block.Norm),
        RuleWave = Lux.initialparameters(rng, block.RuleWave),
        GluProjection = Lux.initialparameters(rng, block.GluProjection),
        LinAttn = Lux.initialparameters(rng, block.LinAttn),
        WaveGate = Lux.initialparameters(rng, block.WaveGate),
        ContentNorm = Lux.initialparameters(rng, block.ContentNorm),
        GateNorm = Lux.initialparameters(rng, block.GateNorm),
        Circuit = Lux.initialparameters(rng, block.Circuit),
        OutputNorm = Lux.initialparameters(rng, block.OutputNorm),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, block::ReasoningDrafterBlock)
    return (
        Norm = Lux.initialstates(rng, block.Norm),
        RuleWave = Lux.initialstates(rng, block.RuleWave),
        GluProjection = Lux.initialstates(rng, block.GluProjection),
        LinAttn = Lux.initialstates(rng, block.LinAttn),
        WaveGate = Lux.initialstates(rng, block.WaveGate),
        ContentNorm = Lux.initialstates(rng, block.ContentNorm),
        GateNorm = Lux.initialstates(rng, block.GateNorm),
        Circuit = Lux.initialstates(rng, block.Circuit),
        OutputNorm = Lux.initialstates(rng, block.OutputNorm),
    )
end

"""
    (block::ReasoningDrafterBlock)((hidden, time_emb), ps, st) → (output, state)

Forward pass:
1. RMSNorm (pre-norm)
2. RuleConditionedWavePDE (VQ situation → modulated wave dynamics)
3. GLU: Dense(dim→2dim) → split → LinAttn(content) ⊙ sigmoid(WavePDE(gate))
4. Residual connection
5. AlgebraicCircuit (structural consistency check)
6. LayerNorm (output stabilization)
"""
function (block::ReasoningDrafterBlock)(inputs::Tuple, ps, st)
    hidden, time_emb = inputs
    dim = block.embedding_dimension
    residual = hidden

    # 1. Pre-norm
    normed, norm_st = block.Norm(hidden, ps.Norm, st.Norm)

    # 2. RuleConditionedWavePDE: quantize situation → retrieve rule →
    #    modulate wave speed/damping → PDE integration → gated inject
    normed, rc_st = block.RuleWave(normed, ps.RuleWave, st.RuleWave)

    # 3. GLU core: project → split → content/gate paths → multiply
    projected, glu_st = block.GluProjection(normed, ps.GluProjection, st.GluProjection)
    content_half = copy(selectdim(projected, 1, 1:dim))
    gate_half = copy(selectdim(projected, 1, (dim + 1):(2 * dim)))

    # Content path: LinearAttention (sequence mixing)
    content_out, la_st = block.LinAttn((content_half, time_emb), ps.LinAttn, st.LinAttn)
    content_out, cn_st = block.ContentNorm(content_out, ps.ContentNorm, st.ContentNorm)

    # Gate path: WavePDE (unmodulated — frequency-selective gating)
    gate_out, wg_st = block.WaveGate(gate_half, ps.WaveGate, st.WaveGate)
    gate_out, gn_st = block.GateNorm(gate_out, ps.GateNorm, st.GateNorm)

    # GLU: content ⊙ sigmoid(gate)
    glu_out = content_out .* NNlib.sigmoid.(gate_out)

    # 4. Residual
    hidden = residual .+ glu_out

    # 5. AlgebraicCircuit: SPN consistency verification
    hidden, cl_st = block.Circuit(hidden, ps.Circuit, st.Circuit)

    # 6. Output normalization
    hidden_flat = reshape(hidden, dim, :)
    hidden_flat, on_st = block.OutputNorm(hidden_flat, ps.OutputNorm, st.OutputNorm)
    hidden = reshape(hidden_flat, size(residual))

    new_st = (
        Norm = norm_st,
        RuleWave = rc_st,
        GluProjection = glu_st,
        LinAttn = la_st,
        WaveGate = wg_st,
        ContentNorm = cn_st,
        GateNorm = gn_st,
        Circuit = cl_st,
        OutputNorm = on_st,
    )

    return hidden, new_st
end

# ============================================================================
# Model
# ============================================================================

struct ReasoningDrafter{TE,PE,BL,FN,OH} <: LuxLayer
    config::ReasoningDrafterConfig
    TokenEmbedding::TE
    PositionEmbedding::PE
    Blocks::BL
    FinalNorm::FN
    OutputHead::OH
end

function ReasoningDrafter(config::ReasoningDrafterConfig)
    blocks = Tuple(ReasoningDrafterBlock(config) for _ in 1:config.number_of_layers)
    return ReasoningDrafter(
        config,
        Lux.Embedding(config.vocab_size => config.embedding_dimension),
        Lux.Embedding(config.max_sequence_length => config.embedding_dimension),
        blocks,
        RMSNorm(config.embedding_dimension),
        Lux.Dense(config.embedding_dimension => config.vocab_size; use_bias = false),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::ReasoningDrafter)
    nl = model.config.number_of_layers
    block_params = NamedTuple{ntuple(i -> Symbol("Block_$i"), nl)}(
        Tuple(Lux.initialparameters(rng, block) for block in model.Blocks)
    )
    # Fixed time embedding (zeros — LinAttn time projection learns a bias)
    time_emb = zeros(Float32, model.config.time_dimension)

    return (
        TokenEmbedding = Lux.initialparameters(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialparameters(rng, model.PositionEmbedding),
        Blocks = block_params,
        FinalNorm = Lux.initialparameters(rng, model.FinalNorm),
        OutputHead = Lux.initialparameters(rng, model.OutputHead),
        TimeEmbedding = time_emb,
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::ReasoningDrafter)
    nl = model.config.number_of_layers
    block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), nl)}(
        Tuple(Lux.initialstates(rng, block) for block in model.Blocks)
    )
    return (
        TokenEmbedding = Lux.initialstates(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialstates(rng, model.PositionEmbedding),
        Blocks = block_states,
        FinalNorm = Lux.initialstates(rng, model.FinalNorm),
        OutputHead = Lux.initialstates(rng, model.OutputHead),
    )
end

"""
    (model::ReasoningDrafter)(token_ids, ps, st) → (logits, state)

Forward pass for the reasoning drafter. Produces logits over vocab for each position.

- `token_ids`: (seq_len,) or (seq_len, batch) integer token IDs
"""
function (model::ReasoningDrafter)(token_ids::AbstractArray{<:Integer}, ps, st)
    was_unbatched = ndims(token_ids) == 1
    tokens = was_unbatched ? reshape(token_ids, :, 1) : token_ids
    seq_len, batch_size = size(tokens)

    # 1. Token + position embeddings
    tok_flat = vec(tokens)
    tok_emb_flat, tok_st = model.TokenEmbedding(tok_flat, ps.TokenEmbedding, st.TokenEmbedding)
    tok_emb = reshape(tok_emb_flat, model.config.embedding_dimension, seq_len, batch_size)

    pos_indices = collect(1:min(seq_len, model.config.max_sequence_length))
    pos_emb_raw, pos_st = model.PositionEmbedding(pos_indices, ps.PositionEmbedding, st.PositionEmbedding)
    pos_emb = reshape(pos_emb_raw, model.config.embedding_dimension, seq_len, 1)

    hidden = tok_emb .+ pos_emb

    # 2. Fixed time embedding → broadcast to (time_dim, batch)
    time_emb = repeat(reshape(ps.TimeEmbedding, :, 1), 1, batch_size)

    # 3. Blocks
    block_states = Vector{Any}(undef, model.config.number_of_layers)
    for (i, block) in enumerate(model.Blocks)
        key = Symbol("Block_$i")
        hidden, block_state = block((hidden, time_emb), ps.Blocks[key], st.Blocks[key])
        block_states[i] = block_state
    end

    # 4. Final norm
    hidden, fn_st = model.FinalNorm(hidden, ps.FinalNorm, st.FinalNorm)

    # 5. Output head → logits
    logits, oh_st = model.OutputHead(hidden, ps.OutputHead, st.OutputHead)

    if was_unbatched
        logits = dropdims(logits, dims = 3)
    end

    new_st = (
        TokenEmbedding = tok_st,
        PositionEmbedding = pos_st,
        Blocks = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.config.number_of_layers)}(
            Tuple(block_states)
        ),
        FinalNorm = fn_st,
        OutputHead = oh_st,
    )

    return logits, new_st
end

# ============================================================================
# Generation utilities
# ============================================================================

"""
    draft_reasoning_tokens(model, prompt_ids, ps, st; num_tokens=8) → (draft_ids, draft_logits)

Autoregressively generate `num_tokens` draft tokens from a prompt.
Returns the full sequence (prompt + drafts) and logits at each draft position.
"""
function draft_reasoning_tokens(
    model::ReasoningDrafter,
    prompt_ids::AbstractVector{<:Integer},
    ps, st;
    num_tokens::Int = 8,
)
    tokens = reshape(collect(Int, prompt_ids), :, 1)
    draft_logits = Vector{Vector{Float32}}()

    for _ in 1:num_tokens
        logits, st = model(tokens, ps, st)
        last_logits = logits[:, end, 1]
        push!(draft_logits, Vector{Float32}(last_logits))
        next_token = argmax(last_logits)
        tokens = vcat(tokens, fill(next_token, 1, 1))
        size(tokens, 1) >= model.config.max_sequence_length && break
    end

    return vec(tokens), draft_logits
end

"""
    estimate_drafter_parameters(config::ReasoningDrafterConfig) → Int

Estimate total trainable parameter count without instantiating the model.
"""
function estimate_drafter_parameters(config::ReasoningDrafterConfig)
    d = config.embedding_dimension
    h = config.number_of_heads
    hd = d ÷ h
    td = config.time_dimension
    cd = config.rc_code_dim
    cs = config.rc_codebook_size
    nc = config.circuit_num_circuits
    nl = config.circuit_num_leaves
    ns = config.circuit_num_sums
    np = nl ÷ config.circuit_product_arity

    # Embeddings
    embed = config.vocab_size * d + config.max_sequence_length * d
    # Output head (no bias)
    head = d * config.vocab_size

    # Per block:
    # RMSNorm: d
    norm = d
    # RuleConditionedWavePDE
    rc = (cd * cs +                 # Codebook
          cd * d + cd +             # Encoder
          cd * cs +                 # RuleBank
          d * cd +                  # SpeedModWeight
          d * cd +                  # DampingModWeight
          d + d +                   # log_wave_speed + log_damping
          d * d + d)                # GateWeight + GateBias
    # GluProjection: Dense(d → 2d)
    glu = d * 2d + 2d
    # LinearAttention: 4 Dense(d→d) + 2 feature maps + pos embeds + time proj + norm
    la = 4 * (d * d + d) + 2 * (hd * 2hd + 2hd + 2hd * hd + hd) +
         2 * (config.max_sequence_length * hd) + (td * hd + hd) + 2 * hd
    # WavePDE (unmodulated gate): 2 * d
    wave = 2d
    # RMSNorm x2: 2d
    norms = 2d
    # AlgebraicCircuit
    circuit = nl * d * nc + nl * nc +       # LeafWeights + LeafBiases
              ns * np * nc +                 # SumLogWeights
              nc +                           # ComposeLogWeights
              d * ns + d +                   # OutputWeight + bias
              d * d + d                      # GateWeight + bias
    # OutputNorm (LayerNorm): 2d
    out_norm = 2d
    # FinalNorm: d
    final_norm = d
    # TimeEmbedding: td
    time_emb = td

    per_block = norm + rc + glu + la + wave + norms + circuit + out_norm
    return embed + config.number_of_layers * per_block + final_norm + head + time_emb
end

# ============================================================================
# Exports
# ============================================================================

export ReasoningDrafterConfig, ReasoningDrafterBlock, ReasoningDrafter
export draft_reasoning_tokens, estimate_drafter_parameters

end # module
