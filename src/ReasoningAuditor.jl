module ReasoningAuditorMod

"""
Reasoning Auditor v3 (Final): Integrated Logic Engine

Architecture:
1. Gated Proposer (GLU with Separation in Dimension):
   - proj_2d = ProposerGluProjection(h_normed) -> split into h_c, h_g
   - attn_out = LinearAttention(h_c)
   - gate_phys = SoftWavePDE(h_g)
   - proposal = attn_out .+ (sigmoid(gate_phys) .* Refinement(attn_out))
2. Auditor Engine (High-Resolution Logic):
   - truth_value = VQ -> Binding -> Block-Wave-PDE -> Logic heads -> Circuit
   - veto = sigmoid(15.0 * (wave_mod * truth_value))
3. Final Veto Fusion:
   - h_new = h + (proposal .* veto)
"""

using Lux, Random, NNlib, FFTW, ChainRulesCore, LinearAlgebra, CUDA
using ..Swamma: LuxLayer, RMSNorm, to_device_like, is_gpu_array
using ..LinearAttention: LinearAttentionLayer
using ..WavePDE: laplacian_batch, leapfrog_step
using ..CircuitLayerMod: AlgebraicCircuitLayer
using ..PredicateEngramMod: _init_rule_bank, vq_quantize

# ============================================================================
# Configuration
# ============================================================================

Base.@kwdef struct ReasoningAuditorConfig
    vocab_size::Int = 32000
    max_sequence_length::Int = 64
    embedding_dimension::Int = 512
    number_of_heads::Int = 8
    number_of_layers::Int = 6
    time_dimension::Int = 64

    # Ablation & Training Phase
    use_gated_proposer::Bool = true # FALSE for Chess Phase 1
    use_adapters::Bool = false          # TRUE for Language Phase 3

    # Shared Auditor Features
    code_dim::Int = 128
    codebook_size::Int = 512
    num_logic_heads::Int = 4
    role_binding_rank::Int = 8
    
    # Physics
    default_time_step::Float32 = 0.1f0
    integration_steps::Int = 8
    
    # Gating
    proposer_gate_gain::Float32 = 1.0f0  # Soft
    veto_gain::Float32 = 15.0f0          # Sharp
end

# ============================================================================
# Shared VQ-VAE & Auditor Readout
# ============================================================================

struct MaskAwareVQVAE <: LuxLayer
    config::ReasoningAuditorConfig
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::MaskAwareVQVAE)
    c = layer.config
    d, cd, cs, nh = c.embedding_dimension, c.code_dim, c.codebook_size, c.num_logic_heads
    rk = c.role_binding_rank
    he = Float32(sqrt(2.0 / (d + cd)))
    return (
        Codebook = randn(rng, Float32, cd, cs) .* 0.1f0,
        EncoderWeight = randn(rng, Float32, cd, d + 1) .* he,
        EncoderBias = zeros(Float32, cd),
        RuleReadout = randn(rng, Float32, (d ÷ nh) * (d ÷ nh) * nh, cd) .* 0.05f0,
        WaveReadout = randn(rng, Float32, 2 * d, cd) .* 0.1f0,
        BindingUReadout = randn(rng, Float32, d, rk * nh, cd) .* 0.01f0,
        BindingVReadout = randn(rng, Float32, d, rk * nh, cd) .* 0.01f0,
    )
end

function (layer::MaskAwareVQVAE)(inputs::Tuple, ps, st)
    h_flat, mask_density = inputs
    feat = vcat(h_flat, mask_density)
    query = ps.EncoderWeight * feat .+ ps.EncoderBias
    quantized, indices = vq_quantize(query, ps.Codebook)
    return (indices, ps.RuleReadout * quantized, ps.WaveReadout * quantized, 
            ps.BindingUReadout * quantized, ps.BindingVReadout * quantized, query), st
end

# ============================================================================
# Integrated Auditor Block
# ============================================================================

struct ReasoningAuditorBlock <: LuxLayer
    config::ReasoningAuditorConfig
    VQVAE::MaskAwareVQVAE
    InputNorm::RMSNorm
    
    # Gated Proposer stack
    ProposerGluProjection::Union{Lux.Dense, Nothing}
    LinAttn::Union{LinearAttentionLayer, Nothing}
    ProposerRefinement::Union{Lux.Dense, Nothing}
    ProposerWave_Lambda::Vector{Float32}
    
    # Auditor Engine stack
    AuditorWave_Lambda::Vector{Float32}
    Circuit::AlgebraicCircuitLayer
    
    OutputNorm::RMSNorm
end

function ReasoningAuditorBlock(config::ReasoningAuditorConfig)
    d = config.embedding_dimension
    m = Float32.(FFTW.fftfreq(d) .* d)
    lambda = @. 2f0 * (cos(2f0 * Float32(pi) * m / d) - 1f0)
    
    if config.use_gated_proposer
        p_glu = Lux.Dense(d => 2 * d)
        lin_attn = LinearAttentionLayer(d, config.max_sequence_length, config.number_of_heads, config.time_dimension)
        refinement = Lux.Dense(d => d)
    else
        p_glu = nothing; lin_attn = nothing; refinement = nothing
    end

    return ReasoningAuditorBlock(
        config, MaskAwareVQVAE(config), RMSNorm(d),
        p_glu, lin_attn, refinement, lambda, lambda,
        AlgebraicCircuitLayer(d; num_leaves=16, num_sums=8), RMSNorm(d)
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, block::ReasoningAuditorBlock)
    d = block.config.embedding_dimension
    return (
        VQVAE = Lux.initialparameters(rng, block.VQVAE),
        InputNorm = Lux.initialparameters(rng, block.InputNorm),
        ProposerGluProjection = block.ProposerGluProjection === nothing ? nothing : Lux.initialparameters(rng, block.ProposerGluProjection),
        LinAttn = block.LinAttn === nothing ? nothing : Lux.initialparameters(rng, block.LinAttn),
        ProposerRefinement = block.ProposerRefinement === nothing ? nothing : Lux.initialparameters(rng, block.ProposerRefinement),
        Circuit = Lux.initialparameters(rng, block.Circuit),
        OutputNorm = Lux.initialparameters(rng, block.OutputNorm),
        prop_log_wave_speed = zeros(Float32, d), prop_log_damping = fill(-2f0, d),
        aud_log_wave_speed = zeros(Float32, d), aud_log_damping = fill(-3f0, d),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, block::ReasoningAuditorBlock)
    return (
        VQVAE = Lux.initialstates(rng, block.VQVAE),
        InputNorm = Lux.initialstates(rng, block.InputNorm),
        ProposerGluProjection = block.ProposerGluProjection === nothing ? nothing : Lux.initialstates(rng, block.ProposerGluProjection),
        LinAttn = block.LinAttn === nothing ? nothing : Lux.initialstates(rng, block.LinAttn),
        ProposerRefinement = block.ProposerRefinement === nothing ? nothing : Lux.initialstates(rng, block.ProposerRefinement),
        Circuit = Lux.initialstates(rng, block.Circuit),
        OutputNorm = Lux.initialstates(rng, block.OutputNorm),
        lambda_cache = nothing,
    )
end

function (block::ReasoningAuditorBlock)(inputs::Tuple, ps, st)
    h, time_emb, mask_density = inputs
    d, seq_len, batch_size = size(h)
    N = seq_len * batch_size
    h_flat = reshape(h, d, N)
    
    normed, in_st = block.InputNorm(h, ps.InputNorm, st.InputNorm)
    normed_flat = reshape(normed, d, N)
    
    # --- 1. Auditor Path (Truth Computation) ---
    (indices, rules, wave_mods, shift_u, shift_v, query), vq_st = block.VQVAE((normed_flat, mask_density), ps.VQVAE, st.VQVAE)
    
    # Dynamic Binding
    shift_u_res = reshape(shift_u, d, block.config.role_binding_rank * block.config.num_logic_heads, N)
    shift_v_res = reshape(shift_v, d, block.config.role_binding_rank * block.config.num_logic_heads, N)
    v_t_h = dropdims(sum(shift_v_res .* reshape(normed_flat, d, 1, N), dims=1), dims=1)
    roles_h = normed_flat .+ dropdims(sum(shift_u_res .* reshape(v_t_h, 1, :, N), dims=2), dims=2)
    
    # Block-Physics (Discovery)
    dt = block.config.default_time_step
    c_aud = clamp.(NNlib.softplus.(ps.aud_log_wave_speed .+ wave_mods[1:d, :]), 0.1f0, 2.0f0)
    γ_aud = clamp.(NNlib.softplus.(ps.aud_log_damping .+ wave_mods[d+1:end, :]), 0.01f0, 1.0f0)
    
    u_pde_aud = ChainRulesCore.ignore_derivatives() do
        u, v = roles_h, zero(roles_h)
        for _ in 1:block.config.integration_steps
            u, v = leapfrog_step(u, v, c_aud.^2, exp.(-γ_aud .* dt ./ 2f0), block.AuditorWave_Lambda, dt)
        end
        u
    end
    
    # Multi-Head Logic (Conclusion)
    rd, nh = d ÷ block.config.num_logic_heads, block.config.num_logic_heads
    rule_mats = reshape(rules, rd, rd, nh, N)
    conclusions = dropdims(sum(reshape(u_pde_aud, rd, nh, N) .* permutedims(rule_mats, (2,1,3,4)), dims=1), dims=1)
    
    # Consistency Audit
    circuit_out, circ_st = block.Circuit(reshape(conclusions, d, seq_len, batch_size), ps.Circuit, st.Circuit)
    truth = reshape(circuit_out, d, N)
    wave_mod_aud = (c_aud ./ (c_aud .+ 1f0)) .* (1f0 ./ (γ_aud .+ 1f0)) 
    
    # Final Sharp Veto Gate
    veto = NNlib.sigmoid.(block.config.veto_gain .* (wave_mod_aud .* truth))
    
    # --- 2. Proposer Path (Gated Refinement GLU) ---
    local p_st, la_st, ref_st, proposal
    if block.config.use_gated_proposer
        # Separate Dimension GLU Split
        proj_2d, p_st = block.ProposerGluProjection(normed_flat, ps.ProposerGluProjection, st.ProposerGluProjection)
        h_c = reshape(proj_2d[1:d, :], d, seq_len, batch_size)
        h_g = proj_2d[d+1:end, :]
        
        # Content: LinAttn
        attn_out, la_st = block.LinAttn((h_c, time_emb), ps.LinAttn, st.LinAttn)
        attn_flat = reshape(attn_out, d, N)
        
        # Gate: Proposer Physics
        c_prop = NNlib.softplus.(ps.prop_log_wave_speed)
        γ_prop = NNlib.softplus.(ps.prop_log_damping)
        gate_phys = ChainRulesCore.ignore_derivatives() do
            u, v = h_g, zero(h_g)
            for _ in 1:block.config.integration_steps
                u, v = leapfrog_step(u, v, c_prop.^2, exp.(-γ_prop .* dt ./ 2f0), block.ProposerWave_Lambda, dt)
            end
            u
        end
        soft_gate = NNlib.sigmoid.(block.config.proposer_gate_gain .* gate_phys)
        
        # Soft Residual GLU: proposal = attn + σ(gate) * refinement(attn)
        refinement, ref_st = block.ProposerRefinement(attn_flat, ps.ProposerRefinement, st.ProposerRefinement)
        proposal = attn_flat .+ (soft_gate .* refinement)
    else
        # Phase 1: Entire proposer ablated
        proposal = normed_flat
        p_st = nothing; la_st = nothing; ref_st = nothing
    end
    
    # --- 3. Global Fusion ---
    h_new = h_flat .+ (proposal .* veto)
    out, out_norm_st = block.OutputNorm(reshape(h_new, d, seq_len, batch_size), ps.OutputNorm, st.OutputNorm)
    
    new_st = (VQVAE=vq_st, LinAttn=la_st, ProposerGluProjection=p_st, ProposerRefinement=ref_st, Circuit=circ_st, OutputNorm=out_norm_st, InputNorm=in_st)
    return out, new_st
end

# ============================================================================
# Backbone Model
# ============================================================================

struct ReasoningAuditor{TE,PE,BL,FN,OH} <: LuxLayer
    config::ReasoningAuditorConfig
    TokenEmbedding::TE
    PositionEmbedding::PE
    Blocks::BL
    FinalNorm::FN
    OutputHead::OH
end

function ReasoningAuditor(config::ReasoningAuditorConfig)
    blocks = Tuple(ReasoningAuditorBlock(config) for _ in 1:config.number_of_layers)
    return ReasoningAuditor(
        config,
        Lux.Embedding(config.vocab_size => config.embedding_dimension),
        Lux.Embedding(config.max_sequence_length => config.embedding_dimension),
        blocks, RMSNorm(config.embedding_dimension),
        Lux.Dense(config.embedding_dimension => config.vocab_size; use_bias = false)
    )
end

function (model::ReasoningAuditor)(token_ids, ps, st; mask_density::Float32 = 0.0f0)
    tokens = ndims(token_ids) == 1 ? reshape(token_ids, :, 1) : token_ids
    seq_len, batch_size = size(tokens)
    tok_emb, tok_st = model.TokenEmbedding(vec(tokens), ps.TokenEmbedding, st.TokenEmbedding)
    pos_emb, pos_st = model.PositionEmbedding(collect(1:seq_len), ps.PositionEmbedding, st.PositionEmbedding)
    hidden = reshape(tok_emb, :, seq_len, batch_size) .+ reshape(pos_emb, :, seq_len, 1)
    time_emb = zeros(Float32, model.config.time_dimension, batch_size)
    m_dens = to_device_like(hidden, fill(mask_density, 1, seq_len * batch_size))
    block_states = []
    for i in 1:model.config.number_of_layers
        b_key = Symbol("Block_$i")
        hidden, b_st = model.Blocks[i]((hidden, time_emb, m_dens), ps.Blocks[b_key], st.Blocks[b_key])
        push!(block_states, b_st)
    end
    hidden, fn_st = model.FinalNorm(hidden, ps.FinalNorm, st.FinalNorm)
    logits, oh_st = model.OutputHead(hidden, ps.OutputHead, st.OutputHead)
    new_st = (TokenEmbedding=tok_st, PositionEmbedding=pos_st, FinalNorm=fn_st, OutputHead=oh_st,
              Blocks=NamedTuple{ntuple(i->Symbol("Block_$i"), model.config.number_of_layers)}(Tuple(block_states)))
    return ndims(token_ids) == 1 ? dropdims(logits, dims=3) : logits, new_st
end

export ReasoningAuditorConfig, ReasoningAuditorBlock, ReasoningAuditor

end # module
