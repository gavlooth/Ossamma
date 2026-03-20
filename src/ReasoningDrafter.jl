module ReasoningDrafterMod

"""
ReasoningDrafter: shared structural front end + proposer stack + audit tail.

Architecture:

    token/position embed
        -> SharedOpcodeFrontend
        -> ReasoningDrafterBlock x N    (gated WavePDE + LinearAttention proposer stack)
        -> ReasoningAuditTail
        -> FinalNorm
        -> OutputHead

This replaces the previous monolithic block with a clearer decomposition:
- **SharedOpcodeFrontend**: one shared VQ codebook plus 4 rule-conditioned
  Wave-PDE heads that build a constraint-aware field.
- **ReasoningDrafterBlock**: global proposer block that fuses a WavePDE branch
  with a LinearAttention branch via GLU-style gating, without local windows.
- **ReasoningAuditTail**: re-quantize after proposal formation, perform dynamic
  role binding and predicate mixing, then run the algebraic circuit and veto the
  final proposal delta.
"""

using Lux
using Random
using NNlib
using FFTW
using ChainRulesCore
using LinearAlgebra

using ..Swamma: LuxLayer, RMSNorm, SwiGLU, to_device_like, state_with_training, state_is_training
using ..LinearAttention: LinearAttentionLayer
using ..PredicateEngramMod: _init_rule_bank, vq_quantize
using ..CircuitLayerMod: AlgebraicCircuitLayer
using ..WavePDE: WavePDELayer, laplacian_batch, leapfrog_step

# ============================================================================
# Configuration
# ============================================================================

Base.@kwdef struct ReasoningDrafterConfig
    vocab_size::Int = 32000
    max_sequence_length::Int = 64
    embedding_dimension::Int = 256
    number_of_heads::Int = 4
    number_of_layers::Int = 3           # proposer blocks
    time_dimension::Int = 64

    # Shared VQ / front-end Wave-PDE
    rc_code_dim::Int = 64
    rc_codebook_size::Int = 512
    rc_integration_steps::Int = 8
    frontend_wave_heads::Int = 4
    default_time_step::Float32 = 0.1f0

    # Legacy fields kept for config compatibility.
    min_frequency::Float32 = 0.1f0
    max_frequency::Float32 = 10.0f0

    # Proposer blocks
    proposer_ffn_expansion::Float32 = 3f0 / 2f0
    frontend_header_expansion::Float32 = 3f0 / 2f0
    audit_input_header_expansion::Float32 = 2f0

    # Audit tail
    predicate_num_heads::Int = 4
    num_roles::Int = 4
    circuit_num_leaves::Int = 16
    circuit_product_arity::Int = 2
    circuit_num_sums::Int = 8
    circuit_num_circuits::Int = 4
    veto_gain::Float32 = 10.0f0

    # Adapter headers
    use_adapters::Bool = false
end

# ============================================================================
# Residual adapter headers
# ============================================================================

struct ResidualAdapterHeader{N,H,O} <: LuxLayer
    embedding_dimension::Int
    expansion_factor::Float32
    InputNorm::N
    Hidden::H
    OutputProjection::O
end

function ResidualAdapterHeader(dim::Int; expansion_factor::Float32 = 3f0 / 2f0)
    return ResidualAdapterHeader(
        dim,
        expansion_factor,
        RMSNorm(dim),
        SwiGLU(dim; expansion_factor = expansion_factor),
        Lux.Dense(dim => dim),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, header::ResidualAdapterHeader)
    dim = header.embedding_dimension
    return (
        InputNorm = Lux.initialparameters(rng, header.InputNorm),
        Hidden = Lux.initialparameters(rng, header.Hidden),
        OutputProjection = Lux.initialparameters(rng, header.OutputProjection),
        GateWeight = randn(rng, Float32, dim, dim) .* Float32(sqrt(2.0 / (2 * dim))) .* 0.1f0,
        GateBias = fill(Float32(-2.0), dim),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, header::ResidualAdapterHeader)
    return (
        InputNorm = Lux.initialstates(rng, header.InputNorm),
        Hidden = Lux.initialstates(rng, header.Hidden),
        OutputProjection = Lux.initialstates(rng, header.OutputProjection),
    )
end

function (header::ResidualAdapterHeader)(hidden, ps, st)
    dim = header.embedding_dimension
    normed, in_st = header.InputNorm(hidden, ps.InputNorm, st.InputNorm)
    hidden_delta, hid_st = header.Hidden(normed, ps.Hidden, st.Hidden)
    projected, out_st = header.OutputProjection(hidden_delta, ps.OutputProjection, st.OutputProjection)
    gate_flat = ps.GateWeight * reshape(normed, dim, :) .+ ps.GateBias
    gate = reshape(NNlib.sigmoid.(gate_flat), size(hidden))
    output = hidden .+ gate .* projected
    new_st = (
        InputNorm = in_st,
        Hidden = hid_st,
        OutputProjection = out_st,
    )
    return output, new_st
end

# ============================================================================
# Shared opcode front end
# ============================================================================

struct SharedOpcodeFrontend{N} <: LuxLayer
    embedding_dimension::Int
    code_dim::Int
    codebook_size::Int
    num_wave_heads::Int
    integration_steps::Int
    time_dimension::Int
    default_time_step::Float32
    lambda::Vector{Float32}
    head_speed_prior::Vector{Float32}
    head_damping_prior::Vector{Float32}
    InputNorm::N
end

function _frontend_head_priors(num_heads::Int)
    if num_heads == 4
        speed = Float32[1.35, 1.0, 0.7, 1.15]
        damping = Float32[0.7, 1.0, 1.45, 0.9]
    else
        speed = collect(Float32, range(1.25f0, 0.8f0; length = num_heads))
        damping = collect(Float32, range(0.8f0, 1.25f0; length = num_heads))
    end
    return speed, damping
end

function SharedOpcodeFrontend(config::ReasoningDrafterConfig)
    dim = config.embedding_dimension
    m = Float32.(FFTW.fftfreq(dim) .* dim)
    lambda = @. 2f0 * (cos(2f0 * Float32(pi) * m / dim) - 1f0)
    head_speed_prior, head_damping_prior = _frontend_head_priors(config.frontend_wave_heads)

    return SharedOpcodeFrontend(
        dim,
        config.rc_code_dim,
        config.rc_codebook_size,
        config.frontend_wave_heads,
        config.rc_integration_steps,
        config.time_dimension,
        config.default_time_step,
        lambda,
        head_speed_prior,
        head_damping_prior,
        RMSNorm(dim),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::SharedOpcodeFrontend)
    dim = layer.embedding_dimension
    cd = layer.code_dim
    cs = layer.codebook_size
    nh = layer.num_wave_heads
    he = Float32(sqrt(2.0 / (dim + cd)))

    return (
        InputNorm = Lux.initialparameters(rng, layer.InputNorm),
        Codebook = randn(rng, Float32, cd, cs) .* 0.1f0,
        EncoderWeight = randn(rng, Float32, cd, dim) .* he,
        EncoderBias = zeros(Float32, cd),
        MaskCodeWeight = randn(rng, Float32, cd, layer.time_dimension) .* he .* 0.1f0,
        MaskCodeBias = zeros(Float32, cd),
        WaveReadoutWeight = randn(rng, Float32, 2 * dim * nh, cd) .* he .* 0.1f0,
        WaveReadoutBias = zeros(Float32, 2 * dim * nh),
        MaskReadoutWeight = randn(rng, Float32, 2 * dim * nh, layer.time_dimension) .* he .* 0.1f0,
        MaskReadoutBias = zeros(Float32, 2 * dim * nh),
        log_wave_speed = zeros(Float32, dim, nh),
        log_damping = fill(-3f0, dim, nh),
        FusionWeight = randn(rng, Float32, dim, dim * nh) .* Float32(sqrt(2.0 / (dim + dim * nh))),
        FusionBias = zeros(Float32, dim),
        GateWeight = randn(rng, Float32, dim, dim) .* Float32(sqrt(2.0 / (2 * dim))),
        GateBias = fill(Float32(-2.0), dim),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::SharedOpcodeFrontend)
    return state_with_training(
        InputNorm = Lux.initialstates(rng, layer.InputNorm),
        ema_cluster_size = ones(Float32, layer.codebook_size),
        ema_embed_sum = zeros(Float32, layer.code_dim, layer.codebook_size),
        lambda_cache = nothing,
    )
end

function _ensure_frontend_lambda_cache(layer::SharedOpcodeFrontend, template, st)
    cache = get(st, :lambda_cache, nothing)
    template_is_gpu = occursin("CuArray", string(typeof(template)))
    cache_matches = cache !== nothing &&
                    occursin("CuArray", string(typeof(cache))) == template_is_gpu &&
                    length(cache) == length(layer.lambda)
    cache_matches && return cache, st

    λ = to_device_like(template, layer.lambda)
    return λ, merge(st, (lambda_cache = λ,))
end

function _shared_frontend_ema_update(layer::SharedOpcodeFrontend, query, indices, st)
    decay = 0.99f0
    query_cpu = Array(query)
    indices_cpu = Array(indices)
    cs = layer.codebook_size
    n = length(indices_cpu)

    new_counts = zeros(Float32, cs)
    for i in 1:n
        new_counts[indices_cpu[i]] += 1.0f0
    end

    new_sums = zeros(Float32, layer.code_dim, cs)
    for i in 1:n
        new_sums[:, indices_cpu[i]] .+= @view query_cpu[:, i]
    end

    ema_cs = Array(st.ema_cluster_size)
    ema_es = Array(st.ema_embed_sum)
    new_ema_cs = decay .* ema_cs .+ (1.0f0 - decay) .* new_counts
    new_ema_es = decay .* ema_es .+ (1.0f0 - decay) .* new_sums

    return (
        ema_cluster_size = typeof(st.ema_cluster_size)(new_ema_cs),
        ema_embed_sum = typeof(st.ema_embed_sum)(new_ema_es),
    )
end

function _shared_frontend_next_state(layer::SharedOpcodeFrontend, query, indices, st, norm_st)
    updated = merge(st, (InputNorm = norm_st,))
    if state_is_training(st)
        ema_updates = _shared_frontend_ema_update(layer, query, indices, st)
        updated = merge(updated, ema_updates)
    end
    return updated
end

ChainRulesCore.@non_differentiable _shared_frontend_next_state(layer, query, indices, st, norm_st)

function apply_shared_frontend_ema_codebook!(ps, st, layer::SharedOpcodeFrontend; laplace_smoothing::Float32 = 1f-5)
    ema_cluster_size = Array(st.ema_cluster_size)
    active_codes = findall(>(0f0), ema_cluster_size)
    isempty(active_codes) && return ps

    ema_embed_sum = Array(st.ema_embed_sum)
    codebook_cpu = Array(ps.Codebook)
    n = sum(ema_cluster_size)
    denom = max(n + layer.codebook_size * laplace_smoothing, 1f-8)
    smoothed = (ema_cluster_size .+ laplace_smoothing) ./ denom .* n

    for c in active_codes
        codebook_cpu[:, c] .= ema_embed_sum[:, c] ./ max(smoothed[c], 1f-8)
    end

    ps.Codebook .= to_device_like(ps.Codebook, codebook_cpu)
    return ps
end

function (layer::SharedOpcodeFrontend)(hidden_state::AbstractMatrix, ps, st)
    hidden_b = reshape(hidden_state, size(hidden_state, 1), size(hidden_state, 2), 1)
    time_emb = zeros(eltype(hidden_state), layer.time_dimension, 1)
    output_b, new_st = layer((hidden_b, time_emb), ps, st)
    return dropdims(output_b, dims=3), new_st
end

function _broadcast_frontend_condition(cond::AbstractMatrix, seq_len::Int)
    channels, batch = size(cond)
    expanded = reshape(cond, channels, 1, batch)
    return reshape(repeat(expanded, 1, seq_len, 1), channels, seq_len * batch)
end

function (layer::SharedOpcodeFrontend)(inputs::Tuple{<:AbstractArray{T,3},<:AbstractMatrix}, ps, st) where {T}
    hidden_b, time_emb = inputs
    dim, seq_len, batch_size = size(hidden_b)
    m = seq_len * batch_size

    normed, norm_st = layer.InputNorm(hidden_b, ps.InputNorm, st.InputNorm)
    hidden_flat = reshape(normed, dim, m)
    mask_code = _broadcast_frontend_condition(ps.MaskCodeWeight * time_emb .+ ps.MaskCodeBias, seq_len)
    mask_readout = _broadcast_frontend_condition(ps.MaskReadoutWeight * time_emb .+ ps.MaskReadoutBias, seq_len)

    query = ps.EncoderWeight * hidden_flat .+ ps.EncoderBias .+ mask_code
    quantized, indices = vq_quantize(query, ps.Codebook)
    readouts = ps.WaveReadoutWeight * quantized .+ ps.WaveReadoutBias .+ mask_readout
    readouts = reshape(readouts, 2 * dim, layer.num_wave_heads, m)

    λ, st = _ensure_frontend_lambda_cache(layer, hidden_flat, st)
    speed_shift = @view readouts[1:dim, :, :]
    damping_shift = @view readouts[(dim + 1):(2 * dim), :, :]
    speed_prior = reshape(to_device_like(hidden_flat, layer.head_speed_prior), 1, layer.num_wave_heads, 1)
    damping_prior = reshape(to_device_like(hidden_flat, layer.head_damping_prior), 1, layer.num_wave_heads, 1)
    base_speed = reshape(ps.log_wave_speed, dim, layer.num_wave_heads, 1)
    base_damping = reshape(ps.log_damping, dim, layer.num_wave_heads, 1)

    c = clamp.(NNlib.softplus.(base_speed .+ speed_shift) .* speed_prior, 0.1f0, 2.0f0)
    γ = clamp.(NNlib.softplus.(base_damping .+ damping_shift) .* damping_prior, 0.01f0, 1.0f0)
    c_sq = c .^ 2
    d = exp.(-γ .* layer.default_time_step ./ 2f0)

    detached_fields = ChainRulesCore.ignore_derivatives() do
        fields = similar(hidden_flat, dim, layer.num_wave_heads, m)

        for head in 1:layer.num_wave_heads
            u_pde = hidden_flat
            v_pde = zero(u_pde)
            c_sq_head = @view c_sq[:, head, :]
            d_head = @view d[:, head, :]
            for _ in 1:layer.integration_steps
                u_pde, v_pde = leapfrog_step(u_pde, v_pde, c_sq_head, d_head, λ, layer.default_time_step)
            end
            @views fields[:, head, :] .= u_pde
        end

        fields
    end

    wave_mod = (c ./ (c .+ 1f0)) .* (1f0 ./ (γ .+ 1f0))
    head_fields = wave_mod .* detached_fields
    fused_input = reshape(head_fields, dim * layer.num_wave_heads, m)
    fused = ps.FusionWeight * fused_input .+ ps.FusionBias
    gate = NNlib.sigmoid.(ps.GateWeight * hidden_flat .+ ps.GateBias)
    output_flat = reshape(hidden_b, dim, m) .+ gate .* fused

    output = reshape(output_flat, dim, seq_len, batch_size)

    new_st = _shared_frontend_next_state(layer, query, indices, st, norm_st)

    return output, new_st
end

# ============================================================================
# Proposer blocks
# ============================================================================

struct ReasoningDrafterBlock{N,GP,LA,WG,AN,GN,FF,ON} <: LuxLayer
    embedding_dimension::Int
    use_adapters::Bool
    InputNorm::N
    GluProjection::GP
    LinAttn::LA
    WaveGateLayer::WG
    AttnNorm::AN
    WaveGateNorm::GN
    FFN::FF
    OutputNorm::ON
end

function ReasoningDrafterBlock(config::ReasoningDrafterConfig)
    dim = config.embedding_dimension
    return ReasoningDrafterBlock(
        dim,
        config.use_adapters,
        RMSNorm(dim),
        Lux.Dense(dim => 2 * dim),
        LinearAttentionLayer(dim, config.max_sequence_length, config.number_of_heads, config.time_dimension),
        WavePDELayer(
            dim,
            dim,
            dim,
            config.min_frequency,
            config.max_frequency,
            config.default_time_step;
            integration_steps = config.rc_integration_steps,
        ),
        RMSNorm(dim),
        RMSNorm(dim),
        SwiGLU(dim; expansion_factor = config.proposer_ffn_expansion),
        RMSNorm(dim),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, block::ReasoningDrafterBlock)
    dim = block.embedding_dimension
    adapters = if block.use_adapters
        (
            ProposalHeaderWeight = Matrix{Float32}(LinearAlgebra.I, dim, dim),
            ProposalHeaderBias = zeros(Float32, dim),
        )
    else
        (
            ProposalHeaderWeight = nothing,
            ProposalHeaderBias = nothing,
        )
    end

    return (
        InputNorm = Lux.initialparameters(rng, block.InputNorm),
        GluProjection = Lux.initialparameters(rng, block.GluProjection),
        LinAttn = Lux.initialparameters(rng, block.LinAttn),
        WaveGateLayer = Lux.initialparameters(rng, block.WaveGateLayer),
        AttnNorm = Lux.initialparameters(rng, block.AttnNorm),
        WaveGateNorm = Lux.initialparameters(rng, block.WaveGateNorm),
        FFN = Lux.initialparameters(rng, block.FFN),
        OutputNorm = Lux.initialparameters(rng, block.OutputNorm),
        adapters...,
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, block::ReasoningDrafterBlock)
    return (
        InputNorm = Lux.initialstates(rng, block.InputNorm),
        GluProjection = Lux.initialstates(rng, block.GluProjection),
        LinAttn = Lux.initialstates(rng, block.LinAttn),
        WaveGateLayer = Lux.initialstates(rng, block.WaveGateLayer),
        AttnNorm = Lux.initialstates(rng, block.AttnNorm),
        WaveGateNorm = Lux.initialstates(rng, block.WaveGateNorm),
        FFN = Lux.initialstates(rng, block.FFN),
        OutputNorm = Lux.initialstates(rng, block.OutputNorm),
    )
end

function (block::ReasoningDrafterBlock)(inputs::Tuple, ps, st)
    hidden, time_emb = inputs
    dim = block.embedding_dimension

    normed, in_st = block.InputNorm(hidden, ps.InputNorm, st.InputNorm)
    glu_proj, gp_st = block.GluProjection(normed, ps.GluProjection, st.GluProjection)
    content_half = copy(selectdim(glu_proj, 1, 1:dim))
    gate_half = copy(selectdim(glu_proj, 1, (dim + 1):(2 * dim)))

    seq_len = size(content_half, 2)
    batch_size = size(content_half, 3)
    flat_tokens = seq_len * batch_size
    lin_attn = block.LinAttn
    lin_ps = ps.LinAttn
    lin_st = st.LinAttn
    head_dim = size(lin_ps.TimeProjection.weight, 1)
    num_heads = dim ÷ head_dim
    flat_heads = num_heads * seq_len * batch_size
    merged_heads = num_heads * batch_size

    flat_content = reshape(content_half, dim, flat_tokens)
    query_flat, _ = lin_attn.QueryProjection(flat_content, lin_ps.QueryProjection, lin_st.QueryProjection)
    key_flat, _ = lin_attn.KeyProjection(flat_content, lin_ps.KeyProjection, lin_st.KeyProjection)
    value_flat, _ = lin_attn.ValueProjection(flat_content, lin_ps.ValueProjection, lin_st.ValueProjection)

    query_heads = reshape(reshape(query_flat, dim, seq_len, batch_size), head_dim, num_heads, seq_len, batch_size)
    key_heads = reshape(reshape(key_flat, dim, seq_len, batch_size), head_dim, num_heads, seq_len, batch_size)
    value_heads = reshape(reshape(value_flat, dim, seq_len, batch_size), head_dim, num_heads, seq_len, batch_size)

    time_proj, _ = lin_attn.TimeProjection(time_emb, lin_ps.TimeProjection, lin_st.TimeProjection)
    time_proj_batch = size(time_proj, 2)
    (time_proj_batch == 1 || time_proj_batch == batch_size) || throw(ArgumentError(
        "ReasoningDrafterBlock proposer time batch mismatch: got $(time_proj_batch), expected 1 or $(batch_size)."
    ))
    time_broadcast = reshape(time_proj, head_dim, 1, 1, time_proj_batch)

    query_feature_flat, _ = lin_attn.QueryFeatureLinear(
        reshape(query_heads .+ time_broadcast, head_dim, flat_heads),
        lin_ps.QueryFeatureLinear,
        lin_st.QueryFeatureLinear,
    )
    key_feature_flat, _ = lin_attn.KeyFeatureLinear(
        reshape(key_heads .+ time_broadcast, head_dim, flat_heads),
        lin_ps.KeyFeatureLinear,
        lin_st.KeyFeatureLinear,
    )

    query_feature_tensor = reshape(query_feature_flat, head_dim, num_heads, seq_len, batch_size)
    key_feature_tensor = reshape(key_feature_flat, head_dim, num_heads, seq_len, batch_size)

    position_indices = 1:seq_len
    pos_cosine_raw, _ = lin_attn.PositionEmbeddingCosine(position_indices, lin_ps.PositionEmbeddingCosine, lin_st.PositionEmbeddingCosine)
    pos_sine_raw, _ = lin_attn.PositionEmbeddingSine(position_indices, lin_ps.PositionEmbeddingSine, lin_st.PositionEmbeddingSine)
    pos_cosine = reshape(pos_cosine_raw, head_dim, 1, seq_len, 1)
    pos_sine = reshape(pos_sine_raw, head_dim, 1, seq_len, 1)

    query_cos_unnorm = softplus.(query_feature_tensor .* pos_cosine)
    key_cos_unnorm = softplus.(key_feature_tensor .* pos_cosine)
    query_sin_unnorm = softplus.(query_feature_tensor .* pos_sine)
    key_sin_unnorm = softplus.(key_feature_tensor .* pos_sine)

    query_cos_flat, _ = lin_attn.FeatureNorm(
        reshape(query_cos_unnorm, head_dim, flat_heads),
        lin_ps.FeatureNorm,
        lin_st.FeatureNorm,
    )
    key_cos_flat, _ = lin_attn.FeatureNorm(
        reshape(key_cos_unnorm, head_dim, flat_heads),
        lin_ps.FeatureNorm,
        lin_st.FeatureNorm,
    )
    query_sin_flat, _ = lin_attn.FeatureNorm(
        reshape(query_sin_unnorm, head_dim, flat_heads),
        lin_ps.FeatureNorm,
        lin_st.FeatureNorm,
    )
    key_sin_flat, _ = lin_attn.FeatureNorm(
        reshape(key_sin_unnorm, head_dim, flat_heads),
        lin_ps.FeatureNorm,
        lin_st.FeatureNorm,
    )

    query_cos = reshape(query_cos_flat, head_dim, num_heads, seq_len, batch_size)
    key_cos = reshape(key_cos_flat, head_dim, num_heads, seq_len, batch_size)
    query_sin = reshape(query_sin_flat, head_dim, num_heads, seq_len, batch_size)
    key_sin = reshape(key_sin_flat, head_dim, num_heads, seq_len, batch_size)
    value_sequence = reshape(value_heads, head_dim, seq_len, merged_heads)

    query_cos_sequence = reshape(query_cos, head_dim, seq_len, merged_heads)
    key_cos_sequence = reshape(key_cos, head_dim, seq_len, merged_heads)
    query_sin_sequence = reshape(query_sin, head_dim, seq_len, merged_heads)
    key_sin_sequence = reshape(key_sin, head_dim, seq_len, merged_heads)

    context_cos = NNlib.batched_mul(value_sequence, permutedims(key_cos_sequence, (2, 1, 3)))
    context_sin = NNlib.batched_mul(value_sequence, permutedims(key_sin_sequence, (2, 1, 3)))
    attn_heads = NNlib.batched_mul(context_cos, query_cos_sequence) .+ NNlib.batched_mul(context_sin, query_sin_sequence)

    attn_out_flat, _ = lin_attn.OutputProjection(
        reshape(attn_heads, dim, flat_tokens),
        lin_ps.OutputProjection,
        lin_st.OutputProjection,
    )
    attn_out = reshape(attn_out_flat, dim, seq_len, batch_size)
    la_st = lin_st
    attn_out, an_st = block.AttnNorm(attn_out, ps.AttnNorm, st.AttnNorm)

    wave_out, wg_st = block.WaveGateLayer(gate_half, ps.WaveGateLayer, st.WaveGateLayer)
    wave_gate, wgn_st = block.WaveGateNorm(wave_out, ps.WaveGateNorm, st.WaveGateNorm)
    wave_speed = NNlib.softplus.(ps.WaveGateLayer.log_wave_speed)
    wave_damping = NNlib.softplus.(ps.WaveGateLayer.log_damping)
    wave_param_mod = reshape(
        (wave_speed ./ (wave_speed .+ 1f0)) .* (1f0 ./ (wave_damping .+ 1f0)),
        dim, 1, 1,
    )

    gated = attn_out .* NNlib.sigmoid.(wave_param_mod .* wave_gate)
    proposal = if block.use_adapters && ps.ProposalHeaderWeight !== nothing
        flat = reshape(gated, dim, :)
        adapted = ps.ProposalHeaderWeight * flat .+ ps.ProposalHeaderBias
        reshape(adapted, size(gated))
    else
        gated
    end

    ffn_out, ffn_st = block.FFN(proposal, ps.FFN, st.FFN)
    output = hidden .+ proposal .+ ffn_out
    output, on_st = block.OutputNorm(output, ps.OutputNorm, st.OutputNorm)

    new_st = (
        InputNorm = in_st,
        GluProjection = gp_st,
        LinAttn = la_st,
        WaveGateLayer = wg_st,
        AttnNorm = an_st,
        WaveGateNorm = wgn_st,
        FFN = ffn_st,
        OutputNorm = on_st,
    )

    return output, new_st
end

# ============================================================================
# Audit tail
# ============================================================================

function _init_multihead_rule_bank(rng::Random.AbstractRNG, num_heads::Int, num_roles::Int, codebook_size::Int)
    bank = zeros(Float32, num_heads * num_roles * num_roles, codebook_size)
    for head in 1:num_heads
        start_idx = (head - 1) * num_roles * num_roles + 1
        stop_idx = head * num_roles * num_roles
        bank[start_idx:stop_idx, :] .= _init_rule_bank(rng, num_roles, codebook_size)
    end
    return bank
end

struct ReasoningAuditTail{N,AIH,CL,C} <: LuxLayer
    embedding_dimension::Int
    code_dim::Int
    codebook_size::Int
    num_roles::Int
    predicate_num_heads::Int
    veto_gain::Float32
    use_adapters::Bool
    InputNorm::N
    AuditInputHeader::AIH
    CircuitLeafProjection::CL
    Circuit::C
end

function ReasoningAuditTail(config::ReasoningDrafterConfig)
    dim = config.embedding_dimension
    dim % config.num_roles == 0 || throw(ArgumentError(
        "embedding_dimension=$(dim) must be divisible by num_roles=$(config.num_roles)."
    ))

    return ReasoningAuditTail(
        dim,
        config.rc_code_dim,
        config.rc_codebook_size,
        config.num_roles,
        config.predicate_num_heads,
        config.veto_gain,
        config.use_adapters,
        RMSNorm(dim),
        config.use_adapters ? ResidualAdapterHeader(dim; expansion_factor = config.audit_input_header_expansion) : nothing,
        Lux.Dense(dim => dim),
        AlgebraicCircuitLayer(
            dim;
            num_leaves = config.circuit_num_leaves,
            product_arity = config.circuit_product_arity,
            num_sums = config.circuit_num_sums,
            num_circuits = config.circuit_num_circuits,
        ),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, tail::ReasoningAuditTail)
    dim = tail.embedding_dimension
    total_role_dim = tail.predicate_num_heads * dim
    he = Float32(sqrt(2.0 / (dim + tail.code_dim)))
    adapters = if tail.use_adapters
        (
            CircuitLeafHeaderWeight = Matrix{Float32}(LinearAlgebra.I, dim, dim),
            CircuitLeafHeaderBias = zeros(Float32, dim),
            CircuitGateBiasShift = fill(10.0f0, dim),
        )
    else
        (
            CircuitLeafHeaderWeight = nothing,
            CircuitLeafHeaderBias = nothing,
            CircuitGateBiasShift = nothing,
        )
    end

    return (
        InputNorm = Lux.initialparameters(rng, tail.InputNorm),
        AuditInputHeader = tail.AuditInputHeader === nothing ? nothing : Lux.initialparameters(rng, tail.AuditInputHeader),
        FineEncoderWeight = randn(rng, Float32, tail.code_dim, dim) .* he,
        FineEncoderBias = zeros(Float32, tail.code_dim),
        RoleBaseWeight = randn(rng, Float32, total_role_dim, dim) .* Float32(sqrt(2.0 / (dim + total_role_dim))),
        RoleBaseBias = zeros(Float32, total_role_dim),
        RoleShiftWeight = randn(rng, Float32, total_role_dim, tail.code_dim) .* he .* 0.1f0,
        PredicateRuleBank = _init_multihead_rule_bank(rng, tail.predicate_num_heads, tail.num_roles, tail.codebook_size),
        PredicateOutputWeight = randn(rng, Float32, dim, total_role_dim) .* Float32(sqrt(2.0 / (dim + total_role_dim))),
        PredicateOutputBias = zeros(Float32, dim),
        CircuitLeafProjection = Lux.initialparameters(rng, tail.CircuitLeafProjection),
        Circuit = Lux.initialparameters(rng, tail.Circuit),
        ScoreWeight = randn(rng, Float32, dim) .* Float32(1 / sqrt(dim)),
        AgreementWeight = randn(rng, Float32, dim) .* Float32(1 / sqrt(dim)),
        ScoreBias = Float32[0.0],
        AgreementBias = Float32[0.0],
        adapters...,
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, tail::ReasoningAuditTail)
    return (
        InputNorm = Lux.initialstates(rng, tail.InputNorm),
        AuditInputHeader = tail.AuditInputHeader === nothing ? nothing : Lux.initialstates(rng, tail.AuditInputHeader),
        CircuitLeafProjection = Lux.initialstates(rng, tail.CircuitLeafProjection),
        Circuit = Lux.initialstates(rng, tail.Circuit),
    )
end

function (tail::ReasoningAuditTail)(inputs::Tuple{<:AbstractMatrix,<:AbstractMatrix,Any}, ps, st)
    base_hidden, proposal_hidden, shared_codebook = inputs
    base_b = reshape(base_hidden, size(base_hidden, 1), size(base_hidden, 2), 1)
    proposal_b = reshape(proposal_hidden, size(proposal_hidden, 1), size(proposal_hidden, 2), 1)
    output_b, new_st = tail((base_b, proposal_b, shared_codebook), ps, st)
    return dropdims(output_b, dims=3), new_st
end

function (tail::ReasoningAuditTail)(inputs::Tuple{<:AbstractArray{<:Any,3},<:AbstractArray{<:Any,3},Any}, ps, st)
    base_b, proposal_b, shared_codebook = inputs
    dim, seq_len, batch_size = size(proposal_b)
    m = seq_len * batch_size
    filler_dim = dim ÷ tail.num_roles

    audit_input, aih_st = if tail.use_adapters && tail.AuditInputHeader !== nothing && ps.AuditInputHeader !== nothing
        tail.AuditInputHeader(proposal_b, ps.AuditInputHeader, st.AuditInputHeader)
    else
        proposal_b, st.AuditInputHeader
    end
    normed, norm_st = tail.InputNorm(audit_input, ps.InputNorm, st.InputNorm)
    hidden_flat = reshape(normed, dim, m)

    fine_query = ps.FineEncoderWeight * hidden_flat .+ ps.FineEncoderBias
    quantized, indices = vq_quantize(fine_query, shared_codebook)

    role_flat = ps.RoleBaseWeight * hidden_flat .+ ps.RoleBaseBias .+ ps.RoleShiftWeight * quantized
    roles = reshape(role_flat, filler_dim, tail.num_roles, tail.predicate_num_heads, m)

    rule_flat = ps.PredicateRuleBank[:, indices]
    rule_mats = reshape(rule_flat, tail.num_roles, tail.num_roles, tail.predicate_num_heads, m)
    logic_heads = ntuple(head -> begin
        fillers = reshape(copy(selectdim(roles, 3, head)), filler_dim, tail.num_roles, m)
        rules = copy(selectdim(rule_mats, 3, head))
        rule_t = permutedims(rules, (2, 1, 3))
        transformed = NNlib.batched_mul(fillers, rule_t)
        reshape(transformed, dim, m)
    end, tail.predicate_num_heads)
    logic_cat = vcat(logic_heads...)
    logic_features = ps.PredicateOutputWeight * logic_cat .+ ps.PredicateOutputBias

    circuit_input = if tail.use_adapters && ps.CircuitLeafHeaderWeight !== nothing
        ps.CircuitLeafHeaderWeight * logic_features .+ ps.CircuitLeafHeaderBias
    else
        logic_features
    end

    circuit_input, clp_st = tail.CircuitLeafProjection(circuit_input, ps.CircuitLeafProjection, st.CircuitLeafProjection)
    circuit_out, circuit_st = tail.Circuit(
        reshape(circuit_input, dim, seq_len, batch_size),
        ps.Circuit,
        st.Circuit,
    )

    base_flat = reshape(base_b, dim, m)
    proposal_flat = reshape(proposal_b, dim, m)
    delta_flat = proposal_flat .- base_flat
    base_norm = sqrt.(sum(abs2, base_flat, dims=1) .+ 1f-6)
    proposal_norm = sqrt.(sum(abs2, proposal_flat, dims=1) .+ 1f-6)
    agreement_features = (base_flat ./ base_norm) .* (proposal_flat ./ proposal_norm)

    circuit_flat = reshape(circuit_out, dim, m)
    gate_shift = if tail.use_adapters && ps.CircuitGateBiasShift !== nothing
        sum(ps.CircuitGateBiasShift) / length(ps.CircuitGateBiasShift)
    else
        0.0f0
    end
    circuit_score = reshape(sum(circuit_flat .* ps.ScoreWeight, dims=1), 1, m)
    agreement_score = reshape(sum(agreement_features .* ps.AgreementWeight, dims=1), 1, m) .+ ps.AgreementBias
    score = circuit_score .+ agreement_score .+ ps.ScoreBias .+ gate_shift
    gate = NNlib.sigmoid.(tail.veto_gain .* score)

    output_flat = base_flat .+ gate .* delta_flat

    output = reshape(output_flat, dim, seq_len, batch_size)

    new_st = (
        AuditInputHeader = aih_st,
        InputNorm = norm_st,
        CircuitLeafProjection = clp_st,
        Circuit = circuit_st,
    )

    return output, new_st
end

# ============================================================================
# Model
# ============================================================================

struct ReasoningDrafter{TE,PE,FE,FEH,BL,AT,FN,OH} <: LuxLayer
    config::ReasoningDrafterConfig
    TokenEmbedding::TE
    PositionEmbedding::PE
    FrontEnd::FE
    FrontEndHeader::FEH
    Blocks::BL
    AuditTail::AT
    FinalNorm::FN
    OutputHead::OH
end

function ReasoningDrafter(config::ReasoningDrafterConfig)
    blocks = Tuple(ReasoningDrafterBlock(config) for _ in 1:config.number_of_layers)
    return ReasoningDrafter(
        config,
        Lux.Embedding(config.vocab_size => config.embedding_dimension),
        Lux.Embedding(config.max_sequence_length => config.embedding_dimension),
        SharedOpcodeFrontend(config),
        config.use_adapters ? ResidualAdapterHeader(config.embedding_dimension; expansion_factor = config.frontend_header_expansion) : nothing,
        blocks,
        ReasoningAuditTail(config),
        RMSNorm(config.embedding_dimension),
        Lux.Dense(config.embedding_dimension => config.vocab_size; use_bias = false),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::ReasoningDrafter)
    nl = model.config.number_of_layers
    block_params = NamedTuple{ntuple(i -> Symbol("Block_$i"), nl)}(
        Tuple(Lux.initialparameters(rng, block) for block in model.Blocks)
    )
    time_emb = ones(Float32, model.config.time_dimension)

    return (
        TokenEmbedding = Lux.initialparameters(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialparameters(rng, model.PositionEmbedding),
        FrontEnd = Lux.initialparameters(rng, model.FrontEnd),
        FrontEndHeader = model.FrontEndHeader === nothing ? nothing : Lux.initialparameters(rng, model.FrontEndHeader),
        Blocks = block_params,
        AuditTail = Lux.initialparameters(rng, model.AuditTail),
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
        FrontEnd = Lux.initialstates(rng, model.FrontEnd),
        FrontEndHeader = model.FrontEndHeader === nothing ? nothing : Lux.initialstates(rng, model.FrontEndHeader),
        Blocks = block_states,
        AuditTail = Lux.initialstates(rng, model.AuditTail),
        FinalNorm = Lux.initialstates(rng, model.FinalNorm),
        OutputHead = Lux.initialstates(rng, model.OutputHead),
    )
end

_apply_reasoning_blocks(::Tuple{}, hidden, time_emb, ::Tuple{}, ::Tuple{}) = hidden, ()

function _normalize_mask_ratio(mask_ratio, batch_size::Int, template)
    t_cpu = if ndims(mask_ratio) == 0
        fill(Float32(mask_ratio), 1, batch_size)
    elseif ndims(mask_ratio) == 1
        reshape(Float32.(mask_ratio), 1, :)
    elseif ndims(mask_ratio) == 2 && size(mask_ratio, 1) == 1
        Float32.(mask_ratio)
    else
        throw(ArgumentError("mask_ratio must be scalar, Vector, or (1, batch) array."))
    end
    size(t_cpu, 2) == batch_size || throw(ArgumentError(
        "mask_ratio batch mismatch: expected batch_size=$(batch_size), got $(size(t_cpu, 2))."
    ))
    return to_device_like(template, t_cpu)
end

function _reasoning_time_embedding(mask_ratio, time_gain, template)
    time_dimension = length(time_gain)
    iseven(time_dimension) || throw(ArgumentError(
        "ReasoningDrafter requires even time_dimension for explicit mask conditioning, got $(time_dimension)."
    ))
    sinusoidal = ChainRulesCore.ignore_derivatives() do
        half_dim = time_dimension ÷ 2
        freqs_cpu = Float32.(exp.(-(log(10000.0f0)) .* collect(0:half_dim-1) ./ half_dim))
        freqs = to_device_like(template, freqs_cpu)
        args = freqs * vec(mask_ratio)'
        vcat(sin.(args), cos.(args))
    end
    gain = to_device_like(template, reshape(time_gain, :, 1))
    return gain .* sinusoidal
end

function _apply_reasoning_blocks(blocks::Tuple, hidden, time_emb, ps_vals::Tuple, st_vals::Tuple)
    block = first(blocks)
    block_ps = first(ps_vals)
    block_st = first(st_vals)

    next_hidden, next_state = block((hidden, time_emb), block_ps, block_st)
    final_hidden, remaining_states = _apply_reasoning_blocks(
        Base.tail(blocks), next_hidden, time_emb, Base.tail(ps_vals), Base.tail(st_vals)
    )

    return final_hidden, (next_state, remaining_states...)
end

function reasoning_hidden(model::ReasoningDrafter, inputs::NamedTuple, ps, st)
    hasproperty(inputs, :token_ids) || throw(ArgumentError("ReasoningDrafter inputs must include `token_ids`."))
    token_ids = inputs.token_ids
    mask_ratio = hasproperty(inputs, :mask_ratio) ? inputs.mask_ratio : 0.0f0
    was_unbatched = ndims(token_ids) == 1
    tokens = was_unbatched ? reshape(token_ids, :, 1) : token_ids
    seq_len, batch_size = size(tokens)
    seq_len <= model.config.max_sequence_length || throw(ArgumentError(
        "ReasoningDrafter received seq_len=$(seq_len), but max_sequence_length=$(model.config.max_sequence_length). " *
        "Truncate or pad inputs before calling the model."
    ))

    tok_flat = vec(tokens)
    tok_emb_flat, tok_st = model.TokenEmbedding(tok_flat, ps.TokenEmbedding, st.TokenEmbedding)
    tok_emb = reshape(tok_emb_flat, model.config.embedding_dimension, seq_len, batch_size)

    pos_indices = collect(1:seq_len)
    pos_emb_raw, pos_st = model.PositionEmbedding(pos_indices, ps.PositionEmbedding, st.PositionEmbedding)
    pos_emb = reshape(pos_emb_raw, model.config.embedding_dimension, seq_len, 1)

    hidden = tok_emb .+ pos_emb
    t_input = _normalize_mask_ratio(mask_ratio, batch_size, hidden)
    time_emb = _reasoning_time_embedding(t_input, ps.TimeEmbedding, hidden)

    hidden, frontend_st = model.FrontEnd((hidden, time_emb), ps.FrontEnd, st.FrontEnd)
    hidden, frontend_header_st = if model.config.use_adapters && model.FrontEndHeader !== nothing && ps.FrontEndHeader !== nothing
        model.FrontEndHeader(hidden, ps.FrontEndHeader, st.FrontEndHeader)
    else
        hidden, st.FrontEndHeader
    end
    proposal_hidden, block_states = _apply_reasoning_blocks(
        model.Blocks, hidden, time_emb, values(ps.Blocks), values(st.Blocks)
    )
    hidden, audit_st = model.AuditTail(
        (hidden, proposal_hidden, ps.FrontEnd.Codebook),
        ps.AuditTail,
        st.AuditTail,
    )

    hidden, fn_st = model.FinalNorm(hidden, ps.FinalNorm, st.FinalNorm)

    new_st = (
        TokenEmbedding = tok_st,
        PositionEmbedding = pos_st,
        FrontEnd = frontend_st,
        FrontEndHeader = frontend_header_st,
        Blocks = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.config.number_of_layers)}(
            block_states
        ),
        AuditTail = audit_st,
        FinalNorm = fn_st,
        OutputHead = st.OutputHead,
    )

    if was_unbatched
        hidden = dropdims(hidden, dims = 3)
    end

    return hidden, new_st
end

function (model::ReasoningDrafter)(inputs::NamedTuple, ps, st)
    hidden, partial_st = reasoning_hidden(model, inputs, ps, st)
    logits, oh_st = model.OutputHead(hidden, ps.OutputHead, st.OutputHead)
    new_st = merge(partial_st, (OutputHead = oh_st,))
    return logits, new_st
end

function (model::ReasoningDrafter)(token_ids::AbstractArray{<:Integer}, ps, st)
    return model((token_ids = token_ids, mask_ratio = 0.0f0), ps, st)
end

function (model::ReasoningDrafter)(token_ids::AbstractArray{<:Integer}, mask_ratio, ps, st)
    return model((token_ids = token_ids, mask_ratio = mask_ratio), ps, st)
end

# ============================================================================
# Generation utilities
# ============================================================================

function draft_reasoning_tokens(
    model::ReasoningDrafter,
    prompt_ids::AbstractVector{<:Integer},
    ps, st;
    num_tokens::Int = 8,
)
    isempty(prompt_ids) && throw(ArgumentError("draft_reasoning_tokens requires a non-empty prompt."))
    prompt_len = length(prompt_ids)
    prompt_len <= model.config.max_sequence_length || throw(ArgumentError(
        "draft_reasoning_tokens received prompt_len=$(prompt_len), but max_sequence_length=$(model.config.max_sequence_length)."
    ))
    max_len = min(prompt_len + num_tokens, model.config.max_sequence_length)
    token_buffer = Matrix{Int}(undef, max_len, 1)
    prompt_len > 0 && (token_buffer[1:prompt_len, 1] .= collect(Int, prompt_ids))
    st = Lux.testmode(st)

    draft_logits = Vector{Vector{Float32}}()
    sizehint!(draft_logits, max(max_len - prompt_len, 0))

    active_len = prompt_len
    while active_len < max_len
        tokens = @view token_buffer[1:active_len, :]
        logits, st = model((token_ids = tokens, mask_ratio = 0.0f0), ps, st)
        last_logits = logits[:, end, 1]
        push!(draft_logits, Vector{Float32}(last_logits))
        next_token = argmax(last_logits)
        active_len += 1
        token_buffer[active_len, 1] = next_token
    end

    return vec(token_buffer[1:active_len, 1]), draft_logits
end

# ============================================================================
# Parameter / EMA helpers
# ============================================================================

function _count_params(x)
    x isa AbstractArray && return length(x)
    x isa NamedTuple && return sum(_count_params(v) for v in values(x))
    x isa Tuple && return sum(_count_params(v) for v in x)
    x isa Nothing && return 0
    return 0
end

function estimate_drafter_parameters(config::ReasoningDrafterConfig)
    rng = Random.MersenneTwister(0)
    return _count_params(Lux.initialparameters(rng, ReasoningDrafter(config)))
end

function apply_reasoning_drafter_ema_codebook!(
    ps, st, model::ReasoningDrafter; laplace_smoothing::Float32 = 1f-5
)
    apply_shared_frontend_ema_codebook!(
        ps.FrontEnd,
        st.FrontEnd,
        model.FrontEnd;
        laplace_smoothing = laplace_smoothing,
    )
    return ps
end

# ============================================================================
# Exports
# ============================================================================

export ReasoningDrafterConfig, ResidualAdapterHeader, ReasoningDrafterBlock, ReasoningDrafter, reasoning_hidden
export draft_reasoning_tokens, estimate_drafter_parameters, apply_reasoning_drafter_ema_codebook!

end # module
