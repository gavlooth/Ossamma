module RuleConditionedWavePDEMod

"""
Rule-Conditioned WavePDE: Reasoning-Modulated Wave Dynamics

Combines VQ-VAE situation recognition with WavePDE parameter modulation:

1. **VQ codebook** quantizes hidden states into discrete "reasoning situations"
2. Each code retrieves a **rule vector** from a learned rule bank
3. The rule vector **modulates WavePDE dynamics** — changing wave speed c(x)
   and damping γ(x) per frequency mode
4. The modulated WavePDE solves the wave equation with rule-specific physics

This replaces PredicateEngram's TPR role-filler mechanism with something
native to the architecture: rules don't directly transform hidden states,
they change **how information propagates** through the wave equation.

Physical interpretation:
- A "transitivity" rule increases propagation speed (consequences flow faster)
  and reduces damping (preserves the inference chain)
- A "contradiction" rule increases damping at specific frequencies
  (suppresses inconsistent modes)
- A "case split" rule creates frequency-selective channels
  (different cases propagate at different speeds)

The wave equation does the reasoning. The rule tells it how.
"""

using Lux
using Random
using NNlib
using FFTW
import CUDA
using ChainRulesCore
using Statistics: mean
using LinearAlgebra: I

using ..Swamma: LuxLayer, to_device_like, state_with_training, state_is_training

# ============================================================================
# RuleConditionedWavePDE
# ============================================================================

struct RuleConditionedWavePDE <: LuxLayer
    state_dimension::Int        # N — spatial/embedding dimension
    code_dim::Int               # VQ codebook vector dimension
    codebook_size::Int          # number of discrete reasoning situations
    default_time_step::Float32
    integration_steps::Int
    use_adapters::Bool          # Phase 3: thin adapter headers for domain transfer
    lambda::Vector{Float32}     # fixed spectral operator (same as WavePDELayer)
end

function RuleConditionedWavePDE(
    state_dimension::Int;
    code_dim::Int = 64,
    codebook_size::Int = 512,
    default_time_step::Float32 = 0.1f0,
    integration_steps::Int = 8,
    use_adapters::Bool = false,
)
    state_dimension > 0 || throw(ArgumentError("state_dimension must be positive"))
    code_dim > 0 || throw(ArgumentError("code_dim must be positive"))
    codebook_size > 0 || throw(ArgumentError("codebook_size must be positive"))

    # Precompute spectral Laplacian eigenvalues: λ_m = 2(cos(2πm/N) - 1)
    m = Float32.(FFTW.fftfreq(state_dimension) .* state_dimension)
    lambda = @. 2f0 * (cos(2f0 * Float32(pi) * m / state_dimension) - 1f0)

    return RuleConditionedWavePDE(
        state_dimension, code_dim, codebook_size,
        default_time_step, integration_steps, use_adapters, lambda,
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::RuleConditionedWavePDE)
    N = layer.state_dimension
    cd = layer.code_dim
    cs = layer.codebook_size
    he = Float32(sqrt(2.0 / (cd + N)))

    # Adapter headers: identity-initialized for domain transfer (Phase 3)
    adapters = if layer.use_adapters
        (
            EncoderHeaderWeight = Matrix{Float32}(I, cd, cd),  # identity init
            EncoderHeaderBias = zeros(Float32, cd),
            RuleBankHeaderWeight = Matrix{Float32}(I, cd, cd),
            RuleBankHeaderBias = zeros(Float32, cd),
            GateBiasShift = zeros(Float32, N),  # additive shift on gate bias
        )
    else
        (
            EncoderHeaderWeight = nothing,
            EncoderHeaderBias = nothing,
            RuleBankHeaderWeight = nothing,
            RuleBankHeaderBias = nothing,
            GateBiasShift = nothing,
        )
    end

    return (
        # VQ-VAE codebook: (code_dim, codebook_size)
        Codebook = randn(rng, Float32, cd, cs) .* 0.1f0,
        # Encoder: state_dimension → code_dim
        EncoderWeight = randn(rng, Float32, cd, N) .* he,
        EncoderBias = zeros(Float32, cd),
        # Rule bank: each code → rule vector of size code_dim
        RuleBank = randn(rng, Float32, cd, cs) .* 0.1f0,
        # Modulation projections: rule_vector → wave speed/damping shifts
        SpeedModWeight = randn(rng, Float32, N, cd) .* Float32(sqrt(2.0 / (N + cd))) .* 0.1f0,
        DampingModWeight = randn(rng, Float32, N, cd) .* Float32(sqrt(2.0 / (N + cd))) .* 0.1f0,
        # Base WavePDE parameters (same as WavePDELayer)
        log_wave_speed = zeros(Float32, N),         # softplus(0) ≈ 0.693
        log_damping = fill(-3f0, N),                # softplus(-3) ≈ 0.049
        # Output gate (starts near-closed)
        GateWeight = randn(rng, Float32, N, N) .* Float32(sqrt(2.0 / (2N))),
        GateBias = fill(Float32(-2.0), N),
        # Adapter headers
        adapters...,
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::RuleConditionedWavePDE)
    return state_with_training(
        # EMA codebook update state
        ema_cluster_size = zeros(Float32, layer.codebook_size),
        ema_embed_sum = zeros(Float32, layer.code_dim, layer.codebook_size),
        lambda_cache = nothing,
    )
end

function _ensure_lambda_cache(layer::RuleConditionedWavePDE, template, st)
    cache = get(st, :lambda_cache, nothing)
    template_is_gpu = occursin("CuArray", string(typeof(template)))
    cache_matches = cache !== nothing &&
                    occursin("CuArray", string(typeof(cache))) == template_is_gpu &&
                    length(cache) == length(layer.lambda)
    cache_matches && return cache, st

    λ = to_device_like(template, layer.lambda)
    return λ, merge(st, (lambda_cache = λ,))
end

# ============================================================================
# VQ quantization (same as PredicateEngram — on-device)
# ============================================================================

function vq_quantize_rc(query, codebook)
    indices = ChainRulesCore.ignore_derivatives() do
        q_sq = sum(query .^ 2, dims=1)
        c_sq = sum(codebook .^ 2, dims=1)
        dots = codebook' * query
        dists = q_sq .+ c_sq' .- 2 .* dots
        idx_cart = argmin(dists, dims=1)
        vec(map(ci -> ci[1], Array(idx_cart)))
    end
    quantized = codebook[:, indices]
    diff = ChainRulesCore.ignore_derivatives() do
        quantized .- query
    end
    quantized_st = query .+ diff
    return quantized_st, indices
end

# ============================================================================
# Spectral PDE integration with modulated parameters
# ============================================================================

_fft_work_type(::Type{T}) where {T<:AbstractFloat} = T
_fft_work_type(::Type{Float16}) = Float32
_fft_work_type(::Type{Core.BFloat16}) = Float32

function laplacian_batch(u::AbstractMatrix, λ::AbstractVector)
    input_type = eltype(u)
    work_type = _fft_work_type(input_type)
    u_fft = work_type === input_type ? u : work_type.(u)
    λ_fft = eltype(λ) === work_type ? λ : work_type.(λ)
    spectral = real.(ifft(λ_fft .* fft(u_fft, 1), 1))
    return work_type === input_type ? spectral : input_type.(spectral)
end

function leapfrog_step(u, v, c_sq, d, λ, dt)
    v_a = d .* v
    v_b = v_a .+ (dt / 2f0) .* (c_sq .* laplacian_batch(u, λ))
    u_next = u .+ dt .* v_b
    v_c = v_b .+ (dt / 2f0) .* (c_sq .* laplacian_batch(u_next, λ))
    v_next = d .* v_c
    return u_next, v_next
end

# ============================================================================
# Forward Pass
# ============================================================================

"""
    (layer::RuleConditionedWavePDE)(hidden_state, ps, st) → (output, state)

1. Encode hidden → VQ code (what reasoning situation?)
2. Retrieve rule vector from bank
3. Modulate wave speed and damping: c(x) = softplus(base + W_c · rule)
4. Run leapfrog PDE integration with modulated parameters
5. Gate and residual inject
"""
function (layer::RuleConditionedWavePDE)(hidden_state, ps, st)
    is_batched = ndims(hidden_state) == 3
    if !is_batched
        hidden_b = reshape(hidden_state, size(hidden_state, 1), size(hidden_state, 2), 1)
    else
        hidden_b = hidden_state
    end

    N, seq_len, batch_size = size(hidden_b)
    M = seq_len * batch_size
    hidden_flat = reshape(hidden_b, N, M)

    # ==================================================================
    # 1. VQ encode: hidden → discrete reasoning situation code
    # ==================================================================
    query = ps.EncoderWeight * hidden_flat .+ ps.EncoderBias  # (code_dim, M)

    # Adapter header: domain-specific correction on encoder output
    if layer.use_adapters && ps.EncoderHeaderWeight !== nothing
        query = ps.EncoderHeaderWeight * query .+ ps.EncoderHeaderBias
    end

    _quantized, indices = vq_quantize_rc(query, ps.Codebook)

    # ==================================================================
    # 2. Retrieve rule vectors from bank
    # ==================================================================
    rule_vectors = ps.RuleBank[:, indices]  # (code_dim, M)

    # Adapter header: domain-specific correction on rule vectors
    if layer.use_adapters && ps.RuleBankHeaderWeight !== nothing
        rule_vectors = ps.RuleBankHeaderWeight * rule_vectors .+ ps.RuleBankHeaderBias
    end

    # ==================================================================
    # 3. Modulate WavePDE parameters — DIFFERENTIABLE
    #    Gradients flow to log_wave_speed, log_damping, SpeedModWeight,
    #    DampingModWeight via the modulation computation.
    # ==================================================================
    dt = layer.default_time_step

    base_speed = ps.log_wave_speed
    base_damping = ps.log_damping
    speed_shift = ps.SpeedModWeight * rule_vectors
    damping_shift = ps.DampingModWeight * rule_vectors
    c = clamp.(NNlib.softplus.(base_speed .+ speed_shift), 0.1f0, 2.0f0)
    γ = clamp.(NNlib.softplus.(base_damping .+ damping_shift), 0.01f0, 1.0f0)
    c_sq = c .^ 2
    d = exp.(-γ .* dt ./ 2f0)

    # ==================================================================
    # 4. PDE integration — DETACHED (FFT leapfrog is the memory hog)
    #    Uses detached copies of c_sq and d so the leapfrog loop itself
    #    doesn't build an AD tape, but modulation params above get gradients
    #    via the gate path (step 5) which uses the original c_sq/d indirectly.
    # ==================================================================
    λ, st = _ensure_lambda_cache(layer, hidden_flat, st)

    u = ChainRulesCore.ignore_derivatives() do
        c_sq_det = c_sq
        d_det = d
        u_pde = hidden_flat
        v_pde = zero(u_pde)

        for _ in 1:layer.integration_steps
            u_pde, v_pde = leapfrog_step(u_pde, v_pde, c_sq_det, d_det, λ, dt)
        end
        u_pde
    end

    # ==================================================================
    # 5. Gate and residual inject
    #    Gradients flow through gate (learned) and hidden_flat (residual).
    #    The PDE output u is detached but modulation params (speed, damping)
    #    still get signal via the gate's dependence on hidden_flat.
    # ==================================================================
    gate_bias = ps.GateBias
    if layer.use_adapters && ps.GateBiasShift !== nothing
        gate_bias = gate_bias .+ ps.GateBiasShift
    end
    gate = NNlib.sigmoid.(ps.GateWeight * hidden_flat .+ gate_bias)
    # Modulate gate per-dimension by wave params — direct gradient path for c/γ.
    # c scales gate up (faster wave = stronger signal), γ scales it down (more damping = weaker).
    # No reduction, no sigmoid saturation — raw (N, M) modulation.
    # Normalize by init values so modulation starts near 1.0 and c/γ changes are relative.
    wave_mod = (c ./ (c .+ 1f0)) .* (1f0 ./ (γ .+ 1f0))   # (N, M), ∈ (0, 1)
    output_flat = hidden_flat .+ (gate .* wave_mod) .* u

    output = reshape(output_flat, N, seq_len, batch_size)
    if !is_batched
        output = dropdims(output, dims=3)
    end

    # ==================================================================
    # 6. EMA codebook update — fully detached copies to avoid Zygote trace issues
    # ==================================================================
    new_st = if state_is_training(st)
        query_det = ChainRulesCore.ignore_derivatives(query)
        indices_det = ChainRulesCore.ignore_derivatives(indices)
        ema_updates = ChainRulesCore.ignore_derivatives() do
            _ema_update(layer, query_det, indices_det, st)
        end
        merge(st, ema_updates)
    else
        st
    end

    return output, new_st
end

function _ema_update(layer::RuleConditionedWavePDE, query, indices, st)
    decay = 0.99f0
    query_cpu = Array(query)
    indices_cpu = Array(indices)
    cs = layer.codebook_size
    M = length(indices_cpu)

    new_counts = zeros(Float32, cs)
    for i in 1:M
        new_counts[indices_cpu[i]] += 1.0f0
    end

    new_sums = zeros(Float32, layer.code_dim, cs)
    for i in 1:M
        new_sums[:, indices_cpu[i]] .+= @view query_cpu[:, i]
    end

    # Move to same device as state before broadcasting
    ema_cs = Array(st.ema_cluster_size)
    ema_es = Array(st.ema_embed_sum)
    new_ema_cs = decay .* ema_cs .+ (1.0f0 - decay) .* new_counts
    new_ema_es = decay .* ema_es .+ (1.0f0 - decay) .* new_sums

    # Convert back to same type as original state
    return (
        ema_cluster_size = typeof(st.ema_cluster_size)(new_ema_cs),
        ema_embed_sum = typeof(st.ema_embed_sum)(new_ema_es),
    )
end

# ============================================================================
# Commitment loss (for training)
# ============================================================================

"""
    rc_wavepde_commitment_loss(hidden_flat, ps, layer) → Float32

VQ-VAE commitment loss for the rule-conditioned WavePDE.
"""
function rc_wavepde_commitment_loss(hidden_flat, ps, layer::RuleConditionedWavePDE)
    query = ps.EncoderWeight * hidden_flat .+ ps.EncoderBias
    _, indices = vq_quantize_rc(query, ps.Codebook)
    quantized = ps.Codebook[:, indices]
    return Float32(mean((query .- ChainRulesCore.ignore_derivatives(quantized)) .^ 2))
end

"""
    apply_rc_ema_codebook!(ps, st, layer; laplace_smoothing=1f-5)

Apply EMA statistics to update codebook vectors in-place.
"""
function apply_rc_ema_codebook!(ps, st, layer::RuleConditionedWavePDE; laplace_smoothing::Float32 = 1f-5)
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

"""
    revive_dead_codes!(ps, st, layer; threshold=1f-2, noise_scale=0.01f0)

Reinitialize dead codebook entries from the most-used code + noise.
Dead = EMA cluster size below threshold. Standard VQ-VAE practice.
Returns number of codes revived.
"""
function revive_dead_codes!(ps, st, layer::RuleConditionedWavePDE;
                            threshold::Float32 = 1f-2, noise_scale::Float32 = 0.01f0)
    ema_cs = Array(st.ema_cluster_size)
    codebook_cpu = Array(ps.Codebook)
    cs = layer.codebook_size

    dead = findall(<=(threshold), ema_cs)
    isempty(dead) && return 0

    # Find the most-used code
    best = argmax(ema_cs)
    best_vec = codebook_cpu[:, best]

    revived = 0
    for d in dead
        # Reinit from best code + uniform noise
        codebook_cpu[:, d] .= best_vec .+ noise_scale .* (rand(Float32, size(best_vec)) .- 0.5f0)
        # Reset EMA stats for this code
        ema_cs[d] = 1f0  # small nonzero so it's "active"
        revived += 1
    end

    ps.Codebook .= to_device_like(ps.Codebook, codebook_cpu)
    st.ema_cluster_size .= to_device_like(st.ema_cluster_size, ema_cs)
    return revived
end

# ============================================================================
# Diagnostics
# ============================================================================

"""
    rc_codebook_diagnostics(ps, st, layer) → NamedTuple

Return codebook health metrics without modifying state:
- active_codes: number of codebook entries with nonzero EMA count
- total_codes: codebook size
- utilization: active_codes / total_codes
- top5_counts: EMA counts of the 5 most-used codes
- bottom5_counts: EMA counts of the 5 least-used active codes
- wave_speed_range: (min, max) of softplus(log_wave_speed)
- damping_range: (min, max) of softplus(log_damping)
"""
function rc_codebook_diagnostics(ps, st, layer::RuleConditionedWavePDE)
    ema_cs = Array(st.ema_cluster_size)
    cs = layer.codebook_size
    active = findall(>(1f-3), ema_cs)
    sorted_counts = sort(ema_cs, rev=true)

    speed = NNlib.softplus.(Array(ps.log_wave_speed))
    damping = NNlib.softplus.(Array(ps.log_damping))

    return (
        active_codes = length(active),
        total_codes = cs,
        utilization = length(active) / cs,
        top5_counts = sorted_counts[1:min(5, cs)],
        bottom5_counts = sorted_counts[max(1, cs-4):cs],
        wave_speed_range = (minimum(speed), maximum(speed)),
        damping_range = (minimum(damping), maximum(damping)),
    )
end

# ============================================================================
# Exports
# ============================================================================

export RuleConditionedWavePDE
export rc_wavepde_commitment_loss, apply_rc_ema_codebook!, rc_codebook_diagnostics, revive_dead_codes!

end # module
