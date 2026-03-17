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

using ..Swamma: LuxLayer

# ============================================================================
# RuleConditionedWavePDE
# ============================================================================

struct RuleConditionedWavePDE <: LuxLayer
    state_dimension::Int        # N — spatial/embedding dimension
    code_dim::Int               # VQ codebook vector dimension
    codebook_size::Int          # number of discrete reasoning situations
    default_time_step::Float32
    integration_steps::Int
    lambda::Vector{Float32}     # fixed spectral operator (same as WavePDELayer)
end

function RuleConditionedWavePDE(
    state_dimension::Int;
    code_dim::Int = 64,
    codebook_size::Int = 512,
    default_time_step::Float32 = 0.1f0,
    integration_steps::Int = 8,
)
    state_dimension > 0 || throw(ArgumentError("state_dimension must be positive"))
    code_dim > 0 || throw(ArgumentError("code_dim must be positive"))
    codebook_size > 0 || throw(ArgumentError("codebook_size must be positive"))

    # Precompute spectral Laplacian eigenvalues: λ_m = 2(cos(2πm/N) - 1)
    m = Float32.(FFTW.fftfreq(state_dimension) .* state_dimension)
    lambda = @. 2f0 * (cos(2f0 * Float32(pi) * m / state_dimension) - 1f0)

    return RuleConditionedWavePDE(
        state_dimension, code_dim, codebook_size,
        default_time_step, integration_steps, lambda,
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::RuleConditionedWavePDE)
    N = layer.state_dimension
    cd = layer.code_dim
    cs = layer.codebook_size
    he = Float32(sqrt(2.0 / (cd + N)))

    return (
        # VQ-VAE codebook: (code_dim, codebook_size)
        Codebook = randn(rng, Float32, cd, cs) .* 0.1f0,
        # Encoder: state_dimension → code_dim
        EncoderWeight = randn(rng, Float32, cd, N) .* he,
        EncoderBias = zeros(Float32, cd),
        # Rule bank: each code → rule vector of size code_dim
        RuleBank = randn(rng, Float32, cd, cs) .* 0.1f0,
        # Modulation projections: rule_vector → wave speed/damping shifts
        # Low-rank: (code_dim → N) via code_dim × N matrix
        SpeedModWeight = randn(rng, Float32, N, cd) .* Float32(sqrt(2.0 / (N + cd))) .* 0.1f0,
        DampingModWeight = randn(rng, Float32, N, cd) .* Float32(sqrt(2.0 / (N + cd))) .* 0.1f0,
        # Base WavePDE parameters (same as WavePDELayer)
        log_wave_speed = zeros(Float32, N),         # softplus(0) ≈ 0.693
        log_damping = fill(-3f0, N),                # softplus(-3) ≈ 0.049
        # Output gate (starts near-closed)
        GateWeight = randn(rng, Float32, N, N) .* Float32(sqrt(2.0 / (2N))),
        GateBias = fill(Float32(-2.0), N),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::RuleConditionedWavePDE)
    return (
        # EMA codebook update state
        ema_cluster_size = ones(Float32, layer.codebook_size),
        ema_embed_sum = zeros(Float32, layer.code_dim, layer.codebook_size),
    )
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

function laplacian_batch(u::AbstractMatrix, λ::AbstractVector)
    U = fft(u, 1)
    real.(ifft(λ .* U, 1))
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
    _quantized, indices = vq_quantize_rc(query, ps.Codebook)

    # ==================================================================
    # 2. Retrieve rule vectors from bank
    # ==================================================================
    rule_vectors = ps.RuleBank[:, indices]  # (code_dim, M)

    # ==================================================================
    # 3. Modulate WavePDE parameters per position
    #    Each position gets its own c(x) and γ(x) based on its rule
    # ==================================================================
    # Base parameters: (N,) → broadcast to (N, M)
    base_speed = ps.log_wave_speed   # (N,)
    base_damping = ps.log_damping    # (N,)

    # Rule modulation: (N, code_dim) × (code_dim, M) → (N, M)
    speed_shift = ps.SpeedModWeight * rule_vectors    # (N, M)
    damping_shift = ps.DampingModWeight * rule_vectors  # (N, M)

    # Modulated parameters (per position)
    c = NNlib.softplus.(base_speed .+ speed_shift)    # (N, M)
    γ = NNlib.softplus.(base_damping .+ damping_shift)  # (N, M)

    dt = layer.default_time_step

    c_sq = c .^ 2        # (N, M)
    d = exp.(-γ .* dt ./ 2f0)  # (N, M)

    # ==================================================================
    # 4. PDE integration: leapfrog with per-position modulated dynamics
    #    u is (N, M) where each column is a sequence position
    # ==================================================================
    u = hidden_flat
    v = zero(u)

    λ = ChainRulesCore.ignore_derivatives() do
        if occursin("CuArray", string(typeof(u)))
            CUDA.CuArray(layer.lambda)
        else
            copy(layer.lambda)
        end
    end

    for _ in 1:layer.integration_steps
        u, v = leapfrog_step(u, v, c_sq, d, λ, dt)
    end

    # ==================================================================
    # 5. Gate and residual inject
    # ==================================================================
    gate = NNlib.sigmoid.(ps.GateWeight * hidden_flat .+ ps.GateBias)
    output_flat = hidden_flat .+ gate .* u

    output = reshape(output_flat, N, seq_len, batch_size)
    if !is_batched
        output = dropdims(output, dims=3)
    end

    # ==================================================================
    # 6. EMA codebook update
    # ==================================================================
    new_st = ChainRulesCore.ignore_derivatives() do
        _ema_update(layer, query, indices, st)
    end

    return output, new_st
end

function _ema_update(layer::RuleConditionedWavePDE, query, indices, st)
    decay = 0.99f0
    query_cpu = Array(query)
    cs = layer.codebook_size
    M = length(indices)

    new_counts = zeros(Float32, cs)
    for i in 1:M
        new_counts[indices[i]] += 1.0f0
    end

    new_sums = zeros(Float32, layer.code_dim, cs)
    for i in 1:M
        new_sums[:, indices[i]] .+= @view query_cpu[:, i]
    end

    return (
        ema_cluster_size = decay .* st.ema_cluster_size .+ (1.0f0 - decay) .* new_counts,
        ema_embed_sum = decay .* st.ema_embed_sum .+ (1.0f0 - decay) .* new_sums,
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
    n = sum(st.ema_cluster_size)
    smoothed = (st.ema_cluster_size .+ laplace_smoothing) ./
               (n + layer.codebook_size * laplace_smoothing) .* n
    for c in 1:layer.codebook_size
        ps.Codebook[:, c] .= st.ema_embed_sum[:, c] ./ max(smoothed[c], 1f-8)
    end
end

# ============================================================================
# Exports
# ============================================================================

export RuleConditionedWavePDE
export rc_wavepde_commitment_loss, apply_rc_ema_codebook!

end # module
