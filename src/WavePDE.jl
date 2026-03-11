module WavePDE

"""
WavePDE — Spectral Wave-PDE sequence layer.

Implements a projection-free damped wave equation layer aligned with the
Wave-PDE Nets specification.

The 2nd-order damped wave equation on N spatial points:
    ∂²u/∂t² = c(x)² · ∂²u/∂x² - γ(x) · ∂u/∂t

Reformulated as a first-order system (u = displacement, v = velocity):
    ∂u/∂t = v
    ∂v/∂t = c(x)² · ∂²u/∂x² - γ(x) · v

The Laplacian ∂²u/∂x² is computed spectrally via FFT:
    ∂²u/∂x² = IFFT( -k² · FFT(u) )
    where k_j = 2π·j/N  (j = 0, 1, ..., N-1 in FFT order)

Trainable parameters per layer:
    c(x) ∈ ℝ^N  — wave speed per spatial mode  (constrained > 0 via softplus)
    γ(x) ∈ ℝ^N  — damping per spatial mode      (constrained ≥ 0 via softplus)

Integration: leapfrog (Störmer-Verlet) with exact damping split.
    d      = exp(-γ · dt / 2)
    v_a    = d ⊙ v
    v_b    = v_a + (dt/2) · c² ⊙ L(u)
    u_next = u + dt · v_b
    v_c    = v_b + (dt/2) · c² ⊙ L(u_next)
    v_next = d ⊙ v_c

Input/output interface matches the legacy wave-gate layer interface:
    input:  (input_dim, T) or (input_dim, T, B)
    output: (output_dim, T) or (output_dim, T, B)
    state:  stateless NamedTuple()
"""

using Lux
using Random
using NNlib
using FFTW
using Zygote: @ignore

# ============================================================================
# Layer Definition
# ============================================================================

struct WavePDELayer <: Lux.AbstractLuxLayer
    input_dimension::Int
    state_dimension::Int   # N — spatial domain size (typically == embedding_dim)
    output_dimension::Int
    minimum_frequency::Float32  # kept for constructor compatibility; not used
    maximum_frequency::Float32  # kept for constructor compatibility; not used
    default_time_step::Float32  # Δt for numerical integration
    integration_steps::Int      # internal PDE integration steps
    lambda::Vector{Float32}     # λ_m = 2(cos(2πm/N)-1), fixed spectral operator
end

function make_lambda(N::Int)
    m = Float32.(FFTW.fftfreq(N) .* N)
    @. 2f0 * (cos(2f0 * Float32(pi) * m / N) - 1f0)
end

function WavePDELayer(
    input_dimension::Int,
    state_dimension::Int,
    output_dimension::Int,
    minimum_frequency::Real,
    maximum_frequency::Real,
    default_time_step::Real;
    integration_steps::Int = 8,
)
    return WavePDELayer(
        input_dimension,
        state_dimension,
        output_dimension,
        Float32(minimum_frequency),
        Float32(maximum_frequency),
        Float32(default_time_step);
        integration_steps = integration_steps,
    )
end

function WavePDELayer(
    input_dimension::Int,
    state_dimension::Int,
    output_dimension::Int,
    minimum_frequency::Float32,
    maximum_frequency::Float32,
    default_time_step::Float32;
    integration_steps::Int = 8,
)
    input_dimension > 0 || throw(ArgumentError("input_dimension must be positive"))
    state_dimension > 0 || throw(ArgumentError("state_dimension must be positive"))
    output_dimension > 0 || throw(ArgumentError("output_dimension must be positive"))
    default_time_step > 0 || throw(ArgumentError("default_time_step must be positive"))
    integration_steps > 0 || throw(ArgumentError("integration_steps must be positive"))

    # Projection-free Wave-PDE core requires matching dimensions.
    if !(input_dimension == state_dimension == output_dimension)
        throw(ArgumentError(
            "WavePDELayer is projection-free: input_dimension ($input_dimension), state_dimension ($state_dimension), and output_dimension ($output_dimension) must match.",
        ))
    end

    return WavePDELayer(
        input_dimension,
        state_dimension,
        output_dimension,
        minimum_frequency,
        maximum_frequency,
        default_time_step,
        integration_steps,
        make_lambda(state_dimension),
    )
end

function laplacian_batch(u::AbstractMatrix, λ::AbstractVector)
    U = fft(u, 1)
    real.(ifft(λ .* U, 1))
end

function leapfrog_step(
    u::AbstractMatrix,
    v::AbstractMatrix,
    c_sq::AbstractMatrix,
    d::AbstractMatrix,
    λ::AbstractVector,
    dt::Float32,
)
    v_a = d .* v
    v_b = v_a .+ (dt / 2f0) .* (c_sq .* laplacian_batch(u, λ))
    u_next = u .+ dt .* v_b
    v_c = v_b .+ (dt / 2f0) .* (c_sq .* laplacian_batch(u_next, λ))
    v_next = d .* v_c
    return u_next, v_next
end

# ============================================================================
# Parameter Initialization
# ============================================================================

function Lux.initialparameters(_rng::Random.AbstractRNG, layer::WavePDELayer)
    N = layer.state_dimension

    # Spec-faithful initialisation:
    # softplus(0)  ≈ 0.693  -> initial c
    # softplus(-3) ≈ 0.049  -> light initial damping
    log_wave_speed = zeros(Float32, N)
    log_damping = fill(-3f0, N)

    return (
        log_wave_speed    = log_wave_speed,
        log_damping       = log_damping,
    )
end

# ============================================================================
# State Initialization
# ============================================================================

function Lux.initialstates(_rng::Random.AbstractRNG, _layer::WavePDELayer)
    # Projection-free Wave-PDE block is stateless across forwards.
    NamedTuple()
end

# ============================================================================
# Forward Pass
# ============================================================================

function (layer::WavePDELayer)(input_sequence::AbstractArray, parameters, state)
    ndims(input_sequence) in (2, 3) || throw(ArgumentError(
        "WavePDELayer expects 2D or 3D input, got shape $(size(input_sequence)).",
    ))
    is_batched = ndims(input_sequence) == 3
    standardized = is_batched ?
        input_sequence :
        reshape(input_sequence, size(input_sequence, 1), size(input_sequence, 2), 1)

    (n_spatial, n_columns, n_batches) = size(standardized)
    N = layer.state_dimension

    n_spatial == layer.input_dimension || throw(ArgumentError(
        "WavePDELayer input dimension mismatch: expected $(layer.input_dimension), got $n_spatial.",
    ))
    n_spatial == N || throw(ArgumentError(
        "WavePDELayer expects first input axis = state_dimension ($N), got $n_spatial.",
    ))

    c = NNlib.softplus.(parameters.log_wave_speed)
    γ = NNlib.softplus.(parameters.log_damping)
    length(c) == N || throw(ArgumentError("Expected $(N) wave-speed params, got $(length(c))."))
    length(γ) == N || throw(ArgumentError("Expected $(N) damping params, got $(length(γ))."))

    dt = layer.default_time_step
    maximum(c) * dt >= 1f0 &&
        @warn "WavePDE stability condition violated: Δt·max(c) = $(dt * maximum(c)) ≥ 1"

    c_sq = reshape(c .^ 2, N, 1)
    d = reshape(exp.(-γ .* dt ./ 2f0), N, 1)

    # Run PDE on each column independently (columns = sequence*batch).
    u = reshape(standardized, N, :)
    v = similar(u)
    fill!(v, zero(eltype(v)))

    # Put λ on the same device/container as parameters.
    λ = @ignore let
        buf = similar(c, N)
        buf .= layer.lambda
        buf
    end

    for _ in 1:layer.integration_steps
        u, v = leapfrog_step(u, v, c_sq, d, λ, dt)
    end

    output_tensor = reshape(u, N, n_columns, n_batches)
    final_output = is_batched ? output_tensor : dropdims(output_tensor; dims = 3)
    return (final_output, state)
end

end # module WavePDE
