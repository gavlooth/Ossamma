module ossm

import LuxCore, Lux, Random, NNlib

struct ossm <: Lux.AbstractLuxLayer
    dim_in::Int
    dim_out::Int
    oscillators_count::Int
end

@inline function ossm_dim(block::ossm)
    2 * block.oscillators_count
end

function apply_oscillation(block, x, ρ, θ)
    H = block.oscillators_count
    @assert size(x) == (2H, 1)

    x_view = reshape(x, 2, H)                 # (2, H)
    slices = eachslice(x_view; dims = 2)      # iterator over columns (each is length-2)

    cols = [
        ρi * [cos(θi) -sin(θi); sin(θi) cos(θi)] * xi for (ρi, θi, xi) in zip(ρ, θ, slices)
    ]                                         # vector of length-H 2-vectors

    X_next = reduce(hcat, cols)               # (2, H)
    return reshape(X_next, 2H, 1)             # (2H, 1)  <-- Convention A
end

function Lux.initialparameters(rng, block::ossm)
    state_dim = ossm_dim(block)
    H = block.oscillators_count

    return (
        θ = randn(rng, H),
        α = randn(rng, H),
        B = randn(rng, state_dim, block.dim_in),
        C = randn(rng, block.dim_out, state_dim),
        D = zeros(Float32, block.dim_out, block.dim_in),
    )
end

function Lux.initialstates(rng, block::ossm)
    (; oscillation_state = zeros(Float32, ossm_dim(block), 1))
end

function oscillator_step(block, params, xt, ut)
    (; θ, α, B, C, D) = params
    ρ = NNlib.sigmoid.(α)

    @assert size(xt) == (ossm_dim(block), 1)
    @assert size(ut) == (block.dim_in, 1)

    x_next = apply_oscillation(block, xt, ρ, θ) + B * ut   # (2H,1)
    y = C * xt + D * ut                               # (dim_out,1)

    return (y, (; oscillation_state = x_next))
end

function (block::ossm)(u, params, state)
    xt = state.oscillation_state
    return oscillator_step(block, params, xt, u)
end

end
