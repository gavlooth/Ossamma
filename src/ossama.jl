module ossama

include("Dlinoss.jl")
include("Attention.jl")
include("ossm.jl")

using ..Attention
using ..Dlinoss
using Lux
using Random
using NNlib


struct GLU <: Lux.AbstractLuxLayer

end



struct Ossama <: Lux.AbstractLuxLayer
    input_dimensions::Int
    ossl::DLinOSS
    conv::Lux.Conv
    Wattention::Attention
    output_dimensions::Int
    #GLU layer
    GluProjection1::Lux.Dense
    GluProjection2::Lux.Dense
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::Ossama)
    return (
        ossl = Lux.initialparameters(rng, layer.ossl),
        conv = Lux.initialparameters(rng, layer.conv),
        Wattention = Lux.initialparameters(rng, layer.Wattention),
        GluProjection1 = Lux.initialparameters(rng, layer.GluProjection1),
        GluProjection2 = Lux.initialparameters(rng, layer.GluProjection2),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::Ossama)
    return (
        ossl = Lux.initialstates(rng, layer.ossl),
        conv = Lux.initialstates(rng, layer.conv),
        Wattention = Lux.initialstates(rng, layer.Wattention),
        GluProjection1 = Lux.initialstates(rng, layer.GluProjection1),
        GluProjection2 = Lux.initialstates(rng, layer.GluProjection2),
    )
end



# Re-export submodules for callers who want direct access.
export Dlinoss, Attention, ossm

# Provide conventional aliases for the main layer types.
const DLinOSS = Dlinoss.DLinOSS
const SWAttention = Attention.SWAttention
const Ossm = ossm.ossm

export DLinOSS, SWAttention, Ossm

end
