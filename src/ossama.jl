module ossama

include("Dlinoss.jl")
include("Attention.jl")
include("ossm.jl")

# Re-export submodules for callers who want direct access.
export Dlinoss, Attention, ossm

# Provide conventional aliases for the main layer types.
const DLinOSS = Dlinoss.DLinOSS
const SWAttention = Attention.SWAttention
const Ossm = ossm.ossm

export DLinOSS, SWAttention, Ossm

end
