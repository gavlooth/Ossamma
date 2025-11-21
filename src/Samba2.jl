module Samba2

import LuxCore, Lux, NNlib


cl = 1024
dim = 128

struct SWAttention <: Lux.AbstractLuxLayer
    sequence_length::Int
    dimension::Int
    number_of_heads::Int
    Q_projection::Lux.Dense
    K_projection::Lux.Dense
    V_projection::Lux.Dense
    OUTPUT_projection::Lux.Dense
end


function Lux.initialparameters(rng::Random.AbstractRNG, block::SWAttention)
    q_projection = Lux.initialparameters(rng, block.Q_projection)
    k_projection = Lux.initialparameters(rng, block.K_projection)
    v_projection = Lux.initialparameters(rng, block.V_projection)
    output_projection = Lux.initialparameters(rng, block.OUTPUT_projection)


    return (
        Q_projection = q_projection,
        K_projection = k_projection,
        V_projection = v_projection,
        OUTPUT_projection = output_projection,
    )
end

function SWAttention (sequence_length::Int, dimension::Int, number_of_heads::Int)
    @assert dimension % number_of_heads == 0 "dimension must be divisible by number_of_heads" # we want perfect division
    sub_dimension = div(layer.dimension, layer.number_of_heads)
    Q=Lux.Dense(dimension => sub_dimension) 
    K=Lux.Dense(dimension => sub_dimension) 
    V=Lux.Dense(dimension => sub_dimension) 
    OUT=Lux.Dense(dimension => dimension) 
    return SWAttention(sequence_length, dimension, number_of_heads, Q, K, V, OUT)
end


function Lux.initialstates(rng::Random.AbstractRNG, block::SWAttention)
    q_projection = Lux.initialparameters(block.dimension, block.Q_projection)
    k_projection = Lux.initialparameters(rng, block.K_projection)
    v_projection = Lux.initialparameters(rng, block.V_projection)
    output_projection = Lux.initialparameters(rng, block.OUTPUT_projection)


    return (
        Q_projection = q_projection,
        K_projection = k_projection,
        V_projection = v_projection,
        OUTPUT_projection = output_projection,
    )
end

@inline function normalized_sigmoids(seq; τ = 1.0, eps = 1e-12)
    sigmoids = NNlib.sigmoid.(seq ./ τ)
    s = reduce(+, sigmoids) + eps
    s = sum(sigmoids) + eps
    map!(x -> x / s, sigmoids, sigmoids)
    return sigmoids
end

function (block::SWAttention)(x, params::NamedTuple, _state::NamedTuple)
    (;Q_projection, K_projection, V_projection, OUTPUT_projection) = params

    Q, _ = Q_projection(x)
    K, _ = K_projection(x)
    V, _ = V_projection(x)
    d_k =   size(Q, 1)  
    # scores = (transpose(Q) * K)/sqrt(d_k)
    scores = (Q' * K)/ √(d_k)
    weights = similar(scores)
  
     for i in axes(scores,1) # 1:size(scores, 1) 
        weights[i,:] =  normalized_sigmoids(@view scores[i,:])
    end

    Y = V*weights

    output, _ = OUTPUT_projection(Y)
    return(output, _state)

end


end
