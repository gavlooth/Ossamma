module Attention

using Lux
using Random
using NNlib
using Zygote
import ChainRulesCore
using ChainRulesCore: NoTangent
import CUDA
using CUDA: @cuda, blockIdx, blockDim, threadIdx

const LuxAttentionSupertype = isdefined(Lux, :AbstractExplicitLayer) ?
                              Lux.AbstractExplicitLayer :
                              Lux.AbstractLuxLayer

struct SWAttention <: LuxAttentionSupertype
    sequence_length::Int
    embedding_dimension::Int
    number_of_heads::Int
    window_size::Int       # The sliding window radius
    head_dimension::Int    # Pre-calculated dimension per head

    # Layers for projections
    QueryProjection::Lux.Dense
    KeyProjection::Lux.Dense
    ValueProjection::Lux.Dense
    OutputProjection::Lux.Dense
end

@inline function sigsoftmax(logits; dims = 1)
    transformed_logits = logits .+ NNlib.logsigmoid.(logits)
    return NNlib.softmax(transformed_logits; dims = dims)
end

@inline function valid_band_range(sequence_length::Int, offset::Int)
    if offset >= 0
        length = sequence_length - offset
        return length <= 0 ? nothing : ((1:length), ((1 + offset):sequence_length))
    else
        length = sequence_length + offset
        return length <= 0 ? nothing : (((1 - offset):sequence_length), (1:length))
    end
end

function build_banded_index_map(sequence_length::Int, window_size::Int)
    offsets = reshape(collect(-window_size:window_size), :, 1)
    positions = reshape(collect(1:sequence_length), 1, :)
    source_positions = positions .+ offsets
    valid_mask = (source_positions .>= 1) .& (source_positions .<= sequence_length)
    source_indices = Int32.(clamp.(source_positions, 1, sequence_length))
    return source_indices, valid_mask
end

function band_index_map_for_input(state, sequence_length::Int, on_gpu::Bool)
    if size(state.window_indices, 2) == sequence_length
        return state.window_indices, state.window_valid
    end

    source_indices, valid_mask = build_banded_index_map(sequence_length, length(state.window_offsets) ÷ 2)
    if on_gpu
        return CUDA.CuArray(source_indices), CUDA.CuArray(valid_mask)
    end
    return source_indices, valid_mask
end

function banded_attention_weights_cpu(query_sequence, key_sequence, window_size::Int, head_dimension::Int)
    (_, sequence_length, batch_heads) = size(query_sequence)
    band_size = 2 * window_size + 1
    element_type = eltype(query_sequence)
    attention_scores = Zygote.Buffer(similar(query_sequence, element_type, band_size, sequence_length, batch_heads))
    negative_infinity = typemin(element_type)
    scale = sqrt(Float32(head_dimension))

    @views for bh in 1:batch_heads
        for target_idx in 1:sequence_length
            for band_idx in 1:band_size
                offset = band_idx - window_size - 1
                source_idx = target_idx + offset

                if 1 <= source_idx <= sequence_length
                    acc = zero(element_type)
                    for d in 1:head_dimension
                        acc += key_sequence[d, source_idx, bh] * query_sequence[d, target_idx, bh]
                    end
                    attention_scores[band_idx, target_idx, bh] = acc / scale
                else
                    attention_scores[band_idx, target_idx, bh] = negative_infinity
                end
            end
        end
    end

    return sigsoftmax(copy(attention_scores); dims = 1)
end

function apply_banded_attention_cpu(value_sequence, attention_weights, window_size::Int)
    (head_dimension, sequence_length, batch_heads) = size(value_sequence)
    output = Zygote.Buffer(similar(value_sequence, eltype(value_sequence), head_dimension, sequence_length, batch_heads))

    @views for bh in 1:batch_heads
        for target_idx in 1:sequence_length
            for d in 1:head_dimension
                acc = zero(eltype(value_sequence))
                for offset in (-window_size):window_size
                    source_idx = target_idx + offset
                    if 1 <= source_idx <= sequence_length
                        band_idx = offset + window_size + 1
                        weight = attention_weights[band_idx, target_idx, bh]
                        acc += value_sequence[d, source_idx, bh] * weight
                    end
                end
                output[d, target_idx, bh] = acc
            end
        end
    end

    return copy(output)
end

function banded_score_kernel!(scores, query_sequence, key_sequence, window_size::Int32, scale::Float32)
    idx = (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x) + Int32(threadIdx().x)
    band_size = Int32(size(scores, 1))
    sequence_length = Int32(size(scores, 2))
    batch_heads = Int32(size(scores, 3))
    total = band_size * sequence_length * batch_heads
    idx > total && return

    band_idx = Int32(mod(idx - Int32(1), band_size) + Int32(1))
    tmp = Int32(div(idx - Int32(1), band_size))
    target_idx = Int32(mod(tmp, sequence_length) + Int32(1))
    batch_head_idx = Int32(div(tmp, sequence_length) + Int32(1))

    offset = band_idx - window_size - Int32(1)
    source_idx = target_idx + offset

    if source_idx >= Int32(1) && source_idx <= sequence_length
        acc = zero(eltype(scores))
        @inbounds for dim_idx in 1:Int32(size(query_sequence, 1))
            acc += key_sequence[dim_idx, source_idx, batch_head_idx] * query_sequence[dim_idx, target_idx, batch_head_idx]
        end
        scores[band_idx, target_idx, batch_head_idx] = acc / scale
    else
        scores[band_idx, target_idx, batch_head_idx] = typemin(eltype(scores))
    end
    return
end

function banded_value_kernel!(output, value_sequence, attention_weights, window_size::Int32)
    idx = (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x) + Int32(threadIdx().x)
    head_dimension = Int32(size(output, 1))
    sequence_length = Int32(size(output, 2))
    batch_heads = Int32(size(output, 3))
    band_size = Int32(size(attention_weights, 1))
    total = head_dimension * sequence_length * batch_heads
    idx > total && return

    dim_idx = Int32(mod(idx - Int32(1), head_dimension) + Int32(1))
    tmp = Int32(div(idx - Int32(1), head_dimension))
    target_idx = Int32(mod(tmp, sequence_length) + Int32(1))
    batch_head_idx = Int32(div(tmp, sequence_length) + Int32(1))

    acc = zero(eltype(output))
    @inbounds for band_idx in 1:band_size
        source_idx = target_idx + band_idx - window_size - Int32(1)
        if source_idx >= Int32(1) && source_idx <= sequence_length
            acc += value_sequence[dim_idx, source_idx, batch_head_idx] * attention_weights[band_idx, target_idx, batch_head_idx]
        end
    end
    output[dim_idx, target_idx, batch_head_idx] = acc
    return
end

function banded_attention_scores_gpu(query_sequence, key_sequence, window_size::Int, head_dimension::Int)
    band_size = 2 * window_size + 1
    sequence_length = size(query_sequence, 2)
    batch_heads = size(query_sequence, 3)
    scores = similar(query_sequence, eltype(query_sequence), band_size, sequence_length, batch_heads)
    threads = 256
    blocks = cld(length(scores), threads)
    @cuda threads = threads blocks = blocks banded_score_kernel!(
        scores,
        query_sequence,
        key_sequence,
        Int32(window_size),
        sqrt(Float32(head_dimension)),
    )
    return scores
end

function banded_attention_weights_gpu(query_sequence, key_sequence, window_size::Int, head_dimension::Int)
    scores = banded_attention_scores_gpu(query_sequence, key_sequence, window_size, head_dimension)
    return sigsoftmax(scores; dims = 1)
end

function apply_banded_attention_gpu(value_sequence, attention_weights, window_size::Int)
    output = similar(value_sequence)
    threads = 256
    blocks = cld(length(output), threads)
    @cuda threads = threads blocks = blocks banded_value_kernel!(
        output,
        value_sequence,
        attention_weights,
        Int32(window_size),
    )
    return output
end

function banded_attention_gpu(query_sequence, key_sequence, value_sequence, window_size::Int, head_dimension::Int)
    attention_weights = banded_attention_weights_gpu(
        query_sequence,
        key_sequence,
        window_size,
        head_dimension,
    )
    return apply_banded_attention_gpu(value_sequence, attention_weights, window_size)
end

function banded_attention_pullback_gpu(
    query_sequence,
    key_sequence,
    value_sequence,
    scores,
    attention_weights,
    output_bar,
    window_size::Int,
    head_dimension::Int,
)
    (_, sequence_length, batch_heads) = size(query_sequence)
    scale = convert(eltype(query_sequence), sqrt(Float32(head_dimension)))

    query_bar = zero(eltype(query_sequence)) .* query_sequence
    key_bar = zero(eltype(key_sequence)) .* key_sequence
    value_bar = zero(eltype(value_sequence)) .* value_sequence
    weight_bar = zero(eltype(attention_weights)) .* attention_weights

    @views for offset in (-window_size):window_size
        ranges = valid_band_range(sequence_length, offset)
        ranges === nothing && continue
        target_range, source_range = ranges
        band_idx = offset + window_size + 1
        weights_slice = reshape(
            attention_weights[band_idx, target_range, :],
            1,
            length(target_range),
            batch_heads,
        )
        value_bar[:, source_range, :] .+= output_bar[:, target_range, :] .* weights_slice
        weight_grad_slice = sum(
            output_bar[:, target_range, :] .* value_sequence[:, source_range, :],
            dims = 1,
        )
        weight_bar[band_idx, target_range, :] .= reshape(
            weight_grad_slice,
            length(target_range),
            batch_heads,
        )
    end

    transformed_grad = 1 .+ NNlib.sigmoid.(-scores)
    softmax_dot = sum(weight_bar .* attention_weights, dims = 1)
    score_bar = attention_weights .* (weight_bar .- softmax_dot) .* transformed_grad

    @views for offset in (-window_size):window_size
        ranges = valid_band_range(sequence_length, offset)
        ranges === nothing && continue
        target_range, source_range = ranges
        band_idx = offset + window_size + 1
        score_slice = reshape(
            score_bar[band_idx, target_range, :],
            1,
            length(target_range),
            batch_heads,
        )
        query_bar[:, target_range, :] .+= key_sequence[:, source_range, :] .* score_slice ./ scale
        key_bar[:, source_range, :] .+= query_sequence[:, target_range, :] .* score_slice ./ scale
    end

    return query_bar, key_bar, value_bar
end

function ChainRulesCore.rrule(
    ::typeof(banded_attention_gpu),
    query_sequence,
    key_sequence,
    value_sequence,
    window_size::Int,
    head_dimension::Int,
)
    scores = banded_attention_scores_gpu(
        query_sequence,
        key_sequence,
        window_size,
        head_dimension,
    )
    attention_weights = sigsoftmax(scores; dims = 1)
    output = apply_banded_attention_gpu(value_sequence, attention_weights, window_size)

    function banded_attention_gpu_pullback(output_bar)
        query_bar, key_bar, value_bar = banded_attention_pullback_gpu(
            query_sequence,
            key_sequence,
            value_sequence,
            scores,
            attention_weights,
            output_bar,
            window_size,
            head_dimension,
        )
        return (
            NoTangent(),
            query_bar,
            key_bar,
            value_bar,
            NoTangent(),
            NoTangent(),
        )
    end

    return output, banded_attention_gpu_pullback
end

# 1. CONSTRUCTOR
function SWAttention(
    sequence_length::Int,
    embedding_dimension::Int,
    number_of_heads::Int;
    window_size::Int = 5,
)
    @assert embedding_dimension % number_of_heads == 0 "Embedding dimension must be divisible by the number of heads."

    calculated_head_dimension = div(embedding_dimension, number_of_heads)

    return SWAttention(
        sequence_length,
        embedding_dimension,
        number_of_heads,
        window_size,
        calculated_head_dimension,
        Lux.Dense(embedding_dimension => embedding_dimension), # Query
        Lux.Dense(embedding_dimension => embedding_dimension), # Key
        Lux.Dense(embedding_dimension => embedding_dimension), # Value
        Lux.Dense(embedding_dimension => embedding_dimension),  # Output
    )
end

# 2. PARAMETER INITIALIZATION
# Explicitly define this to ensure params are structured correctly and accessible by field name
function Lux.initialparameters(rng::Random.AbstractRNG, layer::SWAttention)
    return (
        QueryProjection = Lux.initialparameters(rng, layer.QueryProjection),
        KeyProjection = Lux.initialparameters(rng, layer.KeyProjection),
        ValueProjection = Lux.initialparameters(rng, layer.ValueProjection),
        OutputProjection = Lux.initialparameters(rng, layer.OutputProjection),
    )
end

# 3. STATE INITIALIZATION
# We need a custom initialstates to include both child states and our mask
function Lux.initialstates(rng::Random.AbstractRNG, layer::SWAttention)
    # recursively initialize child states
    child_states = (
        QueryProjection = Lux.initialstates(rng, layer.QueryProjection),
        KeyProjection = Lux.initialstates(rng, layer.KeyProjection),
        ValueProjection = Lux.initialstates(rng, layer.ValueProjection),
        OutputProjection = Lux.initialstates(rng, layer.OutputProjection),
    )

    window_offsets = collect(-layer.window_size:layer.window_size)
    window_indices, window_valid = build_banded_index_map(layer.sequence_length, layer.window_size)
    return merge(child_states, (; window_offsets, window_indices, window_valid))
end

# 4. FORWARD PASS
function (layer::SWAttention)(input_tensor::AbstractArray, params, state)
    # ---------------------------------------------------------
    # A. Handle Dimensions & Masking
    # ---------------------------------------------------------
    # Expected Input: (Features, Time) or (Features, Time, Batch)
    is_input_batched = ndims(input_tensor) == 3
    
    # Standardize to 3D Tensor: (Features, Time, Batch)
    input_3d_tensor =
        is_input_batched ? input_tensor :
        reshape(input_tensor, size(input_tensor, 1), size(input_tensor, 2), 1)

    (feature_dimension, sequence_length, batch_size) = size(input_3d_tensor)
    on_gpu = input_3d_tensor isa CUDA.CuArray

    # ---------------------------------------------------------
    # B. Projections (Q, K, V)
    # ---------------------------------------------------------
    # Reshape to 2D (Features, Time * Batch) for efficient Dense Layer processing
    input_flattened_for_projection = reshape(input_3d_tensor, feature_dimension, :)

    # Apply Dense Layers with correct params/state
    # Params are automatically structured by Lux as (QueryProjection=..., etc.)
    # State is structured by us in initialstates
    
    q_flat, q_st = layer.QueryProjection(input_flattened_for_projection, params.QueryProjection, state.QueryProjection)
    k_flat, k_st = layer.KeyProjection(input_flattened_for_projection, params.KeyProjection, state.KeyProjection)
    v_flat, v_st = layer.ValueProjection(input_flattened_for_projection, params.ValueProjection, state.ValueProjection)

    # Reshape back to 3D: (Features, Time, Batch)
    query_tensor = reshape(q_flat, feature_dimension, sequence_length, batch_size)
    key_tensor   = reshape(k_flat, feature_dimension, sequence_length, batch_size)
    value_tensor = reshape(v_flat, feature_dimension, sequence_length, batch_size)

    # ---------------------------------------------------------
    # C. Multi-Head Splitting
    # ---------------------------------------------------------
    # Target Shape: (Head_Dim, Heads, Time, Batch) -> (Head_Dim, Time, Heads, Batch)
    
    reshape_and_permute = x -> begin
        x_r = reshape(x, layer.head_dimension, layer.number_of_heads, sequence_length, batch_size)
        permutedims(x_r, (1, 3, 2, 4))
    end

    query_permuted = reshape_and_permute(query_tensor)
    key_permuted   = reshape_and_permute(key_tensor)
    value_permuted = reshape_and_permute(value_tensor)

    # ---------------------------------------------------------
    # D. True Banded Attention
    # ---------------------------------------------------------
    merge_batch_dimensions(tensor) = reshape(tensor, layer.head_dimension, sequence_length, :)
    query_sequence = merge_batch_dimensions(query_permuted)
    key_sequence = merge_batch_dimensions(key_permuted)
    value_sequence = merge_batch_dimensions(value_permuted)
    weighted_values = if on_gpu
        banded_attention_gpu(
            query_sequence,
            key_sequence,
            value_sequence,
            layer.window_size,
            layer.head_dimension,
        )
    else
        normalized_attention_weights = banded_attention_weights_cpu(
            query_sequence,
            key_sequence,
            layer.window_size,
            layer.head_dimension,
        )
        apply_banded_attention_cpu(
            value_sequence,
            normalized_attention_weights,
            layer.window_size,
        )
    end

    # ---------------------------------------------------------
    # H. Output Projection
    # ---------------------------------------------------------
    # Permute back: (Head_Dim, Heads, Time, Batch)
    weighted_values_permuted = permutedims(
        reshape(weighted_values, layer.head_dimension, sequence_length, layer.number_of_heads, batch_size),
        (1, 3, 2, 4),
    )

    # Merge Heads: (Feature_Dim, Time, Batch)
    output_merged_heads =
        reshape(weighted_values_permuted, feature_dimension, sequence_length, batch_size)

    # Flatten for Dense Layer
    output_flattened_for_projection = reshape(output_merged_heads, feature_dimension, :)

    final_output_flat, o_st = layer.OutputProjection(
        output_flattened_for_projection,
        params.OutputProjection,
        state.OutputProjection
    )

    # Restore 3D Shape
    final_output_3d =
        reshape(final_output_flat, feature_dimension, sequence_length, batch_size)

    # Handle Batch Dimension
    final_output = is_input_batched ? final_output_3d : dropdims(final_output_3d, dims = 3)
    
    # Update State
    new_state = (
        QueryProjection = q_st,
        KeyProjection = k_st,
        ValueProjection = v_st,
        OutputProjection = o_st,
        window_offsets = state.window_offsets,
        window_indices = state.window_indices,
        window_valid = state.window_valid,
    )

    return final_output, new_state
end

end # module
