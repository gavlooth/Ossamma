module Attention

import Lux
import Random
import NNlib

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

@inline function build_sliding_window_mask(sequence_length::Int, window_size::Int)
    time_indices = collect(1:sequence_length)
    distance_matrix = abs.(time_indices' .- time_indices)
    return distance_matrix .> window_size
end

@inline function apply_sliding_window_mask(attention_scores, window_mask)
    negative_infinity = typemin(eltype(attention_scores))
    return ifelse.(window_mask, negative_infinity, attention_scores)
end

@inline function sigsoftmax(logits; dims = 1)
    transformed_logits = logits .+ NNlib.logsigmoid.(logits)
    return NNlib.softmax(transformed_logits; dims = dims)
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
function Lux.initialparameters(
    random_number_generator::Random.AbstractRNG,
    layer::SWAttention,
)
    return (
        query_projection_params = Lux.initialparameters(
            random_number_generator,
            layer.QueryProjection,
        ),
        key_projection_params = Lux.initialparameters(
            random_number_generator,
            layer.KeyProjection,
        ),
        value_projection_params = Lux.initialparameters(
            random_number_generator,
            layer.ValueProjection,
        ),
        output_projection_params = Lux.initialparameters(
            random_number_generator,
            layer.OutputProjection,
        ),
    )
end

# 3. STATE INITIALIZATION (Stateless layer)
function Lux.initialstates(_rng::Random.AbstractRNG, layer::SWAttention)
    mask = build_sliding_window_mask(layer.sequence_length, layer.window_size)
    return (; window_mask = mask)
end

# 4. FORWARD PASS
function (layer::SWAttention)(input_tensor::AbstractArray, params, state)
    window_mask = state.window_mask

    # ---------------------------------------------------------
    # A. Handle Dimensions
    # ---------------------------------------------------------
    # Expected Input: (Features, Time) or (Features, Time, Batch)
    is_input_batched = ndims(input_tensor) == 3

    # Standardize to 3D Tensor: (Features, Time, Batch)
    input_3d_tensor =
        is_input_batched ? input_tensor :
        reshape(input_tensor, size(input_tensor, 1), size(input_tensor, 2), 1)

    (feature_dimension, sequence_length, batch_size) = size(input_3d_tensor)

    # ---------------------------------------------------------
    # B. Projections (Q, K, V)
    # ---------------------------------------------------------
    # Reshape to 2D (Features, Time * Batch) for efficient Dense Layer processing
    input_flattened_for_projection = reshape(input_3d_tensor, feature_dimension, :)

    # Apply Dense Layers
    # Note: Lux Dense layers return a tuple (output, state), we only need the output [1]
    query_projected_flat = layer.QueryProjection(
        input_flattened_for_projection,
        params.query_projection_params,
        state,
    )[1]
    key_projected_flat = layer.KeyProjection(
        input_flattened_for_projection,
        params.key_projection_params,
        state,
    )[1]
    value_projected_flat = layer.ValueProjection(
        input_flattened_for_projection,
        params.value_projection_params,
        state,
    )[1]

    # Reshape back to 3D: (Features, Time, Batch)
    query_tensor =
        reshape(query_projected_flat, feature_dimension, sequence_length, batch_size)
    key_tensor = reshape(key_projected_flat, feature_dimension, sequence_length, batch_size)
    value_tensor =
        reshape(value_projected_flat, feature_dimension, sequence_length, batch_size)

    # ---------------------------------------------------------
    # C. Multi-Head Splitting
    # ---------------------------------------------------------
    # Current Shape: (Head_Dim * Heads, Time, Batch)
    # Target Shape:  (Head_Dim, Heads, Time, Batch)

    query_reshaped = reshape(
        query_tensor,
        layer.head_dimension,
        layer.number_of_heads,
        sequence_length,
        batch_size,
    )
    key_reshaped = reshape(
        key_tensor,
        layer.head_dimension,
        layer.number_of_heads,
        sequence_length,
        batch_size,
    )
    value_reshaped = reshape(
        value_tensor,
        layer.head_dimension,
        layer.number_of_heads,
        sequence_length,
        batch_size,
    )

    # Permute for Batched Multiplication: (Head_Dim, Time, Heads, Batch)
    # We want Time and Head_Dim to interact in the matrix multiplication
    query_permuted = permutedims(query_reshaped, (1, 3, 2, 4))
    key_permuted = permutedims(key_reshaped, (1, 3, 2, 4))
    value_permuted = permutedims(value_reshaped, (1, 3, 2, 4))

    # ---------------------------------------------------------
    # D. Attention Score Calculation
    # ---------------------------------------------------------
    # Operation: K^T * Q
    # Key Transposed: (Time, Head_Dim, Heads, Batch)
    # Query:          (Head_Dim, Time, Heads, Batch)
    # Result:         (Time, Time, Heads, Batch)

    key_transposed_for_score = permutedims(key_permuted, (2, 1, 3, 4))

    # Compute raw scores scaled by sqrt(d_k)
    scaling_factor = sqrt(Float32(layer.head_dimension))
    attention_scores_raw =
        NNlib.batched_mul(key_transposed_for_score, query_permuted) ./ scaling_factor

    # ---------------------------------------------------------
    # E. Sliding Window Masking
    # ---------------------------------------------------------
    masked_attention_scores =
        apply_sliding_window_mask(attention_scores_raw, window_mask)
    # ---------------------------------------------------------
    # F. Normalization (SigSoftmax Attention)
    # ---------------------------------------------------------
    normalized_attention_weights = sigsoftmax(masked_attention_scores; dims = 1)

    # ---------------------------------------------------------
    # G. Weighted Aggregation
    # ---------------------------------------------------------
    # Operation: Value * Weights
    # Value:   (Head_Dim, Time, Heads, Batch)
    # Weights: (Time, Time, Heads, Batch)
    # Result:  (Head_Dim, Time, Heads, Batch)

    weighted_values = NNlib.batched_mul(value_permuted, normalized_attention_weights)

    # ---------------------------------------------------------
    # H. Output Projection
    # ---------------------------------------------------------
    # Permute back: (Head_Dim, Heads, Time, Batch)
    weighted_values_permuted = permutedims(weighted_values, (1, 3, 2, 4))

    # Merge Heads: (Head_Dim * Heads, Time, Batch) -> (Feature_Dim, Time, Batch)
    output_merged_heads =
        reshape(weighted_values_permuted, feature_dimension, sequence_length, batch_size)

    # Flatten for Dense Layer: (Feature_Dim, Time * Batch)
    output_flattened_for_projection = reshape(output_merged_heads, feature_dimension, :)

    # Apply Output Projection
    final_output_flat = layer.OutputProjection(
        output_flattened_for_projection,
        params.output_projection_params,
        state,
    )[1]

    # Restore 3D Shape
    final_output_3d =
        reshape(final_output_flat, feature_dimension, sequence_length, batch_size)

    # Handle Batch Dimension (drop if input wasn't batched)
    final_output = is_input_batched ? final_output_3d : dropdims(final_output_3d, dims = 3)

    return final_output, state
end

end # module
