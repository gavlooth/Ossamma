#!/usr/bin/env julia
"""
Benchmark the economic relation-extraction changes:
1. True banded SWAttention versus the old dense masked implementation
2. Sampled relation negatives versus the old quadratic negative enumeration

Usage:
    julia --project=. scripts/benchmark_attention_rebel_economic.jl
"""

using Random
using Statistics
using Printf
using Lux
using NNlib
using CUDA

include(joinpath(@__DIR__, "..", "src", "Swamma.jl"))
using .Swamma
using .Swamma.Attention: SWAttention
using .Swamma.RelationExtraction: sample_negative_pairs!, build_token_vocab, build_entity_label_space, build_relation_label_space, prepare_rebel_batch

const RNG = MersenneTwister(42)

struct DenseMaskedSWAttention <: Lux.AbstractLuxLayer
    sequence_length::Int
    embedding_dimension::Int
    number_of_heads::Int
    window_size::Int
    head_dimension::Int
    QueryProjection::Lux.Dense
    KeyProjection::Lux.Dense
    ValueProjection::Lux.Dense
    OutputProjection::Lux.Dense
end

function DenseMaskedSWAttention(
    sequence_length::Int,
    embedding_dimension::Int,
    number_of_heads::Int;
    window_size::Int = 5,
)
    head_dimension = div(embedding_dimension, number_of_heads)
    DenseMaskedSWAttention(
        sequence_length,
        embedding_dimension,
        number_of_heads,
        window_size,
        head_dimension,
        Lux.Dense(embedding_dimension => embedding_dimension),
        Lux.Dense(embedding_dimension => embedding_dimension),
        Lux.Dense(embedding_dimension => embedding_dimension),
        Lux.Dense(embedding_dimension => embedding_dimension),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::DenseMaskedSWAttention)
    return (
        QueryProjection = Lux.initialparameters(rng, layer.QueryProjection),
        KeyProjection = Lux.initialparameters(rng, layer.KeyProjection),
        ValueProjection = Lux.initialparameters(rng, layer.ValueProjection),
        OutputProjection = Lux.initialparameters(rng, layer.OutputProjection),
    )
end

function build_sliding_window_mask(sequence_length::Int, window_size::Int)
    time_indices = collect(1:sequence_length)
    abs.(time_indices' .- time_indices) .> window_size
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::DenseMaskedSWAttention)
    return (
        QueryProjection = Lux.initialstates(rng, layer.QueryProjection),
        KeyProjection = Lux.initialstates(rng, layer.KeyProjection),
        ValueProjection = Lux.initialstates(rng, layer.ValueProjection),
        OutputProjection = Lux.initialstates(rng, layer.OutputProjection),
        window_mask = build_sliding_window_mask(layer.sequence_length, layer.window_size),
    )
end

@inline function sigsoftmax(logits; dims = 1)
    transformed_logits = logits .+ NNlib.logsigmoid.(logits)
    NNlib.softmax(transformed_logits; dims = dims)
end

@inline function apply_sliding_window_mask(attention_scores, window_mask)
    negative_infinity = typemin(eltype(attention_scores))
    ifelse.(window_mask, negative_infinity, attention_scores)
end

function (layer::DenseMaskedSWAttention)(input_tensor::AbstractArray, params, state)
    is_input_batched = ndims(input_tensor) == 3
    current_T = size(input_tensor, 2)
    active_mask = size(state.window_mask, 1) == current_T ?
        state.window_mask :
        build_sliding_window_mask(current_T, layer.window_size)

    input_3d_tensor = is_input_batched ? input_tensor : reshape(input_tensor, size(input_tensor, 1), size(input_tensor, 2), 1)
    feature_dimension, sequence_length, batch_size = size(input_3d_tensor)
    input_flattened = reshape(input_3d_tensor, feature_dimension, :)

    q_flat, q_st = layer.QueryProjection(input_flattened, params.QueryProjection, state.QueryProjection)
    k_flat, k_st = layer.KeyProjection(input_flattened, params.KeyProjection, state.KeyProjection)
    v_flat, v_st = layer.ValueProjection(input_flattened, params.ValueProjection, state.ValueProjection)

    query_tensor = reshape(q_flat, feature_dimension, sequence_length, batch_size)
    key_tensor = reshape(k_flat, feature_dimension, sequence_length, batch_size)
    value_tensor = reshape(v_flat, feature_dimension, sequence_length, batch_size)

    reshape_and_permute = x -> permutedims(
        reshape(x, layer.head_dimension, layer.number_of_heads, sequence_length, batch_size),
        (1, 3, 2, 4),
    )
    query_permuted = reshape_and_permute(query_tensor)
    key_permuted = reshape_and_permute(key_tensor)
    value_permuted = reshape_and_permute(value_tensor)

    key_transposed = permutedims(key_permuted, (2, 1, 3, 4))
    scaling_factor = sqrt(Float32(layer.head_dimension))
    attention_scores = NNlib.batched_mul(key_transposed, query_permuted) ./ scaling_factor
    masked_scores = apply_sliding_window_mask(attention_scores, active_mask)
    normalized_weights = sigsoftmax(masked_scores; dims = 1)
    weighted_values = NNlib.batched_mul(value_permuted, normalized_weights)

    weighted_values_permuted = permutedims(weighted_values, (1, 3, 2, 4))
    merged_heads = reshape(weighted_values_permuted, feature_dimension, sequence_length, batch_size)
    output_flat = reshape(merged_heads, feature_dimension, :)
    final_output_flat, o_st = layer.OutputProjection(output_flat, params.OutputProjection, state.OutputProjection)
    final_output_3d = reshape(final_output_flat, feature_dimension, sequence_length, batch_size)
    final_output = is_input_batched ? final_output_3d : dropdims(final_output_3d, dims = 3)

    return final_output, (
        QueryProjection = q_st,
        KeyProjection = k_st,
        ValueProjection = v_st,
        OutputProjection = o_st,
        window_mask = active_mask,
    )
end

function to_device(x, device::Symbol)
    if x isa NamedTuple
        return NamedTuple{keys(x)}(Tuple(to_device(v, device) for v in values(x)))
    elseif x isa Tuple
        return Tuple(to_device(v, device) for v in x)
    elseif x isa AbstractArray
        return device == :gpu ? CUDA.CuArray(x) : x
    else
        return x
    end
end

function synchronize_if_needed(device::Symbol)
    device == :gpu && CUDA.synchronize()
    return nothing
end

function benchmark_forward(model, params, state, x; iterations::Int, warmup::Int, device::Symbol)
    for _ in 1:warmup
        model(x, params, state)
        synchronize_if_needed(device)
    end

    times_ms = Float64[]
    for _ in 1:iterations
        start = time_ns()
        model(x, params, state)
        synchronize_if_needed(device)
        push!(times_ms, (time_ns() - start) / 1e6)
    end

    return (
        median_ms = median(times_ms),
        min_ms = minimum(times_ms),
        max_ms = maximum(times_ms),
    )
end

function quadratic_negative_enumeration(entity_count::Int, positive_pairs::Set{Tuple{Int, Int}}, target_negatives::Int)
    negative_candidates = Tuple{Int, Int}[]
    for head_idx in 1:entity_count
        for tail_idx in 1:entity_count
            head_idx == tail_idx && continue
            pair = (head_idx, tail_idx)
            pair in positive_pairs && continue
            push!(negative_candidates, pair)
        end
    end
    return collect(Iterators.take(negative_candidates, target_negatives))
end

function benchmark_negative_sampler(entity_count::Int; target_negatives::Int = 256, iterations::Int = 20)
    positive_pairs = Set{Tuple{Int, Int}}((i, i + 1) for i in 1:min(entity_count - 1, max(1, entity_count ÷ 8)))
    relation_pairs = zeros(Int, 2, target_negatives + length(positive_pairs))
    relation_labels = fill(-100, target_negatives + length(positive_pairs))
    relation_mask = falses(target_negatives + length(positive_pairs))
    relation_targets = zeros(Float32, target_negatives + length(positive_pairs))

    old_times = Float64[]
    new_times = Float64[]
    no_relation_id = 1

    for _ in 1:iterations
        start_old = time_ns()
        quadratic_negative_enumeration(entity_count, positive_pairs, target_negatives)
        push!(old_times, (time_ns() - start_old) / 1e6)

        fill!(relation_pairs, 0)
        fill!(relation_labels, -100)
        fill!(relation_mask, false)
        fill!(relation_targets, 0.0f0)

        start_new = time_ns()
        sample_negative_pairs!(
            relation_pairs,
            relation_labels,
            relation_mask,
            relation_targets,
            0,
            entity_count,
            positive_pairs,
            no_relation_id,
            target_negatives,
        )
        push!(new_times, (time_ns() - start_new) / 1e6)
    end

    return (
        old_median_ms = median(old_times),
        new_median_ms = median(new_times),
        speedup = median(old_times) / median(new_times),
    )
end

function benchmark_prepare_rebel_batch(entity_count::Int; iterations::Int = 10)
    tokens = ["t$(i)" for i in 1:max(entity_count + 4, 32)]
    entities = [(start = i, stop = i, label = isodd(i) ? "PERSON" : "ORGANIZATION") for i in 1:entity_count]
    relations = [(head = i, tail = i + 1, label = "LINK") for i in 1:min(entity_count - 1, max(1, entity_count ÷ 8))]
    rows = [(tokens = tokens, entities = entities, relations = relations)]

    vocab = build_token_vocab(rows; max_vocab = 4096)
    entity_label_to_id = build_entity_label_space(rows)
    relation_label_to_id = build_relation_label_space(rows)

    times_ms = Float64[]
    for _ in 1:iterations
        start = time_ns()
        prepare_rebel_batch(
            rows,
            vocab,
            entity_label_to_id,
            relation_label_to_id;
            max_len = length(tokens),
            max_candidate_spans = entity_count,
            max_candidate_pairs = min(entity_count * 4, 1024),
            max_span_width = 8,
            hard_negative_ratio = 2.0f0,
        )
        push!(times_ms, (time_ns() - start) / 1e6)
    end

    return median(times_ms)
end

function print_attention_table(device::Symbol)
    println()
    println("="^88)
    println("Attention Benchmark ($(uppercase(String(device))))")
    println("="^88)
    @printf("%8s %8s %14s %14s %12s\n", "seq_len", "window", "banded_ms", "dense_ms", "speedup")

    embedding_dimension = 512
    number_of_heads = 8
    batch_size = device == :gpu ? 8 : 4
    cases = [
        (64, 8),
        (128, 16),
        (256, 24),
        (512, 32),
    ]

    for (seq_len, window_size) in cases
        banded = SWAttention(seq_len, embedding_dimension, number_of_heads; window_size = window_size)
        dense = DenseMaskedSWAttention(seq_len, embedding_dimension, number_of_heads; window_size = window_size)
        banded_ps, banded_st = Lux.setup(RNG, banded)
        dense_ps, dense_st = Lux.setup(RNG, dense)
        x = randn(RNG, Float32, embedding_dimension, seq_len, batch_size)

        x = to_device(x, device)
        banded_ps = to_device(banded_ps, device)
        banded_st = to_device(banded_st, device)
        dense_ps = to_device(dense_ps, device)
        dense_st = to_device(dense_st, device)

        iterations = seq_len <= 128 ? 20 : seq_len <= 256 ? 12 : 6
        warmup = max(3, iterations ÷ 3)

        banded_stats = benchmark_forward(banded, banded_ps, banded_st, x; iterations = iterations, warmup = warmup, device = device)
        dense_stats = benchmark_forward(dense, dense_ps, dense_st, x; iterations = iterations, warmup = warmup, device = device)

        @printf(
            "%8d %8d %14.3f %14.3f %12.2fx\n",
            seq_len,
            window_size,
            banded_stats.median_ms,
            dense_stats.median_ms,
            dense_stats.median_ms / banded_stats.median_ms,
        )
    end
end

function print_gpu_batch_sweep()
    CUDA.functional() || return

    println()
    println("="^88)
    println("GPU Batch Sweep (Banded SWAttention)")
    println("="^88)
    @printf("%8s %8s %10s %14s %14s\n", "seq_len", "window", "batch", "median_ms", "tokens_per_s")

    embedding_dimension = 512
    number_of_heads = 8
    cases = [
        (256, 24),
        (512, 32),
    ]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    for (seq_len, window_size) in cases
        model = SWAttention(seq_len, embedding_dimension, number_of_heads; window_size = window_size)
        ps, st = Lux.setup(RNG, model)
        ps = to_device(ps, :gpu)
        st = to_device(st, :gpu)

        for batch_size in batch_sizes
            try
                x = CUDA.CuArray(randn(RNG, Float32, embedding_dimension, seq_len, batch_size))
                iterations = batch_size <= 8 ? 10 : 6
                warmup = 3
                stats = benchmark_forward(model, ps, st, x; iterations = iterations, warmup = warmup, device = :gpu)
                tokens_per_s = (seq_len * batch_size) / (stats.median_ms / 1e3)
                @printf(
                    "%8d %8d %10d %14.3f %14.0f\n",
                    seq_len,
                    window_size,
                    batch_size,
                    stats.median_ms,
                    tokens_per_s,
                )
            catch err
                if occursin("out of memory", lowercase(sprint(showerror, err)))
                    @printf("%8d %8d %10d %14s %14s\n", seq_len, window_size, batch_size, "OOM", "-")
                    break
                end
                rethrow(err)
            end
        end
    end
end

function print_negative_sampling_table()
    println()
    println("="^88)
    println("Negative Sampling Benchmark")
    println("="^88)
    @printf("%10s %16s %16s %12s %18s\n", "entities", "old_quad_ms", "new_sample_ms", "speedup", "prepare_batch_ms")

    for entity_count in (32, 64, 128, 256)
        negative_stats = benchmark_negative_sampler(entity_count; target_negatives = min(entity_count * 2, 512))
        prepare_ms = benchmark_prepare_rebel_batch(entity_count)
        @printf(
            "%10d %16.3f %16.3f %12.2fx %18.3f\n",
            entity_count,
            negative_stats.old_median_ms,
            negative_stats.new_median_ms,
            negative_stats.speedup,
            prepare_ms,
        )
    end
end

function main()
    println("CUDA functional: $(CUDA.functional())")
    if CUDA.functional()
        println("CUDA device: $(CUDA.name(CUDA.device()))")
    end

    print_attention_table(:cpu)
    if CUDA.functional()
        print_attention_table(:gpu)
        print_gpu_batch_sweep()
    end
    print_negative_sampling_table()
end

main()
