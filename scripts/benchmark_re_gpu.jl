#!/usr/bin/env julia
"""
End-to-end GPU benchmark for Swamma relation extraction.

Measures:
- Inference throughput with provided spans/pairs on the GPU-safe path
- Training-step throughput with gold spans/pairs and backward pass
- GPU memory footprint across batch sizes / sequence lengths

Usage:
    julia --project=. scripts/benchmark_re_gpu.jl
    julia --project=. scripts/benchmark_re_gpu.jl configs/redfm_base.toml
"""

using Random
using Statistics
using Printf
using Lux
using CUDA
using Optimisers
using Zygote

include(joinpath(@__DIR__, "..", "src", "Swamma.jl"))
using .Swamma

const RNG = MersenneTwister(1234)

Base.@kwdef struct BenchmarkOptions
    seq_lengths::Vector{Int}
    batch_sizes::Vector{Int}
    inference_warmup::Int = 2
    inference_iterations::Int = 5
    train_warmup::Int = 1
    train_iterations::Int = 3
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

function gpu_memory_gb()
    return CUDA.used_memory() / 1e9, CUDA.total_memory() / 1e9
end

function synthetic_re_row(seq_len::Int, entity_count::Int, relation_count::Int)
    tokens = ["tok$(i)" for i in 1:seq_len]
    labels = ("PERSON", "ORGANIZATION", "LOCATION", "EVENT")

    entity_positions = sort(unique(round.(Int, range(1, seq_len - 2; length = entity_count))))
    entities = NamedTuple[]
    for (i, pos) in enumerate(entity_positions)
        stop = min(seq_len, pos + (i % 2))
        push!(entities, (start = pos, stop = stop, label = labels[mod1(i, length(labels))]))
    end

    relations = NamedTuple[]
    relation_total = min(relation_count, max(length(entities) - 1, 0))
    for i in 1:relation_total
        push!(relations, (head = i, tail = i + 1, label = i % 2 == 0 ? "RELATED_TO" : "WORKS_FOR"))
    end

    return (tokens = tokens, entities = entities, relations = relations)
end

function build_synthetic_batch(config::RelationExtractionConfig, batch_size::Int, seq_len::Int)
    entity_budget = min(config.max_candidate_spans, max(8, seq_len ÷ 16))
    relation_budget = min(config.max_candidate_pairs, max(4, entity_budget ÷ 2))
    rows = [
        synthetic_re_row(seq_len, entity_budget, relation_budget)
        for _ in 1:batch_size
    ]

    vocab = build_token_vocab(rows; max_vocab = max(config.vocab_size, 4096))
    entity_label_to_id = build_entity_label_space(rows)
    relation_label_to_id = build_relation_label_space(rows)

    batch = prepare_rebel_batch(
        rows,
        vocab,
        entity_label_to_id,
        relation_label_to_id;
        max_len = seq_len,
        max_candidate_spans = config.max_candidate_spans,
        max_candidate_pairs = config.max_candidate_pairs,
        max_span_width = config.max_span_width,
        hard_negative_ratio = 2.0f0,
    )

    span_scores = Float32.(batch.span_mask)

    inference_inputs = (
        token_ids = batch.token_ids,
        spans = batch.spans,
        span_mask = batch.span_mask,
        span_scores = span_scores,
        mention_spans = batch.mention_spans,
        mention_mask = batch.mention_mask,
        relation_pairs = batch.relation_pairs,
        relation_mask = batch.relation_mask,
    )

    training_inputs = (
        token_ids = batch.token_ids,
        spans = batch.spans,
        span_mask = batch.span_mask,
        span_scores = span_scores,
        mention_spans = batch.mention_spans,
        mention_mask = batch.mention_mask,
        relation_pairs = batch.relation_pairs,
        relation_mask = batch.relation_mask,
    )

    targets = (
        entity_labels = batch.entity_labels,
        boundary_labels = batch.boundary_labels,
        mention_labels = batch.mention_labels,
        mention_mask = batch.mention_mask,
        relation_labels = batch.relation_labels,
        relation_mask = batch.relation_mask,
        relation_targets = batch.relation_targets,
    )

    return inference_inputs, training_inputs, targets
end

function benchmark_inference(model, params, state, inputs; warmup::Int = 5, iterations::Int = 20)
    for _ in 1:warmup
        model(inputs, params, state)
        CUDA.synchronize()
    end

    times_ms = Float64[]
    for _ in 1:iterations
        start = time_ns()
        model(inputs, params, state)
        CUDA.synchronize()
        push!(times_ms, (time_ns() - start) / 1e6)
    end

    return median(times_ms)
end

function relation_loss(outputs, targets)
    loss_entity = entity_cross_entropy(outputs.entity_logits, targets.entity_labels)
    loss_boundary = boundary_bce(outputs.boundary_logits, targets.boundary_labels)
    loss_mention = Swamma.RelationExtraction.mention_bce(
        outputs.mention_logits,
        targets.mention_labels,
        targets.mention_mask,
    )
    loss_relation = relation_cross_entropy(outputs.relation_logits, targets.relation_labels, targets.relation_mask)
    loss_confidence = confidence_bce(outputs.confidence_logits, targets.relation_targets, targets.relation_mask)
    return loss_entity + loss_boundary + loss_mention + loss_relation + loss_confidence
end

function benchmark_train_step(model, params, state, training_inputs, targets; warmup::Int = 3, iterations::Int = 10)
    opt = Optimisers.AdamW(2.0f-4, (0.9f0, 0.999f0), 0.01f0)
    opt_state = Optimisers.setup(opt, params)
    local_params = params
    local_state = state

    for _ in 1:warmup
        loss, grads = Zygote.withgradient(local_params) do p
            outputs, _ = model(training_inputs, p, local_state)
            relation_loss(outputs, targets)
        end
        opt_state, local_params = Optimisers.update(opt_state, local_params, grads[1])
        # Refresh the mutable Lux state outside autodiff to avoid the CUDA
        # device exception triggered by `withgradient(...; aux_state)` on this model.
        _, local_state = model(training_inputs, local_params, local_state)
        CUDA.synchronize()
    end

    times_ms = Float64[]
    losses = Float64[]
    for _ in 1:iterations
        start = time_ns()
        loss, grads = Zygote.withgradient(local_params) do p
            outputs, _ = model(training_inputs, p, local_state)
            relation_loss(outputs, targets)
        end
        opt_state, local_params = Optimisers.update(opt_state, local_params, grads[1])
        _, local_state = model(training_inputs, local_params, local_state)
        CUDA.synchronize()
        push!(times_ms, (time_ns() - start) / 1e6)
        push!(losses, Float64(loss))
    end

    return median(times_ms), mean(losses)
end

function format_mem()
    used, total = gpu_memory_gb()
    return @sprintf("%.1f/%.1fGB", used, total)
end

function benchmark_case(model, params, state, config::RelationExtractionConfig, seq_len::Int, batch_size::Int, opts::BenchmarkOptions)
    inference_inputs_cpu, training_inputs_cpu, targets_cpu = build_synthetic_batch(config, batch_size, seq_len)
    inference_inputs = to_device(inference_inputs_cpu, :gpu)
    training_inputs = to_device(training_inputs_cpu, :gpu)
    targets = to_device(targets_cpu, :gpu)
    inference_state = Lux.testmode(state)

    CUDA.reclaim()
    inference_ms = benchmark_inference(
        model,
        params,
        inference_state,
        inference_inputs;
        warmup = opts.inference_warmup,
        iterations = opts.inference_iterations,
    )
    inference_mem = format_mem()

    train_ms = nothing
    train_tokens_per_s = nothing
    train_mem = "-"
    loss = nothing
    train_error = nothing
    try
        CUDA.reclaim()
        train_ms, loss = benchmark_train_step(
            model,
            params,
            state,
            training_inputs,
            targets;
            warmup = opts.train_warmup,
            iterations = opts.train_iterations,
        )
        train_mem = format_mem()
        train_tokens_per_s = (seq_len * batch_size) / (train_ms / 1e3)
    catch err
        train_error = sprint(showerror, err)
    end

    tokens = seq_len * batch_size
    return (
        inference_ms = inference_ms,
        inference_tokens_per_s = tokens / (inference_ms / 1e3),
        inference_mem = inference_mem,
        train_ms = train_ms,
        train_tokens_per_s = train_tokens_per_s,
        train_mem = train_mem,
        loss = loss,
        train_error = train_error,
    )
end

parse_list_arg(value::AbstractString) = parse.(Int, split(String(value), ","))

function parse_options(config::RelationExtractionConfig, args::Vector{String})
    full = "--full" in args
    quick = "--quick" in args || !full

    seq_lengths = [config.max_sequence_length]
    batch_sizes = quick ? [8, 16] : [1, 2, 4, 8, 16, 32, 64]
    inference_warmup = quick ? 1 : 5
    inference_iterations = quick ? 3 : 20
    train_warmup = quick ? 1 : 3
    train_iterations = quick ? 2 : 10

    for arg in args
        startswith(arg, "--batches=") && (batch_sizes = parse_list_arg(split(arg, "=", limit = 2)[2]))
        startswith(arg, "--seq-lengths=") && (seq_lengths = parse_list_arg(split(arg, "=", limit = 2)[2]))
        startswith(arg, "--infer-warmup=") && (inference_warmup = parse(Int, split(arg, "=", limit = 2)[2]))
        startswith(arg, "--infer-iters=") && (inference_iterations = parse(Int, split(arg, "=", limit = 2)[2]))
        startswith(arg, "--train-warmup=") && (train_warmup = parse(Int, split(arg, "=", limit = 2)[2]))
        startswith(arg, "--train-iters=") && (train_iterations = parse(Int, split(arg, "=", limit = 2)[2]))
    end

    return BenchmarkOptions(
        seq_lengths = seq_lengths,
        batch_sizes = batch_sizes,
        inference_warmup = inference_warmup,
        inference_iterations = inference_iterations,
        train_warmup = train_warmup,
        train_iterations = train_iterations,
    )
end

function main()
    positional_args = [arg for arg in ARGS if !startswith(arg, "--")]
    config_path = isempty(positional_args) ? "configs/redfm_base.toml" : first(positional_args)
    CUDA.functional() || error("CUDA is not functional on this machine.")

    println("CUDA device: $(CUDA.name(CUDA.device()))")
    println("Config: $config_path")

    config = load_relation_extraction_config(config_path)
    opts = parse_options(config, ARGS)
    model = SwammaRelationExtractor(config)
    params, state = Lux.setup(RNG, model)
    params = to_device(params, :gpu)
    state = to_device(state, :gpu)

    println()
    println("="^120)
    println("End-to-End RE GPU Benchmark")
    println("="^120)
    @printf(
        "%8s %8s %14s %14s %16s %14s %14s %16s %10s\n",
        "seq_len",
        "batch",
        "infer_ms",
        "infer_tok/s",
        "infer_mem",
        "train_ms",
        "train_tok/s",
        "train_mem",
        "loss",
    )
    flush(stdout)

    for seq_len in opts.seq_lengths
        for batch_size in opts.batch_sizes
            try
                @printf("Running seq_len=%d batch=%d ...\n", seq_len, batch_size)
                flush(stdout)
                stats = benchmark_case(model, params, state, config, seq_len, batch_size, opts)
                if isnothing(stats.train_ms)
                    @printf(
                        "%8d %8d %14.3f %14.0f %16s %14s %14s %16s %10s\n",
                        seq_len,
                        batch_size,
                        stats.inference_ms,
                        stats.inference_tokens_per_s,
                        stats.inference_mem,
                        "AD_UNSUP",
                        "-",
                        stats.train_mem,
                        "-",
                    )
                    println("  train_error: $(stats.train_error)")
                else
                    @printf(
                        "%8d %8d %14.3f %14.0f %16s %14.3f %14.0f %16s %10.3f\n",
                        seq_len,
                        batch_size,
                        stats.inference_ms,
                        stats.inference_tokens_per_s,
                        stats.inference_mem,
                        stats.train_ms,
                        stats.train_tokens_per_s,
                        stats.train_mem,
                        stats.loss,
                    )
                end
                flush(stdout)
            catch err
                msg = lowercase(sprint(showerror, err))
                if occursin("out of memory", msg) || occursin("oom", msg)
                    @printf(
                        "%8d %8d %14s %14s %16s %14s %14s %16s %10s\n",
                        seq_len,
                        batch_size,
                        "OOM",
                        "-",
                        "-",
                        "OOM",
                        "-",
                        "-",
                        "-",
                    )
                    CUDA.reclaim()
                    flush(stdout)
                    break
                end
                rethrow(err)
            end
        end
    end
end

main()
