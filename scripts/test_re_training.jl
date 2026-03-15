#!/usr/bin/env julia
"""
Quick GPU smoke test for Swamma relation-extraction training.

Usage:
    julia --project=. scripts/test_re_training.jl
"""

using Random
using Statistics
using Printf
using CUDA
using Lux
using Optimisers
using Zygote

include(joinpath(@__DIR__, "..", "src", "Swamma.jl"))
using .Swamma

function to_device(x)
    if x isa NamedTuple
        return NamedTuple{keys(x)}(Tuple(to_device(v) for v in values(x)))
    elseif x isa Tuple
        return Tuple(to_device(v) for v in x)
    elseif x isa AbstractArray
        return CUDA.CuArray(x)
    else
        return x
    end
end

function synthetic_row(seq_len::Int, idx::Int)
    return (
        tokens = ["tok$(i)" for i in 1:seq_len],
        entities = [
            (start = 6, stop = 7, label = "PERSON"),
            (start = 18, stop = 19, label = "ORGANIZATION"),
            (start = 32, stop = 33, label = "LOCATION"),
        ],
        relations = [
            (head = 1, tail = 2, label = idx % 2 == 0 ? "WORKS_FOR" : "RELATED_TO"),
            (head = 1, tail = 3, label = "LOCATED_IN"),
        ],
    )
end

function relation_loss(outputs, targets)
    return entity_cross_entropy(outputs.entity_logits, targets.entity_labels) +
           boundary_bce(outputs.boundary_logits, targets.boundary_labels) +
           relation_cross_entropy(outputs.relation_logits, targets.relation_labels, targets.relation_mask) +
           confidence_bce(outputs.confidence_logits, targets.relation_targets, targets.relation_mask)
end

function main()
    CUDA.functional() || error("CUDA is not functional on this machine.")
    CUDA.allowscalar(false)

    rng = MersenneTwister(42)
    config = RelationExtractionConfig(
        vocab_size = 4096,
        max_sequence_length = 64,
        embedding_dimension = 128,
        number_of_heads = 4,
        number_of_layers = 2,
        num_relations = 8,
        time_dimension = 32,
        state_dimension = 128,
        window_size = 8,
        max_candidate_spans = 16,
        max_candidate_pairs = 24,
        biaffine_rank = 16,
        pair_neighbor_radius = 2,
    )

    rows = [synthetic_row(config.max_sequence_length, i) for i in 1:16]
    vocab = build_token_vocab(rows; max_vocab = config.vocab_size)
    entity_label_to_id = build_entity_label_space(rows)
    relation_label_to_id = build_relation_label_space(rows)

    config = RelationExtractionConfig(
        vocab_size = length(vocab),
        max_sequence_length = config.max_sequence_length,
        embedding_dimension = config.embedding_dimension,
        number_of_heads = config.number_of_heads,
        number_of_layers = config.number_of_layers,
        num_entity_labels = length(entity_label_to_id),
        num_relations = length(relation_label_to_id),
        time_dimension = config.time_dimension,
        state_dimension = config.state_dimension,
        window_size = config.window_size,
        local_operator = config.local_operator,
        residual_mode = config.residual_mode,
        min_frequency = config.min_frequency,
        max_frequency = config.max_frequency,
        default_time_step = config.default_time_step,
        dropout_rate = config.dropout_rate,
        use_ffn = config.use_ffn,
        ffn_expansion = config.ffn_expansion,
        use_parallel_scan = config.use_parallel_scan,
        parallel_chunk_size = config.parallel_chunk_size,
        use_vector_gains = config.use_vector_gains,
        use_per_head_alpha = config.use_per_head_alpha,
        use_branch_projections = config.use_branch_projections,
        max_candidate_spans = config.max_candidate_spans,
        max_candidate_pairs = config.max_candidate_pairs,
        max_span_width = config.max_span_width,
        biaffine_rank = config.biaffine_rank,
        pair_neighbor_radius = config.pair_neighbor_radius,
    )

    model = SwammaRelationExtractor(config)
    params, state = Lux.setup(rng, model)
    params = to_device(params)
    state = to_device(state)

    optimizer = Optimisers.AdamW(2.0f-4, (0.9f0, 0.999f0), 0.01f0)
    opt_state = Optimisers.setup(optimizer, params)

    println("Testing RE GPU training")
    println("  device: $(CUDA.name(CUDA.device()))")
    println("  steps: 3")
    println("  batch: 4")

    losses = Float64[]
    step_times = Float64[]

    for step in 1:3
        batch_rows = rows[(step - 1) * 4 + 1:step * 4]
        batch = prepare_rebel_batch(
            batch_rows,
            vocab,
            entity_label_to_id,
            relation_label_to_id;
            max_len = config.max_sequence_length,
            max_candidate_spans = config.max_candidate_spans,
            max_candidate_pairs = config.max_candidate_pairs,
            max_span_width = config.max_span_width,
            hard_negative_ratio = 2.0f0,
        )

        inputs = to_device((
            token_ids = batch.token_ids,
            token_mask = batch.token_mask,
            span_context_sentence_ids = batch.sentence_ids,
            spans = batch.spans,
            span_mask = batch.span_mask,
            span_scores = Float32.(batch.span_mask),
            mention_spans = batch.mention_spans,
            mention_mask = batch.mention_mask,
            relation_pairs = batch.relation_pairs,
            relation_mask = batch.relation_mask,
        ))
        targets = to_device((
            entity_labels = batch.entity_labels,
            boundary_labels = batch.boundary_labels,
            mention_labels = batch.mention_labels,
            mention_mask = batch.mention_mask,
            relation_labels = batch.relation_labels,
            relation_mask = batch.relation_mask,
            relation_targets = batch.relation_targets,
        ))

        t0 = time_ns()
        (loss, new_state), grads = Zygote.withgradient(params) do p
            outputs, next_state = model(inputs, p, state)
            relation_loss(outputs, targets), next_state
        end
        opt_state, params = Optimisers.update(opt_state, params, grads[1])
        state = new_state
        CUDA.synchronize()

        push!(losses, Float64(loss))
        push!(step_times, (time_ns() - t0) / 1e6)
        @printf("  step %d | loss %.4f | %.1f ms\n", step, loss, step_times[end])
    end

    println("Smoke test passed")
    @printf("  avg loss: %.4f\n", mean(losses))
    @printf("  median step: %.1f ms\n", median(step_times))
end

main()
