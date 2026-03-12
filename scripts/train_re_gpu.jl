#!/usr/bin/env julia
"""
GPU training script for Swamma relation extraction.

Usage:
    julia --project=. scripts/train_re_gpu.jl
    julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base.toml --max-steps 100
    julia --project=. scripts/train_re_gpu.jl --resume checkpoints/redfm_base/checkpoint_last.jls
"""

using Random
using Statistics
using Printf
using Dates
using Serialization
using TOML
using CUDA
using Lux
using Optimisers
using Zygote

include(joinpath(@__DIR__, "..", "src", "Swamma.jl"))
using .Swamma

Base.@kwdef struct RETrainingRunConfig
    config_path::String = "configs/redfm_base.toml"
    checkpoint_dir::String = "checkpoints/redfm_base"
    train_path::String = ""
    val_path::String = ""
    batch_size::Int = 16
    gradient_accumulation_steps::Int = 1
    learning_rate::Float32 = 2.0f-4
    weight_decay::Float32 = 0.01f0
    warmup_steps::Int = 1000
    total_steps::Int = 50000
    log_every::Int = 25
    eval_every::Int = 250
    save_every::Int = 1000
    hard_negative_ratio::Float32 = 2.0f0
    max_len::Int = 256
    seed::Int = 42
    max_eval_batches::Int = 8
    resume_path::Union{Nothing,String} = nothing
end

Base.@kwdef struct RECLIOptions
    config_path::String = "configs/redfm_base.toml"
    resume_path::Union{Nothing,String} = nothing
    max_steps_override::Union{Nothing,Int} = nothing
    eval_checkpoint::Union{Nothing,String} = nothing
    checkpoint_sweep::Vector{String} = String[]
    max_eval_batches_override::Union{Nothing,Int} = nothing
end

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

function to_cpu(x)
    if x isa NamedTuple
        return NamedTuple{keys(x)}(Tuple(to_cpu(v) for v in values(x)))
    elseif x isa Tuple
        return Tuple(to_cpu(v) for v in x)
    elseif x isa CUDA.CuArray
        return Array(x)
    elseif x isa AbstractArray
        return x
    else
        return x
    end
end

tree_to_device(x) = Lux.fmap(v -> v isa CUDA.CuArray ? v : v isa AbstractArray ? CUDA.CuArray(v) : v, x)
tree_to_cpu(x) = Lux.fmap(v -> v isa CUDA.CuArray ? Array(v) : v, x)

function tree_add(a, b)
    if a === nothing
        return b
    elseif b === nothing
        return a
    elseif a isa NamedTuple
        return NamedTuple{keys(a)}(Tuple(tree_add(getfield(a, k), getfield(b, k)) for k in keys(a)))
    elseif a isa Tuple
        return Tuple(tree_add(a[i], b[i]) for i in eachindex(a))
    elseif a isa AbstractArray
        return a .+ b
    else
        return a
    end
end

function tree_scale(x, α::Float32)
    if x === nothing
        return nothing
    elseif x isa NamedTuple
        return NamedTuple{keys(x)}(Tuple(tree_scale(getfield(x, k), α) for k in keys(x)))
    elseif x isa Tuple
        return Tuple(tree_scale(v, α) for v in x)
    elseif x isa AbstractArray
        return x .* α
    else
        return x
    end
end

function count_parameters(params)
    total = 0
    function walk(x)
        if x isa NamedTuple || x isa Tuple
            for v in values(x)
                walk(v)
            end
        elseif x isa AbstractArray
            total += length(x)
        end
    end
    walk(params)
    return total
end

function parse_args(args::Vector{String})
    options = RECLIOptions()

    i = 1
    while i <= length(args)
        if args[i] == "--config" && i < length(args)
            options = RECLIOptions(;
                config_path = args[i + 1],
                resume_path = options.resume_path,
                max_steps_override = options.max_steps_override,
                eval_checkpoint = options.eval_checkpoint,
                checkpoint_sweep = options.checkpoint_sweep,
                max_eval_batches_override = options.max_eval_batches_override,
            )
            i += 2
        elseif args[i] == "--resume" && i < length(args)
            options = RECLIOptions(;
                config_path = options.config_path,
                resume_path = args[i + 1],
                max_steps_override = options.max_steps_override,
                eval_checkpoint = options.eval_checkpoint,
                checkpoint_sweep = options.checkpoint_sweep,
                max_eval_batches_override = options.max_eval_batches_override,
            )
            i += 2
        elseif args[i] == "--max-steps" && i < length(args)
            options = RECLIOptions(;
                config_path = options.config_path,
                resume_path = options.resume_path,
                max_steps_override = parse(Int, args[i + 1]),
                eval_checkpoint = options.eval_checkpoint,
                checkpoint_sweep = options.checkpoint_sweep,
                max_eval_batches_override = options.max_eval_batches_override,
            )
            i += 2
        elseif args[i] == "--eval-checkpoint" && i < length(args)
            options = RECLIOptions(;
                config_path = options.config_path,
                resume_path = options.resume_path,
                max_steps_override = options.max_steps_override,
                eval_checkpoint = args[i + 1],
                checkpoint_sweep = options.checkpoint_sweep,
                max_eval_batches_override = options.max_eval_batches_override,
            )
            i += 2
        elseif args[i] == "--checkpoint-sweep" && i < length(args)
            sweep_paths = filter!(!isempty, strip.(split(args[i + 1], ",")))
            options = RECLIOptions(;
                config_path = options.config_path,
                resume_path = options.resume_path,
                max_steps_override = options.max_steps_override,
                eval_checkpoint = options.eval_checkpoint,
                checkpoint_sweep = sweep_paths,
                max_eval_batches_override = options.max_eval_batches_override,
            )
            i += 2
        elseif args[i] == "--max-eval-batches" && i < length(args)
            options = RECLIOptions(;
                config_path = options.config_path,
                resume_path = options.resume_path,
                max_steps_override = options.max_steps_override,
                eval_checkpoint = options.eval_checkpoint,
                checkpoint_sweep = options.checkpoint_sweep,
                max_eval_batches_override = parse(Int, args[i + 1]),
            )
            i += 2
        else
            error("Unknown argument: $(args[i])")
        end
    end

    return options
end

function load_run_config(path::String; resume_path = nothing, max_steps_override = nothing)
    data = TOML.parsefile(path)
    training = get(data, "training", Dict{String,Any}())
    checkpoints = get(training, "checkpoints", Dict{String,Any}())
    data_cfg = get(data, "data", Dict{String,Any}())

    run_cfg = RETrainingRunConfig(
        config_path = path,
        checkpoint_dir = joinpath("checkpoints", splitext(basename(path))[1]),
        train_path = get(data_cfg, "train_path", ""),
        val_path = get(data_cfg, "val_path", ""),
        batch_size = get(training, "batch_size", 16),
        gradient_accumulation_steps = get(training, "gradient_accumulation_steps", 1),
        learning_rate = Float32(get(training, "learning_rate", 2e-4)),
        weight_decay = Float32(get(training, "weight_decay", 0.01)),
        warmup_steps = get(training, "warmup_steps", 1000),
        total_steps = get(training, "total_steps", 50000),
        log_every = get(checkpoints, "log_every", get(training, "log_every", 25)),
        eval_every = get(checkpoints, "eval_every", get(training, "eval_every", 250)),
        save_every = get(checkpoints, "save_every", get(training, "save_every", 1000)),
        max_len = get(data_cfg, "max_len", get(get(data, "model", Dict{String,Any}()), "max_sequence_length", 256)),
        resume_path = resume_path,
    )

    if max_steps_override !== nothing
        run_cfg = RETrainingRunConfig(
            config_path = run_cfg.config_path,
            checkpoint_dir = run_cfg.checkpoint_dir,
            train_path = run_cfg.train_path,
            val_path = run_cfg.val_path,
            batch_size = run_cfg.batch_size,
            gradient_accumulation_steps = run_cfg.gradient_accumulation_steps,
            learning_rate = run_cfg.learning_rate,
            weight_decay = run_cfg.weight_decay,
            warmup_steps = run_cfg.warmup_steps,
            total_steps = max_steps_override,
            log_every = run_cfg.log_every,
            eval_every = run_cfg.eval_every,
            save_every = run_cfg.save_every,
            hard_negative_ratio = run_cfg.hard_negative_ratio,
            max_len = run_cfg.max_len,
            seed = run_cfg.seed,
            max_eval_batches = run_cfg.max_eval_batches,
            resume_path = run_cfg.resume_path,
        )
    end

    return run_cfg
end

function with_label_counts(config::RelationExtractionConfig, num_entity_labels::Int, num_relations::Int, vocab_size::Int)
    return RelationExtractionConfig(
        vocab_size = vocab_size,
        max_sequence_length = config.max_sequence_length,
        embedding_dimension = config.embedding_dimension,
        number_of_heads = config.number_of_heads,
        number_of_layers = config.number_of_layers,
        number_of_refinement_layers = config.number_of_refinement_layers,
        num_entity_labels = num_entity_labels,
        num_relations = num_relations,
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
end

function relation_loss(outputs, targets)
    return entity_cross_entropy(outputs.entity_logits, targets.entity_labels) +
           boundary_bce(outputs.boundary_logits, targets.boundary_labels) +
           relation_cross_entropy(outputs.relation_logits, targets.relation_labels, targets.relation_mask) +
           confidence_bce(outputs.confidence_logits, targets.relation_targets, targets.relation_mask)
end

function relation_loss_breakdown(outputs, targets)
    entity = Float32(entity_cross_entropy(outputs.entity_logits, targets.entity_labels))
    boundary = Float32(boundary_bce(outputs.boundary_logits, targets.boundary_labels))
    relation = Float32(
        relation_cross_entropy(outputs.relation_logits, targets.relation_labels, targets.relation_mask)
    )
    confidence = Float32(
        confidence_bce(outputs.confidence_logits, targets.relation_targets, targets.relation_mask)
    )
    total = entity + boundary + relation + confidence
    return (
        entity = entity,
        boundary = boundary,
        relation = relation,
        confidence = confidence,
        total = total,
    )
end

function recent_mean(losses::Vector{Float32}, window::Int)
    isempty(losses) && return NaN32
    start_idx = max(1, length(losses) - window + 1)
    return mean(@view(losses[start_idx:end]))
end

function make_batch(rows, vocab, entity_label_to_id, relation_label_to_id, model_config, run_config)
    batch = prepare_rebel_batch(
        rows,
        vocab,
        entity_label_to_id,
        relation_label_to_id;
        max_len = run_config.max_len,
        max_candidate_spans = model_config.max_candidate_spans,
        max_candidate_pairs = model_config.max_candidate_pairs,
        hard_negative_ratio = run_config.hard_negative_ratio,
    )

    inputs = (
        token_ids = batch.token_ids,
        spans = batch.spans,
        span_mask = batch.span_mask,
        span_scores = Float32.(batch.span_mask),
        relation_pairs = batch.relation_pairs,
        relation_mask = batch.relation_mask,
    )
    targets = (
        entity_labels = batch.entity_labels,
        boundary_labels = batch.boundary_labels,
        spans = batch.spans,
        span_mask = batch.span_mask,
        relation_labels = batch.relation_labels,
        relation_pairs = batch.relation_pairs,
        relation_mask = batch.relation_mask,
        relation_targets = batch.relation_targets,
    )
    return to_device(inputs), to_device(targets)
end

@inline safe_rate(numerator::Real, denominator::Real) = denominator > 0 ? Float32(numerator / denominator) : 0.0f0

function collect_span_set(spans, span_mask, batch_idx::Int)
    span_set = Set{Tuple{Int, Int}}()
    max_spans = size(spans, 2)
    for i in 1:max_spans
        span_mask[i, batch_idx] || continue
        push!(span_set, (Int(spans[1, i, batch_idx]), Int(spans[2, i, batch_idx])))
    end
    return span_set
end

function collect_pair_set(spans, relation_pairs, relation_mask, batch_idx::Int)
    pair_set = Set{NTuple{4, Int}}()
    max_pairs = size(relation_pairs, 2)
    max_spans = size(spans, 2)
    for i in 1:max_pairs
        relation_mask[i, batch_idx] || continue
        head_idx = Int(relation_pairs[1, i, batch_idx])
        tail_idx = Int(relation_pairs[2, i, batch_idx])
        if !(1 <= head_idx <= max_spans && 1 <= tail_idx <= max_spans)
            continue
        end
        head = (Int(spans[1, head_idx, batch_idx]), Int(spans[2, head_idx, batch_idx]))
        tail = (Int(spans[1, tail_idx, batch_idx]), Int(spans[2, tail_idx, batch_idx]))
        push!(pair_set, (head[1], head[2], tail[1], tail[2]))
    end
    return pair_set
end

function collect_gold_relation_set(spans, relation_pairs, relation_labels, relation_mask, relation_targets, batch_idx::Int)
    relation_set = Set{NTuple{5, Int}}()
    max_pairs = size(relation_pairs, 2)
    max_spans = size(spans, 2)
    for i in 1:max_pairs
        relation_mask[i, batch_idx] || continue
        relation_targets[i, batch_idx] > 0.5f0 || continue
        head_idx = Int(relation_pairs[1, i, batch_idx])
        tail_idx = Int(relation_pairs[2, i, batch_idx])
        if !(1 <= head_idx <= max_spans && 1 <= tail_idx <= max_spans)
            continue
        end
        head = (Int(spans[1, head_idx, batch_idx]), Int(spans[2, head_idx, batch_idx]))
        tail = (Int(spans[1, tail_idx, batch_idx]), Int(spans[2, tail_idx, batch_idx]))
        label_id = Int(relation_labels[i, batch_idx])
        push!(relation_set, (head[1], head[2], tail[1], tail[2], label_id))
    end
    return relation_set
end

function collect_predicted_relation_set(outputs; no_relation_id::Int, confidence_threshold::Float32)
    spans = to_cpu(outputs.spans)
    relation_pairs = to_cpu(outputs.relation_pairs)
    relation_mask = to_cpu(outputs.relation_mask)
    relation_logits = to_cpu(outputs.relation_logits)
    confidence_logits = Float32.(to_cpu(outputs.confidence_logits))
    confidence_probs = 1.0f0 ./ (1.0f0 .+ exp.(-confidence_logits))

    batch_size = size(spans, 3)
    max_pairs = size(relation_pairs, 2)
    max_spans = size(spans, 2)
    predicted_sets = Vector{Set{NTuple{5, Int}}}(undef, batch_size)

    for b in 1:batch_size
        rel_set = Set{NTuple{5, Int}}()
        for i in 1:max_pairs
            relation_mask[i, b] || continue
            confidence_probs[1, i, b] >= confidence_threshold || continue
            label_id = findmax(@view(relation_logits[:, i, b]))[2]
            label_id == no_relation_id && continue
            head_idx = Int(relation_pairs[1, i, b])
            tail_idx = Int(relation_pairs[2, i, b])
            if !(1 <= head_idx <= max_spans && 1 <= tail_idx <= max_spans)
                continue
            end
            head = (Int(spans[1, head_idx, b]), Int(spans[2, head_idx, b]))
            tail = (Int(spans[1, tail_idx, b]), Int(spans[2, tail_idx, b]))
            push!(rel_set, (head[1], head[2], tail[1], tail[2], Int(label_id)))
        end
        predicted_sets[b] = rel_set
    end

    return predicted_sets
end

function proposal_diagnostics(outputs, targets; no_relation_id::Int, confidence_threshold::Float32 = 0.5f0)
    target_spans = to_cpu(targets.spans)
    target_span_mask = to_cpu(targets.span_mask)
    target_relation_pairs = to_cpu(targets.relation_pairs)
    target_relation_labels = to_cpu(targets.relation_labels)
    target_relation_mask = to_cpu(targets.relation_mask)
    target_relation_targets = Float32.(to_cpu(targets.relation_targets))

    predicted_spans = to_cpu(outputs.spans)
    predicted_span_mask = to_cpu(outputs.span_mask)
    predicted_pairs = to_cpu(outputs.relation_pairs)
    predicted_pair_mask = to_cpu(outputs.relation_mask)
    predicted_relations = collect_predicted_relation_set(
        outputs;
        no_relation_id = no_relation_id,
        confidence_threshold = confidence_threshold,
    )

    batch_size = size(target_spans, 3)
    gold_span_total = 0
    matched_span_total = 0
    gold_pair_total = 0
    matched_pair_total = 0
    predicted_relation_total = 0
    gold_relation_total = 0
    relation_true_positive_total = 0

    for b in 1:batch_size
        gold_span_set = collect_span_set(target_spans, target_span_mask, b)
        pred_span_set = collect_span_set(predicted_spans, predicted_span_mask, b)
        gold_pair_set = collect_pair_set(target_spans, target_relation_pairs, target_relation_mask .& (target_relation_targets .> 0.5f0), b)
        pred_pair_set = collect_pair_set(predicted_spans, predicted_pairs, predicted_pair_mask, b)
        gold_relation_set = collect_gold_relation_set(
            target_spans,
            target_relation_pairs,
            target_relation_labels,
            target_relation_mask,
            target_relation_targets,
            b,
        )
        pred_relation_set = predicted_relations[b]

        gold_span_total += length(gold_span_set)
        matched_span_total += length(intersect(gold_span_set, pred_span_set))
        gold_pair_total += length(gold_pair_set)
        matched_pair_total += length(intersect(gold_pair_set, pred_pair_set))
        gold_relation_total += length(gold_relation_set)
        predicted_relation_total += length(pred_relation_set)
        relation_true_positive_total += length(intersect(gold_relation_set, pred_relation_set))
    end

    return (
        gold_spans = gold_span_total,
        matched_spans = matched_span_total,
        gold_pairs = gold_pair_total,
        matched_pairs = matched_pair_total,
        gold_relations = gold_relation_total,
        predicted_relations = predicted_relation_total,
        true_positive_relations = relation_true_positive_total,
    )
end

function build_proposal_inputs(outputs, inputs, model_config)
    entity_logits_cpu = to_cpu(outputs.entity_logits)
    boundary_logits_cpu = to_cpu(outputs.boundary_logits)
    spans, span_mask, span_scores = Swamma.RelationExtraction.propose_candidate_spans(
        entity_logits_cpu,
        boundary_logits_cpu;
        max_candidate_spans = model_config.max_candidate_spans,
        max_span_width = model_config.max_span_width,
    )
    relation_pairs, relation_mask = Swamma.RelationExtraction.propose_relation_pairs(
        spans,
        span_mask,
        span_scores;
        max_candidate_pairs = model_config.max_candidate_pairs,
        neighbor_radius = model_config.pair_neighbor_radius,
    )
    return (
        token_ids = inputs.token_ids,
        spans = to_device(spans),
        span_mask = to_device(span_mask),
        span_scores = to_device(span_scores),
        relation_pairs = to_device(relation_pairs),
        relation_mask = to_device(relation_mask),
    )
end

function format_eval_summary(eval_stats)
    return @sprintf(
        "val_loss %.4f | entity %.4f | boundary %.4f | relation %.4f | confidence %.4f | span_recall %.4f | pair_recall %.4f | rel_p %.4f | rel_r %.4f | rel_f1 %.4f",
        eval_stats.total_loss,
        eval_stats.entity_loss,
        eval_stats.boundary_loss,
        eval_stats.relation_loss,
        eval_stats.confidence_loss,
        eval_stats.span_recall,
        eval_stats.pair_recall,
        eval_stats.relation_precision,
        eval_stats.relation_recall,
        eval_stats.relation_f1,
    )
end

function evaluate_model(model, params, state, rows, vocab, entity_label_to_id, relation_label_to_id, model_config, run_config)
    isempty(rows) && return (
        total_loss = NaN32,
        entity_loss = NaN32,
        boundary_loss = NaN32,
        relation_loss = NaN32,
        confidence_loss = NaN32,
        span_recall = NaN32,
        pair_recall = NaN32,
        relation_precision = NaN32,
        relation_recall = NaN32,
        relation_f1 = NaN32,
        gold_spans = 0,
        gold_pairs = 0,
        gold_relations = 0,
        predicted_relations = 0,
        true_positive_relations = 0,
        batches = 0,
    )

    total_loss_sum = 0.0f0
    entity_loss_sum = 0.0f0
    boundary_loss_sum = 0.0f0
    relation_loss_sum = 0.0f0
    confidence_loss_sum = 0.0f0
    gold_spans = 0
    matched_spans = 0
    gold_pairs = 0
    matched_pairs = 0
    gold_relations = 0
    predicted_relations = 0
    true_positive_relations = 0
    eval_state = Lux.testmode(state)
    max_batches = min(run_config.max_eval_batches, cld(length(rows), run_config.batch_size))
    no_relation_id = get(relation_label_to_id, "NO_RELATION", 1)

    for batch_idx in 1:max_batches
        start_idx = (batch_idx - 1) * run_config.batch_size + 1
        end_idx = min(batch_idx * run_config.batch_size, length(rows))
        batch_rows = rows[start_idx:end_idx]
        inputs, targets = make_batch(batch_rows, vocab, entity_label_to_id, relation_label_to_id, model_config, run_config)
        outputs, _ = model(inputs, params, eval_state)
        breakdown = relation_loss_breakdown(outputs, targets)
        total_loss_sum += breakdown.total
        entity_loss_sum += breakdown.entity
        boundary_loss_sum += breakdown.boundary
        relation_loss_sum += breakdown.relation
        confidence_loss_sum += breakdown.confidence

        proposal_inputs = build_proposal_inputs(outputs, inputs, model_config)
        proposal_outputs, _ = model(proposal_inputs, params, eval_state)
        diagnostics = proposal_diagnostics(
            proposal_outputs,
            targets;
            no_relation_id = no_relation_id,
            confidence_threshold = 0.5f0,
        )
        gold_spans += diagnostics.gold_spans
        matched_spans += diagnostics.matched_spans
        gold_pairs += diagnostics.gold_pairs
        matched_pairs += diagnostics.matched_pairs
        gold_relations += diagnostics.gold_relations
        predicted_relations += diagnostics.predicted_relations
        true_positive_relations += diagnostics.true_positive_relations
        CUDA.synchronize()
    end

    relation_precision = safe_rate(true_positive_relations, predicted_relations)
    relation_recall = safe_rate(true_positive_relations, gold_relations)
    relation_f1 = relation_precision + relation_recall > 0 ?
        2f0 * relation_precision * relation_recall / (relation_precision + relation_recall) :
        0.0f0

    return (
        total_loss = total_loss_sum / max_batches,
        entity_loss = entity_loss_sum / max_batches,
        boundary_loss = boundary_loss_sum / max_batches,
        relation_loss = relation_loss_sum / max_batches,
        confidence_loss = confidence_loss_sum / max_batches,
        span_recall = safe_rate(matched_spans, gold_spans),
        pair_recall = safe_rate(matched_pairs, gold_pairs),
        relation_precision = relation_precision,
        relation_recall = relation_recall,
        relation_f1 = relation_f1,
        gold_spans = gold_spans,
        gold_pairs = gold_pairs,
        gold_relations = gold_relations,
        predicted_relations = predicted_relations,
        true_positive_relations = true_positive_relations,
        batches = max_batches,
    )
end

function save_re_checkpoint(path; params, state, opt_state, step, epoch, loss, vocab, entity_label_to_id, relation_label_to_id, run_config, model_config)
    mkpath(dirname(path))
    serialize(path, Dict(
        :params => tree_to_cpu(params),
        :state => tree_to_cpu(state),
        :opt_state => tree_to_cpu(opt_state),
        :step => step,
        :epoch => epoch,
        :loss => loss,
        :vocab => vocab,
        :entity_label_to_id => entity_label_to_id,
        :relation_label_to_id => relation_label_to_id,
        :run_config => run_config,
        :model_config => model_config,
        :timestamp => Dates.now(),
    ))
end

function reclaim_device_memory()
    GC.gc(false)
    CUDA.reclaim()
    return nothing
end

function load_eval_context(run_config::RETrainingRunConfig)
    base_config = load_relation_extraction_config(run_config.config_path)
    train_rows = load_rebel_jsonl(run_config.train_path)
    val_rows = isfile(run_config.val_path) ? load_rebel_jsonl(run_config.val_path) : Any[]
    all_rows = isempty(val_rows) ? train_rows : vcat(train_rows, val_rows)

    vocab = build_token_vocab(train_rows; max_vocab = base_config.vocab_size)
    entity_label_to_id = build_entity_label_space(all_rows)
    relation_label_to_id = build_relation_label_space(all_rows)
    model_config = with_label_counts(base_config, length(entity_label_to_id), length(relation_label_to_id), length(vocab))

    return (
        train_rows = train_rows,
        val_rows = val_rows,
        vocab = vocab,
        entity_label_to_id = entity_label_to_id,
        relation_label_to_id = relation_label_to_id,
        model_config = model_config,
    )
end

function step_from_checkpoint(path::String, ckpt)
    step = get(ckpt, :step, nothing)
    if step !== nothing
        return Int(step)
    end
    m = match(r"checkpoint_step_(\d+)\.jls$", basename(path))
    return m === nothing ? 0 : parse(Int, m.captures[1])
end

function run_checkpoint_sweep(run_config::RETrainingRunConfig, checkpoint_paths::Vector{String})
    isempty(checkpoint_paths) && error("No checkpoints provided for sweep.")
    checkpoint_paths = sort(copy(checkpoint_paths); by = path -> begin
        m = match(r"checkpoint_step_(\d+)\.jls$", basename(path))
        m === nothing ? typemax(Int) : parse(Int, m.captures[1])
    end)
    context = load_eval_context(run_config)
    isempty(context.val_rows) && error("Validation data not found for checkpoint sweep.")

    println("=" ^ 120)
    println("RE Checkpoint Sweep")
    println("=" ^ 120)
    println("Config: $(run_config.config_path)")
    println("Val rows: $(length(context.val_rows)) | Max eval batches: $(run_config.max_eval_batches)")
    println()
    println(rpad("checkpoint", 22) *
            lpad("step", 8) *
            lpad("total", 10) *
            lpad("entity", 10) *
            lpad("bound", 10) *
            lpad("rel", 10) *
            lpad("conf", 10) *
            lpad("span_r", 10) *
            lpad("pair_r", 10) *
            lpad("rel_p", 10) *
            lpad("rel_r", 10) *
            lpad("rel_f1", 10))
    println("-" ^ 120)

    for checkpoint_path in checkpoint_paths
        ckpt = deserialize(checkpoint_path)
        model_config = get(ckpt, :model_config, context.model_config)
        vocab = get(ckpt, :vocab, context.vocab)
        entity_label_to_id = get(ckpt, :entity_label_to_id, context.entity_label_to_id)
        relation_label_to_id = get(ckpt, :relation_label_to_id, context.relation_label_to_id)
        model = SwammaRelationExtractor(model_config)
        params = tree_to_device(ckpt[:params])
        state = tree_to_device(ckpt[:state])
        step = step_from_checkpoint(checkpoint_path, ckpt)

        eval_stats = evaluate_model(
            model,
            params,
            state,
            context.val_rows,
            vocab,
            entity_label_to_id,
            relation_label_to_id,
            model_config,
            run_config,
        )

        println(
            rpad(basename(checkpoint_path), 22) *
            lpad(step, 8) *
            @sprintf("%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f",
                eval_stats.total_loss,
                eval_stats.entity_loss,
                eval_stats.boundary_loss,
                eval_stats.relation_loss,
                eval_stats.confidence_loss,
                eval_stats.span_recall,
                eval_stats.pair_recall,
                eval_stats.relation_precision,
                eval_stats.relation_recall,
                eval_stats.relation_f1,
            )
        )
        reclaim_device_memory()
    end
end

function main()
    CUDA.functional() || error("CUDA is not functional on this machine.")
    CUDA.allowscalar(false)

    options = parse_args(ARGS)
    run_config = load_run_config(
        options.config_path;
        resume_path = options.resume_path,
        max_steps_override = options.max_steps_override,
    )
    if options.max_eval_batches_override !== nothing
        run_config = RETrainingRunConfig(
            config_path = run_config.config_path,
            checkpoint_dir = run_config.checkpoint_dir,
            train_path = run_config.train_path,
            val_path = run_config.val_path,
            batch_size = run_config.batch_size,
            gradient_accumulation_steps = run_config.gradient_accumulation_steps,
            learning_rate = run_config.learning_rate,
            weight_decay = run_config.weight_decay,
            warmup_steps = run_config.warmup_steps,
            total_steps = run_config.total_steps,
            log_every = run_config.log_every,
            eval_every = run_config.eval_every,
            save_every = run_config.save_every,
            hard_negative_ratio = run_config.hard_negative_ratio,
            max_len = run_config.max_len,
            seed = run_config.seed,
            max_eval_batches = options.max_eval_batches_override,
            resume_path = run_config.resume_path,
        )
    end

    if options.eval_checkpoint !== nothing
        run_checkpoint_sweep(run_config, [options.eval_checkpoint])
        return
    elseif !isempty(options.checkpoint_sweep)
        run_checkpoint_sweep(run_config, options.checkpoint_sweep)
        return
    end

    isfile(run_config.train_path) || error("Training data not found: $(run_config.train_path). Run scripts/test_re_training.jl first for a smoke test, or provide REBEL-format JSONL data.")

    println("=" ^ 72)
    println("Swamma Relation Extraction GPU Training")
    println("=" ^ 72)
    println("Time:   $(Dates.now())")
    println("Device: $(CUDA.name(CUDA.device()))")
    println("Config: $(run_config.config_path)")

    base_config = load_relation_extraction_config(run_config.config_path)
    train_rows = load_rebel_jsonl(run_config.train_path)
    val_rows = isfile(run_config.val_path) ? load_rebel_jsonl(run_config.val_path) : Any[]
    all_rows = isempty(val_rows) ? train_rows : vcat(train_rows, val_rows)

    vocab = build_token_vocab(train_rows; max_vocab = base_config.vocab_size)
    entity_label_to_id = build_entity_label_space(all_rows)
    relation_label_to_id = build_relation_label_space(all_rows)
    model_config = with_label_counts(base_config, length(entity_label_to_id), length(relation_label_to_id), length(vocab))

    print_relation_extraction_summary(model_config)
    println("Train rows: $(length(train_rows))")
    println("Val rows:   $(length(val_rows))")
    println("Vocab:      $(length(vocab))")

    rng = MersenneTwister(run_config.seed)
    model = SwammaRelationExtractor(model_config)
    params, state = Lux.setup(rng, model)
    optimizer = Optimisers.AdamW(run_config.learning_rate, (0.9f0, 0.999f0), run_config.weight_decay)
    opt_state = nothing

    start_step = 0
    start_epoch = 1
    best_val_loss = Inf32

    if run_config.resume_path !== nothing
        ckpt = deserialize(run_config.resume_path)
        params = ckpt[:params]
        state = ckpt[:state]
        opt_state = ckpt[:opt_state]
        vocab = ckpt[:vocab]
        entity_label_to_id = ckpt[:entity_label_to_id]
        relation_label_to_id = ckpt[:relation_label_to_id]
        start_step = get(ckpt, :step, 0)
        start_epoch = get(ckpt, :epoch, 0) + 1
        ckpt_loss = get(ckpt, :loss, Inf32)
        best_val_loss = ckpt_loss === nothing ? Inf32 : Float32(ckpt_loss)
        println("Resumed from $(run_config.resume_path) at step $start_step")
    end

    params = tree_to_device(params)
    state = tree_to_device(state)
    if opt_state === nothing
        opt_state = Optimisers.setup(optimizer, params)
    else
        opt_state = tree_to_device(opt_state)
    end

    println("Parameters: $(round(count_parameters(params) / 1e6, digits = 2))M")
    println("Batch size: $(run_config.batch_size)")
    println("Grad accum: $(run_config.gradient_accumulation_steps)")
    println("Total step updates: $(run_config.total_steps)")
    println()

    step = start_step
    epoch = start_epoch
    recent_losses = Float32[]

    while step < run_config.total_steps
        shuffled = Random.shuffle(rng, copy(train_rows))
        batch_starts = collect(1:run_config.batch_size:length(shuffled))
        grad_accum = nothing
        accum_count = 0

        for batch_start in batch_starts
            batch_end = min(batch_start + run_config.batch_size - 1, length(shuffled))
            batch_rows = shuffled[batch_start:batch_end]
            inputs, targets = make_batch(batch_rows, vocab, entity_label_to_id, relation_label_to_id, model_config, run_config)

            t0 = time_ns()
            (loss, new_state), grads = Zygote.withgradient(params) do p
                outputs, next_state = model(inputs, p, state)
                relation_loss(outputs, targets), next_state
            end
            CUDA.synchronize()

            grad_accum = tree_add(grad_accum, grads[1])
            accum_count += 1
            state = new_state
            push!(recent_losses, Float32(loss))

            if accum_count == run_config.gradient_accumulation_steps || batch_end == length(shuffled)
                step += 1
                mean_grads = tree_scale(grad_accum, 1.0f0 / Float32(accum_count))
                opt_state, params = Optimisers.update(opt_state, params, mean_grads)
                grad_accum = nothing
                accum_count = 0

                if step % run_config.log_every == 0 || step == 1
                    dt_ms = (time_ns() - t0) / 1e6
                    avg_loss = recent_mean(recent_losses, run_config.log_every)
                    effective_batch = run_config.batch_size * run_config.gradient_accumulation_steps
                    tokens_per_sec = (run_config.max_len * effective_batch) / max(dt_ms / 1e3, 1f-6)
                    @printf(
                        "step %6d | epoch %3d | loss %.4f | %.1f ms/update | %.0f tok/s\n",
                        step, epoch, avg_loss, dt_ms, tokens_per_sec
                    )
                    flush(stdout)
                end

                if !isempty(val_rows) && step % run_config.eval_every == 0
                    eval_stats = evaluate_model(
                        model,
                        params,
                        state,
                        val_rows,
                        vocab,
                        entity_label_to_id,
                        relation_label_to_id,
                        model_config,
                        run_config,
                    )
                    @printf("  eval step %6d | %s\n", step, format_eval_summary(eval_stats))
                    flush(stdout)
                    if eval_stats.total_loss < best_val_loss
                        best_val_loss = eval_stats.total_loss
                        save_re_checkpoint(
                            joinpath(run_config.checkpoint_dir, "checkpoint_best.jls");
                            params = params,
                            state = state,
                            opt_state = opt_state,
                            step = step,
                            epoch = epoch,
                            loss = eval_stats.total_loss,
                            vocab = vocab,
                            entity_label_to_id = entity_label_to_id,
                            relation_label_to_id = relation_label_to_id,
                            run_config = run_config,
                            model_config = model_config,
                        )
                    end
                    reclaim_device_memory()
                end

                if step % run_config.save_every == 0
                    save_re_checkpoint(
                        joinpath(run_config.checkpoint_dir, "checkpoint_step_$(step).jls");
                        params = params,
                        state = state,
                        opt_state = opt_state,
                        step = step,
                        epoch = epoch,
                        loss = recent_mean(recent_losses, run_config.log_every),
                        vocab = vocab,
                        entity_label_to_id = entity_label_to_id,
                        relation_label_to_id = relation_label_to_id,
                        run_config = run_config,
                        model_config = model_config,
                    )
                    save_re_checkpoint(
                        joinpath(run_config.checkpoint_dir, "checkpoint_last.jls");
                        params = params,
                        state = state,
                        opt_state = opt_state,
                        step = step,
                        epoch = epoch,
                        loss = recent_mean(recent_losses, run_config.log_every),
                        vocab = vocab,
                        entity_label_to_id = entity_label_to_id,
                        relation_label_to_id = relation_label_to_id,
                        run_config = run_config,
                        model_config = model_config,
                    )
                    reclaim_device_memory()
                end

                reclaim_device_memory()
                step >= run_config.total_steps && break
            end
        end

        epoch += 1
    end

    save_re_checkpoint(
        joinpath(run_config.checkpoint_dir, "checkpoint_last.jls");
        params = params,
        state = state,
        opt_state = opt_state,
        step = step,
        epoch = epoch,
        loss = isempty(recent_losses) ? nothing : mean(recent_losses),
        vocab = vocab,
        entity_label_to_id = entity_label_to_id,
        relation_label_to_id = relation_label_to_id,
        run_config = run_config,
        model_config = model_config,
    )
    reclaim_device_memory()

    println("Training complete at step $step")
end

main()
