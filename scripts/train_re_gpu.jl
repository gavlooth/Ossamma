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
import ChainRulesCore

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
    mention_negative_ratio::Float32 = 1.0f0
    proposal_train_probability::Float32 = 0.0f0
    proposal_loss_weight::Float32 = 0.0f0
    proposal_warmup_steps::Int = 0
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
    oracle_ladder_checkpoint::Union{Nothing,String} = nothing
    mention_sweep_checkpoint::Union{Nothing,String} = nothing
    threshold_sweep_checkpoint::Union{Nothing,String} = nothing
    margin_sweep_checkpoint::Union{Nothing,String} = nothing
    nonnull_sweep_checkpoint::Union{Nothing,String} = nothing
    auto_calibrate_checkpoint::Union{Nothing,String} = nothing
    evidence_pooling_sweep_checkpoint::Union{Nothing,String} = nothing
    pair_sweep_checkpoint::Union{Nothing,String} = nothing
    checkpoint_sweep::Vector{String} = String[]
    mention_sweep_budgets::Vector{Int} = Int[]
    mention_sweep_modes::Vector{Symbol} = Symbol[]
    threshold_sweep_values::Vector{Float32} = Float32[]
    threshold_sweep_margin::Float32 = 0.0f0
    threshold_sweep_nonnull::Float32 = 0.0f0
    margin_sweep_values::Vector{Float32} = Float32[]
    nonnull_sweep_values::Vector{Float32} = Float32[]
    nonnull_sweep_confidence::Float32 = 0.5f0
    nonnull_sweep_margin::Float32 = 0.0f0
    decode_head_cap::Int = 0
    decode_tail_cap::Int = 0
    per_relation_thresholds::Union{Nothing,String} = nothing
    auto_calibrate_threshold::Float32 = 0.70f0
    auto_calibrate_margin::Float32 = 0.30f0
    auto_calibrate_nonnull::Float32 = 0.0f0
    auto_calibrate_min_predictions::Int = 8
    auto_calibrate_thresholds::Vector{Float32} = Float32[]
    type_constraints_mode::String = "off"
    type_constraints_min_count::Int = 1
    relation_consistency_mode::String = "off"
    relation_consistency_min_count::Int = 1
    evidence_pooling_modes::Vector{Symbol} = Symbol[]
    pair_sweep_budgets::Vector{Int} = Int[]
    pair_sweep_overgenerate::Vector{Int} = Int[]
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

function merge_resume_tree(current, loaded)
    if current isa NamedTuple
        loaded isa NamedTuple || return current, true
        loaded_keys = keys(loaded)
        mismatch = false
        merged_values = map(keys(current)) do key
            if key in loaded_keys
                merged, child_mismatch = merge_resume_tree(getfield(current, key), getfield(loaded, key))
                mismatch |= child_mismatch
                merged
            else
                mismatch = true
                getfield(current, key)
            end
        end
        return NamedTuple{keys(current)}(Tuple(merged_values)), mismatch
    elseif current isa Tuple
        loaded isa Tuple || return current, true
        length(current) == length(loaded) || return current, true
        mismatch = false
        merged_values = map(eachindex(current)) do i
            merged, child_mismatch = merge_resume_tree(current[i], loaded[i])
            mismatch |= child_mismatch
            merged
        end
        return Tuple(merged_values), mismatch
    elseif current isa AbstractArray
        if loaded isa AbstractArray && size(current) == size(loaded)
            return Array(loaded), false
        end
        return current, true
    else
        if loaded isa typeof(current)
            return loaded, false
        end
        return current, true
    end
end

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
    config_path = options.config_path
    resume_path = options.resume_path
    max_steps_override = options.max_steps_override
    eval_checkpoint = options.eval_checkpoint
    oracle_ladder_checkpoint = options.oracle_ladder_checkpoint
    mention_sweep_checkpoint = options.mention_sweep_checkpoint
    threshold_sweep_checkpoint = options.threshold_sweep_checkpoint
    margin_sweep_checkpoint = options.margin_sweep_checkpoint
    nonnull_sweep_checkpoint = options.nonnull_sweep_checkpoint
    auto_calibrate_checkpoint = options.auto_calibrate_checkpoint
    evidence_pooling_sweep_checkpoint = options.evidence_pooling_sweep_checkpoint
    pair_sweep_checkpoint = options.pair_sweep_checkpoint
    checkpoint_sweep = copy(options.checkpoint_sweep)
    mention_sweep_budgets = copy(options.mention_sweep_budgets)
    mention_sweep_modes = copy(options.mention_sweep_modes)
    threshold_sweep_values = copy(options.threshold_sweep_values)
    threshold_sweep_margin = options.threshold_sweep_margin
    threshold_sweep_nonnull = options.threshold_sweep_nonnull
    margin_sweep_values = copy(options.margin_sweep_values)
    nonnull_sweep_values = copy(options.nonnull_sweep_values)
    nonnull_sweep_confidence = options.nonnull_sweep_confidence
    nonnull_sweep_margin = options.nonnull_sweep_margin
    decode_head_cap = options.decode_head_cap
    decode_tail_cap = options.decode_tail_cap
    per_relation_thresholds = options.per_relation_thresholds
    auto_calibrate_threshold = options.auto_calibrate_threshold
    auto_calibrate_margin = options.auto_calibrate_margin
    auto_calibrate_nonnull = options.auto_calibrate_nonnull
    auto_calibrate_min_predictions = options.auto_calibrate_min_predictions
    auto_calibrate_thresholds = copy(options.auto_calibrate_thresholds)
    type_constraints_mode = options.type_constraints_mode
    type_constraints_min_count = options.type_constraints_min_count
    relation_consistency_mode = options.relation_consistency_mode
    relation_consistency_min_count = options.relation_consistency_min_count
    evidence_pooling_modes = copy(options.evidence_pooling_modes)
    pair_sweep_budgets = copy(options.pair_sweep_budgets)
    pair_sweep_overgenerate = copy(options.pair_sweep_overgenerate)
    max_eval_batches_override = options.max_eval_batches_override

    i = 1
    while i <= length(args)
        arg = args[i]
        if (arg == "--config" || arg == "--resume" || arg == "--max-steps" ||
            arg == "--eval-checkpoint" || arg == "--oracle-ladder-checkpoint" ||
            arg == "--mention-sweep-checkpoint" || arg == "--threshold-sweep-checkpoint" ||
            arg == "--margin-sweep-checkpoint" || arg == "--nonnull-sweep-checkpoint" ||
            arg == "--auto-calibrate-checkpoint" ||
            arg == "--evidence-pooling-sweep-checkpoint" || arg == "--evidence-pooling-modes" ||
            arg == "--pair-sweep-checkpoint" || arg == "--checkpoint-sweep" ||
            arg == "--mention-sweep-budgets" || arg == "--mention-sweep-modes" ||
            arg == "--threshold-sweep-values" || arg == "--threshold-sweep-margin" || arg == "--threshold-sweep-nonnull" ||
            arg == "--margin-sweep-values" || arg == "--nonnull-sweep-values" || arg == "--nonnull-sweep-confidence" ||
            arg == "--nonnull-sweep-margin" || arg == "--decode-head-cap" || arg == "--decode-tail-cap" ||
            arg == "--per-relation-thresholds" ||
            arg == "--auto-calibrate-threshold" || arg == "--auto-calibrate-margin" ||
            arg == "--auto-calibrate-nonnull" || arg == "--auto-calibrate-min-predictions" ||
            arg == "--auto-calibrate-thresholds" ||
            arg == "--type-constraints-mode" || arg == "--type-constraints-min-count" ||
            arg == "--relation-consistency-mode" || arg == "--relation-consistency-min-count" ||
            arg == "--pair-sweep-budgets" ||
            arg == "--pair-sweep-overgenerate" || arg == "--max-eval-batches") && i >= length(args)
            error("Missing value for argument: $(arg)")
        end

        if arg == "--config"
            config_path = args[i + 1]
            i += 2
        elseif arg == "--resume"
            resume_path = args[i + 1]
            i += 2
        elseif arg == "--max-steps"
            max_steps_override = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--eval-checkpoint"
            eval_checkpoint = args[i + 1]
            i += 2
        elseif arg == "--oracle-ladder-checkpoint"
            oracle_ladder_checkpoint = args[i + 1]
            i += 2
        elseif arg == "--mention-sweep-checkpoint"
            mention_sweep_checkpoint = args[i + 1]
            i += 2
        elseif arg == "--threshold-sweep-checkpoint"
            threshold_sweep_checkpoint = args[i + 1]
            i += 2
        elseif arg == "--margin-sweep-checkpoint"
            margin_sweep_checkpoint = args[i + 1]
            i += 2
        elseif arg == "--nonnull-sweep-checkpoint"
            nonnull_sweep_checkpoint = args[i + 1]
            i += 2
        elseif arg == "--auto-calibrate-checkpoint"
            auto_calibrate_checkpoint = args[i + 1]
            i += 2
        elseif arg == "--evidence-pooling-sweep-checkpoint"
            evidence_pooling_sweep_checkpoint = args[i + 1]
            i += 2
        elseif arg == "--evidence-pooling-modes"
            evidence_pooling_modes = [Symbol(strip(v)) for v in split(args[i + 1], ",") if !isempty(strip(v))]
            i += 2
        elseif arg == "--pair-sweep-checkpoint"
            pair_sweep_checkpoint = args[i + 1]
            i += 2
        elseif arg == "--checkpoint-sweep"
            checkpoint_sweep = [strip(v) for v in split(args[i + 1], ",") if !isempty(strip(v))]
            i += 2
        elseif arg == "--mention-sweep-budgets"
            mention_sweep_budgets = [parse(Int, strip(v)) for v in split(args[i + 1], ",") if !isempty(strip(v))]
            i += 2
        elseif arg == "--mention-sweep-modes"
            mention_sweep_modes = [Symbol(strip(v)) for v in split(args[i + 1], ",") if !isempty(strip(v))]
            i += 2
        elseif arg == "--threshold-sweep-values"
            threshold_sweep_values = [Float32(parse(Float64, strip(v))) for v in split(args[i + 1], ",") if !isempty(strip(v))]
            i += 2
        elseif arg == "--threshold-sweep-margin"
            threshold_sweep_margin = Float32(parse(Float64, args[i + 1]))
            i += 2
        elseif arg == "--threshold-sweep-nonnull"
            threshold_sweep_nonnull = Float32(parse(Float64, args[i + 1]))
            i += 2
        elseif arg == "--margin-sweep-values"
            margin_sweep_values = [Float32(parse(Float64, strip(v))) for v in split(args[i + 1], ",") if !isempty(strip(v))]
            i += 2
        elseif arg == "--nonnull-sweep-values"
            nonnull_sweep_values = [Float32(parse(Float64, strip(v))) for v in split(args[i + 1], ",") if !isempty(strip(v))]
            i += 2
        elseif arg == "--nonnull-sweep-confidence"
            nonnull_sweep_confidence = Float32(parse(Float64, args[i + 1]))
            i += 2
        elseif arg == "--nonnull-sweep-margin"
            nonnull_sweep_margin = Float32(parse(Float64, args[i + 1]))
            i += 2
        elseif arg == "--decode-head-cap"
            decode_head_cap = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--decode-tail-cap"
            decode_tail_cap = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--per-relation-thresholds"
            per_relation_thresholds = args[i + 1]
            i += 2
        elseif arg == "--auto-calibrate-threshold"
            auto_calibrate_threshold = Float32(parse(Float64, args[i + 1]))
            i += 2
        elseif arg == "--auto-calibrate-margin"
            auto_calibrate_margin = Float32(parse(Float64, args[i + 1]))
            i += 2
        elseif arg == "--auto-calibrate-nonnull"
            auto_calibrate_nonnull = Float32(parse(Float64, args[i + 1]))
            i += 2
        elseif arg == "--auto-calibrate-min-predictions"
            auto_calibrate_min_predictions = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--auto-calibrate-thresholds"
            auto_calibrate_thresholds = [Float32(parse(Float64, strip(v))) for v in split(args[i + 1], ",") if !isempty(strip(v))]
            i += 2
        elseif arg == "--type-constraints-mode"
            type_constraints_mode = lowercase(strip(args[i + 1]))
            i += 2
        elseif arg == "--type-constraints-min-count"
            type_constraints_min_count = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--relation-consistency-mode"
            relation_consistency_mode = lowercase(strip(args[i + 1]))
            i += 2
        elseif arg == "--relation-consistency-min-count"
            relation_consistency_min_count = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--pair-sweep-budgets"
            pair_sweep_budgets = [parse(Int, strip(v)) for v in split(args[i + 1], ",") if !isempty(strip(v))]
            i += 2
        elseif arg == "--pair-sweep-overgenerate"
            pair_sweep_overgenerate = [parse(Int, strip(v)) for v in split(args[i + 1], ",") if !isempty(strip(v))]
            i += 2
        elseif arg == "--max-eval-batches"
            max_eval_batches_override = parse(Int, args[i + 1])
            i += 2
        else
            error("Unknown argument: $(arg)")
        end
    end

    type_constraints_mode in ("off", "hard") ||
        error("Unsupported --type-constraints-mode=$(type_constraints_mode). Supported: off, hard.")
    type_constraints_min_count >= 1 ||
        error("--type-constraints-min-count must be >= 1, got $(type_constraints_min_count).")
    relation_consistency_mode in ("off", "resolve") ||
        error("Unsupported --relation-consistency-mode=$(relation_consistency_mode). Supported: off, resolve.")
    relation_consistency_min_count >= 1 ||
        error("--relation-consistency-min-count must be >= 1, got $(relation_consistency_min_count).")
    if !isempty(evidence_pooling_modes)
        all(mode -> mode in (:token, :sentence, :hybrid), evidence_pooling_modes) ||
            error("Unsupported value in --evidence-pooling-modes. Supported modes: token,sentence,hybrid.")
    end

    return RECLIOptions(;
        config_path = config_path,
        resume_path = resume_path,
        max_steps_override = max_steps_override,
        eval_checkpoint = eval_checkpoint,
        oracle_ladder_checkpoint = oracle_ladder_checkpoint,
        mention_sweep_checkpoint = mention_sweep_checkpoint,
        threshold_sweep_checkpoint = threshold_sweep_checkpoint,
        margin_sweep_checkpoint = margin_sweep_checkpoint,
        nonnull_sweep_checkpoint = nonnull_sweep_checkpoint,
        auto_calibrate_checkpoint = auto_calibrate_checkpoint,
        evidence_pooling_sweep_checkpoint = evidence_pooling_sweep_checkpoint,
        pair_sweep_checkpoint = pair_sweep_checkpoint,
        checkpoint_sweep = checkpoint_sweep,
        mention_sweep_budgets = mention_sweep_budgets,
        mention_sweep_modes = mention_sweep_modes,
        threshold_sweep_values = threshold_sweep_values,
        threshold_sweep_margin = threshold_sweep_margin,
        threshold_sweep_nonnull = threshold_sweep_nonnull,
        margin_sweep_values = margin_sweep_values,
        nonnull_sweep_values = nonnull_sweep_values,
        nonnull_sweep_confidence = nonnull_sweep_confidence,
        nonnull_sweep_margin = nonnull_sweep_margin,
        decode_head_cap = decode_head_cap,
        decode_tail_cap = decode_tail_cap,
        per_relation_thresholds = per_relation_thresholds,
        auto_calibrate_threshold = auto_calibrate_threshold,
        auto_calibrate_margin = auto_calibrate_margin,
        auto_calibrate_nonnull = auto_calibrate_nonnull,
        auto_calibrate_min_predictions = auto_calibrate_min_predictions,
        auto_calibrate_thresholds = auto_calibrate_thresholds,
        type_constraints_mode = type_constraints_mode,
        type_constraints_min_count = type_constraints_min_count,
        relation_consistency_mode = relation_consistency_mode,
        relation_consistency_min_count = relation_consistency_min_count,
        evidence_pooling_modes = evidence_pooling_modes,
        pair_sweep_budgets = pair_sweep_budgets,
        pair_sweep_overgenerate = pair_sweep_overgenerate,
        max_eval_batches_override = max_eval_batches_override,
    )
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
        hard_negative_ratio = Float32(get(training, "hard_negative_ratio", 2.0)),
        mention_negative_ratio = Float32(get(training, "mention_negative_ratio", 1.0)),
        proposal_train_probability = Float32(get(training, "proposal_train_probability", 0.0)),
        proposal_loss_weight = Float32(get(training, "proposal_loss_weight", 0.0)),
        proposal_warmup_steps = get(training, "proposal_warmup_steps", 0),
        max_len = get(data_cfg, "max_len", get(get(data, "model", Dict{String,Any}()), "max_sequence_length", 256)),
        seed = get(training, "seed", 42),
        max_eval_batches = get(training, "max_eval_batches", 8),
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
            mention_negative_ratio = run_cfg.mention_negative_ratio,
            proposal_train_probability = run_cfg.proposal_train_probability,
            proposal_loss_weight = run_cfg.proposal_loss_weight,
            proposal_warmup_steps = run_cfg.proposal_warmup_steps,
            max_len = run_cfg.max_len,
            seed = run_cfg.seed,
            max_eval_batches = run_cfg.max_eval_batches,
            resume_path = run_cfg.resume_path,
        )
    end

    return run_cfg
end

function load_null_relation_weight(config_path::String)::Float32
    data = TOML.parsefile(config_path)
    training = get(data, "training", Dict{String,Any}())
    return Float32(get(training, "null_relation_weight", 1.0))
end

function load_relation_focal_gamma(config_path::String)::Float32
    data = TOML.parsefile(config_path)
    training = get(data, "training", Dict{String,Any}())
    gamma = Float32(get(training, "relation_focal_gamma", 0.0))
    gamma >= 0.0f0 || error("training.relation_focal_gamma must be >= 0, got $(gamma)")
    return gamma
end

function load_positive_relation_weight(config_path::String)::Float32
    data = TOML.parsefile(config_path)
    training = get(data, "training", Dict{String,Any}())
    weight = Float32(get(training, "positive_relation_weight", 1.0))
    weight > 0.0f0 || error("training.positive_relation_weight must be > 0, got $(weight)")
    return weight
end

function load_relation_logit_adjustment_tau(config_path::String)::Float32
    data = TOML.parsefile(config_path)
    training = get(data, "training", Dict{String,Any}())
    tau = Float32(get(training, "relation_logit_adjustment_tau", 0.0))
    tau >= 0.0f0 || error("training.relation_logit_adjustment_tau must be >= 0, got $(tau)")
    return tau
end

function load_distillation_settings(config_path::String)
    data = TOML.parsefile(config_path)
    training = get(data, "training", Dict{String,Any}())
    entity_weight = Float32(get(training, "teacher_entity_loss_weight", 0.0))
    relation_weight = Float32(get(training, "teacher_relation_loss_weight", 0.0))
    confidence_weight = Float32(get(training, "teacher_confidence_loss_weight", 0.0))
    allow_missing_teacher_targets = Bool(get(training, "allow_missing_teacher_targets", false))
    entity_weight >= 0.0f0 || error("training.teacher_entity_loss_weight must be >= 0, got $(entity_weight)")
    relation_weight >= 0.0f0 || error("training.teacher_relation_loss_weight must be >= 0, got $(relation_weight)")
    confidence_weight >= 0.0f0 || error("training.teacher_confidence_loss_weight must be >= 0, got $(confidence_weight)")
    return (
        entity_weight = entity_weight,
        relation_weight = relation_weight,
        confidence_weight = confidence_weight,
        allow_missing_teacher_targets = allow_missing_teacher_targets,
    )
end

function summarize_teacher_payloads(rows)
    rows_with_teacher_entities = 0
    rows_with_teacher_relations = 0
    teacher_entity_total = 0
    teacher_relation_total = 0
    for row in rows
        if haskey(row, :teacher_entities)
            teacher_entities = collect(row.teacher_entities)
            rows_with_teacher_entities += 1
            teacher_entity_total += length(teacher_entities)
        end
        if haskey(row, :teacher_relations)
            teacher_relations = collect(row.teacher_relations)
            rows_with_teacher_relations += 1
            teacher_relation_total += length(teacher_relations)
        end
    end
    return (
        rows_with_teacher_entities = rows_with_teacher_entities,
        rows_with_teacher_relations = rows_with_teacher_relations,
        teacher_entity_total = teacher_entity_total,
        teacher_relation_total = teacher_relation_total,
    )
end

function validate_teacher_payload_coverage(rows, distillation_settings; context::String = "train")
    requested = distillation_settings.entity_weight > 0.0f0 ||
                distillation_settings.relation_weight > 0.0f0 ||
                distillation_settings.confidence_weight > 0.0f0
    requested || return nothing

    coverage = summarize_teacher_payloads(rows)
    println(
        "Teacher payload coverage ($context): " *
        "entity_rows=$(coverage.rows_with_teacher_entities), " *
        "relation_rows=$(coverage.rows_with_teacher_relations), " *
        "entity_targets=$(coverage.teacher_entity_total), " *
        "relation_targets=$(coverage.teacher_relation_total)"
    )

    missing_entity_targets = distillation_settings.entity_weight > 0.0f0 &&
                             coverage.teacher_entity_total == 0
    missing_relation_targets = (distillation_settings.relation_weight > 0.0f0 ||
                                distillation_settings.confidence_weight > 0.0f0) &&
                               coverage.teacher_relation_total == 0

    if !distillation_settings.allow_missing_teacher_targets &&
       (missing_entity_targets || missing_relation_targets)
        missing_parts = String[]
        missing_entity_targets && push!(missing_parts, "teacher_entities")
        missing_relation_targets && push!(missing_parts, "teacher_relations")
        missing_spec = join(missing_parts, " and ")
        error(
            "Distillation losses are enabled but $(missing_spec) are missing in the $context rows. " *
            "Add teacher payloads or set training.allow_missing_teacher_targets=true for plumbing-only smoke runs."
        )
    end

    if distillation_settings.allow_missing_teacher_targets &&
       (missing_entity_targets || missing_relation_targets)
        println("Warning: proceeding without teacher payload coverage because training.allow_missing_teacher_targets=true")
    end

    return coverage
end

function load_edge_ranking_settings(config_path::String)
    data = TOML.parsefile(config_path)
    training = get(data, "training", Dict{String,Any}())
    return (
        weight = Float32(get(training, "edge_ranking_loss_weight", 0.0)),
        margin = Float32(get(training, "edge_ranking_margin", 0.2)),
        hard_negatives = get(training, "edge_ranking_hard_negatives", 16),
        start_step = get(training, "edge_ranking_start_step", 0),
        warmup_steps = get(training, "edge_ranking_warmup_steps", 0),
    )
end

function load_retrieval_bias_settings(config_path::String)
    data = TOML.parsefile(config_path)
    relation = get(data, "relation_extraction", Dict{String,Any}())
    return (
        distance_scale = Float32(get(relation, "retrieval_distance_bias_scale", 0.0)),
        type_scale = Float32(get(relation, "retrieval_type_bias_scale", 0.0)),
        sentence_scale = Float32(get(relation, "retrieval_sentence_bias_scale", 0.0)),
        local_scale = Float32(get(relation, "retrieval_local_bias_scale", 0.0)),
        sentence_embedding_scale = Float32(get(relation, "retrieval_sentence_embedding_scale", 0.0)),
        type_compat_scale = Float32(get(relation, "retrieval_type_compat_bias_scale", 0.0)),
        dot_scale = Float32(get(relation, "retrieval_dot_bias_scale", 0.0)),
        compatibility_scale = Float32(get(relation, "retrieval_compatibility_scale", 0.0)),
        span_context_use_adjacent = get(relation, "span_context_use_adjacent", true),
        span_context_use_sentence = get(relation, "span_context_use_sentence", true),
        span_context_use_semantic = get(relation, "span_context_use_semantic", true),
        span_context_start_step = get(relation, "span_context_start_step", 0),
        span_context_adjacent_start_step = get(relation, "span_context_adjacent_start_step", 0),
        span_context_sentence_start_step = get(relation, "span_context_sentence_start_step", 0),
        span_context_semantic_start_step = get(relation, "span_context_semantic_start_step", 0),
        edge_v2_semantic_topk = get(relation, "edge_v2_semantic_topk", 0),
        edge_v2_reverse_topk = get(relation, "edge_v2_reverse_topk", 0),
        edge_v2_global_reserve = get(relation, "edge_v2_global_reserve", 0),
        edge_v2_semantic_score_scale = Float32(get(relation, "edge_v2_semantic_score_scale", 1.0)),
        edge_v2_span_score_scale = Float32(get(relation, "edge_v2_span_score_scale", 1.0)),
        edge_v2_distance_penalty = Float32(get(relation, "edge_v2_distance_penalty", 0.0)),
        edge_v2_require_mutual = get(relation, "edge_v2_require_mutual", false),
        edge_v2_use_local_neighbors = get(relation, "edge_v2_use_local_neighbors", true),
        edge_v2_use_routed_buckets = get(relation, "edge_v2_use_routed_buckets", true),
        edge_v2_use_semantic_topk = get(relation, "edge_v2_use_semantic_topk", true),
        edge_v2_use_global_reserve = get(relation, "edge_v2_use_global_reserve", true),
    )
end

@inline function with_retrieval_bias_inputs(inputs, settings; step::Union{Nothing,Int} = nothing)
    context_enabled = step === nothing || step >= settings.span_context_start_step
    adjacent_on = Bool(settings.span_context_use_adjacent) &&
                  (step === nothing || step >= settings.span_context_adjacent_start_step)
    sentence_on = Bool(settings.span_context_use_sentence) &&
                  (step === nothing || step >= settings.span_context_sentence_start_step)
    semantic_on = Bool(settings.span_context_use_semantic) &&
                  (step === nothing || step >= settings.span_context_semantic_start_step)
    return merge(
        inputs,
        (
            retrieval_distance_bias_scale = settings.distance_scale,
            retrieval_type_bias_scale = settings.type_scale,
            retrieval_sentence_bias_scale = settings.sentence_scale,
            retrieval_local_bias_scale = settings.local_scale,
            retrieval_sentence_embedding_scale = settings.sentence_embedding_scale,
            retrieval_type_compat_bias_scale = settings.type_compat_scale,
            retrieval_dot_bias_scale = settings.dot_scale,
            retrieval_compatibility_scale = settings.compatibility_scale,
            span_context_enabled = context_enabled,
            span_context_use_adjacent = adjacent_on,
            span_context_use_sentence = sentence_on,
            span_context_use_semantic = semantic_on,
            edge_v2_semantic_topk = settings.edge_v2_semantic_topk,
            edge_v2_reverse_topk = settings.edge_v2_reverse_topk,
            edge_v2_global_reserve = settings.edge_v2_global_reserve,
            edge_v2_semantic_score_scale = settings.edge_v2_semantic_score_scale,
            edge_v2_span_score_scale = settings.edge_v2_span_score_scale,
            edge_v2_distance_penalty = settings.edge_v2_distance_penalty,
            edge_v2_require_mutual = settings.edge_v2_require_mutual,
            edge_v2_use_local_neighbors = settings.edge_v2_use_local_neighbors,
            edge_v2_use_routed_buckets = settings.edge_v2_use_routed_buckets,
            edge_v2_use_semantic_topk = settings.edge_v2_use_semantic_topk,
            edge_v2_use_global_reserve = settings.edge_v2_use_global_reserve,
        ),
    )
end

function edge_ranking_weight_for_step(settings, step::Int)::Float32
    base = settings.weight
    base <= 0.0f0 && return 0.0f0
    start_step = settings.start_step
    warmup_steps = settings.warmup_steps
    step < start_step && return 0.0f0
    warmup_steps <= 0 && return base
    progress = Float32(step - start_step + 1) / Float32(warmup_steps)
    return base * clamp(progress, 0.0f0, 1.0f0)
end

function load_pair_proposer_settings(path::String)
    data = TOML.parsefile(path)
    relation = get(data, "relation_extraction", Dict{String,Any}())
    mode_raw = get(relation, "pair_proposer_mode", "local")
    return (
        mode = mode_raw isa Symbol ? mode_raw : Symbol(mode_raw),
        global_top_spans = get(relation, "pair_global_top_spans", 0),
    )
end

function with_label_counts(config::RelationExtractionConfig, num_entity_labels::Int, num_relations::Int, vocab_size::Int)
    return RelationExtractionConfig(
        vocab_size = vocab_size,
        max_sequence_length = config.max_sequence_length,
        embedding_dimension = config.embedding_dimension,
        number_of_heads = config.number_of_heads,
        number_of_layers = config.number_of_layers,
        number_of_refinement_layers = config.number_of_refinement_layers,
        use_interleaved_local_wave = config.use_interleaved_local_wave,
        interleaved_block_type = config.interleaved_block_type,
        local_wave_ratio = config.local_wave_ratio,
        interleaved_use_local_attention = config.interleaved_use_local_attention,
        interleaved_use_wave_pde = config.interleaved_use_wave_pde,
        num_entity_labels = num_entity_labels,
        num_relations = num_relations,
        time_dimension = config.time_dimension,
        state_dimension = config.state_dimension,
        window_size = config.window_size,
        local_operator = config.local_operator,
        residual_mode = config.residual_mode,
        hyper_connection_width = config.hyper_connection_width,
        hyper_connection_sinkhorn_iterations = config.hyper_connection_sinkhorn_iterations,
        min_frequency = config.min_frequency,
        max_frequency = config.max_frequency,
        default_time_step = config.default_time_step,
        dropout_rate = config.dropout_rate,
        use_ffn = config.use_ffn,
        ffn_expansion = config.ffn_expansion,
        use_output_projection = config.use_output_projection,
        use_parallel_scan = config.use_parallel_scan,
        parallel_chunk_size = config.parallel_chunk_size,
        use_vector_gains = config.use_vector_gains,
        use_per_head_alpha = config.use_per_head_alpha,
        use_branch_projections = config.use_branch_projections,
        max_candidate_spans = config.max_candidate_spans,
        max_candidate_pairs = config.max_candidate_pairs,
        max_span_width = config.max_span_width,
        span_context_layers = config.span_context_layers,
        span_context_neighbor_radius = config.span_context_neighbor_radius,
        span_context_topk = config.span_context_topk,
        biaffine_rank = config.biaffine_rank,
        pair_neighbor_radius = config.pair_neighbor_radius,
        pair_proposer_mode = config.pair_proposer_mode,
        pair_global_top_spans = config.pair_global_top_spans,
        pair_router_dimension = config.pair_router_dimension,
        pair_router_buckets = config.pair_router_buckets,
        pair_router_topk = config.pair_router_topk,
        pair_router_routes_per_span = config.pair_router_routes_per_span,
        pair_router_score_scale = config.pair_router_score_scale,
        pair_overgenerate_factor = config.pair_overgenerate_factor,
        pair_retrieval_dimension = config.pair_retrieval_dimension,
        pair_distance_buckets = config.pair_distance_buckets,
        pair_retrieval_loss_weight = config.pair_retrieval_loss_weight,
        pair_evidence_dimension = config.pair_evidence_dimension,
        relation_decoder_mode = config.relation_decoder_mode,
        relation_decoder_residual_scale = config.relation_decoder_residual_scale,
        mention_score_mode = config.mention_score_mode,
        mention_score_learned_weight = config.mention_score_learned_weight,
    )
end

function retrieval_hard_negative_ranking_loss(
    retrieval_logits,
    relation_targets,
    relation_mask;
    margin::Float32 = 0.2f0,
    hard_negatives::Int = 16,
)
    hard_negatives <= 0 && return 0.0f0

    logits_are_3d = ndims(retrieval_logits) == 3
    logits_cpu = ChainRulesCore.ignore_derivatives() do
        logits_are_3d ? Array(@view(retrieval_logits[1, :, :])) : Array(retrieval_logits)
    end
    targets_cpu = ChainRulesCore.ignore_derivatives() do
        Float32.(to_cpu(relation_targets))
    end
    mask_cpu = ChainRulesCore.ignore_derivatives() do
        Bool.(to_cpu(relation_mask))
    end

    total_loss = 0.0f0
    contributing_batches = 0
    for b in axes(logits_cpu, 2)
        valid_mask = @view(mask_cpu[:, b])
        target_mask = @view(targets_cpu[:, b])
        pos_idx = findall(valid_mask .& (target_mask .> 0.5f0))
        neg_idx = findall(valid_mask .& (target_mask .<= 0.5f0))
        isempty(pos_idx) && continue
        isempty(neg_idx) && continue

        neg_scores_cpu = logits_cpu[neg_idx, b]
        neg_order = sortperm(neg_scores_cpu; rev = true)
        keep = min(hard_negatives, length(neg_order))
        keep > 0 || continue
        hard_neg_idx = neg_idx[neg_order[1:keep]]

        pos_scores = logits_are_3d ?
            reshape(retrieval_logits[1, pos_idx, b], :, 1) :
            reshape(retrieval_logits[pos_idx, b], :, 1)
        neg_scores = logits_are_3d ?
            reshape(retrieval_logits[1, hard_neg_idx, b], 1, :) :
            reshape(retrieval_logits[hard_neg_idx, b], 1, :)
        hinge = max.(0.0f0, margin .- pos_scores .+ neg_scores)
        total_loss += Float32(sum(hinge) / (length(pos_idx) * length(hard_neg_idx)))
        contributing_batches += 1
    end

    return contributing_batches > 0 ? total_loss / Float32(contributing_batches) : 0.0f0
end

function relation_loss(
    outputs,
    targets;
    null_relation_weight::Float32 = 1.0f0,
    positive_relation_weight::Float32 = 1.0f0,
    no_relation_id::Int = 1,
    relation_focal_gamma::Float32 = 0.0f0,
    relation_logit_adjustment_tau::Float32 = 0.0f0,
    relation_logit_adjustment::Union{Nothing,Vector{Float32}} = nothing,
    teacher_entity_loss_weight::Float32 = 0.0f0,
    teacher_relation_loss_weight::Float32 = 0.0f0,
    teacher_confidence_loss_weight::Float32 = 0.0f0,
    edge_ranking_loss_weight::Float32 = 0.0f0,
    edge_ranking_margin::Float32 = 0.2f0,
    edge_ranking_hard_negatives::Int = 16,
)
    retrieval_bce = outputs.retrieval_loss_weight * confidence_bce(outputs.retrieval_logits, targets.relation_targets, targets.relation_mask)
    retrieval_rank = edge_ranking_loss_weight > 0.0f0 ? (
        edge_ranking_loss_weight * retrieval_hard_negative_ranking_loss(
            outputs.retrieval_logits,
            targets.relation_targets,
            targets.relation_mask;
            margin = edge_ranking_margin,
            hard_negatives = edge_ranking_hard_negatives,
        )
    ) : 0.0f0
    supervised = entity_cross_entropy(outputs.entity_logits, targets.entity_labels) +
           boundary_bce(outputs.boundary_logits, targets.boundary_labels) +
           Swamma.RelationExtraction.mention_bce(
               outputs.mention_logits,
               targets.mention_labels,
               targets.mention_mask,
           ) +
           retrieval_bce + retrieval_rank +
           relation_cross_entropy(
               outputs.relation_logits,
               targets.relation_labels,
               targets.relation_mask;
               null_relation_weight = null_relation_weight,
               positive_relation_weight = positive_relation_weight,
               no_relation_id = no_relation_id,
               focal_gamma = relation_focal_gamma,
               logit_adjustment_tau = relation_logit_adjustment_tau,
               logit_adjustment = relation_logit_adjustment,
           ) +
           confidence_bce(outputs.confidence_logits, targets.relation_targets, targets.relation_mask)

    teacher_entity = teacher_entity_loss_weight > 0.0f0 ?
        entity_cross_entropy(outputs.entity_logits, targets.teacher_entity_labels) :
        0.0f0
    teacher_relation = teacher_relation_loss_weight > 0.0f0 ?
        relation_cross_entropy(
            outputs.relation_logits,
            targets.teacher_relation_labels,
            targets.teacher_relation_mask;
            null_relation_weight = 1.0f0,
            positive_relation_weight = 1.0f0,
            no_relation_id = no_relation_id,
            focal_gamma = 0.0f0,
            logit_adjustment_tau = 0.0f0,
            logit_adjustment = nothing,
        ) :
        0.0f0
    teacher_confidence = teacher_confidence_loss_weight > 0.0f0 ?
        confidence_bce(outputs.confidence_logits, targets.teacher_confidence_targets, targets.teacher_confidence_mask) :
        0.0f0
    distillation = teacher_entity_loss_weight * teacher_entity +
                   teacher_relation_loss_weight * teacher_relation +
                   teacher_confidence_loss_weight * teacher_confidence

    return supervised + distillation
end

function relation_loss_breakdown(
    outputs,
    targets;
    null_relation_weight::Float32 = 1.0f0,
    positive_relation_weight::Float32 = 1.0f0,
    no_relation_id::Int = 1,
    relation_focal_gamma::Float32 = 0.0f0,
    relation_logit_adjustment_tau::Float32 = 0.0f0,
    relation_logit_adjustment::Union{Nothing,Vector{Float32}} = nothing,
    teacher_entity_loss_weight::Float32 = 0.0f0,
    teacher_relation_loss_weight::Float32 = 0.0f0,
    teacher_confidence_loss_weight::Float32 = 0.0f0,
    edge_ranking_loss_weight::Float32 = 0.0f0,
    edge_ranking_margin::Float32 = 0.2f0,
    edge_ranking_hard_negatives::Int = 16,
)
    entity = Float32(entity_cross_entropy(outputs.entity_logits, targets.entity_labels))
    boundary = Float32(boundary_bce(outputs.boundary_logits, targets.boundary_labels))
    mention = Float32(
        Swamma.RelationExtraction.mention_bce(
            outputs.mention_logits,
            targets.mention_labels,
            targets.mention_mask,
        )
    )
    retrieval_bce = Float32(
        outputs.retrieval_loss_weight * confidence_bce(outputs.retrieval_logits, targets.relation_targets, targets.relation_mask)
    )
    retrieval_rank = Float32(
        edge_ranking_loss_weight > 0.0f0 ? (
            edge_ranking_loss_weight * retrieval_hard_negative_ranking_loss(
                outputs.retrieval_logits,
                targets.relation_targets,
                targets.relation_mask;
                margin = edge_ranking_margin,
                hard_negatives = edge_ranking_hard_negatives,
            )
        ) : 0.0f0
    )
    retrieval = retrieval_bce + retrieval_rank
    relation = Float32(
        relation_cross_entropy(
            outputs.relation_logits,
            targets.relation_labels,
            targets.relation_mask;
            null_relation_weight = null_relation_weight,
            positive_relation_weight = positive_relation_weight,
            no_relation_id = no_relation_id,
            focal_gamma = relation_focal_gamma,
            logit_adjustment_tau = relation_logit_adjustment_tau,
            logit_adjustment = relation_logit_adjustment,
        )
    )
    confidence = Float32(
        confidence_bce(outputs.confidence_logits, targets.relation_targets, targets.relation_mask)
    )
    teacher_entity = Float32(
        teacher_entity_loss_weight > 0.0f0 ?
        entity_cross_entropy(outputs.entity_logits, targets.teacher_entity_labels) :
        0.0f0
    )
    teacher_relation = Float32(
        teacher_relation_loss_weight > 0.0f0 ?
        relation_cross_entropy(
            outputs.relation_logits,
            targets.teacher_relation_labels,
            targets.teacher_relation_mask;
            null_relation_weight = 1.0f0,
            positive_relation_weight = 1.0f0,
            no_relation_id = no_relation_id,
            focal_gamma = 0.0f0,
            logit_adjustment_tau = 0.0f0,
            logit_adjustment = nothing,
        ) :
        0.0f0
    )
    teacher_confidence = Float32(
        teacher_confidence_loss_weight > 0.0f0 ?
        confidence_bce(outputs.confidence_logits, targets.teacher_confidence_targets, targets.teacher_confidence_mask) :
        0.0f0
    )
    distillation = teacher_entity_loss_weight * teacher_entity +
                   teacher_relation_loss_weight * teacher_relation +
                   teacher_confidence_loss_weight * teacher_confidence
    total = entity + boundary + mention + retrieval + relation + confidence + distillation
    return (
        entity = entity,
        boundary = boundary,
        mention = mention,
        retrieval = retrieval,
        retrieval_rank = retrieval_rank,
        relation = relation,
        confidence = confidence,
        teacher_entity = teacher_entity,
        teacher_relation = teacher_relation,
        teacher_confidence = teacher_confidence,
        distillation = distillation,
        total = total,
    )
end

function proposal_training_loss(
    proposal_outputs,
    proposal_targets;
    null_relation_weight::Float32 = 1.0f0,
    positive_relation_weight::Float32 = 1.0f0,
    no_relation_id::Int = 1,
    relation_focal_gamma::Float32 = 0.0f0,
    relation_logit_adjustment_tau::Float32 = 0.0f0,
    relation_logit_adjustment::Union{Nothing,Vector{Float32}} = nothing,
    edge_ranking_loss_weight::Float32 = 0.0f0,
    edge_ranking_margin::Float32 = 0.2f0,
    edge_ranking_hard_negatives::Int = 16,
)
    retrieval_bce = Float32(
        proposal_outputs.retrieval_loss_weight * confidence_bce(
            proposal_outputs.retrieval_logits,
            proposal_targets.relation_targets,
            proposal_targets.relation_mask,
        )
    )
    retrieval_rank = Float32(
        edge_ranking_loss_weight > 0.0f0 ? (
            edge_ranking_loss_weight * retrieval_hard_negative_ranking_loss(
                proposal_outputs.retrieval_logits,
                proposal_targets.relation_targets,
                proposal_targets.relation_mask;
                margin = edge_ranking_margin,
                hard_negatives = edge_ranking_hard_negatives,
            )
        ) : 0.0f0
    )
    retrieval = retrieval_bce + retrieval_rank
    relation = Float32(
        relation_cross_entropy(
            proposal_outputs.relation_logits,
            proposal_targets.relation_labels,
            proposal_targets.relation_mask,
            null_relation_weight = null_relation_weight,
            positive_relation_weight = positive_relation_weight,
            no_relation_id = no_relation_id,
            focal_gamma = relation_focal_gamma,
            logit_adjustment_tau = relation_logit_adjustment_tau,
            logit_adjustment = relation_logit_adjustment,
        )
    )
    confidence = Float32(
        confidence_bce(
            proposal_outputs.confidence_logits,
            proposal_targets.relation_targets,
            proposal_targets.relation_mask,
        )
    )
    total = retrieval + relation + confidence
    return (
        retrieval = retrieval,
        retrieval_rank = retrieval_rank,
        relation = relation,
        confidence = confidence,
        total = total,
    )
end

function build_exhaustive_relation_pairs(spans, span_mask)
    spans_cpu = to_cpu(spans)
    span_mask_cpu = to_cpu(span_mask)
    batch_size = size(spans_cpu, 3)
    pair_lists = Vector{Vector{Tuple{Int, Int}}}(undef, batch_size)
    max_pairs = 0

    for b in 1:batch_size
        valid_indices = findall(@view(span_mask_cpu[:, b]))
        pairs = Tuple{Int, Int}[]
        for head_idx in valid_indices
            for tail_idx in valid_indices
                head_idx == tail_idx && continue
                push!(pairs, (head_idx, tail_idx))
            end
        end
        pair_lists[b] = pairs
        max_pairs = max(max_pairs, length(pairs))
    end

    relation_pairs = zeros(Int, 2, max_pairs, batch_size)
    relation_mask = falses(max_pairs, batch_size)
    for b in 1:batch_size
        for (pair_idx, (head_idx, tail_idx)) in enumerate(pair_lists[b])
            relation_pairs[1, pair_idx, b] = head_idx
            relation_pairs[2, pair_idx, b] = tail_idx
            relation_mask[pair_idx, b] = true
        end
    end

    if spans isa CUDA.CuArray
        return CUDA.CuArray(relation_pairs), CUDA.CuArray(relation_mask)
    end
    return relation_pairs, relation_mask
end

function oracle_mode_stats(
    outputs,
    targets;
    no_relation_id::Int,
    confidence_threshold::Float32 = 0.5f0,
    no_relation_margin::Float32 = 0.0f0,
    nonnull_probability_threshold::Float32 = 0.0f0,
    max_relations_per_head::Int = 0,
    max_relations_per_tail::Int = 0,
    relation_confidence_thresholds::Dict{Int,Float32} = Dict{Int,Float32}(),
    relation_allowed_type_pairs::Dict{Int,Set{Tuple{Int,Int}}} = Dict{Int,Set{Tuple{Int,Int}}}(),
    span_type_to_token_label_ids::Dict{Int,Tuple{Int,Int}} = Dict{Int,Tuple{Int,Int}}(),
    symmetric_relations::Set{Int} = Set{Int}(),
    inverse_relation_map::Dict{Int,Int} = Dict{Int,Int}(),
)
    entity = Float32(entity_cross_entropy(outputs.entity_logits, targets.entity_labels))
    boundary = Float32(boundary_bce(outputs.boundary_logits, targets.boundary_labels))
    mention = if size(outputs.mention_logits) == size(targets.mention_labels)
        Float32(
            Swamma.RelationExtraction.mention_bce(
                outputs.mention_logits,
                targets.mention_labels,
                targets.mention_mask,
            )
        )
    else
        0.0f0
    end
    proposal_targets = build_proposal_relation_targets(outputs, targets, no_relation_id)
    proposal_losses = proposal_training_loss(outputs, proposal_targets)
    diagnostics = proposal_diagnostics(
        outputs,
        targets;
        no_relation_id = no_relation_id,
        confidence_threshold = confidence_threshold,
        no_relation_margin = no_relation_margin,
        nonnull_probability_threshold = nonnull_probability_threshold,
        max_relations_per_head = max_relations_per_head,
        max_relations_per_tail = max_relations_per_tail,
        relation_confidence_thresholds = relation_confidence_thresholds,
        relation_allowed_type_pairs = relation_allowed_type_pairs,
        span_type_to_token_label_ids = span_type_to_token_label_ids,
        symmetric_relations = symmetric_relations,
        inverse_relation_map = inverse_relation_map,
    )
    return (
        entity = entity,
        boundary = boundary,
        mention = mention,
        retrieval = proposal_losses.retrieval,
        relation = proposal_losses.relation,
        confidence = proposal_losses.confidence,
        total = entity + boundary + mention + proposal_losses.total,
        diagnostics = diagnostics,
    )
end

function zero_oracle_accumulator()
    return Dict{Symbol, Float64}(
        :total_loss => 0.0,
        :entity_loss => 0.0,
        :boundary_loss => 0.0,
        :mention_loss => 0.0,
        :retrieval_loss => 0.0,
        :relation_loss => 0.0,
        :confidence_loss => 0.0,
        :gold_spans => 0.0,
        :matched_spans => 0.0,
        :predicted_spans => 0.0,
        :gold_span_top8_hits => 0.0,
        :gold_span_top16_hits => 0.0,
        :gold_span_top32_hits => 0.0,
        :gold_pairs => 0.0,
        :matched_pairs => 0.0,
        :matched_pair_rank_sum => 0.0,
        :matched_pair_rank_count => 0.0,
        :gold_pair_top8_hits => 0.0,
        :gold_pair_top16_hits => 0.0,
        :gold_pair_top32_hits => 0.0,
        :gold_pair_short_total => 0.0,
        :gold_pair_medium_total => 0.0,
        :gold_pair_long_total => 0.0,
        :matched_pair_short_total => 0.0,
        :matched_pair_medium_total => 0.0,
        :matched_pair_long_total => 0.0,
        :gold_relations => 0.0,
        :predicted_relations => 0.0,
        :true_positive_relations => 0.0,
        :oracle_relation_coverage => 0.0,
    )
end

function accumulate_oracle_mode!(acc, mode_stats)
    acc[:total_loss] += mode_stats.total
    acc[:entity_loss] += mode_stats.entity
    acc[:boundary_loss] += mode_stats.boundary
    acc[:mention_loss] += mode_stats.mention
    acc[:retrieval_loss] += mode_stats.retrieval
    acc[:relation_loss] += mode_stats.relation
    acc[:confidence_loss] += mode_stats.confidence
    for key in keys(mode_stats.diagnostics)
        acc[key] += getproperty(mode_stats.diagnostics, key)
    end
    return acc
end

function finalize_oracle_mode(acc, batches::Int)
    gold_spans = Int(acc[:gold_spans])
    matched_spans = Int(acc[:matched_spans])
    predicted_spans = Int(acc[:predicted_spans])
    gold_pairs = Int(acc[:gold_pairs])
    matched_pairs = Int(acc[:matched_pairs])
    gold_relations = Int(acc[:gold_relations])
    predicted_relations = Int(acc[:predicted_relations])
    true_positive_relations = Int(acc[:true_positive_relations])
    matched_pair_rank_count = Int(acc[:matched_pair_rank_count])
    gold_pair_short_total = Int(acc[:gold_pair_short_total])
    gold_pair_medium_total = Int(acc[:gold_pair_medium_total])
    gold_pair_long_total = Int(acc[:gold_pair_long_total])
    matched_pair_short_total = Int(acc[:matched_pair_short_total])
    matched_pair_medium_total = Int(acc[:matched_pair_medium_total])
    matched_pair_long_total = Int(acc[:matched_pair_long_total])
    oracle_relation_coverage = Int(acc[:oracle_relation_coverage])

    mention_precision = safe_rate(matched_spans, predicted_spans)
    mention_recall = safe_rate(matched_spans, gold_spans)
    mention_f1 = mention_precision + mention_recall > 0 ?
        2f0 * mention_precision * mention_recall / (mention_precision + mention_recall) :
        0.0f0
    relation_precision = safe_rate(true_positive_relations, predicted_relations)
    relation_recall = safe_rate(true_positive_relations, gold_relations)
    relation_f1 = relation_precision + relation_recall > 0 ?
        2f0 * relation_precision * relation_recall / (relation_precision + relation_recall) :
        0.0f0
    matched_pair_rank_mean = matched_pair_rank_count > 0 ? Float32(acc[:matched_pair_rank_sum] / matched_pair_rank_count) : NaN32
    missed_pair_total = max(gold_pairs - matched_pairs, 0)
    missed_pair_short = max(gold_pair_short_total - matched_pair_short_total, 0)
    missed_pair_medium = max(gold_pair_medium_total - matched_pair_medium_total, 0)
    missed_pair_long = max(gold_pair_long_total - matched_pair_long_total, 0)

    return (
        total_loss = Float32(acc[:total_loss] / batches),
        entity_loss = Float32(acc[:entity_loss] / batches),
        boundary_loss = Float32(acc[:boundary_loss] / batches),
        mention_loss = Float32(acc[:mention_loss] / batches),
        retrieval_loss = Float32(acc[:retrieval_loss] / batches),
        relation_loss = Float32(acc[:relation_loss] / batches),
        confidence_loss = Float32(acc[:confidence_loss] / batches),
        mention_precision = mention_precision,
        mention_recall = mention_recall,
        mention_f1 = mention_f1,
        mention_top8_recall = safe_rate(Int(acc[:gold_span_top8_hits]), gold_spans),
        mention_top16_recall = safe_rate(Int(acc[:gold_span_top16_hits]), gold_spans),
        mention_top32_recall = safe_rate(Int(acc[:gold_span_top32_hits]), gold_spans),
        oracle_relation_coverage = safe_rate(oracle_relation_coverage, gold_relations),
        span_recall = safe_rate(matched_spans, gold_spans),
        pair_recall = safe_rate(matched_pairs, gold_pairs),
        pair_top8_recall = safe_rate(Int(acc[:gold_pair_top8_hits]), gold_pairs),
        pair_top16_recall = safe_rate(Int(acc[:gold_pair_top16_hits]), gold_pairs),
        pair_top32_recall = safe_rate(Int(acc[:gold_pair_top32_hits]), gold_pairs),
        matched_pair_rank_mean = matched_pair_rank_mean,
        pair_recall_short = safe_rate(matched_pair_short_total, gold_pair_short_total),
        pair_recall_medium = safe_rate(matched_pair_medium_total, gold_pair_medium_total),
        pair_recall_long = safe_rate(matched_pair_long_total, gold_pair_long_total),
        missed_pair_short_share = safe_rate(missed_pair_short, missed_pair_total),
        missed_pair_medium_share = safe_rate(missed_pair_medium, missed_pair_total),
        missed_pair_long_share = safe_rate(missed_pair_long, missed_pair_total),
        relation_precision = relation_precision,
        relation_recall = relation_recall,
        relation_f1 = relation_f1,
        gold_spans = gold_spans,
        gold_pairs = gold_pairs,
        gold_relations = gold_relations,
        predicted_relations = predicted_relations,
        true_positive_relations = true_positive_relations,
        batches = batches,
    )
end

function recent_mean(losses::Vector{Float32}, window::Int)
    isempty(losses) && return NaN32
    start_idx = max(1, length(losses) - window + 1)
    return mean(@view(losses[start_idx:end]))
end

function make_batch(
    rows,
    vocab,
    entity_label_to_id,
    relation_label_to_id,
    model_config,
    run_config;
    rng::AbstractRNG = Random.default_rng(),
)
    batch = prepare_rebel_batch(
        rows,
        vocab,
        entity_label_to_id,
        relation_label_to_id;
        max_len = run_config.max_len,
        max_candidate_spans = model_config.max_candidate_spans,
        max_candidate_pairs = model_config.max_candidate_pairs,
        max_span_width = model_config.max_span_width,
        hard_negative_ratio = run_config.hard_negative_ratio,
        mention_negative_ratio = run_config.mention_negative_ratio,
        rng = rng,
    )

    inputs = (
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
    )
    targets = (
        entity_labels = batch.entity_labels,
        boundary_labels = batch.boundary_labels,
        mention_labels = batch.mention_labels,
        mention_mask = batch.mention_mask,
        spans = batch.spans,
        span_mask = batch.span_supervision_mask,
        relation_labels = batch.relation_labels,
        relation_pairs = batch.relation_pairs,
        relation_mask = batch.relation_supervision_mask,
        relation_targets = batch.relation_targets,
        teacher_entity_labels = batch.teacher_entity_labels,
        teacher_relation_labels = batch.teacher_relation_labels,
        teacher_relation_mask = batch.teacher_relation_mask,
        teacher_confidence_targets = batch.teacher_confidence_targets,
        teacher_confidence_mask = batch.teacher_confidence_mask,
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

function collect_span_rank_map(spans, span_mask, batch_idx::Int)
    span_ranks = Dict{Tuple{Int, Int}, Int}()
    max_spans = size(spans, 2)
    for i in 1:max_spans
        span_mask[i, batch_idx] || continue
        span = (Int(spans[1, i, batch_idx]), Int(spans[2, i, batch_idx]))
        get!(span_ranks, span, i)
    end
    return span_ranks
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

function collect_pair_rank_map(spans, relation_pairs, relation_mask, batch_idx::Int)
    pair_ranks = Dict{NTuple{4, Int}, Int}()
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
        pair = (head[1], head[2], tail[1], tail[2])
        get!(pair_ranks, pair, i)
    end
    return pair_ranks
end

@inline function pair_distance_bucket(pair::NTuple{4, Int})
    distance = abs(pair[1] - pair[3])
    if distance <= 8
        return :short
    elseif distance <= 24
        return :medium
    else
        return :long
    end
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

function apply_relation_consistency(
    predictions::Vector{NamedTuple{(:relation, :label_id, :confidence),Tuple{NTuple{5,Int},Int,Float32}}},
    symmetric_relations::Set{Int},
    inverse_relation_map::Dict{Int,Int},
)
    (isempty(symmetric_relations) && isempty(inverse_relation_map)) && return predictions
    length(predictions) <= 1 && return predictions

    pair_to_index = Dict{NTuple{4,Int},Int}()
    for (idx, item) in enumerate(predictions)
        rel = item.relation
        pair_to_index[(rel[1], rel[2], rel[3], rel[4])] = idx
    end

    keep = trues(length(predictions))
    for i in eachindex(predictions)
        keep[i] || continue
        rel_i = predictions[i].relation
        reverse_pair = (rel_i[3], rel_i[4], rel_i[1], rel_i[2])
        j = get(pair_to_index, reverse_pair, 0)
        (j <= 0 || j == i || j < i || !keep[j]) && continue

        label_i = predictions[i].label_id
        label_j = predictions[j].label_id
        consistent = true
        if label_i in symmetric_relations
            consistent &= (label_j == label_i)
        elseif haskey(inverse_relation_map, label_i)
            consistent &= (label_j == inverse_relation_map[label_i])
        end
        if label_j in symmetric_relations
            consistent &= (label_i == label_j)
        elseif haskey(inverse_relation_map, label_j)
            consistent &= (label_i == inverse_relation_map[label_j])
        end

        if !consistent
            if predictions[i].confidence >= predictions[j].confidence
                keep[j] = false
            else
                keep[i] = false
            end
        end
    end

    return [predictions[i] for i in eachindex(predictions) if keep[i]]
end

function collect_predicted_relation_candidates(
    outputs;
    no_relation_id::Int,
    confidence_threshold::Float32,
    no_relation_margin::Float32 = 0.0f0,
    nonnull_probability_threshold::Float32 = 0.0f0,
    max_relations_per_head::Int = 0,
    max_relations_per_tail::Int = 0,
    relation_confidence_thresholds::Dict{Int,Float32} = Dict{Int,Float32}(),
    relation_allowed_type_pairs::Dict{Int,Set{Tuple{Int,Int}}} = Dict{Int,Set{Tuple{Int,Int}}}(),
    span_type_to_token_label_ids::Dict{Int,Tuple{Int,Int}} = Dict{Int,Tuple{Int,Int}}(),
    symmetric_relations::Set{Int} = Set{Int}(),
    inverse_relation_map::Dict{Int,Int} = Dict{Int,Int}(),
)
    spans = to_cpu(outputs.spans)
    relation_pairs = to_cpu(outputs.relation_pairs)
    relation_mask = to_cpu(outputs.relation_mask)
    relation_logits = to_cpu(outputs.relation_logits)
    confidence_logits = Float32.(to_cpu(outputs.confidence_logits))
    confidence_probs = 1.0f0 ./ (1.0f0 .+ exp.(-confidence_logits))

    batch_size = size(spans, 3)
    max_pairs = size(relation_pairs, 2)
    max_spans = size(spans, 2)
    span_type_ids = if isempty(relation_allowed_type_pairs) || isempty(span_type_to_token_label_ids)
        zeros(Int, max_spans, batch_size)
    else
        infer_span_type_ids(outputs, span_type_to_token_label_ids)
    end
    predicted_batches = Vector{Vector{NamedTuple{(:relation, :label_id, :confidence),Tuple{NTuple{5,Int},Int,Float32}}}}(undef, batch_size)

    for b in 1:batch_size
        candidates = Tuple{Float32, NTuple{5, Int}, NTuple{2, Int}, NTuple{2, Int}, Int, Float32}[]
        for i in 1:max_pairs
            relation_mask[i, b] || continue
            confidence_prob = confidence_probs[1, i, b]
            relation_view = @view(relation_logits[:, i, b])
            label_id = findmax(relation_view)[2]
            label_id == no_relation_id && continue
            effective_confidence_threshold = get(relation_confidence_thresholds, Int(label_id), confidence_threshold)
            confidence_prob >= effective_confidence_threshold || continue
            max_logit = maximum(relation_view)
            exp_shifted = exp.(Float32.(relation_view) .- Float32(max_logit))
            denom = sum(exp_shifted)
            nonnull_prob = denom > 0 ? Float32(1.0f0 - exp_shifted[no_relation_id] / denom) : 0.0f0
            nonnull_prob >= nonnull_probability_threshold || continue
            margin = Float32(relation_logits[label_id, i, b] - relation_logits[no_relation_id, i, b])
            margin >= no_relation_margin || continue
            head_idx = Int(relation_pairs[1, i, b])
            tail_idx = Int(relation_pairs[2, i, b])
            if !(1 <= head_idx <= max_spans && 1 <= tail_idx <= max_spans)
                continue
            end
            if !isempty(relation_allowed_type_pairs)
                allowed_pairs = get(relation_allowed_type_pairs, Int(label_id), nothing)
                if allowed_pairs !== nothing && !isempty(allowed_pairs)
                    head_type = span_type_ids[head_idx, b]
                    tail_type = span_type_ids[tail_idx, b]
                    if head_type == 0 || tail_type == 0 || !((head_type, tail_type) in allowed_pairs)
                        continue
                    end
                end
            end
            head = (Int(spans[1, head_idx, b]), Int(spans[2, head_idx, b]))
            tail = (Int(spans[1, tail_idx, b]), Int(spans[2, tail_idx, b]))
            label_prob = denom > 0 ? Float32(exp_shifted[label_id] / denom) : 0.0f0
            candidate_score = confidence_prob * label_prob
            rel_tuple = (head[1], head[2], tail[1], tail[2], Int(label_id))
            push!(candidates, (candidate_score, rel_tuple, head, tail, Int(label_id), Float32(confidence_prob)))
        end
        sort!(candidates; by = item -> item[1], rev = true)
        head_counts = Dict{NTuple{2, Int}, Int}()
        tail_counts = Dict{NTuple{2, Int}, Int}()
        batch_predictions = NamedTuple{(:relation, :label_id, :confidence),Tuple{NTuple{5,Int},Int,Float32}}[]
        for (_, rel_tuple, head, tail, label_id, confidence_prob) in candidates
            if max_relations_per_head > 0 && get(head_counts, head, 0) >= max_relations_per_head
                continue
            end
            if max_relations_per_tail > 0 && get(tail_counts, tail, 0) >= max_relations_per_tail
                continue
            end
            push!(batch_predictions, (relation = rel_tuple, label_id = label_id, confidence = confidence_prob))
            head_counts[head] = get(head_counts, head, 0) + 1
            tail_counts[tail] = get(tail_counts, tail, 0) + 1
        end
        predicted_batches[b] = apply_relation_consistency(
            batch_predictions,
            symmetric_relations,
            inverse_relation_map,
        )
    end

    return predicted_batches
end

function collect_predicted_relation_set(
    outputs;
    no_relation_id::Int,
    confidence_threshold::Float32,
    no_relation_margin::Float32 = 0.0f0,
    nonnull_probability_threshold::Float32 = 0.0f0,
    max_relations_per_head::Int = 0,
    max_relations_per_tail::Int = 0,
    relation_confidence_thresholds::Dict{Int,Float32} = Dict{Int,Float32}(),
    relation_allowed_type_pairs::Dict{Int,Set{Tuple{Int,Int}}} = Dict{Int,Set{Tuple{Int,Int}}}(),
    span_type_to_token_label_ids::Dict{Int,Tuple{Int,Int}} = Dict{Int,Tuple{Int,Int}}(),
    symmetric_relations::Set{Int} = Set{Int}(),
    inverse_relation_map::Dict{Int,Int} = Dict{Int,Int}(),
)
    batches = collect_predicted_relation_candidates(
        outputs;
        no_relation_id = no_relation_id,
        confidence_threshold = confidence_threshold,
        no_relation_margin = no_relation_margin,
        nonnull_probability_threshold = nonnull_probability_threshold,
        max_relations_per_head = max_relations_per_head,
        max_relations_per_tail = max_relations_per_tail,
        relation_confidence_thresholds = relation_confidence_thresholds,
        relation_allowed_type_pairs = relation_allowed_type_pairs,
        span_type_to_token_label_ids = span_type_to_token_label_ids,
        symmetric_relations = symmetric_relations,
        inverse_relation_map = inverse_relation_map,
    )
    predicted_sets = Vector{Set{NTuple{5, Int}}}(undef, length(batches))
    for b in eachindex(batches)
        predicted_sets[b] = Set(item.relation for item in batches[b])
    end
    return predicted_sets
end

function proposal_diagnostics(
    outputs,
    targets;
    no_relation_id::Int,
    confidence_threshold::Float32 = 0.5f0,
    no_relation_margin::Float32 = 0.0f0,
    nonnull_probability_threshold::Float32 = 0.0f0,
    max_relations_per_head::Int = 0,
    max_relations_per_tail::Int = 0,
    relation_confidence_thresholds::Dict{Int,Float32} = Dict{Int,Float32}(),
    relation_allowed_type_pairs::Dict{Int,Set{Tuple{Int,Int}}} = Dict{Int,Set{Tuple{Int,Int}}}(),
    span_type_to_token_label_ids::Dict{Int,Tuple{Int,Int}} = Dict{Int,Tuple{Int,Int}}(),
    symmetric_relations::Set{Int} = Set{Int}(),
    inverse_relation_map::Dict{Int,Int} = Dict{Int,Int}(),
)
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
        no_relation_margin = no_relation_margin,
        nonnull_probability_threshold = nonnull_probability_threshold,
        max_relations_per_head = max_relations_per_head,
        max_relations_per_tail = max_relations_per_tail,
        relation_confidence_thresholds = relation_confidence_thresholds,
        relation_allowed_type_pairs = relation_allowed_type_pairs,
        span_type_to_token_label_ids = span_type_to_token_label_ids,
        symmetric_relations = symmetric_relations,
        inverse_relation_map = inverse_relation_map,
    )

    batch_size = size(target_spans, 3)
    gold_span_total = 0
    matched_span_total = 0
    predicted_span_total = 0
    gold_span_top8_hits = 0
    gold_span_top16_hits = 0
    gold_span_top32_hits = 0
    gold_pair_total = 0
    matched_pair_total = 0
    matched_pair_rank_sum = 0
    matched_pair_rank_count = 0
    gold_pair_top8_hits = 0
    gold_pair_top16_hits = 0
    gold_pair_top32_hits = 0
    gold_pair_short_total = 0
    gold_pair_medium_total = 0
    gold_pair_long_total = 0
    matched_pair_short_total = 0
    matched_pair_medium_total = 0
    matched_pair_long_total = 0
    predicted_relation_total = 0
    gold_relation_total = 0
    relation_true_positive_total = 0
    oracle_relation_coverage_total = 0

    for b in 1:batch_size
        gold_span_set = collect_span_set(target_spans, target_span_mask, b)
        pred_span_set = collect_span_set(predicted_spans, predicted_span_mask, b)
        pred_span_ranks = collect_span_rank_map(predicted_spans, predicted_span_mask, b)
        gold_pair_set = collect_pair_set(target_spans, target_relation_pairs, target_relation_mask .& (target_relation_targets .> 0.5f0), b)
        pred_pair_set = collect_pair_set(predicted_spans, predicted_pairs, predicted_pair_mask, b)
        pred_pair_ranks = collect_pair_rank_map(predicted_spans, predicted_pairs, predicted_pair_mask, b)
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
        predicted_span_total += length(pred_span_set)
        gold_pair_total += length(gold_pair_set)
        matched_pair_total += length(intersect(gold_pair_set, pred_pair_set))
        gold_relation_total += length(gold_relation_set)
        predicted_relation_total += length(pred_relation_set)
        relation_true_positive_total += length(intersect(gold_relation_set, pred_relation_set))

        for gold_span in gold_span_set
            rank = get(pred_span_ranks, gold_span, 0)
            if rank > 0
                if rank <= 8
                    gold_span_top8_hits += 1
                end
                if rank <= 16
                    gold_span_top16_hits += 1
                end
                if rank <= 32
                    gold_span_top32_hits += 1
                end
            end
        end

        for gold_pair in gold_pair_set
            bucket = pair_distance_bucket(gold_pair)
            if bucket == :short
                gold_pair_short_total += 1
            elseif bucket == :medium
                gold_pair_medium_total += 1
            else
                gold_pair_long_total += 1
            end

            rank = get(pred_pair_ranks, gold_pair, 0)
            if rank > 0
                matched_pair_rank_sum += rank
                matched_pair_rank_count += 1
                if rank <= 8
                    gold_pair_top8_hits += 1
                end
                if rank <= 16
                    gold_pair_top16_hits += 1
                end
                if rank <= 32
                    gold_pair_top32_hits += 1
                end

                if bucket == :short
                    matched_pair_short_total += 1
                elseif bucket == :medium
                    matched_pair_medium_total += 1
                else
                    matched_pair_long_total += 1
                end
            end
        end

        for gold_relation in gold_relation_set
            head = (gold_relation[1], gold_relation[2])
            tail = (gold_relation[3], gold_relation[4])
            if head in pred_span_set && tail in pred_span_set
                oracle_relation_coverage_total += 1
            end
        end
    end

    return (
        gold_spans = gold_span_total,
        matched_spans = matched_span_total,
        predicted_spans = predicted_span_total,
        gold_span_top8_hits = gold_span_top8_hits,
        gold_span_top16_hits = gold_span_top16_hits,
        gold_span_top32_hits = gold_span_top32_hits,
        gold_pairs = gold_pair_total,
        matched_pairs = matched_pair_total,
        matched_pair_rank_sum = matched_pair_rank_sum,
        matched_pair_rank_count = matched_pair_rank_count,
        gold_pair_top8_hits = gold_pair_top8_hits,
        gold_pair_top16_hits = gold_pair_top16_hits,
        gold_pair_top32_hits = gold_pair_top32_hits,
        gold_pair_short_total = gold_pair_short_total,
        gold_pair_medium_total = gold_pair_medium_total,
        gold_pair_long_total = gold_pair_long_total,
        matched_pair_short_total = matched_pair_short_total,
        matched_pair_medium_total = matched_pair_medium_total,
        matched_pair_long_total = matched_pair_long_total,
        gold_relations = gold_relation_total,
        predicted_relations = predicted_relation_total,
        true_positive_relations = relation_true_positive_total,
        oracle_relation_coverage = oracle_relation_coverage_total,
    )
end

function build_proposal_inputs(
    outputs,
    inputs,
    model_config,
    pair_proposer_settings;
    emit_evidence_diagnostics::Bool = false,
    evidence_pooling_mode::Symbol = :token,
)
    return (
        token_ids = inputs.token_ids,
        token_mask = inputs.token_mask,
        emit_evidence_diagnostics = emit_evidence_diagnostics,
        evidence_pooling_mode = evidence_pooling_mode,
        retrieval_distance_bias_scale = hasproperty(inputs, :retrieval_distance_bias_scale) ? Float32(inputs.retrieval_distance_bias_scale) : 0.0f0,
        retrieval_type_bias_scale = hasproperty(inputs, :retrieval_type_bias_scale) ? Float32(inputs.retrieval_type_bias_scale) : 0.0f0,
        retrieval_sentence_bias_scale = hasproperty(inputs, :retrieval_sentence_bias_scale) ? Float32(inputs.retrieval_sentence_bias_scale) : 0.0f0,
        retrieval_local_bias_scale = hasproperty(inputs, :retrieval_local_bias_scale) ? Float32(inputs.retrieval_local_bias_scale) : 0.0f0,
        retrieval_sentence_embedding_scale = hasproperty(inputs, :retrieval_sentence_embedding_scale) ? Float32(inputs.retrieval_sentence_embedding_scale) : 0.0f0,
        retrieval_type_compat_bias_scale = hasproperty(inputs, :retrieval_type_compat_bias_scale) ? Float32(inputs.retrieval_type_compat_bias_scale) : 0.0f0,
        retrieval_dot_bias_scale = hasproperty(inputs, :retrieval_dot_bias_scale) ? Float32(inputs.retrieval_dot_bias_scale) : 0.0f0,
        retrieval_compatibility_scale = hasproperty(inputs, :retrieval_compatibility_scale) ? Float32(inputs.retrieval_compatibility_scale) : 0.0f0,
        span_context_enabled = hasproperty(inputs, :span_context_enabled) ? Bool(inputs.span_context_enabled) : true,
        span_context_use_adjacent = hasproperty(inputs, :span_context_use_adjacent) ? Bool(inputs.span_context_use_adjacent) : true,
        span_context_use_sentence = hasproperty(inputs, :span_context_use_sentence) ? Bool(inputs.span_context_use_sentence) : true,
        span_context_use_semantic = hasproperty(inputs, :span_context_use_semantic) ? Bool(inputs.span_context_use_semantic) : true,
        span_context_sentence_ids = hasproperty(inputs, :span_context_sentence_ids) ? inputs.span_context_sentence_ids : nothing,
        edge_v2_semantic_topk = hasproperty(inputs, :edge_v2_semantic_topk) ? Int(inputs.edge_v2_semantic_topk) : 0,
        edge_v2_reverse_topk = hasproperty(inputs, :edge_v2_reverse_topk) ? Int(inputs.edge_v2_reverse_topk) : 0,
        edge_v2_global_reserve = hasproperty(inputs, :edge_v2_global_reserve) ? Int(inputs.edge_v2_global_reserve) : 0,
        edge_v2_semantic_score_scale = hasproperty(inputs, :edge_v2_semantic_score_scale) ? Float32(inputs.edge_v2_semantic_score_scale) : 1.0f0,
        edge_v2_span_score_scale = hasproperty(inputs, :edge_v2_span_score_scale) ? Float32(inputs.edge_v2_span_score_scale) : 1.0f0,
        edge_v2_distance_penalty = hasproperty(inputs, :edge_v2_distance_penalty) ? Float32(inputs.edge_v2_distance_penalty) : 0.0f0,
        edge_v2_require_mutual = hasproperty(inputs, :edge_v2_require_mutual) ? Bool(inputs.edge_v2_require_mutual) : false,
        edge_v2_use_local_neighbors = hasproperty(inputs, :edge_v2_use_local_neighbors) ? Bool(inputs.edge_v2_use_local_neighbors) : true,
        edge_v2_use_routed_buckets = hasproperty(inputs, :edge_v2_use_routed_buckets) ? Bool(inputs.edge_v2_use_routed_buckets) : true,
        edge_v2_use_semantic_topk = hasproperty(inputs, :edge_v2_use_semantic_topk) ? Bool(inputs.edge_v2_use_semantic_topk) : true,
        edge_v2_use_global_reserve = hasproperty(inputs, :edge_v2_use_global_reserve) ? Bool(inputs.edge_v2_use_global_reserve) : true,
    )
end

function build_fixed_proposal_inputs(proposal_outputs, inputs)
    return (
        token_ids = inputs.token_ids,
        token_mask = inputs.token_mask,
        spans = proposal_outputs.spans,
        span_mask = proposal_outputs.span_mask,
        span_scores = proposal_outputs.span_scores,
        relation_pairs = proposal_outputs.relation_pairs,
        relation_mask = proposal_outputs.relation_mask,
        retrieval_distance_bias_scale = hasproperty(inputs, :retrieval_distance_bias_scale) ? Float32(inputs.retrieval_distance_bias_scale) : 0.0f0,
        retrieval_type_bias_scale = hasproperty(inputs, :retrieval_type_bias_scale) ? Float32(inputs.retrieval_type_bias_scale) : 0.0f0,
        retrieval_sentence_bias_scale = hasproperty(inputs, :retrieval_sentence_bias_scale) ? Float32(inputs.retrieval_sentence_bias_scale) : 0.0f0,
        retrieval_local_bias_scale = hasproperty(inputs, :retrieval_local_bias_scale) ? Float32(inputs.retrieval_local_bias_scale) : 0.0f0,
        retrieval_sentence_embedding_scale = hasproperty(inputs, :retrieval_sentence_embedding_scale) ? Float32(inputs.retrieval_sentence_embedding_scale) : 0.0f0,
        retrieval_type_compat_bias_scale = hasproperty(inputs, :retrieval_type_compat_bias_scale) ? Float32(inputs.retrieval_type_compat_bias_scale) : 0.0f0,
        retrieval_dot_bias_scale = hasproperty(inputs, :retrieval_dot_bias_scale) ? Float32(inputs.retrieval_dot_bias_scale) : 0.0f0,
        retrieval_compatibility_scale = hasproperty(inputs, :retrieval_compatibility_scale) ? Float32(inputs.retrieval_compatibility_scale) : 0.0f0,
        span_context_enabled = hasproperty(inputs, :span_context_enabled) ? Bool(inputs.span_context_enabled) : true,
        span_context_use_adjacent = hasproperty(inputs, :span_context_use_adjacent) ? Bool(inputs.span_context_use_adjacent) : true,
        span_context_use_sentence = hasproperty(inputs, :span_context_use_sentence) ? Bool(inputs.span_context_use_sentence) : true,
        span_context_use_semantic = hasproperty(inputs, :span_context_use_semantic) ? Bool(inputs.span_context_use_semantic) : true,
        span_context_sentence_ids = hasproperty(inputs, :span_context_sentence_ids) ? inputs.span_context_sentence_ids : nothing,
        edge_v2_semantic_topk = hasproperty(inputs, :edge_v2_semantic_topk) ? Int(inputs.edge_v2_semantic_topk) : 0,
        edge_v2_reverse_topk = hasproperty(inputs, :edge_v2_reverse_topk) ? Int(inputs.edge_v2_reverse_topk) : 0,
        edge_v2_global_reserve = hasproperty(inputs, :edge_v2_global_reserve) ? Int(inputs.edge_v2_global_reserve) : 0,
        edge_v2_semantic_score_scale = hasproperty(inputs, :edge_v2_semantic_score_scale) ? Float32(inputs.edge_v2_semantic_score_scale) : 1.0f0,
        edge_v2_span_score_scale = hasproperty(inputs, :edge_v2_span_score_scale) ? Float32(inputs.edge_v2_span_score_scale) : 1.0f0,
        edge_v2_distance_penalty = hasproperty(inputs, :edge_v2_distance_penalty) ? Float32(inputs.edge_v2_distance_penalty) : 0.0f0,
        edge_v2_require_mutual = hasproperty(inputs, :edge_v2_require_mutual) ? Bool(inputs.edge_v2_require_mutual) : false,
        edge_v2_use_local_neighbors = hasproperty(inputs, :edge_v2_use_local_neighbors) ? Bool(inputs.edge_v2_use_local_neighbors) : true,
        edge_v2_use_routed_buckets = hasproperty(inputs, :edge_v2_use_routed_buckets) ? Bool(inputs.edge_v2_use_routed_buckets) : true,
        edge_v2_use_semantic_topk = hasproperty(inputs, :edge_v2_use_semantic_topk) ? Bool(inputs.edge_v2_use_semantic_topk) : true,
        edge_v2_use_global_reserve = hasproperty(inputs, :edge_v2_use_global_reserve) ? Bool(inputs.edge_v2_use_global_reserve) : true,
    )
end

function build_proposal_relation_targets(proposal_outputs, targets, no_relation_id::Int)
    proposal_spans = to_cpu(proposal_outputs.spans)
    proposal_pairs = to_cpu(proposal_outputs.relation_pairs)
    proposal_mask = to_cpu(proposal_outputs.relation_mask)

    target_spans = to_cpu(targets.spans)
    target_span_mask = to_cpu(targets.span_mask)
    target_relation_pairs = to_cpu(targets.relation_pairs)
    target_relation_labels = to_cpu(targets.relation_labels)
    target_relation_mask = to_cpu(targets.relation_mask)
    target_relation_targets = Float32.(to_cpu(targets.relation_targets))

    max_pairs, batch_size = size(proposal_mask)
    proposal_labels = fill(Int32(-100), max_pairs, batch_size)
    proposal_targets = zeros(Float32, max_pairs, batch_size)

    for b in 1:batch_size
        gold_relation_lookup = Dict{NTuple{4, Int}, Int32}()
        max_target_pairs = size(target_relation_pairs, 2)
        max_target_spans = size(target_spans, 2)
        for i in 1:max_target_pairs
            target_relation_mask[i, b] || continue
            target_relation_targets[i, b] > 0.5f0 || continue
            head_idx = Int(target_relation_pairs[1, i, b])
            tail_idx = Int(target_relation_pairs[2, i, b])
            if !(1 <= head_idx <= max_target_spans && 1 <= tail_idx <= max_target_spans)
                continue
            end
            target_span_mask[head_idx, b] || continue
            target_span_mask[tail_idx, b] || continue
            head = (Int(target_spans[1, head_idx, b]), Int(target_spans[2, head_idx, b]))
            tail = (Int(target_spans[1, tail_idx, b]), Int(target_spans[2, tail_idx, b]))
            gold_relation_lookup[(head[1], head[2], tail[1], tail[2])] = Int32(target_relation_labels[i, b])
        end

        max_proposal_pairs = size(proposal_pairs, 2)
        max_proposal_spans = size(proposal_spans, 2)
        for i in 1:max_proposal_pairs
            proposal_mask[i, b] || continue
            head_idx = Int(proposal_pairs[1, i, b])
            tail_idx = Int(proposal_pairs[2, i, b])
            if !(1 <= head_idx <= max_proposal_spans && 1 <= tail_idx <= max_proposal_spans)
                continue
            end
            head = (Int(proposal_spans[1, head_idx, b]), Int(proposal_spans[2, head_idx, b]))
            tail = (Int(proposal_spans[1, tail_idx, b]), Int(proposal_spans[2, tail_idx, b]))
            pair = (head[1], head[2], tail[1], tail[2])
            if haskey(gold_relation_lookup, pair)
                proposal_labels[i, b] = gold_relation_lookup[pair]
                proposal_targets[i, b] = 1.0f0
            else
                proposal_labels[i, b] = Int32(no_relation_id)
                proposal_targets[i, b] = 0.0f0
            end
        end
    end

    return (
        relation_labels = to_device(proposal_labels),
        relation_targets = to_device(proposal_targets),
        relation_mask = proposal_outputs.relation_mask,
    )
end

function collect_evidence_diagnostics(outputs)
    if !hasproperty(outputs, :evidence_top_token_index) ||
       !hasproperty(outputs, :evidence_attention_entropy) ||
       !hasproperty(outputs, :evidence_attention_max_weight)
        return (
            count = 0,
            entropy_sum = 0.0f0,
            max_weight_sum = 0.0f0,
            effective_tokens_sum = 0.0f0,
            top_token_hist = Dict{Int,Int}(),
        )
    end

    top_index = outputs.evidence_top_token_index
    attention_entropy = outputs.evidence_attention_entropy
    attention_max_weight = outputs.evidence_attention_max_weight
    if top_index === nothing || attention_entropy === nothing || attention_max_weight === nothing
        return (
            count = 0,
            entropy_sum = 0.0f0,
            max_weight_sum = 0.0f0,
            effective_tokens_sum = 0.0f0,
            top_token_hist = Dict{Int,Int}(),
        )
    end

    relation_mask = Bool.(to_cpu(outputs.relation_mask))
    top_index_cpu = Int.(to_cpu(top_index))
    entropy_cpu = Float32.(to_cpu(attention_entropy))
    max_weight_cpu = Float32.(to_cpu(attention_max_weight))

    count = 0
    entropy_sum = 0.0f0
    max_weight_sum = 0.0f0
    effective_tokens_sum = 0.0f0
    top_token_hist = Dict{Int,Int}()
    max_pairs, batch_size = size(relation_mask)
    for b in 1:batch_size
        for pair_idx in 1:max_pairs
            relation_mask[pair_idx, b] || continue
            count += 1
            entropy = entropy_cpu[pair_idx, b]
            max_weight = max_weight_cpu[pair_idx, b]
            entropy_sum += entropy
            max_weight_sum += max_weight
            effective_tokens_sum += exp(entropy)
            token_idx = top_index_cpu[pair_idx, b]
            token_idx > 0 || continue
            top_token_hist[token_idx] = get(top_token_hist, token_idx, 0) + 1
        end
    end

    return (
        count = count,
        entropy_sum = entropy_sum,
        max_weight_sum = max_weight_sum,
        effective_tokens_sum = effective_tokens_sum,
        top_token_hist = top_token_hist,
    )
end

function top_k_token_stats(hist::Dict{Int,Int}; k::Int = 3)
    entries = sort(collect(hist); by = item -> item[2], rev = true)
    token_ids = fill(0, k)
    counts = fill(0, k)
    for idx in 1:min(k, length(entries))
        token_ids[idx] = entries[idx][1]
        counts[idx] = entries[idx][2]
    end
    return token_ids, counts
end

function format_eval_summary(eval_stats)
    return @sprintf(
        "val_loss %.4f | entity %.4f | boundary %.4f | mention %.4f | retrieval %.4f | ret_rank %.4f | relation %.4f | confidence %.4f | prop_ret %.4f | prop_rank %.4f | prop_rel %.4f | prop_conf %.4f | prop_total %.4f | ment_p/r/f1 %.4f/%.4f/%.4f | ment_t16 %.4f | oracle_rel %.4f | pair_recall %.4f | pair_t16 %.4f | pair_rank %.1f | miss_s/m/l %.2f/%.2f/%.2f | rel_p %.4f | rel_r %.4f | rel_f1 %.4f | ev_ent/max/eff %.3f/%.3f/%.1f | ev_top %d:%d,%d:%d,%d:%d",
        eval_stats.total_loss,
        eval_stats.entity_loss,
        eval_stats.boundary_loss,
        eval_stats.mention_loss,
        eval_stats.retrieval_loss,
        eval_stats.retrieval_ranking_loss,
        eval_stats.relation_loss,
        eval_stats.confidence_loss,
        eval_stats.proposal_retrieval_loss,
        eval_stats.proposal_retrieval_ranking_loss,
        eval_stats.proposal_relation_loss,
        eval_stats.proposal_confidence_loss,
        eval_stats.proposal_total_loss,
        eval_stats.mention_precision,
        eval_stats.mention_recall,
        eval_stats.mention_f1,
        eval_stats.mention_top16_recall,
        eval_stats.oracle_relation_coverage,
        eval_stats.pair_recall,
        eval_stats.pair_top16_recall,
        eval_stats.matched_pair_rank_mean,
        eval_stats.missed_pair_short_share,
        eval_stats.missed_pair_medium_share,
        eval_stats.missed_pair_long_share,
        eval_stats.relation_precision,
        eval_stats.relation_recall,
        eval_stats.relation_f1,
        eval_stats.evidence_entropy,
        eval_stats.evidence_max_weight,
        eval_stats.evidence_effective_tokens,
        eval_stats.evidence_top1_token,
        eval_stats.evidence_top1_count,
        eval_stats.evidence_top2_token,
        eval_stats.evidence_top2_count,
        eval_stats.evidence_top3_token,
        eval_stats.evidence_top3_count,
    )
end

function evaluate_model(
    model,
    params,
    state,
    rows,
    vocab,
    entity_label_to_id,
    relation_label_to_id,
    model_config,
    run_config,
    pair_proposer_settings;
    current_step::Union{Nothing,Int} = nothing,
    evidence_pooling_mode::Symbol = :token,
    relation_logit_adjustment_tau::Float32 = 0.0f0,
    relation_logit_adjustment::Union{Nothing,Vector{Float32}} = nothing,
)
    isempty(rows) && return (
        total_loss = NaN32,
        entity_loss = NaN32,
        boundary_loss = NaN32,
        mention_loss = NaN32,
        retrieval_loss = NaN32,
        retrieval_ranking_loss = NaN32,
        relation_loss = NaN32,
        confidence_loss = NaN32,
        proposal_retrieval_loss = NaN32,
        proposal_retrieval_ranking_loss = NaN32,
        proposal_relation_loss = NaN32,
        proposal_confidence_loss = NaN32,
        proposal_total_loss = NaN32,
        mention_precision = NaN32,
        mention_recall = NaN32,
        mention_f1 = NaN32,
        mention_top8_recall = NaN32,
        mention_top16_recall = NaN32,
        mention_top32_recall = NaN32,
        oracle_relation_coverage = NaN32,
        span_recall = NaN32,
        pair_recall = NaN32,
        pair_top8_recall = NaN32,
        pair_top16_recall = NaN32,
        pair_top32_recall = NaN32,
        matched_pair_rank_mean = NaN32,
        pair_recall_short = NaN32,
        pair_recall_medium = NaN32,
        pair_recall_long = NaN32,
        missed_pair_short_share = NaN32,
        missed_pair_medium_share = NaN32,
        missed_pair_long_share = NaN32,
        relation_precision = NaN32,
        relation_recall = NaN32,
        relation_f1 = NaN32,
        evidence_entropy = NaN32,
        evidence_max_weight = NaN32,
        evidence_effective_tokens = NaN32,
        evidence_top1_token = 0,
        evidence_top1_count = 0,
        evidence_top2_token = 0,
        evidence_top2_count = 0,
        evidence_top3_token = 0,
        evidence_top3_count = 0,
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
    mention_loss_sum = 0.0f0
    retrieval_loss_sum = 0.0f0
    retrieval_ranking_loss_sum = 0.0f0
    relation_loss_sum = 0.0f0
    confidence_loss_sum = 0.0f0
    proposal_retrieval_loss_sum = 0.0f0
    proposal_retrieval_ranking_loss_sum = 0.0f0
    proposal_relation_loss_sum = 0.0f0
    proposal_confidence_loss_sum = 0.0f0
    proposal_total_loss_sum = 0.0f0
    gold_spans = 0
    matched_spans = 0
    predicted_spans = 0
    gold_span_top8_hits = 0
    gold_span_top16_hits = 0
    gold_span_top32_hits = 0
    gold_pairs = 0
    matched_pairs = 0
    matched_pair_rank_sum = 0
    matched_pair_rank_count = 0
    gold_pair_top8_hits = 0
    gold_pair_top16_hits = 0
    gold_pair_top32_hits = 0
    gold_pair_short_total = 0
    gold_pair_medium_total = 0
    gold_pair_long_total = 0
    matched_pair_short_total = 0
    matched_pair_medium_total = 0
    matched_pair_long_total = 0
    gold_relations = 0
    predicted_relations = 0
    true_positive_relations = 0
    oracle_relation_coverage = 0
    evidence_entropy_sum = 0.0f0
    evidence_max_weight_sum = 0.0f0
    evidence_effective_tokens_sum = 0.0f0
    evidence_count = 0
    evidence_top_token_hist = Dict{Int,Int}()
    eval_state = Lux.testmode(state)
    max_batches = min(run_config.max_eval_batches, cld(length(rows), run_config.batch_size))
    no_relation_id = get(relation_label_to_id, "NO_RELATION", 1)
    retrieval_bias_settings = load_retrieval_bias_settings(run_config.config_path)
    null_relation_weight = load_null_relation_weight(run_config.config_path)
    relation_focal_gamma = load_relation_focal_gamma(run_config.config_path)
    positive_relation_weight = load_positive_relation_weight(run_config.config_path)
    distillation_settings = load_distillation_settings(run_config.config_path)
    logit_adjustment_tau = relation_logit_adjustment_tau > 0.0f0 ?
        relation_logit_adjustment_tau :
        load_relation_logit_adjustment_tau(run_config.config_path)
    edge_ranking_settings = load_edge_ranking_settings(run_config.config_path)
    eval_rng = MersenneTwister(run_config.seed)

    for batch_idx in 1:max_batches
        start_idx = (batch_idx - 1) * run_config.batch_size + 1
        end_idx = min(batch_idx * run_config.batch_size, length(rows))
        batch_rows = rows[start_idx:end_idx]
        inputs, targets = make_batch(
            batch_rows,
            vocab,
            entity_label_to_id,
            relation_label_to_id,
            model_config,
            run_config;
            rng = eval_rng,
        )
        inputs = with_retrieval_bias_inputs(inputs, retrieval_bias_settings; step = current_step)
        outputs, _ = model(inputs, params, eval_state)
        breakdown = relation_loss_breakdown(
            outputs,
            targets;
            null_relation_weight = null_relation_weight,
            positive_relation_weight = positive_relation_weight,
            no_relation_id = no_relation_id,
            relation_focal_gamma = relation_focal_gamma,
            relation_logit_adjustment_tau = logit_adjustment_tau,
            relation_logit_adjustment = relation_logit_adjustment,
            teacher_entity_loss_weight = distillation_settings.entity_weight,
            teacher_relation_loss_weight = distillation_settings.relation_weight,
            teacher_confidence_loss_weight = distillation_settings.confidence_weight,
            edge_ranking_loss_weight = edge_ranking_settings.weight,
            edge_ranking_margin = edge_ranking_settings.margin,
            edge_ranking_hard_negatives = edge_ranking_settings.hard_negatives,
        )
        total_loss_sum += breakdown.total
        entity_loss_sum += breakdown.entity
        boundary_loss_sum += breakdown.boundary
        mention_loss_sum += breakdown.mention
        retrieval_loss_sum += breakdown.retrieval
        retrieval_ranking_loss_sum += breakdown.retrieval_rank
        relation_loss_sum += breakdown.relation
        confidence_loss_sum += breakdown.confidence

        proposal_inputs = build_proposal_inputs(
            outputs,
            inputs,
            model_config,
            pair_proposer_settings;
            emit_evidence_diagnostics = true,
            evidence_pooling_mode = evidence_pooling_mode,
        )
        proposal_outputs, _ = model(proposal_inputs, params, eval_state)
        proposal_targets = build_proposal_relation_targets(proposal_outputs, targets, no_relation_id)
        proposal_losses = proposal_training_loss(
            proposal_outputs,
            proposal_targets;
            null_relation_weight = null_relation_weight,
            positive_relation_weight = positive_relation_weight,
            no_relation_id = no_relation_id,
            relation_focal_gamma = relation_focal_gamma,
            relation_logit_adjustment_tau = logit_adjustment_tau,
            relation_logit_adjustment = relation_logit_adjustment,
            edge_ranking_loss_weight = edge_ranking_settings.weight,
            edge_ranking_margin = edge_ranking_settings.margin,
            edge_ranking_hard_negatives = edge_ranking_settings.hard_negatives,
        )
        proposal_retrieval_loss_sum += proposal_losses.retrieval
        proposal_retrieval_ranking_loss_sum += proposal_losses.retrieval_rank
        proposal_relation_loss_sum += proposal_losses.relation
        proposal_confidence_loss_sum += proposal_losses.confidence
        proposal_total_loss_sum += breakdown.entity + breakdown.boundary + breakdown.mention + proposal_losses.total
        diagnostics = proposal_diagnostics(
            proposal_outputs,
            targets;
            no_relation_id = no_relation_id,
            confidence_threshold = 0.5f0,
        )
        gold_spans += diagnostics.gold_spans
        matched_spans += diagnostics.matched_spans
        predicted_spans += diagnostics.predicted_spans
        gold_span_top8_hits += diagnostics.gold_span_top8_hits
        gold_span_top16_hits += diagnostics.gold_span_top16_hits
        gold_span_top32_hits += diagnostics.gold_span_top32_hits
        gold_pairs += diagnostics.gold_pairs
        matched_pairs += diagnostics.matched_pairs
        matched_pair_rank_sum += diagnostics.matched_pair_rank_sum
        matched_pair_rank_count += diagnostics.matched_pair_rank_count
        gold_pair_top8_hits += diagnostics.gold_pair_top8_hits
        gold_pair_top16_hits += diagnostics.gold_pair_top16_hits
        gold_pair_top32_hits += diagnostics.gold_pair_top32_hits
        gold_pair_short_total += diagnostics.gold_pair_short_total
        gold_pair_medium_total += diagnostics.gold_pair_medium_total
        gold_pair_long_total += diagnostics.gold_pair_long_total
        matched_pair_short_total += diagnostics.matched_pair_short_total
        matched_pair_medium_total += diagnostics.matched_pair_medium_total
        matched_pair_long_total += diagnostics.matched_pair_long_total
        gold_relations += diagnostics.gold_relations
        predicted_relations += diagnostics.predicted_relations
        true_positive_relations += diagnostics.true_positive_relations
        oracle_relation_coverage += diagnostics.oracle_relation_coverage
        evidence = collect_evidence_diagnostics(proposal_outputs)
        evidence_entropy_sum += evidence.entropy_sum
        evidence_max_weight_sum += evidence.max_weight_sum
        evidence_effective_tokens_sum += evidence.effective_tokens_sum
        evidence_count += evidence.count
        for (token_id, token_count) in evidence.top_token_hist
            evidence_top_token_hist[token_id] = get(evidence_top_token_hist, token_id, 0) + token_count
        end
        CUDA.synchronize()
    end

    mention_precision = safe_rate(matched_spans, predicted_spans)
    mention_recall = safe_rate(matched_spans, gold_spans)
    mention_f1 = mention_precision + mention_recall > 0 ?
        2f0 * mention_precision * mention_recall / (mention_precision + mention_recall) :
        0.0f0
    relation_precision = safe_rate(true_positive_relations, predicted_relations)
    relation_recall = safe_rate(true_positive_relations, gold_relations)
    relation_f1 = relation_precision + relation_recall > 0 ?
        2f0 * relation_precision * relation_recall / (relation_precision + relation_recall) :
        0.0f0
    matched_pair_rank_mean = matched_pair_rank_count > 0 ? Float32(matched_pair_rank_sum / matched_pair_rank_count) : NaN32
    missed_pair_total = max(gold_pairs - matched_pairs, 0)
    missed_pair_short = max(gold_pair_short_total - matched_pair_short_total, 0)
    missed_pair_medium = max(gold_pair_medium_total - matched_pair_medium_total, 0)
    missed_pair_long = max(gold_pair_long_total - matched_pair_long_total, 0)
    evidence_entropy = evidence_count > 0 ? evidence_entropy_sum / evidence_count : NaN32
    evidence_max_weight = evidence_count > 0 ? evidence_max_weight_sum / evidence_count : NaN32
    evidence_effective_tokens = evidence_count > 0 ? evidence_effective_tokens_sum / evidence_count : NaN32
    evidence_top_ids, evidence_top_counts = top_k_token_stats(evidence_top_token_hist; k = 3)

    return (
        total_loss = total_loss_sum / max_batches,
        entity_loss = entity_loss_sum / max_batches,
        boundary_loss = boundary_loss_sum / max_batches,
        mention_loss = mention_loss_sum / max_batches,
        retrieval_loss = retrieval_loss_sum / max_batches,
        retrieval_ranking_loss = retrieval_ranking_loss_sum / max_batches,
        relation_loss = relation_loss_sum / max_batches,
        confidence_loss = confidence_loss_sum / max_batches,
        proposal_retrieval_loss = proposal_retrieval_loss_sum / max_batches,
        proposal_retrieval_ranking_loss = proposal_retrieval_ranking_loss_sum / max_batches,
        proposal_relation_loss = proposal_relation_loss_sum / max_batches,
        proposal_confidence_loss = proposal_confidence_loss_sum / max_batches,
        proposal_total_loss = proposal_total_loss_sum / max_batches,
        mention_precision = mention_precision,
        mention_recall = mention_recall,
        mention_f1 = mention_f1,
        mention_top8_recall = safe_rate(gold_span_top8_hits, gold_spans),
        mention_top16_recall = safe_rate(gold_span_top16_hits, gold_spans),
        mention_top32_recall = safe_rate(gold_span_top32_hits, gold_spans),
        oracle_relation_coverage = safe_rate(oracle_relation_coverage, gold_relations),
        span_recall = safe_rate(matched_spans, gold_spans),
        pair_recall = safe_rate(matched_pairs, gold_pairs),
        pair_top8_recall = safe_rate(gold_pair_top8_hits, gold_pairs),
        pair_top16_recall = safe_rate(gold_pair_top16_hits, gold_pairs),
        pair_top32_recall = safe_rate(gold_pair_top32_hits, gold_pairs),
        matched_pair_rank_mean = matched_pair_rank_mean,
        pair_recall_short = safe_rate(matched_pair_short_total, gold_pair_short_total),
        pair_recall_medium = safe_rate(matched_pair_medium_total, gold_pair_medium_total),
        pair_recall_long = safe_rate(matched_pair_long_total, gold_pair_long_total),
        missed_pair_short_share = safe_rate(missed_pair_short, missed_pair_total),
        missed_pair_medium_share = safe_rate(missed_pair_medium, missed_pair_total),
        missed_pair_long_share = safe_rate(missed_pair_long, missed_pair_total),
        relation_precision = relation_precision,
        relation_recall = relation_recall,
        relation_f1 = relation_f1,
        evidence_entropy = Float32(evidence_entropy),
        evidence_max_weight = Float32(evidence_max_weight),
        evidence_effective_tokens = Float32(evidence_effective_tokens),
        evidence_top1_token = evidence_top_ids[1],
        evidence_top1_count = evidence_top_counts[1],
        evidence_top2_token = evidence_top_ids[2],
        evidence_top2_count = evidence_top_counts[2],
        evidence_top3_token = evidence_top_ids[3],
        evidence_top3_count = evidence_top_counts[3],
        gold_spans = gold_spans,
        gold_pairs = gold_pairs,
        gold_relations = gold_relations,
        predicted_relations = predicted_relations,
        true_positive_relations = true_positive_relations,
        batches = max_batches,
    )
end

function evaluate_oracle_ladder(model, params, state, rows, vocab, entity_label_to_id, relation_label_to_id, model_config, run_config, pair_proposer_settings)
    return evaluate_oracle_ladder(
        model,
        params,
        state,
        rows,
        vocab,
        entity_label_to_id,
        relation_label_to_id,
        model_config,
        run_config,
        pair_proposer_settings;
        confidence_threshold = 0.5f0,
        no_relation_margin = 0.0f0,
        nonnull_probability_threshold = 0.0f0,
        max_relations_per_head = 0,
        max_relations_per_tail = 0,
        relation_confidence_thresholds = Dict{Int,Float32}(),
        relation_allowed_type_pairs = Dict{Int,Set{Tuple{Int,Int}}}(),
        span_type_to_token_label_ids = Dict{Int,Tuple{Int,Int}}(),
        symmetric_relations = Set{Int}(),
        inverse_relation_map = Dict{Int,Int}(),
    )
end

function evaluate_oracle_ladder(
    model,
    params,
    state,
    rows,
    vocab,
    entity_label_to_id,
    relation_label_to_id,
    model_config,
    run_config,
    pair_proposer_settings;
    confidence_threshold::Float32,
    no_relation_margin::Float32 = 0.0f0,
    nonnull_probability_threshold::Float32 = 0.0f0,
    max_relations_per_head::Int = 0,
    max_relations_per_tail::Int = 0,
    relation_confidence_thresholds::Dict{Int,Float32} = Dict{Int,Float32}(),
    relation_allowed_type_pairs::Dict{Int,Set{Tuple{Int,Int}}} = Dict{Int,Set{Tuple{Int,Int}}}(),
    span_type_to_token_label_ids::Dict{Int,Tuple{Int,Int}} = Dict{Int,Tuple{Int,Int}}(),
    symmetric_relations::Set{Int} = Set{Int}(),
    inverse_relation_map::Dict{Int,Int} = Dict{Int,Int}(),
)
    isempty(rows) && error("No validation rows available for oracle ladder evaluation.")
    eval_state = Lux.testmode(state)
    max_batches = min(run_config.max_eval_batches, cld(length(rows), run_config.batch_size))
    no_relation_id = get(relation_label_to_id, "NO_RELATION", 1)
    retrieval_bias_settings = load_retrieval_bias_settings(run_config.config_path)
    eval_rng = MersenneTwister(run_config.seed)

    mode_accumulators = Dict(
        :gold_gold => zero_oracle_accumulator(),
        :gold_pred_pairs => zero_oracle_accumulator(),
        :pred_exhaustive => zero_oracle_accumulator(),
        :pred_pred => zero_oracle_accumulator(),
    )

    for batch_idx in 1:max_batches
        start_idx = (batch_idx - 1) * run_config.batch_size + 1
        end_idx = min(batch_idx * run_config.batch_size, length(rows))
        batch_rows = rows[start_idx:end_idx]
        inputs, targets = make_batch(
            batch_rows,
            vocab,
            entity_label_to_id,
            relation_label_to_id,
            model_config,
            run_config;
            rng = eval_rng,
        )
        inputs = with_retrieval_bias_inputs(inputs, retrieval_bias_settings)

        gold_gold_outputs, _ = model(inputs, params, eval_state)
        accumulate_oracle_mode!(
            mode_accumulators[:gold_gold],
            oracle_mode_stats(
                gold_gold_outputs,
                targets;
                no_relation_id = no_relation_id,
                confidence_threshold = confidence_threshold,
                no_relation_margin = no_relation_margin,
                nonnull_probability_threshold = nonnull_probability_threshold,
                max_relations_per_head = max_relations_per_head,
                max_relations_per_tail = max_relations_per_tail,
                relation_confidence_thresholds = relation_confidence_thresholds,
                relation_allowed_type_pairs = relation_allowed_type_pairs,
                span_type_to_token_label_ids = span_type_to_token_label_ids,
                symmetric_relations = symmetric_relations,
                inverse_relation_map = inverse_relation_map,
            ),
        )

        gold_span_inputs = (
            token_ids = inputs.token_ids,
            token_mask = inputs.token_mask,
            spans = targets.spans,
            span_mask = targets.span_mask,
            span_scores = Float32.(targets.span_mask),
            retrieval_distance_bias_scale = inputs.retrieval_distance_bias_scale,
            retrieval_type_bias_scale = inputs.retrieval_type_bias_scale,
            retrieval_sentence_bias_scale = inputs.retrieval_sentence_bias_scale,
            retrieval_local_bias_scale = inputs.retrieval_local_bias_scale,
            retrieval_sentence_embedding_scale = inputs.retrieval_sentence_embedding_scale,
            retrieval_type_compat_bias_scale = inputs.retrieval_type_compat_bias_scale,
            retrieval_dot_bias_scale = inputs.retrieval_dot_bias_scale,
            retrieval_compatibility_scale = inputs.retrieval_compatibility_scale,
            span_context_enabled = inputs.span_context_enabled,
            span_context_use_adjacent = inputs.span_context_use_adjacent,
            span_context_use_sentence = inputs.span_context_use_sentence,
            span_context_use_semantic = inputs.span_context_use_semantic,
            span_context_sentence_ids = hasproperty(inputs, :span_context_sentence_ids) ? inputs.span_context_sentence_ids : nothing,
            edge_v2_semantic_topk = inputs.edge_v2_semantic_topk,
            edge_v2_reverse_topk = inputs.edge_v2_reverse_topk,
            edge_v2_global_reserve = inputs.edge_v2_global_reserve,
            edge_v2_semantic_score_scale = inputs.edge_v2_semantic_score_scale,
            edge_v2_span_score_scale = inputs.edge_v2_span_score_scale,
            edge_v2_distance_penalty = inputs.edge_v2_distance_penalty,
            edge_v2_require_mutual = inputs.edge_v2_require_mutual,
            edge_v2_use_local_neighbors = inputs.edge_v2_use_local_neighbors,
            edge_v2_use_routed_buckets = inputs.edge_v2_use_routed_buckets,
            edge_v2_use_semantic_topk = inputs.edge_v2_use_semantic_topk,
            edge_v2_use_global_reserve = inputs.edge_v2_use_global_reserve,
        )
        gold_pred_outputs, _ = model(gold_span_inputs, params, eval_state)
        accumulate_oracle_mode!(
            mode_accumulators[:gold_pred_pairs],
            oracle_mode_stats(
                gold_pred_outputs,
                targets;
                no_relation_id = no_relation_id,
                confidence_threshold = confidence_threshold,
                no_relation_margin = no_relation_margin,
                nonnull_probability_threshold = nonnull_probability_threshold,
                max_relations_per_head = max_relations_per_head,
                max_relations_per_tail = max_relations_per_tail,
                relation_confidence_thresholds = relation_confidence_thresholds,
                relation_allowed_type_pairs = relation_allowed_type_pairs,
                span_type_to_token_label_ids = span_type_to_token_label_ids,
                symmetric_relations = symmetric_relations,
                inverse_relation_map = inverse_relation_map,
            ),
        )

        auto_inputs = (
            token_ids = inputs.token_ids,
            token_mask = inputs.token_mask,
            retrieval_distance_bias_scale = inputs.retrieval_distance_bias_scale,
            retrieval_type_bias_scale = inputs.retrieval_type_bias_scale,
            retrieval_sentence_bias_scale = inputs.retrieval_sentence_bias_scale,
            retrieval_local_bias_scale = inputs.retrieval_local_bias_scale,
            retrieval_sentence_embedding_scale = inputs.retrieval_sentence_embedding_scale,
            retrieval_type_compat_bias_scale = inputs.retrieval_type_compat_bias_scale,
            retrieval_dot_bias_scale = inputs.retrieval_dot_bias_scale,
            retrieval_compatibility_scale = inputs.retrieval_compatibility_scale,
            span_context_enabled = inputs.span_context_enabled,
            span_context_use_adjacent = inputs.span_context_use_adjacent,
            span_context_use_sentence = inputs.span_context_use_sentence,
            span_context_use_semantic = inputs.span_context_use_semantic,
            span_context_sentence_ids = hasproperty(inputs, :span_context_sentence_ids) ? inputs.span_context_sentence_ids : nothing,
            edge_v2_semantic_topk = inputs.edge_v2_semantic_topk,
            edge_v2_reverse_topk = inputs.edge_v2_reverse_topk,
            edge_v2_global_reserve = inputs.edge_v2_global_reserve,
            edge_v2_semantic_score_scale = inputs.edge_v2_semantic_score_scale,
            edge_v2_span_score_scale = inputs.edge_v2_span_score_scale,
            edge_v2_distance_penalty = inputs.edge_v2_distance_penalty,
            edge_v2_require_mutual = inputs.edge_v2_require_mutual,
            edge_v2_use_local_neighbors = inputs.edge_v2_use_local_neighbors,
            edge_v2_use_routed_buckets = inputs.edge_v2_use_routed_buckets,
            edge_v2_use_semantic_topk = inputs.edge_v2_use_semantic_topk,
            edge_v2_use_global_reserve = inputs.edge_v2_use_global_reserve,
        )
        pred_pred_outputs, _ = model(auto_inputs, params, eval_state)
        accumulate_oracle_mode!(
            mode_accumulators[:pred_pred],
            oracle_mode_stats(
                pred_pred_outputs,
                targets;
                no_relation_id = no_relation_id,
                confidence_threshold = confidence_threshold,
                no_relation_margin = no_relation_margin,
                nonnull_probability_threshold = nonnull_probability_threshold,
                max_relations_per_head = max_relations_per_head,
                max_relations_per_tail = max_relations_per_tail,
                relation_confidence_thresholds = relation_confidence_thresholds,
                relation_allowed_type_pairs = relation_allowed_type_pairs,
                span_type_to_token_label_ids = span_type_to_token_label_ids,
                symmetric_relations = symmetric_relations,
                inverse_relation_map = inverse_relation_map,
            ),
        )

        exhaustive_pairs, exhaustive_mask = build_exhaustive_relation_pairs(
            pred_pred_outputs.spans,
            pred_pred_outputs.span_mask,
        )
        pred_exhaustive_inputs = (
            token_ids = inputs.token_ids,
            token_mask = inputs.token_mask,
            spans = pred_pred_outputs.spans,
            span_mask = pred_pred_outputs.span_mask,
            span_scores = pred_pred_outputs.span_scores,
            relation_pairs = exhaustive_pairs,
            relation_mask = exhaustive_mask,
            retrieval_distance_bias_scale = inputs.retrieval_distance_bias_scale,
            retrieval_type_bias_scale = inputs.retrieval_type_bias_scale,
            retrieval_sentence_bias_scale = inputs.retrieval_sentence_bias_scale,
            retrieval_local_bias_scale = inputs.retrieval_local_bias_scale,
            retrieval_sentence_embedding_scale = inputs.retrieval_sentence_embedding_scale,
            retrieval_type_compat_bias_scale = inputs.retrieval_type_compat_bias_scale,
            retrieval_dot_bias_scale = inputs.retrieval_dot_bias_scale,
            retrieval_compatibility_scale = inputs.retrieval_compatibility_scale,
            span_context_enabled = inputs.span_context_enabled,
            span_context_use_adjacent = inputs.span_context_use_adjacent,
            span_context_use_sentence = inputs.span_context_use_sentence,
            span_context_use_semantic = inputs.span_context_use_semantic,
            span_context_sentence_ids = hasproperty(inputs, :span_context_sentence_ids) ? inputs.span_context_sentence_ids : nothing,
            edge_v2_semantic_topk = inputs.edge_v2_semantic_topk,
            edge_v2_reverse_topk = inputs.edge_v2_reverse_topk,
            edge_v2_global_reserve = inputs.edge_v2_global_reserve,
            edge_v2_semantic_score_scale = inputs.edge_v2_semantic_score_scale,
            edge_v2_span_score_scale = inputs.edge_v2_span_score_scale,
            edge_v2_distance_penalty = inputs.edge_v2_distance_penalty,
            edge_v2_require_mutual = inputs.edge_v2_require_mutual,
            edge_v2_use_local_neighbors = inputs.edge_v2_use_local_neighbors,
            edge_v2_use_routed_buckets = inputs.edge_v2_use_routed_buckets,
            edge_v2_use_semantic_topk = inputs.edge_v2_use_semantic_topk,
            edge_v2_use_global_reserve = inputs.edge_v2_use_global_reserve,
        )
        pred_exhaustive_outputs, _ = model(pred_exhaustive_inputs, params, eval_state)
        accumulate_oracle_mode!(
            mode_accumulators[:pred_exhaustive],
            oracle_mode_stats(
                pred_exhaustive_outputs,
                targets;
                no_relation_id = no_relation_id,
                confidence_threshold = confidence_threshold,
                no_relation_margin = no_relation_margin,
                nonnull_probability_threshold = nonnull_probability_threshold,
                max_relations_per_head = max_relations_per_head,
                max_relations_per_tail = max_relations_per_tail,
                relation_confidence_thresholds = relation_confidence_thresholds,
                relation_allowed_type_pairs = relation_allowed_type_pairs,
                span_type_to_token_label_ids = span_type_to_token_label_ids,
                symmetric_relations = symmetric_relations,
                inverse_relation_map = inverse_relation_map,
            ),
        )

        CUDA.synchronize()
    end

    return (
        gold_gold = finalize_oracle_mode(mode_accumulators[:gold_gold], max_batches),
        gold_pred_pairs = finalize_oracle_mode(mode_accumulators[:gold_pred_pairs], max_batches),
        pred_exhaustive = finalize_oracle_mode(mode_accumulators[:pred_exhaustive], max_batches),
        pred_pred = finalize_oracle_mode(mode_accumulators[:pred_pred], max_batches),
    )
end

function format_oracle_mode_row(label::String, stats)
    return rpad(label, 24) * @sprintf(
        "%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.1f%10.4f",
        stats.total_loss,
        stats.relation_loss,
        stats.mention_top16_recall,
        stats.span_recall,
        stats.oracle_relation_coverage,
        stats.pair_recall,
        stats.pair_top16_recall,
        stats.relation_precision,
        stats.matched_pair_rank_mean,
        stats.relation_f1,
    )
end

function run_oracle_ladder(
    run_config::RETrainingRunConfig,
    checkpoint_path::String;
    max_relations_per_head::Int = 0,
    max_relations_per_tail::Int = 0,
    per_relation_threshold_spec::Union{Nothing,String} = nothing,
    type_constraints_mode::String = "off",
    type_constraints_min_count::Int = 1,
    relation_consistency_mode::String = "off",
    relation_consistency_min_count::Int = 1,
)
    context = load_eval_context(run_config)
    isempty(context.val_rows) && error("Validation data not found for oracle ladder evaluation.")
    ckpt = deserialize(checkpoint_path)
    model_config = get(ckpt, :model_config, context.model_config)
    vocab = get(ckpt, :vocab, context.vocab)
    entity_label_to_id = get(ckpt, :entity_label_to_id, context.entity_label_to_id)
    relation_label_to_id = get(ckpt, :relation_label_to_id, context.relation_label_to_id)
    relation_conf_thresholds = parse_relation_threshold_overrides(per_relation_threshold_spec, relation_label_to_id)
    type_constraints = resolve_decode_type_constraints(
        type_constraints_mode,
        context.train_rows,
        relation_label_to_id,
        entity_label_to_id;
        min_count = type_constraints_min_count,
    )
    relation_consistency = resolve_relation_consistency_constraints(
        relation_consistency_mode,
        context.train_rows,
        relation_label_to_id;
        min_count = relation_consistency_min_count,
    )
    model = SwammaRelationExtractor(model_config)
    params = tree_to_device(ckpt[:params])
    state = tree_to_device(ckpt[:state])

    ladder = evaluate_oracle_ladder(
        model,
        params,
        state,
        context.val_rows,
        vocab,
        entity_label_to_id,
        relation_label_to_id,
        model_config,
        run_config,
        context.pair_proposer_settings;
        confidence_threshold = 0.5f0,
        max_relations_per_head = max_relations_per_head,
        max_relations_per_tail = max_relations_per_tail,
        relation_confidence_thresholds = relation_conf_thresholds,
        relation_allowed_type_pairs = type_constraints.relation_allowed_type_pairs,
        span_type_to_token_label_ids = type_constraints.span_type_to_token_label_ids,
        symmetric_relations = relation_consistency.symmetric_relations,
        inverse_relation_map = relation_consistency.inverse_relation_map,
    )

    println("=" ^ 132)
    println("Oracle Ladder")
    println("=" ^ 132)
    println("Checkpoint: $(checkpoint_path)")
    println("Config: $(run_config.config_path)")
    println("Val rows: $(length(context.val_rows)) | Max eval batches: $(run_config.max_eval_batches)")
    println("Decode caps: head=$(max_relations_per_head), tail=$(max_relations_per_tail)")
    println("Per-relation thresholds: $(format_relation_threshold_overrides(relation_conf_thresholds, relation_label_to_id))")
    println("Type constraints: $(format_type_constraints_summary(type_constraints))")
    println("Relation consistency: $(format_relation_consistency_summary(relation_consistency))")
    println()
    println(
        rpad("mode", 24) *
        lpad("total", 10) *
        lpad("rel", 10) *
        lpad("ment_t16", 10) *
        lpad("span_r", 10) *
        lpad("oracle_rel", 10) *
        lpad("pair_r", 10) *
        lpad("pair_t16", 10) *
        lpad("rel_p", 10) *
        lpad("pair_rk", 10) *
        lpad("rel_f1", 10)
    )
    println("-" ^ 132)
    println(format_oracle_mode_row("gold spans + gold pairs", ladder.gold_gold))
    println(format_oracle_mode_row("gold spans + pred pairs", ladder.gold_pred_pairs))
    println(format_oracle_mode_row("pred spans + exhaustive", ladder.pred_exhaustive))
    println(format_oracle_mode_row("pred spans + pred pairs", ladder.pred_pred))
    reclaim_device_memory()
end

function format_threshold_sweep_row(threshold::Float32, stats)
    return lpad(@sprintf("%.2f", threshold), 10) *
           @sprintf(
               "%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f",
               stats.relation_precision,
               stats.relation_recall,
               stats.relation_f1,
               stats.oracle_relation_coverage,
               stats.pair_recall,
               stats.pair_top16_recall,
           )
end

function run_threshold_sweep(
    run_config::RETrainingRunConfig,
    checkpoint_path::String;
    thresholds::Vector{Float32} = Float32[],
    no_relation_margin::Float32 = 0.0f0,
    nonnull_probability_threshold::Float32 = 0.0f0,
    max_relations_per_head::Int = 0,
    max_relations_per_tail::Int = 0,
    per_relation_threshold_spec::Union{Nothing,String} = nothing,
    type_constraints_mode::String = "off",
    type_constraints_min_count::Int = 1,
    relation_consistency_mode::String = "off",
    relation_consistency_min_count::Int = 1,
)
    context = load_eval_context(run_config)
    isempty(context.val_rows) && error("Validation data not found for threshold sweep.")
    ckpt = deserialize(checkpoint_path)
    model_config = get(ckpt, :model_config, context.model_config)
    vocab = get(ckpt, :vocab, context.vocab)
    entity_label_to_id = get(ckpt, :entity_label_to_id, context.entity_label_to_id)
    relation_label_to_id = get(ckpt, :relation_label_to_id, context.relation_label_to_id)
    relation_conf_thresholds = parse_relation_threshold_overrides(per_relation_threshold_spec, relation_label_to_id)
    type_constraints = resolve_decode_type_constraints(
        type_constraints_mode,
        context.train_rows,
        relation_label_to_id,
        entity_label_to_id;
        min_count = type_constraints_min_count,
    )
    relation_consistency = resolve_relation_consistency_constraints(
        relation_consistency_mode,
        context.train_rows,
        relation_label_to_id;
        min_count = relation_consistency_min_count,
    )
    model = SwammaRelationExtractor(model_config)
    params = tree_to_device(ckpt[:params])
    state = tree_to_device(ckpt[:state])

    sweep_thresholds = isempty(thresholds) ? Float32[0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90] : sort(unique(thresholds))

    println("=" ^ 120)
    println("Relation Threshold Sweep")
    println("=" ^ 120)
    println("Checkpoint: $(checkpoint_path)")
    println("Config: $(run_config.config_path)")
    println("Val rows: $(length(context.val_rows)) | Max eval batches: $(run_config.max_eval_batches)")
    println("NO_RELATION margin: $(no_relation_margin)")
    println("Min non-null prob: $(nonnull_probability_threshold)")
    println("Decode caps: head=$(max_relations_per_head), tail=$(max_relations_per_tail)")
    println("Per-relation thresholds: $(format_relation_threshold_overrides(relation_conf_thresholds, relation_label_to_id))")
    println("Type constraints: $(format_type_constraints_summary(type_constraints))")
    println("Relation consistency: $(format_relation_consistency_summary(relation_consistency))")
    println()
    ladders = [
        (
            threshold = threshold,
            stats = evaluate_oracle_ladder(
                model,
                params,
                state,
                context.val_rows,
                vocab,
                entity_label_to_id,
                relation_label_to_id,
                model_config,
                run_config,
                context.pair_proposer_settings;
                confidence_threshold = threshold,
                no_relation_margin = no_relation_margin,
                nonnull_probability_threshold = nonnull_probability_threshold,
                max_relations_per_head = max_relations_per_head,
                max_relations_per_tail = max_relations_per_tail,
                relation_confidence_thresholds = relation_conf_thresholds,
                relation_allowed_type_pairs = type_constraints.relation_allowed_type_pairs,
                span_type_to_token_label_ids = type_constraints.span_type_to_token_label_ids,
                symmetric_relations = relation_consistency.symmetric_relations,
                inverse_relation_map = relation_consistency.inverse_relation_map,
            ),
        )
        for threshold in sweep_thresholds
    ]

    println("gold spans + gold pairs")
    println(rpad("threshold", 10) * lpad("rel_p", 10) * lpad("rel_r", 10) * lpad("rel_f1", 10) * lpad("oracle_rel", 10) * lpad("pair_r", 10) * lpad("pair_t16", 10))
    println("-" ^ 70)
    for item in ladders
        println(format_threshold_sweep_row(item.threshold, item.stats.gold_gold))
    end
    println()
    println("pred spans + exhaustive")
    println(rpad("threshold", 10) * lpad("rel_p", 10) * lpad("rel_r", 10) * lpad("rel_f1", 10) * lpad("oracle_rel", 10) * lpad("pair_r", 10) * lpad("pair_t16", 10))
    println("-" ^ 70)
    for item in ladders
        println(format_threshold_sweep_row(item.threshold, item.stats.pred_exhaustive))
    end
    println()
    println("pred spans + pred pairs")
    println(rpad("threshold", 10) * lpad("rel_p", 10) * lpad("rel_r", 10) * lpad("rel_f1", 10) * lpad("oracle_rel", 10) * lpad("pair_r", 10) * lpad("pair_t16", 10))
    println("-" ^ 70)
    for item in ladders
        println(format_threshold_sweep_row(item.threshold, item.stats.pred_pred))
    end
    reclaim_device_memory()
end

function run_margin_sweep(
    run_config::RETrainingRunConfig,
    checkpoint_path::String;
    margins::Vector{Float32} = Float32[],
    confidence_threshold::Float32 = 0.5f0,
    nonnull_probability_threshold::Float32 = 0.0f0,
    max_relations_per_head::Int = 0,
    max_relations_per_tail::Int = 0,
    per_relation_threshold_spec::Union{Nothing,String} = nothing,
    type_constraints_mode::String = "off",
    type_constraints_min_count::Int = 1,
    relation_consistency_mode::String = "off",
    relation_consistency_min_count::Int = 1,
)
    context = load_eval_context(run_config)
    isempty(context.val_rows) && error("Validation data not found for margin sweep.")
    ckpt = deserialize(checkpoint_path)
    model_config = get(ckpt, :model_config, context.model_config)
    vocab = get(ckpt, :vocab, context.vocab)
    entity_label_to_id = get(ckpt, :entity_label_to_id, context.entity_label_to_id)
    relation_label_to_id = get(ckpt, :relation_label_to_id, context.relation_label_to_id)
    relation_conf_thresholds = parse_relation_threshold_overrides(per_relation_threshold_spec, relation_label_to_id)
    type_constraints = resolve_decode_type_constraints(
        type_constraints_mode,
        context.train_rows,
        relation_label_to_id,
        entity_label_to_id;
        min_count = type_constraints_min_count,
    )
    relation_consistency = resolve_relation_consistency_constraints(
        relation_consistency_mode,
        context.train_rows,
        relation_label_to_id;
        min_count = relation_consistency_min_count,
    )
    model = SwammaRelationExtractor(model_config)
    params = tree_to_device(ckpt[:params])
    state = tree_to_device(ckpt[:state])

    sweep_margins = isempty(margins) ? Float32[0.0, 0.25, 0.5, 0.75, 1.0] : sort(unique(margins))
    println("=" ^ 120)
    println("NO_RELATION Margin Sweep")
    println("=" ^ 120)
    println("Checkpoint: $(checkpoint_path)")
    println("Config: $(run_config.config_path)")
    println("Val rows: $(length(context.val_rows)) | Max eval batches: $(run_config.max_eval_batches)")
    println("Confidence threshold fixed at $(confidence_threshold)")
    println("Min non-null prob fixed at $(nonnull_probability_threshold)")
    println("Decode caps fixed at head=$(max_relations_per_head), tail=$(max_relations_per_tail)")
    println("Per-relation thresholds: $(format_relation_threshold_overrides(relation_conf_thresholds, relation_label_to_id))")
    println("Type constraints: $(format_type_constraints_summary(type_constraints))")
    println("Relation consistency: $(format_relation_consistency_summary(relation_consistency))")
    println()

    ladders = [
        (
            margin = margin,
            stats = evaluate_oracle_ladder(
                model,
                params,
                state,
                context.val_rows,
                vocab,
                entity_label_to_id,
                relation_label_to_id,
                model_config,
                run_config,
                context.pair_proposer_settings;
                confidence_threshold = confidence_threshold,
                no_relation_margin = margin,
                nonnull_probability_threshold = nonnull_probability_threshold,
                max_relations_per_head = max_relations_per_head,
                max_relations_per_tail = max_relations_per_tail,
                relation_confidence_thresholds = relation_conf_thresholds,
                relation_allowed_type_pairs = type_constraints.relation_allowed_type_pairs,
                span_type_to_token_label_ids = type_constraints.span_type_to_token_label_ids,
                symmetric_relations = relation_consistency.symmetric_relations,
                inverse_relation_map = relation_consistency.inverse_relation_map,
            ),
        )
        for margin in sweep_margins
    ]

    header = rpad("margin", 10) * lpad("rel_p", 10) * lpad("rel_r", 10) * lpad("rel_f1", 10) * lpad("oracle_rel", 10) * lpad("pair_r", 10) * lpad("pair_t16", 10)
    println("pred spans + pred pairs")
    println(header)
    println("-" ^ 70)
    for item in ladders
        println(format_threshold_sweep_row(item.margin, item.stats.pred_pred))
    end
    println()
    println("pred spans + exhaustive")
    println(header)
    println("-" ^ 70)
    for item in ladders
        println(format_threshold_sweep_row(item.margin, item.stats.pred_exhaustive))
    end
    reclaim_device_memory()
end

function run_nonnull_sweep(
    run_config::RETrainingRunConfig,
    checkpoint_path::String;
    nonnull_values::Vector{Float32} = Float32[],
    confidence_threshold::Float32 = 0.5f0,
    no_relation_margin::Float32 = 0.0f0,
    max_relations_per_head::Int = 0,
    max_relations_per_tail::Int = 0,
    per_relation_threshold_spec::Union{Nothing,String} = nothing,
    type_constraints_mode::String = "off",
    type_constraints_min_count::Int = 1,
    relation_consistency_mode::String = "off",
    relation_consistency_min_count::Int = 1,
)
    context = load_eval_context(run_config)
    isempty(context.val_rows) && error("Validation data not found for non-null sweep.")
    ckpt = deserialize(checkpoint_path)
    model_config = get(ckpt, :model_config, context.model_config)
    vocab = get(ckpt, :vocab, context.vocab)
    entity_label_to_id = get(ckpt, :entity_label_to_id, context.entity_label_to_id)
    relation_label_to_id = get(ckpt, :relation_label_to_id, context.relation_label_to_id)
    relation_conf_thresholds = parse_relation_threshold_overrides(per_relation_threshold_spec, relation_label_to_id)
    type_constraints = resolve_decode_type_constraints(
        type_constraints_mode,
        context.train_rows,
        relation_label_to_id,
        entity_label_to_id;
        min_count = type_constraints_min_count,
    )
    relation_consistency = resolve_relation_consistency_constraints(
        relation_consistency_mode,
        context.train_rows,
        relation_label_to_id;
        min_count = relation_consistency_min_count,
    )
    model = SwammaRelationExtractor(model_config)
    params = tree_to_device(ckpt[:params])
    state = tree_to_device(ckpt[:state])

    sweep_values = isempty(nonnull_values) ? Float32[0.0, 0.2, 0.4, 0.6, 0.8] : sort(unique(nonnull_values))

    println("=" ^ 120)
    println("Non-Null Probability Sweep")
    println("=" ^ 120)
    println("Checkpoint: $(checkpoint_path)")
    println("Config: $(run_config.config_path)")
    println("Val rows: $(length(context.val_rows)) | Max eval batches: $(run_config.max_eval_batches)")
    println("Confidence threshold fixed at $(confidence_threshold)")
    println("NO_RELATION margin fixed at $(no_relation_margin)")
    println("Decode caps fixed at head=$(max_relations_per_head), tail=$(max_relations_per_tail)")
    println("Per-relation thresholds: $(format_relation_threshold_overrides(relation_conf_thresholds, relation_label_to_id))")
    println("Type constraints: $(format_type_constraints_summary(type_constraints))")
    println("Relation consistency: $(format_relation_consistency_summary(relation_consistency))")
    println()

    ladders = [
        (
            nonnull = nonnull,
            stats = evaluate_oracle_ladder(
                model,
                params,
                state,
                context.val_rows,
                vocab,
                entity_label_to_id,
                relation_label_to_id,
                model_config,
                run_config,
                context.pair_proposer_settings;
                confidence_threshold = confidence_threshold,
                no_relation_margin = no_relation_margin,
                nonnull_probability_threshold = nonnull,
                max_relations_per_head = max_relations_per_head,
                max_relations_per_tail = max_relations_per_tail,
                relation_confidence_thresholds = relation_conf_thresholds,
                relation_allowed_type_pairs = type_constraints.relation_allowed_type_pairs,
                span_type_to_token_label_ids = type_constraints.span_type_to_token_label_ids,
                symmetric_relations = relation_consistency.symmetric_relations,
                inverse_relation_map = relation_consistency.inverse_relation_map,
            ),
        )
        for nonnull in sweep_values
    ]

    header = rpad("nonnull", 10) * lpad("rel_p", 10) * lpad("rel_r", 10) * lpad("rel_f1", 10) * lpad("oracle_rel", 10) * lpad("pair_r", 10) * lpad("pair_t16", 10)
    println("pred spans + pred pairs")
    println(header)
    println("-" ^ 70)
    for item in ladders
        println(format_threshold_sweep_row(item.nonnull, item.stats.pred_pred))
    end
    println()
    println("pred spans + exhaustive")
    println(header)
    println("-" ^ 70)
    for item in ladders
        println(format_threshold_sweep_row(item.nonnull, item.stats.pred_exhaustive))
    end
    reclaim_device_memory()
end

function relation_prf(records::Vector{Tuple{Float32,Bool}}, gold_total::Int, threshold::Float32)
    predicted = 0
    true_positive = 0
    for (confidence, is_tp) in records
        confidence >= threshold || continue
        predicted += 1
        true_positive += is_tp ? 1 : 0
    end
    precision = safe_rate(true_positive, predicted)
    recall = safe_rate(true_positive, gold_total)
    f1 = precision + recall > 0 ? 2f0 * precision * recall / (precision + recall) : 0.0f0
    return (precision = precision, recall = recall, f1 = f1, predicted = predicted, true_positive = true_positive)
end

function run_auto_calibration(
    run_config::RETrainingRunConfig,
    checkpoint_path::String;
    confidence_threshold::Float32 = 0.70f0,
    no_relation_margin::Float32 = 0.30f0,
    nonnull_probability_threshold::Float32 = 0.0f0,
    min_predictions::Int = 8,
    candidate_thresholds::Vector{Float32} = Float32[],
    max_relations_per_head::Int = 0,
    max_relations_per_tail::Int = 0,
    per_relation_threshold_spec::Union{Nothing,String} = nothing,
    type_constraints_mode::String = "off",
    type_constraints_min_count::Int = 1,
    relation_consistency_mode::String = "off",
    relation_consistency_min_count::Int = 1,
)
    context = load_eval_context(run_config)
    isempty(context.val_rows) && error("Validation data not found for auto calibration.")
    ckpt = deserialize(checkpoint_path)
    model_config = get(ckpt, :model_config, context.model_config)
    vocab = get(ckpt, :vocab, context.vocab)
    entity_label_to_id = get(ckpt, :entity_label_to_id, context.entity_label_to_id)
    relation_label_to_id = get(ckpt, :relation_label_to_id, context.relation_label_to_id)
    no_relation_id = get(relation_label_to_id, "NO_RELATION", 1)
    base_overrides = parse_relation_threshold_overrides(per_relation_threshold_spec, relation_label_to_id)
    type_constraints = resolve_decode_type_constraints(
        type_constraints_mode,
        context.train_rows,
        relation_label_to_id,
        entity_label_to_id;
        min_count = type_constraints_min_count,
    )
    relation_consistency = resolve_relation_consistency_constraints(
        relation_consistency_mode,
        context.train_rows,
        relation_label_to_id;
        min_count = relation_consistency_min_count,
    )
    model = SwammaRelationExtractor(model_config)
    params = tree_to_device(ckpt[:params])
    state = Lux.testmode(tree_to_device(ckpt[:state]))

    thresholds = isempty(candidate_thresholds) ?
        Float32[confidence_threshold, min(confidence_threshold + 0.05f0, 0.99f0), min(confidence_threshold + 0.10f0, 0.99f0), 0.90f0, 0.95f0, 0.99f0] :
        sort(unique(candidate_thresholds))
    thresholds = sort(unique(vcat(thresholds, [confidence_threshold])))

    gold_per_relation = Dict{Int,Int}()
    records_per_relation = Dict{Int,Vector{Tuple{Float32,Bool}}}()
    max_batches = min(run_config.max_eval_batches, cld(length(context.val_rows), run_config.batch_size))
    retrieval_bias_settings = load_retrieval_bias_settings(run_config.config_path)
    calibration_rng = MersenneTwister(run_config.seed)

    for batch_idx in 1:max_batches
        start_idx = (batch_idx - 1) * run_config.batch_size + 1
        end_idx = min(batch_idx * run_config.batch_size, length(context.val_rows))
        batch_rows = context.val_rows[start_idx:end_idx]
        inputs, targets = make_batch(
            batch_rows,
            vocab,
            entity_label_to_id,
            relation_label_to_id,
            model_config,
            run_config;
            rng = calibration_rng,
        )
        auto_inputs = (
            token_ids = inputs.token_ids,
            token_mask = inputs.token_mask,
            retrieval_distance_bias_scale = retrieval_bias_settings.distance_scale,
            retrieval_type_bias_scale = retrieval_bias_settings.type_scale,
            retrieval_sentence_bias_scale = retrieval_bias_settings.sentence_scale,
            retrieval_local_bias_scale = retrieval_bias_settings.local_scale,
            retrieval_sentence_embedding_scale = retrieval_bias_settings.sentence_embedding_scale,
            retrieval_type_compat_bias_scale = retrieval_bias_settings.type_compat_scale,
            retrieval_dot_bias_scale = retrieval_bias_settings.dot_scale,
            retrieval_compatibility_scale = retrieval_bias_settings.compatibility_scale,
            span_context_enabled = true,
            span_context_use_adjacent = Bool(retrieval_bias_settings.span_context_use_adjacent),
            span_context_use_sentence = Bool(retrieval_bias_settings.span_context_use_sentence),
            span_context_use_semantic = Bool(retrieval_bias_settings.span_context_use_semantic),
            edge_v2_semantic_topk = retrieval_bias_settings.edge_v2_semantic_topk,
            edge_v2_reverse_topk = retrieval_bias_settings.edge_v2_reverse_topk,
            edge_v2_global_reserve = retrieval_bias_settings.edge_v2_global_reserve,
            edge_v2_semantic_score_scale = retrieval_bias_settings.edge_v2_semantic_score_scale,
            edge_v2_span_score_scale = retrieval_bias_settings.edge_v2_span_score_scale,
            edge_v2_distance_penalty = retrieval_bias_settings.edge_v2_distance_penalty,
            edge_v2_require_mutual = retrieval_bias_settings.edge_v2_require_mutual,
            edge_v2_use_local_neighbors = retrieval_bias_settings.edge_v2_use_local_neighbors,
            edge_v2_use_routed_buckets = retrieval_bias_settings.edge_v2_use_routed_buckets,
            edge_v2_use_semantic_topk = retrieval_bias_settings.edge_v2_use_semantic_topk,
            edge_v2_use_global_reserve = retrieval_bias_settings.edge_v2_use_global_reserve,
        )
        outputs, _ = model(auto_inputs, params, state)

        target_spans = to_cpu(targets.spans)
        target_relation_pairs = to_cpu(targets.relation_pairs)
        target_relation_labels = to_cpu(targets.relation_labels)
        target_relation_mask = to_cpu(targets.relation_mask)
        target_relation_targets = Float32.(to_cpu(targets.relation_targets))

        predicted_batches = collect_predicted_relation_candidates(
            outputs;
            no_relation_id = no_relation_id,
            confidence_threshold = confidence_threshold,
            no_relation_margin = no_relation_margin,
            nonnull_probability_threshold = nonnull_probability_threshold,
            max_relations_per_head = max_relations_per_head,
            max_relations_per_tail = max_relations_per_tail,
            relation_confidence_thresholds = base_overrides,
            relation_allowed_type_pairs = type_constraints.relation_allowed_type_pairs,
            span_type_to_token_label_ids = type_constraints.span_type_to_token_label_ids,
            symmetric_relations = relation_consistency.symmetric_relations,
            inverse_relation_map = relation_consistency.inverse_relation_map,
        )

        for b in 1:length(predicted_batches)
            gold_relation_set = collect_gold_relation_set(
                target_spans,
                target_relation_pairs,
                target_relation_labels,
                target_relation_mask,
                target_relation_targets,
                b,
            )
            for rel in gold_relation_set
                rel_id = Int(rel[5])
                rel_id == no_relation_id && continue
                gold_per_relation[rel_id] = get(gold_per_relation, rel_id, 0) + 1
            end
            for item in predicted_batches[b]
                rel_id = item.label_id
                rel_id == no_relation_id && continue
                rel_records = get!(records_per_relation, rel_id, Tuple{Float32,Bool}[])
                is_tp = item.relation in gold_relation_set
                push!(rel_records, (item.confidence, is_tp))
            end
        end
        CUDA.synchronize()
    end

    id_to_label = Dict{Int,String}(id => label for (label, id) in relation_label_to_id)
    suggested_overrides = copy(base_overrides)
    change_rows = NamedTuple[]

    for (rel_id, records) in sort(collect(records_per_relation); by = first)
        rel_id == no_relation_id && continue
        gold_total = get(gold_per_relation, rel_id, 0)
        base_threshold = get(base_overrides, rel_id, confidence_threshold)
        base_stats = relation_prf(records, gold_total, base_threshold)
        base_stats.predicted >= min_predictions || continue

        best_threshold = base_threshold
        best_stats = base_stats
        for threshold in thresholds
            threshold >= base_threshold || continue
            stats = relation_prf(records, gold_total, threshold)
            improves_f1 = stats.f1 > best_stats.f1 + 1f-6
            ties_f1 = abs(stats.f1 - best_stats.f1) <= 1f-6
            improves_precision = stats.precision > best_stats.precision + 1f-6
            if improves_f1 || (ties_f1 && improves_precision) || (ties_f1 && abs(stats.precision - best_stats.precision) <= 1f-6 && threshold > best_threshold)
                best_threshold = threshold
                best_stats = stats
            end
        end

        strict_improves = (
            best_stats.f1 > base_stats.f1 + 1f-6 ||
            (abs(best_stats.f1 - base_stats.f1) <= 1f-6 && best_stats.precision > base_stats.precision + 1f-6)
        )
        if best_threshold > base_threshold + 1f-6 &&
           strict_improves &&
           best_stats.predicted > 0 &&
           (base_stats.true_positive > 0 || best_stats.true_positive > 0)
            suggested_overrides[rel_id] = best_threshold
            push!(change_rows, (
                rel_id = rel_id,
                label = get(id_to_label, rel_id, string(rel_id)),
                base_threshold = base_threshold,
                best_threshold = best_threshold,
                base_precision = base_stats.precision,
                best_precision = best_stats.precision,
                base_recall = base_stats.recall,
                best_recall = best_stats.recall,
                base_f1 = base_stats.f1,
                best_f1 = best_stats.f1,
                predictions = base_stats.predicted,
                gold = gold_total,
            ))
        end
    end

    baseline_ladder = evaluate_oracle_ladder(
        model,
        params,
        state,
        context.val_rows,
        vocab,
        entity_label_to_id,
        relation_label_to_id,
        model_config,
        run_config,
        context.pair_proposer_settings;
        confidence_threshold = confidence_threshold,
        no_relation_margin = no_relation_margin,
        nonnull_probability_threshold = nonnull_probability_threshold,
        max_relations_per_head = max_relations_per_head,
        max_relations_per_tail = max_relations_per_tail,
        relation_confidence_thresholds = base_overrides,
        relation_allowed_type_pairs = type_constraints.relation_allowed_type_pairs,
        span_type_to_token_label_ids = type_constraints.span_type_to_token_label_ids,
        symmetric_relations = relation_consistency.symmetric_relations,
        inverse_relation_map = relation_consistency.inverse_relation_map,
    )
    calibrated_ladder = evaluate_oracle_ladder(
        model,
        params,
        state,
        context.val_rows,
        vocab,
        entity_label_to_id,
        relation_label_to_id,
        model_config,
        run_config,
        context.pair_proposer_settings;
        confidence_threshold = confidence_threshold,
        no_relation_margin = no_relation_margin,
        nonnull_probability_threshold = nonnull_probability_threshold,
        max_relations_per_head = max_relations_per_head,
        max_relations_per_tail = max_relations_per_tail,
        relation_confidence_thresholds = suggested_overrides,
        relation_allowed_type_pairs = type_constraints.relation_allowed_type_pairs,
        span_type_to_token_label_ids = type_constraints.span_type_to_token_label_ids,
        symmetric_relations = relation_consistency.symmetric_relations,
        inverse_relation_map = relation_consistency.inverse_relation_map,
    )

    baseline_global_f1 = baseline_ladder.pred_pred.relation_f1
    calibrated_global_f1 = calibrated_ladder.pred_pred.relation_f1
    global_accept = calibrated_global_f1 >= baseline_global_f1 - 1f-6
    accepted_overrides = global_accept ? suggested_overrides : base_overrides
    accepted_ladder = global_accept ? calibrated_ladder : baseline_ladder

    println("=" ^ 132)
    println("Auto Calibration (Per-Relation Threshold Proposals)")
    println("=" ^ 132)
    println("Checkpoint: $(checkpoint_path)")
    println("Config: $(run_config.config_path)")
    println("Val rows: $(length(context.val_rows)) | Max eval batches: $(run_config.max_eval_batches)")
    println("Base decode: threshold=$(confidence_threshold), margin=$(no_relation_margin), nonnull=$(nonnull_probability_threshold)")
    println("Decode caps: head=$(max_relations_per_head), tail=$(max_relations_per_tail)")
    println("Type constraints: $(format_type_constraints_summary(type_constraints))")
    println("Relation consistency: $(format_relation_consistency_summary(relation_consistency))")
    println("Base per-relation thresholds: $(format_relation_threshold_overrides(base_overrides, relation_label_to_id))")
    println("Suggested per-relation thresholds (raw): $(format_relation_threshold_overrides(suggested_overrides, relation_label_to_id))")
    println("Accepted per-relation thresholds: $(format_relation_threshold_overrides(accepted_overrides, relation_label_to_id))")
    if !global_accept
        println("Global gate: rejected raw suggestions because global rel_f1 dropped ($(round(calibrated_global_f1, digits=4)) < $(round(baseline_global_f1, digits=4))).")
    end
    println()

    println("pred spans + pred pairs (global)")
    println(rpad("setting", 14) * lpad("rel_p", 10) * lpad("rel_r", 10) * lpad("rel_f1", 10) * lpad("oracle_rel", 10) * lpad("pair_r", 10) * lpad("pair_t16", 10))
    println("-" ^ 74)
    baseline_row = format_threshold_sweep_row(0.0f0, baseline_ladder.pred_pred)[11:end]
    calibrated_row = format_threshold_sweep_row(0.0f0, calibrated_ladder.pred_pred)[11:end]
    accepted_row = format_threshold_sweep_row(0.0f0, accepted_ladder.pred_pred)[11:end]
    println(rpad("baseline", 14) * baseline_row)
    println(rpad("calibrated", 14) * calibrated_row)
    if !global_accept
        println(rpad("accepted", 14) * accepted_row)
    end
    println()

    if isempty(change_rows)
        println("No relation-specific threshold updates met the improvement criteria.")
    else
        println("Proposed relation updates (sorted by ΔF1 desc):")
        println(rpad("relation", 16) * lpad("base_t", 8) * lpad("best_t", 8) * lpad("base_p", 9) * lpad("best_p", 9) * lpad("base_r", 9) * lpad("best_r", 9) * lpad("base_f1", 9) * lpad("best_f1", 9) * lpad("pred", 8) * lpad("gold", 8))
        println("-" ^ 110)
        for row in sort(change_rows; by = r -> (r.best_f1 - r.base_f1, r.best_precision - r.base_precision), rev = true)
            println(
                rpad(row.label, 16) *
                @sprintf("%8.2f%8.2f%9.4f%9.4f%9.4f%9.4f%9.4f%9.4f%8d%8d",
                    row.base_threshold,
                    row.best_threshold,
                    row.base_precision,
                    row.best_precision,
                    row.base_recall,
                    row.best_recall,
                    row.base_f1,
                    row.best_f1,
                    row.predictions,
                    row.gold,
                )
            )
        end
    end
    reclaim_device_memory()
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
    # Background HF upload if configured
    hf_repo = get(ENV, "SWAMMA_HF_REPO", "")
    if !isempty(hf_repo)
        upload_script = joinpath(@__DIR__, "upload_checkpoint_hf.py")
        venv_python = joinpath(@__DIR__, "..", ".venv", "bin", "python3")
        if isfile(upload_script) && isfile(venv_python)
            config_path = hasfield(typeof(run_config), :config_path) ? run_config.config_path : ""
            cmd = `$venv_python $upload_script --checkpoint $path --repo $hf_repo --config $config_path --commit $(string("step ", step, " loss ", round(loss, digits=4)))`
            @async begin
                println("[hf] Uploading checkpoint to $hf_repo ...")
                flush(stdout)
                run(pipeline(cmd; stdout=devnull, stderr=devnull))
                println("[hf] Upload complete: $(basename(path))")
                flush(stdout)
            end
        end
    end
end

function reclaim_device_memory()
    GC.gc(false)
    CUDA.reclaim()
    return nothing
end

function load_eval_context(run_config::RETrainingRunConfig)
    base_config = load_relation_extraction_config(run_config.config_path)
    pair_proposer_settings = load_pair_proposer_settings(run_config.config_path)
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
        pair_proposer_settings = pair_proposer_settings,
    )
end

function build_relation_logit_adjustment(
    rows,
    relation_label_to_id::Dict{String,Int};
    smoothing::Float32 = 1.0f0,
)::Vector{Float32}
    num_relations = length(relation_label_to_id)
    num_relations > 0 || return Float32[]
    no_relation_id = get(relation_label_to_id, "NO_RELATION", 1)
    counts = fill(smoothing, num_relations)
    if 1 <= no_relation_id <= num_relations
        counts[no_relation_id] = 1.0f0
    end

    for row in rows
        haskey(row, :relations) || continue
        for rel in row.relations
            rel_id = get(relation_label_to_id, String(rel.label), 0)
            (1 <= rel_id <= num_relations) || continue
            rel_id == no_relation_id && continue
            counts[rel_id] += 1.0f0
        end
    end

    pos_total = sum(counts) - ((1 <= no_relation_id <= num_relations) ? counts[no_relation_id] : 0.0f0)
    pos_total > 0.0f0 || (pos_total = Float32(max(num_relations - 1, 1)))

    log_adjustment = zeros(Float32, num_relations)
    for rel_id in 1:num_relations
        rel_id == no_relation_id && continue
        prior = max(counts[rel_id] / pos_total, 1f-6)
        log_adjustment[rel_id] = log(prior)
    end
    return log_adjustment
end

function override_relation_config(config::RelationExtractionConfig; kwargs...)
    base = (; (field => getfield(config, field) for field in fieldnames(RelationExtractionConfig))...)
    return RelationExtractionConfig(; base..., kwargs...)
end

function parse_relation_threshold_overrides(
    spec::Union{Nothing,String},
    relation_label_to_id::Dict{String,Int},
)::Dict{Int,Float32}
    thresholds = Dict{Int,Float32}()
    spec === nothing && return thresholds
    stripped = strip(spec)
    isempty(stripped) && return thresholds

    for token_raw in split(stripped, ",")
        token = strip(token_raw)
        isempty(token) && continue
        sep = occursin("=", token) ? "=" : occursin(":", token) ? ":" : nothing
        sep === nothing && error("Invalid per-relation threshold entry `$(token)`. Use LABEL=VALUE or ID=VALUE.")
        parts = split(token, sep; limit = 2)
        length(parts) == 2 || error("Invalid per-relation threshold entry `$(token)`.")
        key_raw = strip(parts[1])
        value = Float32(parse(Float64, strip(parts[2])))
        (0.0f0 <= value <= 1.0f0) || error("Per-relation threshold must be in [0,1], got $(value) for `$(token)`.")
        relation_id = begin
            parsed_id = tryparse(Int, key_raw)
            if parsed_id !== nothing
                parsed_id
            else
                get(relation_label_to_id, key_raw, 0)
            end
        end
        relation_id > 0 || error("Unknown relation label/id `$(key_raw)` in per-relation thresholds.")
        thresholds[Int(relation_id)] = value
    end
    return thresholds
end

function format_relation_threshold_overrides(
    thresholds::Dict{Int,Float32},
    relation_label_to_id::Dict{String,Int},
)::String
    isempty(thresholds) && return "none"
    id_to_label = Dict{Int,String}(id => label for (label, id) in relation_label_to_id)
    entries = [
        string(get(id_to_label, relation_id, string(relation_id)), "=", @sprintf("%.2f", threshold))
        for (relation_id, threshold) in sort(collect(thresholds); by = first)
    ]
    return join(entries, ", ")
end

normalize_entity_type_name(raw::AbstractString) = uppercase(replace(String(raw), r"^[BI]-" => ""))

function infer_relation_index_offset(relations, entity_count::Int)::Int
    entity_count <= 0 && return 0
    isempty(relations) && return 0
    valid_offset0 = 0
    valid_offset1 = 0
    saw_zero = false
    saw_upper = false

    for rel in relations
        head_raw = Int(rel.head)
        tail_raw = Int(rel.tail)
        saw_zero |= (head_raw == 0 || tail_raw == 0)
        saw_upper |= (head_raw == entity_count || tail_raw == entity_count)
        if 1 <= head_raw <= entity_count && 1 <= tail_raw <= entity_count
            valid_offset0 += 1
        end
        head_plus = head_raw + 1
        tail_plus = tail_raw + 1
        if 1 <= head_plus <= entity_count && 1 <= tail_plus <= entity_count
            valid_offset1 += 1
        end
    end

    if valid_offset1 > valid_offset0
        return 1
    elseif valid_offset0 > valid_offset1
        return 0
    elseif saw_zero
        return 1
    elseif saw_upper
        return 0
    else
        return 1
    end
end

function build_relation_type_constraints(
    rows,
    relation_label_to_id::Dict{String,Int};
    min_count::Int = 1,
)::Tuple{Dict{String,Int},Dict{Int,Set{Tuple{Int,Int}}},Int}
    min_count = max(min_count, 1)
    no_relation_id = get(relation_label_to_id, "NO_RELATION", 1)
    type_to_id = Dict{String,Int}()
    triple_counts = Dict{Tuple{Int,Int,Int},Int}()

    for row in rows
        entities = haskey(row, :entities) ? collect(row.entities) : Any[]
        entity_count = length(entities)
        entity_type_ids = zeros(Int, entity_count)
        for i in 1:entity_count
            type_name = normalize_entity_type_name(String(entities[i].label))
            type_id = get!(type_to_id, type_name) do
                length(type_to_id) + 1
            end
            entity_type_ids[i] = type_id
        end

        haskey(row, :relations) || continue
        relations = collect(row.relations)
        offset = infer_relation_index_offset(relations, entity_count)
        for rel in relations
            rel_id = get(relation_label_to_id, String(rel.label), 0)
            rel_id == 0 && continue
            rel_id == no_relation_id && continue
            head_raw = Int(rel.head)
            tail_raw = Int(rel.tail)
            head_idx = head_raw + offset
            tail_idx = tail_raw + offset
            if !(1 <= head_idx <= entity_count && 1 <= tail_idx <= entity_count)
                continue
            end
            head_type = entity_type_ids[head_idx]
            tail_type = entity_type_ids[tail_idx]
            head_type > 0 && tail_type > 0 || continue
            key = (rel_id, head_type, tail_type)
            triple_counts[key] = get(triple_counts, key, 0) + 1
        end
    end

    relation_allowed_pairs = Dict{Int,Set{Tuple{Int,Int}}}()
    kept = 0
    for ((rel_id, head_type, tail_type), count) in triple_counts
        count >= min_count || continue
        push!(get!(relation_allowed_pairs, rel_id, Set{Tuple{Int,Int}}()), (head_type, tail_type))
        kept += 1
    end

    return type_to_id, relation_allowed_pairs, kept
end

function build_entity_type_to_token_label_ids(
    type_to_id::Dict{String,Int},
    entity_label_to_id::Dict{String,Int},
)::Dict{Int,Tuple{Int,Int}}
    type_to_token_ids = Dict{Int,Tuple{Int,Int}}()
    for (type_name, type_id) in type_to_id
        b_id = get(entity_label_to_id, "B-$type_name", 0)
        i_id = get(entity_label_to_id, "I-$type_name", 0)
        (b_id > 0 || i_id > 0) || continue
        type_to_token_ids[type_id] = (b_id, i_id)
    end
    return type_to_token_ids
end

function infer_span_type_ids(
    outputs,
    type_to_token_label_ids::Dict{Int,Tuple{Int,Int}},
)
    spans = to_cpu(outputs.spans)
    span_mask = to_cpu(outputs.span_mask)
    entity_logits = Float32.(to_cpu(outputs.entity_logits))
    max_spans = size(spans, 2)
    batch_size = size(spans, 3)
    seq_len = size(entity_logits, 2)
    span_type_ids = zeros(Int, max_spans, batch_size)
    isempty(type_to_token_label_ids) && return span_type_ids

    type_entries = collect(type_to_token_label_ids)
    for b in 1:batch_size
        for span_idx in 1:max_spans
            span_mask[span_idx, b] || continue
            start_idx = clamp(Int(spans[1, span_idx, b]), 1, seq_len)
            end_idx = clamp(Int(spans[2, span_idx, b]), start_idx, seq_len)
            best_type = 0
            best_score = -Inf32
            for (type_id, (b_id, i_id)) in type_entries
                type_score = -Inf32
                if b_id > 0
                    type_score = max(type_score, maximum(@view(entity_logits[b_id, start_idx:end_idx, b])))
                end
                if i_id > 0
                    type_score = max(type_score, maximum(@view(entity_logits[i_id, start_idx:end_idx, b])))
                end
                if type_score > best_score
                    best_score = type_score
                    best_type = type_id
                end
            end
            span_type_ids[span_idx, b] = best_type
        end
    end
    return span_type_ids
end

function build_relation_consistency_rules(
    rows,
    relation_label_to_id::Dict{String,Int};
    min_count::Int = 1,
)
    min_count = max(min_count, 1)
    no_relation_id = get(relation_label_to_id, "NO_RELATION", 1)
    reverse_pair_counts = Dict{Tuple{Int,Int},Int}()
    reverse_total_by_relation = Dict{Int,Int}()

    for row in rows
        entities = haskey(row, :entities) ? collect(row.entities) : Any[]
        entity_count = length(entities)
        entity_count == 0 && continue
        haskey(row, :relations) || continue
        relations = collect(row.relations)
        isempty(relations) && continue
        offset = infer_relation_index_offset(relations, entity_count)

        edge_labels = Dict{Tuple{Int,Int},Int}()
        for rel in relations
            rel_id = get(relation_label_to_id, String(rel.label), 0)
            (rel_id == 0 || rel_id == no_relation_id) && continue
            head_idx = Int(rel.head) + offset
            tail_idx = Int(rel.tail) + offset
            if !(1 <= head_idx <= entity_count && 1 <= tail_idx <= entity_count)
                continue
            end
            edge_labels[(head_idx, tail_idx)] = rel_id
        end

        for ((head_idx, tail_idx), rel_id) in edge_labels
            reverse_key = (tail_idx, head_idx)
            haskey(edge_labels, reverse_key) || continue
            reverse_id = edge_labels[reverse_key]
            key = (rel_id, reverse_id)
            reverse_pair_counts[key] = get(reverse_pair_counts, key, 0) + 1
            reverse_total_by_relation[rel_id] = get(reverse_total_by_relation, rel_id, 0) + 1
        end
    end

    best_reverse = Dict{Int,Tuple{Int,Int}}()
    for ((rel_id, reverse_id), count) in reverse_pair_counts
        prev = get(best_reverse, rel_id, (0, 0))
        if count > prev[2]
            best_reverse[rel_id] = (reverse_id, count)
        end
    end

    symmetric_relations = Set{Int}()
    provisional_inverse = Dict{Int,Int}()
    for (rel_id, (reverse_id, count)) in best_reverse
        count >= min_count || continue
        total = get(reverse_total_by_relation, rel_id, 0)
        total > 0 || continue
        if reverse_id == rel_id
            push!(symmetric_relations, rel_id)
        else
            provisional_inverse[rel_id] = reverse_id
        end
    end

    inverse_relation_map = Dict{Int,Int}()
    for (rel_id, reverse_id) in provisional_inverse
        if get(provisional_inverse, reverse_id, 0) == rel_id
            inverse_relation_map[rel_id] = reverse_id
        end
    end

    paired_inverse_edges = 0
    seen_pairs = Set{Tuple{Int,Int}}()
    for (rel_id, reverse_id) in inverse_relation_map
        pair = rel_id < reverse_id ? (rel_id, reverse_id) : (reverse_id, rel_id)
        if !(pair in seen_pairs)
            push!(seen_pairs, pair)
            paired_inverse_edges += 1
        end
    end

    return (
        symmetric_relations = symmetric_relations,
        inverse_relation_map = inverse_relation_map,
        symmetric_count = length(symmetric_relations),
        inverse_pair_count = paired_inverse_edges,
    )
end

function resolve_decode_type_constraints(
    mode::String,
    train_rows,
    relation_label_to_id::Dict{String,Int},
    entity_label_to_id::Dict{String,Int};
    min_count::Int = 1,
)
    normalized_mode = lowercase(strip(mode))
    normalized_mode in ("off", "hard") ||
        error("Unsupported type constraint mode `$(mode)`. Supported: off, hard.")
    min_count = max(min_count, 1)
    empty_rules = Dict{Int,Set{Tuple{Int,Int}}}()
    empty_type_map = Dict{Int,Tuple{Int,Int}}()

    if normalized_mode == "off"
        return (
            enabled = false,
            mode = normalized_mode,
            min_count = min_count,
            relation_allowed_type_pairs = empty_rules,
            span_type_to_token_label_ids = empty_type_map,
            num_types = 0,
            num_relations = 0,
            num_rules = 0,
        )
    end

    type_to_id, relation_allowed_pairs, kept_rules = build_relation_type_constraints(
        train_rows,
        relation_label_to_id;
        min_count = min_count,
    )
    span_type_to_token_label_ids = build_entity_type_to_token_label_ids(type_to_id, entity_label_to_id)
    enabled = !isempty(relation_allowed_pairs) && !isempty(span_type_to_token_label_ids)
    return (
        enabled = enabled,
        mode = normalized_mode,
        min_count = min_count,
        relation_allowed_type_pairs = enabled ? relation_allowed_pairs : empty_rules,
        span_type_to_token_label_ids = enabled ? span_type_to_token_label_ids : empty_type_map,
        num_types = length(type_to_id),
        num_relations = length(relation_allowed_pairs),
        num_rules = kept_rules,
    )
end

function format_type_constraints_summary(type_constraints)::String
    if type_constraints.mode == "off"
        return "off"
    end
    status = type_constraints.enabled ? "on" : "inactive"
    return @sprintf(
        "%s(mode=%s,min_count=%d,types=%d,relations=%d,rules=%d)",
        status,
        type_constraints.mode,
        type_constraints.min_count,
        type_constraints.num_types,
        type_constraints.num_relations,
        type_constraints.num_rules,
    )
end

function resolve_relation_consistency_constraints(
    mode::String,
    train_rows,
    relation_label_to_id::Dict{String,Int};
    min_count::Int = 1,
)
    normalized_mode = lowercase(strip(mode))
    normalized_mode in ("off", "resolve") ||
        error("Unsupported relation consistency mode `$(mode)`. Supported: off, resolve.")
    min_count = max(min_count, 1)
    empty_symmetric = Set{Int}()
    empty_inverse = Dict{Int,Int}()

    if normalized_mode == "off"
        return (
            enabled = false,
            mode = normalized_mode,
            min_count = min_count,
            symmetric_relations = empty_symmetric,
            inverse_relation_map = empty_inverse,
            symmetric_count = 0,
            inverse_pair_count = 0,
        )
    end

    stats = build_relation_consistency_rules(
        train_rows,
        relation_label_to_id;
        min_count = min_count,
    )
    enabled = !isempty(stats.symmetric_relations) || !isempty(stats.inverse_relation_map)
    return (
        enabled = enabled,
        mode = normalized_mode,
        min_count = min_count,
        symmetric_relations = enabled ? stats.symmetric_relations : empty_symmetric,
        inverse_relation_map = enabled ? stats.inverse_relation_map : empty_inverse,
        symmetric_count = stats.symmetric_count,
        inverse_pair_count = stats.inverse_pair_count,
    )
end

function format_relation_consistency_summary(consistency)::String
    if consistency.mode == "off"
        return "off"
    end
    status = consistency.enabled ? "on" : "inactive"
    return @sprintf(
        "%s(mode=%s,min_count=%d,symmetric=%d,inverse_pairs=%d)",
        status,
        consistency.mode,
        consistency.min_count,
        consistency.symmetric_count,
        consistency.inverse_pair_count,
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
    relation_logit_adjustment_tau = load_relation_logit_adjustment_tau(run_config.config_path)
    relation_logit_adjustment = relation_logit_adjustment_tau > 0.0f0 ?
        build_relation_logit_adjustment(context.train_rows, context.relation_label_to_id) :
        nothing

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
            lpad("prop_rel", 10) *
            lpad("prop_conf", 10) *
            lpad("prop_tot", 10) *
            lpad("span_r", 10) *
            lpad("pair_r", 10) *
            lpad("pair_t16", 10) *
            lpad("pair_rk", 10) *
            lpad("miss_s", 10) *
            lpad("miss_m", 10) *
            lpad("miss_l", 10) *
            lpad("rel_p", 10) *
            lpad("rel_r", 10) *
            lpad("rel_f1", 10) *
            lpad("ev_ent", 10) *
            lpad("ev_max", 10) *
            lpad("ev_eff", 10) *
            lpad("ev_t1", 8))
    println("-" ^ 238)

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
            context.pair_proposer_settings;
            relation_logit_adjustment_tau = relation_logit_adjustment_tau,
            relation_logit_adjustment = relation_logit_adjustment,
        )

        println(
            rpad(basename(checkpoint_path), 22) *
            lpad(step, 8) *
            @sprintf("%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.1f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.2f%8d",
                eval_stats.total_loss,
                eval_stats.entity_loss,
                eval_stats.boundary_loss,
                eval_stats.relation_loss,
                eval_stats.confidence_loss,
                eval_stats.proposal_relation_loss,
                eval_stats.proposal_confidence_loss,
                eval_stats.proposal_total_loss,
                eval_stats.span_recall,
                eval_stats.pair_recall,
                eval_stats.pair_top16_recall,
                eval_stats.matched_pair_rank_mean,
                eval_stats.missed_pair_short_share,
                eval_stats.missed_pair_medium_share,
                eval_stats.missed_pair_long_share,
                eval_stats.relation_precision,
                eval_stats.relation_recall,
                eval_stats.relation_f1,
                eval_stats.evidence_entropy,
                eval_stats.evidence_max_weight,
                eval_stats.evidence_effective_tokens,
                eval_stats.evidence_top1_token,
            )
        )
        reclaim_device_memory()
    end
end

function format_mention_sweep_row(mode::Symbol, span_budget::Int, stats)
    return rpad(String(mode), 12) *
           lpad(span_budget, 8) *
           @sprintf(
               "%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.1f%10.4f",
               stats.total_loss,
               stats.mention_top16_recall,
               stats.mention_top32_recall,
               stats.span_recall,
               stats.oracle_relation_coverage,
               stats.pair_recall,
               stats.pair_top16_recall,
               stats.relation_precision,
               stats.matched_pair_rank_mean,
               stats.relation_f1,
           )
end

function run_mention_sweep(
    run_config::RETrainingRunConfig,
    checkpoint_path::String;
    budgets::Vector{Int} = Int[],
    modes::Vector{Symbol} = Symbol[],
)
    context = load_eval_context(run_config)
    isempty(context.val_rows) && error("Validation data not found for mention sweep.")
    relation_logit_adjustment_tau = load_relation_logit_adjustment_tau(run_config.config_path)
    relation_logit_adjustment = relation_logit_adjustment_tau > 0.0f0 ?
        build_relation_logit_adjustment(context.train_rows, context.relation_label_to_id) :
        nothing
    ckpt = deserialize(checkpoint_path)
    checkpoint_config = get(ckpt, :model_config, context.model_config)
    vocab = get(ckpt, :vocab, context.vocab)
    entity_label_to_id = get(ckpt, :entity_label_to_id, context.entity_label_to_id)
    relation_label_to_id = get(ckpt, :relation_label_to_id, context.relation_label_to_id)
    params = tree_to_device(ckpt[:params])
    state = tree_to_device(ckpt[:state])

    sweep_budgets = isempty(budgets) ? [32, 64, 96, 128, 192] : sort(unique(budgets))
    sweep_modes = isempty(modes) ? [:heuristic, :hybrid, :learned] : unique(modes)

    println("=" ^ 140)
    println("Mention Sweep")
    println("=" ^ 140)
    println("Checkpoint: $(checkpoint_path)")
    println("Config: $(run_config.config_path)")
    println("Val rows: $(length(context.val_rows)) | Max eval batches: $(run_config.max_eval_batches)")
    println()
    println(
        rpad("mode", 12) *
        lpad("budget", 8) *
        lpad("total", 10) *
        lpad("ment_t16", 10) *
        lpad("ment_t32", 10) *
        lpad("span_r", 10) *
        lpad("oracle_rel", 10) *
        lpad("pair_r", 10) *
        lpad("pair_t16", 10) *
        lpad("rel_p", 10) *
        lpad("pair_rk", 10) *
        lpad("rel_f1", 10)
    )
    println("-" ^ 140)

    for mode in sweep_modes
        for span_budget in sweep_budgets
            eval_config = override_relation_config(
                checkpoint_config;
                max_candidate_spans = span_budget,
                mention_score_mode = mode,
            )
            model = SwammaRelationExtractor(eval_config)
            eval_stats = evaluate_model(
                model,
                params,
                state,
                context.val_rows,
                vocab,
                entity_label_to_id,
                relation_label_to_id,
                eval_config,
                run_config,
                context.pair_proposer_settings;
                relation_logit_adjustment_tau = relation_logit_adjustment_tau,
                relation_logit_adjustment = relation_logit_adjustment,
            )
            println(format_mention_sweep_row(mode, span_budget, eval_stats))
            reclaim_device_memory()
        end
    end
end

function format_evidence_pooling_row(mode::Symbol, stats)
    return rpad(String(mode), 10) *
           @sprintf(
               "%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.2f%8d",
               stats.relation_precision,
               stats.relation_recall,
               stats.relation_f1,
               stats.oracle_relation_coverage,
               stats.pair_recall,
               stats.pair_top16_recall,
               stats.evidence_entropy,
               stats.evidence_max_weight,
               stats.evidence_effective_tokens,
               stats.evidence_top1_token,
           )
end

function run_evidence_pooling_sweep(
    run_config::RETrainingRunConfig,
    checkpoint_path::String;
    modes::Vector{Symbol} = Symbol[],
)
    context = load_eval_context(run_config)
    isempty(context.val_rows) && error("Validation data not found for evidence pooling sweep.")
    relation_logit_adjustment_tau = load_relation_logit_adjustment_tau(run_config.config_path)
    relation_logit_adjustment = relation_logit_adjustment_tau > 0.0f0 ?
        build_relation_logit_adjustment(context.train_rows, context.relation_label_to_id) :
        nothing
    ckpt = deserialize(checkpoint_path)
    checkpoint_config = get(ckpt, :model_config, context.model_config)
    vocab = get(ckpt, :vocab, context.vocab)
    entity_label_to_id = get(ckpt, :entity_label_to_id, context.entity_label_to_id)
    relation_label_to_id = get(ckpt, :relation_label_to_id, context.relation_label_to_id)
    params = tree_to_device(ckpt[:params])
    state = tree_to_device(ckpt[:state])

    sweep_modes = isempty(modes) ? [:token, :sentence, :hybrid] : unique(modes)

    println("=" ^ 126)
    println("Evidence Pooling Sweep")
    println("=" ^ 126)
    println("Checkpoint: $(checkpoint_path)")
    println("Config: $(run_config.config_path)")
    println("Val rows: $(length(context.val_rows)) | Max eval batches: $(run_config.max_eval_batches)")
    println()
    println(
        rpad("mode", 10) *
        lpad("rel_p", 10) *
        lpad("rel_r", 10) *
        lpad("rel_f1", 10) *
        lpad("oracle_rel", 10) *
        lpad("pair_r", 10) *
        lpad("pair_t16", 10) *
        lpad("ev_ent", 10) *
        lpad("ev_max", 10) *
        lpad("ev_eff", 10) *
        lpad("ev_t1", 8)
    )
    println("-" ^ 106)

    for mode in sweep_modes
        eval_stats = evaluate_model(
            SwammaRelationExtractor(checkpoint_config),
            params,
            state,
            context.val_rows,
            vocab,
            entity_label_to_id,
            relation_label_to_id,
            checkpoint_config,
            run_config,
            context.pair_proposer_settings;
            evidence_pooling_mode = mode,
            relation_logit_adjustment_tau = relation_logit_adjustment_tau,
            relation_logit_adjustment = relation_logit_adjustment,
        )
        println(format_evidence_pooling_row(mode, eval_stats))
        reclaim_device_memory()
    end
end

function format_pair_sweep_row(pair_budget::Int, overgenerate::Int, ladder)
    pred = ladder.pred_pred
    exhaustive = ladder.pred_exhaustive
    recall_ratio = exhaustive.pair_recall > 0 ? pred.pair_recall / exhaustive.pair_recall : 0.0f0
    return lpad(pair_budget, 8) *
           lpad(overgenerate, 8) *
           @sprintf(
               "%10.4f%10.4f%10.1f%10.4f%10.4f%10.4f%10.4f",
               pred.pair_recall,
               pred.pair_top16_recall,
               pred.matched_pair_rank_mean,
               exhaustive.pair_recall,
               recall_ratio,
               pred.relation_f1,
               pred.oracle_relation_coverage,
           )
end

function run_pair_sweep(
    run_config::RETrainingRunConfig,
    checkpoint_path::String;
    pair_budgets::Vector{Int} = Int[],
    overgenerate_factors::Vector{Int} = Int[],
)
    context = load_eval_context(run_config)
    isempty(context.val_rows) && error("Validation data not found for pair sweep.")
    ckpt = deserialize(checkpoint_path)
    checkpoint_config = get(ckpt, :model_config, context.model_config)
    vocab = get(ckpt, :vocab, context.vocab)
    entity_label_to_id = get(ckpt, :entity_label_to_id, context.entity_label_to_id)
    relation_label_to_id = get(ckpt, :relation_label_to_id, context.relation_label_to_id)
    params = tree_to_device(ckpt[:params])
    state = tree_to_device(ckpt[:state])

    budgets = isempty(pair_budgets) ? [96, 128, 160, 192] : sort(unique(pair_budgets))
    overgens = isempty(overgenerate_factors) ? [2, 4] : sort(unique(overgenerate_factors))

    println("=" ^ 120)
    println("Pair Sweep")
    println("=" ^ 120)
    println("Checkpoint: $(checkpoint_path)")
    println("Config: $(run_config.config_path)")
    println("Val rows: $(length(context.val_rows)) | Max eval batches: $(run_config.max_eval_batches)")
    println()
    println(
        lpad("pairs", 8) *
        lpad("overgen", 8) *
        lpad("pair_r", 10) *
        lpad("pair_t16", 10) *
        lpad("pair_rk", 10) *
        lpad("exh_r", 10) *
        lpad("r/exh", 10) *
        lpad("rel_f1", 10) *
        lpad("oracle_rel", 10)
    )
    println("-" ^ 84)

    for pair_budget in budgets
        for overgenerate in overgens
            eval_config = override_relation_config(
                checkpoint_config;
                max_candidate_pairs = pair_budget,
                pair_overgenerate_factor = overgenerate,
            )
            model = SwammaRelationExtractor(eval_config)
            ladder = evaluate_oracle_ladder(
                model,
                params,
                state,
                context.val_rows,
                vocab,
                entity_label_to_id,
                relation_label_to_id,
                eval_config,
                run_config,
                context.pair_proposer_settings;
                confidence_threshold = 0.5f0,
            )
            println(format_pair_sweep_row(pair_budget, overgenerate, ladder))
            reclaim_device_memory()
        end
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
    Random.seed!(run_config.seed)
    null_relation_weight = load_null_relation_weight(options.config_path)
    relation_focal_gamma = load_relation_focal_gamma(options.config_path)
    positive_relation_weight = load_positive_relation_weight(options.config_path)
    relation_logit_adjustment_tau = load_relation_logit_adjustment_tau(options.config_path)
    distillation_settings = load_distillation_settings(options.config_path)
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
            mention_negative_ratio = run_config.mention_negative_ratio,
            proposal_train_probability = run_config.proposal_train_probability,
            proposal_loss_weight = run_config.proposal_loss_weight,
            proposal_warmup_steps = run_config.proposal_warmup_steps,
            max_len = run_config.max_len,
            seed = run_config.seed,
            max_eval_batches = options.max_eval_batches_override,
            resume_path = run_config.resume_path,
        )
    end

    if options.eval_checkpoint !== nothing
        run_checkpoint_sweep(run_config, [options.eval_checkpoint])
        return
    elseif options.oracle_ladder_checkpoint !== nothing
        run_oracle_ladder(
            run_config,
            options.oracle_ladder_checkpoint;
            max_relations_per_head = options.decode_head_cap,
            max_relations_per_tail = options.decode_tail_cap,
            per_relation_threshold_spec = options.per_relation_thresholds,
            type_constraints_mode = options.type_constraints_mode,
            type_constraints_min_count = options.type_constraints_min_count,
            relation_consistency_mode = options.relation_consistency_mode,
            relation_consistency_min_count = options.relation_consistency_min_count,
        )
        return
    elseif options.mention_sweep_checkpoint !== nothing
        run_mention_sweep(
            run_config,
            options.mention_sweep_checkpoint;
            budgets = options.mention_sweep_budgets,
            modes = options.mention_sweep_modes,
        )
        return
    elseif options.threshold_sweep_checkpoint !== nothing
        run_threshold_sweep(
            run_config,
            options.threshold_sweep_checkpoint;
            thresholds = options.threshold_sweep_values,
            no_relation_margin = options.threshold_sweep_margin,
            nonnull_probability_threshold = options.threshold_sweep_nonnull,
            max_relations_per_head = options.decode_head_cap,
            max_relations_per_tail = options.decode_tail_cap,
            per_relation_threshold_spec = options.per_relation_thresholds,
            type_constraints_mode = options.type_constraints_mode,
            type_constraints_min_count = options.type_constraints_min_count,
            relation_consistency_mode = options.relation_consistency_mode,
            relation_consistency_min_count = options.relation_consistency_min_count,
        )
        return
    elseif options.margin_sweep_checkpoint !== nothing
        run_margin_sweep(
            run_config,
            options.margin_sweep_checkpoint;
            margins = options.margin_sweep_values,
            max_relations_per_head = options.decode_head_cap,
            max_relations_per_tail = options.decode_tail_cap,
            per_relation_threshold_spec = options.per_relation_thresholds,
            type_constraints_mode = options.type_constraints_mode,
            type_constraints_min_count = options.type_constraints_min_count,
            relation_consistency_mode = options.relation_consistency_mode,
            relation_consistency_min_count = options.relation_consistency_min_count,
        )
        return
    elseif options.nonnull_sweep_checkpoint !== nothing
        run_nonnull_sweep(
            run_config,
            options.nonnull_sweep_checkpoint;
            nonnull_values = options.nonnull_sweep_values,
            confidence_threshold = options.nonnull_sweep_confidence,
            no_relation_margin = options.nonnull_sweep_margin,
            max_relations_per_head = options.decode_head_cap,
            max_relations_per_tail = options.decode_tail_cap,
            per_relation_threshold_spec = options.per_relation_thresholds,
            type_constraints_mode = options.type_constraints_mode,
            type_constraints_min_count = options.type_constraints_min_count,
            relation_consistency_mode = options.relation_consistency_mode,
            relation_consistency_min_count = options.relation_consistency_min_count,
        )
        return
    elseif options.auto_calibrate_checkpoint !== nothing
        run_auto_calibration(
            run_config,
            options.auto_calibrate_checkpoint;
            confidence_threshold = options.auto_calibrate_threshold,
            no_relation_margin = options.auto_calibrate_margin,
            nonnull_probability_threshold = options.auto_calibrate_nonnull,
            min_predictions = options.auto_calibrate_min_predictions,
            candidate_thresholds = options.auto_calibrate_thresholds,
            max_relations_per_head = options.decode_head_cap,
            max_relations_per_tail = options.decode_tail_cap,
            per_relation_threshold_spec = options.per_relation_thresholds,
            type_constraints_mode = options.type_constraints_mode,
            type_constraints_min_count = options.type_constraints_min_count,
            relation_consistency_mode = options.relation_consistency_mode,
            relation_consistency_min_count = options.relation_consistency_min_count,
        )
        return
    elseif options.evidence_pooling_sweep_checkpoint !== nothing
        run_evidence_pooling_sweep(
            run_config,
            options.evidence_pooling_sweep_checkpoint;
            modes = options.evidence_pooling_modes,
        )
        return
    elseif options.pair_sweep_checkpoint !== nothing
        run_pair_sweep(
            run_config,
            options.pair_sweep_checkpoint;
            pair_budgets = options.pair_sweep_budgets,
            overgenerate_factors = options.pair_sweep_overgenerate,
        )
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
    pair_proposer_settings = load_pair_proposer_settings(run_config.config_path)
    train_rows = load_rebel_jsonl(run_config.train_path)
    val_rows = isfile(run_config.val_path) ? load_rebel_jsonl(run_config.val_path) : Any[]
    validate_teacher_payload_coverage(train_rows, distillation_settings; context = "train")
    all_rows = isempty(val_rows) ? train_rows : vcat(train_rows, val_rows)

    vocab = build_token_vocab(train_rows; max_vocab = base_config.vocab_size)
    entity_label_to_id = build_entity_label_space(all_rows)
    relation_label_to_id = build_relation_label_space(all_rows)
    model_config = with_label_counts(base_config, length(entity_label_to_id), length(relation_label_to_id), length(vocab))
    relation_logit_adjustment = relation_logit_adjustment_tau > 0.0f0 ?
        build_relation_logit_adjustment(train_rows, relation_label_to_id) :
        nothing

    print_relation_extraction_summary(model_config)
    println("Train rows: $(length(train_rows))")
    println("Val rows:   $(length(val_rows))")
    println("Vocab:      $(length(vocab))")
    edge_ranking_settings = load_edge_ranking_settings(run_config.config_path)
    retrieval_bias_settings = load_retrieval_bias_settings(run_config.config_path)

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
        loaded_params = ckpt[:params]
        loaded_state = ckpt[:state]
        loaded_opt_state = ckpt[:opt_state]
        params, params_mismatch = merge_resume_tree(params, loaded_params)
        state, state_mismatch = merge_resume_tree(state, loaded_state)
        partial_warmstart = params_mismatch || state_mismatch
        opt_state = partial_warmstart ? nothing : loaded_opt_state
        vocab = ckpt[:vocab]
        entity_label_to_id = ckpt[:entity_label_to_id]
        relation_label_to_id = ckpt[:relation_label_to_id]
        start_step = get(ckpt, :step, 0)
        start_epoch = get(ckpt, :epoch, 0) + 1
        ckpt_loss = get(ckpt, :loss, Inf32)
        best_val_loss = ckpt_loss === nothing ? Inf32 : Float32(ckpt_loss)
        if partial_warmstart
            println("Warm-started from $(run_config.resume_path) at step $start_step (partial parameter/state match; optimizer reset)")
        else
            println("Resumed from $(run_config.resume_path) at step $start_step")
        end
    end

    relation_logit_adjustment = relation_logit_adjustment_tau > 0.0f0 ?
        build_relation_logit_adjustment(train_rows, relation_label_to_id) :
        nothing

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
    println("Edge rank wt: $(edge_ranking_settings.weight)")
    println("Edge rank margin: $(edge_ranking_settings.margin)")
    println("Edge rank hard negs: $(edge_ranking_settings.hard_negatives)")
    println("Edge rank start step: $(edge_ranking_settings.start_step)")
    println("Edge rank warmup: $(edge_ranking_settings.warmup_steps)")
    println("Proposal train prob: $(run_config.proposal_train_probability)")
    println("Proposal loss wt: $(run_config.proposal_loss_weight)")
    println("Proposal warmup: $(run_config.proposal_warmup_steps)")
    println("Null relation wt: $(null_relation_weight)")
    println("Relation focal gamma: $(relation_focal_gamma)")
    println("Positive relation wt: $(positive_relation_weight)")
    println("Relation logit-adjust tau: $(relation_logit_adjustment_tau)")
    println("Teacher entity loss wt: $(distillation_settings.entity_weight)")
    println("Teacher relation loss wt: $(distillation_settings.relation_weight)")
    println("Teacher confidence loss wt: $(distillation_settings.confidence_weight)")
    println("Allow missing teacher targets: $(distillation_settings.allow_missing_teacher_targets)")
    println("Total step updates: $(run_config.total_steps)")
    println()

    step = start_step
    epoch = start_epoch
    recent_losses = Float32[]

    println("[train] Entering training loop (step=$step, total=$(run_config.total_steps))")
    flush(stdout)

    while step < run_config.total_steps
        println("[train] Epoch $epoch: shuffling $(length(train_rows)) rows...")
        flush(stdout)
        shuffled = Random.shuffle(rng, copy(train_rows))
        batch_starts = collect(1:run_config.batch_size:length(shuffled))
        grad_accum = nothing
        accum_count = 0

        for batch_start in batch_starts
            batch_end = min(batch_start + run_config.batch_size - 1, length(shuffled))
            batch_rows = shuffled[batch_start:batch_end]
            if step < 3
                println("[train] step=$(step+1): make_batch batch_start=$batch_start...")
                flush(stdout)
            end
            inputs, targets = make_batch(
                batch_rows,
                vocab,
                entity_label_to_id,
                relation_label_to_id,
                model_config,
                run_config;
                rng = rng,
            )
            next_step = step + 1
            inputs = with_retrieval_bias_inputs(inputs, retrieval_bias_settings; step = next_step)
            use_proposal_training = (
                run_config.proposal_loss_weight > 0.0f0 &&
                run_config.proposal_train_probability > 0.0f0 &&
                next_step > run_config.proposal_warmup_steps &&
                rand(rng) < run_config.proposal_train_probability
            )
            no_relation_id = get(relation_label_to_id, "NO_RELATION", 1)
            edge_ranking_weight = edge_ranking_weight_for_step(edge_ranking_settings, next_step)

            # Force GPU memory reclaim before each gradient pass
            GC.gc(true)
            CUDA.reclaim()

            t0 = time_ns()
            if step < 5
                free_mb = round(CUDA.available_memory() / 1e6, digits=0)
                println("[train] step=$(step+1): gradient pass... (GPU free: $(free_mb) MB)")
                flush(stdout)
            end
            (loss, new_state), grads = Zygote.withgradient(params) do p
                outputs, teacher_state = model(inputs, p, state)
                total_loss = relation_loss(
                    outputs,
                    targets;
                    null_relation_weight = null_relation_weight,
                    positive_relation_weight = positive_relation_weight,
                    no_relation_id = no_relation_id,
                    relation_focal_gamma = relation_focal_gamma,
                    relation_logit_adjustment_tau = relation_logit_adjustment_tau,
                    relation_logit_adjustment = relation_logit_adjustment,
                    teacher_entity_loss_weight = distillation_settings.entity_weight,
                    teacher_relation_loss_weight = distillation_settings.relation_weight,
                    teacher_confidence_loss_weight = distillation_settings.confidence_weight,
                    edge_ranking_loss_weight = edge_ranking_weight,
                    edge_ranking_margin = edge_ranking_settings.margin,
                    edge_ranking_hard_negatives = edge_ranking_settings.hard_negatives,
                )
                final_state = teacher_state

                if use_proposal_training
                    proposal_seed_inputs = build_proposal_inputs(outputs, inputs, model_config, pair_proposer_settings)
                    proposal_seed_outputs = ChainRulesCore.ignore_derivatives() do
                        first(model(proposal_seed_inputs, p, teacher_state))
                    end
                    proposal_inputs = ChainRulesCore.ignore_derivatives() do
                        build_fixed_proposal_inputs(proposal_seed_outputs, inputs)
                    end
                    proposal_outputs, proposal_state = model(proposal_inputs, p, teacher_state)
                    proposal_targets = ChainRulesCore.ignore_derivatives() do
                        build_proposal_relation_targets(proposal_outputs, targets, no_relation_id)
                    end
                    proposal_losses = proposal_training_loss(
                        proposal_outputs,
                        proposal_targets;
                        null_relation_weight = null_relation_weight,
                        positive_relation_weight = positive_relation_weight,
                        no_relation_id = no_relation_id,
                        relation_focal_gamma = relation_focal_gamma,
                        relation_logit_adjustment_tau = relation_logit_adjustment_tau,
                        relation_logit_adjustment = relation_logit_adjustment,
                        edge_ranking_loss_weight = edge_ranking_weight,
                        edge_ranking_margin = edge_ranking_settings.margin,
                        edge_ranking_hard_negatives = edge_ranking_settings.hard_negatives,
                    )
                    total_loss += run_config.proposal_loss_weight * proposal_losses.total
                    final_state = proposal_state
                end

                total_loss, final_state
            end
            CUDA.synchronize()

            grad_accum = tree_add(grad_accum, grads[1])
            grads = nothing  # Free AD tape
            accum_count += 1
            state = new_state
            new_state = nothing
            push!(recent_losses, Float32(loss))

            if accum_count == run_config.gradient_accumulation_steps || batch_end == length(shuffled)
                step += 1
                mean_grads = tree_scale(grad_accum, 1.0f0 / Float32(accum_count))
                opt_state, params = Optimisers.update(opt_state, params, mean_grads)
                grad_accum = nothing
                mean_grads = nothing
                accum_count = 0
                GC.gc(false)
                CUDA.reclaim()

                if step % run_config.log_every == 0 || step == 1
                    dt_ms = (time_ns() - t0) / 1e6
                    avg_loss = recent_mean(recent_losses, run_config.log_every)
                    effective_batch = run_config.batch_size * run_config.gradient_accumulation_steps
                    tokens_per_sec = (run_config.max_len * effective_batch) / max(dt_ms / 1e3, 1f-6)
                    @printf(
                        "step %6d | epoch %3d | loss %.4f | edge_w %.3f | %.1f ms/update | %.0f tok/s\n",
                        step, epoch, avg_loss, edge_ranking_weight, dt_ms, tokens_per_sec
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
                        pair_proposer_settings,
                        current_step = step,
                        relation_logit_adjustment_tau = relation_logit_adjustment_tau,
                        relation_logit_adjustment = relation_logit_adjustment,
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

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
