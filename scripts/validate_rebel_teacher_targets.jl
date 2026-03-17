#!/usr/bin/env julia
"""
Validate optional teacher targets embedded in REBEL-style JSONL rows.

Usage:
    julia --project=. scripts/validate_rebel_teacher_targets.jl --data data/rebel/train.jsonl
    julia --project=. scripts/validate_rebel_teacher_targets.jl --config configs/redfm_base.toml --split train
    julia --project=. scripts/validate_rebel_teacher_targets.jl --data data/rebel/train.jsonl --require-teacher --strict
"""

using Printf
using TOML
using JSON3

Base.@kwdef mutable struct ValidatorOptions
    data_path::Union{Nothing,String} = nothing
    config_path::Union{Nothing,String} = nothing
    split::String = "train"
    max_rows::Int = 0
    require_teacher::Bool = false
    strict::Bool = false
end

Base.@kwdef mutable struct ValidationStats
    rows_scanned::Int = 0
    rows_with_teacher_entities::Int = 0
    rows_with_teacher_relations::Int = 0
    teacher_entity_total::Int = 0
    teacher_relation_total::Int = 0
    teacher_entity_invalid_spans::Int = 0
    teacher_entity_unknown_labels::Int = 0
    teacher_entity_non_gold_signatures::Int = 0
    teacher_entity_exact_gold_matches::Int = 0
    teacher_entity_exact_alignment_rows::Int = 0
    teacher_entity_count_mismatch_rows::Int = 0
    teacher_relation_invalid_indices::Int = 0
    teacher_relation_unknown_labels::Int = 0
    teacher_relation_missing_confidence::Int = 0
    teacher_relation_exact_gold_matches::Int = 0
    teacher_relation_non_gold_pairs::Int = 0
    teacher_relation_non_gold_labeled_edges::Int = 0
    teacher_confidence_total::Float64 = 0.0
    teacher_confidence_count::Int = 0
    teacher_confidence_min::Float64 = Inf
    teacher_confidence_max::Float64 = -Inf
end

function usage()
    println("Usage:")
    println("  julia --project=. scripts/validate_rebel_teacher_targets.jl --data <path>")
    println("  julia --project=. scripts/validate_rebel_teacher_targets.jl --config <path> [--split train|val|test]")
    println("Options:")
    println("  --max-rows <n>        Limit scan to the first n rows (0 = all rows)")
    println("  --require-teacher     Exit non-zero if no teacher payloads are present")
    println("  --strict              Exit non-zero if invalid spans/indices/labels are found")
    println("  -h, --help            Show this message")
end

function parse_args(args)::ValidatorOptions
    opts = ValidatorOptions()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--data"
            i += 1
            i <= length(args) || error("--data requires a value")
            opts.data_path = args[i]
        elseif arg == "--config"
            i += 1
            i <= length(args) || error("--config requires a value")
            opts.config_path = args[i]
        elseif arg == "--split"
            i += 1
            i <= length(args) || error("--split requires a value")
            opts.split = lowercase(args[i])
        elseif arg == "--max-rows"
            i += 1
            i <= length(args) || error("--max-rows requires a value")
            opts.max_rows = parse(Int, args[i])
        elseif arg == "--require-teacher"
            opts.require_teacher = true
        elseif arg == "--strict"
            opts.strict = true
        elseif arg == "-h" || arg == "--help"
            usage()
            exit(0)
        else
            error("unknown argument: $arg")
        end
        i += 1
    end
    return opts
end

field(obj, key::Symbol, default = nothing) = haskey(obj, key) ? obj[key] : default

function resolve_data_path(opts::ValidatorOptions)::String
    if opts.data_path !== nothing
        return opts.data_path
    end
    opts.config_path === nothing && error("provide either --data or --config")
    config = TOML.parsefile(opts.config_path)
    haskey(config, "data") || error("config $(opts.config_path) is missing [data]")
    data_cfg = config["data"]
    key = if opts.split == "train"
        "train_path"
    elseif opts.split == "val" || opts.split == "validation"
        "val_path"
    elseif opts.split == "test"
        "test_path"
    else
        error("unsupported split: $(opts.split)")
    end
    haskey(data_cfg, key) || error("config $(opts.config_path) is missing data.$key")
    return String(data_cfg[key])
end

function entity_span_end(entity)::Int
    if haskey(entity, :stop)
        return Int(entity[:stop])
    elseif haskey(entity, :end)
        return Int(entity[:end])
    elseif haskey(entity, :end_)
        return Int(entity[:end_])
    else
        return Int(field(entity, :start, 0))
    end
end

function normalize_entity_signature(entity, token_count::Int)
    token_count > 0 || return nothing
    start_raw = Int(field(entity, :start, 0))
    stop_raw = entity_span_end(entity)
    offset = (start_raw == 0 || stop_raw == 0) ? 1 : 0
    start_idx = start_raw + offset
    stop_idx = stop_raw + offset
    if !(1 <= start_idx <= token_count && start_idx <= stop_idx <= token_count)
        return nothing
    end
    label = uppercase(String(field(entity, :label, "")))
    isempty(label) && return nothing
    return (start_idx, stop_idx, label)
end

function relation_endpoint_span(rel, prefix::Symbol, token_count::Int)
    token_count > 0 || return nothing
    start_key = Symbol(String(prefix) * "_start")
    stop_key = Symbol(String(prefix) * "_stop")
    end_key = Symbol(String(prefix) * "_end")
    span_key = Symbol(String(prefix) * "_span")

    if haskey(rel, start_key)
        start_raw = Int(rel[start_key])
        stop_raw = if haskey(rel, stop_key)
            Int(rel[stop_key])
        elseif haskey(rel, end_key)
            Int(rel[end_key])
        else
            start_raw
        end
        offset = (start_raw == 0 || stop_raw == 0) ? 1 : 0
        return (start_raw + offset, stop_raw + offset)
    elseif haskey(rel, span_key)
        span = rel[span_key]
        start_raw = Int(field(span, :start, 0))
        stop_raw = if haskey(span, :stop)
            Int(span[:stop])
        elseif haskey(span, :end)
            Int(span[:end])
        elseif haskey(span, :end_)
            Int(span[:end_])
        else
            start_raw
        end
        offset = (start_raw == 0 || stop_raw == 0) ? 1 : 0
        return (start_raw + offset, stop_raw + offset)
    end

    return nothing
end

function normalize_relation_signature(rel, entity_count::Int, token_count::Int, span_to_entity_index::Dict{Tuple{Int,Int},Int})
    entity_count > 0 || return nothing
    head_span = relation_endpoint_span(rel, :head, token_count)
    tail_span = relation_endpoint_span(rel, :tail, token_count)
    if head_span !== nothing || tail_span !== nothing
        head_span === nothing && return nothing
        tail_span === nothing && return nothing
        head_idx = get(span_to_entity_index, head_span, 0)
        tail_idx = get(span_to_entity_index, tail_span, 0)
    else
        head_raw = Int(field(rel, :head, 0))
        tail_raw = Int(field(rel, :tail, 0))
        offset = (head_raw == 0 || tail_raw == 0) ? 1 : 0
        head_idx = head_raw + offset
        tail_idx = tail_raw + offset
    end
    if !(1 <= head_idx <= entity_count && 1 <= tail_idx <= entity_count) || head_idx == tail_idx
        return nothing
    end
    label = String(field(rel, :label, ""))
    isempty(label) && return nothing
    return (head_idx, tail_idx, label)
end

function row_tokens(row)::Vector{String}
    if haskey(row, :tokens)
        return [String(tok) for tok in row.tokens]
    end
    text = String(field(row, :text, ""))
    return isempty(text) ? String[] : split(text)
end

function load_rows(path::String; max_rows::Int = 0)
    rows = Any[]
    open(path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            push!(rows, JSON3.read(line))
            max_rows > 0 && length(rows) >= max_rows && break
        end
    end
    return rows
end

function collect_gold_label_sets(rows)
    entity_labels = Set{String}()
    relation_labels = Set(["NO_RELATION"])
    for row in rows
        if haskey(row, :entities)
            for entity in row.entities
                label = uppercase(String(field(entity, :label, "")))
                isempty(label) || push!(entity_labels, label)
            end
        end
        if haskey(row, :relations)
            for rel in row.relations
                label = String(field(rel, :label, ""))
                isempty(label) || push!(relation_labels, label)
            end
        end
    end
    return entity_labels, relation_labels
end

function update_confidence!(stats::ValidationStats, rel)
    value = if haskey(rel, :confidence)
        Float64(rel[:confidence])
    elseif haskey(rel, :score)
        Float64(rel[:score])
    else
        stats.teacher_relation_missing_confidence += 1
        return
    end
    stats.teacher_confidence_total += value
    stats.teacher_confidence_count += 1
    stats.teacher_confidence_min = min(stats.teacher_confidence_min, value)
    stats.teacher_confidence_max = max(stats.teacher_confidence_max, value)
end

function validate_rows(rows, gold_entity_labels::Set{String}, gold_relation_labels::Set{String})
    stats = ValidationStats()
    for row in rows
        stats.rows_scanned += 1
        tokens = row_tokens(row)
        token_count = length(tokens)

        gold_entities = haskey(row, :entities) ? collect(row.entities) : Any[]
        gold_entity_signatures = Tuple{Int,Int,String}[]
        gold_span_to_entity_index = Dict{Tuple{Int,Int},Int}()
        for entity in gold_entities
            sig = normalize_entity_signature(entity, token_count)
            if sig !== nothing
                push!(gold_entity_signatures, sig)
                gold_span_to_entity_index[(sig[1], sig[2])] = length(gold_entity_signatures)
            end
        end
        gold_entity_set = Set(gold_entity_signatures)
        gold_entity_count = length(gold_entities)

        gold_relations = haskey(row, :relations) ? collect(row.relations) : Any[]
        gold_relation_set = Set{Tuple{Int,Int,String}}()
        gold_relation_pair_set = Set{Tuple{Int,Int}}()
        for rel in gold_relations
            sig = normalize_relation_signature(rel, gold_entity_count, token_count, gold_span_to_entity_index)
            sig === nothing && continue
            push!(gold_relation_set, sig)
            push!(gold_relation_pair_set, (sig[1], sig[2]))
        end

        teacher_entity_signatures = Tuple{Int,Int,String}[]
        teacher_span_to_entity_index = Dict{Tuple{Int,Int},Int}()
        if haskey(row, :teacher_entities)
            teacher_entities = collect(row.teacher_entities)
            stats.rows_with_teacher_entities += 1
            stats.teacher_entity_total += length(teacher_entities)

            for entity in teacher_entities
                sig = normalize_entity_signature(entity, token_count)
                if sig === nothing
                    stats.teacher_entity_invalid_spans += 1
                    continue
                end
                push!(teacher_entity_signatures, sig)
                teacher_span_to_entity_index[(sig[1], sig[2])] = length(teacher_entity_signatures)
                if !(sig[3] in gold_entity_labels)
                    stats.teacher_entity_unknown_labels += 1
                end
                if sig in gold_entity_set
                    stats.teacher_entity_exact_gold_matches += 1
                else
                    stats.teacher_entity_non_gold_signatures += 1
                end
            end

            if length(teacher_entities) != gold_entity_count
                stats.teacher_entity_count_mismatch_rows += 1
            end
            if teacher_entity_signatures == gold_entity_signatures
                stats.teacher_entity_exact_alignment_rows += 1
            end
        end

        if haskey(row, :teacher_relations)
            teacher_relations = collect(row.teacher_relations)
            stats.rows_with_teacher_relations += 1
            stats.teacher_relation_total += length(teacher_relations)
            relation_entity_count = isempty(teacher_entity_signatures) ? gold_entity_count : length(teacher_entity_signatures)
            relation_span_to_entity_index = isempty(teacher_span_to_entity_index) ? gold_span_to_entity_index : teacher_span_to_entity_index

            for rel in teacher_relations
                sig = normalize_relation_signature(rel, relation_entity_count, token_count, relation_span_to_entity_index)
                update_confidence!(stats, rel)
                if sig === nothing
                    stats.teacher_relation_invalid_indices += 1
                    continue
                end
                if !(sig[3] in gold_relation_labels)
                    stats.teacher_relation_unknown_labels += 1
                end
                if sig in gold_relation_set
                    stats.teacher_relation_exact_gold_matches += 1
                else
                    stats.teacher_relation_non_gold_labeled_edges += 1
                end
                if !((sig[1], sig[2]) in gold_relation_pair_set)
                    stats.teacher_relation_non_gold_pairs += 1
                end
            end
        end
    end
    return stats
end

function print_summary(path::String, stats::ValidationStats)
    println("============================================================")
    println("Teacher Target Validation")
    println("============================================================")
    println("Data: $path")
    println("Rows scanned: $(stats.rows_scanned)")
    println()
    println("Coverage")
    @printf("  rows with teacher_entities:   %d\n", stats.rows_with_teacher_entities)
    @printf("  rows with teacher_relations:  %d\n", stats.rows_with_teacher_relations)
    @printf("  teacher entities total:       %d\n", stats.teacher_entity_total)
    @printf("  teacher relations total:      %d\n", stats.teacher_relation_total)
    println()
    println("Teacher entity checks")
    @printf("  invalid spans:                %d\n", stats.teacher_entity_invalid_spans)
    @printf("  unknown labels:               %d\n", stats.teacher_entity_unknown_labels)
    @printf("  exact gold signature matches: %d\n", stats.teacher_entity_exact_gold_matches)
    @printf("  non-gold signatures:          %d\n", stats.teacher_entity_non_gold_signatures)
    @printf("  exact alignment rows:         %d\n", stats.teacher_entity_exact_alignment_rows)
    @printf("  count mismatch rows:          %d\n", stats.teacher_entity_count_mismatch_rows)
    println()
    println("Teacher relation checks")
    @printf("  invalid indices:              %d\n", stats.teacher_relation_invalid_indices)
    @printf("  unknown labels:               %d\n", stats.teacher_relation_unknown_labels)
    @printf("  exact gold labeled matches:   %d\n", stats.teacher_relation_exact_gold_matches)
    @printf("  non-gold labeled edges:       %d\n", stats.teacher_relation_non_gold_labeled_edges)
    @printf("  non-gold positive pairs:      %d\n", stats.teacher_relation_non_gold_pairs)
    @printf("  missing confidence/score:     %d\n", stats.teacher_relation_missing_confidence)
    if stats.teacher_confidence_count > 0
        mean_conf = stats.teacher_confidence_total / stats.teacher_confidence_count
        @printf("  confidence mean/min/max:      %.4f / %.4f / %.4f\n", mean_conf, stats.teacher_confidence_min, stats.teacher_confidence_max)
    else
        println("  confidence mean/min/max:      n/a")
    end
end

function has_validation_failures(stats::ValidationStats)::Bool
    return stats.teacher_entity_invalid_spans > 0 ||
           stats.teacher_entity_unknown_labels > 0 ||
           stats.teacher_relation_invalid_indices > 0 ||
           stats.teacher_relation_unknown_labels > 0
end

function main()
    opts = parse_args(ARGS)
    path = resolve_data_path(opts)
    isfile(path) || error("data file not found: $path")

    rows = load_rows(path; max_rows = opts.max_rows)
    gold_entity_labels, gold_relation_labels = collect_gold_label_sets(rows)
    stats = validate_rows(rows, gold_entity_labels, gold_relation_labels)
    print_summary(path, stats)

    teacher_present = stats.rows_with_teacher_entities > 0 || stats.rows_with_teacher_relations > 0
    if !teacher_present
        println()
        println("Warning: no teacher payloads were found in the scanned rows.")
    end
    if stats.teacher_relation_non_gold_pairs > 0
        println()
        println("Warning: some teacher relations use positive pairs that are not gold-positive pairs.")
        println("Those edges now rely on teacher-only candidate injection and are supervised only through distillation losses.")
    end

    if opts.require_teacher && !teacher_present
        exit(1)
    end
    if opts.strict && has_validation_failures(stats)
        exit(1)
    end
end

main()
