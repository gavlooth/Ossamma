#!/usr/bin/env julia

using JSON3

include(joinpath(@__DIR__, "..", "src", "Swamma.jl"))
using .Swamma.RelationExtraction: load_rebel_jsonl

Base.@kwdef mutable struct EvalOptions
    gold_path::Union{Nothing,String} = nothing
    teacher_path::Union{Nothing,String} = nothing
    max_rows::Int = 0
end

function usage()
    println("Usage:")
    println("  julia --project=. scripts/evaluate_rebel_teacher_pilot.jl --gold <rebel.jsonl> --teacher <parsed_teacher.jsonl>")
    println("Options:")
    println("  --max-rows <n>         Limit gold rows considered (0 = all)")
    println("  -h, --help")
end

function parse_args(args)
    opts = EvalOptions()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--gold"
            i += 1
            opts.gold_path = args[i]
        elseif arg == "--teacher"
            i += 1
            opts.teacher_path = args[i]
        elseif arg == "--max-rows"
            i += 1
            opts.max_rows = parse(Int, args[i])
        elseif arg == "-h" || arg == "--help"
            usage()
            exit(0)
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end
    opts.gold_path === nothing && error("--gold is required")
    opts.teacher_path === nothing && error("--teacher is required")
    return opts
end

function row_match_key(row, row_index::Int)
    if haskey(row, :docid)
        return "docid:" * String(row.docid)
    elseif haskey(row, :uri)
        return "uri:" * String(row.uri)
    elseif haskey(row, :title)
        return "title:" * String(row.title)
    end
    return "row_index:" * string(row_index)
end

function read_jsonl(path::String)
    rows = Vector{Any}()
    open(path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            push!(rows, JSON3.read(line))
        end
    end
    return rows
end

function relation_labels(row)
    labels = String[]
    haskey(row, :relations) || return labels
    for rel in row.relations
        push!(labels, String(rel.label))
    end
    return labels
end

function relation_labels_any(row)
    labels = String[]
    if haskey(row, "relations")
        for rel in row["relations"]
            haskey(rel, "label") || continue
            push!(labels, String(rel["label"]))
        end
    end
    return labels
end

function main()
    opts = parse_args(ARGS)
    isfile(opts.gold_path) || error("gold JSONL not found: $(opts.gold_path)")
    isfile(opts.teacher_path) || error("teacher JSONL not found: $(opts.teacher_path)")

    gold_rows = load_rebel_jsonl(opts.gold_path)
    limit = opts.max_rows > 0 ? min(opts.max_rows, length(gold_rows)) : length(gold_rows)
    gold_rows = gold_rows[1:limit]
    teacher_rows = read_jsonl(opts.teacher_path)

    teacher_by_key = Dict{String,Any}()
    for row in teacher_rows
        haskey(row, "match_key") || continue
        teacher_by_key[String(row["match_key"])] = row
    end

    total_rows = length(gold_rows)
    matched_rows = 0
    non_empty_rows = 0
    top1_in_gold = 0
    exact_label_set_match = 0
    total_pred_relations = 0
    pred_label_counts = Dict{String,Int}()
    gold_label_counts = Dict{String,Int}()

    for (idx, row) in enumerate(gold_rows)
        gold_key = row_match_key(row, idx)
        gold_labels = relation_labels(row)
        gold_label_set = Set(gold_labels)
        for label in gold_labels
            gold_label_counts[label] = get(gold_label_counts, label, 0) + 1
        end
        teacher_row = get(teacher_by_key, gold_key, nothing)
        teacher_row === nothing && continue
        matched_rows += 1
        pred_labels = relation_labels_any(teacher_row)
        total_pred_relations += length(pred_labels)
        for label in pred_labels
            pred_label_counts[label] = get(pred_label_counts, label, 0) + 1
        end
        isempty(pred_labels) && continue
        non_empty_rows += 1
        pred_labels[1] in gold_label_set && (top1_in_gold += 1)
        Set(pred_labels) == gold_label_set && (exact_label_set_match += 1)
    end

    println("============================================================")
    println("Teacher Pilot Evaluation")
    println("============================================================")
    println("Gold rows:              $total_rows")
    println("Teacher matched rows:   $matched_rows")
    println("Teacher non-empty rows: $non_empty_rows")
    println("Predicted relations:    $total_pred_relations")
    if matched_rows > 0
        println("Non-empty rate:         $(round(non_empty_rows / matched_rows; digits = 4))")
    end
    if non_empty_rows > 0
        println("Top-1 label in gold:    $(round(top1_in_gold / non_empty_rows; digits = 4))")
        println("Exact label-set match:  $(round(exact_label_set_match / non_empty_rows; digits = 4))")
    end

    println("Pred label counts:      " * JSON3.write(pred_label_counts))
    println("Gold label counts:      " * JSON3.write(gold_label_counts))
end

main()
