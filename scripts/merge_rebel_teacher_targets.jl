#!/usr/bin/env julia
"""
Merge teacher annotations into REBEL-style JSONL rows.

Usage:
    julia --project=. scripts/merge_rebel_teacher_targets.jl \
        --base data/rebel/train.jsonl \
        --teacher teacher/train.jsonl \
        --output data/rebel/train_teacher.jsonl
"""

using JSON3
using Printf

Base.@kwdef mutable struct MergeOptions
    base_path::Union{Nothing,String} = nothing
    teacher_path::Union{Nothing,String} = nothing
    output_path::Union{Nothing,String} = nothing
    match_key::String = "auto"
    teacher_entity_field::String = "entities"
    teacher_relation_field::String = "relations"
    relation_format::String = "spans"
    strict::Bool = false
end

function usage()
    println("Usage:")
    println("  julia --project=. scripts/merge_rebel_teacher_targets.jl --base <base.jsonl> --teacher <teacher.jsonl> --output <out.jsonl>")
    println("Options:")
    println("  --match-key <auto|docid|uri|title|row_index>")
    println("  --teacher-entity-field <entities|teacher_entities>")
    println("  --teacher-relation-field <relations|teacher_relations>")
    println("  --relation-format <spans|indices>")
    println("  --strict")
    println("  -h, --help")
end

function parse_args(args)::MergeOptions
    opts = MergeOptions()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--base"
            i += 1
            opts.base_path = args[i]
        elseif arg == "--teacher"
            i += 1
            opts.teacher_path = args[i]
        elseif arg == "--output"
            i += 1
            opts.output_path = args[i]
        elseif arg == "--match-key"
            i += 1
            opts.match_key = lowercase(args[i])
        elseif arg == "--teacher-entity-field"
            i += 1
            opts.teacher_entity_field = args[i]
        elseif arg == "--teacher-relation-field"
            i += 1
            opts.teacher_relation_field = args[i]
        elseif arg == "--relation-format"
            i += 1
            opts.relation_format = lowercase(args[i])
        elseif arg == "--strict"
            opts.strict = true
        elseif arg == "-h" || arg == "--help"
            usage()
            exit(0)
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end

    opts.base_path === nothing && error("--base is required")
    opts.teacher_path === nothing && error("--teacher is required")
    opts.output_path === nothing && error("--output is required")
    opts.match_key in ("auto", "docid", "uri", "title", "row_index") || error("unsupported --match-key=$(opts.match_key)")
    opts.relation_format in ("spans", "indices") || error("unsupported --relation-format=$(opts.relation_format)")
    return opts
end

function json_to_native(x)
    if x isa JSON3.Object
        out = Dict{String,Any}()
        for (k, v) in pairs(x)
            out[String(k)] = json_to_native(v)
        end
        return out
    elseif x isa AbstractVector
        return Any[json_to_native(v) for v in x]
    else
        return x
    end
end

function read_jsonl(path::String)
    rows = Vector{Dict{String,Any}}()
    open(path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            push!(rows, json_to_native(JSON3.read(line)))
        end
    end
    return rows
end

field(obj::Dict{String,Any}, key::String, default = nothing) = get(obj, key, default)

function entity_stop(entity::Dict{String,Any})
    if haskey(entity, "stop")
        return Int(entity["stop"])
    elseif haskey(entity, "end")
        return Int(entity["end"])
    elseif haskey(entity, "end_")
        return Int(entity["end_"])
    else
        return Int(field(entity, "start", 0))
    end
end

function normalize_entity(entity::Dict{String,Any})
    out = Dict{String,Any}()
    out["start"] = Int(field(entity, "start", 0))
    out["stop"] = entity_stop(entity)
    out["label"] = String(field(entity, "label", ""))
    for extra_key in ("uri", "surfaceform", "text", "confidence", "score")
        haskey(entity, extra_key) && (out[extra_key] = entity[extra_key])
    end
    return out
end

function relation_endpoint_from_span_fields(rel::Dict{String,Any}, prefix::String)
    start_key = prefix * "_start"
    stop_key = prefix * "_stop"
    end_key = prefix * "_end"
    span_key = prefix * "_span"

    if haskey(rel, start_key)
        start_raw = Int(rel[start_key])
        stop_raw = if haskey(rel, stop_key)
            Int(rel[stop_key])
        elseif haskey(rel, end_key)
            Int(rel[end_key])
        else
            start_raw
        end
        return (start_raw, stop_raw)
    elseif haskey(rel, span_key)
        span = rel[span_key]
        span isa Dict{String,Any} || return nothing
        start_raw = Int(field(span, "start", 0))
        stop_raw = if haskey(span, "stop")
            Int(span["stop"])
        elseif haskey(span, "end")
            Int(span["end"])
        elseif haskey(span, "end_")
            Int(span["end_"])
        else
            start_raw
        end
        return (start_raw, stop_raw)
    end

    return nothing
end

function relation_endpoints_from_indices(rel::Dict{String,Any}, entities::Vector{Any})
    haskey(rel, "head") || return nothing
    haskey(rel, "tail") || return nothing
    head_raw = Int(rel["head"])
    tail_raw = Int(rel["tail"])
    offset = (head_raw == 0 || tail_raw == 0) ? 1 : 0
    head_idx = head_raw + offset
    tail_idx = tail_raw + offset
    if !(1 <= head_idx <= length(entities) && 1 <= tail_idx <= length(entities))
        return nothing
    end
    head_entity = entities[head_idx]::Dict{String,Any}
    tail_entity = entities[tail_idx]::Dict{String,Any}
    return (
        Int(field(head_entity, "start", 0)),
        entity_stop(head_entity),
        Int(field(tail_entity, "start", 0)),
        entity_stop(tail_entity),
    )
end

function normalize_relation(rel::Dict{String,Any}, teacher_entities::Vector{Any}, relation_format::String)
    label = String(field(rel, "label", ""))
    isempty(label) && return nothing
    confidence = haskey(rel, "confidence") ? rel["confidence"] : haskey(rel, "score") ? rel["score"] : nothing

    if relation_format == "indices"
        out = Dict{String,Any}()
        out["label"] = label
        if haskey(rel, "head") && haskey(rel, "tail")
            out["head"] = Int(rel["head"])
            out["tail"] = Int(rel["tail"])
        else
            endpoints = relation_endpoint_from_span_fields(rel, "head"), relation_endpoint_from_span_fields(rel, "tail")
            endpoints[1] === nothing && return nothing
            endpoints[2] === nothing && return nothing
            return nothing
        end
        confidence === nothing || (out["confidence"] = confidence)
        return out
    end

    endpoints = if relation_endpoint_from_span_fields(rel, "head") !== nothing ||
                   relation_endpoint_from_span_fields(rel, "tail") !== nothing
        head = relation_endpoint_from_span_fields(rel, "head")
        tail = relation_endpoint_from_span_fields(rel, "tail")
        head === nothing || tail === nothing ? nothing : (head[1], head[2], tail[1], tail[2])
    else
        relation_endpoints_from_indices(rel, teacher_entities)
    end
    endpoints === nothing && return nothing

    out = Dict{String,Any}()
    out["head_start"] = endpoints[1]
    out["head_stop"] = endpoints[2]
    out["tail_start"] = endpoints[3]
    out["tail_stop"] = endpoints[4]
    out["label"] = label
    confidence === nothing || (out["confidence"] = confidence)
    return out
end

function row_key(row::Dict{String,Any}, row_index::Int, match_key::String)
    if match_key == "row_index"
        return "row_index:" * string(row_index)
    elseif match_key == "docid"
        return haskey(row, "docid") ? "docid:" * string(row["docid"]) : nothing
    elseif match_key == "uri"
        return haskey(row, "uri") ? "uri:" * string(row["uri"]) : nothing
    elseif match_key == "title"
        return haskey(row, "title") ? "title:" * string(row["title"]) : nothing
    end

    for key in ("docid", "uri", "title")
        if haskey(row, key)
            return key * ":" * string(row[key])
        end
    end
    return "row_index:" * string(row_index)
end

function main()
    opts = parse_args(ARGS)
    isfile(opts.base_path) || error("base JSONL not found: $(opts.base_path)")
    isfile(opts.teacher_path) || error("teacher JSONL not found: $(opts.teacher_path)")

    base_rows = read_jsonl(opts.base_path)
    teacher_rows = read_jsonl(opts.teacher_path)

    teacher_lookup = Dict{String,Dict{String,Any}}()
    duplicate_keys = String[]
    for (idx, row) in enumerate(teacher_rows)
        key = row_key(row, idx, opts.match_key)
        key === nothing && continue
        if haskey(teacher_lookup, key)
            push!(duplicate_keys, key)
        else
            teacher_lookup[key] = row
        end
    end
    if !isempty(duplicate_keys) && opts.strict
        duplicate_spec = join(unique(duplicate_keys), ", ")
        error("duplicate teacher keys found: $(duplicate_spec)")
    end

    matched_rows = 0
    missing_rows = 0
    teacher_entity_total = 0
    teacher_relation_total = 0
    skipped_relations = 0

    open(opts.output_path, "w") do io
        for (idx, row) in enumerate(base_rows)
            key = row_key(row, idx, opts.match_key)
            teacher_row = key === nothing ? nothing : get(teacher_lookup, key, nothing)
            if teacher_row === nothing
                missing_rows += 1
                println(io, JSON3.write(row))
                continue
            end

            matched_rows += 1
            teacher_entities_raw = haskey(teacher_row, opts.teacher_entity_field) ? teacher_row[opts.teacher_entity_field] : Any[]
            teacher_entities = Any[normalize_entity(entity) for entity in teacher_entities_raw]
            teacher_relations_raw = haskey(teacher_row, opts.teacher_relation_field) ? teacher_row[opts.teacher_relation_field] : Any[]
            teacher_relations = Any[]
            for rel in teacher_relations_raw
                relation = normalize_relation(rel, teacher_entities, opts.relation_format)
                if relation === nothing
                    skipped_relations += 1
                    continue
                end
                push!(teacher_relations, relation)
            end

            row["teacher_entities"] = teacher_entities
            row["teacher_relations"] = teacher_relations
            teacher_entity_total += length(teacher_entities)
            teacher_relation_total += length(teacher_relations)
            println(io, JSON3.write(row))
        end
    end

    println("============================================================")
    println("Teacher Merge Complete")
    println("============================================================")
    println("Base rows:      $(length(base_rows))")
    println("Teacher rows:   $(length(teacher_rows))")
    println("Matched rows:   $matched_rows")
    println("Missing rows:   $missing_rows")
    println("Teacher ents:   $teacher_entity_total")
    println("Teacher rels:   $teacher_relation_total")
    println("Skipped rels:   $skipped_relations")
    println("Output:         $(opts.output_path)")

    if opts.strict && skipped_relations > 0
        error("skipped teacher relations while --strict was enabled")
    end
end

main()
