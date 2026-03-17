#!/usr/bin/env julia
"""
Parse raw teacher responses into normalized REBEL-style teacher annotation rows.

The output is intended to feed `merge_rebel_teacher_targets.jl`.

Usage:
    julia --project=. scripts/parse_rebel_teacher_responses.jl \
        --input teacher_raw.jsonl \
        --output teacher_parsed.jsonl
"""

using JSON3

const REBEL_RELATION_GLOSSES = Dict(
    "P17" => "country",
    "P19" => "place of birth",
    "P26" => "spouse",
    "P27" => "country of citizenship",
    "P31" => "instance of",
    "P36" => "capital",
    "P40" => "child",
    "P47" => "shares border with",
    "P50" => "author",
    "P57" => "director",
    "P106" => "occupation",
    "P112" => "founded by",
    "P118" => "league or competition",
    "P127" => "owned by",
    "P136" => "genre",
    "P138" => "named after",
    "P155" => "follows",
    "P159" => "headquarters location",
    "P161" => "cast member",
    "P176" => "manufacturer",
    "P206" => "located in or next to body of water",
    "P276" => "location",
    "P3373" => "sibling",
    "P361" => "part of",
    "P403" => "mouth of the watercourse",
    "P463" => "member of",
    "P571" => "inception",
    "P641" => "sport",
    "P674" => "characters",
    "P710" => "participant",
    "P800" => "notable work",
    "P1365" => "replaces",
)

const REBEL_RELATION_GLOSS_TO_ID = let out = Dict{String,String}()
    for (pid, gloss) in REBEL_RELATION_GLOSSES
        out[lowercase(gloss)] = pid
    end
    out
end

Base.@kwdef mutable struct ParseOptions
    input_path::Union{Nothing,String} = nothing
    output_path::Union{Nothing,String} = nothing
    response_field::String = "response"
    max_rows::Int = 0
    strict::Bool = false
end

function usage()
    println("Usage:")
    println("  julia --project=. scripts/parse_rebel_teacher_responses.jl --input <raw.jsonl> --output <parsed.jsonl>")
    println("Options:")
    println("  --response-field <name>   Field containing the raw teacher response text or object")
    println("  --max-rows <n>            Limit rows processed (0 = all)")
    println("  --strict                  Exit non-zero if any row fails to parse")
    println("  -h, --help")
end

function parse_args(args)::ParseOptions
    opts = ParseOptions()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--input"
            i += 1
            opts.input_path = args[i]
        elseif arg == "--output"
            i += 1
            opts.output_path = args[i]
        elseif arg == "--response-field"
            i += 1
            opts.response_field = args[i]
        elseif arg == "--max-rows"
            i += 1
            opts.max_rows = parse(Int, args[i])
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

    opts.input_path === nothing && error("--input is required")
    opts.output_path === nothing && error("--output is required")
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

function read_jsonl(path::String; max_rows::Int = 0)
    rows = Vector{Dict{String,Any}}()
    open(path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            push!(rows, json_to_native(JSON3.read(line)))
            max_rows > 0 && length(rows) >= max_rows && break
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

function canonicalize_relation_label(label::AbstractString)
    stripped = strip(String(label))
    isempty(stripped) && return nothing
    if startswith(stripped, "P") && all(isdigit, collect(stripped[2:end]))
        return stripped
    end
    normalized = lowercase(replace(stripped, r"\s+" => " "))
    return get(REBEL_RELATION_GLOSS_TO_ID, normalized, nothing)
end

function extract_json_object_text(text::String)
    start_idx = findfirst(==('{'), text)
    start_idx === nothing && return nothing
    depth = 0
    in_string = false
    escaped = false
    for idx in eachindex(text)
        idx < start_idx && continue
        ch = text[idx]
        if in_string
            if escaped
                escaped = false
            elseif ch == '\\'
                escaped = true
            elseif ch == '"'
                in_string = false
            end
            continue
        end
        if ch == '"'
            in_string = true
        elseif ch == '{'
            depth += 1
        elseif ch == '}'
            depth -= 1
            if depth == 0
                return text[start_idx:idx]
            end
        end
    end
    return nothing
end

function parse_teacher_payload(value)
    if value isa Dict{String,Any}
        return value
    elseif value isa AbstractString
        payload_text = extract_json_object_text(String(value))
        payload_text === nothing && return nothing
        return json_to_native(JSON3.read(payload_text))
    else
        return nothing
    end
end

function normalize_entity(entity::Dict{String,Any})
    label = strip(String(field(entity, "label", "")))
    isempty(label) && return nothing
    start_raw = Int(field(entity, "start", 0))
    stop_raw = entity_stop(entity)
    out = Dict{String,Any}()
    out["start"] = start_raw
    out["stop"] = stop_raw
    out["label"] = label
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

function normalize_relation(rel::Dict{String,Any}, entities::Vector{Any})
    raw_label = strip(String(field(rel, "label", "")))
    label = canonicalize_relation_label(raw_label)
    label === nothing && return nothing

    head_span = relation_endpoint_from_span_fields(rel, "head")
    tail_span = relation_endpoint_from_span_fields(rel, "tail")
    if head_span === nothing || tail_span === nothing
        if haskey(rel, "head") && haskey(rel, "tail")
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
            head_span = (Int(field(head_entity, "start", 0)), entity_stop(head_entity))
            tail_span = (Int(field(tail_entity, "start", 0)), entity_stop(tail_entity))
        else
            return nothing
        end
    end

    out = Dict{String,Any}()
    out["head_start"] = head_span[1]
    out["head_stop"] = head_span[2]
    out["tail_start"] = tail_span[1]
    out["tail_stop"] = tail_span[2]
    out["label"] = label
    if haskey(rel, "confidence")
        out["confidence"] = rel["confidence"]
    elseif haskey(rel, "score")
        out["confidence"] = rel["score"]
    end
    return out
end

function normalize_payload(payload::Dict{String,Any})
    entities_raw = haskey(payload, "entities") ? payload["entities"] : Any[]
    relations_raw = haskey(payload, "relations") ? payload["relations"] : Any[]

    entities = Any[]
    for entity in entities_raw
        entity isa Dict{String,Any} || continue
        norm = normalize_entity(entity)
        norm === nothing || push!(entities, norm)
    end

    relations = Any[]
    for rel in relations_raw
        rel isa Dict{String,Any} || continue
        norm = normalize_relation(rel, entities)
        norm === nothing || push!(relations, norm)
    end

    return entities, relations
end

function carry_metadata!(out::Dict{String,Any}, row::Dict{String,Any})
    for key in ("match_key", "docid", "uri", "title", "row_index")
        haskey(row, key) && (out[key] = row[key])
    end
end

function main()
    opts = parse_args(ARGS)
    isfile(opts.input_path) || error("input JSONL not found: $(opts.input_path)")
    rows = read_jsonl(opts.input_path; max_rows = opts.max_rows)

    parsed_rows = 0
    failed_rows = 0
    entity_total = 0
    relation_total = 0

    open(opts.output_path, "w") do io
        for row in rows
            response_value = if haskey(row, opts.response_field)
                row[opts.response_field]
            elseif haskey(row, "teacher_response")
                row["teacher_response"]
            elseif haskey(row, "output")
                row["output"]
            else
                nothing
            end

            payload = response_value === nothing ? nothing : parse_teacher_payload(response_value)
            if payload === nothing
                failed_rows += 1
                continue
            end

            entities, relations = normalize_payload(payload)
            out = Dict{String,Any}()
            carry_metadata!(out, row)
            out["entities"] = entities
            out["relations"] = relations
            println(io, JSON3.write(out))

            parsed_rows += 1
            entity_total += length(entities)
            relation_total += length(relations)
        end
    end

    println("============================================================")
    println("Teacher Response Parse Complete")
    println("============================================================")
    println("Input rows:     $(length(rows))")
    println("Parsed rows:    $parsed_rows")
    println("Failed rows:    $failed_rows")
    println("Entities total: $entity_total")
    println("Relations total: $relation_total")
    println("Output:         $(opts.output_path)")

    if opts.strict && failed_rows > 0
        error("failed to parse $(failed_rows) teacher response rows")
    end
end

main()
