#!/usr/bin/env julia

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

const REBEL_RELATION_ALIAS_TO_ID = Dict(
    "headquartered in" => "P159",
    "headquartered at" => "P159",
    "headquarters in" => "P159",
    "headquarters at" => "P159",
)

Base.@kwdef mutable struct ApplyOptions
    teacher_path::Union{Nothing,String} = nothing
    selections_path::Union{Nothing,String} = nothing
    output_path::Union{Nothing,String} = nothing
    response_field::String = "response"
    response_prefix::String = "{\"label\":\""
end

function usage()
    println("Usage:")
    println("  julia --project=. scripts/apply_rebel_relation_label_selections.jl --teacher <parsed_teacher.jsonl> --selections <selection_raw.jsonl> --output <updated_teacher.jsonl>")
    println("Options:")
    println("  --response-field <name>   Field containing the raw label-selection response")
    println("  --response-prefix <txt>   Prefix used during generation when responses are stored as suffix text")
    println("  -h, --help")
end

function parse_args(args)
    opts = ApplyOptions()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--teacher"
            i += 1
            opts.teacher_path = args[i]
        elseif arg == "--selections"
            i += 1
            opts.selections_path = args[i]
        elseif arg == "--output"
            i += 1
            opts.output_path = args[i]
        elseif arg == "--response-field"
            i += 1
            opts.response_field = args[i]
        elseif arg == "--response-prefix"
            i += 1
            opts.response_prefix = args[i]
        elseif arg == "-h" || arg == "--help"
            usage()
            exit(0)
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end
    opts.teacher_path === nothing && error("--teacher is required")
    opts.selections_path === nothing && error("--selections is required")
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

function allowed_relation_label_set(row::Dict{String,Any})
    values = get(row, "relation_labels", nothing)
    values isa AbstractVector || return nothing
    labels = Set{String}()
    for value in values
        label = strip(String(value))
        isempty(label) || push!(labels, label)
    end
    return isempty(labels) ? nothing : labels
end

function canonicalize_relation_label(label::AbstractString; allowed_labels = nothing)
    stripped = strip(String(label))
    isempty(stripped) && return nothing
    candidate = if startswith(stripped, "P") && ncodeunits(stripped) > 1 && all(isdigit, collect(stripped[2:end]))
        stripped
    else
        normalized = lowercase(replace(stripped, r"\s+" => " "))
        get(REBEL_RELATION_GLOSS_TO_ID, normalized, get(REBEL_RELATION_ALIAS_TO_ID, normalized, nothing))
    end
    candidate === nothing && return nothing
    allowed_labels === nothing && return candidate
    return candidate in allowed_labels ? candidate : nothing
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

function extract_label_field_text(text::String)
    match = findfirst("\"label\":\"", text)
    match === nothing && return nothing
    idx = last(match) + 1
    buffer = IOBuffer()
    escaped = false
    while idx <= lastindex(text)
        ch = text[idx]
        if escaped
            print(buffer, ch)
            escaped = false
        elseif ch == '\\'
            escaped = true
        elseif ch == '"'
            return String(take!(buffer))
        else
            print(buffer, ch)
        end
        idx = nextind(text, idx)
    end
    return nothing
end

function parse_selection_label(row::Dict{String,Any}, response_field::String, response_prefix::String)
    haskey(row, response_field) || return nothing
    allowed_labels = allowed_relation_label_set(row)
    value = row[response_field]
    payload = if value isa Dict{String,Any}
        value
    elseif value isa AbstractString
        raw_text = String(value)
        payload_text = extract_json_object_text(raw_text)
        if payload_text === nothing && !isempty(response_prefix)
            payload_text = extract_json_object_text(response_prefix * raw_text)
        end
        if payload_text === nothing
            nothing
        else
            try
                json_to_native(JSON3.read(payload_text))
            catch
                nothing
            end
        end
    else
        return nothing
    end
    if payload !== nothing && haskey(payload, "label")
        label = canonicalize_relation_label(String(payload["label"]); allowed_labels = allowed_labels)
        label === nothing || return label
    end
    value isa AbstractString || return nothing
    raw_text = String(value)
    for candidate_text in (raw_text, isempty(response_prefix) ? raw_text : response_prefix * raw_text)
        label_text = extract_label_field_text(candidate_text)
        label_text === nothing && continue
        label = canonicalize_relation_label(label_text; allowed_labels = allowed_labels)
        label === nothing || return label
    end
    return nothing
end

function main()
    opts = parse_args(ARGS)
    isfile(opts.teacher_path) || error("teacher JSONL not found: $(opts.teacher_path)")
    isfile(opts.selections_path) || error("selections JSONL not found: $(opts.selections_path)")

    teacher_rows = read_jsonl(opts.teacher_path)
    selection_rows = read_jsonl(opts.selections_path)
    label_by_candidate = Dict{String,String}()
    for row in selection_rows
        haskey(row, "candidate_key") || continue
        label = parse_selection_label(row, opts.response_field, opts.response_prefix)
        label === nothing && continue
        label_by_candidate[String(row["candidate_key"])] = label
    end

    updated = 0
    total_relations = 0
    open(opts.output_path, "w") do io
        for teacher_row in teacher_rows
            match_key = haskey(teacher_row, "match_key") ? String(teacher_row["match_key"]) : ""
            relations = haskey(teacher_row, "relations") ? teacher_row["relations"] : Any[]
            for (rel_idx, relation_any) in enumerate(relations)
                relation_any isa Dict{String,Any} || continue
                total_relations += 1
                candidate_key = string(match_key, "#rel", rel_idx)
                if haskey(label_by_candidate, candidate_key)
                    relation_any["label"] = label_by_candidate[candidate_key]
                    updated += 1
                end
            end
            println(io, JSON3.write(teacher_row))
        end
    end

    println("============================================================")
    println("Relation Label Merge Complete")
    println("============================================================")
    println("Teacher rows:        $(length(teacher_rows))")
    println("Total relations:     $total_relations")
    println("Updated relations:   $updated")
    println("Output:              $(opts.output_path)")
end

main()
