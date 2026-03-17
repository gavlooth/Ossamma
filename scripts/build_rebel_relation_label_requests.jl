#!/usr/bin/env julia

using JSON3

Base.@kwdef mutable struct BuildOptions
    requests_path::Union{Nothing,String} = nothing
    teacher_path::Union{Nothing,String} = nothing
    output_path::Union{Nothing,String} = nothing
    max_rows::Int = 0
    system_prompt::String = "You are an information extraction system. Return strict JSON only."
end

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

function usage()
    println("Usage:")
    println("  julia --project=. scripts/build_rebel_relation_label_requests.jl --requests <requests.jsonl> --teacher <parsed_teacher.jsonl> --output <label_requests.jsonl>")
    println("Options:")
    println("  --max-rows <n>         Limit number of source teacher rows processed (0 = all)")
    println("  --system-prompt <txt>  Override stored system prompt")
    println("  -h, --help")
end

function parse_args(args)
    opts = BuildOptions()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--requests"
            i += 1
            opts.requests_path = args[i]
        elseif arg == "--teacher"
            i += 1
            opts.teacher_path = args[i]
        elseif arg == "--output"
            i += 1
            opts.output_path = args[i]
        elseif arg == "--max-rows"
            i += 1
            opts.max_rows = parse(Int, args[i])
        elseif arg == "--system-prompt"
            i += 1
            opts.system_prompt = args[i]
        elseif arg == "-h" || arg == "--help"
            usage()
            exit(0)
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end
    opts.requests_path === nothing && error("--requests is required")
    opts.teacher_path === nothing && error("--teacher is required")
    opts.output_path === nothing && error("--output is required")
    return opts
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

function span_text(tokens, start_idx::Int, stop_idx::Int)
    n = length(tokens)
    if !(1 <= start_idx <= n && 1 <= stop_idx <= n && start_idx <= stop_idx)
        return ""
    end
    return join(String.(tokens[start_idx:stop_idx]), " ")
end

function gloss_lines(relation_labels)
    lines = String[]
    for label in relation_labels
        gloss = get(REBEL_RELATION_GLOSSES, String(label), nothing)
        gloss === nothing || push!(lines, string("- ", gloss, " (", label, ")"))
    end
    return lines
end

function build_prompt(request_row::Dict{String,Any}, relation::Dict{String,Any})
    tokens = haskey(request_row, "tokens") ? request_row["tokens"] : Any[]
    relation_labels = haskey(request_row, "relation_labels") ? request_row["relation_labels"] : Any[]
    glosses = gloss_lines(relation_labels)
    head_start = Int(relation["head_start"])
    head_stop = Int(relation["head_stop"])
    tail_start = Int(relation["tail_start"])
    tail_stop = Int(relation["tail_stop"])
    head_text = span_text(tokens, head_start, head_stop)
    tail_text = span_text(tokens, tail_start, tail_stop)
    return """
Choose the best relation meaning for this candidate relation.

Allowed relation choices:
$(join(glosses, "\n"))

Return exactly one JSON object:
{"label":"relation name","confidence":0.95}

Rules:
- copy the relation name exactly from the allowed list, not the Wikidata ID
- choose only one label
- do not invent a new label or paraphrase an allowed label
- do not output an entity type or free-form category such as "city", "single", "nickname", "person", or "country" unless that exact phrase appears in the allowed relation choices
- if no listed relation fits, output {"label":"","confidence":0.0}
- do not output any extra text

Head span:
- indices: $head_start-$head_stop
- text: $(JSON3.write(head_text))

Tail span:
- indices: $tail_start-$tail_stop
- text: $(JSON3.write(tail_text))

tokens = $(JSON3.write(String.(tokens)))
"""
end

function main()
    opts = parse_args(ARGS)
    isfile(opts.requests_path) || error("requests JSONL not found: $(opts.requests_path)")
    isfile(opts.teacher_path) || error("teacher JSONL not found: $(opts.teacher_path)")

    request_rows = read_jsonl(opts.requests_path)
    teacher_rows = read_jsonl(opts.teacher_path)
    request_by_key = Dict{String,Dict{String,Any}}()
    for row in request_rows
        haskey(row, "match_key") || continue
        request_by_key[String(row["match_key"])] = row
    end

    limit = opts.max_rows > 0 ? min(opts.max_rows, length(teacher_rows)) : length(teacher_rows)
    selected_rows = teacher_rows[1:limit]
    written = 0

    open(opts.output_path, "w") do io
        for teacher_row in selected_rows
            haskey(teacher_row, "match_key") || continue
            match_key = String(teacher_row["match_key"])
            request_row = get(request_by_key, match_key, nothing)
            request_row === nothing && continue
            relations = haskey(teacher_row, "relations") ? teacher_row["relations"] : Any[]
            for (rel_idx, relation_any) in enumerate(relations)
                relation_any isa Dict{String,Any} || continue
                relation = relation_any
                prompt = build_prompt(request_row, relation)
                out = Dict{String,Any}()
                out["candidate_key"] = string(match_key, "#rel", rel_idx)
                out["match_key"] = match_key
                out["relation_index"] = rel_idx
                out["prompt"] = prompt
                out["system_prompt"] = opts.system_prompt
                out["relation_labels"] = get(request_row, "relation_labels", Any[])
                out["head_start"] = relation["head_start"]
                out["head_stop"] = relation["head_stop"]
                out["tail_start"] = relation["tail_start"]
                out["tail_stop"] = relation["tail_stop"]
                println(io, JSON3.write(out))
                written += 1
            end
        end
    end

    println("============================================================")
    println("Relation Label Request Export Complete")
    println("============================================================")
    println("Teacher rows:   $(length(selected_rows))")
    println("Exported rows:  $written")
    println("Output:         $(opts.output_path)")
end

main()
