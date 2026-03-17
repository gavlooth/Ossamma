#!/usr/bin/env julia
"""
Build teacher-request JSONL for REBEL relation extraction distillation.

Each output row contains enough metadata to:
1. send the prompt to an external teacher model
2. recover the source row via stable matching keys
3. merge the teacher response back with `merge_rebel_teacher_targets.jl`

Usage:
    julia --project=. scripts/build_rebel_teacher_requests.jl \
        --input data/rebel/train.jsonl \
        --output data/rebel/train_teacher_requests.jsonl
"""

using JSON3

include(joinpath(@__DIR__, "..", "src", "Swamma.jl"))
using .Swamma.RelationExtraction: load_rebel_jsonl

Base.@kwdef mutable struct RequestOptions
    input_path::Union{Nothing,String} = nothing
    output_path::Union{Nothing,String} = nothing
    max_rows::Int = 0
    include_title::Bool = true
    include_schema::Bool = true
    match_key::String = "auto"
    system_prompt::String = "You are an information extraction system. Return strict JSON only."
    prompt_style::String = "verbose"
    relation_label_mode::String = "id_or_name"
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
    println("  julia --project=. scripts/build_rebel_teacher_requests.jl --input <rebel.jsonl> --output <requests.jsonl>")
    println("Options:")
    println("  --max-rows <n>         Limit exported rows (0 = all)")
    println("  --no-title             Omit title context from prompts")
    println("  --no-schema            Omit explicit label schema from prompts")
    println("  --match-key <auto|docid|uri|title|row_index>")
    println("  --system-prompt <txt>  Override stored system prompt")
    println("  --prompt-style <verbose|compact>")
    println("  --relation-label-mode <id|id_or_name|name>")
    println("  -h, --help")
end

function parse_args(args)::RequestOptions
    opts = RequestOptions()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--input"
            i += 1
            opts.input_path = args[i]
        elseif arg == "--output"
            i += 1
            opts.output_path = args[i]
        elseif arg == "--max-rows"
            i += 1
            opts.max_rows = parse(Int, args[i])
        elseif arg == "--no-title"
            opts.include_title = false
        elseif arg == "--no-schema"
            opts.include_schema = false
        elseif arg == "--match-key"
            i += 1
            opts.match_key = lowercase(args[i])
        elseif arg == "--system-prompt"
            i += 1
            opts.system_prompt = args[i]
        elseif arg == "--prompt-style"
            i += 1
            opts.prompt_style = lowercase(args[i])
        elseif arg == "--relation-label-mode"
            i += 1
            opts.relation_label_mode = lowercase(args[i])
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
    opts.match_key in ("auto", "docid", "uri", "title", "row_index") || error("unsupported --match-key=$(opts.match_key)")
    opts.prompt_style in ("verbose", "compact") || error("unsupported --prompt-style=$(opts.prompt_style)")
    opts.relation_label_mode in ("id", "id_or_name", "name") || error("unsupported --relation-label-mode=$(opts.relation_label_mode)")
    return opts
end

function entity_types(rows)
    labels = Set{String}()
    for row in rows
        haskey(row, :entities) || continue
        for entity in row.entities
            push!(labels, uppercase(String(entity.label)))
        end
    end
    return sort!(collect(labels))
end

function relation_types(rows)
    labels = Set{String}()
    for row in rows
        haskey(row, :relations) || continue
        for rel in row.relations
            push!(labels, String(rel.label))
        end
    end
    return sort!(collect(labels))
end

function row_match_key(row, row_index::Int, mode::String)
    if mode == "row_index"
        return "row_index:" * string(row_index)
    elseif mode == "docid" && haskey(row, :docid)
        return "docid:" * String(row.docid)
    elseif mode == "uri" && haskey(row, :uri)
        return "uri:" * String(row.uri)
    elseif mode == "title" && haskey(row, :title)
        return "title:" * String(row.title)
    end

    if haskey(row, :docid)
        return "docid:" * String(row.docid)
    elseif haskey(row, :uri)
        return "uri:" * String(row.uri)
    elseif haskey(row, :title)
        return "title:" * String(row.title)
    end
    return "row_index:" * string(row_index)
end

function numbered_tokens(tokens)
    return join(["$(i): $(String(tok))" for (i, tok) in enumerate(tokens)], "\n")
end

function schema_text(entity_labels::Vector{String}, relation_labels::Vector{String})
    entity_spec = isempty(entity_labels) ? "(none)" : join(entity_labels, ", ")
    relation_spec = isempty(relation_labels) ? "(none)" : join(relation_labels, ", ")
    return """
Allowed entity labels: $entity_spec
Allowed relation labels: $relation_spec

Return a JSON object with exactly two keys:
- "entities": array of objects with keys "start", "stop", and "label"
- "relations": array of objects with keys "head_start", "head_stop", "tail_start", "tail_stop", "label", and "confidence"

Use 1-based token indices matching the token list below.
Use integer numbers for all span indices and a numeric value for confidence.
If no entities or relations are present, return empty arrays.
Never output the words "Int" or "Float".
Never output keys such as "title", "text", or "tokens".
Start the response with "{" and end it with "}".
Do not use markdown, code fences, comments, or explanatory text.
Do not return text outside the JSON object.
"""
end

function compact_schema_text(entity_labels::Vector{String}, relation_labels::Vector{String}; relation_label_mode::String = "id_or_name")
    entity_spec = isempty(entity_labels) ? "[]" : JSON3.write(entity_labels)
    relation_spec = isempty(relation_labels) ? "[]" : JSON3.write(relation_labels)
    relation_glosses = String[]
    for label in relation_labels
        gloss = get(REBEL_RELATION_GLOSSES, label, nothing)
        gloss === nothing || push!(relation_glosses, string(label, "=", gloss))
    end
    relation_gloss_text = isempty(relation_glosses) ? "" :
        "Relation meaning hints: " * join(relation_glosses, "; ") * "\n"
    relation_label_rule = if relation_label_mode == "id"
        "- for relation `label`, output the Wikidata property ID from the allowed list (for example `P127`)\n"
    elseif relation_label_mode == "name"
        "- for relation `label`, output the exact relation name from the hints (for example `owned by`), not the Wikidata ID; downstream canonicalization will map names back to IDs\n"
    else
        "- for relation `label`, you may output either the Wikidata property ID (for example `P127`) or the exact relation name from the hints (for example `owned by`); downstream canonicalization will map names back to IDs\n"
    end
    return """
Allowed entity labels: $entity_spec
Allowed relation labels: $relation_spec
$relation_gloss_text\
Return only one JSON object:
{"entities":[{"start":1,"stop":1,"label":"ENTITY_TYPE"}],"relations":[{"head_start":1,"head_stop":1,"tail_start":2,"tail_stop":2,"label":"RELATION_TYPE","confidence":0.95}]}
Rules:
- indices are 1-based integers over the token array
- confidence is numeric
- use no keys other than the schema above
- use empty arrays if no entities or relations exist
- extract only high-confidence spans that are needed for allowed relations or are clearly salient named mentions
- do not annotate every noun, modifier, or token; omit speculative CONCEPT spans
- prefer fewer annotations; most rows should stay well under 8 entities and 6 relations
- choose the relation ID whose meaning best matches the spans; do not default to one label across rows
$relation_label_rule\
- if uncertain, return fewer annotations or empty arrays instead of guessing
- close the JSON object immediately after the relations array
"""
end

function build_prompt(row, entity_labels::Vector{String}, relation_labels::Vector{String}; include_title::Bool, include_schema::Bool)
    parts = String[]
    push!(parts, "Extract entity and relation annotations for the following tokenized text.")
    if include_title && haskey(row, :title)
        title = strip(String(row.title))
        isempty(title) || push!(parts, "Title: " * title)
    end
    if include_schema
        push!(parts, schema_text(entity_labels, relation_labels))
    else
        push!(parts, """
Return strict JSON with keys "entities" and "relations".
Use 1-based token indices over the provided token list.
Relations must use span fields: head_start, head_stop, tail_start, tail_stop.
Use integer numbers for all span indices and a numeric value for confidence.
If no entities or relations are present, return empty arrays.
Never output the words "Int" or "Float".
Never output keys such as "title", "text", or "tokens".
Start the response with "{" and end it with "}".
Do not use markdown, code fences, comments, or explanatory text.
Do not return text outside the JSON object.
""")
    end
    push!(parts, """
Return JSON in this exact structural shape:
{"entities":[{"start":1,"stop":1,"label":"ENTITY_TYPE"}],"relations":[{"head_start":1,"head_stop":1,"tail_start":2,"tail_stop":2,"label":"RELATION_TYPE","confidence":0.95}]}
""")
    if haskey(row, :text)
        push!(parts, "Text:\n" * String(row.text))
    end
    push!(parts, "Tokens:\n" * numbered_tokens(row.tokens))
    return join(parts, "\n\n")
end

function build_compact_prompt(row, entity_labels::Vector{String}, relation_labels::Vector{String}; include_title::Bool, include_schema::Bool, relation_label_mode::String = "id_or_name")
    parts = String[]
    push!(parts, "Extract entities and relations from this token array.")
    if include_title && haskey(row, :title)
        title = strip(String(row.title))
        isempty(title) || push!(parts, "Title: " * title)
    end
    if include_schema
        push!(parts, compact_schema_text(entity_labels, relation_labels; relation_label_mode = relation_label_mode))
    else
        push!(parts, """
Return only:
{"entities":[{"start":1,"stop":1,"label":"ENTITY_TYPE"}],"relations":[{"head_start":1,"head_stop":1,"tail_start":2,"tail_stop":2,"label":"RELATION_TYPE","confidence":0.95}]}
Rules:
- indices are 1-based integers over the token array
- confidence is numeric
- use empty arrays if no entities or relations exist
- do not return markdown or any extra text
""")
    end
    push!(parts, "tokens = " * JSON3.write([String(tok) for tok in row.tokens]))
    return join(parts, "\n\n")
end

function main()
    opts = parse_args(ARGS)
    isfile(opts.input_path) || error("input JSONL not found: $(opts.input_path)")
    rows = load_rebel_jsonl(opts.input_path)
    limit = opts.max_rows > 0 ? min(opts.max_rows, length(rows)) : length(rows)
    selected = rows[1:limit]

    entity_label_set = entity_types(selected)
    relation_label_set = relation_types(selected)

    written = 0
    open(opts.output_path, "w") do io
        for (idx, row) in enumerate(selected)
            record = Dict{String,Any}()
            record["row_index"] = idx
            record["match_key"] = row_match_key(row, idx, opts.match_key)
            haskey(row, :docid) && (record["docid"] = String(row.docid))
            haskey(row, :uri) && (record["uri"] = String(row.uri))
            haskey(row, :title) && (record["title"] = String(row.title))
            haskey(row, :text) && (record["text"] = String(row.text))
            record["tokens"] = [String(tok) for tok in row.tokens]
            record["system_prompt"] = opts.system_prompt
            record["prompt"] = opts.prompt_style == "compact" ?
                build_compact_prompt(
                    row,
                    entity_label_set,
                    relation_label_set;
                    include_title = opts.include_title,
                    include_schema = opts.include_schema,
                    relation_label_mode = opts.relation_label_mode,
                ) :
                build_prompt(
                    row,
                    entity_label_set,
                    relation_label_set;
                    include_title = opts.include_title,
                    include_schema = opts.include_schema,
                )
            record["response_schema"] = Dict(
                "entities" => [Dict("start" => 1, "stop" => 1, "label" => "ENTITY_TYPE")],
                "relations" => [Dict(
                    "head_start" => 1,
                    "head_stop" => 1,
                    "tail_start" => 2,
                    "tail_stop" => 2,
                    "label" => "RELATION_TYPE",
                    "confidence" => 0.95,
                )],
            )
            # Keep label metadata in the JSONL even when the prompt omits schema text;
            # downstream strict validation and stage-2 relabeling still need it.
            record["entity_labels"] = entity_label_set
            record["relation_labels"] = relation_label_set
            println(io, JSON3.write(record))
            written += 1
        end
    end

    println("============================================================")
    println("Teacher Request Export Complete")
    println("============================================================")
    println("Input rows:     $(length(rows))")
    println("Exported rows:  $written")
    println("Output:         $(opts.output_path)")
    println("Match key mode: $(opts.match_key)")
    println("Schema in prompt: $(opts.include_schema)")
    println("Prompt style:   $(opts.prompt_style)")
end

main()
