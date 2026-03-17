#!/usr/bin/env julia
"""
Generate raw teacher responses for REBEL distillation requests using the
Julia-native Granite path.

This consumes the JSONL emitted by `build_rebel_teacher_requests.jl` and writes
raw response JSONL compatible with `parse_rebel_teacher_responses.jl`.

Example:
    julia --project=. scripts/generate_rebel_teacher_responses_native.jl \
        --input data/rebel/train_teacher_requests.jsonl \
        --output data/rebel/train_teacher_raw_native.jsonl \
        --teacher-model ibm-granite/granite-4.0-micro \
        --local-files-only \
        --resume
"""

using ArgParse
using JSON3
using Random
using Statistics

include(joinpath(@__DIR__, "..", "src", "Swamma.jl"))

using .Swamma
using .Swamma.HFTokenizer: load_tokenizer, has_chat_template, apply_chat_template, encode, decode

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

Base.@kwdef struct NativeTeacherGenOptions
    input_path::String = ""
    output_path::String = ""
    teacher_model::String = "ibm-granite/granite-4.0-micro"
    prompt_field::String = "prompt"
    system_prompt_field::String = "system_prompt"
    response_field::String = "response"
    resume::Bool = false
    overwrite::Bool = false
    max_rows::Int = 0
    max_input_tokens::Int = 2048
    max_new_tokens::Int = 512
    temperature::Float32 = 0f0
    top_p::Float32 = 1f0
    do_sample::Bool = false
    stop_sequences::Vector{String} = String[]
    dtype_name::String = "float32"
    plain_prompt::Bool = false
    local_files_only::Bool = false
    seed::Int = 42
    response_prefix::String = "{\"entities\":"
    require_json::Bool = true
    verbose::Bool = false
    use_json_heuristics::Bool = true
    stop_on_complete_json::Bool = true
    max_entities_hint::Int = 8
    max_relations_hint::Int = 6
end

function parse_args(args)
    settings = ArgParseSettings(description = "Generate raw teacher responses for REBEL requests via NativeTeacherLM.")
    @add_arg_table! settings begin
        "--input"
            help = "Input request JSONL"
            required = true
        "--output"
            help = "Output raw response JSONL"
            required = true
        "--teacher-model"
            help = "Teacher model id/path"
            default = "ibm-granite/granite-4.0-micro"
        "--prompt-field"
            help = "Field containing rendered prompt text"
            default = "prompt"
        "--system-prompt-field"
            help = "Field containing system prompt"
            default = "system_prompt"
        "--response-field"
            help = "Output field name for raw teacher response"
            default = "response"
        "--resume"
            help = "Skip match keys already present in the output JSONL"
            action = :store_true
        "--overwrite"
            help = "Allow overwriting the output JSONL when not using --resume"
            action = :store_true
        "--max-rows"
            help = "Optional cap after resume filtering"
            arg_type = Int
            default = 0
        "--max-input-tokens"
            help = "Prompt truncation length before generation"
            arg_type = Int
            default = 2048
        "--max-new-tokens"
            help = "Maximum teacher completion length"
            arg_type = Int
            default = 512
        "--temperature"
            help = "Generation temperature"
            arg_type = Float64
            default = 0.0
        "--top-p"
            help = "Top-p for sampling"
            arg_type = Float64
            default = 1.0
        "--do-sample"
            help = "Use sampling instead of greedy decoding"
            action = :store_true
        "--stop-sequence"
            help = "Optional stop sequence; may be repeated"
            action = :append_arg
        "--dtype"
            help = "Weight dtype: float16 | float32"
            default = "float32"
        "--plain-prompt"
            help = "Disable chat template usage even if the tokenizer supports it"
            action = :store_true
        "--local-files-only"
            help = "Refuse network fetches and only use local HF cache/files"
            action = :store_true
        "--seed"
            help = "Sampling seed"
            arg_type = Int
            default = 42
        "--response-prefix"
            help = "Prefix appended to the assistant response before generation"
            default = "{\"entities\":"
        "--allow-non-json"
            help = "Allow completions that do not contain a parseable JSON object"
            action = :store_true
        "--verbose"
            help = "Print per-stage timing and per-row generation progress"
            action = :store_true
        "--disable-json-heuristics"
            help = "Disable lightweight JSON-aware token suppression heuristics"
            action = :store_true
        "--disable-stop-on-complete-json"
            help = "Do not stop early when the generated response forms a complete JSON object"
            action = :store_true
        "--max-entities-hint"
            help = "Soft cap for lightweight entity-array close-out heuristics (0 disables)"
            arg_type = Int
            default = 8
        "--max-relations-hint"
            help = "Soft cap for lightweight relation-array close-out heuristics (0 disables)"
            arg_type = Int
            default = 6
    end

    parsed = ArgParse.parse_args(args, settings)
    return NativeTeacherGenOptions(
        input_path = parsed["input"],
        output_path = parsed["output"],
        teacher_model = parsed["teacher-model"],
        prompt_field = parsed["prompt-field"],
        system_prompt_field = parsed["system-prompt-field"],
        response_field = parsed["response-field"],
        resume = parsed["resume"],
        overwrite = parsed["overwrite"],
        max_rows = parsed["max-rows"],
        max_input_tokens = parsed["max-input-tokens"],
        max_new_tokens = parsed["max-new-tokens"],
        temperature = Float32(parsed["temperature"]),
        top_p = Float32(parsed["top-p"]),
        do_sample = parsed["do-sample"],
        stop_sequences = String.(get(parsed, "stop-sequence", String[])),
        dtype_name = parsed["dtype"],
        plain_prompt = parsed["plain-prompt"],
        local_files_only = parsed["local-files-only"],
        seed = parsed["seed"],
        response_prefix = parsed["response-prefix"],
        require_json = !parsed["allow-non-json"],
        verbose = parsed["verbose"],
        use_json_heuristics = !parsed["disable-json-heuristics"],
        stop_on_complete_json = !parsed["disable-stop-on-complete-json"],
        max_entities_hint = parsed["max-entities-hint"],
        max_relations_hint = parsed["max-relations-hint"],
    )
end

function resolve_dtype(name::String)
    lowered = lowercase(strip(name))
    lowered == "float16" && return Float16
    lowered == "float32" && return Float32
    error("Unsupported --dtype=$(repr(name)). Expected float16 or float32.")
end

function load_existing_match_keys(path::String)
    keys = Set{String}()
    isfile(path) || return keys
    open(path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            row = try
                JSON3.read(line)
            catch
                nothing
            end
            row === nothing && continue
            if haskey(row, "match_key")
                key = strip(String(row["match_key"]))
                isempty(key) || push!(keys, key)
            end
        end
    end
    return keys
end

function iter_request_rows(path::String)
    rows = Vector{NamedTuple}()
    open(path, "r") do io
        for (idx, line) in enumerate(eachline(io))
            isempty(strip(line)) && continue
            row = JSON3.read(line)
            push!(rows, (source_index = idx, row = row))
        end
    end
    return rows
end

function build_rendered_prompt(tokenizer, prompt::AbstractString, system_prompt::AbstractString, plain_prompt::Bool)
    if !plain_prompt && has_chat_template(tokenizer)
        messages = Any[]
        isempty(system_prompt) || push!(messages, (role = "system", content = String(system_prompt)))
        push!(messages, (role = "user", content = String(prompt)))
        return apply_chat_template(tokenizer, messages; add_generation_prompt = true)
    end
    if isempty(system_prompt)
        return String(prompt)
    end
    return "System: $(String(system_prompt))\n\nUser: $(String(prompt))\n\nAssistant:"
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

function validate_teacher_json_structure(text::String)
    json_text = extract_json_object_text(text)
    json_text === nothing && return nothing
    payload = try
        JSON3.read(json_text)
    catch
        nothing
    end
    payload === nothing && return nothing
    haskey(payload, "entities") || return nothing
    haskey(payload, "relations") || return nothing
    return json_text
end

function label_set_from_row(row, key::String)
    values = get(row, key, nothing)
    values === nothing && return nothing
    values isa AbstractVector || return nothing
    labels = Set{String}()
    for value in values
        label = strip(String(value))
        isempty(label) || push!(labels, label)
    end
    return isempty(labels) ? nothing : labels
end

function canonicalize_relation_label(label::AbstractString)
    stripped = strip(String(label))
    isempty(stripped) && return nothing
    if startswith(stripped, "P") && ncodeunits(stripped) > 1 && all(isdigit, collect(stripped[2:end]))
        return stripped
    end
    normalized = lowercase(replace(stripped, r"\s+" => " "))
    return get(REBEL_RELATION_GLOSS_TO_ID, normalized, nothing)
end

function payload_labels_allowed(payload, allowed_entity_labels, allowed_relation_labels)
    entities = get(payload, "entities", Any[])
    relations = get(payload, "relations", Any[])

    if allowed_entity_labels !== nothing
        for entity in entities
            entity isa AbstractDict || return false
            label = strip(String(get(entity, "label", "")))
            label in allowed_entity_labels || return false
        end
    end

    if allowed_relation_labels !== nothing
        for relation in relations
            relation isa AbstractDict || return false
            raw_label = strip(String(get(relation, "label", "")))
            label = canonicalize_relation_label(raw_label)
            label === nothing && return false
            label in allowed_relation_labels || return false
        end
    end

    return true
end

function validate_teacher_json(
    text::String;
    allowed_entity_labels = nothing,
    allowed_relation_labels = nothing,
)
    json_text = validate_teacher_json_structure(text)
    json_text === nothing && return nothing
    payload = try
        JSON3.read(json_text)
    catch
        nothing
    end
    payload === nothing && return nothing
    payload_labels_allowed(payload, allowed_entity_labels, allowed_relation_labels) || return nothing
    return json_text
end

function extract_complete_top_level_objects(text::AbstractString)
    objects = String[]
    depth = 0
    in_string = false
    escaped = false
    object_start = nothing
    for idx in eachindex(text)
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
            depth == 0 && (object_start = idx)
            depth += 1
        elseif ch == '}'
            depth == 0 && continue
            depth -= 1
            if depth == 0 && object_start !== nothing
                push!(objects, String(text[object_start:idx]))
                object_start = nothing
            end
        end
    end
    return objects
end

function salvage_relation_only_json(
    text::AbstractString,
    max_relations_hint::Int;
    allowed_relation_labels = nothing,
)
    relation_prefix = "{\"entities\":[],\"relations\":["
    match = findfirst(relation_prefix, text)
    match === nothing && return nothing
    relation_text = text[(last(match) + 1):end]
    objects = extract_complete_top_level_objects(relation_text)
    isempty(objects) && return nothing
    if max_relations_hint > 0
        objects = objects[1:min(length(objects), max_relations_hint)]
    end
    repaired = relation_prefix * join(objects, ",") * "]}"
    return validate_teacher_json(
        repaired;
        allowed_entity_labels = nothing,
        allowed_relation_labels = allowed_relation_labels,
    )
end

function candidate_response_texts(completion_text::AbstractString, response_prefix::AbstractString)
    stripped = String(lstrip(completion_text))
    candidates = String[]
    isempty(stripped) || push!(candidates, stripped)
    prefixed = String(response_prefix) * stripped
    prefixed == stripped || push!(candidates, prefixed)
    return candidates
end

function resolve_response_text(
    completion_text::AbstractString,
    response_prefix::AbstractString;
    require_json::Bool,
    max_relations_hint::Int = 0,
    allowed_entity_labels = nothing,
    allowed_relation_labels = nothing,
)
    candidates = candidate_response_texts(completion_text, response_prefix)
    if require_json
        for candidate in candidates
            json_text = validate_teacher_json(
                candidate;
                allowed_entity_labels = allowed_entity_labels,
                allowed_relation_labels = allowed_relation_labels,
            )
            json_text === nothing || return json_text
            repaired = salvage_relation_only_json(
                candidate,
                max_relations_hint;
                allowed_relation_labels = allowed_relation_labels,
            )
            repaired === nothing || return repaired
        end
        return nothing
    end
    isempty(candidates) && return ""
    return first(candidates)
end

function preview_text(text, limit::Int = 256)
    text === nothing && return ""
    rendered = String(text)
    return first(rendered, min(lastindex(rendered), limit))
end

function apply_stop_sequences(text::String, stop_sequences::Vector{String})
    isempty(stop_sequences) && return strip(text)
    cutoff = lastindex(text)
    found = false
    for stop in stop_sequences
        isempty(stop) && continue
        range = findfirst(stop, text)
        range === nothing && continue
        cutoff = min(cutoff, first(range) - 1)
        found = true
    end
    return found ? rstrip(text[begin:cutoff]) : strip(text)
end

function sample_from_logits(logits::AbstractVector{<:Real}, rng::AbstractRNG; temperature::Float32, top_p::Float32)
    if temperature <= 0f0 || !(0f0 < top_p < 1f0)
        shifted = Float32.(logits) ./ max(temperature, 1f-6)
        max_logit = maximum(shifted)
        probs = exp.(shifted .- max_logit)
        probs ./= sum(probs)
        if top_p < 1f0
            order = sortperm(probs; rev = true)
            cumulative = 0f0
            keep = falses(length(probs))
            for idx in order
                keep[idx] = true
                cumulative += probs[idx]
                cumulative >= top_p && break
            end
            probs .*= keep
            probs ./= sum(probs)
        end
        threshold = rand(rng)
        cumulative = 0f0
        for idx in eachindex(probs)
            cumulative += probs[idx]
            if cumulative >= threshold
                return idx
            end
        end
        return argmax(probs)
    end

    shifted = Float32.(logits) ./ temperature
    max_logit = maximum(shifted)
    probs = exp.(shifted .- max_logit)
    probs ./= sum(probs)
    if top_p < 1f0
        order = sortperm(probs; rev = true)
        cumulative = 0f0
        keep = falses(length(probs))
        for idx in order
            keep[idx] = true
            cumulative += probs[idx]
            cumulative >= top_p && break
        end
        probs .*= keep
        probs ./= sum(probs)
    end
    threshold = rand(rng)
    cumulative = 0f0
    for idx in eachindex(probs)
        cumulative += probs[idx]
        if cumulative >= threshold
            return idx
        end
    end
    return argmax(probs)
end

function choose_next_token(logits::AbstractVector{<:Real}, rng::AbstractRNG, opts::NativeTeacherGenOptions)
    if !opts.do_sample || opts.temperature <= 0f0
        return Int(argmax(logits))
    end
    return Int(sample_from_logits(logits, rng; temperature = opts.temperature, top_p = opts.top_p))
end

function generate_response_tokens(
    model,
    params,
    state,
    prompt_tokens::Vector{Int},
    tokenizer,
    rng::AbstractRNG,
    opts::NativeTeacherGenOptions,
    heuristics,
    allowed_entity_labels,
    allowed_relation_labels,
    relation_label_constraint,
)
    prompt_matrix = reshape(prompt_tokens, :, 1)
    cache = init_decoder_cache(model, 1)
    eos_token_id = tokenizer.eos_token_id === nothing ? nothing : tokenizer.eos_token_id + 1
    generated = Int[]
    prefill_logits, generation_state, cache = forward_with_cache(model, prompt_matrix, params, state, cache)

    for _ in 1:opts.max_new_tokens
        source_logits = if size(prefill_logits, 2) == 1
            view(prefill_logits, :, 1, 1)
        else
            view(prefill_logits, :, size(prefill_logits, 2), 1)
        end
        response_so_far = opts.response_prefix * decode(tokenizer, generated; skip_special_tokens = true)
        constrained_logits = heuristics === nothing ?
            Float32.(source_logits) :
            apply_json_heuristics(source_logits, response_so_far, length(generated), heuristics)
        constrained_logits = apply_relation_label_constraint(
            constrained_logits,
            response_so_far,
            relation_label_constraint,
        )
        next_token = choose_next_token(constrained_logits, rng, opts)
        push!(generated, next_token)
        next_token_matrix = reshape(Int[next_token], :, 1)

        decoded = decode(tokenizer, generated; skip_special_tokens = true)
        if opts.stop_on_complete_json
            json_text = resolve_response_text(
                decoded,
                opts.response_prefix;
                require_json = true,
                max_relations_hint = opts.max_relations_hint,
                allowed_entity_labels = allowed_entity_labels,
                allowed_relation_labels = allowed_relation_labels,
            )
            json_text === nothing || return generated, json_text
        end
        stripped = apply_stop_sequences(decoded, opts.stop_sequences)
        if stripped != strip(decoded)
            return generated, stripped
        end
        if eos_token_id !== nothing && next_token == eos_token_id
            break
        end
        prefill_logits, generation_state, cache =
            forward_with_cache(model, next_token_matrix, params, generation_state, cache)
    end

    return generated, apply_stop_sequences(
        decode(tokenizer, generated; skip_special_tokens = true),
        opts.stop_sequences,
    )
end

function generation_config_dict(opts::NativeTeacherGenOptions)
    return Dict(
        "teacher_model" => opts.teacher_model,
        "max_input_tokens" => opts.max_input_tokens,
        "max_new_tokens" => opts.max_new_tokens,
        "temperature" => opts.temperature,
        "top_p" => opts.top_p,
        "do_sample" => opts.do_sample,
        "stop_sequences" => copy(opts.stop_sequences),
        "dtype" => opts.dtype_name,
        "plain_prompt" => opts.plain_prompt,
        "local_files_only" => opts.local_files_only,
        "native_julia" => true,
        "response_prefix" => opts.response_prefix,
        "require_json" => opts.require_json,
        "max_entities_hint" => opts.max_entities_hint,
        "max_relations_hint" => opts.max_relations_hint,
    )
end

function log_verbose(opts::NativeTeacherGenOptions, message::AbstractString)
    opts.verbose || return nothing
    println(message)
    flush(stdout)
    return nothing
end

Base.@kwdef struct NativeJsonHeuristics
    zero_token_id::Union{Int,Nothing} = nothing
    eos_token_id::Union{Int,Nothing} = nothing
    code_fence_token_id::Union{Int,Nothing} = nothing
    json_token_id::Union{Int,Nothing} = nothing
    quote_colon_token_id::Union{Int,Nothing} = nothing
    quote_space_token_id::Union{Int,Nothing} = nothing
    quote_token_id::Union{Int,Nothing} = nothing
    comma_token_id::Union{Int,Nothing} = nothing
    comma_quote_token_id::Union{Int,Nothing} = nothing
    comma_curly_token_id::Union{Int,Nothing} = nothing
    close_object_comma_curly_token_id::Union{Int,Nothing} = nothing
    close_object_comma_curly_quote_token_id::Union{Int,Nothing} = nothing
    max_entities_hint::Int = 8
    max_relations_hint::Int = 6
end

Base.@kwdef struct RelationLabelConstraint
    next_token_ids_by_prefix::Dict{String,Set{Int}} = Dict{String,Set{Int}}()
    complete_prefixes::Set{String} = Set{String}()
    quote_token_id::Union{Int,Nothing} = nothing
end

function maybe_single_token_id(tokenizer, literal::String)
    ids = encode(tokenizer, literal; add_special_tokens = false)
    length(ids) == 1 || return nothing
    return ids[1]
end

function build_relation_label_constraint(tokenizer, allowed_relation_labels)
    allowed_relation_labels === nothing && return nothing
    next_token_ids_by_prefix = Dict{String,Set{Int}}()
    complete_prefixes = Set{String}()
    for label in sort!(collect(allowed_relation_labels))
        ids = encode(tokenizer, label; add_special_tokens = false)
        isempty(ids) && continue
        for prefix_len in 0:(length(ids) - 1)
            prefix_text = prefix_len == 0 ? "" : decode(tokenizer, ids[1:prefix_len]; skip_special_tokens = true)
            bucket = get!(next_token_ids_by_prefix, prefix_text, Set{Int}())
            push!(bucket, ids[prefix_len + 1])
        end
        push!(complete_prefixes, decode(tokenizer, ids; skip_special_tokens = true))
    end
    isempty(next_token_ids_by_prefix) && return nothing
    return RelationLabelConstraint(
        next_token_ids_by_prefix = next_token_ids_by_prefix,
        complete_prefixes = complete_prefixes,
        quote_token_id = maybe_single_token_id(tokenizer, "\""),
    )
end

function build_json_heuristics(tokenizer, opts::NativeTeacherGenOptions)
    return NativeJsonHeuristics(
        zero_token_id = maybe_single_token_id(tokenizer, "0"),
        eos_token_id = tokenizer.eos_token_id === nothing ? nothing : tokenizer.eos_token_id + 1,
        code_fence_token_id = maybe_single_token_id(tokenizer, "```"),
        json_token_id = maybe_single_token_id(tokenizer, "json"),
        quote_colon_token_id = maybe_single_token_id(tokenizer, "\":"),
        quote_space_token_id = maybe_single_token_id(tokenizer, " \""),
        quote_token_id = maybe_single_token_id(tokenizer, "\""),
        comma_token_id = maybe_single_token_id(tokenizer, ","),
        comma_quote_token_id = maybe_single_token_id(tokenizer, ",\""),
        comma_curly_token_id = maybe_single_token_id(tokenizer, ",{"),
        close_object_comma_curly_token_id = maybe_single_token_id(tokenizer, "},{"),
        close_object_comma_curly_quote_token_id = maybe_single_token_id(tokenizer, "},{\""),
        max_entities_hint = opts.max_entities_hint,
        max_relations_hint = opts.max_relations_hint,
    )
end

function numeric_field_open(response_text::String)
    return occursin(r"\"(?:start|stop|head_start|head_stop|tail_start|tail_stop)\":\s*$", response_text)
end

function numeric_field_value_open(response_text::String)
    return occursin(r"\"(?:start|stop|head_start|head_stop|tail_start|tail_stop)\":\s*[1-9]\d*\s*$", response_text)
end

function count_entity_objects(response_text::String)
    return count(_ -> true, eachmatch(r"\{\"start\":", response_text))
end

function count_relation_objects(response_text::String)
    return count(_ -> true, eachmatch(r"\{\"head_start\":", response_text))
end

function entity_array_open(response_text::String)
    return occursin("\"entities\":[", response_text) && !occursin("\"relations\":[", response_text)
end

function relation_array_open(response_text::String)
    return occursin("\"relations\":[", response_text)
end

function object_boundary_open(response_text::String)
    return occursin(r"\}\s*$", response_text)
end

function object_tail_open(response_text::String)
    return occursin(r"\"label\":\"[^\"]*\"\s*$", response_text) ||
           occursin(r"\"confidence\":\s*[0-9]+(?:\.[0-9]+)?\s*$", response_text)
end

function apply_json_heuristics(
    logits::AbstractVector{<:Real},
    response_text::String,
    generated_count::Int,
    heuristics::NativeJsonHeuristics,
)
    adjusted = Float32.(logits)
    if generated_count < 8
        heuristics.eos_token_id !== nothing && (adjusted[heuristics.eos_token_id] = -Inf32)
        heuristics.code_fence_token_id !== nothing && (adjusted[heuristics.code_fence_token_id] = -Inf32)
        heuristics.json_token_id !== nothing && (adjusted[heuristics.json_token_id] = -Inf32)
    end
    if numeric_field_open(response_text)
        heuristics.zero_token_id !== nothing && (adjusted[heuristics.zero_token_id] = -Inf32)
    end
    if numeric_field_value_open(response_text)
        heuristics.quote_colon_token_id !== nothing && (adjusted[heuristics.quote_colon_token_id] = -Inf32)
        heuristics.quote_space_token_id !== nothing && (adjusted[heuristics.quote_space_token_id] = -Inf32)
        heuristics.quote_token_id !== nothing && (adjusted[heuristics.quote_token_id] = -Inf32)
    end
    if object_boundary_open(response_text) || object_tail_open(response_text)
        if heuristics.max_entities_hint > 0 &&
           entity_array_open(response_text) &&
           count_entity_objects(response_text) >= heuristics.max_entities_hint
            heuristics.comma_token_id !== nothing && (adjusted[heuristics.comma_token_id] = -Inf32)
            heuristics.comma_quote_token_id !== nothing && (adjusted[heuristics.comma_quote_token_id] = -Inf32)
            heuristics.comma_curly_token_id !== nothing && (adjusted[heuristics.comma_curly_token_id] = -Inf32)
            heuristics.close_object_comma_curly_token_id !== nothing && (adjusted[heuristics.close_object_comma_curly_token_id] = -Inf32)
            heuristics.close_object_comma_curly_quote_token_id !== nothing && (adjusted[heuristics.close_object_comma_curly_quote_token_id] = -Inf32)
        elseif heuristics.max_relations_hint > 0 &&
               relation_array_open(response_text) &&
               count_relation_objects(response_text) >= heuristics.max_relations_hint
            heuristics.comma_token_id !== nothing && (adjusted[heuristics.comma_token_id] = -Inf32)
            heuristics.comma_quote_token_id !== nothing && (adjusted[heuristics.comma_quote_token_id] = -Inf32)
            heuristics.comma_curly_token_id !== nothing && (adjusted[heuristics.comma_curly_token_id] = -Inf32)
            heuristics.close_object_comma_curly_token_id !== nothing && (adjusted[heuristics.close_object_comma_curly_token_id] = -Inf32)
            heuristics.close_object_comma_curly_quote_token_id !== nothing && (adjusted[heuristics.close_object_comma_curly_quote_token_id] = -Inf32)
        end
    end
    return adjusted
end

function current_open_label_prefix(response_text::String)
    match = Base.match(r"\"label\":\"([^\"]*)$", response_text)
    match === nothing && return nothing
    return match.captures[1]
end

function apply_relation_label_constraint(
    logits::AbstractVector{<:Real},
    response_text::String,
    constraint::Union{Nothing,RelationLabelConstraint},
)
    constraint === nothing && return Float32.(logits)
    relation_array_open(response_text) || return Float32.(logits)
    prefix = current_open_label_prefix(response_text)
    prefix === nothing && return Float32.(logits)

    allowed_ids = Set{Int}()
    if haskey(constraint.next_token_ids_by_prefix, prefix)
        union!(allowed_ids, constraint.next_token_ids_by_prefix[prefix])
    end
    if prefix in constraint.complete_prefixes && constraint.quote_token_id !== nothing
        push!(allowed_ids, constraint.quote_token_id)
    end
    isempty(allowed_ids) && return Float32.(logits)

    constrained = fill(-Inf32, length(logits))
    for idx in allowed_ids
        constrained[idx] = Float32(logits[idx])
    end
    return constrained
end

function truncate_prompt_tokens(prompt_tokens::Vector{Int}, max_prompt_tokens::Int)
    max_prompt_tokens > 0 || return Int[]
    length(prompt_tokens) <= max_prompt_tokens && return prompt_tokens

    # Preserve both the schema/instruction prefix and the tail token list.
    head_keep = min(length(prompt_tokens), min(128, max(32, max_prompt_tokens ÷ 4)))
    tail_keep = max_prompt_tokens - head_keep
    tail_keep <= 0 && return prompt_tokens[1:max_prompt_tokens]
    head = prompt_tokens[1:head_keep]
    tail = prompt_tokens[(end - tail_keep + 1):end]
    return vcat(head, tail)
end

function main()
    opts = parse_args(ARGS)
    isfile(opts.input_path) || error("Input file not found: $(opts.input_path)")
    output_path = opts.output_path
    error_path = output_path * ".errors.jsonl"

    if isfile(output_path) && !opts.resume && !opts.overwrite
        error("Output already exists: $(output_path). Use --resume or --overwrite.")
    end
    if opts.overwrite && !opts.resume
        isfile(output_path) && rm(output_path)
        isfile(error_path) && rm(error_path)
    end

    mkpath(dirname(output_path))

    rng = Random.MersenneTwister(opts.seed)
    setup_started = time()
    tokenizer = load_tokenizer(opts.teacher_model; local_files_only = opts.local_files_only)
    heuristics = opts.use_json_heuristics ? build_json_heuristics(tokenizer, opts) : nothing
    dtype = resolve_dtype(opts.dtype_name)
    log_verbose(opts, "Loaded tokenizer in $(round(time() - setup_started; digits = 3))s")
    model_started = time()
    model, params, state = load_granite_model(
        opts.teacher_model;
        rng = Random.MersenneTwister(1),
        dtype = dtype,
        local_files_only = opts.local_files_only,
    )
    log_verbose(opts, "Loaded native teacher model in $(round(time() - model_started; digits = 3))s")

    rows = iter_request_rows(opts.input_path)
    existing = opts.resume ? load_existing_match_keys(output_path) : Set{String}()
    mode = opts.resume ? "a" : "w"

    accepted = 0
    skipped_existing = 0
    failed = 0
    gen_cfg = generation_config_dict(opts)

    open(output_path, mode) do out_io
        open(error_path, mode) do err_io
            for request in rows
                row_started = time()
                row = request.row
                prompt = strip(String(get(row, opts.prompt_field, "")))
                isempty(prompt) && continue
                system_prompt = strip(String(get(row, opts.system_prompt_field, "")))
                allowed_entity_labels = label_set_from_row(row, "entity_labels")
                allowed_relation_labels = label_set_from_row(row, "relation_labels")
                relation_label_constraint = build_relation_label_constraint(tokenizer, allowed_relation_labels)
                match_key = strip(String(get(row, "match_key", "row_index:$(request.source_index)")))
                isempty(match_key) && (match_key = "row_index:$(request.source_index)")

                if match_key in existing
                    skipped_existing += 1
                    continue
                end
                if opts.max_rows > 0 && accepted >= opts.max_rows
                    break
                end

                rendered_prompt = build_rendered_prompt(tokenizer, prompt, system_prompt, opts.plain_prompt)
                rendered_prompt_tokens = encode(
                    tokenizer,
                    rendered_prompt;
                    add_special_tokens = false,
                )
                response_prefix_tokens = encode(
                    tokenizer,
                    opts.response_prefix;
                    add_special_tokens = false,
                )
                if opts.max_input_tokens > 0
                    available_prompt_tokens = opts.max_input_tokens - length(response_prefix_tokens)
                    available_prompt_tokens > 0 ||
                        error("response prefix consumes max input budget ($(opts.max_input_tokens) tokens)")
                    rendered_prompt_tokens = truncate_prompt_tokens(rendered_prompt_tokens, available_prompt_tokens)
                end
                prompt_tokens = vcat(rendered_prompt_tokens, response_prefix_tokens)
                isempty(prompt_tokens) && continue
                log_verbose(
                    opts,
                    "row=$(request.source_index) match_key=$(match_key) rendered_prompt_tokens=$(length(rendered_prompt_tokens)) full_prompt_tokens=$(length(prompt_tokens)) max_new_tokens=$(opts.max_new_tokens)",
                )

                completion_text = ""
                response_text = ""
                try
                    generation_started = time()
                    _, completion_text = generate_response_tokens(
                        model,
                        params,
                        state,
                        prompt_tokens,
                        tokenizer,
                        rng,
                        opts,
                        heuristics,
                        allowed_entity_labels,
                        allowed_relation_labels,
                        relation_label_constraint,
                    )
                    generation_elapsed = time() - generation_started
                    response_text = resolve_response_text(
                        completion_text,
                        opts.response_prefix;
                        require_json = opts.require_json,
                        max_relations_hint = opts.max_relations_hint,
                        allowed_entity_labels = allowed_entity_labels,
                        allowed_relation_labels = allowed_relation_labels,
                    )
                    isempty(strip(something(response_text, ""))) && error("teacher generation returned empty text")
                    response_text === nothing && error("teacher generation did not produce parseable JSON")

                    record = Dict{String,Any}()
                    for (key, value) in pairs(row)
                        record[String(key)] = value
                    end
                    record[opts.response_field] = response_text
                    record["teacher_model"] = opts.teacher_model
                    record["teacher_revision"] = nothing
                    record["generation_config"] = gen_cfg

                    println(out_io, JSON3.write(record))
                    flush(out_io)
                    push!(existing, match_key)
                    accepted += 1
                    log_verbose(
                        opts,
                        "accepted match_key=$(match_key) chars=$(ncodeunits(response_text)) gen_s=$(round(generation_elapsed; digits = 3)) total_row_s=$(round(time() - row_started; digits = 3))",
                    )
                catch exc
                    failed += 1
                    println(
                        err_io,
                        JSON3.write(Dict(
                            "match_key" => match_key,
                            "source_index" => request.source_index,
                            "error" => sprint(showerror, exc),
                            "prompt" => prompt,
                            "completion_preview" => preview_text(completion_text),
                            "response_preview" => preview_text(response_text),
                        )),
                    )
                    flush(err_io)
                    log_verbose(
                        opts,
                        "failed match_key=$(match_key) total_row_s=$(round(time() - row_started; digits = 3)) error=$(sprint(showerror, exc))",
                    )
                end
            end
        end
    end

    summary = Dict(
        "input" => opts.input_path,
        "output" => output_path,
        "error_log" => error_path,
        "teacher_model" => opts.teacher_model,
        "accepted" => accepted,
        "skipped_existing" => skipped_existing,
        "failed" => failed,
        "native_julia" => true,
    )
    println(JSON3.write(summary))
end

main()
