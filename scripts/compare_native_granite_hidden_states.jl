#!/usr/bin/env julia

using ArgParse
using JSON3
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "src", "Swamma.jl"))

using .Swamma
using .Swamma.HFTokenizer: load_tokenizer, has_chat_template, apply_chat_template, encode

Base.@kwdef struct CompareHiddenOptions
    input_path::String = ""
    row_index::Int = 1
    model_ref::String = "ibm-granite/granite-4.0-micro"
    prompt_field::String = "prompt"
    system_prompt_field::String = "system_prompt"
    response_prefix::String = "{\"entities\":[{\"start\":"
    max_input_tokens::Int = 256
    dtype_name::String = "float32"
    plain_prompt::Bool = false
    local_files_only::Bool = false
end

function parse_args(args)
    settings = ArgParseSettings(description = "Compare native Granite and HF last-token hidden states layer by layer.")
    @add_arg_table! settings begin
        "--input"
            help = "Input request JSONL"
            required = true
        "--row-index"
            help = "1-based request row index"
            arg_type = Int
            default = 1
        "--model"
            help = "Teacher model id/path"
            default = "ibm-granite/granite-4.0-micro"
        "--prompt-field"
            help = "Field containing rendered prompt text"
            default = "prompt"
        "--system-prompt-field"
            help = "Field containing system prompt"
            default = "system_prompt"
        "--response-prefix"
            help = "Prefix appended before generation"
            default = "{\"entities\":[{\"start\":"
        "--max-input-tokens"
            help = "Prompt truncation budget including response prefix"
            arg_type = Int
            default = 256
        "--dtype"
            help = "Weight dtype: float16 | float32"
            default = "float32"
        "--plain-prompt"
            help = "Disable chat template usage even if supported"
            action = :store_true
        "--local-files-only"
            help = "Refuse network fetches and only use local HF cache/files"
            action = :store_true
    end

    parsed = ArgParse.parse_args(args, settings)
    return CompareHiddenOptions(
        input_path = parsed["input"],
        row_index = parsed["row-index"],
        model_ref = parsed["model"],
        prompt_field = parsed["prompt-field"],
        system_prompt_field = parsed["system-prompt-field"],
        response_prefix = parsed["response-prefix"],
        max_input_tokens = parsed["max-input-tokens"],
        dtype_name = parsed["dtype"],
        plain_prompt = parsed["plain-prompt"],
        local_files_only = parsed["local-files-only"],
    )
end

function resolve_dtype(name::String)
    lowered = lowercase(strip(name))
    lowered == "float16" && return Float16
    lowered == "float32" && return Float32
    error("Unsupported --dtype=$(repr(name)). Expected float16 or float32.")
end

function read_row(path::String, row_index::Int)
    row_index > 0 || error("--row-index must be positive")
    current = 0
    row = nothing
    open(path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            current += 1
            if current == row_index
                row = JSON3.read(line)
                break
            end
        end
    end
    row !== nothing && return row
    error("Row index $(row_index) not found in $(path)")
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

function truncate_prompt_tokens(prompt_tokens::Vector{Int}, max_prompt_tokens::Int)
    max_prompt_tokens > 0 || return Int[]
    length(prompt_tokens) <= max_prompt_tokens && return prompt_tokens
    head_keep = min(length(prompt_tokens), min(128, max(32, max_prompt_tokens ÷ 4)))
    tail_keep = max_prompt_tokens - head_keep
    tail_keep <= 0 && return prompt_tokens[1:max_prompt_tokens]
    return vcat(prompt_tokens[1:head_keep], prompt_tokens[(end - tail_keep + 1):end])
end

function prepare_prompt_tokens(tokenizer, row, opts::CompareHiddenOptions)
    prompt = strip(String(get(row, opts.prompt_field, "")))
    system_prompt = strip(String(get(row, opts.system_prompt_field, "")))
    rendered_prompt = build_rendered_prompt(tokenizer, prompt, system_prompt, opts.plain_prompt)
    prompt_tokens = encode(tokenizer, rendered_prompt; add_special_tokens = false)
    prefix_tokens = encode(tokenizer, opts.response_prefix; add_special_tokens = false)
    if opts.max_input_tokens > 0
        available_prompt_tokens = opts.max_input_tokens - length(prefix_tokens)
        available_prompt_tokens > 0 || error("response prefix consumes max input budget")
        prompt_tokens = truncate_prompt_tokens(prompt_tokens, available_prompt_tokens)
    end
    return vcat(prompt_tokens, prefix_tokens)
end

function native_last_token_states(model, params, state, prompt_tokens::Vector{Int})
    tokens = reshape(prompt_tokens, :, 1)
    hidden, _ = model.TokenEmbedding(tokens, params.TokenEmbedding, state.TokenEmbedding)
    hidden .*= model.config.embedding_multiplier
    states = Vector{Vector{Float32}}()
    push!(states, vec(Float32.(hidden[:, end, 1])))

    block_state = state
    for (block_index, block) in enumerate(model.Blocks)
        hidden, _ = block(hidden, params.Blocks[block_index], block_state.Blocks[block_index])
        push!(states, vec(Float32.(hidden[:, end, 1])))
    end

    hidden, _ = model.FinalNorm(hidden, params.FinalNorm, state.FinalNorm)
    push!(states, vec(Float32.(hidden[:, end, 1])))
    return states
end

function hf_last_token_states(prompt_tokens::Vector{Int}, opts::CompareHiddenOptions)
    prompt_json = JSON3.write(Dict("input_ids" => [token - 1 for token in prompt_tokens]))
    local_files_only_py = opts.local_files_only ? "True" : "False"
    python = """
import json
import sys
from transformers import AutoModelForCausalLM
import torch

payload = json.loads(sys.stdin.read())
input_ids = torch.tensor([payload["input_ids"]], dtype=torch.long)
model = AutoModelForCausalLM.from_pretrained($(repr(opts.model_ref)), local_files_only=$(local_files_only_py), trust_remote_code=True, dtype=torch.float32)
with torch.no_grad():
    outputs = model.model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
hidden_states = outputs.hidden_states
rows = []
for hs in hidden_states:
    rows.append(hs[0, -1, :].detach().cpu().tolist())
print(json.dumps({"states": rows}))
"""
    output = read(pipeline(`python3 -c $python`, stdin = IOBuffer(prompt_json)), String)
    payload = JSON3.read(output)
    return [Float32.(collect(state)) for state in payload["states"]]
end

function cosine_similarity(a::Vector{Float32}, b::Vector{Float32})
    denom = norm(a) * norm(b)
    denom == 0f0 && return 0f0
    return dot(a, b) / denom
end

function main()
    opts = parse_args(ARGS)
    row = read_row(opts.input_path, opts.row_index)
    tokenizer = load_tokenizer(opts.model_ref; local_files_only = opts.local_files_only)
    prompt_tokens = prepare_prompt_tokens(tokenizer, row, opts)
    dtype = resolve_dtype(opts.dtype_name)
    model, params, state = load_granite_model(opts.model_ref; rng = Random.MersenneTwister(1), dtype = dtype, local_files_only = opts.local_files_only)

    native_states = native_last_token_states(model, params, state, prompt_tokens)
    hf_states = hf_last_token_states(prompt_tokens, opts)

    println("Prompt token count: $(length(prompt_tokens))")
    println("Stage | Cosine | L2 | MaxAbs")
    println("------|--------|----|-------")
    for index in 1:min(length(native_states), length(hf_states))
        native = native_states[index]
        hf = hf_states[index]
        l2 = norm(native .- hf)
        maxabs = maximum(abs.(native .- hf))
        label = index == 1 ? "embed" : (index == length(native_states) ? "final_norm" : "block_$(index - 1)")
        println("$(label) | $(round(cosine_similarity(native, hf); digits = 6)) | $(round(l2; digits = 6)) | $(round(maxabs; digits = 6))")
    end
end

main()
