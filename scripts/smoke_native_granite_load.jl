#!/usr/bin/env julia

using ArgParse
using Random

include(joinpath(@__DIR__, "..", "src", "Swamma.jl"))

using .Swamma
using .Swamma.HFTokenizer: load_tokenizer, get_vocab_size, get_pad_token_id
using .Swamma.HFTokenizer: has_chat_template, apply_chat_template, apply_chat_template_tokens

function parse_args()
    settings = ArgParseSettings(description = "Smoke test the native Granite tokenizer + Lux importer path.")
    @add_arg_table! settings begin
        "--model"
            help = "Hugging Face model id or local snapshot directory"
            arg_type = String
            default = "ibm-granite/granite-4.0-micro"
        "--dtype"
            help = "Weight dtype: float16 | float32"
            arg_type = String
            default = "float16"
        "--local-files-only"
            help = "Refuse network fetches and only use local HF cache/files"
            action = :store_true
        "--run-forward"
            help = "After loading weights, run one forward pass on the rendered prompt"
            action = :store_true
        "--run-cached-check"
            help = "Compare cached next-token logits against full-prefix recompute on one appended token"
            action = :store_true
        "--max-prompt-tokens"
            help = "Trim rendered prompt to this many tokens for the forward smoke"
            arg_type = Int
            default = 64
    end
    return ArgParse.parse_args(settings)
end

function resolve_dtype(name::String)
    lowered = lowercase(strip(name))
    lowered == "float16" && return Float16
    lowered == "float32" && return Float32
    error("Unsupported --dtype=$(repr(name)). Expected float16 or float32.")
end

function main()
    args = parse_args()
    dtype = resolve_dtype(args["dtype"])
    model_ref = args["model"]
    local_files_only = args["local-files-only"]

    prompt_tokens = Int[]
    println("Loading tokenizer: $(model_ref)")
    try
        tokenizer = load_tokenizer(model_ref; local_files_only = local_files_only)
        println("  vocab_size=$(get_vocab_size(tokenizer)) pad=$(get_pad_token_id(tokenizer)) eos=$(something(tokenizer.eos_token_id, -1) + 1)")
        println("  has_chat_template=$(has_chat_template(tokenizer))")

        messages = [
            (role = "system", content = "You extract entities and relations."),
            (role = "user", content = "Barack Obama was born in Hawaii."),
        ]
        prompt = apply_chat_template(tokenizer, messages; add_generation_prompt = true)
        prompt_tokens = apply_chat_template_tokens(
            tokenizer,
            messages;
            add_generation_prompt = true,
            max_length = args["max-prompt-tokens"],
        )

        println("Rendered prompt:")
        println(prompt)
        println("Prompt token count: $(length(prompt_tokens))")
    catch exc
        println("  tokenizer smoke skipped: $(typeof(exc))")
        println("  tokenizer error: $(sprint(showerror, exc))")
    end

    println("\nLoading native Granite model...")
    rng = Random.MersenneTwister(1)
    model, params, state = load_granite_model(
        model_ref;
        rng = rng,
        dtype = dtype,
        local_files_only = local_files_only,
    )
    println("  loaded config: vocab=$(model.config.vocab_size) dim=$(model.config.embedding_dimension) layers=$(model.config.number_of_layers) heads=$(model.config.number_of_heads)/$(model.config.number_of_kv_heads)")
    println("  embedding_multiplier=$(model.config.embedding_multiplier) logits_scaling=$(model.config.logits_scaling)")
    println("  token_embedding=$(size(params.TokenEmbedding.weight)) output_head=$(size(params.OutputHead.weight))")

    if args["run-forward"]
        isempty(prompt_tokens) && error("Cannot run forward smoke because tokenizer prompt tokens were not produced.")
        token_matrix = reshape(prompt_tokens, :, 1)
        println("\nRunning native forward on $(size(token_matrix, 1)) tokens...")
        logits, _ = model(token_matrix, params, state)
        println("  logits size=$(size(logits))")
        next_logits, _ = next_token_logits(model, token_matrix, params, state)
        println("  next_token_logits size=$(size(next_logits))")
    end

    if args["run-cached-check"]
        isempty(prompt_tokens) && error("Cannot run cached check because tokenizer prompt tokens were not produced.")
        token_matrix = reshape(prompt_tokens, :, 1)
        appended_token = reshape(Int[prompt_tokens[end]], :, 1)

        println("\nRunning cached decode check...")
        cache = init_decoder_cache(model, 1)
        _, cached_state, filled_cache = forward_with_cache(model, token_matrix, params, state, cache)
        cached_logits, _, advanced_cache =
            next_token_logits_cached(model, appended_token, params, cached_state, filled_cache)
        full_logits, _ = model(vcat(token_matrix, appended_token), params, state)
        max_delta = maximum(abs.(Float32.(cached_logits[:, 1]) .- Float32.(full_logits[:, end, 1])))
        println("  cache sequence length: $(cache_sequence_length(advanced_cache))")
        println("  cached_logits size=$(size(cached_logits))")
        println("  max abs delta vs full recompute=$(max_delta)")
    end
end

main()
