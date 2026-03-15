#!/usr/bin/env julia
"""
Canonical LLaDA training entrypoint.

What this script standardizes:
- Swamma LLaDA model/training config loading from one TOML file
- HuggingFace tokenizer path (via HFTokenizer wrapper)
- deterministic text -> token -> fixed-length chunk pipeline
- periodic eval + checkpointing + resume

Usage:
  julia --project=. scripts/train_llada_canonical.jl \
    --config configs/train_base.toml \
    --train-path data/train.txt \
    --val-path data/val.txt \
    --tokenizer-model ibm-granite/granite-4.0-micro
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using ArgParse
using Dates
using JSON3
using Optimisers
using Printf
using Random
using Serialization
using TOML

include(joinpath(dirname(@__DIR__), "src", "Swamma.jl"))
using .Swamma
using .Swamma.HFTokenizer: load_tokenizer, encode, decode, get_vocab_size

mutable struct BatchLoader
    batches::Vector{Matrix{Int}}
    shuffle::Bool
    rng::Random.AbstractRNG
    order::Vector{Int}
    cursor::Int
end

function BatchLoader(
    batches::Vector{Matrix{Int}};
    shuffle::Bool = true,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    order = collect(1:length(batches))
    if shuffle
        Random.shuffle!(rng, order)
    end
    return BatchLoader(batches, shuffle, rng, order, 1)
end

Base.length(loader::BatchLoader) = length(loader.batches)

function Base.iterate(loader::BatchLoader, state = nothing)
    if loader.cursor > length(loader.order)
        loader.cursor = 1
        if loader.shuffle
            Random.shuffle!(loader.rng, loader.order)
        end
        return nothing
    end

    idx = loader.order[loader.cursor]
    loader.cursor += 1
    return loader.batches[idx], nothing
end

function reset!(loader::BatchLoader)
    loader.cursor = 1
    if loader.shuffle
        Random.shuffle!(loader.rng, loader.order)
    end
end

function parse_commandline()
    s = ArgParseSettings(description = "Canonical LLaDA training with HF tokenization")

    @add_arg_table! s begin
        "--config"
            help = "Path to LLaDA config TOML"
            arg_type = String
            default = "configs/train_base.toml"
        "--train-path"
            help = "Training corpus path (.txt or .jsonl)"
            arg_type = String
            default = ""
        "--val-path"
            help = "Validation corpus path (.txt or .jsonl)"
            arg_type = String
            default = ""
        "--tokenizer-model"
            help = "HF tokenizer model id/path"
            arg_type = String
            default = "ibm-granite/granite-4.0-micro"
        "--checkpoint-dir"
            help = "Checkpoint/output directory"
            arg_type = String
            default = "checkpoints/llada_canonical"
        "--resume"
            help = "Checkpoint to resume from"
            arg_type = String
            default = ""
        "--text-field"
            help = "JSONL primary text field"
            arg_type = String
            default = "text"
        "--alt-text-field"
            help = "JSONL fallback text field"
            arg_type = String
            default = "content"
        "--seed"
            help = "Random seed"
            arg_type = Int
            default = 42
        "--val-split"
            help = "Validation split used when --val-path is missing"
            arg_type = Float64
            default = 0.1
        "--seq-len"
            help = "Optional override for max_sequence_length (<=0 keeps config)"
            arg_type = Int
            default = -1
        "--stride"
            help = "Chunk stride; <=0 means stride=seq_len"
            arg_type = Int
            default = -1
        "--batch-size"
            help = "Optional training batch-size override (<=0 keeps config)"
            arg_type = Int
            default = -1
        "--total-steps"
            help = "Optional total-steps override (<=0 keeps config)"
            arg_type = Int
            default = -1
        "--eval-every"
            help = "Optional eval-every override (<=0 keeps config)"
            arg_type = Int
            default = -1
        "--save-every"
            help = "Optional save-every override (<=0 keeps config)"
            arg_type = Int
            default = -1
        "--log-every"
            help = "Optional log-every override (<=0 keeps config)"
            arg_type = Int
            default = -1
        "--learning-rate"
            help = "Optional learning-rate override (<=0 keeps config)"
            arg_type = Float64
            default = -1.0
        "--max-train-texts"
            help = "Optional cap on loaded train texts (<=0 means unlimited)"
            arg_type = Int
            default = -1
        "--max-val-texts"
            help = "Optional cap on loaded val texts (<=0 means unlimited)"
            arg_type = Int
            default = -1
        "--sample-steps"
            help = "Generation denoising steps for end-of-run sample"
            arg_type = Int
            default = 32
    end

    return parse_args(s)
end

function parse_data_section(path::String)
    cfg = TOML.parsefile(path)
    return get(cfg, "data", Dict{String, Any}())
end

function load_texts(path::String; text_field::String = "text", alt_text_field::String = "content")
    texts = String[]
    if isempty(path)
        return texts
    end
    isfile(path) || error("Data path does not exist: $path")

    if endswith(lowercase(path), ".jsonl")
        for line in eachline(path)
            line = strip(line)
            isempty(line) && continue

            pushed = false
            try
                obj = JSON3.read(line)
                if haskey(obj, Symbol(text_field))
                    value = strip(String(obj[Symbol(text_field)]))
                    if !isempty(value)
                        push!(texts, value)
                        pushed = true
                    end
                elseif haskey(obj, Symbol(alt_text_field))
                    value = strip(String(obj[Symbol(alt_text_field)]))
                    if !isempty(value)
                        push!(texts, value)
                        pushed = true
                    end
                end
            catch
                # Fallback to raw line below.
            end

            if !pushed
                push!(texts, line)
            end
        end
        return texts
    end

    # .txt and other plain-text files: prefer paragraph chunks.
    raw = read(path, String)
    paragraphs = [strip(String(p)) for p in split(raw, r"\n\s*\n+")]
    texts = filter(t -> !isempty(t), paragraphs)

    if isempty(texts)
        # Fallback: one line per sample.
        for line in split(raw, '\n')
            line = strip(line)
            isempty(line) || push!(texts, line)
        end
    end

    return texts
end

function cap_texts(texts::AbstractVector{<:AbstractString}, max_count::Int)
    if max_count > 0 && length(texts) > max_count
        return String.(texts[1:max_count])
    end
    return String.(texts)
end

function text_to_chunks(
    tokenizer,
    texts::AbstractVector{<:AbstractString},
    seq_len::Int;
    stride::Int = seq_len,
)
    chunks = Vector{Vector{Int}}()
    local_stride = stride <= 0 ? seq_len : stride

    for text in texts
        ids = encode(tokenizer, text; add_special_tokens = true)
        if length(ids) < seq_len
            continue
        end

        last_start = length(ids) - seq_len + 1
        for start in 1:local_stride:last_start
            push!(chunks, ids[start:(start + seq_len - 1)])
        end
    end

    return chunks
end

function make_batches(chunks::Vector{Vector{Int}}, batch_size::Int)
    batches = Matrix{Int}[]
    if isempty(chunks)
        return batches
    end

    for i in 1:batch_size:length(chunks)
        j = min(i + batch_size - 1, length(chunks))
        if (j - i + 1) < batch_size
            break
        end
        push!(batches, hcat(chunks[i:j]...))
    end

    return batches
end

function count_parameters(params)
    total = 0
    for (_, v) in pairs(params)
        if v isa AbstractArray
            total += length(v)
        elseif v isa NamedTuple
            total += count_parameters(v)
        end
    end
    return total
end

function maybe_override(base::Int, override::Int)
    return override > 0 ? override : base
end

function maybe_override(base::Float32, override::Float64)
    return override > 0 ? Float32(override) : base
end

function main()
    args = parse_commandline()
    rng = Random.MersenneTwister(args["seed"])

    println("=" ^ 72)
    println("Canonical Swamma LLaDA Training")
    println("=" ^ 72)
    println("Config: $(args["config"])")
    println("Checkpoint dir: $(args["checkpoint-dir"])")
    println("Seed: $(args["seed"])")
    println()

    model_config = load_config(args["config"])
    train_config = load_training_config(args["config"])
    data_cfg = parse_data_section(args["config"])

    train_path = !isempty(args["train-path"]) ? args["train-path"] : get(data_cfg, "train_path", "")
    val_path = !isempty(args["val-path"]) ? args["val-path"] : get(data_cfg, "val_path", "")
    tokenizer_model = args["tokenizer-model"]
    if tokenizer_model == "ibm-granite/granite-4.0-micro"
        tokenizer_model = get(data_cfg, "tokenizer_model", tokenizer_model)
    end

    if isempty(train_path)
        error("No training path provided. Set --train-path or [data].train_path in the config.")
    end

    seq_len = maybe_override(model_config.max_sequence_length, args["seq-len"])
    stride = args["stride"] > 0 ? args["stride"] : seq_len
    batch_size = maybe_override(train_config.batch_size, args["batch-size"])
    total_steps = maybe_override(train_config.total_steps, args["total-steps"])
    eval_every = maybe_override(train_config.eval_every, args["eval-every"])
    save_every = maybe_override(train_config.save_every, args["save-every"])
    log_every = maybe_override(train_config.log_every, args["log-every"])
    learning_rate = maybe_override(train_config.learning_rate, args["learning-rate"])

    println("[1/6] Loading tokenizer: $tokenizer_model")
    tokenizer = try
        load_tokenizer(tokenizer_model)
    catch err
        msg = sprint(showerror, err)
        if occursin("transformers", lowercase(msg))
            error("Could not load HF tokenizer because Python package `transformers` is missing in the PyCall environment.")
        end
        rethrow(err)
    end
    tok_vocab_size = get_vocab_size(tokenizer)
    println("  tokenizer vocab size: $tok_vocab_size")
    println()

    println("[2/6] Loading text corpora")
    train_texts = load_texts(train_path; text_field = args["text-field"], alt_text_field = args["alt-text-field"])
    val_texts = isempty(val_path) ? String[] :
        load_texts(val_path; text_field = args["text-field"], alt_text_field = args["alt-text-field"])
    train_texts = cap_texts(train_texts, args["max-train-texts"])
    val_texts = cap_texts(val_texts, args["max-val-texts"])

    if isempty(train_texts)
        error("Training corpus is empty after loading: $train_path")
    end

    # If val is missing, split from train deterministically.
    if isempty(val_texts)
        val_count = clamp(round(Int, length(train_texts) * args["val-split"]), 1, max(1, length(train_texts) - 1))
        Random.shuffle!(rng, train_texts)
        val_texts = train_texts[end - val_count + 1:end]
        train_texts = train_texts[1:end - val_count]
        println("  val split generated from train: $(length(train_texts)) train / $(length(val_texts)) val texts")
    else
        println("  loaded: $(length(train_texts)) train / $(length(val_texts)) val texts")
    end
    println()

    println("[3/6] Tokenizing and chunking")
    train_chunks = text_to_chunks(tokenizer, train_texts, seq_len; stride = stride)
    val_chunks = text_to_chunks(tokenizer, val_texts, seq_len; stride = stride)

    if isempty(train_chunks)
        error("No train chunks produced. Reduce --seq-len or provide larger text corpus.")
    end
    if isempty(val_chunks)
        error("No val chunks produced. Reduce --seq-len or provide larger validation corpus.")
    end

    train_batches = make_batches(train_chunks, batch_size)
    val_batches = make_batches(val_chunks, batch_size)

    if isempty(train_batches)
        error("No full training batches produced. Lower batch size or add more data.")
    end
    if isempty(val_batches)
        error("No full validation batches produced. Lower batch size or add more validation data.")
    end

    println("  chunks: $(length(train_chunks)) train / $(length(val_chunks)) val")
    println("  batches: $(length(train_batches)) train / $(length(val_batches)) val")
    println()

    resolved_vocab_size = max(model_config.vocab_size, tok_vocab_size)
    resolved_window = min(model_config.window_size, seq_len)

    model_config = LLaDAConfig(
        vocab_size = resolved_vocab_size,
        max_sequence_length = seq_len,
        embedding_dimension = model_config.embedding_dimension,
        number_of_heads = model_config.number_of_heads,
        number_of_layers = model_config.number_of_layers,
        time_dimension = model_config.time_dimension,
        state_dimension = model_config.state_dimension,
        window_size = resolved_window,
        min_frequency = model_config.min_frequency,
        max_frequency = model_config.max_frequency,
        default_time_step = model_config.default_time_step,
        default_num_steps = model_config.default_num_steps,
        default_temperature = model_config.default_temperature,
        mask_schedule = train_config.mask_schedule,
        prime_subtoken_length = model_config.prime_subtoken_length,
        prime_subtoken_base = model_config.prime_subtoken_base,
    )

    run_train_config = TrainingConfig(
        batch_size = batch_size,
        learning_rate = learning_rate,
        min_learning_rate = train_config.min_learning_rate,
        warmup_steps = train_config.warmup_steps,
        total_steps = total_steps,
        eval_every = eval_every,
        log_every = log_every,
        save_every = save_every,
        mask_schedule = train_config.mask_schedule,
        gradient_clip = train_config.gradient_clip,
    )

    println("[4/6] Building model")
    model = LLaDAModel(model_config)
    optimizer = Optimisers.Adam(run_train_config.learning_rate)
    state = create_train_state(model, optimizer; rng = rng)

    if !isempty(args["resume"])
        println("  Resuming from: $(args["resume"])")
        checkpoint = load_checkpoint(args["resume"])
        checkpoint.params === nothing && error("Checkpoint missing params")
        checkpoint.state === nothing && error("Checkpoint missing state")
        state.params = checkpoint.params
        state.state = checkpoint.state
        if checkpoint.opt_state !== nothing
            state.optimizer_state = checkpoint.opt_state
        end
        state.step = checkpoint.epoch
        state.best_loss = checkpoint.loss === nothing ? state.best_loss : Float32(checkpoint.loss)
        println("  resumed at step $(state.step), best_loss=$(state.best_loss)")
    end

    param_count = count_parameters(state.params)
    println("  parameters: $(round(param_count / 1e6, digits = 2))M")
    println()

    train_loader = BatchLoader(train_batches; shuffle = true, rng = rng)
    # Keep validation data as a plain vector so each evaluation pass starts fresh.
    val_loader = val_batches

    println("[5/6] Preparing checkpoints and metadata")
    ckpt_dir = args["checkpoint-dir"]
    mkpath(ckpt_dir)

    open(joinpath(ckpt_dir, "run_config.toml"), "w") do io
        save_config(model_config, io)
        println(io)
        println(io, "[training]")
        println(io, "batch_size = ", run_train_config.batch_size)
        println(io, "learning_rate = ", run_train_config.learning_rate)
        println(io, "min_learning_rate = ", run_train_config.min_learning_rate)
        println(io, "warmup_steps = ", run_train_config.warmup_steps)
        println(io, "total_steps = ", run_train_config.total_steps)
        println(io, "mask_schedule = \"", run_train_config.mask_schedule, "\"")
        println(io, "gradient_clip = ", run_train_config.gradient_clip)
    end

    metadata = Dict(
        "timestamp_utc" => string(now(UTC)),
        "tokenizer_model" => tokenizer_model,
        "tokenizer_vocab_size" => tok_vocab_size,
        "resolved_vocab_size" => model_config.vocab_size,
        "sequence_length" => seq_len,
        "stride" => stride,
        "batch_size" => batch_size,
        "train_texts" => length(train_texts),
        "val_texts" => length(val_texts),
        "train_chunks" => length(train_chunks),
        "val_chunks" => length(val_chunks),
    )
    open(joinpath(ckpt_dir, "run_metadata.json"), "w") do io
        JSON3.pretty(io, metadata)
    end

    serialize(joinpath(ckpt_dir, "tokenizer_ref.jls"), Dict(
        :tokenizer_model => tokenizer_model,
        :vocab_size => tok_vocab_size,
    ))
    println("  saved metadata to $ckpt_dir")
    println()

    callbacks = Dict{Symbol, Function}(
        :on_log => (ts, avg_loss, lr) -> begin
            ppl = exp(clamp(Float64(avg_loss), -20.0, 20.0))
            println(@sprintf("  train ppl≈%.3f", ppl))
            flush(stdout)
        end,
        :on_eval => (ts, val_loss) -> begin
            ppl = exp(clamp(Float64(val_loss), -20.0, 20.0))
            println(@sprintf("  val ppl≈%.3f", ppl))
            flush(stdout)
        end,
        :on_best => (ts) -> begin
            path = joinpath(ckpt_dir, "best.jls")
            save_checkpoint(path;
                params = ts.params,
                state = ts.state,
                opt_state = ts.optimizer_state,
                epoch = ts.step,
                loss = ts.best_loss
            )
            println("  saved new best checkpoint: $path")
            flush(stdout)
        end,
        :on_save => (ts) -> begin
            path = joinpath(ckpt_dir, "step_$(ts.step).jls")
            save_checkpoint(path;
                params = ts.params,
                state = ts.state,
                opt_state = ts.optimizer_state,
                epoch = ts.step,
                loss = ts.best_loss
            )
            cp(path, joinpath(ckpt_dir, "latest.jls"), force = true)
            println("  saved checkpoint: $path")
            flush(stdout)
        end,
    )

    println("[6/6] Training")
    println("  steps=$(run_train_config.total_steps), lr=$(run_train_config.learning_rate), warmup=$(run_train_config.warmup_steps)")
    println("  eval_every=$(run_train_config.eval_every), save_every=$(run_train_config.save_every), log_every=$(run_train_config.log_every)")
    println()

    train!(
        model,
        state,
        train_loader,
        run_train_config;
        val_data = val_loader,
        callbacks = callbacks,
        rng = rng,
    )

    final_path = joinpath(ckpt_dir, "final.jls")
    save_checkpoint(final_path;
        params = state.params,
        state = state.state,
        opt_state = state.optimizer_state,
        epoch = state.step,
        loss = state.best_loss
    )
    cp(final_path, joinpath(ckpt_dir, "latest.jls"), force = true)
    println("\nFinal checkpoint: $final_path")

    println("\nGeneration sample:")
    sample_ids = generate(
        model,
        state.params,
        state.state,
        seq_len;
        num_steps = args["sample-steps"],
        batch_size = 1,
        rng = rng,
    )
    text = decode(tokenizer, vec(sample_ids))
    snippet = length(text) > 300 ? text[1:300] * "..." : text
    println(snippet)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
