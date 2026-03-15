#!/usr/bin/env julia

using Pkg
Pkg.activate(dirname(@__DIR__))

using ArgParse
using TOML
using Random
using Lux
using CUDA
using JSON3
using Statistics: mean
using Printf
using Dates

include(joinpath(dirname(@__DIR__), "src", "Swamma.jl"))
using .Swamma
using .Swamma.HFTokenizer: load_tokenizer, encode
include(joinpath(@__DIR__, "long_context_models.jl"))
using .LongContextModels

function parse_cli_args()
    s = ArgParseSettings(description = "Long-context evaluation (Swamma vs Transformer baseline)")
    @add_arg_table! s begin
        "--config"
            help = "Path to evaluation TOML config"
            arg_type = String
            default = "configs/swamma_vs_transformer/eval_long_context.toml"
        "--output"
            help = "Output CSV path"
            arg_type = String
            default = "benchmarks/long_context_eval.csv"
        "--swamma-checkpoint"
            help = "Optional checkpoint for Swamma model"
            arg_type = String
            default = ""
        "--transformer-checkpoint"
            help = "Optional checkpoint for Transformer baseline"
            arg_type = String
            default = ""
        "--seed"
            help = "Random seed override (negative keeps config)"
            arg_type = Int
            default = -1
        "--device"
            help = "Execution device: cpu | gpu"
            arg_type = String
            default = "gpu"
    end
    return ArgParse.parse_args(s)
end

function to_device(x, device::Symbol)
    if x isa NamedTuple
        return NamedTuple{keys(x)}(Tuple(to_device(v, device) for v in values(x)))
    elseif x isa Tuple
        return Tuple(to_device(v, device) for v in x)
    elseif x isa AbstractArray
        return device == :gpu ? CUDA.CuArray(x) : x
    end
    return x
end

function resolve_device(device_arg::String)
    device = Symbol(lowercase(strip(device_arg)))
    device in (:cpu, :gpu) || error("Invalid --device value '$device_arg'. Use cpu or gpu.")
    if device == :gpu && !CUDA.functional()
        println("CUDA not functional. Falling back to CPU.")
        return :cpu
    end
    return device
end

function parse_config(path::String)
    cfg = TOML.parsefile(path)
    common = get(cfg, "common", Dict{String, Any}())
    sweep = get(cfg, "sweep", Dict{String, Any}())
    eval_cfg = get(cfg, "eval", Dict{String, Any}())

    arch_raw = get(sweep, "architectures", ["swamma", "transformer"])
    archs = Symbol.(String.(arch_raw))

    return (
        common = (
            vocab_size = Int(get(common, "vocab_size", 32000)),
            embedding_dimension = Int(get(common, "embedding_dimension", 512)),
            number_of_heads = Int(get(common, "number_of_heads", 8)),
            number_of_layers = Int(get(common, "number_of_layers", 12)),
            time_dimension = Int(get(common, "time_dimension", 128)),
            state_dimension = Int(get(common, "state_dimension", get(common, "embedding_dimension", 512))),
            window_size = Int(get(common, "window_size", 128)),
            min_frequency = Float32(get(common, "min_frequency", 0.1)),
            max_frequency = Float32(get(common, "max_frequency", 10.0)),
            default_time_step = Float32(get(common, "default_time_step", 0.1)),
            prime_subtoken_length = Int(get(common, "prime_subtoken_length", 4)),
            prime_subtoken_base = Int(get(common, "prime_subtoken_base", 16)),
            batch_size = Int(get(common, "batch_size", 2)),
        ),
        sweep = (
            context_lengths = Int.(get(sweep, "context_lengths", [1024, 2048, 4096, 8192, 16384])),
            architectures = archs,
        ),
        eval = (
            seed = Int(get(eval_cfg, "seed", 42)),
            mask_ratio = Float32(get(eval_cfg, "mask_ratio", 0.5)),
            eval_batches = Int(get(eval_cfg, "eval_batches", 16)),
            run_text_eval = Bool(get(eval_cfg, "run_text_eval", false)),
            text_path = String(get(eval_cfg, "text_path", "")),
            tokenizer_model = String(get(eval_cfg, "tokenizer_model", "ibm-granite/granite-4.0-micro")),
            run_needle_eval = Bool(get(eval_cfg, "run_needle_eval", true)),
            needle_batches = Int(get(eval_cfg, "needle_batches", 64)),
        ),
    )
end

function maybe_load_checkpoint!(params, state, checkpoint_path::String)
    if isempty(checkpoint_path)
        return params, state, :random_init, ""
    end

    try
        ckpt = load_checkpoint(checkpoint_path)
        ckpt.params === nothing && return params, state, :random_init, "missing_params"
        ckpt.state === nothing && return params, state, :random_init, "missing_state"
        return ckpt.params, ckpt.state, :checkpoint, ""
    catch err
        return params, state, :random_init, sprint(showerror, err)
    end
end

function load_texts(path::String)
    isempty(path) && return String[]
    isfile(path) || error("Text path does not exist: $path")

    if endswith(lowercase(path), ".jsonl")
        texts = String[]
        for line in eachline(path)
            line = strip(line)
            isempty(line) && continue
            obj = try
                JSON3.read(line)
            catch
                nothing
            end
            if obj === nothing
                push!(texts, line)
            elseif haskey(obj, :text)
                push!(texts, strip(String(obj[:text])))
            elseif haskey(obj, :content)
                push!(texts, strip(String(obj[:content])))
            end
        end
        return filter(t -> !isempty(t), texts)
    end

    raw = read(path, String)
    paras = [strip(String(p)) for p in split(raw, r"\n\s*\n+")]
    texts = filter(t -> !isempty(t), paras)
    if isempty(texts)
        texts = filter(t -> !isempty(t), strip.(split(raw, '\n')))
    end
    return texts
end

function text_to_batches(tokenizer, texts::Vector{String}, seq_len::Int, batch_size::Int, max_batches::Int)
    chunks = Vector{Vector{Int}}()
    for text in texts
        ids = encode(tokenizer, text; add_special_tokens = true)
        if length(ids) < seq_len
            continue
        end
        last_start = length(ids) - seq_len + 1
        stride = seq_len
        for start in 1:stride:last_start
            push!(chunks, ids[start:(start + seq_len - 1)])
            if length(chunks) >= max_batches * batch_size
                break
            end
        end
        if length(chunks) >= max_batches * batch_size
            break
        end
    end

    batches = Matrix{Int}[]
    n_full = fld(length(chunks), batch_size)
    n_keep = min(n_full, max_batches)
    for i in 1:n_keep
        start = (i - 1) * batch_size + 1
        stop = i * batch_size
        batch_chunks = chunks[start:stop]
        batch = hcat(batch_chunks...)
        push!(batches, batch)
    end
    return batches
end

function evaluate_text(model, params, state, batches, mask_ratio::Float32, rng)
    isempty(batches) && return (
        loss = Float32(NaN),
        ppl = Float32(NaN),
        acc = Float32(NaN),
        acc_early = Float32(NaN),
        acc_middle = Float32(NaN),
        acc_late = Float32(NaN),
    )

    total_weight = 0
    weighted_loss = 0.0
    weighted_acc = 0.0
    weighted_acc_early = 0.0
    weighted_acc_middle = 0.0
    weighted_acc_late = 0.0

    local_state = Lux.testmode(state)

    for batch in batches
        logits, local_state, token_mask, _ = masked_forward(model, params, local_state, batch, mask_ratio, rng)
        metrics = masked_metrics(logits, batch, token_mask)
        w = max(metrics.masked_positions, 1)

        total_weight += w
        weighted_loss += Float64(metrics.loss) * w
        weighted_acc += Float64(metrics.acc) * w
        weighted_acc_early += Float64(metrics.acc_early) * w
        weighted_acc_middle += Float64(metrics.acc_middle) * w
        weighted_acc_late += Float64(metrics.acc_late) * w
    end

    if total_weight == 0
        return (
            loss = Float32(NaN),
            ppl = Float32(NaN),
            acc = Float32(NaN),
            acc_early = Float32(NaN),
            acc_middle = Float32(NaN),
            acc_late = Float32(NaN),
        )
    end

    loss = Float32(weighted_loss / total_weight)
    ppl = Float32(exp(clamp(Float64(loss), -20.0, 20.0)))
    return (
        loss = loss,
        ppl = ppl,
        acc = Float32(weighted_acc / total_weight),
        acc_early = Float32(weighted_acc_early / total_weight),
        acc_middle = Float32(weighted_acc_middle / total_weight),
        acc_late = Float32(weighted_acc_late / total_weight),
    )
end

function evaluate_needle(model, params, state, needle_batches::Int, seq_len::Int, batch_size::Int, rng)
    needle_batches <= 0 && return Float32(NaN)

    local_state = Lux.testmode(state)
    accs = Float32[]
    for _ in 1:needle_batches
        acc, local_state = needle_eval_step(model, params, local_state, seq_len, batch_size, rng)
        push!(accs, acc)
    end
    return mean(accs)
end

function effective_device_for_point(requested_device::Symbol, arch::Symbol, context_length::Int)
    _ = arch
    _ = context_length
    return requested_device, ""
end

function write_csv(rows, path::String)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "timestamp_utc,architecture,context_length,init_mode,text_loss,text_ppl,text_acc,text_acc_early,text_acc_middle,text_acc_late,needle_acc,device,run_note,checkpoint_error")
        for r in rows
            println(io,
                r.timestamp_utc, ",",
                r.architecture, ",",
                r.context_length, ",",
                r.init_mode, ",",
                @sprintf("%.6f", r.text_loss), ",",
                @sprintf("%.6f", r.text_ppl), ",",
                @sprintf("%.6f", r.text_acc), ",",
                @sprintf("%.6f", r.text_acc_early), ",",
                @sprintf("%.6f", r.text_acc_middle), ",",
                @sprintf("%.6f", r.text_acc_late), ",",
                @sprintf("%.6f", r.needle_acc), ",",
                r.device, ",",
                replace(r.run_note, ',' => ';'), ",",
                replace(r.checkpoint_error, ',' => ';')
            )
        end
    end
end

function main()
    args = parse_cli_args()
    cfg = parse_config(args["config"])
    device = resolve_device(args["device"])

    seed = args["seed"] >= 0 ? args["seed"] : cfg.eval.seed
    rng = Random.MersenneTwister(seed)

    println("="^72)
    println("Long-Context Evaluation")
    println("="^72)
    println("Config: $(args["config"])")
    println("Architectures: $(join(string.(cfg.sweep.architectures), ", "))")
    println("Contexts: $(join(string.(cfg.sweep.context_lengths), ", "))")
    println("Text eval: $(cfg.eval.run_text_eval)")
    println("Needle eval: $(cfg.eval.run_needle_eval)")
    println("Device: $device")
    if device == :gpu
        println("CUDA device: $(CUDA.name(CUDA.device()))")
    end
    println("Seed: $seed")
    println()

    tokenizer = nothing
    texts = String[]
    if cfg.eval.run_text_eval
        if isempty(cfg.eval.text_path)
            error("run_text_eval=true but eval.text_path is empty")
        end

        texts = load_texts(cfg.eval.text_path)
        isempty(texts) && error("No texts loaded from $(cfg.eval.text_path)")

        try
            tokenizer = load_tokenizer(cfg.eval.tokenizer_model)
        catch err
            msg = sprint(showerror, err)
            if occursin("transformers", lowercase(msg))
                error("Could not load tokenizer because Python package `transformers` is missing in the PyCall environment.")
            end
            rethrow(err)
        end
    end

    rows = NamedTuple[]

    for arch in cfg.sweep.architectures
        ckpt_path = arch == :swamma ? args["swamma-checkpoint"] : args["transformer-checkpoint"]
        println("[Architecture] $arch")

        for context_length in cfg.sweep.context_lengths
            run_device, run_note = effective_device_for_point(device, arch, context_length)
            if run_device == :skip
                row = (
                    timestamp_utc = string(now(UTC)),
                    architecture = String(arch),
                    context_length = context_length,
                    init_mode = "skipped",
                    text_loss = Float32(NaN),
                    text_ppl = Float32(NaN),
                    text_acc = Float32(NaN),
                    text_acc_early = Float32(NaN),
                    text_acc_middle = Float32(NaN),
                    text_acc_late = Float32(NaN),
                    needle_acc = Float32(NaN),
                    device = "skipped",
                    run_note = run_note,
                    checkpoint_error = "",
                )
                push!(rows, row)
                println("  N=$(lpad(context_length, 6)) | skipped | note: $run_note")
                continue
            end

            spec = ModelSpec(
                architecture = arch,
                vocab_size = cfg.common.vocab_size,
                max_sequence_length = context_length,
                embedding_dimension = cfg.common.embedding_dimension,
                number_of_heads = cfg.common.number_of_heads,
                number_of_layers = cfg.common.number_of_layers,
                time_dimension = cfg.common.time_dimension,
                state_dimension = cfg.common.state_dimension,
                window_size = min(cfg.common.window_size, context_length),
                min_frequency = cfg.common.min_frequency,
                max_frequency = cfg.common.max_frequency,
                default_time_step = cfg.common.default_time_step,
                prime_subtoken_length = cfg.common.prime_subtoken_length,
                prime_subtoken_base = cfg.common.prime_subtoken_base,
            )

            model = build_model(spec)
            params, state = Lux.setup(rng, model)
            params, state, init_mode, ckpt_error = maybe_load_checkpoint!(params, state, ckpt_path)
            params = to_device(params, run_device)
            state = to_device(state, run_device)

            text_metrics = (
                loss = Float32(NaN),
                ppl = Float32(NaN),
                acc = Float32(NaN),
                acc_early = Float32(NaN),
                acc_middle = Float32(NaN),
                acc_late = Float32(NaN),
            )
            if cfg.eval.run_text_eval
                batches = text_to_batches(tokenizer, texts, context_length, cfg.common.batch_size, cfg.eval.eval_batches)
                text_metrics = evaluate_text(model, params, state, batches, cfg.eval.mask_ratio, rng)
            end

            needle_acc = Float32(NaN)
            if cfg.eval.run_needle_eval
                needle_acc = evaluate_needle(
                    model, params, state,
                    cfg.eval.needle_batches,
                    context_length,
                    cfg.common.batch_size,
                    rng,
                )
            end

            row = (
                timestamp_utc = string(now(UTC)),
                architecture = String(arch),
                context_length = context_length,
                init_mode = String(init_mode),
                text_loss = text_metrics.loss,
                text_ppl = text_metrics.ppl,
                text_acc = text_metrics.acc,
                text_acc_early = text_metrics.acc_early,
                text_acc_middle = text_metrics.acc_middle,
                text_acc_late = text_metrics.acc_late,
                needle_acc = needle_acc,
                device = String(run_device),
                run_note = run_note,
                checkpoint_error = ckpt_error,
            )
            push!(rows, row)

            println(@sprintf(
                "  N=%6d | init=%s | text_acc=%.4f | text_ppl=%.3f | needle=%.4f | dev=%s",
                context_length,
                String(init_mode),
                row.text_acc,
                row.text_ppl,
                row.needle_acc,
                String(run_device),
            ))
            if !isempty(ckpt_error)
                println("    checkpoint note: $ckpt_error")
            end
        end
        println()
    end

    write_csv(rows, args["output"])
    println("Saved evaluation CSV: $(args["output"])")
end

main()
