#!/usr/bin/env julia

using Pkg
Pkg.activate(dirname(@__DIR__))

using ArgParse
using TOML
using Random
using Lux
using Zygote
using Optimisers
using NNlib
using Statistics: mean
using Printf
using Dates
using ChainRulesCore
using CUDA

include(joinpath(dirname(@__DIR__), "src", "Swamma.jl"))
using .Swamma
include(joinpath(@__DIR__, "long_context_models.jl"))
using .LongContextModels

function parse_cli_args()
    s = ArgParseSettings(description = "Train Swamma vs Transformer on synthetic needle task and save checkpoints.")
    @add_arg_table! s begin
        "--config"
            help = "Path to training TOML config"
            arg_type = String
            default = "configs/swamma_vs_transformer/train_needle_quick.toml"
        "--device"
            help = "Execution device: cpu | gpu"
            arg_type = String
            default = "gpu"
        "--seed"
            help = "Global seed override (negative keeps config)"
            arg_type = Int
            default = -1
        "--steps"
            help = "Step override (<=0 keeps config)"
            arg_type = Int
            default = -1
        "--output-dir"
            help = "Checkpoint output root override"
            arg_type = String
            default = ""
        "--allow-gpu-training"
            help = "Allow GPU training attempts (may fail with current Zygote/CUDA PRIME path)"
            action = :store_true
    end
    return ArgParse.parse_args(s)
end

function parse_config(path::String)
    cfg = TOML.parsefile(path)
    common = get(cfg, "common", Dict{String, Any}())
    sweep = get(cfg, "sweep", Dict{String, Any}())
    train = get(cfg, "train", Dict{String, Any}())

    arch_raw = get(sweep, "architectures", ["swamma", "transformer"])
    archs = Symbol.(String.(arch_raw))

    return (
        common = (
            vocab_size = Int(get(common, "vocab_size", 32000)),
            context_length = Int(get(common, "context_length", 1024)),
            embedding_dimension = Int(get(common, "embedding_dimension", 256)),
            number_of_heads = Int(get(common, "number_of_heads", 8)),
            number_of_layers = Int(get(common, "number_of_layers", 8)),
            time_dimension = Int(get(common, "time_dimension", 128)),
            state_dimension = Int(get(common, "state_dimension", get(common, "embedding_dimension", 256))),
            window_size = Int(get(common, "window_size", 128)),
            min_frequency = Float32(get(common, "min_frequency", 0.1)),
            max_frequency = Float32(get(common, "max_frequency", 10.0)),
            default_time_step = Float32(get(common, "default_time_step", 0.1)),
            prime_subtoken_length = Int(get(common, "prime_subtoken_length", 4)),
            prime_subtoken_base = Int(get(common, "prime_subtoken_base", 16)),
        ),
        sweep = (
            architectures = archs,
        ),
        train = (
            seed = Int(get(train, "seed", 42)),
            steps = Int(get(train, "steps", 1000)),
            batch_size = Int(get(train, "batch_size", 8)),
            learning_rate = Float32(get(train, "learning_rate", 1e-3)),
            weight_decay = Float32(get(train, "weight_decay", 0.0)),
            log_every = Int(get(train, "log_every", 20)),
            eval_every = Int(get(train, "eval_every", 100)),
            eval_batches = Int(get(train, "eval_batches", 16)),
            save_every = Int(get(train, "save_every", 200)),
            output_dir = String(get(train, "output_dir", "checkpoints/long_context_needle_quick")),
        ),
    )
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

function to_device_like(target, x::AbstractArray)
    target_type = string(typeof(target))
    if occursin("CuArray", target_type)
        cuda_mod = parentmodule(typeof(target))
        while cuda_mod !== Main && !isdefined(cuda_mod, :CuArray)
            cuda_mod = parentmodule(cuda_mod)
        end
        if isdefined(cuda_mod, :CuArray)
            return cuda_mod.CuArray(x)
        end
    end
    return x
end

function first_array_leaf(x)
    if x isa AbstractArray
        return x
    elseif x isa NamedTuple
        for v in values(x)
            leaf = first_array_leaf(v)
            leaf === nothing || return leaf
        end
    elseif x isa Tuple
        for v in x
            leaf = first_array_leaf(v)
            leaf === nothing || return leaf
        end
    end
    return nothing
end

function synchronize_if_needed(device::Symbol)
    if device == :gpu
        CUDA.synchronize()
    end
end

function resolve_device(device_arg::String; allow_gpu_training::Bool = false)
    device = Symbol(lowercase(strip(device_arg)))
    device in (:cpu, :gpu) || error("Invalid --device value '$device_arg'. Use cpu or gpu.")
    if device == :gpu && !CUDA.functional()
        println("CUDA not functional. Falling back to CPU.")
        return :cpu
    elseif device == :gpu && !allow_gpu_training
        println("GPU training fallback: current PRIME+Zygote path triggers CUDA AD errors. Using CPU.")
        return :cpu
    end
    return device
end

function final_token_ce_loss(last_logits, targets::AbstractVector{<:Integer})
    vocab_size, batch_size = size(last_logits)
    log_probs = NNlib.logsoftmax(last_logits, dims = 1)

    one_hot = ChainRulesCore.ignore_derivatives() do
        oh = zeros(Float32, vocab_size, batch_size)
        for b in 1:batch_size
            t = Int(targets[b])
            if 1 <= t <= vocab_size
                oh[t, b] = 1f0
            end
        end
        to_device_like(last_logits, oh)
    end

    return -sum(log_probs .* one_hot) / batch_size
end

function needle_train_loss(model, params, state, token_ids::AbstractMatrix{<:Integer})
    seq_len, batch_size = size(token_ids)

    subtoken_state = token_ids_to_subtokens(token_ids, model.prime_code_table)
    masked = copy(subtoken_state)
    masked[:, seq_len, :] .= model.prime_mask_subtoken_id
    targets = vec(@view(token_ids[seq_len, :]))

    ref = first_array_leaf(params)
    masked_device = ref === nothing ? masked : to_device_like(ref, masked)
    inputs = (subtoken_state = masked_device, mask_ratio = Float32(1 / seq_len))

    logits, new_state = model(inputs, params, state)
    last_logits = logits[:, seq_len, :]
    loss = final_token_ce_loss(last_logits, targets)
    return loss, new_state
end

function evaluate_needle_accuracy(model, params, state, eval_batches::Int, seq_len::Int, batch_size::Int, rng)
    local_state = Lux.testmode(state)
    accs = Float32[]
    for _ in 1:eval_batches
        acc, local_state = needle_eval_step(model, params, local_state, seq_len, batch_size, rng)
        push!(accs, acc)
    end
    return mean(accs)
end

function train_one_architecture(
    arch::Symbol,
    cfg,
    device::Symbol,
    output_root::String,
    seed::Int,
)
    arch_rng = Random.MersenneTwister(seed)

    spec = ModelSpec(
        architecture = arch,
        vocab_size = cfg.common.vocab_size,
        max_sequence_length = cfg.common.context_length,
        embedding_dimension = cfg.common.embedding_dimension,
        number_of_heads = cfg.common.number_of_heads,
        number_of_layers = cfg.common.number_of_layers,
        time_dimension = cfg.common.time_dimension,
        state_dimension = cfg.common.state_dimension,
        window_size = min(cfg.common.window_size, cfg.common.context_length),
        min_frequency = cfg.common.min_frequency,
        max_frequency = cfg.common.max_frequency,
        default_time_step = cfg.common.default_time_step,
        prime_subtoken_length = cfg.common.prime_subtoken_length,
        prime_subtoken_base = cfg.common.prime_subtoken_base,
    )

    model = build_model(spec)
    params, state = Lux.setup(arch_rng, model)
    params = to_device(params, device)
    state = to_device(state, device)

    optimizer = Optimisers.AdamW(; eta = cfg.train.learning_rate, lambda = cfg.train.weight_decay)
    opt_state = Optimisers.setup(optimizer, params)

    best_acc = -Inf32
    losses = Float32[]
    ckpt_dir = joinpath(output_root, String(arch))
    mkpath(ckpt_dir)

    println("[Train] architecture=$arch")
    println("  output_dir=$ckpt_dir")

    for step in 1:cfg.train.steps
        tokens, _ = synthetic_needle_batch(
            spec.vocab_size,
            spec.max_sequence_length,
            cfg.train.batch_size,
            arch_rng,
        )

        (loss, new_state), grads = Zygote.withgradient(params) do ps
            needle_train_loss(model, ps, state, tokens)
        end
        state = new_state
        opt_state, params = Optimisers.update(opt_state, params, grads[1])
        push!(losses, Float32(loss))

        if step % cfg.train.log_every == 0 || step == 1
            recent = losses[max(1, end - min(length(losses), cfg.train.log_every) + 1):end]
            println(@sprintf("  step=%5d | loss=%.5f | loss_recent=%.5f", step, Float32(loss), mean(recent)))
        end

        if step % cfg.train.eval_every == 0 || step == cfg.train.steps
            eval_acc = evaluate_needle_accuracy(
                model, params, state,
                cfg.train.eval_batches,
                spec.max_sequence_length,
                cfg.train.batch_size,
                arch_rng,
            )
            println(@sprintf("  step=%5d | eval_needle_acc=%.4f", step, eval_acc))

            if eval_acc > best_acc
                best_acc = eval_acc
                best_path = joinpath(ckpt_dir, "checkpoint_best.jls")
                save_checkpoint(best_path;
                    params = params,
                    state = state,
                    opt_state = opt_state,
                    epoch = step,
                    loss = Float32(1 - eval_acc),
                )
                println("    saved best checkpoint: $best_path")
            end
        end

        if step % cfg.train.save_every == 0 || step == cfg.train.steps
            step_path = joinpath(ckpt_dir, "checkpoint_step_$(step).jls")
            save_checkpoint(step_path;
                params = params,
                state = state,
                opt_state = opt_state,
                epoch = step,
                loss = Float32(loss),
            )
            println("    saved checkpoint: $step_path")
        end

        synchronize_if_needed(device)
    end

    last_path = joinpath(ckpt_dir, "checkpoint_last.jls")
    save_checkpoint(last_path;
        params = params,
        state = state,
        opt_state = opt_state,
        epoch = cfg.train.steps,
        loss = isempty(losses) ? nothing : losses[end],
    )
    println("  saved last checkpoint: $last_path")
    println(@sprintf("  best_eval_needle_acc=%.4f", best_acc))
    println()
end

function main()
    args = parse_cli_args()
    cfg = parse_config(args["config"])
    device = resolve_device(args["device"]; allow_gpu_training = args["allow-gpu-training"])

    seed = args["seed"] >= 0 ? args["seed"] : cfg.train.seed
    steps_override = args["steps"] > 0 ? args["steps"] : cfg.train.steps
    output_root = isempty(args["output-dir"]) ? cfg.train.output_dir : args["output-dir"]
    cfg = (common = cfg.common, sweep = cfg.sweep, train = merge(cfg.train, (steps = steps_override,)))

    println("="^72)
    println("Synthetic Needle Training (Swamma vs Transformer)")
    println("="^72)
    println("Config: $(args["config"])")
    println("Architectures: $(join(string.(cfg.sweep.architectures), ", "))")
    println("Context length: $(cfg.common.context_length)")
    println("Steps: $(cfg.train.steps)")
    println("Batch size: $(cfg.train.batch_size)")
    println("Device: $device")
    if device == :gpu
        println("CUDA device: $(CUDA.name(CUDA.device()))")
    end
    println("Seed: $seed")
    println("Output root: $output_root")
    println()

    for (i, arch) in enumerate(cfg.sweep.architectures)
        arch_seed = seed + 1000 * i
        train_one_architecture(arch, cfg, device, output_root, arch_seed)
    end

    println("Training run complete.")
end

main()
