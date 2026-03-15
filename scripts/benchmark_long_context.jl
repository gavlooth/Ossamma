#!/usr/bin/env julia

using Pkg
Pkg.activate(dirname(@__DIR__))

using ArgParse
using TOML
using Random
using Lux
using CUDA
using Statistics: mean, std
using Printf
using Dates

include(joinpath(dirname(@__DIR__), "src", "Swamma.jl"))
using .Swamma
include(joinpath(@__DIR__, "long_context_models.jl"))
using .LongContextModels

function parse_cli_args()
    s = ArgParseSettings(description = "Long-context complexity benchmark (Swamma vs Transformer baseline)")
    @add_arg_table! s begin
        "--config"
            help = "Path to benchmark TOML config"
            arg_type = String
            default = "configs/swamma_vs_transformer/benchmark_long_context.toml"
        "--output"
            help = "Output CSV path"
            arg_type = String
            default = "benchmarks/long_context_benchmark.csv"
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

function synchronize_if_needed(device::Symbol)
    if device == :gpu
        CUDA.synchronize()
    end
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
    bench = get(cfg, "benchmark", Dict{String, Any}())

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
            batch_size = Int(get(common, "batch_size", 1)),
        ),
        sweep = (
            context_lengths = Int.(get(sweep, "context_lengths", [1024, 2048, 4096, 8192, 16384])),
            architectures = archs,
            warmup = Int(get(sweep, "warmup", 2)),
            iterations = Int(get(sweep, "iterations", 5)),
        ),
        benchmark = (
            mask_ratio = Float32(get(bench, "mask_ratio", 0.5)),
            seed = Int(get(bench, "seed", 42)),
        ),
    )
end

function parameter_count(params)
    return LongContextModels.count_parameters(params)
end

function fit_scaling_exponent(rows, arch::Symbol)
    points = [
        (Float64(r.context_length), Float64(r.mean_forward_ms))
        for r in rows if r.architecture == arch && isfinite(r.mean_forward_ms) && r.mean_forward_ms > 0
    ]
    length(points) >= 2 || return NaN

    xs = log.([p[1] for p in points])
    ys = log.([p[2] for p in points])
    xbar = mean(xs)
    ybar = mean(ys)
    num = sum((x - xbar) * (y - ybar) for (x, y) in zip(xs, ys))
    den = sum((x - xbar)^2 for x in xs)
    den == 0 && return NaN
    return num / den
end

function write_csv(rows, path::String)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "timestamp_utc,architecture,context_length,batch_size,params,mean_forward_ms,std_forward_ms,tokens_per_sec,device,run_note")
        for r in rows
            println(io,
                string(r.timestamp_utc), ",",
                string(r.architecture), ",",
                r.context_length, ",",
                r.batch_size, ",",
                r.params, ",",
                @sprintf("%.6f", r.mean_forward_ms), ",",
                @sprintf("%.6f", r.std_forward_ms), ",",
                @sprintf("%.2f", r.tokens_per_sec), ",",
                r.device, ",",
                replace(r.run_note, ',' => ';')
            )
        end
    end
end

function effective_device_for_point(requested_device::Symbol, arch::Symbol, context_length::Int)
    _ = arch
    _ = context_length
    return requested_device, ""
end

function main()
    args = parse_cli_args()
    cfg = parse_config(args["config"])
    device = resolve_device(args["device"])

    seed = args["seed"] >= 0 ? args["seed"] : cfg.benchmark.seed
    rng = Random.MersenneTwister(seed)

    println("="^72)
    println("Long-Context Complexity Benchmark")
    println("="^72)
    println("Config: $(args["config"])")
    println("Architectures: $(join(string.(cfg.sweep.architectures), ", "))")
    println("Contexts: $(join(string.(cfg.sweep.context_lengths), ", "))")
    println("Warmup/iters: $(cfg.sweep.warmup)/$(cfg.sweep.iterations)")
    println("Device: $device")
    if device == :gpu
        println("CUDA device: $(CUDA.name(CUDA.device()))")
    end
    println("Seed: $seed")
    println()

    rows = NamedTuple[]

    for arch in cfg.sweep.architectures
        println("[Architecture] $arch")

        for context_length in cfg.sweep.context_lengths
            run_device, run_note = effective_device_for_point(device, arch, context_length)
            if run_device == :skip
                row = (
                    timestamp_utc = string(now(UTC)),
                    architecture = arch,
                    context_length = context_length,
                    batch_size = cfg.common.batch_size,
                    params = 0,
                    mean_forward_ms = NaN,
                    std_forward_ms = NaN,
                    tokens_per_sec = NaN,
                    device = "skipped",
                    run_note = run_note,
                )
                push!(rows, row)
                println("  N=$(lpad(context_length, 6)) | skipped | note: $run_note")
                continue
            elseif run_device != device
                println("  N=$(lpad(context_length, 6)) | note: $run_note")
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
            params = to_device(params, run_device)
            bench_state = Lux.testmode(to_device(state, run_device))

            token_ids = rand(rng, 1:spec.vocab_size, context_length, cfg.common.batch_size)
            subtoken_state = token_ids_to_subtokens(token_ids, model.prime_code_table)
            masked_subtokens, _, _ = apply_subtoken_mask(
                subtoken_state,
                cfg.benchmark.mask_ratio,
                model.prime_mask_subtoken_id;
                rng = rng,
            )
            masked_subtokens = to_device(masked_subtokens, run_device)
            inputs = (subtoken_state = masked_subtokens, mask_ratio = cfg.benchmark.mask_ratio)

            for _ in 1:cfg.sweep.warmup
                _, bench_state = model(inputs, params, bench_state)
            end
            synchronize_if_needed(run_device)

            times_ms = Float64[]
            for _ in 1:cfg.sweep.iterations
                synchronize_if_needed(run_device)
                elapsed_s = @elapsed begin
                    _, bench_state = model(inputs, params, bench_state)
                    synchronize_if_needed(run_device)
                end
                push!(times_ms, 1000.0 * elapsed_s)
            end

            mean_ms = mean(times_ms)
            std_ms = length(times_ms) > 1 ? std(times_ms) : 0.0
            toks_per_s = (context_length * cfg.common.batch_size) / (mean_ms / 1000.0)
            params_n = parameter_count(params)

            row = (
                timestamp_utc = string(now(UTC)),
                architecture = arch,
                context_length = context_length,
                batch_size = cfg.common.batch_size,
                params = params_n,
                mean_forward_ms = mean_ms,
                std_forward_ms = std_ms,
                tokens_per_sec = toks_per_s,
                device = String(run_device),
                run_note = run_note,
            )
            push!(rows, row)

            println(@sprintf(
                "  N=%6d | params=%8.2fM | mean=%.2fms ± %.2f | %.1f tok/s | dev=%s",
                context_length,
                params_n / 1e6,
                mean_ms,
                std_ms,
                toks_per_s,
                String(run_device),
            ))
        end

        slope = fit_scaling_exponent(rows, arch)
        if isfinite(slope)
            println(@sprintf("  log-log fitted time exponent for %s: %.3f", string(arch), slope))
        end
        println()
    end

    write_csv(rows, args["output"])
    println("Saved benchmark CSV: $(args["output"])")

    if :swamma in cfg.sweep.architectures && :transformer in cfg.sweep.architectures
        latest_swamma = filter(r -> r.architecture == :swamma, rows)
        latest_transformer = filter(r -> r.architecture == :transformer, rows)
        by_n_swamma = Dict(r.context_length => r for r in latest_swamma)
        by_n_transformer = Dict(r.context_length => r for r in latest_transformer)

        println("\nRelative Speedup (Transformer / Swamma)")
        for n in cfg.sweep.context_lengths
            if haskey(by_n_swamma, n) && haskey(by_n_transformer, n)
                sw = by_n_swamma[n]
                tr = by_n_transformer[n]
                if isfinite(sw.mean_forward_ms) && isfinite(tr.mean_forward_ms) && sw.mean_forward_ms > 0
                    speedup = tr.mean_forward_ms / sw.mean_forward_ms
                    println(@sprintf("  N=%6d | %.2fx", n, speedup))
                else
                    println(@sprintf("  N=%6d | n/a", n))
                end
            end
        end
    end
end

main()
