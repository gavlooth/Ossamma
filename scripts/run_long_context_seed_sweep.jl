#!/usr/bin/env julia

using Pkg
Pkg.activate(dirname(@__DIR__))

using ArgParse
using Printf

function parse_cli_args()
    s = ArgParseSettings(description = "Run long-context benchmark/eval across seeds and aggregate outputs.")
    @add_arg_table! s begin
        "--seeds"
            help = "Comma-separated seed list"
            arg_type = String
            default = "42,7,19"
        "--device"
            help = "Execution device: cpu | gpu"
            arg_type = String
            default = "gpu"
        "--benchmark-config"
            help = "Benchmark TOML path"
            arg_type = String
            default = "configs/swamma_vs_transformer/benchmark_long_context.toml"
        "--eval-config"
            help = "Eval TOML path"
            arg_type = String
            default = "configs/swamma_vs_transformer/eval_long_context_quick.toml"
        "--output-dir"
            help = "Output directory for per-seed files"
            arg_type = String
            default = "benchmarks"
        "--benchmark-prefix"
            help = "Per-seed benchmark filename prefix"
            arg_type = String
            default = "long_context_benchmark_seed"
        "--eval-prefix"
            help = "Per-seed eval filename prefix"
            arg_type = String
            default = "long_context_eval_seed"
        "--run-benchmark"
            help = "Run benchmark script for each seed"
            action = :store_true
        "--run-eval"
            help = "Run eval script for each seed"
            action = :store_true
        "--aggregate"
            help = "Run aggregation after seed runs"
            action = :store_true
        "--skip-existing"
            help = "Skip per-seed runs when output CSV already exists"
            action = :store_true
        "--dry-run"
            help = "Print commands only"
            action = :store_true
        "--swamma-checkpoint"
            help = "Optional Swamma checkpoint path for eval"
            arg_type = String
            default = ""
        "--transformer-checkpoint"
            help = "Optional Transformer checkpoint path for eval"
            arg_type = String
            default = ""
        "--aggregate-benchmark-csv"
            help = "Aggregated benchmark CSV output path"
            arg_type = String
            default = "benchmarks/long_context_benchmark_agg_3seed.csv"
        "--aggregate-eval-csv"
            help = "Aggregated eval CSV output path"
            arg_type = String
            default = "benchmarks/long_context_eval_agg_3seed.csv"
        "--aggregate-md"
            help = "Aggregated markdown summary output path"
            arg_type = String
            default = "benchmarks/long_context_aggregate_summary_3seed.md"
    end
    return ArgParse.parse_args(s)
end

function parse_seed_list(spec::String)
    seeds = Int[]
    for part in split(spec, ",")
        token = strip(part)
        isempty(token) && continue
        push!(seeds, parse(Int, token))
    end
    isempty(seeds) && error("No seeds parsed from --seeds")
    return seeds
end

function maybe_run(cmd::Cmd, dry_run::Bool)
    println("CMD: $cmd")
    dry_run && return
    run(cmd)
end

function main()
    args = parse_cli_args()
    seeds = parse_seed_list(args["seeds"])

    run_benchmark = args["run-benchmark"]
    run_eval = args["run-eval"]
    run_aggregate = args["aggregate"]
    if !run_benchmark && !run_eval && !run_aggregate
        run_benchmark = true
        run_eval = true
        run_aggregate = true
    end

    out_dir = args["output-dir"]
    mkpath(out_dir)
    proj = "--project=."
    dry_run = args["dry-run"]
    skip_existing = args["skip-existing"]

    benchmark_paths = String[]
    eval_paths = String[]

    println("="^72)
    println("Long-Context Seed Sweep")
    println("="^72)
    println("Seeds: $(join(string.(seeds), ", "))")
    println("Device: $(args["device"])")
    println("Run benchmark: $run_benchmark")
    println("Run eval: $run_eval")
    println("Aggregate: $run_aggregate")
    println("Skip existing: $skip_existing")
    println("Dry run: $dry_run")
    println()

    for seed in seeds
        bench_out = joinpath(out_dir, "$(args["benchmark-prefix"])$(seed).csv")
        eval_out = joinpath(out_dir, "$(args["eval-prefix"])$(seed).csv")

        push!(benchmark_paths, bench_out)
        push!(eval_paths, eval_out)

        if run_benchmark
            if skip_existing && isfile(bench_out)
                println("Skipping benchmark seed $seed (exists): $bench_out")
            else
                cmd = `julia $proj scripts/benchmark_long_context.jl --config $(args["benchmark-config"]) --output $bench_out --device $(args["device"]) --seed $(string(seed))`
                maybe_run(cmd, dry_run)
            end
        end

        if run_eval
            if skip_existing && isfile(eval_out)
                println("Skipping eval seed $seed (exists): $eval_out")
            else
                cmd = `julia $proj scripts/eval_long_context.jl --config $(args["eval-config"]) --output $eval_out --device $(args["device"]) --seed $(string(seed))`
                if !isempty(args["swamma-checkpoint"])
                    cmd = `$cmd --swamma-checkpoint $(args["swamma-checkpoint"])`
                end
                if !isempty(args["transformer-checkpoint"])
                    cmd = `$cmd --transformer-checkpoint $(args["transformer-checkpoint"])`
                end
                maybe_run(cmd, dry_run)
            end
        end
    end

    if run_aggregate
        bench_csvs = join(benchmark_paths, ",")
        eval_csvs = join(eval_paths, ",")
        cmd = `julia $proj scripts/aggregate_long_context_seeds.jl --benchmark-csvs $bench_csvs --eval-csvs $eval_csvs --output-benchmark-csv $(args["aggregate-benchmark-csv"]) --output-eval-csv $(args["aggregate-eval-csv"]) --output-md $(args["aggregate-md"])`
        maybe_run(cmd, dry_run)
    end

    println("\nSweep finished.")
end

main()

