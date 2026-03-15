#!/usr/bin/env julia

using Pkg
Pkg.activate(dirname(@__DIR__))

using ArgParse
using Dates
using Statistics: mean, std
using Printf

function parse_cli_args()
    s = ArgParseSettings(description = "Aggregate long-context benchmark/eval results across multiple seed CSVs.")
    @add_arg_table! s begin
        "--benchmark-csvs"
            help = "Comma-separated benchmark CSV paths"
            arg_type = String
            default = "benchmarks/long_context_benchmark.csv"
        "--eval-csvs"
            help = "Comma-separated eval CSV paths"
            arg_type = String
            default = "benchmarks/long_context_eval_full64.csv"
        "--output-benchmark-csv"
            help = "Output aggregated benchmark CSV path"
            arg_type = String
            default = "benchmarks/long_context_benchmark_agg.csv"
        "--output-eval-csv"
            help = "Output aggregated eval CSV path"
            arg_type = String
            default = "benchmarks/long_context_eval_agg.csv"
        "--output-md"
            help = "Output markdown summary path"
            arg_type = String
            default = "benchmarks/long_context_aggregate_summary.md"
        "--swamma-arch"
            help = "Architecture label for Swamma rows"
            arg_type = String
            default = "swamma"
        "--transformer-arch"
            help = "Architecture label for Transformer rows"
            arg_type = String
            default = "transformer"
    end
    return ArgParse.parse_args(s)
end

function parse_input_paths(spec::String)
    paths = String[strip(p) for p in split(spec, ",")]
    paths = filter(p -> !isempty(p), paths)
    isempty(paths) && error("No input paths provided.")
    for p in paths
        isfile(p) || error("Input file not found: $p")
    end
    return paths
end

function parse_csv(path::String)
    lines = readlines(path)
    isempty(lines) && return Dict{String, String}[]
    header = split(lines[1], ",")

    rows = Dict{String, String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        vals = split(line, ",")
        length(vals) == length(header) || continue
        row = Dict{String, String}()
        for (k, v) in zip(header, vals)
            row[k] = strip(v)
        end
        push!(rows, row)
    end
    return rows
end

safe_parse_int(x::AbstractString) = try parse(Int, x) catch; nothing end
safe_parse_float(x::AbstractString) = try parse(Float64, x) catch; NaN end

function finite_stats(values::Vector{Float64})
    finite_vals = filter(isfinite, values)
    if isempty(finite_vals)
        return NaN, NaN, 0
    end
    m = mean(finite_vals)
    s = length(finite_vals) > 1 ? std(finite_vals) : 0.0
    return m, s, length(finite_vals)
end

function aggregate_benchmark(paths::Vector{String})
    groups = Dict{Tuple{String, Int}, Vector{Dict{String, String}}}()
    for p in paths
        for row in parse_csv(p)
            arch = lowercase(get(row, "architecture", ""))
            n = safe_parse_int(get(row, "context_length", ""))
            n === nothing && continue
            key = (arch, n)
            if !haskey(groups, key)
                groups[key] = Dict{String, String}[]
            end
            push!(groups[key], row)
        end
    end

    out_rows = NamedTuple[]
    for ((arch, n), rows) in sort(collect(groups), by = x -> (x[1][1], x[1][2]))
        mean_ms, std_ms, n_finite_ms = finite_stats([safe_parse_float(get(r, "mean_forward_ms", "NaN")) for r in rows])
        tps_mean, tps_std, n_finite_tps = finite_stats([safe_parse_float(get(r, "tokens_per_sec", "NaN")) for r in rows])
        params_mean, params_std, _ = finite_stats([safe_parse_float(get(r, "params", "NaN")) for r in rows])

        push!(out_rows, (
            architecture = arch,
            context_length = n,
            n_runs = length(rows),
            n_finite_ms = n_finite_ms,
            n_finite_tps = n_finite_tps,
            mean_forward_ms_mean = mean_ms,
            mean_forward_ms_std = std_ms,
            tokens_per_sec_mean = tps_mean,
            tokens_per_sec_std = tps_std,
            params_mean = params_mean,
            params_std = params_std,
        ))
    end

    return out_rows
end

function aggregate_eval(paths::Vector{String})
    groups = Dict{Tuple{String, Int}, Vector{Dict{String, String}}}()
    for p in paths
        for row in parse_csv(p)
            arch = lowercase(get(row, "architecture", ""))
            n = safe_parse_int(get(row, "context_length", ""))
            n === nothing && continue
            key = (arch, n)
            if !haskey(groups, key)
                groups[key] = Dict{String, String}[]
            end
            push!(groups[key], row)
        end
    end

    out_rows = NamedTuple[]
    metrics = (
        "text_loss",
        "text_ppl",
        "text_acc",
        "text_acc_early",
        "text_acc_middle",
        "text_acc_late",
        "needle_acc",
    )
    for ((arch, n), rows) in sort(collect(groups), by = x -> (x[1][1], x[1][2]))
        stats = Dict{String, Tuple{Float64, Float64, Int}}()
        for m in metrics
            stats[m] = finite_stats([safe_parse_float(get(r, m, "NaN")) for r in rows])
        end

        push!(out_rows, (
            architecture = arch,
            context_length = n,
            n_runs = length(rows),
            text_loss_mean = stats["text_loss"][1],
            text_loss_std = stats["text_loss"][2],
            text_ppl_mean = stats["text_ppl"][1],
            text_ppl_std = stats["text_ppl"][2],
            text_acc_mean = stats["text_acc"][1],
            text_acc_std = stats["text_acc"][2],
            text_acc_early_mean = stats["text_acc_early"][1],
            text_acc_early_std = stats["text_acc_early"][2],
            text_acc_middle_mean = stats["text_acc_middle"][1],
            text_acc_middle_std = stats["text_acc_middle"][2],
            text_acc_late_mean = stats["text_acc_late"][1],
            text_acc_late_std = stats["text_acc_late"][2],
            needle_acc_mean = stats["needle_acc"][1],
            needle_acc_std = stats["needle_acc"][2],
        ))
    end

    return out_rows
end

function fit_exponent(rows, arch::String)
    points = Tuple{Float64, Float64}[]
    for r in rows
        r.architecture == lowercase(arch) || continue
        isfinite(r.mean_forward_ms_mean) && r.mean_forward_ms_mean > 0 || continue
        push!(points, (Float64(r.context_length), r.mean_forward_ms_mean))
    end
    length(points) >= 2 || return NaN

    xs = log.([p[1] for p in points])
    ys = log.([p[2] for p in points])
    xbar = mean(xs)
    ybar = mean(ys)
    den = sum((x - xbar)^2 for x in xs)
    den == 0 && return NaN
    num = sum((x - xbar) * (y - ybar) for (x, y) in zip(xs, ys))
    return num / den
end

function write_benchmark_csv(path::String, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "architecture,context_length,n_runs,n_finite_ms,n_finite_tps,mean_forward_ms_mean,mean_forward_ms_std,tokens_per_sec_mean,tokens_per_sec_std,params_mean,params_std")
        for r in rows
            println(io,
                "$(r.architecture),$(r.context_length),$(r.n_runs),$(r.n_finite_ms),$(r.n_finite_tps),",
                @sprintf("%.6f", r.mean_forward_ms_mean), ",",
                @sprintf("%.6f", r.mean_forward_ms_std), ",",
                @sprintf("%.6f", r.tokens_per_sec_mean), ",",
                @sprintf("%.6f", r.tokens_per_sec_std), ",",
                @sprintf("%.2f", r.params_mean), ",",
                @sprintf("%.2f", r.params_std)
            )
        end
    end
end

function write_eval_csv(path::String, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "architecture,context_length,n_runs,text_loss_mean,text_loss_std,text_ppl_mean,text_ppl_std,text_acc_mean,text_acc_std,text_acc_early_mean,text_acc_early_std,text_acc_middle_mean,text_acc_middle_std,text_acc_late_mean,text_acc_late_std,needle_acc_mean,needle_acc_std")
        for r in rows
            println(io,
                "$(r.architecture),$(r.context_length),$(r.n_runs),",
                @sprintf("%.6f", r.text_loss_mean), ",",
                @sprintf("%.6f", r.text_loss_std), ",",
                @sprintf("%.6f", r.text_ppl_mean), ",",
                @sprintf("%.6f", r.text_ppl_std), ",",
                @sprintf("%.6f", r.text_acc_mean), ",",
                @sprintf("%.6f", r.text_acc_std), ",",
                @sprintf("%.6f", r.text_acc_early_mean), ",",
                @sprintf("%.6f", r.text_acc_early_std), ",",
                @sprintf("%.6f", r.text_acc_middle_mean), ",",
                @sprintf("%.6f", r.text_acc_middle_std), ",",
                @sprintf("%.6f", r.text_acc_late_mean), ",",
                @sprintf("%.6f", r.text_acc_late_std), ",",
                @sprintf("%.6f", r.needle_acc_mean), ",",
                @sprintf("%.6f", r.needle_acc_std)
            )
        end
    end
end

function write_md(path::String, bench_rows, eval_rows, sw_arch::String, tr_arch::String, bench_paths, eval_paths)
    bench_map = Dict((r.architecture, r.context_length) => r for r in bench_rows)
    eval_map = Dict((r.architecture, r.context_length) => r for r in eval_rows)
    contexts = sort(unique(vcat([r.context_length for r in bench_rows], [r.context_length for r in eval_rows])))

    sw_exp = fit_exponent(bench_rows, sw_arch)
    tr_exp = fit_exponent(bench_rows, tr_arch)

    speed_lines = String[]
    for n in contexts
        sw = get(bench_map, (lowercase(sw_arch), n), nothing)
        tr = get(bench_map, (lowercase(tr_arch), n), nothing)
        if sw === nothing || tr === nothing
            push!(speed_lines, "| $n | n/a | n/a | n/a |")
            continue
        end
        ratio = (isfinite(sw.mean_forward_ms_mean) && sw.mean_forward_ms_mean > 0 && isfinite(tr.mean_forward_ms_mean)) ?
            tr.mean_forward_ms_mean / sw.mean_forward_ms_mean : NaN
        ratio_s = isfinite(ratio) ? @sprintf("%.3f", ratio) : "n/a"
        push!(speed_lines, @sprintf("| %d | %.2f ± %.2f | %.2f ± %.2f | %s |",
            n,
            sw.tokens_per_sec_mean, sw.tokens_per_sec_std,
            tr.tokens_per_sec_mean, tr.tokens_per_sec_std,
            ratio_s
        ))
    end

    needle_lines = String[]
    for n in contexts
        sw = get(eval_map, (lowercase(sw_arch), n), nothing)
        tr = get(eval_map, (lowercase(tr_arch), n), nothing)
        if sw === nothing || tr === nothing
            push!(needle_lines, "| $n | n/a | n/a | n/a |")
            continue
        end
        delta = (isfinite(sw.needle_acc_mean) && isfinite(tr.needle_acc_mean)) ? sw.needle_acc_mean - tr.needle_acc_mean : NaN
        delta_s = isfinite(delta) ? @sprintf("%.4f", delta) : "n/a"
        push!(needle_lines, @sprintf("| %d | %.4f ± %.4f | %.4f ± %.4f | %s |",
            n,
            sw.needle_acc_mean, sw.needle_acc_std,
            tr.needle_acc_mean, tr.needle_acc_std,
            delta_s
        ))
    end

    md = """
# Long-Context Multi-Seed Aggregate

Generated: $(string(now(UTC))) UTC

## Inputs
- benchmark CSVs:
$(join(["  - `$p`" for p in bench_paths], "\n"))
- eval CSVs:
$(join(["  - `$p`" for p in eval_paths], "\n"))

## Scaling Exponents (From Aggregated Means)
- Swamma (`$sw_arch`): $(isfinite(sw_exp) ? @sprintf("%.4f", sw_exp) : "n/a")
- Transformer (`$tr_arch`): $(isfinite(tr_exp) ? @sprintf("%.4f", tr_exp) : "n/a")

## Throughput + Latency Ratio
| Context | Swamma tok/s (mean ± std) | Transformer tok/s (mean ± std) | Latency ratio (Transformer / Swamma) |
|---:|---:|---:|---:|
$(join(speed_lines, "\n"))

## Needle Accuracy
| Context | Swamma needle_acc (mean ± std) | Transformer needle_acc (mean ± std) | Delta (Swamma - Transformer) |
|---:|---:|---:|---:|
$(join(needle_lines, "\n"))
"""

    mkpath(dirname(path))
    open(path, "w") do io
        write(io, md)
    end
end

function main()
    args = parse_cli_args()
    bench_paths = parse_input_paths(args["benchmark-csvs"])
    eval_paths = parse_input_paths(args["eval-csvs"])

    bench_rows = aggregate_benchmark(bench_paths)
    eval_rows = aggregate_eval(eval_paths)

    write_benchmark_csv(args["output-benchmark-csv"], bench_rows)
    write_eval_csv(args["output-eval-csv"], eval_rows)
    write_md(
        args["output-md"],
        bench_rows,
        eval_rows,
        args["swamma-arch"],
        args["transformer-arch"],
        bench_paths,
        eval_paths,
    )

    println("Saved aggregated benchmark: $(args["output-benchmark-csv"])")
    println("Saved aggregated eval: $(args["output-eval-csv"])")
    println("Saved aggregate summary: $(args["output-md"])")
end

main()
