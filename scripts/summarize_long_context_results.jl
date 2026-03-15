#!/usr/bin/env julia

using Pkg
Pkg.activate(dirname(@__DIR__))

using ArgParse
using Dates
using Statistics: mean
using Printf

function parse_cli_args()
    s = ArgParseSettings(description = "Summarize long-context benchmark/eval CSV outputs.")
    @add_arg_table! s begin
        "--benchmark-csv"
            help = "Path to long-context benchmark CSV"
            arg_type = String
            default = "benchmarks/long_context_benchmark.csv"
        "--eval-csv"
            help = "Path to long-context eval CSV"
            arg_type = String
            default = "benchmarks/long_context_eval_full64.csv"
        "--output-md"
            help = "Path to markdown summary output"
            arg_type = String
            default = "benchmarks/long_context_summary.md"
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

function parse_csv(path::String)
    isfile(path) || error("CSV not found: $path")
    lines = readlines(path)
    isempty(lines) && return Dict{String, String}[]

    header = split(lines[1], ",")
    rows = Dict{String, String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = split(line, ",")
        length(values) == length(header) || continue
        row = Dict{String, String}()
        for (k, v) in zip(header, values)
            row[k] = strip(v)
        end
        push!(rows, row)
    end
    return rows
end

safe_parse_int(x::AbstractString) = try parse(Int, x) catch; nothing end
safe_parse_float(x::AbstractString) = try parse(Float64, x) catch; NaN end

function fit_exponent(rows, arch::String)
    points = Tuple{Float64, Float64}[]
    for r in rows
        lowercase(get(r, "architecture", "")) == lowercase(arch) || continue
        n = safe_parse_int(get(r, "context_length", ""))
        t = safe_parse_float(get(r, "mean_forward_ms", "NaN"))
        if n !== nothing && isfinite(t) && t > 0
            push!(points, (Float64(n), t))
        end
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

function by_arch_and_context(rows)
    out = Dict{Tuple{String, Int}, Dict{String, String}}()
    for r in rows
        arch = lowercase(get(r, "architecture", ""))
        n = safe_parse_int(get(r, "context_length", ""))
        n === nothing && continue
        out[(arch, n)] = r
    end
    return out
end

function finite_contexts(rows, arch::String; metric::String = "mean_forward_ms")
    contexts = Int[]
    for r in rows
        lowercase(get(r, "architecture", "")) == lowercase(arch) || continue
        n = safe_parse_int(get(r, "context_length", ""))
        n === nothing && continue
        m = safe_parse_float(get(r, metric, "NaN"))
        if isfinite(m)
            push!(contexts, n)
        end
    end
    sort!(unique!(contexts))
    return contexts
end

function write_summary_md(path::String, content::String)
    mkpath(dirname(path))
    open(path, "w") do io
        write(io, content)
    end
end

function main()
    args = parse_cli_args()
    bench_rows = parse_csv(args["benchmark-csv"])
    eval_rows = parse_csv(args["eval-csv"])

    sw_arch = lowercase(args["swamma-arch"])
    tr_arch = lowercase(args["transformer-arch"])

    bench_index = by_arch_and_context(bench_rows)
    eval_index = by_arch_and_context(eval_rows)

    sw_exp = fit_exponent(bench_rows, sw_arch)
    tr_exp = fit_exponent(bench_rows, tr_arch)

    all_contexts = sort(unique(filter(!isnothing, [safe_parse_int(get(r, "context_length", "")) for r in bench_rows])))

    speed_lines = String[]
    for n in all_contexts
        sw = get(bench_index, (sw_arch, n), nothing)
        tr = get(bench_index, (tr_arch, n), nothing)
        if sw === nothing || tr === nothing
            push!(speed_lines, "| $n | n/a | n/a | n/a |")
            continue
        end
        sw_ms = safe_parse_float(get(sw, "mean_forward_ms", "NaN"))
        tr_ms = safe_parse_float(get(tr, "mean_forward_ms", "NaN"))
        sw_tps = safe_parse_float(get(sw, "tokens_per_sec", "NaN"))
        tr_tps = safe_parse_float(get(tr, "tokens_per_sec", "NaN"))
        ratio = (isfinite(sw_ms) && isfinite(tr_ms) && sw_ms > 0) ? tr_ms / sw_ms : NaN
        ratio_str = isfinite(ratio) ? @sprintf("%.3f", ratio) : "n/a"
        push!(speed_lines, @sprintf("| %d | %.2f | %.2f | %s |", n, sw_tps, tr_tps, ratio_str))
    end

    eval_contexts = sort(unique(filter(!isnothing, [safe_parse_int(get(r, "context_length", "")) for r in eval_rows])))
    needle_lines = String[]
    for n in eval_contexts
        sw = get(eval_index, (sw_arch, n), nothing)
        tr = get(eval_index, (tr_arch, n), nothing)
        sw_n = sw === nothing ? NaN : safe_parse_float(get(sw, "needle_acc", "NaN"))
        tr_n = tr === nothing ? NaN : safe_parse_float(get(tr, "needle_acc", "NaN"))
        delta = (isfinite(sw_n) && isfinite(tr_n)) ? (sw_n - tr_n) : NaN
        sw_s = isfinite(sw_n) ? @sprintf("%.4f", sw_n) : "n/a"
        tr_s = isfinite(tr_n) ? @sprintf("%.4f", tr_n) : "n/a"
        d_s = isfinite(delta) ? @sprintf("%.4f", delta) : "n/a"
        push!(needle_lines, "| $n | $sw_s | $tr_s | $d_s |")
    end

    sw_max_ctx = isempty(finite_contexts(bench_rows, sw_arch)) ? "n/a" : string(maximum(finite_contexts(bench_rows, sw_arch)))
    tr_max_ctx = isempty(finite_contexts(bench_rows, tr_arch)) ? "n/a" : string(maximum(finite_contexts(bench_rows, tr_arch)))

    summary = """
# Long-Context Summary

Generated: $(string(now(UTC))) UTC

## Inputs
- benchmark: `$(args["benchmark-csv"])`
- eval: `$(args["eval-csv"])`
- swamma arch label: `$(args["swamma-arch"])`
- transformer arch label: `$(args["transformer-arch"])`

## Scaling Exponents
- Swamma log-log time exponent: $(isfinite(sw_exp) ? @sprintf("%.4f", sw_exp) : "n/a")
- Transformer log-log time exponent: $(isfinite(tr_exp) ? @sprintf("%.4f", tr_exp) : "n/a")

## Max Finite Benchmark Context
- Swamma: $sw_max_ctx
- Transformer: $tr_max_ctx

## Benchmark Table
| Context | Swamma tok/s | Transformer tok/s | Latency ratio (Transformer / Swamma) |
|---:|---:|---:|---:|
$(join(speed_lines, "\n"))

## Needle Accuracy Table
| Context | Swamma needle_acc | Transformer needle_acc | Delta (Swamma - Transformer) |
|---:|---:|---:|---:|
$(join(needle_lines, "\n"))
"""

    write_summary_md(args["output-md"], summary)
    println("Saved summary: $(args["output-md"])")
end

main()

