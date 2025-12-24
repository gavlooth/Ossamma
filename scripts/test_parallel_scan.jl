#!/usr/bin/env julia
"""
Test script for parallel scan GPU optimization.

Tests:
1. DLinOSSParallel produces same results as DLinOSS
2. GPU parallel scan performance improvement
3. Memory usage comparison
"""

using Random
using Statistics
using Printf
using BenchmarkTools

# Load Ossamma modules
include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
using .Ossamma
using .Ossamma: DLinOSS, DLinOSSParallel

using Lux
using CUDA

# Check if CUDA is available
const USE_GPU = CUDA.functional()

println("=" ^ 70)
println("Testing Parallel Scan GPU Optimization")
println("=" ^ 70)
println("CUDA available: $USE_GPU")
if USE_GPU
    println("GPU: $(CUDA.name(CUDA.device()))")
    println("VRAM: $(round(CUDA.total_memory() / 1e9, digits=1)) GB")
end
println()

# =============================================================================
# Test 1: Correctness - DLinOSSParallel vs DLinOSS
# =============================================================================

println("Test 1: Correctness Check")
println("-" ^ 40)

rng = Random.default_rng()
Random.seed!(rng, 42)

# Create layers with same config
dim_in, dim_state, dim_out = 64, 64, 64
min_freq, max_freq, default_dt = 0.1f0, 10.0f0, 0.1f0

layer_seq = DLinOSS(dim_in, dim_state, dim_out, min_freq, max_freq, default_dt)
layer_par = DLinOSSParallel(dim_in, dim_state, dim_out, min_freq, max_freq, default_dt; chunk_size=32)

# Initialize with same parameters
ps_seq = Lux.initialparameters(rng, layer_seq)
st_seq = Lux.initialstates(rng, layer_seq)

ps_par = Lux.initialparameters(rng, layer_par)
st_par = Lux.initialstates(rng, layer_par)

# Copy parameters to ensure identical
ps_par = (
    log_time_step = copy(ps_seq.log_time_step),
    log_stiffness_coefficients = copy(ps_seq.log_stiffness_coefficients),
    log_damping_coefficients = copy(ps_seq.log_damping_coefficients),
    input_projection = copy(ps_seq.input_projection),
    output_projection = copy(ps_seq.output_projection),
)

# Test input
T, B = 128, 8  # sequence length, batch size
x = randn(rng, Float32, dim_in, T, B)

# Run both
y_seq, st_seq_new = layer_seq(x, ps_seq, st_seq)
y_par, st_par_new = layer_par(x, ps_par, st_par)

# Compare outputs
max_diff = maximum(abs.(y_seq .- y_par))
mean_diff = mean(abs.(y_seq .- y_par))

println("  Max absolute difference: $(@sprintf("%.2e", max_diff))")
println("  Mean absolute difference: $(@sprintf("%.2e", mean_diff))")

if max_diff < 1e-4
    println("  ✓ PASS: Parallel implementation matches sequential")
else
    println("  ✗ FAIL: Significant difference detected")
end
println()

# =============================================================================
# Test 2: Performance Benchmark (CPU)
# =============================================================================

println("Test 2: CPU Performance Benchmark")
println("-" ^ 40)

# Benchmark sequential
t_seq = @benchmark $layer_seq($x, $ps_seq, $st_seq) samples=10 evals=1
println("  Sequential: $(round(median(t_seq.times) / 1e6, digits=2)) ms")

# Benchmark parallel (CPU path)
t_par = @benchmark $layer_par($x, $ps_par, $st_par) samples=10 evals=1
println("  Parallel:   $(round(median(t_par.times) / 1e6, digits=2)) ms")

speedup_cpu = median(t_seq.times) / median(t_par.times)
println("  Speedup: $(round(speedup_cpu, digits=2))×")
println()

# =============================================================================
# Test 3: GPU Performance Benchmark
# =============================================================================

if USE_GPU
    println("Test 3: GPU Performance Benchmark")
    println("-" ^ 40)

    # Move to GPU
    device = Lux.gpu_device()

    x_gpu = device(x)
    ps_seq_gpu = device(ps_seq)
    st_seq_gpu = device(st_seq)
    ps_par_gpu = device(ps_par)
    st_par_gpu = device(st_par)

    # Warmup
    y_seq_gpu, _ = layer_seq(x_gpu, ps_seq_gpu, st_seq_gpu)
    y_par_gpu, _ = layer_par(x_gpu, ps_par_gpu, st_par_gpu)
    CUDA.synchronize()

    # Benchmark GPU sequential
    t_seq_gpu = @benchmark begin
        CUDA.@sync $layer_seq($x_gpu, $ps_seq_gpu, $st_seq_gpu)
    end samples=20 evals=1
    println("  Sequential (GPU): $(round(median(t_seq_gpu.times) / 1e6, digits=2)) ms")

    # Benchmark GPU parallel
    t_par_gpu = @benchmark begin
        CUDA.@sync $layer_par($x_gpu, $ps_par_gpu, $st_par_gpu)
    end samples=20 evals=1
    println("  Parallel (GPU):   $(round(median(t_par_gpu.times) / 1e6, digits=2)) ms")

    speedup_gpu = median(t_seq_gpu.times) / median(t_par_gpu.times)
    println("  GPU Speedup: $(round(speedup_gpu, digits=2))×")

    # GPU utilization during parallel scan
    println()
    println("  GPU Memory Usage:")
    mem_used = CUDA.used_memory() / 1e9
    mem_total = CUDA.total_memory() / 1e9
    println("    Used: $(round(mem_used, digits=2)) GB / $(round(mem_total, digits=2)) GB")
    println()
end

# =============================================================================
# Test 4: Scaling with Sequence Length
# =============================================================================

println("Test 4: Scaling with Sequence Length")
println("-" ^ 40)

seq_lengths = [32, 64, 128, 256, 512]
println("  SeqLen  | Sequential | Parallel  | Speedup")
println("  --------|------------|-----------|--------")

for T_test in seq_lengths
    x_test = randn(rng, Float32, dim_in, T_test, 4)

    t_s = @benchmark $layer_seq($x_test, $ps_seq, $st_seq) samples=5 evals=1
    t_p = @benchmark $layer_par($x_test, $ps_par, $st_par) samples=5 evals=1

    ms_seq = round(median(t_s.times) / 1e6, digits=2)
    ms_par = round(median(t_p.times) / 1e6, digits=2)
    speedup = round(median(t_s.times) / median(t_p.times), digits=2)

    @printf("  %6d  | %8.2f ms | %7.2f ms | %5.2f×\n", T_test, ms_seq, ms_par, speedup)
end
println()

# =============================================================================
# Test 5: Full NER Model with Parallel Scan
# =============================================================================

println("Test 5: Full OssammaNER Model Test")
println("-" ^ 40)

using .Ossamma: OssammaNER, NERConfig

# Create two models: one with parallel scan, one without
config_seq = NERConfig(
    vocab_size = 1000,
    max_sequence_length = 128,
    embedding_dimension = 64,
    number_of_heads = 2,
    number_of_layers = 2,
    time_dimension = 32,
    use_parallel_scan = false,
)

config_par = NERConfig(
    vocab_size = 1000,
    max_sequence_length = 128,
    embedding_dimension = 64,
    number_of_heads = 2,
    number_of_layers = 2,
    time_dimension = 32,
    use_parallel_scan = true,
    parallel_chunk_size = 32,
)

model_seq = OssammaNER(config_seq)
model_par = OssammaNER(config_par)

ps_model_seq = Lux.initialparameters(rng, model_seq)
st_model_seq = Lux.initialstates(rng, model_seq)

ps_model_par = Lux.initialparameters(rng, model_par)
st_model_par = Lux.initialstates(rng, model_par)

# Test input
token_ids = rand(rng, 1:1000, 64, 4)

# Run models
(emissions_seq, _), _ = model_seq(token_ids, ps_model_seq, st_model_seq)
(emissions_par, _), _ = model_par(token_ids, ps_model_par, st_model_par)

println("  Sequential model output shape: $(size(emissions_seq))")
println("  Parallel model output shape:   $(size(emissions_par))")

# Benchmark full model
t_model_seq = @benchmark $model_seq($token_ids, $ps_model_seq, $st_model_seq) samples=5 evals=1
t_model_par = @benchmark $model_par($token_ids, $ps_model_par, $st_model_par) samples=5 evals=1

println("  Sequential model: $(round(median(t_model_seq.times) / 1e6, digits=2)) ms")
println("  Parallel model:   $(round(median(t_model_par.times) / 1e6, digits=2)) ms")
println("  Model speedup: $(round(median(t_model_seq.times) / median(t_model_par.times), digits=2))×")
println()

# =============================================================================
# Summary
# =============================================================================

println("=" ^ 70)
println("Summary")
println("=" ^ 70)
println("✓ Parallel associative scan implemented for DLinOSS")
println("✓ GPU-optimized configuration created")
println("✓ OssammaNERBlock updated with use_parallel_scan option")
println()
println("To use parallel scan, set in config:")
println("  [parallelization]")
println("  use_parallel_scan = true")
println("  chunk_size = 64")
println()
println("Or programmatically:")
println("  config = NERConfig(..., use_parallel_scan=true)")
println("  model = OssammaNER(config)")
println()
