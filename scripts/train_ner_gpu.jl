#!/usr/bin/env julia
"""
GPU-accelerated NER training using Intel Arc (oneAPI) or NVIDIA (CUDA).

Usage:
    julia --project=. scripts/train_ner_gpu.jl
    julia --project=. scripts/train_ner_gpu.jl --config configs/ner_dev.toml
"""

using Random
using Statistics
using Printf
using Dates

# Load the main module
include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
using .Ossamma

using Lux
using NNlib
using Optimisers
using Zygote

# Load GPU backends at top level
using oneAPI
using GPUArrays

# Detect GPU backend
const GPU_BACKEND = if oneAPI.functional()
    @info "Using Intel Arc GPU (oneAPI)"
    # Allow scalar indexing for now (model has some CPU-style operations)
    # TODO: Optimize model to avoid scalar indexing for full GPU performance
    GPUArrays.allowscalar(true)
    :oneAPI
else
    @warn "oneAPI not functional, using CPU"
    :cpu
end

# Device transfer functions
function to_device(x)
    if GPU_BACKEND == :oneAPI
        return oneAPI.oneArray(x)
    elseif GPU_BACKEND == :CUDA
        return CUDA.CuArray(x)
    else
        return x
    end
end

function to_cpu(x)
    if GPU_BACKEND == :oneAPI
        return Array(x)
    elseif GPU_BACKEND == :CUDA
        return Array(x)
    else
        return x
    end
end

function count_parameters(params)
    total = 0
    function count_nested(x)
        if x isa NamedTuple || x isa Tuple
            for v in values(x)
                count_nested(v)
            end
        elseif x isa AbstractArray
            total += length(x)
        end
    end
    count_nested(params)
    return total
end

# Transfer parameters to GPU
function params_to_device(params)
    if params isa NamedTuple
        return NamedTuple{keys(params)}(Tuple(params_to_device(v) for v in values(params)))
    elseif params isa Tuple
        return Tuple(params_to_device(v) for v in params)
    elseif params isa AbstractArray
        return to_device(params)
    else
        return params
    end
end

"""
Prepare one-hot targets and mask on GPU from CPU labels.
Call this OUTSIDE the gradient computation.
"""
function prepare_targets(labels_cpu, num_labels; ignore_index::Int = -100)
    seq_len, batch_size = size(labels_cpu)
    n_tokens = seq_len * batch_size

    labels_flat = vec(labels_cpu)

    # Create mask on CPU
    mask_cpu = Float32.(labels_flat .!= ignore_index)
    valid_count = sum(mask_cpu)

    # Clamp labels to valid range for one-hot
    safe_labels = clamp.(labels_flat, 1, num_labels)

    # Create one-hot encoding on CPU
    one_hot = zeros(Float32, num_labels, n_tokens)
    for i in 1:n_tokens
        one_hot[safe_labels[i], i] = 1.0f0
    end

    # Transfer to GPU
    one_hot_gpu = to_device(one_hot)
    mask_gpu = to_device(reshape(mask_cpu, 1, :))

    return one_hot_gpu, mask_gpu, valid_count
end

"""
GPU-compatible cross-entropy loss for NER.
Takes pre-computed one-hot targets and mask (both on GPU).
"""
function gpu_cross_entropy_with_targets(logits, one_hot_gpu, mask_gpu, valid_count, num_labels)
    if valid_count == 0
        return 0.0f0
    end

    # Flatten logits
    logits_flat = reshape(logits, num_labels, :)

    # Compute log softmax (on GPU)
    log_probs = NNlib.logsoftmax(logits_flat, dims=1)

    # Negative log likelihood: -sum(one_hot .* log_probs, dims=1)
    nll = -sum(one_hot_gpu .* log_probs, dims=1)

    # Apply mask and compute mean
    masked_loss = sum(nll .* mask_gpu) / valid_count

    return masked_loss
end

function main()
    println("=" ^ 60)
    println("OssammaNER GPU Training")
    println("=" ^ 60)
    println("Backend: ", GPU_BACKEND)
    println("Time: ", Dates.now())

    # =========================================================================
    # Load configuration
    # =========================================================================
    config_path = get(ARGS, 1, nothing)
    if config_path === nothing || !occursin("--config", config_path)
        # Use minimal config for GPU testing (due to driver limitations)
        config_path = joinpath(@__DIR__, "..", "configs", "ner_minimal.toml")
    else
        # Parse --config path
        idx = findfirst(x -> x == "--config", ARGS)
        if idx !== nothing && idx < length(ARGS)
            config_path = ARGS[idx + 1]
        end
    end

    println("\nLoading config: ", config_path)
    config = load_ner_config(config_path)
    print_config_summary(config)

    # =========================================================================
    # Create model
    # =========================================================================
    println("\n[1/5] Creating model...")
    model = OssammaNER(config)

    rng = Random.default_rng()
    Random.seed!(rng, 42)

    # Initialize on CPU first
    params_cpu = Lux.initialparameters(rng, model)
    state = Lux.initialstates(rng, model)

    actual_params = count_parameters(params_cpu)
    println("  Parameters: $(round(actual_params / 1e6, digits=2))M")

    # =========================================================================
    # Transfer to GPU
    # =========================================================================
    println("\n[2/5] Transferring to GPU...")
    t0 = time()

    if GPU_BACKEND != :cpu
        params = params_to_device(params_cpu)
        println("  Transfer time: $(round(time() - t0, digits=2))s")
        println("  Model on GPU: YES")
    else
        params = params_cpu
        println("  Running on CPU")
    end

    # =========================================================================
    # Setup training
    # =========================================================================
    println("\n[3/5] Setting up training...")

    batch_size = 2  # Reduced for GPU driver stability
    seq_len = min(16, config.max_sequence_length)  # Reduced
    n_steps = 20
    lr = 1e-3

    println("  Batch size: $batch_size")
    println("  Sequence length: $seq_len")
    println("  Training steps: $n_steps")
    println("  Learning rate: $lr")

    opt = Optimisers.AdamW(Float32(lr), (0.9f0, 0.999f0), 0.01f0)
    opt_state = Optimisers.setup(opt, params)

    # =========================================================================
    # Training loop
    # =========================================================================
    println("\n[4/5] Training...")
    println("-" ^ 60)

    losses = Float64[]
    times = Float64[]

    for step in 1:n_steps
        t_step = time()

        # Generate batch (on CPU, then transfer)
        token_ids_cpu = rand(rng, 1:config.vocab_size, seq_len, batch_size)
        labels_cpu = rand(rng, 1:config.num_labels, seq_len, batch_size)
        labels_cpu[end-2:end, :] .= -100  # Padding

        # Transfer tokens to GPU
        token_ids = to_device(token_ids_cpu)

        # Prepare one-hot targets on GPU (outside gradient)
        one_hot_gpu, mask_gpu, valid_count = prepare_targets(labels_cpu, config.num_labels)

        # Training step - everything stays on GPU during gradient computation
        (loss, _), grads = Zygote.withgradient(params) do p
            (emissions, _), st = model(token_ids, p, state)
            # GPU-compatible loss function with pre-computed targets
            l = gpu_cross_entropy_with_targets(emissions, one_hot_gpu, mask_gpu, valid_count, config.num_labels)
            return l, st
        end

        # Update parameters
        opt_state, params = Optimisers.update(opt_state, params, grads[1])

        step_time = time() - t_step
        push!(losses, loss)
        push!(times, step_time)

        if step == 1 || step % 10 == 0 || step == n_steps
            avg_time = mean(times)
            samples_per_sec = batch_size / avg_time
            @printf("  Step %3d/%d: loss=%.4f, time=%.2fs, %.1f samples/s\n",
                    step, n_steps, loss, step_time, samples_per_sec)
        end
    end

    println("-" ^ 60)

    # =========================================================================
    # Summary
    # =========================================================================
    println("\n[5/5] Training complete!")
    println("=" ^ 60)

    total_time = sum(times)
    avg_time = mean(times)
    throughput = batch_size * n_steps / total_time

    println("  Total time:      $(round(total_time, digits=1))s")
    println("  Avg step time:   $(round(avg_time * 1000, digits=1))ms")
    println("  Throughput:      $(round(throughput, digits=1)) samples/s")
    println()
    println("  Initial loss:    $(round(losses[1], digits=4))")
    println("  Final loss:      $(round(losses[end], digits=4))")
    println("  Loss reduction:  $(round((losses[1] - losses[end]) / losses[1] * 100, digits=1))%")
    println()

    if GPU_BACKEND != :cpu
        println("  GPU Backend:     $GPU_BACKEND")
        println("  GPU Acceleration: ENABLED")
    else
        println("  Running on:      CPU")
    end

    println("=" ^ 60)
    println("SUCCESS!")

    return true
end

# Run
main()
