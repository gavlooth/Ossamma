#!/usr/bin/env julia
"""
Optimized CPU NER training with multi-threading.

Usage:
    julia --project=. -t auto scripts/train_ner_optimized.jl
    julia --project=. -t 8 scripts/train_ner_optimized.jl --config configs/ner_dev.toml

The -t flag enables multi-threading (auto uses all available cores).
"""

using Random
using Statistics
using Printf
using Dates
using Serialization
using LinearAlgebra

# Enable multi-threaded BLAS
BLAS.set_num_threads(Threads.nthreads())

# Load the main module
include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
using .Ossamma

using Lux
using Optimisers
using Zygote

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

function main()
    println("=" ^ 60)
    println("OssammaNER Optimized CPU Training")
    println("=" ^ 60)
    println("Time: ", Dates.now())
    println("Julia threads: ", Threads.nthreads())
    println("BLAS threads:  ", BLAS.get_num_threads())

    # =========================================================================
    # Load configuration
    # =========================================================================
    config_path = get(ARGS, 1, nothing)
    if config_path === nothing || !occursin("--config", config_path)
        # Use dev config by default
        config_path = joinpath(@__DIR__, "..", "configs", "ner_dev.toml")
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

    params = Lux.initialparameters(rng, model)
    state = Lux.initialstates(rng, model)

    actual_params = count_parameters(params)
    println("  Parameters: $(round(actual_params / 1e6, digits=2))M")

    # =========================================================================
    # Setup training
    # =========================================================================
    println("\n[2/5] Setting up training...")

    batch_size = 8
    seq_len = min(64, config.max_sequence_length)
    n_epochs = 5
    steps_per_epoch = 20
    lr = 1e-3

    println("  Batch size:       $batch_size")
    println("  Sequence length:  $seq_len")
    println("  Epochs:           $n_epochs")
    println("  Steps per epoch:  $steps_per_epoch")
    println("  Learning rate:    $lr")

    opt = Optimisers.AdamW(Float32(lr), (0.9f0, 0.999f0), 0.01f0)
    opt_state = Optimisers.setup(opt, params)

    # =========================================================================
    # Warmup
    # =========================================================================
    println("\n[3/5] Warmup (compiling)...")

    # Single warmup step to trigger JIT compilation
    token_ids = rand(rng, 1:config.vocab_size, seq_len, batch_size)
    labels = rand(rng, 1:config.num_labels, seq_len, batch_size)

    t0 = time()
    (loss, _), grads = Zygote.withgradient(params) do p
        (emissions, _), st = model(token_ids, p, state)
        l = ner_cross_entropy(emissions, labels)
        return l, st
    end
    opt_state, params = Optimisers.update(opt_state, params, grads[1])
    warmup_time = time() - t0
    println("  Warmup complete: $(round(warmup_time, digits=1))s")

    # =========================================================================
    # Training loop
    # =========================================================================
    println("\n[4/5] Training...")
    println("-" ^ 60)

    all_losses = Float64[]
    epoch_times = Float64[]

    for epoch in 1:n_epochs
        epoch_start = time()
        epoch_losses = Float64[]

        for step in 1:steps_per_epoch
            # Generate batch
            token_ids = rand(rng, 1:config.vocab_size, seq_len, batch_size)
            labels = rand(rng, 1:config.num_labels, seq_len, batch_size)
            labels[end-2:end, :] .= -100  # Padding

            # Training step
            (loss, _), grads = Zygote.withgradient(params) do p
                (emissions, _), st = model(token_ids, p, state)
                l = ner_cross_entropy(emissions, labels)
                return l, st
            end

            # Update parameters
            opt_state, params = Optimisers.update(opt_state, params, grads[1])

            push!(epoch_losses, loss)
            push!(all_losses, loss)
        end

        epoch_time = time() - epoch_start
        push!(epoch_times, epoch_time)

        avg_loss = mean(epoch_losses)
        samples_per_sec = (batch_size * steps_per_epoch) / epoch_time

        @printf("  Epoch %d/%d: avg_loss=%.4f, time=%.1fs, %.1f samples/s\n",
                epoch, n_epochs, avg_loss, epoch_time, samples_per_sec)
    end

    println("-" ^ 60)

    # =========================================================================
    # Summary
    # =========================================================================
    println("\n[5/5] Training complete!")
    println("=" ^ 60)

    total_time = sum(epoch_times)
    total_samples = batch_size * steps_per_epoch * n_epochs
    throughput = total_samples / total_time

    println("  Total time:      $(round(total_time, digits=1))s")
    println("  Total samples:   $total_samples")
    println("  Throughput:      $(round(throughput, digits=1)) samples/s")
    println()
    println("  Initial loss:    $(round(all_losses[1], digits=4))")
    println("  Final loss:      $(round(all_losses[end], digits=4))")

    if all_losses[1] > all_losses[end]
        reduction = (all_losses[1] - all_losses[end]) / all_losses[1] * 100
        println("  Loss reduction:  $(round(reduction, digits=1))%")
    end

    println()
    println("  Hardware:        CPU ($(Threads.nthreads()) threads)")

    # =========================================================================
    # Save checkpoint
    # =========================================================================
    checkpoint_dir = joinpath(@__DIR__, "..", "checkpoints", "ner_optimized")
    isdir(checkpoint_dir) || mkpath(checkpoint_dir)

    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    checkpoint_path = joinpath(checkpoint_dir, "checkpoint_$(timestamp).jls")

    serialize(checkpoint_path, Dict(
        :params => params,
        :state => state,
        :config => config,
        :final_loss => all_losses[end],
        :total_steps => n_epochs * steps_per_epoch,
    ))
    println("  Checkpoint:      $checkpoint_path")

    println("=" ^ 60)
    println("SUCCESS!")

    return true
end

# Run
main()
