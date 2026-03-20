#!/usr/bin/env julia
"""
Phase 3b: Granite Distillation

Distills the reasoning drafter to match Granite's output distribution
on reasoning-heavy prompts. Uses KL divergence between drafter and
verifier logits.

The drafter's chess-learned reasoning backbone stays frozen. Only adapters,
codebook, and embeddings are updated — same freeze strategy as Phase 3a.

This is the final training step before deployment in the speculative
decoding pipeline (TiDAR).

Requires:
  - Granite model weights (downloaded via HuggingFace)
  - Phase 3a checkpoint (language-adapted drafter)
  - PyCall + safetensors Python packages for Granite weight loading

CUDA/GPU rules: no try/catch in loops, println() only, grads=nothing after step.

Usage:
  julia --project=. scripts/distill_granite.jl \
    --drafter-checkpoint checkpoints/reasoning_drafter/phase3a/best.jld2 \
    --granite-model ibm-granite/granite-4.0-micro \
    --data-dir data/reasoning \
    --output-dir checkpoints/reasoning_drafter/phase3b
"""

using Swamma
using Swamma.ReasoningDrafterMod
using Swamma.ReasoningDrafterMod: apply_reasoning_drafter_ema_codebook!
using Swamma.RuleConditionedWavePDEMod
using Swamma.ReasoningDataset
using Swamma.NativeTeacherLM

using Lux
using Random
using NNlib
using Optimisers
using Zygote
using JLD2

const USE_GPU = try
    using CUDA
    CUDA.functional()
catch
    false
end

if !isdefined(@__MODULE__, :_SWAMMA_GPU_BANNER_PRINTED)
    global _SWAMMA_GPU_BANNER_PRINTED = false
end

if !_SWAMMA_GPU_BANNER_PRINTED
    if USE_GPU
        println("GPU: $(CUDA.name(CUDA.device())), $(round(CUDA.total_memory() / 1e9, digits=1))GB")
    else
        println("GPU: not available, using CPU")
    end
    global _SWAMMA_GPU_BANNER_PRINTED = true
end

to_dev(x) = USE_GPU ? CUDA.cu(x) : x

# Import freeze logic from Phase 3a
include("train_reasoning_language.jl")

# ============================================================================
# Granite teacher forward
# ============================================================================

"""
Run Granite on a batch of token sequences, return logits.
Granite expects (seq_len, batch) integer token IDs.
"""
function granite_forward(granite_model, granite_ps, granite_st, tokens)
    logits, _ = granite_model(tokens, granite_ps, granite_st)
    return logits  # (vocab, seq, batch)
end

function maybe_fallback_teacher_to_cpu(
    granite_model,
    granite_ps,
    granite_st,
    requested_device::Symbol,
    probe_tokens,
)
    if !(USE_GPU && requested_device == :gpu)
        return granite_ps, granite_st, requested_device
    end

    try
        teacher_logits = Zygote.@ignore granite_forward(granite_model, granite_ps, granite_st, probe_tokens)
        teacher_logits = nothing
        return granite_ps, granite_st, requested_device
    catch err
        println("WARNING: Granite GPU teacher probe failed; falling back to CPU")
        println("  Probe error: $(sprint(showerror, err))")
        return cpu_device()(granite_ps), cpu_device()(granite_st), :cpu
    end
end

# ============================================================================
# Distillation loss
# ============================================================================

"""
KL(teacher || student) on next-token predictions.
Only computed on non-padding positions.
"""
function distillation_kl_loss(
    drafter, drafter_ps, drafter_st,
    teacher_logits,
    tokens;
    temperature::Float32 = 2.0f0,
)
    seq_len = size(tokens, 1)
    input_tokens = tokens[1:seq_len-1, :]
    target_tokens = tokens[2:seq_len, :]

    # Drafter logits
    drafter_logits, new_st = drafter(input_tokens, drafter_ps, drafter_st)

    # Slice both models to the shared active vocab when checkpoint and teacher differ.
    shared_vocab = min(size(drafter_logits, 1), size(teacher_logits, 1))
    drafter_slice = drafter_logits[1:shared_vocab, :, :]
    teacher_slice = teacher_logits[1:shared_vocab, 1:seq_len-1, :]
    if USE_GPU
        teacher_slice = to_dev(teacher_slice)
    end

    # Temperature scaling
    d_scaled = drafter_slice ./ temperature
    t_scaled = teacher_slice ./ temperature

    # KL divergence: sum_v p_teacher(v) * [log p_teacher(v) - log p_drafter(v)]
    t_probs = NNlib.softmax(t_scaled, dims=1)
    d_logprobs = NNlib.logsoftmax(d_scaled, dims=1)
    t_logprobs = NNlib.logsoftmax(t_scaled, dims=1)

    # Mask padding
    target_mask = Float32.(target_tokens .> 1)
    mask_broadcast = reshape(target_mask, 1, size(target_mask)...)

    kl_per_position = sum(t_probs .* (t_logprobs .- d_logprobs), dims=1)  # (1, seq-1, batch)
    kl_masked = kl_per_position .* mask_broadcast
    n_valid = max(sum(target_mask), 1)

    loss = sum(kl_masked) / n_valid * temperature^2

    return loss, new_st
end

function _save_distill_checkpoint(path, drafter_ps, drafter_st, opt_state, config, global_step, epoch; best_loss = nothing)
    ps_cpu = cpu_device()(drafter_ps)
    st_cpu = cpu_device()(drafter_st)
    opt_state_cpu = _optimizer_state_to_cpu(opt_state)
    training_stage = "phase3b_distill"
    if best_loss === nothing
        JLD2.@save path ps_cpu st_cpu opt_state_cpu config global_step epoch training_stage
    else
        JLD2.@save path ps_cpu st_cpu opt_state_cpu config global_step epoch best_loss training_stage
    end
end

function _load_phase3b_state(checkpoint_path::String, rng::Random.AbstractRNG, learning_rate::Float32)
    ckpt = JLD2.load(checkpoint_path)
    drafter_ps = ckpt["ps_cpu"]
    config = _coerce_reasoning_drafter_config(ckpt["config"])
    drafter_model = ReasoningDrafter(config)
    drafter_st = haskey(ckpt, "st_cpu") ? ckpt["st_cpu"] : Lux.initialstates(rng, drafter_model)
    opt = Optimisers.Adam(learning_rate)
    stage = get(ckpt, "training_stage", "")
    resumed = stage == "phase3b_distill" && (haskey(ckpt, "st_cpu") || haskey(ckpt, "opt_state_cpu"))
    opt_state = resumed && haskey(ckpt, "opt_state_cpu") ? ckpt["opt_state_cpu"] : Optimisers.setup(opt, drafter_ps)
    global_step = resumed && haskey(ckpt, "global_step") ? ckpt["global_step"] : 0
    start_epoch = resumed && haskey(ckpt, "epoch") ? ckpt["epoch"] : 0
    best_loss = resumed && haskey(ckpt, "best_loss") ? ckpt["best_loss"] : Inf
    return (;
        drafter_ps,
        config,
        drafter_model,
        drafter_st,
        opt_state,
        global_step,
        start_epoch,
        best_loss,
        resumed,
    )
end

# ============================================================================
# Training
# ============================================================================

function train_phase3b(;
    drafter_checkpoint::String,
    granite_model_ref::String,
    data_dir::String,
    output_dir::String,
    batch_size::Int = 16,
    num_epochs::Int = 5,
    learning_rate::Float32 = 1f-4,
    max_seq_length::Int = 256,
    temperature::Float32 = 2.0f0,
    checkpoint_every::Int = 200,
    max_per_dataset::Union{Int,Nothing} = nothing,
    max_steps::Union{Int,Nothing} = nothing,
    log_every::Int = 20,
    local_files_only::Bool = false,
    teacher_device::Symbol = :cpu,
    seed::Int = 42,
)
    rng = Random.MersenneTwister(seed)
    mkpath(output_dir)

    println("=== Phase 3b: Granite Distillation ===")
    println("Drafter: $drafter_checkpoint")
    println("Granite: $granite_model_ref")
    println("Temperature: $temperature")

    # Load drafter
    println("Loading drafter checkpoint...")
    loaded = _load_phase3b_state(drafter_checkpoint, rng, learning_rate)
    drafter_ps = loaded.drafter_ps
    config = loaded.config
    drafter_model = loaded.drafter_model
    drafter_st = loaded.drafter_st
    opt_state = loaded.opt_state
    global_step = loaded.global_step
    start_epoch = loaded.start_epoch
    best_loss = loaded.best_loss
    println(loaded.resumed ? "  Mode: resume Phase 3b state" : "  Mode: initialize from Phase 3a params")

    # Load Granite teacher
    println("Loading Granite model: $granite_model_ref")
    println(local_files_only ? "  Using local_files_only=true" : "  Remote downloads allowed")
    println("  Teacher device: $(teacher_device)")
    local granite_model_dir
    try
        granite_model_dir = resolve_hf_model_dir(granite_model_ref; local_files_only = local_files_only)
    catch err
        throw(ArgumentError(
            "Could not resolve Granite teacher $(repr(granite_model_ref)). " *
            (local_files_only ?
                "The model is not available in the local HuggingFace cache. Disable local_files_only or pre-download it first." :
                "Teacher resolution failed before distillation started.") *
            " Underlying error: $(sprint(showerror, err))"
        ))
    end
    granite_model, granite_ps, granite_st = load_granite_model(
        granite_model_dir;
        rng = rng,
        local_files_only = false,
    )
    println("  Granite loaded: $(granite_model.config.number_of_layers) layers, vocab=$(granite_model.config.vocab_size)")

    # Verify vocab match
    if config.vocab_size != granite_model.config.vocab_size
        println("WARNING: Drafter vocab $(config.vocab_size) != Granite vocab $(granite_model.config.vocab_size)")
        println("  Distillation will use min(drafter, granite) vocab for KL computation")
    end

    if USE_GPU
        drafter_ps = to_dev(drafter_ps)
        drafter_st = to_dev(drafter_st)
        opt_state = _optimizer_state_to_device(opt_state)
        if teacher_device == :gpu
            granite_ps = to_dev(granite_ps)
            granite_st = to_dev(granite_st)
        end
    end

    # Load reasoning data
    println("Loading reasoning datasets...")
    examples = load_all_reasoning_datasets(data_dir; max_per_dataset=max_per_dataset, max_seq_length=max_seq_length)
    println("Total: $(length(examples)) examples")
    isempty(examples) && throw(ArgumentError("No reasoning examples loaded from $data_dir"))

    teacher_device_runtime = teacher_device
    if USE_GPU && teacher_device_runtime == :gpu
        probe_batch = first(iterate_reasoning_batches(examples, 1; shuffle=false, rng=rng))
        probe_tokens = to_dev(probe_batch.tokens)
        granite_ps, granite_st, teacher_device_runtime = maybe_fallback_teacher_to_cpu(
            granite_model,
            granite_ps,
            granite_st,
            teacher_device_runtime,
            probe_tokens,
        )
    end

    steps_run = 0
    stop_requested = false

    for epoch in 1:num_epochs
        current_epoch = start_epoch + epoch
        batches = iterate_reasoning_batches(examples, batch_size; shuffle=true, rng=rng)
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in batches
            if max_steps !== nothing && steps_run >= max_steps
                println("Reached max_steps=$max_steps; stopping Phase 3b early.")
                stop_requested = true
                break
            end
            global_step += 1
            steps_run += 1
            epoch_steps += 1

            b_tokens = to_dev(batch.tokens)
            teacher_tokens = teacher_device_runtime == :gpu ? b_tokens : batch.tokens

            if USE_GPU
                GC.gc(false)
            end

            # Teacher forward (no gradient — frozen)
            teacher_logits = Zygote.@ignore granite_forward(granite_model, granite_ps, granite_st, teacher_tokens)

            # Drafter gradient
            (loss_val, new_st), grads = Zygote.withgradient(drafter_ps) do p
                distillation_kl_loss(drafter_model, p, drafter_st, teacher_logits, b_tokens;
                    temperature=temperature)
            end
            grads = grads[1]

            # Apply freeze mask (same as Phase 3a)
            zero_frozen_grads!(grads, config)

            opt_state, drafter_ps = Optimisers.update(opt_state, drafter_ps, grads)
            grads = nothing
            drafter_st = new_st
            apply_reasoning_drafter_ema_codebook!(drafter_ps, drafter_st, drafter_model)

            epoch_loss += Float64(loss_val)

            if global_step % log_every == 0
                println("step=$global_step  kl_loss=$(round(Float64(loss_val), digits=4))")
            end

            if global_step % checkpoint_every == 0
                cp_path = joinpath(output_dir, "checkpoint_last.jld2")
                _save_distill_checkpoint(cp_path, drafter_ps, drafter_st, opt_state, config, global_step, current_epoch; best_loss = best_loss)
                println("Checkpoint: $cp_path")
            end
        end

        epoch_steps == 0 && (stop_requested && break)
        avg_loss = epoch_loss / max(epoch_steps, 1)
        println("--- Epoch $current_epoch  avg_kl=$(round(avg_loss, digits=4)) ---")

        if avg_loss < best_loss
            best_loss = avg_loss
            best_path = joinpath(output_dir, "best.jld2")
            _save_distill_checkpoint(best_path, drafter_ps, drafter_st, opt_state, config, global_step, current_epoch; best_loss = best_loss)
            println("  New best! Saved to $best_path")
        end

        cp_path = joinpath(output_dir, "checkpoint_last.jld2")
        _save_distill_checkpoint(cp_path, drafter_ps, drafter_st, opt_state, config, global_step, current_epoch; best_loss = best_loss)

        stop_requested && break
    end

    println("\n=== Phase 3b complete. Best KL loss: $(round(best_loss, digits=4)) ===")
    return (
        best_loss = best_loss,
        global_step = global_step,
        steps_run = steps_run,
        num_examples = length(examples),
    )
end

# CLI
function main()
    drafter_cp = "checkpoints/reasoning_drafter/phase3a/best.jld2"
    granite_ref = "ibm-granite/granite-4.0-micro"
    data_dir = "data/reasoning"
    output_dir = "checkpoints/reasoning_drafter/phase3b"
    epochs = 5
    batch_size = 16
    learning_rate = 1f-4
    max_seq_length = 256
    temperature = 2.0f0
    checkpoint_every = 200
    max_per_dataset = nothing
    max_steps = nothing
    log_every = 20
    local_files_only = false
    teacher_device = :cpu
    seed = 42

    args = ARGS
    i = 1
    while i <= length(args)
        if args[i] == "--drafter-checkpoint" && i < length(args)
            drafter_cp = args[i+1]; i += 2
        elseif args[i] == "--granite-model" && i < length(args)
            granite_ref = args[i+1]; i += 2
        elseif args[i] == "--data-dir" && i < length(args)
            data_dir = args[i+1]; i += 2
        elseif args[i] == "--output-dir" && i < length(args)
            output_dir = args[i+1]; i += 2
        elseif args[i] == "--epochs" && i < length(args)
            epochs = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--batch-size" && i < length(args)
            batch_size = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--learning-rate" && i < length(args)
            learning_rate = parse(Float32, args[i+1]); i += 2
        elseif args[i] == "--max-seq-length" && i < length(args)
            max_seq_length = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--temperature" && i < length(args)
            temperature = parse(Float32, args[i+1]); i += 2
        elseif args[i] == "--checkpoint-every" && i < length(args)
            checkpoint_every = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--max-per-dataset" && i < length(args)
            max_per_dataset = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--max-steps" && i < length(args)
            max_steps = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--log-every" && i < length(args)
            log_every = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--local-files-only" && i < length(args)
            local_files_only = parse(Bool, args[i+1]); i += 2
        elseif args[i] == "--teacher-device" && i < length(args)
            teacher_device = Symbol(args[i+1]); i += 2
        elseif args[i] == "--seed" && i < length(args)
            seed = parse(Int, args[i+1]); i += 2
        else
            i += 1
        end
    end

    train_phase3b(;
        drafter_checkpoint = drafter_cp,
        granite_model_ref = granite_ref,
        data_dir, output_dir,
        batch_size = batch_size,
        num_epochs = epochs,
        learning_rate = learning_rate,
        max_seq_length = max_seq_length,
        temperature = temperature,
        checkpoint_every = checkpoint_every,
        max_per_dataset = max_per_dataset,
        max_steps = max_steps,
        log_every = log_every,
        local_files_only = local_files_only,
        teacher_device = teacher_device,
        seed = seed,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
