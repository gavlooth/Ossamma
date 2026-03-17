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

if USE_GPU
    println("GPU: $(CUDA.name(CUDA.device())), $(round(CUDA.total_memory() / 1e9, digits=1))GB")
else
    println("GPU: not available, using CPU")
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

    # Teacher logits (already computed, sliced to match)
    teacher_slice = teacher_logits[:, 1:seq_len-1, :]

    # Temperature scaling
    d_scaled = drafter_logits ./ temperature
    t_scaled = teacher_slice ./ temperature

    # KL divergence: sum_v p_teacher(v) * [log p_teacher(v) - log p_drafter(v)]
    t_probs = NNlib.softmax(t_scaled, dims=1)
    d_logprobs = NNlib.logsoftmax(d_scaled, dims=1)
    t_logprobs = NNlib.logsoftmax(t_scaled, dims=1)

    # Mask padding
    mask = target_tokens .> 1
    mask_broadcast = reshape(mask, 1, size(mask)...)

    kl_per_position = sum(t_probs .* (t_logprobs .- d_logprobs), dims=1)  # (1, seq-1, batch)
    kl_masked = kl_per_position .* mask_broadcast
    n_valid = max(sum(mask), 1)

    loss = sum(kl_masked) / n_valid * temperature^2

    return loss, new_st
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
    seed::Int = 42,
)
    rng = Random.MersenneTwister(seed)
    mkpath(output_dir)

    println("=== Phase 3b: Granite Distillation ===")
    println("Drafter: $drafter_checkpoint")
    println("Granite: $granite_model_ref")
    println("Temperature: $temperature")

    # Load drafter
    println("Loading Phase 3a drafter checkpoint...")
    phase3a = JLD2.load(drafter_checkpoint)
    drafter_ps = phase3a["ps_cpu"]
    config = phase3a["config"]
    drafter_model = ReasoningDrafter(config)
    drafter_st = Lux.initialstates(rng, drafter_model)

    # Load Granite teacher
    println("Loading Granite model: $granite_model_ref")
    println("  (This may download model weights from HuggingFace)")
    granite_model, granite_ps, granite_st = load_granite_model(
        granite_model_ref;
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
        granite_ps = to_dev(granite_ps)
        granite_st = to_dev(granite_st)
    end

    # Load reasoning data
    println("Loading reasoning datasets...")
    examples = load_all_reasoning_datasets(data_dir; max_seq_length=max_seq_length)
    println("Total: $(length(examples)) examples")

    # Optimizer (same freeze strategy as Phase 3a)
    opt = Optimisers.Adam(learning_rate)
    opt_state = Optimisers.setup(opt, drafter_ps)

    global_step = 0
    best_loss = Inf

    for epoch in 1:num_epochs
        batches = iterate_reasoning_batches(examples, batch_size; shuffle=true, rng=rng)
        epoch_loss = 0.0

        for batch in batches
            global_step += 1

            b_tokens = to_dev(batch.tokens)

            if USE_GPU
                GC.gc(false)
            end

            # Teacher forward (no gradient — frozen)
            teacher_logits = Zygote.@ignore granite_forward(granite_model, granite_ps, granite_st, b_tokens)

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

            epoch_loss += Float64(loss_val)

            if global_step % 20 == 0
                println("step=$global_step  kl_loss=$(round(Float64(loss_val), digits=4))")
            end

            if global_step % checkpoint_every == 0
                cp_path = joinpath(output_dir, "step_$(global_step).jld2")
                ps_cpu = Lux.cpu(drafter_ps)
                JLD2.@save cp_path ps_cpu config global_step epoch
                println("Checkpoint: $cp_path")
            end
        end

        avg_loss = epoch_loss / max(length(batches), 1)
        println("--- Epoch $epoch/$num_epochs  avg_kl=$(round(avg_loss, digits=4)) ---")

        if avg_loss < best_loss
            best_loss = avg_loss
            best_path = joinpath(output_dir, "best.jld2")
            ps_cpu = Lux.cpu(drafter_ps)
            JLD2.@save best_path ps_cpu config global_step epoch
            println("  New best! Saved to $best_path")
        end
    end

    println("\n=== Phase 3b complete. Best KL loss: $(round(best_loss, digits=4)) ===")
end

# CLI
function main()
    drafter_cp = "checkpoints/reasoning_drafter/phase3a/best.jld2"
    granite_ref = "ibm-granite/granite-4.0-micro"
    data_dir = "data/reasoning"
    output_dir = "checkpoints/reasoning_drafter/phase3b"
    epochs = 5

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
        else
            i += 1
        end
    end

    train_phase3b(;
        drafter_checkpoint = drafter_cp,
        granite_model_ref = granite_ref,
        data_dir, output_dir,
        num_epochs = epochs,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
