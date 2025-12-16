#!/usr/bin/env julia
"""
Quick trainability test for Ossamma models.

Tests:
1. Model instantiation
2. Forward pass
3. Backward pass (gradient computation)
4. Training step

Usage:
    julia --project=. scripts/test_trainability.jl
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

println("=" ^ 60)
println("TRAINABILITY TEST - Ossamma")
println("=" ^ 60)
println()

using Random
using Lux
using Zygote
using NNlib
using Optimisers

include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
using .Ossamma

# ============================================================================
# Test Configuration
# ============================================================================

# Use small config for fast testing, or production_config() for full test
const USE_PRODUCTION = get(ENV, "TEST_PRODUCTION", "false") == "true"

println("[0/5] Configuration")
if USE_PRODUCTION
    println("  Using PRODUCTION config (slow but realistic)")
    config = production_config()
else
    println("  Using SMALL config (fast test)")
    config = small_config()
end
println()

# ============================================================================
# Test 1: Model Creation
# ============================================================================

println("[1/5] Creating model...")
model = LLaDAModel(config)

rng = Random.MersenneTwister(42)
ps, st = Lux.setup(rng, model)

# Count parameters
function count_params(p)
    total = 0
    for (_, v) in pairs(p)
        if v isa AbstractArray
            total += length(v)
        elseif v isa NamedTuple
            total += count_params(v)
        end
    end
    return total
end

param_count = count_params(ps)
println("  Layers: $(config.number_of_layers)")
println("  Embedding dim: $(config.embedding_dimension)")
println("  Heads: $(config.number_of_heads)")
println("  Vocab size: $(model.vocab_size)")
println("  Parameters: $(round(param_count / 1e6, digits=2))M")
println("  ✓ Model created successfully")
println()

# ============================================================================
# Test 2: Forward Pass
# ============================================================================

println("[2/5] Testing forward pass...")

batch_size = 2
seq_len = min(config.max_sequence_length, 64)  # Use shorter seq for speed
vocab_size = model.vocab_size
mask_token_id = model.mask_token_id

# Create dummy batch
token_ids = rand(rng, 1:vocab_size-1, seq_len, batch_size)
mask_ratio = Float32(0.5)

# Apply masking
mask = rand(rng, Float32, size(token_ids)) .< mask_ratio
masked_ids = ifelse.(mask, mask_token_id, token_ids)

inputs = (token_ids = masked_ids, mask_ratio = mask_ratio)

forward_time = @elapsed begin
    logits, new_st = model(inputs, ps, st)
end

println("  Input shape: (seq=$seq_len, batch=$batch_size)")
println("  Output shape: $(size(logits))")
println("  Time: $(round(forward_time, digits=3))s")

expected_shape = (vocab_size, seq_len, batch_size)
if size(logits) != expected_shape
    error("Shape mismatch! Expected $expected_shape, got $(size(logits))")
end
println("  ✓ Forward pass successful")
println()

# ============================================================================
# Test 3: Backward Pass
# ============================================================================

println("[3/5] Testing backward pass (gradient computation)...")

function compute_loss(ps, model, inputs, targets, mask, st)
    logits, _ = model(inputs, ps, st)

    # Cross-entropy on masked positions (Zygote-compatible)
    vocab_size, seq_len, batch_size = size(logits)
    log_probs = NNlib.logsoftmax(logits, dims=1)

    # Create one-hot encoding outside gradient tape
    targets_onehot = Zygote.@ignore begin
        oh = zeros(Float32, vocab_size, seq_len, batch_size)
        for b in 1:batch_size
            for s in 1:seq_len
                t = targets[s, b]
                if 1 <= t <= vocab_size
                    oh[t, s, b] = Float32(1.0)
                end
            end
        end
        oh
    end

    # Differentiable part: multiply and sum
    target_log_probs = dropdims(sum(log_probs .* targets_onehot, dims=1), dims=1)

    mask_float = Float32.(mask)
    n_masked = sum(mask_float)
    loss = n_masked > 0 ? -sum(target_log_probs .* mask_float) / n_masked : Float32(0.0)
    return loss
end

backward_time = @elapsed begin
    loss, grads = Zygote.withgradient(
        p -> compute_loss(p, model, inputs, token_ids, mask, st),
        ps
    )
end

println("  Loss: $(round(loss, digits=4))")
println("  Time: $(round(backward_time, digits=3))s")

# Check gradients are finite
function check_grads(g, path="")
    for (k, v) in pairs(g)
        new_path = isempty(path) ? string(k) : "$path.$k"
        if v isa AbstractArray
            if any(isnan, v) || any(isinf, v)
                error("NaN/Inf gradient at $new_path")
            end
        elseif v isa NamedTuple
            check_grads(v, new_path)
        end
    end
end

check_grads(grads[1])
println("  ✓ Gradients computed successfully (no NaN/Inf)")
println()

# ============================================================================
# Test 4: Optimizer Step
# ============================================================================

println("[4/5] Testing optimizer step...")

optimizer = Optimisers.Adam(Float32(1e-4))
opt_state = Optimisers.setup(optimizer, ps)

step_time = @elapsed begin
    opt_state, ps_new = Optimisers.update(opt_state, ps, grads[1])
end

println("  Time: $(round(step_time, digits=3))s")
println("  ✓ Optimizer step successful")
println()

# ============================================================================
# Test 5: Full Training Step
# ============================================================================

println("[5/5] Testing full training step...")

# Use the Training module's train_step!
train_state = create_train_state(model, optimizer; rng=rng)

full_step_time = @elapsed begin
    step_loss = train_step!(
        train_state, model, token_ids, mask_token_id;
        rng=rng, schedule=:cosine
    )
end

println("  Loss: $(round(step_loss, digits=4))")
println("  Time: $(round(full_step_time, digits=3))s")
println("  ✓ Full training step successful")
println()

# ============================================================================
# Summary
# ============================================================================

println("=" ^ 60)
println("✓ ALL TESTS PASSED - MODEL IS TRAINABLE!")
println("=" ^ 60)
println()
println("Performance Summary:")
println("  Forward pass:  $(round(forward_time * 1000, digits=1))ms")
println("  Backward pass: $(round(backward_time * 1000, digits=1))ms")
println("  Optimizer:     $(round(step_time * 1000, digits=1))ms")
println("  Full step:     $(round(full_step_time * 1000, digits=1))ms")
println()
println("Ready to train with:")
println("  julia --project=. scripts/train_colab.jl")
