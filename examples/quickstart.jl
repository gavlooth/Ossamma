#!/usr/bin/env julia
"""
Quickstart example for LLaDA text diffusion model.

Demonstrates:
1. Loading a model from config
2. Forward pass
3. Masking and unmasking
4. Generation
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Lux

# Load module
include(joinpath(@__DIR__, "..", "src", "Swamma.jl"))
using .Swamma

println("LLaDA Quickstart Example")
println("=" ^ 40)

# ============================================================================
# 1. Create Model from Config
# ============================================================================
println("\n1. Creating model...")

# Option A: From preset config
config = small_config()

# Option B: From TOML file
# config = load_config("configs/small.toml")

# Option C: Custom config
# config = LLaDAConfig(
#     vocab_size = 500,
#     max_sequence_length = 32,
#     embedding_dimension = 64,
#     number_of_heads = 2,
#     number_of_layers = 2,
# )

model = LLaDAModel(config)
println("   Model created with $(config.number_of_layers) layers")

# Initialize parameters
rng = Random.default_rng()
Random.seed!(rng, 42)
params, state = Lux.setup(rng, model)
println("   Parameters initialized")

# ============================================================================
# 2. Forward Pass
# ============================================================================
println("\n2. Forward pass...")

# Create dummy input
seq_len = 32
batch_size = 2
token_ids = rand(rng, 1:config.vocab_size, seq_len, batch_size)
mask_ratio = 0.5f0  # 50% masked

println("   Input shape: (seq=$seq_len, batch=$batch_size)")
println("   Mask ratio: $mask_ratio")

# PRIME sub-token masking
subtoken_state = token_ids_to_subtokens(token_ids, model.prime_code_table)
masked_subtokens, subtoken_mask, token_mask = apply_subtoken_mask(
    subtoken_state, mask_ratio, model.prime_mask_subtoken_id; rng = rng
)
println("   Masked subtokens: $(sum(subtoken_mask)) / $(length(subtoken_mask))")
println("   Masked tokens: $(sum(token_mask)) / $(length(token_mask))")

# Forward pass
inputs = (subtoken_state = masked_subtokens, mask_ratio = mask_ratio)
logits, new_state = model(inputs, params, state)
println("   Output logits shape: $(size(logits))")

# ============================================================================
# 3. Compute Loss (for training)
# ============================================================================
println("\n3. Computing loss...")

loss, _ = diffusion_loss(model, params, state, token_ids; rng=rng)
println("   Diffusion loss: $(round(loss, digits=4))")

# ============================================================================
# 4. Generation (Iterative Denoising)
# ============================================================================
println("\n4. Generation...")

generated = generate(
    model, params, state, 16;  # Generate 16 tokens
    num_steps = 5,
    batch_size = 1,
    rng = rng,
)

println("   Generated $(length(generated)) tokens:")
println("   $generated")

# ============================================================================
# 5. Step-by-step Unmasking (Manual Control)
# ============================================================================
println("\n5. Manual unmasking demo...")

# Start with fully masked subtokens for 8 positions.
current_subtokens = fill(model.prime_mask_subtoken_id, model.prime_subtoken_length, 8, 1)
remaining_masks() = [
    count(==(model.prime_mask_subtoken_id), @view current_subtokens[:, i, 1]) for i in 1:8
]
println("   Start masked-subtokens-per-position: $(remaining_masks())")

# Unmask in steps
for step in 1:4
    t = 1.0f0 - (step - 1) / 4
    inputs = (subtoken_state = current_subtokens, mask_ratio = t)
    logits, state = model(inputs, params, state)

    # Reveal 2 subtokens per step.
    current_subtokens = unmask_subtoken_step(
        logits, current_subtokens, 2,
        model.prime_code_table, model.prime_mask_subtoken_id
    )

    println("   Step $step masked-subtokens-per-position: $(remaining_masks())")
end

# ============================================================================
# 6. Save/Load Config
# ============================================================================
println("\n6. Config save/load...")

# Save config to file
config_path = joinpath(@__DIR__, "..", "configs", "test_output.toml")
save_config(config, config_path)
println("   Saved config to: $config_path")

# Load it back
loaded_config = load_config(config_path)
println("   Loaded config: vocab=$(loaded_config.vocab_size), layers=$(loaded_config.number_of_layers)")

# Cleanup
rm(config_path)

println("\n" * "=" ^ 40)
println("Quickstart complete!")
