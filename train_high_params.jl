#!/usr/bin/env julia
"""
High-parameter training script for LLaDA on Google Colab L4 GPU.
"""

using Pkg
Pkg.activate("/content/Ossamma")

# ============================================================================
# Configuration - HIGH PARAMETERS for L4 GPU (23GB VRAM)
# ============================================================================

const CONFIG = (
    # Dataset - load lots of data
    dataset = :tinystories,
    num_train_rows = 100000,   # 100k training samples
    num_val_rows = 5000,       # 5k validation samples

    # Model - BASE config (fits in 23GB)
    model_size = :base,
    vocab_size = 10000,        # Character-level vocab

    # Training - high capacity
    seq_length = 256,          # Longer sequences
    batch_size = 32,           # Good batch size for L4
    learning_rate = Float32(3e-4),    # Slightly higher LR
    warmup_steps = 1000,
    total_steps = 20000,       # Train for 20k steps
    eval_every = 1000,
    log_every = 100,

    # Output
    save_checkpoints = true,
    checkpoint_dir = "/content/Ossamma/checkpoints",
)

# ============================================================================
# Setup
# ============================================================================

println("=" ^ 70)
println("LLaDA High-Parameter Training on Google Colab")
println("=" ^ 70)
println()

# Check CUDA
using CUDA
if CUDA.functional()
    println("✓ CUDA available: $(CUDA.name(CUDA.device()))")
    println("  Memory: $(round(CUDA.total_memory() / 1e9, digits=2)) GB")
else
    println("⚠ CUDA not available, using CPU")
end
println()

# Load the Ossamma package (this includes LLaDA, Training, etc.)
using Ossamma
using Random
using Optimisers
using Lux

# We still need DataLoader separately since it's not exported fully
include("/content/Ossamma/src/DataLoader.jl")
using .DataLoader

# Set random seed
rng = Random.MersenneTwister(42)

# ============================================================================
# Load Data
# ============================================================================

println("[1/4] Loading TinyStories dataset...")
println("  Train rows: $(CONFIG.num_train_rows)")
println("  Val rows: $(CONFIG.num_val_rows)")
println()

train_loader, val_loader, tokenizer = prepare_tinystories(;
    num_train_rows = CONFIG.num_train_rows,
    num_val_rows = CONFIG.num_val_rows,
    seq_length = CONFIG.seq_length,
    batch_size = CONFIG.batch_size,
    max_vocab_size = CONFIG.vocab_size,
    rng = rng,
)

actual_vocab_size = get_vocab_size(tokenizer)
mask_token_id = get_mask_token_id(tokenizer) + 1  # +1 for Julia 1-indexing

println("\n✓ Dataset loaded!")
println("  Vocab size: $actual_vocab_size")
println("  Mask token ID: $mask_token_id")
println("  Train batches: $(length(train_loader))")
println("  Val batches: $(length(val_loader))")
println()

# ============================================================================
# Create Model - BASE Config
# ============================================================================

println("[2/4] Creating BASE model...")

# BASE config: 512 dim, 8 heads, 12 layers
model_config = LLaDAConfig(
    vocab_size = actual_vocab_size,
    max_sequence_length = CONFIG.seq_length,
    embedding_dimension = 512,
    number_of_heads = 8,
    number_of_layers = 12,
    time_dimension = 128,
    state_dimension = 512,
    mask_token_id = mask_token_id,
)

model = LLaDAModel(model_config)

# Initialize
params, state = Lux.setup(rng, model)

println("✓ Model created!")
println("  Embedding dim: $(model_config.embedding_dimension)")
println("  Layers: $(model_config.number_of_layers)")
println("  Heads: $(model_config.number_of_heads)")
println("  Sequence length: $(model_config.max_sequence_length)")
println()

# ============================================================================
# Setup Training
# ============================================================================

println("[3/4] Setting up training...")

optimizer = Optimisers.Adam(CONFIG.learning_rate)
train_state = create_train_state(model, optimizer; rng=rng)

train_config = TrainingConfig(
    batch_size = CONFIG.batch_size,
    learning_rate = CONFIG.learning_rate,
    min_learning_rate = Float32(1e-6),
    warmup_steps = CONFIG.warmup_steps,
    total_steps = CONFIG.total_steps,
    eval_every = CONFIG.eval_every,
    log_every = CONFIG.log_every,
    mask_schedule = :cosine,
    gradient_clip = Float32(1.0),
)

# Setup checkpoint directory
if CONFIG.save_checkpoints
    mkpath(CONFIG.checkpoint_dir)
end

callbacks = Dict{Symbol, Function}()
callbacks[:on_save] = function(ts)
    println("  [Checkpoint] Step $(ts.step), Best Loss $(round(ts.best_loss, digits=4))")
end

println("✓ Training configured!")
println("  Total steps: $(train_config.total_steps)")
println("  Batch size: $(train_config.batch_size)")
println("  Learning rate: $(train_config.learning_rate)")
println("  Warmup steps: $(train_config.warmup_steps)")
println()

# ============================================================================
# Train!
# ============================================================================

println("[4/4] Starting training...")
println("=" ^ 70)
println()

start_time = time()

train!(
    model,
    train_state,
    train_loader,
    train_config;
    val_data = val_loader,
    callbacks = callbacks,
    rng = rng,
)

elapsed = time() - start_time
println()
println("=" ^ 70)
println("✓ Training complete!")
println("  Total time: $(round(elapsed / 60, digits=1)) minutes")
println("  Final step: $(train_state.step)")
println("  Best validation loss: $(round(train_state.best_loss, digits=4))")
println("=" ^ 70)

# ============================================================================
# Generate Samples
# ============================================================================

println()
println("[Bonus] Generating sample text...")
println()

for i in 1:3
    try
        generated_ids = generate(
            model,
            train_state.params,
            train_state.state,
            CONFIG.seq_length;
            num_steps = 15,
            batch_size = 1,
            rng = Random.MersenneTwister(42 + i),
        )

        # Decode
        sample_ids = vec(generated_ids) .- 1
        sample_text = DataLoader.decode(tokenizer, sample_ids)

        println("Sample $i:")
        println("-" ^ 50)
        println(sample_text[1:min(300, length(sample_text))])
        println("-" ^ 50)
        println()
    catch e
        println("Generation $i failed: $e")
    end
end

println("\n✓ Done!")
