#!/usr/bin/env julia
"""
Production training script for Ossamma LLaDA model.

Optimized for GPUs with 24GB+ VRAM (L4, A100, RTX 4090, etc.)

Usage (local):
    julia --project=. scripts/train_production.jl

Usage (Google Colab - see instructions below):
    # In a Colab cell:
    # !git clone https://github.com/YOUR_REPO/ossamma.git
    # !cd ossamma && julia --project=. scripts/train_production.jl
"""

# ============================================================================
# Setup
# ============================================================================

using Pkg

# Handle both direct execution and REPL/include
const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = if isempty(SCRIPT_DIR) || SCRIPT_DIR == "."
    # Running from REPL or include_string
    get(ENV, "OSSAMMA_ROOT", pwd())
else
    dirname(SCRIPT_DIR)
end

Pkg.activate(PROJECT_ROOT)

println("=" ^ 70)
println("OSSAMMA Production Training")
println("=" ^ 70)
println()

# Load packages
println("[Setup] Loading packages...")
flush(stdout)

using Random
using Lux
using Zygote
using Optimisers
using Statistics: mean

include(joinpath(PROJECT_ROOT, "src", "Ossamma.jl"))
using .Ossamma

include(joinpath(PROJECT_ROOT, "src", "DataLoader.jl"))
using .DataLoader

println("[Setup] Packages loaded successfully")
println()

# ============================================================================
# Configuration
# ============================================================================

# Training configuration - adjust based on your GPU memory
const CONFIG = (
    # Model config
    use_production_config = false,  # Set true for full production (149M params)
                                    # false uses smaller config for faster iteration

    # Dataset options: :gutenberg (reliable), :tinystories, :wikitext, :synthetic
    dataset = :gutenberg,  # Use :gutenberg for reliable text data
    num_train_samples = 5000,   # Keep under 10k for HF API (not used for gutenberg)
    num_val_samples = 500,

    # Training hyperparameters
    batch_size = 16,           # Reduce if OOM
    seq_length = 256,          # Sequence length (reduce if OOM)
    learning_rate = 3e-4,
    min_lr = 1e-6,
    warmup_steps = 500,
    total_steps = 5000,        # Increase for longer training

    # Logging
    log_every = 50,
    eval_every = 500,
    save_every = 1000,

    # Output
    output_dir = "checkpoints",
    experiment_name = "ossamma_production",
)

println("[Config] Training configuration:")
println("  Production config: $(CONFIG.use_production_config)")
println("  Dataset: $(CONFIG.dataset)")
println("  Batch size: $(CONFIG.batch_size)")
println("  Sequence length: $(CONFIG.seq_length)")
println("  Total steps: $(CONFIG.total_steps)")
println()

# ============================================================================
# Model Setup
# ============================================================================

println("[Model] Creating model...")
flush(stdout)

# Choose config based on setting
model_config = if CONFIG.use_production_config
    production_config()
else
    # Smaller config for faster iteration
    LLaDAConfig(
        vocab_size = 10000,
        max_sequence_length = CONFIG.seq_length,
        embedding_dimension = 256,
        number_of_heads = 4,
        number_of_layers = 6,
        time_dimension = 64,
        state_dimension = 256,
        window_size = 32,
        mask_schedule = :cosine,
    )
end

# Override sequence length from config
model_config = LLaDAConfig(
    vocab_size = model_config.vocab_size,
    max_sequence_length = CONFIG.seq_length,
    embedding_dimension = model_config.embedding_dimension,
    number_of_heads = model_config.number_of_heads,
    number_of_layers = model_config.number_of_layers,
    time_dimension = model_config.time_dimension,
    state_dimension = model_config.state_dimension,
    window_size = min(model_config.window_size, CONFIG.seq_length),
    min_frequency = model_config.min_frequency,
    max_frequency = model_config.max_frequency,
    default_time_step = model_config.default_time_step,
    mask_schedule = model_config.mask_schedule,
)

rng = Random.MersenneTwister(42)

# We'll create the model after loading data (to get actual vocab size)

# ============================================================================
# Data Loading
# ============================================================================

println("[Data] Loading dataset: $(CONFIG.dataset)")
flush(stdout)

train_loader, val_loader, tokenizer = if CONFIG.dataset == :gutenberg
    prepare_gutenberg(;
        books = :all,  # All 10 classic novels for more training data
        seq_length = CONFIG.seq_length,
        batch_size = CONFIG.batch_size,
        max_vocab_size = model_config.vocab_size,
        rng = rng,
    )
elseif CONFIG.dataset == :tinystories
    prepare_tinystories(;
        num_train_rows = CONFIG.num_train_samples,
        num_val_rows = CONFIG.num_val_samples,
        seq_length = CONFIG.seq_length,
        batch_size = CONFIG.batch_size,
        max_vocab_size = model_config.vocab_size,
        rng = rng,
    )
elseif CONFIG.dataset == :wikitext
    prepare_wikitext(;
        num_train_rows = CONFIG.num_train_samples,
        num_val_rows = CONFIG.num_val_samples,
        seq_length = CONFIG.seq_length,
        batch_size = CONFIG.batch_size,
        max_vocab_size = model_config.vocab_size,
        rng = rng,
    )
elseif CONFIG.dataset == :synthetic
    # Synthetic data for quick testing (no network required)
    println("[Data] Generating synthetic data...")
    vocab_size = model_config.vocab_size

    # Create synthetic batches (just random token sequences)
    struct SyntheticLoader
        vocab_size::Int
        seq_length::Int
        batch_size::Int
        num_batches::Int
        rng::Random.AbstractRNG
    end

    function Base.iterate(loader::SyntheticLoader, state=1)
        if state > loader.num_batches
            return nothing
        end
        batch = rand(loader.rng, 1:loader.vocab_size, loader.seq_length, loader.batch_size)
        return batch, state + 1
    end
    Base.length(loader::SyntheticLoader) = loader.num_batches

    # Create synthetic tokenizer (minimal)
    tokenizer = Tokenizer()
    tokenizer.vocab_size = vocab_size

    train_loader = SyntheticLoader(vocab_size, CONFIG.seq_length, CONFIG.batch_size,
                                   CONFIG.num_train_samples ÷ CONFIG.batch_size, rng)
    val_loader = SyntheticLoader(vocab_size, CONFIG.seq_length, CONFIG.batch_size,
                                 CONFIG.num_val_samples ÷ CONFIG.batch_size, rng)

    (train_loader, val_loader, tokenizer)
else
    error("Unknown dataset: $(CONFIG.dataset). Options: :gutenberg, :tinystories, :wikitext, :synthetic")
end

actual_vocab_size = get_vocab_size(tokenizer)
mask_token_id = get_mask_token_id(tokenizer) + 1  # +1 for Julia 1-indexing

println("[Data] Dataset ready:")
println("  Vocab size: $actual_vocab_size")
println("  Mask token: $mask_token_id")
println("  Train batches: $(length(train_loader))")
println("  Val batches: $(length(val_loader))")
println()

# ============================================================================
# Create Model with actual vocab size
# ============================================================================

final_config = LLaDAConfig(
    vocab_size = actual_vocab_size,
    max_sequence_length = CONFIG.seq_length,
    embedding_dimension = model_config.embedding_dimension,
    number_of_heads = model_config.number_of_heads,
    number_of_layers = model_config.number_of_layers,
    mask_token_id = mask_token_id,
    time_dimension = model_config.time_dimension,
    state_dimension = model_config.state_dimension,
    window_size = model_config.window_size,
    min_frequency = model_config.min_frequency,
    max_frequency = model_config.max_frequency,
    default_time_step = model_config.default_time_step,
    mask_schedule = model_config.mask_schedule,
)

model = LLaDAModel(final_config)
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
println("[Model] Model created:")
println("  Embedding dim: $(final_config.embedding_dimension)")
println("  Layers: $(final_config.number_of_layers)")
println("  Heads: $(final_config.number_of_heads)")
println("  Parameters: $(round(param_count / 1e6, digits=2))M")
println()

# ============================================================================
# Training Setup
# ============================================================================

println("[Train] Setting up optimizer...")
flush(stdout)

optimizer = Optimisers.Adam(Float32(CONFIG.learning_rate))
train_state = create_train_state(model, optimizer; rng=rng)

train_config = TrainingConfig(
    batch_size = CONFIG.batch_size,
    learning_rate = Float32(CONFIG.learning_rate),
    min_learning_rate = Float32(CONFIG.min_lr),
    warmup_steps = CONFIG.warmup_steps,
    total_steps = CONFIG.total_steps,
    eval_every = CONFIG.eval_every,
    log_every = CONFIG.log_every,
    save_every = CONFIG.save_every,
    mask_schedule = :cosine,
)

# Create output directory
mkpath(CONFIG.output_dir)

# Callbacks for saving
callbacks = Dict{Symbol, Function}(
    :on_save => function(ts)
        println("  [Checkpoint] Step $(ts.step)")
        # Could add model saving here
    end,
    :on_best => function(ts)
        println("  [Best] New best loss at step $(ts.step)!")
    end,
)

println("[Train] Ready to train!")
println("  Steps: $(train_config.total_steps)")
println("  LR: $(train_config.learning_rate)")
println("  Warmup: $(train_config.warmup_steps)")
println()

# ============================================================================
# Training Loop
# ============================================================================

println("=" ^ 70)
println("Starting training...")
println("=" ^ 70)
println()
flush(stdout)

train!(
    model,
    train_state,
    train_loader,
    train_config;
    val_data = val_loader,
    callbacks = callbacks,
    rng = rng,
)

println()
println("=" ^ 70)
println("Training Complete!")
println("=" ^ 70)

# ============================================================================
# Generation Sample
# ============================================================================

println()
println("[Generate] Generating sample text...")
flush(stdout)

try
    generated_ids = generate(
        model,
        train_state.params,
        train_state.state,
        min(64, CONFIG.seq_length);  # Short sample
        num_steps = 20,
        batch_size = 1,
        rng = rng,
    )

    # Decode
    sample_ids = vec(generated_ids) .- 1  # Back to 0-indexed for tokenizer
    sample_text = decode(tokenizer, sample_ids)

    println()
    println("Generated sample:")
    println("-" ^ 50)
    println(sample_text[1:min(200, length(sample_text))])
    println("-" ^ 50)
catch e
    println("Generation failed: $e")
end

println()
println("Done!")
