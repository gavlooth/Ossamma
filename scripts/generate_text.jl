#!/usr/bin/env julia
"""
Text generation script for Ossamma LLaDA model.
Trains on Gutenberg data and generates text samples.
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

println("Loading packages...")
using Random
using Lux
using Zygote
using Optimisers
using Statistics: mean

include(joinpath(dirname(@__DIR__), "src", "Ossamma.jl"))
using .Ossamma

include(joinpath(dirname(@__DIR__), "src", "DataLoader.jl"))
using .DataLoader

# Configuration
const SEQ_LENGTH = 128  # Shorter for faster training
const BATCH_SIZE = 32
const TRAIN_STEPS = 1000  # More steps for better generation
const WARMUP_STEPS = 100

println("=" ^ 60)
println("Ossamma Text Generation")
println("=" ^ 60)

# Load data
println("\n[1/4] Loading Gutenberg dataset...")
rng = Random.MersenneTwister(42)

train_loader, val_loader, tokenizer = prepare_gutenberg(;
    books = :all,
    seq_length = SEQ_LENGTH,
    batch_size = BATCH_SIZE,
    max_vocab_size = 10000,
    rng = rng,
)

vocab_size = get_vocab_size(tokenizer)
mask_token_id = get_mask_token_id(tokenizer) + 1

println("  Vocab size: $vocab_size")
println("  Mask token: $mask_token_id")

# Create model
println("\n[2/4] Creating model...")
config = LLaDAConfig(
    vocab_size = vocab_size,
    max_sequence_length = SEQ_LENGTH,
    embedding_dimension = 256,
    number_of_heads = 4,
    number_of_layers = 6,
    mask_token_id = mask_token_id,
    time_dimension = 64,
    state_dimension = 256,
    window_size = 16,
    mask_schedule = :cosine,
)

model = LLaDAModel(config)
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

println("  Parameters: $(round(count_params(ps) / 1e6, digits=2))M")

# Training
println("\n[3/4] Training for $TRAIN_STEPS steps...")
optimizer = Optimisers.Adam(Float32(3e-4))
train_state = create_train_state(model, optimizer; rng=rng)

train_config = TrainingConfig(
    batch_size = BATCH_SIZE,
    learning_rate = Float32(3e-4),
    min_learning_rate = Float32(1e-6),
    warmup_steps = WARMUP_STEPS,
    total_steps = TRAIN_STEPS,
    eval_every = 500,
    log_every = 100,
    save_every = 10000,
    mask_schedule = :cosine,
)

train!(
    model,
    train_state,
    train_loader,
    train_config;
    val_data = val_loader,
    rng = rng,
)

# Text Generation
println("\n[4/4] Generating text samples...")
println("=" ^ 60)

function generate_and_decode(model, ps, st, tokenizer, seq_len; num_steps=20, rng=Random.default_rng())
    generated_ids = generate(
        model, ps, st, seq_len;
        num_steps = num_steps,
        batch_size = 1,
        rng = rng,
    )

    # Decode (subtract 1 for 0-indexed tokenizer)
    ids = vec(generated_ids) .- 1
    return decode(tokenizer, ids)
end

# Generate multiple samples
println("\n--- Sample 1 (64 chars, 10 denoising steps) ---")
sample1 = generate_and_decode(model, train_state.params, train_state.state, tokenizer, 64; num_steps=10, rng=rng)
println(sample1)

println("\n--- Sample 2 (128 chars, 20 denoising steps) ---")
sample2 = generate_and_decode(model, train_state.params, train_state.state, tokenizer, 128; num_steps=20, rng=rng)
println(sample2)

println("\n--- Sample 3 (128 chars, 50 denoising steps) ---")
sample3 = generate_and_decode(model, train_state.params, train_state.state, tokenizer, 128; num_steps=50, rng=rng)
println(sample3)

println("\n--- Sample 4 (256 chars, 30 denoising steps) ---")
sample4 = generate_and_decode(model, train_state.params, train_state.state, tokenizer, 256; num_steps=30, rng=rng)
println(sample4)

println("\n" * "=" ^ 60)
println("Generation complete!")
println("=" ^ 60)
