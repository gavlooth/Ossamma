#!/usr/bin/env julia
"""
Fast training script with smaller model for reasonable training times.
Uses smaller model dimensions to reduce JIT compilation time.
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

println("=" ^ 70); flush(stdout)
println("OSSAMMA Fast Training"); flush(stdout)
println("=" ^ 70); flush(stdout)

println("\nLoading packages..."); flush(stdout)
using Random
using Lux
using Zygote
using Optimisers
using Statistics: mean
using Serialization
using NNlib: logsoftmax

include(joinpath(dirname(@__DIR__), "src", "Ossamma.jl"))
using .Ossamma

include(joinpath(dirname(@__DIR__), "src", "DataLoader.jl"))
using .DataLoader

println("Packages loaded!"); flush(stdout)

# Configuration - smaller for faster JIT
const CONFIG = (
    embedding_dim = 128,  # Reduced from 256
    num_heads = 4,
    num_layers = 4,       # Reduced from 6
    seq_length = 64,      # Reduced from 128
    batch_size = 16,      # Reduced from 32
    num_epochs = 10,
    learning_rate = 3e-4,
    warmup_steps = 100,
    checkpoint_dir = "checkpoints",
    log_every = 25,
)

println("\nConfig:"); flush(stdout)
println("  Embedding: $(CONFIG.embedding_dim)"); flush(stdout)
println("  Layers: $(CONFIG.num_layers)"); flush(stdout)
println("  Seq length: $(CONFIG.seq_length)"); flush(stdout)
println("  Batch size: $(CONFIG.batch_size)"); flush(stdout)
println("  Epochs: $(CONFIG.num_epochs)"); flush(stdout)

mkpath(CONFIG.checkpoint_dir)

# Load dataset
println("\n[1/3] Loading dataset..."); flush(stdout)
rng = Random.MersenneTwister(42)

train_loader, val_loader, tokenizer = prepare_gutenberg(;
    books = :all,
    seq_length = CONFIG.seq_length,
    batch_size = CONFIG.batch_size,
    max_vocab_size = 5000,
    rng = rng,
)

vocab_size = get_vocab_size(tokenizer)
mask_token_id = get_mask_token_id(tokenizer) + 1
num_batches = length(train_loader)
println("  Vocab: $vocab_size, Batches/epoch: $num_batches"); flush(stdout)

# Save tokenizer
serialize(joinpath(CONFIG.checkpoint_dir, "tokenizer.jls"), Dict(
    :vocab => tokenizer.vocab,
    :inverse_vocab => tokenizer.inverse_vocab,
    :special_tokens => tokenizer.special_tokens,
    :vocab_size => tokenizer.vocab_size,
))
println("  Tokenizer saved"); flush(stdout)

# Create model
println("\n[2/3] Creating model..."); flush(stdout)
model_config = LLaDAConfig(
    vocab_size = vocab_size,
    max_sequence_length = CONFIG.seq_length,
    embedding_dimension = CONFIG.embedding_dim,
    number_of_heads = CONFIG.num_heads,
    number_of_layers = CONFIG.num_layers,
    mask_token_id = mask_token_id,
    time_dimension = 32,
    state_dimension = CONFIG.embedding_dim,
    window_size = 16,
    mask_schedule = :cosine,
)

model = LLaDAModel(model_config)
ps, st = Lux.setup(rng, model)

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
println("  Parameters: $(round(count_params(ps) / 1e6, digits=2))M"); flush(stdout)

# Loss function
function compute_loss(model, params, state, batch, mask_token_id, mask_ratio)
    inputs = (token_ids = batch, mask_ratio = mask_ratio)
    logits, new_state = model(inputs, params, state)

    # Cross-entropy loss on masked positions
    vocab_size = size(logits, 1)
    seq_len, batch_size = size(batch)

    # Flatten for loss computation
    logits_flat = reshape(logits, vocab_size, :)  # (vocab, seq*batch)
    targets_flat = reshape(batch, :)               # (seq*batch,)

    # Numerically stable log softmax
    log_probs = logsoftmax(logits_flat, dims=1)

    # Gather correct class log probs
    loss = 0.0f0
    for i in 1:length(targets_flat)
        loss -= log_probs[targets_flat[i], i]
    end
    loss = loss / length(targets_flat)

    # Clamp to avoid NaN
    loss = isnan(loss) ? 10.0f0 : loss

    return loss, new_state
end

# Training
println("\n[3/3] Training..."); flush(stdout)
println("=" ^ 70); flush(stdout)

optimizer = Optimisers.Adam(Float32(CONFIG.learning_rate))
opt_state = Optimisers.setup(optimizer, ps)

global_step = 0
best_loss = Inf32

for epoch in 1:CONFIG.num_epochs
    global global_step, best_loss, ps, st, opt_state

    println("\n--- Epoch $epoch / $(CONFIG.num_epochs) ---"); flush(stdout)
    epoch_loss = 0.0f0
    num_steps = 0

    reset!(train_loader)

    for (batch_idx, batch) in enumerate(train_loader)
        global_step += 1

        # LR warmup
        lr = global_step < CONFIG.warmup_steps ?
            Float32(CONFIG.learning_rate * global_step / CONFIG.warmup_steps) :
            Float32(CONFIG.learning_rate)
        Optimisers.adjust!(opt_state, lr)

        # Random mask ratio (cosine schedule sampling)
        u = rand(rng)
        mask_ratio = Float32(1.0 - cos(π * u / 2))
        mask_ratio = clamp(mask_ratio, 0.1f0, 0.9f0)

        # Forward/backward
        (loss, new_st), grads = Zygote.withgradient(ps) do p
            compute_loss(model, p, st, batch, mask_token_id, mask_ratio)
        end
        st = new_st

        # Update
        opt_state, ps = Optimisers.update(opt_state, ps, grads[1])

        epoch_loss += loss
        num_steps += 1

        if global_step % CONFIG.log_every == 0
            println("  Step $global_step | Loss: $(round(loss, digits=4)) | LR: $(round(lr, sigdigits=3))"); flush(stdout)
        end
    end

    avg_loss = epoch_loss / num_steps
    println("Epoch $epoch done | Avg Loss: $(round(avg_loss, digits=4))"); flush(stdout)

    # Validation
    println("Validating..."); flush(stdout)
    reset!(val_loader)
    val_loss = 0.0f0
    val_steps = 0
    for batch in val_loader
        loss, _ = compute_loss(model, ps, st, batch, mask_token_id, 0.5f0)
        val_loss += loss
        val_steps += 1
    end
    avg_val_loss = val_loss / val_steps
    println("  Val Loss: $(round(avg_val_loss, digits=4))"); flush(stdout)

    # Save checkpoint
    checkpoint_path = joinpath(CONFIG.checkpoint_dir, "checkpoint_epoch_$(epoch).jls")
    serialize(checkpoint_path, Dict(:epoch => epoch, :params => ps, :state => st, :loss => avg_val_loss))
    println("  Saved: $checkpoint_path"); flush(stdout)

    if avg_val_loss < best_loss
        best_loss = avg_val_loss
        serialize(joinpath(CONFIG.checkpoint_dir, "checkpoint_best.jls"), Dict(
            :epoch => epoch, :params => ps, :state => st, :loss => best_loss
        ))
        println("  New best!"); flush(stdout)
    end

    # Generate sample
    println("Sample:"); flush(stdout)
    try
        gen_ids = generate(model, ps, st, 32; num_steps=10, batch_size=1, rng=rng)
        text = decode(tokenizer, vec(gen_ids) .- 1)
        println("  \"$(text[1:min(60, length(text))])...\""); flush(stdout)
    catch e
        println("  Gen failed: $e"); flush(stdout)
    end
end

println("\n" * "=" ^ 70); flush(stdout)
println("Training Complete! Best val loss: $(round(best_loss, digits=4))"); flush(stdout)
println("Checkpoints in: $(CONFIG.checkpoint_dir)/"); flush(stdout)

# Final samples
println("\n--- Final Samples ---"); flush(stdout)
for i in 1:3
    try
        gen_ids = generate(model, ps, st, 64; num_steps=15, batch_size=1, rng=Random.MersenneTwister(i))
        println("$i: $(decode(tokenizer, vec(gen_ids) .- 1))"); flush(stdout)
    catch e
        println("$i: Generation failed: $e"); flush(stdout)
    end
end
