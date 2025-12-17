#!/usr/bin/env julia
"""
Extended training script with checkpoint saving.
Trains for multiple epochs and saves checkpoints.
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

println("Loading packages..."); flush(stdout)
using Random
using Lux
using Zygote
using Optimisers
using Statistics: mean
using Serialization

include(joinpath(dirname(@__DIR__), "src", "Ossamma.jl"))
using .Ossamma

include(joinpath(dirname(@__DIR__), "src", "DataLoader.jl"))
using .DataLoader

include(joinpath(dirname(@__DIR__), "src", "Training.jl"))
using .Training: masked_cross_entropy_vectorized

# Configuration
const CONFIG = (
    embedding_dim = 256,
    num_heads = 4,
    num_layers = 6,
    seq_length = 128,
    batch_size = 32,
    num_epochs = 10,
    learning_rate = 3e-4,
    warmup_steps = 200,
    checkpoint_every_epoch = 1,
    checkpoint_dir = "checkpoints",
    log_every = 50,
)

println("=" ^ 70); flush(stdout)
println("OSSAMMA Extended Training"); flush(stdout)
println("=" ^ 70); flush(stdout)
println("Config: $(CONFIG.num_epochs) epochs, batch=$(CONFIG.batch_size), seq=$(CONFIG.seq_length)"); flush(stdout)

mkpath(CONFIG.checkpoint_dir)

println("\n[1/3] Loading Gutenberg dataset..."); flush(stdout)
rng = Random.MersenneTwister(42)

train_loader, val_loader, tokenizer = prepare_gutenberg(;
    books = :all,
    seq_length = CONFIG.seq_length,
    batch_size = CONFIG.batch_size,
    max_vocab_size = 10000,
    rng = rng,
)

vocab_size = get_vocab_size(tokenizer)
mask_token_id = get_mask_token_id(tokenizer) + 1
num_batches = length(train_loader)
println("  Vocab: $vocab_size, Batches/epoch: $num_batches, Total: $(num_batches * CONFIG.num_epochs)"); flush(stdout)

println("\n[2/3] Creating model..."); flush(stdout)
model_config = LLaDAConfig(
    vocab_size = vocab_size,
    max_sequence_length = CONFIG.seq_length,
    embedding_dimension = CONFIG.embedding_dim,
    number_of_heads = CONFIG.num_heads,
    number_of_layers = CONFIG.num_layers,
    mask_token_id = mask_token_id,
    time_dimension = 64,
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

# Save tokenizer
function save_tokenizer(tok, path)
    serialize(path, Dict(:vocab => tok.vocab, :inverse_vocab => tok.inverse_vocab, :special_tokens => tok.special_tokens, :vocab_size => tok.vocab_size))
end
save_tokenizer(tokenizer, joinpath(CONFIG.checkpoint_dir, "tokenizer.jls"))

# Checkpoint functions
function save_checkpoint(epoch, params, state, opt_state, loss, path)
    serialize(path, Dict(:epoch => epoch, :params => params, :state => state, :opt_state => opt_state, :loss => loss))
    println("  Saved: $path"); flush(stdout)
end

println("\n[3/3] Starting training..."); flush(stdout)
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

        # LR schedule
        lr = global_step < CONFIG.warmup_steps ? Float32(CONFIG.learning_rate * global_step / CONFIG.warmup_steps) : Float32(CONFIG.learning_rate)
        Optimisers.adjust!(opt_state, lr)

        # Sample mask ratio
        mask_ratio = sample_mask_ratio(rng; schedule=:cosine)

        # Forward/backward
        (loss, new_st), grads = Zygote.withgradient(ps) do p
            inputs = (token_ids = batch, mask_ratio = mask_ratio)
            logits, st_out = model(inputs, p, st)
            l = masked_cross_entropy_vectorized(logits, batch, mask_ratio, mask_token_id)
            (l, st_out)
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

    avg_epoch_loss = epoch_loss / num_steps
    println("Epoch $epoch done | Avg Loss: $(round(avg_epoch_loss, digits=4))"); flush(stdout)

    # Validation
    println("Validating..."); flush(stdout)
    reset!(val_loader)
    val_loss = 0.0f0
    val_steps = 0
    for batch in val_loader
        mask_ratio = 0.5f0
        inputs = (token_ids = batch, mask_ratio = mask_ratio)
        logits, _ = model(inputs, ps, st)
        loss = masked_cross_entropy_vectorized(logits, batch, mask_ratio, mask_token_id)
        val_loss += loss
        val_steps += 1
    end
    avg_val_loss = val_loss / val_steps
    println("  Val Loss: $(round(avg_val_loss, digits=4))"); flush(stdout)

    # Save checkpoint
    checkpoint_path = joinpath(CONFIG.checkpoint_dir, "checkpoint_epoch_$(epoch).jls")
    save_checkpoint(epoch, ps, st, opt_state, avg_val_loss, checkpoint_path)

    if avg_val_loss < best_loss
        best_loss = avg_val_loss
        save_checkpoint(epoch, ps, st, opt_state, avg_val_loss, joinpath(CONFIG.checkpoint_dir, "checkpoint_best.jls"))
        println("  New best!"); flush(stdout)
    end

    # Generate sample
    println("Sample:"); flush(stdout)
    try
        gen_ids = generate(model, ps, st, 64; num_steps=15, batch_size=1, rng=rng)
        text = decode(tokenizer, vec(gen_ids) .- 1)
        println("  \"$(text[1:min(80, length(text))])...\""); flush(stdout)
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
    gen_ids = generate(model, ps, st, 128; num_steps=25, batch_size=1, rng=Random.MersenneTwister(i))
    println("$i: $(decode(tokenizer, vec(gen_ids) .- 1))"); flush(stdout)
end
