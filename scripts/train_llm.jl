#!/usr/bin/env julia
"""
Meaningful LLM training with higher complexity.
Uses word-level tokenization and larger model architecture.
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

println("=" ^ 70); flush(stdout)
println("OSSAMMA LLM Training - Higher Complexity"); flush(stdout)
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

# Higher complexity configuration
const CONFIG = (
    # Model architecture - meaningful LLM size
    embedding_dim = 512,      # Increased from 128
    num_heads = 8,            # Increased from 4
    num_layers = 8,           # Increased from 4
    seq_length = 256,         # Longer context

    # Training
    batch_size = 8,           # Smaller batch for larger model
    num_epochs = 20,          # More epochs
    learning_rate = 1e-4,     # Lower LR for stability
    min_lr = 1e-6,
    warmup_steps = 500,

    # Tokenization
    max_vocab_size = 8000,    # Word-level vocab
    min_word_freq = 3,        # Filter rare words

    # Checkpointing
    checkpoint_dir = "checkpoints_llm",
    log_every = 50,
    save_every_epoch = 1,
)

println("\nModel Configuration:"); flush(stdout)
println("  Embedding dim: $(CONFIG.embedding_dim)"); flush(stdout)
println("  Attention heads: $(CONFIG.num_heads)"); flush(stdout)
println("  Transformer layers: $(CONFIG.num_layers)"); flush(stdout)
println("  Sequence length: $(CONFIG.seq_length)"); flush(stdout)
println("  Vocab size: $(CONFIG.max_vocab_size)"); flush(stdout)
println("  Batch size: $(CONFIG.batch_size)"); flush(stdout)
println("  Epochs: $(CONFIG.num_epochs)"); flush(stdout)

mkpath(CONFIG.checkpoint_dir)

# ============================================================================
# Word-level Tokenizer
# ============================================================================

mutable struct WordTokenizer
    word_to_id::Dict{String, Int}
    id_to_word::Dict{Int, String}
    vocab_size::Int
    pad_id::Int
    unk_id::Int
    bos_id::Int
    eos_id::Int
    mask_id::Int
end

function build_word_tokenizer(texts::Vector{String}; max_vocab::Int=8000, min_freq::Int=3)
    # Count word frequencies
    word_counts = Dict{String, Int}()
    for text in texts
        words = split(lowercase(text))
        for word in words
            # Clean word - keep alphanumeric and basic punctuation
            clean = replace(word, r"[^a-z0-9',.\-!?]" => "")
            if !isempty(clean)
                word_counts[clean] = get(word_counts, clean, 0) + 1
            end
        end
    end

    # Filter by frequency and sort by count
    filtered = [(w, c) for (w, c) in word_counts if c >= min_freq]
    sort!(filtered, by=x -> -x[2])

    # Build vocabulary with special tokens
    word_to_id = Dict{String, Int}()
    id_to_word = Dict{Int, String}()

    # Special tokens (1-indexed for Julia)
    special = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]
    for (i, tok) in enumerate(special)
        word_to_id[tok] = i
        id_to_word[i] = tok
    end

    # Add vocabulary words
    vocab_words = min(max_vocab - length(special), length(filtered))
    for i in 1:vocab_words
        word = filtered[i][1]
        idx = i + length(special)
        word_to_id[word] = idx
        id_to_word[idx] = word
    end

    vocab_size = length(word_to_id)
    println("  Built vocabulary: $vocab_size words"); flush(stdout)

    return WordTokenizer(
        word_to_id, id_to_word, vocab_size,
        1, 2, 3, 4, 5  # pad, unk, bos, eos, mask
    )
end

function encode_text(tokenizer::WordTokenizer, text::String; max_len::Int=256)
    words = split(lowercase(text))
    ids = Int[]
    push!(ids, tokenizer.bos_id)

    for word in words
        clean = replace(word, r"[^a-z0-9',.\-!?]" => "")
        if !isempty(clean)
            id = get(tokenizer.word_to_id, clean, tokenizer.unk_id)
            push!(ids, id)
        end
        if length(ids) >= max_len - 1
            break
        end
    end

    push!(ids, tokenizer.eos_id)

    # Pad to max_len
    while length(ids) < max_len
        push!(ids, tokenizer.pad_id)
    end

    return ids[1:max_len]
end

function decode_ids(tokenizer::WordTokenizer, ids::Vector{Int})
    words = String[]
    for id in ids
        if id in [tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.mask_id]
            continue
        end
        word = get(tokenizer.id_to_word, id, "<unk>")
        push!(words, word)
    end
    return join(words, " ")
end

# ============================================================================
# Data Loading
# ============================================================================

println("\n[1/4] Loading and processing Gutenberg texts..."); flush(stdout)
rng = Random.MersenneTwister(42)

# Download and combine all texts
const GUTENBERG_URLS = Dict(
    :great_expectations => "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
    :tale_two_cities => "https://www.gutenberg.org/cache/epub/98/pg98.txt",
    :frankenstein => "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    :sherlock => "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    :dracula => "https://www.gutenberg.org/cache/epub/345/pg345.txt",
    :alice => "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    :moby_dick => "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    :pride => "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    :war_peace => "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
    :emma => "https://www.gutenberg.org/cache/epub/158/pg158.txt",
)

function download_gutenberg(url::String)
    cache_dir = joinpath(homedir(), ".cache", "ossamma", "texts")
    mkpath(cache_dir)

    filename = replace(url, r"[:/.]" => "_") * ".txt"
    cache_path = joinpath(cache_dir, filename)

    if isfile(cache_path)
        return read(cache_path, String)
    end

    println("    Downloading: $url"); flush(stdout)
    text = try
        read(download(url), String)
    catch e
        println("    Download failed: $e"); flush(stdout)
        return ""
    end

    write(cache_path, text)
    return text
end

function clean_gutenberg_text(text::String)
    # Find start and end markers
    start_markers = ["*** START OF", "***START OF", "*** START OF THE PROJECT"]
    end_markers = ["*** END OF", "***END OF", "End of Project Gutenberg"]

    start_idx = 1
    for marker in start_markers
        idx = findfirst(marker, text)
        if idx !== nothing
            # Find next newline after marker
            nl = findnext('\n', text, last(idx))
            if nl !== nothing
                start_idx = nl + 1
            end
            break
        end
    end

    end_idx = length(text)
    for marker in end_markers
        idx = findfirst(marker, text)
        if idx !== nothing
            end_idx = first(idx) - 1
            break
        end
    end

    clean = text[start_idx:end_idx]
    # Normalize whitespace
    clean = replace(clean, r"\r\n" => "\n")
    clean = replace(clean, r"\n{3,}" => "\n\n")
    return strip(clean)
end

# Load all texts
all_texts = String[]
for (name, url) in GUTENBERG_URLS
    println("  Loading: $name"); flush(stdout)
    raw = download_gutenberg(url)
    if !isempty(raw)
        clean = clean_gutenberg_text(raw)
        push!(all_texts, clean)
        println("    $(length(clean)) characters"); flush(stdout)
    end
end

# Split into paragraphs/chunks for training
println("\n  Splitting into training chunks..."); flush(stdout)
chunks = String[]
for text in all_texts
    # Split by double newlines (paragraphs)
    paragraphs = split(text, r"\n\n+")
    for para in paragraphs
        para = strip(String(para))
        # Only keep substantial paragraphs
        if length(para) > 100
            push!(chunks, para)
        end
    end
end
println("    Total chunks: $(length(chunks))"); flush(stdout)

# Shuffle and split
Random.shuffle!(rng, chunks)
split_idx = Int(floor(0.9 * length(chunks)))
train_chunks = chunks[1:split_idx]
val_chunks = chunks[split_idx+1:end]
println("    Train: $(length(train_chunks)), Val: $(length(val_chunks))"); flush(stdout)

# Build tokenizer
println("\n[2/4] Building word-level tokenizer..."); flush(stdout)
tokenizer = build_word_tokenizer(train_chunks; max_vocab=CONFIG.max_vocab_size, min_freq=CONFIG.min_word_freq)

# Save tokenizer
serialize(joinpath(CONFIG.checkpoint_dir, "tokenizer.jls"), Dict(
    :word_to_id => tokenizer.word_to_id,
    :id_to_word => tokenizer.id_to_word,
    :vocab_size => tokenizer.vocab_size,
    :special_ids => (tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.mask_id),
))
println("  Tokenizer saved"); flush(stdout)

# Tokenize datasets
println("  Tokenizing training data..."); flush(stdout)
train_ids = [encode_text(tokenizer, chunk; max_len=CONFIG.seq_length) for chunk in train_chunks]
val_ids = [encode_text(tokenizer, chunk; max_len=CONFIG.seq_length) for chunk in val_chunks]

# Filter out chunks that are mostly padding
train_ids = filter(ids -> count(x -> x != tokenizer.pad_id, ids) > CONFIG.seq_length ÷ 4, train_ids)
val_ids = filter(ids -> count(x -> x != tokenizer.pad_id, ids) > CONFIG.seq_length ÷ 4, val_ids)
println("  Filtered train: $(length(train_ids)), val: $(length(val_ids))"); flush(stdout)

# Create batches
function create_batches(data::Vector{Vector{Int}}, batch_size::Int)
    batches = []
    for i in 1:batch_size:length(data)
        end_idx = min(i + batch_size - 1, length(data))
        batch_data = data[i:end_idx]
        if length(batch_data) == batch_size
            # Stack into (seq_length, batch_size) matrix
            batch = hcat([ids for ids in batch_data]...)
            push!(batches, batch)
        end
    end
    return batches
end

train_batches = create_batches(train_ids, CONFIG.batch_size)
val_batches = create_batches(val_ids, CONFIG.batch_size)
println("  Train batches: $(length(train_batches)), Val batches: $(length(val_batches))"); flush(stdout)

# ============================================================================
# Model Creation
# ============================================================================

println("\n[3/4] Creating model..."); flush(stdout)

model_config = LLaDAConfig(
    vocab_size = tokenizer.vocab_size,
    max_sequence_length = CONFIG.seq_length,
    embedding_dimension = CONFIG.embedding_dim,
    number_of_heads = CONFIG.num_heads,
    number_of_layers = CONFIG.num_layers,
    mask_token_id = tokenizer.mask_id,
    time_dimension = 128,
    state_dimension = CONFIG.embedding_dim,
    window_size = 64,
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

param_count = count_params(ps)
println("  Parameters: $(round(param_count / 1e6, digits=2))M"); flush(stdout)
println("  Architecture: $(CONFIG.num_layers) layers, $(CONFIG.num_heads) heads, $(CONFIG.embedding_dim) dim"); flush(stdout)

# ============================================================================
# Loss Function with Padding Mask
# ============================================================================

function compute_loss_with_padding(model, params, state, batch, tokenizer, mask_ratio)
    pad_id = tokenizer.pad_id
    mask_id = tokenizer.mask_id

    inputs = (token_ids = batch, mask_ratio = mask_ratio)
    logits, new_state = model(inputs, params, state)

    vocab_size = size(logits, 1)
    seq_len, batch_size = size(batch)

    # Flatten
    logits_flat = reshape(logits, vocab_size, :)
    targets_flat = reshape(batch, :)

    # Numerically stable log softmax
    log_probs = logsoftmax(logits_flat, dims=1)

    # Compute loss only on non-padding tokens
    loss = 0.0f0
    count = 0
    for i in 1:length(targets_flat)
        if targets_flat[i] != pad_id
            loss -= log_probs[targets_flat[i], i]
            count += 1
        end
    end

    loss = count > 0 ? loss / count : 0.0f0
    return loss, new_state
end

# ============================================================================
# Training Loop
# ============================================================================

println("\n[4/4] Starting training..."); flush(stdout)
println("=" ^ 70); flush(stdout)
println("Note: First gradient computation will take several minutes (JIT)"); flush(stdout)
println("=" ^ 70); flush(stdout)

optimizer = Optimisers.Adam(Float32(CONFIG.learning_rate))
opt_state = Optimisers.setup(optimizer, ps)

global_step = 0
best_val_loss = Inf32
start_time = time()

for epoch in 1:CONFIG.num_epochs
    global global_step, best_val_loss, ps, st, opt_state

    epoch_start = time()
    println("\n--- Epoch $epoch / $(CONFIG.num_epochs) ---"); flush(stdout)

    epoch_loss = 0.0f0
    num_steps = 0

    # Shuffle training batches
    shuffled_batches = Random.shuffle(rng, copy(train_batches))

    for (batch_idx, batch) in enumerate(shuffled_batches)
        global_step += 1

        # Learning rate schedule with warmup and cosine decay
        if global_step < CONFIG.warmup_steps
            lr = Float32(CONFIG.learning_rate * global_step / CONFIG.warmup_steps)
        else
            progress = Float32(global_step - CONFIG.warmup_steps) / Float32(CONFIG.num_epochs * length(train_batches) - CONFIG.warmup_steps)
            lr = Float32(CONFIG.min_lr + 0.5 * (CONFIG.learning_rate - CONFIG.min_lr) * (1 + cos(π * progress)))
        end
        Optimisers.adjust!(opt_state, lr)

        # Sample mask ratio from cosine schedule
        u = rand(rng)
        mask_ratio = Float32(1.0 - cos(π * u / 2))
        mask_ratio = clamp(mask_ratio, 0.15f0, 0.85f0)

        # Forward/backward
        (loss, new_st), grads = Zygote.withgradient(ps) do p
            compute_loss_with_padding(model, p, st, batch, tokenizer, mask_ratio)
        end
        st = new_st

        # Skip if loss is NaN
        if isnan(loss)
            println("  Warning: NaN loss at step $global_step, skipping"); flush(stdout)
            continue
        end

        # Gradient clipping
        grads_flat, restructure = Optimisers.destructure(grads[1])
        grad_norm = sqrt(sum(grads_flat .^ 2))
        max_norm = 1.0f0
        if grad_norm > max_norm
            grads_flat = grads_flat .* (max_norm / grad_norm)
        end
        grads_clipped = (restructure(grads_flat),)

        # Update
        opt_state, ps = Optimisers.update(opt_state, ps, grads_clipped[1])

        epoch_loss += loss
        num_steps += 1

        if global_step % CONFIG.log_every == 0
            elapsed = time() - start_time
            steps_per_sec = global_step / elapsed
            println("  Step $global_step | Loss: $(round(loss, digits=4)) | LR: $(round(lr, sigdigits=3)) | $(round(steps_per_sec, digits=2)) steps/s"); flush(stdout)
        end
    end

    avg_loss = num_steps > 0 ? epoch_loss / num_steps : 0.0f0
    epoch_time = time() - epoch_start
    println("Epoch $epoch done | Avg Loss: $(round(avg_loss, digits=4)) | Time: $(round(epoch_time/60, digits=1)) min"); flush(stdout)

    # Validation
    println("Validating..."); flush(stdout)
    val_loss = 0.0f0
    val_steps = 0
    for batch in val_batches
        loss, _ = compute_loss_with_padding(model, ps, st, batch, tokenizer, 0.5f0)
        if !isnan(loss)
            val_loss += loss
            val_steps += 1
        end
    end
    avg_val_loss = val_steps > 0 ? val_loss / val_steps : Inf32
    println("  Val Loss: $(round(avg_val_loss, digits=4))"); flush(stdout)

    # Save checkpoint
    checkpoint_path = joinpath(CONFIG.checkpoint_dir, "checkpoint_epoch_$(epoch).jls")
    serialize(checkpoint_path, Dict(
        :epoch => epoch,
        :step => global_step,
        :params => ps,
        :state => st,
        :train_loss => avg_loss,
        :val_loss => avg_val_loss,
        :config => CONFIG,
    ))
    println("  Saved: $checkpoint_path"); flush(stdout)

    if avg_val_loss < best_val_loss
        best_val_loss = avg_val_loss
        serialize(joinpath(CONFIG.checkpoint_dir, "checkpoint_best.jls"), Dict(
            :epoch => epoch,
            :step => global_step,
            :params => ps,
            :state => st,
            :val_loss => best_val_loss,
        ))
        println("  New best model!"); flush(stdout)
    end

    # Generate sample
    println("Sample generation:"); flush(stdout)
    try
        gen_ids = generate(model, ps, st, CONFIG.seq_length; num_steps=50, batch_size=1, rng=rng)
        text = decode_ids(tokenizer, vec(gen_ids))
        # Show first 150 chars
        sample = length(text) > 150 ? text[1:150] * "..." : text
        println("  \"$sample\""); flush(stdout)
    catch e
        println("  Generation failed: $e"); flush(stdout)
    end
end

total_time = (time() - start_time) / 60
println("\n" * "=" ^ 70); flush(stdout)
println("Training Complete!"); flush(stdout)
println("  Total time: $(round(total_time, digits=1)) minutes"); flush(stdout)
println("  Best val loss: $(round(best_val_loss, digits=4))"); flush(stdout)
println("  Parameters: $(round(param_count / 1e6, digits=2))M"); flush(stdout)
println("  Checkpoints: $(CONFIG.checkpoint_dir)/"); flush(stdout)
println("=" ^ 70); flush(stdout)

# Final generation samples
println("\n--- Final Generated Samples ---"); flush(stdout)
for i in 1:5
    try
        gen_ids = generate(model, ps, st, CONFIG.seq_length; num_steps=75, batch_size=1, rng=Random.MersenneTwister(i * 42))
        text = decode_ids(tokenizer, vec(gen_ids))
        println("\nSample $i:"); flush(stdout)
        println(text[1:min(300, length(text))]); flush(stdout)
    catch e
        println("Sample $i failed: $e"); flush(stdout)
    end
end
