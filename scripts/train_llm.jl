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
using NNlib: logsoftmax, onehotbatch
using Printf

# GPU support
using CUDA
using LuxCUDA

# Check GPU availability
if CUDA.functional()
    println("✓ CUDA available: $(CUDA.name(CUDA.device()))"); flush(stdout)
    println("  Memory: $(round(CUDA.total_memory() / 1e9, digits=1)) GB"); flush(stdout)
    # Allow scalar operations - some operations fall back to CPU
    # This is slower but necessary until all ops are GPU-native
    CUDA.allowscalar(true)
    const GPU_DEVICE = gpu_device()
else
    println("✗ CUDA not available, using CPU"); flush(stdout)
    const GPU_DEVICE = cpu_device()
end

include(joinpath(dirname(@__DIR__), "src", "Ossamma.jl"))
using .Ossamma

include(joinpath(dirname(@__DIR__), "src", "DataLoader.jl"))
using .DataLoader

println("Packages loaded!"); flush(stdout)

# ============================================================================
# Progress Bar
# ============================================================================

mutable struct ProgressBar
    total::Int
    current::Int
    width::Int
    start_time::Float64
    desc::String
    last_print_time::Float64
end

function ProgressBar(total::Int; width::Int=40, desc::String="Progress")
    ProgressBar(total, 0, width, time(), desc, 0.0)
end

function update!(pb::ProgressBar, current::Int; loss::Float32=0.0f0, lr::Float32=0.0f0)
    pb.current = current
    now = time()

    # Only update display every 0.5 seconds to reduce flicker
    if now - pb.last_print_time < 0.5 && current < pb.total
        return
    end
    pb.last_print_time = now

    # Calculate progress
    progress = current / pb.total
    filled = Int(round(progress * pb.width))
    empty = pb.width - filled

    # Calculate ETA
    elapsed = now - pb.start_time
    if current > 0
        eta = elapsed / current * (pb.total - current)
        eta_str = format_time(eta)
    else
        eta_str = "--:--"
    end
    elapsed_str = format_time(elapsed)

    # Build progress bar
    bar = "█" ^ filled * "░" ^ empty
    pct = @sprintf("%5.1f%%", progress * 100)

    # Build status line
    status = @sprintf("\r%s |%s| %s [%s<%s] Loss: %.4f LR: %.2e",
                      pb.desc, bar, pct, elapsed_str, eta_str, loss, lr)

    print(status)
    flush(stdout)

    if current >= pb.total
        println()  # New line when complete
        flush(stdout)
    end
end

function format_time(seconds::Float64)
    if seconds < 60
        return @sprintf("%02d:%02d", 0, Int(floor(seconds)))
    elseif seconds < 3600
        mins = Int(floor(seconds / 60))
        secs = Int(floor(seconds % 60))
        return @sprintf("%02d:%02d", mins, secs)
    else
        hours = Int(floor(seconds / 3600))
        mins = Int(floor((seconds % 3600) / 60))
        return @sprintf("%dh%02dm", hours, mins)
    end
end

function finish!(pb::ProgressBar)
    update!(pb, pb.total)
end

# Higher complexity configuration - balanced for compilation time
const CONFIG = (
    # Model architecture - 4 layers to reduce JIT compilation time
    embedding_dim = 384,      # Reasonable size
    num_heads = 6,            # 6 heads
    num_layers = 4,           # Reduced from 8 for faster compilation
    seq_length = 128,         # Shorter for faster training

    # Training - CPU-friendly batch size
    batch_size = 16,          # Smaller batch
    num_epochs = 15,          # Fewer epochs
    learning_rate = 2e-4,     # Slightly higher LR
    min_lr = 1e-6,
    warmup_steps = 200,

    # Tokenization
    max_vocab_size = 8000,    # Word-level vocab
    min_word_freq = 3,        # Filter rare words

    # Checkpointing
    checkpoint_dir = "checkpoints_llm",
    log_every = 25,
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
    time_dimension = 96,
    state_dimension = CONFIG.embedding_dim,
    window_size = 32,  # Smaller window for 128 seq length
    mask_schedule = :cosine,
)

model = LLaDAModel(model_config)
ps, st = Lux.setup(rng, model)

# Move to GPU
ps = ps |> GPU_DEVICE
st = st |> GPU_DEVICE
println("  Model moved to: $(typeof(GPU_DEVICE))"); flush(stdout)

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

function compute_loss_with_padding(model, params, state, batch_cpu, tokenizer, mask_ratio)
    pad_id = tokenizer.pad_id
    vocab_size = tokenizer.vocab_size

    # Move batch to GPU
    batch = CuArray(batch_cpu)

    inputs = (token_ids = batch, mask_ratio = mask_ratio)
    logits, new_state = model(inputs, params, state)

    seq_len, batch_sz = size(batch)
    n = seq_len * batch_sz

    # Flatten
    logits_flat = reshape(logits, vocab_size, :)  # (vocab, n)
    targets_flat = reshape(batch, :)               # (n,) - on GPU

    # Numerically stable log softmax
    log_probs = logsoftmax(logits_flat, dims=1)   # (vocab, n)

    # GPU-friendly one-hot: create on CPU then transfer as Float32 array
    targets_cpu = Array(targets_flat)
    one_hot_cpu = Float32.(onehotbatch(targets_cpu, 1:vocab_size))  # (vocab, n) Float32 on CPU
    one_hot = CuArray(one_hot_cpu)  # Move to GPU

    # Gather target log probs via element-wise multiply and sum
    target_log_probs = sum(log_probs .* one_hot, dims=1)  # (1, n)

    # Create padding mask (1 for non-pad, 0 for pad) on GPU
    mask_cpu = Float32.(targets_cpu .!= pad_id)  # (n,) on CPU
    mask = CuArray(reshape(mask_cpu, 1, :))  # (1, n) on GPU

    # Masked cross-entropy loss
    num_tokens = sum(mask)
    loss = -sum(target_log_probs .* mask) / max(num_tokens, 1.0f0)

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
    current_loss = 0.0f0
    current_lr = 0.0f0

    # Shuffle training batches
    shuffled_batches = Random.shuffle(rng, copy(train_batches))
    num_batches = length(shuffled_batches)

    # Create progress bar for this epoch
    pb = ProgressBar(num_batches; width=30, desc=@sprintf("Epoch %2d", epoch))

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
        current_lr = lr

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
            println("\n  Warning: NaN loss at step $global_step, skipping"); flush(stdout)
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
        current_loss = loss

        # Update progress bar
        update!(pb, batch_idx; loss=current_loss, lr=current_lr)
    end

    # Finish progress bar
    finish!(pb)

    avg_loss = num_steps > 0 ? epoch_loss / num_steps : 0.0f0
    epoch_time = time() - epoch_start
    println("Epoch $epoch done | Avg Loss: $(round(avg_loss, digits=4)) | Time: $(round(epoch_time/60, digits=1)) min"); flush(stdout)

    # Validation with progress bar
    val_loss = 0.0f0
    val_steps = 0
    val_pb = ProgressBar(length(val_batches); width=30, desc="Validate ")
    for (i, batch) in enumerate(val_batches)
        loss, _ = compute_loss_with_padding(model, ps, st, batch, tokenizer, 0.5f0)
        if !isnan(loss)
            val_loss += loss
            val_steps += 1
        end
        update!(val_pb, i; loss=val_steps > 0 ? val_loss / val_steps : 0.0f0, lr=0.0f0)
    end
    finish!(val_pb)
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
