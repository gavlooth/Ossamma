#!/usr/bin/env julia
"""
Train OssammaNER model on converted MultiNERD data.

Usage:
    julia --project=. scripts/train_ner.jl --data data/rag/ --output models/ner/
    julia --project=. scripts/train_ner.jl --test  # Quick test with synthetic data
"""

using Random
using Statistics
using Printf
using Dates
using JSON3

# Load Ossamma
include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
using .Ossamma
using .Ossamma: OssammaNER, NERConfig, tiny_ner, small_ner, base_ner
using .Ossamma: ner_cross_entropy, predict_labels, extract_entities
using .Ossamma: RAG_LABELS, LABEL_TO_ID, ID_TO_LABEL, NUM_LABELS

using Lux
using Optimisers
using Zygote

# =============================================================================
# Data Loading
# =============================================================================

"""
Load NER data from JSONL file.
Expected format: {"tokens": [...], "ner_tags": ["O", "B-PERSON", ...]}
"""
function load_jsonl(filepath::String)
    data = []
    open(filepath, "r") do f
        for line in eachline(f)
            if !isempty(strip(line))
                push!(data, JSON3.read(line))
            end
        end
    end
    return data
end

"""
Create vocabulary from data.
"""
function build_vocab(data; min_freq::Int=1, max_vocab::Int=32000)
    word_counts = Dict{String, Int}()

    for example in data
        for token in example.tokens
            word_counts[token] = get(word_counts, token, 0) + 1
        end
    end

    # Sort by frequency
    sorted_words = sort(collect(word_counts), by=x->-x[2])

    # Build vocab with special tokens
    vocab = Dict{String, Int}()
    vocab["[PAD]"] = 1
    vocab["[UNK]"] = 2
    vocab["[CLS]"] = 3
    vocab["[SEP]"] = 4

    idx = 5
    for (word, count) in sorted_words
        if count >= min_freq && idx <= max_vocab
            vocab[word] = idx
            idx += 1
        end
    end

    return vocab
end

"""
Convert tokens to IDs using vocabulary.
"""
function tokenize(tokens::Vector, vocab::Dict{String, Int})
    unk_id = vocab["[UNK]"]
    return [get(vocab, String(t), unk_id) for t in tokens]
end

"""
Convert NER tags to label IDs.
"""
function tags_to_ids(tags::Vector)
    return [get(LABEL_TO_ID, String(t), 1) for t in tags]  # Default to "O" (id=1)
end

"""
Prepare batch from examples.
"""
function prepare_batch(examples, vocab::Dict{String, Int}, max_len::Int)
    batch_size = length(examples)

    # Initialize with padding
    token_ids = ones(Int, max_len, batch_size)  # PAD = 1
    label_ids = fill(-100, max_len, batch_size)  # Ignore index

    for (i, ex) in enumerate(examples)
        tokens = collect(ex.tokens)
        tags = collect(ex.ner_tags)
        seq_len = min(length(tokens), max_len)

        token_ids[1:seq_len, i] = tokenize(tokens[1:seq_len], vocab)
        label_ids[1:seq_len, i] = tags_to_ids(tags[1:seq_len])
    end

    return token_ids, label_ids
end

# =============================================================================
# Training Loop
# =============================================================================

"""
Single training step.
"""
function train_step(model, params, state, opt_state, token_ids, label_ids)
    # Compute loss and gradients
    (loss, new_state), grads = Zygote.withgradient(params) do p
        logits, st = model(token_ids, p, state)
        l = ner_cross_entropy(logits, label_ids)
        return l, st
    end

    # Update parameters
    opt_state, params = Optimisers.update(opt_state, params, grads[1])

    return loss, params, new_state, opt_state
end

"""
Evaluate on validation set.
"""
function evaluate(model, params, state, data, vocab, max_len, batch_size)
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch_start in 1:batch_size:length(data)
        batch_end = min(batch_start + batch_size - 1, length(data))
        batch = data[batch_start:batch_end]

        token_ids, label_ids = prepare_batch(batch, vocab, max_len)

        logits, _ = model(token_ids, params, state)
        loss = ner_cross_entropy(logits, label_ids)
        total_loss += loss * length(batch)

        # Compute accuracy (only on non-padded positions)
        predictions = mapslices(argmax, logits, dims=1)
        mask = label_ids .!= -100
        total_correct += sum((predictions .== label_ids) .& mask)
        total_tokens += sum(mask)
    end

    avg_loss = total_loss / length(data)
    accuracy = total_correct / max(total_tokens, 1)

    return avg_loss, accuracy
end

"""
Main training function.
"""
function train_ner(;
    data_dir::String,
    output_dir::String,
    model_size::Symbol = :small,
    epochs::Int = 10,
    batch_size::Int = 16,
    learning_rate::Float64 = 1e-4,
    max_len::Int = 128,
    eval_every::Int = 100,
    save_every::Int = 1000,
)
    # Create output directory
    mkpath(output_dir)

    # Load data
    println("Loading data from $data_dir...")
    train_data = load_jsonl(joinpath(data_dir, "train.jsonl"))
    val_data = if isfile(joinpath(data_dir, "validation.jsonl"))
        load_jsonl(joinpath(data_dir, "validation.jsonl"))
    elseif isfile(joinpath(data_dir, "val.jsonl"))
        load_jsonl(joinpath(data_dir, "val.jsonl"))
    else
        train_data[1:min(1000, length(train_data))]  # Use subset of train
    end

    println("  Train: $(length(train_data)) examples")
    println("  Val: $(length(val_data)) examples")

    # Build vocabulary
    println("Building vocabulary...")
    vocab = build_vocab(train_data; max_vocab=32000)
    vocab_size = length(vocab)
    println("  Vocabulary size: $vocab_size")

    # Save vocabulary
    open(joinpath(output_dir, "vocab.json"), "w") do f
        JSON3.write(f, vocab)
    end

    # Create model
    println("Creating model (size=$model_size)...")
    model = if model_size == :tiny
        tiny_ner(vocab_size=vocab_size, max_sequence_length=max_len)
    elseif model_size == :small
        small_ner(vocab_size=vocab_size, max_sequence_length=max_len)
    elseif model_size == :base
        base_ner(vocab_size=vocab_size, max_sequence_length=max_len)
    else
        error("Unknown model size: $model_size")
    end

    # Initialize
    rng = Random.default_rng()
    params = Lux.initialparameters(rng, model)
    state = Lux.initialstates(rng, model)

    # Count parameters
    param_count = sum(length, Lux.parameterlength(params))
    println("  Parameters: $(param_count)")

    # Optimizer
    opt = Optimisers.Adam(learning_rate)
    opt_state = Optimisers.setup(opt, params)

    # Training loop
    println("\nStarting training...")
    println("  Epochs: $epochs")
    println("  Batch size: $batch_size")
    println("  Learning rate: $learning_rate")
    println("-" ^ 60)

    global_step = 0
    best_val_loss = Inf

    for epoch in 1:epochs
        epoch_loss = 0.0
        epoch_steps = 0

        # Shuffle training data
        shuffled_data = shuffle(rng, train_data)

        for batch_start in 1:batch_size:length(shuffled_data)
            batch_end = min(batch_start + batch_size - 1, length(shuffled_data))
            batch = shuffled_data[batch_start:batch_end]

            # Prepare batch
            token_ids, label_ids = prepare_batch(batch, vocab, max_len)

            # Training step
            loss, params, state, opt_state = train_step(
                model, params, state, opt_state, token_ids, label_ids
            )

            epoch_loss += loss
            epoch_steps += 1
            global_step += 1

            # Evaluate periodically
            if global_step % eval_every == 0
                val_loss, val_acc = evaluate(model, params, state, val_data, vocab, max_len, batch_size)
                @printf("Step %d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%%\n",
                    global_step, loss, val_loss, val_acc * 100)

                # Save best model
                if val_loss < best_val_loss
                    best_val_loss = val_loss
                    # TODO: Save model checkpoint
                end
            end

            # Save checkpoint
            if global_step % save_every == 0
                # TODO: Save checkpoint
            end
        end

        avg_epoch_loss = epoch_loss / epoch_steps
        val_loss, val_acc = evaluate(model, params, state, val_data, vocab, max_len, batch_size)

        @printf("\nEpoch %d/%d | Avg Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%%\n",
            epoch, epochs, avg_epoch_loss, val_loss, val_acc * 100)
        println("-" ^ 60)
    end

    println("\nTraining complete!")
    println("Best validation loss: $best_val_loss")

    return model, params, state, vocab
end

# =============================================================================
# Quick Test Mode
# =============================================================================

function run_test()
    println("Running quick test with synthetic data...\n")

    # Create tiny model
    model = tiny_ner(vocab_size=100, max_sequence_length=32)
    rng = Random.default_rng()
    params = Lux.initialparameters(rng, model)
    state = Lux.initialstates(rng, model)

    # Optimizer
    opt = Optimisers.Adam(1e-3)
    opt_state = Optimisers.setup(opt, params)

    println("Model created. Training on synthetic data...")

    # Synthetic training
    for step in 1:100
        # Random batch
        token_ids = rand(1:100, 32, 8)
        label_ids = rand(1:NUM_LABELS, 32, 8)

        loss, params, state, opt_state = train_step(
            model, params, state, opt_state, token_ids, label_ids
        )

        if step % 20 == 0
            println("  Step $step | Loss: $(@sprintf("%.4f", loss))")
        end
    end

    # Test inference
    println("\nTesting inference...")
    test_tokens = rand(1:100, 32)
    labels = predict_labels(model, params, state, test_tokens)

    # Count label distribution
    label_counts = Dict{String, Int}()
    for l in labels
        label_counts[l] = get(label_counts, l, 0) + 1
    end

    println("Predicted label distribution:")
    for (label, count) in sort(collect(label_counts), by=x->-x[2])
        println("  $label: $count")
    end

    println("\n✅ Test passed!")
end

# =============================================================================
# Main
# =============================================================================

function main()
    if "--test" in ARGS
        run_test()
        return
    end

    # Parse arguments
    data_dir = ""
    output_dir = "models/ner"
    model_size = :small
    epochs = 10
    batch_size = 16
    learning_rate = 1e-4

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--data"
            data_dir = ARGS[i+1]
            i += 2
        elseif arg == "--output"
            output_dir = ARGS[i+1]
            i += 2
        elseif arg == "--model"
            model_size = Symbol(ARGS[i+1])
            i += 2
        elseif arg == "--epochs"
            epochs = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--batch-size"
            batch_size = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--lr"
            learning_rate = parse(Float64, ARGS[i+1])
            i += 2
        else
            i += 1
        end
    end

    if isempty(data_dir)
        println("""
Usage:
    julia --project=. scripts/train_ner.jl --data DATA_DIR --output OUTPUT_DIR [OPTIONS]
    julia --project=. scripts/train_ner.jl --test

Options:
    --data DIR        Input data directory (with train.jsonl, validation.jsonl)
    --output DIR      Output directory for model (default: models/ner)
    --model SIZE      Model size: tiny, small, base (default: small)
    --epochs N        Number of epochs (default: 10)
    --batch-size N    Batch size (default: 16)
    --lr RATE         Learning rate (default: 1e-4)
    --test            Run quick test with synthetic data
        """)
        return
    end

    train_ner(
        data_dir=data_dir,
        output_dir=output_dir,
        model_size=model_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
end

main()
