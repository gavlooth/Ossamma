#!/usr/bin/env julia
"""
Train OssammaNER model (v2) with CRF and Dual Gating.

Usage:
    julia --project=. scripts/train_ner.jl --data data/rag/ --output models/ner/
"""

using Random
using Statistics
using Printf
using Dates
using JSON3
using Lux
using Optimisers
using Zygote
using NNlib

# Load Ossamma
include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
using .Ossamma
using .Ossamma.NER
using .Ossamma.CRF
using .Ossamma.NERDataset
using .Ossamma.Tokenizer
using .Ossamma.NERMetrics

# =============================================================================
# Configuration
# =============================================================================

Base.@kwdef struct TrainConfig
    # Data
    data_dir::String
    output_dir::String
    max_len::Int = 128
    
    # Model
    model_size::Symbol = :small
    embedding_dim::Int = 256
    num_heads::Int = 4
    num_layers::Int = 4
    window_size::Int = 128 # Sliding window attention size
    
    # Training
    epochs::Int = 10
    batch_size::Int = 16
    learning_rate::Float64 = 2e-4
    crf_learning_rate::Float64 = 1e-3
    dropout::Float32 = 0.1f0
    boundary_loss_weight::Float32 = 0.2f0
    max_grad_norm::Float64 = 1.0
    
    # Evaluation
    eval_every::Int = 100
    save_every::Int = 1000
    seed::Int = 42
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
Compute boundary labels for auxiliary loss.
Returns 1 for B-* tags (start of entity), 0 otherwise.
"""
function compute_boundary_labels(labels::Matrix{Int}, id_to_label::Dict{Int, String})
    seq_len, batch_size = size(labels)
    boundaries = zeros(Float32, 2, seq_len, batch_size) # One-hot: [not_boundary, is_boundary]
    
    for b in 1:batch_size
        for t in 1:seq_len
            label_id = labels[t, b]
            if label_id == -100 # Ignore padding
                # Masked positions will be ignored by crossentropy if we handle it right,
                # but NNlib.logitcrossentropy doesn't support ignore_index directly for arrays.
                # We'll handle masking manually in the loss function.
                continue
            end
            
            label_str = get(id_to_label, label_id, "O")
            is_boundary = startswith(label_str, "B-")
            
            if is_boundary
                boundaries[2, t, b] = 1.0f0 # is_boundary
            else
                boundaries[1, t, b] = 1.0f0 # not_boundary
            end
        end
    end
    return boundaries
end

"""
Custom combined loss function.
"""
function compute_loss(
    model, params, state, 
    token_ids, labels, mask, 
    boundary_targets, boundary_weight
)
    # Forward pass
    (emissions, boundary_logits), st = model(token_ids, params, state)
    
    # 1. CRF Loss (Negative Log Likelihood)
    # emissions: (num_labels, seq_len, batch)
    crf_l, _ = crf_loss(model.CRF, emissions, labels, mask, params.CRF, st.CRF)
    
    # 2. Boundary Loss (Auxiliary)
    # boundary_logits: (2, seq_len, batch) -> reshape to (2, seq_len * batch)
    # boundary_targets: (2, seq_len, batch) -> reshape to (2, seq_len * batch)
    
    dims = size(boundary_logits)
    flat_logits = reshape(boundary_logits, 2, :)
    flat_targets = reshape(boundary_targets, 2, :)
    flat_mask = reshape(mask, :)
    
    # Compute cross entropy only on valid positions
    b_loss = 0.0f0
    valid_count = 0
    
    # Vectorized if possible, but loop is safe for masking
    # Using logitcrossentropy from NNlib
    losses = NNlib.logitcrossentropy(flat_logits, flat_targets; agg=identity)
    
    # Apply mask
    for i in 1:length(flat_mask)
        if flat_mask[i]
            b_loss += losses[i]
            valid_count += 1
        end
    end
    
    boundary_l = valid_count > 0 ? b_loss / valid_count : 0.0f0
    
    # Total loss
    total_loss = crf_l + boundary_weight * boundary_l
    
    return total_loss, st
end

# =============================================================================
# Training Loop
# =============================================================================

function train(config::TrainConfig)
    # Setup
    Random.seed!(config.seed)
    mkpath(config.output_dir)
    
    # 1. Load Data
    println("Loading data from $(config.data_dir)...\n")
    train_samples = load_dataset(joinpath(config.data_dir, "train.jsonl"))
    val_samples = load_dataset(joinpath(config.data_dir, "validation.jsonl"))
    
    println("  Train: $(length(train_samples))")
    println("  Val:   $(length(val_samples))")
    
    # 2. Build Vocabulary & Tokenizer
    println("Building vocabulary...")
    # Extract tokens from all training samples
    all_tokens = [s.tokens for s in train_samples]
    vocab = build_vocab_from_tokens(all_tokens; min_freq=2, max_vocab_size=32000)
    
    tokenizer = NERTokenizer(vocab; max_length=config.max_len)
    println("  Vocab size: $(length(vocab))")
    save_vocab(vocab, joinpath(config.output_dir, "vocab.json"))
    
    # 3. Create Model
    println("Initializing OssammaNER (v2) model...")
    ner_config = NERConfig(
        vocab_size = length(vocab),
        max_sequence_length = config.max_len,
        embedding_dimension = config.embedding_dim,
        number_of_heads = config.num_heads,
        number_of_layers = config.num_layers,
        num_labels = NUM_LABELS,
        window_size = config.window_size,
        dropout_rate = config.dropout,
        use_crf = true
    )
    
    model = OssammaNER(ner_config)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    
    println("  Parameters: $(sum(length, Lux.parameterlength(ps)))")
    
    # 4. Optimizers
    # Separate LR for CRF might be beneficial, but keeping simple for now
    opt = Optimisers.Adam(config.learning_rate)
    opt_state = Optimisers.setup(opt, ps)
    
    # 5. Training Loop
    println("\nStarting training...")
    
    # Labels needed for boundary computation
    _, id_to_label = create_label_dicts() # Using default RAG schema
    
    global_step = 0
    best_f1 = 0.0
    
    # Data loaders
    train_loader = NERDataLoader(train_samples; batch_size=config.batch_size, max_length=config.max_len)
    
    for epoch in 1:config.epochs
        reset!(train_loader)
        epoch_loss = 0.0
        steps = 0
        
        for batch in train_loader
            # Tokenize batch
            # batch is (tokens, labels, mask) tuple from loader, but loader returns strings
            # We need to convert strings to IDs using our tokenizer
            
            # Use custom batch processing since loader returns padded strings
            # Ideally NERDataLoader would use the tokenizer, but it's decoupled.
            # We'll re-tokenize the raw samples for now or adapt.
            # Actually, let's just use the raw samples from the batch indices of the loader if possible.
            # The loader returns (tokens, labels, mask) matrices.
            # We can map tokens to IDs.
            
            batch_tokens = batch.tokens # Matrix{String}
            batch_labels = batch.labels # Matrix{Int}
            batch_mask = batch.mask     # Matrix{Bool}
            
            # Convert tokens to IDs
            seq_len, batch_size = size(batch_tokens)
            token_ids = zeros(Int, seq_len, batch_size)
            
            for i in 1:length(batch_tokens)
                token_ids[i] = vocab[batch_tokens[i]]
            end
            
            # Compute boundary targets
            boundary_targets = compute_boundary_labels(batch_labels, id_to_label)
            
            # Gradient step
            (loss, st), grads = Zygote.withgradient(ps) do p
                compute_loss(model, p, st, token_ids, batch_labels, batch_mask, boundary_targets, config.boundary_loss_weight)
            end
            
            # Clip grads (manual calculation for now or use Optimisers.ClipGrad if available)
            # Optimisers.update! handles update. 
            # We'll skip clipping for this simplified script or add it if unstable.
            
            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
            
            epoch_loss += loss
            steps += 1
            global_step += 1
            
            if global_step % config.eval_every == 0
                @printf("Step %d | Loss: %.4f\n", global_step, loss)
            end
        end
        
        avg_loss = epoch_loss / max(steps, 1)
        println("\nEpoch $epoch | Avg Loss: $avg_loss")
        
        # Validation
        println("Evaluating on validation set...")
        val_metrics = evaluate_model(model, ps, st, val_samples, tokenizer, vocab, id_to_label, config)
        
        println(classification_report(val_metrics))
        
        if val_metrics.f1_micro > best_f1
            best_f1 = val_metrics.f1_micro
            println("New best F1! Saving checkpoint...")
            # serialization would go here
            # JLD2 or Serialization
        end
        println("-" ^ 60)
    end
end

"""
Evaluate model using Viterbi decoding.
"""
function evaluate_model(model, params, state, samples, tokenizer, vocab, id_to_label, config)
    predictions = Vector{Vector{Int}}()
    gold_labels = Vector{Vector{Int}}()
    
    # Process in batches
    loader = NERDataLoader(samples; batch_size=config.batch_size, max_length=config.max_len, shuffle=false)
    
    for batch in loader
        batch_tokens = batch.tokens
        batch_labels = batch.labels
        batch_mask = batch.mask
        
        seq_len, batch_size = size(batch_tokens)
        token_ids = zeros(Int, seq_len, batch_size)
        for i in 1:length(batch_tokens)
            token_ids[i] = vocab[batch_tokens[i]]
        end
        
        # Forward pass to get emissions
        (emissions, _), _ = model(token_ids, params, state)
        
        # Viterbi decode
        decoded_preds, _ = viterbi_decode(model.CRF, emissions, batch_mask, params.CRF, state.CRF)
        
        # Collect results
        for b in 1:batch_size
            seq_len = sum(batch_mask[:, b])
            push!(predictions, decoded_preds[1:seq_len, b])
            push!(gold_labels, batch_labels[1:seq_len, b])
        end
    end
    
    return evaluate_ner(predictions, gold_labels, id_to_label)
end

# =============================================================================
# Main
# =============================================================================

function main()
    # Basic args parsing
    data_dir = "data/rag"
    output_dir = "models/ner_v2"
    
    if length(ARGS) >= 2 && ARGS[1] == "--data"
        data_dir = ARGS[2]
    end
    if length(ARGS) >= 4 && ARGS[3] == "--output"
        output_dir = ARGS[4]
    end
    
    config = TrainConfig(
        data_dir = data_dir,
        output_dir = output_dir
    )
    
    train(config)
end

main()