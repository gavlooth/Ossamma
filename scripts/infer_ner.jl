#!/usr/bin/env julia
"""
OssammaNER Inference Script

Simple script for running NER inference on text using the trained model.

Usage:
    julia --project=. scripts/infer_ner.jl "John Smith works at Google in New York."
    julia --project=. scripts/infer_ner.jl --file input.txt
    julia --project=. scripts/infer_ner.jl --interactive
"""

using Random
using Serialization
using JSON3
using ArgParse

# Load Ossamma
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Ossamma
using Ossamma: OssammaNER, NERConfig
using Ossamma.NER: ID_TO_LABEL, LABEL_TO_ID, extract_entities

using Lux

# =============================================================================
# Configuration
# =============================================================================

const DEFAULT_CHECKPOINT = "checkpoints/ner_110m/checkpoint_best.jls"
const DEFAULT_VOCAB = "checkpoints/ner_110m/vocab.json"

# =============================================================================
# Model Loading
# =============================================================================

function load_model(checkpoint_path::String, vocab_path::String)
    println("Loading vocabulary: $vocab_path")
    vocab = JSON3.read(read(vocab_path, String), Dict{String,Int})
    id_to_token = Dict(v => k for (k, v) in vocab)
    println("  Vocab size: $(length(vocab))")

    println("Loading checkpoint: $checkpoint_path")
    checkpoint = deserialize(checkpoint_path)
    config = checkpoint[:config]
    println("  Loaded from step: $(checkpoint[:step])")

    # Create model with inference settings
    ner_config = NERConfig(
        vocab_size = length(vocab),
        max_sequence_length = config.max_sequence_length,
        embedding_dimension = config.embedding_dimension,
        number_of_heads = config.number_of_heads,
        number_of_layers = config.number_of_layers,
        time_dimension = config.time_dimension,
        state_dimension = config.state_dimension,
        window_size = config.window_size,
        dropout_rate = 0.0f0,  # No dropout for inference
        use_ffn = config.use_ffn,
        ffn_expansion = config.ffn_expansion,
    )

    model = OssammaNER(ner_config)
    params = checkpoint[:params]
    state = Lux.testmode(checkpoint[:state])

    return model, params, state, vocab, id_to_token, config
end

# =============================================================================
# Tokenization
# =============================================================================

function tokenize(text::String, vocab::Dict{String,Int})
    # Simple word tokenization with punctuation handling
    unk_id = get(vocab, "[UNK]", get(vocab, "<UNK>", 2))

    # Split on whitespace, handle punctuation
    raw_tokens = String[]
    for word in split(text)
        # Separate leading/trailing punctuation
        m = match(r"^([.,!?;:\"'()\[\]{}]*)(.+?)([.,!?;:\"'()\[\]{}]*)$", word)
        if m !== nothing
            !isempty(m[1]) && push!(raw_tokens, m[1])
            push!(raw_tokens, m[2])
            !isempty(m[3]) && push!(raw_tokens, m[3])
        else
            push!(raw_tokens, String(word))
        end
    end

    # Convert to IDs (lowercase for lookup)
    token_ids = Int[]
    original_tokens = String[]
    for token in raw_tokens
        push!(original_tokens, token)
        token_lower = lowercase(token)
        push!(token_ids, get(vocab, token_lower, get(vocab, token, unk_id)))
    end

    return token_ids, original_tokens
end

# =============================================================================
# Inference
# =============================================================================

function predict(model, params, state, token_ids::Vector{Int}, max_seq_len::Int)
    # Pad sequence
    pad_id = 1  # Assuming PAD is index 1
    seq_len = min(length(token_ids), max_seq_len)
    padded = fill(pad_id, max_seq_len)
    padded[1:seq_len] = token_ids[1:seq_len]

    # Create batch (seq_len, batch_size)
    input_batch = reshape(padded, :, 1)

    # Run model
    (emissions, boundary_logits), _ = model(input_batch, params, state)

    # Get predictions (argmax)
    preds = vec(mapslices(argmax, Array(emissions[:, 1:seq_len, 1]), dims=1))

    return preds
end

function format_entities(tokens::Vector{String}, labels::Vector{String})
    entities = NamedTuple{(:text, :label, :start, :end_), Tuple{String, String, Int, Int}}[]

    current_entity = String[]
    current_type = ""
    current_start = 0

    for (i, (token, label)) in enumerate(zip(tokens, labels))
        if startswith(label, "B-")
            # Save previous entity
            if !isempty(current_entity)
                push!(entities, (
                    text = join(current_entity, " "),
                    label = current_type,
                    start = current_start,
                    end_ = i - 1,
                ))
            end
            # Start new entity
            current_entity = [token]
            current_type = label[3:end]
            current_start = i
        elseif startswith(label, "I-") && !isempty(current_entity)
            push!(current_entity, token)
        else
            # End entity
            if !isempty(current_entity)
                push!(entities, (
                    text = join(current_entity, " "),
                    label = current_type,
                    start = current_start,
                    end_ = i - 1,
                ))
                current_entity = String[]
                current_type = ""
            end
        end
    end

    # Final entity
    if !isempty(current_entity)
        push!(entities, (
            text = join(current_entity, " "),
            label = current_type,
            start = current_start,
            end_ = length(tokens),
        ))
    end

    return entities
end

function run_inference(text::String, model, params, state, vocab, id_to_token, max_seq_len)
    # Tokenize
    token_ids, tokens = tokenize(text, vocab)

    if isempty(token_ids)
        return String[], NamedTuple[]
    end

    # Predict
    preds = predict(model, params, state, token_ids, max_seq_len)

    # Convert to labels
    labels = [get(ID_TO_LABEL, p, "O") for p in preds]

    # Extract entities
    entities = format_entities(tokens, labels)

    return labels, entities
end

# =============================================================================
# Main
# =============================================================================

function parse_args()
    s = ArgParseSettings(description = "OssammaNER Inference")

    @add_arg_table! s begin
        "text"
            help = "Text to analyze"
            required = false
        "--checkpoint"
            help = "Path to model checkpoint"
            default = DEFAULT_CHECKPOINT
        "--vocab"
            help = "Path to vocabulary JSON"
            default = DEFAULT_VOCAB
        "--file", "-f"
            help = "Input file (one sentence per line)"
        "--interactive", "-i"
            help = "Interactive mode"
            action = :store_true
        "--json"
            help = "Output in JSON format"
            action = :store_true
    end

    return ArgParse.parse_args(s)
end

function main()
    args = parse_args()

    # Load model
    println("=" ^ 60)
    println("OssammaNER Inference")
    println("=" ^ 60)

    model, params, state, vocab, id_to_token, config = load_model(
        args["checkpoint"],
        args["vocab"]
    )
    max_seq_len = config.max_sequence_length

    println("\nModel ready!\n")

    if args["interactive"]
        # Interactive mode
        println("Enter text (or 'quit' to exit):")
        while true
            print("> ")
            text = readline()
            text = strip(text)

            if lowercase(text) == "quit" || lowercase(text) == "exit"
                break
            end

            if isempty(text)
                continue
            end

            labels, entities = run_inference(text, model, params, state, vocab, id_to_token, max_seq_len)

            if args["json"]
                result = Dict(
                    "text" => text,
                    "entities" => [Dict("text" => e.text, "label" => e.label, "start" => e.start, "end" => e.end_) for e in entities]
                )
                println(JSON3.write(result))
            else
                println("\nEntities found:")
                if isempty(entities)
                    println("  (none)")
                else
                    for e in entities
                        println("  - $(e.text) [$(e.label)]")
                    end
                end
                println()
            end
        end

    elseif args["file"] !== nothing
        # File mode
        for line in eachline(args["file"])
            text = strip(line)
            isempty(text) && continue

            labels, entities = run_inference(text, model, params, state, vocab, id_to_token, max_seq_len)

            if args["json"]
                result = Dict(
                    "text" => text,
                    "entities" => [Dict("text" => e.text, "label" => e.label, "start" => e.start, "end" => e.end_) for e in entities]
                )
                println(JSON3.write(result))
            else
                println("Text: $text")
                for e in entities
                    println("  - $(e.text) [$(e.label)]")
                end
                println()
            end
        end

    elseif args["text"] !== nothing
        # Single text mode
        text = args["text"]
        labels, entities = run_inference(text, model, params, state, vocab, id_to_token, max_seq_len)

        if args["json"]
            result = Dict(
                "text" => text,
                "entities" => [Dict("text" => e.text, "label" => e.label, "start" => e.start, "end" => e.end_) for e in entities]
            )
            println(JSON3.write(result))
        else
            println("Input: $text\n")
            println("Entities:")
            if isempty(entities)
                println("  (none detected)")
            else
                for e in entities
                    println("  - $(e.text) [$(e.label)]")
                end
            end
        end
    else
        println("Usage:")
        println("  julia --project=. scripts/infer_ner.jl \"Your text here\"")
        println("  julia --project=. scripts/infer_ner.jl --interactive")
        println("  julia --project=. scripts/infer_ner.jl --file input.txt")
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
