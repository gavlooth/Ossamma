#!/usr/bin/env julia
"""
Synthetic Data Augmentation Script for OssammaNER.

1. "Pumps up" specific categories (like WORK) using templates + word lists.
2. Adds "Robustness" by generating lowercased/noisy versions of existing data.

Usage:
    julia scripts/augment_synthetic.jl --pump-work --robustness --input data/rag/final_train.jsonl
"""

using JSON3
using Random
using ProgressMeter

# =============================================================================
# 1. TEMPLATES FOR "WORK" (Pumping Up)
# =============================================================================

const WORK_TEMPLATES = [
    # Reading
    (["I", "read", "{WORK}", "yesterday", "."], ["O", "O", "B-WORK", "O", "O"]),
    (["Have", "you", "seen", "{WORK}", "?"], ["O", "O", "O", "B-WORK", "O"]),
    (["The", "plot", "of", "{WORK}", "is", "complex", "."], ["O", "O", "O", "B-WORK", "O", "O", "O"]),
    (["In", "{WORK}", ",", "the", "author", "argues", "that", "..."], ["O", "B-WORK", "O", "O", "O", "O", "O", "O"]),
    (["Chapter", "3", "of", "{WORK}", "discusses", "this", "."], ["O", "O", "O", "B-WORK", "O", "O", "O"]),
    # Watching/Listening
    (["We", "watched", "{WORK}", "at", "the", "cinema", "."], ["O", "O", "B-WORK", "O", "O", "O", "O"]),
    (["I", "listened", "to", "{WORK}", "on", "Spotify", "."], ["O", "O", "O", "B-WORK", "O", "O", "O"]),
    (["The", "song", "{WORK}", "is", "stuck", "in", "my", "head", "."], ["O", "O", "B-WORK", "O", "O", "O", "O", "O", "O"]),
    (["The", "movie", "{WORK}", "won", "an", "award", "."], ["O", "O", "B-WORK", "O", "O", "O", "O"]),
    # Academic/Professional
    (["According", "to", "{WORK}", ",", "the", "results", "are", "valid", "."], ["O", "O", "B-WORK", "O", "O", "O", "O", "O", "O"]),
    (["The", "documentation", "for", "{WORK}", "is", "excellent", "."], ["O", "O", "O", "B-WORK", "O", "O", "O"]),
]

# A small seed list of works. 
# INSTRUCTION: The user can replace this with a massive list loaded from a file.
const SEED_WORKS = [
    # Books
    "1984", "The Great Gatsby", "Pride and Prejudice", "Dune", "The Hobbit",
    "Thinking, Fast and Slow", "Sapiens", "Clean Code", "Introduction to Algorithms",
    # Movies/TV
    "The Matrix", "Inception", "Breaking Bad", "The Wire", "Star Wars",
    "Pulp Fiction", "The Godfather", "Interstellar", "Parasite",
    # Papers/Journals
    "Attention Is All You Need", "Nature", "Science", "The New York Times",
    "ResNet", "BERT: Pre-training of Deep Bidirectional Transformers",
    # Software (can be WORK or INSTRUMENT, context dependent)
    "TensorFlow", "PyTorch", "React", "Linux"
]

function generate_work_synthetic(n_samples::Int=1000)
    println("Generating $n_samples synthetic examples for WORK...")
    data = []
    
    for _ in 1:n_samples
        template_tokens, template_labels = rand(WORK_TEMPLATES)
        work_title = rand(SEED_WORKS)
        
        # Tokenize the title simply by splitting on space (naïve but works for synthetic)
        title_parts = split(work_title)
        n_parts = length(title_parts)
        
        # Construct new sentence
        new_tokens = String[]
        new_labels = String[]
        
        for (tok, lbl) in zip(template_tokens, template_labels)
            if tok == "{WORK}"
                append!(new_tokens, title_parts)
                # B-WORK, then I-WORK for the rest
                push!(new_labels, "B-WORK")
                for _ in 2:n_parts
                    push!(new_labels, "I-WORK")
                end
            else
                push!(new_tokens, tok)
                push!(new_labels, lbl)
            end
        end
        
        push!(data, Dict("tokens" => new_tokens, "ner_tags" => new_labels))
    end
    
    return data
end

# =============================================================================
# 2. ROBUSTNESS (Lowercasing)
# =============================================================================

function make_robust(file_path::String, output_path::String, probability::Float64=0.15)
    println("Creating robust version of $file_path (p=$probability)...")
    
    open(output_path, "w") do out_io
        lines = readlines(file_path)
        count = 0
        
        @showprogress for line in lines
            record = JSON3.read(line)
            tokens = String.(record.tokens)
            labels = String.(record.ner_tags)
            
            # 1. Write original
            println(out_io, JSON3.write(Dict("tokens" => tokens, "ner_tags" => labels)))
            
            # 2. Maybe create a lowercase version
            if rand() < probability
                # Lowercase all tokens
                lower_tokens = lowercase.(tokens)
                
                # Write lowercase version
                println(out_io, JSON3.write(Dict("tokens" => lower_tokens, "ner_tags" => labels)))
                count += 1
            end
        end
        println("  Added $count lowercase examples.")
    end
end

# =============================================================================
# MAIN
# =============================================================================

function main()
    args = ARGS
    
    if isempty(args)
        println("Usage: julia scripts/augment_synthetic.jl --pump-work --robustness --input <file>")
        return
    end
    
    output_dir = "data/rag"
    
    # 1. PUMP WORK
    if "--pump-work" in args
        synthetic_data = generate_work_synthetic(2000) # Generate 2000 examples
        
        out_path = joinpath(output_dir, "synthetic_work.jsonl")
        open(out_path, "w") do io
            for row in synthetic_data
                println(io, JSON3.write(row))
            end
        end
        println("Saved synthetic WORK data to $out_path")
    end
    
    # 2. ROBUSTNESS
    input_idx = findfirst(==("--input"), args)
    if "--robustness" in args && input_idx !== nothing
        input_file = args[input_idx + 1]
        
        # Determine output filename
        base, ext = splitext(basename(input_file))
        output_file = joinpath(dirname(input_file), "$(base)_robust$(ext)")
        
        make_robust(input_file, output_file, 0.15) # 15% duplication rate
        println("Saved robust dataset to $output_file")
    end
end

main()
