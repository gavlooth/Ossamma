#!/usr/bin/env julia
"""
Export model parameters for ONNX conversion.
Uses the existing checkpoint loader from training script.
"""

using Serialization
using NPZ
using JSON3
using Lux
using CUDA
using LuxCUDA

# Load Ossamma
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Ossamma
using Ossamma.NER: NERConfig, load_ner_config

# Include the training script's TrainingConfig (must match the one used for saving)
# We'll extract just the struct definition
Base.@kwdef mutable struct TrainingConfig
    vocab_size::Int = 32000
    max_sequence_length::Int = 128
    embedding_dimension::Int = 256
    number_of_heads::Int = 4
    number_of_layers::Int = 4
    time_dimension::Int = 128
    state_dimension::Int = 256
    window_size::Int = 32
    dropout_rate::Float32 = 0.1f0
    batch_size::Int = 32
    gradient_accumulation_steps::Int = 2
    learning_rate::Float64 = 2e-4
    min_learning_rate::Float64 = 1e-6
    warmup_steps::Int = 500
    total_steps::Int = 10000
    gradient_clip::Float64 = 1.0
    weight_decay::Float64 = 0.01
    eval_every::Int = 500
    log_every::Int = 50
    save_every::Int = 2000
    push_every::Int = 5000
    data_dir::String = "data/ner"
    checkpoint_dir::String = "checkpoints/ner_110m"
    use_crf::Bool = true
    use_boundary_head::Bool = true
    use_ffn::Bool = true
    ffn_expansion::Float32 = 1.333333f0
    use_parallel_scan::Bool = false
    parallel_chunk_size::Int = 64
end

# Simple CPU device
cpu_dev(x) = x
cpu_dev(x::CUDA.CuArray) = Array(x)

function cpu_recursive(x::NamedTuple)
    NamedTuple{keys(x)}(Tuple(cpu_recursive(v) for v in values(x)))
end
cpu_recursive(x::AbstractArray) = cpu_dev(x)
cpu_recursive(x) = x

function extract_arrays(data, prefix="")
    result = Dict{String, Array}()

    if data isa NamedTuple
        for k in keys(data)
            v = data[k]
            key = prefix == "" ? string(k) : prefix * "." * string(k)
            merge!(result, extract_arrays(v, key))
        end
    elseif data isa AbstractDict
        for (k, v) in data
            key = prefix == "" ? string(k) : prefix * "." * string(k)
            merge!(result, extract_arrays(v, key))
        end
    elseif data isa AbstractArray{<:Number}
        arr = data isa CUDA.CuArray ? Array(data) : Array(data)
        result[prefix] = arr
    elseif data isa Number
        result[prefix] = [data]
    end

    return result
end

function try_load_checkpoint(path::String)
    println("Attempting to load checkpoint: $path")

    # Direct deserialize - we have the matching TrainingConfig defined above
    try
        data = deserialize(path)
        println("Loaded successfully")
        return data
    catch e
        println("Deserialize failed: $e")
        return nothing
    end
end

function main()
    if length(ARGS) < 2
        println("Usage: julia export_for_onnx.jl <checkpoint.jls> <output.npz>")
        println("       julia export_for_onnx.jl <config.toml> <checkpoint.jls> <output.npz>")
        return
    end

    # Determine if first arg is a config or checkpoint
    if endswith(ARGS[1], ".toml")
        config_path = ARGS[1]
        checkpoint_path = ARGS[2]
        output_path = ARGS[3]

        println("Loading config: $config_path")
        config = load_ner_config(config_path)
    else
        checkpoint_path = ARGS[1]
        output_path = ARGS[2]
        config = nothing
    end

    data = try_load_checkpoint(checkpoint_path)

    if data === nothing
        println("ERROR: Could not load checkpoint")
        println("\nThe checkpoint format may be incompatible.")
        println("Creating a model from config and saving architecture info instead.")

        if config !== nothing
            # Save config as JSON for PyTorch recreation
            config_path = replace(output_path, ".npz" => ".config.json")
            config_dict = Dict(
                "vocab_size" => config.vocab_size,
                "max_sequence_length" => config.max_sequence_length,
                "embedding_dimension" => config.embedding_dimension,
                "number_of_heads" => config.number_of_heads,
                "number_of_layers" => config.number_of_layers,
                "num_labels" => config.num_labels,
                "time_dimension" => config.time_dimension,
                "state_dimension" => config.state_dimension == -1 ? config.embedding_dimension : config.state_dimension,
                "window_size" => config.window_size,
                "dropout_rate" => config.dropout_rate,
                "use_ffn" => config.use_ffn,
                "ffn_expansion" => config.ffn_expansion,
            )

            open(config_path, "w") do f
                JSON3.pretty(f, config_dict)
            end
            println("Saved config to: $config_path")
        end
        return
    end

    println("Checkpoint type: $(typeof(data))")
    if data isa Dict || data isa NamedTuple
        ks = data isa Dict ? keys(data) : keys(data)
        println("Keys: $(collect(ks))")
    end

    # Extract params
    params = if data isa Dict
        get(data, :params, nothing)
    elseif data isa NamedTuple
        hasproperty(data, :params) ? data.params : nothing
    else
        nothing
    end

    if params === nothing
        println("ERROR: Could not extract params from checkpoint")
        return
    end

    # Convert to CPU
    println("Converting to CPU arrays...")
    params_cpu = cpu_recursive(params)

    println("Extracting arrays...")
    arrays = extract_arrays(params_cpu)

    println("Found $(length(arrays)) parameter arrays")
    total_params = sum(length(v) for (k, v) in arrays)
    println("Total parameters: $(total_params) (~$(round(total_params / 1e6, digits=1))M)")

    println("\nSaving to: $output_path")
    NPZ.npzwrite(output_path, arrays)

    println("Done!")
end

main()
