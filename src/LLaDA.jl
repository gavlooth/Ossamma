module LLaDA

"""
LLaDA-style Text Diffusion Model using Swamma architecture.

Forward (masking):   "The cat sat on mat" → "The [M] sat [M] mat" → "[M] [M] [M] [M] [M]"
Reverse (denoising): "[M] [M] [M] [M] [M]" → "The [M] sat [M] mat" → "The cat sat on mat"

Each step, model predicts masked tokens, we unmask some based on confidence.
"""

using Lux
using Random
using NNlib
using TOML
using ChainRulesCore

# Import parent module components
using ..Swamma: SwammaBlock, LuxLayer, to_device_like, is_gpu_array
using ..EngramMod: EngramModule, subtokens_to_token_ids

function required_subtoken_base(vocab_size::Int, subtoken_length::Int)
    subtoken_length >= 1 || throw(ArgumentError("subtoken_length must be >= 1"))
    base = 2
    while base^subtoken_length < vocab_size
        base += 1
    end
    return base
end

function build_vocab_code_table(vocab_size::Int, base::Int, subtoken_length::Int)
    table = Matrix{Int}(undef, subtoken_length, vocab_size)
    for token_id in 1:vocab_size
        value = token_id - 1
        for j in 1:subtoken_length
            table[j, token_id] = (value % base) + 1
            value ÷= base
        end
    end
    return table
end

function build_subtoken_value_masks(code_table::AbstractMatrix{<:Integer}, base::Int)
    subtoken_length, vocab_size = size(code_table)
    masks = [ [falses(vocab_size) for _ in 1:base] for _ in 1:subtoken_length ]
    for token_id in 1:vocab_size
        for j in 1:subtoken_length
            digit = Int(code_table[j, token_id])
            masks[j][digit][token_id] = true
        end
    end
    return masks
end

function token_ids_to_subtokens(token_ids::AbstractArray{<:Integer}, code_table::AbstractMatrix{<:Integer})
    subtoken_length, vocab_size = size(code_table)
    if ndims(token_ids) == 1
        seq_len = size(token_ids, 1)
        out = Matrix{Int}(undef, subtoken_length, seq_len)
        for s in 1:seq_len
            tid = clamp(Int(token_ids[s]), 1, vocab_size)
            for j in 1:subtoken_length
                out[j, s] = code_table[j, tid]
            end
        end
        return out
    elseif ndims(token_ids) == 2
        seq_len, batch_size = size(token_ids)
        out = Array{Int}(undef, subtoken_length, seq_len, batch_size)
        for b in 1:batch_size
            for s in 1:seq_len
                tid = clamp(Int(token_ids[s, b]), 1, vocab_size)
                for j in 1:subtoken_length
                    out[j, s, b] = code_table[j, tid]
                end
            end
        end
        return out
    else
        throw(ArgumentError("token_ids_to_subtokens expects 1D or 2D token ids, got ndims=$(ndims(token_ids))"))
    end
end

# ============================================================================
# Configuration
# ============================================================================

"""
Configuration struct for LLaDA model, loadable from TOML.
"""
Base.@kwdef struct LLaDAConfig
    # Model architecture
    vocab_size::Int = 32000
    max_sequence_length::Int = 512
    embedding_dimension::Int = 256
    number_of_heads::Int = 4
    number_of_layers::Int = 6

    # Internal dimensions
    time_dimension::Int = 128
    state_dimension::Int = -1  # -1 means use embedding_dimension

    # Attention
    window_size::Int = 5

    # Wave Gate (WavePDE)
    min_frequency::Float32 = 0.1f0
    max_frequency::Float32 = 10.0f0
    default_time_step::Float32 = 0.1f0

    # Generation defaults
    default_num_steps::Int = 10
    default_temperature::Float32 = 1.0f0

    # Training defaults
    mask_schedule::Symbol = :uniform  # :uniform, :cosine

    # MDM-Prime style sub-token parameterization
    prime_subtoken_length::Int = 4
    prime_subtoken_base::Int = 16

    # Engram conditional memory (Ma et al., 2026)
    use_engram::Bool = false
    engram_layers::Vector{Int} = Int[]       # which layers get engram; empty = auto-select
    engram_num_heads::Int = 8                # hash heads per N-gram order
    engram_ngram_orders::Vector{Int} = [2, 3] # N-gram granularities
    engram_table_size::Int = 65536           # entries per hash head
    engram_head_dim::Int = 0                 # 0 = auto (embedding_dim ÷ num_heads)
end

"""
    load_config(path::String) -> LLaDAConfig

Load model configuration from a TOML file.
"""
function load_config(path::String)
    data = TOML.parsefile(path)
    return config_from_dict(data)
end

"""
    load_config(io::IO) -> LLaDAConfig

Load model configuration from an IO stream.
"""
function load_config(io::IO)
    data = TOML.parse(io)
    return config_from_dict(data)
end

"""
    config_from_dict(data::Dict) -> LLaDAConfig

Create config from a parsed TOML dictionary.
"""
function config_from_dict(data::Dict)
    # Recursively flatten nested sections
    flat = Dict{Symbol, Any}()

    function flatten!(d::Dict, prefix::String = "")
        for (key, val) in d
            if val isa Dict
                # Recurse into nested dicts
                flatten!(val, prefix * key * "_")
            else
                # Use flattened key-values directly.
                flat[Symbol(key)] = val
            end
        end
    end

    flatten!(data)

    # Convert mask_schedule string to Symbol
    if haskey(flat, :mask_schedule) && flat[:mask_schedule] isa String
        flat[:mask_schedule] = Symbol(flat[:mask_schedule])
    end

    # Convert vector fields from TOML (Vector{Any} → Vector{Int})
    for key in (:engram_layers, :engram_ngram_orders)
        if haskey(flat, key) && flat[key] isa Vector
            flat[key] = Int.(flat[key])
        end
    end

    # Build config with defaults for missing keys
    return LLaDAConfig(;
        vocab_size = get(flat, :vocab_size, 32000),
        max_sequence_length = get(flat, :max_sequence_length, 512),
        embedding_dimension = get(flat, :embedding_dimension, 256),
        number_of_heads = get(flat, :number_of_heads, 4),
        number_of_layers = get(flat, :number_of_layers, 6),
        time_dimension = get(flat, :time_dimension, 128),
        state_dimension = get(flat, :state_dimension, -1),
        window_size = get(flat, :window_size, 5),
        min_frequency = Float32(get(flat, :min_frequency, 0.1)),
        max_frequency = Float32(get(flat, :max_frequency, 10.0)),
        default_time_step = Float32(get(flat, :default_time_step, 0.1)),
        default_num_steps = get(flat, :default_num_steps, 10),
        default_temperature = Float32(get(flat, :default_temperature, 1.0)),
        mask_schedule = get(flat, :mask_schedule, :uniform),
        prime_subtoken_length = Int(get(flat, :prime_subtoken_length, 4)),
        prime_subtoken_base = Int(get(flat, :prime_subtoken_base, 16)),
        # Engram
        use_engram = get(flat, :use_engram, false),
        engram_layers = Int.(get(flat, :engram_layers, Int[])),
        engram_num_heads = Int(get(flat, :engram_num_heads, 8)),
        engram_ngram_orders = Int.(get(flat, :engram_ngram_orders, [2, 3])),
        engram_table_size = Int(get(flat, :engram_table_size, 65536)),
        engram_head_dim = Int(get(flat, :engram_head_dim, 0)),
    )
end

"""
    save_config(config::LLaDAConfig, path::String)

Save configuration to a TOML file.
"""
function save_config(config::LLaDAConfig, path::String)
    open(path, "w") do io
        save_config(config, io)
    end
end

"""
    save_config(config::LLaDAConfig, io::IO)

Write configuration to an IO stream in TOML format.
"""
function save_config(config::LLaDAConfig, io::IO)
    println(io, "# LLaDA Model Configuration")
    println(io, "# Generated automatically\n")

    println(io, "[model]")
    println(io, "vocab_size = ", config.vocab_size)
    println(io, "max_sequence_length = ", config.max_sequence_length)
    println(io, "embedding_dimension = ", config.embedding_dimension)
    println(io, "number_of_heads = ", config.number_of_heads)
    println(io, "number_of_layers = ", config.number_of_layers)
    println(io)

    println(io, "[model.prime]")
    println(io, "prime_subtoken_length = ", config.prime_subtoken_length)
    println(io, "prime_subtoken_base = ", config.prime_subtoken_base)
    println(io)

    println(io, "[model.dimensions]")
    println(io, "time_dimension = ", config.time_dimension)
    println(io, "state_dimension = ", config.state_dimension)
    println(io)

    println(io, "[model.attention]")
    println(io, "window_size = ", config.window_size)
    println(io)

    println(io, "[model.wave_gate]")
    println(io, "min_frequency = ", config.min_frequency)
    println(io, "max_frequency = ", config.max_frequency)
    println(io, "default_time_step = ", config.default_time_step)
    println(io)

    println(io, "[generation]")
    println(io, "default_num_steps = ", config.default_num_steps)
    println(io, "default_temperature = ", config.default_temperature)
    println(io)

    println(io, "[training]")
    println(io, "mask_schedule = \"", config.mask_schedule, "\"")
end

"""
    default_config() -> LLaDAConfig

Return default configuration.
"""
default_config() = LLaDAConfig()

"""
    small_config() -> LLaDAConfig

Return configuration for a small model (for testing/debugging).
"""
function small_config()
    LLaDAConfig(
        vocab_size = 1000,
        max_sequence_length = 64,
        embedding_dimension = 64,
        number_of_heads = 2,
        number_of_layers = 2,
        time_dimension = 32,
        state_dimension = 64,
    )
end

"""
    base_config() -> LLaDAConfig

Return configuration for a base-sized model.
"""
function base_config()
    LLaDAConfig(
        vocab_size = 32000,
        max_sequence_length = 512,
        embedding_dimension = 512,
        number_of_heads = 8,
        number_of_layers = 12,
        time_dimension = 128,
        state_dimension = 512,
    )
end

"""
    large_config() -> LLaDAConfig

Return configuration for a large model.
"""
function large_config()
    LLaDAConfig(
        vocab_size = 32000,
        max_sequence_length = 1024,
        embedding_dimension = 1024,
        number_of_heads = 16,
        number_of_layers = 24,
        time_dimension = 256,
        state_dimension = 1024,
    )
end

"""
    production_config() -> LLaDAConfig

Return a production-ready configuration optimized for 24GB+ GPUs.
Well-balanced for serious training with sensible oscillator dynamics.
"""
function production_config()
    LLaDAConfig(
        vocab_size = 32000,
        max_sequence_length = 1024,
        embedding_dimension = 768,
        number_of_heads = 12,
        number_of_layers = 12,
        time_dimension = 256,
        state_dimension = 768,
        window_size = 128,
        min_frequency = 0.01f0,
        max_frequency = 5.0f0,
        default_time_step = 0.05f0,
        default_num_steps = 50,
        default_temperature = 0.8f0,
        mask_schedule = :cosine,
    )
end

# ============================================================================
# Sinusoidal Time Embedding (for mask ratio t ∈ [0,1])
# ============================================================================
struct SinusoidalTimeEmbedding <: LuxLayer
    time_dimension::Int
    max_period::Float32
end

function SinusoidalTimeEmbedding(time_dimension::Int; max_period::Float32 = 10000.0f0)
    @assert time_dimension % 2 == 0 "time_dimension must be even"
    return SinusoidalTimeEmbedding(time_dimension, max_period)
end

function Lux.initialparameters(::Random.AbstractRNG, ::SinusoidalTimeEmbedding)
    return (;)  # No learnable parameters
end

function Lux.initialstates(::Random.AbstractRNG, ::SinusoidalTimeEmbedding)
    return (;)
end

function (layer::SinusoidalTimeEmbedding)(t, params, state)
    # t: scalar or (1, batch) - mask ratio in [0, 1]
    half_dim = layer.time_dimension ÷ 2

    # Frequency bands - create on CPU then transfer to match input device
    freqs_cpu = Float32.(exp.(-(log(layer.max_period)) .* collect(0:half_dim-1) ./ half_dim))
    freqs = to_device_like(t, freqs_cpu)

    # Handle batched input
    if ndims(t) == 0 || (ndims(t) == 2 && size(t, 1) == 1)
        t_flat = ndims(t) == 0 ? reshape([t[1]], :) : vec(t)
        # (half_dim,) * (batch,)' → (half_dim, batch)
        args = freqs * t_flat'
    else
        args = freqs .* t
    end

    # Sinusoidal embedding: [sin; cos]
    embedding = vcat(sin.(args), cos.(args))  # (time_dim, batch)

    return embedding, state
end

# ============================================================================
# Time MLP: Projects sinusoidal embedding to model dimension
# ============================================================================
struct TimeMLPEmbedding <: LuxLayer
    time_dimension::Int
    embedding_dimension::Int
    SinusoidalEmbed::SinusoidalTimeEmbedding
    MLP1::Lux.Dense
    MLP2::Lux.Dense
end

function TimeMLPEmbedding(time_dimension::Int, embedding_dimension::Int)
    return TimeMLPEmbedding(
        time_dimension,
        embedding_dimension,
        SinusoidalTimeEmbedding(time_dimension),
        Lux.Dense(time_dimension => embedding_dimension, NNlib.gelu),
        Lux.Dense(embedding_dimension => embedding_dimension),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::TimeMLPEmbedding)
    return (
        SinusoidalEmbed = Lux.initialparameters(rng, layer.SinusoidalEmbed),
        MLP1 = Lux.initialparameters(rng, layer.MLP1),
        MLP2 = Lux.initialparameters(rng, layer.MLP2),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::TimeMLPEmbedding)
    return (
        SinusoidalEmbed = Lux.initialstates(rng, layer.SinusoidalEmbed),
        MLP1 = Lux.initialstates(rng, layer.MLP1),
        MLP2 = Lux.initialstates(rng, layer.MLP2),
    )
end

function (layer::TimeMLPEmbedding)(t, params, state)
    sinusoidal, sin_state = layer.SinusoidalEmbed(t, params.SinusoidalEmbed, state.SinusoidalEmbed)
    hidden, mlp1_state = layer.MLP1(sinusoidal, params.MLP1, state.MLP1)
    output, mlp2_state = layer.MLP2(hidden, params.MLP2, state.MLP2)

    new_state = (
        SinusoidalEmbed = sin_state,
        MLP1 = mlp1_state,
        MLP2 = mlp2_state,
    )
    return output, new_state
end

# ============================================================================
# LLaDA Model: Full text diffusion architecture
# ============================================================================
struct LLaDAModel{S,P,T,B,N,O,M,V,E} <: LuxLayer
    vocab_size::Int
    max_sequence_length::Int
    embedding_dimension::Int
    number_of_heads::Int
    number_of_layers::Int
    prime_subtoken_base::Int
    prime_subtoken_length::Int
    prime_mask_subtoken_id::Int
    prime_subtoken_embedding_dimension::Int
    prime_code_table::M
    prime_value_masks::V

    # Embeddings
    SubtokenEmbeddings::S
    PositionEmbedding::P
    TimeEmbedding::T

    # Transformer blocks
    Blocks::B

    # Output
    FinalNorm::N
    OutputHead::O

    # Engram conditional memory
    use_engram::Bool
    engram_layer_map::Vector{Int}   # length = number_of_layers; value = engram index (0 = none)
    EngramModules::E                # Tuple of EngramModule or nothing
end

"""
    LLaDAModel(config::LLaDAConfig)

Create a LLaDA model from a configuration struct.
"""
function LLaDAModel(config::LLaDAConfig)
    state_dimension = config.state_dimension == -1 ? config.embedding_dimension : config.state_dimension

    return LLaDAModel(;
        vocab_size = config.vocab_size,
        max_sequence_length = config.max_sequence_length,
        embedding_dimension = config.embedding_dimension,
        number_of_heads = config.number_of_heads,
        number_of_layers = config.number_of_layers,
        time_dimension = config.time_dimension,
        state_dimension = state_dimension,
        window_size = config.window_size,
        min_frequency = config.min_frequency,
        max_frequency = config.max_frequency,
        default_time_step = config.default_time_step,
        prime_subtoken_length = config.prime_subtoken_length,
        prime_subtoken_base = config.prime_subtoken_base,
        # Engram
        use_engram = config.use_engram,
        engram_layers = config.engram_layers,
        engram_num_heads = config.engram_num_heads,
        engram_ngram_orders = config.engram_ngram_orders,
        engram_table_size = config.engram_table_size,
        engram_head_dim = config.engram_head_dim,
    )
end

"""
    LLaDAModel(config_path::String)

Create a LLaDA model from a TOML configuration file.
"""
function LLaDAModel(config_path::String)
    config = load_config(config_path)
    return LLaDAModel(config)
end

function LLaDAModel(;
    vocab_size::Int,
    max_sequence_length::Int,
    embedding_dimension::Int,
    number_of_heads::Int,
    number_of_layers::Int,
    time_dimension::Int = 128,
    state_dimension::Int = embedding_dimension,
    window_size::Int = 5,
    min_frequency::Float32 = 0.1f0,
    max_frequency::Float32 = 10.0f0,
    default_time_step::Float32 = 0.1f0,
    prime_subtoken_length::Int = 4,
    prime_subtoken_base::Int = 16,
    # Engram
    use_engram::Bool = false,
    engram_layers::Vector{Int} = Int[],
    engram_num_heads::Int = 8,
    engram_ngram_orders::Vector{Int} = [2, 3],
    engram_table_size::Int = 65536,
    engram_head_dim::Int = 0,
)
    resolved_subtoken_base = 0
    resolved_subtoken_length = 0
    resolved_mask_subtoken_id = 0
    resolved_subtoken_embedding_dimension = 0
    code_table = zeros(Int, 0, 0)
    value_masks = Vector{Vector{BitVector}}()
    subtoken_embeddings = nothing

    prime_subtoken_length >= 1 || throw(ArgumentError("prime_subtoken_length must be >= 1"))
    resolved_subtoken_length = prime_subtoken_length
    resolved_subtoken_base = max(prime_subtoken_base, 2)
    if resolved_subtoken_base^resolved_subtoken_length < vocab_size
        resolved_subtoken_base = required_subtoken_base(vocab_size, resolved_subtoken_length)
    end
    embedding_dimension % resolved_subtoken_length == 0 || throw(ArgumentError(
        "embedding_dimension=$(embedding_dimension) must be divisible by prime_subtoken_length=$(resolved_subtoken_length)"
    ))

    resolved_mask_subtoken_id = resolved_subtoken_base + 1
    resolved_subtoken_embedding_dimension = embedding_dimension ÷ resolved_subtoken_length
    code_table = build_vocab_code_table(vocab_size, resolved_subtoken_base, resolved_subtoken_length)
    value_masks = build_subtoken_value_masks(code_table, resolved_subtoken_base)
    subtoken_embeddings = Tuple(
        Lux.Embedding((resolved_subtoken_base + 1) => resolved_subtoken_embedding_dimension)
        for _ in 1:resolved_subtoken_length
    )

    # Build stack of SwammaBlocks
    blocks = Tuple([
        SwammaBlock(
            embedding_dimension,
            max_sequence_length,
            number_of_heads,
            time_dimension;
            state_dimension = state_dimension,
            window_size = window_size,
            min_frequency = min_frequency,
            max_frequency = max_frequency,
            default_time_step = default_time_step,
        )
        for _ in 1:number_of_layers
    ])

    # Build engram modules if enabled
    resolved_engram_layers = if use_engram
        if isempty(engram_layers)
            # Auto-select: early layer + ~42% depth (paper: layers 2 and 15 of 36 ≈ 6%/42%)
            first_layer = min(2, number_of_layers)
            mid = max(first_layer + 1, round(Int, 0.42 * number_of_layers))
            auto = Int[first_layer]
            if mid <= number_of_layers
                push!(auto, mid)
            end
            auto
        else
            filter(l -> 1 <= l <= number_of_layers, engram_layers)
        end
    else
        Int[]
    end

    engram_layer_map = zeros(Int, number_of_layers)
    for (idx, layer_num) in enumerate(resolved_engram_layers)
        engram_layer_map[layer_num] = idx
    end

    engram_modules = if use_engram && !isempty(resolved_engram_layers)
        Tuple([
            EngramModule(
                embedding_dimension;
                num_heads = engram_num_heads,
                ngram_orders = engram_ngram_orders,
                table_size = engram_table_size,
                head_dim = engram_head_dim,
            )
            for _ in 1:length(resolved_engram_layers)
        ])
    else
        nothing
    end

    return LLaDAModel(
        vocab_size,
        max_sequence_length,
        embedding_dimension,
        number_of_heads,
        number_of_layers,
        resolved_subtoken_base,
        resolved_subtoken_length,
        resolved_mask_subtoken_id,
        resolved_subtoken_embedding_dimension,
        code_table,
        value_masks,
        # Embeddings
        subtoken_embeddings,
        Lux.Embedding(max_sequence_length => embedding_dimension),
        TimeMLPEmbedding(time_dimension, time_dimension),
        # Blocks
        blocks,
        # Output
        Lux.LayerNorm((embedding_dimension,)),
        Lux.Dense(embedding_dimension => vocab_size; use_bias = false),
        # Engram
        use_engram,
        engram_layer_map,
        engram_modules,
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::LLaDAModel)
    block_params = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialparameters(rng, block) for block in model.Blocks)
    )
    keys = ntuple(i -> Symbol("Subtoken_$i"), model.prime_subtoken_length)
    vals = Tuple(
        Lux.initialparameters(rng, model.SubtokenEmbeddings[i])
        for i in 1:model.prime_subtoken_length
    )
    subtoken_params = NamedTuple{keys}(vals)

    engram_params = if model.use_engram && model.EngramModules !== nothing
        num_engrams = length(model.EngramModules)
        NamedTuple{ntuple(i -> Symbol("Engram_$i"), num_engrams)}(
            Tuple(Lux.initialparameters(rng, mod) for mod in model.EngramModules)
        )
    else
        NamedTuple()
    end

    return (
        SubtokenEmbeddings = subtoken_params,
        PositionEmbedding = Lux.initialparameters(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialparameters(rng, model.TimeEmbedding),
        Blocks = block_params,
        Engrams = engram_params,
        FinalNorm = Lux.initialparameters(rng, model.FinalNorm),
        OutputHead = Lux.initialparameters(rng, model.OutputHead),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::LLaDAModel)
    block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialstates(rng, block) for block in model.Blocks)
    )
    keys = ntuple(i -> Symbol("Subtoken_$i"), model.prime_subtoken_length)
    vals = Tuple(
        Lux.initialstates(rng, model.SubtokenEmbeddings[i])
        for i in 1:model.prime_subtoken_length
    )
    subtoken_states = NamedTuple{keys}(vals)

    engram_states = if model.use_engram && model.EngramModules !== nothing
        num_engrams = length(model.EngramModules)
        NamedTuple{ntuple(i -> Symbol("Engram_$i"), num_engrams)}(
            Tuple(Lux.initialstates(rng, mod) for mod in model.EngramModules)
        )
    else
        NamedTuple()
    end

    return (
        SubtokenEmbeddings = subtoken_states,
        PositionEmbedding = Lux.initialstates(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialstates(rng, model.TimeEmbedding),
        Blocks = block_states,
        Engrams = engram_states,
        FinalNorm = Lux.initialstates(rng, model.FinalNorm),
        OutputHead = Lux.initialstates(rng, model.OutputHead),
    )
end

function prime_compatibility_mask(model::LLaDAModel, subtoken_state_cpu::Array{<:Integer,3})
    subtoken_length, seq_len, batch_size = size(subtoken_state_cpu)
    subtoken_length == model.prime_subtoken_length || throw(ArgumentError(
        "Expected subtoken_state first dim $(model.prime_subtoken_length), got $(subtoken_length)"
    ))
    vocab_size = model.vocab_size
    compat = trues(vocab_size, seq_len, batch_size)

    for b in 1:batch_size
        for s in 1:seq_len
            allowed = trues(vocab_size)
            for j in 1:model.prime_subtoken_length
                digit = Int(subtoken_state_cpu[j, s, b])
                if digit != model.prime_mask_subtoken_id
                    if 1 <= digit <= model.prime_subtoken_base
                        allowed .&= model.prime_value_masks[j][digit]
                    else
                        # Invalid digits should not zero out support and create NaNs.
                        fill!(allowed, true)
                        break
                    end
                end
            end
            if !any(allowed)
                # Keep at least one finite logit per position for stable softmax.
                fill!(allowed, true)
            end
            compat[:, s, b] .= allowed
        end
    end

    return compat
end

function apply_prime_carryover_filter(model::LLaDAModel, logits, subtoken_state_batched)
    compat_cpu = ChainRulesCore.ignore_derivatives() do
        subtoken_state_cpu = Array(subtoken_state_batched)
        prime_compatibility_mask(model, subtoken_state_cpu)
    end
    compat = to_device_like(logits, compat_cpu)
    neg_large = convert(eltype(logits), -1.0f9)
    return ifelse.(compat, logits, neg_large)
end

function (model::LLaDAModel)(inputs::NamedTuple, params, state)
    has_subtoken_state = haskey(inputs, :subtoken_state)
    mask_ratio = inputs.mask_ratio

    has_subtoken_state || throw(ArgumentError("LLaDA PRIME expects `subtoken_state` in inputs."))

    subtoken_state = inputs.subtoken_state
    subtoken_state_batched = if ndims(subtoken_state) == 2
        size(subtoken_state, 1) == model.prime_subtoken_length || throw(ArgumentError(
            "Prime input must have first dim = $(model.prime_subtoken_length), got $(size(subtoken_state, 1))."
        ))
        reshape(subtoken_state, model.prime_subtoken_length, size(subtoken_state, 2), 1)
    elseif ndims(subtoken_state) == 3
        size(subtoken_state, 1) == model.prime_subtoken_length || throw(ArgumentError(
            "Prime input must have first dim = $(model.prime_subtoken_length), got $(size(subtoken_state, 1))."
        ))
        subtoken_state
    else
        throw(ArgumentError("Prime input expects 2D or 3D subtoken_state, got ndims=$(ndims(subtoken_state))."))
    end
    was_unbatched = ndims(subtoken_state) == 2

    seq_len = size(subtoken_state_batched, 2)
    batch_size = size(subtoken_state_batched, 3)
    source_for_device = subtoken_state_batched

    subtoken_pairs = ntuple(j -> begin
        key = Symbol("Subtoken_$j")
        subtoken_flat = vec(subtoken_state_batched[j, :, :])
        emb_flat, emb_state = model.SubtokenEmbeddings[j](
            subtoken_flat, params.SubtokenEmbeddings[key], state.SubtokenEmbeddings[key]
        )
        emb = reshape(emb_flat, model.prime_subtoken_embedding_dimension, seq_len, batch_size)
        (emb, emb_state)
    end, model.prime_subtoken_length)

    sub_embs = map(p -> p[1], subtoken_pairs)
    token_emb = cat(sub_embs...; dims = 1)
    sub_states = map(p -> p[2], subtoken_pairs)
    subtoken_states_out = NamedTuple{ntuple(i -> Symbol("Subtoken_$i"), model.prime_subtoken_length)}(
        sub_states
    )

    # =========================================================================
    # 2. Position Embedding (ensure same device as input)
    # =========================================================================
    position_indices_cpu = collect(1:seq_len)
    position_indices = to_device_like(source_for_device, position_indices_cpu)
    pos_emb_raw, pos_state = model.PositionEmbedding(position_indices, params.PositionEmbedding, state.PositionEmbedding)
    pos_emb = reshape(pos_emb_raw, model.embedding_dimension, seq_len, 1)

    # =========================================================================
    # 3. Combine Embeddings
    # =========================================================================
    hidden = token_emb .+ pos_emb

    # =========================================================================
    # 4. Time Embedding (mask ratio conditioning)
    # =========================================================================
    # Ensure mask_ratio is (1, batch) shape and on same device as input
    t_input_cpu = if ndims(mask_ratio) == 0
        fill(Float32(mask_ratio), 1, batch_size)
    elseif ndims(mask_ratio) == 1
        reshape(Float32.(mask_ratio), 1, :)
    else
        Float32.(mask_ratio)
    end
    t_input = to_device_like(source_for_device, t_input_cpu)

    time_emb, time_state = model.TimeEmbedding(t_input, params.TimeEmbedding, state.TimeEmbedding)

    # =========================================================================
    # 5. Reconstruct token IDs for engram (if enabled)
    # =========================================================================
    token_ids_for_engram = if model.use_engram && model.EngramModules !== nothing
        ChainRulesCore.ignore_derivatives() do
            subtokens_to_token_ids(Array(subtoken_state_batched), model.prime_subtoken_base)
        end
    else
        nothing
    end

    # =========================================================================
    # 6. Process through SwammaBlocks (Zygote-compatible using foldl)
    # =========================================================================
    # Use foldl to avoid mutation (push!) which Zygote doesn't support
    (hidden, block_states) = foldl(
        enumerate(model.Blocks);
        init = (hidden, ())
    ) do (h, states), (i, block)
        # Apply engram before block at selected layers
        h = if model.use_engram && model.EngramModules !== nothing && model.engram_layer_map[i] > 0
            engram_idx = model.engram_layer_map[i]
            engram_key = Symbol("Engram_$engram_idx")
            engram_mod = model.EngramModules[engram_idx]
            h_new, _ = engram_mod(
                (token_ids_for_engram, h),
                params.Engrams[engram_key],
                state.Engrams[engram_key],
            )
            h_new
        else
            h
        end

        block_key = Symbol("Block_$i")
        block_params = params.Blocks[block_key]
        block_state = state.Blocks[block_key]

        new_h, new_block_state = block((h, time_emb), block_params, block_state)
        (new_h, (states..., new_block_state))
    end

    # =========================================================================
    # 7. Final Normalization
    # =========================================================================
    # LayerNorm expects 2D input, flatten 3D → 2D → normalize → reshape back
    hidden_shape = size(hidden)
    hidden_flat = reshape(hidden, model.embedding_dimension, :)
    normalized_flat, norm_state = model.FinalNorm(hidden_flat, params.FinalNorm, state.FinalNorm)
    normalized = reshape(normalized_flat, hidden_shape)

    # =========================================================================
    # 8. Output Head → Logits
    # =========================================================================
    # Dense layer can handle 3D input directly
    logits, out_state = model.OutputHead(normalized, params.OutputHead, state.OutputHead)
    logits = apply_prime_carryover_filter(model, logits, subtoken_state_batched)

    final_logits = was_unbatched ? dropdims(logits, dims = 3) : logits

    # =========================================================================
    # 9. Update State
    # =========================================================================
    # block_states is already a tuple from foldl
    new_block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        block_states
    )

    # Engram states are read-only (hash multipliers), pass through unchanged
    engram_states_out = if model.use_engram && model.EngramModules !== nothing
        num_engrams = length(model.EngramModules)
        NamedTuple{ntuple(i -> Symbol("Engram_$i"), num_engrams)}(
            ntuple(i -> state.Engrams[Symbol("Engram_$i")], num_engrams)
        )
    else
        NamedTuple()
    end

    new_state = (
        SubtokenEmbeddings = subtoken_states_out,
        PositionEmbedding = pos_state,
        TimeEmbedding = time_state,
        Blocks = new_block_states,
        Engrams = engram_states_out,
        FinalNorm = norm_state,
        OutputHead = out_state,
    )

    return final_logits, new_state
end

# ============================================================================
# Diffusion Utilities
# ============================================================================

"""
    apply_subtoken_mask(subtoken_state, mask_ratio, mask_subtoken_id; rng)

Apply random masking at the sub-token level.
Returns `(masked_subtokens, subtoken_mask, token_mask)` where:
- `subtoken_mask` marks masked sub-token cells.
- `token_mask` marks positions with at least one masked sub-token.
"""
function apply_subtoken_mask(
    subtoken_state::AbstractArray,
    mask_ratio::Real,
    mask_subtoken_id::Int;
    rng = Random.default_rng()
)
    ndims(subtoken_state) in (2, 3) || throw(ArgumentError(
        "apply_subtoken_mask expects 2D or 3D subtoken state, got ndims=$(ndims(subtoken_state))."
    ))

    subtoken_mask = rand(rng, Float32, size(subtoken_state)) .< mask_ratio
    masked_subtokens = ifelse.(subtoken_mask, mask_subtoken_id, subtoken_state)

    token_mask = if ndims(subtoken_state) == 2
        vec(dropdims(any(subtoken_mask, dims = 1), dims = 1))
    else
        dropdims(any(subtoken_mask, dims = 1), dims = 1)
    end

    return masked_subtokens, subtoken_mask, token_mask
end

"""
    sample_mask_ratio(rng; schedule = :uniform)

Sample a mask ratio t ∈ [0, 1] for training.
"""
function sample_mask_ratio(rng::Random.AbstractRNG; schedule::Symbol = :uniform)
    if schedule == :uniform
        return rand(rng, Float32)
    elseif schedule == :cosine
        # Cosine schedule: more samples near t=0 and t=1
        u = rand(rng, Float32)
        return (1 - cos(π * u)) / 2
    else
        return rand(rng, Float32)
    end
end

"""
    unmask_subtoken_step(logits, current_subtokens, num_to_unmask, code_table, mask_subtoken_id)

Reveal masked sub-tokens using predicted token codewords and token confidence.
"""
function unmask_subtoken_step(
    logits::AbstractArray,
    current_subtokens::AbstractArray,
    num_to_unmask::Int,
    code_table::AbstractMatrix{<:Integer},
    mask_subtoken_id::Int,
)
    num_to_unmask <= 0 && return current_subtokens

    logits_batched = ndims(logits) == 2 ? reshape(logits, size(logits, 1), size(logits, 2), 1) : logits
    was_unbatched = ndims(current_subtokens) == 2
    state_batched = was_unbatched ?
        reshape(current_subtokens, size(current_subtokens, 1), size(current_subtokens, 2), 1) :
        current_subtokens

    predictions_raw = argmax(logits_batched, dims = 1)
    predictions = map(ci -> ci[1], predictions_raw)
    predictions = dropdims(predictions, dims = 1)  # (seq, batch)

    probs = NNlib.softmax(logits_batched, dims = 1)
    confidence = maximum(probs, dims = 1)
    confidence = dropdims(confidence, dims = 1)    # (seq, batch)

    subtoken_length, seq_len, batch_size = size(state_batched)
    new_state = copy(state_batched)
    quota_per_batch = max(1, ceil(Int, num_to_unmask / batch_size))

    for b in 1:batch_size
        remaining = quota_per_batch
        sorted_indices = sortperm(vec(confidence[:, b]), rev = true)

        for idx in sorted_indices
            remaining <= 0 && break
            if !any(@view(new_state[:, idx, b]) .== mask_subtoken_id)
                continue
            end

            pred_id = clamp(Int(predictions[idx, b]), 1, size(code_table, 2))
            for j in 1:subtoken_length
                if new_state[j, idx, b] == mask_subtoken_id
                    new_state[j, idx, b] = code_table[j, pred_id]
                    remaining -= 1
                    break
                end
            end
        end
    end

    return was_unbatched ? dropdims(new_state, dims = 3) : new_state
end

"""
    generate(model, params, state, seq_len;
             num_steps, batch_size, rng)

Generate text by iterative denoising from fully masked sequence.
"""
function generate(
    model::LLaDAModel,
    params,
    state,
    seq_len::Int;
    num_steps::Int = 10,
    batch_size::Int = 1,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    current_subtokens = fill(
        model.prime_mask_subtoken_id, model.prime_subtoken_length, seq_len, batch_size
    )
    subtokens_per_step = max(1, ceil(Int, (seq_len * model.prime_subtoken_length) / num_steps))

    for step in 1:num_steps
        mask_ratio = Float32(1.0 - (step - 1) / num_steps)
        inputs = (subtoken_state = current_subtokens, mask_ratio = mask_ratio)
        logits, state = model(inputs, params, state)

        remaining_masked = count(==(model.prime_mask_subtoken_id), current_subtokens)
        num_to_unmask = min(subtokens_per_step, remaining_masked)
        if num_to_unmask > 0
            current_subtokens = unmask_subtoken_step(
                logits, current_subtokens, num_to_unmask,
                model.prime_code_table, model.prime_mask_subtoken_id
            )
        end

        if count(==(model.prime_mask_subtoken_id), current_subtokens) == 0
            break
        end
    end

    inputs = (subtoken_state = current_subtokens, mask_ratio = 0.0f0)
    logits, state = model(inputs, params, state)
    predictions = map(ci -> ci[1], argmax(logits, dims = 1))
    predictions = dropdims(predictions, dims = 1)
    return batch_size == 1 ? vec(predictions) : predictions
end

# ============================================================================
# Exports
# ============================================================================

# Model
export LLaDAModel, TimeMLPEmbedding, SinusoidalTimeEmbedding

# Configuration
export LLaDAConfig, load_config, save_config, config_from_dict
export default_config, small_config, base_config, large_config, production_config

# Diffusion utilities
export apply_subtoken_mask, sample_mask_ratio, unmask_subtoken_step, generate
export token_ids_to_subtokens

end # module
