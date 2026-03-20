module NativeTeacherLM

using JSON3
using Lux
using LuxCore
using NNlib
using Random

# PyCall is loaded lazily — only when HF weight loading functions are called
function _pyimport(name::String)
    PyCall = Base.require(Base.PkgId(Base.UUID("438e738f-606a-5dbb-bf0a-cddfbfd45ab0"), "PyCall"))
    return PyCall.pyimport(name)
end

function _PyDict()
    PyCall = Base.require(Base.PkgId(Base.UUID("438e738f-606a-5dbb-bf0a-cddfbfd45ab0"), "PyCall"))
    return PyCall.PyDict()
end

function _PyArray(x)
    PyCall = Base.require(Base.PkgId(Base.UUID("438e738f-606a-5dbb-bf0a-cddfbfd45ab0"), "PyCall"))
    return PyCall.PyArray(x)
end

function _PyObject_check(x)
    PyCall = Base.require(Base.PkgId(Base.UUID("438e738f-606a-5dbb-bf0a-cddfbfd45ab0"), "PyCall"))
    return x isa PyCall.PyObject
end

using ..Swamma: RMSNorm, LuxLayer

export NativeTeacherConfig
export RotaryEmbedding, CausalSelfAttention, GatedMLP, DecoderBlock, NativeCausalLM
export build_causal_attention_bias, next_token_logits, greedy_generate
export AttentionKVCache, NativeDecoderCache, init_decoder_cache, cache_sequence_length
export forward_with_cache, next_token_logits_cached, greedy_generate_cached
export resolve_hf_model_dir, granite_config_from_hf, load_granite_weights, load_granite_model

Base.@kwdef struct NativeTeacherConfig
    vocab_size::Int = 49160
    max_sequence_length::Int = 1024
    embedding_dimension::Int = 512
    number_of_layers::Int = 8
    number_of_heads::Int = 8
    number_of_kv_heads::Int = number_of_heads
    mlp_hidden_dimension::Int = 1536
    rope_theta::Float32 = 10_000f0
    rms_norm_eps::Float32 = 1f-6
    embedding_multiplier::Float32 = 1f0
    logits_scaling::Float32 = 1f0
    tie_word_embeddings::Bool = false
    dropout_rate::Float32 = 0f0
    use_bias::Bool = false
    attention_multiplier::Float32 = NaN32
    residual_multiplier::Float32 = 1f0
end

function validate_config(config::NativeTeacherConfig)
    config.vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    config.max_sequence_length > 0 || throw(ArgumentError("max_sequence_length must be positive"))
    config.embedding_dimension > 0 || throw(ArgumentError("embedding_dimension must be positive"))
    config.number_of_layers > 0 || throw(ArgumentError("number_of_layers must be positive"))
    config.number_of_heads > 0 || throw(ArgumentError("number_of_heads must be positive"))
    config.number_of_kv_heads > 0 || throw(ArgumentError("number_of_kv_heads must be positive"))
    config.embedding_dimension % config.number_of_heads == 0 ||
        throw(ArgumentError("embedding_dimension must be divisible by number_of_heads"))
    config.number_of_heads % config.number_of_kv_heads == 0 ||
        throw(ArgumentError("number_of_heads must be divisible by number_of_kv_heads"))
    head_dim = config.embedding_dimension ÷ config.number_of_heads
    iseven(head_dim) || throw(ArgumentError("head dimension must be even for RoPE"))
    config.mlp_hidden_dimension > 0 || throw(ArgumentError("mlp_hidden_dimension must be positive"))
    config.rms_norm_eps > 0 || throw(ArgumentError("rms_norm_eps must be positive"))
    config.embedding_multiplier > 0 || throw(ArgumentError("embedding_multiplier must be positive"))
    config.logits_scaling > 0 || throw(ArgumentError("logits_scaling must be positive"))
    (isnan(config.attention_multiplier) || config.attention_multiplier > 0) ||
        throw(ArgumentError("attention_multiplier must be positive when provided"))
    config.residual_multiplier > 0 || throw(ArgumentError("residual_multiplier must be positive"))
    return config
end

struct RotaryEmbedding <: LuxLayer
    head_dimension::Int
    theta::Float32
end

RotaryEmbedding(head_dimension::Int; theta::Float32 = 10_000f0) = begin
    iseven(head_dimension) || throw(ArgumentError("Rotary head dimension must be even"))
    RotaryEmbedding(head_dimension, theta)
end

Lux.initialparameters(::Random.AbstractRNG, ::RotaryEmbedding) = (;)
Lux.initialstates(::Random.AbstractRNG, ::RotaryEmbedding) = (;)

_device_like(x, ref) = x
function _device_like(x::AbstractArray, ref::AbstractArray)
    target = similar(ref, eltype(x), size(x)...)
    copyto!(target, x)
    return target
end

function rope_frequencies(layer::RotaryEmbedding, sequence_length::Int; position_offset::Int = 0)
    pair_count = layer.head_dimension ÷ 2
    positions = reshape(Float32.(position_offset:(position_offset + sequence_length - 1)), 1, sequence_length)
    exponents = Float32.(collect(0:(pair_count - 1))) ./ Float32(pair_count)
    inverse_frequency = layer.theta .^ (-exponents)
    return inverse_frequency .* positions
end

function apply_rotary_embedding(layer::RotaryEmbedding, x; position_offset::Int = 0)
    head_dimension, head_count, sequence_length, batch_size = size(x)
    pair_count = head_dimension ÷ 2
    angles = rope_frequencies(layer, sequence_length; position_offset = position_offset)
    # Broadcast-friendly: (pair_count, 1, seq, 1)
    cos_a = reshape(cos.(angles), pair_count, 1, sequence_length, 1)
    sin_a = reshape(sin.(angles), pair_count, 1, sequence_length, 1)
    cos_a = _device_like(cos_a, x)
    sin_a = _device_like(sin_a, x)

    x_first  = x[1:pair_count, :, :, :]
    x_second = x[(pair_count + 1):head_dimension, :, :, :]

    rotated_first  = x_first .* cos_a .- x_second .* sin_a
    rotated_second = x_second .* cos_a .+ x_first .* sin_a

    return vcat(rotated_first, rotated_second)
end

function (layer::RotaryEmbedding)(x, ps, st)
    return apply_rotary_embedding(layer, x), st
end

struct AttentionKVCache
    keys::Union{Nothing,AbstractArray}
    values::Union{Nothing,AbstractArray}
end

AttentionKVCache() = AttentionKVCache(nothing, nothing)

struct NativeDecoderCache
    layers::Vector{AttentionKVCache}
end

function init_decoder_cache(model, batch_size::Int)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    return NativeDecoderCache([AttentionKVCache() for _ in 1:model.config.number_of_layers])
end

function cache_sequence_length(cache::AttentionKVCache)
    cache.keys === nothing && return 0
    return size(cache.keys, 3)
end

function cache_sequence_length(cache::NativeDecoderCache)
    isempty(cache.layers) && return 0
    return cache_sequence_length(cache.layers[1])
end

struct CausalSelfAttention{QN,KN,VN,ON,RO,DO} <: LuxLayer
    embedding_dimension::Int
    number_of_heads::Int
    number_of_kv_heads::Int
    head_dimension::Int
    attention_multiplier::Float32
    QueryProjection::QN
    KeyProjection::KN
    ValueProjection::VN
    OutputProjection::ON
    Rope::RO
    Dropout::DO
end

function CausalSelfAttention(
    embedding_dimension::Int,
    number_of_heads::Int;
    number_of_kv_heads::Int = number_of_heads,
    rope_theta::Float32 = 10_000f0,
    dropout_rate::Float32 = 0f0,
    use_bias::Bool = false,
    attention_multiplier::Union{Nothing,Float32} = nothing,
)
    embedding_dimension % number_of_heads == 0 ||
        throw(ArgumentError("embedding_dimension must be divisible by number_of_heads"))
    number_of_heads % number_of_kv_heads == 0 ||
        throw(ArgumentError("number_of_heads must be divisible by number_of_kv_heads"))
    head_dimension = embedding_dimension ÷ number_of_heads
    iseven(head_dimension) || throw(ArgumentError("head dimension must be even for RoPE"))

    kv_dimension = number_of_kv_heads * head_dimension
    attention_multiplier = something(attention_multiplier, inv(sqrt(Float32(head_dimension))))
    return CausalSelfAttention(
        embedding_dimension,
        number_of_heads,
        number_of_kv_heads,
        head_dimension,
        attention_multiplier,
        Lux.Dense(embedding_dimension => embedding_dimension; use_bias = use_bias),
        Lux.Dense(embedding_dimension => kv_dimension; use_bias = use_bias),
        Lux.Dense(embedding_dimension => kv_dimension; use_bias = use_bias),
        Lux.Dense(embedding_dimension => embedding_dimension; use_bias = use_bias),
        RotaryEmbedding(head_dimension; theta = rope_theta),
        Lux.Dropout(dropout_rate),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::CausalSelfAttention)
    return (
        QueryProjection = Lux.initialparameters(rng, layer.QueryProjection),
        KeyProjection = Lux.initialparameters(rng, layer.KeyProjection),
        ValueProjection = Lux.initialparameters(rng, layer.ValueProjection),
        OutputProjection = Lux.initialparameters(rng, layer.OutputProjection),
        Rope = Lux.initialparameters(rng, layer.Rope),
        Dropout = Lux.initialparameters(rng, layer.Dropout),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::CausalSelfAttention)
    return (
        QueryProjection = Lux.initialstates(rng, layer.QueryProjection),
        KeyProjection = Lux.initialstates(rng, layer.KeyProjection),
        ValueProjection = Lux.initialstates(rng, layer.ValueProjection),
        OutputProjection = Lux.initialstates(rng, layer.OutputProjection),
        Rope = Lux.initialstates(rng, layer.Rope),
        Dropout = Lux.initialstates(rng, layer.Dropout),
    )
end

function build_causal_attention_bias(query_length::Int, key_length::Int = query_length; query_offset::Int = 0)
    # Vectorized: query position q can attend to key position k iff k <= query_offset + q
    q_pos = reshape(Float32.(1:query_length), query_length, 1) .+ Float32(query_offset)
    k_pos = reshape(Float32.(1:key_length), 1, key_length)
    return ifelse.(k_pos .<= q_pos, 0f0, -1f9)
end

function reshape_heads(x, head_dimension::Int, head_count::Int)
    sequence_length = size(x, 2)
    batch_size = size(x, 3)
    return reshape(x, head_dimension, head_count, sequence_length, batch_size)
end

function expand_kv_heads(x, expand_factor::Int)
    expand_factor == 1 && return x
    head_dimension, kv_head_count, sequence_length, batch_size = size(x)
    # Insert a repeat dimension: (head_dim, kv_heads, 1, seq, batch) → repeat along dim 3
    x5 = reshape(x, head_dimension, kv_head_count, 1, sequence_length, batch_size)
    repeated = repeat(x5, 1, 1, expand_factor, 1, 1)  # (hd, kv, factor, seq, batch)
    return reshape(repeated, head_dimension, kv_head_count * expand_factor, sequence_length, batch_size)
end

function append_attention_cache(cache::AttentionKVCache, keys, values)
    if cache.keys === nothing
        return AttentionKVCache(keys, values)
    end
    return AttentionKVCache(
        cat(cache.keys, keys; dims = 3),
        cat(cache.values, values; dims = 3),
    )
end

function causal_attention_forward(
    layer::CausalSelfAttention,
    x,
    ps,
    st;
    cache::Union{Nothing,AttentionKVCache} = nothing,
    position_offset::Int = 0,
)
    sequence_length = size(x, 2)
    batch_size = size(x, 3)

    query_projection, query_state = layer.QueryProjection(x, ps.QueryProjection, st.QueryProjection)
    key_projection, key_state = layer.KeyProjection(x, ps.KeyProjection, st.KeyProjection)
    value_projection, value_state = layer.ValueProjection(x, ps.ValueProjection, st.ValueProjection)

    queries = reshape_heads(query_projection, layer.head_dimension, layer.number_of_heads)
    keys = reshape_heads(key_projection, layer.head_dimension, layer.number_of_kv_heads)
    values = reshape_heads(value_projection, layer.head_dimension, layer.number_of_kv_heads)

    queries = apply_rotary_embedding(layer.Rope, queries; position_offset = position_offset)
    keys = apply_rotary_embedding(layer.Rope, keys; position_offset = position_offset)

    expand_factor = layer.number_of_heads ÷ layer.number_of_kv_heads
    keys = expand_kv_heads(keys, expand_factor)
    values = expand_kv_heads(values, expand_factor)

    updated_cache = cache === nothing ? nothing : append_attention_cache(cache, keys, values)
    full_keys = updated_cache === nothing ? keys : updated_cache.keys
    full_values = updated_cache === nothing ? values : updated_cache.values

    key_length = size(full_keys, 3)
    bias = build_causal_attention_bias(sequence_length, key_length; query_offset = position_offset)

    # Input shape: (head_dim, num_heads, seq, batch)
    # Permute to (head_dim, seq, num_heads, batch) then flatten heads*batch
    nh = layer.number_of_heads
    q_perm = permutedims(queries, (1, 3, 2, 4))      # (head_dim, seq_q, nh, batch)
    k_perm = permutedims(full_keys, (1, 3, 2, 4))    # (head_dim, key_len, nh, batch)
    v_perm = permutedims(full_values, (1, 3, 2, 4))  # (head_dim, key_len, nh, batch)

    q_flat = reshape(q_perm, layer.head_dimension, sequence_length, nh * batch_size)
    k_flat = reshape(k_perm, layer.head_dimension, key_length, nh * batch_size)
    v_flat = reshape(v_perm, layer.head_dimension, key_length, nh * batch_size)

    # scores = Q^T K * scale → (seq_q, key_len, nh*batch)
    scores = NNlib.batched_mul(
        permutedims(q_flat, (2, 1, 3)),  # (seq_q, head_dim, nh*batch)
        k_flat,                           # (head_dim, key_len, nh*batch)
    ) .* layer.attention_multiplier
    bias = _device_like(bias, scores)

    # Add causal bias and softmax
    weights = NNlib.softmax(scores .+ bias; dims = 2)

    # context = weights * V^T → (seq_q, head_dim, nh*batch)
    context = NNlib.batched_mul(
        weights,                           # (seq_q, key_len, nh*batch)
        permutedims(v_flat, (2, 1, 3)),    # (key_len, head_dim, nh*batch)
    )

    # Reshape back: (seq_q, head_dim, nh*batch) → (head_dim, seq, nh, batch) → (head_dim, nh, seq, batch)
    context_4d = reshape(permutedims(context, (2, 1, 3)), layer.head_dimension, sequence_length, nh, batch_size)
    attended = permutedims(context_4d, (1, 3, 2, 4))  # (head_dim, nh, seq, batch)
    merged = reshape(attended, layer.embedding_dimension, sequence_length, batch_size)
    projected, output_state = layer.OutputProjection(merged, ps.OutputProjection, st.OutputProjection)
    projected, dropout_state = layer.Dropout(projected, ps.Dropout, st.Dropout)

    return projected, (
        QueryProjection = query_state,
        KeyProjection = key_state,
        ValueProjection = value_state,
        OutputProjection = output_state,
        Rope = st.Rope,
        Dropout = dropout_state,
    ), updated_cache
end

function (layer::CausalSelfAttention)(x, ps, st)
    projected, attention_state, _ = causal_attention_forward(layer, x, ps, st)
    return projected, attention_state
end

struct GatedMLP{GP,UP,DP,DO} <: LuxLayer
    embedding_dimension::Int
    hidden_dimension::Int
    GateProjection::GP
    UpProjection::UP
    DownProjection::DP
    Dropout::DO
end

function GatedMLP(
    embedding_dimension::Int,
    hidden_dimension::Int;
    dropout_rate::Float32 = 0f0,
    use_bias::Bool = false,
)
    return GatedMLP(
        embedding_dimension,
        hidden_dimension,
        Lux.Dense(embedding_dimension => hidden_dimension; use_bias = use_bias),
        Lux.Dense(embedding_dimension => hidden_dimension; use_bias = use_bias),
        Lux.Dense(hidden_dimension => embedding_dimension; use_bias = use_bias),
        Lux.Dropout(dropout_rate),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::GatedMLP)
    return (
        GateProjection = Lux.initialparameters(rng, layer.GateProjection),
        UpProjection = Lux.initialparameters(rng, layer.UpProjection),
        DownProjection = Lux.initialparameters(rng, layer.DownProjection),
        Dropout = Lux.initialparameters(rng, layer.Dropout),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::GatedMLP)
    return (
        GateProjection = Lux.initialstates(rng, layer.GateProjection),
        UpProjection = Lux.initialstates(rng, layer.UpProjection),
        DownProjection = Lux.initialstates(rng, layer.DownProjection),
        Dropout = Lux.initialstates(rng, layer.Dropout),
    )
end

function (layer::GatedMLP)(x, ps, st)
    gate, gate_state = layer.GateProjection(x, ps.GateProjection, st.GateProjection)
    up, up_state = layer.UpProjection(x, ps.UpProjection, st.UpProjection)
    hidden = NNlib.swish.(gate) .* up
    output, down_state = layer.DownProjection(hidden, ps.DownProjection, st.DownProjection)
    output, dropout_state = layer.Dropout(output, ps.Dropout, st.Dropout)

    return output, (
        GateProjection = gate_state,
        UpProjection = up_state,
        DownProjection = down_state,
        Dropout = dropout_state,
    )
end

struct DecoderBlock{AN,FN,AT,FF} <: LuxLayer
    residual_multiplier::Float32
    AttentionNorm::AN
    FeedForwardNorm::FN
    Attention::AT
    FeedForward::FF
end

function DecoderBlock(config::NativeTeacherConfig)
    validate_config(config)
    attention_multiplier = isnan(config.attention_multiplier) ?
        inv(sqrt(Float32(config.embedding_dimension ÷ config.number_of_heads))) :
        config.attention_multiplier
    return DecoderBlock(
        config.residual_multiplier,
        RMSNorm(config.embedding_dimension; eps = config.rms_norm_eps),
        RMSNorm(config.embedding_dimension; eps = config.rms_norm_eps),
        CausalSelfAttention(
            config.embedding_dimension,
            config.number_of_heads;
            number_of_kv_heads = config.number_of_kv_heads,
            rope_theta = config.rope_theta,
            dropout_rate = config.dropout_rate,
            use_bias = config.use_bias,
            attention_multiplier = attention_multiplier,
        ),
        GatedMLP(
            config.embedding_dimension,
            config.mlp_hidden_dimension;
            dropout_rate = config.dropout_rate,
            use_bias = config.use_bias,
        ),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::DecoderBlock)
    return (
        AttentionNorm = Lux.initialparameters(rng, layer.AttentionNorm),
        FeedForwardNorm = Lux.initialparameters(rng, layer.FeedForwardNorm),
        Attention = Lux.initialparameters(rng, layer.Attention),
        FeedForward = Lux.initialparameters(rng, layer.FeedForward),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::DecoderBlock)
    return (
        AttentionNorm = Lux.initialstates(rng, layer.AttentionNorm),
        FeedForwardNorm = Lux.initialstates(rng, layer.FeedForwardNorm),
        Attention = Lux.initialstates(rng, layer.Attention),
        FeedForward = Lux.initialstates(rng, layer.FeedForward),
    )
end

function (layer::DecoderBlock)(x, ps, st)
    normalized_attention, attention_norm_state =
        layer.AttentionNorm(x, ps.AttentionNorm, st.AttentionNorm)
    attention_output, attention_state =
        layer.Attention(normalized_attention, ps.Attention, st.Attention)
    residual = x .+ attention_output .* layer.residual_multiplier

    normalized_ffn, feed_forward_norm_state =
        layer.FeedForwardNorm(residual, ps.FeedForwardNorm, st.FeedForwardNorm)
    feed_forward_output, feed_forward_state =
        layer.FeedForward(normalized_ffn, ps.FeedForward, st.FeedForward)
    output = residual .+ feed_forward_output .* layer.residual_multiplier

    return output, (
        AttentionNorm = attention_norm_state,
        FeedForwardNorm = feed_forward_norm_state,
        Attention = attention_state,
        FeedForward = feed_forward_state,
    )
end

function forward_decoder_block_with_cache(
    layer::DecoderBlock,
    x,
    ps,
    st;
    cache::Union{Nothing,AttentionKVCache} = nothing,
    position_offset::Int = 0,
)
    normalized_attention, attention_norm_state =
        layer.AttentionNorm(x, ps.AttentionNorm, st.AttentionNorm)
    attention_output, attention_state, updated_cache =
        causal_attention_forward(
            layer.Attention,
            normalized_attention,
            ps.Attention,
            st.Attention;
            cache = cache,
            position_offset = position_offset,
        )
    residual = x .+ attention_output .* layer.residual_multiplier

    normalized_ffn, feed_forward_norm_state =
        layer.FeedForwardNorm(residual, ps.FeedForwardNorm, st.FeedForwardNorm)
    feed_forward_output, feed_forward_state =
        layer.FeedForward(normalized_ffn, ps.FeedForward, st.FeedForward)
    output = residual .+ feed_forward_output .* layer.residual_multiplier

    return output, (
        AttentionNorm = attention_norm_state,
        FeedForwardNorm = feed_forward_norm_state,
        Attention = attention_state,
        FeedForward = feed_forward_state,
    ), updated_cache
end

struct NativeCausalLM{TE,BL,FN,OH} <: LuxLayer
    config::NativeTeacherConfig
    TokenEmbedding::TE
    Blocks::BL
    FinalNorm::FN
    OutputHead::OH
end

function NativeCausalLM(config::NativeTeacherConfig)
    validate_config(config)
    blocks = Tuple(DecoderBlock(config) for _ in 1:config.number_of_layers)
    return NativeCausalLM(
        config,
        Lux.Embedding(config.vocab_size => config.embedding_dimension),
        blocks,
        RMSNorm(config.embedding_dimension; eps = config.rms_norm_eps),
        Lux.Dense(config.embedding_dimension => config.vocab_size; use_bias = false),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::NativeCausalLM)
    return (
        TokenEmbedding = Lux.initialparameters(rng, model.TokenEmbedding),
        Blocks = Tuple(Lux.initialparameters(rng, block) for block in model.Blocks),
        FinalNorm = Lux.initialparameters(rng, model.FinalNorm),
        OutputHead = Lux.initialparameters(rng, model.OutputHead),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::NativeCausalLM)
    return (
        TokenEmbedding = Lux.initialstates(rng, model.TokenEmbedding),
        Blocks = Tuple(Lux.initialstates(rng, block) for block in model.Blocks),
        FinalNorm = Lux.initialstates(rng, model.FinalNorm),
        OutputHead = Lux.initialstates(rng, model.OutputHead),
    )
end

function ensure_token_batch(token_ids::AbstractVector{<:Integer})
    return reshape(collect(Int, token_ids), :, 1)
end

function ensure_token_batch(token_ids::AbstractMatrix{<:Integer})
    return Matrix{Int}(token_ids)
end

function (model::NativeCausalLM)(token_ids::AbstractArray{<:Integer}, ps, st)
    tokens = ensure_token_batch(token_ids)
    size(tokens, 1) <= model.config.max_sequence_length ||
        throw(ArgumentError("token sequence exceeds max_sequence_length"))

    hidden, embedding_state = model.TokenEmbedding(tokens, ps.TokenEmbedding, st.TokenEmbedding)
    hidden .*= model.config.embedding_multiplier
    block_states = Vector{Any}(undef, length(model.Blocks))

    for (block_index, block) in enumerate(model.Blocks)
        hidden, block_state = block(hidden, ps.Blocks[block_index], st.Blocks[block_index])
        block_states[block_index] = block_state
    end

    hidden, final_norm_state = model.FinalNorm(hidden, ps.FinalNorm, st.FinalNorm)
    logits, output_state = model.OutputHead(hidden, ps.OutputHead, st.OutputHead)
    logits = logits ./ model.config.logits_scaling

    return logits, (
        TokenEmbedding = embedding_state,
        Blocks = Tuple(block_states),
        FinalNorm = final_norm_state,
        OutputHead = output_state,
    )
end

function forward_with_cache(
    model::NativeCausalLM,
    token_ids::AbstractArray{<:Integer},
    ps,
    st,
    cache::NativeDecoderCache,
)
    tokens = ensure_token_batch(token_ids)
    size(tokens, 1) <= model.config.max_sequence_length ||
        throw(ArgumentError("token sequence exceeds max_sequence_length"))

    position_offset = cache_sequence_length(cache)
    hidden, embedding_state = model.TokenEmbedding(tokens, ps.TokenEmbedding, st.TokenEmbedding)
    hidden .*= model.config.embedding_multiplier
    block_states = Vector{Any}(undef, length(model.Blocks))
    updated_layer_caches = Vector{AttentionKVCache}(undef, length(model.Blocks))

    for (block_index, block) in enumerate(model.Blocks)
        hidden, block_state, updated_cache =
            forward_decoder_block_with_cache(
                block,
                hidden,
                ps.Blocks[block_index],
                st.Blocks[block_index];
                cache = cache.layers[block_index],
                position_offset = position_offset,
            )
        block_states[block_index] = block_state
        updated_layer_caches[block_index] = updated_cache
    end

    hidden, final_norm_state = model.FinalNorm(hidden, ps.FinalNorm, st.FinalNorm)
    logits, output_state = model.OutputHead(hidden, ps.OutputHead, st.OutputHead)
    logits = logits ./ model.config.logits_scaling

    return logits, (
        TokenEmbedding = embedding_state,
        Blocks = Tuple(block_states),
        FinalNorm = final_norm_state,
        OutputHead = output_state,
    ), NativeDecoderCache(updated_layer_caches)
end

function resolve_hf_model_dir(
    model_ref::String;
    allow_patterns::Union{Nothing,Vector{String}} = nothing,
    local_files_only::Bool = false,
)
    if isdir(model_ref)
        return abspath(model_ref)
    end

    huggingface_hub = _pyimport("huggingface_hub")
    kwargs = Dict{Symbol,Any}(
        :repo_id => model_ref,
        :local_files_only => local_files_only,
    )
    if allow_patterns !== nothing
        kwargs[:allow_patterns] = allow_patterns
    end
    return String(huggingface_hub.snapshot_download(; kwargs...))
end

function load_json_file(path::String)
    return JSON3.read(read(path, String))
end

function granite_config_from_hf(model_ref::String; local_files_only::Bool = false)
    model_dir = resolve_hf_model_dir(
        model_ref;
        allow_patterns = ["config.json"],
        local_files_only = local_files_only,
    )
    config_path = joinpath(model_dir, "config.json")
    isfile(config_path) || error("Granite config not found at $(config_path)")
    config_data = load_json_file(config_path)

    model_type = String(get(config_data, "model_type", ""))
    model_type == "granitemoehybrid" ||
        error("Unsupported model_type=$(repr(model_type)). Expected \"granitemoehybrid\".")

    layer_types = collect(String.(get(config_data, "layer_types", String[])))
    all(layer_type -> layer_type == "attention", layer_types) ||
        error("NativeTeacherLM currently supports attention-only Granite checkpoints. layer_types=$(layer_types)")

    Int(get(config_data, "num_local_experts", 0)) == 0 ||
        error("NativeTeacherLM does not support Granite MoE experts yet.")
    Int(get(config_data, "num_experts_per_tok", 0)) == 0 ||
        error("NativeTeacherLM does not support Granite MoE experts yet.")

    return NativeTeacherConfig(
        vocab_size = Int(get(config_data, "vocab_size", 0)),
        max_sequence_length = Int(get(config_data, "max_position_embeddings", 0)),
        embedding_dimension = Int(get(config_data, "hidden_size", 0)),
        number_of_layers = Int(get(config_data, "num_hidden_layers", length(layer_types))),
        number_of_heads = Int(get(config_data, "num_attention_heads", 0)),
        number_of_kv_heads = Int(get(config_data, "num_key_value_heads", get(config_data, "num_attention_heads", 0))),
        mlp_hidden_dimension = Int(get(config_data, "shared_intermediate_size", get(config_data, "intermediate_size", 0))),
        rope_theta = Float32(get(config_data, "rope_theta", 10_000.0)),
        rms_norm_eps = Float32(get(config_data, "rms_norm_eps", 1.0f-6)),
        embedding_multiplier = Float32(get(config_data, "embedding_multiplier", 1.0)),
        logits_scaling = Float32(get(config_data, "logits_scaling", 1.0)),
        tie_word_embeddings = Bool(get(config_data, "tie_word_embeddings", false)),
        dropout_rate = Float32(get(config_data, "attention_dropout", 0.0)),
        use_bias = Bool(get(config_data, "attention_bias", false)),
        attention_multiplier = Float32(get(config_data, "attention_multiplier", NaN)),
        residual_multiplier = Float32(get(config_data, "residual_multiplier", 1.0)),
    )
end

function load_granite_index(model_dir::String)
    index_path = joinpath(model_dir, "model.safetensors.index.json")
    isfile(index_path) || error("Granite safetensors index not found at $(index_path)")
    index_data = load_json_file(index_path)
    weight_map_json = get(index_data, "weight_map", nothing)
    weight_map_json === nothing && error("Granite index is missing weight_map")

    weight_map = Dict{String,String}()
    for (tensor_name, shard_name) in pairs(weight_map_json)
        weight_map[String(tensor_name)] = String(shard_name)
    end
    return weight_map
end

function torch_dtype_for(::Type{Float32})
    return "float32"
end

function torch_dtype_for(::Type{Float16})
    return "float16"
end

function py_tensor_to_julia(py_tensor, dtype::Type{T}) where {T <: AbstractFloat}
    torch = _pyimport("torch")
    np_array = py_tensor.to(getproperty(torch, Symbol(torch_dtype_for(dtype)))).cpu().numpy()
    if _PyObject_check(np_array)
        return Array{dtype}(_PyArray(np_array))
    end
    return Array{dtype}(np_array)
end

function load_shard_tensors(model_dir::String, shard_name::String, tensor_names::Vector{String}, dtype::Type{T}) where {T <: AbstractFloat}
    shard_path = joinpath(model_dir, shard_name)
    isfile(shard_path) || error("Granite shard missing at $(shard_path)")

    safetensors = _pyimport("safetensors")
    loaded = Dict{String,Array{T}}()
    safe_handle = safetensors.safe_open(shard_path; framework = "pt", device = "cpu")
    for tensor_name in tensor_names
        loaded[tensor_name] = py_tensor_to_julia(safe_handle.get_tensor(tensor_name), dtype)
    end

    return loaded
end

function block_tensor_names(layer_index::Int)
    prefix = "model.layers.$(layer_index - 1)"
    return (
        input_norm = "$(prefix).input_layernorm.weight",
        post_norm = "$(prefix).post_attention_layernorm.weight",
        q_proj = "$(prefix).self_attn.q_proj.weight",
        k_proj = "$(prefix).self_attn.k_proj.weight",
        v_proj = "$(prefix).self_attn.v_proj.weight",
        o_proj = "$(prefix).self_attn.o_proj.weight",
        mlp_input = "$(prefix).shared_mlp.input_linear.weight",
        mlp_output = "$(prefix).shared_mlp.output_linear.weight",
    )
end

function load_granite_weights(
    model::NativeCausalLM,
    params,
    model_ref::String;
    dtype::Type{T} = Float32,
    local_files_only::Bool = false,
) where {T <: AbstractFloat}
    model_dir = resolve_hf_model_dir(
        model_ref;
        allow_patterns = ["config.json", "model.safetensors.index.json", "model-*.safetensors"],
        local_files_only = local_files_only,
    )
    weight_map = load_granite_index(model_dir)

    required_names = String[
        "model.embed_tokens.weight",
        "model.norm.weight",
    ]
    if !model.config.tie_word_embeddings
        push!(required_names, "lm_head.weight")
    end
    for layer_index in 1:model.config.number_of_layers
        names = block_tensor_names(layer_index)
        append!(
            required_names,
            [
                names.input_norm,
                names.post_norm,
                names.q_proj,
                names.k_proj,
                names.v_proj,
                names.o_proj,
                names.mlp_input,
                names.mlp_output,
            ],
        )
    end

    shard_groups = Dict{String,Vector{String}}()
    for tensor_name in required_names
        shard_name = get(weight_map, tensor_name, nothing)
        shard_name === nothing && error("Missing Granite tensor mapping for $(tensor_name)")
        push!(get!(shard_groups, shard_name, String[]), tensor_name)
    end

    loaded_tensors = Dict{String,Array{T}}()
    for (shard_name, tensor_names) in sort!(collect(shard_groups); by = first)
        merge!(loaded_tensors, load_shard_tensors(model_dir, shard_name, tensor_names, dtype))
    end

    embed_weight = loaded_tensors["model.embed_tokens.weight"]
    token_embedding = (
        weight = permutedims(embed_weight, (2, 1)),
    )

    block_params = Vector{Any}(undef, model.config.number_of_layers)
    hidden_size = model.config.mlp_hidden_dimension
    for layer_index in 1:model.config.number_of_layers
        names = block_tensor_names(layer_index)
        mlp_input = loaded_tensors[names.mlp_input]
        size(mlp_input, 1) == 2 * hidden_size ||
            error("Unexpected Granite MLP input shape $(size(mlp_input)) for layer $(layer_index)")

        block_params[layer_index] = (
            AttentionNorm = (scale = vec(loaded_tensors[names.input_norm]),),
            FeedForwardNorm = (scale = vec(loaded_tensors[names.post_norm]),),
            Attention = (
                QueryProjection = (weight = loaded_tensors[names.q_proj],),
                KeyProjection = (weight = loaded_tensors[names.k_proj],),
                ValueProjection = (weight = loaded_tensors[names.v_proj],),
                OutputProjection = (weight = loaded_tensors[names.o_proj],),
                Rope = params.Blocks[layer_index].Attention.Rope,
                Dropout = params.Blocks[layer_index].Attention.Dropout,
            ),
            FeedForward = (
                GateProjection = (weight = mlp_input[1:hidden_size, :],),
                UpProjection = (weight = mlp_input[(hidden_size + 1):(2 * hidden_size), :],),
                DownProjection = (weight = loaded_tensors[names.mlp_output],),
                Dropout = params.Blocks[layer_index].FeedForward.Dropout,
            ),
        )
    end

    output_weight = model.config.tie_word_embeddings ? embed_weight : loaded_tensors["lm_head.weight"]

    return (
        TokenEmbedding = token_embedding,
        Blocks = Tuple(block_params),
        FinalNorm = (scale = vec(loaded_tensors["model.norm.weight"]),),
        OutputHead = (weight = output_weight,),
    )
end

function load_granite_model(
    model_ref::String;
    rng::Random.AbstractRNG = Random.default_rng(),
    dtype::Type{T} = Float32,
    local_files_only::Bool = false,
) where {T <: AbstractFloat}
    config = granite_config_from_hf(model_ref; local_files_only = local_files_only)
    model = NativeCausalLM(config)
    params, state = Lux.setup(rng, model)
    loaded_params = load_granite_weights(
        model,
        params,
        model_ref;
        dtype = dtype,
        local_files_only = local_files_only,
    )
    return model, loaded_params, Lux.testmode(state)
end

function next_token_logits(model::NativeCausalLM, token_ids::AbstractArray{<:Integer}, ps, st)
    logits, new_state = model(token_ids, ps, st)
    return logits[:, end, :], new_state
end

function next_token_logits_cached(
    model::NativeCausalLM,
    token_ids::AbstractArray{<:Integer},
    ps,
    st,
    cache::NativeDecoderCache,
)
    logits, new_state, new_cache = forward_with_cache(model, token_ids, ps, st, cache)
    return logits[:, end, :], new_state, new_cache
end

function greedy_generate(
    model::NativeCausalLM,
    token_ids::AbstractArray{<:Integer},
    ps,
    st;
    max_new_tokens::Int,
    eos_token_id::Union{Int, Nothing} = nothing,
)
    tokens, generation_state, _ = greedy_generate_cached(
        model, token_ids, ps, st;
        max_new_tokens = max_new_tokens,
        eos_token_id = eos_token_id,
    )
    return tokens, generation_state
end

function greedy_generate_cached(
    model::NativeCausalLM,
    token_ids::AbstractArray{<:Integer},
    ps,
    st;
    max_new_tokens::Int,
    eos_token_id::Union{Int, Nothing} = nothing,
)
    tokens = ensure_token_batch(token_ids)
    batch_size = size(tokens, 2)
    finished = falses(batch_size)
    generation_state = st
    cache = init_decoder_cache(model, batch_size)

    prefill_logits, generation_state, cache = forward_with_cache(model, tokens, ps, generation_state, cache)

    for _ in 1:max_new_tokens
        next_tokens = similar(tokens, 1, batch_size)
        @inbounds for batch_index in 1:batch_size
            source_logits = if size(prefill_logits, 2) == 1
                view(prefill_logits, :, 1, batch_index)
            else
                view(prefill_logits, :, size(prefill_logits, 2), batch_index)
            end
            next_tokens[1, batch_index] = finished[batch_index] && eos_token_id !== nothing ?
                eos_token_id : argmax(source_logits)
        end
        tokens = vcat(tokens, next_tokens)
        if eos_token_id !== nothing
            finished .|= vec(next_tokens[1, :] .== eos_token_id)
            all(finished) && break
        end
        prefill_logits, generation_state, cache =
            forward_with_cache(model, next_tokens, ps, generation_state, cache)
        size(tokens, 1) < model.config.max_sequence_length || break
    end

    return tokens, generation_state, cache
end

end
