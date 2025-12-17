module Classification

"""
Classification model using Ossamma architecture.

Adapts OssammaBlock for sequence classification tasks by:
1. Using a fixed time embedding (no diffusion conditioning)
2. Pooling sequence representations to a single vector
3. Projecting to class logits

Supports multiple pooling strategies:
- :mean  - Average over sequence dimension
- :cls   - Use first token (requires prepending CLS token)
- :last  - Use last token representation
- :max   - Max pooling over sequence
"""

using Lux
using Random
using NNlib
using Statistics: mean

# Import parent module components (assumes we're included from Ossamma module)
import ..OssammaBlock
import ..TimeConditionedLayerNorm

const LuxLayer = Lux.AbstractLuxLayer

# ============================================================================
# Configuration
# ============================================================================

"""
Configuration for classification model.
"""
Base.@kwdef struct ClassifierConfig
    # Architecture
    vocab_size::Int = 32000
    max_sequence_length::Int = 512
    embedding_dimension::Int = 256
    number_of_heads::Int = 4
    number_of_layers::Int = 6
    num_classes::Int = 2

    # Internal dimensions
    time_dimension::Int = 128
    state_dimension::Int = -1  # -1 means use embedding_dimension

    # Attention
    window_size::Int = 5

    # Oscillator SSM
    min_frequency::Float32 = 0.1f0
    max_frequency::Float32 = 10.0f0
    default_time_step::Float32 = 0.1f0

    # Classification
    pooling::Symbol = :mean  # :mean, :cls, :last, :max
    dropout_rate::Float32 = 0.1f0
    use_cls_token::Bool = false  # Whether to prepend CLS token
end

# ============================================================================
# Pooling Layer
# ============================================================================

struct SequencePooling <: LuxLayer
    strategy::Symbol

    function SequencePooling(strategy::Symbol = :mean)
        @assert strategy in (:mean, :cls, :last, :max) "Pooling strategy must be :mean, :cls, :last, or :max"
        return new(strategy)
    end
end

Lux.initialparameters(::Random.AbstractRNG, ::SequencePooling) = (;)
Lux.initialstates(::Random.AbstractRNG, ::SequencePooling) = (;)

function (layer::SequencePooling)(x, params, state)
    # x: (embedding_dim, seq_len, batch) or (embedding_dim, seq_len)
    is_batched = ndims(x) == 3
    seq_dim = 2

    pooled = if layer.strategy == :mean
        dropdims(mean(x, dims=seq_dim), dims=seq_dim)
    elseif layer.strategy == :cls
        # First token
        is_batched ? x[:, 1, :] : x[:, 1]
    elseif layer.strategy == :last
        # Last token
        is_batched ? x[:, end, :] : x[:, end]
    elseif layer.strategy == :max
        dropdims(maximum(x, dims=seq_dim), dims=seq_dim)
    end

    return pooled, state
end

# ============================================================================
# Fixed Time Embedding (constant, no diffusion)
# ============================================================================

struct FixedTimeEmbedding <: LuxLayer
    time_dimension::Int
    fixed_value::Float32
end

function FixedTimeEmbedding(time_dimension::Int; fixed_value::Float32 = 0.5f0)
    return FixedTimeEmbedding(time_dimension, fixed_value)
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::FixedTimeEmbedding)
    # Learnable time embedding vector (initialized from fixed sinusoidal)
    half_dim = layer.time_dimension ÷ 2
    freqs = exp.(-(log(10000.0f0)) .* collect(0:half_dim-1) ./ half_dim)
    args = freqs .* layer.fixed_value
    init_embedding = vcat(sin.(args), cos.(args))
    return (embedding = init_embedding,)
end

Lux.initialstates(::Random.AbstractRNG, ::FixedTimeEmbedding) = (;)

function (layer::FixedTimeEmbedding)(batch_size::Int, params, state)
    # Return fixed embedding repeated for batch
    # Output: (time_dim, batch)
    embedding = repeat(reshape(params.embedding, :, 1), 1, batch_size)
    return embedding, state
end

# ============================================================================
# Ossamma Classifier
# ============================================================================

struct OssammaClassifier{E, P, T, B, PO, C} <: LuxLayer
    vocab_size::Int
    max_sequence_length::Int
    embedding_dimension::Int
    number_of_heads::Int
    number_of_layers::Int
    num_classes::Int
    pooling_strategy::Symbol
    use_cls_token::Bool

    # Embeddings
    TokenEmbedding::E
    PositionEmbedding::P
    TimeEmbedding::T

    # Encoder blocks
    Blocks::B

    # Classification head
    Pooling::PO
    Classifier::C
end

"""
    OssammaClassifier(config::ClassifierConfig)

Create a classifier from configuration.
"""
function OssammaClassifier(config::ClassifierConfig)
    state_dimension = config.state_dimension == -1 ? config.embedding_dimension : config.state_dimension

    return OssammaClassifier(;
        vocab_size = config.vocab_size,
        max_sequence_length = config.max_sequence_length,
        embedding_dimension = config.embedding_dimension,
        number_of_heads = config.number_of_heads,
        number_of_layers = config.number_of_layers,
        num_classes = config.num_classes,
        time_dimension = config.time_dimension,
        state_dimension = state_dimension,
        window_size = config.window_size,
        min_frequency = config.min_frequency,
        max_frequency = config.max_frequency,
        default_time_step = config.default_time_step,
        pooling = config.pooling,
        use_cls_token = config.use_cls_token,
    )
end

function OssammaClassifier(;
    vocab_size::Int,
    max_sequence_length::Int,
    embedding_dimension::Int,
    number_of_heads::Int,
    number_of_layers::Int,
    num_classes::Int,
    time_dimension::Int = 128,
    state_dimension::Int = embedding_dimension,
    window_size::Int = 5,
    min_frequency::Float32 = 0.1f0,
    max_frequency::Float32 = 10.0f0,
    default_time_step::Float32 = 0.1f0,
    pooling::Symbol = :mean,
    use_cls_token::Bool = false,
)
    # Add CLS token to vocab if needed
    actual_vocab_size = use_cls_token ? vocab_size + 1 : vocab_size

    # Build stack of OssammaBlocks
    blocks = Tuple([
        OssammaBlock(
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

    return OssammaClassifier(
        actual_vocab_size,
        max_sequence_length,
        embedding_dimension,
        number_of_heads,
        number_of_layers,
        num_classes,
        pooling,
        use_cls_token,
        # Embeddings
        Lux.Embedding(actual_vocab_size => embedding_dimension),
        Lux.Embedding(max_sequence_length => embedding_dimension),
        FixedTimeEmbedding(time_dimension),
        # Encoder blocks
        blocks,
        # Classification head
        SequencePooling(pooling),
        Lux.Chain(
            Lux.LayerNorm((embedding_dimension,)),
            Lux.Dense(embedding_dimension => embedding_dimension, NNlib.gelu),
            Lux.Dense(embedding_dimension => num_classes),
        ),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::OssammaClassifier)
    block_params = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialparameters(rng, block) for block in model.Blocks)
    )

    return (
        TokenEmbedding = Lux.initialparameters(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialparameters(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialparameters(rng, model.TimeEmbedding),
        Blocks = block_params,
        Pooling = Lux.initialparameters(rng, model.Pooling),
        Classifier = Lux.initialparameters(rng, model.Classifier),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::OssammaClassifier)
    block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialstates(rng, block) for block in model.Blocks)
    )

    return (
        TokenEmbedding = Lux.initialstates(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialstates(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialstates(rng, model.TimeEmbedding),
        Blocks = block_states,
        Pooling = Lux.initialstates(rng, model.Pooling),
        Classifier = Lux.initialstates(rng, model.Classifier),
    )
end

function (model::OssammaClassifier)(token_ids::AbstractArray, params, state)
    # token_ids: (seq_len,) or (seq_len, batch)

    is_batched = ndims(token_ids) == 2
    seq_len = size(token_ids, 1)
    batch_size = is_batched ? size(token_ids, 2) : 1

    # Standardize to batched format
    token_ids_batched = is_batched ? token_ids : reshape(token_ids, :, 1)

    # =========================================================================
    # 1. Token Embedding
    # =========================================================================
    token_flat = vec(token_ids_batched)
    token_emb_flat, tok_state = model.TokenEmbedding(token_flat, params.TokenEmbedding, state.TokenEmbedding)
    token_emb = reshape(token_emb_flat, model.embedding_dimension, seq_len, batch_size)

    # =========================================================================
    # 2. Position Embedding
    # =========================================================================
    position_indices = collect(1:seq_len)
    pos_emb_raw, pos_state = model.PositionEmbedding(position_indices, params.PositionEmbedding, state.PositionEmbedding)
    pos_emb = reshape(pos_emb_raw, model.embedding_dimension, seq_len, 1)

    # =========================================================================
    # 3. Combine Embeddings
    # =========================================================================
    hidden = token_emb .+ pos_emb

    # =========================================================================
    # 4. Fixed Time Embedding (no mask ratio conditioning)
    # =========================================================================
    time_emb, time_state = model.TimeEmbedding(batch_size, params.TimeEmbedding, state.TimeEmbedding)

    # =========================================================================
    # 5. Process through OssammaBlocks
    # =========================================================================
    (hidden, block_states) = foldl(
        enumerate(model.Blocks);
        init = (hidden, ())
    ) do (h, states), (i, block)
        block_key = Symbol("Block_$i")
        block_params = params.Blocks[block_key]
        block_state = state.Blocks[block_key]

        new_h, new_block_state = block((h, time_emb), block_params, block_state)
        (new_h, (states..., new_block_state))
    end

    # =========================================================================
    # 6. Pool sequence to single vector
    # =========================================================================
    pooled, pool_state = model.Pooling(hidden, params.Pooling, state.Pooling)
    # pooled: (embedding_dim, batch)

    # =========================================================================
    # 7. Classification head
    # =========================================================================
    logits, classifier_state = model.Classifier(pooled, params.Classifier, state.Classifier)
    # logits: (num_classes, batch)

    # Remove batch dim if input wasn't batched
    final_logits = is_batched ? logits : dropdims(logits, dims=2)

    # =========================================================================
    # 8. Update State
    # =========================================================================
    new_block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        block_states
    )

    new_state = (
        TokenEmbedding = tok_state,
        PositionEmbedding = pos_state,
        TimeEmbedding = time_state,
        Blocks = new_block_states,
        Pooling = pool_state,
        Classifier = classifier_state,
    )

    return final_logits, new_state
end

# ============================================================================
# Convenience constructors for common configurations
# ============================================================================

"""
    tiny_classifier(; num_classes, vocab_size, kwargs...)

Tiny classifier for debugging and quick tests.
"""
function tiny_classifier(;
    num_classes::Int = 2,
    vocab_size::Int = 1000,
    max_sequence_length::Int = 64,
    kwargs...
)
    config = ClassifierConfig(;
        vocab_size = vocab_size,
        max_sequence_length = max_sequence_length,
        embedding_dimension = 64,
        number_of_heads = 2,
        number_of_layers = 2,
        num_classes = num_classes,
        time_dimension = 32,
        kwargs...
    )
    return OssammaClassifier(config)
end

"""
    small_classifier(; num_classes, vocab_size, kwargs...)

Small classifier suitable for fine-tuning experiments.
"""
function small_classifier(;
    num_classes::Int = 2,
    vocab_size::Int = 32000,
    max_sequence_length::Int = 256,
    kwargs...
)
    config = ClassifierConfig(;
        vocab_size = vocab_size,
        max_sequence_length = max_sequence_length,
        embedding_dimension = 256,
        number_of_heads = 4,
        number_of_layers = 4,
        num_classes = num_classes,
        time_dimension = 64,
        kwargs...
    )
    return OssammaClassifier(config)
end

"""
    base_classifier(; num_classes, vocab_size, kwargs...)

Base-sized classifier for production use.
"""
function base_classifier(;
    num_classes::Int = 2,
    vocab_size::Int = 32000,
    max_sequence_length::Int = 512,
    kwargs...
)
    config = ClassifierConfig(;
        vocab_size = vocab_size,
        max_sequence_length = max_sequence_length,
        embedding_dimension = 512,
        number_of_heads = 8,
        number_of_layers = 8,
        num_classes = num_classes,
        time_dimension = 128,
        kwargs...
    )
    return OssammaClassifier(config)
end

# ============================================================================
# Exports
# ============================================================================

export OssammaClassifier, ClassifierConfig
export SequencePooling, FixedTimeEmbedding
export tiny_classifier, small_classifier, base_classifier

end # module
