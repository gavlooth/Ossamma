module RelationExtraction

using Lux
using Random
using TOML
using JSON3
using NNlib
using Statistics: mean
import CUDA
import ChainRulesCore
import Zygote

import ..Swamma: SwammaBlock, SwammaBlockConfig, LocalWaveRefinementBlock, LuxLayer

const DEFAULT_ENTITY_TYPES = [
    "PERSON", "ORGANIZATION", "LOCATION", "EVENT", "MISC"
]

const DEFAULT_ENTITY_LABELS = vcat(
    ["O"],
    reduce(vcat, [["B-$label", "I-$label"] for label in DEFAULT_ENTITY_TYPES])
)

Base.@kwdef struct RelationExtractionConfig
    vocab_size::Int = 64000
    max_sequence_length::Int = 512
    embedding_dimension::Int = 768
    number_of_heads::Int = 12
    number_of_layers::Int = 24
    number_of_refinement_layers::Int = 0
    num_entity_labels::Int = length(DEFAULT_ENTITY_LABELS)
    num_relations::Int = 64
    time_dimension::Int = 192
    state_dimension::Int = -1
    window_size::Int = 24
    local_operator::Symbol = :swattention
    residual_mode::Symbol = :plain
    min_frequency::Float32 = 0.01f0
    max_frequency::Float32 = 5.0f0
    default_time_step::Float32 = 0.05f0
    dropout_rate::Float32 = 0.1f0
    use_ffn::Bool = true
    ffn_expansion::Float32 = 4f0 / 3f0
    use_parallel_scan::Bool = false
    parallel_chunk_size::Int = 64
    use_vector_gains::Bool = false
    use_per_head_alpha::Bool = false
    use_branch_projections::Bool = false
    max_candidate_spans::Int = 64
    max_candidate_pairs::Int = 256
    max_span_width::Int = 8
    biaffine_rank::Int = 64
    pair_neighbor_radius::Int = 4
end

struct FixedTimeEmbedding <: LuxLayer
    time_dimension::Int
    fixed_value::Float32
end

function FixedTimeEmbedding(time_dimension::Int; fixed_value::Float32 = 0.5f0)
    FixedTimeEmbedding(time_dimension, fixed_value)
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::FixedTimeEmbedding)
    half_dim = layer.time_dimension ÷ 2
    freqs = exp.(-(log(10000.0f0)) .* collect(Float32, 0:half_dim-1) ./ max(half_dim, 1))
    args = freqs .* layer.fixed_value
    embedding = vcat(sin.(args), cos.(args))
    if length(embedding) < layer.time_dimension
        embedding = vcat(embedding, zeros(Float32, layer.time_dimension - length(embedding)))
    end
    return (embedding = embedding,)
end

Lux.initialstates(::Random.AbstractRNG, ::FixedTimeEmbedding) = (;)

function (layer::FixedTimeEmbedding)(batch_size::Int, params, state)
    embedding = repeat(reshape(params.embedding, :, 1), 1, batch_size)
    return embedding, state
end

struct LowRankBiaffineRelationHead{HP,TP,IP,LP} <: LuxLayer
    embedding_dimension::Int
    rank::Int
    num_relations::Int
    HeadProjection::HP
    TailProjection::TP
    InteractionProjection::IP
    LinearProjection::LP
end

function LowRankBiaffineRelationHead(
    embedding_dimension::Int,
    num_relations::Int;
    rank::Int = min(64, embedding_dimension),
)
    return LowRankBiaffineRelationHead(
        embedding_dimension,
        rank,
        num_relations,
        Lux.Dense(embedding_dimension => rank),
        Lux.Dense(embedding_dimension => rank),
        Lux.Dense(rank => num_relations; use_bias = false),
        Lux.Dense(2 * embedding_dimension => num_relations),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::LowRankBiaffineRelationHead)
    return (
        HeadProjection = Lux.initialparameters(rng, layer.HeadProjection),
        TailProjection = Lux.initialparameters(rng, layer.TailProjection),
        InteractionProjection = Lux.initialparameters(rng, layer.InteractionProjection),
        LinearProjection = Lux.initialparameters(rng, layer.LinearProjection),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::LowRankBiaffineRelationHead)
    return (
        HeadProjection = Lux.initialstates(rng, layer.HeadProjection),
        TailProjection = Lux.initialstates(rng, layer.TailProjection),
        InteractionProjection = Lux.initialstates(rng, layer.InteractionProjection),
        LinearProjection = Lux.initialstates(rng, layer.LinearProjection),
    )
end

function (layer::LowRankBiaffineRelationHead)(inputs::Tuple, params, state)
    head_vectors, tail_vectors = inputs

    head_proj, head_state = layer.HeadProjection(head_vectors, params.HeadProjection, state.HeadProjection)
    tail_proj, tail_state = layer.TailProjection(tail_vectors, params.TailProjection, state.TailProjection)

    bilinear_scores, interaction_state = layer.InteractionProjection(
        head_proj .* tail_proj,
        params.InteractionProjection,
        state.InteractionProjection,
    )
    affine_scores, linear_state = layer.LinearProjection(
        vcat(head_vectors, tail_vectors),
        params.LinearProjection,
        state.LinearProjection,
    )

    new_state = (
        HeadProjection = head_state,
        TailProjection = tail_state,
        InteractionProjection = interaction_state,
        LinearProjection = linear_state,
    )

    return bilinear_scores .+ affine_scores, new_state
end

struct SwammaRelationExtractor{E,P,T,B,RB,D,EH,BH,SP,RH,CH} <: LuxLayer
    vocab_size::Int
    max_sequence_length::Int
    embedding_dimension::Int
    number_of_layers::Int
    number_of_refinement_layers::Int
    num_entity_labels::Int
    num_relations::Int
    max_candidate_spans::Int
    max_candidate_pairs::Int
    max_span_width::Int
    pair_neighbor_radius::Int
    TokenEmbedding::E
    PositionEmbedding::P
    TimeEmbedding::T
    Blocks::B
    RefinementBlocks::RB
    Dropout::D
    EntityHead::EH
    BoundaryHead::BH
    SpanProjection::SP
    RelationHead::RH
    ConfidenceHead::CH
end

function load_relation_extraction_config(path::String)::RelationExtractionConfig
    toml = TOML.parsefile(path)
    model = get(toml, "model", Dict())
    dims = get(model, "dimensions", Dict())
    attn = get(model, "attention", Dict())
    refinement = get(model, "refinement", Dict())
    residual = get(model, "residual", Dict())
    osc = get(model, "wave_gate", Dict())
    reg = get(model, "regularization", Dict())
    ablation = get(model, "ablation", Dict())
    parallel = get(toml, "parallelization", Dict())
    relation = get(toml, "relation_extraction", Dict())

    return RelationExtractionConfig(
        vocab_size = get(model, "vocab_size", 64000),
        max_sequence_length = get(model, "max_sequence_length", 512),
        embedding_dimension = get(model, "embedding_dimension", 768),
        number_of_heads = get(model, "number_of_heads", 12),
        number_of_layers = get(model, "number_of_layers", 24),
        number_of_refinement_layers = get(refinement, "number_of_layers", 0),
        num_entity_labels = get(model, "num_entity_labels", length(DEFAULT_ENTITY_LABELS)),
        num_relations = get(model, "num_relations", 64),
        time_dimension = get(dims, "time_dimension", 192),
        state_dimension = get(dims, "state_dimension", -1),
        window_size = get(attn, "window_size", 24),
        local_operator = begin
            raw = get(attn, "local_operator", "swattention")
            raw isa Symbol ? raw : Symbol(raw)
        end,
        residual_mode = begin
            raw = get(residual, "mode", "plain")
            raw isa Symbol ? raw : Symbol(raw)
        end,
        min_frequency = Float32(get(osc, "min_frequency", 0.01)),
        max_frequency = Float32(get(osc, "max_frequency", 5.0)),
        default_time_step = Float32(get(osc, "default_time_step", 0.05)),
        dropout_rate = Float32(get(reg, "dropout_rate", 0.1)),
        use_ffn = get(ablation, "use_ffn", true),
        ffn_expansion = Float32(get(ablation, "ffn_expansion", 4.0 / 3.0)),
        use_parallel_scan = get(parallel, "use_parallel_scan", false),
        parallel_chunk_size = get(parallel, "chunk_size", 64),
        use_vector_gains = get(ablation, "use_vector_gains", false),
        use_per_head_alpha = get(ablation, "use_per_head_alpha", false),
        use_branch_projections = get(ablation, "use_branch_projections", false),
        max_candidate_spans = get(relation, "max_candidate_spans", 64),
        max_candidate_pairs = get(relation, "max_candidate_pairs", 256),
        max_span_width = get(relation, "max_span_width", 8),
        biaffine_rank = get(relation, "biaffine_rank", 64),
        pair_neighbor_radius = get(relation, "pair_neighbor_radius", 4),
    )
end

function entity_span_end(entity)
    if haskey(entity, :stop)
        return Int(entity.stop)
    elseif haskey(entity, Symbol("end"))
        return Int(getproperty(entity, Symbol("end")))
    else
        throw(ArgumentError("Entity span must define either `stop` or `end`."))
    end
end

function print_relation_extraction_summary(config::RelationExtractionConfig)
    println("=" ^ 60)
    println("Swamma Relation Extraction Summary")
    println("=" ^ 60)
    println("Backbone:")
    println("  vocab_size:           $(config.vocab_size)")
    println("  max_sequence_length:  $(config.max_sequence_length)")
    println("  embedding_dimension:  $(config.embedding_dimension)")
    println("  number_of_heads:      $(config.number_of_heads)")
    println("  number_of_layers:     $(config.number_of_layers)")
    println("  refinement_layers:    $(config.number_of_refinement_layers)")
    println("  num_entity_labels:    $(config.num_entity_labels)")
    println("  num_relations:        $(config.num_relations)")
    println("  window_size:          $(config.window_size)")
    println("  local_operator:       $(config.local_operator)")
    println("  residual_mode:        $(config.residual_mode)")
    println("Heads:")
    println("  max_candidate_spans:  $(config.max_candidate_spans)")
    println("  max_candidate_pairs:  $(config.max_candidate_pairs)")
    println("  max_span_width:       $(config.max_span_width)")
    println("  biaffine_rank:        $(config.biaffine_rank)")
    println("  pair_neighbor_radius: $(config.pair_neighbor_radius)")
    println("=" ^ 60)
end

function SwammaRelationExtractor(config::RelationExtractionConfig)
    state_dimension = config.state_dimension == -1 ? config.embedding_dimension : config.state_dimension
    block_config = SwammaBlockConfig(
        embedding_dimension = config.embedding_dimension,
        sequence_length = config.max_sequence_length,
        number_of_heads = config.number_of_heads,
        time_dimension = config.time_dimension,
        state_dimension = state_dimension,
        window_size = config.window_size,
        local_operator = config.local_operator,
        residual_mode = config.residual_mode,
        min_frequency = config.min_frequency,
        max_frequency = config.max_frequency,
        default_time_step = config.default_time_step,
        dropout_rate = config.dropout_rate,
        use_ffn = config.use_ffn,
        ffn_expansion = config.ffn_expansion,
        use_parallel_scan = config.use_parallel_scan,
        parallel_chunk_size = config.parallel_chunk_size,
        use_vector_gains = config.use_vector_gains,
        use_per_head_alpha = config.use_per_head_alpha,
        use_branch_projections = config.use_branch_projections,
    )
    blocks = Tuple([
        SwammaBlock(block_config)
        for _ in 1:config.number_of_layers
    ])
    refinement_blocks = Tuple([
        LocalWaveRefinementBlock(
            config.embedding_dimension,
            config.max_sequence_length,
            config.number_of_heads,
            config.time_dimension;
            state_dimension = state_dimension,
            window_size = config.window_size,
            min_frequency = config.min_frequency,
            max_frequency = config.max_frequency,
            default_time_step = config.default_time_step,
            dropout_rate = config.dropout_rate,
        )
        for _ in 1:config.number_of_refinement_layers
    ])

    d = config.embedding_dimension
    return SwammaRelationExtractor(
        config.vocab_size,
        config.max_sequence_length,
        d,
        config.number_of_layers,
        config.number_of_refinement_layers,
        config.num_entity_labels,
        config.num_relations,
        config.max_candidate_spans,
        config.max_candidate_pairs,
        config.max_span_width,
        config.pair_neighbor_radius,
        Lux.Embedding(config.vocab_size => d),
        Lux.Embedding(config.max_sequence_length => d),
        FixedTimeEmbedding(config.time_dimension),
        blocks,
        refinement_blocks,
        Lux.Dropout(config.dropout_rate),
        Lux.Chain(
            Lux.LayerNorm((d,)),
            Lux.Dropout(config.dropout_rate),
            Lux.Dense(d => config.num_entity_labels),
        ),
        Lux.Chain(
            Lux.LayerNorm((d,)),
            Lux.Dense(d => 2; use_bias = false),
        ),
        Lux.Chain(
            Lux.LayerNorm((3 * d,)),
            Lux.Dense(3 * d => d, gelu),
        ),
        LowRankBiaffineRelationHead(
            d,
            config.num_relations;
            rank = min(config.biaffine_rank, d),
        ),
        Lux.Chain(
            Lux.LayerNorm((4 * d,)),
            Lux.Dense(4 * d => d ÷ 2, gelu; use_bias = false),
            Lux.Dense(d ÷ 2 => 1; use_bias = false),
        ),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::SwammaRelationExtractor)
    block_params = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialparameters(rng, block) for block in model.Blocks)
    )
    refinement_params = NamedTuple{ntuple(i -> Symbol("RefinementBlock_$i"), model.number_of_refinement_layers)}(
        Tuple(Lux.initialparameters(rng, block) for block in model.RefinementBlocks)
    )

    return (
        TokenEmbedding = Lux.initialparameters(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialparameters(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialparameters(rng, model.TimeEmbedding),
        Blocks = block_params,
        RefinementBlocks = refinement_params,
        Dropout = Lux.initialparameters(rng, model.Dropout),
        EntityHead = Lux.initialparameters(rng, model.EntityHead),
        BoundaryHead = Lux.initialparameters(rng, model.BoundaryHead),
        SpanProjection = Lux.initialparameters(rng, model.SpanProjection),
        RelationHead = Lux.initialparameters(rng, model.RelationHead),
        ConfidenceHead = Lux.initialparameters(rng, model.ConfidenceHead),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::SwammaRelationExtractor)
    block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialstates(rng, block) for block in model.Blocks)
    )
    refinement_states = NamedTuple{ntuple(i -> Symbol("RefinementBlock_$i"), model.number_of_refinement_layers)}(
        Tuple(Lux.initialstates(rng, block) for block in model.RefinementBlocks)
    )
    return (
        TokenEmbedding = Lux.initialstates(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialstates(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialstates(rng, model.TimeEmbedding),
        Blocks = block_states,
        RefinementBlocks = refinement_states,
        Dropout = Lux.initialstates(rng, model.Dropout),
        EntityHead = Lux.initialstates(rng, model.EntityHead),
        BoundaryHead = Lux.initialstates(rng, model.BoundaryHead),
        SpanProjection = Lux.initialstates(rng, model.SpanProjection),
        RelationHead = Lux.initialstates(rng, model.RelationHead),
        ConfidenceHead = Lux.initialstates(rng, model.ConfidenceHead),
        position_indices = collect(1:model.max_sequence_length),
    )
end

function encode_tokens(model::SwammaRelationExtractor, token_ids::AbstractArray, params, state)
    is_batched = ndims(token_ids) == 2
    seq_len = size(token_ids, 1)
    batch_size = is_batched ? size(token_ids, 2) : 1
    token_ids_batched = is_batched ? token_ids : reshape(token_ids, :, 1)

    token_flat = vec(token_ids_batched)
    token_emb_flat, tok_state = model.TokenEmbedding(token_flat, params.TokenEmbedding, state.TokenEmbedding)
    token_emb = reshape(token_emb_flat, model.embedding_dimension, seq_len, batch_size)

    position_indices = copy(state.position_indices[1:seq_len])
    pos_emb_raw, pos_state = model.PositionEmbedding(position_indices, params.PositionEmbedding, state.PositionEmbedding)
    pos_emb = reshape(pos_emb_raw, model.embedding_dimension, seq_len, 1)
    hidden = token_emb .+ pos_emb

    time_emb, time_state = model.TimeEmbedding(batch_size, params.TimeEmbedding, state.TimeEmbedding)

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

    new_block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(block_states)
    (hidden, refinement_states) = foldl(
        enumerate(model.RefinementBlocks);
        init = (hidden, ())
    ) do (h, states), (i, block)
        block_key = Symbol("RefinementBlock_$i")
        block_params = params.RefinementBlocks[block_key]
        block_state = state.RefinementBlocks[block_key]
        new_h, new_block_state = block((h, time_emb), block_params, block_state)
        (new_h, (states..., new_block_state))
    end

    new_refinement_states = NamedTuple{
        ntuple(i -> Symbol("RefinementBlock_$i"), model.number_of_refinement_layers)
    }(refinement_states)
    new_state = (
        TokenEmbedding = tok_state,
        PositionEmbedding = pos_state,
        TimeEmbedding = time_state,
        Blocks = new_block_states,
        RefinementBlocks = new_refinement_states,
        Dropout = state.Dropout,
        EntityHead = state.EntityHead,
        BoundaryHead = state.BoundaryHead,
        SpanProjection = state.SpanProjection,
        RelationHead = state.RelationHead,
        ConfidenceHead = state.ConfidenceHead,
        position_indices = state.position_indices,
    )
    return hidden, new_state
end

function token_entity_mass(entity_logits)
    seq_len = size(entity_logits, 2)
    batch_size = size(entity_logits, 3)
    if size(entity_logits, 1) <= 1
        return zeros(eltype(entity_logits), seq_len, batch_size)
    end
    probs = NNlib.softmax(entity_logits, dims = 1)
    return reshape(maximum(probs[2:end, :, :], dims = 1), seq_len, batch_size)
end

function score_span(start_scores, end_scores, entity_mass, start_idx::Int, end_idx::Int)
    width = end_idx - start_idx + 1
    entity_score = sum(@view(entity_mass[start_idx:end_idx])) / max(width, 1)
    return start_scores[start_idx] + end_scores[end_idx] + entity_score
end

function propose_candidate_spans(
    entity_logits,
    boundary_logits;
    max_candidate_spans::Int,
    max_span_width::Int,
)
    seq_len = size(entity_logits, 2)
    batch_size = size(entity_logits, 3)
    spans = zeros(Int, 2, max_candidate_spans, batch_size)
    span_mask = falses(max_candidate_spans, batch_size)
    span_scores = fill(typemin(Float32), max_candidate_spans, batch_size)
    entity_mass = token_entity_mass(entity_logits)

    for b in 1:batch_size
        start_scores = vec(NNlib.sigmoid.(boundary_logits[1, :, b]))
        end_scores = vec(NNlib.sigmoid.(boundary_logits[2, :, b]))
        candidates = Tuple{Float32, Int, Int}[]

        for start_idx in 1:seq_len
            max_end = min(seq_len, start_idx + max_span_width - 1)
            for end_idx in start_idx:max_end
                push!(
                    candidates,
                    (
                        Float32(score_span(start_scores, end_scores, @view(entity_mass[:, b]), start_idx, end_idx)),
                        start_idx,
                        end_idx,
                    ),
                )
            end
        end

        isempty(candidates) && continue
        top_count = min(max_candidate_spans, length(candidates))
        ordered = sort(candidates, by = x -> x[1], rev = true)
        for i in 1:top_count
            score, start_idx, end_idx = ordered[i]
            spans[1, i, b] = start_idx
            spans[2, i, b] = end_idx
            span_mask[i, b] = true
            span_scores[i, b] = score
        end
    end

    return spans, span_mask, span_scores
end

function score_existing_spans(entity_logits, boundary_logits, spans, span_mask)
    max_spans = size(spans, 2)
    batch_size = size(spans, 3)
    span_scores = fill(typemin(Float32), max_spans, batch_size)
    entity_mass = token_entity_mass(entity_logits)

    for b in 1:batch_size
        start_scores = vec(NNlib.sigmoid.(boundary_logits[1, :, b]))
        end_scores = vec(NNlib.sigmoid.(boundary_logits[2, :, b]))
        for i in 1:max_spans
            span_mask[i, b] || continue
            start_idx = spans[1, i, b]
            end_idx = spans[2, i, b]
            span_scores[i, b] = Float32(score_span(start_scores, end_scores, @view(entity_mass[:, b]), start_idx, end_idx))
        end
    end

    return span_scores
end

function propose_relation_pairs(
    spans,
    span_mask,
    span_scores;
    max_candidate_pairs::Int,
    neighbor_radius::Int,
)
    batch_size = size(spans, 3)
    relation_pairs = zeros(Int, 2, max_candidate_pairs, batch_size)
    relation_mask = falses(max_candidate_pairs, batch_size)

    for b in 1:batch_size
        valid_indices = findall(@view(span_mask[:, b]))
        isempty(valid_indices) && continue

        ordered_by_position = sort(valid_indices, by = i -> (spans[1, i, b], spans[2, i, b], -span_scores[i, b]))
        ordered_by_score = sort(valid_indices, by = i -> span_scores[i, b], rev = true)
        position_lookup = Dict(idx => pos for (pos, idx) in enumerate(ordered_by_position))
        seen_pairs = Set{Tuple{Int, Int}}()
        pair_idx = 0

        for anchor_idx in ordered_by_score
            pair_idx >= max_candidate_pairs && break
            anchor_pos = position_lookup[anchor_idx]

            for delta in 1:neighbor_radius
                pair_idx >= max_candidate_pairs && break

                if anchor_pos + delta <= length(ordered_by_position)
                    neighbor_idx = ordered_by_position[anchor_pos + delta]
                    for pair in ((anchor_idx, neighbor_idx), (neighbor_idx, anchor_idx))
                        pair_idx >= max_candidate_pairs && break
                        pair in seen_pairs && continue
                        push!(seen_pairs, pair)
                        pair_idx += 1
                        relation_pairs[1, pair_idx, b] = pair[1]
                        relation_pairs[2, pair_idx, b] = pair[2]
                        relation_mask[pair_idx, b] = true
                    end
                end

                if anchor_pos - delta >= 1
                    neighbor_idx = ordered_by_position[anchor_pos - delta]
                    for pair in ((anchor_idx, neighbor_idx), (neighbor_idx, anchor_idx))
                        pair_idx >= max_candidate_pairs && break
                        pair in seen_pairs && continue
                        push!(seen_pairs, pair)
                        pair_idx += 1
                        relation_pairs[1, pair_idx, b] = pair[1]
                        relation_pairs[2, pair_idx, b] = pair[2]
                        relation_mask[pair_idx, b] = true
                    end
                end
            end
        end
    end

    return relation_pairs, relation_mask
end

function build_span_representations(model::SwammaRelationExtractor, hidden, spans, span_mask, params, state)
    d, seq_len, batch_size = size(hidden)
    max_spans = size(spans, 2)
    on_gpu = hidden isa CUDA.CuArray

    start_idx = clamp.(spans[1, :, :], 1, seq_len)
    end_idx = max.(start_idx, clamp.(spans[2, :, :], 1, seq_len))
    width = max.(end_idx .- start_idx .+ 1, 1)

    span_offsets = reshape(Int.(0:max_spans:(batch_size - 1) * max_spans), 1, batch_size)
    seq_offsets = reshape(Int.(0:seq_len:(batch_size - 1) * seq_len), 1, batch_size)
    padded_offsets = reshape(Int.(0:(seq_len + 1):(batch_size - 1) * (seq_len + 1)), 1, batch_size)
    if on_gpu
        span_offsets = CUDA.CuArray(span_offsets)
        seq_offsets = CUDA.CuArray(seq_offsets)
        padded_offsets = CUDA.CuArray(padded_offsets)
    end

    hidden_flat = reshape(hidden, d, :)
    start_linear = vec(start_idx .+ seq_offsets)
    end_linear = vec(end_idx .+ seq_offsets)
    start_vecs = hidden_flat[:, start_linear]
    end_vecs = hidden_flat[:, end_linear]

    zero_pad = zero(eltype(hidden)) .* hidden[:, 1:1, :]
    cumulative = cat(zero_pad, cumsum(hidden, dims = 2); dims = 2)
    cumulative_flat = reshape(cumulative, d, :)
    prefix_start = vec(start_idx .+ padded_offsets)
    prefix_end = vec((end_idx .+ 1) .+ padded_offsets)
    sum_vecs = cumulative_flat[:, prefix_end] .- cumulative_flat[:, prefix_start]
    mean_vecs = sum_vecs ./ reshape(Float32.(vec(width)), 1, :)

    mask_values = reshape(Float32.(vec(span_mask)), 1, :)
    span_inputs = vcat(
        start_vecs .* mask_values,
        end_vecs .* mask_values,
        mean_vecs .* mask_values,
    )

    projected, span_state = model.SpanProjection(span_inputs, params.SpanProjection, state.SpanProjection)
    span_reps = reshape(projected, d, max_spans, batch_size)
    return span_reps, span_state
end

function gather_pair_span_vectors(span_reps, relation_pairs, relation_mask)
    d, max_spans, batch_size = size(span_reps)
    max_pairs = size(relation_pairs, 2)
    on_gpu = span_reps isa CUDA.CuArray

    pair_offsets = reshape(Int.(0:max_spans:(batch_size - 1) * max_spans), 1, batch_size)
    if on_gpu
        pair_offsets = CUDA.CuArray(pair_offsets)
    end

    head_idx = clamp.(relation_pairs[1, :, :], 1, max_spans)
    tail_idx = clamp.(relation_pairs[2, :, :], 1, max_spans)
    head_linear = vec(head_idx .+ pair_offsets)
    tail_linear = vec(tail_idx .+ pair_offsets)

    span_flat = reshape(span_reps, d, :)
    mask_values = reshape(Float32.(vec(relation_mask)), 1, :)
    head_vectors = span_flat[:, head_linear] .* mask_values
    tail_vectors = span_flat[:, tail_linear] .* mask_values
    return head_vectors, tail_vectors
end

function build_pair_features(head_vectors, tail_vectors)
    return vcat(
        head_vectors,
        tail_vectors,
        abs.(head_vectors .- tail_vectors),
        head_vectors .* tail_vectors,
    )
end

function build_pair_features(span_reps, relation_pairs, relation_mask)
    head_vectors, tail_vectors = gather_pair_span_vectors(span_reps, relation_pairs, relation_mask)
    return build_pair_features(head_vectors, tail_vectors)
end

function (model::SwammaRelationExtractor)(inputs::NamedTuple, params, state)
    token_ids = inputs.token_ids

    hidden, encoder_state = encode_tokens(model, token_ids, params, state)
    d, seq_len, batch_size = size(hidden)

    hidden_flat = reshape(hidden, d, :)
    hidden_flat, dropout_state = model.Dropout(hidden_flat, params.Dropout, state.Dropout)

    entity_logits_flat, entity_state = model.EntityHead(hidden_flat, params.EntityHead, state.EntityHead)
    entity_logits = reshape(entity_logits_flat, model.num_entity_labels, seq_len, batch_size)

    boundary_logits_flat, boundary_state = model.BoundaryHead(hidden_flat, params.BoundaryHead, state.BoundaryHead)
    boundary_logits = reshape(boundary_logits_flat, 2, seq_len, batch_size)

    spans, span_mask, span_scores = if hasproperty(inputs, :spans) && hasproperty(inputs, :span_mask)
        provided_scores = hasproperty(inputs, :span_scores) ?
            inputs.span_scores :
            score_existing_spans(entity_logits, boundary_logits, inputs.spans, inputs.span_mask)
        (inputs.spans, inputs.span_mask, provided_scores)
    else
        propose_candidate_spans(
            entity_logits,
            boundary_logits;
            max_candidate_spans = model.max_candidate_spans,
            max_span_width = model.max_span_width,
        )
    end

    relation_pairs, relation_mask = if hasproperty(inputs, :relation_pairs) && hasproperty(inputs, :relation_mask)
        (inputs.relation_pairs, inputs.relation_mask)
    else
        propose_relation_pairs(
            spans,
            span_mask,
            span_scores;
            max_candidate_pairs = model.max_candidate_pairs,
            neighbor_radius = model.pair_neighbor_radius,
        )
    end

    span_reps, span_state = build_span_representations(
        model, hidden, spans, span_mask, params, state
    )
    head_vectors, tail_vectors = gather_pair_span_vectors(span_reps, relation_pairs, relation_mask)
    pair_features = build_pair_features(head_vectors, tail_vectors)

    relation_logits_flat, relation_state = model.RelationHead(
        (head_vectors, tail_vectors),
        params.RelationHead,
        state.RelationHead,
    )
    confidence_flat, confidence_state = model.ConfidenceHead(pair_features, params.ConfidenceHead, state.ConfidenceHead)

    relation_logits = reshape(relation_logits_flat, model.num_relations, size(relation_pairs, 2), batch_size)
    confidence_logits = reshape(confidence_flat, 1, size(relation_pairs, 2), batch_size)

    new_state = (
        TokenEmbedding = encoder_state.TokenEmbedding,
        PositionEmbedding = encoder_state.PositionEmbedding,
        TimeEmbedding = encoder_state.TimeEmbedding,
        Blocks = encoder_state.Blocks,
        RefinementBlocks = encoder_state.RefinementBlocks,
        Dropout = dropout_state,
        EntityHead = entity_state,
        BoundaryHead = boundary_state,
        SpanProjection = span_state,
        RelationHead = relation_state,
        ConfidenceHead = confidence_state,
        position_indices = state.position_indices,
    )

    return (
        entity_logits = entity_logits,
        boundary_logits = boundary_logits,
        spans = spans,
        span_mask = span_mask,
        span_scores = span_scores,
        span_representations = span_reps,
        relation_pairs = relation_pairs,
        relation_mask = relation_mask,
        relation_logits = relation_logits,
        confidence_logits = confidence_logits,
    ), new_state
end

function entity_cross_entropy(logits, labels; ignore_index::Int = -100)
    num_labels = size(logits, 1)
    logits_flat = reshape(logits, num_labels, :)
    labels_flat = vec(Zygote.dropgrad(labels))
    valid_mask = labels_flat .!= ignore_index
    valid_count = Int(sum(valid_mask))
    valid_count == 0 && return 0.0f0
    log_probs = NNlib.logsoftmax(logits_flat, dims=1)
    safe_labels = clamp.(labels_flat, 1, num_labels)
    label_ids = reshape(collect(1:num_labels), :, 1)
    if logits_flat isa CUDA.CuArray
        label_ids = CUDA.CuArray(label_ids)
    end
    selected = sum(log_probs .* (label_ids .== reshape(safe_labels, 1, :)), dims = 1)
    weights = reshape(Float32.(valid_mask), 1, :)
    total = -sum(selected .* weights)
    return total / Float32(valid_count)
end

function boundary_bce(logits, targets; ignore_index::Int = -100)
    targets_const = Zygote.dropgrad(targets)
    valid_mask = targets_const .!= ignore_index
    count = Int(sum(valid_mask))
    count == 0 && return 0.0f0
    y = Float32.(targets_const)
    z = Float32.(logits)
    losses = NNlib.softplus.(z) .- z .* y
    return sum(losses .* Float32.(valid_mask)) / Float32(count)
end

function ChainRulesCore.rrule(::typeof(boundary_bce), logits, targets; ignore_index::Int = -100)
    targets_const = Zygote.dropgrad(targets)
    valid_mask = targets_const .!= ignore_index
    count = Int(sum(valid_mask))
    if count == 0
        function boundary_bce_zero_pullback(ȳ)
            logits_bar = ChainRulesCore.@thunk(zero(logits))
            return ChainRulesCore.NoTangent(), logits_bar, ChainRulesCore.NoTangent()
        end
        return 0.0f0, boundary_bce_zero_pullback
    end

    y = Float32.(targets_const)
    z = Float32.(logits)
    mask_values = Float32.(valid_mask)
    losses = NNlib.softplus.(z) .- z .* y
    value = sum(losses .* mask_values) / Float32(count)

    function boundary_bce_pullback(ȳ)
        scale = Float32(ȳ) / Float32(count)
        logits_bar = ChainRulesCore.@thunk(scale .* mask_values .* (NNlib.sigmoid.(z) .- y))
        return ChainRulesCore.NoTangent(), logits_bar, ChainRulesCore.NoTangent()
    end

    return value, boundary_bce_pullback
end

function relation_cross_entropy(logits, labels, mask; ignore_index::Int = -100, null_relation_weight::Float32 = 1.0f0)
    num_relations = size(logits, 1)
    logits_flat = reshape(logits, num_relations, :)
    labels_flat = vec(Zygote.dropgrad(labels))
    mask_flat = vec(Zygote.dropgrad(mask))
    valid_mask = mask_flat .& (labels_flat .!= ignore_index)
    valid_count = Int(sum(valid_mask))
    valid_count == 0 && return 0.0f0
    log_probs = NNlib.logsoftmax(logits_flat, dims=1)
    safe_labels = clamp.(labels_flat, 1, num_relations)
    label_ids = reshape(collect(1:num_relations), :, 1)
    if logits_flat isa CUDA.CuArray
        label_ids = CUDA.CuArray(label_ids)
    end
    selected = sum(log_probs .* (label_ids .== reshape(safe_labels, 1, :)), dims = 1)
    weights = Float32.(valid_mask) .* ifelse.(safe_labels .== 1, null_relation_weight, 1.0f0)
    total = -sum(vec(selected) .* weights)
    total_weight = sum(weights)
    return total_weight > 0 ? total / total_weight : 0.0f0
end

function confidence_bce(logits, targets, mask)
    logits_flat = Float32.(vec(logits))
    targets_flat = Float32.(vec(Zygote.dropgrad(targets)))
    mask_flat = vec(Zygote.dropgrad(mask))
    count = Int(sum(mask_flat))
    count == 0 && return 0.0f0
    losses = NNlib.softplus.(logits_flat) .- logits_flat .* targets_flat
    return sum(losses .* Float32.(mask_flat)) / Float32(count)
end

function ChainRulesCore.rrule(::typeof(confidence_bce), logits, targets, mask)
    logits_shape = size(logits)
    z = Float32.(vec(logits))
    y = Float32.(vec(Zygote.dropgrad(targets)))
    mask_const = vec(Zygote.dropgrad(mask))
    count = Int(sum(mask_const))
    if count == 0
        function confidence_bce_zero_pullback(ȳ)
            logits_bar = ChainRulesCore.@thunk(zero(logits))
            return ChainRulesCore.NoTangent(), logits_bar, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
        end
        return 0.0f0, confidence_bce_zero_pullback
    end

    mask_values = Float32.(mask_const)
    losses = NNlib.softplus.(z) .- z .* y
    value = sum(losses .* mask_values) / Float32(count)

    function confidence_bce_pullback(ȳ)
        scale = Float32(ȳ) / Float32(count)
        logits_bar = ChainRulesCore.@thunk(reshape(scale .* mask_values .* (NNlib.sigmoid.(z) .- y), logits_shape))
        return ChainRulesCore.NoTangent(), logits_bar, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end

    return value, confidence_bce_pullback
end

function load_rebel_jsonl(path::String)
    rows = Vector{Any}()
    open(path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            push!(rows, JSON3.read(line))
        end
    end
    return rows
end

function build_token_vocab(rows; max_vocab::Int = 64000, min_freq::Int = 1)
    counts = Dict{String, Int}()
    for row in rows
        for token in row.tokens
            token_str = String(token)
            counts[token_str] = get(counts, token_str, 0) + 1
        end
    end
    ordered = sort(collect(counts), by = x -> -x[2])
    vocab = Dict{String, Int}(
        "[PAD]" => 1,
        "[UNK]" => 2,
        "[CLS]" => 3,
        "[SEP]" => 4,
    )
    idx = 5
    for (token, count) in ordered
        if count < min_freq || idx > max_vocab
            continue
        end
        vocab[token] = idx
        idx += 1
    end
    return vocab
end

function build_entity_label_space(rows)
    labels = Set(["O"])
    for row in rows
        if !haskey(row, :entities)
            continue
        end
        for entity in row.entities
            label = uppercase(String(entity.label))
            push!(labels, "B-$label")
            push!(labels, "I-$label")
        end
    end
    ordered = ["O"; sort(collect(setdiff(labels, Set(["O"]))))...]
    return Dict(label => i for (i, label) in enumerate(ordered))
end

function build_relation_label_space(rows)
    labels = ["NO_RELATION"]
    seen = Set(labels)
    for row in rows
        if !haskey(row, :relations)
            continue
        end
        for rel in row.relations
            label = String(rel.label)
            if !(label in seen)
                push!(labels, label)
                push!(seen, label)
            end
        end
    end
    return Dict(label => i for (i, label) in enumerate(labels))
end

@inline function directed_pair_from_index(idx::Int, entity_count::Int)
    head_idx = fld(idx - 1, entity_count - 1) + 1
    tail_offset = mod(idx - 1, entity_count - 1) + 1
    tail_idx = tail_offset >= head_idx ? tail_offset + 1 : tail_offset
    return head_idx, tail_idx
end

function sample_negative_pairs!(
    relation_pairs,
    relation_labels,
    relation_mask,
    relation_targets,
    pair_idx::Int,
    entity_count::Int,
    positive_pairs::Set{Tuple{Int, Int}},
    no_relation_id::Int,
    target_negatives::Int,
)
    if entity_count < 2 || target_negatives <= 0
        return pair_idx
    end

    total_possible = entity_count * (entity_count - 1)
    start_idx = rand(1:total_possible)
    stride = total_possible > 1 ? total_possible - 1 : 1
    sampled_pairs = Set{Tuple{Int, Int}}()
    emitted = 0

    for step in 0:(total_possible - 1)
        emitted >= target_negatives && break
        candidate_idx = mod(start_idx - 1 + step * stride, total_possible) + 1
        pair = directed_pair_from_index(candidate_idx, entity_count)
        pair in positive_pairs && continue
        pair in sampled_pairs && continue
        push!(sampled_pairs, pair)
        pair_idx += 1
        relation_pairs[1, pair_idx] = pair[1]
        relation_pairs[2, pair_idx] = pair[2]
        relation_labels[pair_idx] = no_relation_id
        relation_targets[pair_idx] = 0.0f0
        relation_mask[pair_idx] = true
        emitted += 1
    end

    return pair_idx
end

function prepare_rebel_batch(
    rows,
    vocab::Dict{String, Int},
    entity_label_to_id::Dict{String, Int},
    relation_label_to_id::Dict{String, Int};
    max_len::Int,
    max_candidate_spans::Int,
    max_candidate_pairs::Int,
    hard_negative_ratio::Float32 = 0.0f0,
)
    batch_size = length(rows)
    token_ids = fill(vocab["[PAD]"], max_len, batch_size)
    entity_labels = fill(-100, max_len, batch_size)
    boundary_labels = fill(-100, 2, max_len, batch_size)
    spans = zeros(Int, 2, max_candidate_spans, batch_size)
    span_mask = falses(max_candidate_spans, batch_size)
    relation_pairs = zeros(Int, 2, max_candidate_pairs, batch_size)
    relation_labels = fill(-100, max_candidate_pairs, batch_size)
    relation_mask = falses(max_candidate_pairs, batch_size)
    relation_targets = zeros(Float32, max_candidate_pairs, batch_size)
    no_relation_id = get(relation_label_to_id, "NO_RELATION", 1)

    for (b, row) in enumerate(rows)
        tokens = [String(tok) for tok in row.tokens]
        seq_len = min(length(tokens), max_len)
        entity_labels[1:seq_len, b] .= entity_label_to_id["O"]
        boundary_labels[:, 1:seq_len, b] .= 0

        for i in 1:seq_len
            token_ids[i, b] = get(vocab, tokens[i], vocab["[UNK]"])
        end

        entities = haskey(row, :entities) ? collect(row.entities) : Any[]
        for (i, entity) in enumerate(entities)
            i > max_candidate_spans && break
            start_raw = Int(entity.start)
            stop_raw = entity_span_end(entity)
            offset = start_raw == 0 || stop_raw == 0 ? 1 : 0
            start_idx = clamp(start_raw + offset, 1, seq_len)
            stop_idx = clamp(stop_raw + offset, start_idx, seq_len)
            label = uppercase(String(entity.label))
            entity_labels[start_idx, b] = entity_label_to_id["B-$label"]
            for pos in (start_idx + 1):stop_idx
                entity_labels[pos, b] = entity_label_to_id["I-$label"]
            end
            boundary_labels[1, start_idx, b] = 1
            boundary_labels[2, stop_idx, b] = 1
            spans[1, i, b] = start_idx
            spans[2, i, b] = stop_idx
            span_mask[i, b] = true
        end

        relations = haskey(row, :relations) ? collect(row.relations) : Any[]
        positive_pairs = Set{Tuple{Int, Int}}()
        pair_idx = 0
        for rel in relations
            pair_idx >= max_candidate_pairs && break
            head_raw = Int(rel.head)
            tail_raw = Int(rel.tail)
            offset = head_raw == 0 || tail_raw == 0 ? 1 : 0
            head_idx = head_raw + offset
            tail_idx = tail_raw + offset
            if !(1 <= head_idx <= max_candidate_spans && 1 <= tail_idx <= max_candidate_spans)
                continue
            end
            if !(span_mask[head_idx, b] && span_mask[tail_idx, b]) || head_idx == tail_idx
                continue
            end
            pair = (head_idx, tail_idx)
            pair in positive_pairs && continue
            pair_idx += 1
            push!(positive_pairs, pair)
            relation_pairs[1, pair_idx, b] = head_idx
            relation_pairs[2, pair_idx, b] = tail_idx
            relation_labels[pair_idx, b] = get(relation_label_to_id, String(rel.label), no_relation_id)
            relation_targets[pair_idx, b] = 1.0f0
            relation_mask[pair_idx, b] = true
        end

        entity_count = sum(span_mask[:, b])
        if entity_count >= 2 && pair_idx < max_candidate_pairs && hard_negative_ratio > 0.0f0
            available_negatives = max(entity_count * (entity_count - 1) - length(positive_pairs), 0)
            target_negatives = if !isempty(positive_pairs)
                ceil(Int, length(positive_pairs) * hard_negative_ratio)
            else
                min(available_negatives, max(1, round(Int, hard_negative_ratio)))
            end
            target_negatives = min(target_negatives, available_negatives, max_candidate_pairs - pair_idx)

            pair_idx = sample_negative_pairs!(
                @view(relation_pairs[:, :, b]),
                @view(relation_labels[:, b]),
                @view(relation_mask[:, b]),
                @view(relation_targets[:, b]),
                pair_idx,
                entity_count,
                positive_pairs,
                no_relation_id,
                target_negatives,
            )
        end
    end

    return (
        token_ids = token_ids,
        entity_labels = entity_labels,
        boundary_labels = boundary_labels,
        spans = spans,
        span_mask = span_mask,
        relation_pairs = relation_pairs,
        relation_labels = relation_labels,
        relation_mask = relation_mask,
        relation_targets = relation_targets,
    )
end

export RelationExtractionConfig, SwammaRelationExtractor
export load_relation_extraction_config, print_relation_extraction_summary
export entity_cross_entropy, boundary_bce, relation_cross_entropy, confidence_bce
export load_rebel_jsonl, build_token_vocab, build_entity_label_space, build_relation_label_space
export prepare_rebel_batch, DEFAULT_ENTITY_LABELS, DEFAULT_ENTITY_TYPES

end # module RelationExtraction
