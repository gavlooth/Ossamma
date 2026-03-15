module RelationExtraction

using Lux
using Random
using TOML
using JSON3
using NNlib
using Statistics: mean, std
import CUDA
import ChainRulesCore

import ..Swamma: SwammaBlock, SwammaBlockConfig, LocalWaveRefinementBlock, LinearLocalRefinementBlock, LuxLayer, sinkhorn_bistochastic

detach_constant(x) = ChainRulesCore.ignore_derivatives() do
    x
end

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
    use_interleaved_local_wave::Bool = false
    interleaved_block_type::Symbol = :local_wave
    local_wave_ratio::Int = 3
    interleaved_use_local_attention::Bool = true
    interleaved_use_wave_pde::Bool = true
    num_entity_labels::Int = length(DEFAULT_ENTITY_LABELS)
    num_relations::Int = 64
    time_dimension::Int = 192
    state_dimension::Int = -1
    window_size::Int = 24
    local_operator::Symbol = :swattention
    residual_mode::Symbol = :plain
    hyper_connection_width::Int = 2
    hyper_connection_sinkhorn_iterations::Int = 4
    min_frequency::Float32 = 0.01f0
    max_frequency::Float32 = 5.0f0
    default_time_step::Float32 = 0.05f0
    dropout_rate::Float32 = 0.1f0
    use_ffn::Bool = true
    ffn_expansion::Float32 = 4f0 / 3f0
    use_output_projection::Bool = false
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
    pair_proposer_mode::Symbol = :local
    pair_global_top_spans::Int = 0
    pair_router_dimension::Int = 64
    pair_router_buckets::Int = 8
    pair_router_topk::Int = 6
    pair_router_routes_per_span::Int = 2
    pair_router_score_scale::Float32 = 0.5f0
    pair_overgenerate_factor::Int = 2
    pair_retrieval_dimension::Int = 64
    pair_distance_buckets::Int = 16
    pair_retrieval_loss_weight::Float32 = 1.0f0
    pair_evidence_dimension::Int = 64
    relation_decoder_mode::Symbol = :biaffine
    relation_decoder_residual_scale::Float32 = 0.25f0
    mention_score_mode::Symbol = :hybrid
    mention_score_learned_weight::Float32 = 0.25f0
    span_context_layers::Int = 0
    span_context_neighbor_radius::Int = 1
    span_context_topk::Int = 4
end

struct PairRetrievalHead{HP,TP,DE,FP,OP} <: LuxLayer
    embedding_dimension::Int
    retrieval_dimension::Int
    distance_buckets::Int
    HeadProjection::HP
    TailProjection::TP
    DistanceEmbedding::DE
    FeatureProjection::FP
    OutputProjection::OP
end

struct PairEvidenceSelectorHead{QP,KP,VP,OP} <: LuxLayer
    embedding_dimension::Int
    evidence_dimension::Int
    QueryProjection::QP
    KeyProjection::KP
    ValueProjection::VP
    OutputProjection::OP
end

function PairEvidenceSelectorHead(
    embedding_dimension::Int;
    evidence_dimension::Int = min(64, embedding_dimension),
)
    return PairEvidenceSelectorHead(
        embedding_dimension,
        evidence_dimension,
        Lux.Dense(4 * embedding_dimension => evidence_dimension; use_bias = false),
        Lux.Dense(embedding_dimension => evidence_dimension; use_bias = false),
        Lux.Dense(embedding_dimension => embedding_dimension; use_bias = false),
        Lux.Chain(
            Lux.LayerNorm((5 * embedding_dimension,)),
            Lux.Dense(5 * embedding_dimension => embedding_dimension, gelu; use_bias = false),
        ),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::PairEvidenceSelectorHead)
    return (
        QueryProjection = Lux.initialparameters(rng, layer.QueryProjection),
        KeyProjection = Lux.initialparameters(rng, layer.KeyProjection),
        ValueProjection = Lux.initialparameters(rng, layer.ValueProjection),
        OutputProjection = Lux.initialparameters(rng, layer.OutputProjection),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::PairEvidenceSelectorHead)
    return (
        QueryProjection = Lux.initialstates(rng, layer.QueryProjection),
        KeyProjection = Lux.initialstates(rng, layer.KeyProjection),
        ValueProjection = Lux.initialstates(rng, layer.ValueProjection),
        OutputProjection = Lux.initialstates(rng, layer.OutputProjection),
    )
end

function (layer::PairEvidenceSelectorHead)(inputs::Tuple, params, state)
    if length(inputs) == 4
        pair_features, token_states, token_mask, pair_mask = inputs
        emit_diagnostics = false
        evidence_pooling_mode = :token
    elseif length(inputs) == 5
        pair_features, token_states, token_mask, pair_mask, emit_diagnostics = inputs
        evidence_pooling_mode = :token
    elseif length(inputs) == 6
        pair_features, token_states, token_mask, pair_mask, emit_diagnostics, evidence_pooling_mode = inputs
    else
        throw(ArgumentError("PairEvidenceSelectorHead expects 4, 5, or 6 inputs, got $(length(inputs))."))
    end
    emit_diagnostics = Bool(emit_diagnostics)
    evidence_pooling_mode = evidence_pooling_mode isa Symbol ? evidence_pooling_mode : Symbol(evidence_pooling_mode)
    evidence_pooling_mode in (:token, :sentence, :hybrid) ||
        throw(ArgumentError("Unsupported evidence_pooling_mode=$(repr(evidence_pooling_mode)); expected :token, :sentence, or :hybrid."))
    pair_feature_dim, max_pairs, batch_size = size(pair_features)
    d, seq_len, _ = size(token_states)
    evidence_dim = layer.evidence_dimension
    on_gpu = token_states isa CUDA.CuArray

    pair_flat = reshape(pair_features, pair_feature_dim, :)
    query_flat, query_state = layer.QueryProjection(pair_flat, params.QueryProjection, state.QueryProjection)
    key_flat, key_state = layer.KeyProjection(reshape(token_states, d, :), params.KeyProjection, state.KeyProjection)
    value_flat, value_state = layer.ValueProjection(reshape(token_states, d, :), params.ValueProjection, state.ValueProjection)

    queries = reshape(query_flat, evidence_dim, max_pairs, batch_size)
    keys = reshape(key_flat, evidence_dim, seq_len, batch_size)
    values = reshape(value_flat, d, seq_len, batch_size)

    large_negative = oftype(zero(eltype(token_states)), -1.0f4)
    token_evidence_batches = map(1:batch_size) do b
        raw_scores = transpose(@view(queries[:, :, b])) * @view(keys[:, :, b]) ./ sqrt(Float32(evidence_dim))
        masked_scores = ifelse.(reshape(@view(token_mask[:, b]), 1, :), raw_scores, large_negative)
        attention_weights = NNlib.softmax(masked_scores, dims = 2)
        summary = @view(values[:, :, b]) * transpose(attention_weights)
        summary .* reshape(Float32.(@view(pair_mask[:, b])), 1, :)
    end
    token_evidence_summary = cat(token_evidence_batches...; dims = 3)

    sentence_evidence_batches = map(1:batch_size) do b
        mask_weights = Float32.(@view(token_mask[:, b]))
        denom = max(sum(mask_weights), 1f-6)
        mean_vec = @view(values[:, :, b]) * reshape(mask_weights ./ denom, :, 1)
        repeat(mean_vec, 1, max_pairs) .* reshape(Float32.(@view(pair_mask[:, b])), 1, :)
    end
    sentence_evidence_summary = cat(sentence_evidence_batches...; dims = 3)

    evidence_summary = if evidence_pooling_mode == :token
        token_evidence_summary
    elseif evidence_pooling_mode == :sentence
        sentence_evidence_summary
    else
        0.5f0 .* (token_evidence_summary .+ sentence_evidence_summary)
    end

    evidence_top_token_index = nothing
    evidence_attention_entropy = nothing
    evidence_attention_max_weight = nothing
    if emit_diagnostics
        top_index_batches = Vector{Vector{Int32}}(undef, batch_size)
        entropy_batches = Vector{Vector{Float32}}(undef, batch_size)
        max_weight_batches = Vector{Vector{Float32}}(undef, batch_size)
        for b in 1:batch_size
            raw_scores = transpose(@view(queries[:, :, b])) * @view(keys[:, :, b]) ./ sqrt(Float32(evidence_dim))
            masked_scores = ifelse.(reshape(@view(token_mask[:, b]), 1, :), raw_scores, large_negative)
            attention_weights = NNlib.softmax(masked_scores, dims = 2)
            token_mask_cpu = ChainRulesCore.ignore_derivatives() do
                Bool.(Array(@view(token_mask[:, b])))
            end
            pair_mask_cpu = ChainRulesCore.ignore_derivatives() do
                Bool.(Array(@view(pair_mask[:, b])))
            end
            attention_cpu = ChainRulesCore.ignore_derivatives() do
                Float32.(Array(attention_weights))
            end
            top_idx = zeros(Int32, max_pairs)
            entropy = zeros(Float32, max_pairs)
            max_weight = zeros(Float32, max_pairs)
            for pair_idx in 1:max_pairs
                pair_mask_cpu[pair_idx] || continue
                row = @view(attention_cpu[pair_idx, :])
                masked_row = similar(row)
                for token_idx in 1:seq_len
                    masked_row[token_idx] = token_mask_cpu[token_idx] ? row[token_idx] : 0.0f0
                end
                score_sum = sum(masked_row)
                if score_sum > 0
                    masked_row ./= score_sum
                end
                top_idx[pair_idx] = Int32(argmax(masked_row))
                max_weight[pair_idx] = maximum(masked_row)
                entropy[pair_idx] = -sum(masked_row .* log.(masked_row .+ 1f-8))
            end
            top_index_batches[b] = top_idx
            entropy_batches[b] = entropy
            max_weight_batches[b] = max_weight
        end
        evidence_top_token_index = hcat(top_index_batches...)
        evidence_attention_entropy = hcat(entropy_batches...)
        evidence_attention_max_weight = hcat(max_weight_batches...)
        if on_gpu
            evidence_top_token_index = CUDA.CuArray(evidence_top_token_index)
            evidence_attention_entropy = CUDA.CuArray(evidence_attention_entropy)
            evidence_attention_max_weight = CUDA.CuArray(evidence_attention_max_weight)
        end
    end

    fused_inputs = vcat(
        reshape(pair_features, pair_feature_dim, :),
        reshape(evidence_summary, d, :),
    )
    projected, output_state = layer.OutputProjection(fused_inputs, params.OutputProjection, state.OutputProjection)

    new_state = (
        QueryProjection = query_state,
        KeyProjection = key_state,
        ValueProjection = value_state,
        OutputProjection = output_state,
    )
    return (
        summary = reshape(projected, d, max_pairs, batch_size),
        top_token_index = evidence_top_token_index,
        attention_entropy = evidence_attention_entropy,
        attention_max_weight = evidence_attention_max_weight,
    ), new_state
end

function PairRetrievalHead(
    embedding_dimension::Int;
    retrieval_dimension::Int = min(64, embedding_dimension),
    distance_buckets::Int = 16,
)
    return PairRetrievalHead(
        embedding_dimension,
        retrieval_dimension,
        distance_buckets,
        Lux.Dense(embedding_dimension => retrieval_dimension; use_bias = false),
        Lux.Dense(embedding_dimension => retrieval_dimension; use_bias = false),
        Lux.Embedding(distance_buckets => retrieval_dimension),
        Lux.Dense(4 * embedding_dimension => retrieval_dimension, gelu; use_bias = false),
        Lux.Dense(4 * retrieval_dimension + 2 => 1; use_bias = false),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::PairRetrievalHead)
    return (
        HeadProjection = Lux.initialparameters(rng, layer.HeadProjection),
        TailProjection = Lux.initialparameters(rng, layer.TailProjection),
        DistanceEmbedding = Lux.initialparameters(rng, layer.DistanceEmbedding),
        FeatureProjection = Lux.initialparameters(rng, layer.FeatureProjection),
        OutputProjection = Lux.initialparameters(rng, layer.OutputProjection),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::PairRetrievalHead)
    return (
        HeadProjection = Lux.initialstates(rng, layer.HeadProjection),
        TailProjection = Lux.initialstates(rng, layer.TailProjection),
        DistanceEmbedding = Lux.initialstates(rng, layer.DistanceEmbedding),
        FeatureProjection = Lux.initialstates(rng, layer.FeatureProjection),
        OutputProjection = Lux.initialstates(rng, layer.OutputProjection),
    )
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

struct FusedRelationDecoderHead{BH,PP,GP} <: LuxLayer
    num_relations::Int
    residual_scale::Float32
    BiaffineHead::BH
    PairProjection::PP
    GateProjection::GP
end

struct EvidenceAwareRelationDecoderHead{BH,PP,GP} <: LuxLayer
    num_relations::Int
    residual_scale::Float32
    BiaffineHead::BH
    PairProjection::PP
    GateProjection::GP
end

function EvidenceAwareRelationDecoderHead(
    embedding_dimension::Int,
    num_relations::Int;
    rank::Int = min(64, embedding_dimension),
    residual_scale::Float32 = 0.25f0,
)
    hidden_dim = max(embedding_dimension, 1)
    return EvidenceAwareRelationDecoderHead(
        num_relations,
        residual_scale,
        LowRankBiaffineRelationHead(
            embedding_dimension,
            num_relations;
            rank = rank,
        ),
        Lux.Chain(
            Lux.LayerNorm((5 * embedding_dimension + 1,)),
            Lux.Dense(5 * embedding_dimension + 1 => hidden_dim, gelu; use_bias = false),
            Lux.Dense(hidden_dim => num_relations; use_bias = false),
        ),
        Lux.Chain(
            Lux.LayerNorm((5 * embedding_dimension + 1,)),
            Lux.Dense(5 * embedding_dimension + 1 => max(hidden_dim ÷ 2, 1), gelu; use_bias = false),
            Lux.Dense(max(hidden_dim ÷ 2, 1) => 1),
        ),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::EvidenceAwareRelationDecoderHead)
    return (
        BiaffineHead = Lux.initialparameters(rng, layer.BiaffineHead),
        PairProjection = Lux.initialparameters(rng, layer.PairProjection),
        GateProjection = Lux.initialparameters(rng, layer.GateProjection),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::EvidenceAwareRelationDecoderHead)
    return (
        BiaffineHead = Lux.initialstates(rng, layer.BiaffineHead),
        PairProjection = Lux.initialstates(rng, layer.PairProjection),
        GateProjection = Lux.initialstates(rng, layer.GateProjection),
    )
end

function (layer::EvidenceAwareRelationDecoderHead)(inputs::Tuple, params, state)
    head_vectors, tail_vectors, pair_features, evidence_summary, retrieval_logits = inputs

    biaffine_logits, biaffine_state = layer.BiaffineHead(
        (head_vectors, tail_vectors),
        params.BiaffineHead,
        state.BiaffineHead,
    )
    decoder_inputs = vcat(pair_features, evidence_summary, retrieval_logits)
    pair_logits, pair_state = layer.PairProjection(
        decoder_inputs,
        params.PairProjection,
        state.PairProjection,
    )
    gate_logits, gate_state = layer.GateProjection(
        decoder_inputs,
        params.GateProjection,
        state.GateProjection,
    )
    gate = NNlib.sigmoid.(gate_logits)
    fused_logits = biaffine_logits .+ layer.residual_scale .* gate .* pair_logits

    new_state = (
        BiaffineHead = biaffine_state,
        PairProjection = pair_state,
        GateProjection = gate_state,
    )
    return fused_logits, new_state
end

function FusedRelationDecoderHead(
    embedding_dimension::Int,
    num_relations::Int;
    rank::Int = min(64, embedding_dimension),
    residual_scale::Float32 = 0.25f0,
)
    hidden_dim = max(embedding_dimension, 1)
    return FusedRelationDecoderHead(
        num_relations,
        residual_scale,
        LowRankBiaffineRelationHead(
            embedding_dimension,
            num_relations;
            rank = rank,
        ),
        Lux.Chain(
            Lux.LayerNorm((4 * embedding_dimension + 1,)),
            Lux.Dense(4 * embedding_dimension + 1 => hidden_dim, gelu; use_bias = false),
            Lux.Dense(hidden_dim => num_relations; use_bias = false),
        ),
        Lux.Chain(
            Lux.LayerNorm((4 * embedding_dimension + 1,)),
            Lux.Dense(4 * embedding_dimension + 1 => max(hidden_dim ÷ 2, 1), gelu; use_bias = false),
            Lux.Dense(max(hidden_dim ÷ 2, 1) => 1),
        ),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::FusedRelationDecoderHead)
    return (
        BiaffineHead = Lux.initialparameters(rng, layer.BiaffineHead),
        PairProjection = Lux.initialparameters(rng, layer.PairProjection),
        GateProjection = Lux.initialparameters(rng, layer.GateProjection),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::FusedRelationDecoderHead)
    return (
        BiaffineHead = Lux.initialstates(rng, layer.BiaffineHead),
        PairProjection = Lux.initialstates(rng, layer.PairProjection),
        GateProjection = Lux.initialstates(rng, layer.GateProjection),
    )
end

function (layer::FusedRelationDecoderHead)(inputs::Tuple, params, state)
    head_vectors, tail_vectors, pair_features, retrieval_logits = inputs

    biaffine_logits, biaffine_state = layer.BiaffineHead(
        (head_vectors, tail_vectors),
        params.BiaffineHead,
        state.BiaffineHead,
    )
    pair_logits, pair_state = layer.PairProjection(
        vcat(pair_features, retrieval_logits),
        params.PairProjection,
        state.PairProjection,
    )
    gate_logits, gate_state = layer.GateProjection(
        vcat(pair_features, retrieval_logits),
        params.GateProjection,
        state.GateProjection,
    )
    gate = NNlib.sigmoid.(gate_logits)
    fused_logits = biaffine_logits .+ layer.residual_scale .* gate .* pair_logits

    new_state = (
        BiaffineHead = biaffine_state,
        PairProjection = pair_state,
        GateProjection = gate_state,
    )
    return fused_logits, new_state
end

struct SparsePairProposalHead{HP,TP,BP} <: LuxLayer
    embedding_dimension::Int
    router_dimension::Int
    bucket_count::Int
    HeadProjection::HP
    TailProjection::TP
    BucketProjection::BP
end

function SparsePairProposalHead(
    embedding_dimension::Int;
    router_dimension::Int = min(64, embedding_dimension),
    bucket_count::Int = 8,
)
    return SparsePairProposalHead(
        embedding_dimension,
        router_dimension,
        bucket_count,
        Lux.Dense(embedding_dimension => router_dimension; use_bias = false),
        Lux.Dense(embedding_dimension => router_dimension; use_bias = false),
        Lux.Dense(embedding_dimension => bucket_count),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::SparsePairProposalHead)
    return (
        HeadProjection = Lux.initialparameters(rng, layer.HeadProjection),
        TailProjection = Lux.initialparameters(rng, layer.TailProjection),
        BucketProjection = Lux.initialparameters(rng, layer.BucketProjection),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::SparsePairProposalHead)
    return (
        HeadProjection = Lux.initialstates(rng, layer.HeadProjection),
        TailProjection = Lux.initialstates(rng, layer.TailProjection),
        BucketProjection = Lux.initialstates(rng, layer.BucketProjection),
    )
end

function (layer::SparsePairProposalHead)(span_vectors, params, state)
    head_router, head_state = layer.HeadProjection(span_vectors, params.HeadProjection, state.HeadProjection)
    tail_router, tail_state = layer.TailProjection(span_vectors, params.TailProjection, state.TailProjection)
    bucket_logits, bucket_state = layer.BucketProjection(span_vectors, params.BucketProjection, state.BucketProjection)

    new_state = (
        HeadProjection = head_state,
        TailProjection = tail_state,
        BucketProjection = bucket_state,
    )

    return (
        head_router = head_router,
        tail_router = tail_router,
        bucket_logits = bucket_logits,
    ), new_state
end

struct SparseSpanContextBlock{IN,QP,KP,VP,OP,FF} <: LuxLayer
    embedding_dimension::Int
    neighbor_radius::Int
    semantic_topk::Int
    InputNorm::IN
    QueryProjection::QP
    KeyProjection::KP
    ValueProjection::VP
    OutputProjection::OP
    FeedForward::FF
end

function SparseSpanContextBlock(
    embedding_dimension::Int;
    neighbor_radius::Int = 1,
    semantic_topk::Int = 4,
)
    ff_hidden = max(embedding_dimension * 2, 1)
    return SparseSpanContextBlock(
        embedding_dimension,
        neighbor_radius,
        semantic_topk,
        Lux.LayerNorm((embedding_dimension,)),
        Lux.Dense(embedding_dimension => embedding_dimension; use_bias = false),
        Lux.Dense(embedding_dimension => embedding_dimension; use_bias = false),
        Lux.Dense(embedding_dimension => embedding_dimension; use_bias = false),
        Lux.Dense(embedding_dimension => embedding_dimension; use_bias = false),
        Lux.Chain(
            Lux.LayerNorm((embedding_dimension,)),
            Lux.Dense(embedding_dimension => ff_hidden, gelu; use_bias = false),
            Lux.Dense(ff_hidden => embedding_dimension; use_bias = false),
        ),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::SparseSpanContextBlock)
    return (
        InputNorm = Lux.initialparameters(rng, layer.InputNorm),
        QueryProjection = Lux.initialparameters(rng, layer.QueryProjection),
        KeyProjection = Lux.initialparameters(rng, layer.KeyProjection),
        ValueProjection = Lux.initialparameters(rng, layer.ValueProjection),
        OutputProjection = Lux.initialparameters(rng, layer.OutputProjection),
        FeedForward = Lux.initialparameters(rng, layer.FeedForward),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::SparseSpanContextBlock)
    return (
        InputNorm = Lux.initialstates(rng, layer.InputNorm),
        QueryProjection = Lux.initialstates(rng, layer.QueryProjection),
        KeyProjection = Lux.initialstates(rng, layer.KeyProjection),
        ValueProjection = Lux.initialstates(rng, layer.ValueProjection),
        OutputProjection = Lux.initialstates(rng, layer.OutputProjection),
        FeedForward = Lux.initialstates(rng, layer.FeedForward),
    )
end

struct SwammaRelationExtractor{E,P,T,B,HC,RB,D,EH,BH,SP,MH,SC,PP,PR,PE,RH,CH} <: LuxLayer
    vocab_size::Int
    max_sequence_length::Int
    embedding_dimension::Int
    number_of_layers::Int
    number_of_refinement_layers::Int
    use_interleaved_local_wave::Bool
    interleaved_block_type::Symbol
    local_wave_ratio::Int
    interleaved_use_local_attention::Bool
    interleaved_use_wave_pde::Bool
    num_entity_labels::Int
    num_relations::Int
    residual_mode::Symbol
    hyper_connection_width::Int
    max_candidate_spans::Int
    max_candidate_pairs::Int
    max_span_width::Int
    span_context_layers::Int
    pair_neighbor_radius::Int
    pair_proposer_mode::Symbol
    pair_global_top_spans::Int
    pair_anchor_top_spans::Int
    pair_router_buckets::Int
    pair_router_topk::Int
    pair_router_routes_per_span::Int
    pair_router_score_scale::Float32
    pair_overgenerate_factor::Int
    pair_retrieval_dimension::Int
    pair_distance_buckets::Int
    pair_retrieval_loss_weight::Float32
    pair_evidence_dimension::Int
    relation_decoder_mode::Symbol
    relation_decoder_residual_scale::Float32
    mention_score_mode::Symbol
    mention_score_learned_weight::Float32
    TokenEmbedding::E
    PositionEmbedding::P
    TimeEmbedding::T
    Blocks::B
    HyperConnections::HC
    RefinementBlocks::RB
    Dropout::D
    EntityHead::EH
    BoundaryHead::BH
    SpanProjection::SP
    MentionHead::MH
    SpanContextBlocks::SC
    PairProposalHead::PP
    PairRetrievalHead::PR
    PairEvidenceHead::PE
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
        use_interleaved_local_wave = get(refinement, "interleave", false),
        interleaved_block_type = begin
            raw = get(refinement, "block_type", "local_wave")
            raw isa Symbol ? raw : Symbol(raw)
        end,
        local_wave_ratio = get(refinement, "swamma_ratio", 3),
        interleaved_use_local_attention = get(refinement, "use_local_attention", true),
        interleaved_use_wave_pde = get(refinement, "use_wave_pde", true),
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
        hyper_connection_width = get(residual, "hyper_width", 2),
        hyper_connection_sinkhorn_iterations = get(residual, "sinkhorn_iterations", 4),
        min_frequency = Float32(get(osc, "min_frequency", 0.01)),
        max_frequency = Float32(get(osc, "max_frequency", 5.0)),
        default_time_step = Float32(get(osc, "default_time_step", 0.05)),
        dropout_rate = Float32(get(reg, "dropout_rate", 0.1)),
        use_ffn = get(ablation, "use_ffn", true),
        ffn_expansion = Float32(get(ablation, "ffn_expansion", 4.0 / 3.0)),
        use_output_projection = get(ablation, "use_output_projection", false),
        use_parallel_scan = get(parallel, "use_parallel_scan", false),
        parallel_chunk_size = get(parallel, "chunk_size", 64),
        use_vector_gains = get(ablation, "use_vector_gains", false),
        use_per_head_alpha = get(ablation, "use_per_head_alpha", false),
        use_branch_projections = get(ablation, "use_branch_projections", false),
        max_candidate_spans = get(relation, "max_candidate_spans", 64),
        max_candidate_pairs = get(relation, "max_candidate_pairs", 256),
        max_span_width = get(relation, "max_span_width", 8),
        span_context_layers = get(relation, "span_context_layers", 0),
        span_context_neighbor_radius = get(relation, "span_context_neighbor_radius", 1),
        span_context_topk = get(relation, "span_context_topk", 4),
        biaffine_rank = get(relation, "biaffine_rank", 64),
        pair_neighbor_radius = get(relation, "pair_neighbor_radius", 4),
        pair_proposer_mode = begin
            raw = get(relation, "pair_proposer_mode", "local")
            raw isa Symbol ? raw : Symbol(raw)
        end,
        pair_global_top_spans = get(relation, "pair_global_top_spans", 0),
        pair_router_dimension = get(relation, "pair_router_dimension", 64),
        pair_router_buckets = get(relation, "pair_router_buckets", 8),
        pair_router_topk = get(relation, "pair_router_topk", 6),
        pair_router_routes_per_span = get(relation, "pair_router_routes_per_span", 2),
        pair_router_score_scale = Float32(get(relation, "pair_router_score_scale", 0.5)),
        pair_overgenerate_factor = get(relation, "pair_overgenerate_factor", 2),
        pair_retrieval_dimension = get(relation, "pair_retrieval_dimension", 64),
        pair_distance_buckets = get(relation, "pair_distance_buckets", 16),
        pair_retrieval_loss_weight = Float32(get(relation, "pair_retrieval_loss_weight", 1.0)),
        pair_evidence_dimension = get(relation, "pair_evidence_dimension", 64),
        relation_decoder_mode = begin
            raw = get(relation, "relation_decoder_mode", "biaffine")
            raw isa Symbol ? raw : Symbol(raw)
        end,
        relation_decoder_residual_scale = Float32(get(relation, "relation_decoder_residual_scale", 0.25)),
        mention_score_mode = begin
            raw = get(relation, "mention_score_mode", "hybrid")
            raw isa Symbol ? raw : Symbol(raw)
        end,
        mention_score_learned_weight = Float32(get(relation, "mention_score_learned_weight", 0.25)),
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

function interleaved_local_wave_positions(total_layers::Int, swamma_ratio::Int)
    total_layers > 0 || return Int[]
    swamma_ratio > 0 || throw(ArgumentError("swamma_ratio must be positive."))

    stride = swamma_ratio + 1
    positions = collect(stride:stride:total_layers)
    if !isempty(positions) && positions[end] == total_layers
        positions[end] = max(total_layers - 1, 1)
    end
    positions = unique(sort(filter(pos -> 1 <= pos < total_layers, positions)))
    return positions
end

function initial_hyper_connection_logits(width::Int)
    logits = fill(-2.0f0, width, width)
    for i in 1:width
        logits[i, i] = 2.0f0
    end
    return logits
end

function apply_manifold_hyper_connection(primary, skip, logits, sinkhorn_iterations::Int)
    mix = sinkhorn_bistochastic(logits, sinkhorn_iterations)
    stack_dim = ndims(primary) + 1
    stream_tensor = cat(primary, skip; dims = stack_dim)
    mixed_shape = size(stream_tensor)
    stream_matrix = reshape(stream_tensor, :, 2)
    mixed_matrix = stream_matrix * transpose(mix)
    mixed_tensor = reshape(mixed_matrix, mixed_shape)
    return copy(selectdim(mixed_tensor, stack_dim, 1)), copy(selectdim(mixed_tensor, stack_dim, 2))
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
    println("  interleave_local_wave: $(config.use_interleaved_local_wave)")
    if config.use_interleaved_local_wave
        positions = interleaved_local_wave_positions(config.number_of_layers, config.local_wave_ratio)
        position_text = isempty(positions) ? "none" : join(positions, ", ")
        println("  local_wave_ratio:     $(config.local_wave_ratio):1")
        println("  local_wave_positions: $position_text")
        println("  interleaved_block:    $(config.interleaved_block_type)")
        println("  local_wave_local_attn: $(config.interleaved_use_local_attention)")
        println("  local_wave_wave_pde:  $(config.interleaved_use_wave_pde)")
    end
    println("  num_entity_labels:    $(config.num_entity_labels)")
    println("  num_relations:        $(config.num_relations)")
    println("  window_size:          $(config.window_size)")
    println("  local_operator:       $(config.local_operator)")
    println("  residual_mode:        $(config.residual_mode)")
    if config.residual_mode == :mhc
        println("  hyper_width:          $(config.hyper_connection_width)")
        println("  sinkhorn_iters:       $(config.hyper_connection_sinkhorn_iterations)")
    end
    println("Heads:")
    println("  max_candidate_spans:  $(config.max_candidate_spans)")
    println("  max_candidate_pairs:  $(config.max_candidate_pairs)")
    println("  max_span_width:       $(config.max_span_width)")
    println("  span_context_layers:  $(config.span_context_layers)")
    if config.span_context_layers > 0
        println("  span_ctx_radius:      $(config.span_context_neighbor_radius)")
        println("  span_ctx_topk:        $(config.span_context_topk)")
    end
    println("  biaffine_rank:        $(config.biaffine_rank)")
    println("  pair_neighbor_radius: $(config.pair_neighbor_radius)")
    println("  pair_proposer_mode:   $(config.pair_proposer_mode)")
    println("  pair_global_top_spans: $(config.pair_global_top_spans)")
    if config.pair_proposer_mode in (:sparse, :sparse_hybrid, :edge_retrieval_v2)
        println("  pair_router_dimension: $(config.pair_router_dimension)")
        println("  pair_router_buckets:  $(config.pair_router_buckets)")
        println("  pair_router_topk:     $(config.pair_router_topk)")
        println("  pair_router_routes:   $(config.pair_router_routes_per_span)")
        println("  pair_router_scale:    $(config.pair_router_score_scale)")
        println("  pair_overgen_factor:  $(config.pair_overgenerate_factor)")
        println("  pair_retrieval_dim:   $(config.pair_retrieval_dimension)")
        println("  pair_dist_buckets:    $(config.pair_distance_buckets)")
        println("  pair_retrieval_wt:    $(config.pair_retrieval_loss_weight)")
    end
    println("  pair_evidence_dim:    $(config.pair_evidence_dimension)")
    println("  relation_decoder:     $(config.relation_decoder_mode)")
    if config.relation_decoder_mode == :fused_residual
        println("  relation_residual:    $(config.relation_decoder_residual_scale)")
    elseif config.relation_decoder_mode == :fused_evidence
        println("  relation_residual:    $(config.relation_decoder_residual_scale)")
    end
    println("  mention_score_mode:   $(config.mention_score_mode)")
    println("  mention_score_weight: $(config.mention_score_learned_weight)")
    println("=" ^ 60)
end

function SwammaRelationExtractor(config::RelationExtractionConfig)
    state_dimension = config.state_dimension == -1 ? config.embedding_dimension : config.state_dimension
    config.residual_mode in (:plain, :mhc) || throw(ArgumentError(
        "Unsupported residual_mode=$(repr(config.residual_mode)). Expected :plain or :mhc."
    ))
    if config.residual_mode == :mhc && config.hyper_connection_width != 2
        throw(ArgumentError("This mHC implementation currently supports hyper_connection_width = 2."))
    end

    block_config = SwammaBlockConfig(
        embedding_dimension = config.embedding_dimension,
        sequence_length = config.max_sequence_length,
        number_of_heads = config.number_of_heads,
        time_dimension = config.time_dimension,
        state_dimension = state_dimension,
        window_size = config.window_size,
        local_operator = config.local_operator,
        residual_mode = :plain,
        min_frequency = config.min_frequency,
        max_frequency = config.max_frequency,
        default_time_step = config.default_time_step,
        dropout_rate = config.dropout_rate,
        use_ffn = config.use_ffn,
        ffn_expansion = config.ffn_expansion,
        use_output_projection = config.use_output_projection,
        use_parallel_scan = config.use_parallel_scan,
        parallel_chunk_size = config.parallel_chunk_size,
        use_vector_gains = config.use_vector_gains,
        use_per_head_alpha = config.use_per_head_alpha,
        use_branch_projections = config.use_branch_projections,
    )
    local_wave_positions = config.use_interleaved_local_wave ?
        Set(interleaved_local_wave_positions(config.number_of_layers, config.local_wave_ratio)) :
        Set{Int}()
    blocks = Tuple([
        if i in local_wave_positions
            if config.interleaved_block_type == :linear_window
                LinearLocalRefinementBlock(
                    config.embedding_dimension,
                    config.max_sequence_length,
                    config.number_of_heads,
                    config.time_dimension;
                    window_size = config.window_size,
                    dropout_rate = config.dropout_rate,
                )
            elseif config.interleaved_block_type == :local_wave
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
                    use_local_attention = config.interleaved_use_local_attention,
                    use_wave_pde = config.interleaved_use_wave_pde,
                )
            else
                throw(ArgumentError("Unsupported interleaved_block_type=$(repr(config.interleaved_block_type))."))
            end
        else
            SwammaBlock(block_config)
        end
        for i in 1:config.number_of_layers
    ])
    hyper_connections = ntuple(
        _ -> config.hyper_connection_sinkhorn_iterations,
        config.residual_mode == :mhc ? config.number_of_layers : 0,
    )
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
    span_context_blocks = Tuple([
        SparseSpanContextBlock(
            config.embedding_dimension;
            neighbor_radius = config.span_context_neighbor_radius,
            semantic_topk = config.span_context_topk,
        )
        for _ in 1:config.span_context_layers
    ])

    d = config.embedding_dimension
    anchor_top_spans = max(
        config.pair_global_top_spans,
        config.pair_router_topk * max(config.pair_router_routes_per_span, 1),
    )
    return SwammaRelationExtractor(
        config.vocab_size,
        config.max_sequence_length,
        d,
        config.number_of_layers,
        config.number_of_refinement_layers,
        config.use_interleaved_local_wave,
        config.interleaved_block_type,
        config.local_wave_ratio,
        config.interleaved_use_local_attention,
        config.interleaved_use_wave_pde,
        config.num_entity_labels,
        config.num_relations,
        config.residual_mode,
        config.hyper_connection_width,
        config.max_candidate_spans,
        config.max_candidate_pairs,
        config.max_span_width,
        config.span_context_layers,
        config.pair_neighbor_radius,
        config.pair_proposer_mode,
        config.pair_global_top_spans,
        anchor_top_spans,
        config.pair_router_buckets,
        config.pair_router_topk,
        config.pair_router_routes_per_span,
        config.pair_router_score_scale,
        config.pair_overgenerate_factor,
        config.pair_retrieval_dimension,
        config.pair_distance_buckets,
        config.pair_retrieval_loss_weight,
        config.pair_evidence_dimension,
        config.relation_decoder_mode,
        config.relation_decoder_residual_scale,
        config.mention_score_mode,
        config.mention_score_learned_weight,
        Lux.Embedding(config.vocab_size => d),
        Lux.Embedding(config.max_sequence_length => d),
        FixedTimeEmbedding(config.time_dimension),
        blocks,
        hyper_connections,
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
        Lux.Chain(
            Lux.LayerNorm((d,)),
            Lux.Dense(d => max(d ÷ 2, 1), gelu; use_bias = false),
            Lux.Dense(max(d ÷ 2, 1) => 1; use_bias = false),
        ),
        span_context_blocks,
        SparsePairProposalHead(
            d;
            router_dimension = min(config.pair_router_dimension, d),
            bucket_count = max(config.pair_router_buckets, 1),
        ),
        PairRetrievalHead(
            d;
            retrieval_dimension = min(config.pair_retrieval_dimension, d),
            distance_buckets = max(config.pair_distance_buckets, 2),
        ),
        PairEvidenceSelectorHead(
            d;
            evidence_dimension = min(config.pair_evidence_dimension, d),
        ),
        begin
            if config.relation_decoder_mode == :biaffine
                LowRankBiaffineRelationHead(
                    d,
                    config.num_relations;
                    rank = min(config.biaffine_rank, d),
                )
            elseif config.relation_decoder_mode == :fused_residual
                FusedRelationDecoderHead(
                    d,
                    config.num_relations;
                    rank = min(config.biaffine_rank, d),
                    residual_scale = config.relation_decoder_residual_scale,
                )
            elseif config.relation_decoder_mode == :fused_evidence
                EvidenceAwareRelationDecoderHead(
                    d,
                    config.num_relations;
                    rank = min(config.biaffine_rank, d),
                    residual_scale = config.relation_decoder_residual_scale,
                )
            else
                throw(ArgumentError(
                    "Unsupported relation_decoder_mode=$(repr(config.relation_decoder_mode)). Expected :biaffine, :fused_residual, or :fused_evidence."
                ))
            end
        end,
        begin
            confidence_input_dim = config.relation_decoder_mode == :fused_evidence ? 5 * d + 1 : 4 * d
            Lux.Chain(
                Lux.LayerNorm((confidence_input_dim,)),
                Lux.Dense(confidence_input_dim => d ÷ 2, gelu; use_bias = false),
                Lux.Dense(d ÷ 2 => 1; use_bias = false),
            )
        end,
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::SwammaRelationExtractor)
    block_params = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialparameters(rng, block) for block in model.Blocks)
    )
    hyper_params = NamedTuple{ntuple(i -> Symbol("HyperConnection_$i"), length(model.HyperConnections))}(
        Tuple((logits = initial_hyper_connection_logits(model.hyper_connection_width),) for _ in model.HyperConnections)
    )
    refinement_params = NamedTuple{ntuple(i -> Symbol("RefinementBlock_$i"), model.number_of_refinement_layers)}(
        Tuple(Lux.initialparameters(rng, block) for block in model.RefinementBlocks)
    )
    span_context_params = NamedTuple{ntuple(i -> Symbol("SpanContextBlock_$i"), model.span_context_layers)}(
        Tuple(Lux.initialparameters(rng, block) for block in model.SpanContextBlocks)
    )

    return (
        TokenEmbedding = Lux.initialparameters(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialparameters(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialparameters(rng, model.TimeEmbedding),
        Blocks = block_params,
        HyperConnections = hyper_params,
        RefinementBlocks = refinement_params,
        Dropout = Lux.initialparameters(rng, model.Dropout),
        EntityHead = Lux.initialparameters(rng, model.EntityHead),
        BoundaryHead = Lux.initialparameters(rng, model.BoundaryHead),
        SpanProjection = Lux.initialparameters(rng, model.SpanProjection),
        MentionHead = Lux.initialparameters(rng, model.MentionHead),
        SpanContextBlocks = span_context_params,
        PairProposalHead = Lux.initialparameters(rng, model.PairProposalHead),
        PairRetrievalHead = Lux.initialparameters(rng, model.PairRetrievalHead),
        PairEvidenceHead = Lux.initialparameters(rng, model.PairEvidenceHead),
        RelationHead = Lux.initialparameters(rng, model.RelationHead),
        ConfidenceHead = Lux.initialparameters(rng, model.ConfidenceHead),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::SwammaRelationExtractor)
    block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialstates(rng, block) for block in model.Blocks)
    )
    hyper_states = NamedTuple{ntuple(i -> Symbol("HyperConnection_$i"), length(model.HyperConnections))}(
        Tuple((;) for _ in model.HyperConnections)
    )
    refinement_states = NamedTuple{ntuple(i -> Symbol("RefinementBlock_$i"), model.number_of_refinement_layers)}(
        Tuple(Lux.initialstates(rng, block) for block in model.RefinementBlocks)
    )
    span_context_states = NamedTuple{ntuple(i -> Symbol("SpanContextBlock_$i"), model.span_context_layers)}(
        Tuple(Lux.initialstates(rng, block) for block in model.SpanContextBlocks)
    )
    return (
        TokenEmbedding = Lux.initialstates(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialstates(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialstates(rng, model.TimeEmbedding),
        Blocks = block_states,
        HyperConnections = hyper_states,
        RefinementBlocks = refinement_states,
        Dropout = Lux.initialstates(rng, model.Dropout),
        EntityHead = Lux.initialstates(rng, model.EntityHead),
        BoundaryHead = Lux.initialstates(rng, model.BoundaryHead),
        SpanProjection = Lux.initialstates(rng, model.SpanProjection),
        MentionHead = Lux.initialstates(rng, model.MentionHead),
        SpanContextBlocks = span_context_states,
        PairProposalHead = Lux.initialstates(rng, model.PairProposalHead),
        PairRetrievalHead = Lux.initialstates(rng, model.PairRetrievalHead),
        PairEvidenceHead = Lux.initialstates(rng, model.PairEvidenceHead),
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

    block_states = ()
    hyper_states = ()
    if model.residual_mode == :mhc
        primary_stream = hidden
        skip_stream = hidden
        for (i, block) in enumerate(model.Blocks)
            block_key = Symbol("Block_$i")
            block_params = params.Blocks[block_key]
            block_state = state.Blocks[block_key]
            new_h, new_block_state = block((primary_stream, time_emb), block_params, block_state)
            block_states = (block_states..., new_block_state)

            hc_key = Symbol("HyperConnection_$i")
            hc_params = params.HyperConnections[hc_key]
            hc_state = state.HyperConnections[hc_key]
            primary_stream, skip_stream = apply_manifold_hyper_connection(
                new_h,
                skip_stream,
                hc_params.logits,
                model.HyperConnections[i],
            )
            new_hc_state = hc_state
            hyper_states = (hyper_states..., new_hc_state)
        end
        hidden = primary_stream
    else
        for (i, block) in enumerate(model.Blocks)
            block_key = Symbol("Block_$i")
            block_params = params.Blocks[block_key]
            block_state = state.Blocks[block_key]
            hidden, new_block_state = block((hidden, time_emb), block_params, block_state)
            block_states = (block_states..., new_block_state)
        end
    end

    new_block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(block_states)
    new_hyper_states = NamedTuple{ntuple(i -> Symbol("HyperConnection_$i"), length(model.HyperConnections))}(hyper_states)

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
        HyperConnections = new_hyper_states,
        RefinementBlocks = new_refinement_states,
        Dropout = state.Dropout,
        EntityHead = state.EntityHead,
        BoundaryHead = state.BoundaryHead,
        SpanProjection = state.SpanProjection,
        MentionHead = state.MentionHead,
        SpanContextBlocks = state.SpanContextBlocks,
        PairProposalHead = state.PairProposalHead,
        PairRetrievalHead = state.PairRetrievalHead,
        PairEvidenceHead = state.PairEvidenceHead,
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

function heuristic_span_scores(entity_logits, boundary_logits, spans, span_mask)
    entity_logits_const = detach_constant(entity_logits)
    boundary_logits_const = detach_constant(boundary_logits)
    entity_logits_cpu = entity_logits_const isa CUDA.CuArray ? Array(entity_logits_const) : entity_logits_const
    boundary_logits_cpu = boundary_logits_const isa CUDA.CuArray ? Array(boundary_logits_const) : boundary_logits_const
    max_spans = size(spans, 2)
    batch_size = size(spans, 3)
    span_scores = fill(typemin(Float32), max_spans, batch_size)
    entity_mass = token_entity_mass(entity_logits_cpu)

    for b in 1:batch_size
        start_scores = vec(NNlib.sigmoid.(boundary_logits_cpu[1, :, b]))
        end_scores = vec(NNlib.sigmoid.(boundary_logits_cpu[2, :, b]))
        for i in 1:max_spans
            span_mask[i, b] || continue
            start_idx = Int(spans[1, i, b])
            end_idx = Int(spans[2, i, b])
            span_scores[i, b] = Float32(
                score_span(start_scores, end_scores, @view(entity_mass[:, b]), start_idx, end_idx)
            )
        end
    end

    return span_scores
end

function normalize_span_scores(scores, span_mask)
    normalized = fill(typemin(Float32), size(scores))
    batch_size = size(scores, 2)

    for b in 1:batch_size
        valid_indices = findall(@view(span_mask[:, b]))
        isempty(valid_indices) && continue
        valid_scores = Float32[scores[i, b] for i in valid_indices]
        μ = mean(valid_scores)
        σ = std(valid_scores; corrected = false)
        scale = σ > 1.0f-5 ? σ : 1.0f0
        for i in valid_indices
            normalized[i, b] = (scores[i, b] - μ) / scale
        end
    end

    return normalized
end

function combine_span_scores(
    learned_scores,
    heuristic_scores,
    span_mask;
    mode::Symbol,
    learned_weight::Float32,
)
    mode in (:heuristic, :learned, :hybrid) || throw(ArgumentError(
        "Unsupported mention_score_mode=$(repr(mode)). Expected :heuristic, :learned, or :hybrid."
    ))
    if mode == :heuristic
        return heuristic_scores
    elseif mode == :learned
        return learned_scores
    end

    α = clamp(learned_weight, 0.0f0, 1.0f0)
    learned_norm = normalize_span_scores(learned_scores, span_mask)
    heuristic_norm = normalize_span_scores(heuristic_scores, span_mask)
    return α .* learned_norm .+ (1.0f0 - α) .* heuristic_norm
end

function enumerate_candidate_span_list(seq_len::Int, max_span_width::Int)
    candidates = Tuple{Int, Int}[]
    for start_idx in 1:seq_len
        max_end = min(seq_len, start_idx + max_span_width - 1)
        for end_idx in start_idx:max_end
            push!(candidates, (start_idx, end_idx))
        end
    end
    return candidates
end

function candidate_span_tensor(seq_len::Int, max_span_width::Int, batch_size::Int)
    candidates = enumerate_candidate_span_list(seq_len, max_span_width)
    candidate_count = length(candidates)
    spans = zeros(Int, 2, candidate_count, batch_size)
    span_mask = trues(candidate_count, batch_size)

    for (idx, (start_idx, end_idx)) in enumerate(candidates)
        spans[1, idx, :] .= start_idx
        spans[2, idx, :] .= end_idx
    end

    return spans, span_mask
end

function score_span_representations(model::SwammaRelationExtractor, span_reps, span_mask, params, state)
    d, max_spans, batch_size = size(span_reps)
    mention_logits_flat, mention_state = model.MentionHead(
        reshape(span_reps, d, :),
        params.MentionHead,
        state.MentionHead,
    )
    mention_logits = reshape(mention_logits_flat, 1, max_spans, batch_size)
    mention_scores = reshape(vec(mention_logits), max_spans, batch_size) .* Float32.(span_mask)
    return mention_scores, mention_logits, mention_state
end

function propose_candidate_spans(
    model::SwammaRelationExtractor,
    hidden,
    entity_logits,
    boundary_logits;
    params,
    state,
    max_candidate_spans::Int,
    max_span_width::Int,
)
    seq_len = size(entity_logits, 2)
    batch_size = size(entity_logits, 3)
    spans = zeros(Int, 2, max_candidate_spans, batch_size)
    span_mask = falses(max_candidate_spans, batch_size)
    span_scores = fill(typemin(Float32), max_candidate_spans, batch_size)
    candidate_spans, candidate_mask = candidate_span_tensor(seq_len, max_span_width, batch_size)
    candidate_spans_device = hidden isa CUDA.CuArray ? CUDA.CuArray(candidate_spans) : candidate_spans
    candidate_mask_device = hidden isa CUDA.CuArray ? CUDA.CuArray(candidate_mask) : candidate_mask
    candidate_reps, span_state = build_span_representations(
        model,
        hidden,
        candidate_spans_device,
        candidate_mask_device,
        params,
        state,
    )
    mention_scores, _, mention_state = score_span_representations(
        model,
        candidate_reps,
        candidate_mask_device,
        params,
        state,
    )
    mention_scores_cpu = mention_scores isa CUDA.CuArray ? Array(mention_scores) : mention_scores
    heuristic_scores = heuristic_span_scores(entity_logits, boundary_logits, candidate_spans, candidate_mask)
    combined_scores = ChainRulesCore.ignore_derivatives() do
        combine_span_scores(
            mention_scores_cpu,
            heuristic_scores,
            candidate_mask;
            mode = model.mention_score_mode,
            learned_weight = model.mention_score_learned_weight,
        )
    end

    for b in 1:batch_size
        candidate_indices = sortperm(@view(combined_scores[:, b]); rev = true)
        top_count = min(max_candidate_spans, length(candidate_indices))
        for i in 1:top_count
            candidate_idx = candidate_indices[i]
            spans[1, i, b] = candidate_spans[1, candidate_idx, b]
            spans[2, i, b] = candidate_spans[2, candidate_idx, b]
            span_mask[i, b] = true
            span_scores[i, b] = combined_scores[candidate_idx, b]
        end
    end

    return spans, span_mask, span_scores, span_state, mention_state
end

function score_existing_spans(model::SwammaRelationExtractor, span_reps, span_mask, entity_logits, boundary_logits, spans, params, state)
    learned_scores, mention_logits, mention_state = score_span_representations(model, span_reps, span_mask, params, state)
    learned_scores_cpu = learned_scores isa CUDA.CuArray ? Array(learned_scores) : learned_scores
    span_mask_cpu = span_mask isa CUDA.CuArray ? Array(span_mask) : span_mask
    spans_cpu = spans isa CUDA.CuArray ? Array(spans) : spans
    heuristic_scores = heuristic_span_scores(entity_logits, boundary_logits, spans_cpu, span_mask_cpu)
    combined_scores = ChainRulesCore.ignore_derivatives() do
        combine_span_scores(
            learned_scores_cpu,
            heuristic_scores,
            span_mask_cpu;
            mode = model.mention_score_mode,
            learned_weight = model.mention_score_learned_weight,
        )
    end
    return combined_scores, mention_logits, mention_state
end

@inline function build_pair_candidate_base(spans, span_scores, head_idx::Int, tail_idx::Int, batch_idx::Int)
    head_start = Int(spans[1, head_idx, batch_idx])
    tail_start = Int(spans[1, tail_idx, batch_idx])
    return (
        score = Float32(span_scores[head_idx, batch_idx] + span_scores[tail_idx, batch_idx]),
        distance = abs(head_start - tail_start),
        head_start = head_start,
        tail_start = tail_start,
        head_idx = head_idx,
        tail_idx = tail_idx,
    )
end

@inline function build_pair_candidate(
    spans,
    span_scores,
    head_idx::Int,
    tail_idx::Int,
    batch_idx::Int;
    router_score::Float32 = 0.0f0,
    routing_bonus::Float32 = 0.0f0,
    router_score_scale::Float32 = 0.0f0,
)
    base = build_pair_candidate_base(spans, span_scores, head_idx, tail_idx, batch_idx)
    total_score = base.score + router_score_scale * (router_score + 0.25f0 * routing_bonus)
    return merge(base, (score = Float32(total_score),))
end

@inline function build_edge_v2_pair_candidate(
    spans,
    span_scores,
    head_idx::Int,
    tail_idx::Int,
    batch_idx::Int;
    semantic_score::Float32 = 0.0f0,
    semantic_score_scale::Float32 = 1.0f0,
    span_score_scale::Float32 = 1.0f0,
    distance_penalty::Float32 = 0.0f0,
)
    base = build_pair_candidate_base(spans, span_scores, head_idx, tail_idx, batch_idx)
    total_score = (
        span_score_scale * base.score +
        semantic_score_scale * semantic_score -
        distance_penalty * Float32(base.distance)
    )
    return merge(base, (score = Float32(total_score),))
end

@inline pair_candidate_is_better(candidate, existing) = (
    candidate.score > existing.score ||
    (candidate.score == existing.score && candidate.distance < existing.distance) ||
    (candidate.score == existing.score && candidate.distance == existing.distance && candidate.head_start < existing.head_start) ||
    (
        candidate.score == existing.score &&
        candidate.distance == existing.distance &&
        candidate.head_start == existing.head_start &&
        candidate.tail_start < existing.tail_start
    )
)

function register_pair_candidate!(candidate_lookup, candidate)
    pair = (candidate.head_idx, candidate.tail_idx)
    if !haskey(candidate_lookup, pair) || pair_candidate_is_better(candidate, candidate_lookup[pair])
        candidate_lookup[pair] = candidate
    end
    return nothing
end

pair_proposer_uses_router(mode::Symbol) = mode in (:sparse, :sparse_hybrid, :edge_retrieval_v2)
pair_proposer_uses_semantic_retrieval(mode::Symbol) = mode in (:sparse, :sparse_hybrid, :edge_retrieval_v2)

function reshape_router_outputs(router_outputs, max_spans::Int, batch_size::Int)
    return (
        head_router = reshape(router_outputs.head_router, :, max_spans, batch_size),
        tail_router = reshape(router_outputs.tail_router, :, max_spans, batch_size),
        bucket_logits = reshape(router_outputs.bucket_logits, :, max_spans, batch_size),
    )
end

function build_retrieval_projection_outputs(model::SwammaRelationExtractor, span_reps, params, state)
    d, max_spans, batch_size = size(span_reps)
    head_proj_flat, head_state = model.PairRetrievalHead.HeadProjection(
        reshape(span_reps, d, :),
        params.PairRetrievalHead.HeadProjection,
        state.PairRetrievalHead.HeadProjection,
    )
    tail_proj_flat, tail_state = model.PairRetrievalHead.TailProjection(
        reshape(span_reps, d, :),
        params.PairRetrievalHead.TailProjection,
        state.PairRetrievalHead.TailProjection,
    )
    return (
        head = reshape(head_proj_flat, :, max_spans, batch_size),
        tail = reshape(tail_proj_flat, :, max_spans, batch_size),
    ), (
        HeadProjection = head_state,
        TailProjection = tail_state,
    )
end

function propose_relation_pairs(
    spans,
    span_mask,
    span_scores;
    max_candidate_pairs::Int,
    neighbor_radius::Int,
    proposer_mode::Symbol = :local,
    global_top_spans::Int = 0,
    anchor_top_spans::Int = 0,
    router_outputs = nothing,
    semantic_outputs = nothing,
    router_topk::Int = 6,
    router_routes_per_span::Int = 2,
    router_score_scale::Float32 = 0.5f0,
    edge_v2_semantic_topk::Int = 0,
    edge_v2_reverse_topk::Int = 0,
    edge_v2_global_reserve::Int = 0,
    edge_v2_semantic_score_scale::Float32 = 1.0f0,
    edge_v2_span_score_scale::Float32 = 1.0f0,
    edge_v2_distance_penalty::Float32 = 0.0f0,
    edge_v2_require_mutual::Bool = false,
    edge_v2_use_local_neighbors::Bool = true,
    edge_v2_use_routed_buckets::Bool = true,
    edge_v2_use_semantic_topk::Bool = true,
    edge_v2_use_global_reserve::Bool = true,
)
    batch_size = size(spans, 3)
    relation_pairs = zeros(Int, 2, max_candidate_pairs, batch_size)
    relation_mask = falses(max_candidate_pairs, batch_size)
    proposer_mode in (:local, :global, :hybrid, :sparse, :sparse_hybrid, :edge_retrieval_v2) || throw(ArgumentError(
        "Unsupported proposer_mode=$(repr(proposer_mode)). Expected :local, :global, :hybrid, :sparse, :sparse_hybrid, or :edge_retrieval_v2."
    ))

    spans_cpu = spans isa CUDA.CuArray ? Array(spans) : spans
    span_mask_cpu = span_mask isa CUDA.CuArray ? Array(span_mask) : span_mask
    span_scores_cpu = span_scores isa CUDA.CuArray ? Array(span_scores) : span_scores
    router_cpu = if router_outputs === nothing
        nothing
    else
        (
            head_router = router_outputs.head_router isa CUDA.CuArray ? Array(router_outputs.head_router) : router_outputs.head_router,
            tail_router = router_outputs.tail_router isa CUDA.CuArray ? Array(router_outputs.tail_router) : router_outputs.tail_router,
            bucket_logits = router_outputs.bucket_logits isa CUDA.CuArray ? Array(router_outputs.bucket_logits) : router_outputs.bucket_logits,
        )
    end
    semantic_cpu = if semantic_outputs === nothing
        nothing
    else
        (
            head = semantic_outputs.head isa CUDA.CuArray ? Array(semantic_outputs.head) : semantic_outputs.head,
            tail = semantic_outputs.tail isa CUDA.CuArray ? Array(semantic_outputs.tail) : semantic_outputs.tail,
        )
    end
    if proposer_mode == :edge_retrieval_v2 && semantic_cpu === nothing
        throw(ArgumentError("semantic_outputs must be provided when proposer_mode=:edge_retrieval_v2."))
    end

    for b in 1:batch_size
        valid_indices = findall(@view(span_mask_cpu[:, b]))
        isempty(valid_indices) && continue

        ordered_by_position = sort(valid_indices, by = i -> (spans_cpu[1, i, b], spans_cpu[2, i, b], -span_scores_cpu[i, b]))
        ordered_by_score = sort(valid_indices, by = i -> span_scores_cpu[i, b], rev = true)
        position_lookup = Dict(idx => pos for (pos, idx) in enumerate(ordered_by_position))
        candidate_lookup = Dict{Tuple{Int, Int}, NamedTuple{(:score, :distance, :head_start, :tail_start, :head_idx, :tail_idx), Tuple{Float32, Int, Int, Int, Int, Int}}}()

        if proposer_mode in (:global, :hybrid) || (proposer_mode == :sparse_hybrid && global_top_spans != 0)
            top_global_spans = global_top_spans > 0 ?
                min(length(ordered_by_score), global_top_spans) :
                min(length(ordered_by_score), floor(Int, (1 + sqrt(1 + 4 * max_candidate_pairs)) / 2))
            for head_offset in 1:top_global_spans
                head_idx = ordered_by_score[head_offset]
                for tail_offset in 1:top_global_spans
                    head_offset == tail_offset && continue
                    tail_idx = ordered_by_score[tail_offset]
                    register_pair_candidate!(candidate_lookup, build_pair_candidate(spans_cpu, span_scores_cpu, head_idx, tail_idx, b))
                end
            end
        end

        if proposer_mode in (:local, :hybrid, :sparse_hybrid) ||
           (proposer_mode == :edge_retrieval_v2 && edge_v2_use_local_neighbors)
            for anchor_idx in ordered_by_score
                anchor_pos = position_lookup[anchor_idx]

                for delta in 1:neighbor_radius
                    if anchor_pos + delta <= length(ordered_by_position)
                        neighbor_idx = ordered_by_position[anchor_pos + delta]
                        for pair in ((anchor_idx, neighbor_idx), (neighbor_idx, anchor_idx))
                            if proposer_mode == :edge_retrieval_v2
                                register_pair_candidate!(
                                    candidate_lookup,
                                    build_edge_v2_pair_candidate(
                                        spans_cpu,
                                        span_scores_cpu,
                                        pair[1],
                                        pair[2],
                                        b;
                                        semantic_score = 0.0f0,
                                        semantic_score_scale = edge_v2_semantic_score_scale,
                                        span_score_scale = edge_v2_span_score_scale,
                                        distance_penalty = edge_v2_distance_penalty,
                                    ),
                                )
                            else
                                register_pair_candidate!(
                                    candidate_lookup,
                                    build_pair_candidate(spans_cpu, span_scores_cpu, pair[1], pair[2], b),
                                )
                            end
                        end
                    end

                    if anchor_pos - delta >= 1
                        neighbor_idx = ordered_by_position[anchor_pos - delta]
                        for pair in ((anchor_idx, neighbor_idx), (neighbor_idx, anchor_idx))
                            if proposer_mode == :edge_retrieval_v2
                                register_pair_candidate!(
                                    candidate_lookup,
                                    build_edge_v2_pair_candidate(
                                        spans_cpu,
                                        span_scores_cpu,
                                        pair[1],
                                        pair[2],
                                        b;
                                        semantic_score = 0.0f0,
                                        semantic_score_scale = edge_v2_semantic_score_scale,
                                        span_score_scale = edge_v2_span_score_scale,
                                        distance_penalty = edge_v2_distance_penalty,
                                    ),
                                )
                            else
                                register_pair_candidate!(
                                    candidate_lookup,
                                    build_pair_candidate(spans_cpu, span_scores_cpu, pair[1], pair[2], b),
                                )
                            end
                        end
                    end
                end
            end
        end

        if anchor_top_spans > 0 && proposer_mode != :edge_retrieval_v2
            top_anchor_count = min(length(ordered_by_score), anchor_top_spans)
            anchor_fanout = max(4, 2 * max(router_topk, 1))
            for anchor_pos in 1:top_anchor_count
                anchor_idx = ordered_by_score[anchor_pos]
                if semantic_cpu !== nothing
                    route_width = max(size(semantic_cpu.head, 1), 1)
                    semantic_neighbors = Tuple{Int, Float32}[]
                    head_projection = @view(semantic_cpu.head[:, anchor_idx, b])
                    tail_projection = @view(semantic_cpu.tail[:, anchor_idx, b])
                    for other_idx in valid_indices
                        anchor_idx == other_idx && continue
                        forward_score = Float32(
                            sum(head_projection .* @view(semantic_cpu.tail[:, other_idx, b])) / sqrt(Float32(route_width))
                        )
                        reverse_score = Float32(
                            sum(@view(semantic_cpu.head[:, other_idx, b]) .* tail_projection) / sqrt(Float32(route_width))
                        )
                        push!(semantic_neighbors, (other_idx, max(forward_score, reverse_score)))
                    end
                    sort!(semantic_neighbors; by = item -> item[2], rev = true)
                    for (other_idx, semantic_score) in Iterators.take(semantic_neighbors, anchor_fanout)
                        register_pair_candidate!(
                            candidate_lookup,
                            build_pair_candidate(
                                spans_cpu,
                                span_scores_cpu,
                                anchor_idx,
                                other_idx,
                                b;
                                router_score = semantic_score,
                                router_score_scale = router_score_scale,
                            ),
                        )
                        register_pair_candidate!(
                            candidate_lookup,
                            build_pair_candidate(
                                spans_cpu,
                                span_scores_cpu,
                                other_idx,
                                anchor_idx,
                                b;
                                router_score = semantic_score,
                                router_score_scale = router_score_scale,
                            ),
                        )
                    end
                else
                    for other_idx in Iterators.take(ordered_by_score, anchor_fanout)
                        anchor_idx == other_idx && continue
                        register_pair_candidate!(
                            candidate_lookup,
                            build_pair_candidate(spans_cpu, span_scores_cpu, anchor_idx, other_idx, b),
                        )
                        register_pair_candidate!(
                            candidate_lookup,
                            build_pair_candidate(spans_cpu, span_scores_cpu, other_idx, anchor_idx, b),
                        )
                    end
                end
            end
        end

        if pair_proposer_uses_router(proposer_mode) &&
           !(proposer_mode == :edge_retrieval_v2 && !edge_v2_use_routed_buckets)
            router_cpu === nothing && throw(ArgumentError(
                "router_outputs must be provided when proposer_mode=$(repr(proposer_mode))."
            ))
            bucket_count = size(router_cpu.bucket_logits, 1)
            route_width = max(size(router_cpu.head_router, 1), 1)
            top_routes = max(1, min(router_routes_per_span, bucket_count))
            bucket_topk = max(1, router_topk)
            bucket_members = [Vector{Tuple{Int, Float32}}() for _ in 1:bucket_count]

            for span_idx in ordered_by_score
                bucket_scores = vec(@view(router_cpu.bucket_logits[:, span_idx, b]))
                ranked_buckets = sortperm(bucket_scores; rev = true)
                for bucket_idx in Iterators.take(ranked_buckets, top_routes)
                    confidence = Float32(NNlib.sigmoid(bucket_scores[bucket_idx]))
                    push!(bucket_members[bucket_idx], (span_idx, confidence))
                end
            end

            for bucket_idx in 1:bucket_count
                isempty(bucket_members[bucket_idx]) && continue
                sort!(
                    bucket_members[bucket_idx];
                    by = item -> (item[2], span_scores_cpu[item[1], b]),
                    rev = true,
                )
                active_members = collect(Iterators.take(bucket_members[bucket_idx], bucket_topk))
                for head_pos in eachindex(active_members)
                    head_idx, head_conf = active_members[head_pos]
                    for tail_pos in eachindex(active_members)
                        head_pos == tail_pos && continue
                        tail_idx, tail_conf = active_members[tail_pos]
                        router_score = Float32(
                            sum(
                                @view(router_cpu.head_router[:, head_idx, b]) .* @view(router_cpu.tail_router[:, tail_idx, b])
                            ) / sqrt(Float32(route_width))
                        )
                        if proposer_mode == :edge_retrieval_v2
                            routed_semantic = router_score + 0.25f0 * (head_conf + tail_conf)
                            register_pair_candidate!(
                                candidate_lookup,
                                build_edge_v2_pair_candidate(
                                    spans_cpu,
                                    span_scores_cpu,
                                    head_idx,
                                    tail_idx,
                                    b;
                                    semantic_score = routed_semantic,
                                    semantic_score_scale = edge_v2_semantic_score_scale,
                                    span_score_scale = edge_v2_span_score_scale,
                                    distance_penalty = edge_v2_distance_penalty,
                                ),
                            )
                        else
                            register_pair_candidate!(
                                candidate_lookup,
                                build_pair_candidate(
                                    spans_cpu,
                                    span_scores_cpu,
                                    head_idx,
                                    tail_idx,
                                    b;
                                    router_score = router_score,
                                    routing_bonus = head_conf + tail_conf,
                                    router_score_scale = router_score_scale,
                                ),
                            )
                        end
                    end
                end
            end
        end

        if proposer_mode in (:sparse, :sparse_hybrid) && semantic_cpu !== nothing
            route_width = max(size(semantic_cpu.head, 1), 1)
            semantic_topk = max(1, router_topk)
            for head_idx in ordered_by_score
                semantic_candidates = Tuple{Int, Float32}[]
                head_projection = @view(semantic_cpu.head[:, head_idx, b])
                for tail_idx in valid_indices
                    head_idx == tail_idx && continue
                    semantic_score = Float32(
                        sum(head_projection .* @view(semantic_cpu.tail[:, tail_idx, b])) / sqrt(Float32(route_width))
                    )
                    push!(semantic_candidates, (tail_idx, semantic_score))
                end
                isempty(semantic_candidates) && continue
                sort!(semantic_candidates; by = candidate -> candidate[2], rev = true)
                for (tail_idx, semantic_score) in Iterators.take(semantic_candidates, semantic_topk)
                    register_pair_candidate!(
                        candidate_lookup,
                        build_pair_candidate(
                            spans_cpu,
                            span_scores_cpu,
                            head_idx,
                            tail_idx,
                            b;
                            router_score = semantic_score,
                            router_score_scale = router_score_scale,
                        ),
                    )
                end
            end
        elseif proposer_mode == :edge_retrieval_v2 && semantic_cpu !== nothing
            route_width = max(size(semantic_cpu.head, 1), 1)
            semantic_topk = edge_v2_semantic_topk > 0 ? edge_v2_semantic_topk : max(1, router_topk)
            reverse_topk = edge_v2_reverse_topk > 0 ? edge_v2_reverse_topk : semantic_topk
            reserve_count = if edge_v2_global_reserve > 0
                min(length(ordered_by_score), edge_v2_global_reserve)
            elseif global_top_spans > 0
                min(length(ordered_by_score), global_top_spans)
            else
                min(length(ordered_by_score), max(2, semantic_topk))
            end

            if edge_v2_use_semantic_topk
                forward_neighbors = Dict{Int, Vector{Tuple{Int, Float32}}}()
                reverse_neighbor_sets = Dict{Int, Set{Int}}()

                for head_idx in ordered_by_score
                    candidates = Tuple{Int, Float32}[]
                    head_projection = @view(semantic_cpu.head[:, head_idx, b])
                    for tail_idx in valid_indices
                        head_idx == tail_idx && continue
                        semantic_score = Float32(
                            sum(head_projection .* @view(semantic_cpu.tail[:, tail_idx, b])) / sqrt(Float32(route_width))
                        )
                        push!(candidates, (tail_idx, semantic_score))
                    end
                    isempty(candidates) && continue
                    sort!(candidates; by = item -> item[2], rev = true)
                    forward_neighbors[head_idx] = collect(Iterators.take(candidates, semantic_topk))
                end

                for tail_idx in ordered_by_score
                    candidates = Tuple{Int, Float32}[]
                    tail_projection = @view(semantic_cpu.tail[:, tail_idx, b])
                    for head_idx in valid_indices
                        head_idx == tail_idx && continue
                        semantic_score = Float32(
                            sum(@view(semantic_cpu.head[:, head_idx, b]) .* tail_projection) / sqrt(Float32(route_width))
                        )
                        push!(candidates, (head_idx, semantic_score))
                    end
                    isempty(candidates) && continue
                    sort!(candidates; by = item -> item[2], rev = true)
                    reverse_neighbor_sets[tail_idx] = Set(item[1] for item in Iterators.take(candidates, reverse_topk))
                end

                for (head_idx, neighbors) in forward_neighbors
                    for (tail_idx, semantic_score) in neighbors
                        if edge_v2_require_mutual
                            haskey(reverse_neighbor_sets, tail_idx) || continue
                            (head_idx in reverse_neighbor_sets[tail_idx]) || continue
                        end
                        register_pair_candidate!(
                            candidate_lookup,
                            build_edge_v2_pair_candidate(
                                spans_cpu,
                                span_scores_cpu,
                                head_idx,
                                tail_idx,
                                b;
                                semantic_score = semantic_score,
                                semantic_score_scale = edge_v2_semantic_score_scale,
                                span_score_scale = edge_v2_span_score_scale,
                                distance_penalty = edge_v2_distance_penalty,
                            ),
                        )
                    end
                end
            end

            if edge_v2_use_global_reserve
                reserve = collect(Iterators.take(ordered_by_score, reserve_count))
                for head_idx in reserve
                    for tail_idx in reserve
                        head_idx == tail_idx && continue
                        register_pair_candidate!(
                            candidate_lookup,
                            build_edge_v2_pair_candidate(
                                spans_cpu,
                                span_scores_cpu,
                                head_idx,
                                tail_idx,
                                b;
                                semantic_score = 0.0f0,
                                semantic_score_scale = edge_v2_semantic_score_scale,
                                span_score_scale = edge_v2_span_score_scale,
                                distance_penalty = edge_v2_distance_penalty,
                            ),
                        )
                    end
                end
            end
        end

        candidates = collect(values(candidate_lookup))

        sort!(
            candidates;
            by = candidate -> (-candidate.score, candidate.distance, candidate.head_start, candidate.tail_start),
        )

        for (pair_idx, candidate) in enumerate(Iterators.take(candidates, max_candidate_pairs))
            relation_pairs[1, pair_idx, b] = candidate.head_idx
            relation_pairs[2, pair_idx, b] = candidate.tail_idx
            relation_mask[pair_idx, b] = true
        end
    end

    return relation_pairs, relation_mask
end

function select_top_relation_pairs(relation_pairs, relation_mask, retrieval_logits; max_candidate_pairs::Int)
    batch_size = size(relation_pairs, 3)
    selected_pairs = zeros(Int, 2, max_candidate_pairs, batch_size)
    selected_mask = falses(max_candidate_pairs, batch_size)

    pairs_cpu = relation_pairs isa CUDA.CuArray ? Array(relation_pairs) : relation_pairs
    mask_cpu = relation_mask isa CUDA.CuArray ? Array(relation_mask) : relation_mask
    logits_cpu = retrieval_logits isa CUDA.CuArray ? Array(retrieval_logits) : retrieval_logits

    for b in 1:batch_size
        valid_indices = findall(@view(mask_cpu[:, b]))
        isempty(valid_indices) && continue
        ranked = sort(
            valid_indices;
            by = idx -> (Float32(logits_cpu[1, idx, b]), -idx),
            rev = true,
        )
        for (out_idx, candidate_idx) in enumerate(Iterators.take(ranked, max_candidate_pairs))
            selected_pairs[:, out_idx, b] .= pairs_cpu[:, candidate_idx, b]
            selected_mask[out_idx, b] = true
        end
    end

    if relation_pairs isa CUDA.CuArray
        return CUDA.CuArray(selected_pairs), CUDA.CuArray(selected_mask)
    end
    return selected_pairs, selected_mask
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

function build_span_context_adjacency(
    scores,
    spans,
    span_mask;
    neighbor_radius::Int,
    semantic_topk::Int,
    use_adjacent::Bool = true,
    use_sentence::Bool = true,
    use_semantic::Bool = true,
    sentence_ids = nothing,
)
    max_spans, _, batch_size = size(scores)
    adjacency = falses(max_spans, max_spans, batch_size)
    for b in 1:batch_size
        valid_indices = findall(@view(span_mask[:, b]))
        isempty(valid_indices) && continue
        ordered_indices = sort(valid_indices; by = idx -> (Int(spans[1, idx, b]), Int(spans[2, idx, b]), idx))

        for span_idx in ordered_indices
            adjacency[span_idx, span_idx, b] = true
        end

        if use_adjacent
            for (order_pos, span_idx) in enumerate(ordered_indices)
                start_pos = max(1, order_pos - max(neighbor_radius, 0))
                end_pos = min(length(ordered_indices), order_pos + max(neighbor_radius, 0))
                for neighbor_pos in start_pos:end_pos
                    neighbor_idx = ordered_indices[neighbor_pos]
                    adjacency[span_idx, neighbor_idx, b] = true
                end
            end
        end

        if use_sentence && sentence_ids !== nothing
            sentence_ids_cpu = sentence_ids isa CUDA.CuArray ? Array(sentence_ids) : sentence_ids
            sentence_column = if ndims(sentence_ids_cpu) == 1
                sentence_ids_cpu
            elseif ndims(sentence_ids_cpu) == 2
                @view(sentence_ids_cpu[:, b])
            else
                throw(ArgumentError("sentence_ids must be rank-1 or rank-2; got ndims=$(ndims(sentence_ids_cpu))."))
            end
            sentence_groups = Dict{Int, Vector{Int}}()
            seq_len = length(sentence_column)
            for span_idx in ordered_indices
                start_idx = clamp(Int(spans[1, span_idx, b]), 1, seq_len)
                sentence_id = Int(sentence_column[start_idx])
                push!(get!(sentence_groups, sentence_id, Int[]), span_idx)
            end
            for group in values(sentence_groups)
                for head_idx in group
                    for tail_idx in group
                        adjacency[head_idx, tail_idx, b] = true
                    end
                end
            end
        end

        if use_semantic && semantic_topk > 0
            for span_idx in ordered_indices
                semantic_candidates = Int[]
                semantic_scores = Float32[]
                for neighbor_idx in valid_indices
                    neighbor_idx == span_idx && continue
                    push!(semantic_candidates, neighbor_idx)
                    push!(semantic_scores, Float32(scores[span_idx, neighbor_idx, b]))
                end
                ranked = sortperm(semantic_scores; rev = true)
                for rank_idx in Iterators.take(ranked, min(semantic_topk, length(ranked)))
                    adjacency[span_idx, semantic_candidates[rank_idx], b] = true
                end
            end
        end
    end
    return adjacency
end

function (layer::SparseSpanContextBlock)(inputs::Tuple, params, state)
    span_reps, spans, span_mask, context_options = if length(inputs) == 3
        (inputs[1], inputs[2], inputs[3], nothing)
    elseif length(inputs) == 4
        (inputs[1], inputs[2], inputs[3], inputs[4])
    else
        throw(ArgumentError("SparseSpanContextBlock expects 3 or 4 inputs, got $(length(inputs))."))
    end
    use_adjacent = context_options === nothing ? true : (
        hasproperty(context_options, :use_adjacent) ? Bool(context_options.use_adjacent) : true
    )
    use_sentence = context_options === nothing ? true : (
        hasproperty(context_options, :use_sentence) ? Bool(context_options.use_sentence) : true
    )
    use_semantic = context_options === nothing ? true : (
        hasproperty(context_options, :use_semantic) ? Bool(context_options.use_semantic) : true
    )
    sentence_ids = context_options === nothing ? nothing : (
        hasproperty(context_options, :sentence_ids) ? context_options.sentence_ids : nothing
    )

    d, max_spans, batch_size = size(span_reps)
    on_gpu = span_reps isa CUDA.CuArray

    span_flat = reshape(span_reps, d, :)
    normalized_flat, norm_state = layer.InputNorm(span_flat, params.InputNorm, state.InputNorm)
    query_flat, query_state = layer.QueryProjection(normalized_flat, params.QueryProjection, state.QueryProjection)
    key_flat, key_state = layer.KeyProjection(normalized_flat, params.KeyProjection, state.KeyProjection)
    value_flat, value_state = layer.ValueProjection(normalized_flat, params.ValueProjection, state.ValueProjection)

    queries = reshape(query_flat, d, max_spans, batch_size)
    keys = reshape(key_flat, d, max_spans, batch_size)
    values = reshape(value_flat, d, max_spans, batch_size)

    adjacency_device = ChainRulesCore.ignore_derivatives() do
        scores_cpu = Array{Float32}(undef, max_spans, max_spans, batch_size)
        queries_cpu = queries isa CUDA.CuArray ? Array(queries) : queries
        keys_cpu = keys isa CUDA.CuArray ? Array(keys) : keys
        for b in 1:batch_size
            scores_cpu[:, :, b] .= Float32.(transpose(@view(queries_cpu[:, :, b])) * @view(keys_cpu[:, :, b]) ./ sqrt(Float32(d)))
        end
        spans_cpu = spans isa CUDA.CuArray ? Array(spans) : spans
        span_mask_cpu = span_mask isa CUDA.CuArray ? Array(span_mask) : span_mask
        adjacency = build_span_context_adjacency(
            scores_cpu,
            spans_cpu,
            span_mask_cpu;
            neighbor_radius = layer.neighbor_radius,
            semantic_topk = layer.semantic_topk,
            use_adjacent = use_adjacent,
            use_sentence = use_sentence,
            use_semantic = use_semantic,
            sentence_ids = sentence_ids,
        )
        on_gpu ? CUDA.CuArray(adjacency) : adjacency
    end

    large_negative = oftype(zero(eltype(span_reps)), -1.0f4)
    attention_batches = map(1:batch_size) do b
        raw_scores = transpose(@view(queries[:, :, b])) * @view(keys[:, :, b]) ./ sqrt(Float32(d))
        masked_scores = ifelse.(@view(adjacency_device[:, :, b]), raw_scores, large_negative)
        attention_weights = NNlib.softmax(masked_scores, dims = 2)
        message = @view(values[:, :, b]) * transpose(attention_weights)
        message .* reshape(Float32.(@view(span_mask[:, b])), 1, :)
    end
    attention_messages = cat(attention_batches...; dims = 3)

    message_flat = reshape(attention_messages, d, :)
    output_flat, output_state = layer.OutputProjection(message_flat, params.OutputProjection, state.OutputProjection)
    mask_values = reshape(Float32.(vec(span_mask)), 1, :)
    residual_flat = span_flat .+ output_flat .* mask_values
    feedforward_flat, feedforward_state = layer.FeedForward(residual_flat, params.FeedForward, state.FeedForward)
    updated_flat = residual_flat .+ feedforward_flat .* mask_values

    new_state = (
        InputNorm = norm_state,
        QueryProjection = query_state,
        KeyProjection = key_state,
        ValueProjection = value_state,
        OutputProjection = output_state,
        FeedForward = feedforward_state,
    )
    return reshape(updated_flat, d, max_spans, batch_size), new_state
end

function apply_span_context(
    model::SwammaRelationExtractor,
    span_reps,
    spans,
    span_mask,
    params,
    state;
    enabled::Bool = true,
    use_adjacent::Bool = true,
    use_sentence::Bool = true,
    use_semantic::Bool = true,
    sentence_ids = nothing,
)
    (!enabled || model.span_context_layers == 0) && return span_reps, state.SpanContextBlocks

    block_states = ()
    contextualized = span_reps
    context_options = (
        use_adjacent = use_adjacent,
        use_sentence = use_sentence,
        use_semantic = use_semantic,
        sentence_ids = sentence_ids,
    )
    for (i, block) in enumerate(model.SpanContextBlocks)
        block_key = Symbol("SpanContextBlock_$i")
        contextualized, block_state = block(
            (contextualized, spans, span_mask, context_options),
            params.SpanContextBlocks[block_key],
            state.SpanContextBlocks[block_key],
        )
        block_states = (block_states..., block_state)
    end

    new_states = NamedTuple{ntuple(i -> Symbol("SpanContextBlock_$i"), model.span_context_layers)}(block_states)
    return contextualized, new_states
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

@inline function pair_distance_bucket_id(distance::Int, bucket_count::Int)
    bucket_count <= 1 && return 1
    boundaries = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
    max_bucket = bucket_count
    for (idx, boundary) in enumerate(boundaries)
        if distance <= boundary
            return min(idx, max_bucket)
        end
    end
    return max_bucket
end

function gather_pair_aux_features(
    spans,
    span_scores,
    relation_pairs,
    relation_mask;
    distance_buckets::Int,
    entity_logits = nothing,
    sentence_ids = nothing,
    local_radius::Int = 0,
)
    max_pairs = size(relation_pairs, 2)
    batch_size = size(relation_pairs, 3)
    distance_ids = ones(Int32, max_pairs, batch_size)
    sentence_distance_ids = ones(Int32, max_pairs, batch_size)
    score_features = zeros(Float32, 2, max_pairs, batch_size)
    distance_bias_base = zeros(Float32, 1, max_pairs, batch_size)
    type_bias_base = zeros(Float32, 1, max_pairs, batch_size)
    sentence_bias_base = zeros(Float32, 1, max_pairs, batch_size)
    local_bias_base = zeros(Float32, 1, max_pairs, batch_size)
    type_compat_bias_base = zeros(Float32, 1, max_pairs, batch_size)

    spans_cpu = spans isa CUDA.CuArray ? Array(spans) : spans
    span_scores_cpu = span_scores isa CUDA.CuArray ? Array(span_scores) : span_scores
    relation_pairs_cpu = relation_pairs isa CUDA.CuArray ? Array(relation_pairs) : relation_pairs
    relation_mask_cpu = relation_mask isa CUDA.CuArray ? Array(relation_mask) : relation_mask
    sentence_ids_cpu = sentence_ids isa CUDA.CuArray ? Array(sentence_ids) : sentence_ids
    entity_mass_cpu = nothing
    entity_type_probs_cpu = nothing
    if entity_logits !== nothing
        entity_logits_cpu = entity_logits isa CUDA.CuArray ? Array(entity_logits) : entity_logits
        entity_mass_cpu = token_entity_mass(entity_logits_cpu)
        if size(entity_logits_cpu, 1) > 1
            type_probs = NNlib.softmax(entity_logits_cpu[2:end, :, :], dims = 1)
            type_norm = sum(type_probs, dims = 1) .+ 1.0f-8
            entity_type_probs_cpu = type_probs ./ type_norm
        end
    end
    max_spans = size(spans_cpu, 2)
    max_distance_denom = max(distance_buckets - 1, 1)

    for b in 1:batch_size
        for pair_idx in 1:max_pairs
            relation_mask_cpu[pair_idx, b] || continue
            head_idx = Int(relation_pairs_cpu[1, pair_idx, b])
            tail_idx = Int(relation_pairs_cpu[2, pair_idx, b])
            if !(1 <= head_idx <= max_spans && 1 <= tail_idx <= max_spans)
                continue
            end
            head_start = Int(spans_cpu[1, head_idx, b])
            tail_start = Int(spans_cpu[1, tail_idx, b])
            distance_bucket = pair_distance_bucket_id(abs(head_start - tail_start), distance_buckets)
            distance_ids[pair_idx, b] = Int32(distance_bucket)
            distance_bias_base[1, pair_idx, b] = -Float32(distance_bucket - 1) / Float32(max_distance_denom)
            score_features[1, pair_idx, b] = Float32(span_scores_cpu[head_idx, b])
            score_features[2, pair_idx, b] = Float32(span_scores_cpu[tail_idx, b])
            if entity_mass_cpu !== nothing
                head_mass = Float32(entity_mass_cpu[head_start, b])
                tail_mass = Float32(entity_mass_cpu[tail_start, b])
                type_bias_base[1, pair_idx, b] = 0.5f0 * (head_mass + tail_mass)
                if entity_type_probs_cpu !== nothing
                    head_type_probs = @view(entity_type_probs_cpu[:, head_start, b])
                    tail_type_probs = @view(entity_type_probs_cpu[:, tail_start, b])
                    type_overlap = Float32(sum(head_type_probs .* tail_type_probs))
                    type_compat_bias_base[1, pair_idx, b] = type_overlap * sqrt(max(0.0f0, head_mass * tail_mass))
                end
            end
            if sentence_ids_cpu !== nothing
                sentence_column = if ndims(sentence_ids_cpu) == 1
                    sentence_ids_cpu
                elseif ndims(sentence_ids_cpu) == 2
                    @view(sentence_ids_cpu[:, b])
                else
                    throw(ArgumentError("sentence_ids must be rank-1 or rank-2; got ndims=$(ndims(sentence_ids_cpu))."))
                end
                seq_len = length(sentence_column)
                head_sentence = Int(sentence_column[clamp(head_start, 1, seq_len)])
                tail_sentence = Int(sentence_column[clamp(tail_start, 1, seq_len)])
                sentence_distance = abs(head_sentence - tail_sentence)
                sentence_distance_ids[pair_idx, b] = Int32(pair_distance_bucket_id(sentence_distance, distance_buckets))
                sentence_bias_base[1, pair_idx, b] = -Float32(min(sentence_distance, 4)) / 4.0f0
            end
            if local_radius > 0
                token_distance = abs(head_start - tail_start)
                local_bias_base[1, pair_idx, b] = max(0.0f0, 1.0f0 - Float32(token_distance) / Float32(local_radius + 1))
            end
        end
    end

    return (
        distance_ids,
        sentence_distance_ids,
        score_features,
        distance_bias_base,
        type_bias_base,
        sentence_bias_base,
        local_bias_base,
        type_compat_bias_base,
    )
end

function (layer::PairRetrievalHead)(inputs::Tuple, params, state)
    if length(inputs) == 5
        head_vectors, tail_vectors, pair_features, distance_ids, score_features = inputs
        retrieval_bias = nothing
        sentence_distance_ids = nothing
        sentence_embedding_scale = 0.0f0
        compatibility_scale = 0.0f0
    elseif length(inputs) == 6
        head_vectors, tail_vectors, pair_features, distance_ids, score_features, retrieval_bias = inputs
        sentence_distance_ids = nothing
        sentence_embedding_scale = 0.0f0
        compatibility_scale = 0.0f0
    elseif length(inputs) == 7
        head_vectors, tail_vectors, pair_features, distance_ids, sentence_distance_ids, score_features, sentence_embedding_scale = inputs
        retrieval_bias = nothing
        compatibility_scale = 0.0f0
    elseif length(inputs) == 8
        head_vectors, tail_vectors, pair_features, distance_ids, sentence_distance_ids, score_features, retrieval_bias, sentence_embedding_scale = inputs
        compatibility_scale = 0.0f0
    elseif length(inputs) == 9
        head_vectors, tail_vectors, pair_features, distance_ids, sentence_distance_ids, score_features, retrieval_bias, sentence_embedding_scale, compatibility_scale = inputs
    else
        throw(ArgumentError("PairRetrievalHead expects 5, 6, 7, 8, or 9 inputs, got $(length(inputs))."))
    end
    r = layer.retrieval_dimension
    pair_count = size(pair_features, 2)
    sentence_embedding_scale = Float32(sentence_embedding_scale)
    compatibility_scale = Float32(compatibility_scale)

    head_proj, head_state = layer.HeadProjection(head_vectors, params.HeadProjection, state.HeadProjection)
    tail_proj, tail_state = layer.TailProjection(tail_vectors, params.TailProjection, state.TailProjection)
    feature_proj, feature_state = layer.FeatureProjection(pair_features, params.FeatureProjection, state.FeatureProjection)

    distance_flat = vec(distance_ids)
    if head_vectors isa CUDA.CuArray && !(distance_flat isa CUDA.CuArray)
        distance_flat = CUDA.CuArray(distance_flat)
    end
    distance_emb, distance_state = layer.DistanceEmbedding(distance_flat, params.DistanceEmbedding, state.DistanceEmbedding)
    distance_emb = reshape(distance_emb, r, pair_count)
    if sentence_distance_ids !== nothing && sentence_embedding_scale != 0.0f0
        sentence_distance_flat = vec(sentence_distance_ids)
        if head_vectors isa CUDA.CuArray && !(sentence_distance_flat isa CUDA.CuArray)
            sentence_distance_flat = CUDA.CuArray(sentence_distance_flat)
        end
        sentence_emb, _ = layer.DistanceEmbedding(sentence_distance_flat, params.DistanceEmbedding, state.DistanceEmbedding)
        sentence_emb = reshape(sentence_emb, r, pair_count)
        distance_emb = distance_emb .+ sentence_embedding_scale .* sentence_emb
    end

    retrieval_inputs = vcat(
        head_proj .* tail_proj,
        abs.(head_proj .- tail_proj),
        feature_proj,
        distance_emb,
        score_features,
    )
    logits, output_state = layer.OutputProjection(retrieval_inputs, params.OutputProjection, state.OutputProjection)
    if compatibility_scale != 0.0f0
        compatibility_logits = sum(feature_proj .* (head_proj .* tail_proj), dims = 1) ./ sqrt(Float32(r))
        logits = logits .+ compatibility_scale .* compatibility_logits
    end
    if retrieval_bias !== nothing
        logits = logits .+ retrieval_bias
    end

    new_state = (
        HeadProjection = head_state,
        TailProjection = tail_state,
        DistanceEmbedding = distance_state,
        FeatureProjection = feature_state,
        OutputProjection = output_state,
    )
    return logits, new_state
end

function build_router_outputs(model::SwammaRelationExtractor, span_reps, params, state)
    d, max_spans, batch_size = size(span_reps)
    router_outputs, router_state = model.PairProposalHead(
        reshape(span_reps, d, :),
        params.PairProposalHead,
        state.PairProposalHead,
    )
    return reshape_router_outputs(router_outputs, max_spans, batch_size), router_state
end

function (model::SwammaRelationExtractor)(inputs::NamedTuple, params, state)
    token_ids = inputs.token_ids
    emit_evidence_diagnostics = hasproperty(inputs, :emit_evidence_diagnostics) ? Bool(inputs.emit_evidence_diagnostics) : false
    evidence_pooling_mode = hasproperty(inputs, :evidence_pooling_mode) ? inputs.evidence_pooling_mode : :token
    retrieval_distance_bias_scale = hasproperty(inputs, :retrieval_distance_bias_scale) ? Float32(inputs.retrieval_distance_bias_scale) : 0.0f0
    retrieval_type_bias_scale = hasproperty(inputs, :retrieval_type_bias_scale) ? Float32(inputs.retrieval_type_bias_scale) : 0.0f0
    retrieval_sentence_bias_scale = hasproperty(inputs, :retrieval_sentence_bias_scale) ? Float32(inputs.retrieval_sentence_bias_scale) : 0.0f0
    retrieval_local_bias_scale = hasproperty(inputs, :retrieval_local_bias_scale) ? Float32(inputs.retrieval_local_bias_scale) : 0.0f0
    retrieval_type_compat_bias_scale = hasproperty(inputs, :retrieval_type_compat_bias_scale) ? Float32(inputs.retrieval_type_compat_bias_scale) : 0.0f0
    retrieval_dot_bias_scale = hasproperty(inputs, :retrieval_dot_bias_scale) ? Float32(inputs.retrieval_dot_bias_scale) : 0.0f0
    retrieval_sentence_embedding_scale = hasproperty(inputs, :retrieval_sentence_embedding_scale) ? Float32(inputs.retrieval_sentence_embedding_scale) : 0.0f0
    retrieval_compatibility_scale = hasproperty(inputs, :retrieval_compatibility_scale) ? Float32(inputs.retrieval_compatibility_scale) : 0.0f0
    span_context_enabled = hasproperty(inputs, :span_context_enabled) ? Bool(inputs.span_context_enabled) : true
    edge_v2_semantic_topk = hasproperty(inputs, :edge_v2_semantic_topk) ? Int(inputs.edge_v2_semantic_topk) : 0
    edge_v2_reverse_topk = hasproperty(inputs, :edge_v2_reverse_topk) ? Int(inputs.edge_v2_reverse_topk) : 0
    edge_v2_global_reserve = hasproperty(inputs, :edge_v2_global_reserve) ? Int(inputs.edge_v2_global_reserve) : 0
    edge_v2_semantic_score_scale = hasproperty(inputs, :edge_v2_semantic_score_scale) ? Float32(inputs.edge_v2_semantic_score_scale) : 1.0f0
    edge_v2_span_score_scale = hasproperty(inputs, :edge_v2_span_score_scale) ? Float32(inputs.edge_v2_span_score_scale) : 1.0f0
    edge_v2_distance_penalty = hasproperty(inputs, :edge_v2_distance_penalty) ? Float32(inputs.edge_v2_distance_penalty) : 0.0f0
    edge_v2_require_mutual = hasproperty(inputs, :edge_v2_require_mutual) ? Bool(inputs.edge_v2_require_mutual) : false
    edge_v2_use_local_neighbors = hasproperty(inputs, :edge_v2_use_local_neighbors) ? Bool(inputs.edge_v2_use_local_neighbors) : true
    edge_v2_use_routed_buckets = hasproperty(inputs, :edge_v2_use_routed_buckets) ? Bool(inputs.edge_v2_use_routed_buckets) : true
    edge_v2_use_semantic_topk = hasproperty(inputs, :edge_v2_use_semantic_topk) ? Bool(inputs.edge_v2_use_semantic_topk) : true
    edge_v2_use_global_reserve = hasproperty(inputs, :edge_v2_use_global_reserve) ? Bool(inputs.edge_v2_use_global_reserve) : true
    span_context_use_adjacent = hasproperty(inputs, :span_context_use_adjacent) ? Bool(inputs.span_context_use_adjacent) : true
    span_context_use_sentence = hasproperty(inputs, :span_context_use_sentence) ? Bool(inputs.span_context_use_sentence) : true
    span_context_use_semantic = hasproperty(inputs, :span_context_use_semantic) ? Bool(inputs.span_context_use_semantic) : true
    span_context_sentence_ids = hasproperty(inputs, :span_context_sentence_ids) ? inputs.span_context_sentence_ids : nothing

    hidden, encoder_state = encode_tokens(model, token_ids, params, state)
    d, seq_len, batch_size = size(hidden)
    token_mask = hasproperty(inputs, :token_mask) ? inputs.token_mask : trues(seq_len, batch_size)
    if hidden isa CUDA.CuArray && !(token_mask isa CUDA.CuArray)
        token_mask = CUDA.CuArray(token_mask)
    end

    hidden_flat = reshape(hidden, d, :)
    hidden_flat, dropout_state = model.Dropout(hidden_flat, params.Dropout, state.Dropout)

    entity_logits_flat, entity_state = model.EntityHead(hidden_flat, params.EntityHead, state.EntityHead)
    entity_logits = reshape(entity_logits_flat, model.num_entity_labels, seq_len, batch_size)

    boundary_logits_flat, boundary_state = model.BoundaryHead(hidden_flat, params.BoundaryHead, state.BoundaryHead)
    boundary_logits = reshape(boundary_logits_flat, 2, seq_len, batch_size)

    spans, span_mask, span_scores, span_reps, span_state, mention_state = if hasproperty(inputs, :spans) && hasproperty(inputs, :span_mask)
        provided_span_reps, provided_span_state = build_span_representations(
            model,
            hidden,
            inputs.spans,
            inputs.span_mask,
            params,
            state,
        )
        learned_scores, _, provided_mention_state = score_span_representations(
            model,
            provided_span_reps,
            inputs.span_mask,
            params,
            state,
        )
        provided_scores = hasproperty(inputs, :span_scores) ? inputs.span_scores : learned_scores
        (inputs.spans, inputs.span_mask, provided_scores, provided_span_reps, provided_span_state, provided_mention_state)
    else
        proposed_spans, proposed_span_mask, proposed_span_scores, proposed_span_state, proposed_mention_state = propose_candidate_spans(
            model,
            hidden,
            entity_logits,
            boundary_logits;
            params = params,
            state = state,
            max_candidate_spans = model.max_candidate_spans,
            max_span_width = model.max_span_width,
        )
        if hidden isa CUDA.CuArray
            proposed_spans = CUDA.CuArray(proposed_spans)
            proposed_span_mask = CUDA.CuArray(proposed_span_mask)
            proposed_span_scores = CUDA.CuArray(proposed_span_scores)
        end
        proposed_span_reps, refined_span_state = build_span_representations(
            model,
            hidden,
            proposed_spans,
            proposed_span_mask,
            params,
            state,
        )
        (
            proposed_spans,
            proposed_span_mask,
            proposed_span_scores,
            proposed_span_reps,
            refined_span_state,
            proposed_mention_state,
        )
    end

    mention_logits, mention_output_state = if hasproperty(inputs, :mention_spans) && hasproperty(inputs, :mention_mask)
        mention_reps, mention_span_state = build_span_representations(
            model,
            hidden,
            inputs.mention_spans,
            inputs.mention_mask,
            params,
            state,
        )
        _, mention_scores_logits, mention_head_state = score_span_representations(
            model,
            mention_reps,
            inputs.mention_mask,
            params,
            state,
        )
        span_state = mention_span_state
        (mention_scores_logits, mention_head_state)
    else
        (reshape(span_scores, 1, size(span_scores, 1), batch_size), mention_state)
    end
    contextualized_span_reps, span_context_state = apply_span_context(
        model,
        span_reps,
        spans,
        span_mask,
        params,
        state,
        enabled = span_context_enabled,
        use_adjacent = span_context_use_adjacent,
        use_sentence = span_context_use_sentence,
        use_semantic = span_context_use_semantic,
        sentence_ids = span_context_sentence_ids,
    )
    relation_pairs, relation_mask, pair_proposal_state = if hasproperty(inputs, :relation_pairs) && hasproperty(inputs, :relation_mask)
        (inputs.relation_pairs, inputs.relation_mask, state.PairProposalHead)
    else
        proposal_budget = max(model.max_candidate_pairs, model.max_candidate_pairs * max(model.pair_overgenerate_factor, 1))
        router_outputs, router_state = pair_proposer_uses_router(model.pair_proposer_mode) ?
            build_router_outputs(model, contextualized_span_reps, params, state) :
            (nothing, state.PairProposalHead)
        semantic_outputs, _ = pair_proposer_uses_semantic_retrieval(model.pair_proposer_mode) ?
            build_retrieval_projection_outputs(model, contextualized_span_reps, params, state) :
            (nothing, nothing)
        proposed_pairs, proposed_mask = propose_relation_pairs(
            spans,
            span_mask,
            span_scores;
            max_candidate_pairs = proposal_budget,
            neighbor_radius = model.pair_neighbor_radius,
            proposer_mode = model.pair_proposer_mode,
            global_top_spans = model.pair_global_top_spans,
            anchor_top_spans = model.pair_anchor_top_spans,
            router_outputs = router_outputs,
            semantic_outputs = semantic_outputs,
            router_topk = model.pair_router_topk,
            router_routes_per_span = model.pair_router_routes_per_span,
            router_score_scale = model.pair_router_score_scale,
            edge_v2_semantic_topk = edge_v2_semantic_topk,
            edge_v2_reverse_topk = edge_v2_reverse_topk,
            edge_v2_global_reserve = edge_v2_global_reserve,
            edge_v2_semantic_score_scale = edge_v2_semantic_score_scale,
            edge_v2_span_score_scale = edge_v2_span_score_scale,
            edge_v2_distance_penalty = edge_v2_distance_penalty,
            edge_v2_require_mutual = edge_v2_require_mutual,
            edge_v2_use_local_neighbors = edge_v2_use_local_neighbors,
            edge_v2_use_routed_buckets = edge_v2_use_routed_buckets,
            edge_v2_use_semantic_topk = edge_v2_use_semantic_topk,
            edge_v2_use_global_reserve = edge_v2_use_global_reserve,
        )
        if hidden isa CUDA.CuArray
            proposed_pairs = CUDA.CuArray(proposed_pairs)
            proposed_mask = CUDA.CuArray(proposed_mask)
        end
        if size(proposed_pairs, 2) > model.max_candidate_pairs
            draft_head_vectors, draft_tail_vectors = gather_pair_span_vectors(contextualized_span_reps, proposed_pairs, proposed_mask)
            draft_pair_features = build_pair_features(draft_head_vectors, draft_tail_vectors)
            draft_distance_ids,
            draft_sentence_distance_ids,
            draft_score_features,
            draft_distance_bias_base,
            draft_type_bias_base,
            draft_sentence_bias_base,
            draft_local_bias_base,
            draft_type_compat_bias_base = ChainRulesCore.ignore_derivatives() do
                gather_pair_aux_features(
                    spans,
                    span_scores,
                    proposed_pairs,
                    proposed_mask;
                    distance_buckets = model.pair_distance_buckets,
                    entity_logits = entity_logits,
                    sentence_ids = span_context_sentence_ids,
                    local_radius = model.pair_neighbor_radius,
                )
            end
            draft_dot_bias_base = ChainRulesCore.ignore_derivatives() do
                dot_values = sum(draft_head_vectors .* draft_tail_vectors, dims = 1) ./ sqrt(Float32(size(draft_head_vectors, 1)))
                dot_cpu = dot_values isa CUDA.CuArray ? Array(dot_values) : dot_values
                reshape(Float32.(dot_cpu), 1, :)
            end
            draft_retrieval_bias = retrieval_distance_bias_scale .* reshape(draft_distance_bias_base, 1, :) .+
                                  retrieval_type_bias_scale .* reshape(draft_type_bias_base, 1, :) .+
                                  retrieval_sentence_bias_scale .* reshape(draft_sentence_bias_base, 1, :) .+
                                  retrieval_local_bias_scale .* reshape(draft_local_bias_base, 1, :) .+
                                  retrieval_type_compat_bias_scale .* reshape(draft_type_compat_bias_base, 1, :) .+
                                  retrieval_dot_bias_scale .* draft_dot_bias_base
            if draft_head_vectors isa CUDA.CuArray
                draft_distance_ids = CUDA.CuArray(draft_distance_ids)
                draft_sentence_distance_ids = CUDA.CuArray(draft_sentence_distance_ids)
                draft_score_features = CUDA.CuArray(draft_score_features)
                draft_retrieval_bias = CUDA.CuArray(draft_retrieval_bias)
            end
            draft_retrieval_flat, _ = model.PairRetrievalHead(
                (
                    draft_head_vectors,
                    draft_tail_vectors,
                    draft_pair_features,
                    draft_distance_ids,
                    draft_sentence_distance_ids,
                    reshape(draft_score_features, 2, :),
                    draft_retrieval_bias,
                    retrieval_sentence_embedding_scale,
                    retrieval_compatibility_scale,
                ),
                params.PairRetrievalHead,
                state.PairRetrievalHead,
            )
            draft_retrieval_logits = reshape(draft_retrieval_flat, 1, size(proposed_pairs, 2), batch_size)
            selected_pairs, selected_mask = select_top_relation_pairs(
                proposed_pairs,
                proposed_mask,
                draft_retrieval_logits;
                max_candidate_pairs = model.max_candidate_pairs,
            )
            (selected_pairs, selected_mask, router_state)
        else
            (proposed_pairs, proposed_mask, router_state)
        end
    end
    head_vectors, tail_vectors = gather_pair_span_vectors(contextualized_span_reps, relation_pairs, relation_mask)
    pair_features = build_pair_features(head_vectors, tail_vectors)
    distance_ids,
    sentence_distance_ids,
    score_features,
    distance_bias_base,
    type_bias_base,
    sentence_bias_base,
    local_bias_base,
    type_compat_bias_base = ChainRulesCore.ignore_derivatives() do
        gather_pair_aux_features(
            spans,
            span_scores,
            relation_pairs,
            relation_mask;
            distance_buckets = model.pair_distance_buckets,
            entity_logits = entity_logits,
            sentence_ids = span_context_sentence_ids,
            local_radius = model.pair_neighbor_radius,
        )
    end
    dot_bias_base = ChainRulesCore.ignore_derivatives() do
        dot_values = sum(head_vectors .* tail_vectors, dims = 1) ./ sqrt(Float32(size(head_vectors, 1)))
        dot_cpu = dot_values isa CUDA.CuArray ? Array(dot_values) : dot_values
        reshape(Float32.(dot_cpu), 1, :)
    end
    retrieval_bias = retrieval_distance_bias_scale .* reshape(distance_bias_base, 1, :) .+
                     retrieval_type_bias_scale .* reshape(type_bias_base, 1, :) .+
                     retrieval_sentence_bias_scale .* reshape(sentence_bias_base, 1, :) .+
                     retrieval_local_bias_scale .* reshape(local_bias_base, 1, :) .+
                     retrieval_type_compat_bias_scale .* reshape(type_compat_bias_base, 1, :) .+
                     retrieval_dot_bias_scale .* dot_bias_base
    if head_vectors isa CUDA.CuArray
        distance_ids = CUDA.CuArray(distance_ids)
        sentence_distance_ids = CUDA.CuArray(sentence_distance_ids)
        score_features = CUDA.CuArray(score_features)
        retrieval_bias = CUDA.CuArray(retrieval_bias)
    end
    retrieval_flat, retrieval_state = model.PairRetrievalHead(
        (
            head_vectors,
            tail_vectors,
            pair_features,
            distance_ids,
            sentence_distance_ids,
            reshape(score_features, 2, :),
            retrieval_bias,
            retrieval_sentence_embedding_scale,
            retrieval_compatibility_scale,
        ),
        params.PairRetrievalHead,
        state.PairRetrievalHead,
    )
    pair_feature_grid = reshape(pair_features, size(pair_features, 1), size(relation_pairs, 2), batch_size)
    evidence_outputs, evidence_state = model.PairEvidenceHead(
        (
            pair_feature_grid,
            hidden,
            token_mask,
            relation_mask,
            emit_evidence_diagnostics,
            evidence_pooling_mode,
        ),
        params.PairEvidenceHead,
        state.PairEvidenceHead,
    )
    evidence_summary = evidence_outputs.summary
    evidence_top_token_index = evidence_outputs.top_token_index
    evidence_attention_entropy = evidence_outputs.attention_entropy
    evidence_attention_max_weight = evidence_outputs.attention_max_weight
    evidence_summary_flat = reshape(evidence_summary, d, :)

    relation_logits_flat, relation_state = if model.relation_decoder_mode == :fused_evidence
        model.RelationHead(
            (
                head_vectors,
                tail_vectors,
                pair_features,
                evidence_summary_flat,
                retrieval_flat,
            ),
            params.RelationHead,
            state.RelationHead,
        )
    else
        model.RelationHead(
            (
                head_vectors,
                tail_vectors,
                pair_features,
                retrieval_flat,
            ),
            params.RelationHead,
            state.RelationHead,
        )
    end
    confidence_inputs = if model.relation_decoder_mode == :fused_evidence
        vcat(pair_features, evidence_summary_flat, retrieval_flat)
    else
        pair_features
    end
    confidence_flat, confidence_state = model.ConfidenceHead(confidence_inputs, params.ConfidenceHead, state.ConfidenceHead)

    retrieval_logits = reshape(retrieval_flat, 1, size(relation_pairs, 2), batch_size)
    relation_logits = reshape(relation_logits_flat, model.num_relations, size(relation_pairs, 2), batch_size)
    confidence_logits = reshape(confidence_flat, 1, size(relation_pairs, 2), batch_size)

    new_state = (
        TokenEmbedding = encoder_state.TokenEmbedding,
        PositionEmbedding = encoder_state.PositionEmbedding,
        TimeEmbedding = encoder_state.TimeEmbedding,
        Blocks = encoder_state.Blocks,
        HyperConnections = encoder_state.HyperConnections,
        RefinementBlocks = encoder_state.RefinementBlocks,
        Dropout = dropout_state,
        EntityHead = entity_state,
        BoundaryHead = boundary_state,
        SpanProjection = span_state,
        MentionHead = mention_output_state,
        SpanContextBlocks = span_context_state,
        PairProposalHead = pair_proposal_state,
        PairRetrievalHead = retrieval_state,
        PairEvidenceHead = evidence_state,
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
        mention_logits = mention_logits,
        span_representations = contextualized_span_reps,
        relation_pairs = relation_pairs,
        relation_mask = relation_mask,
        retrieval_loss_weight = model.pair_retrieval_loss_weight,
        retrieval_logits = retrieval_logits,
        evidence_summary = evidence_summary,
        evidence_top_token_index = evidence_top_token_index,
        evidence_attention_entropy = evidence_attention_entropy,
        evidence_attention_max_weight = evidence_attention_max_weight,
        relation_logits = relation_logits,
        confidence_logits = confidence_logits,
    ), new_state
end

function entity_cross_entropy(logits, labels; ignore_index::Int = -100)
    num_labels = size(logits, 1)
    logits_flat = reshape(logits, num_labels, :)
    labels_flat = vec(detach_constant(labels))
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
    targets_const = detach_constant(targets)
    valid_mask = targets_const .!= ignore_index
    count = Int(sum(valid_mask))
    count == 0 && return 0.0f0
    y = Float32.(targets_const)
    z = Float32.(logits)
    losses = NNlib.softplus.(z) .- z .* y
    return sum(losses .* Float32.(valid_mask)) / Float32(count)
end

function ChainRulesCore.rrule(::typeof(boundary_bce), logits, targets; ignore_index::Int = -100)
    targets_const = detach_constant(targets)
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
    labels_flat = vec(detach_constant(labels))
    mask_flat = vec(detach_constant(mask))
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
    targets_flat = Float32.(vec(detach_constant(targets)))
    mask_flat = vec(detach_constant(mask))
    count = Int(sum(mask_flat))
    count == 0 && return 0.0f0
    losses = NNlib.softplus.(logits_flat) .- logits_flat .* targets_flat
    return sum(losses .* Float32.(mask_flat)) / Float32(count)
end

function ChainRulesCore.rrule(::typeof(confidence_bce), logits, targets, mask)
    logits_shape = size(logits)
    z = Float32.(vec(logits))
    y = Float32.(vec(detach_constant(targets)))
    mask_const = vec(detach_constant(mask))
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

function mention_bce(logits, targets, mask)
    logits_flat = Float32.(vec(logits))
    targets_flat = Float32.(vec(detach_constant(targets)))
    mask_flat = vec(detach_constant(mask))
    count = Int(sum(mask_flat))
    count == 0 && return 0.0f0
    losses = NNlib.softplus.(logits_flat) .- logits_flat .* targets_flat
    return sum(losses .* Float32.(mask_flat)) / Float32(count)
end

function ChainRulesCore.rrule(::typeof(mention_bce), logits, targets, mask)
    logits_shape = size(logits)
    z = Float32.(vec(logits))
    y = Float32.(vec(detach_constant(targets)))
    mask_const = vec(detach_constant(mask))
    count = Int(sum(mask_const))
    if count == 0
        function mention_bce_zero_pullback(ȳ)
            logits_bar = ChainRulesCore.@thunk(zero(logits))
            return ChainRulesCore.NoTangent(), logits_bar, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
        end
        return 0.0f0, mention_bce_zero_pullback
    end

    mask_values = Float32.(mask_const)
    losses = NNlib.softplus.(z) .- z .* y
    value = sum(losses .* mask_values) / Float32(count)

    function mention_bce_pullback(ȳ)
        scale = Float32(ȳ) / Float32(count)
        logits_bar = ChainRulesCore.@thunk(reshape(scale .* mask_values .* (NNlib.sigmoid.(z) .- y), logits_shape))
        return ChainRulesCore.NoTangent(), logits_bar, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end

    return value, mention_bce_pullback
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

@inline function is_sentence_ending_token(token::AbstractString)
    token in (".", "!", "?", "。", "！", "？")
end

function infer_sentence_ids_from_tokens(tokens::Vector{String}, seq_len::Int)
    sentence_ids = ones(Int32, seq_len)
    sentence_id = Int32(1)
    for idx in 1:seq_len
        sentence_ids[idx] = sentence_id
        if is_sentence_ending_token(tokens[idx])
            sentence_id += 1
        end
    end
    return sentence_ids
end

function normalize_sentence_ids(raw_sentence_ids, seq_len::Int)
    seq_len <= 0 && return Int32[]
    isempty(raw_sentence_ids) && return Int32[]

    normalized = Int32[]
    sizehint!(normalized, seq_len)

    min_id = typemax(Int)
    for raw in raw_sentence_ids
        id = Int(raw)
        min_id = min(min_id, id)
    end
    offset = min_id <= 0 ? 1 : 0

    for raw in Iterators.take(raw_sentence_ids, seq_len)
        id = max(Int(raw) + offset, 1)
        push!(normalized, Int32(id))
    end
    if isempty(normalized)
        return Int32[]
    end
    if length(normalized) < seq_len
        append!(normalized, fill(normalized[end], seq_len - length(normalized)))
    end
    return normalized
end

function sentence_ids_for_row(row, tokens::Vector{String}, seq_len::Int)
    seq_len <= 0 && return Int32[]
    if haskey(row, :sentence_ids)
        normalized = normalize_sentence_ids(collect(row.sentence_ids), seq_len)
        !isempty(normalized) && return normalized
    end
    return infer_sentence_ids_from_tokens(tokens, seq_len)
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
    rng::AbstractRNG = Random.default_rng(),
)
    if entity_count < 2 || target_negatives <= 0
        return pair_idx
    end

    total_possible = entity_count * (entity_count - 1)
    start_idx = rand(rng, 1:total_possible)
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
    max_span_width::Int,
    hard_negative_ratio::Float32 = 0.0f0,
    mention_negative_ratio::Float32 = 1.0f0,
    rng::AbstractRNG = Random.default_rng(),
)
    batch_size = length(rows)
    token_ids = fill(vocab["[PAD]"], max_len, batch_size)
    token_mask = falses(max_len, batch_size)
    sentence_ids = ones(Int32, max_len, batch_size)
    entity_labels = fill(-100, max_len, batch_size)
    boundary_labels = fill(-100, 2, max_len, batch_size)
    spans = zeros(Int, 2, max_candidate_spans, batch_size)
    span_mask = falses(max_candidate_spans, batch_size)
    mention_spans = zeros(Int, 2, max_candidate_spans, batch_size)
    mention_mask = falses(max_candidate_spans, batch_size)
    mention_labels = zeros(Float32, max_candidate_spans, batch_size)
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
        token_mask[1:seq_len, b] .= true
        row_sentence_ids = sentence_ids_for_row(row, tokens, seq_len)
        if !isempty(row_sentence_ids)
            sentence_ids[1:seq_len, b] .= row_sentence_ids
        end
        if seq_len < max_len
            pad_fill = seq_len > 0 ? sentence_ids[seq_len, b] : Int32(1)
            sentence_ids[(seq_len + 1):max_len, b] .= pad_fill
        end

        entities = haskey(row, :entities) ? collect(row.entities) : Any[]
        positive_mentions = Tuple{Int, Int}[]
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
            mention_spans[1, i, b] = start_idx
            mention_spans[2, i, b] = stop_idx
            mention_mask[i, b] = true
            mention_labels[i, b] = 1.0f0
            push!(positive_mentions, (start_idx, stop_idx))
        end

        positive_mention_set = Set(positive_mentions)
        mention_idx = min(length(positive_mentions), max_candidate_spans)
        if mention_idx < max_candidate_spans && seq_len > 0
            negative_mentions = filter(
                span -> !(span in positive_mention_set),
                enumerate_candidate_span_list(seq_len, max_span_width),
            )
            available_negatives = length(negative_mentions)
            target_negatives = if !isempty(positive_mentions)
                ceil(Int, length(positive_mentions) * mention_negative_ratio)
            else
                min(available_negatives, max(1, round(Int, mention_negative_ratio)))
            end
            target_negatives = min(target_negatives, available_negatives, max_candidate_spans - mention_idx)

            if target_negatives > 0
                sampled_indices = randperm(rng, length(negative_mentions))[1:target_negatives]
                for negative_idx in sampled_indices
                    mention_idx += 1
                    start_idx, stop_idx = negative_mentions[negative_idx]
                    mention_spans[1, mention_idx, b] = start_idx
                    mention_spans[2, mention_idx, b] = stop_idx
                    mention_mask[mention_idx, b] = true
                end
            end
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
                rng,
            )
        end
    end

    return (
        token_ids = token_ids,
        token_mask = token_mask,
        sentence_ids = sentence_ids,
        entity_labels = entity_labels,
        boundary_labels = boundary_labels,
        spans = spans,
        span_mask = span_mask,
        mention_spans = mention_spans,
        mention_mask = mention_mask,
        mention_labels = mention_labels,
        relation_pairs = relation_pairs,
        relation_labels = relation_labels,
        relation_mask = relation_mask,
        relation_targets = relation_targets,
    )
end

export RelationExtractionConfig, SwammaRelationExtractor
export load_relation_extraction_config, print_relation_extraction_summary
export entity_cross_entropy, boundary_bce, relation_cross_entropy, confidence_bce, mention_bce
export load_rebel_jsonl, build_token_vocab, build_entity_label_space, build_relation_label_space
export prepare_rebel_batch, DEFAULT_ENTITY_LABELS, DEFAULT_ENTITY_TYPES

end # module RelationExtraction
