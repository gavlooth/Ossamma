using Lux
using Random
using Test

include("../src/Swamma.jl")
using .Swamma

const SW = Swamma

@testset "Relation Extraction Auto Proposal" begin
    config = SW.RelationExtractionConfig(
        vocab_size = 128,
        max_sequence_length = 16,
        embedding_dimension = 32,
        number_of_heads = 4,
        number_of_layers = 2,
        num_entity_labels = 5,
        num_relations = 4,
        time_dimension = 16,
        state_dimension = 32,
        window_size = 2,
        dropout_rate = 0.0f0,
        max_candidate_spans = 6,
        max_candidate_pairs = 8,
        max_span_width = 4,
        biaffine_rank = 8,
        pair_neighbor_radius = 2,
    )

    model = SW.SwammaRelationExtractor(config)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    token_ids = reshape(rand(rng, 1:config.vocab_size, 12), 12, 1)
    outputs, _ = model((token_ids = token_ids,), ps, st)

    @test size(outputs.entity_logits) == (config.num_entity_labels, 12, 1)
    @test size(outputs.boundary_logits) == (2, 12, 1)
    @test size(outputs.mention_logits) == (1, config.max_candidate_spans, 1)
    @test size(outputs.spans) == (2, config.max_candidate_spans, 1)
    @test size(outputs.relation_pairs) == (2, config.max_candidate_pairs, 1)
    @test size(outputs.retrieval_logits) == (1, config.max_candidate_pairs, 1)
    @test size(outputs.evidence_summary) == (config.embedding_dimension, config.max_candidate_pairs, 1)
    @test size(outputs.relation_logits) == (config.num_relations, config.max_candidate_pairs, 1)
    @test sum(outputs.span_mask[:, 1]) <= config.max_candidate_spans
    @test sum(outputs.relation_mask[:, 1]) <= config.max_candidate_pairs
end

@testset "Relation Extraction Sparse Routed Proposal" begin
    config = SW.RelationExtractionConfig(
        vocab_size = 128,
        max_sequence_length = 16,
        embedding_dimension = 32,
        number_of_heads = 4,
        number_of_layers = 2,
        num_entity_labels = 5,
        num_relations = 4,
        time_dimension = 16,
        state_dimension = 32,
        window_size = 2,
        dropout_rate = 0.0f0,
        max_candidate_spans = 6,
        max_candidate_pairs = 8,
        max_span_width = 4,
        biaffine_rank = 8,
        pair_neighbor_radius = 2,
        pair_proposer_mode = :sparse_hybrid,
        pair_global_top_spans = 4,
        pair_router_dimension = 8,
        pair_router_buckets = 3,
        pair_router_topk = 3,
        pair_router_routes_per_span = 2,
        pair_router_score_scale = 0.5f0,
    )

    model = SW.SwammaRelationExtractor(config)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    token_ids = reshape(rand(rng, 1:config.vocab_size, 12), 12, 1)
    outputs, _ = model((token_ids = token_ids,), ps, st)

    @test size(outputs.mention_logits) == (1, config.max_candidate_spans, 1)
    @test size(outputs.relation_pairs) == (2, config.max_candidate_pairs, 1)
    @test size(outputs.retrieval_logits) == (1, config.max_candidate_pairs, 1)
    @test size(outputs.relation_logits) == (config.num_relations, config.max_candidate_pairs, 1)
    @test sum(outputs.relation_mask[:, 1]) <= config.max_candidate_pairs
end

@testset "Relation Extraction Edge Retrieval v2 Proposal" begin
    config = SW.RelationExtractionConfig(
        vocab_size = 128,
        max_sequence_length = 16,
        embedding_dimension = 32,
        number_of_heads = 4,
        number_of_layers = 2,
        num_entity_labels = 5,
        num_relations = 4,
        time_dimension = 16,
        state_dimension = 32,
        window_size = 2,
        dropout_rate = 0.0f0,
        max_candidate_spans = 6,
        max_candidate_pairs = 8,
        max_span_width = 4,
        biaffine_rank = 8,
        pair_neighbor_radius = 2,
        pair_proposer_mode = :edge_retrieval_v2,
        pair_global_top_spans = 4,
        pair_router_dimension = 8,
        pair_router_buckets = 3,
        pair_router_topk = 3,
        pair_router_routes_per_span = 2,
        pair_router_score_scale = 0.5f0,
    )

    model = SW.SwammaRelationExtractor(config)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    token_ids = reshape(rand(rng, 1:config.vocab_size, 12), 12, 1)
    outputs, _ = model((token_ids = token_ids,), ps, st)

    @test size(outputs.mention_logits) == (1, config.max_candidate_spans, 1)
    @test size(outputs.relation_pairs) == (2, config.max_candidate_pairs, 1)
    @test size(outputs.retrieval_logits) == (1, config.max_candidate_pairs, 1)
    @test size(outputs.relation_logits) == (config.num_relations, config.max_candidate_pairs, 1)
    @test sum(outputs.relation_mask[:, 1]) <= config.max_candidate_pairs
end

@testset "Edge Retrieval v2 Family Gating" begin
    spans = zeros(Int32, 2, 4, 1)
    spans[:, 1, 1] .= Int32[1, 1]
    spans[:, 2, 1] .= Int32[3, 3]
    spans[:, 3, 1] .= Int32[6, 6]
    spans[:, 4, 1] .= Int32[9, 9]
    span_mask = trues(4, 1)
    span_scores = reshape(Float32[4.0, 3.0, 2.0, 1.0], 4, 1)
    semantic_outputs = (
        head = zeros(Float32, 2, 4, 1),
        tail = zeros(Float32, 2, 4, 1),
    )

    local_pairs, local_mask = SW.RelationExtraction.propose_relation_pairs(
        spans,
        span_mask,
        span_scores;
        max_candidate_pairs = 16,
        neighbor_radius = 1,
        proposer_mode = :edge_retrieval_v2,
        semantic_outputs = semantic_outputs,
        edge_v2_use_local_neighbors = true,
        edge_v2_use_routed_buckets = false,
        edge_v2_use_semantic_topk = false,
        edge_v2_use_global_reserve = false,
    )
    local_set = Set(
        (Int(local_pairs[1, i, 1]), Int(local_pairs[2, i, 1]))
        for i in findall(@view(local_mask[:, 1]))
    )
    @test (1, 2) in local_set
    @test (2, 1) in local_set

    router_outputs = (
        head_router = reshape(Float32[
            1, 0,
            0, 1,
            0, 1,
            1, 0,
        ], 2, 4, 1),
        tail_router = reshape(Float32[
            1, 0,
            0, 1,
            0, 1,
            1, 0,
        ], 2, 4, 1),
        bucket_logits = reshape(Float32[
            10, 0,
            0, 10,
            0, 10,
            10, 0,
        ], 2, 4, 1),
    )
    routed_pairs, routed_mask = SW.RelationExtraction.propose_relation_pairs(
        spans,
        span_mask,
        span_scores;
        max_candidate_pairs = 16,
        neighbor_radius = 1,
        proposer_mode = :edge_retrieval_v2,
        router_outputs = router_outputs,
        semantic_outputs = semantic_outputs,
        router_topk = 4,
        router_routes_per_span = 1,
        edge_v2_use_local_neighbors = false,
        edge_v2_use_routed_buckets = true,
        edge_v2_use_semantic_topk = false,
        edge_v2_use_global_reserve = false,
    )
    routed_set = Set(
        (Int(routed_pairs[1, i, 1]), Int(routed_pairs[2, i, 1]))
        for i in findall(@view(routed_mask[:, 1]))
    )
    @test (1, 4) in routed_set || (4, 1) in routed_set
    @test !((1, 2) in routed_set)
end

@testset "Relation Extraction Span Context" begin
    config = SW.RelationExtractionConfig(
        vocab_size = 128,
        max_sequence_length = 16,
        embedding_dimension = 32,
        number_of_heads = 4,
        number_of_layers = 2,
        num_entity_labels = 5,
        num_relations = 4,
        time_dimension = 16,
        state_dimension = 32,
        window_size = 2,
        dropout_rate = 0.0f0,
        max_candidate_spans = 6,
        max_candidate_pairs = 8,
        max_span_width = 4,
        span_context_layers = 1,
        span_context_neighbor_radius = 1,
        span_context_topk = 2,
        biaffine_rank = 8,
        pair_neighbor_radius = 2,
    )

    model = SW.SwammaRelationExtractor(config)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    token_ids = reshape(rand(rng, 1:config.vocab_size, 12), 12, 1)
    sentence_ids = reshape(Int32[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 12, 1)
    outputs, _ = model(
        (
            token_ids = token_ids,
            span_context_enabled = true,
            span_context_use_adjacent = false,
            span_context_use_sentence = true,
            span_context_use_semantic = false,
            span_context_sentence_ids = sentence_ids,
        ),
        ps,
        st,
    )

    @test size(outputs.span_representations) == (config.embedding_dimension, config.max_candidate_spans, 1)
    @test size(outputs.retrieval_logits) == (1, config.max_candidate_pairs, 1)
    @test size(outputs.evidence_summary) == (config.embedding_dimension, config.max_candidate_pairs, 1)
    @test size(outputs.relation_logits) == (config.num_relations, config.max_candidate_pairs, 1)

    outputs_disabled, _ = model(
        (
            token_ids = token_ids,
            span_context_enabled = false,
            span_context_use_adjacent = false,
            span_context_use_sentence = true,
            span_context_use_semantic = false,
            span_context_sentence_ids = sentence_ids,
        ),
        ps,
        st,
    )
    @test size(outputs_disabled.span_representations) == (config.embedding_dimension, config.max_candidate_spans, 1)
    @test size(outputs_disabled.retrieval_logits) == (1, config.max_candidate_pairs, 1)
end

@testset "Span Context Edge Family Controls" begin
    scores = zeros(Float32, 4, 4, 1)
    scores[1, 3, 1] = 10.0f0
    scores[2, 4, 1] = 9.0f0
    scores[3, 1, 1] = 8.0f0
    scores[4, 2, 1] = 7.0f0

    spans = zeros(Int32, 2, 4, 1)
    spans[:, 1, 1] .= Int32[1, 1]
    spans[:, 2, 1] .= Int32[2, 2]
    spans[:, 3, 1] .= Int32[4, 4]
    spans[:, 4, 1] .= Int32[5, 5]
    span_mask = trues(4, 1)
    sentence_ids = reshape(Int32[1, 1, 1, 2, 2], 5, 1)

    sentence_adj = SW.RelationExtraction.build_span_context_adjacency(
        scores,
        spans,
        span_mask;
        neighbor_radius = 0,
        semantic_topk = 0,
        use_adjacent = false,
        use_sentence = true,
        use_semantic = false,
        sentence_ids = sentence_ids,
    )
    @test sentence_adj[1, 2, 1]
    @test sentence_adj[2, 1, 1]
    @test sentence_adj[3, 4, 1]
    @test sentence_adj[4, 3, 1]
    @test !sentence_adj[1, 3, 1]

    semantic_adj = SW.RelationExtraction.build_span_context_adjacency(
        scores,
        spans,
        span_mask;
        neighbor_radius = 0,
        semantic_topk = 1,
        use_adjacent = false,
        use_sentence = false,
        use_semantic = true,
        sentence_ids = nothing,
    )
    @test semantic_adj[1, 3, 1]
    @test semantic_adj[2, 4, 1]
    @test semantic_adj[3, 1, 1]
    @test semantic_adj[4, 2, 1]

    self_only_adj = SW.RelationExtraction.build_span_context_adjacency(
        scores,
        spans,
        span_mask;
        neighbor_radius = 0,
        semantic_topk = 0,
        use_adjacent = false,
        use_sentence = false,
        use_semantic = false,
        sentence_ids = sentence_ids,
    )
    @test all(self_only_adj[i, i, 1] for i in 1:4)
    @test !self_only_adj[1, 2, 1]
    @test !self_only_adj[2, 4, 1]
end

@testset "Pair Aux Bias Bases" begin
    spans = zeros(Int32, 2, 2, 1)
    spans[:, 1, 1] .= Int32[1, 1]
    spans[:, 2, 1] .= Int32[4, 4]
    span_scores = reshape(Float32[1.0, 0.5], 2, 1)
    relation_pairs = zeros(Int32, 2, 2, 1)
    relation_pairs[:, 1, 1] .= Int32[1, 2]
    relation_pairs[:, 2, 1] .= Int32[1, 1]
    relation_mask = trues(2, 1)
    sentence_ids = reshape(Int32[1, 1, 2, 2], 4, 1)
    entity_logits = fill(-6.0f0, 5, 4, 1)
    entity_logits[2, 1, 1] = 6.0f0
    entity_logits[2, 4, 1] = 6.0f0

    _, sentence_distance_ids, _, _, _, sentence_bias, local_bias, type_compat_bias = SW.RelationExtraction.gather_pair_aux_features(
        spans,
        span_scores,
        relation_pairs,
        relation_mask;
        distance_buckets = 16,
        entity_logits = entity_logits,
        sentence_ids = sentence_ids,
        local_radius = 2,
    )

    @test sentence_bias[1, 1, 1] < 0.0f0
    @test sentence_bias[1, 2, 1] == 0.0f0
    @test sentence_distance_ids[1, 1] > 1
    @test sentence_distance_ids[2, 1] == 1
    @test local_bias[1, 1, 1] == 0.0f0
    @test local_bias[1, 2, 1] == 1.0f0
    @test type_compat_bias[1, 1, 1] > 0.0f0
end

@testset "Pair Retrieval Compatibility Scale Hook" begin
    config = SW.RelationExtractionConfig(
        vocab_size = 128,
        max_sequence_length = 16,
        embedding_dimension = 32,
        number_of_heads = 4,
        number_of_layers = 2,
        num_entity_labels = 5,
        num_relations = 4,
        time_dimension = 16,
        state_dimension = 32,
        window_size = 2,
        dropout_rate = 0.0f0,
        max_candidate_spans = 4,
        max_candidate_pairs = 4,
        max_span_width = 4,
        biaffine_rank = 8,
        pair_neighbor_radius = 2,
    )

    model = SW.SwammaRelationExtractor(config)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    token_ids = reshape(rand(rng, 1:config.vocab_size, 8), 8, 1)
    spans = zeros(Int32, 2, 2, 1)
    spans[:, 1, 1] .= Int32[1, 1]
    spans[:, 2, 1] .= Int32[4, 4]
    span_mask = trues(2, 1)
    span_scores = reshape(Float32[1.0, 0.8], 2, 1)
    relation_pairs = zeros(Int32, 2, 2, 1)
    relation_pairs[:, 1, 1] .= Int32[1, 2]
    relation_pairs[:, 2, 1] .= Int32[2, 1]
    relation_mask = trues(2, 1)

    base_inputs = (
        token_ids = token_ids,
        spans = spans,
        span_mask = span_mask,
        span_scores = span_scores,
        relation_pairs = relation_pairs,
        relation_mask = relation_mask,
    )
    outputs_default, _ = model(base_inputs, ps, st)
    outputs_zero, _ = model(merge(base_inputs, (retrieval_compatibility_scale = 0.0f0,)), ps, st)
    outputs_scaled, _ = model(merge(base_inputs, (retrieval_compatibility_scale = 0.5f0,)), ps, st)

    @test outputs_default.retrieval_logits ≈ outputs_zero.retrieval_logits atol = 1.0f-6 rtol = 1.0f-6
    diff = abs.(outputs_scaled.retrieval_logits .- outputs_zero.retrieval_logits)
    @test maximum(Array(diff)) > 1.0f-6
end

@testset "Relation Extraction Residual Fused Decoder" begin
    config = SW.RelationExtractionConfig(
        vocab_size = 128,
        max_sequence_length = 16,
        embedding_dimension = 32,
        number_of_heads = 4,
        number_of_layers = 2,
        num_entity_labels = 5,
        num_relations = 4,
        time_dimension = 16,
        state_dimension = 32,
        window_size = 2,
        dropout_rate = 0.0f0,
        max_candidate_spans = 6,
        max_candidate_pairs = 8,
        max_span_width = 4,
        biaffine_rank = 8,
        pair_neighbor_radius = 2,
        relation_decoder_mode = :fused_residual,
        relation_decoder_residual_scale = 0.25f0,
    )

    model = SW.SwammaRelationExtractor(config)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    token_ids = reshape(rand(rng, 1:config.vocab_size, 12), 12, 1)
    outputs, _ = model((token_ids = token_ids,), ps, st)

    @test size(outputs.retrieval_logits) == (1, config.max_candidate_pairs, 1)
    @test size(outputs.relation_logits) == (config.num_relations, config.max_candidate_pairs, 1)
end

@testset "Relation Extraction Evidence Decoder" begin
    config = SW.RelationExtractionConfig(
        vocab_size = 128,
        max_sequence_length = 16,
        embedding_dimension = 32,
        number_of_heads = 4,
        number_of_layers = 2,
        num_entity_labels = 5,
        num_relations = 4,
        time_dimension = 16,
        state_dimension = 32,
        window_size = 2,
        dropout_rate = 0.0f0,
        max_candidate_spans = 6,
        max_candidate_pairs = 8,
        max_span_width = 4,
        biaffine_rank = 8,
        pair_neighbor_radius = 2,
        pair_evidence_dimension = 16,
        relation_decoder_mode = :fused_evidence,
        relation_decoder_residual_scale = 0.25f0,
    )

    model = SW.SwammaRelationExtractor(config)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    token_ids = reshape(rand(rng, 1:config.vocab_size, 12), 12, 1)
    token_mask = trues(12, 1)
    outputs, _ = model((token_ids = token_ids, token_mask = token_mask), ps, st)

    @test size(outputs.evidence_summary) == (config.embedding_dimension, config.max_candidate_pairs, 1)
    @test size(outputs.retrieval_logits) == (1, config.max_candidate_pairs, 1)
    @test size(outputs.relation_logits) == (config.num_relations, config.max_candidate_pairs, 1)
end

@testset "prepare_rebel_batch Sampled Negatives" begin
    rows = [
        (
            tokens = ["Ada", "joined", "Acme", "Conference"],
            entities = [
                (start = 1, stop = 1, label = "PERSON"),
                (start = 3, stop = 3, label = "ORGANIZATION"),
                (start = 4, stop = 4, label = "EVENT"),
            ],
            relations = [
                (head = 1, tail = 2, label = "WORKS_FOR"),
            ],
        ),
    ]

    vocab = SW.build_token_vocab(rows; max_vocab = 64)
    entity_label_to_id = SW.build_entity_label_space(rows)
    relation_label_to_id = SW.build_relation_label_space(rows)

    batch = SW.prepare_rebel_batch(
        rows,
        vocab,
        entity_label_to_id,
        relation_label_to_id;
        max_len = 8,
        max_candidate_spans = 8,
        max_candidate_pairs = 8,
        max_span_width = 4,
        hard_negative_ratio = 2.0f0,
    )

    active_pairs = findall(@view(batch.relation_mask[:, 1]))
    @test length(active_pairs) == 3
    @test sum(batch.mention_mask[:, 1]) >= 3
    @test sum(batch.mention_labels[:, 1]) == 3.0f0

    positive_pairs = Set{Tuple{Int, Int}}()
    negative_pairs = Set{Tuple{Int, Int}}()
    for idx in active_pairs
        pair = (batch.relation_pairs[1, idx, 1], batch.relation_pairs[2, idx, 1])
        if batch.relation_targets[idx, 1] == 1.0f0
            push!(positive_pairs, pair)
        else
            push!(negative_pairs, pair)
        end
    end

    @test positive_pairs == Set([(1, 2)])
    @test length(negative_pairs) == 2
    @test isempty(intersect(positive_pairs, negative_pairs))
    @test size(batch.sentence_ids) == (8, 1)
    @test all(batch.sentence_ids[1:4, 1] .== Int32(1))

    rng_a = MersenneTwister(11)
    rng_b = MersenneTwister(11)
    det_a = SW.prepare_rebel_batch(
        rows,
        vocab,
        entity_label_to_id,
        relation_label_to_id;
        max_len = 8,
        max_candidate_spans = 8,
        max_candidate_pairs = 8,
        max_span_width = 4,
        hard_negative_ratio = 2.0f0,
        mention_negative_ratio = 2.0f0,
        rng = rng_a,
    )
    det_b = SW.prepare_rebel_batch(
        rows,
        vocab,
        entity_label_to_id,
        relation_label_to_id;
        max_len = 8,
        max_candidate_spans = 8,
        max_candidate_pairs = 8,
        max_span_width = 4,
        hard_negative_ratio = 2.0f0,
        mention_negative_ratio = 2.0f0,
        rng = rng_b,
    )

    @test det_a.mention_spans == det_b.mention_spans
    @test det_a.mention_mask == det_b.mention_mask
    @test det_a.relation_pairs == det_b.relation_pairs
    @test det_a.relation_mask == det_b.relation_mask

    punctuation_rows = [
        (
            tokens = ["Ada", ".", "Bob", "!"],
            entities = Any[],
            relations = Any[],
        ),
    ]
    punctuation_vocab = SW.build_token_vocab(punctuation_rows; max_vocab = 64)
    punctuation_entity_labels = SW.build_entity_label_space(punctuation_rows)
    punctuation_relation_labels = SW.build_relation_label_space(punctuation_rows)
    punct_batch = SW.prepare_rebel_batch(
        punctuation_rows,
        punctuation_vocab,
        punctuation_entity_labels,
        punctuation_relation_labels;
        max_len = 6,
        max_candidate_spans = 4,
        max_candidate_pairs = 4,
        max_span_width = 3,
    )
    @test vec(punct_batch.sentence_ids[1:4, 1]) == Int32[1, 1, 2, 2]

    explicit_sentence_rows = [
        (
            tokens = ["Ada", "met", "Bob", "."],
            sentence_ids = [0, 0, 1, 1],
            entities = Any[],
            relations = Any[],
        ),
    ]
    explicit_vocab = SW.build_token_vocab(explicit_sentence_rows; max_vocab = 64)
    explicit_entity_labels = SW.build_entity_label_space(explicit_sentence_rows)
    explicit_relation_labels = SW.build_relation_label_space(explicit_sentence_rows)
    explicit_batch = SW.prepare_rebel_batch(
        explicit_sentence_rows,
        explicit_vocab,
        explicit_entity_labels,
        explicit_relation_labels;
        max_len = 6,
        max_candidate_spans = 4,
        max_candidate_pairs = 4,
        max_span_width = 3,
    )
    @test vec(explicit_batch.sentence_ids[1:4, 1]) == Int32[1, 1, 2, 2]
end
