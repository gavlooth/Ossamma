#!/usr/bin/env julia
"""
Smoke test: load model + sample data, run one forward pass on CPU,
decode predictions, print human-readable output.

Usage:
    julia --project=. scripts/smoke_inference_re.jl --config configs/redfm_1b_distill_qwen7b.toml
"""

include(joinpath(@__DIR__, "..", "src", "Swamma.jl"))
using .Swamma.RelationExtraction:
    RelationExtractionConfig, SwammaRelationExtractor,
    load_relation_extraction_config, load_rebel_jsonl,
    build_token_vocab, build_entity_label_space, build_relation_label_space,
    prepare_rebel_batch

using Lux
using Random
using Printf

function main()
    config_path = "configs/redfm_1b_distill_qwen7b.toml"
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--config" && i < length(ARGS)
            config_path = ARGS[i+1]
            i += 2
        else
            i += 1
        end
    end

    println("=" ^ 60)
    println("Inference Smoke Test")
    println("=" ^ 60)
    println("Config: $config_path")

    # Load config — override to small dimensions for CPU smoke test
    base_config = load_relation_extraction_config(config_path)

    # Load a few rows of data
    data_path = base_config.vocab_size > 32000 ? "data/rebel_full/train.jsonl" : "data/rebel/train.jsonl"
    if !isfile(data_path)
        # Fallback
        for p in ["data/rebel_full/train.jsonl", "data/rebel/train.jsonl"]
            if isfile(p)
                data_path = p
                break
            end
        end
    end
    println("Data: $data_path")

    rows = load_rebel_jsonl(data_path)
    sample_rows = rows[1:min(2, length(rows))]
    println("Sample rows: $(length(sample_rows))")

    vocab = build_token_vocab(rows[1:min(1000, length(rows))]; max_vocab = base_config.vocab_size)
    entity_label_to_id = build_entity_label_space(sample_rows)
    relation_label_to_id = build_relation_label_space(rows[1:min(1000, length(rows))])

    # Build a tiny model (override dimensions for CPU speed)
    small_config = RelationExtractionConfig(
        vocab_size = length(vocab),
        max_sequence_length = 128,
        embedding_dimension = 64,
        number_of_heads = 4,
        number_of_layers = 2,
        num_entity_labels = length(entity_label_to_id),
        num_relations = length(relation_label_to_id),
        time_dimension = 16,
        state_dimension = 64,
        window_size = 8,
        max_candidate_spans = 32,
        max_candidate_pairs = 32,
        max_span_width = 6,
        biaffine_rank = 16,
        pair_neighbor_radius = 3,
        pair_proposer_mode = :local,
        relation_decoder_mode = :pair_evidence_mlp,
        pair_evidence_dimension = 16,
        pair_retrieval_dimension = 16,
        mention_score_mode = :learned,
        dropout_rate = 0.0f0,
        use_ffn = true,
        ffn_expansion = 1.333333f0,
    )

    println("\nBuilding tiny model (d=64, L=2, H=4) for CPU smoke test...")
    model = SwammaRelationExtractor(small_config)
    rng = MersenneTwister(42)
    params, state = Lux.setup(rng, model)

    param_count = 0
    Lux.fmap(params) do x
        if x isa AbstractArray
            param_count += length(x)
        end
        x
    end
    println("Parameters: $(round(param_count / 1e6, digits=2))M")

    # Prepare batch
    println("\nPreparing batch from sample rows...")
    batch = prepare_rebel_batch(
        sample_rows,
        vocab,
        entity_label_to_id,
        relation_label_to_id;
        max_len = 128,
        max_candidate_spans = 32,
        max_candidate_pairs = 32,
        max_span_width = 6,
        rng = rng,
    )

    inputs = (
        token_ids = batch.token_ids,
        token_mask = batch.token_mask,
        spans = batch.spans,
        span_mask = batch.span_mask,
        span_scores = Float32.(batch.span_mask),
        mention_spans = batch.mention_spans,
        mention_mask = batch.mention_mask,
        relation_pairs = batch.relation_pairs,
        relation_mask = batch.relation_mask,
    )

    println("\nInput shapes:")
    println("  token_ids:      $(size(inputs.token_ids))")
    println("  spans:          $(size(inputs.spans))")
    println("  relation_pairs: $(size(inputs.relation_pairs))")

    # Forward pass
    println("\nRunning forward pass (CPU)...")
    t0 = time()
    outputs, new_state = model(inputs, params, state)
    dt = time() - t0
    @printf("Forward pass: %.1f ms\n", dt * 1000)

    println("\nOutput shapes:")
    println("  entity_logits:     $(size(outputs.entity_logits))")
    println("  boundary_logits:   $(size(outputs.boundary_logits))")
    println("  relation_logits:   $(size(outputs.relation_logits))")
    println("  confidence_logits: $(size(outputs.confidence_logits))")
    println("  spans:             $(size(outputs.spans))")
    println("  relation_pairs:    $(size(outputs.relation_pairs))")

    # Decode predictions for batch element 1
    id_to_entity_label = Dict(v => k for (k, v) in entity_label_to_id)
    id_to_relation_label = Dict(v => k for (k, v) in relation_label_to_id)
    no_relation_id = get(relation_label_to_id, "NO_RELATION", 1)

    batch_size = size(outputs.entity_logits, 3)
    seq_len = size(outputs.entity_logits, 2)
    num_rels = size(outputs.relation_logits, 1)

    println("\nDecoding batch element 1:")
    println("  Sequence length: $seq_len")
    println("  Num relations classes: $num_rels (including NO_RELATION=$no_relation_id)")

    # Entity predictions (argmax per token)
    entity_preds = outputs.entity_logits[:, :, 1]
    entity_ids = [findmax(entity_preds[:, t])[2] for t in 1:seq_len]
    non_o = [(t, id_to_entity_label[entity_ids[t]]) for t in 1:seq_len if get(id_to_entity_label, entity_ids[t], "O") != "O"]
    println("\n  Entity predictions (non-O tokens):")
    for (t, label) in non_o[1:min(10, length(non_o))]
        println("    token $t: $label")
    end
    if length(non_o) > 10
        println("    ... $(length(non_o) - 10) more")
    end
    if isempty(non_o)
        println("    (all O — untrained model)")
    end

    # Relation predictions (argmax per pair, skip NO_RELATION)
    rel_logits = outputs.relation_logits[:, :, 1]
    rel_pairs = outputs.relation_pairs[:, :, 1]
    rel_mask = outputs.relation_mask[:, 1]
    spans_out = outputs.spans[:, :, 1]
    num_pairs = size(rel_logits, 2)

    predicted_rels = Tuple{Int,Int,Int,Int,String,Float32}[]
    for i in 1:num_pairs
        rel_mask[i] || continue
        label_id = findmax(rel_logits[:, i])[2]
        label_id == no_relation_id && continue
        head_idx = Int(rel_pairs[1, i])
        tail_idx = Int(rel_pairs[2, i])
        if !(1 <= head_idx <= size(spans_out, 2) && 1 <= tail_idx <= size(spans_out, 2))
            continue
        end
        h_start, h_end = Int(spans_out[1, head_idx]), Int(spans_out[2, head_idx])
        t_start, t_end = Int(spans_out[1, tail_idx]), Int(spans_out[2, tail_idx])
        label = get(id_to_relation_label, label_id, "UNK_$label_id")
        # Softmax probability
        exp_logits = exp.(Float32.(rel_logits[:, i]) .- maximum(rel_logits[:, i]))
        prob = exp_logits[label_id] / sum(exp_logits)
        push!(predicted_rels, (h_start, h_end, t_start, t_end, label, prob))
    end

    println("\n  Relation predictions (non-NO_RELATION):")
    if isempty(predicted_rels)
        println("    (none — untrained model predicts all NO_RELATION)")
    else
        for (hs, he, ts, te, label, prob) in predicted_rels[1:min(10, length(predicted_rels))]
            @printf("    [%d:%d] --%s--> [%d:%d]  (prob=%.3f)\n", hs, he, label, ts, te, prob)
        end
        if length(predicted_rels) > 10
            println("    ... $(length(predicted_rels) - 10) more")
        end
    end

    # Sanity checks
    println("\n" * "=" ^ 60)
    println("Sanity Checks")
    println("=" ^ 60)
    checks_passed = 0
    checks_total = 0

    # Check output dimensions
    checks_total += 1
    if size(outputs.entity_logits, 1) == length(entity_label_to_id)
        println("  [PASS] entity_logits dim matches label count ($(length(entity_label_to_id)))")
        checks_passed += 1
    else
        println("  [FAIL] entity_logits dim $(size(outputs.entity_logits, 1)) != label count $(length(entity_label_to_id))")
    end

    checks_total += 1
    if size(outputs.relation_logits, 1) == length(relation_label_to_id)
        println("  [PASS] relation_logits dim matches relation count ($(length(relation_label_to_id)))")
        checks_passed += 1
    else
        println("  [FAIL] relation_logits dim $(size(outputs.relation_logits, 1)) != relation count $(length(relation_label_to_id))")
    end

    checks_total += 1
    if size(outputs.boundary_logits, 1) == 2
        println("  [PASS] boundary_logits has 2 classes (B/I)")
        checks_passed += 1
    else
        println("  [FAIL] boundary_logits dim $(size(outputs.boundary_logits, 1)) != 2")
    end

    checks_total += 1
    if size(outputs.confidence_logits, 1) == 1
        println("  [PASS] confidence_logits is scalar per pair")
        checks_passed += 1
    else
        println("  [FAIL] confidence_logits dim $(size(outputs.confidence_logits, 1)) != 1")
    end

    checks_total += 1
    if all(isfinite, outputs.entity_logits)
        println("  [PASS] entity_logits all finite")
        checks_passed += 1
    else
        println("  [FAIL] entity_logits contains NaN/Inf")
    end

    checks_total += 1
    if all(isfinite, outputs.relation_logits)
        println("  [PASS] relation_logits all finite")
        checks_passed += 1
    else
        println("  [FAIL] relation_logits contains NaN/Inf")
    end

    println("\n$checks_passed/$checks_total checks passed")
    println("=" ^ 60)

    return checks_passed == checks_total ? 0 : 1
end

exit(main())
