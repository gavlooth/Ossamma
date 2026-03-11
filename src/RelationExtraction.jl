module RelationExtraction

using Lux
using Random
using TOML
using JSON3
using NNlib
using Statistics: mean

import ..Swamma: SwammaBlock, SwammaBlockConfig, LuxLayer

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

struct SwammaRelationExtractor{E,P,T,B,D,EH,BH,SP,RH,CH} <: LuxLayer
    vocab_size::Int
    max_sequence_length::Int
    embedding_dimension::Int
    number_of_layers::Int
    num_entity_labels::Int
    num_relations::Int
    max_candidate_spans::Int
    max_candidate_pairs::Int
    max_span_width::Int
    TokenEmbedding::E
    PositionEmbedding::P
    TimeEmbedding::T
    Blocks::B
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
    println("  num_entity_labels:    $(config.num_entity_labels)")
    println("  num_relations:        $(config.num_relations)")
    println("  window_size:          $(config.window_size)")
    println("  local_operator:       $(config.local_operator)")
    println("  residual_mode:        $(config.residual_mode)")
    println("Heads:")
    println("  max_candidate_spans:  $(config.max_candidate_spans)")
    println("  max_candidate_pairs:  $(config.max_candidate_pairs)")
    println("  max_span_width:       $(config.max_span_width)")
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

    d = config.embedding_dimension
    return SwammaRelationExtractor(
        config.vocab_size,
        config.max_sequence_length,
        d,
        config.number_of_layers,
        config.num_entity_labels,
        config.num_relations,
        config.max_candidate_spans,
        config.max_candidate_pairs,
        config.max_span_width,
        Lux.Embedding(config.vocab_size => d),
        Lux.Embedding(config.max_sequence_length => d),
        FixedTimeEmbedding(config.time_dimension),
        blocks,
        Lux.Dropout(config.dropout_rate),
        Lux.Chain(
            Lux.LayerNorm((d,)),
            Lux.Dropout(config.dropout_rate),
            Lux.Dense(d => config.num_entity_labels),
        ),
        Lux.Chain(
            Lux.LayerNorm((d,)),
            Lux.Dense(d => 2),
        ),
        Lux.Chain(
            Lux.LayerNorm((3 * d,)),
            Lux.Dense(3 * d => d, gelu),
        ),
        Lux.Chain(
            Lux.LayerNorm((4 * d,)),
            Lux.Dense(4 * d => d, gelu),
            Lux.Dense(d => config.num_relations),
        ),
        Lux.Chain(
            Lux.LayerNorm((4 * d,)),
            Lux.Dense(4 * d => d ÷ 2, gelu),
            Lux.Dense(d ÷ 2 => 1),
        ),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::SwammaRelationExtractor)
    block_params = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialparameters(rng, block) for block in model.Blocks)
    )

    return (
        TokenEmbedding = Lux.initialparameters(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialparameters(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialparameters(rng, model.TimeEmbedding),
        Blocks = block_params,
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
    return (
        TokenEmbedding = Lux.initialstates(rng, model.TokenEmbedding),
        PositionEmbedding = Lux.initialstates(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialstates(rng, model.TimeEmbedding),
        Blocks = block_states,
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
    new_state = (
        TokenEmbedding = tok_state,
        PositionEmbedding = pos_state,
        TimeEmbedding = time_state,
        Blocks = new_block_states,
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

function build_span_representations(model::SwammaRelationExtractor, hidden, spans, span_mask, params, state)
    d, seq_len, batch_size = size(hidden)
    max_spans = size(spans, 2)
    span_inputs = ntuple(batch_size * max_spans) do idx
        b = fld(idx - 1, max_spans) + 1
        i = mod(idx - 1, max_spans) + 1
        if span_mask[i, b]
            start_idx = clamp(spans[1, i, b], 1, seq_len)
            end_idx = clamp(spans[2, i, b], start_idx, seq_len)
            start_vec = hidden[:, start_idx, b]
            end_vec = hidden[:, end_idx, b]
            mean_vec = vec(mean(hidden[:, start_idx:end_idx, b], dims=2))
            vcat(start_vec, end_vec, mean_vec)
        else
            zeros(eltype(hidden), 3 * d)
        end
    end
    span_inputs = hcat(span_inputs...)

    projected, span_state = model.SpanProjection(span_inputs, params.SpanProjection, state.SpanProjection)
    span_reps = reshape(projected, d, max_spans, batch_size)
    return span_reps, span_state
end

function build_pair_features(span_reps, relation_pairs, relation_mask)
    d, max_spans, batch_size = size(span_reps)
    max_pairs = size(relation_pairs, 2)
    pair_features = ntuple(batch_size * max_pairs) do idx
        b = fld(idx - 1, max_pairs) + 1
        i = mod(idx - 1, max_pairs) + 1
        if relation_mask[i, b]
            head_idx = clamp(relation_pairs[1, i, b], 1, max_spans)
            tail_idx = clamp(relation_pairs[2, i, b], 1, max_spans)
            head_vec = span_reps[:, head_idx, b]
            tail_vec = span_reps[:, tail_idx, b]
            vcat(
                head_vec,
                tail_vec,
                abs.(head_vec .- tail_vec),
                head_vec .* tail_vec,
            )
        else
            zeros(eltype(span_reps), 4 * d)
        end
    end
    return hcat(pair_features...)
end

function (model::SwammaRelationExtractor)(inputs::NamedTuple, params, state)
    token_ids = inputs.token_ids
    spans = inputs.spans
    span_mask = inputs.span_mask
    relation_pairs = inputs.relation_pairs
    relation_mask = inputs.relation_mask

    hidden, encoder_state = encode_tokens(model, token_ids, params, state)
    d, seq_len, batch_size = size(hidden)

    hidden_flat = reshape(hidden, d, :)
    hidden_flat, dropout_state = model.Dropout(hidden_flat, params.Dropout, state.Dropout)

    entity_logits_flat, entity_state = model.EntityHead(hidden_flat, params.EntityHead, state.EntityHead)
    entity_logits = reshape(entity_logits_flat, model.num_entity_labels, seq_len, batch_size)

    boundary_logits_flat, boundary_state = model.BoundaryHead(hidden_flat, params.BoundaryHead, state.BoundaryHead)
    boundary_logits = reshape(boundary_logits_flat, 2, seq_len, batch_size)

    span_reps, span_state = build_span_representations(
        model, hidden, spans, span_mask, params, state
    )
    pair_features = build_pair_features(span_reps, relation_pairs, relation_mask)

    relation_logits_flat, relation_state = model.RelationHead(pair_features, params.RelationHead, state.RelationHead)
    confidence_flat, confidence_state = model.ConfidenceHead(pair_features, params.ConfidenceHead, state.ConfidenceHead)

    relation_logits = reshape(relation_logits_flat, model.num_relations, size(relation_pairs, 2), batch_size)
    confidence_logits = reshape(confidence_flat, 1, size(relation_pairs, 2), batch_size)

    new_state = (
        TokenEmbedding = encoder_state.TokenEmbedding,
        PositionEmbedding = encoder_state.PositionEmbedding,
        TimeEmbedding = encoder_state.TimeEmbedding,
        Blocks = encoder_state.Blocks,
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
        span_representations = span_reps,
        relation_logits = relation_logits,
        confidence_logits = confidence_logits,
    ), new_state
end

function entity_cross_entropy(logits, labels; ignore_index::Int = -100)
    num_labels = size(logits, 1)
    logits_flat = reshape(logits, num_labels, :)
    labels_flat = vec(labels)
    valid_mask = labels_flat .!= ignore_index
    valid_count = sum(valid_mask)
    valid_count == 0 && return 0.0f0
    log_probs = NNlib.logsoftmax(logits_flat, dims=1)
    total = 0.0f0
    for i in eachindex(labels_flat)
        if valid_mask[i]
            total -= log_probs[labels_flat[i], i]
        end
    end
    return total / valid_count
end

function boundary_bce(logits, targets; ignore_index::Int = -100)
    total = 0.0f0
    count = 0
    for idx in eachindex(logits)
        target = targets[idx]
        if target == ignore_index
            continue
        end
        y = Float32(target)
        z = Float32(logits[idx])
        total += max(z, 0f0) - z * y + log1p(exp(-abs(z)))
        count += 1
    end
    return count > 0 ? total / count : 0.0f0
end

function relation_cross_entropy(logits, labels, mask; ignore_index::Int = -100, null_relation_weight::Float32 = 1.0f0)
    num_relations = size(logits, 1)
    logits_flat = reshape(logits, num_relations, :)
    labels_flat = vec(labels)
    mask_flat = vec(mask)
    log_probs = NNlib.logsoftmax(logits_flat, dims=1)
    total = 0.0f0
    total_weight = 0.0f0
    for i in eachindex(labels_flat)
        if mask_flat[i] && labels_flat[i] != ignore_index
            weight = labels_flat[i] == 1 ? null_relation_weight : 1.0f0
            total -= weight * log_probs[labels_flat[i], i]
            total_weight += weight
        end
    end
    return total_weight > 0 ? total / total_weight : 0.0f0
end

function confidence_bce(logits, targets, mask)
    logits_flat = vec(logits)
    targets_flat = vec(targets)
    mask_flat = vec(mask)
    total = 0.0f0
    count = 0
    for i in eachindex(logits_flat)
        if mask_flat[i]
            y = Float32(targets_flat[i])
            z = Float32(logits_flat[i])
            total += max(z, 0f0) - z * y + log1p(exp(-abs(z)))
            count += 1
        end
    end
    return count > 0 ? total / count : 0.0f0
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
            negative_candidates = Tuple{Int, Int}[]
            for head_idx in 1:entity_count
                for tail_idx in 1:entity_count
                    head_idx == tail_idx && continue
                    pair = (head_idx, tail_idx)
                    pair in positive_pairs && continue
                    push!(negative_candidates, pair)
                end
            end

            target_negatives = if !isempty(positive_pairs)
                ceil(Int, length(positive_pairs) * hard_negative_ratio)
            else
                min(length(negative_candidates), max(1, round(Int, hard_negative_ratio)))
            end
            target_negatives = min(target_negatives, length(negative_candidates), max_candidate_pairs - pair_idx)

            for negative_pair in Iterators.take(negative_candidates, target_negatives)
                pair_idx += 1
                relation_pairs[1, pair_idx, b] = negative_pair[1]
                relation_pairs[2, pair_idx, b] = negative_pair[2]
                relation_labels[pair_idx, b] = no_relation_id
                relation_targets[pair_idx, b] = 0.0f0
                relation_mask[pair_idx, b] = true
            end
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
