module LongContextModels

using Lux
using Random
using NNlib
using ChainRulesCore
using Statistics: mean

using Main.Swamma: LLaDAModel, LLaDAConfig, TimeMLPEmbedding
using Main.Swamma: token_ids_to_subtokens, apply_subtoken_mask

export ModelSpec
export FullAttentionLLaDA
export build_model
export count_parameters
export masked_forward
export masked_metrics
export synthetic_needle_batch
export needle_eval_step

const LuxLayerType = isdefined(Lux, :AbstractExplicitLayer) ? Lux.AbstractExplicitLayer : Lux.AbstractLuxLayer

function to_device_like(target, x::AbstractArray)
    target_type = string(typeof(target))
    if occursin("CuArray", target_type)
        cuda_mod = parentmodule(typeof(target))
        while cuda_mod !== Main && !isdefined(cuda_mod, :CuArray)
            cuda_mod = parentmodule(cuda_mod)
        end
        if isdefined(cuda_mod, :CuArray)
            return cuda_mod.CuArray(x)
        end
    end
    return x
end

is_gpu_array(x) = occursin("CuArray", string(typeof(x)))

function first_array_leaf(x)
    if x isa AbstractArray
        return x
    elseif x isa NamedTuple
        for v in values(x)
            leaf = first_array_leaf(v)
            leaf === nothing || return leaf
        end
    elseif x isa Tuple
        for v in x
            leaf = first_array_leaf(v)
            leaf === nothing || return leaf
        end
    end
    return nothing
end

Base.@kwdef struct ModelSpec
    architecture::Symbol = :swamma      # :swamma or :transformer
    vocab_size::Int = 32000
    max_sequence_length::Int = 2048
    embedding_dimension::Int = 512
    number_of_heads::Int = 8
    number_of_layers::Int = 12
    time_dimension::Int = 128
    state_dimension::Int = 512
    window_size::Int = 128
    min_frequency::Float32 = 0.1f0
    max_frequency::Float32 = 10.0f0
    default_time_step::Float32 = 0.1f0
    prime_subtoken_length::Int = 4
    prime_subtoken_base::Int = 16
end

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
    masks = [[falses(vocab_size) for _ in 1:base] for _ in 1:subtoken_length]
    for token_id in 1:vocab_size
        for j in 1:subtoken_length
            digit = Int(code_table[j, token_id])
            masks[j][digit][token_id] = true
        end
    end
    return masks
end

function prime_compatibility_mask(
    prime_subtoken_length::Int,
    prime_subtoken_base::Int,
    prime_mask_subtoken_id::Int,
    prime_value_masks,
    vocab_size::Int,
    subtoken_state_cpu::Array{<:Integer,3},
)
    subtoken_length, seq_len, batch_size = size(subtoken_state_cpu)
    subtoken_length == prime_subtoken_length || throw(ArgumentError(
        "Expected subtoken_state first dim $(prime_subtoken_length), got $(subtoken_length)"
    ))

    compat = trues(vocab_size, seq_len, batch_size)
    for b in 1:batch_size
        for s in 1:seq_len
            allowed = trues(vocab_size)
            for j in 1:prime_subtoken_length
                digit = Int(subtoken_state_cpu[j, s, b])
                if digit != prime_mask_subtoken_id
                    if 1 <= digit <= prime_subtoken_base
                        allowed .&= prime_value_masks[j][digit]
                    else
                        fill!(allowed, true)
                        break
                    end
                end
            end
            if !any(allowed)
                fill!(allowed, true)
            end
            compat[:, s, b] .= allowed
        end
    end

    return compat
end

function apply_prime_carryover_filter(model, logits, subtoken_state_batched)
    compat_cpu = ChainRulesCore.ignore_derivatives() do
        subtoken_state_cpu = Array(subtoken_state_batched)
        prime_compatibility_mask(
            model.prime_subtoken_length,
            model.prime_subtoken_base,
            model.prime_mask_subtoken_id,
            model.prime_value_masks,
            model.vocab_size,
            subtoken_state_cpu,
        )
    end
    compat = to_device_like(logits, compat_cpu)
    neg_large = convert(eltype(logits), -1.0f9)
    return ifelse.(compat, logits, neg_large)
end

struct FullAttentionBlock <: LuxLayerType
    embedding_dimension::Int
    time_dimension::Int
    Norm1::Lux.LayerNorm
    Attention::LuxLayerType
    Norm2::Lux.LayerNorm
    FF1::Lux.Dense
    FF2::Lux.Dense
end

struct DenseSelfAttention <: LuxLayerType
    embedding_dimension::Int
    number_of_heads::Int
    head_dimension::Int
    QueryProjection::Lux.Dense
    KeyProjection::Lux.Dense
    ValueProjection::Lux.Dense
    OutputProjection::Lux.Dense
end

function DenseSelfAttention(embedding_dimension::Int, number_of_heads::Int)
    embedding_dimension % number_of_heads == 0 || throw(ArgumentError(
        "embedding_dimension must be divisible by number_of_heads"
    ))
    head_dimension = embedding_dimension ÷ number_of_heads
    return DenseSelfAttention(
        embedding_dimension,
        number_of_heads,
        head_dimension,
        Lux.Dense(embedding_dimension => embedding_dimension),
        Lux.Dense(embedding_dimension => embedding_dimension),
        Lux.Dense(embedding_dimension => embedding_dimension),
        Lux.Dense(embedding_dimension => embedding_dimension),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::DenseSelfAttention)
    return (
        QueryProjection = Lux.initialparameters(rng, layer.QueryProjection),
        KeyProjection = Lux.initialparameters(rng, layer.KeyProjection),
        ValueProjection = Lux.initialparameters(rng, layer.ValueProjection),
        OutputProjection = Lux.initialparameters(rng, layer.OutputProjection),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::DenseSelfAttention)
    return (
        QueryProjection = Lux.initialstates(rng, layer.QueryProjection),
        KeyProjection = Lux.initialstates(rng, layer.KeyProjection),
        ValueProjection = Lux.initialstates(rng, layer.ValueProjection),
        OutputProjection = Lux.initialstates(rng, layer.OutputProjection),
    )
end

function (layer::DenseSelfAttention)(input_tensor::AbstractArray, params, state)
    query, q_state = layer.QueryProjection(input_tensor, params.QueryProjection, state.QueryProjection)
    key, k_state = layer.KeyProjection(input_tensor, params.KeyProjection, state.KeyProjection)
    value, v_state = layer.ValueProjection(input_tensor, params.ValueProjection, state.ValueProjection)

    seq_len = size(input_tensor, 2)
    batch_size = size(input_tensor, 3)
    heads = layer.number_of_heads
    head_dim = layer.head_dimension

    qh = reshape(query, head_dim, heads, seq_len, batch_size)
    kh = reshape(key, head_dim, heads, seq_len, batch_size)
    vh = reshape(value, head_dim, heads, seq_len, batch_size)

    qf = reshape(permutedims(qh, (1, 3, 2, 4)), head_dim, seq_len, heads * batch_size)
    kf = reshape(permutedims(kh, (1, 3, 2, 4)), head_dim, seq_len, heads * batch_size)
    vf = reshape(permutedims(vh, (1, 3, 2, 4)), head_dim, seq_len, heads * batch_size)

    scale = convert(eltype(input_tensor), inv(sqrt(Float32(head_dim))))
    scores = NNlib.batched_mul(permutedims(qf, (2, 1, 3)), kf) .* scale
    weights = NNlib.softmax(scores, dims = 2)
    contextf = NNlib.batched_mul(vf, permutedims(weights, (2, 1, 3)))

    context_h = permutedims(reshape(contextf, head_dim, seq_len, heads, batch_size), (1, 3, 2, 4))
    context = reshape(context_h, layer.embedding_dimension, seq_len, batch_size)
    output, out_state = layer.OutputProjection(context, params.OutputProjection, state.OutputProjection)

    new_state = (
        QueryProjection = q_state,
        KeyProjection = k_state,
        ValueProjection = v_state,
        OutputProjection = out_state,
    )
    return output, new_state
end

function FullAttentionBlock(
    embedding_dimension::Int,
    time_dimension::Int,
    sequence_length::Int,
    number_of_heads::Int,
)
    return FullAttentionBlock(
        embedding_dimension,
        time_dimension,
        Lux.LayerNorm((embedding_dimension,)),
        DenseSelfAttention(embedding_dimension, number_of_heads),
        Lux.LayerNorm((embedding_dimension,)),
        Lux.Dense(embedding_dimension => 4 * embedding_dimension, NNlib.gelu),
        Lux.Dense(4 * embedding_dimension => embedding_dimension),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, block::FullAttentionBlock)
    return (
        Norm1 = Lux.initialparameters(rng, block.Norm1),
        Attention = Lux.initialparameters(rng, block.Attention),
        Norm2 = Lux.initialparameters(rng, block.Norm2),
        FF1 = Lux.initialparameters(rng, block.FF1),
        FF2 = Lux.initialparameters(rng, block.FF2),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, block::FullAttentionBlock)
    return (
        Norm1 = Lux.initialstates(rng, block.Norm1),
        Attention = Lux.initialstates(rng, block.Attention),
        Norm2 = Lux.initialstates(rng, block.Norm2),
        FF1 = Lux.initialstates(rng, block.FF1),
        FF2 = Lux.initialstates(rng, block.FF2),
    )
end

function (block::FullAttentionBlock)(inputs::Tuple, params, state)
    x, _time_emb = inputs

    x_shape = size(x)
    x_flat = reshape(x, block.embedding_dimension, :)
    n1_flat, n1_state = block.Norm1(x_flat, params.Norm1, state.Norm1)
    n1 = reshape(n1_flat, x_shape)

    attn_out, attn_state = block.Attention(n1, params.Attention, state.Attention)
    h = x .+ attn_out

    h_shape = size(h)
    h_flat = reshape(h, block.embedding_dimension, :)
    n2_flat, n2_state = block.Norm2(h_flat, params.Norm2, state.Norm2)
    n2 = reshape(n2_flat, h_shape)

    ff_hidden, ff1_state = block.FF1(n2, params.FF1, state.FF1)
    ff_out, ff2_state = block.FF2(ff_hidden, params.FF2, state.FF2)

    out = h .+ ff_out

    new_state = (
        Norm1 = n1_state,
        Attention = attn_state,
        Norm2 = n2_state,
        FF1 = ff1_state,
        FF2 = ff2_state,
    )

    return out, new_state
end

struct FullAttentionLLaDA{S,P,T,B,N,O,M,V} <: LuxLayerType
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

    SubtokenEmbeddings::S
    PositionEmbedding::P
    TimeEmbedding::T
    Blocks::B
    FinalNorm::N
    OutputHead::O
end

function FullAttentionLLaDA(spec::ModelSpec)
    resolved_subtoken_length = spec.prime_subtoken_length
    resolved_subtoken_base = max(spec.prime_subtoken_base, 2)
    if resolved_subtoken_base^resolved_subtoken_length < spec.vocab_size
        resolved_subtoken_base = required_subtoken_base(spec.vocab_size, resolved_subtoken_length)
    end
    spec.embedding_dimension % resolved_subtoken_length == 0 || throw(ArgumentError(
        "embedding_dimension=$(spec.embedding_dimension) must be divisible by prime_subtoken_length=$(resolved_subtoken_length)"
    ))

    resolved_mask_subtoken_id = resolved_subtoken_base + 1
    resolved_subtoken_embedding_dimension = spec.embedding_dimension ÷ resolved_subtoken_length
    code_table = build_vocab_code_table(spec.vocab_size, resolved_subtoken_base, resolved_subtoken_length)
    value_masks = build_subtoken_value_masks(code_table, resolved_subtoken_base)

    subtoken_embeddings = Tuple(
        Lux.Embedding((resolved_subtoken_base + 1) => resolved_subtoken_embedding_dimension)
        for _ in 1:resolved_subtoken_length
    )

    blocks = Tuple([
        FullAttentionBlock(
            spec.embedding_dimension,
            spec.time_dimension,
            spec.max_sequence_length,
            spec.number_of_heads,
        )
        for _ in 1:spec.number_of_layers
    ])

    return FullAttentionLLaDA(
        spec.vocab_size,
        spec.max_sequence_length,
        spec.embedding_dimension,
        spec.number_of_heads,
        spec.number_of_layers,
        resolved_subtoken_base,
        resolved_subtoken_length,
        resolved_mask_subtoken_id,
        resolved_subtoken_embedding_dimension,
        code_table,
        value_masks,
        subtoken_embeddings,
        Lux.Embedding(spec.max_sequence_length => spec.embedding_dimension),
        TimeMLPEmbedding(spec.time_dimension, spec.time_dimension),
        blocks,
        Lux.LayerNorm((spec.embedding_dimension,)),
        Lux.Dense(spec.embedding_dimension => spec.vocab_size; use_bias = false),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::FullAttentionLLaDA)
    block_params = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialparameters(rng, block) for block in model.Blocks)
    )
    subtoken_keys = ntuple(i -> Symbol("Subtoken_$i"), model.prime_subtoken_length)
    subtoken_vals = Tuple(
        Lux.initialparameters(rng, model.SubtokenEmbeddings[i])
        for i in 1:model.prime_subtoken_length
    )
    subtoken_params = NamedTuple{subtoken_keys}(subtoken_vals)

    return (
        SubtokenEmbeddings = subtoken_params,
        PositionEmbedding = Lux.initialparameters(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialparameters(rng, model.TimeEmbedding),
        Blocks = block_params,
        FinalNorm = Lux.initialparameters(rng, model.FinalNorm),
        OutputHead = Lux.initialparameters(rng, model.OutputHead),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::FullAttentionLLaDA)
    block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(
        Tuple(Lux.initialstates(rng, block) for block in model.Blocks)
    )
    subtoken_keys = ntuple(i -> Symbol("Subtoken_$i"), model.prime_subtoken_length)
    subtoken_vals = Tuple(
        Lux.initialstates(rng, model.SubtokenEmbeddings[i])
        for i in 1:model.prime_subtoken_length
    )
    subtoken_states = NamedTuple{subtoken_keys}(subtoken_vals)

    return (
        SubtokenEmbeddings = subtoken_states,
        PositionEmbedding = Lux.initialstates(rng, model.PositionEmbedding),
        TimeEmbedding = Lux.initialstates(rng, model.TimeEmbedding),
        Blocks = block_states,
        FinalNorm = Lux.initialstates(rng, model.FinalNorm),
        OutputHead = Lux.initialstates(rng, model.OutputHead),
    )
end

function (model::FullAttentionLLaDA)(inputs::NamedTuple, params, state)
    haskey(inputs, :subtoken_state) || throw(ArgumentError("Expected `subtoken_state` in inputs."))
    mask_ratio = inputs.mask_ratio
    subtoken_state = inputs.subtoken_state

    subtoken_state_batched = if ndims(subtoken_state) == 2
        reshape(subtoken_state, model.prime_subtoken_length, size(subtoken_state, 2), 1)
    elseif ndims(subtoken_state) == 3
        subtoken_state
    else
        throw(ArgumentError("Expected 2D or 3D subtoken_state, got ndims=$(ndims(subtoken_state))."))
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
    subtoken_states_out = NamedTuple{ntuple(i -> Symbol("Subtoken_$i"), model.prime_subtoken_length)}(sub_states)

    pos_idx = to_device_like(source_for_device, collect(1:seq_len))
    pos_emb_raw, pos_state = model.PositionEmbedding(pos_idx, params.PositionEmbedding, state.PositionEmbedding)
    pos_emb = reshape(pos_emb_raw, model.embedding_dimension, seq_len, 1)

    hidden = token_emb .+ pos_emb

    t_input_cpu = if ndims(mask_ratio) == 0
        fill(Float32(mask_ratio), 1, batch_size)
    elseif ndims(mask_ratio) == 1
        reshape(Float32.(mask_ratio), 1, :)
    else
        Float32.(mask_ratio)
    end
    t_input = to_device_like(source_for_device, t_input_cpu)
    time_emb, time_state = model.TimeEmbedding(t_input, params.TimeEmbedding, state.TimeEmbedding)

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

    hidden_shape = size(hidden)
    hidden_flat = reshape(hidden, model.embedding_dimension, :)
    normalized_flat, norm_state = model.FinalNorm(hidden_flat, params.FinalNorm, state.FinalNorm)
    normalized = reshape(normalized_flat, hidden_shape)

    logits, out_state = model.OutputHead(normalized, params.OutputHead, state.OutputHead)
    logits = apply_prime_carryover_filter(model, logits, subtoken_state_batched)

    final_logits = was_unbatched ? dropdims(logits, dims = 3) : logits

    new_block_states = NamedTuple{ntuple(i -> Symbol("Block_$i"), model.number_of_layers)}(block_states)
    new_state = (
        SubtokenEmbeddings = subtoken_states_out,
        PositionEmbedding = pos_state,
        TimeEmbedding = time_state,
        Blocks = new_block_states,
        FinalNorm = norm_state,
        OutputHead = out_state,
    )

    return final_logits, new_state
end

function build_model(spec::ModelSpec)
    if spec.architecture == :swamma
        cfg = LLaDAConfig(
            vocab_size = spec.vocab_size,
            max_sequence_length = spec.max_sequence_length,
            embedding_dimension = spec.embedding_dimension,
            number_of_heads = spec.number_of_heads,
            number_of_layers = spec.number_of_layers,
            time_dimension = spec.time_dimension,
            state_dimension = spec.state_dimension,
            window_size = spec.window_size,
            min_frequency = spec.min_frequency,
            max_frequency = spec.max_frequency,
            default_time_step = spec.default_time_step,
            prime_subtoken_length = spec.prime_subtoken_length,
            prime_subtoken_base = spec.prime_subtoken_base,
        )
        return LLaDAModel(cfg)
    elseif spec.architecture == :transformer
        return FullAttentionLLaDA(spec)
    else
        throw(ArgumentError("Unsupported architecture $(spec.architecture). Use :swamma or :transformer."))
    end
end

function count_parameters(params)
    total = 0
    for v in values(params)
        if v isa AbstractArray
            total += length(v)
        elseif v isa NamedTuple
            total += count_parameters(v)
        elseif v isa Tuple
            total += sum(count_parameters, v)
        end
    end
    return total
end

function masked_forward(model, params, state, token_ids::AbstractArray, mask_ratio::Float32, rng)
    subtoken_state = token_ids_to_subtokens(token_ids, model.prime_code_table)
    masked_subtokens, subtoken_mask, token_mask = apply_subtoken_mask(
        subtoken_state, mask_ratio, model.prime_mask_subtoken_id; rng = rng
    )
    ref = first_array_leaf(params)
    masked_subtokens_device = ref === nothing ? masked_subtokens : to_device_like(ref, masked_subtokens)
    logits, new_state = model((subtoken_state = masked_subtokens_device, mask_ratio = mask_ratio), params, state)
    return logits, new_state, token_mask, subtoken_mask
end

function masked_metrics(logits, token_ids::AbstractMatrix{<:Integer}, token_mask::AbstractMatrix{Bool})
    logits_cpu = is_gpu_array(logits) ? Array(logits) : logits
    token_mask_cpu = is_gpu_array(token_mask) ? Array(token_mask) : token_mask

    vocab_size, seq_len, batch_size = size(logits_cpu)
    log_probs = NNlib.logsoftmax(logits_cpu, dims = 1)
    predictions = dropdims(argmax(logits_cpu, dims = 1), dims = 1)

    nll = 0.0
    correct = 0
    total = 0

    bin_edges = [1, max(2, fld(seq_len, 3) + 1), max(2, fld(2 * seq_len, 3) + 1), seq_len + 1]
    bin_correct = zeros(Int, 3)
    bin_total = zeros(Int, 3)

    for b in 1:batch_size
        for s in 1:seq_len
            if token_mask_cpu[s, b]
                target = Int(token_ids[s, b])
                if 1 <= target <= vocab_size
                    nll -= Float64(log_probs[target, s, b])
                    total += 1
                    if Int(predictions[s, b]) == target
                        correct += 1
                    end
                    bin = s < bin_edges[2] ? 1 : (s < bin_edges[3] ? 2 : 3)
                    bin_total[bin] += 1
                    if Int(predictions[s, b]) == target
                        bin_correct[bin] += 1
                    end
                end
            end
        end
    end

    loss = total > 0 ? Float32(nll / total) : 0.0f0
    acc = total > 0 ? Float32(correct / total) : 0.0f0
    bin_acc = ntuple(i -> (bin_total[i] > 0 ? Float32(bin_correct[i] / bin_total[i]) : 0.0f0), 3)
    return (
        loss = loss,
        ppl = Float32(exp(clamp(Float64(loss), -20.0, 20.0))),
        acc = acc,
        acc_early = bin_acc[1],
        acc_middle = bin_acc[2],
        acc_late = bin_acc[3],
        masked_positions = total,
    )
end

function synthetic_needle_batch(vocab_size::Int, seq_len::Int, batch_size::Int, rng::Random.AbstractRNG)
    vocab_size >= 32 || throw(ArgumentError("Need vocab_size >= 32 for synthetic needle task."))
    seq_len >= 64 || throw(ArgumentError("Need seq_len >= 64 for synthetic needle task."))

    KEY_MARK = 3
    QUERY_MARK = 4
    filler_low = 16
    filler_high = vocab_size

    tokens = rand(rng, filler_low:filler_high, seq_len, batch_size)
    targets = zeros(Int, batch_size)

    for b in 1:batch_size
        key = rand(rng, filler_low:filler_high)
        value = rand(rng, filler_low:filler_high)

        anchor = rand(rng, 8:(seq_len - 16))
        tokens[anchor, b] = KEY_MARK
        tokens[anchor + 1, b] = key
        tokens[anchor + 2, b] = value

        tokens[seq_len - 2, b] = QUERY_MARK
        tokens[seq_len - 1, b] = key
        tokens[seq_len, b] = value

        targets[b] = value
    end

    return tokens, targets
end

function needle_eval_step(model, params, state, seq_len::Int, batch_size::Int, rng::Random.AbstractRNG)
    tokens, targets = synthetic_needle_batch(model.vocab_size, seq_len, batch_size, rng)

    subtoken_state = token_ids_to_subtokens(tokens, model.prime_code_table)
    masked = copy(subtoken_state)
    token_mask = falses(seq_len, batch_size)

    for b in 1:batch_size
        masked[:, seq_len, b] .= model.prime_mask_subtoken_id
        token_mask[seq_len, b] = true
    end

    ref = first_array_leaf(params)
    masked_device = ref === nothing ? masked : to_device_like(ref, masked)

    logits, new_state = model((subtoken_state = masked_device, mask_ratio = Float32(1 / seq_len)), params, state)
    preds = Array(dropdims(argmax(logits, dims = 1), dims = 1))

    correct = 0
    for b in 1:batch_size
        correct += Int(preds[seq_len, b] == targets[b])
    end

    acc = Float32(correct / batch_size)
    return acc, new_state
end

end # module
