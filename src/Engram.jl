module EngramMod

"""
Engram Conditional Memory Module

Implements N-gram-based conditional memory lookup for LLMs, based on
"Pooling Engram Conditional Memory in LLMs using CXL" (Ma et al., 2026).

Engram decouples static lexical knowledge (N-gram co-occurrence patterns)
from dynamic computation. For each token position, it:
1. Extracts causal multi-granular N-grams (e.g., bigrams, trigrams)
2. Hashes each N-gram via multi-head hashing into embedding table indices
3. Retrieves sparse embeddings from a large table (O(1) per lookup)
4. Fuses retrieved knowledge with hidden states via a learned gate

Placement: inserted before attention blocks at select transformer layers.
"""

using Lux
using Random
using NNlib
using ChainRulesCore

# Import shared helpers from parent Swamma module
using ..Swamma: to_device_like, is_gpu_array, LuxLayer

# ============================================================================
# EngramModule
# ============================================================================

struct EngramModule <: LuxLayer
    embedding_dim::Int
    num_heads::Int              # hash heads per N-gram order
    ngram_orders::Vector{Int}   # e.g., [2, 3] for bigrams + trigrams
    table_size::Int             # entries per hash head
    head_dim::Int               # embedding dim per head
    total_tables::Int           # num_heads * length(ngram_orders)
end

function EngramModule(
    embedding_dim::Int;
    num_heads::Int = 8,
    ngram_orders::Vector{Int} = [2, 3],
    table_size::Int = 65536,
    head_dim::Int = 0,
)
    if head_dim <= 0
        head_dim = max(embedding_dim ÷ num_heads, 16)
    end
    total = num_heads * length(ngram_orders)
    return EngramModule(embedding_dim, num_heads, ngram_orders, table_size, head_dim, total)
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::EngramModule)
    total_entries = layer.table_size * layer.total_tables
    scale = Float32(0.01)
    he_scale = Float32(sqrt(2.0 / (layer.embedding_dim + layer.head_dim)))
    return (
        # Combined embedding table: all heads × all N-gram orders in one array
        EmbeddingTable = randn(rng, Float32, layer.head_dim, total_entries) .* scale,
        # Project summed head embeddings → model dimension
        OutputWeight = randn(rng, Float32, layer.embedding_dim, layer.head_dim) .* he_scale,
        OutputBias = zeros(Float32, layer.embedding_dim),
        # Gating: sigmoid(W·hidden + b) controls injection strength
        GateWeight = randn(rng, Float32, layer.embedding_dim, layer.embedding_dim) .* he_scale,
        GateBias = fill(Float32(-2.0), layer.embedding_dim),  # init gate near 0 (conservative injection)
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::EngramModule)
    max_order = maximum(layer.ngram_orders)
    # Random odd multipliers for polynomial hashing (fixed after init)
    hash_mults = Matrix{Int64}(undef, max_order, layer.total_tables)
    for t in 1:layer.total_tables
        for j in 1:max_order
            m = rand(rng, 10007:999983)
            hash_mults[j, t] = isodd(m) ? m : m + 1
        end
    end
    return (hash_multipliers = hash_mults,)
end

# ============================================================================
# Hash index computation (CPU, no gradient)
# ============================================================================

"""
    compute_engram_indices(layer, token_ids, hash_mults)

Compute hash table indices for all positions, N-gram orders, and heads.
Uses causal N-grams: for position s and order n, the N-gram is
(token[s-n+1], ..., token[s]), padded with 0 at sequence boundaries.

Returns flat index array suitable for indexing into the combined table.
Shape: (total_tables, seq_len, batch_size)
"""
function compute_engram_indices(
    layer::EngramModule,
    token_ids::AbstractMatrix{<:Integer},
    hash_mults::Matrix{Int64},
)
    seq_len, batch_size = size(token_ids)
    ts = layer.table_size
    indices = Array{Int}(undef, layer.total_tables, seq_len, batch_size)

    table_idx = 0
    for order in layer.ngram_orders
        for _h in 1:layer.num_heads
            table_idx += 1
            table_offset = (table_idx - 1) * ts
            for b in 1:batch_size
                for s in 1:seq_len
                    hash_val = Int64(0)
                    for j in 1:order
                        pos = s - order + j  # causal: (s-order+1, ..., s)
                        tok = pos >= 1 ? Int64(token_ids[pos, b]) : Int64(0)
                        hash_val += tok * hash_mults[j, table_idx]
                    end
                    # Map to [1, table_size] then offset for combined table
                    indices[table_idx, s, b] = mod(hash_val, ts) + 1 + table_offset
                end
            end
        end
    end

    return indices
end

function prepare_engram_indices(
    layer::EngramModule,
    token_ids::AbstractMatrix{<:Integer},
    hash_mults::AbstractMatrix{<:Integer};
    reference_device = token_ids,
)
    token_ids_cpu = ChainRulesCore.ignore_derivatives() do
        is_gpu_array(token_ids) ? Array(token_ids) : token_ids
    end
    indices_cpu = ChainRulesCore.ignore_derivatives() do
        compute_engram_indices(layer, token_ids_cpu, Matrix{Int64}(hash_mults))
    end
    return to_device_like(reference_device, vec(indices_cpu))
end

# ============================================================================
# Subtoken → token ID reconstruction
# ============================================================================

"""
    subtokens_to_token_ids(subtokens, base)

Reconstruct effective token IDs from PRIME subtoken representation.
subtokens: (subtoken_length, seq_len, batch) integer array
Returns: (seq_len, batch) integer array
"""
function subtokens_to_token_ids(subtokens::AbstractArray{<:Integer, 3}, base::Int)
    subtoken_length, seq_len, batch_size = size(subtokens)
    result = zeros(Int, seq_len, batch_size)
    for b in 1:batch_size
        for s in 1:seq_len
            val = 0
            multiplier = 1
            for j in 1:subtoken_length
                val += Int(subtokens[j, s, b]) * multiplier
                multiplier *= base
            end
            result[s, b] = val
        end
    end
    return result
end

# ============================================================================
# Forward pass
# ============================================================================

"""
    (layer::EngramModule)((token_ids, hidden_state), ps, st)

Apply engram conditional memory injection.

- token_ids: (seq_len,) or (seq_len, batch) integer token IDs
- hidden_state: (dim, seq_len) or (dim, seq_len, batch) hidden representations

Returns: (output, state) where output = hidden + gate ⊙ project(engram_lookup)
"""
function (layer::EngramModule)(inputs::Tuple, ps, st)
    token_ids, hidden_state = if length(inputs) == 2
        (inputs[1], inputs[2])
    elseif length(inputs) == 3
        (inputs[1], inputs[2])
    else
        throw(ArgumentError("EngramModule expects 2 or 3 inputs, got $(length(inputs))."))
    end
    precomputed_indices = length(inputs) == 3 ? inputs[3] : nothing
    is_batched = ndims(hidden_state) == 3

    if !is_batched
        token_ids_b = ndims(token_ids) == 1 ? reshape(token_ids, :, 1) : token_ids
        hidden_b = reshape(hidden_state, size(hidden_state, 1), size(hidden_state, 2), 1)
    else
        token_ids_b = token_ids
        hidden_b = hidden_state
    end

    seq_len = size(token_ids_b, 1)
    batch_size = size(token_ids_b, 2)
    nt = layer.total_tables

    # 1. Compute or reuse hash indices outside the hot lookup path.
    indices = if precomputed_indices === nothing
        prepare_engram_indices(layer, token_ids_b, st.hash_multipliers; reference_device = ps.EmbeddingTable)
    else
        to_device_like(ps.EmbeddingTable, precomputed_indices)
    end

    # 2. Gather embeddings from combined table: single vectorized gather
    all_embs = ps.EmbeddingTable[:, indices]  # (head_dim, nt * seq * batch)

    # 3. Reshape to (head_dim, nt, seq * batch) and sum across tables
    all_embs_3d = reshape(all_embs, layer.head_dim, nt, seq_len * batch_size)
    engram_sum = dropdims(sum(all_embs_3d, dims=2), dims=2)  # (head_dim, seq * batch)

    # 4. Project: head_dim → embedding_dim
    projected = ps.OutputWeight * engram_sum .+ ps.OutputBias  # (embedding_dim, seq * batch)

    # 5. Gate: sigmoid(W·hidden + b) — controls how much engram knowledge to inject
    hidden_flat = reshape(hidden_b, layer.embedding_dim, seq_len * batch_size)
    gate = NNlib.sigmoid.(ps.GateWeight * hidden_flat .+ ps.GateBias)

    # 6. Gated injection
    engram_out = gate .* projected
    engram_out = reshape(engram_out, layer.embedding_dim, seq_len, batch_size)

    # 7. Residual connection
    output = hidden_b .+ engram_out

    if !is_batched
        output = dropdims(output, dims=3)
    end

    return output, st
end

# ============================================================================
# Collision diagnostics
# ============================================================================

"""
    engram_collision_rate(layer, token_ids, hash_mults) → Float64

Compute the fraction of hash collisions across all tables for the given
token sequence. Useful for monitoring hash quality during development.

Returns a value in [0, 1] where 0 = no collisions, 1 = all collide.
"""
function engram_collision_rate(
    layer::EngramModule,
    token_ids::AbstractMatrix{<:Integer},
    hash_mults::Matrix{Int64},
)
    indices = compute_engram_indices(layer, token_ids, hash_mults)
    total = length(indices)
    unique_count = length(Set(vec(indices)))
    return 1.0 - unique_count / total
end

# ============================================================================
# Exports
# ============================================================================

export EngramModule
export subtokens_to_token_ids
export prepare_engram_indices
export engram_collision_rate

end # module
