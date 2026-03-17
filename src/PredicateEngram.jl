module PredicateEngramMod

"""
Predicate Engram: Reasoning Pattern Memory with Variable Binding

Combines three ideas into a single differentiable module:
1. **VQ-VAE codebook** — quantizes hidden states into discrete "reasoning situation"
   codes, making them hashable for O(1) rule lookup (like Engram for N-grams)
2. **Soft TPR (Tensor Product Representations)** — decomposes hidden states into
   role-filler bindings, enabling variable-aware rule application
3. **Gated injection** — sigmoid-gated residual, so the module starts silent and
   learns where reasoning shortcuts help

Architecture:
    hidden → encode → VQ quantize → rule lookup (O(1))
    hidden → unbind into (filler₁, ..., fillerₖ) per role
    rule_matrix × fillers → transformed fillers (rule applied to variables)
    rebind → project → gate ⊙ projected → residual add

The rule bank stores small (num_roles × num_roles) mixing matrices — each rule
encodes how roles interact (e.g., transitivity routes the "consequent" filler
from one relation to the "antecedent" of another).
"""

using Lux
using Random
using NNlib
using ChainRulesCore

const LuxLayer =
    isdefined(Lux, :AbstractExplicitLayer) ? Lux.AbstractExplicitLayer :
    Lux.AbstractLuxLayer

# ============================================================================
# PredicateEngram
# ============================================================================

struct PredicateEngram <: LuxLayer
    embedding_dim::Int
    code_dim::Int           # VQ codebook vector dimension
    codebook_size::Int      # number of discrete reasoning codes
    num_roles::Int          # TPR roles for variable binding
    filler_dim::Int         # dimension per filler
    commitment_cost::Float32  # VQ-VAE commitment loss weight (for training)
end

function PredicateEngram(
    embedding_dim::Int;
    code_dim::Int = 64,
    codebook_size::Int = 512,
    num_roles::Int = 4,
    filler_dim::Int = 0,
    commitment_cost::Float32 = 0.25f0,
)
    if filler_dim <= 0
        filler_dim = embedding_dim ÷ num_roles
    end
    embedding_dim == filler_dim * num_roles || throw(ArgumentError(
        "embedding_dim ($embedding_dim) must equal filler_dim ($filler_dim) × num_roles ($num_roles)"
    ))
    return PredicateEngram(embedding_dim, code_dim, codebook_size, num_roles, filler_dim, commitment_cost)
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::PredicateEngram)
    he_enc = Float32(sqrt(2.0 / (layer.code_dim + layer.embedding_dim)))
    he_out = Float32(sqrt(2.0 / (layer.embedding_dim * 2)))
    total_filler = layer.filler_dim * layer.num_roles  # == embedding_dim

    return (
        # VQ-VAE codebook: (code_dim, codebook_size)
        Codebook = randn(rng, Float32, layer.code_dim, layer.codebook_size) .* 0.1f0,
        # Encoder: embedding_dim → code_dim
        EncoderWeight = randn(rng, Float32, layer.code_dim, layer.embedding_dim) .* he_enc,
        EncoderBias = zeros(Float32, layer.code_dim),
        # Rule bank: each code → (num_roles, num_roles) mixing matrix
        # Stored as (num_roles * num_roles, codebook_size) for efficient gather
        RuleBank = _init_rule_bank(rng, layer.num_roles, layer.codebook_size),
        # Filler extraction (unbinding): embedding_dim → filler_dim * num_roles
        FillerWeight = randn(rng, Float32, total_filler, layer.embedding_dim) .* he_out,
        FillerBias = zeros(Float32, total_filler),
        # Output projection: total_filler → embedding_dim
        OutputWeight = randn(rng, Float32, layer.embedding_dim, total_filler) .* he_out,
        OutputBias = zeros(Float32, layer.embedding_dim),
        # Gate: embedding_dim → embedding_dim
        GateWeight = randn(rng, Float32, layer.embedding_dim, layer.embedding_dim) .* he_out,
        GateBias = fill(Float32(-2.0), layer.embedding_dim),  # starts near-closed
    )
end

function _init_rule_bank(rng::Random.AbstractRNG, num_roles::Int, codebook_size::Int)
    # Initialize rule matrices near identity (each rule starts as "pass-through")
    bank = zeros(Float32, num_roles * num_roles, codebook_size)
    for c in 1:codebook_size
        for r in 1:num_roles
            bank[(r - 1) * num_roles + r, c] = 1.0f0  # diagonal = identity
        end
        # Add small noise for symmetry breaking
        bank[:, c] .+= randn(rng, Float32, num_roles * num_roles) .* 0.02f0
    end
    return bank
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::PredicateEngram)
    return (;)  # no mutable state
end

# ============================================================================
# VQ-VAE Straight-Through Quantization
# ============================================================================

"""
    vq_quantize(query, codebook) → (quantized_st, indices)

Vector quantization with straight-through estimator.
Forward: returns nearest codebook vector. Backward: gradient passes through
as identity on `query` (codebook gets gradients via the gather).
"""
function vq_quantize(query, codebook)
    # query: (code_dim, N), codebook: (code_dim, codebook_size)

    # Find nearest codebook entry (no gradient through argmin)
    indices = ChainRulesCore.ignore_derivatives() do
        q_cpu = Array(query)
        c_cpu = Array(codebook)
        q_sq = sum(q_cpu .^ 2, dims=1)        # (1, N)
        c_sq = sum(c_cpu .^ 2, dims=1)        # (1, codebook_size)
        dots = c_cpu' * q_cpu                   # (codebook_size, N)
        dists = q_sq .+ c_sq' .- 2 .* dots     # (codebook_size, N)
        vec(map(ci -> ci[1], argmin(dists, dims=1)))  # (N,)
    end

    # Gather quantized vectors (differentiable w.r.t. codebook)
    quantized = codebook[:, indices]  # (code_dim, N)

    # Straight-through: forward = quantized, backward = identity on query
    diff = ChainRulesCore.ignore_derivatives() do
        quantized .- query
    end
    quantized_st = query .+ diff

    return quantized_st, indices
end

# ============================================================================
# Forward Pass
# ============================================================================

"""
    (layer::PredicateEngram)(hidden_state, ps, st) → (output, state)

Apply predicate engram reasoning injection.

1. Encode hidden → codebook space, quantize to discrete code
2. Look up rule mixing matrix from rule bank (O(1))
3. Unbind hidden into per-role fillers
4. Apply rule: batched matmul of rule_matrix × fillers (permutes/mixes roles)
5. Rebind, project, gate, residual inject
"""
function (layer::PredicateEngram)(hidden_state, ps, st)
    is_batched = ndims(hidden_state) == 3

    if !is_batched
        hidden_b = reshape(hidden_state, size(hidden_state, 1), size(hidden_state, 2), 1)
    else
        hidden_b = hidden_state
    end

    dim, seq_len, batch_size = size(hidden_b)
    N = seq_len * batch_size
    hidden_flat = reshape(hidden_b, dim, N)

    # =====================================================================
    # 1. Encode → VQ quantize → discrete reasoning code
    # =====================================================================
    query = ps.EncoderWeight * hidden_flat .+ ps.EncoderBias  # (code_dim, N)
    _quantized, indices = vq_quantize(query, ps.Codebook)

    # =====================================================================
    # 2. Look up rule mixing matrices from bank (O(1) per position)
    # =====================================================================
    # RuleBank: (num_roles², codebook_size)
    rule_flat = ps.RuleBank[:, indices]  # (num_roles², N)
    nr = layer.num_roles
    rule_matrices = reshape(rule_flat, nr, nr, N)  # (num_roles, num_roles, N)

    # =====================================================================
    # 3. Unbind: extract per-role fillers
    # =====================================================================
    fillers_flat = ps.FillerWeight * hidden_flat .+ ps.FillerBias  # (filler_dim * num_roles, N)
    fillers = reshape(fillers_flat, layer.filler_dim, nr, N)  # (filler_dim, num_roles, N)

    # =====================================================================
    # 4. Apply rule: batched matmul — rule mixes roles
    #    transformed[:, r_out, n] = Σ_r_in rule[r_out, r_in, n] * fillers[:, r_in, n]
    #    = fillers × rule^T  (batched)
    # =====================================================================
    rule_t = permutedims(rule_matrices, (2, 1, 3))  # (num_roles, num_roles, N) transposed
    transformed = NNlib.batched_mul(fillers, rule_t)  # (filler_dim, num_roles, N)

    # =====================================================================
    # 5. Rebind: flatten roles back to embedding dim
    # =====================================================================
    rebound = reshape(transformed, dim, N)  # (embedding_dim, N)

    # =====================================================================
    # 6. Output projection
    # =====================================================================
    projected = ps.OutputWeight * rebound .+ ps.OutputBias  # (embedding_dim, N)

    # =====================================================================
    # 7. Gate (starts near-closed, learns to open)
    # =====================================================================
    gate = NNlib.sigmoid.(ps.GateWeight * hidden_flat .+ ps.GateBias)

    # =====================================================================
    # 8. Residual injection
    # =====================================================================
    output_flat = hidden_flat .+ gate .* projected
    output = reshape(output_flat, dim, seq_len, batch_size)

    if !is_batched
        output = dropdims(output, dims=3)
    end

    return output, st
end

# ============================================================================
# Commitment loss (for training — call separately, not in forward pass)
# ============================================================================

"""
    predicate_engram_commitment_loss(hidden, ps, layer) → Float32

VQ-VAE commitment loss: encourages encoder outputs to stay close to codebook.
Call this in the training loop and add to the main loss with weight β.
"""
function predicate_engram_commitment_loss(hidden_flat, ps, layer)
    query = ps.EncoderWeight * hidden_flat .+ ps.EncoderBias
    _, indices = vq_quantize(query, ps.Codebook)
    quantized = ps.Codebook[:, indices]
    # ||z_e - sg(z_q)||² — encoder should commit to codebook
    return Float32(mean((query .- ChainRulesCore.ignore_derivatives(quantized)) .^ 2))
end

using Statistics: mean

# ============================================================================
# Exports
# ============================================================================

export PredicateEngram
export vq_quantize
export predicate_engram_commitment_loss

end # module
