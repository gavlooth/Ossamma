module CircuitLayerMod

"""
Algebraic Circuit Layer: Structured Reasoning via Sum-Product Networks

Implements a bank of decomposable algebraic circuits as einsum-native tensor
operations within a transformer layer. Each circuit is a 2-level sum-product
network with guaranteed structural properties:

- **Decomposable**: product node children have disjoint variable scopes
- **Smooth**: sum node children share the same scope
- **Composable**: circuits in the bank can be mixed (sum) or multiplied (product)

Architecture per circuit:
    hidden → leaf predicates (sigmoid → [0,1] truth values)
    → product layer (grouped multiply — logical AND)
    → sum layer (weighted combine — logical OR / soft rules)
    → circuit output vector

Multiple circuits run in parallel. Their outputs compose via:
- :mix   — weighted sum (mixture of circuit conclusions)
- :product — element-wise product (conjunction of circuit conclusions)

The composed output is projected and gated into the hidden state.

This is equivalent to evaluating a bank of tractable probabilistic models
where leaf distributions are neurally parameterized.
"""

using Lux
using Random
using NNlib
using ChainRulesCore

using ..Swamma: LuxLayer

# ============================================================================
# AlgebraicCircuitLayer
# ============================================================================

struct AlgebraicCircuitLayer <: LuxLayer
    embedding_dim::Int
    num_leaves::Int         # leaf predicates per circuit
    product_arity::Int      # leaves per product group (must divide num_leaves)
    num_products::Int       # = num_leaves ÷ product_arity
    num_sums::Int           # sum nodes per circuit (circuit output dim)
    num_circuits::Int       # parallel circuits in the bank
    compose_mode::Symbol    # :mix or :product
end

function AlgebraicCircuitLayer(
    embedding_dim::Int;
    num_leaves::Int = 16,
    product_arity::Int = 2,
    num_sums::Int = 8,
    num_circuits::Int = 4,
    compose_mode::Symbol = :mix,
)
    num_leaves % product_arity == 0 || throw(ArgumentError(
        "num_leaves ($num_leaves) must be divisible by product_arity ($product_arity)"
    ))
    num_products = num_leaves ÷ product_arity
    compose_mode in (:mix, :product) || throw(ArgumentError(
        "compose_mode must be :mix or :product, got $(repr(compose_mode))"
    ))
    return AlgebraicCircuitLayer(
        embedding_dim, num_leaves, product_arity, num_products,
        num_sums, num_circuits, compose_mode,
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::AlgebraicCircuitLayer)
    nc = layer.num_circuits
    he = Float32(sqrt(2.0 / (layer.embedding_dim + layer.num_leaves)))

    return (
        # Leaf predicates: embedding_dim → num_leaves, per circuit
        # Shape: (num_leaves, embedding_dim, num_circuits)
        LeafWeights = randn(rng, Float32, layer.num_leaves, layer.embedding_dim, nc) .* he,
        LeafBiases = zeros(Float32, layer.num_leaves, nc),
        # Sum weights: (num_sums, num_products, num_circuits) — non-negative via softplus
        SumLogWeights = randn(rng, Float32, layer.num_sums, layer.num_products, nc) .* 0.1f0,
        # Circuit composition weights (for :mix mode): (num_circuits,)
        ComposeLogWeights = zeros(Float32, nc),
        # Output projection: num_sums → embedding_dim
        OutputWeight = randn(rng, Float32, layer.embedding_dim, layer.num_sums) .*
            Float32(sqrt(2.0 / (layer.embedding_dim + layer.num_sums))),
        OutputBias = zeros(Float32, layer.embedding_dim),
        # Gate
        GateWeight = randn(rng, Float32, layer.embedding_dim, layer.embedding_dim) .*
            Float32(sqrt(2.0 / (layer.embedding_dim * 2))),
        GateBias = fill(Float32(-2.0), layer.embedding_dim),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::AlgebraicCircuitLayer)
    return (;)
end

# ============================================================================
# Forward Pass
# ============================================================================

function _algebraic_circuit_forward(layer::AlgebraicCircuitLayer, hidden_b, ps)
    dim, seq_len, batch_size = size(hidden_b)
    N = seq_len * batch_size
    hidden_flat = reshape(hidden_b, dim, N)  # (dim, N)
    nc = layer.num_circuits

    # =====================================================================
    # 1. Leaf predicates: neural truth values in [0, 1]
    #    For each circuit c: leaves_c = sigmoid(W_c * hidden + b_c)
    # =====================================================================
    # Batched across circuits: (num_leaves, N, num_circuits)
    leaves = _compute_leaves(hidden_flat, ps.LeafWeights, ps.LeafBiases, nc)

    # =====================================================================
    # 2. Product layer: decomposable conjunction (group + multiply)
    #    Groups of `product_arity` leaves are multiplied.
    #    Uses log-space for numerical stability.
    # =====================================================================
    # Reshape: (product_arity, num_products, N, nc) → sum logs → exp
    products = _compute_products(leaves, layer.product_arity, layer.num_products, N, nc)

    # =====================================================================
    # 3. Sum layer: smooth disjunction (weighted sum)
    #    Non-negative weights via softplus ensure valid mixture.
    # =====================================================================
    sum_weights_norm = _normalize_sum_weights(ps.SumLogWeights)

    # Batched matmul: for each circuit c, sums_c = W_c * products_c
    # products: (num_products, N, nc) → need (num_products, N) per circuit
    # sums: (num_sums, N, nc)
    sums = _batched_sum_layer(sum_weights_norm, products, layer.num_sums, N, nc)

    # =====================================================================
    # 4. Compose circuits: mix or multiply
    # =====================================================================
    composed = _compose_circuits(Val(layer.compose_mode), sums, ps.ComposeLogWeights, nc)

    # =====================================================================
    # 5. Output projection + gate + inject
    # =====================================================================
    output_flat = _project_and_gate(
        hidden_flat,
        composed,
        ps.OutputWeight,
        ps.OutputBias,
        ps.GateWeight,
        ps.GateBias,
    )

    return reshape(output_flat, dim, seq_len, batch_size)
end

"""
    (layer::AlgebraicCircuitLayer)(hidden_state, ps, st) → (output, state)

Evaluate a bank of algebraic circuits on neural leaf predicates derived
from the hidden state, compose their outputs, and gate-inject the result.

All operations are standard tensor ops (einsum-native), fully GPU-compatible.
"""
function (layer::AlgebraicCircuitLayer)(hidden_state::AbstractMatrix, ps, st)
    hidden_b = reshape(hidden_state, size(hidden_state, 1), size(hidden_state, 2), 1)
    output_b = _algebraic_circuit_forward(layer, hidden_b, ps)
    return dropdims(output_b, dims=3), st
end

function (layer::AlgebraicCircuitLayer)(hidden_state::AbstractArray{<:Any, 3}, ps, st)
    return _algebraic_circuit_forward(layer, hidden_state, ps), st
end

# ============================================================================
# Internal helpers (Zygote-friendly, no mutation)
# ============================================================================

function _compute_leaves(hidden_flat, leaf_weights, leaf_biases, nc)
    # hidden_flat: (dim, N)
    # leaf_weights: (num_leaves, dim, nc)
    # leaf_biases: (num_leaves, nc)
    # Returns: (num_leaves, N, nc)
    dim, N = size(hidden_flat)
    num_leaves = size(leaf_weights, 1)

    # Compute all circuits at once via reshape + batched_mul
    # W: (num_leaves, dim, nc) → treat as batch of (num_leaves, dim) matrices
    # h: (dim, N) → broadcast across circuits
    # result: (num_leaves, N, nc)

    # Manual loop-free: reshape W to (num_leaves * nc, dim), multiply, reshape back
    W_flat = reshape(leaf_weights, num_leaves * nc, dim)  # (num_leaves*nc, dim)
    logits_flat = W_flat * hidden_flat  # (num_leaves*nc, N)
    logits = reshape(logits_flat, num_leaves, nc, N)  # (num_leaves, nc, N)
    logits = permutedims(logits, (1, 3, 2))  # (num_leaves, N, nc)

    # Add bias: (num_leaves, nc) → (num_leaves, 1, nc)
    bias = reshape(leaf_biases, num_leaves, 1, nc)
    logits = logits .+ bias

    return NNlib.sigmoid.(logits)
end

function _compute_products(leaves, product_arity, num_products, N, nc)
    leaves = clamp.(leaves, 1f-6, 1f0 - 1f-6)
    log_leaves = log.(leaves)
    log_leaves_grouped = reshape(log_leaves, product_arity, num_products, N, nc)
    log_products = dropdims(sum(log_leaves_grouped, dims=1), dims=1)
    return exp.(log_products)
end

function _normalize_sum_weights(sum_log_weights)
    sum_weights = NNlib.softplus.(sum_log_weights)
    return sum_weights ./ (sum(sum_weights, dims=2) .+ 1f-8)
end

function _batched_sum_layer(weights_norm, products, num_sums, N, nc)
    # weights_norm: (num_sums, num_products, nc)
    # products: (num_products, N, nc)
    # Returns: (num_sums, N, nc)

    # For each circuit c: sums[:, :, c] = weights[:, :, c] * products[:, :, c]
    # This is a batched matmul with batch dim = nc

    # Reshape for NNlib.batched_mul:
    # A: (num_sums, num_products, nc) — already correct
    # B: (num_products, N, nc) — already correct
    # Result: (num_sums, N, nc)
    return NNlib.batched_mul(weights_norm, products)
end

function _compose_circuits(::Val{:mix}, sums, compose_log_weights, nc)
    mix_weights = NNlib.softmax(compose_log_weights)
    mix_w = reshape(mix_weights, 1, 1, nc)
    return dropdims(sum(sums .* mix_w, dims=3), dims=3)
end

function _compose_circuits(::Val{:product}, sums, compose_log_weights, nc)
    return dropdims(prod(sums, dims=3), dims=3)
end

function _project_and_gate(hidden_flat, composed, output_weight, output_bias, gate_weight, gate_bias)
    projected = output_weight * composed .+ output_bias
    gate = NNlib.sigmoid.(gate_weight * hidden_flat .+ gate_bias)
    return hidden_flat .+ gate .* projected
end

# ============================================================================
# Circuit inspection utilities
# ============================================================================

"""
    circuit_leaf_activations(hidden_flat, ps, layer) → (num_leaves, N, num_circuits)

Extract the neural predicate truth values without running the full circuit.
Useful for interpretability — see which predicates fire for which inputs.
"""
function circuit_leaf_activations(hidden_flat, ps, layer::AlgebraicCircuitLayer)
    return _compute_leaves(hidden_flat, ps.LeafWeights, ps.LeafBiases, layer.num_circuits)
end

"""
    circuit_structure_summary(layer) → String

Print the circuit topology.
"""
function circuit_structure_summary(layer::AlgebraicCircuitLayer)
    return string(
        "AlgebraicCircuitLayer:\n",
        "  $(layer.num_circuits) parallel circuits\n",
        "  $(layer.num_leaves) leaves → ",
        "$(layer.num_products) products (arity $(layer.product_arity)) → ",
        "$(layer.num_sums) sums\n",
        "  Compose: $(layer.compose_mode)\n",
        "  Structural: decomposable ✓  smooth ✓",
    )
end

# ============================================================================
# Exports
# ============================================================================

export AlgebraicCircuitLayer
export circuit_leaf_activations, circuit_structure_summary

end # module
