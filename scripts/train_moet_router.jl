#!/usr/bin/env julia
"""
Minimal MoET router training on synthetic token categories.
"""

using Random
using Printf

include(joinpath(@__DIR__, "..", "src", "Ossamma.jl"))
using .Ossamma
using .Ossamma: MoETConfig, MoETModel
using .Ossamma: heuristic_labels_batch, router_loss, router_metrics

using Lux
using Optimisers
using Zygote

println("=" ^ 60)
println("MoET Router Training (Synthetic)")
println("=" ^ 60)

const LOGIC_TOKENS = ["forall", "exists", "->", "=>", "implies", "proof", "iff"]
const MATH_TOKENS = ["1", "2", "3", "+", "-", "=", "equation", "solve"]
const MEMORY_TOKENS = ["who", "what", "where", "when", "which", "define", "capital"]
const LANGUAGE_TOKENS = ["the", "and", "is", "we", "this", "that", "people", "story"]

const VOCAB = vcat(LANGUAGE_TOKENS, LOGIC_TOKENS, MATH_TOKENS, MEMORY_TOKENS)
const VOCAB_SIZE = length(VOCAB)
const TOKEN_TO_ID = Dict(tok => i for (i, tok) in enumerate(VOCAB))

function sample_sequence(seq_len::Int; p_logic::Float32 = 0.2f0, p_math::Float32 = 0.2f0, p_memory::Float32 = 0.2f0)
    tokens = Vector{String}(undef, seq_len)
    for t in 1:seq_len
        r = rand()
        if r < p_logic
            tokens[t] = rand(LOGIC_TOKENS)
        elseif r < p_logic + p_math
            tokens[t] = rand(MATH_TOKENS)
        elseif r < p_logic + p_math + p_memory
            tokens[t] = rand(MEMORY_TOKENS)
        else
            tokens[t] = rand(LANGUAGE_TOKENS)
        end
    end
    return tokens
end

function tokens_to_ids(tokens::Vector{String})
    ids = Vector{Int}(undef, length(tokens))
    for i in eachindex(tokens)
        ids[i] = TOKEN_TO_ID[tokens[i]]
    end
    return ids
end

function make_batch(seq_len::Int, batch_size::Int)
    batch_tokens = Vector{Vector{String}}(undef, batch_size)
    token_ids = Array{Int}(undef, seq_len, batch_size)
    for b in 1:batch_size
        tokens = sample_sequence(seq_len)
        batch_tokens[b] = tokens
        token_ids[:, b] = tokens_to_ids(tokens)
    end
    return token_ids, batch_tokens
end

config = MoETConfig(
    vocab_size = VOCAB_SIZE,
    max_sequence_length = 64,
    embedding_dimension = 64,
    number_of_heads = 4,
    number_of_experts = 4,
    layers_per_expert = 2,
    time_dimension = 32,
    router_hidden_dim = 64,
    top_k = 0,
    dropout_rate = 0.0f0,
)

println("\nCreating MoET model...")
model = MoETModel(config)

rng = Random.default_rng()
params, state = Lux.setup(rng, model)

opt = Optimisers.AdamW(1e-3, (0.9, 0.999), 0.01f0)
opt_state = Optimisers.setup(opt, params)

steps = 50
seq_len = 32
batch_size = 8
log_every = 10

println("\nTraining router for $(steps) steps...")
for step in 1:steps
    token_ids, batch_tokens = make_batch(seq_len, batch_size)
    labels = heuristic_labels_batch(batch_tokens)

    (loss, new_state), grads = Zygote.withgradient(params) do p
        _, gates, st = model((token_ids = token_ids,), p, state; return_gates = true)
        l = router_loss(gates, labels; λ_balance = 0.01f0, λ_entropy = 0.01f0, hard = false)
        return l, st
    end

    opt_state, params = Optimisers.update(opt_state, params, grads[1])
    state = new_state

    if step % log_every == 0 || step == 1
        _, gates, _ = model((token_ids = token_ids,), params, state; return_gates = true)
        metrics = router_metrics(gates, labels)
        @printf("  Step %d: loss=%.4f, acc=%.3f, balance=%.4f, entropy=%.4f\n",
            step, loss, metrics.accuracy, metrics.balance_loss, metrics.entropy)
    end
end

println("\nDone.")
