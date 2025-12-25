module CRF

"""
Linear-chain Conditional Random Field (CRF) for BIO sequence labeling.

Enforces valid BIO transitions:
- O can go to O or B-*
- B-X can go to O, B-*, or I-X
- I-X can go to O, B-*, or I-X (same type only)

This prevents invalid sequences like O → I-PERSON or I-PERSON → I-AGENCY.
"""

using Lux
using Random
using NNlib
using Statistics: mean

# Import label mappings from parent
import ..NER: RAG_LABELS, LABEL_TO_ID, ID_TO_LABEL, NUM_LABELS, ENTITY_TYPES

const LuxLayer = Lux.AbstractLuxLayer

# =============================================================================
# Transition Validity
# =============================================================================

"""
    is_valid_transition(from_label::String, to_label::String) -> Bool

Check if transitioning from `from_label` to `to_label` is valid in BIO scheme.
"""
function is_valid_transition(from_label::String, to_label::String)
    # O can go to O or B-*
    if from_label == "O"
        return to_label == "O" || startswith(to_label, "B-")
    end

    # B-X can go to O, B-*, or I-X (same entity type)
    if startswith(from_label, "B-")
        entity_type = from_label[3:end]
        return to_label == "O" ||
               startswith(to_label, "B-") ||
               to_label == "I-$entity_type"
    end

    # I-X can go to O, B-*, or I-X (same entity type)
    if startswith(from_label, "I-")
        entity_type = from_label[3:end]
        return to_label == "O" ||
               startswith(to_label, "B-") ||
               to_label == "I-$entity_type"
    end

    return true
end

"""
    is_valid_transition(from_id::Int, to_id::Int) -> Bool

Check transition validity using label IDs.
"""
function is_valid_transition(from_id::Int, to_id::Int)
    from_label = ID_TO_LABEL[from_id]
    to_label = ID_TO_LABEL[to_id]
    return is_valid_transition(from_label, to_label)
end

"""
    build_transition_mask() -> Matrix{Float32}

Build a mask matrix where valid transitions are 0 and invalid are -Inf.
"""
function build_transition_mask()
    n = NUM_LABELS
    mask = zeros(Float32, n, n)

    for i in 1:n
        for j in 1:n
            if !is_valid_transition(i, j)
                mask[i, j] = -Inf32
            end
        end
    end

    return mask
end

# Pre-compute the transition mask
const TRANSITION_MASK = build_transition_mask()

# =============================================================================
# LinearChainCRF Layer
# =============================================================================

"""
    LinearChainCRF <: LuxLayer

Linear-chain CRF for sequence labeling with learned transition scores.

The CRF models:
    P(y|x) ∝ exp(∑ᵢ emission[yᵢ, i] + ∑ᵢ transition[yᵢ₋₁, yᵢ])

Training uses negative log-likelihood:
    loss = log Z(x) - score(x, y*)

where Z(x) is the partition function (sum over all possible label sequences).
"""
struct LinearChainCRF <: LuxLayer
    num_labels::Int

    function LinearChainCRF(num_labels::Int = NUM_LABELS)
        new(num_labels)
    end
end

function Lux.initialparameters(rng::Random.AbstractRNG, crf::LinearChainCRF)
    n = crf.num_labels

    # Initialize transitions with small random values
    # Valid transitions start near 0, invalid transitions are masked during forward
    transitions = randn(rng, Float32, n, n) * 0.1f0

    # Start transitions: score for starting with each label
    # O and B-* are valid starts, I-* are not
    start_transitions = randn(rng, Float32, n) * 0.1f0
    for i in 1:n
        label = ID_TO_LABEL[i]
        if startswith(label, "I-")
            start_transitions[i] = -10000.0f0
        end
    end

    # End transitions: score for ending with each label
    # All labels are valid ends
    end_transitions = randn(rng, Float32, n) * 0.1f0

    return (
        transitions = transitions,
        start_transitions = start_transitions,
        end_transitions = end_transitions,
    )
end

Lux.initialstates(::Random.AbstractRNG, ::LinearChainCRF) = (;)

# =============================================================================
# Forward Algorithm (Log Partition Function)
# =============================================================================

"""
    log_sum_exp(x; dims=1)

Numerically stable log-sum-exp.
"""
function log_sum_exp(x::AbstractArray; dims=1)
    max_x = maximum(x, dims=dims)
    return max_x .+ log.(sum(exp.(x .- max_x), dims=dims))
end

"""
    forward_algorithm(emissions, mask, params) -> log_partition

Compute log partition function Z(x) using forward algorithm.
Vectorized implementation for better GPU performance.

Arguments:
- emissions: (num_labels, seq_len, batch) - emission scores from encoder
- mask: (seq_len, batch) - attention mask (true = valid token)
- params: CRF parameters

Returns:
- log_partition: (batch,) - log Z(x) for each sequence
"""
function forward_algorithm(
    emissions::AbstractArray{T, 3},
    mask::AbstractMatrix{Bool},
    params
) where T
    num_labels, seq_len, batch_size = size(emissions)

    # Get masked transitions (valid transitions + learned scores)
    # (num_labels, num_labels)
    trans = params.transitions .+ TRANSITION_MASK

    # Initialize with start transitions + first emissions
    # alpha: (num_labels, batch)
    alpha = params.start_transitions .+ emissions[:, 1, :]

    # Iterate over sequence length
    for t in 2:seq_len
        # Broadcast alpha: (num_labels, 1, batch)
        # trans: (num_labels, num_labels, 1)
        # Sum: (source_label, target_label, batch)
        
        # log_sum_exp over source labels (dim 1)
        # This computes: log(sum(exp(alpha[i] + trans[i, j]))) for each j
        
        # We need to do this carefully to avoid huge allocations
        # shape: (num_labels, batch)
        # Using a custom reduction or NNlib if possible
        
        # Expand for broadcasting
        alpha_exp = reshape(alpha, num_labels, 1, batch_size)
        trans_exp = reshape(trans, num_labels, num_labels, 1)
        
        # (num_labels, num_labels, batch)
        scores = alpha_exp .+ trans_exp
        
        # LogSumExp over source labels (dim 1) -> (1, num_labels, batch)
        # Result is log prob of reaching each target label
        new_alpha = log_sum_exp(scores, dims=1)
        
        # Reshape back to (num_labels, batch)
        new_alpha = reshape(new_alpha, num_labels, batch_size)
        
        # Add emissions: (num_labels, batch)
        new_alpha = new_alpha .+ emissions[:, t, :]
        
        # Masking: if mask[t, b] is false, keep old alpha
        # mask[t, :]: (batch,)
        mask_t = reshape(mask[t, :], 1, batch_size)
        
        # If masked, we just carry forward the previous alpha (effectively ignoring this step)
        # But wait, if it's padding, we shouldn't have updated it at all.
        # The standard way is: alpha = mask * new_alpha + (1-mask) * alpha
        alpha = ifelse.(mask_t, new_alpha, alpha)
    end

    # Add end transitions: (num_labels, batch)
    alpha = alpha .+ params.end_transitions

    # Final log sum exp: (batch,)
    log_partition = dropdims(log_sum_exp(alpha, dims=1), dims=1)

    return log_partition
end

# =============================================================================
# Gold Score
# =============================================================================

"""
    compute_gold_score(emissions, labels, mask, params) -> gold_score

Compute the score of the gold label sequence.

Arguments:
- emissions: (num_labels, seq_len, batch)
- labels: (seq_len, batch) - gold label IDs
- mask: (seq_len, batch) - attention mask
- params: CRF parameters

Returns:
- gold_score: (batch,) - score of gold sequence for each batch element
"""
function compute_gold_score(
    emissions::AbstractArray{T, 3},
    labels::AbstractMatrix{<:Integer},
    mask::AbstractMatrix{Bool},
    params
) where T
    num_labels, seq_len, batch_size = size(emissions)
    
    # transitions: (num_labels, num_labels)
    # Get indices for start transitions
    # labels[1, :]: (batch_size,)
    start_scores = params.start_transitions[labels[1, :]] # (batch_size,)
    
    # First emission scores
    # emissions: (num_labels, seq_len, batch)
    # We need emissions[labels[1, b], 1, b] for each b
    # Use linear indexing or a clever slice
    
    # emissions_1: (num_labels, batch)
    emissions_1 = emissions[:, 1, :]
    # Get scores for the first label in each batch
    # emission_scores[b] = emissions_1[labels[1, b], b]
    # In Julia, we can use a loop for this specific indexing or sub2ind-style indexing
    # For Zygote compatibility, a simple map/broadcast is often best
    first_emission_scores = [emissions_1[labels[1, b], b] for b in 1:batch_size]
    
    score = start_scores .+ first_emission_scores

    # Accumulate emission and transition scores
    for t in 2:seq_len
        # mask[t, :]: (batch_size,)
        mask_t = mask[t, :]
        
        # prev_labels: (batch_size,)
        prev_labels = labels[t-1, :]
        # curr_labels: (batch_size,)
        curr_labels = labels[t, :]
        
        # Transition scores: params.transitions[prev, curr]
        # t_scores[b] = params.transitions[prev_labels[b], curr_labels[b]]
        t_scores = [params.transitions[prev_labels[b], curr_labels[b]] for b in 1:batch_size]
        
        # Emission scores: emissions[curr_label, t, batch]
        # e_scores[b] = emissions[curr_labels[b], t, b]
        emissions_t = emissions[:, t, :]
        e_scores = [emissions_t[curr_labels[b], b] for b in 1:batch_size]
        
        # Update score only where mask is true
        step_score = t_scores .+ e_scores
        score = score .+ (mask_t .* step_score)
    end

    # Add end transitions
    # We need to find the last valid label for each sequence
    # end_labels[b] = labels[last_pos[b], b]
    last_positions = [findlast(mask[:, b]) for b in 1:batch_size]
    # If no valid position, default to 1 (though should not happen with valid mask)
    end_labels = [labels[something(pos, 1), b] for (b, pos) in enumerate(last_positions)]
    
    end_scores = params.end_transitions[end_labels]
    score = score .+ end_scores

    return score
end

function something(x, default)
    return x === nothing ? default : x
end

# =============================================================================
# CRF Loss (Negative Log Likelihood)
# =============================================================================

"""
    crf_loss(crf, emissions, labels, mask, params, state) -> (loss, state)

Compute CRF negative log-likelihood loss.

Arguments:
- crf: LinearChainCRF layer
- emissions: (num_labels, seq_len, batch) - logits from encoder
- labels: (seq_len, batch) - gold label IDs (use ignore_index for padding)
- mask: (seq_len, batch) - attention mask (true = valid)
- params: CRF parameters
- state: CRF state (unused)

Returns:
- loss: scalar mean NLL
- state: unchanged state
"""
function crf_loss(
    crf::LinearChainCRF,
    emissions::AbstractArray{T, 3},
    labels::AbstractMatrix{<:Integer},
    mask::AbstractMatrix{Bool},
    params,
    state
) where T
    # Log partition function (sum over all possible sequences)
    log_partition = forward_algorithm(emissions, mask, params)

    # Gold sequence score
    gold_score = compute_gold_score(emissions, labels, mask, params)

    # NLL = log Z - gold_score
    nll = log_partition .- gold_score

    return mean(nll), state
end

# =============================================================================
# Viterbi Decoding
# =============================================================================

"""
    viterbi_decode(crf, emissions, mask, params, state) -> (predictions, state)

Find the most likely label sequence using Viterbi algorithm.

Arguments:
- crf: LinearChainCRF layer
- emissions: (num_labels, seq_len, batch)
- mask: (seq_len, batch) - attention mask
- params: CRF parameters
- state: CRF state

Returns:
- predictions: (seq_len, batch) - best label sequence
- state: unchanged state
"""
function viterbi_decode(
    crf::LinearChainCRF,
    emissions::AbstractArray{T, 3},
    mask::AbstractMatrix{Bool},
    params,
    state
) where T
    num_labels, seq_len, batch_size = size(emissions)

    # Get masked transitions: (num_labels, num_labels)
    trans = params.transitions .+ TRANSITION_MASK

    # Viterbi scores: (num_labels, seq_len, batch)
    viterbi = zeros(T, num_labels, seq_len, batch_size)
    # Backpointers: (num_labels, seq_len, batch)
    backpointers = zeros(Int, num_labels, seq_len, batch_size)

    # Initialize with start transitions + first emissions
    viterbi[:, 1, :] = params.start_transitions .+ emissions[:, 1, :]

    # Forward pass (find best paths)
    for t in 2:seq_len
        # mask[t, :]: (batch_size,)
        mask_t = mask[t, :]
        
        # Broadcast viterbi: (from_label, 1, batch)
        v_prev = reshape(viterbi[:, t-1, :], num_labels, 1, batch_size)
        # trans: (from_label, to_label, 1)
        t_exp = reshape(trans, num_labels, num_labels, 1)
        
        # scores: (from_label, to_label, batch)
        scores = v_prev .+ t_exp
        
        # Find best source label for each target label
        # best_prev_scores: (1, to_label, batch)
        best_prev_scores = maximum(scores, dims=1)
        # best_prev_idx: (1, to_label, batch)
        best_prev_idx = mapslices(argmax, scores, dims=1)
        
        # Update viterbi scores and backpointers
        # emissions_t: (to_label, batch)
        emissions_t = emissions[:, t, :]
        
        new_v = reshape(best_prev_scores, num_labels, batch_size) .+ emissions_t
        new_bp = reshape(best_prev_idx, num_labels, batch_size)
        
        # Apply mask: if !mask_t[b], keep previous values
        mask_exp = reshape(mask_t, 1, batch_size)
        viterbi[:, t, :] = ifelse.(mask_exp, new_v, viterbi[:, t-1, :])
        backpointers[:, t, :] = new_bp # Masking backpointers is less critical but good for clarity
    end

    # Backtracking (remains sequential per batch, but we can iterate over batches)
    predictions = zeros(Int, seq_len, batch_size)

    for b in 1:batch_size
        # Find last valid position
        last_pos = findlast(mask[:, b])
        if last_pos === nothing
            last_pos = 1
        end

        # Best final label
        final_scores = viterbi[:, last_pos, b] .+ params.end_transitions
        best_last = argmax(final_scores)
        predictions[last_pos, b] = best_last

        # Backtrack
        for t in (last_pos-1):-1:1
            predictions[t, b] = backpointers[predictions[t+1, b], t+1, b]
        end

        # Fill padding with O (label 1)
        if last_pos < seq_len
            predictions[(last_pos+1):seq_len, b] .= 1
        end
    end

    return predictions, state
end

# =============================================================================
# Combined CRF + Encoder Model
# =============================================================================

"""
    CRFTagger <: LuxLayer

Combines an encoder (producing emissions) with a CRF layer for sequence labeling.
"""
struct CRFTagger{E} <: LuxLayer
    encoder::E
    crf::LinearChainCRF
end

function CRFTagger(encoder; num_labels::Int = NUM_LABELS)
    return CRFTagger(encoder, LinearChainCRF(num_labels))
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::CRFTagger)
    return (
        encoder = Lux.initialparameters(rng, model.encoder),
        crf = Lux.initialparameters(rng, model.crf),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::CRFTagger)
    return (
        encoder = Lux.initialstates(rng, model.encoder),
        crf = Lux.initialstates(rng, model.crf),
    )
end

"""
Forward pass returning emissions (for training with crf_loss).
"""
function (model::CRFTagger)(token_ids, params, state)
    emissions, encoder_state = model.encoder(token_ids, params.encoder, state.encoder)
    new_state = (encoder = encoder_state, crf = state.crf)
    return emissions, new_state
end

"""
    predict(model::CRFTagger, token_ids, mask, params, state)

Get predictions using Viterbi decoding.
"""
function predict(model::CRFTagger, token_ids, mask, params, state)
    emissions, new_state = model(token_ids, params, state)
    predictions, _ = viterbi_decode(model.crf, emissions, mask, params.crf, state.crf)
    return predictions, new_state
end

# =============================================================================
# Exports
# =============================================================================

export LinearChainCRF, CRFTagger
export crf_loss, viterbi_decode, predict
export is_valid_transition, build_transition_mask
export log_sum_exp, forward_algorithm, compute_gold_score

end # module
