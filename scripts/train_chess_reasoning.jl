#!/usr/bin/env julia
"""
Phase 1: Chess Logic Pre-Training for ReasoningDrafter

Trains the ReasoningDrafter on Stockfish-evaluated positions to learn
the hidden features of constrained reasoning.

Architecture: RuleConditionedWavePDE → GLU(LinAttn ⊙ sigmoid(WavePDE)) → AlgebraicCircuit

Multi-task loss:
  α · CE(move_head, best_move)      — next-step prediction
  β · MSE(eval_head, stockfish_eval) — constraint satisfaction scoring
  γ · CE(board_head, next_position)  — board state prediction (optional)

Usage:
  julia --project=. scripts/train_chess_reasoning.jl [--config path.toml]
  julia --project=. scripts/train_chess_reasoning.jl --data data/chess/lichess_eval_sample_100k.jsonl --steps 1000

CUDA/GPU rules from CLAUDE.md:
  - NO try/catch in training loops (CUDA.jl #2197)
  - NO @info with interpolation in loops (implicit try/catch)
  - Use println() for logging
  - Set grads = nothing after withgradient
  - GC.gc(true); CUDA.reclaim() before gradient passes for large models
"""

using Swamma
using Swamma.ChessTokenizer
using Swamma.ChessDataset
using Swamma.ReasoningDrafterMod
using Swamma.RuleConditionedWavePDEMod

using Lux
using Random
using NNlib
using Optimisers
using Zygote

# ============================================================================
# Chess-specific model wrapper (adds move + eval heads)
# ============================================================================

struct ChessReasoningModel{D,MH,EH} <: Lux.AbstractLuxLayer
    drafter::D
    MoveHead::MH      # dim → MOVE_VOCAB_SIZE
    EvalHead::EH       # dim → 1 (scalar eval)
end

function ChessReasoningModel(config::ReasoningDrafterConfig)
    drafter = ReasoningDrafter(config)
    return ChessReasoningModel(
        drafter,
        Lux.Dense(config.embedding_dimension => MOVE_VOCAB_SIZE),
        Lux.Dense(config.embedding_dimension => 1),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, model::ChessReasoningModel)
    return (
        Drafter = Lux.initialparameters(rng, model.drafter),
        MoveHead = Lux.initialparameters(rng, model.MoveHead),
        EvalHead = Lux.initialparameters(rng, model.EvalHead),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, model::ChessReasoningModel)
    return (
        Drafter = Lux.initialstates(rng, model.drafter),
        MoveHead = Lux.initialstates(rng, model.MoveHead),
        EvalHead = Lux.initialstates(rng, model.EvalHead),
    )
end

"""
Forward pass: board tokens → (move_logits, eval_score, drafter_logits)

Uses the drafter's hidden representation (before output head) for move + eval prediction.
The drafter's own output head predicts board tokens (next piece at each square).
"""
function forward_chess(model::ChessReasoningModel, board_tokens, ps, st)
    # Run drafter forward to get hidden states
    # We need the pre-output-head hidden — extract from the drafter internals
    drafter = model.drafter
    config = drafter.config

    tokens = board_tokens  # (64, batch)
    seq_len, batch_size = size(tokens)

    # 1. Embeddings
    tok_flat = vec(tokens)
    tok_emb_flat, tok_st = drafter.TokenEmbedding(tok_flat, ps.Drafter.TokenEmbedding, st.Drafter.TokenEmbedding)
    tok_emb = reshape(tok_emb_flat, config.embedding_dimension, seq_len, batch_size)

    pos_indices = collect(1:min(seq_len, config.max_sequence_length))
    pos_emb_raw, pos_st = drafter.PositionEmbedding(pos_indices, ps.Drafter.PositionEmbedding, st.Drafter.PositionEmbedding)
    pos_emb = reshape(pos_emb_raw, config.embedding_dimension, seq_len, 1)

    hidden = tok_emb .+ pos_emb

    # 2. Time embedding
    time_emb = repeat(reshape(ps.Drafter.TimeEmbedding, :, 1), 1, batch_size)

    # 3. Blocks
    block_states = Vector{Any}(undef, config.number_of_layers)
    for (i, block) in enumerate(drafter.Blocks)
        key = Symbol("Block_$i")
        hidden, block_state = block((hidden, time_emb), ps.Drafter.Blocks[key], st.Drafter.Blocks[key])
        block_states[i] = block_state
    end

    # 4. Final norm
    hidden, fn_st = drafter.FinalNorm(hidden, ps.Drafter.FinalNorm, st.Drafter.FinalNorm)

    # 5. Task heads on pooled hidden (mean over sequence for global prediction)
    hidden_pooled = dropdims(sum(hidden, dims=2), dims=2) ./ Float32(seq_len)  # (dim, batch)

    move_logits, mh_st = model.MoveHead(hidden_pooled, ps.MoveHead, st.MoveHead)  # (MOVE_VOCAB, batch)
    eval_pred, eh_st = model.EvalHead(hidden_pooled, ps.EvalHead, st.EvalHead)     # (1, batch)

    # 6. Board token logits (per-square piece prediction via drafter's output head)
    board_logits, oh_st = drafter.OutputHead(hidden, ps.Drafter.OutputHead, st.Drafter.OutputHead)  # (piece_vocab, 64, batch)

    new_st = (
        Drafter = (
            TokenEmbedding = tok_st,
            PositionEmbedding = pos_st,
            Blocks = NamedTuple{ntuple(i -> Symbol("Block_$i"), config.number_of_layers)}(Tuple(block_states)),
            FinalNorm = fn_st,
            OutputHead = oh_st,
        ),
        MoveHead = mh_st,
        EvalHead = eh_st,
    )

    return move_logits, eval_pred, board_logits, new_st
end

# ============================================================================
# Loss functions
# ============================================================================

function chess_multi_task_loss(model, ps, st, batch; α=1.0f0, β=0.5f0)
    move_logits, eval_pred, _board_logits, new_st = forward_chess(
        model, batch.board_tokens, ps, st
    )

    # Move prediction loss (cross-entropy)
    move_targets = batch.best_move_ids  # (batch,)
    log_probs = NNlib.logsoftmax(move_logits, dims=1)  # (MOVE_VOCAB, batch)
    move_loss = -sum(log_probs[CartesianIndex.(move_targets, 1:length(move_targets))]) / length(move_targets)

    # Eval prediction loss (MSE on normalized eval)
    eval_targets = reshape(batch.eval_scores, 1, :)  # (1, batch)
    eval_loss = sum((eval_pred .- eval_targets) .^ 2) / length(batch.eval_scores)

    total = α * move_loss + β * eval_loss

    return total, (move_loss = move_loss, eval_loss = eval_loss, new_st = new_st)
end

# ============================================================================
# Training loop
# ============================================================================

function train_chess_phase1(;
    data_path::String,
    max_positions::Int = 100_000,
    min_depth::Int = 10,
    batch_size::Int = 64,
    num_epochs::Int = 10,
    learning_rate::Float32 = 1f-3,
    checkpoint_dir::String = "checkpoints/chess_reasoning",
    checkpoint_every::Int = 500,
    seed::Int = 42,
)
    rng = Random.MersenneTwister(seed)

    println("=== Phase 1: Chess Logic Pre-Training ===")
    println("Data: $data_path")
    println("Max positions: $max_positions, min depth: $min_depth")

    # Load data
    println("Loading positions...")
    positions = load_lichess_jsonl(data_path; max_positions=max_positions, min_depth=min_depth)
    println("Loaded $(length(positions)) positions")
    length(positions) == 0 && error("No positions loaded")

    # Model config — 64 squares, 13 piece types
    config = ReasoningDrafterConfig(
        vocab_size = PIECE_VOCAB_SIZE,    # 13
        max_sequence_length = NUM_SQUARES, # 64
        embedding_dimension = 256,
        number_of_heads = 4,
        number_of_layers = 2,
        time_dimension = 64,
        rc_code_dim = 64,
        rc_codebook_size = 512,
        circuit_num_leaves = 16,
        circuit_product_arity = 2,
        circuit_num_sums = 8,
        circuit_num_circuits = 4,
    )

    model = ChessReasoningModel(config)
    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)

    # Count parameters
    function count_params(x)
        x isa AbstractArray && return length(x)
        x isa NamedTuple && return sum(count_params(v) for v in values(x))
        x isa Tuple && return sum(count_params(v) for v in x)
        x isa Nothing && return 0
        return 0
    end
    total_params = count_params(ps)
    println("Model parameters: $(round(total_params / 1e6, digits=3))M")

    # Optimizer
    opt = Optimisers.Adam(learning_rate)
    opt_state = Optimisers.setup(opt, ps)

    # Training
    mkpath(checkpoint_dir)
    global_step = 0
    best_move_acc = 0.0

    for epoch in 1:num_epochs
        batches = iterate_batches(positions, batch_size; shuffle=true, rng=rng)
        epoch_move_loss = 0.0
        epoch_eval_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for (batch_idx, batch) in enumerate(batches)
            global_step += 1
            bs = size(batch.board_tokens, 2)

            # Gradient computation
            (loss_val, aux), grads = Zygote.withgradient(ps) do p
                chess_multi_task_loss(model, p, st, batch)
            end
            grads = grads[1]

            # Update parameters
            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            grads = nothing  # help GC free AD tape

            # Update state (EMA codebook)
            st = aux.new_st

            # Track metrics
            epoch_move_loss += Float64(aux.move_loss)
            epoch_eval_loss += Float64(aux.eval_loss)

            # Move accuracy (top-1)
            move_logits, _, _, _ = forward_chess(model, batch.board_tokens, ps, st)
            predicted_moves = vec(map(ci -> ci[1], argmax(move_logits, dims=1)))
            correct = sum(predicted_moves .== batch.best_move_ids)
            epoch_correct += correct
            epoch_total += bs

            # Logging
            if global_step % 50 == 0
                println("step=$global_step  loss=$(round(Float64(loss_val), digits=4))  " *
                        "move_loss=$(round(Float64(aux.move_loss), digits=4))  " *
                        "eval_loss=$(round(Float64(aux.eval_loss), digits=4))  " *
                        "move_acc=$(round(correct/bs, digits=3))")
            end

            # Checkpoint
            if global_step % checkpoint_every == 0
                cp_path = joinpath(checkpoint_dir, "step_$(global_step).jld2")
                println("Saving checkpoint: $cp_path")
                # JLD2 save would go here — requires JLD2 dependency
                # JLD2.@save cp_path ps st config global_step
            end
        end

        # Epoch summary
        n_batches = length(batches)
        avg_move_loss = epoch_move_loss / n_batches
        avg_eval_loss = epoch_eval_loss / n_batches
        move_acc = epoch_correct / max(epoch_total, 1)
        println("--- Epoch $epoch/$num_epochs ---")
        println("  avg_move_loss=$(round(avg_move_loss, digits=4))  avg_eval_loss=$(round(avg_eval_loss, digits=4))")
        println("  move_accuracy=$(round(move_acc, digits=4)) ($epoch_correct/$epoch_total)")

        if move_acc > best_move_acc
            best_move_acc = move_acc
            println("  New best move accuracy!")
        end
    end

    println("\n=== Training complete ===")
    println("Best move accuracy: $(round(best_move_acc, digits=4))")

    return model, ps, st
end

# ============================================================================
# CLI entry point
# ============================================================================

function main()
    data_path = "data/chess/lichess_db_eval.jsonl"
    max_positions = 100_000
    steps = nothing

    # Simple arg parsing
    args = ARGS
    i = 1
    while i <= length(args)
        if args[i] == "--data" && i < length(args)
            data_path = args[i+1]
            i += 2
        elseif args[i] == "--max-positions" && i < length(args)
            max_positions = parse(Int, args[i+1])
            i += 2
        elseif args[i] == "--steps" && i < length(args)
            steps = parse(Int, args[i+1])
            i += 2
        else
            i += 1
        end
    end

    num_epochs = steps === nothing ? 10 : max(1, steps ÷ max(max_positions ÷ 64, 1))

    train_chess_phase1(;
        data_path = data_path,
        max_positions = max_positions,
        num_epochs = num_epochs,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
