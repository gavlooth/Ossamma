#!/usr/bin/env julia
"""
Phase 1: Chess Logic Pre-Training for ReasoningDrafter

Trains the ReasoningDrafter on Stockfish-evaluated chess positions to learn
the hidden features of constrained reasoning via multi-task prediction.

Architecture: RuleConditionedWavePDE → GLU(LinAttn ⊙ sigmoid(WavePDE)) → AlgebraicCircuit

Multi-task loss:
  α · CE(move_head, best_move)       — next-step prediction
  β · MSE(eval_head, stockfish_eval) — constraint satisfaction scoring

Target hardware: NVIDIA Spark GB10, 130GB unified memory, CUDA 13.0, aarch64

CUDA/GPU rules (CLAUDE.md):
  - NO try/catch inside training loops (CUDA.jl #2197)
  - NO @info with interpolation in loops (implicit try/catch)
  - Use println() for logging
  - Set grads = nothing after withgradient
  - GC.gc(true); CUDA.reclaim() before gradient passes for large models

Usage:
  julia --project=. scripts/train_chess_reasoning.jl --data data/chess/lichess_db_eval.jsonl
  julia --project=. scripts/train_chess_reasoning.jl --data data/chess/sample_100k.jsonl --steps 200
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
using JLD2

# Optional GPU
const USE_GPU = try
    using CUDA
    CUDA.functional()
catch
    false
end

if USE_GPU
    println("GPU: $(CUDA.name(CUDA.device())), $(round(CUDA.total_memory() / 1e9, digits=1))GB")
else
    println("GPU: not available, using CPU")
end

function to_device(x)
    USE_GPU ? CUDA.cu(x) : x
end

# ============================================================================
# Chess-specific model wrapper
# ============================================================================

struct ChessReasoningModel{D,MH,EH}
    drafter::D
    MoveHead::MH
    EvalHead::EH
end

function ChessReasoningModel(config::ReasoningDrafterConfig)
    drafter = ReasoningDrafter(config)
    return ChessReasoningModel(
        drafter,
        Lux.Dense(config.embedding_dimension => MOVE_VOCAB_SIZE),
        Lux.Dense(config.embedding_dimension => 1),
    )
end

function init_chess_model(rng, config)
    model = ChessReasoningModel(config)
    drafter_ps = Lux.initialparameters(rng, model.drafter)
    drafter_st = Lux.initialstates(rng, model.drafter)
    move_ps = Lux.initialparameters(rng, model.MoveHead)
    move_st = Lux.initialstates(rng, model.MoveHead)
    eval_ps = Lux.initialparameters(rng, model.EvalHead)
    eval_st = Lux.initialstates(rng, model.EvalHead)

    ps = (Drafter = drafter_ps, MoveHead = move_ps, EvalHead = eval_ps)
    st = (Drafter = drafter_st, MoveHead = move_st, EvalHead = eval_st)
    return model, ps, st
end

function forward_chess(model, board_tokens, ps, st)
    drafter = model.drafter
    config = drafter.config
    seq_len, batch_size = size(board_tokens)

    tok_flat = vec(board_tokens)
    tok_emb_flat, tok_st = drafter.TokenEmbedding(tok_flat, ps.Drafter.TokenEmbedding, st.Drafter.TokenEmbedding)
    tok_emb = reshape(tok_emb_flat, config.embedding_dimension, seq_len, batch_size)

    pos_indices = collect(1:min(seq_len, config.max_sequence_length))
    pos_indices = to_device(pos_indices)
    pos_emb_raw, pos_st = drafter.PositionEmbedding(pos_indices, ps.Drafter.PositionEmbedding, st.Drafter.PositionEmbedding)
    pos_emb = reshape(pos_emb_raw, config.embedding_dimension, seq_len, 1)

    hidden = tok_emb .+ pos_emb
    time_emb = repeat(reshape(ps.Drafter.TimeEmbedding, :, 1), 1, batch_size)

    block_states = Vector{Any}(undef, config.number_of_layers)
    for (i, block) in enumerate(drafter.Blocks)
        key = Symbol("Block_$i")
        hidden, block_state = block((hidden, time_emb), ps.Drafter.Blocks[key], st.Drafter.Blocks[key])
        block_states[i] = block_state
    end

    hidden, fn_st = drafter.FinalNorm(hidden, ps.Drafter.FinalNorm, st.Drafter.FinalNorm)

    # Pool over sequence for global predictions
    hidden_pooled = dropdims(sum(hidden, dims=2), dims=2) ./ Float32(seq_len)

    move_logits, mh_st = model.MoveHead(hidden_pooled, ps.MoveHead, st.MoveHead)
    eval_pred, eh_st = model.EvalHead(hidden_pooled, ps.EvalHead, st.EvalHead)

    return move_logits, eval_pred
end

function chess_loss(model, ps, st, batch_tokens, batch_moves, batch_evals; α=1.0f0, β=0.5f0)
    move_logits, eval_pred = forward_chess(model, batch_tokens, ps, st)

    log_probs = NNlib.logsoftmax(move_logits, dims=1)
    move_loss = -sum(log_probs[CartesianIndex.(batch_moves, 1:length(batch_moves))]) / length(batch_moves)

    eval_targets = reshape(batch_evals, 1, :)
    eval_loss = sum((eval_pred .- eval_targets) .^ 2) / length(batch_evals)

    return α * move_loss + β * eval_loss, (move_loss, eval_loss)
end

# ============================================================================
# Training loop
# ============================================================================

function train_phase1(;
    data_path::String,
    max_positions::Int = 1_000_000,
    min_depth::Int = 10,
    batch_size::Int = 64,
    num_epochs::Int = 10,
    learning_rate::Float32 = 1f-3,
    checkpoint_dir::String = "checkpoints/reasoning_drafter/phase1",
    checkpoint_every::Int = 500,
    seed::Int = 42,
)
    rng = Random.MersenneTwister(seed)
    mkpath(checkpoint_dir)

    println("=== Phase 1: Chess Logic Pre-Training ===")
    println("Data: $data_path")
    println("Max positions: $max_positions, min depth: $min_depth")
    println("Batch size: $batch_size, LR: $learning_rate")

    # Load data
    println("Loading positions...")
    positions = load_lichess_jsonl(data_path; max_positions=max_positions, min_depth=min_depth)
    println("Loaded $(length(positions)) positions")
    length(positions) == 0 && error("No positions loaded")

    # Model — 64 squares, 13 piece types (chess-specific)
    config = ReasoningDrafterConfig(
        vocab_size = PIECE_VOCAB_SIZE,
        max_sequence_length = NUM_SQUARES,
        embedding_dimension = 256,
        number_of_heads = 4,
        number_of_layers = 2,
        time_dimension = 64,
        rc_code_dim = 64,
        rc_codebook_size = 512,
        rc_integration_steps = 8,
        circuit_num_leaves = 16,
        circuit_product_arity = 2,
        circuit_num_sums = 8,
        circuit_num_circuits = 4,
        use_adapters = false,
    )

    model, ps, st = init_chess_model(rng, config)

    if USE_GPU
        ps = to_device(ps)
        st = to_device(st)
    end

    function count_params(x)
        x isa AbstractArray && return length(x)
        x isa NamedTuple && return sum(count_params(v) for v in values(x))
        x isa Tuple && return sum(count_params(v) for v in x)
        x isa Nothing && return 0
        return 0
    end
    println("Parameters: $(round(count_params(ps) / 1e6, digits=3))M")

    opt = Optimisers.Adam(learning_rate)
    opt_state = Optimisers.setup(opt, ps)

    global_step = 0
    best_loss = Inf

    for epoch in 1:num_epochs
        batches = iterate_batches(positions, batch_size; shuffle=true, rng=rng)
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch in batches
            global_step += 1
            bs = size(batch.board_tokens, 2)

            # Transfer batch to device
            b_tokens = to_device(batch.board_tokens)
            b_moves = to_device(batch.best_move_ids)
            b_evals = to_device(batch.eval_scores)

            if USE_GPU
                GC.gc(false)
            end

            (loss_val, (ml, el)), grads = Zygote.withgradient(ps) do p
                chess_loss(model, p, st, b_tokens, b_moves, b_evals)
            end
            grads = grads[1]

            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            grads = nothing

            epoch_loss += Float64(loss_val)

            if global_step % 50 == 0
                println("step=$global_step  loss=$(round(Float64(loss_val), digits=4))  move_loss=$(round(Float64(ml), digits=4))  eval_loss=$(round(Float64(el), digits=4))")
            end

            if global_step % checkpoint_every == 0
                cp_path = joinpath(checkpoint_dir, "step_$(global_step).jld2")
                println("Checkpoint: $cp_path")
                ps_cpu = Lux.cpu(ps)
                JLD2.@save cp_path ps_cpu config global_step epoch
            end
        end

        avg_loss = epoch_loss / length(batches)
        println("--- Epoch $epoch/$num_epochs  avg_loss=$(round(avg_loss, digits=4)) ---")

        if avg_loss < best_loss
            best_loss = avg_loss
            best_path = joinpath(checkpoint_dir, "best.jld2")
            ps_cpu = Lux.cpu(ps)
            JLD2.@save best_path ps_cpu config global_step epoch
            println("  New best! Saved to $best_path")
        end
    end

    println("\n=== Phase 1 complete. Best loss: $(round(best_loss, digits=4)) ===")
    return model, ps, st, config
end

# ============================================================================
# CLI
# ============================================================================

function main()
    data_path = "data/chess/lichess_db_eval.jsonl"
    max_positions = 1_000_000
    checkpoint_dir = "checkpoints/reasoning_drafter/phase1"
    num_epochs = 10

    args = ARGS
    i = 1
    while i <= length(args)
        if args[i] == "--data" && i < length(args)
            data_path = args[i+1]; i += 2
        elseif args[i] == "--max-positions" && i < length(args)
            max_positions = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--checkpoint-dir" && i < length(args)
            checkpoint_dir = args[i+1]; i += 2
        elseif args[i] == "--steps" && i < length(args)
            steps = parse(Int, args[i+1])
            num_epochs = steps > 0 ? max(1, steps ÷ max(max_positions ÷ 64, 1)) : 10
            i += 2
        else
            i += 1
        end
    end

    train_phase1(;
        data_path = data_path,
        max_positions = max_positions,
        checkpoint_dir = checkpoint_dir,
        num_epochs = num_epochs,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
