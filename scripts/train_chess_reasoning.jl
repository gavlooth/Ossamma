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
using Swamma.ChessDataset: _open_stream!
using Swamma.ReasoningDrafterMod
using Swamma.ReasoningDrafterMod: apply_reasoning_drafter_ema_codebook!
using Swamma.RuleConditionedWavePDEMod

using Lux
using Random
using NNlib
using Optimisers
using Zygote
using JLD2
using ChainRulesCore

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

function _apply_chess_blocks(drafter, hidden, time_emb, ps_blocks, st_blocks, i::Int = 1)
    if i > drafter.config.number_of_layers
        return hidden, ()
    end

    key = Symbol("Block_$i")
    next_hidden, block_state = drafter.Blocks[i]((hidden, time_emb), ps_blocks[key], st_blocks[key])
    final_hidden, remaining_states = _apply_chess_blocks(
        drafter, next_hidden, time_emb, ps_blocks, st_blocks, i + 1
    )
    return final_hidden, (block_state, remaining_states...)
end

function forward_chess(model, board_tokens, ps, st)
    drafter = model.drafter
    config = drafter.config
    seq_len, batch_size = size(board_tokens)
    seq_len <= config.max_sequence_length || throw(ArgumentError(
        "ChessReasoningModel received seq_len=$(seq_len), but max_sequence_length=$(config.max_sequence_length)."
    ))

    # Detach entire backbone from AD tape — only heads are differentiated.
    # This drops GPU memory from ~28GB to ~1GB for a 6.7M model.
    hidden_pooled, block_states, tok_st, pos_st, fn_st = ChainRulesCore.ignore_derivatives() do
        tok_flat = vec(board_tokens)
        tok_emb_flat, _tok_st = drafter.TokenEmbedding(tok_flat, ps.Drafter.TokenEmbedding, st.Drafter.TokenEmbedding)
        tok_emb = reshape(tok_emb_flat, config.embedding_dimension, seq_len, batch_size)

        pos_indices = to_device(collect(1:seq_len))
        pos_emb_raw, _pos_st = drafter.PositionEmbedding(pos_indices, ps.Drafter.PositionEmbedding, st.Drafter.PositionEmbedding)
        pos_emb = reshape(pos_emb_raw, config.embedding_dimension, seq_len, 1)

        h = tok_emb .+ pos_emb
        t_emb = repeat(reshape(ps.Drafter.TimeEmbedding, :, 1), 1, batch_size)

        h, _block_states = _apply_chess_blocks(drafter, h, t_emb, ps.Drafter.Blocks, st.Drafter.Blocks)
        h, _fn_st = drafter.FinalNorm(h, ps.Drafter.FinalNorm, st.Drafter.FinalNorm)

        pooled = dropdims(sum(h, dims=2), dims=2) ./ Float32(seq_len)
        pooled, _block_states, _tok_st, _pos_st, _fn_st
    end

    # Only MoveHead and EvalHead are traced by Zygote
    move_logits, mh_st = model.MoveHead(hidden_pooled, ps.MoveHead, st.MoveHead)
    eval_pred, eh_st = model.EvalHead(hidden_pooled, ps.EvalHead, st.EvalHead)

    drafter_st = (
        TokenEmbedding = tok_st,
        PositionEmbedding = pos_st,
        Blocks = NamedTuple{ntuple(i -> Symbol("Block_$i"), drafter.config.number_of_layers)}(
            block_states
        ),
        FinalNorm = fn_st,
        OutputHead = st.Drafter.OutputHead,
    )
    new_st = (Drafter = drafter_st, MoveHead = mh_st, EvalHead = eh_st)

    return move_logits, eval_pred, new_st
end

function forward_chess_backbone(model, board_tokens, ps, st)
    drafter = model.drafter
    config = drafter.config
    seq_len, batch_size = size(board_tokens)

    tok_flat = vec(board_tokens)
    tok_emb_flat, tok_st = drafter.TokenEmbedding(tok_flat, ps.Drafter.TokenEmbedding, st.Drafter.TokenEmbedding)
    tok_emb = reshape(tok_emb_flat, config.embedding_dimension, seq_len, batch_size)

    pos_indices = to_device(collect(1:seq_len))
    pos_emb_raw, pos_st = drafter.PositionEmbedding(pos_indices, ps.Drafter.PositionEmbedding, st.Drafter.PositionEmbedding)
    pos_emb = reshape(pos_emb_raw, config.embedding_dimension, seq_len, 1)

    h = tok_emb .+ pos_emb
    t_emb = repeat(reshape(ps.Drafter.TimeEmbedding, :, 1), 1, batch_size)
    h, block_states = _apply_chess_blocks(drafter, h, t_emb, ps.Drafter.Blocks, st.Drafter.Blocks)
    h, fn_st = drafter.FinalNorm(h, ps.Drafter.FinalNorm, st.Drafter.FinalNorm)

    pooled = dropdims(sum(h, dims=2), dims=2) ./ Float32(seq_len)

    drafter_st = (
        TokenEmbedding = tok_st,
        PositionEmbedding = pos_st,
        Blocks = NamedTuple{ntuple(i -> Symbol("Block_$i"), config.number_of_layers)}(block_states),
        FinalNorm = fn_st,
        OutputHead = st.Drafter.OutputHead,
    )
    new_st = (Drafter = drafter_st, MoveHead = st.MoveHead, EvalHead = st.EvalHead)
    return pooled, block_states, new_st
end

function make_onehot(indices, n_classes, device_fn)
    # Build one-hot matrix on the correct device (CPU or GPU), outside gradient
    bs = length(indices)
    idx_cpu = Array(indices)
    oh = zeros(Float32, n_classes, bs)
    for j in 1:bs
        oh[idx_cpu[j], j] = 1.0f0
    end
    return device_fn(oh)
end

function chess_loss(model, ps, st, batch_tokens, batch_moves, batch_evals, target_onehot; α=1.0f0, β=0.5f0)
    move_logits, eval_pred, new_st = forward_chess(model, batch_tokens, ps, st)

    # Cross-entropy: dot(one_hot, logsoftmax) — Zygote-safe on GPU
    bs = size(move_logits, 2)
    log_probs = NNlib.logsoftmax(move_logits, dims=1)
    move_loss = -sum(log_probs .* target_onehot) / bs

    eval_targets = reshape(batch_evals, 1, :)
    eval_loss = sum((eval_pred .- eval_targets) .^ 2) / length(batch_evals)

    return α * move_loss + β * eval_loss, (move_loss, eval_loss, new_st)
end

# ============================================================================
# Training loop
# ============================================================================

function train_phase1(;
    data_path::String,
    max_positions::Int = 1_000_000,
    min_depth::Int = 10,
    batch_size::Int = 64,
    num_epochs::Int = 3,
    learning_rate::Float32 = 1f-3,
    checkpoint_dir::String = "checkpoints/reasoning_drafter/phase1",
    checkpoint_every::Int = 300,
    seed::Int = 42,
    resume_path::String = "",
)
    rng = Random.MersenneTwister(seed)
    mkpath(checkpoint_dir)

    println("=== Phase 1: Chess Logic Pre-Training ===")
    println("Data: $data_path")
    println("Max positions: $max_positions, min depth: $min_depth")
    println("Batch size: $batch_size, LR: $learning_rate")

    # Streaming data — no preload, reads from disk per batch
    println("Streaming from: $data_path (max $max_positions positions, min_depth=$min_depth)")

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
        rc_integration_steps = 4,
        circuit_num_leaves = 16,
        circuit_product_arity = 2,
        circuit_num_sums = 8,
        circuit_num_circuits = 4,
        use_adapters = false,
    )

    model, ps, st = init_chess_model(rng, config)

    start_step = 0
    start_epoch = 1

    # Resume from checkpoint if available
    loaded_opt_state = nothing
    if !isempty(resume_path) && isfile(resume_path)
        println("Resuming from: $resume_path")
        ckpt = JLD2.load(resume_path)
        ps = ckpt["ps_cpu"]
        if haskey(ckpt, "st_cpu")
            st = ckpt["st_cpu"]
        end
        start_step = get(ckpt, "global_step", 0)
        start_epoch = get(ckpt, "epoch", 1)
        if haskey(ckpt, "opt_state_cpu")
            loaded_opt_state = ckpt["opt_state_cpu"]
        end
        println("  Resumed at step=$start_step, epoch=$start_epoch")
    end

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
    if loaded_opt_state !== nothing
        opt_state = USE_GPU ? to_device(loaded_opt_state) : loaded_opt_state
        println("  Restored optimizer state")
    else
        opt_state = Optimisers.setup(opt, ps)
    end

    global_step = start_step
    best_loss = Inf

    for epoch in start_epoch:num_epochs
        stream = StreamingBatchIterator(data_path; batch_size=batch_size, min_depth=min_depth, max_positions=max_positions)
        _open_stream!(stream)
        epoch_loss = 0.0
        epoch_batches = 0

        while true
            batch = next_batch!(stream)
            batch === nothing && break

            global_step += 1
            epoch_batches += 1

            # Transfer batch to device
            b_tokens = to_device(batch.board_tokens)
            b_moves = to_device(batch.best_move_ids)
            b_evals = to_device(batch.eval_scores)

            # Build one-hot target outside gradient (avoids Zygote GPU issues)
            b_onehot = make_onehot(batch.best_move_ids, MOVE_VOCAB_SIZE, to_device)

            if USE_GPU
                GC.gc(false)
            end

            # Forward the backbone outside AD (no tape, minimal memory)
            hidden_pooled, _block_states, new_st = forward_chess_backbone(model, b_tokens, ps, st)

            # Only differentiate through the heads
            (loss_val, (ml, el)), grads = Zygote.withgradient(ps) do p
                move_logits, _mh_st = model.MoveHead(hidden_pooled, p.MoveHead, st.MoveHead)
                eval_pred, _eh_st = model.EvalHead(hidden_pooled, p.EvalHead, st.EvalHead)

                bs = size(move_logits, 2)
                log_probs = NNlib.logsoftmax(move_logits, dims=1)
                m_loss = -sum(log_probs .* b_onehot) / bs
                eval_targets = reshape(b_evals, 1, :)
                e_loss = sum((eval_pred .- eval_targets) .^ 2) / length(b_evals)
                1.0f0 * m_loss + 0.5f0 * e_loss, (m_loss, e_loss)
            end
            grads = grads[1]

            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            grads = nothing
            st = new_st
            apply_reasoning_drafter_ema_codebook!(ps.Drafter, st.Drafter, model.drafter)

            epoch_loss += Float64(loss_val)

            if global_step % 50 == 0
                println("step=$global_step  loss=$(round(Float64(loss_val), digits=4))  move_loss=$(round(Float64(ml), digits=4))  eval_loss=$(round(Float64(el), digits=4))")
            end

            if global_step % checkpoint_every == 0
                cp_path = joinpath(checkpoint_dir, "checkpoint_last.jld2")
                println("Checkpoint (step $global_step): $cp_path")
                ps_cpu = cpu_device()(ps)
                st_cpu = cpu_device()(st)
                opt_state_cpu = cpu_device()(opt_state)
                JLD2.@save cp_path ps_cpu st_cpu opt_state_cpu config global_step epoch
            end
        end

        stream.io !== nothing && close(stream.io)
        avg_loss = epoch_batches > 0 ? epoch_loss / epoch_batches : Inf
        println("--- Epoch $epoch/$num_epochs  avg_loss=$(round(avg_loss, digits=4))  batches=$epoch_batches ---")

        if avg_loss < best_loss
            best_loss = avg_loss
            best_path = joinpath(checkpoint_dir, "best.jld2")
            ps_cpu = cpu_device()(ps)
            st_cpu = cpu_device()(st)
            JLD2.@save best_path ps_cpu st_cpu config global_step epoch
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
    resume_path = ""

    args = ARGS
    i = 1
    while i <= length(args)
        if args[i] == "--data" && i < length(args)
            data_path = args[i+1]; i += 2
        elseif args[i] == "--max-positions" && i < length(args)
            max_positions = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--checkpoint-dir" && i < length(args)
            checkpoint_dir = args[i+1]; i += 2
        elseif args[i] == "--resume" && i < length(args)
            resume_path = args[i+1]; i += 2
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
        resume_path = resume_path,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
