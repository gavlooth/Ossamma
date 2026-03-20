#!/usr/bin/env julia
"""
Phase 1: Chess Logic Pre-Training for ReasoningDrafter

Trains the ReasoningDrafter on Stockfish-evaluated chess positions to learn
the hidden features of constrained reasoning via multi-task prediction.

Architecture: shared FrontEnd → FrontEndHeader → proposer blocks → AuditInputHeader → audit tail

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
using Swamma.ReasoningDrafterMod: apply_reasoning_drafter_ema_codebook!, reasoning_hidden

using Lux
using Random
using NNlib
using Optimisers
using TOML
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

function _precision_type(name::AbstractString)
    lowered = lowercase(strip(String(name)))
    lowered in ("float32", "fp32") && return Float32
    lowered in ("float16", "fp16", "half") && return Float16
    lowered in ("bfloat16", "bf16") && return Core.BFloat16
    error("Unsupported precision: $name. Expected float32, float16, or bfloat16.")
end

_cast_float_array(x::AbstractArray, ::Type{T}) where {T} = eltype(x) <: AbstractFloat ? T.(x) : x
_cast_float_array(x, ::Type{T}) where {T} = x

function _cast_float_tree(x, ::Type{T}) where {T}
    if x isa NamedTuple
        return NamedTuple{keys(x)}(map(v -> _cast_float_tree(v, T), values(x)))
    elseif x isa Tuple
        return map(v -> _cast_float_tree(v, T), x)
    elseif x isa Optimisers.Leaf
        return Optimisers.Leaf(x.rule, _cast_float_tree(x.state, T), x.frozen)
    else
        return _cast_float_array(x, T)
    end
end

_map_optimizer_state_arrays(x, mover) = x
_map_optimizer_state_arrays(x::AbstractArray, mover) = mover(x)
_map_optimizer_state_arrays(x::Tuple, mover) = map(v -> _map_optimizer_state_arrays(v, mover), x)
function _map_optimizer_state_arrays(x::NamedTuple, mover)
    return NamedTuple{keys(x)}(map(v -> _map_optimizer_state_arrays(v, mover), values(x)))
end
function _map_optimizer_state_arrays(x::Optimisers.Leaf, mover)
    return Optimisers.Leaf(x.rule, _map_optimizer_state_arrays(x.state, mover), x.frozen)
end

_optimizer_state_to_cpu(opt_state) = _map_optimizer_state_arrays(opt_state, x -> cpu_device()(x))
_optimizer_state_to_device(opt_state) = _map_optimizer_state_arrays(opt_state, to_device)

function _save_phase1_checkpoint(path, ps, st, opt_state, config, global_step, epoch; best_loss = nothing)
    ps_cpu = cpu_device()(ps)
    st_cpu = cpu_device()(st)
    opt_state_cpu = _optimizer_state_to_cpu(opt_state)
    if best_loss === nothing
        JLD2.@save path ps_cpu st_cpu opt_state_cpu config global_step epoch
    else
        JLD2.@save path ps_cpu st_cpu opt_state_cpu config global_step epoch best_loss
    end
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

function cast_params_for_forward(ps, ::Type{Float32})
    return ps
end

function cast_params_for_forward(ps, precision_type::Type{T}) where {T<:AbstractFloat}
    return _cast_float_tree(ps, precision_type)
end

function _iterate_named_blocks(blocks::NamedTuple)
    return enumerate(values(blocks))
end

function _scale_nested!(x, scale::Float32)
    x isa AbstractArray && (x .*= scale)
    x isa NamedTuple && for v in values(x)
        _scale_nested!(v, scale)
    end
    return x
end

function _scale_proposer_block_grads!(grads, scale::Float32)
    haskey(grads, :Drafter) || return grads
    haskey(grads.Drafter, :Blocks) || return grads
    for block_grads in values(grads.Drafter.Blocks)
        _scale_nested!(block_grads, scale)
    end
    return grads
end

function forward_chess(model, board_tokens, ps, st)
    hidden, drafter_st = reasoning_hidden(
        model.drafter,
        (token_ids = board_tokens, mask_ratio = 0.0f0),
        ps.Drafter,
        st.Drafter,
    )
    pooled = dropdims(sum(hidden, dims=2), dims=2) ./ Float32(size(board_tokens, 1))

    # Only MoveHead and EvalHead are traced by Zygote
    move_logits, mh_st = model.MoveHead(pooled, ps.MoveHead, st.MoveHead)
    eval_pred, eh_st = model.EvalHead(pooled, ps.EvalHead, st.EvalHead)
    new_st = (Drafter = drafter_st, MoveHead = mh_st, EvalHead = eh_st)

    return move_logits, eval_pred, new_st
end

function forward_chess_backbone(model, board_tokens, ps, st)
    hidden, drafter_st = reasoning_hidden(
        model.drafter,
        (token_ids = board_tokens, mask_ratio = 0.0f0),
        ps.Drafter,
        st.Drafter,
    )
    pooled = dropdims(sum(hidden, dims=2), dims=2) ./ Float32(size(board_tokens, 1))
    new_st = (Drafter = drafter_st, MoveHead = st.MoveHead, EvalHead = st.EvalHead)
    return pooled, nothing, new_st
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

function add_grads(g1, g2)
    g1 === nothing && return g2
    g2 === nothing && return g1
    if g1 isa NamedTuple
        return NamedTuple{keys(g1)}(map(add_grads, values(g1), values(g2)))
    elseif g1 isa Tuple
        return map(add_grads, g1, g2)
    else
        return g1 .+ g2
    end
end

function tree_scale(g, factor::Float32)
    g === nothing && return nothing
    if g isa NamedTuple
        return NamedTuple{keys(g)}(map(v -> tree_scale(v, factor), values(g)))
    elseif g isa Tuple
        return map(v -> tree_scale(v, factor), g)
    else
        return g .* factor
    end
end

function sanitize_gradient_tree(g)
    g === nothing && return nothing
    if g isa NamedTuple
        return NamedTuple{keys(g)}(map(sanitize_gradient_tree, values(g)))
    elseif g isa Tuple
        return map(sanitize_gradient_tree, g)
    elseif g isa AbstractArray && eltype(g) <: AbstractFloat
        return ifelse.(isfinite.(g), g, zero(eltype(g)))
    else
        return g
    end
end

function phase1_default_options()
    return Dict{String,Any}(
        "data_path" => "data/chess/lichess_db_eval.jsonl",
        "max_positions" => 1_000_000,
        "checkpoint_dir" => "checkpoints/reasoning_drafter/phase1",
        "num_epochs" => 10,
        "resume_path" => "",
        "checkpoint_every" => 300,
        "log_every" => 50,
        "debug_every" => 1000,
        "revive_every" => 500,
        "revive_start_step" => 1000,
        "batch_size" => 64,
        "gradient_accumulation_steps" => 1,
        "min_depth" => 10,
        "learning_rate" => 1f-3,
        "proposer_lr_scale" => 0.25f0,
        "move_loss_weight" => 1.0f0,
        "eval_loss_weight" => 0.5f0,
        "seed" => 42,
        "max_steps" => nothing,
        "mixed_precision" => false,
        "precision" => "float32",
        "embedding_dimension" => 512,
        "number_of_heads" => 8,
        "number_of_layers" => 16,
        "time_dimension" => 128,
        "rc_code_dim" => 128,
        "rc_codebook_size" => 512,
        "rc_integration_steps" => 8,
        "frontend_wave_heads" => 4,
        "circuit_num_leaves" => 16,
        "circuit_product_arity" => 2,
        "circuit_num_sums" => 8,
        "circuit_num_circuits" => 4,
        "use_adapters" => false,
    )
end

function _phase1_apply_section!(opts::Dict{String,Any}, table, mapping)
    for (src, dst) in mapping
        haskey(table, src) || continue
        opts[dst] = table[src]
    end
end

function load_phase1_config(path::String)
    data = TOML.parsefile(path)
    opts = phase1_default_options()

    haskey(data, "data") && _phase1_apply_section!(opts, data["data"], Dict(
        "path" => "data_path",
        "max_positions" => "max_positions",
        "min_depth" => "min_depth",
    ))

    haskey(data, "training") && _phase1_apply_section!(opts, data["training"], Dict(
        "batch_size" => "batch_size",
        "gradient_accumulation_steps" => "gradient_accumulation_steps",
        "num_epochs" => "num_epochs",
        "learning_rate" => "learning_rate",
        "proposer_lr_scale" => "proposer_lr_scale",
        "move_loss_weight" => "move_loss_weight",
        "eval_loss_weight" => "eval_loss_weight",
        "checkpoint_dir" => "checkpoint_dir",
        "checkpoint_every" => "checkpoint_every",
        "log_every" => "log_every",
        "debug_every" => "debug_every",
        "revive_every" => "revive_every",
        "revive_start_step" => "revive_start_step",
        "seed" => "seed",
        "max_steps" => "max_steps",
        "resume_path" => "resume_path",
        "mixed_precision" => "mixed_precision",
        "precision" => "precision",
    ))

    haskey(data, "hardware") && _phase1_apply_section!(opts, data["hardware"], Dict(
        "mixed_precision" => "mixed_precision",
        "precision" => "precision",
    ))

    haskey(data, "model") && _phase1_apply_section!(opts, data["model"], Dict(
        "embedding_dimension" => "embedding_dimension",
        "number_of_heads" => "number_of_heads",
        "number_of_layers" => "number_of_layers",
        "time_dimension" => "time_dimension",
        "rc_code_dim" => "rc_code_dim",
        "rc_codebook_size" => "rc_codebook_size",
        "rc_integration_steps" => "rc_integration_steps",
        "frontend_wave_heads" => "frontend_wave_heads",
        "circuit_num_leaves" => "circuit_num_leaves",
        "circuit_product_arity" => "circuit_product_arity",
        "circuit_num_sums" => "circuit_num_sums",
        "circuit_num_circuits" => "circuit_num_circuits",
        "use_adapters" => "use_adapters",
    ))

    return opts
end

function validate_phase1_options!(opts::AbstractDict{String,<:Any})
    if opts["embedding_dimension"] % opts["number_of_heads"] != 0
        error("embedding_dimension=$(opts["embedding_dimension"]) must be divisible by number_of_heads=$(opts["number_of_heads"])")
    end
    opts["batch_size"] > 0 || error("batch_size must be > 0")
    opts["gradient_accumulation_steps"] > 0 || error("gradient_accumulation_steps must be > 0")
    opts["number_of_layers"] > 0 || error("number_of_layers must be > 0")
    _precision_type(string(get(opts, "precision", "float32")))
    return opts
end

function apply_legal_move_mask(move_logits, legal_move_mask)
    penalty = eltype(move_logits)(-1f9)
    return move_logits .+ (1.0f0 .- legal_move_mask) .* penalty
end

function legal_move_top1_accuracy(move_logits, legal_move_mask, target_ids)
    masked_logits = apply_legal_move_mask(move_logits, legal_move_mask)
    masked_logits_cpu = Array(cpu_device()(masked_logits))
    pred_idx = vec(argmax(masked_logits_cpu, dims=1))
    pred_move_ids = getindex.(Tuple.(pred_idx), 1)
    return sum(pred_move_ids .== target_ids) / length(target_ids)
end

function chess_loss(model, ps, st, batch_tokens, batch_moves, batch_evals, target_onehot, legal_move_mask; α=1.0f0, β=0.5f0)
    move_logits, eval_pred, new_st = forward_chess(model, batch_tokens, ps, st)

    # Cross-entropy over the legal move set only.
    bs = size(move_logits, 2)
    log_probs = NNlib.logsoftmax(apply_legal_move_mask(move_logits, legal_move_mask), dims=1)
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
    gradient_accumulation_steps::Int = 1,
    num_epochs::Int = 3,
    learning_rate::Float32 = 1f-3,
    proposer_lr_scale::Float32 = 0.25f0,
    move_loss_weight::Float32 = 1.0f0,
    eval_loss_weight::Float32 = 0.5f0,
    mixed_precision::Bool = false,
    precision::String = "float32",
    checkpoint_dir::String = "checkpoints/reasoning_drafter/phase1",
    checkpoint_every::Int = 300,
    log_every::Int = 50,
    debug_every::Int = 1000,
    revive_every::Int = 500,
    revive_start_step::Int = 1000,
    seed::Int = 42,
    resume_path::String = "",
    max_steps::Union{Int,Nothing} = nothing,
    embedding_dimension::Int = 512,
    number_of_heads::Int = 8,
    number_of_layers::Int = 16,
    time_dimension::Int = 128,
    rc_code_dim::Int = 128,
    rc_codebook_size::Int = 512,
    rc_integration_steps::Int = 8,
    frontend_wave_heads::Int = 4,
    circuit_num_leaves::Int = 16,
    circuit_product_arity::Int = 2,
    circuit_num_sums::Int = 8,
    circuit_num_circuits::Int = 4,
    use_adapters::Bool = false,
)
    rng = Random.MersenneTwister(seed)
    mkpath(checkpoint_dir)

    println("=== Phase 1: Chess Logic Pre-Training ===")
    println("Data: $data_path")
    println("Max positions: $max_positions, min depth: $min_depth")
    println("Batch size: $batch_size, LR: $learning_rate")
    println("Gradient accumulation: $gradient_accumulation_steps (effective batch=$(batch_size * gradient_accumulation_steps))")
    println("Loss weights: move=$move_loss_weight eval=$eval_loss_weight")
    println("Precision: $(mixed_precision ? precision : "float32")")

    # Streaming data — no preload, reads from disk per batch
    println("Streaming from: $data_path (max $max_positions positions, min_depth=$min_depth)")

    # Model — 64 squares, 13 piece types (chess-specific)
    config = ReasoningDrafterConfig(
        vocab_size = PIECE_VOCAB_SIZE,
        max_sequence_length = NUM_SQUARES,
        embedding_dimension = embedding_dimension,
        number_of_heads = number_of_heads,
        number_of_layers = number_of_layers,
        time_dimension = time_dimension,
        rc_code_dim = rc_code_dim,
        rc_codebook_size = rc_codebook_size,
        rc_integration_steps = rc_integration_steps,
        frontend_wave_heads = frontend_wave_heads,
        circuit_num_leaves = circuit_num_leaves,
        circuit_product_arity = circuit_product_arity,
        circuit_num_sums = circuit_num_sums,
        circuit_num_circuits = circuit_num_circuits,
        use_adapters = use_adapters,
    )

    model, ps, st = init_chess_model(rng, config)

    param_precision = mixed_precision ? _precision_type(precision) : Float32
    if mixed_precision && !USE_GPU
        println("Mixed precision requested on CPU; falling back to float32.")
        mixed_precision = false
        param_precision = Float32
    end
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
        ps = _cast_float_tree(ps, Float32)
        st = _cast_float_tree(st, Float32)
        loaded_opt_state = loaded_opt_state === nothing ? nothing : _cast_float_tree(loaded_opt_state, Float32)
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

    opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0f0), Optimisers.Adam(learning_rate))
    if loaded_opt_state !== nothing
        opt_state = USE_GPU ? _optimizer_state_to_device(loaded_opt_state) : loaded_opt_state
        println("  Restored optimizer state")
    else
        opt_state = Optimisers.setup(opt, ps)
    end

    println("  Proposer LR scale: $(proposer_lr_scale)")

    global_step = start_step
    best_loss = Inf
    steps_run = 0

    for epoch in start_epoch:num_epochs
        stream = StreamingBatchIterator(data_path; batch_size=batch_size, min_depth=min_depth, max_positions=max_positions)
        _open_stream!(stream)
        epoch_loss = 0.0
        epoch_updates = 0
        accum_grads = nothing
        accum_count = 0
        pending_loss_sum = 0.0
        pending_move_loss_sum = 0.0
        pending_eval_loss_sum = 0.0
        pending_legal_top1_sum = 0.0
        pending_avg_legal_sum = 0.0
        pending_target_legal_sum = 0.0
        stop_requested = false

        while true
            if max_steps !== nothing && steps_run >= max_steps
                println("Reached max_steps=$max_steps; stopping Phase 1 early.")
                stop_requested = true
                break
            end
            batch = next_batch!(stream)
            batch === nothing && break

            # Transfer batch to device
            b_tokens = to_device(batch.board_tokens)
            b_evals = to_device(batch.eval_scores)
            b_legal_mask = to_device(_cast_float_array(batch.legal_move_mask, param_precision))

            # Build one-hot target outside gradient (avoids Zygote GPU issues)
            b_onehot = make_onehot(batch.best_move_ids, MOVE_VOCAB_SIZE, x -> to_device(_cast_float_array(x, param_precision)))

            if USE_GPU
                GC.gc(false)
            end

            (loss_val, (ml, el, new_st, move_logits)), grads = Zygote.withgradient(ps) do p
                forward_params = cast_params_for_forward(p, param_precision)
                hidden, next_drafter_st = reasoning_hidden(
                    model.drafter,
                    (token_ids = b_tokens, mask_ratio = 0.0f0),
                    forward_params.Drafter,
                    st.Drafter,
                )
                pooled = dropdims(sum(hidden, dims=2), dims=2) ./ eltype(hidden)(size(b_tokens, 1))

                move_logits, mh_st = model.MoveHead(pooled, forward_params.MoveHead, st.MoveHead)
                eval_pred, eh_st = model.EvalHead(pooled, forward_params.EvalHead, st.EvalHead)

                bs = size(move_logits, 2)
                move_logits_loss = Float32.(move_logits)
                legal_mask_loss = Float32.(b_legal_mask)
                onehot_loss = Float32.(b_onehot)
                eval_pred_loss = Float32.(eval_pred)
                eval_targets = reshape(Float32.(b_evals), 1, :)
                log_probs = NNlib.logsoftmax(apply_legal_move_mask(move_logits_loss, legal_mask_loss), dims=1)
                m_loss = -sum(log_probs .* onehot_loss) / bs
                e_loss = sum((eval_pred_loss .- eval_targets) .^ 2) / length(b_evals)

                next_st = (
                    Drafter = next_drafter_st,
                    MoveHead = mh_st,
                    EvalHead = eh_st,
                )

                move_loss_weight * m_loss + eval_loss_weight * e_loss, (m_loss, e_loss, next_st, move_logits)
            end
            # Debug: check gradient flow only on the microbatch that will trigger an optimizer update.
            if debug_every > 0 &&
               (accum_count + 1 >= gradient_accumulation_steps) &&
               ((global_step + 1) % debug_every == 0)
                g = grads[1]
                function gnorm(x)
                    x === nothing && return "nil"
                    x isa NamedTuple && return "nt"
                    return round(Float64(sum(abs2, Array(x))), digits=6)
                end
                println("  [grad] mh=$(gnorm(g.MoveHead === nothing ? nothing : g.MoveHead.weight))  te=$(gnorm(g.Drafter === nothing ? nothing : (g.Drafter.TokenEmbedding === nothing ? nothing : g.Drafter.TokenEmbedding.weight)))")
                if g.Drafter !== nothing && g.Drafter.Blocks !== nothing
                    for (i, bg) in _iterate_named_blocks(g.Drafter.Blocks)
                        bg === nothing && (println("  [Block $i] nil"); continue)
                        lin_attn_weight = bg.LinAttn === nothing ? nothing :
                            (haskey(bg.LinAttn, :Query) ? bg.LinAttn.Query.weight : nothing)
                        wave_speed = bg.WaveGateLayer === nothing ? nothing : bg.WaveGateLayer.log_wave_speed
                        println("  [Block $i] glu=$(gnorm(bg.GluProjection === nothing ? nothing : bg.GluProjection.weight))  la=$(gnorm(lin_attn_weight))  wave=$(gnorm(wave_speed))  header=$(gnorm(get(bg, :ProposalHeaderWeight, nothing)))")
                    end
                end
            end
            accum_grads = add_grads(accum_grads, grads[1])
            grads = nothing
            st = new_st
            accum_count += 1

            legal_top1 = legal_move_top1_accuracy(move_logits, b_legal_mask, batch.best_move_ids)
            avg_legal_moves = sum(batch.legal_move_counts) / length(batch.legal_move_counts)
            target_legal_rate = sum(batch.target_legal_flags) / length(batch.target_legal_flags)
            pending_loss_sum += Float64(loss_val)
            pending_move_loss_sum += Float64(ml)
            pending_eval_loss_sum += Float64(el)
            pending_legal_top1_sum += Float64(legal_top1)
            pending_avg_legal_sum += Float64(avg_legal_moves)
            pending_target_legal_sum += Float64(target_legal_rate)
            move_logits = nothing

            if accum_count >= gradient_accumulation_steps
                global_step += 1
                steps_run += 1
                epoch_updates += 1

                mean_grads = tree_scale(accum_grads, 1.0f0 / Float32(accum_count))
                block_scaled_grads = _scale_proposer_block_grads!(mean_grads, proposer_lr_scale)
                safe_grads = sanitize_gradient_tree(block_scaled_grads)
                opt_state, ps = Optimisers.update(opt_state, ps, safe_grads)
                apply_reasoning_drafter_ema_codebook!(ps.Drafter, st.Drafter, model.drafter)
                epoch_loss += pending_loss_sum / accum_count

                if global_step % log_every == 0 || global_step == 1
                    println("step=$global_step  loss=$(round(pending_loss_sum / accum_count, digits=4))  move_loss=$(round(pending_move_loss_sum / accum_count, digits=4))  eval_loss=$(round(pending_eval_loss_sum / accum_count, digits=4))  legal_top1=$(round(pending_legal_top1_sum / accum_count, digits=4))  avg_legal=$(round(pending_avg_legal_sum / accum_count, digits=2))  target_legal=$(round(100 * pending_target_legal_sum / accum_count, digits=1))%  accum=$accum_count")
                end

                if global_step % checkpoint_every == 0
                    cp_path = joinpath(checkpoint_dir, "checkpoint_last.jld2")
                    println("Checkpoint (step $global_step): $cp_path")
                    _save_phase1_checkpoint(cp_path, ps, st, opt_state, config, global_step, epoch; best_loss = best_loss)
                end

                accum_grads = nothing
                accum_count = 0
                pending_loss_sum = 0.0
                pending_move_loss_sum = 0.0
                pending_eval_loss_sum = 0.0
                pending_legal_top1_sum = 0.0
                pending_avg_legal_sum = 0.0
                pending_target_legal_sum = 0.0
                GC.gc(false)
            end
        end

        if accum_count > 0 && !(max_steps !== nothing && steps_run >= max_steps)
            global_step += 1
            steps_run += 1
            epoch_updates += 1

            mean_grads = tree_scale(accum_grads, 1.0f0 / Float32(accum_count))
            block_scaled_grads = _scale_proposer_block_grads!(mean_grads, proposer_lr_scale)
            safe_grads = sanitize_gradient_tree(block_scaled_grads)
            opt_state, ps = Optimisers.update(opt_state, ps, safe_grads)
            apply_reasoning_drafter_ema_codebook!(ps.Drafter, st.Drafter, model.drafter)
            epoch_loss += pending_loss_sum / accum_count

            if global_step % log_every == 0 || global_step == 1
                println("step=$global_step  loss=$(round(pending_loss_sum / accum_count, digits=4))  move_loss=$(round(pending_move_loss_sum / accum_count, digits=4))  eval_loss=$(round(pending_eval_loss_sum / accum_count, digits=4))  legal_top1=$(round(pending_legal_top1_sum / accum_count, digits=4))  avg_legal=$(round(pending_avg_legal_sum / accum_count, digits=2))  target_legal=$(round(100 * pending_target_legal_sum / accum_count, digits=1))%  accum=$accum_count")
            end

            if global_step % checkpoint_every == 0
                cp_path = joinpath(checkpoint_dir, "checkpoint_last.jld2")
                println("Checkpoint (step $global_step): $cp_path")
                _save_phase1_checkpoint(cp_path, ps, st, opt_state, config, global_step, epoch; best_loss = best_loss)
            end
        end

        stream.io !== nothing && close(stream.io)
        avg_loss = epoch_updates > 0 ? epoch_loss / epoch_updates : Inf
        println("--- Epoch $epoch/$num_epochs  avg_loss=$(round(avg_loss, digits=4))  updates=$epoch_updates ---")

        if avg_loss < best_loss
            best_loss = avg_loss
            best_path = joinpath(checkpoint_dir, "best.jld2")
            _save_phase1_checkpoint(best_path, ps, st, opt_state, config, global_step, epoch; best_loss = best_loss)
            println("  New best! Saved to $best_path")
        end

        cp_path = joinpath(checkpoint_dir, "checkpoint_last.jld2")
        _save_phase1_checkpoint(cp_path, ps, st, opt_state, config, global_step, epoch; best_loss = best_loss)

        if (max_steps !== nothing && steps_run >= max_steps) || stop_requested
            break
        end
    end

    println("\n=== Phase 1 complete. Best loss: $(round(best_loss, digits=4)) ===")
    return model, ps, st, config
end

# ============================================================================
# CLI
# ============================================================================

function main()
    args = ARGS
    config_path = ""
    for i in 1:length(args)-1
        if args[i] == "--config"
            config_path = args[i+1]
            break
        end
    end

    opts = isempty(config_path) ? phase1_default_options() : load_phase1_config(config_path)
    isempty(config_path) || println("Loaded config: $config_path")

    data_path = opts["data_path"]
    max_positions = opts["max_positions"]
    checkpoint_dir = opts["checkpoint_dir"]
    num_epochs = opts["num_epochs"]
    resume_path = opts["resume_path"]
    checkpoint_every = opts["checkpoint_every"]
    log_every = opts["log_every"]
    debug_every = opts["debug_every"]
    revive_every = opts["revive_every"]
    revive_start_step = opts["revive_start_step"]
    batch_size = opts["batch_size"]
    gradient_accumulation_steps = opts["gradient_accumulation_steps"]
    min_depth = opts["min_depth"]
    learning_rate = Float32(opts["learning_rate"])
    proposer_lr_scale = Float32(opts["proposer_lr_scale"])
    move_loss_weight = Float32(opts["move_loss_weight"])
    eval_loss_weight = Float32(opts["eval_loss_weight"])
    mixed_precision = Bool(opts["mixed_precision"])
    precision = string(opts["precision"])
    seed = opts["seed"]
    max_steps = opts["max_steps"]
    embedding_dimension = opts["embedding_dimension"]
    number_of_heads = opts["number_of_heads"]
    number_of_layers = opts["number_of_layers"]
    time_dimension = opts["time_dimension"]
    rc_code_dim = opts["rc_code_dim"]
    rc_codebook_size = opts["rc_codebook_size"]
    rc_integration_steps = opts["rc_integration_steps"]
    frontend_wave_heads = opts["frontend_wave_heads"]
    circuit_num_leaves = opts["circuit_num_leaves"]
    circuit_product_arity = opts["circuit_product_arity"]
    circuit_num_sums = opts["circuit_num_sums"]
    circuit_num_circuits = opts["circuit_num_circuits"]
    use_adapters = opts["use_adapters"]

    i = 1
    while i <= length(args)
        if args[i] == "--config" && i < length(args)
            i += 2
        elseif args[i] == "--data" && i < length(args)
            data_path = args[i+1]; i += 2
        elseif args[i] == "--max-positions" && i < length(args)
            max_positions = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--min-depth" && i < length(args)
            min_depth = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--batch-size" && i < length(args)
            batch_size = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--gradient-accumulation-steps" && i < length(args)
            gradient_accumulation_steps = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--learning-rate" && i < length(args)
            learning_rate = parse(Float32, args[i+1]); i += 2
        elseif args[i] == "--proposer-lr-scale" && i < length(args)
            proposer_lr_scale = parse(Float32, args[i+1]); i += 2
        elseif args[i] == "--move-loss-weight" && i < length(args)
            move_loss_weight = parse(Float32, args[i+1]); i += 2
        elseif args[i] == "--eval-loss-weight" && i < length(args)
            eval_loss_weight = parse(Float32, args[i+1]); i += 2
        elseif args[i] == "--mixed-precision" && i < length(args)
            mixed_precision = parse(Bool, args[i+1]); i += 2
        elseif args[i] == "--precision" && i < length(args)
            precision = args[i+1]; i += 2
        elseif args[i] == "--checkpoint-dir" && i < length(args)
            checkpoint_dir = args[i+1]; i += 2
        elseif args[i] == "--checkpoint-every" && i < length(args)
            checkpoint_every = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--log-every" && i < length(args)
            log_every = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--debug-every" && i < length(args)
            debug_every = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--revive-every" && i < length(args)
            revive_every = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--revive-start-step" && i < length(args)
            revive_start_step = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--resume" && i < length(args)
            resume_path = args[i+1]; i += 2
        elseif args[i] == "--steps" && i < length(args)
            steps = parse(Int, args[i+1])
            num_epochs = steps > 0 ? max(1, steps ÷ max(max_positions ÷ 64, 1)) : 10
            i += 2
        elseif args[i] == "--max-steps" && i < length(args)
            max_steps = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--seed" && i < length(args)
            seed = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--embedding-dim" && i < length(args)
            embedding_dimension = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--heads" && i < length(args)
            number_of_heads = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--layers" && i < length(args)
            number_of_layers = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--time-dim" && i < length(args)
            time_dimension = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--rc-code-dim" && i < length(args)
            rc_code_dim = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--rc-codebook-size" && i < length(args)
            rc_codebook_size = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--rc-steps" && i < length(args)
            rc_integration_steps = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--frontend-wave-heads" && i < length(args)
            frontend_wave_heads = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--circuit-leaves" && i < length(args)
            circuit_num_leaves = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--circuit-arity" && i < length(args)
            circuit_product_arity = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--circuit-sums" && i < length(args)
            circuit_num_sums = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--circuit-circuits" && i < length(args)
            circuit_num_circuits = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--use-adapters" && i < length(args)
            use_adapters = parse(Bool, args[i+1]); i += 2
        else
            i += 1
        end
    end

    validate_phase1_options!(Dict(
        "embedding_dimension" => embedding_dimension,
        "number_of_heads" => number_of_heads,
        "batch_size" => batch_size,
        "gradient_accumulation_steps" => gradient_accumulation_steps,
        "number_of_layers" => number_of_layers,
        "precision" => precision,
    ))

    train_phase1(;
        data_path = data_path,
        max_positions = max_positions,
        min_depth = min_depth,
        batch_size = batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        learning_rate = learning_rate,
        proposer_lr_scale = proposer_lr_scale,
        move_loss_weight = move_loss_weight,
        eval_loss_weight = eval_loss_weight,
        mixed_precision = mixed_precision,
        precision = precision,
        checkpoint_dir = checkpoint_dir,
        num_epochs = num_epochs,
        checkpoint_every = checkpoint_every,
        log_every = log_every,
        debug_every = debug_every,
        revive_every = revive_every,
        revive_start_step = revive_start_step,
        seed = seed,
        resume_path = resume_path,
        max_steps = max_steps,
        embedding_dimension = embedding_dimension,
        number_of_heads = number_of_heads,
        number_of_layers = number_of_layers,
        time_dimension = time_dimension,
        rc_code_dim = rc_code_dim,
        rc_codebook_size = rc_codebook_size,
        rc_integration_steps = rc_integration_steps,
        frontend_wave_heads = frontend_wave_heads,
        circuit_num_leaves = circuit_num_leaves,
        circuit_product_arity = circuit_product_arity,
        circuit_num_sums = circuit_num_sums,
        circuit_num_circuits = circuit_num_circuits,
        use_adapters = use_adapters,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
