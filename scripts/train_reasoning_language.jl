#!/usr/bin/env julia
"""
Phase 3a: Language Fine-Tuning for ReasoningDrafter

Trains the adapter headers and thawed components on reasoning datasets
while keeping the chess-learned reasoning backbone frozen.

Datasets (downloaded by scripts/download_reasoning_datasets.sh):
  ┌────────────────┬──────────┬─────────────────────────────────────────┐
  │    Dataset     │ Examples │             Reasoning type              │
  ├────────────────┼──────────┼─────────────────────────────────────────┤
  │ LogicNLI       │   16,000 │ Logical entailment (premise→hypothesis) │
  │ GSM8K          │    7,473 │ Arithmetic chain-of-thought             │
  │ ReClor         │    4,638 │ Argumentation (law exam reasoning)      │
  │ ARC-Challenge  │    1,119 │ Science reasoning with multi-hop        │
  │ bAbI-deduction │    1,000 │ Syllogistic deduction                   │
  │ bAbI-induction │    1,000 │ Inductive reasoning                     │
  │ Total          │   31,230 │                                         │
  └────────────────┴──────────┴─────────────────────────────────────────┘

Freeze strategy:
  PERMANENTLY FROZEN: full shared front-end backbone (including codebook),
    proposer core (GluProjection, LinearAttention, WaveGate, FFN, norms),
    and audit-tail logic/circuit core (fine encoder, role base/shift,
    predicate bank/output, circuit leaf projection, circuit), plus FinalNorm
  TRAINABLE AT BASE LR: `FrontEndHeader`, proposer headers,
    `AuditInputHeader`, circuit leaf headers, veto/score calibration heads,
    token/position/time embeddings, and OutputHead

Mask-ratio handling:
  Phase 3a is next-token language tuning, not masked denoising. We still call
  ReasoningDrafter with an explicit `mask_ratio = 0.0f0` so the new
  mask-conditioned time embedding path stays aligned with the model API.

CUDA/GPU rules: no try/catch in loops, println() only, grads=nothing after step.

  Usage:
    julia --project=. scripts/train_reasoning_language.jl \
      --checkpoint checkpoints/reasoning_drafter/phase2/surgery.jld2 \
      --data-dir data/reasoning \
      --output-dir checkpoints/reasoning_drafter/phase3a

  Bounded smoke:
    julia --project=. scripts/train_reasoning_language.jl \
      --checkpoint checkpoints/reasoning_drafter/phase2/surgery.jld2 \
      --data-dir data/reasoning \
      --output-dir checkpoints/reasoning_drafter/phase3a_smoke \
      --epochs 1 --batch-size 1 --max-seq-length 32 --max-per-dataset 1 --max-steps 1
"""

using Swamma
using Swamma.ReasoningDrafterMod
using Swamma.ReasoningDrafterMod: apply_reasoning_drafter_ema_codebook!
using Swamma.RuleConditionedWavePDEMod
using Swamma.ReasoningDataset

using Lux
using Random
using NNlib
using Optimisers
using Zygote
using JLD2

const USE_GPU = try
    using CUDA
    CUDA.functional()
catch
    false
end

if !isdefined(@__MODULE__, :_SWAMMA_GPU_BANNER_PRINTED)
    global _SWAMMA_GPU_BANNER_PRINTED = false
end

if !_SWAMMA_GPU_BANNER_PRINTED
    if USE_GPU
        println("GPU: $(CUDA.name(CUDA.device())), $(round(CUDA.total_memory() / 1e9, digits=1))GB")
    else
        println("GPU: not available, using CPU")
    end
    global _SWAMMA_GPU_BANNER_PRINTED = true
end

to_dev(x) = USE_GPU ? CUDA.cu(x) : x

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
_optimizer_state_to_device(opt_state) = _map_optimizer_state_arrays(opt_state, to_dev)

# ============================================================================
# Freeze mask: returns OptimiserRule that zeros gradients for frozen params
# ============================================================================

"""
Build a parameter-wise learning rate map.
Frozen params get LR=0, adapters/header params get base_lr.
"""
function build_optimizer(ps, base_lr::Float32)
    # We use Optimisers.Adam with a freeze mask via Optimisers.Descent(0) trick
    # Simpler: use a flat Adam and manually zero frozen gradients
    return Optimisers.Adam(base_lr)
end

"""
Zero out gradients for frozen parameters in-place.
Frozen = full shared front-end backbone, proposer core, full audit logic/circuit
         core, and FinalNorm.
Trainable = front-end header, proposer headers, audit input/circuit headers,
            audit score/agreement calibration, token/position/time embeddings,
            and OutputHead.
"""
function zero_frozen_grads!(grads, config)
    haskey(grads, :FrontEnd) || return grads

    # Freeze the full shared front-end backbone, including the codebook.
    fe = grads.FrontEnd
    _zero_if_exists!(fe, :Codebook)
    _zero_nested!(fe.InputNorm)
    _zero_if_exists!(fe, :EncoderWeight)
    _zero_if_exists!(fe, :EncoderBias)
    _zero_if_exists!(fe, :MaskCodeWeight)
    _zero_if_exists!(fe, :MaskCodeBias)
    _zero_if_exists!(fe, :WaveReadoutWeight)
    _zero_if_exists!(fe, :WaveReadoutBias)
    _zero_if_exists!(fe, :MaskReadoutWeight)
    _zero_if_exists!(fe, :MaskReadoutBias)
    _zero_if_exists!(fe, :log_wave_speed)
    _zero_if_exists!(fe, :log_damping)
    _zero_if_exists!(fe, :FusionWeight)
    _zero_if_exists!(fe, :FusionBias)
    _zero_if_exists!(fe, :GateWeight)
    _zero_if_exists!(fe, :GateBias)

    nl = config.number_of_layers
    for i in 1:nl
        key = Symbol("Block_$i")
        block_grads = grads.Blocks[key]

        # Freeze proposer core; keep proposal header trainable.
        _zero_nested!(block_grads.InputNorm)
        _zero_nested!(block_grads.GluProjection)
        _zero_nested!(block_grads.LinAttn)
        _zero_nested!(block_grads.WaveGateLayer)
        _zero_nested!(block_grads.AttnNorm)
        _zero_nested!(block_grads.WaveGateNorm)
        _zero_nested!(block_grads.FFN)
        _zero_nested!(block_grads.OutputNorm)
    end

    # Freeze audit-tail logic/circuit core; keep only headers and score heads trainable.
    audit = grads.AuditTail
    _zero_nested!(audit.InputNorm)
    _zero_if_exists!(audit, :FineEncoderWeight)
    _zero_if_exists!(audit, :FineEncoderBias)
    _zero_if_exists!(audit, :RoleBaseWeight)
    _zero_if_exists!(audit, :RoleBaseBias)
    _zero_if_exists!(audit, :RoleShiftWeight)
    _zero_if_exists!(audit, :PredicateRuleBank)
    _zero_if_exists!(audit, :PredicateOutputWeight)
    _zero_if_exists!(audit, :PredicateOutputBias)
    _zero_nested!(audit.CircuitLeafProjection)
    _zero_nested!(audit.Circuit)

    # Freeze final normalization.
    _zero_nested!(grads.FinalNorm)

    return grads
end

function _zero_if_exists!(nt, field::Symbol)
    nt isa NamedTuple || return
    if haskey(nt, field) && nt[field] isa AbstractArray
        fill!(nt[field], 0)
    end
end

function _zero_nested!(x)
    x isa AbstractArray && fill!(x, 0)
    x isa NamedTuple && for v in values(x); _zero_nested!(v); end
end

function _scale_if_exists!(nt, field::Symbol, s::Float32)
    nt isa NamedTuple || return
    if haskey(nt, field) && nt[field] isa AbstractArray
        nt[field] .*= s
    end
end

function _scale_nested!(x, s::Float32)
    x isa AbstractArray && (x .*= s)
    x isa NamedTuple && for v in values(x); _scale_nested!(v, s); end
end

function make_language_batch(tokens, vocab_size::Int, to_dev_fn)
    seq_len = size(tokens, 1)
    seq_len >= 2 || throw(ArgumentError("Need at least 2 tokens per sequence for next-token language loss."))
    device_tokens = to_dev_fn(tokens)
    target_tokens_flat = vec(Array(tokens[2:seq_len, :]))
    safe_targets_flat = Int.(clamp.(target_tokens_flat, 1, vocab_size))
    target_mask = Float32.(target_tokens_flat .> 1)
    n_valid = Float32(max(sum(target_mask), 1))
    positions = collect(1:length(safe_targets_flat))

    return (
        input_tokens = device_tokens[1:seq_len-1, :],
        target_indices = CartesianIndex.(safe_targets_flat, positions),
        target_mask = to_dev_fn(target_mask),
        n_valid = n_valid,
        mask_ratio = 0.0f0,
    )
end

# ============================================================================
# Training
# ============================================================================

function language_loss(model, ps, st, batch)
    logits, new_st = phase3a_logits(model, ps, st, batch)

    vocab = size(logits, 1)
    logits_flat = reshape(logits, vocab, :)

    log_probs = NNlib.logsoftmax(logits_flat, dims=1)
    target_log_probs = log_probs[batch.target_indices]
    nll = -sum(target_log_probs .* batch.target_mask) / batch.n_valid

    return nll, new_st
end

phase3a_output_vocab_size(config::ReasoningDrafterConfig) = min(config.vocab_size, REASONING_CHAR_VOCAB_SIZE)

function phase3a_logits(model::ReasoningDrafter, ps, st, batch)
    active_vocab_size = phase3a_output_vocab_size(model.config)
    inputs = (token_ids = batch.input_tokens, mask_ratio = batch.mask_ratio)
    hidden, partial_st = reasoning_hidden(model, inputs, ps, st)

    if active_vocab_size == model.config.vocab_size
        logits, oh_st = model.OutputHead(hidden, ps.OutputHead, st.OutputHead)
    else
        head_weight = @view ps.OutputHead.weight[1:active_vocab_size, :]
        hidden_flat = reshape(hidden, size(hidden, 1), :)
        logits_flat = head_weight * hidden_flat
        logits = reshape(logits_flat, active_vocab_size, size(hidden, 2), size(hidden, 3))
        oh_st = st.OutputHead
    end

    new_st = merge(partial_st, (OutputHead = oh_st,))
    return logits, new_st
end

function estimate_phase3a_footprint(
    config::ReasoningDrafterConfig;
    batch_size::Int,
    max_seq_length::Int,
)
    effective_seq_len = min(max_seq_length, config.max_sequence_length)
    input_seq_len = max(effective_seq_len - 1, 0)
    full_logits_elements = Int(config.vocab_size) * input_seq_len * batch_size
    full_logits_mebibytes = Float64(full_logits_elements * sizeof(Float32)) / 1024^2
    active_vocab_size = phase3a_output_vocab_size(config)
    active_logits_elements = Int(active_vocab_size) * input_seq_len * batch_size
    active_logits_mebibytes = Float64(active_logits_elements * sizeof(Float32)) / 1024^2
    token_embedding_params = Int(config.vocab_size) * Int(config.embedding_dimension)
    output_head_params = Int(config.embedding_dimension) * Int(config.vocab_size)
    position_embedding_params = Int(config.max_sequence_length) * Int(config.embedding_dimension)
    idle_vocab_rows = max(config.vocab_size - REASONING_CHAR_VOCAB_SIZE, 0)
    vocab_multiplier_vs_char = Float64(config.vocab_size) / REASONING_CHAR_VOCAB_SIZE

    return (
        effective_seq_len = effective_seq_len,
        input_seq_len = input_seq_len,
        full_logits_elements = full_logits_elements,
        full_logits_mebibytes = full_logits_mebibytes,
        active_vocab_size = active_vocab_size,
        active_logits_elements = active_logits_elements,
        active_logits_mebibytes = active_logits_mebibytes,
        token_embedding_params = token_embedding_params,
        output_head_params = output_head_params,
        position_embedding_params = position_embedding_params,
        idle_vocab_rows = idle_vocab_rows,
        vocab_multiplier_vs_char = vocab_multiplier_vs_char,
        char_vocab_mismatch = config.vocab_size != REASONING_CHAR_VOCAB_SIZE,
    )
end

function print_phase3a_resource_summary(
    config::ReasoningDrafterConfig;
    batch_size::Int,
    max_seq_length::Int,
)
    summary = estimate_phase3a_footprint(
        config;
        batch_size = batch_size,
        max_seq_length = max_seq_length,
    )
    println("Phase 3a footprint:")
    println(
        "  full output-head logits: $(summary.full_logits_elements) Float32 values " *
        "(~$(round(summary.full_logits_mebibytes, digits=1)) MiB) " *
        "for input_seq=$(summary.input_seq_len), batch=$batch_size, vocab=$(config.vocab_size)"
    )
    println(
        "  token/output params: " *
        "$(round(summary.token_embedding_params / 1e6, digits=3))M + " *
        "$(round(summary.output_head_params / 1e6, digits=3))M"
    )
    println(
        "  position params: $(round(summary.position_embedding_params / 1e6, digits=3))M " *
        "(max_sequence_length=$(config.max_sequence_length))"
    )
    if summary.char_vocab_mismatch
        println(
            "  WARNING: Phase 3a currently uses the char-level reasoning loader " *
            "(vocab=$(REASONING_CHAR_VOCAB_SIZE)), but the checkpoint config uses " *
            "vocab=$(config.vocab_size)."
        )
        println(
            "           That leaves $(summary.idle_vocab_rows) token/output rows unused " *
            "for this stage and would inflate full logits by " *
            "$(round(summary.vocab_multiplier_vs_char, digits=1))x."
        )
        println(
            "           Phase 3a will train with a sliced output head of size " *
            "$(summary.active_vocab_size), reducing per-step logits to " *
            "~$(round(summary.active_logits_mebibytes, digits=1)) MiB."
        )
    else
        println(
            "  active Phase 3a logits: $(summary.active_logits_elements) Float32 values " *
            "(~$(round(summary.active_logits_mebibytes, digits=1)) MiB)"
        )
    end
    return summary
end

function _save_phase3a_checkpoint(path, ps, st, opt_state, config, global_step, epoch; best_loss = nothing)
    ps_cpu = cpu_device()(ps)
    st_cpu = cpu_device()(st)
    opt_state_cpu = _optimizer_state_to_cpu(opt_state)
    if best_loss === nothing
        JLD2.@save path ps_cpu st_cpu opt_state_cpu config global_step epoch
    else
        JLD2.@save path ps_cpu st_cpu opt_state_cpu config global_step epoch best_loss
    end
end

function _legacy_config_value(raw_config, defaults::ReasoningDrafterConfig, field::Symbol)
    value = hasproperty(raw_config, field) ? getproperty(raw_config, field) : getproperty(defaults, field)
    return convert(fieldtype(ReasoningDrafterConfig, field), value)
end

function _coerce_reasoning_drafter_config(raw_config)
    raw_config isa ReasoningDrafterConfig && return raw_config
    defaults = ReasoningDrafterConfig()
    return ReasoningDrafterConfig(
        vocab_size = _legacy_config_value(raw_config, defaults, :vocab_size),
        max_sequence_length = _legacy_config_value(raw_config, defaults, :max_sequence_length),
        embedding_dimension = _legacy_config_value(raw_config, defaults, :embedding_dimension),
        number_of_heads = _legacy_config_value(raw_config, defaults, :number_of_heads),
        number_of_layers = _legacy_config_value(raw_config, defaults, :number_of_layers),
        time_dimension = _legacy_config_value(raw_config, defaults, :time_dimension),
        rc_code_dim = _legacy_config_value(raw_config, defaults, :rc_code_dim),
        rc_codebook_size = _legacy_config_value(raw_config, defaults, :rc_codebook_size),
        rc_integration_steps = _legacy_config_value(raw_config, defaults, :rc_integration_steps),
        frontend_wave_heads = _legacy_config_value(raw_config, defaults, :frontend_wave_heads),
        default_time_step = _legacy_config_value(raw_config, defaults, :default_time_step),
        min_frequency = _legacy_config_value(raw_config, defaults, :min_frequency),
        max_frequency = _legacy_config_value(raw_config, defaults, :max_frequency),
        proposer_ffn_expansion = _legacy_config_value(raw_config, defaults, :proposer_ffn_expansion),
        frontend_header_expansion = _legacy_config_value(raw_config, defaults, :frontend_header_expansion),
        audit_input_header_expansion = _legacy_config_value(raw_config, defaults, :audit_input_header_expansion),
        predicate_num_heads = _legacy_config_value(raw_config, defaults, :predicate_num_heads),
        num_roles = _legacy_config_value(raw_config, defaults, :num_roles),
        circuit_num_leaves = _legacy_config_value(raw_config, defaults, :circuit_num_leaves),
        circuit_product_arity = _legacy_config_value(raw_config, defaults, :circuit_product_arity),
        circuit_num_sums = _legacy_config_value(raw_config, defaults, :circuit_num_sums),
        circuit_num_circuits = _legacy_config_value(raw_config, defaults, :circuit_num_circuits),
        veto_gain = _legacy_config_value(raw_config, defaults, :veto_gain),
        use_adapters = _legacy_config_value(raw_config, defaults, :use_adapters),
    )
end

function _namedtuple_has_fields(x, fields::Tuple{Vararg{Symbol}})
    x isa NamedTuple || return false
    return all(field -> haskey(x, field), fields)
end

function _namedtuple_field_keys(x)
    x isa NamedTuple || return ()
    return keys(x)
end

function _phase3a_checkpoint_layout(ps)
    current_roots = (
        :TokenEmbedding,
        :PositionEmbedding,
        :FrontEnd,
        :FrontEndHeader,
        :Blocks,
        :AuditTail,
        :FinalNorm,
        :OutputHead,
        :TimeEmbedding,
    )
    legacy_roots = (
        :TokenEmbedding,
        :PositionEmbedding,
        :Blocks,
        :FinalNorm,
        :OutputHead,
        :TimeEmbedding,
    )
    legacy_block_fields = (
        :Norm,
        :RuleWave,
        :GluProjection,
        :LinAttn,
        :WaveGate,
        :ContentNorm,
        :GateNorm,
        :Circuit,
        :OutputNorm,
    )

    if _namedtuple_has_fields(ps, current_roots)
        return :current
    end

    if _namedtuple_has_fields(ps, legacy_roots) &&
       haskey(ps.Blocks, :Block_1) &&
       _namedtuple_has_fields(ps.Blocks.Block_1, legacy_block_fields)
        return :legacy_monolithic_phase2
    end

    if _namedtuple_has_fields(ps, (:Drafter, :MoveHead, :EvalHead))
        drafter = ps.Drafter
        if _namedtuple_has_fields(drafter, legacy_roots) &&
           haskey(drafter.Blocks, :Block_1) &&
           _namedtuple_has_fields(drafter.Blocks.Block_1, legacy_block_fields)
            return :legacy_phase1_with_heads
        end
    end

    return :unknown
end

function _phase3a_checkpoint_error(checkpoint_path::String, ps, layout::Symbol)
    root_keys = Tuple(_namedtuple_field_keys(ps))
    block_keys = if ps isa NamedTuple && haskey(ps, :Blocks) && haskey(ps.Blocks, :Block_1)
        Tuple(_namedtuple_field_keys(ps.Blocks.Block_1))
    elseif ps isa NamedTuple && haskey(ps, :Drafter) && haskey(ps.Drafter, :Blocks) && haskey(ps.Drafter.Blocks, :Block_1)
        Tuple(_namedtuple_field_keys(ps.Drafter.Blocks.Block_1))
    else
        ()
    end

    if layout == :legacy_monolithic_phase2
        return ArgumentError(
            "Checkpoint $checkpoint_path uses the legacy monolithic Phase 2 drafter layout " *
            "(roots=$(root_keys), block_1=$(block_keys)). " *
            "Current Phase 3a training expects a current-architecture checkpoint with " *
            "`FrontEnd`, `FrontEndHeader`, `Blocks`, `AuditTail`, and `FinalNorm`. " *
            "The checked-in legacy Phase 2 artifacts cannot be loaded safely into the split drafter. " *
            "Regenerate Phase 2 with a current-architecture Phase 1 checkpoint, or resume from a current Phase 3a checkpoint created on this branch."
        )
    elseif layout == :legacy_phase1_with_heads
        return ArgumentError(
            "Checkpoint $checkpoint_path is a legacy Phase 1 training checkpoint with top-level chess heads. " *
            "Phase 3a training does not accept raw Phase 1 checkpoints directly. " *
            "Run `scripts/transfer_surgery.jl` on a current-architecture Phase 1 drafter checkpoint first."
        )
    else
        return ArgumentError(
            "Checkpoint $checkpoint_path has an unrecognized parameter layout " *
            "(roots=$(root_keys), block_1=$(block_keys)). " *
            "Expected either a current split ReasoningDrafter checkpoint or a resumable Phase 3a checkpoint from this branch."
        )
    end
end

function _load_phase3a_state(checkpoint_path::String, rng::Random.AbstractRNG, learning_rate::Float32)
    ckpt = JLD2.load(checkpoint_path)
    ps = ckpt["ps_cpu"]
    layout = _phase3a_checkpoint_layout(ps)
    layout == :current || throw(_phase3a_checkpoint_error(checkpoint_path, ps, layout))
    config = _coerce_reasoning_drafter_config(ckpt["config"])
    model = ReasoningDrafter(config)
    st = haskey(ckpt, "st_cpu") ? ckpt["st_cpu"] : Lux.initialstates(rng, model)
    opt = build_optimizer(ps, learning_rate)
    opt_state = haskey(ckpt, "opt_state_cpu") ? ckpt["opt_state_cpu"] : Optimisers.setup(opt, ps)
    global_step = haskey(ckpt, "global_step") ? ckpt["global_step"] : 0
    start_epoch = haskey(ckpt, "epoch") ? ckpt["epoch"] : 0
    best_loss = haskey(ckpt, "best_loss") ? ckpt["best_loss"] : Inf
    resumed = haskey(ckpt, "st_cpu") || haskey(ckpt, "opt_state_cpu")
    return (;
        ps,
        config,
        model,
        st,
        opt_state,
        global_step,
        start_epoch,
        best_loss,
        resumed,
    )
end

function train_phase3a(;
    checkpoint_path::String,
    data_dir::String,
    output_dir::String,
    batch_size::Int = 32,
    num_epochs::Int = 10,
    learning_rate::Float32 = 3f-4,
    max_seq_length::Int = 256,
    checkpoint_every::Int = 300,
    max_per_dataset::Union{Int,Nothing} = nothing,
    max_steps::Union{Int,Nothing} = nothing,
    log_every::Int = 50,
    seed::Int = 42,
)
    rng = Random.MersenneTwister(seed)
    mkpath(output_dir)

    println("=== Phase 3a: Language Fine-Tuning ===")
    println("Checkpoint: $checkpoint_path")
    println("Data: $data_dir")
    println("Max seq: $max_seq_length, batch: $batch_size, LR: $learning_rate")

    # Load surgery checkpoint or resume Phase 3a checkpoint
    println("Loading checkpoint...")
    loaded = _load_phase3a_state(checkpoint_path, rng, learning_rate)
    ps = loaded.ps
    config = loaded.config
    model = loaded.model
    st = loaded.st
    opt_state = loaded.opt_state
    global_step = loaded.global_step
    start_epoch = loaded.start_epoch
    best_loss = loaded.best_loss
    println(loaded.resumed ? "  Mode: resume Phase 3a state" : "  Mode: initialize from Phase 2 params")
    println("  Config: dim=$(config.embedding_dimension), layers=$(config.number_of_layers), vocab=$(config.vocab_size)")
    println("  Adapters: $(config.use_adapters)")
    effective_max_seq_length = min(max_seq_length, config.max_sequence_length)
    if effective_max_seq_length != max_seq_length
        println("  Requested max_seq_length=$max_seq_length exceeds model limit=$(config.max_sequence_length); clamping to $effective_max_seq_length")
    end
    footprint = print_phase3a_resource_summary(
        config;
        batch_size = batch_size,
        max_seq_length = effective_max_seq_length,
    )

    if USE_GPU
        ps = to_dev(ps)
        st = to_dev(st)
        opt_state = _optimizer_state_to_device(opt_state)
    end

    # Load reasoning data
    println("Loading reasoning datasets from $data_dir...")
    examples = load_all_reasoning_datasets(
        data_dir;
        max_per_dataset = max_per_dataset,
        max_seq_length = effective_max_seq_length,
    )
    println("Total: $(length(examples)) examples")
    isempty(examples) && throw(ArgumentError("No reasoning examples loaded from $data_dir"))

    steps_run = 0
    stop_requested = false

    for epoch in 1:num_epochs
        current_epoch = start_epoch + epoch
        batches = iterate_reasoning_batches(examples, batch_size; shuffle=true, rng=rng)
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in batches
            if max_steps !== nothing && steps_run >= max_steps
                println("Reached max_steps=$max_steps; stopping Phase 3a early.")
                stop_requested = true
                break
            end
            global_step += 1
            steps_run += 1
            epoch_steps += 1

            language_batch = make_language_batch(batch.tokens, config.vocab_size, to_dev)

            if USE_GPU
                GC.gc(false)
            end

            (loss_val, new_st), grads = Zygote.withgradient(ps) do p
                language_loss(model, p, st, language_batch)
            end
            grads = grads[1]

            # Apply freeze mask
            zero_frozen_grads!(grads, config)

            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            grads = nothing
            st = new_st
            apply_reasoning_drafter_ema_codebook!(ps, st, model)

            epoch_loss += Float64(loss_val)

            if global_step % log_every == 0
                println("step=$global_step  loss=$(round(Float64(loss_val), digits=4))")
            end

            if global_step % checkpoint_every == 0
                cp_path = joinpath(output_dir, "checkpoint_last.jld2")
                _save_phase3a_checkpoint(cp_path, ps, st, opt_state, config, global_step, current_epoch; best_loss = best_loss)
                println("Checkpoint (step $global_step): $cp_path")
            end
        end

        if epoch_steps > 0
            avg_loss = epoch_loss / epoch_steps
            println("--- Epoch $current_epoch  avg_loss=$(round(avg_loss, digits=4)) ---")

            if avg_loss < best_loss
                best_loss = avg_loss
                best_path = joinpath(output_dir, "best.jld2")
                _save_phase3a_checkpoint(best_path, ps, st, opt_state, config, global_step, current_epoch; best_loss = best_loss)
                println("  New best! Saved to $best_path")
            end

            cp_path = joinpath(output_dir, "checkpoint_last.jld2")
            _save_phase3a_checkpoint(cp_path, ps, st, opt_state, config, global_step, current_epoch; best_loss = best_loss)
        end

        stop_requested && break
    end

    println("\n=== Phase 3a complete. Best loss: $(round(best_loss, digits=4)) ===")
    return (
        best_loss = best_loss,
        global_step = global_step,
        steps_run = steps_run,
        num_examples = length(examples),
        effective_max_seq_length = effective_max_seq_length,
        footprint = footprint,
    )
end

# CLI
function main()
    checkpoint = "checkpoints/reasoning_drafter/phase2/surgery.jld2"
    data_dir = "data/reasoning"
    output_dir = "checkpoints/reasoning_drafter/phase3a"
    epochs = 10
    batch_size = 32
    learning_rate = 3f-4
    max_seq_length = 256
    checkpoint_every = 300
    max_per_dataset = nothing
    max_steps = nothing
    log_every = 50
    seed = 42

    args = ARGS
    i = 1
    while i <= length(args)
        if args[i] == "--checkpoint" && i < length(args)
            checkpoint = args[i+1]; i += 2
        elseif args[i] == "--data-dir" && i < length(args)
            data_dir = args[i+1]; i += 2
        elseif args[i] == "--output-dir" && i < length(args)
            output_dir = args[i+1]; i += 2
        elseif args[i] == "--epochs" && i < length(args)
            epochs = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--batch-size" && i < length(args)
            batch_size = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--learning-rate" && i < length(args)
            learning_rate = parse(Float32, args[i+1]); i += 2
        elseif args[i] == "--max-seq-length" && i < length(args)
            max_seq_length = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--checkpoint-every" && i < length(args)
            checkpoint_every = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--max-per-dataset" && i < length(args)
            max_per_dataset = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--max-steps" && i < length(args)
            max_steps = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--log-every" && i < length(args)
            log_every = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--seed" && i < length(args)
            seed = parse(Int, args[i+1]); i += 2
        else
            i += 1
        end
    end

    train_phase3a(
        ;
        checkpoint_path = checkpoint,
        data_dir,
        output_dir,
        batch_size = batch_size,
        num_epochs = epochs,
        learning_rate = learning_rate,
        max_seq_length = max_seq_length,
        checkpoint_every = checkpoint_every,
        max_per_dataset = max_per_dataset,
        max_steps = max_steps,
        log_every = log_every,
        seed = seed,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
