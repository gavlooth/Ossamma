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
  PERMANENTLY FROZEN: SpeedModWeight, DampingModWeight, log_wave_speed,
    log_damping, GluProjection, WavePDE gate, Norms, SumLogWeights,
    ComposeLogWeights, Encoder, RuleBank, LeafWeights, Gate weights
  ADAPTERS (1x LR): EncoderHeader, RuleBankHeader, GateBiasShift,
    CircuitLeafHeader, CircuitGateBiasShift
  THAWED (0.1x LR): Codebook, LinearAttention, Circuit OutputWeight
  FULL (1x LR): TokenEmbedding, OutputHead

CUDA/GPU rules: no try/catch in loops, println() only, grads=nothing after step.

Usage:
  julia --project=. scripts/train_reasoning_language.jl \
    --checkpoint checkpoints/reasoning_drafter/phase2/surgery.jld2 \
    --data-dir data/reasoning \
    --output-dir checkpoints/reasoning_drafter/phase3a
"""

using Swamma
using Swamma.ReasoningDrafterMod
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

if USE_GPU
    println("GPU: $(CUDA.name(CUDA.device())), $(round(CUDA.total_memory() / 1e9, digits=1))GB")
else
    println("GPU: not available, using CPU")
end

to_dev(x) = USE_GPU ? CUDA.cu(x) : x

# ============================================================================
# Freeze mask: returns OptimiserRule that zeros gradients for frozen params
# ============================================================================

"""
Build a parameter-wise learning rate map.
Frozen params get LR=0, adapters get base_lr, thawed get 0.1*base_lr.
"""
function build_optimizer(ps, base_lr::Float32)
    # We use Optimisers.Adam with a freeze mask via Optimisers.Descent(0) trick
    # Simpler: use a flat Adam and manually zero frozen gradients
    return Optimisers.Adam(base_lr)
end

"""
Zero out gradients for frozen parameters in-place.
Frozen = everything in RuleWave except Codebook and adapter headers,
         everything in Circuit except OutputWeight/OutputBias and adapter headers,
         GluProjection, WaveGate, Norms.
"""
function zero_frozen_grads!(grads, config)
    nl = config.number_of_layers
    for i in 1:nl
        key = Symbol("Block_$i")
        block_grads = grads.Blocks[key]

        # Freeze: RuleWave backbone (not codebook, not adapters)
        rw = block_grads.RuleWave
        _zero_if_exists!(rw, :EncoderWeight)
        _zero_if_exists!(rw, :EncoderBias)
        _zero_if_exists!(rw, :RuleBank)
        _zero_if_exists!(rw, :SpeedModWeight)
        _zero_if_exists!(rw, :DampingModWeight)
        _zero_if_exists!(rw, :log_wave_speed)
        _zero_if_exists!(rw, :log_damping)
        _zero_if_exists!(rw, :GateWeight)
        _zero_if_exists!(rw, :GateBias)

        # Freeze: GluProjection, WaveGate, Norms
        _zero_nested!(block_grads.GluProjection)
        _zero_nested!(block_grads.WaveGate)
        _zero_nested!(block_grads.ContentNorm)
        _zero_nested!(block_grads.GateNorm)
        _zero_nested!(block_grads.Norm)
        _zero_nested!(block_grads.OutputNorm)

        # Freeze: Circuit backbone (not OutputWeight, not adapters)
        circ = block_grads.Circuit
        _zero_if_exists!(circ, :LeafWeights)
        _zero_if_exists!(circ, :LeafBiases)
        _zero_if_exists!(circ, :SumLogWeights)
        _zero_if_exists!(circ, :ComposeLogWeights)
        _zero_if_exists!(circ, :GateWeight)
        _zero_if_exists!(circ, :GateBias)

        # Scale thawed params to 0.1x (Codebook, LinAttn, Circuit OutputWeight)
        _scale_if_exists!(rw, :Codebook, 0.1f0)
        _scale_nested!(block_grads.LinAttn, 0.1f0)
        _scale_if_exists!(circ, :OutputWeight, 0.1f0)
        _scale_if_exists!(circ, :OutputBias, 0.1f0)
    end

    # Freeze FinalNorm
    _zero_nested!(grads.FinalNorm)

    return grads
end

function _zero_if_exists!(nt, field::Symbol)
    if haskey(nt, field) && nt[field] isa AbstractArray
        fill!(nt[field], 0)
    end
end

function _zero_nested!(x)
    x isa AbstractArray && fill!(x, 0)
    x isa NamedTuple && for v in values(x); _zero_nested!(v); end
end

function _scale_if_exists!(nt, field::Symbol, s::Float32)
    if haskey(nt, field) && nt[field] isa AbstractArray
        nt[field] .*= s
    end
end

function _scale_nested!(x, s::Float32)
    x isa AbstractArray && (x .*= s)
    x isa NamedTuple && for v in values(x); _scale_nested!(v, s); end
end

# ============================================================================
# Training
# ============================================================================

function language_loss(model, ps, st, tokens)
    # Next-token prediction: predict token[t+1] from position t
    seq_len = size(tokens, 1)
    input_tokens = tokens[1:seq_len-1, :]
    target_tokens = tokens[2:seq_len, :]

    logits, new_st = model(input_tokens, ps, st)

    vocab = size(logits, 1)
    logits_flat = reshape(logits, vocab, :)
    targets_flat = vec(target_tokens)

    # Ignore padding tokens (PAD_TOKEN = 1)
    mask = targets_flat .> 1
    n_valid = max(sum(mask), 1)

    log_probs = NNlib.logsoftmax(logits_flat, dims=1)
    nll = -sum(log_probs[CartesianIndex.(targets_flat, 1:length(targets_flat))] .* mask) / n_valid

    return nll, new_st
end

function train_phase3a(;
    checkpoint_path::String,
    data_dir::String,
    output_dir::String,
    batch_size::Int = 32,
    num_epochs::Int = 10,
    learning_rate::Float32 = 3f-4,
    max_seq_length::Int = 256,
    checkpoint_every::Int = 200,
    seed::Int = 42,
)
    rng = Random.MersenneTwister(seed)
    mkpath(output_dir)

    println("=== Phase 3a: Language Fine-Tuning ===")
    println("Checkpoint: $checkpoint_path")
    println("Data: $data_dir")
    println("Max seq: $max_seq_length, batch: $batch_size, LR: $learning_rate")

    # Load surgery checkpoint
    println("Loading Phase 2 checkpoint...")
    phase2 = JLD2.load(checkpoint_path)
    ps = phase2["ps_cpu"]
    config = phase2["config"]
    println("  Config: dim=$(config.embedding_dimension), layers=$(config.number_of_layers), vocab=$(config.vocab_size)")
    println("  Adapters: $(config.use_adapters)")

    # Build model
    model = ReasoningDrafter(config)
    st = Lux.initialstates(rng, model)

    if USE_GPU
        ps = to_dev(ps)
        st = to_dev(st)
    end

    # Load reasoning data
    println("Loading reasoning datasets from $data_dir...")
    examples = load_all_reasoning_datasets(data_dir; max_seq_length=max_seq_length)
    println("Total: $(length(examples)) examples")

    # Optimizer
    opt = build_optimizer(ps, learning_rate)
    opt_state = Optimisers.setup(opt, ps)

    global_step = 0
    best_loss = Inf

    for epoch in 1:num_epochs
        batches = iterate_reasoning_batches(examples, batch_size; shuffle=true, rng=rng)
        epoch_loss = 0.0

        for batch in batches
            global_step += 1

            b_tokens = to_dev(batch.tokens)

            if USE_GPU
                GC.gc(false)
            end

            (loss_val, new_st), grads = Zygote.withgradient(ps) do p
                language_loss(model, p, st, b_tokens)
            end
            grads = grads[1]

            # Apply freeze mask
            zero_frozen_grads!(grads, config)

            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            grads = nothing
            st = new_st

            epoch_loss += Float64(loss_val)

            if global_step % 50 == 0
                println("step=$global_step  loss=$(round(Float64(loss_val), digits=4))")
            end

            if global_step % checkpoint_every == 0
                cp_path = joinpath(output_dir, "step_$(global_step).jld2")
                ps_cpu = Lux.cpu(ps)
                JLD2.@save cp_path ps_cpu config global_step epoch
                println("Checkpoint: $cp_path")
            end
        end

        avg_loss = epoch_loss / max(length(batches), 1)
        println("--- Epoch $epoch/$num_epochs  avg_loss=$(round(avg_loss, digits=4)) ---")

        if avg_loss < best_loss
            best_loss = avg_loss
            best_path = joinpath(output_dir, "best.jld2")
            ps_cpu = Lux.cpu(ps)
            JLD2.@save best_path ps_cpu config global_step epoch
            println("  New best! Saved to $best_path")
        end
    end

    println("\n=== Phase 3a complete. Best loss: $(round(best_loss, digits=4)) ===")
end

# CLI
function main()
    checkpoint = "checkpoints/reasoning_drafter/phase2/surgery.jld2"
    data_dir = "data/reasoning"
    output_dir = "checkpoints/reasoning_drafter/phase3a"
    epochs = 10

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
        else
            i += 1
        end
    end

    train_phase3a(; checkpoint_path=checkpoint, data_dir, output_dir, num_epochs=epochs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
