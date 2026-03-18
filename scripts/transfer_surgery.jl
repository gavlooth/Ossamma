#!/usr/bin/env julia
"""
Phase 2: Transfer Surgery

Loads a Phase 1 chess checkpoint, freezes the reasoning backbone,
adds adapter headers, and reinitializes embeddings for the target vocab
(Granite, vocab_size=49160).

This script produces a checkpoint ready for Phase 3a language fine-tuning.

What is frozen (chess-learned reasoning backbone):
  - SpeedModWeight, DampingModWeight (how rules → wave dynamics)
  - log_wave_speed, log_damping (default propagation physics)
  - GluProjection, WavePDE gate, all Norms
  - SumLogWeights, ComposeLogWeights (circuit combination logic)
  - Encoder, RuleBank, LeafWeights, Gate weights

What is added (adapters, identity-initialized):
  - EncoderHeader, RuleBankHeader, GateBiasShift (in RuleConditionedWavePDE)
  - CircuitLeafHeader, CircuitGateBiasShift (in block)

What is reinitialized:
  - TokenEmbedding (new vocab)
  - OutputHead (new vocab)

Usage:
  julia --project=. scripts/transfer_surgery.jl \
    --input checkpoints/reasoning_drafter/phase1/best.jld2 \
    --output checkpoints/reasoning_drafter/phase2/surgery.jld2 \
    --target-vocab 49160
"""

using Swamma
using Swamma.ReasoningDrafterMod
using Swamma.RuleConditionedWavePDEMod

using Lux
using Random
using JLD2
using LinearAlgebra: I

function transfer_surgery(;
    input_path::String,
    output_path::String,
    target_vocab::Int = 49160,   # Granite vocab size
    max_seq_length::Int = 512,   # Granite context length (up from chess's 64)
    seed::Int = 42,
)
    rng = Random.MersenneTwister(seed)

    println("=== Phase 2: Transfer Surgery ===")
    println("Input:  $input_path")
    println("Output: $output_path")
    println("Target vocab: $target_vocab, max_seq: $max_seq_length")

    # Load Phase 1 checkpoint
    println("Loading Phase 1 checkpoint...")
    phase1 = JLD2.load(input_path)
    ps_loaded = phase1["ps_cpu"]
    ps_chess = haskey(ps_loaded, :Drafter) ? ps_loaded.Drafter : ps_loaded
    config_chess = phase1["config"]
    chess_step = get(phase1, "global_step", 0)
    println("  Chess config: dim=$(config_chess.embedding_dimension), layers=$(config_chess.number_of_layers)")
    println("  Chess step: $chess_step")

    # Build new config with adapters enabled and target vocab
    config_new = ReasoningDrafterConfig(
        vocab_size = target_vocab,
        max_sequence_length = max_seq_length,
        embedding_dimension = config_chess.embedding_dimension,
        number_of_heads = config_chess.number_of_heads,
        number_of_layers = config_chess.number_of_layers,
        time_dimension = config_chess.time_dimension,
        min_frequency = config_chess.min_frequency,
        max_frequency = config_chess.max_frequency,
        default_time_step = config_chess.default_time_step,
        rc_code_dim = config_chess.rc_code_dim,
        rc_codebook_size = config_chess.rc_codebook_size,
        rc_integration_steps = config_chess.rc_integration_steps,
        circuit_num_leaves = config_chess.circuit_num_leaves,
        circuit_product_arity = config_chess.circuit_product_arity,
        circuit_num_sums = config_chess.circuit_num_sums,
        circuit_num_circuits = config_chess.circuit_num_circuits,
        use_adapters = true,   # <-- enable adapters
    )

    # Build new model with adapters
    model_new = ReasoningDrafter(config_new)
    ps_new = Lux.initialparameters(rng, model_new)
    st_new = Lux.initialstates(rng, model_new)

    println("Performing surgery...")

    # Copy chess-trained block parameters (everything except adapters)
    nl = config_new.number_of_layers
    for i in 1:nl
        key = Symbol("Block_$i")
        chess_block = ps_chess.Blocks[key]
        new_block = ps_new.Blocks[key]

        # RuleWave: copy all chess params, adapters stay identity-initialized
        for field in (:Codebook, :EncoderWeight, :EncoderBias, :RuleBank,
                      :SpeedModWeight, :DampingModWeight,
                      :log_wave_speed, :log_damping, :GateWeight, :GateBias)
            if haskey(chess_block.RuleWave, field) && chess_block.RuleWave[field] !== nothing
                copyto!(ps_new.Blocks[key].RuleWave[field], chess_block.RuleWave[field])
            end
        end

        # GLU, norms, LinAttn, WaveGate, Circuit — copy from chess
        for comp in (:GluProjection, :LinAttn, :WaveGate, :ContentNorm, :GateNorm, :Circuit, :OutputNorm, :Norm)
            _recursive_copy!(ps_new.Blocks[key][comp], chess_block[comp])
        end
    end

    # Copy FinalNorm
    _recursive_copy!(ps_new.FinalNorm, ps_chess.FinalNorm)

    # TimeEmbedding: copy from chess
    copyto!(ps_new.TimeEmbedding, ps_chess.TimeEmbedding)

    # TokenEmbedding and OutputHead: KEEP RANDOMLY INITIALIZED (new vocab)
    # PositionEmbedding: reinit for new max_seq_length (keep random)
    println("  TokenEmbedding: reinitialized for vocab=$target_vocab")
    println("  PositionEmbedding: reinitialized for max_seq=$max_seq_length")
    println("  OutputHead: reinitialized for vocab=$target_vocab")

    # Verify adapters are identity
    cd = config_new.rc_code_dim
    dim = config_new.embedding_dimension
    for i in 1:nl
        key = Symbol("Block_$i")
        enc_header = ps_new.Blocks[key].RuleWave.EncoderHeaderWeight
        if enc_header !== nothing
            is_identity = isapprox(enc_header, Matrix{Float32}(I, cd, cd); atol=1f-6)
            println("  Block $i EncoderHeader: $(is_identity ? "identity ✓" : "NOT identity ✗")")
        end
        leaf_header = ps_new.Blocks[key].CircuitLeafHeaderWeight
        if leaf_header !== nothing
            is_identity = isapprox(leaf_header, Matrix{Float32}(I, dim, dim); atol=1f-6)
            println("  Block $i CircuitLeafHeader: $(is_identity ? "identity ✓" : "NOT identity ✗")")
        end
    end

    # Save
    mkpath(dirname(output_path))
    ps_cpu = ps_new
    config = config_new
    JLD2.@save output_path ps_cpu config chess_step
    println("\nSaved surgery checkpoint: $output_path")

    # Summary
    function count_params(x)
        x isa AbstractArray && return length(x)
        x isa NamedTuple && return sum(count_params(v) for v in values(x))
        x isa Tuple && return sum(count_params(v) for v in x)
        x isa Nothing && return 0
        return 0
    end
    println("Total params: $(round(count_params(ps_new) / 1e6, digits=3))M")
    println("=== Phase 2 complete ===")
end

# Recursive copy helper
function _recursive_copy!(dst, src)
    if dst isa AbstractArray && src isa AbstractArray
        size(dst) == size(src) && copyto!(dst, src)
    elseif dst isa NamedTuple && src isa NamedTuple
        for k in keys(dst)
            if haskey(src, k)
                _recursive_copy!(dst[k], src[k])
            end
        end
    end
end

# CLI
function main()
    input_path = "checkpoints/reasoning_drafter/phase1/best.jld2"
    output_path = "checkpoints/reasoning_drafter/phase2/surgery.jld2"
    target_vocab = 49160

    args = ARGS
    i = 1
    while i <= length(args)
        if args[i] == "--input" && i < length(args)
            input_path = args[i+1]; i += 2
        elseif args[i] == "--output" && i < length(args)
            output_path = args[i+1]; i += 2
        elseif args[i] == "--target-vocab" && i < length(args)
            target_vocab = parse(Int, args[i+1]); i += 2
        else
            i += 1
        end
    end

    transfer_surgery(; input_path, output_path, target_vocab)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
