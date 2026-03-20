#!/usr/bin/env julia
"""
Phase 2: Transfer Surgery

Loads a Phase 1 chess checkpoint, freezes the reasoning backbone,
adds adapter headers, and reinitializes embeddings for the target vocab
(Granite, vocab_size=49160).

This script produces a checkpoint ready for Phase 3a language fine-tuning.

What is frozen (chess-learned reasoning backbone):
  - FrontEnd encoder / wave / fusion backbone
  - proposer LinearAttention / FFN backbone
  - audit-tail predicate / circuit backbone
  - FinalNorm

What is added (adapters, identity-initialized):
  - FrontEndHeader after the shared front end
  - ProposalHeaderWeight / ProposalHeaderBias in each proposer block
  - AuditInputHeader before the frozen audit core
  - CircuitLeafHeaderWeight / CircuitLeafHeaderBias / CircuitGateBiasShift
    in the audit tail

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

using Lux
using Random
using JLD2
using LinearAlgebra: I

function _namedtuple_has_fields(x, fields::Tuple{Vararg{Symbol}})
    x isa NamedTuple || return false
    return all(field -> haskey(x, field), fields)
end

function _namedtuple_field_keys(x)
    x isa NamedTuple || return ()
    return keys(x)
end

function _phase1_drafter_layout(ps)
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
    current_block_fields = (
        :InputNorm,
        :GluProjection,
        :LinAttn,
        :WaveGateLayer,
        :AttnNorm,
        :WaveGateNorm,
        :FFN,
        :OutputNorm,
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

    if _namedtuple_has_fields(ps, current_roots) &&
       haskey(ps.Blocks, :Block_1) &&
       _namedtuple_has_fields(ps.Blocks.Block_1, current_block_fields)
        return :current
    end

    if _namedtuple_has_fields(ps, legacy_roots) &&
       haskey(ps.Blocks, :Block_1) &&
       _namedtuple_has_fields(ps.Blocks.Block_1, legacy_block_fields)
        return :legacy_monolithic
    end

    return :unknown
end

function _transfer_surgery_compatibility_error(input_path::String, ps, layout::Symbol)
    root_keys = Tuple(_namedtuple_field_keys(ps))
    block_keys = if ps isa NamedTuple && haskey(ps, :Blocks) && haskey(ps.Blocks, :Block_1)
        Tuple(_namedtuple_field_keys(ps.Blocks.Block_1))
    else
        ()
    end

    if layout == :legacy_monolithic
        return ArgumentError(
            "Checkpoint $input_path uses the legacy monolithic ReasoningDrafter layout " *
            "(roots=$(root_keys), block_1=$(block_keys)). " *
            "Current transfer surgery expects a current-architecture Phase 1 drafter checkpoint with " *
            "`FrontEnd`, `FrontEndHeader`, `Blocks`, `AuditTail`, and `FinalNorm`. " *
            "There is no safe 1:1 transfer from the old per-block `RuleWave`/`Circuit` layout into the split front-end plus audit-tail model. " *
            "Regenerate the Phase 1 checkpoint on the current branch before running Phase 2 surgery."
        )
    end

    return ArgumentError(
        "Checkpoint $input_path has an unrecognized drafter layout " *
        "(roots=$(root_keys), block_1=$(block_keys)). " *
        "Transfer surgery only supports current-architecture Phase 1 drafter checkpoints."
    )
end

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
    layout = _phase1_drafter_layout(ps_chess)
    layout == :current || throw(_transfer_surgery_compatibility_error(input_path, ps_chess, layout))
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
        rc_code_dim = _cfgprop(config_chess, :rc_code_dim, 64),
        rc_codebook_size = _cfgprop(config_chess, :rc_codebook_size, 512),
        rc_integration_steps = _cfgprop(config_chess, :rc_integration_steps, 8),
        frontend_wave_heads = _cfgprop(config_chess, :frontend_wave_heads, 4),
        default_time_step = _cfgprop(config_chess, :default_time_step, 0.1f0),
        min_frequency = _cfgprop(config_chess, :min_frequency, 0.1f0),
        max_frequency = _cfgprop(config_chess, :max_frequency, 10.0f0),
        proposer_ffn_expansion = _cfgprop(config_chess, :proposer_ffn_expansion, 3f0 / 2f0),
        predicate_num_heads = _cfgprop(config_chess, :predicate_num_heads, 4),
        num_roles = _cfgprop(config_chess, :num_roles, 4),
        circuit_num_leaves = _cfgprop(config_chess, :circuit_num_leaves, 16),
        circuit_product_arity = _cfgprop(config_chess, :circuit_product_arity, 2),
        circuit_num_sums = _cfgprop(config_chess, :circuit_num_sums, 8),
        circuit_num_circuits = _cfgprop(config_chess, :circuit_num_circuits, 4),
        use_adapters = true,   # <-- enable adapters
    )

    # Build new model with adapters
    model_new = ReasoningDrafter(config_new)
    ps_new = Lux.initialparameters(rng, model_new)
    st_new = Lux.initialstates(rng, model_new)

    println("Performing surgery...")

    # Copy chess-trained block parameters. If the source checkpoint already has
    # adapters, keep them; otherwise the new headers stay identity-initialized.
    nl = config_new.number_of_layers
    for i in 1:nl
        key = Symbol("Block_$i")
        chess_block = ps_chess.Blocks[key]
        new_block = ps_new.Blocks[key]

        # Proposer: copy the learned backbone; keep the proposal header identity-initialized.
        for field in (:InputNorm, :GluProjection, :LinAttn, :WaveGateLayer, :AttnNorm, :WaveGateNorm, :FFN, :OutputNorm)
            _recursive_copy!(ps_new.Blocks[key][field], chess_block[field])
        end

        _recursive_copy!(ps_new.Blocks[key].ProposalHeaderWeight, get(chess_block, :ProposalHeaderWeight, nothing))
        _recursive_copy!(ps_new.Blocks[key].ProposalHeaderBias, get(chess_block, :ProposalHeaderBias, nothing))
    end

    # Shared front end: copy learned backbone and any adapter header if present.
    for field in (:Codebook, :InputNorm, :EncoderWeight, :EncoderBias,
                  :MaskCodeWeight, :MaskCodeBias,
                  :WaveReadoutWeight, :WaveReadoutBias,
                  :MaskReadoutWeight, :MaskReadoutBias,
                  :log_wave_speed, :log_damping,
                  :FusionWeight, :FusionBias, :GateWeight, :GateBias)
        _recursive_copy!(ps_new.FrontEnd[field], ps_chess.FrontEnd[field])
    end
    _recursive_copy!(ps_new.FrontEndHeader, get(ps_chess, :FrontEndHeader, nothing))

    # Audit tail: copy the learned logic backbone and any adapter headers if present.
    for field in (:InputNorm, :FineEncoderWeight, :FineEncoderBias,
                  :RoleBaseWeight, :RoleBaseBias, :RoleShiftWeight,
                  :PredicateRuleBank, :PredicateOutputWeight, :PredicateOutputBias,
                  :CircuitLeafProjection, :Circuit,
                  :ScoreWeight, :AgreementWeight, :ScoreBias, :AgreementBias)
        _recursive_copy!(ps_new.AuditTail[field], ps_chess.AuditTail[field])
    end
    _recursive_copy!(ps_new.AuditTail.AuditInputHeader, get(ps_chess.AuditTail, :AuditInputHeader, nothing))
    _recursive_copy!(ps_new.AuditTail.CircuitLeafHeaderWeight, get(ps_chess.AuditTail, :CircuitLeafHeaderWeight, nothing))
    _recursive_copy!(ps_new.AuditTail.CircuitLeafHeaderBias, get(ps_chess.AuditTail, :CircuitLeafHeaderBias, nothing))
    _recursive_copy!(ps_new.AuditTail.CircuitGateBiasShift, get(ps_chess.AuditTail, :CircuitGateBiasShift, nothing))

    # Copy FinalNorm
    _recursive_copy!(ps_new.FinalNorm, ps_chess.FinalNorm)

    # TimeEmbedding: copy from chess
    copyto!(ps_new.TimeEmbedding, ps_chess.TimeEmbedding)

    # TokenEmbedding and OutputHead: KEEP RANDOMLY INITIALIZED (new vocab)
    # PositionEmbedding: reinit for new max_seq_length (keep random)
    println("  TokenEmbedding: reinitialized for vocab=$target_vocab")
    println("  PositionEmbedding: reinitialized for max_seq=$max_seq_length")
    println("  OutputHead: reinitialized for vocab=$target_vocab")

    # Verify adapters are initialized sensibly.
    dim = config_new.embedding_dimension
    if ps_new.FrontEndHeader !== nothing
        println("  FrontEndHeader: $(get(ps_chess, :FrontEndHeader, nothing) === nothing ? "fresh ✓" : "copied ✓")")
    end
    for i in 1:nl
        key = Symbol("Block_$i")
        prop_header = ps_new.Blocks[key].ProposalHeaderWeight
        if prop_header !== nothing
            if get(ps_chess.Blocks[key], :ProposalHeaderWeight, nothing) === nothing
                is_identity = isapprox(prop_header, Matrix{Float32}(I, dim, dim); atol=1f-6)
                println("  Block $i ProposalHeader: $(is_identity ? "identity ✓" : "NOT identity ✗")")
            else
                println("  Block $i ProposalHeader: copied ✓")
            end
        end
    end

    if ps_new.AuditTail.AuditInputHeader !== nothing
        println("  AuditTail AuditInputHeader: $(get(ps_chess.AuditTail, :AuditInputHeader, nothing) === nothing ? "fresh ✓" : "copied ✓")")
    end
    leaf_header = ps_new.AuditTail.CircuitLeafHeaderWeight
    if leaf_header !== nothing
        if get(ps_chess.AuditTail, :CircuitLeafHeaderWeight, nothing) === nothing
            is_identity = isapprox(leaf_header, Matrix{Float32}(I, dim, dim); atol=1f-6)
            println("  AuditTail CircuitLeafHeader: $(is_identity ? "identity ✓" : "NOT identity ✗")")
        else
            println("  AuditTail CircuitLeafHeader: copied ✓")
        end
    end
    gate_shift = ps_new.AuditTail.CircuitGateBiasShift
    if gate_shift !== nothing
        println("  AuditTail CircuitGateBiasShift: $(all(gate_shift .== 10f0) ? "identity-ish ✓" : "non-default ✗")")
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

function _cfgprop(cfg, name::Symbol, default)
    hasproperty(cfg, name) ? getproperty(cfg, name) : default
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
