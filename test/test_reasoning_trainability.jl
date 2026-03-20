module ReasoningTrainabilitySmoke

using Swamma
using Swamma.ReasoningDrafterMod
using Swamma.ReasoningDrafterMod: apply_reasoning_drafter_ema_codebook!
using Random
using Lux
using NNlib
using Optimisers
using Zygote
using JLD2
using Test

const CPU_DEV = cpu_device()

include(joinpath(dirname(@__DIR__), "scripts", "train_reasoning_language.jl"))
include(joinpath(dirname(@__DIR__), "scripts", "transfer_surgery.jl"))

function reasoning_language_loss(model, ps, st, tokens)
    seq_len = size(tokens, 1)
    input_tokens = tokens[1:seq_len-1, :]
    target_tokens = tokens[2:seq_len, :]

    logits, new_st = model(input_tokens, ps, st)
    vocab = size(logits, 1)
    logits_flat = reshape(logits, vocab, :)
    targets_flat = vec(target_tokens)

    mask = targets_flat .> 1
    n_valid = max(sum(mask), 1)
    log_probs = NNlib.logsoftmax(logits_flat, dims=1)
    nll = -sum(log_probs[CartesianIndex.(targets_flat, 1:length(targets_flat))] .* mask) / n_valid

    return nll, new_st
end

@testset "Reasoning Phase 3a Trainability Smoke" begin
    rng = Random.MersenneTwister(7)
    config = ReasoningDrafterConfig(
        vocab_size = 64,
        max_sequence_length = 12,
        embedding_dimension = 32,
        number_of_heads = 2,
        number_of_layers = 1,
        time_dimension = 16,
        rc_code_dim = 16,
        rc_codebook_size = 32,
        rc_integration_steps = 4,
        circuit_num_leaves = 8,
        circuit_product_arity = 2,
        circuit_num_sums = 4,
        circuit_num_circuits = 2,
    )

    model = ReasoningDrafter(config)
    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)
    opt = Optimisers.Adam(1f-3)
    opt_state = Optimisers.setup(opt, ps)

    batch_tokens = rand(rng, 2:config.vocab_size, config.max_sequence_length, 2)
    initial_output_head = copy(ps.OutputHead.weight)
    losses = Float32[]

    for _ in 1:2
        (loss, new_st), grads = Zygote.withgradient(ps) do p
            reasoning_language_loss(model, p, st, batch_tokens)
        end
        grad_ps = grads[1]

        @test isfinite(loss)
        @test grad_ps !== nothing
        @test grad_ps.OutputHead.weight !== nothing
        @test all(isfinite, grad_ps.OutputHead.weight)

        opt_state, ps = Optimisers.update(opt_state, ps, grad_ps)
        st = new_st
        apply_reasoning_drafter_ema_codebook!(ps, st, model)
        push!(losses, Float32(loss))
    end

    @test length(losses) == 2
    @test all(isfinite, losses)
    @test !isapprox(sum(abs.(ps.OutputHead.weight .- initial_output_head)), 0.0f0; atol = 1f-8)

    mktempdir() do tmpdir
        checkpoint_path = joinpath(tmpdir, "reasoning_phase3a_smoke.jld2")
        ps_cpu = CPU_DEV(ps)
        st_cpu = CPU_DEV(st)
        opt_state_cpu = _optimizer_state_to_cpu(opt_state)
        JLD2.@save checkpoint_path ps_cpu st_cpu opt_state_cpu config losses

        @test isfile(checkpoint_path)
        ckpt = JLD2.load(checkpoint_path)
        @test haskey(ckpt, "ps_cpu")
        @test haskey(ckpt, "st_cpu")
        @test haskey(ckpt, "opt_state_cpu")
        @test ckpt["config"].max_sequence_length == config.max_sequence_length
        @test length(ckpt["losses"]) == 2
    end
end

@testset "Phase 3a language helpers" begin
    rng = Random.MersenneTwister(11)
    config = ReasoningDrafterConfig(
        vocab_size = 32,
        max_sequence_length = 10,
        embedding_dimension = 16,
        number_of_heads = 2,
        number_of_layers = 1,
        time_dimension = 8,
        rc_code_dim = 8,
        rc_codebook_size = 16,
        rc_integration_steps = 2,
        circuit_num_leaves = 4,
        circuit_product_arity = 2,
        circuit_num_sums = 2,
        circuit_num_circuits = 2,
    )

    model = ReasoningDrafter(config)
    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)
    opt_state = Optimisers.setup(Optimisers.Adam(1f-3), ps)

    tokens = reshape(Int[2, 3, 4, 1, 6, 7, 8, 9, 10, 11], :, 1)
    batch = make_language_batch(tokens, config.vocab_size, identity)

    @test batch.input_tokens == tokens[1:end-1, :]
    @test length(batch.target_indices) == size(tokens, 1) - 1
    @test eltype(batch.target_indices) == CartesianIndex{2}
    @test !haskey(batch, :target_onehot)
    @test batch.target_mask == Float32[1, 1, 0, 1, 1, 1, 1, 1, 1]
    @test batch.n_valid == 8.0f0

    loss, st2 = language_loss(model, ps, st, batch)
    logits, _ = model(batch.input_tokens, ps, st)
    log_probs = NNlib.logsoftmax(reshape(logits, size(logits, 1), :), dims = 1)
    expected_loss = -sum(log_probs[batch.target_indices] .* batch.target_mask) / batch.n_valid
    @test isfinite(loss)
    @test isapprox(loss, expected_loss; atol = 0.0, rtol = 0.0)
    @test st2 !== nothing

    mktempdir() do tmpdir
        resume_path = joinpath(tmpdir, "phase3a_resume.jld2")
        (step_loss, step_st), step_grads = Zygote.withgradient(ps) do p
            language_loss(model, p, st, batch)
        end
        step_grad = step_grads[1]
        step_opt_state, step_ps = Optimisers.update(opt_state, ps, step_grad)
        _save_phase3a_checkpoint(resume_path, step_ps, step_st, step_opt_state, config, 7, 3; best_loss = 1.25)

        resumed = _load_phase3a_state(resume_path, rng, 1f-3)
        @test resumed.resumed
        @test resumed.global_step == 7
        @test resumed.start_epoch == 3
        @test resumed.best_loss == 1.25
        @test haskey(resumed.st, :Blocks)
        @test resumed.ps.OutputHead.weight == step_ps.OutputHead.weight
        @test resumed.st == step_st
        @test resumed.opt_state == step_opt_state

        (live_next_loss, live_next_st), live_next_grads = Zygote.withgradient(step_ps) do p
            language_loss(model, p, step_st, batch)
        end
        (resume_next_loss, resume_next_st), resume_next_grads = Zygote.withgradient(resumed.ps) do p
            language_loss(resumed.model, p, resumed.st, batch)
        end
        @test isapprox(live_next_loss, resume_next_loss; atol = 0.0, rtol = 0.0)
        @test live_next_st == resume_next_st
        @test live_next_grads[1].OutputHead.weight == resume_next_grads[1].OutputHead.weight

        legacy_path = joinpath(tmpdir, "phase2_legacy.jld2")
        ps_cpu = CPU_DEV(ps)
        JLD2.@save legacy_path ps_cpu config

        legacy = _load_phase3a_state(legacy_path, rng, 1f-3)
        @test !legacy.resumed
        @test legacy.global_step == 0
        @test legacy.start_epoch == 0
        @test legacy.best_loss == Inf
        @test haskey(legacy.st, :Blocks)
    end
end

@testset "Phase 3a bounded train run" begin
    rng = Random.MersenneTwister(23)
    config = ReasoningDrafterConfig(
        vocab_size = REASONING_CHAR_VOCAB_SIZE,
        max_sequence_length = 12,
        embedding_dimension = 16,
        number_of_heads = 2,
        number_of_layers = 1,
        time_dimension = 8,
        rc_code_dim = 8,
        rc_codebook_size = 16,
        rc_integration_steps = 2,
        frontend_wave_heads = 2,
        circuit_num_leaves = 4,
        circuit_product_arity = 2,
        circuit_num_sums = 2,
        circuit_num_circuits = 2,
    )

    model = ReasoningDrafter(config)
    ps = Lux.initialparameters(rng, model)

    mktempdir() do tmpdir
        checkpoint_path = joinpath(tmpdir, "phase2_legacy.jld2")
        data_dir = joinpath(tmpdir, "reasoning")
        output_dir = joinpath(tmpdir, "phase3a")
        mkpath(data_dir)

        ps_cpu = CPU_DEV(ps)
        JLD2.@save checkpoint_path ps_cpu config

        dataset_path = joinpath(data_dir, "gsm8k.jsonl")
        open(dataset_path, "w") do io
            write(io, "{\"source\":\"gsm8k\",\"reasoning_type\":\"arithmetic_chain\",\"question\":\"2+2?\",\"answer\":\"4\"}\n")
            write(io, "{\"source\":\"gsm8k\",\"reasoning_type\":\"arithmetic_chain\",\"question\":\"3+5?\",\"answer\":\"8\"}\n")
        end

        result = train_phase3a(
            ;
            checkpoint_path = checkpoint_path,
            data_dir = data_dir,
            output_dir = output_dir,
            batch_size = 1,
            num_epochs = 3,
            learning_rate = 1f-3,
            max_seq_length = 12,
            checkpoint_every = 1,
            max_per_dataset = 1,
            max_steps = 1,
            log_every = 1,
            seed = 23,
        )

        @test result.steps_run == 1
        @test result.num_examples == 1
        @test result.effective_max_seq_length == 12
        @test result.footprint.input_seq_len == 11
        @test !result.footprint.char_vocab_mismatch
        @test isfile(joinpath(output_dir, "best.jld2"))
        @test result.best_loss < Inf
    end
end

@testset "Phase 3a bounded resume run" begin
    rng = Random.MersenneTwister(29)
    config = ReasoningDrafterConfig(
        vocab_size = REASONING_CHAR_VOCAB_SIZE,
        max_sequence_length = 12,
        embedding_dimension = 16,
        number_of_heads = 2,
        number_of_layers = 1,
        time_dimension = 8,
        rc_code_dim = 8,
        rc_codebook_size = 16,
        rc_integration_steps = 2,
        frontend_wave_heads = 2,
        circuit_num_leaves = 4,
        circuit_product_arity = 2,
        circuit_num_sums = 2,
        circuit_num_circuits = 2,
    )

    model = ReasoningDrafter(config)
    ps = Lux.initialparameters(rng, model)

    mktempdir() do tmpdir
        checkpoint_path = joinpath(tmpdir, "phase2_legacy.jld2")
        data_dir = joinpath(tmpdir, "reasoning")
        output_dir = joinpath(tmpdir, "phase3a")
        mkpath(data_dir)

        ps_cpu = CPU_DEV(ps)
        JLD2.@save checkpoint_path ps_cpu config

        dataset_path = joinpath(data_dir, "gsm8k.jsonl")
        open(dataset_path, "w") do io
            write(io, "{\"source\":\"gsm8k\",\"reasoning_type\":\"arithmetic_chain\",\"question\":\"1+1?\",\"answer\":\"2\"}\n")
            write(io, "{\"source\":\"gsm8k\",\"reasoning_type\":\"arithmetic_chain\",\"question\":\"4+3?\",\"answer\":\"7\"}\n")
        end

        first_result = train_phase3a(
            ;
            checkpoint_path = checkpoint_path,
            data_dir = data_dir,
            output_dir = output_dir,
            batch_size = 1,
            num_epochs = 2,
            learning_rate = 1f-3,
            max_seq_length = 12,
            checkpoint_every = 1,
            max_per_dataset = 1,
            max_steps = 1,
            log_every = 1,
            seed = 29,
        )

        resume_checkpoint = joinpath(output_dir, "checkpoint_last.jld2")
        best_checkpoint = joinpath(output_dir, "best.jld2")
        @test isfile(resume_checkpoint)
        @test isfile(best_checkpoint)
        @test first_result.global_step == 1
        first_ckpt = JLD2.load(resume_checkpoint)
        first_best = JLD2.load(best_checkpoint)
        @test first_ckpt["best_loss"] == first_result.best_loss
        @test first_best["best_loss"] == first_result.best_loss

        resumed_result = train_phase3a(
            ;
            checkpoint_path = resume_checkpoint,
            data_dir = data_dir,
            output_dir = output_dir,
            batch_size = 1,
            num_epochs = 3,
            learning_rate = 1f-3,
            max_seq_length = 12,
            checkpoint_every = 1,
            max_per_dataset = 1,
            max_steps = 2,
            log_every = 1,
            seed = 29,
        )

        resumed_ckpt = JLD2.load(resume_checkpoint)
        resumed_best = JLD2.load(best_checkpoint)
        @test resumed_result.steps_run == 2
        @test resumed_result.global_step == 3
        @test resumed_ckpt["global_step"] == 3
        @test resumed_ckpt["epoch"] >= 2
        @test haskey(resumed_ckpt, "opt_state_cpu")
        @test resumed_ckpt["best_loss"] == resumed_result.best_loss
        @test resumed_best["best_loss"] == resumed_result.best_loss
        @test resumed_result.best_loss < Inf
    end
end

@testset "Legacy checkpoint compatibility guards" begin
    rng = Random.MersenneTwister(37)
    config = ReasoningDrafterConfig(
        vocab_size = REASONING_CHAR_VOCAB_SIZE,
        max_sequence_length = 12,
        embedding_dimension = 16,
        number_of_heads = 2,
        number_of_layers = 1,
        time_dimension = 8,
        rc_code_dim = 8,
        rc_codebook_size = 16,
        rc_integration_steps = 2,
        frontend_wave_heads = 2,
        circuit_num_leaves = 4,
        circuit_product_arity = 2,
        circuit_num_sums = 2,
        circuit_num_circuits = 2,
    )

    legacy_phase2_ps = (
        TokenEmbedding = (weight = zeros(Float32, config.vocab_size, config.embedding_dimension),),
        PositionEmbedding = (weight = zeros(Float32, config.max_sequence_length, config.embedding_dimension),),
        Blocks = (
            Block_1 = (
                Norm = (scale = zeros(Float32, config.embedding_dimension),),
                RuleWave = (Codebook = zeros(Float32, config.rc_code_dim, config.rc_codebook_size),),
                GluProjection = (weight = zeros(Float32, 2 * config.embedding_dimension, config.embedding_dimension), bias = zeros(Float32, 2 * config.embedding_dimension)),
                LinAttn = (QueryProjection = (weight = zeros(Float32, config.embedding_dimension, div(config.embedding_dimension, config.number_of_heads)), bias = zeros(Float32, config.embedding_dimension)),),
                WaveGate = (log_wave_speed = zeros(Float32, div(config.embedding_dimension, config.number_of_heads)), log_damping = zeros(Float32, div(config.embedding_dimension, config.number_of_heads))),
                ContentNorm = (scale = zeros(Float32, div(config.embedding_dimension, config.number_of_heads)),),
                GateNorm = (scale = zeros(Float32, div(config.embedding_dimension, config.number_of_heads)),),
                Circuit = (GateBias = zeros(Float32, config.embedding_dimension),),
                OutputNorm = (scale = zeros(Float32, config.embedding_dimension), bias = zeros(Float32, config.embedding_dimension)),
                CircuitLeafHeaderWeight = nothing,
                CircuitLeafHeaderBias = nothing,
                CircuitGateBiasShift = nothing,
            ),
        ),
        FinalNorm = (scale = zeros(Float32, config.embedding_dimension),),
        OutputHead = (weight = zeros(Float32, config.vocab_size, config.embedding_dimension),),
        TimeEmbedding = zeros(Float32, config.time_dimension),
    )

    legacy_phase1_ps = (
        Drafter = legacy_phase2_ps,
        MoveHead = (weight = zeros(Float32, 4, config.embedding_dimension), bias = zeros(Float32, 4)),
        EvalHead = (weight = zeros(Float32, 1, config.embedding_dimension), bias = zeros(Float32, 1)),
    )

    mktempdir() do tmpdir
        legacy_phase2_path = joinpath(tmpdir, "legacy_phase2.jld2")
        JLD2.@save legacy_phase2_path ps_cpu = legacy_phase2_ps config

        phase3a_err = try
            _load_phase3a_state(legacy_phase2_path, rng, 1f-3)
            nothing
        catch err
            err
        end
        @test phase3a_err isa ArgumentError
        @test occursin("legacy monolithic Phase 2 drafter layout", sprint(showerror, phase3a_err))

        legacy_phase1_path = joinpath(tmpdir, "legacy_phase1.jld2")
        JLD2.@save legacy_phase1_path ps_cpu = legacy_phase1_ps config

        surgery_err = try
            transfer_surgery(
                ;
                input_path = legacy_phase1_path,
                output_path = joinpath(tmpdir, "unused_surgery.jld2"),
                target_vocab = 49160,
                seed = 37,
            )
            nothing
        catch err
            err
        end
        @test surgery_err isa ArgumentError
        @test occursin("legacy monolithic ReasoningDrafter layout", sprint(showerror, surgery_err))
    end
end

end # module ReasoningTrainabilitySmoke
