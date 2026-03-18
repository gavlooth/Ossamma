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
        opt_state_cpu = CPU_DEV(opt_state)
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

end # module ReasoningTrainabilitySmoke
