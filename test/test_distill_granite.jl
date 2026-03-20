module DistillGraniteSmoke

using Test
using Random
using Lux
using Optimisers
using JLD2
using Swamma

include(joinpath(dirname(@__DIR__), "scripts", "distill_granite.jl"))

@testset "Phase 3b teacher backend probe" begin
    if USE_GPU
        probe_tokens = to_dev(reshape(Int[2, 3, 4], :, 1))
        gpu_ps = to_dev((weight = rand(Float32, 2, 2),))
        gpu_st = to_dev((cache = rand(Float32, 2),))

        failing_teacher = (tokens, ps, st) -> error("synthetic probe failure")
        cpu_ps, cpu_st, runtime_device = maybe_fallback_teacher_to_cpu(
            failing_teacher,
            gpu_ps,
            gpu_st,
            :gpu,
            probe_tokens,
        )

        @test runtime_device == :cpu
        @test !occursin("CuArray", string(typeof(cpu_ps.weight)))
        @test !occursin("CuArray", string(typeof(cpu_st.cache)))

        successful_teacher = (tokens, ps, st) -> (to_dev(zeros(Float32, 4, size(tokens, 1), size(tokens, 2))), st)
        kept_ps, kept_st, kept_device = maybe_fallback_teacher_to_cpu(
            successful_teacher,
            gpu_ps,
            gpu_st,
            :gpu,
            probe_tokens,
        )

        @test kept_device == :gpu
        @test occursin("CuArray", string(typeof(kept_ps.weight)))
        @test occursin("CuArray", string(typeof(kept_st.cache)))
    else
        probe_tokens = reshape(Int[2, 3, 4], :, 1)
        cpu_ps = (weight = rand(Float32, 2, 2),)
        cpu_st = (cache = rand(Float32, 2),)

        successful_teacher = (tokens, ps, st) -> (zeros(Float32, 4, size(tokens, 1), size(tokens, 2)), st)
        kept_ps, kept_st, kept_device = maybe_fallback_teacher_to_cpu(
            successful_teacher,
            cpu_ps,
            cpu_st,
            :cpu,
            probe_tokens,
        )

        @test kept_device == :cpu
        @test kept_ps == cpu_ps
        @test kept_st == cpu_st
    end
end

@testset "Phase 3b checkpoint metadata sync" begin
    rng = Random.MersenneTwister(43)
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
    st = Lux.initialstates(rng, model)
    opt_state = Optimisers.setup(Optimisers.Adam(1f-3), ps)

    mktempdir() do tmpdir
        checkpoint_path = joinpath(tmpdir, "checkpoint_last.jld2")
        best_path = joinpath(tmpdir, "best.jld2")

        _save_distill_checkpoint(checkpoint_path, ps, st, opt_state, config, 2, 1; best_loss = Inf)
        _save_distill_checkpoint(best_path, ps, st, opt_state, config, 2, 1; best_loss = 2.5)
        _save_distill_checkpoint(checkpoint_path, ps, st, opt_state, config, 2, 1; best_loss = 2.5)

        last_ckpt = JLD2.load(checkpoint_path)
        best_ckpt = JLD2.load(best_path)

        @test last_ckpt["best_loss"] == best_ckpt["best_loss"] == 2.5
        @test last_ckpt["global_step"] == best_ckpt["global_step"] == 2
        @test last_ckpt["epoch"] == best_ckpt["epoch"] == 1
        @test last_ckpt["training_stage"] == "phase3b_distill"
        @test best_ckpt["training_stage"] == "phase3b_distill"
        @test haskey(last_ckpt, "opt_state_cpu")
    end
end

end # module DistillGraniteSmoke
