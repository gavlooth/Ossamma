using Swamma
using Swamma.ReasoningDrafterMod
using Swamma.RuleConditionedWavePDEMod
using Random
using Lux
using Test

@testset "RuleConditionedWavePDE" begin
    rng = Random.MersenneTwister(42)

    @testset "construction" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        @test rc.state_dimension == 64
        @test rc.code_dim == 32
        @test rc.codebook_size == 128
        @test length(rc.lambda) == 64
    end

    @testset "parameter shapes" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)

        @test size(ps.Codebook) == (32, 128)
        @test size(ps.EncoderWeight) == (32, 64)
        @test size(ps.RuleBank) == (32, 128)
        @test size(ps.SpeedModWeight) == (64, 32)
        @test size(ps.DampingModWeight) == (64, 32)
        @test size(ps.log_wave_speed) == (64,)
        @test size(ps.log_damping) == (64,)
        @test size(ps.GateWeight) == (64, 64)
        @test all(ps.GateBias .< 0)
    end

    @testset "forward pass - batched" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)
        st = Lux.initialstates(rng, rc)

        hidden = randn(rng, Float32, 64, 16, 2)
        out, st2 = rc(hidden, ps, st)
        @test size(out) == (64, 16, 2)
        @test eltype(out) == Float32
        @test all(isfinite, out)
    end

    @testset "forward pass - unbatched" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)
        st = Lux.initialstates(rng, rc)

        hidden = randn(rng, Float32, 64, 16)
        out, st2 = rc(hidden, ps, st)
        @test size(out) == (64, 16)
    end

    @testset "gate starts near-closed" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)
        st = Lux.initialstates(rng, rc)

        hidden = randn(rng, Float32, 64, 8, 2)
        out, _ = rc(hidden, ps, st)
        relative_diff = sum(abs.(out .- hidden)) / sum(abs.(hidden))
        @test relative_diff < 0.5
    end

    @testset "EMA state updates" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)
        st = Lux.initialstates(rng, rc)

        @test size(st.ema_cluster_size) == (128,)
        hidden = randn(rng, Float32, 64, 8, 2)
        _, st2 = rc(hidden, ps, st)
        @test st2.ema_cluster_size !== st.ema_cluster_size
    end

    @testset "different rules produce different dynamics" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)
        st = Lux.initialstates(rng, rc)

        # Two very different inputs should get different VQ codes → different dynamics
        h1 = ones(Float32, 64, 4, 1) .* 5f0
        h2 = ones(Float32, 64, 4, 1) .* -5f0
        out1, _ = rc(h1, ps, st)
        out2, _ = rc(h2, ps, st)
        # Outputs should differ (different rules modulate differently)
        @test out1 != out2
    end
end

@testset "ReasoningDrafter" begin
    rng = Random.MersenneTwister(99)

    config = ReasoningDrafterConfig(
        vocab_size = 500,
        max_sequence_length = 32,
        embedding_dimension = 64,
        number_of_heads = 2,
        number_of_layers = 2,
        time_dimension = 32,
        rc_code_dim = 32,
        rc_codebook_size = 64,
        circuit_num_leaves = 8,
        circuit_product_arity = 2,
        circuit_num_sums = 4,
        circuit_num_circuits = 2,
    )

    @testset "construction" begin
        model = ReasoningDrafter(config)
        @test length(model.Blocks) == 2
        @test model.config.vocab_size == 500
    end

    @testset "block has RuleWave not PredEngram" begin
        model = ReasoningDrafter(config)
        block = model.Blocks[1]
        @test block.RuleWave isa RuleConditionedWavePDE
        @test block.Circuit isa AlgebraicCircuitLayer
    end

    @testset "forward pass - batched" begin
        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        tokens = rand(rng, 1:config.vocab_size, 16, 2)
        logits, st2 = model(tokens, ps, st)
        @test size(logits) == (500, 16, 2)
        @test eltype(logits) == Float32
        @test all(isfinite, logits)
    end

    @testset "forward pass - unbatched" begin
        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        tokens = rand(rng, 1:config.vocab_size, 16)
        logits, st2 = model(tokens, ps, st)
        @test size(logits) == (500, 16)
        @test all(isfinite, logits)
    end

    @testset "draft generation" begin
        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        prompt = collect(1:8)
        draft_tokens, draft_logits = draft_reasoning_tokens(
            model, prompt, ps, st; num_tokens = 4
        )
        @test length(draft_tokens) == 12
        @test length(draft_logits) == 4
        @test all(1 .<= draft_tokens .<= config.vocab_size)
    end

    @testset "parameter estimate" begin
        est = estimate_drafter_parameters(config)
        @test est > 0

        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        function count_params(x)
            x isa AbstractArray && return length(x)
            x isa NamedTuple && return sum(count_params(v) for v in values(x))
            x isa Tuple && return sum(count_params(v) for v in x)
            x isa Nothing && return 0
            return 0
        end
        actual = count_params(ps)
        println("  Estimated params: $est")
        println("  Actual params:    $actual")
        @test 0.75 * actual <= est <= 1.25 * actual
    end
end

println("\nAll ReasoningDrafter tests passed!")
