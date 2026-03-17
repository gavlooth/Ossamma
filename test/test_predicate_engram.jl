using Swamma
using Swamma.PredicateEngramMod
using Random
using Lux
using Test

@testset "PredicateEngram" begin
    rng = Random.MersenneTwister(123)

    @testset "construction" begin
        pe = PredicateEngram(64; code_dim=32, codebook_size=256, num_roles=4)
        @test pe.embedding_dim == 64
        @test pe.code_dim == 32
        @test pe.codebook_size == 256
        @test pe.num_roles == 4
        @test pe.filler_dim == 16  # 64 ÷ 4

        # Mismatch should error
        @test_throws ArgumentError PredicateEngram(64; num_roles=3, filler_dim=10)
    end

    @testset "parameter shapes" begin
        pe = PredicateEngram(64; code_dim=32, codebook_size=128, num_roles=4, filler_dim=16)
        ps = Lux.initialparameters(rng, pe)

        @test size(ps.Codebook) == (32, 128)
        @test size(ps.EncoderWeight) == (32, 64)
        @test size(ps.RuleBank) == (16, 128)  # num_roles² × codebook_size
        @test size(ps.FillerWeight) == (64, 64)
        @test size(ps.OutputWeight) == (64, 64)
        @test size(ps.GateWeight) == (64, 64)
        @test all(ps.GateBias .< 0)  # starts near-closed
    end

    @testset "rule bank initialization" begin
        pe = PredicateEngram(64; codebook_size=128, num_roles=4)
        ps = Lux.initialparameters(rng, pe)

        # Rule matrices should be near-identity at init
        rule_0 = reshape(ps.RuleBank[:, 1], 4, 4)
        for i in 1:4
            @test abs(rule_0[i, i] - 1.0) < 0.1  # diagonal near 1
        end
    end

    @testset "forward pass - batched" begin
        pe = PredicateEngram(64; code_dim=32, codebook_size=128, num_roles=4)
        ps = Lux.initialparameters(rng, pe)
        st = Lux.initialstates(rng, pe)

        hidden = randn(rng, Float32, 64, 16, 2)  # (dim=64, seq=16, batch=2)
        out, st2 = pe(hidden, ps, st)
        @test size(out) == (64, 16, 2)
        @test eltype(out) == Float32
    end

    @testset "forward pass - unbatched" begin
        pe = PredicateEngram(64; code_dim=32, codebook_size=128, num_roles=4)
        ps = Lux.initialparameters(rng, pe)
        st = Lux.initialstates(rng, pe)

        hidden = randn(rng, Float32, 64, 16)  # (dim=64, seq=16)
        out, st2 = pe(hidden, ps, st)
        @test size(out) == (64, 16)
    end

    @testset "VQ quantization" begin
        codebook = Float32[1 0 -1; 0 1 -1]  # 3 codes of dim 2
        query = Float32[0.9 -0.8; 0.1 -0.9]  # 2 queries

        quantized, indices = vq_quantize(query, codebook)
        # Query 1 (0.9, 0.1) should be nearest to code 1 (1, 0)
        @test indices[1] == 1
        # Query 2 (-0.8, -0.9) should be nearest to code 3 (-1, -1)
        @test indices[2] == 3
    end

    @testset "output close to input at init (gate near-closed)" begin
        pe = PredicateEngram(64; code_dim=32, codebook_size=128, num_roles=4)
        ps = Lux.initialparameters(rng, pe)
        st = Lux.initialstates(rng, pe)

        hidden = randn(rng, Float32, 64, 8, 2)
        out, _ = pe(hidden, ps, st)

        # With gate bias at -2.0, sigmoid(-2) ≈ 0.12
        # Output should be close to input (small perturbation)
        relative_diff = sum(abs.(out .- hidden)) / sum(abs.(hidden))
        @test relative_diff < 0.5  # less than 50% relative change
    end

    @testset "EMA state tracking" begin
        pe = PredicateEngram(64; code_dim=32, codebook_size=128, num_roles=4)
        ps = Lux.initialparameters(rng, pe)
        st = Lux.initialstates(rng, pe)

        # EMA state should be initialized
        @test size(st.ema_cluster_size) == (128,)
        @test size(st.ema_embed_sum) == (32, 128)
        @test all(st.ema_cluster_size .== 1.0f0)  # initialized to ones

        # Forward pass should update EMA state
        hidden = randn(rng, Float32, 64, 8, 2)
        _, st2 = pe(hidden, ps, st)
        @test st2.ema_cluster_size !== st.ema_cluster_size  # state was updated
    end

    @testset "EMA codebook application" begin
        pe = PredicateEngram(64; code_dim=32, codebook_size=128, num_roles=4)
        ps = Lux.initialparameters(rng, pe)
        st = Lux.initialstates(rng, pe)

        # Run a forward pass to accumulate EMA stats
        hidden = randn(rng, Float32, 64, 16, 4)
        _, st2 = pe(hidden, ps, st)

        codebook_before = copy(ps.Codebook)
        apply_ema_codebook!(ps, st2, pe)
        # Codebook should have changed
        @test ps.Codebook != codebook_before
    end

    @testset "EMA disabled" begin
        pe = PredicateEngram(64; code_dim=32, codebook_size=128, num_roles=4, ema_decay=0.0f0)
        ps = Lux.initialparameters(rng, pe)
        st = Lux.initialstates(rng, pe)

        hidden = randn(rng, Float32, 64, 8, 2)
        _, st2 = pe(hidden, ps, st)
        # With ema_decay=0, state should pass through unchanged
        @test st2.ema_cluster_size == st.ema_cluster_size
    end
end

println("All PredicateEngram tests passed!")
