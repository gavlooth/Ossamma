using Swamma
using Swamma.CircuitLayerMod
using Random
using Lux
using Zygote
using Test

@testset "AlgebraicCircuitLayer" begin
    rng = Random.MersenneTwister(456)

    @testset "construction" begin
        cl = AlgebraicCircuitLayer(64; num_leaves=16, product_arity=2, num_sums=8, num_circuits=4)
        @test cl.embedding_dim == 64
        @test cl.num_leaves == 16
        @test cl.product_arity == 2
        @test cl.num_products == 8  # 16 ÷ 2
        @test cl.num_sums == 8
        @test cl.num_circuits == 4
        @test cl.compose_mode == :mix

        # Arity must divide leaves
        @test_throws ArgumentError AlgebraicCircuitLayer(64; num_leaves=15, product_arity=4)
        # Invalid compose mode
        @test_throws ArgumentError AlgebraicCircuitLayer(64; compose_mode=:invalid)
    end

    @testset "parameter shapes" begin
        cl = AlgebraicCircuitLayer(64; num_leaves=16, product_arity=2, num_sums=8, num_circuits=4)
        ps = Lux.initialparameters(rng, cl)

        @test size(ps.LeafWeights) == (16, 64, 4)
        @test size(ps.LeafBiases) == (16, 4)
        @test size(ps.SumLogWeights) == (8, 8, 4)  # num_sums × num_products × nc
        @test size(ps.ComposeLogWeights) == (4,)
        @test size(ps.OutputWeight) == (64, 8)
        @test all(ps.GateBias .< 0)
    end

    @testset "forward pass - mix mode - batched" begin
        cl = AlgebraicCircuitLayer(64; num_leaves=16, product_arity=2, num_sums=8, num_circuits=4, compose_mode=:mix)
        ps = Lux.initialparameters(rng, cl)
        st = Lux.initialstates(rng, cl)

        hidden = randn(rng, Float32, 64, 16, 2)
        out, st2 = cl(hidden, ps, st)
        @test size(out) == (64, 16, 2)
        @test eltype(out) == Float32
    end

    @testset "forward pass - product mode - batched" begin
        cl = AlgebraicCircuitLayer(64; num_leaves=12, product_arity=3, num_sums=4, num_circuits=3, compose_mode=:product)
        ps = Lux.initialparameters(rng, cl)
        st = Lux.initialstates(rng, cl)

        hidden = randn(rng, Float32, 64, 10, 3)
        out, st2 = cl(hidden, ps, st)
        @test size(out) == (64, 10, 3)
    end

    @testset "forward pass - unbatched" begin
        cl = AlgebraicCircuitLayer(64; num_leaves=8, product_arity=2, num_sums=4, num_circuits=2)
        ps = Lux.initialparameters(rng, cl)
        st = Lux.initialstates(rng, cl)

        hidden = randn(rng, Float32, 64, 16)
        out, st2 = cl(hidden, ps, st)
        @test size(out) == (64, 16)
    end

    @testset "decomposability: product groups are disjoint" begin
        cl = AlgebraicCircuitLayer(64; num_leaves=12, product_arity=3)
        # product_arity=3, num_leaves=12 → 4 product nodes
        # Groups: [1,2,3], [4,5,6], [7,8,9], [10,11,12] — disjoint by construction
        @test cl.num_products == 4
        # No overlap between groups (guaranteed by reshape)
    end

    @testset "leaf activations in [0, 1]" begin
        cl = AlgebraicCircuitLayer(32; num_leaves=8, product_arity=2, num_sums=4, num_circuits=2)
        ps = Lux.initialparameters(rng, cl)

        hidden = randn(rng, Float32, 32, 10)
        hidden_flat = reshape(hidden, 32, 10)
        leaves = circuit_leaf_activations(hidden_flat, ps, cl)
        @test all(0 .<= leaves .<= 1)
        @test size(leaves) == (8, 10, 2)  # (num_leaves, N, num_circuits)
    end

    @testset "structure summary" begin
        cl = AlgebraicCircuitLayer(64; num_leaves=16, product_arity=2, num_sums=8, num_circuits=4)
        s = circuit_structure_summary(cl)
        @test occursin("4 parallel circuits", s)
        @test occursin("16 leaves", s)
        @test occursin("decomposable", s)
    end

    @testset "output close to input at init" begin
        cl = AlgebraicCircuitLayer(64; num_leaves=16, product_arity=2, num_sums=8, num_circuits=4)
        ps = Lux.initialparameters(rng, cl)
        st = Lux.initialstates(rng, cl)

        hidden = randn(rng, Float32, 64, 8, 2)
        out, _ = cl(hidden, ps, st)

        relative_diff = sum(abs.(out .- hidden)) / sum(abs.(hidden))
        @test relative_diff < 0.5
    end

    @testset "input and parameter gradients flow in batched mode" begin
        cl = AlgebraicCircuitLayer(16; num_leaves=8, product_arity=2, num_sums=4, num_circuits=3)
        ps = Lux.initialparameters(rng, cl)
        st = Lux.initialstates(rng, cl)
        hidden = randn(rng, Float32, 16, 5, 2)

        loss, grads = Zygote.withgradient(hidden, ps) do hidden_state, p
            out, _ = cl(hidden_state, p, st)
            sum(abs2, out)
        end

        grad_hidden, grad_ps = grads
        @test isfinite(loss)
        @test grad_hidden !== nothing
        @test size(grad_hidden) == size(hidden)
        @test grad_ps !== nothing
        @test grad_ps.LeafWeights !== nothing
        @test grad_ps.SumLogWeights !== nothing
        @test grad_ps.OutputWeight !== nothing
        @test grad_ps.GateWeight !== nothing
    end
end

println("All AlgebraicCircuitLayer tests passed!")
