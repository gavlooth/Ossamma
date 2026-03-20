import Swamma
using Random
using Lux
using Test

@testset "WavePDELayer state contract" begin
    rng = Random.MersenneTwister(7)
    layer = Swamma.WavePDELayer(32, 32, 32, 0.1f0, 5.0f0, 0.05f0; integration_steps = 2)
    ps = Lux.initialparameters(rng, layer)
    st = Lux.initialstates(rng, layer)

    @test st.lambda_cache === nothing

    eval_st = Lux.testmode(st)
    train_st = Lux.trainmode(eval_st)

    @test !hasproperty(st, :training)
    @test eval_st == st
    @test train_st == st

    hidden = randn(rng, Float32, 32, 8, 2)
    out_eval, eval_st2 = layer(hidden, ps, eval_st)
    out_train, train_st2 = layer(hidden, ps, train_st)
    _, eval_st3 = layer(hidden, ps, eval_st2)

    @test size(out_eval) == size(hidden)
    @test out_eval == out_train
    @test eval_st2.lambda_cache !== nothing
    @test train_st2.lambda_cache !== nothing
    @test eval_st3.lambda_cache === eval_st2.lambda_cache
end

@testset "WavePDELayer supports low-precision spectral inputs" begin
    rng = Random.MersenneTwister(11)
    layer = Swamma.WavePDELayer(32, 32, 32, 0.1f0, 5.0f0, 0.05f0; integration_steps = 2)
    ps = Lux.initialparameters(rng, layer)
    st = Lux.initialstates(rng, layer)

    hidden = Core.BFloat16.(randn(rng, Float32, 32, 8, 2))
    out, st2 = layer(hidden, ps, st)

    @test size(out) == size(hidden)
    @test all(isfinite, Float32.(out))
    @test st2.lambda_cache !== nothing
end
