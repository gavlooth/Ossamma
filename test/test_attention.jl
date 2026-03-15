using Lux
using Random
using Test

include("../src/Attention.jl")
using .Attention

@testset "SWAttention Soundness & Dynamic Tests" begin
    T_init = 10
    D = 16
    Heads = 4
    Window = 2
    
    model = Attention.SWAttention(T_init, D, Heads; window_size=Window)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    println("Initial State keys: ", keys(st))

    # Check 1: State Composition
    # True banded attention only needs the offset schedule, not a quadratic mask.
    @test haskey(st, :QueryProjection)
    @test haskey(st, :window_offsets)
    @test st.window_offsets == collect(-Window:Window)
    println("Check 1: State Composition - PASSED")

    # Check 2: Sequence Length Mismatch (Dynamic Masking)
    # We initialized with T=10.
    # Let's try input with T=5.
    
    x_short = randn(Float32, D, 5, 1) # (Features, Time, Batch)
    
    println("\nTesting Input Length T=5 (Model initialized with T=10)...")
    
    y_short, st_short = model(x_short, ps, st)
    @test size(y_short) == (D, 5, 1)
    @test st_short.window_offsets == collect(-Window:Window)
    
    println("Check 2: Dynamic Length T=5 - PASSED")

    # Check 3: Input Length T=10 (Should reuse or recreate mask)
    x_correct = randn(Float32, D, 10, 1)
    println("\nTesting Input Length T=10...")
    
    y_long, st_long = model(x_correct, ps, st)
    @test size(y_long) == (D, 10, 1)
    @test st_long.window_offsets == collect(-Window:Window)

    println("Check 3: Correct Length T=10 - PASSED")
    
    # Check 4: Batched vs Unbatched
    x_unbatched = randn(Float32, D, 8)
    y_unb, st_unb = model(x_unbatched, ps, st)
    @test size(y_unb) == (D, 8)
    @test st_unb.window_offsets == collect(-Window:Window)
    println("Check 4: Unbatched Input - PASSED")

end

@testset "SWAttention Locality" begin
    D = 12
    T = 8
    Heads = 3
    Window = 1

    model = Attention.SWAttention(T, D, Heads; window_size = Window)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    x_base = randn(Float32, D, T, 1)
    x_perturbed = copy(x_base)
    x_perturbed[:, end, 1] .+= 50.0f0

    y_base, _ = model(x_base, ps, st)
    y_perturbed, _ = model(x_perturbed, ps, st)

    @test y_base[:, 1:6, 1] ≈ y_perturbed[:, 1:6, 1]
    @test !(y_base[:, end - 1, 1] ≈ y_perturbed[:, end - 1, 1])
end
