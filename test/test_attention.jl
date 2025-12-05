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
    # New implementation should have child states + window_mask
    @test haskey(st, :QueryProjection)
    @test haskey(st, :window_mask)
    println("Check 1: State Composition - PASSED")

    # Check 2: Sequence Length Mismatch (Dynamic Masking)
    # We initialized with T=10.
    # Let's try input with T=5.
    
    x_short = randn(Float32, D, 5, 1) # (Features, Time, Batch)
    
    println("\nTesting Input Length T=5 (Model initialized with T=10)...")
    
    y_short, st_short = model(x_short, ps, st)
    @test size(y_short) == (D, 5, 1)
    # Check that the mask in the new state is 5x5
    @test size(st_short.window_mask) == (5, 5)
    
    println("Check 2: Dynamic Length T=5 - PASSED")

    # Check 3: Input Length T=10 (Should reuse or recreate mask)
    x_correct = randn(Float32, D, 10, 1)
    println("\nTesting Input Length T=10...")
    
    y_long, st_long = model(x_correct, ps, st)
    @test size(y_long) == (D, 10, 1)
    @test size(st_long.window_mask) == (10, 10)

    println("Check 3: Correct Length T=10 - PASSED")
    
    # Check 4: Batched vs Unbatched
    x_unbatched = randn(Float32, D, 8)
    y_unb, st_unb = model(x_unbatched, ps, st)
    @test size(y_unb) == (D, 8)
    @test size(st_unb.window_mask) == (8, 8)
    println("Check 4: Unbatched Input - PASSED")

end
