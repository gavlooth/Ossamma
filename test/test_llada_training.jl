using Test
using Random
using Lux
using Optimisers
using Swamma

@testset "LLaDA Training Smoke" begin
    rng = Random.MersenneTwister(123)
    cfg = Swamma.small_config()
    model = Swamma.LLaDA.LLaDAModel(cfg)

    # Small synthetic batch in the expected token-id range.
    batch = rand(rng, 1:cfg.vocab_size, cfg.max_sequence_length, 2)

    @testset "Diffusion Loss Is Finite" begin
        params, state = Lux.setup(rng, model)
        eval_state = Lux.testmode(state)
        loss, _ = Swamma.Training.diffusion_loss(
            model, params, eval_state, batch;
            rng = rng, schedule = :cosine
        )
        @test isfinite(loss)
        @test loss > 0
    end

    @testset "Single Train Step Advances State" begin
        optimizer = Optimisers.Adam(1f-3)
        train_state = Swamma.Training.create_train_state(model, optimizer; rng = rng)

        loss = Swamma.Training.train_step!(
            train_state, model, batch;
            rng = rng, schedule = :cosine, gradient_clip = 1.0f0
        )

        @test isfinite(loss)
        @test train_state.step == 1
    end
end

@testset "LLaDA PRIME Subtoken Smoke" begin
    rng = Random.MersenneTwister(321)
    cfg = Swamma.LLaDAConfig(
        vocab_size = 512,
        max_sequence_length = 32,
        embedding_dimension = 64,
        number_of_heads = 2,
        number_of_layers = 2,
        time_dimension = 32,
        state_dimension = 64,
        window_size = 4,
        mask_schedule = :cosine,
        prime_subtoken_length = 4,
        prime_subtoken_base = 8,
    )
    model = Swamma.LLaDA.LLaDAModel(cfg)
    batch = rand(rng, 1:cfg.vocab_size, cfg.max_sequence_length, 2)

    @test size(model.prime_code_table) == (cfg.prime_subtoken_length, cfg.vocab_size)
    @test model.prime_subtoken_base^model.prime_subtoken_length >= cfg.vocab_size

    params, state = Lux.setup(rng, model)
    eval_state = Lux.testmode(state)
    loss, _ = Swamma.Training.diffusion_loss(
        model, params, eval_state, batch;
        rng = rng, schedule = :cosine
    )
    @test isfinite(loss)
    @test loss > 0

    optimizer = Optimisers.Adam(1f-3)
    train_state = Swamma.Training.create_train_state(model, optimizer; rng = rng)
    step_loss = Swamma.Training.train_step!(
        train_state, model, batch;
        rng = rng, schedule = :cosine, gradient_clip = 1.0f0
    )
    @test isfinite(step_loss)
    @test train_state.step == 1
end
