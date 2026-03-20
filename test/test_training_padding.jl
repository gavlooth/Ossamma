using Test
using Random
using Lux
using NNlib
using Statistics: mean
using Swamma

@testset "Training Padding Helpers" begin
    @testset "masked_cross_entropy_vectorized matches manual NLL" begin
        logits = Float32[
            2.0  0.5
            0.0  1.5
            -1.0 -0.5
        ]
        logits = reshape(logits, 3, 2, 1)
        targets = reshape(Int[1, 2], 2, 1)
        mask = reshape(Bool[true, true], 2, 1)

        loss = Swamma.Training.masked_cross_entropy_vectorized(logits, targets, mask)
        log_probs = NNlib.logsoftmax(logits, dims = 1)
        expected = -Float32(mean([
            log_probs[1, 1, 1],
            log_probs[2, 2, 1],
        ]))

        @test isapprox(loss, expected; atol = 1f-6)
    end

    @testset "diffusion_loss_with_padding masks all-pad batches" begin
        rng = Random.MersenneTwister(7)
        cfg = Swamma.LLaDAConfig(
            vocab_size = 64,
            max_sequence_length = 8,
            embedding_dimension = 32,
            number_of_heads = 2,
            number_of_layers = 2,
            time_dimension = 16,
            prime_subtoken_length = 4,
            prime_subtoken_base = 4,
        )
        model = Swamma.LLaDA.LLaDAModel(cfg)
        params, state = Lux.setup(rng, model)
        eval_state = Lux.testmode(state)

        pad_id = cfg.vocab_size
        batch = fill(pad_id, cfg.max_sequence_length, 2)
        loss, _ = Swamma.Training.diffusion_loss_with_padding(
            model,
            params,
            eval_state,
            batch,
            pad_id;
            rng = rng,
            mask_ratio = 1.0f0,
        )

        @test loss == 0.0f0
    end

    @testset "diffusion_loss_with_padding is finite on mixed batches" begin
        rng = Random.MersenneTwister(11)
        cfg = Swamma.LLaDAConfig(
            vocab_size = 96,
            max_sequence_length = 8,
            embedding_dimension = 32,
            number_of_heads = 2,
            number_of_layers = 2,
            time_dimension = 16,
            prime_subtoken_length = 4,
            prime_subtoken_base = 4,
        )
        model = Swamma.LLaDA.LLaDAModel(cfg)
        params, state = Lux.setup(rng, model)
        eval_state = Lux.testmode(state)

        pad_id = cfg.vocab_size
        batch = rand(rng, 1:cfg.vocab_size-1, cfg.max_sequence_length, 2)
        batch[end-1:end, 2] .= pad_id

        loss, _ = Swamma.Training.diffusion_loss_with_padding(
            model,
            params,
            eval_state,
            batch,
            pad_id;
            rng = rng,
            mask_ratio = 0.5f0,
        )

        @test isfinite(loss)
        @test loss >= 0.0f0
    end
end
