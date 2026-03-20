module TrainChessReasoningSmoke

using Test
using Random
using Lux
using Optimisers
using JLD2
using Swamma
using Swamma.ChessTokenizer
using Swamma.ReasoningDrafterMod

include(joinpath(dirname(@__DIR__), "scripts", "train_chess_reasoning.jl"))

@testset "Phase 1 checkpoint metadata sync" begin
    rng = Random.MersenneTwister(41)
    config = ReasoningDrafterConfig(
        vocab_size = PIECE_VOCAB_SIZE,
        max_sequence_length = NUM_SQUARES,
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

    _, ps, st = init_chess_model(rng, config)
    opt_state = Optimisers.setup(Optimisers.Adam(1f-3), ps)

    mktempdir() do tmpdir
        checkpoint_path = joinpath(tmpdir, "checkpoint_last.jld2")
        best_path = joinpath(tmpdir, "best.jld2")

        _save_phase1_checkpoint(checkpoint_path, ps, st, opt_state, config, 1, 1; best_loss = Inf)
        _save_phase1_checkpoint(best_path, ps, st, opt_state, config, 1, 1; best_loss = 3.25)
        _save_phase1_checkpoint(checkpoint_path, ps, st, opt_state, config, 1, 1; best_loss = 3.25)

        last_ckpt = JLD2.load(checkpoint_path)
        best_ckpt = JLD2.load(best_path)

        @test last_ckpt["best_loss"] == best_ckpt["best_loss"] == 3.25
        @test last_ckpt["global_step"] == best_ckpt["global_step"] == 1
        @test last_ckpt["epoch"] == best_ckpt["epoch"] == 1
        @test haskey(last_ckpt, "opt_state_cpu")
    end
end

@testset "Phase 1 config loading" begin
    cfg = load_phase1_config(joinpath(dirname(@__DIR__), "configs", "chess_phase1_260m.toml"))
    @test cfg["data_path"] == "data/chess/lichess_db_eval.jsonl"
    @test cfg["batch_size"] == 4
    @test cfg["gradient_accumulation_steps"] == 8
    @test cfg["mixed_precision"] == true
    @test cfg["precision"] == "bfloat16"
    @test cfg["embedding_dimension"] == 1024
    @test cfg["number_of_heads"] == 16
    @test cfg["number_of_layers"] == 24
    @test validate_phase1_options!(Dict(
        "embedding_dimension" => 1024,
        "number_of_heads" => 16,
        "batch_size" => 4,
        "gradient_accumulation_steps" => 8,
        "number_of_layers" => 24,
    ))["batch_size"] == 4
end

end # module TrainChessReasoningSmoke
