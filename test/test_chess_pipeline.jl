using Swamma
using Swamma.ChessTokenizer
using Swamma.ChessDataset
using Swamma.ReasoningDrafterMod
using Random
using Lux
using Test

@testset "ChessTokenizer" begin
    @testset "FEN parsing" begin
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board, stm, cast, ep, hm, fm = parse_fen(fen)
        @test length(board) == 64
        @test stm == 0  # white
        @test cast == [1, 1, 1, 1]
        @test ep == 0
        @test board[1] == 4   # wR at a1
        @test board[5] == 6   # wK at e1
        @test board[57] == 10 # bR at a8
        @test board[32] == 0  # empty at h4
    end

    @testset "token encoding (1-based)" begin
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        tokens = fen_to_tokens(fen)
        @test length(tokens) == 64
        @test all(1 .<= tokens .<= PIECE_VOCAB_SIZE)
        @test tokens[1] == 5  # wR + 1
        @test tokens[32] == 1 # empty + 1
    end

    @testset "FEN roundtrip" begin
        fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "8/8/8/8/8/8/8/4K2k w - - 0 1",
        ]
        for fen in fens
            board = fen_to_board(split(fen)[1])
            back = board_to_fen(board)
            @test back == split(fen)[1]
        end
    end

    @testset "move encoding" begin
        moves = ["e2e4", "g1f3", "a7a8q", "b2b1n", "e1g1"]
        for uci in moves
            id = uci_to_move_id(uci)
            @test 1 <= id <= MOVE_VOCAB_SIZE
            @test move_id_to_uci(id) == uci
        end
    end

    @testset "legal move generation" begin
        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        start_ids = legal_move_ids(start_fen)
        @test length(start_ids) == 20
        @test uci_to_move_id("e2e4") in start_ids
        @test uci_to_move_id("g1f3") in start_ids

        castle_fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        castle_ids = legal_move_ids(castle_fen)
        @test uci_to_move_id("e1g1") in castle_ids
        @test uci_to_move_id("e1c1") in castle_ids

        ep_fen = "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1"
        ep_ids = legal_move_ids(ep_fen)
        @test uci_to_move_id("e5d6") in ep_ids

        mask = legal_move_mask(start_fen)
        @test size(mask) == (MOVE_VOCAB_SIZE,)
        @test mask[uci_to_move_id("e2e4")] == 1.0f0
        @test mask[uci_to_move_id("e1e2")] == 0.0f0
    end

    @testset "metadata" begin
        meta = fen_to_metadata("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        @test meta[1] == 1.0f0  # black to move
        @test meta[6] == 5.0f0  # en passant file e = 5
    end

    @testset "encode/decode position" begin
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        tokens, meta = encode_position(fen)
        @test length(tokens) == 64
        @test length(meta) == 6
    end
end

@testset "ChessDataset" begin
    @testset "parse JSONL line" begin
        line = """{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","evals":[{"pvs":[{"cp":15,"line":"e2e4 e7e5"}],"knodes":1234,"depth":20}]}"""
        pos = ChessDataset.parse_lichess_eval_line(line)
        @test pos !== nothing
        @test pos.best_move == "e2e4"
        @test pos.eval_cp == 15.0f0
        @test pos.depth == 20
        @test length(pos.tokens) == 64
    end

    @testset "malformed lines return nothing" begin
        @test ChessDataset.parse_lichess_eval_line("") === nothing
        @test ChessDataset.parse_lichess_eval_line("not json") === nothing
        @test ChessDataset.parse_lichess_eval_line("""{"fen":"bad"}""") === nothing
    end

    @testset "batch preparation" begin
        line = """{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","evals":[{"pvs":[{"cp":15,"line":"e2e4 e7e5"}],"knodes":1234,"depth":20}]}"""
        pos = ChessDataset.parse_lichess_eval_line(line)
        batch = prepare_batch([pos, pos, pos])
        @test size(batch.board_tokens) == (64, 3)
        @test size(batch.metadata) == (6, 3)
        @test length(batch.best_move_ids) == 3
        @test length(batch.eval_scores) == 3
        @test size(batch.legal_move_mask) == (MOVE_VOCAB_SIZE, 3)
        @test length(batch.legal_move_counts) == 3
        @test length(batch.target_legal_flags) == 3
        @test all(batch.target_legal_flags)
        @test all(batch.legal_move_counts .== 20)
        for i in 1:3
            @test batch.legal_move_mask[pos.best_move_id, i] == 1.0f0
        end
    end

    @testset "eval normalization" begin
        @test normalize_eval(0) ≈ 0.0f0 atol=1f-6
        @test normalize_eval(400) > 0.7f0
        @test normalize_eval(-400) < -0.7f0
        @test abs(normalize_eval(10000)) > 0.99f0  # near mate
    end

    @testset "real sample targets stay legal" begin
        sample_positions = Swamma.ChessDataset.ChessPosition[]
        open(joinpath(@__DIR__, "..", "data", "chess", "sample_100k.jsonl"), "r") do io
            for line in eachline(io)
                pos = ChessDataset.parse_lichess_eval_line(line)
                pos === nothing && continue
                push!(sample_positions, pos)
                length(sample_positions) >= 5 && break
            end
        end
        @test length(sample_positions) == 5
        batch = prepare_batch(sample_positions)
        @test all(batch.target_legal_flags)
        @test all(batch.legal_move_counts .> 0)
    end
end

@testset "Chess → ReasoningDrafter integration" begin
    rng = Random.MersenneTwister(77)

    config = ReasoningDrafterConfig(
        vocab_size = PIECE_VOCAB_SIZE,    # 13
        max_sequence_length = NUM_SQUARES, # 64
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

    model = ReasoningDrafter(config)
    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)

    @testset "forward on chess tokens" begin
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        tokens = fen_to_tokens(fen)
        # Batch of 2 identical positions
        batch_tokens = hcat(tokens, tokens)  # (64, 2)
        logits, st2 = model(batch_tokens, ps, st)
        @test size(logits) == (PIECE_VOCAB_SIZE, 64, 2)
        @test all(isfinite, logits)
    end

    @testset "different positions produce different logits" begin
        fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        fen2 = "8/8/8/8/8/8/8/4K2k w - - 0 1"
        t1 = fen_to_tokens(fen1)
        t2 = fen_to_tokens(fen2)
        batch = hcat(t1, t2)
        logits, _ = model(batch, ps, st)
        @test logits[:, :, 1] != logits[:, :, 2]
    end
end

println("\nAll chess pipeline tests passed!")
