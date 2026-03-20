using Swamma
using Swamma.LLaDA
using Swamma.EngramMod
using Random
using Lux
using Test

@testset "Engram Conditional Memory" begin
    rng = Random.MersenneTwister(42)

    @testset "EngramModule standalone" begin
        engram = EngramModule(64; num_heads=4, ngram_orders=[2,3], table_size=1024, head_dim=16)
        ps = Lux.initialparameters(rng, engram)
        st = Lux.initialstates(rng, engram)

        # Test total_tables
        @test engram.total_tables == 4 * 2  # 4 heads × 2 orders

        # Batched input
        token_ids = rand(rng, 1:100, 16, 2)  # (seq=16, batch=2)
        hidden = randn(rng, Float32, 64, 16, 2)
        out, st2 = engram((token_ids, hidden), ps, st)
        @test size(out) == size(hidden)
        @test eltype(out) == Float32

        # Unbatched input
        token_ids_1d = rand(rng, 1:100, 16)
        hidden_2d = randn(rng, Float32, 64, 16)
        out_2d, _ = engram((token_ids_1d, hidden_2d), ps, st)
        @test size(out_2d) == size(hidden_2d)

        # Gate bias initialized negative → engram injection starts small
        @test all(ps.GateBias .< 0)

        # Collision rate should be reasonable for diverse tokens
        collision_rate = EngramMod.engram_collision_rate(engram, token_ids, st.hash_multipliers)
        @test 0.0 <= collision_rate <= 1.0
        @test collision_rate < 0.5  # should not be mostly collisions

        precomputed_indices = EngramMod.prepare_engram_indices(
            engram,
            token_ids,
            st.hash_multipliers;
            reference_device = ps.EmbeddingTable,
        )
        out_cached, _ = engram((token_ids, hidden, precomputed_indices), ps, st)
        @test out_cached == out
    end

    @testset "EngramModule cache state is mode-invariant" begin
        engram = EngramModule(64; num_heads=4, ngram_orders=[2, 3], table_size=1024, head_dim=16)
        ps = Lux.initialparameters(rng, engram)
        st = Lux.initialstates(rng, engram)
        eval_st = Lux.testmode(st)
        train_st = Lux.trainmode(eval_st)

        @test !hasproperty(st, :training)
        @test eval_st == st
        @test train_st == st

        token_ids = rand(rng, 1:100, 16, 2)
        hidden = randn(rng, Float32, 64, 16, 2)
        out_eval, eval_st2 = engram((token_ids, hidden), ps, eval_st)
        out_train, train_st2 = engram((token_ids, hidden), ps, train_st)

        @test out_eval == out_train
        @test eval_st2.hash_multipliers == eval_st.hash_multipliers
        @test train_st2.hash_multipliers == train_st.hash_multipliers
    end

    @testset "subtokens_to_token_ids" begin
        subtokens = zeros(Int, 4, 3, 2)  # (subtoken_len=4, seq=3, batch=2)
        subtokens[:, 1, 1] = [1, 2, 3, 4]
        base = 10
        ids = EngramMod.subtokens_to_token_ids(subtokens, base)
        @test size(ids) == (3, 2)
        @test ids[1, 1] == 1 + 2*10 + 3*100 + 4*1000  # 4321
    end

    @testset "LLaDA with engram disabled (backward compat)" begin
        config = LLaDAConfig(
            vocab_size=1000, max_sequence_length=32, embedding_dimension=64,
            number_of_heads=2, number_of_layers=4, time_dimension=32,
            use_engram=false,
        )
        model = LLaDAModel(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        @test model.use_engram == false
        @test model.EngramModules === nothing
        @test all(model.engram_layer_map .== 0)

        subtoken_state = rand(rng, 1:config.prime_subtoken_base, model.prime_subtoken_length, 32, 2)
        inputs = (subtoken_state=subtoken_state, mask_ratio=0.5f0)
        logits, st_out = model(inputs, ps, st)
        @test size(logits) == (1000, 32, 2)
    end

    @testset "LLaDA with engram enabled (auto layers)" begin
        config = LLaDAConfig(
            vocab_size=1000, max_sequence_length=32, embedding_dimension=64,
            number_of_heads=2, number_of_layers=6, time_dimension=32,
            use_engram=true, engram_layers=Int[],
            engram_num_heads=4, engram_ngram_orders=[2, 3],
            engram_table_size=1024, engram_head_dim=16,
        )
        model = LLaDAModel(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        @test model.use_engram == true
        @test model.EngramModules !== nothing
        @test length(model.EngramModules) == 2  # auto: layer 2 + layer 3
        @test model.engram_layer_map[2] > 0
        @test model.engram_layer_map[3] > 0

        subtoken_state = rand(rng, 1:config.prime_subtoken_base, model.prime_subtoken_length, 32, 2)
        inputs = (subtoken_state=subtoken_state, mask_ratio=0.5f0)
        logits, st_out = model(inputs, ps, st)
        @test size(logits) == (1000, 32, 2)
        println("  Engram layer map: ", model.engram_layer_map)
    end

    @testset "LLaDA with engram enabled (explicit layers)" begin
        config = LLaDAConfig(
            vocab_size=1000, max_sequence_length=32, embedding_dimension=64,
            number_of_heads=2, number_of_layers=4, time_dimension=32,
            use_engram=true, engram_layers=[1, 3],
            engram_num_heads=4, engram_ngram_orders=[2],
            engram_table_size=512, engram_head_dim=16,
        )
        model = LLaDAModel(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        @test model.engram_layer_map == [1, 0, 2, 0]
        @test length(model.EngramModules) == 2

        subtoken_state = rand(rng, 1:config.prime_subtoken_base, model.prime_subtoken_length, 32, 1)
        inputs = (subtoken_state=subtoken_state, mask_ratio=0.3f0)
        logits, st_out = model(inputs, ps, st)
        @test size(logits) == (1000, 32, 1)
    end

    @testset "LLaDA PRIME runtime precompute matches implicit path" begin
        config = LLaDAConfig(
            vocab_size=1000, max_sequence_length=32, embedding_dimension=64,
            number_of_heads=2, number_of_layers=4, time_dimension=32,
            use_engram=true, engram_layers=[2],
            engram_num_heads=4, engram_ngram_orders=[2, 3],
            engram_table_size=512, engram_head_dim=16,
        )
        model = LLaDAModel(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        subtoken_state = rand(rng, 1:config.prime_subtoken_base, model.prime_subtoken_length, 32, 2)
        implicit_inputs = (subtoken_state = subtoken_state, mask_ratio = 0.4f0)
        runtime_inputs = LLaDA.prepare_prime_forward_inputs(model, subtoken_state)
        engram_runtime = Swamma.LLaDA.prepare_engram_forward_inputs(
            model,
            runtime_inputs.token_ids_for_engram,
            st;
            reference_device = subtoken_state,
        )
        explicit_inputs = (
            subtoken_state = subtoken_state,
            mask_ratio = 0.4f0,
            prime_compatibility_mask = runtime_inputs.prime_compatibility_mask,
            token_ids_for_engram = runtime_inputs.token_ids_for_engram,
            engram_indices = engram_runtime,
        )

        logits_implicit, _ = model(implicit_inputs, ps, st)
        logits_explicit, _ = model(explicit_inputs, ps, st)

        @test logits_explicit == logits_implicit
    end

    @testset "config_from_dict with engram" begin
        data = Dict(
            "model" => Dict(
                "vocab_size" => 500,
                "max_sequence_length" => 16,
                "embedding_dimension" => 64,
                "number_of_heads" => 2,
                "number_of_layers" => 4,
            ),
            "engram" => Dict(
                "use_engram" => true,
                "engram_layers" => [2, 4],
                "engram_num_heads" => 4,
                "engram_table_size" => 512,
            ),
        )
        config = config_from_dict(data)
        @test config.use_engram == true
        @test config.engram_layers == [2, 4]
        @test config.engram_num_heads == 4
        @test config.engram_table_size == 512
    end
end

println("\nAll engram tests passed!")
