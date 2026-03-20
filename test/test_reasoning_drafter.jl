using Swamma
using Swamma.ReasoningDrafterMod
using Swamma.RuleConditionedWavePDEMod
using Random
using Lux
using NNlib
using Zygote
using Test
using LinearAlgebra

@testset "RuleConditionedWavePDE" begin
    rng = Random.MersenneTwister(42)

    @testset "construction" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        @test rc.state_dimension == 64
        @test rc.code_dim == 32
        @test rc.codebook_size == 128
        @test length(rc.lambda) == 64
    end

    @testset "parameter shapes" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)

        @test size(ps.Codebook) == (32, 128)
        @test size(ps.EncoderWeight) == (32, 64)
        @test size(ps.RuleBank) == (32, 128)
        @test size(ps.SpeedModWeight) == (64, 32)
        @test size(ps.DampingModWeight) == (64, 32)
        @test size(ps.log_wave_speed) == (64,)
        @test size(ps.log_damping) == (64,)
        @test size(ps.GateWeight) == (64, 64)
        @test all(ps.GateBias .< 0)
    end

    @testset "forward pass - batched" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)
        st = Lux.initialstates(rng, rc)

        hidden = randn(rng, Float32, 64, 16, 2)
        out, st2 = rc(hidden, ps, st)
        @test size(out) == (64, 16, 2)
        @test eltype(out) == Float32
        @test all(isfinite, out)
    end

    @testset "forward pass - low precision input" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)
        st = Lux.initialstates(rng, rc)

        hidden = Core.BFloat16.(randn(rng, Float32, 64, 16, 2))
        out, st2 = rc(hidden, ps, st)
        @test size(out) == (64, 16, 2)
        @test all(isfinite, Float32.(out))
        @test st2.lambda_cache !== nothing
    end

    @testset "forward pass - unbatched" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)
        st = Lux.initialstates(rng, rc)

        hidden = randn(rng, Float32, 64, 16)
        out, st2 = rc(hidden, ps, st)
        @test size(out) == (64, 16)
    end

    @testset "gate starts near-closed" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)
        st = Lux.initialstates(rng, rc)

        hidden = randn(rng, Float32, 64, 8, 2)
        out, _ = rc(hidden, ps, st)
        relative_diff = sum(abs.(out .- hidden)) / sum(abs.(hidden))
        @test relative_diff < 0.5
    end

    @testset "EMA state updates" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)
        st = Lux.initialstates(rng, rc)

        @test size(st.ema_cluster_size) == (128,)
        hidden = randn(rng, Float32, 64, 8, 2)
        _, st2 = rc(hidden, ps, st)
        @test st2.ema_cluster_size !== st.ema_cluster_size
        @test sum(st2.ema_cluster_size) > 0
    end

    @testset "train/test mode state contract" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        st = Lux.initialstates(rng, rc)
        eval_st = Lux.testmode(st)
        train_st = Lux.trainmode(eval_st)

        @test st.training === Val(true)
        @test eval_st.training === Val(false)
        @test train_st.training === Val(true)
    end

    @testset "eval mode skips EMA and reuses lambda cache" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)
        st = Lux.testmode(Lux.initialstates(rng, rc))

        hidden = randn(rng, Float32, 64, 8, 2)
        _, st2 = rc(hidden, ps, st)
        _, st3 = rc(hidden, ps, st2)

        @test st2.ema_cluster_size == st.ema_cluster_size
        @test st2.lambda_cache !== nothing
        @test st3.lambda_cache === st2.lambda_cache
    end

    @testset "trainmode re-enables EMA updates" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)
        eval_st = Lux.testmode(Lux.initialstates(rng, rc))
        train_st = Lux.trainmode(eval_st)

        hidden = randn(rng, Float32, 64, 8, 2)
        _, st2 = rc(hidden, ps, train_st)

        @test st2.training === Val(true)
        @test st2.ema_cluster_size !== train_st.ema_cluster_size
    end

    @testset "EMA application mutates active codebook" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)
        st = Lux.initialstates(rng, rc)

        hidden = randn(rng, Float32, 64, 8, 2)
        _, st2 = rc(hidden, ps, st)
        codebook_before = copy(ps.Codebook)
        apply_rc_ema_codebook!(ps, st2, rc)

        @test codebook_before != ps.Codebook
    end

    @testset "different rules produce different dynamics" begin
        rc = RuleConditionedWavePDE(64; code_dim=32, codebook_size=128)
        ps = Lux.initialparameters(rng, rc)
        st = Lux.initialstates(rng, rc)

        # Two very different inputs should get different VQ codes → different dynamics
        h1 = ones(Float32, 64, 4, 1) .* 5f0
        h2 = ones(Float32, 64, 4, 1) .* -5f0
        out1, _ = rc(h1, ps, st)
        out2, _ = rc(h2, ps, st)
        # Outputs should differ (different rules modulate differently)
        @test out1 != out2
    end
end

@testset "RMSNorm" begin
    rng = Random.MersenneTwister(5)

    @testset "backward pass" begin
        layer = Swamma.RMSNorm(16)
        ps = Lux.initialparameters(rng, layer)
        st = Lux.initialstates(rng, layer)
        x = randn(rng, Float32, 16, 8, 1)

        loss, grads = Zygote.withgradient(ps) do p
            y, _ = layer(x, p, st)
            sum(abs2, y)
        end

        grad_ps = grads[1]
        @test isfinite(loss)
        @test grad_ps !== nothing
        @test grad_ps.scale !== nothing
        @test all(isfinite, grad_ps.scale)
    end
end

@testset "ReasoningDrafter" begin
    rng = Random.MersenneTwister(99)

    config = ReasoningDrafterConfig(
        vocab_size = 500,
        max_sequence_length = 32,
        embedding_dimension = 64,
        number_of_heads = 2,
        number_of_layers = 2,
        time_dimension = 32,
        rc_code_dim = 32,
        rc_codebook_size = 64,
        rc_integration_steps = 4,
        frontend_wave_heads = 4,
        circuit_num_leaves = 8,
        circuit_product_arity = 2,
        circuit_num_sums = 4,
        circuit_num_circuits = 2,
    )

    @testset "construction" begin
        model = ReasoningDrafter(config)
        @test length(model.Blocks) == 2
        @test model.config.vocab_size == 500
        @test model.FrontEnd isa Swamma.ReasoningDrafterMod.SharedOpcodeFrontend
        @test model.AuditTail isa Swamma.ReasoningDrafterMod.ReasoningAuditTail
        @test all(block -> block isa Swamma.ReasoningDrafterMod.ReasoningDrafterBlock, model.Blocks)
        @test all(block -> block.LinAttn isa Swamma.LinearAttention.LinearAttentionLayer, model.Blocks)
        @test all(block -> block.WaveGateLayer isa Swamma.WavePDE.WavePDELayer, model.Blocks)
    end

    @testset "adapter initialization preserves front-end and audit-tail identity" begin
        adapter_config = ReasoningDrafterConfig(
            vocab_size = config.vocab_size,
            max_sequence_length = config.max_sequence_length,
            embedding_dimension = config.embedding_dimension,
            number_of_heads = config.number_of_heads,
            number_of_layers = config.number_of_layers,
            time_dimension = config.time_dimension,
            rc_code_dim = config.rc_code_dim,
            rc_codebook_size = config.rc_codebook_size,
            rc_integration_steps = config.rc_integration_steps,
            circuit_num_leaves = config.circuit_num_leaves,
            circuit_product_arity = config.circuit_product_arity,
            circuit_num_sums = config.circuit_num_sums,
            circuit_num_circuits = config.circuit_num_circuits,
            use_adapters = true,
        )
        model = ReasoningDrafter(adapter_config)
        ps = Lux.initialparameters(rng, model)

        @test model.FrontEndHeader isa Swamma.ReasoningDrafterMod.ResidualAdapterHeader
        @test model.AuditTail.AuditInputHeader isa Swamma.ReasoningDrafterMod.ResidualAdapterHeader
        @test size(ps.FrontEnd.Codebook) == (adapter_config.rc_code_dim, adapter_config.rc_codebook_size)
        @test ps.FrontEnd.InputNorm isa NamedTuple
        @test ps.FrontEndHeader !== nothing
        @test ps.FrontEndHeader.InputNorm isa NamedTuple
        @test all(ps.FrontEndHeader.GateBias .== -2f0)
        @test ps.AuditTail.AuditInputHeader !== nothing
        @test ps.AuditTail.AuditInputHeader.InputNorm isa NamedTuple
        @test ps.AuditTail.CircuitLeafHeaderWeight == Matrix{Float32}(I, adapter_config.embedding_dimension, adapter_config.embedding_dimension)
        @test all(ps.AuditTail.CircuitLeafHeaderBias .== 0f0)
        @test all(ps.AuditTail.CircuitGateBiasShift .== 10f0)
        @test all(NNlib.sigmoid.(ps.AuditTail.CircuitGateBiasShift) .> 0.999f0)
    end

    @testset "front-end and proposer fields exist" begin
        model = ReasoningDrafter(config)
        @test model.FrontEnd isa Swamma.ReasoningDrafterMod.SharedOpcodeFrontend
        @test hasproperty(model.FrontEnd, :InputNorm)
        block = model.Blocks[1]
        @test block.InputNorm isa Swamma.RMSNorm
        @test block.GluProjection isa Lux.Dense
        @test block.LinAttn isa Swamma.LinearAttention.LinearAttentionLayer
        @test block.WaveGateLayer isa Swamma.WavePDE.WavePDELayer
        @test block.WaveGateNorm isa Swamma.RMSNorm
        @test block.FFN isa Swamma.SwiGLU
        @test block.OutputNorm isa Swamma.RMSNorm
        @test model.AuditTail.Circuit isa Swamma.AlgebraicCircuitLayer
    end

    @testset "forward pass - batched" begin
        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        tokens = rand(rng, 1:config.vocab_size, 16, 2)
        logits, st2 = model((token_ids = tokens, mask_ratio = Float32[0.25, 0.75]), ps, st)
        @test size(logits) == (500, 16, 2)
        @test eltype(logits) == Float32
        @test all(isfinite, logits)
    end

    @testset "forward pass - unbatched" begin
        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        tokens = rand(rng, 1:config.vocab_size, 16)
        logits, st2 = model((token_ids = tokens, mask_ratio = 0.5f0), ps, st)
        @test size(logits) == (500, 16)
        @test all(isfinite, logits)
    end

    @testset "mask ratio explicitly conditions logits" begin
        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.testmode(Lux.initialstates(rng, model))
        tokens = rand(rng, 1:config.vocab_size, 12, 2)

        logits_low, _ = model((token_ids = tokens, mask_ratio = Float32[0.1, 0.1]), ps, st)
        logits_high, _ = model((token_ids = tokens, mask_ratio = Float32[0.9, 0.9]), ps, st)

        @test logits_low != logits_high
    end

    @testset "overlength input throws" begin
        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        tokens = rand(rng, 1:config.vocab_size, config.max_sequence_length + 1, 2)
        @test_throws ArgumentError model((token_ids = tokens, mask_ratio = 0.5f0), ps, st)
    end

    @testset "draft generation" begin
        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        prompt = collect(1:8)
        draft_tokens, draft_logits = draft_reasoning_tokens(
            model, prompt, ps, st; num_tokens = 4
        )
        @test length(draft_tokens) == 12
        @test length(draft_logits) == 4
        @test all(1 .<= draft_tokens .<= config.vocab_size)
    end

    @testset "draft generation does not accumulate frontend EMA stats" begin
        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        prompt = collect(1:8)
        draft_reasoning_tokens(model, prompt, ps, st; num_tokens = 2)

        eval_st = Lux.testmode(st)
        tokens = reshape(prompt, :, 1)
        _, st2 = model((token_ids = tokens, mask_ratio = 0.0f0), ps, eval_st)
        @test st2.FrontEnd.ema_cluster_size == eval_st.FrontEnd.ema_cluster_size
    end

    @testset "front-end eval mode reuses lambda cache" begin
        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.testmode(Lux.initialstates(rng, model))
        tokens = rand(rng, 1:config.vocab_size, 8, 2)

        _, st2 = model((token_ids = tokens, mask_ratio = Float32[0.4, 0.6]), ps, st)
        _, st3 = model((token_ids = tokens, mask_ratio = Float32[0.4, 0.6]), ps, st2)

        @test st2.FrontEnd.lambda_cache !== nothing
        @test st3.FrontEnd.lambda_cache === st2.FrontEnd.lambda_cache
    end

    @testset "model train/test mode round-trip preserves frontend contract" begin
        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)
        eval_st = Lux.testmode(st)
        train_st = Lux.trainmode(eval_st)

        @test st.FrontEnd.training === Val(true)
        @test eval_st.FrontEnd.training === Val(false)
        @test train_st.FrontEnd.training === Val(true)

        tokens = rand(rng, 1:config.vocab_size, 8, 2)
        _, st2 = model((token_ids = tokens, mask_ratio = Float32[0.3, 0.8]), ps, train_st)
        @test st2.FrontEnd.ema_cluster_size !== train_st.FrontEnd.ema_cluster_size
    end

    @testset "draft generation respects prompt constraints" begin
        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        capped_prompt = collect(1:config.max_sequence_length)
        capped_tokens, capped_logits = draft_reasoning_tokens(
            model, capped_prompt, ps, st; num_tokens = 4
        )
        @test length(capped_tokens) == config.max_sequence_length
        @test isempty(capped_logits)

        @test_throws ArgumentError draft_reasoning_tokens(model, Int[], ps, st; num_tokens = 1)
        @test_throws ArgumentError draft_reasoning_tokens(
            model, collect(1:(config.max_sequence_length + 1)), ps, st; num_tokens = 1
        )
    end

    @testset "gradient pass reaches front-end proposer and audit params" begin
        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)
        tokens = rand(rng, 1:config.vocab_size, 8, 1)

        loss, grads = Zygote.withgradient(ps) do p
            logits = first(model((token_ids = tokens, mask_ratio = 0.5f0), p, st))
            sum(abs2, logits)
        end
        grad_ps = grads[1]

        @test isfinite(loss)
        @test grad_ps !== nothing
        @test grad_ps.FrontEnd.InputNorm.scale !== nothing
        @test grad_ps.FrontEnd.EncoderWeight !== nothing
        @test grad_ps.FrontEnd.MaskCodeWeight !== nothing
        @test grad_ps.FrontEnd.WaveReadoutWeight !== nothing
        @test grad_ps.FrontEnd.MaskReadoutWeight !== nothing
        @test grad_ps.FrontEnd.FusionWeight !== nothing
        @test grad_ps.FrontEnd.GateWeight !== nothing
        @test grad_ps.TimeEmbedding !== nothing
        @test grad_ps.Blocks.Block_1.InputNorm.scale !== nothing
        @test grad_ps.Blocks.Block_1.GluProjection.weight !== nothing
        @test grad_ps.Blocks.Block_1.LinAttn.QueryProjection.weight !== nothing
        @test grad_ps.Blocks.Block_1.LinAttn.OutputProjection.weight !== nothing
        @test grad_ps.Blocks.Block_1.WaveGateLayer.log_wave_speed !== nothing
        @test grad_ps.Blocks.Block_1.WaveGateNorm.scale !== nothing
        @test grad_ps.Blocks.Block_1.FFN.Expand.weight !== nothing
        @test grad_ps.Blocks.Block_1.FFN.Contract.weight !== nothing
        @test grad_ps.AuditTail.InputNorm.scale !== nothing
        @test grad_ps.AuditTail.FineEncoderWeight !== nothing
        @test grad_ps.AuditTail.RoleBaseWeight !== nothing
        @test grad_ps.AuditTail.RoleShiftWeight !== nothing
        @test grad_ps.AuditTail.PredicateOutputWeight !== nothing
        @test grad_ps.AuditTail.ScoreWeight !== nothing
        @test grad_ps.AuditTail.AgreementWeight !== nothing
        @test grad_ps.OutputHead.weight !== nothing
    end

    @testset "drafter EMA helper updates front-end codebook" begin
        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        st = Lux.initialstates(rng, model)

        tokens = rand(rng, 1:config.vocab_size, 16, 2)
        _, st2 = model(tokens, ps, st)
        codebook_before = copy(ps.FrontEnd.Codebook)
        apply_reasoning_drafter_ema_codebook!(ps, st2, model)

        @test codebook_before != ps.FrontEnd.Codebook
    end

    @testset "parameter estimate" begin
        est = estimate_drafter_parameters(config)
        @test est > 0

        model = ReasoningDrafter(config)
        ps = Lux.initialparameters(rng, model)
        function count_params(x)
            x isa AbstractArray && return length(x)
            x isa NamedTuple && return sum(count_params(v) for v in values(x))
            x isa Tuple && return sum(count_params(v) for v in x)
            x isa Nothing && return 0
            return 0
        end
        actual = count_params(ps)
        println("  Estimated params: $est")
        println("  Actual params:    $actual")
        @test 0.75 * actual <= est <= 1.25 * actual
    end
end

println("\nAll ReasoningDrafter tests passed!")
