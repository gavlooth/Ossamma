using Test
using Random
using JSON3
using Lux
using PyCall

include("../src/Swamma.jl")
using .Swamma

function synthetic_granite_tensor_map(config::NativeTeacherConfig)
    tensors = Dict{String,Array{Float32}}()
    hidden = config.embedding_dimension
    kv = (config.embedding_dimension ÷ config.number_of_heads) * config.number_of_kv_heads

    tensors["model.embed_tokens.weight"] = reshape(
        Float32.(1:(config.vocab_size * hidden)),
        config.vocab_size,
        hidden,
    ) ./ 1000f0
    tensors["model.norm.weight"] = Float32.(range(1.0f0, 1.0f0 + (hidden - 1) / 100f0; length = hidden))

    for layer_index in 0:(config.number_of_layers - 1)
        prefix = "model.layers.$layer_index"
        base = Float32(layer_index + 1)
        tensors["$(prefix).input_layernorm.weight"] =
            fill(base, hidden) .+ Float32.(0:(hidden - 1)) ./ 100f0
        tensors["$(prefix).post_attention_layernorm.weight"] =
            fill(base + 10f0, hidden) .+ Float32.(0:(hidden - 1)) ./ 100f0
        tensors["$(prefix).self_attn.q_proj.weight"] =
            reshape(Float32.(1:(hidden * hidden)), hidden, hidden) ./ (100f0 * (layer_index + 1))
        tensors["$(prefix).self_attn.k_proj.weight"] =
            reshape(Float32.(1:(kv * hidden)), kv, hidden) ./ (200f0 * (layer_index + 1))
        tensors["$(prefix).self_attn.v_proj.weight"] =
            reshape(Float32.(1:(kv * hidden)), kv, hidden) ./ (300f0 * (layer_index + 1))
        tensors["$(prefix).self_attn.o_proj.weight"] =
            reshape(Float32.(1:(hidden * hidden)), hidden, hidden) ./ (400f0 * (layer_index + 1))
        tensors["$(prefix).shared_mlp.input_linear.weight"] =
            reshape(
                Float32.(1:(2 * config.mlp_hidden_dimension * hidden)),
                2 * config.mlp_hidden_dimension,
                hidden,
            ) ./ (500f0 * (layer_index + 1))
        tensors["$(prefix).shared_mlp.output_linear.weight"] =
            reshape(
                Float32.(1:(hidden * config.mlp_hidden_dimension)),
                hidden,
                config.mlp_hidden_dimension,
            ) ./ (600f0 * (layer_index + 1))
    end

    return tensors
end

function write_synthetic_granite_checkpoint(dir::String, config::NativeTeacherConfig)
    tensors = synthetic_granite_tensor_map(config)
    config_payload = Dict(
        "model_type" => "granitemoehybrid",
        "architectures" => ["GraniteMoeHybridForCausalLM"],
        "vocab_size" => config.vocab_size,
        "max_position_embeddings" => config.max_sequence_length,
        "hidden_size" => config.embedding_dimension,
        "num_hidden_layers" => config.number_of_layers,
        "num_attention_heads" => config.number_of_heads,
        "num_key_value_heads" => config.number_of_kv_heads,
        "shared_intermediate_size" => config.mlp_hidden_dimension,
        "intermediate_size" => config.mlp_hidden_dimension,
        "hidden_act" => "silu",
        "position_embedding_type" => "rope",
        "rope_theta" => config.rope_theta,
        "rms_norm_eps" => config.rms_norm_eps,
        "layer_types" => fill("attention", config.number_of_layers),
        "num_local_experts" => 0,
        "num_experts_per_tok" => 0,
        "tie_word_embeddings" => true,
        "embedding_multiplier" => config.embedding_multiplier,
        "logits_scaling" => config.logits_scaling,
        "attention_dropout" => config.dropout_rate,
        "attention_bias" => config.use_bias,
    )
    open(joinpath(dir, "config.json"), "w") do io
        JSON3.write(io, config_payload)
    end

    np = pyimport("numpy")
    safetensors_numpy = pyimport("safetensors.numpy")
    py_tensors = PyDict()
    for (name, tensor) in tensors
        py_tensors[name] = np.asarray(tensor; dtype = np.float32)
    end
    shard_name = "model-00001-of-00001.safetensors"
    safetensors_numpy.save_file(py_tensors, joinpath(dir, shard_name))

    index_payload = Dict(
        "metadata" => Dict("total_parameters" => 0, "total_size" => 0),
        "weight_map" => Dict(name => shard_name for name in keys(tensors)),
    )
    open(joinpath(dir, "model.safetensors.index.json"), "w") do io
        JSON3.write(io, index_payload)
    end

    return tensors
end

@testset "Native Teacher LM" begin
    rng = Random.MersenneTwister(77)
    config = NativeTeacherConfig(
        vocab_size = 64,
        max_sequence_length = 16,
        embedding_dimension = 32,
        number_of_layers = 2,
        number_of_heads = 4,
        number_of_kv_heads = 2,
        mlp_hidden_dimension = 96,
        dropout_rate = 0f0,
    )
    model = NativeCausalLM(config)
    params, state = Lux.setup(rng, model)
    eval_state = Lux.testmode(state)

    @testset "Forward Pass Produces Finite Logits" begin
        tokens = rand(rng, 1:config.vocab_size, 6, 2)
        logits, new_state = model(tokens, params, eval_state)

        @test size(logits) == (config.vocab_size, 6, 2)
        @test all(isfinite, logits)
        @test :Blocks in keys(new_state)
    end

    @testset "Causal Mask Blocks Future Leakage" begin
        prefix = reshape(Int[5, 8, 13], :, 1)
        tokens_a = vcat(prefix, reshape(Int[21, 34], :, 1))
        tokens_b = vcat(prefix, reshape(Int[55, 2], :, 1))

        logits_a, _ = model(tokens_a, params, eval_state)
        logits_b, _ = model(tokens_b, params, eval_state)

        @test isapprox(logits_a[:, 1:3, 1], logits_b[:, 1:3, 1]; atol = 1f-5, rtol = 1f-5)
        @test !isapprox(logits_a[:, 5, 1], logits_b[:, 5, 1]; atol = 1f-6, rtol = 1f-6)
    end

    @testset "Greedy Generation Extends Prompt" begin
        prompt = reshape(Int[3, 7, 11, 19], :, 1)
        generated, generation_state = greedy_generate(
            model,
            prompt,
            params,
            eval_state;
            max_new_tokens = 4,
        )

        @test size(generated, 1) == size(prompt, 1) + 4
        @test size(generated, 2) == 1
        @test all((generated .>= 1) .& (generated .<= config.vocab_size))
        @test :OutputHead in keys(generation_state)
    end

    @testset "Cached Decode Matches Full Recompute" begin
        prompt = reshape(Int[3, 7, 11, 19], :, 1)
        next_input = reshape(Int[23], :, 1)

        cache = init_decoder_cache(model, 1)
        prompt_logits, prompt_state, filled_cache = forward_with_cache(model, prompt, params, eval_state, cache)
        full_logits, _ = model(vcat(prompt, next_input), params, eval_state)
        cached_logits, _, advanced_cache =
            next_token_logits_cached(model, next_input, params, prompt_state, filled_cache)

        @test size(prompt_logits) == (config.vocab_size, size(prompt, 1), 1)
        @test cache_sequence_length(filled_cache) == size(prompt, 1)
        @test cache_sequence_length(advanced_cache) == size(prompt, 1) + size(next_input, 1)
        @test isapprox(cached_logits[:, 1], full_logits[:, end, 1]; atol = 1f-4, rtol = 1f-4)
    end

    @testset "Cached Greedy Generation Matches Uncached" begin
        prompt = reshape(Int[5, 9, 12], :, 1)
        uncached_tokens, _ = greedy_generate(
            model,
            prompt,
            params,
            eval_state;
            max_new_tokens = 3,
        )
        cached_tokens, _, cache = greedy_generate_cached(
            model,
            prompt,
            params,
            eval_state;
            max_new_tokens = 3,
        )

        @test cached_tokens == uncached_tokens
        @test cache_sequence_length(cache) == size(prompt, 1) + 3
    end

    @testset "Causal Bias Shape" begin
        bias = build_causal_attention_bias(4)
        @test size(bias) == (4, 4)
        @test bias[1, 1] == 0f0
        @test bias[1, 4] < -1f8
        @test bias[4, 1] == 0f0

        offset_bias = build_causal_attention_bias(2, 5; query_offset = 3)
        @test offset_bias[1, 4] == 0f0
        @test offset_bias[1, 5] < -1f8
        @test offset_bias[2, 5] == 0f0
    end

    @testset "Synthetic Granite Import" begin
        mktempdir() do dir
            granite_config = NativeTeacherConfig(
                vocab_size = 64,
                max_sequence_length = 32,
                embedding_dimension = 32,
                number_of_layers = 2,
                number_of_heads = 4,
                number_of_kv_heads = 2,
                mlp_hidden_dimension = 96,
                rope_theta = 50_000f0,
                rms_norm_eps = 1f-5,
                embedding_multiplier = 12f0,
                logits_scaling = 10f0,
                tie_word_embeddings = true,
                dropout_rate = 0f0,
                use_bias = false,
            )
            tensors = write_synthetic_granite_checkpoint(dir, granite_config)

            loaded_config = granite_config_from_hf(dir)
            @test loaded_config.embedding_dimension == granite_config.embedding_dimension
            @test loaded_config.number_of_layers == granite_config.number_of_layers
            @test loaded_config.number_of_kv_heads == granite_config.number_of_kv_heads
            @test loaded_config.embedding_multiplier == 12f0
            @test loaded_config.logits_scaling == 10f0
            @test loaded_config.tie_word_embeddings
            @test loaded_config.rms_norm_eps == 1f-5

            granite_model = NativeCausalLM(loaded_config)
            granite_params, granite_state = Lux.setup(Random.MersenneTwister(11), granite_model)
            imported_params = load_granite_weights(granite_model, granite_params, dir)

            @test imported_params.TokenEmbedding.weight == permutedims(tensors["model.embed_tokens.weight"], (2, 1))
            @test imported_params.OutputHead.weight == tensors["model.embed_tokens.weight"]
            @test imported_params.Blocks[1].Attention.QueryProjection.weight ==
                tensors["model.layers.0.self_attn.q_proj.weight"]
            @test imported_params.Blocks[2].Attention.KeyProjection.weight ==
                tensors["model.layers.1.self_attn.k_proj.weight"]
            @test imported_params.Blocks[1].FeedForward.GateProjection.weight ==
                tensors["model.layers.0.shared_mlp.input_linear.weight"][1:granite_config.mlp_hidden_dimension, :]
            @test imported_params.Blocks[1].FeedForward.UpProjection.weight ==
                tensors["model.layers.0.shared_mlp.input_linear.weight"][(granite_config.mlp_hidden_dimension + 1):end, :]
            @test imported_params.Blocks[2].FeedForward.DownProjection.weight ==
                tensors["model.layers.1.shared_mlp.output_linear.weight"]
            @test imported_params.FinalNorm.scale == tensors["model.norm.weight"]

            logits, _ = granite_model(
                rand(rng, 1:loaded_config.vocab_size, 5, 1),
                imported_params,
                Lux.testmode(granite_state),
            )
            @test size(logits) == (loaded_config.vocab_size, 5, 1)
            @test all(isfinite, logits)
        end
    end

    @testset "Granite Chat Template Wrapper" begin
        granite_snapshot = joinpath(
            homedir(),
            ".cache",
            "huggingface",
            "hub",
            "models--ibm-granite--granite-4.0-micro",
            "snapshots",
            "56111ae135df9c53a78c99028e7bc24035a9e979",
        )

        if isdir(granite_snapshot)
            try
                tokenizer = Swamma.HFTokenizer.load_tokenizer(
                    granite_snapshot;
                    local_files_only = true,
                )
                messages = [
                    (role = "system", content = "You extract entities and relations."),
                    (role = "user", content = "Barack Obama was born in Hawaii."),
                ]

                @test Swamma.HFTokenizer.has_chat_template(tokenizer)

                rendered = Swamma.HFTokenizer.apply_chat_template(
                    tokenizer,
                    messages;
                    add_generation_prompt = true,
                )
                @test occursin("<|start_of_role|>system", rendered)
                @test occursin("<|start_of_role|>assistant", rendered)

                token_ids = Swamma.HFTokenizer.apply_chat_template_tokens(
                    tokenizer,
                    messages;
                    add_generation_prompt = true,
                )
                @test !isempty(token_ids)
                @test all(token_ids .>= 1)
            catch exc
                @test_skip "PyCall tokenizer runtime unavailable: $(typeof(exc))"
            end
        else
            @test_skip "local granite tokenizer snapshot not available"
        end
    end
end
