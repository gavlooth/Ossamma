# HFTokenizer.jl - HuggingFace Tokenizer wrapper for Julia
#
# Wraps HuggingFace tokenizers via PyCall for use with Granite/Qwen3/Llama.
# Provides a Julia-native interface for tokenization.

module HFTokenizer

using JSON3
using PyCall

export HuggingFaceTokenizer, load_tokenizer
export encode, decode, batch_encode, batch_decode
export get_vocab_size, get_mask_token_id, get_pad_token_id
export has_chat_template, apply_chat_template, apply_chat_template_tokens

# =============================================================================
# Tokenizer Wrapper
# =============================================================================

"""
    HuggingFaceTokenizer

Wrapper around HuggingFace AutoTokenizer.
"""
struct HuggingFaceTokenizer
    py_tokenizer::PyObject
    model_name::String
    backend::Symbol
    vocab_size::Int
    pad_token_id::Int
    mask_token_id::Int
    eos_token_id::Union{Int, Nothing}
    bos_token_id::Union{Int, Nothing}
    chat_template::Union{String, Nothing}
    model_dir::Union{String, Nothing}
end

function resolve_model_dir(model_name::String; local_files_only::Bool = false)
    if isdir(model_name)
        return abspath(model_name)
    end
    huggingface_hub = pyimport("huggingface_hub")
    path = huggingface_hub.snapshot_download(
        repo_id = model_name,
        local_files_only = local_files_only,
        allow_patterns = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"],
    )
    return String(path)
end

function load_json(path::String)
    return JSON3.read(read(path, String))
end

function parse_special_token_content(value)
    value === nothing && return nothing
    if value isa AbstractString
        return String(value)
    end
    if haskey(value, "content")
        return String(value["content"])
    end
    return nothing
end

function token_id_from_content(py_tokenizer::PyObject, token_content::Union{String, Nothing})
    token_content === nothing && return nothing
    token_id = py_tokenizer.token_to_id(token_content)
    token_id === nothing && return nothing
    return Int(token_id)
end

function load_tokenizer_via_tokenizers(model_name::String; local_files_only::Bool = false)
    model_dir = resolve_model_dir(model_name; local_files_only = local_files_only)
    tokenizer_path = joinpath(model_dir, "tokenizer.json")
    isfile(tokenizer_path) || error("Tokenizer fallback requires tokenizer.json at $(tokenizer_path)")

    tokenizers = pyimport("tokenizers")
    py_tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)

    tokenizer_config_path = joinpath(model_dir, "tokenizer_config.json")
    tokenizer_config = isfile(tokenizer_config_path) ? load_json(tokenizer_config_path) : Dict{String,Any}()
    special_tokens_path = joinpath(model_dir, "special_tokens_map.json")
    special_tokens_map = isfile(special_tokens_path) ? load_json(special_tokens_path) : Dict{String,Any}()

    vocab_size = Int(py_tokenizer.get_vocab_size())
    pad_token_content = parse_special_token_content(
        haskey(tokenizer_config, "pad_token") ? tokenizer_config["pad_token"] :
        (haskey(special_tokens_map, "pad_token") ? special_tokens_map["pad_token"] : nothing),
    )
    eos_token_content = parse_special_token_content(
        haskey(tokenizer_config, "eos_token") ? tokenizer_config["eos_token"] :
        (haskey(special_tokens_map, "eos_token") ? special_tokens_map["eos_token"] : nothing),
    )
    bos_token_content = parse_special_token_content(
        haskey(tokenizer_config, "bos_token") ? tokenizer_config["bos_token"] :
        (haskey(special_tokens_map, "bos_token") ? special_tokens_map["bos_token"] : nothing),
    )
    mask_token_content = parse_special_token_content(
        haskey(tokenizer_config, "mask_token") ? tokenizer_config["mask_token"] :
        (haskey(special_tokens_map, "mask_token") ? special_tokens_map["mask_token"] : nothing),
    )

    pad_token_id = something(
        token_id_from_content(py_tokenizer, pad_token_content),
        token_id_from_content(py_tokenizer, eos_token_content),
        0,
    )
    eos_token_id = token_id_from_content(py_tokenizer, eos_token_content)
    bos_token_id = token_id_from_content(py_tokenizer, bos_token_content)
    mask_token_id = something(token_id_from_content(py_tokenizer, mask_token_content), vocab_size - 1)

    chat_template_path = joinpath(model_dir, "chat_template.jinja")
    chat_template = isfile(chat_template_path) ? read(chat_template_path, String) : nothing

    return HuggingFaceTokenizer(
        py_tokenizer,
        model_name,
        :tokenizers,
        vocab_size,
        pad_token_id,
        mask_token_id,
        eos_token_id,
        bos_token_id,
        chat_template,
        model_dir,
    )
end

"""
    load_tokenizer(model_name::String; trust_remote_code=true) -> HuggingFaceTokenizer

Load a HuggingFace tokenizer by model name.

# Examples
```julia
# Granite 4.0
tokenizer = load_tokenizer("ibm-granite/granite-4.0-micro")

# Qwen3
tokenizer = load_tokenizer("Qwen/Qwen3-4B")

# Llama 3
tokenizer = load_tokenizer("meta-llama/Llama-3.1-8B")
```
"""
function load_tokenizer(
    model_name::String;
    trust_remote_code::Bool = true,
    local_files_only::Bool = false,
)
    try
        auto_tokenization = pyimport("transformers")
        py_tokenizer = auto_tokenization.AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code = trust_remote_code,
            local_files_only = local_files_only,
        )

        vocab_size = Int(py_tokenizer.vocab_size)

        pad_token_id = try
            py_tokenizer.pad_token_id
        catch
            nothing
        end

        eos_token_id = try
            py_tokenizer.eos_token_id
        catch
            nothing
        end

        if pad_token_id === nothing
            sep_token_id = try
                py_tokenizer.sep_token_id
            catch
                nothing
            end
            pad_token_id = something(eos_token_id, sep_token_id, 0)
        end

        mask_token_id = try
            py_tokenizer.mask_token_id
        catch
            vocab_size - 1
        end

        bos_token_id = try
            py_tokenizer.bos_token_id
        catch
            nothing
        end

        chat_template = try
            tmpl = py_tokenizer.chat_template
            tmpl === nothing ? nothing : String(tmpl)
        catch
            nothing
        end

        model_dir = try
            resolve_model_dir(model_name; local_files_only = local_files_only)
        catch
            isdir(model_name) ? abspath(model_name) : nothing
        end

        return HuggingFaceTokenizer(
            py_tokenizer,
            model_name,
            :transformers,
            vocab_size,
            Int(pad_token_id),
            Int(mask_token_id !== nothing ? mask_token_id : vocab_size - 1),
            eos_token_id === nothing ? nothing : Int(eos_token_id),
            bos_token_id === nothing ? nothing : Int(bos_token_id),
            chat_template,
            model_dir,
        )
    catch
        return load_tokenizer_via_tokenizers(model_name; local_files_only = local_files_only)
    end
end

# =============================================================================
# Tokenizer Interface
# =============================================================================

"""
    encode(tokenizer, text; add_special_tokens=true, max_length=nothing) -> Vector{Int}

Encode text to token IDs.
"""
function encode(
    tokenizer::HuggingFaceTokenizer,
    text::String;
    add_special_tokens::Bool = true,
    max_length::Union{Int, Nothing} = nothing
)
    kwargs = Dict{Symbol,Any}(
        :add_special_tokens => add_special_tokens,
        :return_tensors => nothing
    )

    if max_length !== nothing
        kwargs[:max_length] = max_length
        kwargs[:truncation] = true
    end

    if tokenizer.backend == :transformers
        encoding = tokenizer.py_tokenizer(text; kwargs...)
        input_ids = encoding["input_ids"]
        return [Int(id) + 1 for id in input_ids]
    else
        encoding = tokenizer.py_tokenizer.encode(text; add_special_tokens = add_special_tokens)
        input_ids = [Int(id) + 1 for id in encoding.ids]
        if max_length !== nothing
            return input_ids[1:min(end, max_length)]
        end
        return input_ids
    end
end

"""
    decode(tokenizer, token_ids; skip_special_tokens=true) -> String

Decode token IDs to text.
"""
function decode(
    tokenizer::HuggingFaceTokenizer,
    token_ids::AbstractVector{<:Integer};
    skip_special_tokens::Bool = true
)
    # Convert back to 0-indexed for Python
    py_ids = [id - 1 for id in token_ids]
    return String(tokenizer.py_tokenizer.decode(py_ids, skip_special_tokens=skip_special_tokens))
end

"""
    batch_encode(tokenizer, texts; add_special_tokens=true, max_length=nothing, padding=true)

Batch encode multiple texts.

Returns:
- `input_ids`: Matrix of token IDs (seq_len, batch)
- `attention_mask`: Matrix of attention masks (seq_len, batch)
"""
function batch_encode(
    tokenizer::HuggingFaceTokenizer,
    texts::Vector{String};
    add_special_tokens::Bool = true,
    max_length::Union{Int, Nothing} = nothing,
    padding::Bool = true
)
    kwargs = Dict{Symbol,Any}(
        :add_special_tokens => add_special_tokens,
        :padding => padding,
        :return_tensors => "np"  # NumPy for easy conversion
    )

    if max_length !== nothing
        kwargs[:max_length] = max_length
        kwargs[:truncation] = true
    end

    if tokenizer.backend == :transformers
        encoding = tokenizer.py_tokenizer(texts; kwargs...)

        np = pyimport("numpy")
        input_ids_np = encoding["input_ids"]
        attention_mask_np = encoding["attention_mask"]

        input_ids = Array{Int}(PyArray(input_ids_np)) .+ 1
        attention_mask = Array{Bool}(PyArray(attention_mask_np))

        return (
            input_ids = permutedims(input_ids, (2, 1)),
            attention_mask = permutedims(attention_mask, (2, 1))
        )
    else
        encodings = tokenizer.py_tokenizer.encode_batch(texts; add_special_tokens = add_special_tokens)
        sequences = Vector{Vector{Int}}(undef, length(texts))
        masks = Vector{Vector{Bool}}(undef, length(texts))
        target_length = 0
        for (index, encoding) in enumerate(encodings)
            ids = [Int(id) + 1 for id in encoding.ids]
            attention = [Bool(v) for v in encoding.attention_mask]
            if max_length !== nothing
                trunc_length = min(length(ids), max_length)
                ids = ids[1:trunc_length]
                attention = attention[1:trunc_length]
            end
            sequences[index] = ids
            masks[index] = attention
            target_length = max(target_length, length(ids))
        end
        if !padding
            target_length = isempty(sequences) ? 0 : minimum(length.(sequences))
        end
        input_ids = fill(tokenizer.pad_token_id + 1, target_length, length(texts))
        attention_mask = fill(false, target_length, length(texts))
        for batch_index in eachindex(sequences)
            ids = sequences[batch_index]
            attention = masks[batch_index]
            limit = min(length(ids), target_length)
            limit == 0 && continue
            input_ids[1:limit, batch_index] .= ids[1:limit]
            attention_mask[1:limit, batch_index] .= attention[1:limit]
        end
        return (input_ids = input_ids, attention_mask = attention_mask)
    end
end

"""
    batch_decode(tokenizer, token_ids; skip_special_tokens=true) -> Vector{String}

Batch decode token IDs to texts.
"""
function batch_decode(
    tokenizer::HuggingFaceTokenizer,
    token_ids::AbstractMatrix{<:Integer};
    skip_special_tokens::Bool = true
)
    batch_size = size(token_ids, 2)
    results = String[]

    for b in 1:batch_size
        ids = token_ids[:, b]
        push!(results, decode(tokenizer, ids; skip_special_tokens))
    end

    return results
end

function has_chat_template(tokenizer::HuggingFaceTokenizer)
    return tokenizer.chat_template !== nothing
end

function normalize_chat_messages(messages)
    py_messages = PyObject[]
    for message in messages
        if message isa NamedTuple
            role = get(message, :role, nothing)
            content = get(message, :content, nothing)
            role === nothing && throw(ArgumentError("chat messages must provide :role"))
            content === nothing && throw(ArgumentError("chat messages must provide :content"))
            push!(py_messages, PyDict(Dict("role" => String(role), "content" => String(content))))
        elseif message isa AbstractDict
            haskey(message, "role") || haskey(message, :role) ||
                throw(ArgumentError("chat messages must provide role"))
            haskey(message, "content") || haskey(message, :content) ||
                throw(ArgumentError("chat messages must provide content"))
            role = haskey(message, "role") ? message["role"] : message[:role]
            content = haskey(message, "content") ? message["content"] : message[:content]
            push!(py_messages, PyDict(Dict("role" => String(role), "content" => String(content))))
        else
            throw(ArgumentError("Unsupported chat message type $(typeof(message))"))
        end
    end
    return py_messages
end

function apply_chat_template(
    tokenizer::HuggingFaceTokenizer,
    messages;
    add_generation_prompt::Bool = true,
)
    has_chat_template(tokenizer) ||
        throw(ArgumentError("Tokenizer $(repr(tokenizer.model_name)) does not expose a chat template"))
    if tokenizer.backend == :transformers
        py_messages = normalize_chat_messages(messages)
        rendered = tokenizer.py_tokenizer.apply_chat_template(
            py_messages;
            tokenize = false,
            add_generation_prompt = add_generation_prompt,
        )
        return String(rendered)
    else
        jinja2 = pyimport("jinja2")
        env = jinja2.Environment(trim_blocks = true, lstrip_blocks = true)
        template = env.from_string(tokenizer.chat_template)
        py_messages = normalize_chat_messages(messages)
        rendered = template.render(
            messages = py_messages,
            add_generation_prompt = add_generation_prompt,
            tools = nothing,
            available_tools = nothing,
            documents = nothing,
        )
        return String(rendered)
    end
end

function apply_chat_template_tokens(
    tokenizer::HuggingFaceTokenizer,
    messages;
    add_generation_prompt::Bool = true,
    max_length::Union{Int, Nothing} = nothing,
)
    rendered = apply_chat_template(
        tokenizer,
        messages;
        add_generation_prompt = add_generation_prompt,
    )
    return encode(
        tokenizer,
        rendered;
        add_special_tokens = false,
        max_length = max_length,
    )
end

# =============================================================================
# Utility Functions
# =============================================================================

get_vocab_size(t::HuggingFaceTokenizer) = t.vocab_size
get_mask_token_id(t::HuggingFaceTokenizer) = t.mask_token_id + 1  # Julia 1-indexed
get_pad_token_id(t::HuggingFaceTokenizer) = t.pad_token_id + 1    # Julia 1-indexed

function Base.show(io::IO, t::HuggingFaceTokenizer)
    print(io, "HuggingFaceTokenizer(\"$(t.model_name)\", vocab_size=$(t.vocab_size))")
end

# =============================================================================
# Preset Tokenizers
# =============================================================================

"""
    load_granite_tokenizer(; model="ibm-granite/granite-4.0-micro")

Load the Granite 4.0 tokenizer.
"""
function load_granite_tokenizer(; model::String = "ibm-granite/granite-4.0-micro")
    return load_tokenizer(model)
end

"""
    load_qwen3_tokenizer(; model="Qwen/Qwen3-4B")

Load the Qwen3 tokenizer.
"""
function load_qwen3_tokenizer(; model::String = "Qwen/Qwen3-4B")
    return load_tokenizer(model)
end

export load_granite_tokenizer, load_qwen3_tokenizer

end # module HFTokenizer
