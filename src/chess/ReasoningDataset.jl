module ReasoningDataset

"""
Reasoning Dataset: Loader for Phase 3a Language Fine-Tuning

Loads and tokenizes reasoning examples from six datasets:
- PrOntoQA (syllogistic chains)
- ReClor (argumentation)
- GSM8K (arithmetic reasoning)
- BoardgameQA (rule-based deduction)
- FOLIO (first-order logic)
- LogiQA (formal logic)

Each example is converted to a text sequence with explicit reasoning structure:
    Context: <context>
    Question: <question>
    Reasoning: <chain_of_thought or answer>

The text is then tokenized to integer sequences for the drafter.
"""

using JSON3
using Random

export ReasoningExample, load_reasoning_dataset, load_all_reasoning_datasets
export prepare_reasoning_batch, iterate_reasoning_batches
export text_to_char_tokens, REASONING_CHAR_VOCAB_SIZE

# ============================================================================
# Character-level tokenizer (simple, no external dependencies)
# ============================================================================

# Printable ASCII (32-126) + special tokens
const PAD_TOKEN = 1
const BOS_TOKEN = 2
const EOS_TOKEN = 3
const UNK_TOKEN = 4
const CHAR_OFFSET = 4  # first 4 IDs are special tokens
const REASONING_CHAR_VOCAB_SIZE = 128 + CHAR_OFFSET  # ASCII 0-127 + specials

"""
    text_to_char_tokens(text; max_length=512) → Vector{Int}

Character-level tokenization. Each character maps to its ASCII code + offset.
Pads or truncates to max_length. Adds BOS/EOS.
"""
function text_to_char_tokens(text::AbstractString; max_length::Int = 512)
    tokens = Int[BOS_TOKEN]
    for ch in text
        code = Int(ch)
        if 0 <= code <= 127
            push!(tokens, code + CHAR_OFFSET + 1)  # 1-based
        else
            push!(tokens, UNK_TOKEN)
        end
        length(tokens) >= max_length - 1 && break
    end
    push!(tokens, EOS_TOKEN)

    # Pad to max_length
    while length(tokens) < max_length
        push!(tokens, PAD_TOKEN)
    end

    return tokens[1:max_length]
end

# ============================================================================
# Example struct
# ============================================================================

struct ReasoningExample
    source::String           # dataset name
    reasoning_type::String   # syllogistic, argumentation, arithmetic_chain, etc.
    text::String             # formatted reasoning text
    tokens::Vector{Int}      # character-level token IDs
    label::Int               # answer label (for MCQ datasets, -1 if N/A)
end

# ============================================================================
# Per-dataset formatters
# ============================================================================

function format_prontoqa(data)
    ctx = get(data, :context, get(data, "context", ""))
    q = get(data, :question, get(data, "question", ""))
    cot = get(data, :chain_of_thought, get(data, "chain_of_thought", ""))
    ans = get(data, :answer, get(data, "answer", ""))
    reasoning = isempty(cot) ? ans : cot
    return "Context: $ctx\nQuestion: $q\nReasoning: $reasoning"
end

function format_reclor(data)
    ctx = get(data, :context, get(data, "context", ""))
    q = get(data, :question, get(data, "question", ""))
    answers = get(data, :answers, get(data, "answers", String[]))
    label = get(data, :label, get(data, "label", -1))
    opts = join(["($('A'+i-1)) $a" for (i, a) in enumerate(answers)], "\n")
    correct = 0 <= label < length(answers) ? answers[label + 1] : ""
    return "Context: $ctx\nQuestion: $q\n$opts\nAnswer: $correct"
end

function format_gsm8k(data)
    q = get(data, :question, get(data, "question", ""))
    a = get(data, :answer, get(data, "answer", ""))
    return "Question: $q\nReasoning: $a"
end

function format_boardgameqa(data)
    ctx = get(data, :context, get(data, "context", ""))
    q = get(data, :question, get(data, "question", ""))
    a = get(data, :answer, get(data, "answer", ""))
    return "Rules: $ctx\nQuestion: $q\nReasoning: $a"
end

function format_folio(data)
    premises = get(data, :premises, get(data, "premises", ""))
    conclusion = get(data, :conclusion, get(data, "conclusion", ""))
    label = get(data, :label, get(data, "label", ""))
    return "Premises: $premises\nConclusion: $conclusion\nJudgment: $label"
end

function format_logiqa(data)
    ctx = get(data, :context, get(data, "context", ""))
    q = get(data, :question, get(data, "question", ""))
    opts = get(data, :options, get(data, "options", String[]))
    label = get(data, :label, get(data, "label", -1))
    opts_str = join(["($('A'+i-1)) $o" for (i, o) in enumerate(opts)], "\n")
    return "Context: $ctx\nQuestion: $q\n$opts_str"
end

function format_logicnli(data)
    ctx = get(data, :context, get(data, "context", ""))
    q = get(data, :question, get(data, "question", ""))
    a = get(data, :answer, get(data, "answer", ""))
    return "Premise: $ctx\nHypothesis: $q\nEntailment: $a"
end

function format_arc_challenge(data)
    q = get(data, :question, get(data, "question", ""))
    opts = get(data, :options, get(data, "options", String[]))
    a = get(data, :answer, get(data, "answer", ""))
    opts_str = join(opts, "\n")
    return "Question: $q\n$opts_str\nAnswer: $a"
end

function format_babi(data)
    ctx = get(data, :context, get(data, "context", ""))
    q = get(data, :question, get(data, "question", ""))
    a = get(data, :answer, get(data, "answer", ""))
    return "Facts: $ctx\nStatement: $q\nValid: $a"
end

const FORMATTERS = Dict{String, Function}(
    "prontoqa" => format_prontoqa,
    "reclor" => format_reclor,
    "gsm8k" => format_gsm8k,
    "boardgameqa" => format_boardgameqa,
    "folio" => format_folio,
    "logiqa" => format_logiqa,
    "logicnli" => format_logicnli,
    "arc_challenge" => format_arc_challenge,
    "babi_deduction" => format_babi,
    "babi_induction" => format_babi,
)

# ============================================================================
# Loading
# ============================================================================

"""
    load_reasoning_dataset(path; max_examples=nothing, max_seq_length=512) → Vector{ReasoningExample}

Load a single JSONL reasoning dataset file.
"""
function load_reasoning_dataset(path::String;
                                 max_examples::Union{Int,Nothing} = nothing,
                                 max_seq_length::Int = 512)
    examples = ReasoningExample[]
    open(path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            data = try; JSON3.read(line); catch; continue; end

            source = String(get(data, :source, get(data, "source", "unknown")))
            rtype = String(get(data, :reasoning_type, get(data, "reasoning_type", "unknown")))
            label = Int(get(data, :label, get(data, "label", -1)))

            formatter = get(FORMATTERS, source, nothing)
            text = formatter !== nothing ? formatter(data) : string(data)

            tokens = text_to_char_tokens(text; max_length = max_seq_length)
            push!(examples, ReasoningExample(source, rtype, text, tokens, label))

            if max_examples !== nothing && length(examples) >= max_examples
                break
            end
        end
    end
    return examples
end

"""
    load_all_reasoning_datasets(dir; max_per_dataset=nothing, max_seq_length=512) → Vector{ReasoningExample}

Load all JSONL files from a directory and merge with balanced sampling.
"""
function load_all_reasoning_datasets(dir::String;
                                      max_per_dataset::Union{Int,Nothing} = nothing,
                                      max_seq_length::Int = 512)
    all_examples = ReasoningExample[]
    for fname in readdir(dir)
        endswith(fname, ".jsonl") || continue
        path = joinpath(dir, fname)
        examples = load_reasoning_dataset(path;
            max_examples = max_per_dataset,
            max_seq_length = max_seq_length)
        append!(all_examples, examples)
        println("  Loaded $(length(examples)) from $fname")
    end
    println("  Total: $(length(all_examples)) reasoning examples")
    return all_examples
end

# ============================================================================
# Batching
# ============================================================================

"""
    prepare_reasoning_batch(examples) → NamedTuple

Collate reasoning examples into training-ready arrays:
- tokens: (max_seq_length, batch) Int matrix
- labels: (batch,) Int vector (MCQ label, -1 if N/A)
- sources: Vector{String} (dataset names)
"""
function prepare_reasoning_batch(examples::AbstractVector{ReasoningExample})
    batch_size = length(examples)
    seq_len = length(examples[1].tokens)
    tokens = Matrix{Int}(undef, seq_len, batch_size)
    labels = Vector{Int}(undef, batch_size)
    sources = Vector{String}(undef, batch_size)

    for (i, ex) in enumerate(examples)
        tokens[:, i] = ex.tokens
        labels[i] = ex.label
        sources[i] = ex.source
    end

    return (tokens = tokens, labels = labels, sources = sources)
end

"""
    iterate_reasoning_batches(examples, batch_size; shuffle=true, rng=nothing) → Vector{NamedTuple}
"""
function iterate_reasoning_batches(examples::Vector{ReasoningExample}, batch_size::Int;
                                    shuffle::Bool = true, rng = nothing)
    n = length(examples)
    indices = collect(1:n)
    if shuffle
        rng !== nothing ? Random.shuffle!(rng, indices) : Random.shuffle!(indices)
    end

    batches = NamedTuple[]
    for start in 1:batch_size:n
        stop = min(start + batch_size - 1, n)
        push!(batches, prepare_reasoning_batch(examples[indices[start:stop]]))
    end
    return batches
end

end # module
