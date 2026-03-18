module ChessDataset

"""
Chess Dataset: Lichess Evaluation Database Loader

Loads positions from the Lichess evaluation database (JSONL format).
Each line is a JSON object:
    {"fen": "rnbqkbnr/...", "evals": [{"pvs": [{"cp": 15, "line": "e2e4 e7e5"}], "knodes": 1234, "depth": 20}]}

Produces batches of:
- board_tokens: (64, batch) — piece tokens for each square
- metadata: (6, batch) — side to move, castling, en passant
- best_move_id: (batch,) — UCI best move encoded as integer
- eval_score: (batch,) — centipawn evaluation (clamped, normalized)
"""

using JSON3

using ..ChessTokenizer

export LichessEvalReader, read_positions, prepare_batch
export normalize_eval, load_lichess_jsonl, iterate_batches, StreamingBatchIterator, next_batch!

# ============================================================================
# Evaluation normalization
# ============================================================================

"""
    normalize_eval(cp::Real; scale=400.0) → Float32

Normalize centipawn evaluation to approximately [-1, 1] using tanh scaling.
cp=0 → 0.0 (equal), cp=400 → ~0.76 (winning), mate → ±1.0.
"""
function normalize_eval(cp::Real; scale::Float64 = 400.0)
    return Float32(tanh(cp / scale))
end

"""
    normalize_eval_mate(mate_in::Int) → Float32

Convert mate-in-N to normalized eval. Positive = white mates, negative = black mates.
"""
function normalize_eval_mate(mate_in::Int)
    return mate_in > 0 ? 1.0f0 : -1.0f0
end

# ============================================================================
# Position struct
# ============================================================================

struct ChessPosition
    fen::String
    tokens::Vector{Int}       # length 64, 1-based piece IDs
    metadata::Vector{Float32} # length 6
    best_move::String         # UCI format
    best_move_id::Int         # encoded move ID (1-based)
    eval_cp::Float32          # raw centipawn score
    eval_normalized::Float32  # tanh-scaled to [-1, 1]
    depth::Int                # Stockfish search depth
end

# ============================================================================
# JSONL reader
# ============================================================================

"""
    parse_lichess_eval_line(line::String) → Union{ChessPosition, Nothing}

Parse one line from the Lichess evaluation JSONL database.
Returns nothing if the line is malformed or has no usable evaluation.
"""
function parse_lichess_eval_line(line::AbstractString)
    isempty(strip(line)) && return nothing

    data = try
        JSON3.read(line)
    catch
        return nothing
    end

    haskey(data, :fen) || return nothing
    haskey(data, :evals) || return nothing
    fen = String(data.fen)

    evals = data.evals
    isempty(evals) && return nothing

    # Take the deepest evaluation
    best_eval = nothing
    best_depth = 0
    for ev in evals
        d = get(ev, :depth, 0)
        if d > best_depth
            best_depth = d
            best_eval = ev
        end
    end
    best_eval === nothing && return nothing

    pvs = get(best_eval, :pvs, nothing)
    (pvs === nothing || isempty(pvs)) && return nothing
    pv = pvs[1]

    # Get evaluation (centipawns or mate)
    has_cp = haskey(pv, :cp)
    has_mate = haskey(pv, :mate)
    (!has_cp && !has_mate) && return nothing

    eval_cp = if has_cp
        Float32(pv.cp)
    else
        # Convert mate to large centipawn value
        mate_in = Int(pv.mate)
        Float32(mate_in > 0 ? 10000 : -10000)
    end
    eval_normalized = has_cp ? normalize_eval(eval_cp) : normalize_eval_mate(Int(pv.mate))

    # Get best move (first move in principal variation)
    line_str = get(pv, :line, "")
    isempty(line_str) && return nothing
    best_move = String(split(line_str)[1])

    # Validate move format
    (4 <= length(best_move) <= 5) || return nothing

    # Encode
    tokens = try; fen_to_tokens(fen); catch; return nothing; end
    metadata = try; fen_to_metadata(fen); catch; return nothing; end
    move_id = try; uci_to_move_id(best_move); catch; return nothing; end

    return ChessPosition(fen, tokens, metadata, best_move, move_id,
                          eval_cp, eval_normalized, best_depth)
end

"""
    load_lichess_jsonl(path::String; max_positions=nothing, min_depth=10) → Vector{ChessPosition}

Load positions from a Lichess evaluation JSONL file.
Filters by minimum Stockfish depth.
"""
function load_lichess_jsonl(path::String; max_positions::Union{Int,Nothing} = nothing, min_depth::Int = 10)
    positions = ChessPosition[]
    open(path, "r") do io
        for line in eachline(io)
            pos = parse_lichess_eval_line(line)
            pos === nothing && continue
            pos.depth < min_depth && continue
            push!(positions, pos)
            if max_positions !== nothing && length(positions) >= max_positions
                break
            end
        end
    end
    return positions
end

# ============================================================================
# Batching
# ============================================================================

"""
    prepare_batch(positions::Vector{ChessPosition}) → NamedTuple

Collate a vector of ChessPositions into training-ready batch arrays:
- board_tokens: (64, batch) Int matrix
- metadata: (6, batch) Float32 matrix
- best_move_ids: (batch,) Int vector
- eval_scores: (batch,) Float32 vector (normalized)
"""
function prepare_batch(positions::AbstractVector{ChessPosition})
    batch_size = length(positions)
    board_tokens = Matrix{Int}(undef, 64, batch_size)
    metadata = Matrix{Float32}(undef, 6, batch_size)
    best_move_ids = Vector{Int}(undef, batch_size)
    eval_scores = Vector{Float32}(undef, batch_size)

    for (i, pos) in enumerate(positions)
        board_tokens[:, i] = pos.tokens
        metadata[:, i] = pos.metadata
        best_move_ids[i] = pos.best_move_id
        eval_scores[i] = pos.eval_normalized
    end

    return (
        board_tokens = board_tokens,
        metadata = metadata,
        best_move_ids = best_move_ids,
        eval_scores = eval_scores,
    )
end

"""
    iterate_batches(positions, batch_size; shuffle=true, rng=Random.default_rng())

Yield batches from a position vector. Shuffles each epoch if requested.
"""
function iterate_batches(positions::Vector{ChessPosition}, batch_size::Int;
                          shuffle::Bool = true, rng = nothing)
    n = length(positions)
    indices = collect(1:n)
    if shuffle && rng !== nothing
        Random.shuffle!(rng, indices)
    elseif shuffle
        Random.shuffle!(indices)
    end

    batches = NamedTuple[]
    for start in 1:batch_size:n
        stop = min(start + batch_size - 1, n)
        batch_positions = positions[indices[start:stop]]
        push!(batches, prepare_batch(batch_positions))
    end
    return batches
end

using Random

# ============================================================================
# Streaming batch iterator (reads from disk, no preload)
# ============================================================================

mutable struct StreamingBatchIterator
    path::String
    batch_size::Int
    min_depth::Int
    max_positions::Int
    io::Union{IOStream, Nothing}
    positions_read::Int
end

function StreamingBatchIterator(path::String; batch_size::Int=64, min_depth::Int=10, max_positions::Int=typemax(Int))
    StreamingBatchIterator(path, batch_size, min_depth, max_positions, nothing, 0)
end

function _open_stream!(it::StreamingBatchIterator)
    it.io !== nothing && close(it.io)
    it.io = open(it.path, "r")
    it.positions_read = 0
end

"""
    next_batch!(it::StreamingBatchIterator) → Union{NamedTuple, Nothing}

Read the next batch from disk. Returns nothing at end of file/max_positions.
Call `_open_stream!` to reset for a new epoch.
"""
function next_batch!(it::StreamingBatchIterator)
    it.io === nothing && _open_stream!(it)
    buf = ChessPosition[]
    while length(buf) < it.batch_size
        it.positions_read >= it.max_positions && break
        eof(it.io) && break
        line = readline(it.io)
        pos = parse_lichess_eval_line(line)
        pos === nothing && continue
        pos.depth < it.min_depth && continue
        push!(buf, pos)
        it.positions_read += 1
    end
    isempty(buf) && return nothing
    return prepare_batch(buf)
end

end # module
