module ChessTokenizer

"""
Chess Tokenizer: FEN ↔ 64-Square Token Grid

Encodes chess positions as 64-token sequences for the ReasoningDrafter.
Each token represents one square (a1=1, a2=2, ..., h8=64) with a piece
type from a vocabulary of 13 tokens:

    0  = empty
    1  = white pawn     7  = black pawn
    2  = white knight   8  = black knight
    3  = white bishop   9  = black bishop
    4  = white rook    10  = black rook
    5  = white queen   11  = black queen
    6  = white king    12  = black king

Board layout: rank 1 (white's home) first, file a-h within each rank.
Token index = (rank-1)*8 + file, so a1=1, b1=2, ..., h1=8, a2=9, ..., h8=64.

Side metadata (side to move, castling, en passant, halfmove) encoded as
a separate conditioning vector, not as tokens.

Move encoding: UCI string → (from_square, to_square, promotion) integer triple.
"""

export PIECE_VOCAB_SIZE, EMPTY, PIECE_CHARS, PIECE_TO_ID, ID_TO_PIECE
export NUM_SQUARES, MOVE_VOCAB_SIZE
export fen_to_board, board_to_fen
export fen_to_tokens, tokens_to_fen
export fen_to_metadata, parse_fen
export uci_to_move_id, move_id_to_uci
export encode_position, decode_position

# ============================================================================
# Piece vocabulary
# ============================================================================

const EMPTY = 0
const PIECE_VOCAB_SIZE = 13   # 0-12 inclusive
const NUM_SQUARES = 64

# FEN character → piece ID
const PIECE_TO_ID = Dict{Char, Int}(
    'P' => 1, 'N' => 2, 'B' => 3, 'R' => 4, 'Q' => 5, 'K' => 6,
    'p' => 7, 'n' => 8, 'b' => 9, 'r' => 10, 'q' => 11, 'k' => 12,
)

const ID_TO_PIECE = Dict{Int, Char}(v => k for (k, v) in PIECE_TO_ID)
ID_TO_PIECE[0] = '.'

const PIECE_CHARS = "PNBRQKpnbrqk"

# ============================================================================
# FEN parsing
# ============================================================================

"""
    fen_to_board(fen_piece_placement::String) → Vector{Int} (length 64)

Parse the piece placement part of a FEN string into a 64-element board array.
Index 1 = a1, index 64 = h8. Rank 1 first (white's home rank).
"""
function fen_to_board(piece_placement::AbstractString)
    board = zeros(Int, 64)
    ranks = split(piece_placement, '/')
    length(ranks) == 8 || throw(ArgumentError("FEN piece placement must have 8 ranks, got $(length(ranks))"))

    for (rank_from_top, rank_str) in enumerate(ranks)
        rank = 9 - rank_from_top  # FEN starts from rank 8 (top)
        file = 1
        for ch in rank_str
            if isdigit(ch)
                file += parse(Int, ch)
            else
                haskey(PIECE_TO_ID, ch) || throw(ArgumentError("Unknown piece character: $ch"))
                idx = (rank - 1) * 8 + file
                board[idx] = PIECE_TO_ID[ch]
                file += 1
            end
        end
    end
    return board
end

"""
    board_to_fen(board::Vector{Int}) → String

Convert a 64-element board array back to FEN piece placement string.
"""
function board_to_fen(board::AbstractVector{<:Integer})
    length(board) == 64 || throw(ArgumentError("Board must have 64 squares"))
    ranks = String[]
    for rank in 8:-1:1  # FEN starts from rank 8
        rank_str = IOBuffer()
        empty_count = 0
        for file in 1:8
            idx = (rank - 1) * 8 + file
            piece = board[idx]
            if piece == EMPTY
                empty_count += 1
            else
                if empty_count > 0
                    print(rank_str, empty_count)
                    empty_count = 0
                end
                print(rank_str, ID_TO_PIECE[piece])
            end
        end
        if empty_count > 0
            print(rank_str, empty_count)
        end
        push!(ranks, String(take!(rank_str)))
    end
    return join(ranks, '/')
end

"""
    parse_fen(fen::String) → (board, side_to_move, castling, en_passant, halfmove, fullmove)

Parse a complete FEN string into all components.
"""
function parse_fen(fen::AbstractString)
    parts = split(strip(fen))
    length(parts) >= 4 || throw(ArgumentError("FEN must have at least 4 fields, got $(length(parts))"))

    board = fen_to_board(parts[1])
    side_to_move = parts[2] == "w" ? 0 : 1  # 0=white, 1=black

    # Castling rights as 4-bit vector [K, Q, k, q]
    castling = zeros(Int, 4)
    castle_str = parts[3]
    if castle_str != "-"
        'K' in castle_str && (castling[1] = 1)
        'Q' in castle_str && (castling[2] = 1)
        'k' in castle_str && (castling[3] = 1)
        'q' in castle_str && (castling[4] = 1)
    end

    # En passant target square (0 = none, 1-64 = square index)
    ep_str = parts[4]
    en_passant = if ep_str == "-"
        0
    else
        file = Int(ep_str[1]) - Int('a') + 1
        rank = parse(Int, ep_str[2:2])
        (rank - 1) * 8 + file
    end

    halfmove = length(parts) >= 5 ? parse(Int, parts[5]) : 0
    fullmove = length(parts) >= 6 ? parse(Int, parts[6]) : 1

    return board, side_to_move, castling, en_passant, halfmove, fullmove
end

"""
    fen_to_tokens(fen::String) → Vector{Int} (length 64, values 1-13)

Convert FEN to token sequence. Adds 1 to piece IDs so tokens are in [1, 13]
(empty=1, wP=2, ..., bK=13). This is suitable for Lux.Embedding which expects
1-based indices.
"""
function fen_to_tokens(fen::AbstractString)
    board, _, _, _, _, _ = parse_fen(fen)
    return board .+ 1  # shift to 1-based for embedding lookup
end

"""
    fen_to_metadata(fen::String) → Vector{Float32} (length 6)

Extract side metadata as a conditioning vector:
[side_to_move, castle_K, castle_Q, castle_k, castle_q, en_passant_file]

en_passant_file is 0 if none, 1-8 for files a-h.
"""
function fen_to_metadata(fen::AbstractString)
    _, stm, castling, ep, _, _ = parse_fen(fen)
    ep_file = ep == 0 ? 0.0f0 : Float32(((ep - 1) % 8) + 1)
    return Float32[stm, castling[1], castling[2], castling[3], castling[4], ep_file]
end

# ============================================================================
# Move encoding
# ============================================================================

# Square name ↔ index
function square_to_index(sq::AbstractString)
    length(sq) == 2 || throw(ArgumentError("Square must be 2 chars: $sq"))
    file = Int(sq[1]) - Int('a') + 1
    rank = parse(Int, sq[2:2])
    (1 <= file <= 8 && 1 <= rank <= 8) || throw(ArgumentError("Invalid square: $sq"))
    return (rank - 1) * 8 + file
end

function index_to_square(idx::Int)
    (1 <= idx <= 64) || throw(ArgumentError("Square index must be 1-64, got $idx"))
    rank = (idx - 1) ÷ 8 + 1
    file = (idx - 1) % 8 + 1
    return string(Char(Int('a') + file - 1), rank)
end

# Promotion piece encoding: 0=none, 1=knight, 2=bishop, 3=rook, 4=queen
const PROMO_CHAR_TO_ID = Dict{Char, Int}('n' => 1, 'b' => 2, 'r' => 3, 'q' => 4)
const PROMO_ID_TO_CHAR = Dict{Int, Char}(v => k for (k, v) in PROMO_CHAR_TO_ID)

# Move vocabulary: from(64) × to(64) × promo(5) = 20480
# But we compress: move_id = (from-1)*64*5 + (to-1)*5 + promo
# promo: 0=none, 1-4=nbrq
const MOVE_VOCAB_SIZE = 64 * 64 * 5  # 20480

"""
    uci_to_move_id(uci::String) → Int (1-based)

Encode a UCI move string (e.g., "e2e4", "a7a8q") as an integer in [1, MOVE_VOCAB_SIZE].
"""
function uci_to_move_id(uci::AbstractString)
    uci = lowercase(strip(uci))
    len = length(uci)
    (4 <= len <= 5) || throw(ArgumentError("UCI move must be 4-5 chars: $uci"))

    from = square_to_index(uci[1:2])
    to = square_to_index(uci[3:4])
    promo = len == 5 ? get(PROMO_CHAR_TO_ID, uci[5], 0) : 0

    return (from - 1) * 64 * 5 + (to - 1) * 5 + promo + 1  # 1-based
end

"""
    move_id_to_uci(move_id::Int) → String

Decode an integer move ID back to UCI string.
"""
function move_id_to_uci(move_id::Int)
    (1 <= move_id <= MOVE_VOCAB_SIZE) || throw(ArgumentError("Move ID out of range: $move_id"))
    id0 = move_id - 1
    promo = id0 % 5
    id0 = id0 ÷ 5
    to = id0 % 64 + 1
    from = id0 ÷ 64 + 1

    uci = index_to_square(from) * index_to_square(to)
    if promo > 0
        uci *= string(PROMO_ID_TO_CHAR[promo])
    end
    return uci
end

# ============================================================================
# Combined encoding for training
# ============================================================================

"""
    encode_position(fen::String) → (tokens, metadata)

Full encoding for one position:
- tokens: Vector{Int} length 64, values in [1, 13] (for Embedding layer)
- metadata: Vector{Float32} length 6 (side, castling, ep)
"""
function encode_position(fen::AbstractString)
    return fen_to_tokens(fen), fen_to_metadata(fen)
end

"""
    decode_position(tokens::Vector{Int}, metadata::Vector{Float32}) → String

Reconstruct a FEN string from tokens + metadata (piece placement + side only).
"""
function decode_position(tokens::AbstractVector{<:Integer}, metadata::AbstractVector{<:Real})
    board = tokens .- 1  # back to 0-based piece IDs
    placement = board_to_fen(board)
    side = metadata[1] < 0.5 ? "w" : "b"

    castle = ""
    metadata[2] > 0.5 && (castle *= "K")
    metadata[3] > 0.5 && (castle *= "Q")
    metadata[4] > 0.5 && (castle *= "k")
    metadata[5] > 0.5 && (castle *= "q")
    isempty(castle) && (castle = "-")

    ep_file = round(Int, metadata[6])
    ep = if ep_file == 0
        "-"
    else
        ep_rank = metadata[1] < 0.5 ? 6 : 3  # white's ep target is rank 6, black's rank 3
        index_to_square((ep_rank - 1) * 8 + ep_file)
    end

    return "$placement $side $castle $ep 0 1"
end

end # module
