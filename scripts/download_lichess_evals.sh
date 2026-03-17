#!/usr/bin/env bash
# Download the Lichess evaluation database
# Source: https://database.lichess.org/#evals
#
# The file is ~3GB compressed, ~30GB uncompressed (JSONL format).
# Each line: {"fen": "...", "evals": [{"pvs": [{"cp": 15, "line": "e2e4 ..."}], "depth": 20}]}
#
# Usage:
#   ./scripts/download_lichess_evals.sh [output_dir]
#   Default output: data/chess/

set -euo pipefail

OUTPUT_DIR="${1:-data/chess}"
mkdir -p "$OUTPUT_DIR"

URL="https://database.lichess.org/lichess_db_eval.jsonl.zst"
COMPRESSED="$OUTPUT_DIR/lichess_db_eval.jsonl.zst"
OUTPUT="$OUTPUT_DIR/lichess_db_eval.jsonl"

if [ -f "$OUTPUT" ]; then
    echo "Already exists: $OUTPUT"
    echo "Lines: $(wc -l < "$OUTPUT")"
    exit 0
fi

echo "Downloading Lichess evaluation database..."
echo "URL: $URL"
echo "Output: $COMPRESSED"

if command -v curl &>/dev/null; then
    curl -L -o "$COMPRESSED" "$URL"
elif command -v wget &>/dev/null; then
    wget -O "$COMPRESSED" "$URL"
else
    echo "ERROR: Neither curl nor wget found. Install one and retry."
    exit 1
fi

echo "Decompressing (zstd)..."
if command -v zstd &>/dev/null; then
    zstd -d "$COMPRESSED" -o "$OUTPUT"
    rm "$COMPRESSED"
elif command -v unzstd &>/dev/null; then
    unzstd "$COMPRESSED" -o "$OUTPUT"
    rm "$COMPRESSED"
else
    echo "ERROR: zstd not found. Install with: sudo pacman -S zstd (or apt install zstd)"
    echo "Compressed file kept at: $COMPRESSED"
    exit 1
fi

echo "Done. Output: $OUTPUT"
echo "Lines: $(wc -l < "$OUTPUT")"
echo ""
echo "To create a small sample for testing:"
echo "  head -100000 $OUTPUT > $OUTPUT_DIR/lichess_eval_sample_100k.jsonl"
