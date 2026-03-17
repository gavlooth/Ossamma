#!/usr/bin/env bash
# =============================================================================
# ReasoningDrafter Full Training Pipeline
# =============================================================================
# Target: NVIDIA Spark GB10, 130GB unified memory, CUDA 13.0, aarch64
#
# This script runs the complete 4-phase training pipeline:
#   Phase 1: Chess pre-training (learn logic from Stockfish evaluations)
#   Phase 2: Transfer surgery (freeze backbone, add adapters, swap vocab)
#   Phase 3a: Language fine-tuning (reasoning datasets)
#   Phase 3b: Granite distillation (match verifier distribution)
#
# Prerequisites:
#   - Julia 1.12+ with project deps: julia --project=. -e 'using Pkg; Pkg.instantiate()'
#   - Python 3 with: pip install datasets (for reasoning dataset download)
#   - zstd for decompressing Lichess data: sudo apt install zstd
#   - ~50GB disk for Lichess data, ~1GB for reasoning datasets
#   - Granite model weights (downloaded automatically via HuggingFace)
#
# Usage:
#   ./scripts/launch_reasoning_pipeline.sh                    # full pipeline
#   ./scripts/launch_reasoning_pipeline.sh --phase 1          # chess only
#   ./scripts/launch_reasoning_pipeline.sh --phase 3a         # language only
#   ./scripts/launch_reasoning_pipeline.sh --smoke            # quick smoke test
#
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:---all}"
CHECKPOINT_DIR="checkpoints/reasoning_drafter"
DATA_DIR="data"
mkdir -p "$CHECKPOINT_DIR"

echo "=================================================="
echo " ReasoningDrafter Training Pipeline"
echo " Target: NVIDIA Spark GB10 (130GB unified memory)"
echo "=================================================="
echo ""

# ---------------------------------------------------------------------------
# Phase 0: Data preparation
# ---------------------------------------------------------------------------
prepare_data() {
    echo "=== Phase 0: Data Preparation ==="

    # Chess data
    if [ ! -f "$DATA_DIR/chess/lichess_db_eval.jsonl" ]; then
        echo "Downloading Lichess evaluation database (~19GB compressed)..."
        mkdir -p "$DATA_DIR/chess"
        if [ ! -f "$DATA_DIR/chess/lichess_db_eval.jsonl.zst" ]; then
            curl -L -o "$DATA_DIR/chess/lichess_db_eval.jsonl.zst" \
                https://database.lichess.org/lichess_db_eval.jsonl.zst
        fi
        echo "Decompressing (requires zstd)..."
        zstd -d "$DATA_DIR/chess/lichess_db_eval.jsonl.zst" -o "$DATA_DIR/chess/lichess_db_eval.jsonl"
        rm -f "$DATA_DIR/chess/lichess_db_eval.jsonl.zst"
    fi
    echo "Chess data: $(wc -l < "$DATA_DIR/chess/lichess_db_eval.jsonl") positions"

    # Create sample for smoke testing
    if [ ! -f "$DATA_DIR/chess/sample_100k.jsonl" ]; then
        head -100000 "$DATA_DIR/chess/lichess_db_eval.jsonl" > "$DATA_DIR/chess/sample_100k.jsonl"
    fi

    # Reasoning datasets
    if [ ! -d "$DATA_DIR/reasoning" ] || [ "$(ls -1 "$DATA_DIR/reasoning"/*.jsonl 2>/dev/null | wc -l)" -lt 3 ]; then
        echo "Downloading reasoning datasets..."
        ./scripts/download_reasoning_datasets.sh "$DATA_DIR/reasoning"
    fi
    echo "Reasoning data: $(cat "$DATA_DIR/reasoning"/*.jsonl 2>/dev/null | wc -l) examples"
    echo ""
}

# ---------------------------------------------------------------------------
# Phase 1: Chess pre-training
# ---------------------------------------------------------------------------
run_phase1() {
    echo "=== Phase 1: Chess Logic Pre-Training ==="
    local DATA_FILE="$DATA_DIR/chess/lichess_db_eval.jsonl"
    local MAX_POS=10000000

    if [ "$PHASE" = "--smoke" ]; then
        DATA_FILE="$DATA_DIR/chess/sample_100k.jsonl"
        MAX_POS=10000
    fi

    julia --project=. scripts/train_chess_reasoning.jl \
        --data "$DATA_FILE" \
        --max-positions "$MAX_POS" \
        --checkpoint-dir "$CHECKPOINT_DIR/phase1" \
        --steps 0

    echo "Phase 1 complete. Checkpoint: $CHECKPOINT_DIR/phase1/"
    echo ""
}

# ---------------------------------------------------------------------------
# Phase 2: Transfer surgery
# ---------------------------------------------------------------------------
run_phase2() {
    echo "=== Phase 2: Transfer Surgery ==="
    julia --project=. scripts/transfer_surgery.jl \
        --input "$CHECKPOINT_DIR/phase1/best.jld2" \
        --output "$CHECKPOINT_DIR/phase2/surgery.jld2" \
        --target-vocab 49160

    echo "Phase 2 complete. Checkpoint: $CHECKPOINT_DIR/phase2/"
    echo ""
}

# ---------------------------------------------------------------------------
# Phase 3a: Language fine-tuning
# ---------------------------------------------------------------------------
run_phase3a() {
    echo "=== Phase 3a: Language Fine-Tuning ==="
    julia --project=. scripts/train_reasoning_language.jl \
        --checkpoint "$CHECKPOINT_DIR/phase2/surgery.jld2" \
        --data-dir "$DATA_DIR/reasoning" \
        --output-dir "$CHECKPOINT_DIR/phase3a" \
        --epochs 10

    echo "Phase 3a complete. Checkpoint: $CHECKPOINT_DIR/phase3a/"
    echo ""
}

# ---------------------------------------------------------------------------
# Phase 3b: Granite distillation
# ---------------------------------------------------------------------------
run_phase3b() {
    echo "=== Phase 3b: Granite Distillation ==="
    julia --project=. scripts/distill_granite.jl \
        --drafter-checkpoint "$CHECKPOINT_DIR/phase3a/best.jld2" \
        --granite-model "ibm-granite/granite-4.0-micro" \
        --data-dir "$DATA_DIR/reasoning" \
        --output-dir "$CHECKPOINT_DIR/phase3b" \
        --epochs 5

    echo "Phase 3b complete. Final checkpoint: $CHECKPOINT_DIR/phase3b/"
    echo ""
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
case "$PHASE" in
    --all)
        prepare_data
        run_phase1
        run_phase2
        run_phase3a
        run_phase3b
        ;;
    --smoke)
        prepare_data
        run_phase1
        ;;
    --phase)
        PHASE_NUM="${2:-1}"
        prepare_data
        case "$PHASE_NUM" in
            1)  run_phase1 ;;
            2)  run_phase2 ;;
            3a) run_phase3a ;;
            3b) run_phase3b ;;
            *)  echo "Unknown phase: $PHASE_NUM (use 1, 2, 3a, 3b)"; exit 1 ;;
        esac
        ;;
    *)
        echo "Usage: $0 [--all|--smoke|--phase <1|2|3a|3b>]"
        exit 1
        ;;
esac

echo "=================================================="
echo " Pipeline complete!"
echo " Final model: $CHECKPOINT_DIR/phase3b/"
echo "=================================================="
