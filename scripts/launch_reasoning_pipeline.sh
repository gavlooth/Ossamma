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
#   ./scripts/launch_reasoning_pipeline.sh --bounded-medium   # bounded end-to-end pipeline
#   ./scripts/launch_reasoning_pipeline.sh --phase 1          # chess only
#   ./scripts/launch_reasoning_pipeline.sh --phase 3a         # language only
#   ./scripts/launch_reasoning_pipeline.sh --smoke            # quick smoke test
#   ./scripts/launch_reasoning_pipeline.sh --phase 3a --bounded-medium
#   ./scripts/launch_reasoning_pipeline.sh --phase 3b --bounded-medium
#       practical bounded Phase 3b validation profile
#
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="all"
PHASE_NUM=""
SMOKE=false
RESUME=false
BOUNDED_MEDIUM=false
ARGS=("$@")
idx=0
while [ $idx -lt ${#ARGS[@]} ]; do
    arg="${ARGS[$idx]}"
    case "$arg" in
        --all)
            MODE="all"
            ;;
        --smoke)
            SMOKE=true
            ;;
        --resume)
            RESUME=true
            ;;
        --bounded-medium)
            BOUNDED_MEDIUM=true
            ;;
        --phase)
            MODE="phase"
            idx=$((idx + 1))
            if [ $idx -ge ${#ARGS[@]} ]; then
                echo "Missing value after --phase"
                exit 1
            fi
            PHASE_NUM="${ARGS[$idx]}"
            ;;
        *)
            echo "Usage: $0 [--all] [--smoke] [--bounded-medium] [--resume] [--phase <1|2|3a|3b>]"
            exit 1
            ;;
    esac
    idx=$((idx + 1))
done

if [ "$SMOKE" = true ] && [ "$BOUNDED_MEDIUM" = true ]; then
    echo "ERROR: --smoke and --bounded-medium are mutually exclusive."
    exit 1
fi

if [ "$BOUNDED_MEDIUM" = true ] && [ "$MODE" = "phase" ] && [ "${PHASE_NUM:-}" != "3a" ] && [ "${PHASE_NUM:-}" != "3b" ]; then
    echo "ERROR: --bounded-medium phase mode is currently supported only with --phase 3a or --phase 3b."
    exit 1
fi

CHECKPOINT_DIR="${REASONING_CHECKPOINT_DIR:-}"
if [ -z "$CHECKPOINT_DIR" ]; then
    if [ "$SMOKE" = true ]; then
        CHECKPOINT_DIR="checkpoints/reasoning_drafter_smoke"
    elif [ "$BOUNDED_MEDIUM" = true ]; then
        CHECKPOINT_DIR="checkpoints/reasoning_drafter_medium"
    else
        CHECKPOINT_DIR="checkpoints/reasoning_drafter"
    fi
fi
DATA_DIR="data"
mkdir -p "$CHECKPOINT_DIR"

echo "=================================================="
echo " ReasoningDrafter Training Pipeline"
echo " Target: NVIDIA Spark GB10 (130GB unified memory)"
echo "=================================================="
echo ""

require_file() {
    local path="$1"
    local message="$2"
    if [ ! -f "$path" ]; then
        echo "ERROR: $message"
        echo "Missing file: $path"
        exit 1
    fi
}

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
    local REQUIRED_REASONING_FILES=3
    if [ "$SMOKE" = true ]; then
        REQUIRED_REASONING_FILES=1
    elif [ "$BOUNDED_MEDIUM" = true ]; then
        REQUIRED_REASONING_FILES=2
    fi
    if [ ! -d "$DATA_DIR/reasoning" ] || [ "$(ls -1 "$DATA_DIR/reasoning"/*.jsonl 2>/dev/null | wc -l)" -lt "$REQUIRED_REASONING_FILES" ]; then
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
    local EXTRA_ARGS=()
    local RESUME_ARGS=()

    if [ "$SMOKE" = true ]; then
        DATA_FILE="$DATA_DIR/chess/smoke.jsonl"
        MAX_POS=128
        local PHASE1_MAX_STEPS=1
        EXTRA_ARGS=(
            --batch-size 8
            --learning-rate 1e-3
            --checkpoint-every 1
            --log-every 1
            --max-steps "$PHASE1_MAX_STEPS"
            --embedding-dim 64
            --heads 4
            --layers 2
            --time-dim 32
            --rc-code-dim 32
            --rc-codebook-size 64
            --rc-steps 4
            --frontend-wave-heads 2
            --circuit-leaves 8
            --circuit-sums 4
            --circuit-circuits 2
            --seed 41
        )
    elif [ "$BOUNDED_MEDIUM" = true ]; then
        DATA_FILE="$DATA_DIR/chess/smoke.jsonl"
        MAX_POS=128
        local PHASE1_MAX_STEPS=3
        EXTRA_ARGS=(
            --batch-size 8
            --learning-rate 1e-3
            --checkpoint-every 1
            --log-every 1
            --max-steps "$PHASE1_MAX_STEPS"
            --embedding-dim 64
            --heads 4
            --layers 2
            --time-dim 32
            --rc-code-dim 32
            --rc-codebook-size 64
            --rc-steps 4
            --frontend-wave-heads 2
            --circuit-leaves 8
            --circuit-sums 4
            --circuit-circuits 2
            --seed 41
        )
    fi

    if [ "$RESUME" = true ]; then
        require_file \
            "$CHECKPOINT_DIR/phase1/checkpoint_last.jld2" \
            "Phase 1 resume requested, but no checkpoint_last.jld2 exists in the selected checkpoint root. Run a fresh Phase 1 smoke first."
        RESUME_ARGS=(--resume "$CHECKPOINT_DIR/phase1/checkpoint_last.jld2")
    fi

    julia --project=. scripts/train_chess_reasoning.jl \
        --data "$DATA_FILE" \
        --max-positions "$MAX_POS" \
        --checkpoint-dir "$CHECKPOINT_DIR/phase1" \
        --steps 0 \
        "${RESUME_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"

    echo "Phase 1 complete. Checkpoint: $CHECKPOINT_DIR/phase1/"
    echo ""
}

# ---------------------------------------------------------------------------
# Phase 2: Transfer surgery
# ---------------------------------------------------------------------------
run_phase2() {
    echo "=== Phase 2: Transfer Surgery ==="
    require_file \
        "$CHECKPOINT_DIR/phase1/best.jld2" \
        "Phase 2 requires a Phase 1 best checkpoint. Run Phase 1 first in the same checkpoint root."
    if [ "$SMOKE" = true ] || [ "$BOUNDED_MEDIUM" = true ]; then
        julia --project=. scripts/transfer_surgery.jl \
            --input "$CHECKPOINT_DIR/phase1/best.jld2" \
            --output "$CHECKPOINT_DIR/phase2/surgery.jld2" \
            --target-vocab 132
    else
        julia --project=. scripts/transfer_surgery.jl \
            --input "$CHECKPOINT_DIR/phase1/best.jld2" \
            --output "$CHECKPOINT_DIR/phase2/surgery.jld2" \
            --target-vocab 49160
    fi

    echo "Phase 2 complete. Checkpoint: $CHECKPOINT_DIR/phase2/"
    echo ""
}

# ---------------------------------------------------------------------------
# Phase 3a: Language fine-tuning
# ---------------------------------------------------------------------------
run_phase3a() {
    echo "=== Phase 3a: Language Fine-Tuning ==="
    local CHECKPOINT_PATH="$CHECKPOINT_DIR/phase2/surgery.jld2"
    if [ "$RESUME" = true ]; then
        require_file \
            "$CHECKPOINT_DIR/phase3a/checkpoint_last.jld2" \
            "Phase 3a resume requested, but no Phase 3a checkpoint_last.jld2 exists in the selected checkpoint root. Run a fresh Phase 3a smoke first."
        CHECKPOINT_PATH="$CHECKPOINT_DIR/phase3a/checkpoint_last.jld2"
    else
        require_file \
            "$CHECKPOINT_DIR/phase2/surgery.jld2" \
            "Phase 3a requires a Phase 2 surgery checkpoint. Run Phase 2 first in the same checkpoint root."
    fi
    if [ "$SMOKE" = true ]; then
        local PHASE3A_MAX_STEPS=2
        julia --project=. scripts/train_reasoning_language.jl \
            --checkpoint "$CHECKPOINT_PATH" \
            --data-dir "$DATA_DIR/reasoning" \
            --output-dir "$CHECKPOINT_DIR/phase3a" \
            --epochs 2 \
            --batch-size 1 \
            --max-seq-length 64 \
            --max-per-dataset 1 \
            --max-steps "$PHASE3A_MAX_STEPS" \
            --checkpoint-every 1 \
            --log-every 1 \
            --seed 41
    elif [ "$BOUNDED_MEDIUM" = true ]; then
        echo "Using bounded-medium Phase 3a profile: batch_size=2, max_per_dataset=32, max_steps=40, max_seq_length=64"
        julia --project=. scripts/train_reasoning_language.jl \
            --checkpoint "$CHECKPOINT_PATH" \
            --data-dir "$DATA_DIR/reasoning" \
            --output-dir "$CHECKPOINT_DIR/phase3a" \
            --epochs 2 \
            --batch-size 2 \
            --max-seq-length 64 \
            --max-per-dataset 32 \
            --max-steps 40 \
            --checkpoint-every 4 \
            --log-every 1 \
            --seed 41
    else
        julia --project=. scripts/train_reasoning_language.jl \
            --checkpoint "$CHECKPOINT_PATH" \
            --data-dir "$DATA_DIR/reasoning" \
            --output-dir "$CHECKPOINT_DIR/phase3a" \
            --epochs 10
    fi

    echo "Phase 3a complete. Checkpoint: $CHECKPOINT_DIR/phase3a/"
    echo ""
}

# ---------------------------------------------------------------------------
# Phase 3b: Granite distillation
# ---------------------------------------------------------------------------
run_phase3b() {
    echo "=== Phase 3b: Granite Distillation ==="
    local DRAFTER_CHECKPOINT="$CHECKPOINT_DIR/phase3a/best.jld2"
    if [ "$RESUME" = true ]; then
        require_file \
            "$CHECKPOINT_DIR/phase3b/checkpoint_last.jld2" \
            "Phase 3b resume requested, but no Phase 3b checkpoint_last.jld2 exists in the selected checkpoint root. Run a fresh Phase 3b smoke first."
        DRAFTER_CHECKPOINT="$CHECKPOINT_DIR/phase3b/checkpoint_last.jld2"
    else
        require_file \
            "$CHECKPOINT_DIR/phase3a/best.jld2" \
            "Phase 3b requires a Phase 3a best checkpoint. Run Phase 3a first in the same checkpoint root."
    fi
    if [ "$SMOKE" = true ]; then
        julia --project=. scripts/distill_granite.jl \
            --drafter-checkpoint "$DRAFTER_CHECKPOINT" \
            --granite-model "ibm-granite/granite-4.0-micro" \
            --data-dir "$DATA_DIR/reasoning" \
            --output-dir "$CHECKPOINT_DIR/phase3b" \
            --epochs 1 \
            --batch-size 1 \
            --max-seq-length 64 \
            --max-per-dataset 1 \
            --max-steps 1 \
            --checkpoint-every 1 \
            --log-every 1 \
            --local-files-only true \
            --teacher-device gpu \
            --seed 41
    elif [ "$BOUNDED_MEDIUM" = true ]; then
        echo "Using bounded-medium Phase 3b profile: batch_size=2, max_per_dataset=32, max_steps=40, max_seq_length=64"
        julia --project=. scripts/distill_granite.jl \
            --drafter-checkpoint "$DRAFTER_CHECKPOINT" \
            --granite-model "ibm-granite/granite-4.0-micro" \
            --data-dir "$DATA_DIR/reasoning" \
            --output-dir "$CHECKPOINT_DIR/phase3b" \
            --epochs 2 \
            --batch-size 2 \
            --max-seq-length 64 \
            --max-per-dataset 32 \
            --max-steps 40 \
            --checkpoint-every 4 \
            --log-every 1 \
            --local-files-only true \
            --teacher-device gpu \
            --seed 41
    else
        julia --project=. scripts/distill_granite.jl \
            --drafter-checkpoint "$DRAFTER_CHECKPOINT" \
            --granite-model "ibm-granite/granite-4.0-micro" \
            --data-dir "$DATA_DIR/reasoning" \
            --output-dir "$CHECKPOINT_DIR/phase3b" \
            --epochs 5
    fi

    echo "Phase 3b complete. Final checkpoint: $CHECKPOINT_DIR/phase3b/"
    echo ""
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
case "$MODE" in
    all)
        prepare_data
        run_phase1
        run_phase2
        run_phase3a
        run_phase3b
        ;;
    phase)
        if [ -z "$PHASE_NUM" ]; then
            echo "Missing phase number (use --phase <1|2|3a|3b>)"
            exit 1
        fi
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
        echo "Usage: $0 [--all] [--smoke] [--bounded-medium] [--resume] [--phase <1|2|3a|3b>]"
        exit 1
        ;;
esac

FINAL_ARTIFACT="$CHECKPOINT_DIR/phase3b/"
if [ "$MODE" = "phase" ]; then
    case "${PHASE_NUM:-1}" in
        1) FINAL_ARTIFACT="$CHECKPOINT_DIR/phase1/" ;;
        2) FINAL_ARTIFACT="$CHECKPOINT_DIR/phase2/" ;;
        3a) FINAL_ARTIFACT="$CHECKPOINT_DIR/phase3a/" ;;
        3b) FINAL_ARTIFACT="$CHECKPOINT_DIR/phase3b/" ;;
    esac
elif [ "$SMOKE" = true ]; then
    FINAL_ARTIFACT="$CHECKPOINT_DIR/phase3b/"
fi

echo "=================================================="
echo " Pipeline complete!"
echo " Final model: $FINAL_ARTIFACT"
echo "=================================================="
