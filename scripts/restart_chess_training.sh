#!/usr/bin/env bash
# Auto-restart chess reasoning Phase 1 training with resume from checkpoint
# Polls exit status and relaunches until clean exit or manual Ctrl-C

set -u
cd "$(dirname "$0")/.."

DATA="${1:-data/chess/lichess_db_eval.jsonl}"
MAX_POS="${2:-10000000}"
CKPT_DIR="checkpoints/reasoning_drafter/phase1"
CKPT_LAST="$CKPT_DIR/checkpoint_last.jld2"
MAX_RETRIES=50
RETRY=0

mkdir -p "$CKPT_DIR"

while [ $RETRY -lt $MAX_RETRIES ]; do
    RETRY=$((RETRY + 1))

    # Build command — add --resume if checkpoint exists
    CMD="julia --project=. scripts/train_chess_reasoning.jl --data $DATA --max-positions $MAX_POS --checkpoint-dir $CKPT_DIR"
    if [ -f "$CKPT_LAST" ]; then
        CMD="$CMD --resume $CKPT_LAST"
        echo "=== Attempt $RETRY / $MAX_RETRIES  $(date)  [RESUMING] ==="
    else
        echo "=== Attempt $RETRY / $MAX_RETRIES  $(date)  [FRESH] ==="
    fi

    $CMD
    EXIT=$?

    if [ $EXIT -eq 0 ]; then
        echo "=== Clean exit. Training complete. ==="
        exit 0
    fi

    echo "=== Crashed (exit $EXIT). Restarting in 5s... ==="
    sleep 5
done

echo "=== Exhausted $MAX_RETRIES retries. Giving up. ==="
exit 1
