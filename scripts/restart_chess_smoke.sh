#!/usr/bin/env bash
# Auto-restart chess reasoning smoke test on crash
# Polls exit status and relaunches until clean exit or manual Ctrl-C

set -u
cd "$(dirname "$0")/.."

CMD="julia --project=. scripts/train_chess_reasoning.jl --data data/chess/smoke.jsonl --max-positions 1000 --steps 50 --checkpoint-dir checkpoints/reasoning_drafter/phase1_smoke"
MAX_RETRIES=10
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    RETRY=$((RETRY + 1))
    echo "=== Attempt $RETRY / $MAX_RETRIES  $(date) ==="
    $CMD
    EXIT=$?
    if [ $EXIT -eq 0 ]; then
        echo "=== Clean exit. Done. ==="
        exit 0
    fi
    echo "=== Crashed (exit $EXIT). Restarting in 5s... ==="
    sleep 5
done

echo "=== Exhausted $MAX_RETRIES retries. Giving up. ==="
exit 1
