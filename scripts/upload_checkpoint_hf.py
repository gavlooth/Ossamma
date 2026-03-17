#!/usr/bin/env python3
"""
Upload Swamma RE checkpoint to HuggingFace Hub.

Usage:
    python3 scripts/upload_checkpoint_hf.py \
        --checkpoint checkpoints/redfm_1b_distill_qwen7b/checkpoint_step_1000.jls \
        --repo christos-chatzifountas/swamma-re-1b \
        --commit "step 1000, loss 1.55"

    # Auto-upload latest checkpoint:
    python3 scripts/upload_checkpoint_hf.py \
        --checkpoint-dir checkpoints/redfm_1b_distill_qwen7b \
        --repo christos-chatzifountas/swamma-re-1b
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload Swamma checkpoint to HF Hub.")
    parser.add_argument("--checkpoint", help="Path to specific .jls checkpoint")
    parser.add_argument("--checkpoint-dir", help="Directory to find latest checkpoint")
    parser.add_argument("--repo", required=True, help="HF repo id (e.g. user/swamma-re-1b)")
    parser.add_argument("--commit", default="", help="Commit message")
    parser.add_argument("--config", default="configs/redfm_1b_distill_qwen7b.toml", help="Config file to include")
    parser.add_argument("--log", default="logs/train_1b_rebel_full.log", help="Training log to include")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir: str) -> Path | None:
    d = Path(checkpoint_dir)
    if not d.exists():
        return None
    checkpoints = sorted(d.glob("checkpoint_step_*.jls"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        return None
    return checkpoints[-1]


def main() -> int:
    args = parse_args()
    api = HfApi()

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.checkpoint_dir:
        ckpt_path = find_latest_checkpoint(args.checkpoint_dir)
        if ckpt_path is None:
            print(f"No checkpoints found in {args.checkpoint_dir}")
            return 1
    else:
        print("Provide --checkpoint or --checkpoint-dir")
        return 1

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    print(f"Checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / 1e9:.2f} GB)")

    # Create repo if needed
    try:
        create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
        print(f"Repo: https://huggingface.co/{args.repo}")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload checkpoint
    commit_msg = args.commit or f"Upload {ckpt_path.name}"
    print(f"Uploading checkpoint...")
    api.upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo=f"checkpoints/{ckpt_path.name}",
        repo_id=args.repo,
        commit_message=commit_msg,
    )
    print(f"  Uploaded: checkpoints/{ckpt_path.name}")

    # Upload best checkpoint if exists
    best_path = ckpt_path.parent / "checkpoint_best.jls"
    if best_path.exists():
        api.upload_file(
            path_or_fileobj=str(best_path),
            path_in_repo="checkpoints/checkpoint_best.jls",
            repo_id=args.repo,
            commit_message=commit_msg,
        )
        print(f"  Uploaded: checkpoints/checkpoint_best.jls")

    # Upload config
    config_path = Path(args.config)
    if config_path.exists():
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="config.toml",
            repo_id=args.repo,
            commit_message=commit_msg,
        )
        print(f"  Uploaded: config.toml")

    # Upload training log (tail only — full log can be huge)
    log_path = Path(args.log)
    if log_path.exists():
        # Upload last 1000 lines
        lines = log_path.read_text().splitlines()
        tail = "\n".join(lines[-1000:])
        tail_path = Path("/tmp/train_log_tail.txt")
        tail_path.write_text(tail)
        api.upload_file(
            path_or_fileobj=str(tail_path),
            path_in_repo="train_log.txt",
            repo_id=args.repo,
            commit_message=commit_msg,
        )
        print(f"  Uploaded: train_log.txt (last {min(1000, len(lines))} lines)")

    # Upload a model card
    card = f"""---
tags:
- relation-extraction
- swamma
- julia
- lux
license: apache-2.0
---

# Swamma RE 1B

Relation extraction model based on the Swamma architecture (Spectral Wave-PDE Attention Masked Mixer).

## Architecture
- **Parameters**: ~882M
- **Backbone**: d=1408, L=28, H=22 SwammaBlocks with hierarchical wave-gate frequencies
- **Decoder**: pair_evidence_mlp with edge_retrieval_v2 pair proposer
- **Training data**: Full REBEL dataset (2.49M rows, 1144 relation types)
- **Framework**: Lux.jl + Zygote.jl (Julia)

## Key Features
- O(T log T) sequence complexity (vs O(T²) for transformers)
- Per-layer spectral diversity via hierarchical wave-PDE frequencies
- Single-pass span + pair classification (no autoregressive decoding)

## Checkpoint Format
Julia Serialization (.jls) — load with:
```julia
using Serialization
ckpt = deserialize("checkpoint_step_N.jls")
params = ckpt[:params]
state = ckpt[:state]
model_config = ckpt[:model_config]
```
"""
    card_path = Path("/tmp/README.md")
    card_path.write_text(card)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.repo,
        commit_message=commit_msg,
    )
    print(f"  Uploaded: README.md")

    print(f"\nDone! https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
