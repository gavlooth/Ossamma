#!/usr/bin/env python3
"""
Package cleaned teacher corpus rows into the JSONL format expected by
scripts/train_llada_canonical.jl.

The canonical trainer only needs a `text` field, so this script renders either:
- prompt + teacher response
- teacher response only

It can also mix in a raw corpus for the first full offline-distillation run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package cleaned teacher corpus for PRIME/LLaDA training.")
    parser.add_argument("--teacher-train", required=True, help="Cleaned teacher-train JSONL")
    parser.add_argument("--teacher-val", required=True, help="Cleaned teacher-val JSONL")
    parser.add_argument("--output-train", required=True, help="Packaged train JSONL")
    parser.add_argument("--output-val", required=True, help="Packaged val JSONL")
    parser.add_argument(
        "--render-mode",
        choices=("prompt_response", "response_only"),
        default="prompt_response",
        help="How to render teacher rows into final training text",
    )
    parser.add_argument(
        "--raw-train",
        default="",
        help="Optional raw train corpus (.txt or .jsonl with text/content field)",
    )
    parser.add_argument(
        "--raw-val",
        default="",
        help="Optional raw val corpus (.txt or .jsonl with text/content field)",
    )
    parser.add_argument(
        "--teacher-repeat",
        type=int,
        default=1,
        help="How many times to repeat teacher rows in the packaged corpus",
    )
    parser.add_argument(
        "--raw-repeat",
        type=int,
        default=1,
        help="How many times to repeat raw rows in the packaged corpus",
    )
    parser.add_argument(
        "--manifest-output",
        default="",
        help="Optional JSON path for packaging manifest",
    )
    parser.add_argument(
        "--fallback-val-ratio",
        type=float,
        default=0.05,
        help="When teacher-val is empty, split this ratio from teacher-train into validation",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_raw_texts(path: Path) -> List[str]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        texts: List[str] = []
        for row in iter_jsonl(path):
            text = str(row.get("text", "") or row.get("content", "")).strip()
            if text:
                texts.append(text)
        return texts

    raw = path.read_text(encoding="utf-8")
    texts = [chunk.strip() for chunk in raw.split("\n\n") if chunk.strip()]
    if texts:
        return texts
    return [line.strip() for line in raw.splitlines() if line.strip()]


def render_teacher_row(row: Dict[str, object], mode: str) -> str:
    prompt = str(row.get("prompt", "")).strip()
    teacher_text = str(row.get("teacher_text", "")).strip()
    if mode == "response_only":
        return teacher_text
    return f"User: {prompt}\n\nAssistant: {teacher_text}"


def stable_bucket(key: str, val_ratio: float) -> str:
    cutoff = int(val_ratio * 10000)
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 10000
    return "validation" if bucket < cutoff else "train"


def ensure_teacher_validation(
    teacher_train_rows: List[Dict[str, object]],
    teacher_val_rows: List[Dict[str, object]],
    fallback_val_ratio: float,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if teacher_val_rows or len(teacher_train_rows) <= 1:
        return teacher_train_rows, teacher_val_rows

    promoted_val: List[Dict[str, object]] = []
    kept_train: List[Dict[str, object]] = []

    for row in teacher_train_rows:
        key = str(row.get("prompt_hash", "")) or str(row.get("prompt", "")) or json.dumps(row, sort_keys=True)
        if stable_bucket(key, fallback_val_ratio) == "validation":
            promoted_val.append(row)
        else:
            kept_train.append(row)

    if not promoted_val:
        promoted_val.append(teacher_train_rows[-1])
        kept_train = teacher_train_rows[:-1]

    if not kept_train:
        kept_train.append(promoted_val.pop())
        promoted_val.append(teacher_train_rows[0])

    return kept_train, promoted_val


def write_packaged_output(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()

    teacher_train_path = Path(args.teacher_train)
    teacher_val_path = Path(args.teacher_val)
    if not teacher_train_path.exists():
        raise SystemExit(f"Teacher train file not found: {teacher_train_path}")
    if not teacher_val_path.exists():
        raise SystemExit(f"Teacher val file not found: {teacher_val_path}")

    teacher_train_rows = list(iter_jsonl(teacher_train_path))
    teacher_val_rows = list(iter_jsonl(teacher_val_path))
    teacher_train_rows, teacher_val_rows = ensure_teacher_validation(
        teacher_train_rows,
        teacher_val_rows,
        args.fallback_val_ratio,
    )
    raw_train_texts = load_raw_texts(Path(args.raw_train)) if args.raw_train else []
    raw_val_texts = load_raw_texts(Path(args.raw_val)) if args.raw_val else []

    packaged_train: List[Dict[str, object]] = []
    packaged_val: List[Dict[str, object]] = []

    for _ in range(max(args.teacher_repeat, 1)):
        for row in teacher_train_rows:
            packaged_train.append(
                {
                    "text": render_teacher_row(row, args.render_mode),
                    "source": "teacher",
                    "prompt_hash": row.get("prompt_hash"),
                    "source_task": row.get("source_task"),
                    "teacher_model": row.get("teacher_model"),
                }
            )
        for row in teacher_val_rows:
            packaged_val.append(
                {
                    "text": render_teacher_row(row, args.render_mode),
                    "source": "teacher",
                    "prompt_hash": row.get("prompt_hash"),
                    "source_task": row.get("source_task"),
                    "teacher_model": row.get("teacher_model"),
                }
            )

    for _ in range(max(args.raw_repeat, 1)):
        for text in raw_train_texts:
            packaged_train.append({"text": text, "source": "raw"})
        for text in raw_val_texts:
            packaged_val.append({"text": text, "source": "raw"})

    output_train = Path(args.output_train)
    output_val = Path(args.output_val)
    write_packaged_output(packaged_train, output_train)
    write_packaged_output(packaged_val, output_val)

    manifest = {
        "teacher_train_rows": len(teacher_train_rows),
        "teacher_val_rows": len(teacher_val_rows),
        "raw_train_rows": len(raw_train_texts),
        "raw_val_rows": len(raw_val_texts),
        "teacher_repeat": args.teacher_repeat,
        "raw_repeat": args.raw_repeat,
        "fallback_val_ratio": args.fallback_val_ratio,
        "render_mode": args.render_mode,
        "packaged_train_rows": len(packaged_train),
        "packaged_val_rows": len(packaged_val),
        "output_train": str(output_train),
        "output_val": str(output_val),
    }

    if args.manifest_output:
        manifest_path = Path(args.manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
