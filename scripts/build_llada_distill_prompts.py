#!/usr/bin/env python3
"""
Build a prompt manifest for offline PRIME/LLaDA distillation from raw text or JSONL.

The first intended use is continuation-style distillation: sample source text
fragments, cut a deterministic prefix, and ask the teacher to continue it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterator, List

SPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build distillation prompts from raw text.")
    parser.add_argument("--input", required=True, help="Raw corpus (.txt or .jsonl)")
    parser.add_argument("--output", required=True, help="Prompt manifest JSONL")
    parser.add_argument(
        "--text-field",
        default="text",
        help="Primary text field when reading JSONL",
    )
    parser.add_argument(
        "--alt-text-field",
        default="content",
        help="Fallback text field when reading JSONL",
    )
    parser.add_argument(
        "--task",
        default="continuation",
        choices=("continuation",),
        help="Prompt family to build",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a precise, fluent assistant. Continue the user's text naturally and coherently.",
        help="System prompt label to store alongside generated prompts",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=80,
        help="Minimum source text length in words before it can yield a prompt",
    )
    parser.add_argument(
        "--prefix-words",
        type=int,
        default=48,
        help="Prefix length in words used as the teacher prompt seed",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=5000,
        help="Maximum prompts to emit",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation ratio for deterministic split assignment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for source text shuffling",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.strip())


def iter_source_texts(path: Path, text_field: str, alt_text_field: str) -> Iterator[str]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = str(row.get(text_field, "") or row.get(alt_text_field, "")).strip()
                if text:
                    yield normalize_text(text)
        return

    raw = path.read_text(encoding="utf-8")
    paragraphs = [normalize_text(chunk) for chunk in raw.split("\n\n")]
    emitted = False
    for paragraph in paragraphs:
        if paragraph:
            emitted = True
            yield paragraph
    if not emitted:
        for line in raw.splitlines():
            line = normalize_text(line)
            if line:
                yield line


def stable_split(key: str, val_ratio: float) -> str:
    cutoff = int(val_ratio * 10000)
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 10000
    return "validation" if bucket < cutoff else "train"


def build_continuation_prompt(prefix_words: List[str]) -> str:
    prefix = " ".join(prefix_words).strip()
    return (
        "Continue the following passage in a coherent, informative style.\n\n"
        f"{prefix}"
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    rng = random.Random(args.seed)
    texts = list(iter_source_texts(input_path, args.text_field, args.alt_text_field))
    rng.shuffle(texts)

    prompts: List[Dict[str, object]] = []
    for source_index, text in enumerate(texts):
        words = text.split()
        if len(words) < max(args.min_words, args.prefix_words + 8):
            continue

        prefix_words = words[: args.prefix_words]
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        prompt = build_continuation_prompt(prefix_words)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        prompts.append(
            {
                "id": f"{args.task}_{source_index}",
                "prompt": prompt,
                "task": args.task,
                "split": stable_split(source_hash, args.val_ratio),
                "source_hash": source_hash,
                "prompt_hash": prompt_hash,
                "prefix_word_count": len(prefix_words),
                "source_word_count": len(words),
                "system_prompt": args.system_prompt,
            }
        )
        if len(prompts) >= args.max_prompts:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in prompts:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "prompts": len(prompts),
                "task": args.task,
                "prefix_words": args.prefix_words,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
