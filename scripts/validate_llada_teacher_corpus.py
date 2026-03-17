#!/usr/bin/env python3
"""
Validate and clean a generated teacher corpus for PRIME/LLaDA distillation.

This script removes empty/short rows, exact duplicates, high-frequency
boilerplate, and optional near-duplicates, then writes cleaned train/val JSONL
files suitable for subsequent packaging into canonical PRIME training corpora.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
SPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and clean teacher corpus JSONL.")
    parser.add_argument("--input", required=True, help="Generated teacher corpus JSONL")
    parser.add_argument("--output-train", required=True, help="Cleaned train JSONL output")
    parser.add_argument("--output-val", required=True, help="Cleaned val JSONL output")
    parser.add_argument(
        "--rejections-output",
        default="",
        help="Optional JSONL path for rejected rows",
    )
    parser.add_argument(
        "--stats-output",
        default="",
        help="Optional JSON path for validation stats",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=32,
        help="Reject teacher_text shorter than this many characters",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=8,
        help="Reject teacher_text shorter than this many word tokens",
    )
    parser.add_argument(
        "--max-normalized-repeat",
        type=int,
        default=3,
        help="Reject rows whose normalized teacher text appears more than this many times",
    )
    parser.add_argument(
        "--near-dup-hamming-bits",
        type=int,
        default=3,
        help="Max SimHash Hamming distance to treat as near-duplicate; <0 disables",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation ratio used when no explicit split field is present",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.strip().lower())


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def stable_bucket(prompt_hash: str, val_ratio: float) -> str:
    cutoff = int(val_ratio * 10000)
    bucket = int(hashlib.sha256(prompt_hash.encode("utf-8")).hexdigest()[:8], 16) % 10000
    return "val" if bucket < cutoff else "train"


def simhash(text: str) -> int:
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        return 0
    shingles = tokens if len(tokens) < 3 else [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    weights = [0] * 64
    for shingle in shingles:
        digest = hashlib.sha256(shingle.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big")
        for bit in range(64):
            weights[bit] += 1 if ((value >> bit) & 1) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def maybe_near_duplicate(
    normalized: str,
    max_hamming_bits: int,
    buckets: Dict[int, List[int]],
) -> bool:
    if max_hamming_bits < 0:
        return False
    code = simhash(normalized)
    bucket_key = code >> 48
    candidates = buckets[bucket_key]
    for prev_code in candidates:
        if hamming_distance(code, prev_code) <= max_hamming_bits:
            return True
    candidates.append(code)
    return False


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    rows = list(iter_jsonl(input_path))
    normalized_counts = Counter(
        normalize_text(str(row.get("teacher_text", "")))
        for row in rows
        if str(row.get("teacher_text", "")).strip()
    )

    accepted_train: List[Dict[str, object]] = []
    accepted_val: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []

    exact_seen: set[Tuple[str, str]] = set()
    near_dup_buckets: Dict[int, List[int]] = defaultdict(list)
    stats = Counter(total_rows=len(rows))
    source_task_counts = Counter()
    split_counts = Counter()
    word_length_buckets = Counter()

    for row in rows:
        prompt_hash = str(row.get("prompt_hash", "")).strip()
        teacher_text = str(row.get("teacher_text", "")).strip()
        normalized = normalize_text(teacher_text)

        reject_reason = None
        if not teacher_text:
            reject_reason = "empty_teacher_text"
        elif len(teacher_text) < args.min_chars:
            reject_reason = "too_short_chars"
        elif word_count(teacher_text) < args.min_words:
            reject_reason = "too_short_words"
        elif normalized_counts[normalized] > args.max_normalized_repeat:
            reject_reason = "boilerplate_repeat"
        else:
            exact_key = (prompt_hash, normalized)
            if exact_key in exact_seen:
                reject_reason = "exact_duplicate"
            else:
                exact_seen.add(exact_key)
                if maybe_near_duplicate(normalized, args.near_dup_hamming_bits, near_dup_buckets):
                    reject_reason = "near_duplicate"

        if reject_reason is not None:
            stats[f"rejected_{reject_reason}"] += 1
            rejected.append(
                {
                    "reject_reason": reject_reason,
                    "row": row,
                }
            )
            continue

        source_split = str(row.get("source_split", "")).strip().lower()
        target_split = source_split if source_split in {"train", "val", "validation"} else stable_bucket(prompt_hash or normalized, args.val_ratio)
        cleaned = dict(row)
        wc = word_count(teacher_text)
        cleaned["teacher_text_normalized"] = normalized
        cleaned["teacher_text_word_count"] = wc
        source_task = str(cleaned.get("source_task", "general"))
        source_task_counts[source_task] += 1
        split_counts[target_split] += 1
        if wc < 32:
            word_length_buckets["lt32"] += 1
        elif wc < 64:
            word_length_buckets["32_63"] += 1
        elif wc < 128:
            word_length_buckets["64_127"] += 1
        elif wc < 256:
            word_length_buckets["128_255"] += 1
        else:
            word_length_buckets["ge256"] += 1

        if target_split in {"val", "validation"}:
            accepted_val.append(cleaned)
        else:
            accepted_train.append(cleaned)

    output_train = Path(args.output_train)
    output_val = Path(args.output_val)
    output_train.parent.mkdir(parents=True, exist_ok=True)
    output_val.parent.mkdir(parents=True, exist_ok=True)

    with output_train.open("w", encoding="utf-8") as handle:
        for row in accepted_train:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with output_val.open("w", encoding="utf-8") as handle:
        for row in accepted_val:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.rejections_output:
        rejection_path = Path(args.rejections_output)
        rejection_path.parent.mkdir(parents=True, exist_ok=True)
        with rejection_path.open("w", encoding="utf-8") as handle:
            for row in rejected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(input_path),
        "output_train": str(output_train),
        "output_val": str(output_val),
        "accepted_train": len(accepted_train),
        "accepted_val": len(accepted_val),
        "rejected": len(rejected),
        "stats": dict(stats),
        "split_balance": dict(split_counts),
        "task_balance": dict(source_task_counts),
        "word_length_distribution": dict(word_length_buckets),
    }

    if args.stats_output:
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
