#!/usr/bin/env python3
"""
Download and convert REDFM into the JSONL format expected by train_re_gpu.jl.

By default this writes:
    data/rebel/train.jsonl
    data/rebel/validation.jsonl
    data/rebel/test.jsonl

Usage:
    python3 scripts/download_redfm.py
    python3 scripts/download_redfm.py --languages en
    python3 scripts/download_redfm.py --languages en,de,fr --output-dir data/rebel_multi
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

BASE_URL = "https://huggingface.co/datasets/Babelscape/REDFM/resolve/main/data"
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
SPLIT_MAP = {
    "train": "train",
    "dev": "validation",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and convert REDFM to Swamma RE JSONL.")
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated REDFM languages to download, e.g. en or en,de,fr",
    )
    parser.add_argument(
        "--output-dir",
        default="data/rebel",
        help="Directory for converted train/validation/test JSONL files.",
    )
    parser.add_argument(
        "--relation-label",
        choices=("uri", "surfaceform"),
        default="uri",
        help="Which REDFM predicate field to use as the relation label.",
    )
    return parser.parse_args()


def iter_jsonl(url: str) -> Iterable[dict]:
    with urllib.request.urlopen(url, timeout=60) as response:
        for raw_line in response:
            if not raw_line:
                continue
            line = raw_line.decode("utf-8").strip()
            if line:
                yield json.loads(line)


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def char_span_to_token_span(
    token_spans: List[Tuple[str, int, int]],
    start_char: int,
    end_char: int,
) -> Optional[Tuple[int, int]]:
    overlapping = [
        idx
        for idx, (_, tok_start, tok_end) in enumerate(token_spans)
        if tok_end > start_char and tok_start < end_char
    ]
    if not overlapping:
        return None
    return overlapping[0] + 1, overlapping[-1] + 1


def entity_key(entity: dict) -> Tuple:
    boundaries = entity.get("boundaries") or [None, None]
    return (
        entity.get("uri"),
        boundaries[0],
        boundaries[1],
        entity.get("surfaceform"),
        entity.get("type"),
    )


def relation_label(relation: dict, mode: str) -> Optional[str]:
    predicate = relation.get("predicate") or {}
    if mode == "uri":
        return predicate.get("uri")
    return predicate.get("surfaceform")


def convert_row(row: dict, label_mode: str) -> Optional[dict]:
    text = row.get("text") or ""
    token_spans = tokenize_with_spans(text)
    if not token_spans:
        return None

    tokens = [token for token, _, _ in token_spans]
    entities_out: List[dict] = []
    entity_index_map: Dict[Tuple, int] = {}

    for entity in row.get("entities", []):
        boundaries = entity.get("boundaries")
        if not boundaries or len(boundaries) != 2:
            continue
        span = char_span_to_token_span(token_spans, int(boundaries[0]), int(boundaries[1]))
        if span is None:
            continue
        key = entity_key(entity)
        if key in entity_index_map:
            continue
        entity_index_map[key] = len(entities_out) + 1
        entities_out.append(
            {
                "start": span[0],
                "stop": span[1],
                "label": entity.get("type", "MISC"),
                "uri": entity.get("uri"),
                "surfaceform": entity.get("surfaceform"),
            }
        )

    relations_out: List[dict] = []
    seen_relations = set()
    for relation in row.get("relations", []):
        subject = relation.get("subject") or {}
        obj = relation.get("object") or {}
        subj_idx = entity_index_map.get(entity_key(subject))
        obj_idx = entity_index_map.get(entity_key(obj))
        if subj_idx is None or obj_idx is None or subj_idx == obj_idx:
            continue
        label = relation_label(relation, label_mode)
        if not label:
            continue
        rel_key = (subj_idx, obj_idx, label)
        if rel_key in seen_relations:
            continue
        seen_relations.add(rel_key)
        relations_out.append(
            {
                "head": subj_idx,
                "tail": obj_idx,
                "label": label,
                "confidence": relation.get("confidence"),
            }
        )

    return {
        "docid": row.get("docid"),
        "title": row.get("title"),
        "uri": row.get("uri"),
        "language": row.get("lan"),
        "text": text,
        "tokens": tokens,
        "entities": entities_out,
        "relations": relations_out,
    }


def download_split(source_split: str, languages: List[str], output_path: str, label_mode: str) -> Dict[str, int]:
    stats = Counter()
    with open(output_path, "w", encoding="utf-8") as out:
        for language in languages:
            url = f"{BASE_URL}/{source_split}.{language}.jsonl"
            print(f"Downloading {url}", flush=True)
            for row in iter_jsonl(url):
                converted = convert_row(row, label_mode)
                if converted is None:
                    stats["skipped_empty"] += 1
                    continue
                if not converted["entities"]:
                    stats["rows_without_entities"] += 1
                if not converted["relations"]:
                    stats["rows_without_relations"] += 1
                stats["rows"] += 1
                stats["entities"] += len(converted["entities"])
                stats["relations"] += len(converted["relations"])
                out.write(json.dumps(converted, ensure_ascii=False) + "\n")
    return dict(stats)


def main() -> int:
    args = parse_args()
    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    if not languages:
        print("No languages provided.", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Languages: {', '.join(languages)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Relation label mode: {args.relation_label}")

    all_stats: Dict[str, Dict[str, int]] = {}
    for source_split, target_name in SPLIT_MAP.items():
        output_path = os.path.join(args.output_dir, f"{target_name}.jsonl")
        stats = download_split(source_split, languages, output_path, args.relation_label)
        all_stats[target_name] = stats
        print(
            f"{target_name}: rows={stats.get('rows', 0)} "
            f"entities={stats.get('entities', 0)} "
            f"relations={stats.get('relations', 0)} "
            f"rows_without_relations={stats.get('rows_without_relations', 0)}"
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
