#!/usr/bin/env python3
"""
Convert the full Babelscape/rebel-dataset from character-boundary format
to token-indexed span format compatible with the Swamma RE training pipeline.

Input:  REBEL JSONL with character-level boundaries and triples
Output: Span-indexed JSONL with token-level entities and relations

Usage:
    python3 scripts/convert_rebel_full_to_spans.py \
        --input data/rebel_full/en_train.jsonl \
        --output data/rebel_full/train.jsonl \
        --max-rows 0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert full REBEL to span-indexed format.")
    parser.add_argument("--input", required=True, help="Input REBEL JSONL")
    parser.add_argument("--output", required=True, help="Output span-indexed JSONL")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows (0=all)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Drop rows longer than this")
    parser.add_argument("--min-triples", type=int, default=1, help="Require at least N triples")
    return parser.parse_args()


def tokenize_simple(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer matching REBEL conventions."""
    # Split on whitespace, then separate leading/trailing punctuation
    raw = text.split()
    tokens = []
    for word in raw:
        # Split punctuation from word boundaries
        # e.g. "hello," -> ["hello", ","], "(test)" -> ["(", "test", ")"]
        parts = re.findall(r"""[A-Za-z0-9]+(?:'[A-Za-z]+)*|[^\s]""", word)
        tokens.extend(parts)
    return tokens


def build_char_to_token_map(text: str, tokens: list[str]) -> list[int]:
    """Map each character position to its token index (1-based), or 0 if whitespace."""
    char_map = [0] * len(text)
    pos = 0
    for tok_idx, token in enumerate(tokens):
        # Find this token in the text starting from pos
        loc = text.find(token, pos)
        if loc == -1:
            # Try case-insensitive
            loc = text.lower().find(token.lower(), pos)
        if loc == -1:
            # Skip unaligned token
            continue
        for c in range(loc, loc + len(token)):
            if c < len(text):
                char_map[c] = tok_idx + 1  # 1-based
        pos = loc + len(token)
    return char_map


def char_span_to_token_span(
    boundaries: list[int],
    char_map: list[int],
) -> Optional[tuple[int, int]]:
    """Convert [char_start, char_end) to (token_start, token_stop) 1-based inclusive."""
    if boundaries is None or len(boundaries) != 2:
        return None
    char_start, char_end = boundaries
    if char_start < 0 or char_end < 0:
        return None
    char_end = min(char_end, len(char_map) - 1)
    char_start = min(char_start, len(char_map) - 1)

    # Find first non-zero token in the char range
    tok_start = 0
    for c in range(char_start, min(char_end + 1, len(char_map))):
        if char_map[c] > 0:
            tok_start = char_map[c]
            break
    if tok_start == 0:
        return None

    # Find last non-zero token in the char range
    tok_stop = tok_start
    for c in range(min(char_end, len(char_map) - 1), char_start - 1, -1):
        if char_map[c] > 0:
            tok_stop = char_map[c]
            break

    if tok_start > tok_stop:
        tok_start, tok_stop = tok_stop, tok_start

    return (tok_start, tok_stop)


def entity_label_from_uri(uri: str) -> str:
    """Heuristic entity type from Wikidata URI. Falls back to MISC."""
    # We don't have entity types in the full REBEL data — only Wikidata URIs.
    # Return a generic label; the model learns entity types from context.
    return "ENTITY"


def convert_row(row: dict, max_tokens: int) -> Optional[dict]:
    """Convert one REBEL row to span-indexed format."""
    text = row.get("text", "")
    if not text:
        return None

    tokens = tokenize_simple(text)
    if len(tokens) == 0 or len(tokens) > max_tokens:
        return None

    char_map = build_char_to_token_map(text, tokens)

    # Collect entities from triples (deduplicated by span)
    entity_spans: dict[tuple[int, int], dict] = {}
    triples = row.get("triples", [])

    for triple in triples:
        for role in ("subject", "object"):
            ent = triple.get(role, {})
            boundaries = ent.get("boundaries")
            if boundaries is None:
                continue
            span = char_span_to_token_span(boundaries, char_map)
            if span is None:
                continue
            if span not in entity_spans:
                entity_spans[span] = {
                    "start": span[0],
                    "stop": span[1],
                    "label": entity_label_from_uri(ent.get("uri", "")),
                    "uri": ent.get("uri", ""),
                    "surfaceform": ent.get("surfaceform", ""),
                }

    # Build relations using span-based entity indices
    span_list = sorted(entity_spans.keys())
    span_to_idx = {s: i for i, s in enumerate(span_list)}
    entities = [entity_spans[s] for s in span_list]

    relations = []
    for triple in triples:
        subj = triple.get("subject", {})
        obj = triple.get("object", {})
        subj_bounds = subj.get("boundaries")
        obj_bounds = obj.get("boundaries")
        if subj_bounds is None or obj_bounds is None:
            continue

        subj_span = char_span_to_token_span(subj_bounds, char_map)
        obj_span = char_span_to_token_span(obj_bounds, char_map)
        if subj_span is None or obj_span is None:
            continue
        if subj_span not in span_to_idx or obj_span not in span_to_idx:
            continue

        pred = triple.get("predicate", {})
        pred_uri = pred.get("uri", "")
        # Confidence from triple or default
        confidence = triple.get("confidence")
        if confidence is None:
            confidence = 1.0

        relations.append({
            "head": span_to_idx[subj_span],
            "tail": span_to_idx[obj_span],
            "label": pred_uri,
            "confidence": confidence,
        })

    if len(relations) == 0:
        return None

    out = {
        "docid": row.get("docid", ""),
        "title": row.get("title", ""),
        "uri": row.get("uri", ""),
        "language": "en",
        "text": text,
        "tokens": tokens,
        "entities": entities,
        "relations": relations,
    }
    return out


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped_no_triples = 0
    skipped_too_long = 0
    skipped_alignment = 0
    total = 0
    relation_labels: set[str] = set()

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            row = json.loads(line)
            triples = row.get("triples", [])
            if len(triples) < args.min_triples:
                skipped_no_triples += 1
                if args.max_rows > 0 and total >= args.max_rows:
                    break
                continue

            result = convert_row(row, args.max_tokens)
            if result is None:
                skipped_alignment += 1
                if args.max_rows > 0 and total >= args.max_rows:
                    break
                continue

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            for rel in result["relations"]:
                relation_labels.add(rel["label"])
            converted += 1

            if args.max_rows > 0 and total >= args.max_rows:
                break

            if converted % 100000 == 0:
                print(f"  {converted} converted, {total} scanned...", flush=True)

    print("=" * 60)
    print("REBEL Full → Span Conversion Complete")
    print("=" * 60)
    print(f"Input:              {input_path}")
    print(f"Output:             {output_path}")
    print(f"Rows scanned:       {total}")
    print(f"Converted:          {converted}")
    print(f"Skipped (no triple):{skipped_no_triples}")
    print(f"Skipped (alignment):{skipped_alignment}")
    print(f"Unique relations:   {len(relation_labels)}")
    print(f"Conversion rate:    {100*converted/total:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
