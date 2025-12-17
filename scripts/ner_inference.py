#!/usr/bin/env python3
"""
Simple NER inference with RAG-optimized 9-label schema.

Usage:
    # Single text
    python ner_inference.py --model models/rag_ner/final "Einstein worked at Princeton"

    # From file
    python ner_inference.py --model models/rag_ner/final --input texts.txt --output entities.jsonl

    # Interactive mode
    python ner_inference.py --model models/rag_ner/final --interactive
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# =============================================================================
# Schema
# =============================================================================

RAG_LABELS = {
    "PERSON": "Named individuals, historical figures",
    "AGENCY": "Companies, governments, groups, institutions",
    "PLACE": "Locations, celestial bodies, addresses",
    "ORGANISM": "Animals, plants, microbes, mythological beings",
    "EVENT": "Named events, wars, eras, incidents",
    "INSTRUMENT": "Tools, products, vehicles, devices, materials",
    "WORK": "Books, papers, films, music, datasets, artworks",
    "DOMAIN": "Sciences, methods, technologies, diseases, fields",
    "MEASURE": "Numbers, dates, money, percentages, durations",
}


# =============================================================================
# Model Loading
# =============================================================================

_model = None
_model_type = None


def load_model(model_path: str):
    """Load NER model (SpanMarker or Token Classification)."""
    global _model, _model_type

    model_path = Path(model_path)

    # Try SpanMarker first
    try:
        from span_marker import SpanMarkerModel
        _model = SpanMarkerModel.from_pretrained(model_path)
        _model_type = "spanmarker"
        print(f"Loaded SpanMarker model from {model_path}")
        return
    except Exception:
        pass

    # Try token classification
    try:
        from transformers import pipeline
        _model = pipeline("ner", model=str(model_path), aggregation_strategy="simple")
        _model_type = "token"
        print(f"Loaded token classification model from {model_path}")
        return
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def extract_entities(text: str) -> list[dict]:
    """Extract entities from text."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    if _model_type == "spanmarker":
        entities = _model.predict(text)
        return [
            {
                "text": ent["span"],
                "label": ent["label"],
                "start": ent["char_start_index"],
                "end": ent["char_end_index"],
                "score": ent.get("score", 1.0),
            }
            for ent in entities
        ]
    else:
        entities = _model(text)
        return [
            {
                "text": ent["word"],
                "label": ent["entity_group"],
                "start": ent["start"],
                "end": ent["end"],
                "score": ent["score"],
            }
            for ent in entities
        ]


def extract_entities_batch(texts: list[str]) -> list[list[dict]]:
    """Extract entities from multiple texts."""
    return [extract_entities(text) for text in texts]


# =============================================================================
# Output Formatting
# =============================================================================

def format_entities_inline(text: str, entities: list[dict]) -> str:
    """Format text with inline entity annotations."""
    if not entities:
        return text

    # Sort by start position, reverse to replace from end
    sorted_ents = sorted(entities, key=lambda x: x["start"], reverse=True)

    result = text
    for ent in sorted_ents:
        before = result[:ent["start"]]
        after = result[ent["end"]:]
        result = f"{before}[{ent['text']}]({ent['label']}){after}"

    return result


def format_entities_table(entities: list[dict]) -> str:
    """Format entities as a simple table."""
    if not entities:
        return "  No entities found"

    lines = []
    for ent in entities:
        score_str = f"{ent['score']:.2f}" if ent.get('score') else ""
        lines.append(f"  {ent['label']:15s}  {ent['text']:30s}  {score_str}")
    return "\n".join(lines)


def format_for_rag(text: str, entities: list[dict]) -> dict:
    """Format output for RAG indexing."""
    # Group by label
    by_label = {}
    for ent in entities:
        label = ent["label"]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(ent["text"])

    # Deduplicate
    for label in by_label:
        by_label[label] = list(set(by_label[label]))

    return {
        "text": text,
        "entities": entities,
        "entities_by_type": by_label,
        "entity_count": len(entities),
    }


# =============================================================================
# CLI
# =============================================================================

def interactive_mode():
    """Run interactive entity extraction."""
    print("\nRAG NER Interactive Mode")
    print("Enter text to extract entities (Ctrl+C to exit)")
    print("-" * 50)

    while True:
        try:
            text = input("\n> ").strip()
            if not text:
                continue

            entities = extract_entities(text)
            print(format_entities_table(entities))
            print(f"\nAnnotated: {format_entities_inline(text, entities)}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def process_file(input_path: Path, output_path: Optional[Path] = None):
    """Process file and extract entities."""
    with open(input_path, 'r') as f:
        texts = [line.strip() for line in f if line.strip()]

    results = []
    for text in texts:
        entities = extract_entities(text)
        results.append(format_for_rag(text, entities))

    if output_path:
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Saved {len(results)} results to {output_path}")
    else:
        for result in results:
            print(json.dumps(result, ensure_ascii=False))

    return results


def main():
    parser = argparse.ArgumentParser(description="NER inference with RAG schema")

    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("text", nargs="?", help="Text to process")
    parser.add_argument("--input", type=Path, help="Input file (one text per line)")
    parser.add_argument("--output", type=Path, help="Output JSONL file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--format", choices=["table", "inline", "json"], default="table")
    parser.add_argument("--labels", action="store_true", help="Show label descriptions")

    args = parser.parse_args()

    if args.labels:
        print("\nRAG NER Labels:")
        print("-" * 50)
        for label, desc in RAG_LABELS.items():
            print(f"  {label:15s}  {desc}")
        return

    # Load model
    load_model(args.model)

    if args.interactive:
        interactive_mode()
    elif args.input:
        process_file(args.input, args.output)
    elif args.text:
        entities = extract_entities(args.text)

        if args.format == "table":
            print(format_entities_table(entities))
        elif args.format == "inline":
            print(format_entities_inline(args.text, entities))
        else:
            print(json.dumps(format_for_rag(args.text, entities), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
