#!/usr/bin/env python3
"""
Convert MultiNERD dataset to 9-label RAG-optimized schema.

MultiNERD (15 labels) → RAG Schema (9 labels):
    PERSON        ← PER
    AGENCY        ← ORG
    PLACE         ← LOC, CEL
    ORGANISM      ← ANIM, PLANT, MYTH
    EVENT         ← EVE
    INSTRUMENT    ← INST, VEHI, FOOD
    CREATIVE_WORK ← MEDIA
    DOMAIN        ← BIO, DIS
    MEASURE       ← TIME

Usage:
    python convert_multinerd.py --input data/multinerd_train.conll --output data/rag_train.conll
    python convert_multinerd.py --input data/ --output converted/ --format jsonl
    python convert_multinerd.py --huggingface Babelscape/multinerd --output data/rag_schema/
"""

import argparse
import json
from pathlib import Path
from typing import Iterator
from collections import Counter

# =============================================================================
# Label Mapping
# =============================================================================

MULTINERD_TO_RAG = {
    # Direct mappings
    "PER": "PERSON",
    "ORG": "AGENCY",
    "LOC": "PLACE",
    "EVE": "EVENT",
    "INST": "INSTRUMENT",
    "MEDIA": "WORK",
    "TIME": "MEASURE",

    # Merged into PLACE
    "CEL": "PLACE",           # Celestial bodies are locations

    # Merged into ORGANISM
    "ANIM": "ORGANISM",       # Animals
    "PLANT": "ORGANISM",      # Plants
    "MYTH": "ORGANISM",       # Mythological beings (fictional organisms)

    # Merged into INSTRUMENT
    "VEHI": "INSTRUMENT",     # Vehicles are tools
    "FOOD": "INSTRUMENT",     # Consumables

    # Merged into DOMAIN
    "BIO": "DOMAIN",          # Biological concepts (genes, proteins)
    "DIS": "DOMAIN",          # Diseases (medical domain)
}

RAG_LABELS = [
    "O",
    "B-PERSON", "I-PERSON",
    "B-AGENCY", "I-AGENCY",
    "B-PLACE", "I-PLACE",
    "B-ORGANISM", "I-ORGANISM",
    "B-EVENT", "I-EVENT",
    "B-INSTRUMENT", "I-INSTRUMENT",
    "B-WORK", "I-WORK",
    "B-DOMAIN", "I-DOMAIN",
    "B-MEASURE", "I-MEASURE",
]

LABEL_DESCRIPTIONS = {
    "PERSON": "Named individuals, historical figures",
    "AGENCY": "Companies, governments, groups, institutions",
    "PLACE": "Locations, celestial bodies, addresses",
    "ORGANISM": "Animals, plants, microbes, mythological beings",
    "EVENT": "Named events, wars, eras, incidents",
    "INSTRUMENT": "Tools, products, vehicles, devices, materials, food",
    "WORK": "Books, papers, films, music, datasets, artworks",
    "DOMAIN": "Sciences, methods, technologies, diseases, fields",
    "MEASURE": "Numbers, dates, money, percentages, durations",
}


def convert_label(label: str) -> str:
    """Convert a single MultiNERD label to RAG schema."""
    if label == "O":
        return "O"

    if "-" not in label:
        # Handle cases where label might not have B-/I- prefix
        return MULTINERD_TO_RAG.get(label, label)

    prefix, tag = label.split("-", 1)
    new_tag = MULTINERD_TO_RAG.get(tag, tag)
    return f"{prefix}-{new_tag}"


def convert_labels_batch(labels: list[str]) -> list[str]:
    """Convert a batch of labels."""
    return [convert_label(label) for label in labels]


# =============================================================================
# CoNLL Format Handling
# =============================================================================

def read_conll(filepath: Path) -> Iterator[list[tuple[str, str]]]:
    """
    Read CoNLL format file.
    Yields sentences as list of (token, label) tuples.
    """
    current_sentence = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                if current_sentence:
                    yield current_sentence
                    current_sentence = []
                continue

            # Handle different CoNLL formats
            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                label = parts[-1]  # Label is typically last column
                current_sentence.append((token, label))
            elif len(parts) == 1:
                # Token only, no label
                current_sentence.append((parts[0], "O"))

    if current_sentence:
        yield current_sentence


def write_conll(filepath: Path, sentences: list[list[tuple[str, str]]]):
    """Write sentences to CoNLL format."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for token, label in sentence:
                f.write(f"{token}\t{label}\n")
            f.write("\n")


def convert_conll_file(input_path: Path, output_path: Path) -> dict:
    """Convert a single CoNLL file. Returns statistics."""
    stats = {
        "sentences": 0,
        "tokens": 0,
        "original_labels": Counter(),
        "converted_labels": Counter(),
    }

    converted_sentences = []

    for sentence in read_conll(input_path):
        converted_sentence = []
        for token, label in sentence:
            new_label = convert_label(label)
            converted_sentence.append((token, new_label))

            stats["tokens"] += 1
            stats["original_labels"][label] += 1
            stats["converted_labels"][new_label] += 1

        converted_sentences.append(converted_sentence)
        stats["sentences"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_conll(output_path, converted_sentences)

    return stats


# =============================================================================
# JSONL Format Handling
# =============================================================================

def read_jsonl(filepath: Path) -> Iterator[dict]:
    """Read JSONL file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(filepath: Path, records: list[dict]):
    """Write records to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert_jsonl_file(input_path: Path, output_path: Path) -> dict:
    """Convert a JSONL file with tokens and ner_tags fields."""
    stats = {
        "sentences": 0,
        "tokens": 0,
        "original_labels": Counter(),
        "converted_labels": Counter(),
    }

    converted_records = []

    for record in read_jsonl(input_path):
        tokens = record.get("tokens", [])
        labels = record.get("ner_tags", record.get("labels", []))

        # Handle integer labels (HuggingFace format)
        if labels and isinstance(labels[0], int):
            # Would need label mapping from dataset info
            # For now, skip or handle separately
            converted_records.append(record)
            continue

        new_labels = convert_labels_batch(labels)

        new_record = {**record}
        new_record["ner_tags"] = new_labels
        if "labels" in new_record:
            new_record["labels"] = new_labels

        converted_records.append(new_record)

        stats["sentences"] += 1
        stats["tokens"] += len(tokens)
        for orig, conv in zip(labels, new_labels):
            stats["original_labels"][orig] += 1
            stats["converted_labels"][conv] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, converted_records)

    return stats


# =============================================================================
# HuggingFace Dataset Handling
# =============================================================================

def convert_huggingface_dataset(dataset_name: str, output_dir: Path, subset: str = "en"):
    """Download and convert a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: Please install datasets: pip install datasets")
        return

    print(f"Loading dataset: {dataset_name}")

    # MultiNERD has language subsets
    try:
        dataset = load_dataset(dataset_name, subset)
    except:
        dataset = load_dataset(dataset_name)

    # Get label names from dataset
    label_names = dataset["train"].features["ner_tags"].feature.names
    print(f"Original labels: {label_names}")

    # Create integer to string mapping
    id_to_label = {i: label for i, label in enumerate(label_names)}

    # Convert each split
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_stats = {
        "original_labels": Counter(),
        "converted_labels": Counter(),
    }

    for split in dataset.keys():
        print(f"\nConverting {split} split...")

        records = []
        for example in dataset[split]:
            tokens = example["tokens"]
            # Convert integer tags to string labels, then to new schema
            str_labels = [id_to_label[tag_id] for tag_id in example["ner_tags"]]
            new_labels = convert_labels_batch(str_labels)

            records.append({
                "tokens": tokens,
                "ner_tags": new_labels,
                "original_ner_tags": str_labels,
            })

            for orig, conv in zip(str_labels, new_labels):
                total_stats["original_labels"][orig] += 1
                total_stats["converted_labels"][conv] += 1

        output_path = output_dir / f"{split}.jsonl"
        write_jsonl(output_path, records)
        print(f"  Saved {len(records)} examples to {output_path}")

    # Save label info
    label_info = {
        "schema": "rag_9_labels",
        "labels": RAG_LABELS,
        "label_descriptions": LABEL_DESCRIPTIONS,
        "mapping": MULTINERD_TO_RAG,
        "source": dataset_name,
    }

    with open(output_dir / "label_info.json", "w") as f:
        json.dump(label_info, f, indent=2)

    print_stats(total_stats)

    return total_stats


# =============================================================================
# Utilities
# =============================================================================

def print_stats(stats: dict):
    """Print conversion statistics."""
    print("\n" + "=" * 60)
    print("CONVERSION STATISTICS")
    print("=" * 60)

    if "sentences" in stats:
        print(f"Sentences: {stats['sentences']:,}")
        print(f"Tokens: {stats['tokens']:,}")

    print("\nOriginal label distribution:")
    for label, count in sorted(stats["original_labels"].items(), key=lambda x: -x[1]):
        print(f"  {label:20s} {count:>10,}")

    print("\nConverted label distribution:")
    for label, count in sorted(stats["converted_labels"].items(), key=lambda x: -x[1]):
        print(f"  {label:20s} {count:>10,}")

    # Show compression ratio
    orig_unique = len([l for l in stats["original_labels"] if l != "O"])
    conv_unique = len([l for l in stats["converted_labels"] if l != "O"])
    print(f"\nLabel reduction: {orig_unique} → {conv_unique} ({orig_unique - conv_unique} merged)")


def print_schema_info():
    """Print the RAG schema information."""
    print("\n" + "=" * 60)
    print("RAG-OPTIMIZED NER SCHEMA (9 Labels)")
    print("=" * 60)

    for label, description in LABEL_DESCRIPTIONS.items():
        sources = [k for k, v in MULTINERD_TO_RAG.items() if v == label]
        print(f"\n{label}")
        print(f"  Description: {description}")
        print(f"  From MultiNERD: {', '.join(sources)}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert MultiNERD dataset to RAG-optimized 9-label schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert CoNLL file
  python convert_multinerd.py -i train.conll -o train_rag.conll

  # Convert directory of JSONL files
  python convert_multinerd.py -i data/ -o converted/ -f jsonl

  # Download and convert from HuggingFace
  python convert_multinerd.py --huggingface Babelscape/multinerd -o data/rag/

  # Show schema info
  python convert_multinerd.py --info
        """
    )

    parser.add_argument("-i", "--input", type=Path, help="Input file or directory")
    parser.add_argument("-o", "--output", type=Path, help="Output file or directory")
    parser.add_argument("-f", "--format", choices=["conll", "jsonl"], default="conll",
                        help="File format (default: conll)")
    parser.add_argument("--huggingface", type=str, help="HuggingFace dataset name")
    parser.add_argument("--subset", type=str, default="en", help="Dataset subset/language")
    parser.add_argument("--info", action="store_true", help="Print schema information")

    args = parser.parse_args()

    if args.info:
        print_schema_info()
        return

    if args.huggingface:
        if not args.output:
            args.output = Path(f"data/{args.huggingface.split('/')[-1]}_rag")
        convert_huggingface_dataset(args.huggingface, args.output, args.subset)
        return

    if not args.input or not args.output:
        parser.print_help()
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        # Single file conversion
        if args.format == "conll":
            stats = convert_conll_file(input_path, output_path)
        else:
            stats = convert_jsonl_file(input_path, output_path)
        print_stats(stats)

    elif input_path.is_dir():
        # Directory conversion
        pattern = "*.conll" if args.format == "conll" else "*.jsonl"
        files = list(input_path.glob(pattern))

        if not files:
            print(f"No {pattern} files found in {input_path}")
            return

        total_stats = {
            "sentences": 0,
            "tokens": 0,
            "original_labels": Counter(),
            "converted_labels": Counter(),
        }

        for file in files:
            out_file = output_path / file.name
            print(f"Converting {file} → {out_file}")

            if args.format == "conll":
                stats = convert_conll_file(file, out_file)
            else:
                stats = convert_jsonl_file(file, out_file)

            total_stats["sentences"] += stats["sentences"]
            total_stats["tokens"] += stats["tokens"]
            total_stats["original_labels"].update(stats["original_labels"])
            total_stats["converted_labels"].update(stats["converted_labels"])

        print_stats(total_stats)

    else:
        print(f"Error: {input_path} not found")


if __name__ == "__main__":
    main()
