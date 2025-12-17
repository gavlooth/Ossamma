#!/usr/bin/env python3
"""
Fine-tune SpanMarker NER model with RAG-optimized 9-label schema.

This script:
1. Loads converted MultiNERD data (9 labels)
2. Fine-tunes a SpanMarker model
3. Exports to ONNX for fast inference

Usage:
    # Train from converted data
    python train_ner_spanmarker.py --data data/multinerd_rag/ --output models/rag_ner/

    # Train from HuggingFace (auto-converts)
    python train_ner_spanmarker.py --huggingface Babelscape/multinerd --output models/rag_ner/

    # Export existing model to ONNX
    python train_ner_spanmarker.py --export models/rag_ner/ --onnx models/rag_ner.onnx
"""

import argparse
import json
from pathlib import Path

# =============================================================================
# RAG Schema Definition
# =============================================================================

RAG_LABELS = [
    "O",
    "PERSON",
    "AGENCY",
    "PLACE",
    "ORGANISM",
    "EVENT",
    "INSTRUMENT",
    "WORK",
    "DOMAIN",
    "MEASURE",
]

# Without O, just entity types
RAG_ENTITY_LABELS = RAG_LABELS[1:]

MULTINERD_TO_RAG = {
    "PER": "PERSON",
    "ORG": "AGENCY",
    "LOC": "PLACE",
    "CEL": "PLACE",
    "ANIM": "ORGANISM",
    "PLANT": "ORGANISM",
    "MYTH": "ORGANISM",
    "BIO": "DOMAIN",
    "DIS": "DOMAIN",
    "EVE": "EVENT",
    "TIME": "MEASURE",
    "INST": "INSTRUMENT",
    "VEHI": "INSTRUMENT",
    "FOOD": "INSTRUMENT",
    "MEDIA": "WORK",
}


def convert_example(example, id_to_label: dict) -> dict:
    """Convert a single example's labels to RAG schema."""
    tokens = example["tokens"]

    # Convert integer tags to RAG labels
    new_tags = []
    for tag_id in example["ner_tags"]:
        orig_label = id_to_label[tag_id]

        if orig_label == "O":
            new_tags.append("O")
        else:
            # Handle B-/I- prefix
            prefix, tag = orig_label.split("-", 1)
            new_tag = MULTINERD_TO_RAG.get(tag, tag)
            new_tags.append(f"{prefix}-{new_tag}")

    return {
        "tokens": tokens,
        "ner_tags": new_tags,
    }


# =============================================================================
# Data Loading
# =============================================================================

def load_converted_data(data_dir: Path):
    """Load pre-converted JSONL data."""
    from datasets import Dataset, DatasetDict

    data_dir = Path(data_dir)
    splits = {}

    for split_file in data_dir.glob("*.jsonl"):
        split_name = split_file.stem

        records = []
        with open(split_file, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        splits[split_name] = Dataset.from_list(records)

    return DatasetDict(splits)


def load_and_convert_multinerd(subset: str = "en"):
    """Load MultiNERD from HuggingFace and convert to RAG schema."""
    from datasets import load_dataset

    print(f"Loading MultiNERD ({subset})...")
    dataset = load_dataset("Babelscape/multinerd", subset)

    # Get label mapping
    label_names = dataset["train"].features["ner_tags"].feature.names
    id_to_label = {i: label for i, label in enumerate(label_names)}

    print(f"Original labels: {label_names}")
    print(f"Converting to RAG schema...")

    # Convert all splits
    converted = dataset.map(
        lambda x: convert_example(x, id_to_label),
        remove_columns=dataset["train"].column_names,
    )

    return converted


# =============================================================================
# SpanMarker Training
# =============================================================================

def train_spanmarker(
    dataset,
    output_dir: Path,
    base_model: str = "bert-base-cased",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    max_length: int = 256,
):
    """Train SpanMarker model."""
    try:
        from span_marker import SpanMarkerModel, Trainer
        from transformers import TrainingArguments
    except ImportError:
        print("Error: Please install span-marker: pip install span-marker")
        return None

    print(f"\nInitializing SpanMarker with {base_model}...")

    # Initialize model with our labels
    model = SpanMarkerModel.from_pretrained(
        base_model,
        labels=RAG_ENTITY_LABELS,
        model_max_length=max_length,
    )

    # Training arguments
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1",
        greater_is_better=True,
        logging_steps=100,
        fp16=True,
        dataloader_num_workers=4,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("val")),
    )

    print("\nStarting training...")
    trainer.train()

    # Save final model
    trainer.save_model(output_dir / "final")

    # Evaluate
    if "test" in dataset:
        print("\nEvaluating on test set...")
        results = trainer.evaluate(dataset["test"])
        print(f"Test results: {results}")

        with open(output_dir / "test_results.json", "w") as f:
            json.dump(results, f, indent=2)

    return model


# =============================================================================
# Token Classification Alternative
# =============================================================================

def train_token_classifier(
    dataset,
    output_dir: Path,
    base_model: str = "bert-base-cased",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
):
    """Train standard token classification model (alternative to SpanMarker)."""
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForTokenClassification,
            TrainingArguments,
            Trainer,
            DataCollatorForTokenClassification,
        )
        import evaluate
        import numpy as np
    except ImportError:
        print("Error: Please install transformers and evaluate")
        return None

    # Create label mappings
    all_labels = ["O"] + [f"{p}-{l}" for l in RAG_ENTITY_LABELS for p in ["B", "I"]]
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"Labels ({len(all_labels)}): {all_labels}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
    )

    # Tokenize and align labels
    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=256,
        )

        labels = []
        for i, label_list in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id.get(label_list[word_idx], 0))
                else:
                    # For subword tokens, use I- tag or -100
                    orig_label = label_list[word_idx]
                    if orig_label.startswith("B-"):
                        label_ids.append(label2id.get("I-" + orig_label[2:], 0))
                    else:
                        label_ids.append(label2id.get(orig_label, 0))

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized["labels"] = labels
        return tokenized

    # Process dataset
    tokenized_dataset = dataset.map(
        tokenize_and_align,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Metrics
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Training arguments
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        fp16=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation", tokenized_dataset.get("val")),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    trainer.save_model(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")

    return model, tokenizer


# =============================================================================
# ONNX Export
# =============================================================================

def export_to_onnx(model_path: Path, onnx_path: Path, quantize: bool = True):
    """Export trained model to ONNX format."""
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: Please install optimum[onnxruntime]: pip install optimum[onnxruntime]")
        return

    print(f"Exporting {model_path} to ONNX...")

    # Load and export
    model = ORTModelForTokenClassification.from_pretrained(
        model_path,
        export=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Save
    onnx_path = Path(onnx_path)
    onnx_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(onnx_path)
    tokenizer.save_pretrained(onnx_path)

    print(f"Saved ONNX model to {onnx_path}")

    if quantize:
        print("Quantizing model...")
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(onnx_path)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)

        quantizer.quantize(
            save_dir=onnx_path / "quantized",
            quantization_config=qconfig,
        )
        print(f"Saved quantized model to {onnx_path / 'quantized'}")


# =============================================================================
# Inference Example
# =============================================================================

def run_inference_example(model_path: Path):
    """Run example inference to verify model works."""
    try:
        from span_marker import SpanMarkerModel
    except ImportError:
        print("SpanMarker not installed, trying token classification...")
        return run_token_inference_example(model_path)

    print(f"\nLoading model from {model_path}...")
    model = SpanMarkerModel.from_pretrained(model_path)

    test_texts = [
        "Albert Einstein developed the theory of relativity at Princeton University in 1915.",
        "Apple announced the new iPhone at their headquarters in Cupertino, California.",
        "The Lord of the Rings was written by J.R.R. Tolkien and published in 1954.",
        "COVID-19 vaccines were developed by Pfizer and Moderna using mRNA technology.",
        "The Great Wall of China was built during the Ming Dynasty.",
    ]

    print("\n" + "=" * 60)
    print("INFERENCE EXAMPLES")
    print("=" * 60)

    for text in test_texts:
        print(f"\nText: {text}")
        entities = model.predict(text)
        for ent in entities:
            print(f"  {ent['label']:15s} → {ent['span']}")


def run_token_inference_example(model_path: Path):
    """Run inference with token classification model."""
    from transformers import pipeline

    print(f"\nLoading model from {model_path}...")
    ner = pipeline("ner", model=str(model_path), aggregation_strategy="simple")

    test_texts = [
        "Albert Einstein developed the theory of relativity at Princeton University in 1915.",
        "Apple announced the new iPhone at their headquarters in Cupertino, California.",
    ]

    print("\n" + "=" * 60)
    print("INFERENCE EXAMPLES")
    print("=" * 60)

    for text in test_texts:
        print(f"\nText: {text}")
        entities = ner(text)
        for ent in entities:
            print(f"  {ent['entity_group']:15s} → {ent['word']}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train NER model with RAG-optimized 9-label schema"
    )

    parser.add_argument("--data", type=Path, help="Path to converted data directory")
    parser.add_argument("--huggingface", type=str, help="HuggingFace dataset to load")
    parser.add_argument("--subset", type=str, default="en", help="Dataset subset")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")

    parser.add_argument("--model", type=str, default="bert-base-cased",
                        help="Base model (default: bert-base-cased)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)

    parser.add_argument("--method", choices=["spanmarker", "token"], default="spanmarker",
                        help="Training method")

    parser.add_argument("--export", type=Path, help="Export model to ONNX")
    parser.add_argument("--onnx", type=Path, help="ONNX output path")
    parser.add_argument("--test", type=Path, help="Run inference test on model")

    args = parser.parse_args()

    # Export mode
    if args.export:
        if not args.onnx:
            args.onnx = args.export.parent / f"{args.export.name}_onnx"
        export_to_onnx(args.export, args.onnx)
        return

    # Test mode
    if args.test:
        run_inference_example(args.test)
        return

    # Training mode
    if args.data:
        dataset = load_converted_data(args.data)
    elif args.huggingface:
        dataset = load_and_convert_multinerd(args.subset)
    else:
        parser.error("Either --data or --huggingface required")

    print(f"Dataset: {dataset}")

    args.output.mkdir(parents=True, exist_ok=True)

    # Save schema info
    schema_info = {
        "labels": RAG_LABELS,
        "entity_labels": RAG_ENTITY_LABELS,
        "mapping": MULTINERD_TO_RAG,
    }
    with open(args.output / "schema.json", "w") as f:
        json.dump(schema_info, f, indent=2)

    # Train
    if args.method == "spanmarker":
        train_spanmarker(
            dataset,
            args.output,
            base_model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    else:
        train_token_classifier(
            dataset,
            args.output,
            base_model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

    # Test inference
    run_inference_example(args.output / "final")


if __name__ == "__main__":
    main()
