#!/usr/bin/env bash
# Download reasoning datasets for Phase 3a language fine-tuning
#
# Datasets:
#   PrOntoQA  — Syllogistic chains (synthetic, generate from HF)
#   ReClor    — Argumentation (law school logical reasoning)
#   GSM8K     — Arithmetic reasoning chains
#   BoardgameQA — Rule-based deduction under constraints
#   FOLIO     — First-order logic NLI
#   LogiQA    — Formal logical reasoning
#
# Usage:
#   ./scripts/download_reasoning_datasets.sh [output_dir]
#   Default output: data/reasoning/

set -euo pipefail

OUTPUT_DIR="${1:-data/reasoning}"
mkdir -p "$OUTPUT_DIR"

echo "=== Downloading reasoning datasets for Phase 3a ==="
echo "Output: $OUTPUT_DIR"
echo ""

# Requires: pip install datasets
python3 -c "import datasets" 2>/dev/null || {
    echo "ERROR: Python 'datasets' package not found."
    echo "Install with: pip install datasets"
    exit 1
}

export REASONING_OUTPUT_DIR="$OUTPUT_DIR"
python3 << 'PYEOF'
import json
import os

output_dir = os.environ.get("REASONING_OUTPUT_DIR", "data/reasoning")
os.makedirs(output_dir, exist_ok=True)

from datasets import load_dataset

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved {len(data)} examples to {path}")

# =========================================================================
# 1. PrOntoQA — Syllogistic reasoning chains
# =========================================================================
print("1/6 PrOntoQA...")
try:
    ds = load_dataset("rencos/PrOntoQA", split="train")
    examples = []
    for row in ds:
        examples.append({
            "source": "prontoqa",
            "context": row.get("context", ""),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "chain_of_thought": row.get("chain_of_thought", ""),
            "reasoning_type": "syllogistic",
        })
    save_jsonl(examples, os.path.join(output_dir, "prontoqa.jsonl"))
except Exception as e:
    print(f"  SKIP PrOntoQA: {e}")

# =========================================================================
# 2. ReClor — Logical reasoning from law exams
# =========================================================================
print("2/6 ReClor...")
try:
    ds = load_dataset("metaeval/reclor", split="train")
    examples = []
    for row in ds:
        examples.append({
            "source": "reclor",
            "context": row.get("context", ""),
            "question": row.get("question", ""),
            "answers": row.get("answers", []),
            "label": row.get("label", -1),
            "reasoning_type": "argumentation",
        })
    save_jsonl(examples, os.path.join(output_dir, "reclor.jsonl"))
except Exception as e:
    print(f"  SKIP ReClor: {e}")

# =========================================================================
# 3. GSM8K — Grade school math with chain-of-thought
# =========================================================================
print("3/6 GSM8K...")
try:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    examples = []
    for row in ds:
        examples.append({
            "source": "gsm8k",
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "reasoning_type": "arithmetic_chain",
        })
    save_jsonl(examples, os.path.join(output_dir, "gsm8k.jsonl"))
except Exception as e:
    print(f"  SKIP GSM8K: {e}")

# =========================================================================
# 4. BoardgameQA — Rule-based deduction with exceptions
# =========================================================================
print("4/6 BoardgameQA...")
try:
    ds = load_dataset("Mihail195/BoardgameQA", split="train")
    examples = []
    for row in ds:
        examples.append({
            "source": "boardgameqa",
            "context": row.get("context", row.get("input", "")),
            "question": row.get("question", ""),
            "answer": row.get("answer", row.get("target", "")),
            "reasoning_type": "rule_deduction",
        })
    save_jsonl(examples, os.path.join(output_dir, "boardgameqa.jsonl"))
except Exception as e:
    print(f"  SKIP BoardgameQA: {e}")

# =========================================================================
# 5. FOLIO — First-order logic NLI
# =========================================================================
print("5/6 FOLIO...")
try:
    ds = load_dataset("yale-nlp/FOLIO", split="train")
    examples = []
    for row in ds:
        examples.append({
            "source": "folio",
            "premises": row.get("premises", ""),
            "conclusion": row.get("conclusion", ""),
            "label": row.get("label", ""),
            "reasoning_type": "first_order_logic",
        })
    save_jsonl(examples, os.path.join(output_dir, "folio.jsonl"))
except Exception as e:
    print(f"  SKIP FOLIO: {e}")

# =========================================================================
# 6. LogiQA — Logical reasoning MCQ
# =========================================================================
print("6/6 LogiQA...")
try:
    ds = load_dataset("lucasmccabe/logiqa", split="train")
    examples = []
    for row in ds:
        examples.append({
            "source": "logiqa",
            "context": row.get("context", ""),
            "question": row.get("query", row.get("question", "")),
            "options": row.get("options", []),
            "label": row.get("correct_option", row.get("answer", -1)),
            "reasoning_type": "formal_logic",
        })
    save_jsonl(examples, os.path.join(output_dir, "logiqa.jsonl"))
except Exception as e:
    print(f"  SKIP LogiQA: {e}")

# =========================================================================
# Summary
# =========================================================================
print("\n=== Summary ===")
total = 0
for fname in sorted(os.listdir(output_dir)):
    if fname.endswith('.jsonl'):
        path = os.path.join(output_dir, fname)
        count = sum(1 for _ in open(path))
        total += count
        print(f"  {fname}: {count:,} examples")
print(f"  TOTAL: {total:,} examples")

PYEOF

echo ""
echo "Done. All datasets in: $OUTPUT_DIR"
