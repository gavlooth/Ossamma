#!/usr/bin/env python3
"""
Augment NER dataset with regex-based entities.
Primarily used to tag MEASURE entities (money, percentage, physical quantity) 
that are often missed or not labeled in standard datasets like MultiNERD.

Usage:
    python augment_with_regex.py --input data/rag/train.jsonl --output data/rag/train_augmented.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

# Regex patterns for MEASURE
# These are designed to be high-precision to avoid false positives.
PATTERNS = [
    # Percentage: 10%, 10.5 percent
    (re.compile(r'^\d+(\.\d+)?%$'), "MEASURE"),
    (re.compile(r'^\d+(\.\d+)?$'), "MEASURE_NUM"), # Intermediate, check next token for "percent"
    
    # Money: $10, €500, 100USD
    (re.compile(r'^[$€£¥]\d+(\.\d+)?([kKmMbB])?$'), "MEASURE"),
    
    # Physical: 10kg, 100km, 50mm (attached)
    (re.compile(r'^\d+(\.\d+)?[kK]?[mMgG]$'), "MEASURE"),
    
    # Years/Dates: 1990s, 2024 (often already TIME, but good to catch)
    (re.compile(r'^(19|20)\d{2}s?$'), "MEASURE"),
]

UNITS = {
    "percent", "percentage", "%",
    "dollar", "dollars", "usd", "eur", "euro", "euros",
    "km", "meter", "meters", "kg", "kilogram", "ton", "tons",
    "mile", "miles", "inch", "inches", "cm", "mm",
    "degree", "degrees",
    "second", "seconds", "minute", "minutes", "hour", "hours",
    "day", "days", "week", "weeks", "month", "months", "year", "years"
}

def is_measure(token: str, next_token: str = None) -> bool:
    """Check if a token (or token pair) is a measurement."""
    
    # Check attached patterns ($10, 10%)
    for pattern, label in PATTERNS:
        if pattern.match(token) and label == "MEASURE":
            return True
            
    # Check separated patterns (10 percent, 10 km)
    if token.replace('.', '', 1).isdigit():
        if next_token and next_token.lower() in UNITS:
            return True
            
    return False

def augment_sentence(tokens: List[str], labels: List[str]) -> Tuple[List[str], int]:
    """Augment a single sentence's labels."""
    new_labels = list(labels)
    changes = 0
    
    for i in range(len(tokens)):
        # Only overwrite 'O' labels
        if labels[i] != "O":
            continue
            
        token = tokens[i]
        next_token = tokens[i+1] if i + 1 < len(tokens) else None
        
        # Check for MEASURE
        if is_measure(token, next_token):
            new_labels[i] = "B-MEASURE"
            changes += 1
            
            # If we used the next token (e.g., "10" "percent"), label it too
            if next_token and next_token.lower() in UNITS and i + 1 < len(tokens):
                if labels[i+1] == "O":
                    new_labels[i+1] = "I-MEASURE"
    
    return new_labels, changes

def process_file(input_path: Path, output_path: Path):
    print(f"Augmenting {input_path} -> {output_path}")
    
    total_changes = 0
    sentences = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip():
                continue
                
            record = json.loads(line)
            tokens = record['tokens']
            labels = record.get('ner_tags', record.get('labels', []))
            
            new_labels, changes = augment_sentence(tokens, labels)
            
            record['ner_tags'] = new_labels
            if 'labels' in record:
                record['labels'] = new_labels
                
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            total_changes += changes
            sentences += 1
            
    print(f"  Processed {sentences} sentences")
    print(f"  Added {total_changes} new MEASURE labels")

def main():
    parser = argparse.ArgumentParser(description="Augment dataset with regex")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    
    process_file(args.input, args.output)

if __name__ == "__main__":
    main()
