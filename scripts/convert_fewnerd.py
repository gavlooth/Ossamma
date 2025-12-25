#!/usr/bin/env python3
"""
Convert Few-NERD dataset to RAG-optimized 9-label schema.
Used to bolster DOMAIN (fields of study) and EVENT categories.

Usage:
    python convert_fewnerd.py --input data/ner/fewnerd/data/supervised/train.txt --output data/rag/fewnerd_train.jsonl
"""

import argparse
import json
from pathlib import Path
from collections import Counter

# =============================================================================
# Label Mapping: Few-NERD -> RAG (9 Labels)
# =============================================================================

FEWNERD_TO_RAG = {
    # --- DOMAIN (The main reason we are here) ---
    "other-astronomything": "DOMAIN",
    "other-biologything": "DOMAIN",
    "other-chemicalthing": "DOMAIN",
    "other-law": "DOMAIN",
    "other-medical": "DOMAIN",
    "other-language": "DOMAIN", # Linguistics
    "other-educationaldegree": "DOMAIN", # Often related to a field

    # --- EVENT (Strong expansion) ---
    "event-attack/battle/war/militaryconflict": "EVENT",
    "event-disaster": "EVENT",
    "event-election": "EVENT",
    "event-protest": "EVENT",
    "event-sportsevent": "EVENT",
    "event-other": "EVENT",

    # --- MEASURE (Additional coverage) ---
    "other-currency": "MEASURE", 
    # Few-NERD doesn't have explicit 'quantity', usually in 'other'

    # --- OTHERS (To align with existing schema) ---
    "person-actor": "PERSON",
    "person-artist/author": "PERSON",
    "person-athlete": "PERSON",
    "person-director": "PERSON",
    "person-politician": "PERSON",
    "person-scholar": "PERSON",
    "person-soldier": "PERSON",
    "person-other": "PERSON",

    "organization-company": "AGENCY",
    "organization-education": "AGENCY",
    "organization-government/governmentagency": "AGENCY",
    "organization-media/newspaper": "AGENCY",
    "organization-politicalparty": "AGENCY",
    "organization-religion": "AGENCY",
    "organization-sportsleague": "AGENCY",
    "organization-other": "AGENCY",

    "location-GPE": "PLACE",
    "location-bodiesofwater": "PLACE",
    "location-island": "PLACE",
    "location-mountain": "PLACE",
    "location-park": "PLACE",
    "location-road/railway/highway/transit": "PLACE",
    "location-other": "PLACE",
    
    "building-airport": "PLACE",
    "building-hospital": "PLACE",
    "building-hotel": "PLACE",
    "building-library": "PLACE",
    "building-restaurant": "PLACE",
    "building-sportsfacility": "PLACE",
    "building-theater": "PLACE",
    "building-other": "PLACE",

    "art-broadcastprogram": "WORK",
    "art-film": "WORK",
    "art-music": "WORK",
    "art-painting": "WORK",
    "art-writtenart": "WORK",
    "art-other": "WORK",
    
    "product-airplane": "INSTRUMENT",
    "product-car": "INSTRUMENT",
    "product-computer": "INSTRUMENT",
    "product-food": "INSTRUMENT", # Merged into INSTRUMENT/Product in our schema
    "product-game": "INSTRUMENT", # Or WORK, but products are tools/objects
    "product-ship": "INSTRUMENT",
    "product-software": "INSTRUMENT", # Tool
    "product-train": "INSTRUMENT",
    "product-weapon": "INSTRUMENT",
    "product-other": "INSTRUMENT",

    # Ignoring 'other-god' (Could be PERSON or MYTH->ORGANISM)
    "other-god": "ORGANISM", # Mythological being -> ORGANISM
    "other-livingthing": "ORGANISM",
}

def convert_label(label: str) -> str:
    """Convert Few-NERD label to RAG schema."""
    if label == "O":
        return "O"
    
    return FEWNERD_TO_RAG.get(label, "O") # Default to O if not mapped

def process_file(input_path: Path, output_path: Path):
    print(f"Converting {input_path} -> {output_path}")
    
    stats = Counter()
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        current_tokens = []
        current_labels = []
        
        for line in fin:
            line = line.strip()
            if not line:
                if current_tokens:
                    # Save sentence
                    # Convert labels to BIO format (Few-NERD is just raw labels usually)
                    # Actually Few-NERD is usually tab-separated: token \t label
                    # And labels are coarse-fine.
                    
                    # Convert to B- I- format
                    rag_labels = []
                    prev_label = "O"
                    
                    for lbl in current_labels:
                        target = convert_label(lbl)
                        if target == "O":
                            rag_labels.append("O")
                        else:
                            if target != prev_label:
                                rag_labels.append(f"B-{target}")
                            else:
                                # Simple heuristic: if same type, assume I-. 
                                # (Strictly checking B- tags in source would be better but Few-NERD is usually sentence-split well)
                                rag_labels.append(f"I-{target}")
                        prev_label = target
                        stats[target] += 1

                    record = {
                        "tokens": current_tokens,
                        "ner_tags": rag_labels,
                        "original_tags": current_labels
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
                    current_tokens = []
                    current_labels = []
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                label = parts[1]
                current_tokens.append(token)
                current_labels.append(label)

    print("\nLabel Distribution (Mapped):")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")

def main():
    parser = argparse.ArgumentParser(description="Convert Few-NERD to RAG schema")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    
    process_file(args.input, args.output)

if __name__ == "__main__":
    main()
