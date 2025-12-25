# NER Data Preparation Guide

This guide details the process to create a robust, 9-label training dataset for the RAG-optimized NER model. It combines the massive **MultiNERD** dataset with granular data from **Few-NERD**, applies **Regex Augmentation** for measurements, generates **Synthetic Data** for titles, and adds **Robustness** for messy user inputs.

## Prerequisites

Ensure you have the required Python packages:

```bash
pip install datasets
```

And Julia dependencies instantiated:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Step 1: Download & Convert MultiNERD (Base Layer)

We start with MultiNERD as the foundation due to its size (~2.6M sentences). We convert it immediately to our 9-label schema.

```bash
# Downloads from HuggingFace and converts to data/rag/
python3 scripts/convert_multinerd.py --huggingface Babelscape/multinerd --output data/rag/
```

*   **Output:** `data/rag/train.jsonl`, `data/rag/validation.jsonl`, `data/rag/test.jsonl`
*   **Result:** Strong coverage for PERSON, AGENCY, PLACE, WORK, ORGANISM.

## Step 2: Augment with Regex (Measures Layer)

MultiNERD often leaves metrics (money, distances, percentages) unlabelled. We run a regex pass to "fill in" these `MEASURE` labels.

```bash
# Augment Training Data
python3 scripts/augment_with_regex.py \
    --input data/rag/train.jsonl \
    --output data/rag/train_augmented.jsonl

# Augment Validation Data
python3 scripts/augment_with_regex.py \
    --input data/rag/validation.jsonl \
    --output data/rag/validation_augmented.jsonl
```

*   **Result:** Adds thousands of high-precision `MEASURE` entities.

## Step 3: Integrate Few-NERD (Domain & Event Layer)

We bring in Few-NERD to specifically bolster the `DOMAIN` (fields of study) and `EVENT` categories.

```bash
# 3.1 Download Few-NERD
julia --project=. scripts/download_ner_data.jl --fewnerd

# 3.2 Convert to RAG Schema
python3 scripts/convert_fewnerd.py \
    --input data/ner/fewnerd/data/supervised/train.txt \
    --output data/rag/fewnerd_train.jsonl
```

## Step 4: Pump up WORK (Synthetic Layer)

Titles of books/films (WORK) are often confused with regular words. We generate synthetic examples using templates and title lists to make the category "un-ignorable."

```bash
# Generates 2000+ examples in data/rag/synthetic_work.jsonl
julia --project=. scripts/augment_synthetic.jl --pump-work
```

## Step 5: Robustness Pass (Lowercase Layer)

Real-world RAG queries are often lowercase and messy. This step duplicates ~15% of the data in lowercase to ensure the model isn't "case-dependent."

```bash
# 5.1 Combine all current sources
cat data/rag/train_augmented.jsonl \
    data/rag/fewnerd_train.jsonl \
    data/rag/synthetic_work.jsonl > data/rag/combined_pre_robust.jsonl

# 5.2 Generate robust (lowercased) duplicates
julia --project=. scripts/augment_synthetic.jl --robustness --input data/rag/combined_pre_robust.jsonl
```

*   **Output:** `data/rag/combined_pre_robust_robust.jsonl`

## Step 6: Finalize

Shuffle the final dataset to ensure the model sees a balanced mix of sources in every batch.

```bash
shuf data/rag/combined_pre_robust_robust.jsonl > data/rag/final_train.jsonl
```

## Summary of Dataset Composition

| Category | Primary Source | Improvement Step | Quality Note |
| :--- | :--- | :--- | :--- |
| **WORK** | MultiNERD | **Step 4: Synthetic** | Now robust to ambiguous titles |
| **MEASURE** | MultiNERD | **Step 2: Regex** | High precision on metrics/money |
| **DOMAIN** | Few-NERD | Step 3 | Expanded to law, science, medicine |
| **EVENT** | Few-NERD | Step 3 | Granular coverage of disasters/wars |
| **ALL** | All above | **Step 5: Robust** | **Learns lowercase/messy input** |

## Training

```bash
julia --project=. scripts/train_ner.jl \
    --data data/rag/final_train.jsonl \
    --output models/ner_final/ \
    --model production
```