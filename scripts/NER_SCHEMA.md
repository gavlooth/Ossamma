# RAG-Optimized NER Schema (9 Labels)

Named Entity Recognition schema optimized for Retrieval-Augmented Generation.

**Uses OssammaNER** - Token-level NER model built on OssammaBlocks (SWAttention + DLinOSS oscillators).

## Labels

| Label | Description | Examples |
|-------|-------------|----------|
| **PERSON** | Named individuals, historical figures | Einstein, Marie Curie, Obama |
| **AGENCY** | Companies, governments, groups, institutions | Google, NATO, WHO, Congress |
| **PLACE** | Locations, celestial bodies, addresses | Paris, Mars, 123 Main St |
| **ORGANISM** | Animals, plants, microbes, mythological beings | dolphin, oak tree, dragon |
| **EVENT** | Named events, wars, eras, incidents | WW2, Olympics, Renaissance |
| **INSTRUMENT** | Tools, products, vehicles, devices, materials | iPhone, hammer, insulin, pizza |
| **WORK** | Books, papers, films, music, datasets, art | "1984", Nature, Mona Lisa |
| **DOMAIN** | Sciences, methods, technologies, diseases | physics, COVID-19, blockchain |
| **MEASURE** | Numbers, dates, money, percentages | 50%, 2024, $1M, 3 hours |

## MultiNERD Mapping

```
PERSON        ← PER
AGENCY        ← ORG
PLACE         ← LOC, CEL
ORGANISM      ← ANIM, PLANT, MYTH
EVENT         ← EVE
INSTRUMENT    ← INST, VEHI, FOOD
WORK          ← MEDIA
DOMAIN        ← BIO, DIS
MEASURE       ← TIME
```

## Why These Labels for RAG?

| Label | Links Docs? | Query Freq | RAG Use Case |
|-------|-------------|------------|--------------|
| PERSON | ✅ High | ✅ High | "papers by X", "who invented" |
| AGENCY | ✅ High | ✅ High | "X company products" |
| PLACE | ✅ Medium | ✅ High | "events in X" |
| ORGANISM | ✅ Medium | Medium | "habitat of X" |
| EVENT | ✅ High | ✅ High | "causes of X" |
| INSTRUMENT | ✅ Medium | ✅ High | "how X works" |
| WORK | ✅ Very High | ✅ High | "cited by", "summary of" |
| DOMAIN | ✅ High | ✅ High | "advances in X" |
| MEASURE | ❌ Low | Medium | Fact grounding |

## Quick Start (Julia + OssammaBlocks)

### 1. Train with OssammaNER

```julia
using Ossamma

# Create NER model
model = small_ner(vocab_size=32000, max_sequence_length=256)

# Or configure manually
config = NERConfig(
    vocab_size = 32000,
    embedding_dimension = 256,
    number_of_heads = 4,
    number_of_layers = 4,
)
model = OssammaNER(config)

# Initialize
rng = Random.default_rng()
ps = Lux.initialparameters(rng, model)
st = Lux.initialstates(rng, model)

# Forward pass: (seq_len, batch) → (num_labels, seq_len, batch)
token_ids = rand(1:32000, 128, 8)
logits, new_st = model(token_ids, ps, st)

# Predict labels
labels = predict_labels(model, ps, st, token_ids[:, 1])

# Extract entities
tokens = ["Einstein", "worked", "at", "MIT"]
entities = extract_entities(tokens, labels[1:4])
```

### 2. Train from Command Line

```bash
# Quick test
julia --project=. scripts/train_ner.jl --test

# Train on converted data
julia --project=. scripts/train_ner.jl \
    --data data/rag/ \
    --output models/ner/ \
    --model small \
    --epochs 10
```

## Python Tools (for data conversion)

### 1. Convert MultiNERD Data

```bash
# From HuggingFace
python convert_multinerd.py --huggingface Babelscape/multinerd --output data/rag/

# From local CoNLL file
python convert_multinerd.py -i train.conll -o train_rag.conll
```

### 2. Train Model

```bash
# SpanMarker (recommended)
python train_ner_spanmarker.py \
    --huggingface Babelscape/multinerd \
    --output models/rag_ner/ \
    --epochs 3

# Token classification
python train_ner_spanmarker.py \
    --huggingface Babelscape/multinerd \
    --output models/rag_ner/ \
    --method token
```

### 3. Run Inference

```bash
# Single text
python ner_inference.py --model models/rag_ner/final "Einstein worked at Princeton"

# Interactive
python ner_inference.py --model models/rag_ner/final --interactive

# Batch process
python ner_inference.py --model models/rag_ner/final --input texts.txt --output entities.jsonl
```

### 4. Export to ONNX

```bash
python train_ner_spanmarker.py --export models/rag_ner/final --onnx models/rag_ner_onnx/
```

## Output Format for RAG

```json
{
  "text": "Einstein developed relativity at Princeton in 1915.",
  "entities": [
    {"text": "Einstein", "label": "PERSON", "start": 0, "end": 8},
    {"text": "relativity", "label": "DOMAIN", "start": 19, "end": 29},
    {"text": "Princeton", "label": "AGENCY", "start": 33, "end": 42},
    {"text": "1915", "label": "MEASURE", "start": 46, "end": 50}
  ],
  "entities_by_type": {
    "PERSON": ["Einstein"],
    "DOMAIN": ["relativity"],
    "AGENCY": ["Princeton"],
    "MEASURE": ["1915"]
  }
}
```

## Integration with RAG Pipeline

```python
from ner_inference import load_model, extract_entities, format_for_rag

# Load once
load_model("models/rag_ner/final")

# Process documents for indexing
def process_document(doc_text: str) -> dict:
    entities = extract_entities(doc_text)
    rag_data = format_for_rag(doc_text, entities)

    # Use for:
    # 1. Metadata filtering: filter by entity type
    # 2. Query expansion: add entity synonyms
    # 3. Chunk boundaries: split at entity-dense regions
    # 4. Knowledge graph: link entities across docs

    return rag_data
```

## Edge Cases

| Entity | Label | Reasoning |
|--------|-------|-----------|
| "COVID-19" | DOMAIN | Disease = medical domain |
| "Python" | DOMAIN | Technology/language |
| "Tesla" (car) | INSTRUMENT | Product |
| "Tesla" (company) | AGENCY | Organization |
| "Tesla" (person) | PERSON | Individual |
| "World War II" | EVENT | Named historical event |
| "2024" | MEASURE | Temporal anchor |
| "dragons" | ORGANISM | Mythological being |
| "The Bible" | WORK | Religious text = work |
| "pizza" | INSTRUMENT | Consumable item |
