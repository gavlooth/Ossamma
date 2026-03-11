# SWAMMA (Spectral Wave-Attention Masked Mixer Architecture)

**Spectral Wave-PDE Attention Masked Mixer Architecture** - A Julia-based neural network framework for efficient many-layer sequence models built around `LinearAttention`, `WavePDE`, and sharp local attention.

Canonical naming in this repo is now **SWAMMA/Wave-PDE-first**.
Preferred API aliases include `SwammaBlock`, `SwammaClassifier`, `SwammaNER`, and `wave_gate(...)`.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SwammaNER Architecture                             │
└─────────────────────────────────────────────────────────────────────────────┘

  Input: "Barack Obama visited Paris"
                    │
                    ▼
    ┌───────────────────────────────┐
    │     Token Embeddings          │  vocab_size → embedding_dim
    │     + Position Embeddings     │  seq_len → embedding_dim
    │     + Time Embedding          │  (fixed t=0.5 for NER)
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                    SwammaBlock (×N layers)                   │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │            Time-Conditioned LayerNorm                   │  │
    │  │         returns normalized x and α_bias(t)              │  │
    │  └────────────────────┬────────────────────────────────────┘  │
    │                       │                                       │
    │          ┌────────────┴────────────┐                          │
    │          │                         │                          │
    │          ▼                         ▼                          │
    │  ┌────────────────┐      ┌───────────────────────┐           │
    │  │ Global Branch  │      │ Local Branch          │           │
    │  │                │      │                       │           │
    │  │ Dense(d→2d)    │      │ normalized            │           │
    │  │  ├─ content ──►│      │    ⊙ local_gate       │           │
    │  │  │ LinearAttn  │      │         ↓             │           │
    │  │  │ RMSNorm     │      │    SWAttention        │           │
    │  │  └─ gate ─────►│      │                       │           │
    │  │    WavePDE     │      └───────────┬───────────┘           │
    │  │    RMSNorm     │                  │                       │
    │  │    sigmoid     │                  │                       │
    │  │ content ⊙ gate │ = glu_out        │                       │
    │  └────────┬───────┘                  │                       │
    │           │                          │                       │
    │           │   global -> local control                         │
    │           ▼                          │                       │
    │    ┌───────────────────┐             │                       │
    │    │ InputGate(glu_out)│ = local_gate│                       │
    │    └─────────┬─────────┘             │                       │
    │              └───────────────┬───────┘                       │
    │                           ▼                                  │
    │                ┌─────────────────────────┐                   │
    │                │ Adaptive Mix            │                   │
    │                │ α·global + (1-α)·local  │                   │
    │                │ α = σ(Wα·x_norm+α_bias) │                   │
    │                └───────────┬─────────────┘                   │
    │                            ▼                                 │
    │                ┌─────────────────────────┐                   │
    │                │ Dropout → SwiGLU FFN    │                   │
    │                │ → Residual → LayerNorm  │                   │
    │                └─────────────────────────┘                   │
    └───────────────────────────────────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │    Classification Head        │  LayerNorm → Dense(emb → 19)
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │    CRF Layer (optional)       │  Viterbi decoding for valid
    │                               │  BIO sequences
    └───────────────┬───────────────┘
                    │
                    ▼
  Output: [B-PERSON, I-PERSON, O, B-PLACE]
```

## Architectural Thesis

`SwammaBlock` is built around a division of labor rather than a single generic
attention stack.

- `LinearAttention` is the scalable global content path. It moves information
  across the full sequence without paying the full `O(n^2)` cost of dense attention.
- `WavePDE` is not a replacement for attention. It provides a structured smooth
  dynamical prior that gates the global path, so the model does not collapse into
  a plain attention-only encoder.
- `InputGate(glu_out)` is the explicit global-to-local connection. The global
  branch decides which features are worth presenting to the local operator before
  `SWAttention` spends capacity on neighborhood refinement.
- `SWAttention` is the sharp local corrector. It handles boundary-sensitive and
  token-local structure that the smooth Wave-PDE gate should not be expected to model.
- `α`-mixing is the arbitration layer. It learns how much of the final update
  should come from the global structured path versus the local corrective path.

The intended effect is not "more mechanisms". It is a hierarchy:

```text
global interpretation -> local conditioning -> local refinement -> adaptive merge
```

## Core Components

### WavePDE Gate

Spectral structured gate based on a damped wave equation:

```
u_t = v
v_t = c²Δu - γv

Forward pass uses `u(0)=x`, `v(0)=0` and integrates a small number of internal
Wave-PDE steps spectrally, without a recurrent token-by-token scan.
```

In the current `SwammaBlock`, `WavePDE` is used as the gate that modulates the
global `LinearAttention` content path across the active model families in this
repo.

### SWAttention (Sliding Window Attention)

Local attention restricted to a window around each position:

```
Attention(Q, K, V) = sigsoftmax(QK^T / √d · mask) · V

where mask[i,j] = -∞ if |i-j| > window_size
```

Uses `sigsoftmax` (sigmoid-enhanced softmax) for sharper attention patterns.

### LinearAttention

O(n) global attention using the kernel trick:

```
Instead of:  softmax(QK^T)V     → O(n²)
Use:         φ(Q)(φ(K)^T V)    → O(n)
```

Provides global context efficiently, complementing local SWAttention.

### SwiGLU FFN

Swish-Gated Linear Unit feed-forward network from "GLU Variants Improve Transformer" (Shazeer, 2020):

```
FFN(x) = Dense(Swish(a) ⊙ b) where [a, b] = split(Dense(x))

Expansion: d → 3d/2 → split → swish(half) ⊙ other → d
```

Provides transform-type nonlinearity after the α-mixing step. The 3/2 expansion factor (e.g., 384 → 576 → 288 split → 384).

## NER Label Schema

SwammaNER uses a RAG-optimized 9-entity-type schema with BIO tagging (**19 labels** total).

### Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| **PERSON** | Individual people | "Barack Obama", "Marie Curie" |
| **AGENCY** | Organizations, companies, institutions | "Google", "United Nations", "FDA" |
| **PLACE** | Locations, geographic entities | "Paris", "Mount Everest", "Europe" |
| **ORGANISM** | Living things: animals, plants, species | "dolphin", "oak tree", "E. coli" |
| **EVENT** | Occurrences, happenings | "World War II", "Olympics", "IPO" |
| **INSTRUMENT** | Tools, devices, equipment | "microscope", "Python", "MRI scanner" |
| **WORK** | Creative outputs, publications | "Hamlet", "Nature journal", "GPT-4" |
| **DOMAIN** | Fields, categories, topics | "astrology", "media", "quantum physics" |
| **MEASURE** | Quantities, dates, money, time | "500kg", "2024", "$1M", "3 hours" |

### Semantic Coverage

```
Who?        → PERSON, AGENCY
What?       → ORGANISM, INSTRUMENT, WORK
Where?      → PLACE
When/How?   → MEASURE
What happened? → EVENT
What field?    → DOMAIN
```

### BIO Tagging Example

```
"Barack Obama visited Paris"
 B-PERSON I-PERSON O      B-PLACE
```

### Design Rationale

| This Schema | Standard NER Equivalent |
|-------------|-------------------------|
| AGENCY | ORG (same semantics) |
| MEASURE | DATE + TIME + QUANTITY + MONEY (consolidated) |
| DOMAIN | No direct equivalent (catch-all for categorical concepts) |

### Possible Improvements

1. **Split MEASURE** - Separate DATE/TIME from QUANTITY/MONEY for finer temporal queries
2. **Add NORP** - Nationalities, religious, political groups ("Republicans", "French citizens")
3. **PRODUCT vs INSTRUMENT** - Commercial products may warrant separate handling
4. **LANGUAGE type** - For multilingual RAG ("English", "Mandarin")
5. **Clarify DOMAIN boundaries** - Is "AI" a DOMAIN or INSTRUMENT?
6. **Nested entity support** - "New York Times" is both AGENCY and WORK

## Project Structure

```
Swamma/
├── src/
│   ├── Swamma.jl           # Main module, SwammaBlock, SwammaNERBlock
│   ├── WavePDE.jl           # Spectral damped-wave gate used by SwammaBlock
│   ├── Attention.jl         # SWAttention (Sliding Window)
│   ├── linearAttention.jl   # O(n) Linear Attention
│   ├── NER.jl               # SwammaNER model
│   ├── RelationExtraction.jl # Structured relation extraction model
│   ├── CRF.jl               # Conditional Random Field
│   ├── Training.jl          # Loss functions, training utilities
│   ├── data/
│   │   ├── NERDataset.jl    # Data loading and batching
│   │   ├── Tokenizer.jl     # BPE tokenization
│   │   └── Augmentation.jl  # Data augmentation
│   ├── evaluation/
│   │   └── NERMetrics.jl    # F1, precision, recall
│   └── serve/
│       ├── InferenceServer.jl  # HTTP API
│       └── Monitoring.jl       # GPU monitoring
├── scripts/
│   ├── train_ner_production.jl  # Production training
│   ├── train_rebel.jl           # Structured REBEL-style relation extraction training
│   ├── export_model.jl          # Model serialization
│   └── download_ner_data.jl     # Data utilities
├── configs/
│   ├── ner_production_110m.toml # Production config
│   └── ner_dev.toml             # Development config
├── checkpoints/                  # Saved model weights
└── docs/
    ├── SWAMMA_NER_ARCHITECTURE.md
    └── NER_TRAINING_PLAN.md
```

## Quick Start

### Installation

```bash
cd Swamma
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Training

```bash
# Start training on GPU
julia --project=. scripts/train_ner_production.jl --synthetic

# Or with config file
julia --project=. scripts/train_ner_production.jl --config configs/ner_production_110m.toml
```

### Inference

```julia
using Swamma

# Load model
config = load_ner_config("configs/ner_production_110m.toml")
model = SwammaNER(config)
ps, st = load_checkpoint("checkpoints/ner_110m/latest.jls")

# Predict
text = "Barack Obama visited Paris"
tokens, labels, entities = predict(model, ps, st, text)
# entities: [(text="Barack Obama", label="PERSON"), (text="Paris", label="PLACE")]
```

## Model Configurations

| Config | Embedding | Layers | Heads | Params | Use Case |
|--------|-----------|--------|-------|--------|----------|
| `tiny` | 64 | 2 | 2 | ~500K | Debugging |
| `small` | 256 | 4 | 4 | ~5M | Experiments |
| `base` | 384 | 6 | 6 | ~15M | Production |
| `large` | 512 | 12 | 8 | ~50M | High accuracy |

## Training Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Text    │────►│  Tokenizer   │────►│  NERDataset  │
│  + Labels    │     │  (BPE 32k)   │     │  (batching)  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐
│  Checkpoint  │◄────│   Optimizer  │◄────│   Model      │
│  (every 1k)  │     │ (AdamW+cos)  │     │  Forward     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌──────────────┐     ┌──────▼───────┐
                     │   Metrics    │◄────│  NER Loss    │
                     │  (F1, etc)   │     │  + CRF Loss  │
                     └──────────────┘     └──────────────┘
```

## Dependencies

- **Lux.jl** - Neural network framework
- **NNlib.jl** - Neural network primitives
- **Zygote.jl** - Automatic differentiation
- **CUDA.jl** - GPU support
- **Optimisers.jl** - Adam, learning rate schedules

## License

MIT

## Citation

```bibtex
@software{ossamma2024,
  title={Swamma: Oscillatory State Space Attention for NER},
  year={2024},
  url={https://github.com/your-repo/ossamma}
}
```
