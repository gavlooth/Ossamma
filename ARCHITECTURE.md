# Architecture Documentation

## Overview

The active architecture in this repo is built around a composite `SwammaBlock`:
1. **SWAttention**: multi-head sliding-window attention for sharp local structure
2. **WavePDE**: projection-free spectral wave dynamics used as a gate path
3. **LinearAttention**: efficient global content path

The old legacy state-space path is no longer part of the active block design.

---

## LLaDA Model (Text Diffusion LLM)

The main model to be trained is **LLaDAModel** - a discrete text diffusion language model using Swamma architecture.

```
                          ┌──────────────────────────────────────────────────────────────┐
                          │                      LLaDAModel                               │
                          │              Text Diffusion Language Model                    │
                          └──────────────────────────────────────────────────────────────┘
                                                      │
                          ┌───────────────────────────┼───────────────────────────┐
                          │                           │                           │
                    ┌─────▼─────┐             ┌───────▼───────┐           ┌───────▼───────┐
                    │  Token    │             │   Position    │           │    Time       │
                    │ Embedding │             │   Embedding   │           │  Embedding    │
                    │(vocab→dim)│             │  (pos→dim)    │           │ (sinusoidal   │
                    └─────┬─────┘             └───────┬───────┘           │  + MLP)       │
                          │                           │                   └───────┬───────┘
                          └────────────┬──────────────┘                           │
                                       │ + (add)                        mask_ratio t ∈ [0,1]
                                       ▼                                          │
                          ┌────────────────────────┐                              │
                          │     hidden (d, L, B)   │◄─────────────────────────────┘
                          └────────────┬───────────┘
                                       │
                          ╔════════════▼════════════╗
                          ║                         ║
                          ║  SwammaBlock × N       ║  (N = number_of_layers)
                          ║                         ║
                          ╚════════════╤════════════╝
                                       │
                          ┌────────────▼────────────┐
                          │      LayerNorm          │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   Output Head (Dense)   │
                          │     dim → vocab_size    │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   Logits (V, L, B)      │
                          └─────────────────────────┘
```

### SwammaBlock Detail

``` 
╔═════════════════════════════════════════════════════════════════════════════════════════════╗
║                                      SwammaBlock                                            ║
║       Spectral Wave-Attention Masked Mixer Architecture                                     ║
╠═════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                             ║
║   Input ──────────────────────────────────────────────────────────────────── Residual       ║
║      │                                                                           │          ║
║      ▼                                                                           │          ║
║   ┌───────────────────────────────┐                                              │          ║
║   │  Time-Conditioned LayerNorm   │◄── time_emb                                  │          ║
║   │ returns x_norm and α_bias(t)  │                                              │          ║
║   └───────────────┬───────────────┘                                              │          ║
║                   │ x_norm                                                       │          ║
║      ┌────────────┴────────────┐                                                 │          ║
║      │                         │                                                 │          ║
║      ▼                         ▼                                                 │          ║
║ ┌──────────────────────────────────────────┐      ┌────────────────────────────┐ │          ║
║ │       Global GLU-Gated Branch            │      │        Local Branch        │ │          ║
║ │                                          │      │                            │ │          ║
║ │ Dense(d→2d)                              │      │ normalized                 │ │          ║
║ │   ├─ content_half ──► LinearAttention    │      │   ⊙ local_gate            │ │          ║
║ │   │                    ↓                 │      │          ↓                │ │          ║
║ │   │                 RMSNorm              │      │      SWAttention          │ │          ║
║ │   └─ gate_half ─────► WavePDE            │      │                            │ │          ║
║ │                        ↓                 │      └──────────────┬─────────────┘ │          ║
║ │                     RMSNorm                              ▲      │               │          ║
║ │                        ↓                                 │      │               │          ║
║ │                     sigmoid                              │      │               │          ║
║ │                        ↓                                 │      │               │          ║
║ │          global = content_norm ⊙ gate_norm               │      │               │          ║
║ └──────────────────────────┬───────────────────────────────┘      │               │          ║
║                            │                                      │               │          ║
║                            │  global -> local control             │               │          ║
║                            ▼                                      │               │          ║
║                 ┌────────────────────────┐                        │               │          ║
║                 │ InputGate(glu_out)     │ = local_gate          │               │          ║
║                 └─────────────┬──────────┘                        │               │          ║
║                               └──────────────────┬────────────────┘               │          ║
║                                                  ▼                                           ║
║                                   ┌────────────────────────────┐                             ║
║                                   │   Adaptive Token Mixing    │                             ║
║                                   │ α·global + (1-α)·local     │                             ║
║                                   │ α = σ(Wα·x_norm + α_bias)  │                             ║
║                                   └─────────────┬──────────────┘                             ║
║                                                 ▼                                            ║
║                                   ┌────────────────────────────┐                             ║
║                                   │    Dropout → SwiGLU FFN    │                             ║
║                                   └─────────────┬──────────────┘                             ║
║                                                 │                                            ║
║                                                 └─────────────────┬──────────────────────────╣
║                                                                   │ + (residual)            ║
║                                                                   ▼                         ║
║                                                        ┌────────────────────────┐           ║
║                                                        │       LayerNorm        │           ║
║                                                        └───────────┬────────────┘           ║
║                                                                    │                        ║
║                                                                    ▼                        ║
║                                                                 Output                      ║
╚═════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Key Components

| Component | Description |
|-----------|-------------|
| **WavePDE** | Projection-free spectral wave gate with internal PDE integration |
| **LinearAttention** | O(L) complexity, ELU+1 feature map |
| **RMSNorm** | Root Mean Square normalization before GLU gating (stabilizes training) |
| **Token-wise α** | Per-token mixing: α_t = σ(Wα·h_t + bias), not sequence-global |
| **SWAttention** | Local window softmax attention with causal masking |

### Current Architectural Thesis

The current block is designed around complementary roles, not interchangeable
mechanisms:

- `LinearAttention` is the scalable global content carrier.
- `WavePDE` gates that global path with a structured smooth dynamical prior, so
  the encoder does not reduce to a conventional attention-only stack.
- `InputGate(glu_out)` is the explicit global-to-local control channel. The
  global branch enriches and filters what the local branch should examine.
- `SWAttention` is the local refinement operator. It handles sharp neighborhood
  corrections after the global branch has already contextualized the token stream.
- `α`-mixing is the arbitration step between global structure and local correction.

The intended processing order is:

```text
global interpretation -> local conditioning -> local refinement -> adaptive merge
```

### Training Mode (Text Diffusion)

```
  "The cat sat on mat"                 Fully Masked               Iterative Denoising
          │                                 │                            │
          ▼                                 ▼                            ▼
  ┌───────────────┐                ┌─────────────────┐          ┌───────────────┐
  │ Forward pass  │   t→1          │ [M] [M] [M] [M] │  t→0     │ Reverse pass  │
  │ (masking)     │ ──────────────►│ [M]             │ ────────►│ (denoising)   │
  └───────────────┘                └─────────────────┘          └───────────────┘
```

The model follows the **LLaDA** paradigm - a discrete text diffusion model:
1. **Forward**: progressively masks tokens (clean → fully masked)
2. **Reverse**: iteratively predicts and unmasks tokens based on confidence (masked → clean)

### Model Configurations

| Config | vocab | embed_dim | heads | layers | seq_len |
|--------|-------|-----------|-------|--------|---------|
| small | 1000 | 64 | 2 | 2 | 64 |
| default | 32000 | 256 | 4 | 6 | 512 |
| base | 32000 | 512 | 8 | 12 | 512 |
| large | 32000 | 1024 | 16 | 24 | 1024 |
| **production** | 32000 | 768 | 12 | 12 | 1024 |

### Core Innovation

The **SwammaBlock** combines:
- **Global structured path**: `LinearAttention(content)` gated by `sigmoid(WavePDE(gate))`
- **Local sharp path**: sliding-window attention over a GLU-conditioned local input
- **Adaptive mixing**: learns when to use global vs local based on content and diffusion timestep `t`

---

## Current Architecture

### SWAttention (Sliding Window Attention)

**Core Design:**
```
Input (dimension, T)
    ↓
[Q, K, V] Dense Projections (dimension → dimension)
    ↓
Split into H heads (d_k per head, where d_k = dimension / H)
    ↓
Per-head computation:
    - Attention scores: Q' * K / √d_k → (T, T)
    - Normalize with sigmoid instead of softmax
    - Weighted values: V * attention_weights → (d_k, T)
    ↓
Concatenate heads (dimension, T)
    ↓
Output projection (dimension → dimension)
    ↓
Output (dimension, T)
```

**Key Innovation:**
- Uses `normalized_sigmoids` instead of `softmax` for attention weights
- Temperature-scaled sigmoid: `σ(x/τ)` normalized to sum to 1
- Each row of attention matrix is independently normalized

**Current Implementation Details:**
- Stateless layer (no recurrence)
- Requires `dimension % number_of_heads == 0`
- `sequence_length` parameter stored but not enforced
- All projections are same dimension (no bottlenecks)

---

## SwammaMLM: Triple Hybrid Architecture with Mask-Predict

### Overview

SwammaMLM combines three complementary mechanisms with discrete diffusion (mask-predict) training for an efficient LLM alternative.

**Design Goals:**
- Linear complexity O(n) for large context windows
- Expressivity via multiple complementary mechanisms
- Iterative refinement through partial mask/unmask (discrete diffusion)
- **Semantic understanding** - learned relationships, not fixed transforms

### Design Philosophy: Smart LLM, Not Signal Processing

```
┌─────────────────────────────────────────────────────────────────┐
│  KEY INSIGHT: For language, we need LEARNED relationships      │
│                                                                 │
│  "The cat sat on the mat"                                      │
│       ↑         ↑                                               │
│       └────┬────┘                                               │
│            │                                                    │
│   Relationship is SEMANTIC (subject-verb), not frequency-based │
│                                                                 │
│   ✗ FNet (FFT) - fixed transform, no semantic learning         │
│   ✓ Cosformer  - learned Q/K/V, captures meaning               │
└─────────────────────────────────────────────────────────────────┘
```

### The Three Components

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Cosformer (Global Learned Attention) - O(n)                │
│                                                                 │
│     "What tokens relate SEMANTICALLY?"                         │
│                                                                 │
│     - Learned Q/K/V projections (not fixed like FFT)           │
│     - Linear attention via kernel decomposition                │
│     - cos/sin reweighting for position awareness               │
│     - Captures long-range semantic dependencies                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  2. WavePDE (Damped Linear Oscillatory SSM) - O(n)             │
│                                                                 │
│     "What's the narrative state? What patterns over time?"     │
│                                                                 │
│     - Stateful - carries context across sequence               │
│     - Physics-based temporal memory (spring-damper dynamics)   │
│     - Tracks "the story so far" in oscillator state            │
│     - Multi-frequency response to different pattern timescales │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  3. SWAttention (Sliding Window Attention) - O(n·w)            │
│                                                                 │
│     "What are the PRECISE local relationships?"                │
│                                                                 │
│     - Hard window for exact neighbor attention                 │
│     - Sigsoftmax for sharper attention patterns                │
│     - Captures syntax, grammar, local coherence                │
│     - "the [adjective] [noun]" - precise local structure       │
└─────────────────────────────────────────────────────────────────┘
```

### Why NOT FNet for LLMs?

| Aspect | FNet | Cosformer |
|--------|------|-----------|
| **Transform** | Fixed (FFT) | Learned (Q/K/V) |
| **Semantics** | None - frequency mixing | Yes - learns what to attend to |
| **Relationships** | Based on position frequency | Based on meaning |
| **Best for** | Signals, audio, time series | Language, semantics |
| **For Smart LLM** | ✗ Not appropriate | ✓ Designed for this |

FNet is elegant for signal processing where frequency matters. But language understanding requires **learned semantic relationships** - that's what Cosformer provides.

### Gating Strategy

**Key Insight:** Use GLU-style gating for similar mechanisms, mixture gating for different ones.

```
                     ┌─────────────────────────────────────┐
                     │  Similarity Principle               │
                     │                                     │
                     │  Cosformer ←──→ WavePDE            │
                     │  (both O(n), both recurrent-form)   │
                     │  → GLU-style gating                 │
                     │                                     │
                     │  (Cos+DLIN) ←──→ SWAttention       │
                     │  (different mechanisms)             │
                     │  → Mixture gating                   │
                     └─────────────────────────────────────┘
```

#### Full Forward Pass

```
Input: x (Features, SeqLen, Batch)

┌──────────────────────────────────────────────────────────────────┐
│ 1. DOUBLE PROJECTION (not split - each sees full input)         │
│                                                                  │
│    x ──┬──→ W_cosformer ──→ x_cos                               │
│        │                                                         │
│        ├──→ W_dlinoss ────→ x_dlin                              │
│        │                                                         │
│        └──→ W_attention ──→ x_attn                              │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. PARALLEL PROCESSING                                           │
│                                                                  │
│    x_cos  ──→ Cosformer ────→ y_cos      (O(n), global)         │
│                                                                  │
│    x_dlin ──→ WavePDE ──────→ y_dlin     (O(n), stateful)       │
│                                                                  │
│    x_attn ──→ SWAttention ──→ y_attn     (O(n·w), local)        │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. GLU GATE (Cosformer + WavePDE - similar mechanisms)          │
│                                                                  │
│    y_linear = y_cos ⊙ σ(y_dlin)                                 │
│                                                                  │
│    Intuition: WavePDE temporal state gates what global          │
│               information from Cosformer passes through         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. MIXTURE GATE (Linear + Attention - different mechanisms)     │
│                                                                  │
│    g = σ(W_mix · x + b_mix)     # learned, input-dependent      │
│                                                                  │
│    y_combined = g ⊙ y_linear + (1 - g) ⊙ y_attn                 │
│                                                                  │
│    Intuition: Model learns when to use global-linear            │
│               vs local-precise processing                       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. OUTPUT HEADS                                                  │
│                                                                  │
│    logits     = unmask_head(y_combined)    # → vocab_size       │
│    confidence = σ(confidence_head(y_combined))  # → scalar      │
│                                                                  │
│    confidence helps decide which tokens to unmask               │
└──────────────────────────────────────────────────────────────────┘

Output: (logits, confidence), new_state
```

### Why This Gating Makes Sense

| Gate Type | Components | Reasoning |
|-----------|------------|-----------|
| **GLU** | Cosformer + WavePDE | Both O(n), both have recurrent interpretations. One naturally modulates the other. |
| **Mixture** | GLU_output + SWAttention | Fundamentally different operations. Model should learn when each is useful. |

**GLU is wrong when:**
- Components do fundamentally different things
- One component's output doesn't naturally "gate" the other
- You need both outputs to contribute information (not just modulate)

**Mixture is wrong when:**
- Components are so similar that gating makes more sense
- You want multiplicative interaction (GLU) not additive mixing

### Mask-Predict Training (Discrete Diffusion)

#### Training Phase

```
┌────────────────────────────────────────────────────────────────┐
│ Training Step                                                  │
│                                                                │
│ 1. Full sequence:    [The] [cat] [sat] [on] [the] [mat]       │
│                                                                │
│ 2. Random mask:      [The] [MASK] [sat] [MASK] [the] [MASK]   │
│    (e.g., 40%)                                                 │
│                                                                │
│ 3. Forward pass  →  predict all [MASK] positions              │
│                                                                │
│ 4. Loss = CrossEntropy(predictions, targets)                  │
│           only on masked positions                             │
└────────────────────────────────────────────────────────────────┘
```

**Mask ratio strategies:**
- Fixed: 15% (BERT-style) or 40-50% (more generative)
- Curriculum: Start easy (15%) → increase to hard (50%+)
- Random: Sample mask ratio uniformly each batch

#### Inference Phase (Iterative Unmasking)

```
┌────────────────────────────────────────────────────────────────┐
│ Iterative Unmasking (K steps)                                  │
│                                                                │
│ Step 0: [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]             │
│         (fully masked or partially prompted)                   │
│                 │                                              │
│                 ▼ forward pass                                 │
│         predictions: [The:0.9] [dog:0.3] [sat:0.8] ...        │
│         confidence:  [0.95]    [0.40]    [0.88]   ...         │
│                 │                                              │
│                 ▼ unmask top-k confident                       │
│                                                                │
│ Step 1: [The] [MASK] [sat] [MASK] [the] [MASK]                │
│                 │                                              │
│                 ▼ forward pass (refined context!)              │
│                 │                                              │
│                 ▼ unmask top-k confident                       │
│                                                                │
│ Step 2: [The] [MASK] [sat] [on] [the] [MASK]                  │
│                 │                                              │
│                 ▼ ...                                          │
│                                                                │
│ Step K: [The] [cat] [sat] [on] [the] [mat]  ✓ done            │
└────────────────────────────────────────────────────────────────┘
```

**Key insight:** Each step has more context → better predictions → iterative refinement

#### Why Mask-Predict + This Architecture?

| Component | Role in Mask-Predict |
|-----------|---------------------|
| **Cosformer** | Aggregate global context from revealed tokens efficiently |
| **WavePDE** | Track "state of knowledge" as tokens are progressively revealed |
| **SWAttention** | Ensure local coherence between adjacent revealed tokens |
| **confidence_head** | Decide which predictions are reliable enough to commit |

### Struct Definition (Julia/Lux)

```julia
struct SwammaMLM <: Lux.AbstractLuxLayer
    # Dimensions
    input_dim::Int
    hidden_dim::Int
    vocab_size::Int

    # Input projections (double projection - each sees full input)
    proj_cosformer::Lux.Dense    # input_dim → hidden_dim
    proj_dlinoss::Lux.Dense      # input_dim → hidden_dim
    proj_attention::Lux.Dense    # input_dim → hidden_dim

    # Core components
    cosformer::Cosformer         # O(n) global linear attention
    dlinoss::WavePDE             # O(n) oscillatory SSM
    swattention::SWAttention     # O(n·w) local attention

    # Gating
    mixture_gate::Lux.Dense      # hidden_dim → hidden_dim (for sigmoid)

    # Output heads
    unmask_head::Lux.Dense       # hidden_dim → vocab_size
    confidence_head::Lux.Dense   # hidden_dim → 1
end
```

### Cosformer: Linear Attention with cos/sin Reweighting

#### The Problem with Standard Attention

```
Standard:  Attention(Q,K,V) = softmax(QK^T / √d) · V

           QK^T is (SeqLen × SeqLen) → O(n²) memory and compute
```

#### Cosformer Solution

```
Key insight: softmax(QK^T) ≈ φ(Q) · φ(K)^T  for some kernel φ

Cosformer uses:
    φ(x) = ReLU(x) ⊙ cos(π·pos / 2·max_pos)

Then:
    Attention(Q,K,V) = φ(Q) · (φ(K)^T · V) / (φ(Q) · φ(K)^T · 1)
                       \_____/  \________/
                       (d × n)   (n × d)
                              ↓
                           (d × d) intermediate!

    This is O(n) instead of O(n²)
```

#### Why cos/sin Reweighting?

```
Position 0:   cos(0) = 1.0      (full weight)
Position T/4: cos(π/4) ≈ 0.71
Position T/2: cos(π/2) = 0.0    (zero weight)

Creates position-dependent decay: nearby positions contribute more
Without explicit position encodings!
```

#### Cosformer Struct (Conceptual)

```julia
struct Cosformer <: Lux.AbstractLuxLayer
    dim::Int
    num_heads::Int
    head_dim::Int
    max_seq_len::Int

    # Projections
    query_proj::Lux.Dense
    key_proj::Lux.Dense
    value_proj::Lux.Dense
    output_proj::Lux.Dense
end

# Key operation: linear attention with cos reweighting
function linear_attention(Q, K, V, cos_weights)
    # Apply ReLU and cos reweighting
    Q_prime = relu.(Q) .* cos_weights  # (head_dim, seq_len)
    K_prime = relu.(K) .* cos_weights  # (head_dim, seq_len)

    # Compute in O(n): φ(Q) · (φ(K)^T · V)
    KV = K_prime * V'                   # (head_dim, head_dim)
    QKV = Q_prime' * KV                 # (seq_len, head_dim)

    # Normalize
    K_sum = sum(K_prime, dims=2)        # (head_dim, 1)
    normalizer = Q_prime' * K_sum       # (seq_len, 1)

    return QKV ./ (normalizer .+ ε)
end
```

### Training Details

#### Loss Function

```
L_total = L_mlm + λ · L_confidence

where:
    L_mlm = CrossEntropy(predicted_tokens, true_tokens)
            # only on masked positions

    L_confidence = BinaryCrossEntropy(confidence, was_correct)
            # calibration: high confidence should mean correct
```

#### Hyperparameters

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `mask_ratio` | 0.15 - 0.50 | Higher = harder, more generative |
| `num_unmasking_steps` | 4 - 12 | More = better quality, slower |
| `unmask_per_step` | 1/num_steps | Fraction to reveal each iteration |
| `temperature` | 0.7 - 1.0 | For sampling during inference |
| `confidence_threshold` | 0.8 - 0.95 | Minimum confidence to unmask |

#### Curriculum Learning (Optional)

```
Epoch 1-10:   mask_ratio = 0.15   # Easy (BERT-style)
Epoch 11-20:  mask_ratio = 0.30   # Medium
Epoch 21-30:  mask_ratio = 0.50   # Hard
Epoch 31+:    mask_ratio ~ U(0.15, 0.60)  # Random for robustness
```

### Comparison: AR vs Mask-Predict

| Aspect | Autoregressive (GPT) | Mask-Predict (SwammaMLM) |
|--------|---------------------|--------------------------|
| **Generation** | Left-to-right, one token at a time | Parallel, iterative refinement |
| **Speed** | O(n) sequential steps | O(K) steps where K << n |
| **Can fix mistakes?** | No (committed once generated) | Yes (iterative refinement) |
| **Bidirectional context?** | No (only left context) | Yes (sees all revealed tokens) |
| **Variable compute?** | No (always n steps) | Yes (more steps = better) |

### Architecture Synergy

```
┌───────────────────────────────────────────────────────────────┐
│                    Why These Three?                           │
│                                                               │
│  Challenge              Component         How it Helps        │
│  ─────────────────────────────────────────────────────────── │
│  Long-range deps        Cosformer         O(n) global mixing  │
│                                                               │
│  Sequential patterns    WavePDE           Stateful oscillator │
│                                           memory              │
│                                                               │
│  Local coherence        SWAttention       Precise local       │
│                                           attention           │
│                                                               │
│  Iterative refinement   Mask-Predict      Progressive         │
│                                           unmasking           │
│                                                               │
│  Flexible compute       confidence_head   Variable steps      │
│                                           based on certainty  │
└───────────────────────────────────────────────────────────────┘
```

### Implementation Roadmap

```
Phase 1: Core Components
├── [ ] Implement Cosformer (linear attention)
├── [ ] Wire existing WavePDE
├── [ ] Wire existing SWAttention
└── [ ] Verify each component works standalone

Phase 2: SwammaMLM Layer
├── [ ] Create SwammaMLM struct
├── [ ] Implement double projection
├── [ ] Implement GLU gate (Cos + DLIN)
├── [ ] Implement mixture gate (Linear + Attn)
└── [ ] Add output heads (unmask + confidence)

Phase 3: Mask-Predict Training
├── [ ] Implement masking utilities
├── [ ] Implement MLM loss (masked positions only)
├── [ ] Implement confidence loss
└── [ ] Training loop with curriculum

Phase 4: Inference
├── [ ] Implement iterative unmasking loop
├── [ ] Add temperature sampling
├── [ ] Add confidence thresholding
└── [ ] Benchmark generation quality vs speed
```

---

## Alternative: FNet-Style Global Mixing

### Why Consider FNet Over Cosformer?

FNet (Google, 2021) replaces attention entirely with Fourier transforms. For SwammaMLM, this creates an elegant synergy with WavePDE.

### FNet vs Cosformer Comparison

| Aspect | Cosformer | FNet |
|--------|-----------|------|
| **Mechanism** | Linear attention + cos/sin | Pure FFT |
| **Complexity** | O(n) | O(n log n) |
| **Learnable mixing** | Yes (Q/K/V) | No (fixed FFT) |
| **Parameters** | ~4 × d² | 0 (or minimal) |
| **Expressivity** | Higher | Lower |
| **Speed** | Fast | Faster |

### The Frequency-Domain Synergy

```
┌─────────────────────────────────────────────────────────────────┐
│                  Why FNet + WavePDE Works                       │
│                                                                 │
│  FNet:    "What frequencies are present in the input?"          │
│           Static decomposition via FFT                          │
│                                                                 │
│  WavePDE: "How should I respond to each frequency over time?"   │
│           Dynamic filtering via learned oscillators             │
│                                                                 │
│  Together: Analysis (FNet) → Filtering (WavePDE)                │
│            Both speak "frequency language"                      │
└─────────────────────────────────────────────────────────────────┘
```

### FNet Mixer Implementation

```julia
struct FNetMixer <: Lux.AbstractLuxLayer
    dim::Int
    use_freq_weights::Bool  # learnable frequency modulation
end

function FNetMixer(dim::Int; use_freq_weights::Bool = true)
    return FNetMixer(dim, use_freq_weights)
end

function Lux.initialparameters(rng::Random.AbstractRNG, layer::FNetMixer)
    if layer.use_freq_weights
        # Learnable per-frequency weights (complex or real)
        return (freq_weights = ones(Float32, layer.dim),)
    else
        return (;)  # no parameters
    end
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::FNetMixer)
    return (;)  # stateless
end

function (layer::FNetMixer)(x, params, state)
    # x: (features, seq_len, batch) or (features, seq_len)

    # 1. FFT along sequence dimension (dim 2)
    x_fft = fft(x, 2)

    # 2. Optional: learnable frequency modulation
    if layer.use_freq_weights
        # Broadcast weights across sequence positions
        x_fft = x_fft .* reshape(params.freq_weights, :, 1, 1)
    end

    # 3. IFFT back to sequence domain
    x_mixed = real(ifft(x_fft, 2))

    return x_mixed, state
end
```

### FNet Variants

#### 1. Pure FNet (Original)
```
x → FFT → IFFT → output
```
- No learnable parameters in mixing
- Simplest, fastest
- 92-97% of BERT performance

#### 2. FNet + Frequency Weights
```
x → FFT → W_freq ⊙ X_fft → IFFT → output
```
- Learnable per-frequency scaling
- Allows model to emphasize/suppress certain frequencies
- Minimal parameter overhead

#### 3. FNet + Frequency MLP
```
x → FFT → MLP(X_fft) → IFFT → output
```
- Full learnable transform in frequency domain
- More expressive, more parameters
- Still O(n log n)

#### 4. Hybrid: FNet + Sparse Attention
```
x → FNet (global) + SWAttention (local) → gated combine
```
- FNet handles global mixing cheaply
- Attention only for local precision
- Best of both worlds

### Updated SwammaMLM with FNet

```
┌──────────────────────────────────────────────────────────────────┐
│ SwammaMLM (FNet Variant)                                         │
│                                                                  │
│ Input: x (Features, SeqLen, Batch)                               │
│                                                                  │
│ 1. PROJECTIONS                                                   │
│    x ──┬──→ W_fnet ────→ x_fnet     (or skip if pure FNet)      │
│        ├──→ W_dlinoss ──→ x_dlin                                │
│        └──→ W_attention → x_attn                                │
│                                                                  │
│ 2. PARALLEL PROCESSING                                           │
│    x_fnet ──→ FNetMixer ───→ y_fft    (O(n log n), global)      │
│    x_dlin ──→ WavePDE ─────→ y_dlin   (O(n), temporal)          │
│    x_attn ──→ SWAttention ─→ y_attn   (O(n·w), local)           │
│                                                                  │
│ 3. GLU GATE (FNet + WavePDE)                                    │
│    y_freq = y_fft ⊙ σ(y_dlin)                                   │
│                                                                  │
│    ↑ Both in frequency domain - natural pairing!                │
│                                                                  │
│ 4. MIXTURE GATE                                                  │
│    g = σ(W_mix · x)                                             │
│    y_combined = g ⊙ y_freq + (1-g) ⊙ y_attn                     │
│                                                                  │
│ 5. OUTPUT HEADS                                                  │
│    logits = unmask_head(y_combined)                             │
│    confidence = σ(confidence_head(y_combined))                  │
└──────────────────────────────────────────────────────────────────┘
```

### Why GLU Makes Even More Sense Now

With Cosformer + WavePDE, GLU worked because both were O(n) and had recurrent forms.

With FNet + WavePDE, GLU is **even more natural**:

```
FNet output:   Frequency-decomposed representation
               "Here are the frequency components"

WavePDE output: Oscillator responses (σ applied)
                "Here's how important each frequency is right now"

GLU combination: FNet ⊙ σ(WavePDE)
                 "Pass frequencies that matter, gate others"
```

This is essentially **learned frequency-domain gating**.

### Struct Definition (FNet Variant)

```julia
struct SwammaMLM_FNet <: Lux.AbstractLuxLayer
    # Dimensions
    input_dim::Int
    hidden_dim::Int
    vocab_size::Int

    # Input projections
    proj_fnet::Lux.Dense        # optional, can skip for pure FNet
    proj_dlinoss::Lux.Dense
    proj_attention::Lux.Dense

    # Core components
    fnet::FNetMixer             # O(n log n) global frequency mixing
    dlinoss::WavePDE            # O(n) oscillatory SSM
    swattention::SWAttention    # O(n·w) local attention

    # Gating
    mixture_gate::Lux.Dense

    # Output heads
    unmask_head::Lux.Dense
    confidence_head::Lux.Dense
end
```

### When to Use FNet vs Cosformer

| Use Case | Recommendation |
|----------|----------------|
| **Maximum speed** | FNet (pure) |
| **Minimum parameters** | FNet (pure) |
| **Strong frequency patterns** | FNet + WavePDE (natural synergy) |
| **Need learned attention** | Cosformer |
| **Complex token relationships** | Cosformer |
| **Research/exploration** | Try both, compare |

### Performance Expectations

```
Speed (relative):
  Cosformer:     1.0x (baseline)
  FNet (pure):   1.5-2x faster
  FNet + weights: 1.3-1.7x faster

Parameters (relative):
  Cosformer:     1.0x (baseline, ~4d² for Q/K/V/O)
  FNet (pure):   0x (no mixing params)
  FNet + weights: 0.01x (just d parameters)

Quality (estimated):
  Cosformer:     1.0x
  FNet (pure):   0.92-0.97x (per FNet paper)
  FNet + WavePDE: possibly better for frequency-rich data
```

### References

- **FNet**: Lee-Thorp et al., ["FNet: Mixing Tokens with Fourier Transforms"](https://arxiv.org/abs/2105.03824) (NAACL 2022)
- **Cosformer**: Qin et al., ["COSFORMER: Rethinking Softmax in Attention"](https://arxiv.org/abs/2202.08791) (ICLR 2022)
- **Linear Attention**: Katharopoulos et al., ["Transformers are RNNs"](https://arxiv.org/abs/2006.16236) (ICML 2020)
- **Mask-Predict**: Ghazvininejad et al., ["Mask-Predict: Parallel Decoding"](https://arxiv.org/abs/1904.09324) (EMNLP 2019)
- **MaskGIT**: Chang et al., ["MaskGIT: Masked Generative Image Transformer"](https://arxiv.org/abs/2202.04200) (CVPR 2022)
- **Discrete Diffusion**: Austin et al., ["Structured Denoising Diffusion Models in Discrete State-Spaces"](https://arxiv.org/abs/2107.03006) (NeurIPS 2021)

---

## Deep Scaling Strategies (DeepScaling.jl)

Swamma's O(T) complexity (vs Transformer's O(T²)) allows **4-8× more layers** for the same compute budget. The DeepScaling module implements strategies to leverage this advantage.

### Core Insight: Depth vs Width Trade-off

```
Transformer:  Layers = C / (T² × d)
Swamma:      Layers = C / (T × d²)

Ratio = T / d

For T=2048, d=512: Swamma can afford 4× more layers
For T=4096, d=512: Swamma can afford 8× more layers
```

### 1. Hierarchical Frequency Ranges

Different oscillator frequency ranges per layer depth:

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1-12 (early):   freq ∈ [1.0, 100.0]  ← Fast oscillations │
│                        Captures: local syntax, adjacent words   │
│                                                                 │
│  Layer 13-24 (mid):    freq ∈ [0.1, 22.0]   ← Medium            │
│                        Captures: phrases, clauses               │
│                                                                 │
│  Layer 25-48 (late):   freq ∈ [0.02, 5.0]   ← Slow oscillations │
│                        Captures: document-level, long-range     │
└─────────────────────────────────────────────────────────────────┘
```

**Usage:**
```julia
using Swamma

# Configure hierarchical frequencies
freq_config = HierarchicalFrequencyConfig(
    base_min_freq = 0.01f0,
    base_max_freq = 100.0f0,
    decay_rate = 3.0f0,
    scaling_type = :exponential  # or :linear, :logarithmic
)

# Get frequency range for a specific layer
min_freq, max_freq = compute_layer_frequencies(24, 48, freq_config)

# Print summary
frequency_summary(48, freq_config)
```

**Scaling Types:**
| Type | Behavior | Best For |
|------|----------|----------|
| `:exponential` | Fast decay early, slow late | Most cases |
| `:linear` | Uniform transition | Moderate depth |
| `:logarithmic` | Gradual decay | Very deep (96+) |

### 2. Layer Scale Initialization

Stabilizes very deep networks by scaling residual contributions:

```julia
# residual + layer_scale * block_output
# where layer_scale starts small and is learnable
```

**Configuration:**
```julia
scale_config = LayerScaleConfig(
    init_value = 0.1f0,    # Starting scale (smaller for deeper)
    learnable = true,       # Allow scale to be learned
    per_channel = true      # Per-dimension vs scalar
)
```

**Recommended init values:**
| Depth | init_value |
|-------|------------|
| 24L | 0.1 |
| 48L | 0.1 |
| 96L | 0.01 |
| 192L | 1e-4 |

### 3. Stochastic Depth (Drop Path)

Regularization by randomly skipping layers during training:

```julia
depth_config = StochasticDepthConfig(
    drop_rate = 0.1f0,     # Max drop probability (for deepest layer)
    mode = :linear         # or :uniform
)

# Linear mode: Layer 1 = 0% drop, Layer 48 = 10% drop
# Uniform mode: All layers = 10% drop
```

**Benefits:**
- Regularization (prevents overfitting)
- Faster training (fewer layers computed)
- Implicit ensemble effect

### 4. Gradient Checkpointing

Memory-efficient training by recomputing activations:

```julia
checkpoint_config = CheckpointConfig(
    checkpoint_every = 4,   # Checkpoint every 4 layers
    enabled = true
)

# Memory reduction:
# 48L without checkpoint: 48 × activations
# 48L with checkpoint (every 4): 12 × activations + recompute
```

### 5. SwammaBlockDeep

Deep-optimized block variant with all strategies built-in:

```julia
block = SwammaBlockDeep(
    384,                  # embedding_dimension
    4096,                 # sequence_length
    6,                    # number_of_heads
    64;                   # time_dimension
    layer_idx = 24,
    total_layers = 48,
    block_type = :global_only,  # :full, :global_only, :local_only
    freq_config = HierarchicalFrequencyConfig(),
    use_layer_scale = true,
    layer_scale_init = 0.1f0,
    use_stochastic_depth = true,
    stochastic_depth_rate = 0.1f0,
    use_parallel_scan = true,
)
```

**Block Types:**
| Type | Components | Use Case |
|------|------------|----------|
| `:full` | LinearAttn + WavePDE + SWAttention | Full expressivity |
| `:global_only` | LinearAttn + WavePDE | Semantic layers |
| `:local_only` | SWAttention only | Syntax layers |

### 6. Block Type Schedules

Different layer types at different depths:

```julia
# PROGRESSIVE: local → global → full
# Early layers: :local_only (syntax)
# Mid layers: :global_only (semantics)
# Late layers: :full (integration)

# SANDWICH: full at edges, lightweight in middle
# First/last 15%: :full
# Middle: :global_only

# ALTERNATING: cycle through types
# Every 4th: :full
# Even: :global_only
# Odd: :local_only
```

### 7. Deep Model Configurations

Pre-built configurations for common use cases:

```julia
# 48-layer deep model (~120M params)
config = deep_48L_config(
    vocab_size = 32000,
    max_sequence_length = 4096
)

# 96-layer ultra-deep (~100M params)
config = ultra_96L_config()

# Long context optimized (16K+ sequences)
config = long_context_config(
    max_sequence_length = 16384
)

# Create blocks from config
blocks = create_deep_blocks(config)

# Print summary
print_model_summary(config)
```

### Configuration Comparison

| Config | Layers | Dim | Heads | Params | Best For |
|--------|--------|-----|-------|--------|----------|
| `deep_48L` | 48 | 384 | 6 | ~120M | Starting point |
| `ultra_96L` | 96 | 256 | 4 | ~100M | Research |
| `long_context` | 32 | 512 | 8 | ~130M | 16K+ sequences |

### Example: Building a Deep Swamma Model

```julia
using Swamma
using Lux
using Random

# Create configuration
config = deep_48L_config(vocab_size = 32000)

# Print summary
print_model_summary(config)

# Create blocks
blocks = create_deep_blocks(config)

# Initialize
rng = Random.default_rng()
params = [Lux.initialparameters(rng, b) for b in blocks]
states = [Lux.initialstates(rng, b) for b in blocks]

# Forward pass with checkpointing (pseudo-code)
hidden = embeddings
time_emb = sinusoidal_embedding(t)

for (i, (block, ps, st)) in enumerate(zip(blocks, params, states))
    if should_checkpoint(i, CheckpointConfig())
        hidden, st = Zygote.checkpointed(block, (hidden, time_emb), ps, st)
    else
        hidden, st = block((hidden, time_emb), ps, st)
    end
    states[i] = st
end
```

### Performance Expectations

| Mode | GPU Util | Speed |
|------|----------|-------|
| Global-only WavePDE drafter | 80-90% | 0.5-1 sec/step |
| + Diffusion | 80-90% | 8-16× faster generation |

### References

- **Layer Scale**: Touvron et al., ["Going deeper with Image Transformers"](https://arxiv.org/abs/2103.17239) (CaiT, 2021)
- **Stochastic Depth**: Huang et al., ["Deep Networks with Stochastic Depth"](https://arxiv.org/abs/1603.09382) (ECCV 2016)
- **Gradient Checkpointing**: Chen et al., ["Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) (2016)

---

## TiDAR: Speculative Decoding with Granite (TiDAR.jl & Drafter.jl)

TiDAR (Token-level Iterative Drafting with AR Refinement) implements speculative decoding
that pairs a fast, lightweight **SwammaDrafter** model with a large **Granite** autoregressive verifier.

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TiDAR Generation Loop                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐                      ┌─────────────────────────┐      │
│   │  SwammaDrafter │                      │   Granite AR Verifier   │      │
│   │  (~40-100M)     │                      │   (2B/3B/8B params)     │      │
│   │                 │                      │                         │      │
│   │  • O(T) complex │──── K tokens ───────>│  • O(T²) attention      │      │
│   │  • Parallel     │     drafted          │  • Sequential           │      │
│   │  • Diffusion    │                      │  • High quality         │      │
│   └────────┬────────┘                      └───────────┬─────────────┘      │
│            │                                           │                    │
│            │ [MASK] → predictions                      │ verify logits      │
│            │                                           │                    │
│            v                                           v                    │
│   ┌────────────────────────────────────────────────────────────────┐        │
│   │                    Rejection Sampling                          │        │
│   │  • Compare drafter vs verifier predictions                     │        │
│   │  • Accept: matching tokens (or p_ar/p_draft sampling)          │        │
│   │  • Reject: use verifier token, re-draft from that point        │        │
│   └────────────────────────────────────────────────────────────────┘        │
│                               │                                             │
│                               v                                             │
│                     [accepted tokens appended]                              │
│                               │                                             │
│                         (loop until EOS)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### SwammaDrafter Full Architecture

The drafter is built from **SwammaDrafterBlock** layers with time conditioning for diffusion.
This is intentionally not the full `SwammaBlock`: it is the verifier-backed
global-only proposer path.

```
                              token_ids (seq_len, batch)
                                        │
                                        v
┌───────────────────────────────────────────────────────────────────────────────┐
│                           TOKEN EMBEDDING                                     │
│                     Embedding(vocab_size → d)                                 │
│                     vocab = 49155 (Granite 3.1) or 49160 (Granite 4.0)        │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        v
┌───────────────────────────────────────────────────────────────────────────────┐
│                         POSITION EMBEDDING                                    │
│                     Embedding(max_seq_len → d)                                │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        + (elementwise add)
                                        │
                                        v
                              hidden (d, seq, batch)
                                        │
┌───────────────────────────────────────┴───────────────────────────────────────┐
│                                                                               │
│         t ∈ [0,1]                     │                                       │
│             │                         │                                       │
│             v                         │                                       │
│   ┌─────────────────────┐             │                                       │
│   │ SINUSOIDAL TIME EMB │             │                                       │
│   │  sin/cos encoding   │             │                                       │
│   │  (time_dim = 64)    │             │                                       │
│   └─────────┬───────────┘             │                                       │
│             │                         │                                       │
│             v                         │                                       │
│   ┌─────────────────────┐             │                                       │
│   │ TIME MLP EMBEDDING  │             │                                       │
│   │ Dense → GELU → Dense│             │                                       │
│   │ (time_dim → d)      │             │                                       │
│   └─────────┬───────────┘             │                                       │
│             │                         │                                       │
│   sinusoidal_emb (for blocks)         │                                       │
│             │                         │                                       │
└─────────────┼─────────────────────────┼───────────────────────────────────────┘
              │                         │
              v                         v
        ╔═════════════════════════════════════════════════════════════════╗
        ║                  N × SwammaDrafterBlock                        ║
        ║         (6-96 layers depending on configuration)                ║
        ╚═════════════════════════════════════════════════════════════════╝
                                        │
                                        v
┌───────────────────────────────────────────────────────────────────────────────┐
│                            FINAL LAYERNORM                                    │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        v
┌───────────────────────────────────────────────────────────────────────────────┐
│                              LM HEAD                                          │
│                         Dense(d → vocab_size)                                 │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        v
                           logits (vocab, seq, batch)
```

### SwammaDrafterBlock (Single Block Detail)

Each block uses GLU-style gating between `LinearAttention` and `WavePDE`, with
no local branch and no `α`-mixing:

```
                    Input (d, seq, batch)
                            │
            ┌───────────────┴───────────────┐
            │                               │ (residual connection)
            v                               │
┌───────────────────────────────┐           │
│  TIME-CONDITIONED LAYERNORM   │           │
│                               │           │
│  LN(x) → scale(t)·x + shift(t)│◄──── t (time embedding)
│                               │           │
│  Also outputs α_bias (unused   │           │
│  here; retained because the    │           │
│  same norm module is shared)   │           │
└───────────────────────────────┘           │
            │                               │
            v                               │
┌───────────────────────────────┐           │
│     GLU PROJECTION            │           │
│     Dense(d → 2d)             │           │
└───────────────────────────────┘           │
            │                               │
      ┌─────┴─────┐                         │
      │   split   │                         │
      v           v                         │
   path_a      path_b                       │
   (d,seq)     (d,seq)                      │
      │           │                         │
      v           v                         │
┌───────────┐ ┌───────────────────┐         │
│  LINEAR   │ │     WavePDE       │         │
│ ATTENTION │ │   (wave gate)     │         │
│           │ │                   │         │
│ O(n) glob │ │ Sequential state  │         │
│ context   │ │ evolution over    │         │
│           │ │ spectral modes    │         │
│ Q,K,V,O   │ │                   │         │
│ projections│ │                  │         │
└─────┬─────┘ └─────────┬─────────┘         │
      │                 │                   │
      │            sigmoid(·)               │
      │                 │                   │
      └────── ⊙ ────────┘                   │
              │                             │
      (Hadamard product)                    │
              │                             │
              v                             │
┌───────────────────────────────┐           │
│          DROPOUT              │           │
└───────────────────────────────┘           │
              │                             │
              v                             │
┌───────────────────────────────┐           │
│         SwiGLU FFN            │           │
│                               │           │
│  Dense(d → 1.5d) → split      │           │
│       SiLU(a) ⊙ b             │           │
│  Dense(1.5d/2 → d)            │           │
└───────────────────────────────┘           │
              │                             │
              └──────── + ──────────────────┘
                        │
                        v
┌───────────────────────────────┐
│      OUTPUT LAYERNORM         │
└───────────────────────────────┘
                        │
                        v
               Output (d, seq, batch)
```

### Key Components Explained

#### Drafter Design Choice

`SwammaDrafterBlock` is deliberately simpler than `SwammaBlock`.

- The main model keeps `SWAttention` because it must enforce local precision itself.
- The drafter drops the local branch because local correctness can be enforced by
  the AR verifier through rejection.
- That makes the drafter a cheap global proposer rather than a full standalone encoder.

#### 1. LinearAttention (O(n) Global Context)
- Provides **global context** across the sequence without O(n²) cost
- Uses linear attention mechanism (no softmax)
- Processes `path_a` from the GLU split

#### 2. WavePDE (Spectral Wave Gate)
- Projection-free spectral wave dynamics used as a gate over the global path
- Evolves a smooth field with internal PDE integration steps rather than a token-by-token recurrent scan
- Uses learned wave speed and damping over spectral modes
- Provides structured global refinement that complements `LinearAttention`

#### 3. GLU Gating
```
output = LinearAttention(path_a) ⊙ sigmoid(WavePDE(path_b))
```
- Oscillator output gates the attention output
- Sigmoid provides soft gating ∈ (0, 1)

#### 4. Time Conditioning (for Diffusion)
- Drafter uses diffusion-style prediction: `t=0` means "predict all masks"
- Sinusoidal embeddings encode timestep `t ∈ [0, 1]`
- LayerNorm is modulated by time: `scale(t)·LN(x) + shift(t)`

### Granite Verifier Role

**Granite** is IBM's open-source LLM family used as the AR verifier:

| Model | Params | Hidden | Layers | Vocab |
|-------|--------|--------|--------|-------|
| Granite 3.1 2B | 2B | 2048 | 40 | 49155 |
| Granite 3B MoE | 3B (800M active) | 1536 | 32 | 49155 |
| Granite 4.0 | varies | varies | varies | 49160 |
| Granite 8B | 8B | 4096 | 32 | 49155 |

**Critical**: Drafter vocabulary **must match** verifier vocabulary exactly.

### TiDAR Generation Flow

```
Step 1: DRAFTING
────────────────
prefix = [token₁, token₂, ..., tokenₙ]
                    │
                    v
input = [token₁, token₂, ..., tokenₙ, [MASK], [MASK], ..., [MASK]]
                                       ╰───────── K masks ─────────╯
                    │
                    v
            SwammaDrafter(input, t=0)
                    │
                    v
            logits → sample → [draft₁, draft₂, ..., draftₖ]


Step 2: VERIFICATION
────────────────────
full_seq = [token₁, ..., tokenₙ, draft₁, ..., draftₖ]
                    │
                    v
            GraniteVerifier(full_seq)
                    │
                    v
            verifier_logits


Step 3: ACCEPTANCE (Rejection Sampling)
───────────────────────────────────────
For i = 1 to K:
  │
  ├─ p_draft = drafter_probs[draft_i]
  ├─ p_verifier = verifier_probs[draft_i]
  │
  ├─ acceptance_prob = min(1, p_verifier / p_draft)
  │
  └─ if rand() < acceptance_prob:
        ACCEPT draft_i
     else:
        REJECT → use verifier's token, stop


Step 4: RESULT
──────────────
new_prefix = [token₁, ..., tokenₙ, accepted_drafts..., (verifier_token if rejected)]
                    │
              (loop to Step 1)
```

### Configuration Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        TiDARConfig                              │
├─────────────────────────────────────────────────────────────────┤
│  ar_model: "granite_3b" | "granite_8b" | "granite4_3b"          │
│  vocab_size: 49155 (Granite 3.1) or 49160 (Granite 4.0)         │
│  mask_token_id: vocab_size (uses last token position)           │
├─────────────────────────────────────────────────────────────────┤
│  DRAFTER ARCHITECTURE                                           │
│  ├─ embedding_dimension: 384 (narrow for speed)                 │
│  ├─ number_of_layers: 24-96 (deep due to O(T) complexity)       │
│  ├─ number_of_heads: 6                                          │
│  └─ max_sequence_length: 4096-8192                              │
├─────────────────────────────────────────────────────────────────┤
│  DEEP SCALING OPTIMIZATIONS                                     │
│  ├─ HierarchicalFrequencyConfig: layer-wise oscillator freqs    │
│  ├─ LayerScale: learnable per-layer output scaling (init=0.1)   │
│  └─ StochasticDepth: random layer dropping during training      │
├─────────────────────────────────────────────────────────────────┤
│  INFERENCE SETTINGS                                             │
│  ├─ draft_length: 8-12 tokens per step                          │
│  ├─ temperature: 0.9                                            │
│  └─ confidence_threshold: 0.8                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

1. **Speed**: Drafter uses O(T) complexity (WavePDE + LinearAttention), enabling 48-96 layers where a Transformer would be impractical

2. **Quality**: Granite verifier ensures output quality matches the large model

3. **Efficiency**: Most tokens are accepted, so the slow verifier runs infrequently

4. **Parallel Drafting**: Unlike AR models, drafter predicts K tokens simultaneously

### Why Deep Swamma for TiDAR?

| Aspect | Transformer Drafter | Swamma Drafter |
|--------|---------------------|-----------------|
| Complexity | O(T² × d) | **O(T × d²)** |
| Layers for 80M params | 12L × 512d | **48L × 384d** |
| Parallel draft | Limited | **Full (diffusion)** |
| GPU utilization | 80% | **80%+ (parallel scan)** |

### Quick Start

```julia
using Swamma

# Create deep drafter for Granite 3B
config = granite_3b_drafter_deep_config()
print_tidar_config(config)

# Create model
drafter = SwammaDrafterDeep(config)

# Initialize
rng = Random.default_rng()
params = Lux.initialparameters(rng, drafter)
state = Lux.initialstates(rng, drafter)

# Draft tokens
prefix = [1, 2, 3, 4, 5]  # Token IDs
drafted_ids, logits, new_state = draft_tokens(
    drafter, prefix, 8, params, state;
    temperature = 0.9f0
)
```

### TiDAR Inference Loop

```julia
# Pseudo-code for full TiDAR generation

function tidar_generate(drafter, verifier, prompt_ids, max_length)
    prefix = prompt_ids
    drafter_state = initial_state

    while length(prefix) < max_length
        # One TiDAR step
        prefix, accepted, drafter_state = tidar_generate_step(
            drafter, drafter_params, drafter_state,
            verifier,  # Function: ids -> logits
            prefix,
            config.draft_length;
            temperature = config.temperature
        )

        # Stats
        total_tokens += accepted + 1  # accepted + verifier's correction
    end

    return prefix
end
```

### Configuration Options

```julia
# Standard drafter (24L, ~50M params)
config = granite_3b_drafter_config()

# Deep drafter (48L, ~80M params) - recommended
config = granite_3b_drafter_deep_config()

# Custom configuration
config = TiDARConfig(
    ar_model = "granite_3b",
    vocab_size = GRANITE_VOCAB_SIZE,  # 49155
    embedding_dimension = 384,
    number_of_layers = 48,
    number_of_heads = 6,
    max_sequence_length = 4096,

    # Deep scaling
    use_layer_scale = true,
    layer_scale_init = 0.1f0,
    use_stochastic_depth = true,
    stochastic_depth_rate = 0.1f0,
    freq_config = HierarchicalFrequencyConfig(
        base_min_freq = 0.01f0,
        base_max_freq = 100.0f0,
        scaling_type = :exponential
    ),

    # TiDAR settings
    draft_length = 12,
    confidence_threshold = 0.8f0,
    temperature = 0.9f0,
)
```

### Drafter Model Variants

| Config Function | Layers | Dim | Params | Best For |
|-----------------|--------|-----|--------|----------|
| `granite_2b_drafter_config()` | 24 | 384 | ~40M | Granite 2B verifier |
| `granite_3b_drafter_config()` | 32 | 384 | ~60M | Granite 3B verifier |
| `granite_4_3b_drafter_config()` | 32 | 384 | ~60M | Granite 4.0 verifier |
| `granite_8b_drafter_config()` | 48 | 384 | ~80M | Granite 8B verifier |
| `granite_drafter_deep_config()` | 48-96 | 384 | ~80-100M | Maximum acceptance |

### Expected Performance

| Metric | Value |
|--------|-------|
| Drafter params | ~40-100M |
| Verifier params | 2B-8B (Granite) |
| Draft length | 8-12 tokens |
| Acceptance rate | ~60-80% |
| Speedup vs AR | **2-4×** |

### Training the Drafter

The drafter is trained with MLM loss on the same data as the verifier:

```julia
using Swamma

# Create drafter
config = granite_3b_drafter_deep_config()
drafter = SwammaDrafterDeep(config)

# Training config
train_config = DrafterTrainingConfig(
    learning_rate = 1e-4,
    batch_size = 32,
    mask_ratio = 0.15,  # 15% tokens masked
    # ...
)

# Training loop uses drafter_mlm_loss from DrafterTraining module
```

### Key Differences: SwammaDrafter vs Full SwammaBlock

| Feature | Full SwammaBlock | SwammaDrafterBlock |
|---------|-------------------|---------------------|
| SWAttention (local) | ✓ | ✗ (verifier handles local) |
| α-mixing gating | ✓ | ✗ (standard residual) |
| Branches | 2 (Global + Local) | 1 (Global only) |
| Use case | General LM | Fast drafting |

### References

- **Speculative Decoding**: Leviathan et al., ["Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) (ICML 2023)
- **TiDAR (conceptual basis)**: Token-level iterative draft and refine
- **Granite Models**: IBM's Granite 3.1/4.0 family

---

## NER Model Benchmark Findings (2024-12-28)

### Checkpoint: `checkpoints/ner_110m/checkpoint_best.jls`

**Model Architecture** (from checkpoint step 48500):
| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 640 |
| Number of Layers | 10 |
| Number of Heads | 10 |
| Time Dimension | 192 |
| State Dimension | 640 |
| Window Size | 32 |
| Max Sequence Length | 256 |
| FFN Expansion | 1.334375 |
| Vocab Size | 2004 |

**Speed Benchmark** (CPU only, no GPU):
- Throughput: **15.3 tokens/sec** (1.5 sentences/sec)
- Average inference time: **653ms ± 132ms** per sentence

**Accuracy Benchmark**:
- F1 Score: **0%**
- Recall: 0%
- Precision: 0%

### Root Cause: Synthetic Vocabulary

The model was trained with the `--synthetic` flag, which generates placeholder tokens:
```julia
# From generate_synthetic_data() in train_ner_production.jl:269
tokens = ["token_$i" for i in rand(1:vocab_size, seq_len)]
```

**Vocab Contents** (from checkpoint):
```
"token_1406" => 210
"token_1202" => 325
"token_1032" => 1681
...
```

This means:
1. All real English words map to `[UNK]` (token ID 2)
2. The model only learned patterns on synthetic token IDs
3. Embeddings are random for real language input
4. The checkpoint is **not usable for real NER tasks**

### Required Actions

To make the NER model functional:

1. **Retrain with real data**: Use `data/rag/synthetic_work.jsonl` (contains actual English words despite the name):
   ```bash
   julia --project=. scripts/train_ner_production.jl  # Without --synthetic flag
   ```

2. **Ensure data path exists**: The training script falls back to synthetic data if `config.data_dir` doesn't exist

3. **Build proper vocabulary**: From the actual training corpus using `build_vocab()` function

### Files Modified for Benchmark

- `scripts/benchmark_ner.jl` - NER speed/accuracy benchmark
- `scripts/debug_predictions.jl` - Debugging script for model outputs
- `scripts/train_ner_production.jl` - Added `use_ffn`/`ffn_expansion` fields to `TrainingConfig` for checkpoint compatibility
