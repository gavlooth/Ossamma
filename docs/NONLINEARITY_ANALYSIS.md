# Nonlinearity Analysis for OssammaNERBlock

## Question

Does the Local branch (SWAttention) have sufficient nonlinearity, or do we need to add FFN-like capacity?

---

## Current Nonlinearity Inventory

### GLU Branch (Has Transform-Type Nonlinearity)

```
Dense(dim → 2dim)                    ← EXPAND
       │
  ┌────┴────┐
  ▼         ▼
path_a    path_b
  │         │
  ▼         ▼
LinearAttn  DLinOSS
  │         │
  │         ▼
  │      sigmoid                     ← NONLIN: sigmoid activation
  │         │
  ▼         ▼
  a    ⊙    sigmoid(b)               ← NONLIN: multiplicative gating
            │
            ▼
      Dense(dim → dim)               ← CONTRACT
            │
            ▼
        glu_out

STRUCTURE: Expand → Nonlinear Transform → Gate → Contract
SIMILAR TO: SwiGLU / GLU FFN variants
NONLINEARITY TYPE: Transform (creates new feature combinations)
```

### Local Branch (Has Selection-Type Nonlinearity)

```
normalized
    │
    ▼
sigmoid(W_input @ glu_out)           ← NONLIN: sigmoid (selection)
    │
    ▼
gate ⊙ normalized                    ← NONLIN: multiplicative (selection)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      SWAttention                            │
│                                                             │
│  Q = W_q @ x                       linear                   │
│  K = W_k @ x                       linear                   │
│  V = W_v @ x                       linear                   │
│                                                             │
│  scores = Q @ K^T / sqrt(d)        linear                   │
│                                                             │
│  attn = sigsoftmax(scores)         ← NONLIN: sigsoftmax     │
│                                      (selection/weighting)  │
│                                                             │
│  out = attn @ V                    linear                   │
│                                                             │
│  out = W_o @ out                   linear                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
local_out
    │
    ▼
+ sigmoid(W_output @ glu_out) ⊙ glu_out    ← NONLIN: sigmoid gate
    │                           │
    │                           └── BUT: glu_out carries GLU's
    │                                    transform nonlinearity!
    ▼
local_final

STRUCTURE: Gate → Linear Attention → Gate
NONLINEARITY TYPE: Selection (scales features, doesn't transform)
```

### Mixing Stage

```
output = α · glu_out + (1-α) · local_final
              │                    │
              │                    └─→ Contains glu_out via output gate
              │
              └──────────────────────→ Direct glu_out injection

OBSERVATION: Both paths carry GLU's transform-type nonlinearity
```

---

## Key Insight

The Local branch's nonlinearities are all **selection-type**:
- Sigmoid gates scale features between 0-1
- Softmax/sigsoftmax weights values
- Neither creates NEW feature combinations

However, the final output receives **transform-type** nonlinearity from GLU via:
1. Output gate: `sigmoid(·) ⊙ glu_out` injects transformed features
2. Alpha mixing: `α · glu_out` adds transformed features directly

**Question**: Is this "borrowed" nonlinearity sufficient, or does Local branch need its own transform capacity?

---

## Types of Nonlinearity

### Selection-Type (Current Local Branch)

```
output[i] = sigmoid(gate[i]) × input[i]

- Scales each feature independently
- Values stay in same "space"
- Good for: filtering, attention, gating
- Limited for: feature extraction, representation learning
```

### Transform-Type (Current GLU Branch, Standard FFN)

```
hidden = GELU(W₁ @ input)      # Nonlinear activation on combinations
output = W₂ @ hidden           # Project back

- Creates NEW feature combinations
- Maps to different representation space
- Good for: feature extraction, learning complex patterns
- Standard in: Transformer FFN, MLP layers
```

---

## Options Analysis

### Option A: No Change (Rely on GLU Injection)

```
Architecture: Keep current design

Data flow:
                     glu_out (has transform nonlin)
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
     input gate    output gate    α-mixing
          │             │             │
          ▼             │             │
     SWAttention        │             │
          │             │             │
          ▼             ▼             ▼
     local_out + (gate ⊙ glu_out)     │
                        │             │
                        └──────┬──────┘
                               ▼
                  α·glu + (1-α)·local_final
```

| Metric | Value |
|--------|-------|
| Added Parameters | 0 |
| Added FLOPs | 0 |
| Implementation | None needed |

**Benefits:**
- No additional complexity
- GLU's nonlinearity reaches output via gating and mixing
- Every position receives transformed features (weighted)
- Simplest to test - just train current model

**Risks:**
- Local branch may not develop position-specific representations
- "Borrowed" nonlinearity may not be enough for complex patterns
- Attention output stays in linear space

**Verdict:** BASELINE - Test empirically first

---

### Option B: SiLU Activation After Attention

```
Architecture change:

Current:                        Proposed:
────────                        ─────────

Q, K, V projections             Q, K, V projections
       │                               │
       ▼                               ▼
scores = QK^T/√d                scores = QK^T/√d
       │                               │
       ▼                               ▼
attn = sigsoftmax               attn = sigsoftmax
       │                               │
       ▼                               ▼
out = attn @ V                  out = attn @ V
       │                               │
       ▼                               ▼
out = W_o @ out                 out = SiLU(W_o @ out)    ← ADD THIS
       │                               │
       ▼                               ▼
   local_out                       local_out
```

| Metric | Value |
|--------|-------|
| Added Parameters | 0 |
| Added FLOPs | O(dim × seq) ≈ negligible |
| Implementation | One line change |

**Benefits:**
- Zero parameter cost
- Adds transform-type nonlinearity directly in attention
- Each position's features get nonlinearly transformed
- Trivial to implement and test

**Risks:**
- Non-standard (attention output usually kept linear)
- May distort value representations
- Could interfere with residual learning
- Untested in literature

**Implementation:**
```julia
# In SWAttention forward pass, after output projection:
output = W_o @ (attn @ V)
output = NNlib.silu.(output)    # ← Add this line
```

**Verdict:** CHEAP EXPERIMENT - Easy to try, might not work

---

### Option C: FFN After Mixing (Standard Approach)

```
Architecture change:

Current:                        Proposed:
────────                        ─────────

α·glu + (1-α)·local             α·glu + (1-α)·local
       │                               │
       ▼                               ▼
  + residual                    ┌──────────────────┐
       │                        │ Dense(dim→2dim)  │
       ▼                        │       ↓          │
   LayerNorm                    │  SiLU(a) ⊙ b     │   ← SwiGLU FFN
       │                        │       ↓          │
       ▼                        │ Dense(dim→dim)   │
    output                      └────────┬─────────┘
                                         │
                                         ▼
                                    + residual
                                         │
                                         ▼
                                     LayerNorm
                                         │
                                         ▼
                                      output
```

| Metric | Value |
|--------|-------|
| Added Parameters | ~2 × dim² per block |
| | For dim=384: 294,912 × 6 = ~1.77M total |
| Added FLOPs | O(2 × dim² × seq) per block |
| Implementation | Add FFN sublayer |

**Benefits:**
- Standard transformer architecture
- Well-understood, proven effective
- Full transform capacity on combined output
- Clear gradient path for both branches

**Risks:**
- Significant parameter increase (~1.8M)
- May be redundant with GLU's existing FFN-like structure
- Adds compute and memory
- Block becomes more complex

**Implementation:**
```julia
# New struct field:
FFN::SwiGLU  # or Chain of Dense layers

# In forward pass, after mixing:
mixed = α .* glu_out .+ (1-α) .* local_final
ffn_out = FFN(mixed)
output = LayerNorm(x .+ ffn_out)
```

**Verdict:** SAFE BUT EXPENSIVE - Known to work, but significant cost

---

### Option D: Expand Gate Computation (Hybrid Approach)

```
Architecture change:

Current gate:                   Proposed gate:
─────────────                   ───────────────

glu_out                         glu_out
   │                               │
   ▼                               ▼
Dense(dim→dim)                  Dense(dim→2dim)      ← EXPAND
   │                               │
   ▼                            ┌──┴──┐
sigmoid                         ▼     ▼
   │                           a     b
   ▼                           │     │
 gate                          ▼     │
                            SiLU(a)  │               ← NONLINEAR
                               │     │
                               ▼     ▼
                               a  ⊙  b               ← GATE
                                  │
                                  ▼
                            Dense(dim→dim)           ← CONTRACT (optional)
                                  │
                                  ▼
                               sigmoid
                                  │
                                  ▼
                                gate
```

| Metric | Value |
|--------|-------|
| Added Parameters | ~dim² per block |
| | For dim=384: 147,456 × 6 = ~885K total |
| Added FLOPs | O(dim² × seq) per block |
| Implementation | Modify gate computation |

**Benefits:**
- Gate becomes mini-FFN with transform capacity
- Fits the architecture's gating philosophy
- Moderate parameter increase
- "Smarter" gates that understand complex patterns

**Risks:**
- Gates traditionally should be simple signals
- Complex gate might be harder to train
- Nonlinearity in unexpected place
- Novel architecture choice

**Implementation:**
```julia
# Replace InputGate Dense with SwiGLU-style:
struct GateFFN
    expand::Dense      # dim → 2dim
    contract::Dense    # dim → dim
end

function (g::GateFFN)(x, ps, st)
    expanded, st1 = g.expand(x, ps.expand, st.expand)
    a = expanded[1:dim, ...]
    b = expanded[dim+1:end, ...]
    gated = NNlib.silu.(a) .* b
    out, st2 = g.contract(gated, ps.contract, st.contract)
    return NNlib.sigmoid.(out), (expand=st1, contract=st2)
end
```

**Verdict:** INTERESTING HYBRID - Adds capacity where architecture is unique

---

## Comparison Summary

| Option | Params | Compute | Nonlin Location | Confidence | Risk |
|--------|--------|---------|-----------------|------------|------|
| A: No change | 0 | 0 | Via GLU injection | Medium | Low |
| B: SiLU in attn | 0 | Minimal | Inside attention | Low | Medium |
| C: FFN after mix | ~1.8M | High | After mixing | High | Low |
| D: Expand gate | ~885K | Medium | In gate path | Medium | Medium |

---

## Recommended Testing Order

### Phase 1: Baseline (Option A)

Train current architecture with output gate ablation already implemented.

**Success criteria:**
- Loss converges below 1.0
- F1 score competitive with similar models
- No gradient issues

**If succeeds:** Current nonlinearity is sufficient. Ship it.

### Phase 2: Cheap Experiment (Option B)

If Phase 1 underperforms, add SiLU after attention output.

```julia
# One line change in SWAttention:
output = NNlib.silu.(W_o @ (attn @ V))
```

**Success criteria:**
- Improvement over baseline
- No training instability

**If succeeds:** Minimal cost solution found.
**If fails:** SiLU in attention doesn't help, revert.

### Phase 3: Hybrid Approach (Option D)

If Phase 2 doesn't help, expand gate computation.

```julia
# Replace simple gate with SwiGLU-style gate
gate = sigmoid(contract(silu(expand_a) ⊙ expand_b))
```

**Success criteria:**
- Clear improvement over baseline
- Gates show meaningful patterns

**If succeeds:** Novel architecture contribution.
**If fails:** Move to standard FFN.

### Phase 4: Standard FFN (Option C)

If nothing else works, add standard FFN after mixing.

```julia
# Add SwiGLU FFN sublayer
ffn_out = Dense(silu(Dense(mixed, dim→2dim)), 2dim→dim)
output = LayerNorm(x + ffn_out)
```

**Success criteria:**
- Significant improvement
- Justifies parameter cost

**If succeeds:** Standard solution works.
**If fails:** Architecture may have deeper issues.

---

## Appendix A: GLU Variants and Activation Functions

### Background: GLU Variants Paper

The paper "GLU Variants Improve Transformer" (Shazeer, 2020) systematically compared different gating mechanisms for FFN layers. Key findings:

1. Gated activations outperform standard FFN (Dense → ReLU → Dense)
2. SwiGLU and GEGLU performed best
3. With GLU variants, you can use **2/3 of the hidden dimension** and match performance

### GLU Variant Comparison

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         GLU ACTIVATION VARIANTS                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Standard FFN (no gating):                                                 │
│  ─────────────────────────                                                 │
│                                                                            │
│    x ──→ Dense(d, 4d) ──→ ReLU ──→ Dense(4d, d) ──→ output                │
│                                                                            │
│    Structure: expand → activate → contract                                 │
│    Params: 4d² + 4d² = 8d²                                                 │
│                                                                            │
│                                                                            │
│  GLU (Gated Linear Unit, original):                                        │
│  ───────────────────────────────────                                       │
│                                                                            │
│    x ──→ Dense(d, 2h) ──→ split(a, b) ──→ σ(a) ⊙ b ──→ Dense(h, d)        │
│                                                                            │
│    Gate: sigmoid(a)                                                        │
│    Params: 2dh + hd = 3dh                                                  │
│                                                                            │
│                                                                            │
│  ReGLU:                                                                    │
│  ──────                                                                    │
│                                                                            │
│    x ──→ Dense(d, 2h) ──→ split(a, b) ──→ ReLU(a) ⊙ b ──→ Dense(h, d)     │
│                                                                            │
│    Gate: ReLU(a)                                                           │
│    Simpler, but can zero out gradients                                     │
│                                                                            │
│                                                                            │
│  GEGLU:                                                                    │
│  ──────                                                                    │
│                                                                            │
│    x ──→ Dense(d, 2h) ──→ split(a, b) ──→ GELU(a) ⊙ b ──→ Dense(h, d)     │
│                                                                            │
│    Gate: GELU(a)                                                           │
│    Smooth, good gradient flow                                              │
│                                                                            │
│                                                                            │
│  SwiGLU (Swish-Gated Linear Unit):                                         │
│  ──────────────────────────────────                                        │
│                                                                            │
│    x ──→ Dense(d, 2h) ──→ split(a, b) ──→ SiLU(a) ⊙ b ──→ Dense(h, d)     │
│                                                                            │
│    Gate: SiLU(a) = a × σ(a)   (also called Swish)                          │
│    Best performance in paper                                               │
│                                                                            │
│                                                                            │
│  Bilinear (no activation):                                                 │
│  ─────────────────────────                                                 │
│                                                                            │
│    x ──→ Dense(d, 2h) ──→ split(a, b) ──→ a ⊙ b ──→ Dense(h, d)           │
│                                                                            │
│    Gate: identity (just multiply)                                          │
│    Still nonlinear due to multiplication!                                  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Activation Function Details

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ACTIVATION FUNCTION FORMULAS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Sigmoid:   σ(x) = 1 / (1 + e^(-x))                                         │
│                                                                             │
│             Range: (0, 1)                                                   │
│             Saturates at extremes                                           │
│             Used in: GLU (original)                                         │
│                                                                             │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│  ReLU:      ReLU(x) = max(0, x)                                             │
│                                                                             │
│             Range: [0, ∞)                                                   │
│             Zero gradient for x < 0 (dying ReLU problem)                    │
│             Used in: ReGLU                                                  │
│                                                                             │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│  GELU:      GELU(x) = x × Φ(x)  where Φ is standard normal CDF              │
│                     ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))                │
│                                                                             │
│             Range: (-0.17, ∞)                                               │
│             Smooth, probabilistic interpretation                            │
│             Used in: GEGLU, BERT, GPT-2                                     │
│                                                                             │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│  SiLU/Swish: SiLU(x) = x × σ(x) = x / (1 + e^(-x))                          │
│                                                                             │
│             Range: (-0.28, ∞)                                               │
│             Self-gated: input modulates itself                              │
│             Smooth, non-monotonic                                           │
│             Used in: SwiGLU, LLaMA, modern models                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Performance Ranking (from paper)

```
Quality ranking (same parameter budget):

  SwiGLU ≈ GEGLU > ReGLU > GLU > Standard FFN > Bilinear

Parameter efficiency (to match standard FFN quality):

  Standard FFN:  d → 4d → d     (8d² params)
  SwiGLU:        d → 8d/3 → d   (5.33d² params)  ← 33% fewer params!
```

### Implications for OssammaNERBlock

```
Current GLU Branch:
───────────────────

  LinearAttn(a) ⊙ sigmoid(DLinOSS(b))
                    ↑
                 Uses sigmoid gate

This is similar to original GLU, but with complex paths for a and b.


Options for Gate Computation (Option D):
────────────────────────────────────────

D-GLU:    sigmoid(expand_a) ⊙ expand_b     ← Original style
D-ReGLU:  ReLU(expand_a) ⊙ expand_b        ← Simple
D-GEGLU:  GELU(expand_a) ⊙ expand_b        ← Smooth
D-SwiGLU: SiLU(expand_a) ⊙ expand_b        ← Best (recommended)
D-Bilin:  expand_a ⊙ expand_b              ← No activation


Recommendation: Use SwiGLU for the expanded gate:

  glu_out
     │
     ▼
  Dense(d → 2h)      where h = 2d/3 (reduced expansion)
     │
  ┌──┴──┐
  ▼     ▼
  a     b
  │     │
  ▼     │
SiLU(a) │            ← SwiGLU activation
  │     │
  ▼     ▼
  a  ⊙  b
     │
     ▼
  Dense(h → d)
     │
     ▼
  sigmoid            ← Final gate signal in [0,1]
     │
     ▼
   gate
```

### Code for Each GLU Variant

```julia
# GLU (original)
gated = NNlib.sigmoid.(a) .* b

# ReGLU
gated = NNlib.relu.(a) .* b

# GEGLU
gated = NNlib.gelu.(a) .* b

# SwiGLU (recommended)
gated = NNlib.silu.(a) .* b

# Bilinear (no activation)
gated = a .* b
```

---

## Appendix B: Expansion Factor Optimization

### The Idea

Standard SwiGLU uses 2x expansion, but we can use smaller factors to save parameters while keeping nonlinearity:

```
SwiGLU structure:

input (d) → Expand (d → E) → Split (E/2, E/2) → SiLU(a) ⊙ b → Contract (E/2 → d) → output (d)

where E = expansion dimension
```

### Expansion Factor Comparison

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     EXPANSION FACTOR OPTIONS                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  2x Expansion (Standard):                                                  │
│  ─────────────────────────                                                 │
│                                                                            │
│    d ──→ Dense(d, 2d) ──→ split ──→ SiLU(a) ⊙ b ──→ Dense(d, d) ──→ d     │
│           │                  │              │              │               │
│           │             (d) (d)            (d)            (d)              │
│           │                                                                │
│           └── 2d² params                   └── d² params                   │
│                                                                            │
│    Total: 3d² params                                                       │
│    For d=384: 442,368 params                                               │
│                                                                            │
│                                                                            │
│  1.5x Expansion (Reduced):                                                 │
│  ─────────────────────────                                                 │
│                                                                            │
│    d ──→ Dense(d, 1.5d) ──→ split ──→ SiLU(a) ⊙ b ──→ Dense(0.75d, d) ──→ d│
│           │                   │              │              │              │
│           │            (0.75d)(0.75d)     (0.75d)          (d)             │
│           │                                                                │
│           └── 1.5d² params                 └── 0.75d² params               │
│                                                                            │
│    Total: 2.25d² params                                                    │
│    For d=384: 331,776 params                                               │
│    Savings: 25%                                                            │
│                                                                            │
│                                                                            │
│  1.25x Expansion (Minimal):                                                │
│  ──────────────────────────                                                │
│                                                                            │
│    d ──→ Dense(d, 1.25d) ──→ split ──→ SiLU(a) ⊙ b ──→ Dense(0.625d, d) ──→│
│           │                    │              │              │             │
│           │            (0.625d)(0.625d)   (0.625d)          (d)            │
│           │                                                                │
│           └── 1.25d² params                └── 0.625d² params              │
│                                                                            │
│    Total: 1.875d² params                                                   │
│    For d=384: 276,480 params                                               │
│    Savings: 37.5%                                                          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Parameter Comparison Table

| Expansion | Expand Params | Contract Params | Total | vs 2x |
|-----------|---------------|-----------------|-------|-------|
| 2x | 2d² | d² | 3d² | baseline |
| 1.5x | 1.5d² | 0.75d² | 2.25d² | -25% |
| 1.25x | 1.25d² | 0.625d² | 1.875d² | -37.5% |
| 1x (linear) | d² | 0 | d² | -67% |

### For Full Model (d=384, 6 blocks)

| Expansion | Params/Block | Total Added | Savings vs 2x |
|-----------|--------------|-------------|---------------|
| 2x | 442K | 2.65M | - |
| 1.5x | 332K | 1.99M | 663K |
| 1.25x | 276K | 1.66M | 995K |

### Practical Considerations

**Alignment**: For GPU efficiency, dimensions should be multiples of 8 or 16:

```
For d=384:
  - 2x:    expansion = 768 → split = 384, 384     ✓ aligned
  - 1.5x:  expansion = 576 → split = 288, 288     ✓ aligned
  - 1.25x: expansion = 480 → split = 240, 240     ✓ aligned
  - 4/3x:  expansion = 512 → split = 256, 256     ✓ aligned (nice!)
```

**Recommended**: Use 4/3x expansion (512 for d=384)
- Clean power-of-2 split dimension (256)
- Good balance of capacity vs params
- GPU-friendly alignment

```
4/3x Expansion (Recommended):
─────────────────────────────

  d ──→ Dense(d, 4d/3) ──→ split ──→ SiLU(a) ⊙ b ──→ Dense(2d/3, d) ──→ d
         │                   │              │              │
         │            (2d/3)(2d/3)       (2d/3)           (d)
         │
         └── 4d²/3 params               └── 2d²/3 params

  Total: 2d² params
  For d=384: 294,912 params
  Savings: 33% vs 2x
  Split dim: 256 (power of 2!)
```

### Updated Option D with Expansion Factor

```
Option D Variants:
──────────────────

D1: Full expansion (2x)      → +442K params/block  → +2.65M total
D2: 4/3 expansion            → +295K params/block  → +1.77M total  ← RECOMMENDED
D3: 1.25x expansion          → +276K params/block  → +1.66M total
D4: 1.5x expansion           → +332K params/block  → +1.99M total
```

---

## Appendix B: Code Snippets for Each Option

### Option B: SiLU in Attention

```julia
# In src/Attention.jl, SWAttention forward pass

function (block::SWAttention)(x, params, state)
    # ... existing Q, K, V computation ...

    # Attention computation
    attn_weights = sigsoftmax(scores .* mask)
    attn_out = batched_mul(attn_weights, V)

    # Output projection WITH activation
    output, out_state = block.OutputProjection(attn_out, params.OutputProjection, state.OutputProjection)
    output = NNlib.silu.(output)    # ← ADD THIS

    return output, new_state
end
```

### Option C: FFN After Mixing

```julia
# New SwiGLU FFN layer

struct SwiGLU <: LuxLayer
    dim::Int
    expand::Dense
    contract::Dense
end

function SwiGLU(dim::Int; expansion_factor=2)
    SwiGLU(
        dim,
        Dense(dim => expansion_factor * dim),
        Dense(expansion_factor * dim ÷ 2 => dim)
    )
end

function (ffn::SwiGLU)(x, ps, st)
    expanded, st1 = ffn.expand(x, ps.expand, st.expand)

    half = size(expanded, 1) ÷ 2
    a = expanded[1:half, ...]
    b = expanded[half+1:end, ...]

    gated = NNlib.silu.(a) .* b

    output, st2 = ffn.contract(gated, ps.contract, st.contract)
    return output, (expand=st1, contract=st2)
end

# Add to OssammaNERBlock struct and forward pass
```

### Option D: Expanded Gate

```julia
# Modified gate computation in OssammaNERBlock

struct GatedFFN <: LuxLayer
    dim::Int
    expand::Dense
    contract::Dense
end

function GatedFFN(dim::Int)
    GatedFFN(
        dim,
        Dense(dim => 2 * dim),
        Dense(dim => dim; use_bias=false)
    )
end

function (g::GatedFFN)(x, ps, st)
    expanded, st1 = g.expand(x, ps.expand, st.expand)

    a = expanded[1:g.dim, ...]
    b = expanded[g.dim+1:end, ...]

    gated = NNlib.silu.(a) .* b

    logits, st2 = g.contract(gated, ps.contract, st.contract)
    gate = NNlib.sigmoid.(logits)

    return gate, (expand=st1, contract=st2)
end

# Replace InputGate with GatedFFN in OssammaNERBlock
```
