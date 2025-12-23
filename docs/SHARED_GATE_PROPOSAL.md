# Shared Gate Proposal for OssammaNERBlock

## Current Architecture (5 Gates, 2 Gate Projections)

```
INPUT: x (dim, seq, batch)
       t (time_dim, batch)
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         TimeCondLayerNorm                                 │
│                                                                           │
│   normalized = LayerNorm(x) * (1 + scale(t)) + shift(t)                  │
│                                                                           │
│   normalized: (dim, seq, batch)                                          │
└───────────────────────────────────────────────────────────────────────────┘
                │
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
┌──────────────┐  ┌─────────────────────────────────────────────────────────┐
│              │  │                                                         │
│  GLU BRANCH  │  │                    LOCAL BRANCH                         │
│              │  │                                                         │
│              │  │                                                         │
└──────────────┘  └─────────────────────────────────────────────────────────┘
        │                              │
        ▼                              │
┌──────────────────────┐               │
│ GluProjection        │               │
│                      │               │
│ Dense(dim → 2*dim)   │               │
│                      │               │
│ proj: (2*dim, seq, batch)            │
└──────────────────────┘               │
        │                              │
        ▼                              │
┌──────────────────────┐               │
│ SPLIT                │               │
│                      │               │
│ path_a = proj[1:dim]     → (dim, seq, batch)
│ path_b = proj[dim+1:end] → (dim, seq, batch)
└──────────────────────┘               │
        │                              │
   ┌────┴────┐                         │
   │         │                         │
   ▼         ▼                         │
┌──────┐  ┌──────────┐                 │
│Linear│  │  DLinOSS │                 │
│Attn  │  │          │                 │
│      │  │oscillator│                 │
│(dim) │  │  (dim)   │                 │
└──┬───┘  └────┬─────┘                 │
   │           │                       │
   │           ▼                       │
   │    ┌────────────┐                 │
   │    │  sigmoid   │                 │
   │    └─────┬──────┘                 │
   │          │                        │
   ▼          ▼                        │
┌─────────────────────┐                │
│                     │                │
│  GATE #1: GLU Gate  │                │
│                     │                │
│  attn_out ⊙ sigmoid(osc_out)         │
│                     │                │
│  (dim) ⊙ (dim) = (dim, seq, batch)   │
│                     │                │
└──────────┬──────────┘                │
           │                           │
           ▼                           │
┌──────────────────────┐               │
│ GluOutputProjection  │               │
│                      │               │
│ Dense(dim → dim)     │               │
│                      │               │
│ glu_out: (dim, seq, batch)           │
└──────────┬───────────┘               │
           │                           │
           │    ┌──────────────────────┘
           │    │
           ▼    ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                         INPUT GATE                              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │   input_gate_logits = W_input @ glu_out                 │   │
│  │                                                         │   │
│  │   W_input: (dim, dim)     ← LEARNABLE PARAMS            │   │
│  │   glu_out: (dim, seq, batch)                            │   │
│  │   logits:  (dim, seq, batch)                            │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │   g_input = sigmoid(input_gate_logits)                  │   │
│  │                                                         │   │
│  │   g_input: (dim, seq, batch)  values in [0, 1]          │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │   GATE #2: Input Gating                                 │   │
│  │                                                         │   │
│  │   gated_x = normalized ⊙ g_input                        │   │
│  │                                                         │   │
│  │   (dim, seq, batch) ⊙ (dim, seq, batch)                 │   │
│  │            ↓                                            │   │
│  │   (dim, seq, batch)                                     │   │
│  │                                                         │   │
│  │   Effect: For each position & feature dimension,        │   │
│  │           scale the input by how much "global help"     │   │
│  │           that position needs (learned from glu_out)    │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                      SLIDING WINDOW ATTENTION                   │
│                                                                 │
│   Input: gated_x (dim, seq, batch)                              │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Q = W_q @ gated_x    (dim, seq, batch)                 │   │
│   │  K = W_k @ gated_x    (dim, seq, batch)                 │   │
│   │  V = W_v @ gated_x    (dim, seq, batch)                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                         │   │
│   │  scores = Q @ K^T / sqrt(d)                             │   │
│   │                                                         │   │
│   │  Apply window mask: scores[|i-j| > window] = -∞         │   │
│   │                                                         │   │
│   │  GATE #3: attn_weights = sigsoftmax(scores)             │   │
│   │                                                         │   │
│   │  local_out = attn_weights @ V                           │   │
│   │                                                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Output: local_out (dim, seq, batch)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                           │
                           │         glu_out (from GLU branch)
                           │              │
                           ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                        OUTPUT GATE                              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │   output_gate_logits = W_output @ glu_out               │   │
│  │                                                         │   │
│  │   W_output: (dim, dim)    ← SEPARATE LEARNABLE PARAMS!  │   │
│  │   glu_out:  (dim, seq, batch)                           │   │
│  │   logits:   (dim, seq, batch)                           │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │   g_output = sigmoid(output_gate_logits)                │   │
│  │                                                         │   │
│  │   g_output: (dim, seq, batch)  values in [0, 1]         │   │
│  │                                                         │   │
│  │   NOTE: g_output ≠ g_input (different W matrices!)      │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │   GATE #4: Output Gating                                │   │
│  │                                                         │   │
│  │   global_injection = g_output ⊙ glu_out                 │   │
│  │                                                         │   │
│  │   local_final = local_out + global_injection            │   │
│  │                                                         │   │
│  │   (dim, seq, batch) + (dim, seq, batch)                 │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                           │
                           │         glu_out (from GLU branch)
                           │              │
                           ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                        ALPHA MIXING                             │
│                                                                 │
│   α = sigmoid(W_alpha @ mean(normalized) + time_bias)           │
│                                                                 │
│   GATE #5:                                                      │
│   output = α · glu_out + (1 - α) · local_final                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                    + residual (x)
                           │
                           ▼
                      LayerNorm
                           │
                           ▼
                    BLOCK OUTPUT
```

---

## Proposed: Shared Gate Architecture

```
INPUT: x (dim, seq, batch)
       t (time_dim, batch)
                │
                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         TimeCondLayerNorm                                 │
│                                                                           │
│   normalized = LayerNorm(x) * (1 + scale(t)) + shift(t)                  │
│                                                                           │
│   normalized: (dim, seq, batch)                                          │
└───────────────────────────────────────────────────────────────────────────┘
                │
                │
        ┌───────┴───────┐
        │               │
        ▼               │
┌──────────────────────────────────────────┐
│              GLU BRANCH                  │
│                                          │
│ Dense(dim→2dim) → split                  │
│        │                                 │
│   ┌────┴────┐                            │
│   ▼         ▼                            │
│ path_a   path_b                          │
│   │         │                            │
│   ▼         ▼                            │
│ LinearAttn  DLinOSS                      │
│   │         │                            │
│   │         ▼                            │
│   │      sigmoid                         │
│   │         │                            │
│   ▼         ▼                            │
│   └────⊙────┘   ← GATE #1: GLU gating    │
│        │                                 │
│        ▼                                 │
│  Dense(dim→dim)                          │
│        │                                 │
│        ▼                                 │
│    glu_out (dim, seq, batch)             │
└────────────────────┬─────────────────────┘
                     │
                     │◄────────────────────────────────────────┐
                     │                                         │
                     ▼                                         │
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                    SHARED GATE COMPUTATION                               │
│                                                                          │
│   ┌────────────────────────────────────────────────────────────────┐     │
│   │                                                                │     │
│   │   gate_logits = W_shared @ glu_out                             │     │
│   │                                                                │     │
│   │   W_shared: (dim, dim)  ← SINGLE WEIGHT MATRIX                 │     │
│   │                                                                │     │
│   │   This replaces both W_input AND W_output                      │     │
│   │                                                                │     │
│   └────────────────────────────────────────────────────────────────┘     │
│                              │                                           │
│                              ▼                                           │
│   ┌────────────────────────────────────────────────────────────────┐     │
│   │                                                                │     │
│   │   g = sigmoid(gate_logits)                                     │     │
│   │                                                                │     │
│   │   g: (dim, seq, batch)                                         │     │
│   │                                                                │     │
│   │   This gate value will be REUSED twice:                        │     │
│   │     1. To gate the input before SWAttention                    │     │
│   │     2. To gate the global context injection after SWAttention  │     │
│   │                                                                │     │
│   └────────────────────────────────────────────────────────────────┘     │
│                              │                                           │
│                              │                                           │
│              ┌───────────────┴───────────────┐                           │
│              │                               │                           │
│              ▼                               ▼                           │
│         [USE #1]                        [USE #2]                         │
│       (see below)                    (save for later)                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
               │                               │
               │                               │
               ▼                               │
┌──────────────────────────────────────────┐   │
│                                          │   │
│   USE #1: Gate the input to attention    │   │
│                                          │   │
│   ┌────────────────────────────────┐     │   │
│   │                                │     │   │
│   │   gated_x = g ⊙ normalized     │     │   │
│   │                                │     │   │
│   │   GATE #2 (using shared g)     │     │   │
│   │                                │     │   │
│   │   Example per position:        │     │   │
│   │                                │     │   │
│   │   Position "Barack":           │     │   │
│   │     g = 0.9 (needs global)     │     │   │
│   │     normalized features get    │     │   │
│   │     scaled to 90%              │     │   │
│   │                                │     │   │
│   │   Position "visited":          │     │   │
│   │     g = 0.2 (local is fine)    │     │   │
│   │     normalized features get    │     │   │
│   │     scaled to 20%              │     │   │
│   │                                │     │   │
│   └────────────────────────────────┘     │   │
│                                          │   │
│   gated_x: (dim, seq, batch)             │   │
│                                          │   │
└──────────────────────────────────────────┘   │
               │                               │
               ▼                               │
┌──────────────────────────────────────────┐   │
│                                          │   │
│   SLIDING WINDOW ATTENTION               │   │
│                                          │   │
│   Q, K, V projections on gated_x         │   │
│                                          │   │
│   scores = QK^T / sqrt(d)                │   │
│   mask out |i-j| > window_size           │   │
│                                          │   │
│   GATE #3: attn = sigsoftmax(scores)     │   │
│                                          │   │
│   local_out = attn @ V                   │   │
│                                          │   │
│   local_out: (dim, seq, batch)           │   │
│                                          │   │
└──────────────────────────────────────────┘   │
               │                               │
               │◄──────────────────────────────┘
               │         g (reused!)
               │              │
               ▼              ▼
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   USE #2: Gate the global context injection                      │
│                                                                  │
│   ┌────────────────────────────────────────────────────────┐     │
│   │                                                        │     │
│   │   global_injection = g ⊙ glu_out                       │     │
│   │                                                        │     │
│   │   GATE #4 (using SAME shared g)                        │     │
│   │                                                        │     │
│   │   Uses the SAME gate value computed earlier!           │     │
│   │                                                        │     │
│   │   Example per position:                                │     │
│   │                                                        │     │
│   │   Position "Barack":                                   │     │
│   │     g = 0.9 (same as before)                           │     │
│   │     Inject 90% of glu_out (global context)             │     │
│   │     GLU branch knows this is a PERSON entity           │     │
│   │                                                        │     │
│   │   Position "visited":                                  │     │
│   │     g = 0.2 (same as before)                           │     │
│   │     Only inject 20% of glu_out                         │     │
│   │     Trust local attention for common verbs             │     │
│   │                                                        │     │
│   └────────────────────────────────────────────────────────┘     │
│                              │                                   │
│                              ▼                                   │
│   ┌────────────────────────────────────────────────────────┐     │
│   │                                                        │     │
│   │   local_final = local_out + global_injection           │     │
│   │                                                        │     │
│   │             = local_out + (g ⊙ glu_out)                │     │
│   │                                                        │     │
│   │   Combines local attention with gated global context   │     │
│   │                                                        │     │
│   └────────────────────────────────────────────────────────┘     │
│                                                                  │
│   local_final: (dim, seq, batch)                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
               │
               │         glu_out (from GLU branch)
               │              │
               ▼              ▼
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   ALPHA MIXING                                                   │
│                                                                  │
│   input_pooled = mean(normalized, dim=seq)                       │
│                                                                  │
│   α = sigmoid(W_alpha @ input_pooled + time_bias)                │
│                                                                  │
│   α: (1, batch) - scalar per batch item                          │
│                                                                  │
│   GATE #5: Blend global and local branches                       │
│                                                                  │
│   output = α · glu_out + (1 - α) · local_final                   │
│                                                                  │
│   output: (dim, seq, batch)                                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   RESIDUAL + OUTPUT NORM                                         │
│                                                                  │
│   output = LayerNorm(x + output)                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
               │
               ▼
         BLOCK OUTPUT
         (dim, seq, batch)
```

---

## Comparison: Current vs Shared Gate

### Parameter Count

| Component | Current | Shared Gate | Savings |
|-----------|---------|-------------|---------|
| W_input | dim × dim | - | - |
| W_output | dim × dim | - | - |
| W_shared | - | dim × dim | - |
| **Total gate params** | **2 × dim²** | **1 × dim²** | **dim²** |

For dim=384 (production config):
- Current: 2 × 384² = 294,912 params per block
- Shared: 1 × 384² = 147,456 params per block
- Savings: 147,456 params per block × 6 blocks = **~885K params total**

### Gate Count

| Gate | Current | Shared Gate |
|------|---------|-------------|
| #1: GLU (LinearAttn ⊙ sigmoid(DLinOSS)) | ✓ | ✓ |
| #2: Input gate (g ⊙ normalized) | g_input | g_shared |
| #3: SWAttention sigsoftmax | ✓ | ✓ |
| #4: Output gate (g ⊙ glu_out) | g_output | g_shared (reused) |
| #5: Alpha mixing | ✓ | ✓ |
| **Unique gate computations** | **5** | **4** |

---

## Semantic Interpretation

The shared gate `g` answers a single question per (position, feature):

> "How much does this position need help from the global (GLU) branch?"

### When g is HIGH (e.g., 0.9):

```
USE #1 (Input):
  - Suppress local features going into SWAttention
  - "Don't rely on local patterns here"

USE #2 (Output):
  - Inject more global context from GLU
  - "Trust the global branch's understanding"

Together:
  - This position relies heavily on global context
  - Example: Entity names like "Barack Obama" need global knowledge
```

### When g is LOW (e.g., 0.2):

```
USE #1 (Input):
  - Keep most local features for SWAttention
  - "Local patterns are informative here"

USE #2 (Output):
  - Inject little global context
  - "Trust the local attention's result"

Together:
  - This position can be understood from local context
  - Example: Common words like "visited" or "the"
```

---

## Code Diff

```julia
# BEFORE (current implementation)
# ================================

# Step 3a: Input Gate
input_gate_logits, input_gate_state = block.InputGate(
    glu_out, params.InputGate, state.InputGate
)
input_gate = NNlib.sigmoid.(input_gate_logits)
gated_x = normalized .* input_gate

# Step 3b: SWAttention
local_out, sw_attn_state = block.SlidingWindowAttention(
    gated_x, params.SlidingWindowAttention, state.SlidingWindowAttention
)

# Step 3c: Output Gate (SEPARATE computation)
output_gate_logits, output_gate_state = block.OutputGate(
    glu_out, params.OutputGate, state.OutputGate
)
output_gate = NNlib.sigmoid.(output_gate_logits)
local_final = local_out .+ (output_gate .* glu_out)


# AFTER (shared gate implementation)
# ==================================

# Step 3a: Shared Gate (computed ONCE)
gate_logits, gate_state = block.SharedGate(
    glu_out, params.SharedGate, state.SharedGate
)
gate = NNlib.sigmoid.(gate_logits)

# Step 3b: Use gate for input gating
gated_x = normalized .* gate

# Step 3c: SWAttention
local_out, sw_attn_state = block.SlidingWindowAttention(
    gated_x, params.SlidingWindowAttention, state.SlidingWindowAttention
)

# Step 3d: Reuse SAME gate for output (no new computation!)
local_final = local_out .+ (gate .* glu_out)
```

---

## Open Questions for Discussion

1. **Is the shared gate too constrained?**
   - Current: Input and output gating can learn different patterns
   - Shared: Same pattern must work for both uses
   - Counter-argument: Semantically they should be aligned anyway

2. **Should we add a learned scaling factor?**
   ```julia
   # Option: Different scaling for each use
   gated_x = normalized .* (gate * scale_input)
   local_final = local_out .+ (gate * scale_output .* glu_out)
   ```
   This adds 2 scalars instead of dim² params

3. **Alternative: Share the projection but not the gate?**
   ```julia
   # Compute logits once, but apply different biases
   logits = W_shared @ glu_out
   g_input = sigmoid(logits + b_input)
   g_output = sigmoid(logits + b_output)
   ```
   This adds 2 × dim params (biases) instead of dim² params

4. **Does this interact well with the α-mixing?**
   - α is a scalar that blends entire branches
   - g is per-position that controls local vs global
   - They operate at different granularities, should be complementary
