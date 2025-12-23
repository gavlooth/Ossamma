# Option D Architecture Critique

## The Proposed Architecture (Option D)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                              OssammaNERBlock                                 │
│                                                                              │
│  INPUT                                                                       │
│    │                                                                         │
│    ▼                                                                         │
│  TimeCondLayerNorm                                                           │
│    │                                                                         │
│    ├─────────────────────────────────┐                                       │
│    │                                 │                                       │
│    ▼                                 ▼                                       │
│                                                                              │
│  ┌─────────────┐               ┌──────────────────────────────────────────┐  │
│  │ GLU BRANCH  │               │            LOCAL BRANCH                  │  │
│  │             │               │                                          │  │
│  │ LinearAttn  │               │  ┌────────────────────────────────────┐  │  │
│  │     ⊙       │               │  │  OPTION D: SwiGLU Gate             │  │  │
│  │ sigmoid(    │               │  │                                    │  │  │
│  │   DLinOSS)  │               │  │  glu_out → Dense(d→4d/3) → split   │  │  │
│  │     │       │               │  │              → SiLU(a) ⊙ b         │  │  │
│  │     ▼       │               │  │              → Dense(2d/3→d)       │  │  │
│  │  glu_out ───┼───────────────┼──│              → sigmoid             │  │  │
│  │             │               │  │              → gate                │  │  │
│  │             │               │  │                                    │  │  │
│  └─────────────┘               │  └──────────────────┬─────────────────┘  │  │
│                                │                     │                    │  │
│                                │                     ▼                    │  │
│                                │         ┌───────────────────────┐        │  │
│                                │         │                       │        │  │
│                                │         │  gate ⊙ normalized    │        │  │
│                                │         │         │             │        │  │
│                                │         │         ▼             │        │  │
│                                │         │   ┌───────────┐       │        │  │
│                                │         │   │ SWAttention│       │        │  │
│                                │         │   └─────┬─────┘       │        │  │
│                                │         │         │             │        │  │
│                                │         │         ▼             │        │  │
│                                │         │   local_out           │        │  │
│                                │         │         │             │        │  │
│                                │         │         ▼             │        │  │
│                                │         │   + gate ⊙ glu_out    │        │  │
│                                │         │         │             │        │  │
│                                │         │         ▼             │        │  │
│                                │         │   local_final         │        │  │
│                                │         │                       │        │  │
│                                │         └───────────────────────┘        │  │
│                                │                     │                    │  │
│                                └─────────────────────┼────────────────────┘  │
│                                                      │                       │
│    glu_out ──────────────────────────────────────────┤                       │
│                                                      │                       │
│                                                      ▼                       │
│                                  ┌───────────────────────────────────┐       │
│                                  │         ALPHA MIXING              │       │
│                                  │                                   │       │
│                                  │  α · glu_out + (1-α) · local_final│       │
│                                  │                                   │       │
│                                  └─────────────────┬─────────────────┘       │
│                                                    │                         │
│                                                    ▼                         │
│                                              + residual                      │
│                                                    │                         │
│                                                    ▼                         │
│                                               LayerNorm                      │
│                                                    │                         │
│                                                    ▼                         │
│                                                 OUTPUT                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## The Critique: Nonlinearity Placement

### Standard Transformer Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  INPUT                                                         │
│    │                                                           │
│    ▼                                                           │
│  LayerNorm                                                     │
│    │                                                           │
│    ▼                                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  ATTENTION (operates on FULL features)                   │  │
│  │                                                          │  │
│  │  Q = W_q @ x       ← Linear, sees all features           │  │
│  │  K = W_k @ x       ← Linear, sees all features           │  │
│  │  V = W_v @ x       ← Linear, sees all features           │  │
│  │                                                          │  │
│  │  attn = softmax(QK^T)                                    │  │
│  │  out = attn @ V                                          │  │
│  │                                                          │  │
│  │  NO NONLINEARITY BEFORE ATTENTION                        │  │
│  │                                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│    │                                                           │
│    ▼                                                           │
│  + residual                                                    │
│    │                                                           │
│    ▼                                                           │
│  LayerNorm                                                     │
│    │                                                           │
│    ▼                                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  FFN (nonlinearity AFTER attention)                      │  │
│  │                                                          │  │
│  │  hidden = GELU(W1 @ x)    ← Nonlinear transform          │  │
│  │  out = W2 @ hidden                                       │  │
│  │                                                          │  │
│  │  NONLINEARITY HERE - on attended features                │  │
│  │                                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│    │                                                           │
│    ▼                                                           │
│  + residual                                                    │
│    │                                                           │
│    ▼                                                           │
│  OUTPUT                                                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘


Order: Input → Attention (linear access) → FFN (nonlinear transform) → Output
```

### Why This Order Matters

```
ATTENTION NEEDS FULL FEATURE ACCESS
═══════════════════════════════════

The attention mechanism computes:

  Q = W_q @ x      "What am I looking for?"
  K = W_k @ x      "What do I contain?"
  V = W_v @ x      "What do I offer?"

For each query position to find relevant keys, it needs to see ALL features.

If we gate/filter features BEFORE attention:

  x_gated = gate ⊙ x           ← Some features suppressed
  Q = W_q @ x_gated            ← Query can't see suppressed features
  K = W_k @ x_gated            ← Keys can't represent suppressed features

The attention might miss important patterns because the gate
decided (before seeing the attention result) what was "important".


FFN AFTER ATTENTION MAKES SENSE
═══════════════════════════════

After attention, each position has gathered information from other positions.
NOW it makes sense to nonlinearly transform this aggregated information:

  attended = Attention(x)       ← Gathered context
  transformed = FFN(attended)   ← Transform the gathered context

The FFN operates on "complete" information (post-attention).
```

---

## Option D's Problem

```
OPTION D PUTS NONLINEARITY BEFORE ATTENTION
═══════════════════════════════════════════

  glu_out
     │
     ▼
  SwiGLU gate computation        ← Nonlinear transform HERE
     │
     ▼
  gate ⊙ normalized              ← Features FILTERED before attention
     │
     ▼
  SWAttention                    ← Attention sees filtered features
     │
     ▼
  local_out


Problem:
  - The gate decides which features matter BEFORE attention runs
  - Attention can't discover patterns in gated-out features
  - The gate is learned from glu_out, not from attention's needs


Example:
────────

Suppose glu_out (global branch) thinks "this is a common word, gate=0.2"
The gate suppresses 80% of local features.
SWAttention now only sees 20% of the information.

But what if SWAttention could have found a local pattern that
indicates this IS an entity? It never gets the chance because
the gate already decided.
```

---

## Counter-Arguments (Why Option D Might Still Work)

### 1. The Gate Comes From Global Context

```
The gate is computed from glu_out, which contains:
  - LinearAttention (global view)
  - DLinOSS (temporal dynamics)

So the gate is "informed" by global context before filtering local input.
It's not a blind filter - it's a context-aware filter.
```

### 2. There's Still Residual Injection

```
local_final = local_out + gate ⊙ glu_out
                              ↑
              Global features still get injected

Even if SWAttention misses something, glu_out can compensate.
```

### 3. Alpha Mixing Provides Escape Hatch

```
output = α · glu_out + (1-α) · local_final

If α is high, the model bypasses the problematic gating entirely.
```

### 4. This Is Not Standard Attention

```
SWAttention is LOCAL (windowed) attention.
It's not meant to see everything - only a local window.
Maybe pre-filtering for local attention is OK?
```

---

## Alternative: Option E - FFN After Mixing

If Option D's placement is problematic, consider putting the FFN AFTER mixing:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  INPUT                                                                       │
│    │                                                                         │
│    ▼                                                                         │
│  TimeCondLayerNorm                                                           │
│    │                                                                         │
│    ├─────────────────────────┐                                               │
│    │                         │                                               │
│    ▼                         ▼                                               │
│  GLU Branch              Local Branch                                        │
│  (unchanged)             (simple gate, no SwiGLU)                            │
│    │                         │                                               │
│    ▼                         ▼                                               │
│  glu_out                 local_final                                         │
│    │                         │                                               │
│    └───────────┬─────────────┘                                               │
│                │                                                             │
│                ▼                                                             │
│  ┌──────────────────────────────────────┐                                    │
│  │         ALPHA MIXING                 │                                    │
│  │                                      │                                    │
│  │  mixed = α·glu + (1-α)·local         │                                    │
│  │                                      │                                    │
│  └──────────────────┬───────────────────┘                                    │
│                     │                                                        │
│                     ▼                                                        │
│  ┌──────────────────────────────────────┐                                    │
│  │                                      │                                    │
│  │  OPTION E: FFN AFTER MIXING          │   ← Nonlinearity AFTER attention   │
│  │                                      │                                    │
│  │  Dense(d → 4d/3) → split             │                                    │
│  │  → SiLU(a) ⊙ b                       │                                    │
│  │  → Dense(2d/3 → d)                   │                                    │
│  │                                      │                                    │
│  └──────────────────┬───────────────────┘                                    │
│                     │                                                        │
│                     ▼                                                        │
│               + residual                                                     │
│                     │                                                        │
│                     ▼                                                        │
│                LayerNorm                                                     │
│                     │                                                        │
│                     ▼                                                        │
│                  OUTPUT                                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘


This follows standard transformer pattern:
  1. Attention (both GLU+LinearAttn and Local+SWAttn)
  2. Mix results
  3. FFN on mixed result
  4. Residual + Norm
```

---

## Comparison: Option D vs Option E

```
┌─────────────────────┬─────────────────────────────┬─────────────────────────────┐
│                     │         OPTION D            │         OPTION E            │
│                     │   (Gate before SWAttn)      │   (FFN after mixing)        │
├─────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ Nonlinearity        │ Before local attention      │ After all attention         │
│ placement           │                             │                             │
├─────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ What gets           │ Gate computation            │ Mixed output                │
│ transformed         │ (glu_out → gate)            │ (attended features)         │
├─────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ SWAttention         │ Sees filtered features      │ Sees all features           │
│ input               │                             │                             │
├─────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ Standard            │ No                          │ Yes                         │
│ transformer-like    │                             │                             │
├─────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ Gate still          │ Yes (but with SwiGLU)       │ Yes (simple sigmoid)        │
│ present             │                             │                             │
├─────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ Params added        │ ~295K/block in gate         │ ~295K/block after mix       │
├─────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ Risk                │ Attention misses patterns   │ More standard, less risk    │
└─────────────────────┴─────────────────────────────┴─────────────────────────────┘
```

---

## Recommendation

**Option E (FFN after mixing) is more architecturally sound** because:

1. Follows standard transformer pattern (attention → FFN)
2. All attention mechanisms see full features
3. Nonlinear transform operates on complete (post-attention) information
4. Lower risk of limiting attention's pattern discovery

**Option D might still work** because:

1. Gate is informed by global context (not blind)
2. Residual paths provide escape hatches
3. SWAttention is local anyway, maybe filtering is OK

**Suggested approach:**

1. Implement Option E first (safer)
2. Compare with Option D empirically
3. If Option D performs better despite architectural concerns, that's interesting data

---

## Summary

```
Standard wisdom:  Attention (linear) → FFN (nonlinear)
Option D:         Nonlinear gate → Attention → mix
Option E:         Attention → mix → FFN (nonlinear)

Option E aligns better with established transformer design principles.
```

---

## Implementation Plan: Option E

### Overview

Add SwiGLU FFN after α-mixing, following standard transformer pattern.
Also make output gate ablation the default (remove redundant gate).

### Changes Required

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           IMPLEMENTATION TASKS                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  1. Create SwiGLU layer                                                    │
│     ─────────────────────                                                  │
│     - New struct: SwiGLU                                                   │
│     - Parameters: expansion_factor (default 4/3)                           │
│     - Structure: Dense(d→2h) → split → SiLU(a)⊙b → Dense(h→d)             │
│     - Location: src/Ossamma.jl or new src/SwiGLU.jl                        │
│                                                                            │
│  2. Add SwiGLU to OssammaNERBlock                                          │
│     ─────────────────────────────                                          │
│     - New field: FFN::SwiGLU                                               │
│     - New config: use_ffn::Bool = true                                     │
│     - New config: ffn_expansion::Float32 = 4f0/3f0                         │
│                                                                            │
│  3. Update forward pass                                                    │
│     ────────────────────                                                   │
│     - Apply FFN after α-mixing, before residual                            │
│     - Conditional on use_ffn flag                                          │
│                                                                            │
│  4. Update defaults                                                        │
│     ───────────────────                                                    │
│     - use_output_gate = false (ablation is now default)                    │
│     - use_ffn = true (new FFN is default)                                  │
│                                                                            │
│  5. Update configs                                                         │
│     ──────────────────                                                     │
│     - Add [model.ffn] section to TOML                                      │
│     - Add ffn_expansion parameter                                          │
│                                                                            │
│  6. Update NERConfig                                                       │
│     ────────────────────                                                   │
│     - Add use_ffn::Bool = true                                             │
│     - Add ffn_expansion::Float32 = 4f0/3f0                                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### New Architecture (Final)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                    OssammaNERBlock (Option E, Final)                         │
│                                                                              │
│  INPUT: x                                                                    │
│    │                                                                         │
│    ▼                                                                         │
│  TimeCondLayerNorm(x, t) → normalized                                        │
│    │                                                                         │
│    ├──────────────────────────────────────┐                                  │
│    │                                      │                                  │
│    ▼                                      ▼                                  │
│  ┌────────────────────────┐    ┌────────────────────────────────────────┐   │
│  │      GLU BRANCH        │    │           LOCAL BRANCH                 │   │
│  │                        │    │                                        │   │
│  │  Dense(d → 2d)         │    │  gate = sigmoid(Dense(glu_out))        │   │
│  │       │                │    │           │                            │   │
│  │    ┌──┴──┐             │    │           ▼                            │   │
│  │    ▼     ▼             │    │  gated_x = gate ⊙ normalized           │   │
│  │  path_a  path_b        │    │           │                            │   │
│  │    │      │            │    │           ▼                            │   │
│  │    ▼      ▼            │    │     SWAttention(gated_x)               │   │
│  │ LinAttn  DLinOSS       │    │           │                            │   │
│  │    │      │            │    │           ▼                            │   │
│  │    │      ▼            │    │      local_out                         │   │
│  │    │   sigmoid         │    │           │                            │   │
│  │    │      │            │    │           │  (NO output gate anymore)  │   │
│  │    ▼      ▼            │    │           │                            │   │
│  │    a   ⊙  b            │    │           ▼                            │   │
│  │       │                │    │      local_final                       │   │
│  │       ▼                │    │                                        │   │
│  │  Dense(d → d)          │    │                                        │   │
│  │       │                │    │                                        │   │
│  │       ▼                │    │                                        │   │
│  │   glu_out ─────────────┼────┼───→ (to gate computation above)        │   │
│  │                        │    │                                        │   │
│  └────────────────────────┘    └────────────────────────────────────────┘   │
│           │                                      │                           │
│           │                                      │                           │
│           ▼                                      ▼                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │                         ALPHA MIXING                                  │  │
│  │                                                                       │  │
│  │   α = sigmoid(Dense(mean(normalized)) + time_bias)                    │  │
│  │                                                                       │  │
│  │   mixed = α · glu_out + (1 - α) · local_final                         │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │                    SwiGLU FFN (NEW - Option E)                        │  │
│  │                                                                       │  │
│  │   ┌─────────────────────────────────────────────────────────────┐     │  │
│  │   │                                                             │     │  │
│  │   │  Dense(d → 4d/3)     expand (e.g., 384 → 512)               │     │  │
│  │   │       │                                                     │     │  │
│  │   │    ┌──┴──┐                                                  │     │  │
│  │   │    ▼     ▼                                                  │     │  │
│  │   │    a     b           split (e.g., 256, 256)                 │     │  │
│  │   │    │     │                                                  │     │  │
│  │   │    ▼     │                                                  │     │  │
│  │   │  SiLU(a) │           nonlinear activation                   │     │  │
│  │   │    │     │                                                  │     │  │
│  │   │    ▼     ▼                                                  │     │  │
│  │   │    a  ⊙  b           gated combination                      │     │  │
│  │   │       │                                                     │     │  │
│  │   │       ▼                                                     │     │  │
│  │   │  Dense(2d/3 → d)     contract (e.g., 256 → 384)             │     │  │
│  │   │       │                                                     │     │  │
│  │   │       ▼                                                     │     │  │
│  │   │   ffn_out                                                   │     │  │
│  │   │                                                             │     │  │
│  │   └─────────────────────────────────────────────────────────────┘     │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │                      RESIDUAL + LAYERNORM                             │  │
│  │                                                                       │  │
│  │   output = LayerNorm(x + ffn_out)                                     │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│                                 OUTPUT                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Code Structure

```julia
# =============================================================================
# New SwiGLU Layer
# =============================================================================

struct SwiGLU <: LuxLayer
    in_dim::Int
    hidden_dim::Int
    expand::Dense
    contract::Dense
end

function SwiGLU(dim::Int; expansion_factor::Float32 = 4f0/3f0)
    hidden = round(Int, dim * expansion_factor)
    # Ensure hidden is even for split
    hidden = hidden + (hidden % 2)

    SwiGLU(
        dim,
        hidden,
        Dense(dim => hidden),           # d → 4d/3 (e.g., 384 → 512)
        Dense(hidden ÷ 2 => dim)        # 2d/3 → d (e.g., 256 → 384)
    )
end

function (ffn::SwiGLU)(x, ps, st)
    # Expand
    expanded, st_expand = ffn.expand(x, ps.expand, st.expand)

    # Split
    half = ffn.hidden_dim ÷ 2
    a = expanded[1:half, ...]
    b = expanded[half+1:end, ...]

    # SwiGLU: SiLU(a) ⊙ b
    gated = NNlib.silu.(a) .* b

    # Contract
    output, st_contract = ffn.contract(gated, ps.contract, st.contract)

    return output, (expand = st_expand, contract = st_contract)
end


# =============================================================================
# Updated OssammaNERBlock
# =============================================================================

struct OssammaNERBlock <: LuxLayer
    # ... existing fields ...

    use_output_gate::Bool           # Default: false (ablated)
    use_ffn::Bool                   # Default: true (new)

    # ... existing layers ...

    FFN::Union{SwiGLU, Nothing}     # New: SwiGLU after mixing
end


# =============================================================================
# Updated Forward Pass
# =============================================================================

function (block::OssammaNERBlock)(inputs::Tuple, params, state)
    # ... existing code through α-mixing ...

    # Alpha mixing
    mixed_output = alpha .* glu_out .+ (1 .- alpha) .* local_final

    # NEW: Apply SwiGLU FFN after mixing
    if block.use_ffn && block.FFN !== nothing
        ffn_out, ffn_state = block.FFN(mixed_output, params.FFN, state.FFN)
    else
        ffn_out = mixed_output
        ffn_state = NamedTuple()
    end

    # Residual + LayerNorm (on FFN output, not mixed_output)
    output = LayerNorm(x + ffn_out)

    # ... rest of code ...
end
```

### Config Changes

```toml
# configs/ner_production_110m.toml

[model]
vocab_size = 32000
max_sequence_length = 256
embedding_dimension = 384
number_of_heads = 6
number_of_layers = 6
num_labels = 19

[model.ffn]
use_ffn = true                    # NEW: Enable SwiGLU FFN
expansion_factor = 1.333333       # 4/3 expansion

[model.ablation]
use_output_gate = false           # CHANGED: Now false by default
```

### Parameter Impact

```
PARAMETER CHANGES
═════════════════

Removed (output gate ablation):
  - W_output: dim × dim = 147,456 × 6 = 884,736 params removed

Added (SwiGLU FFN with 4/3 expansion):
  - W_expand:   dim × (4dim/3) = 384 × 512 = 196,608 per block
  - W_contract: (2dim/3) × dim = 256 × 384 = 98,304 per block
  - Total: 294,912 × 6 = 1,769,472 params added

Net change: +884,736 params (about +6% for 15M model)


For dim=384, 6 blocks:
┌───────────────────┬───────────────┐
│ Component         │ Params        │
├───────────────────┼───────────────┤
│ Output gate (old) │ -884,736      │
│ SwiGLU FFN (new)  │ +1,769,472    │
├───────────────────┼───────────────┤
│ Net change        │ +884,736      │
└───────────────────┴───────────────┘
```

### Implementation Order

```
┌─────┬────────────────────────────────────────────────────────────────────────┐
│ Step│ Task                                                                   │
├─────┼────────────────────────────────────────────────────────────────────────┤
│  1  │ Change use_output_gate default to false in:                            │
│     │   - OssammaNERBlock constructor                                        │
│     │   - NERConfig struct                                                   │
│     │   - Production config files                                            │
├─────┼────────────────────────────────────────────────────────────────────────┤
│  2  │ Create SwiGLU struct and forward pass                                  │
│     │   - In src/Ossamma.jl or new file                                      │
│     │   - Include initialparameters and initialstates                        │
├─────┼────────────────────────────────────────────────────────────────────────┤
│  3  │ Add SwiGLU to OssammaNERBlock                                          │
│     │   - Add FFN field                                                      │
│     │   - Add use_ffn and ffn_expansion config                               │
│     │   - Update constructor                                                 │
├─────┼────────────────────────────────────────────────────────────────────────┤
│  4  │ Update forward pass                                                    │
│     │   - Apply FFN after α-mixing                                           │
│     │   - Before residual connection                                         │
├─────┼────────────────────────────────────────────────────────────────────────┤
│  5  │ Update NERConfig and TOML loading                                      │
│     │   - Add use_ffn and ffn_expansion fields                               │
│     │   - Update load_ner_config function                                    │
├─────┼────────────────────────────────────────────────────────────────────────┤
│  6  │ Update production config files                                         │
│     │   - Add [model.ffn] section                                            │
│     │   - Set use_output_gate = false                                        │
├─────┼────────────────────────────────────────────────────────────────────────┤
│  7  │ Test compilation and forward pass                                      │
│     │   - Verify shapes                                                      │
│     │   - Compare parameter counts                                           │
├─────┼────────────────────────────────────────────────────────────────────────┤
│  8  │ Update README diagram                                                  │
│     │   - Reflect new architecture                                           │
└─────┴────────────────────────────────────────────────────────────────────────┘
```
