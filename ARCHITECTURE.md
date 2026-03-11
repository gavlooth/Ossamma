# Architecture Documentation

## Overview

The active architecture in this repo is built around a composite `SwammaBlock`:
1. **SWAttention**: multi-head sliding-window attention for sharp local structure
2. **WavePDE**: projection-free spectral wave dynamics used as a gate path
3. **LinearAttention**: efficient global content path

The old `OSSM` path is no longer the active block design.

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

### OSSM (Oscillatory State Space Model)

**Core Design:**
```
Input u (dim_in, T)
    ↓
Input Gating: g_in = σ(Dense(u))
u_gated = g_in ⊙ u
    ↓
SSM Processing (for each timestep t):
    State xt: (2H, 1) - H oscillators, 2 coords each

    Selective Step Sizing:
        Δt = softplus(WΔ * ut + bΔ)  → (H,)

    Damped Rotation:
        ρ = exp(-softplus(α) * Δt)   → (H,) damping
        θ = ω * Δt                    → (H,) rotation

    State Update:
        x_{t+1} = A(ρ,θ) * xt + B * ut
        where A(ρ,θ) applies per-oscillator 2D rotation

    Output:
        yt = C * xt + D * ut
    ↓
Collect Y from all timesteps → (dim_out, T)
    ↓
Mixture Gating:
    g_mix = σ(Dense(u))
    out = g_mix ⊙ Y + (1 - g_mix) ⊙ u
    ↓
Output (dim_out, T)
```

**Key Innovations:**
1. **Learnable Oscillators**: Each of H oscillators has independent frequency ω and damping α
2. **Selective Step Sizing**: Step size Δt depends on input (like Mamba's selective SSM)
3. **Stable Dynamics**: Damping forced positive via softplus ensures ρ ∈ (0,1]
4. **Gated Skip Connections**: Both input and output are gated

**State Convention:**
- State is (2H, 1) column vector
- Reshaped to (2, H) for per-oscillator operations
- Each oscillator has 2D state (x, y) representing phase space coordinates

---

## OSSM vs. Mamba Architecture: Detailed Comparison

### Overview

Both OSSM and [Mamba](https://arxiv.org/abs/2312.00752) are selective state space models designed for efficient sequence processing with linear-time complexity. However, they differ fundamentally in their mathematical foundations, state dynamics, and implementation strategies.

### Core Similarities

| Aspect | Both Architectures |
|--------|-------------------|
| **Paradigm** | Selective State Space Models (parameters depend on input) |
| **Complexity** | Linear time O(T) vs. quadratic O(T²) in transformers |
| **Sequential Processing** | Recurrent state updates across time |
| **Input-Dependent Dynamics** | State transitions adapt based on current input |
| **Gating Mechanisms** | Control information flow |

### Fundamental Differences

#### 1. **State Dynamics & Mathematical Foundation**

**OSSM (Oscillatory):**
```
State Update: x_{t+1} = A(ρ, θ) * x_t + B * u_t
where A(ρ, θ) = block_diag(ρ_1 * R(θ_1), ..., ρ_H * R(θ_H))
R(θ) = [cos(θ) -sin(θ)]  # 2D rotation matrix
       [sin(θ)  cos(θ)]

ρ = exp(-α * Δt)  # damping factor ∈ (0, 1]
θ = ω * Δt        # rotation angle
```

- **Foundation**: Coupled damped harmonic oscillators from physics
- **A Matrix**: Block-diagonal with 2×2 rotation blocks (non-diagonal, non-separable)
- **State Space**: Explicitly 2D phase space per oscillator (position + velocity)
- **Dynamics**: Rotational + damping (spiral trajectories)
- **Interpretability**: Physical meaning (frequency ω, damping α, phase)

**Mamba-1 (Selective S6):**
```
State Update: x_{t+1} = A̅ * x_t + B̅ * u_t
where A̅ = exp(Δ * A), B̅ = (Δ * A)^{-1} * (exp(Δ * A) - I) * Δ * B
A is diagonal (or low-rank + diagonal in S4)

Discretization: Zero-Order Hold (ZOH)
```

- **Foundation**: Continuous-time linear time-invariant (LTI) systems, discretized
- **A Matrix**: Diagonal or structured (HiPPO initialization)
- **State Space**: Abstract N-dimensional latent space
- **Dynamics**: Exponential decay/growth along principal axes
- **Interpretability**: Less direct physical meaning, more learned representations

**Mamba-2 (SSD - State Space Dual):**
```
A = -α * I  # scalar times identity (even simpler!)

Structured matrix multiplication formulation
Bridges SSMs and attention via duality
```

- **Foundation**: Structured state space duality theory
- **A Matrix**: Scalar multiple of identity (maximum simplicity)
- **Efficiency**: Leverages matrix multiplication primitives (faster than Mamba-1)

#### 2. **Selective Mechanism (Input-Dependent Parameters)**

**OSSM:**
- **What's Selective**: Step size `Δt` only
- **How**: `Δt = softplus(W_Δ * u_t + b_Δ)` → (H,) per oscillator
- **Fixed**: Frequencies `ω`, damping `α`, B, C, D (after training)
- **Intuition**: Adapt temporal resolution per oscillator based on input

**Mamba-1:**
- **What's Selective**: `Δ, B, C` (all three!)
- **How**:
  - `Δ = softplus(Linear_Δ(u_t))` → (N,) or (D,)
  - `B = Linear_B(u_t)` → (N,)
  - `C = Linear_C(u_t)` → (N,)
- **Fixed**: A matrix structure (HiPPO initialization)
- **Intuition**: Fully adaptive filtering (what to remember, what to forget, what to output)

**Mamba-2:**
- Similar to Mamba-1 but with simplified A matrix
- Focus on efficient matrix multiplication formulation

#### 3. **State Dimension**

| Model | State Dim | Typical Values | Notes |
|-------|-----------|---------------|-------|
| **OSSM** | 2H | 2×4 = 8 to 2×64 = 128 | Paired (2D per oscillator), grows with H |
| **Mamba-1** | N | 16 (standard) | Fixed per layer, independent of model dim |
| **Mamba-2** | N | 64-256 | Much larger thanks to efficient SSD algorithm |

**Key Insight**: Mamba-2 can use 16× larger state dimension than Mamba-1 while being faster, thanks to the SSD formulation. OSSM's state grows with oscillator count.

#### 4. **Hardware-Aware Implementation**

**OSSM:**
- **Algorithm**: Sequential `foldl` scan over time
- **Parallelism**: None across time (inherently sequential)
- **Memory**: Stores full output buffer Y: (dim_out, T)
- **Optimization Level**: Basic Julia (not hardware-optimized)
- **Speed**: Standard, no special kernels

**Mamba-1:**
- **Algorithm**: [Parallel associative scan](https://github.com/state-spaces/mamba) with kernel fusion
- **Parallelism**: Work-efficient parallel scan O(log T) depth
- **Memory**: Recomputation strategy (don't store intermediate states)
- **Optimization**: Custom CUDA kernels, kernel fusion
- **Speed**: ~40× faster than naive implementation
- **Implementation**: ~3000 lines of optimized CUDA

**Mamba-2:**
- **Algorithm**: Structured matrix multiplication (SSD)
- **Parallelism**: Leverages optimized GEMM primitives
- **Memory**: More memory-efficient than Mamba-1
- **Optimization**: Uses existing optimized BLAS/cuBLAS
- **Speed**: 2-8× faster than Mamba-1 in training
- **Implementation**: ~25 lines of minimal code (much simpler!)

#### 5. **Gating Architecture**

**OSSM:**
```
u_gated = σ(InputGate(u)) ⊙ u          # input gating
Y = SSM(u_gated)                        # SSM processing
out = σ(MixGate(u)) ⊙ Y + (1-σ) ⊙ u    # mixture + residual
```
- **Two gates**: Input gate and mixture gate
- **Explicit skip**: Gated residual connection around SSM
- **Design**: Similar to gated RNNs (GRU-style)

**Mamba:**
```
x_proj = Linear(x)                      # project input
x, gate = split(x_proj)                 # split into data + gate
x_conv = Conv1d(x)                      # short convolution
x_ssm = SSM(x_conv)                     # selective scan
out = x_ssm ⊙ σ(gate)                   # gated output
```
- **Single gate**: Output gating only
- **Conv layer**: 1D convolution before SSM (not in OSSM)
- **Design**: Similar to Gated Linear Unit (GLU)

#### 6. **Additional Architectural Components**

| Component | OSSM | Mamba |
|-----------|------|-------|
| **Convolution** | ❌ None | ✅ 1D conv (kernel size 4) |
| **Normalization** | ❌ None (should add) | ✅ RMSNorm |
| **Skip Connections** | ⚠️ Gated (requires dim match) | ✅ Direct residual |
| **Projection Layers** | ✅ Input/output gates | ✅ Input/output projections |

#### 7. **Theoretical Properties**

**OSSM:**
- **Stability**: Guaranteed stable (ρ < 1 via softplus on α)
- **Frequency Selectivity**: Explicit via learnable ω
- **Long-term Memory**: Depends on damping α (can decay quickly)
- **Inductive Bias**: Periodic/oscillatory patterns

**Mamba:**
- **Stability**: Depends on A matrix eigenvalues (HiPPO initialization helps)
- **Frequency Selectivity**: Implicit in state space
- **Long-term Memory**: Optimized via HiPPO basis (designed for long sequences)
- **Inductive Bias**: General sequence modeling

#### 8. **Computational Complexity**

For sequence length T, state dimension N/2H, model dimension D:

| Operation | OSSM | Mamba-1 | Mamba-2 |
|-----------|------|---------|---------|
| **Forward Pass** | O(T · H) | O(T · N) | O(T · N) |
| **Scan Algorithm** | O(T) sequential | O(T) parallel | O(T) via matmul |
| **Memory (Training)** | O(T · D) | O(1) per step† | O(T · N) |
| **Memory (Inference)** | O(H) state | O(N) state | O(N) state |

† Mamba-1 uses selective recomputation to save memory

### Performance Comparison (Estimated)

**Speed (Relative to Mamba-1 = 1.0×):**
- OSSM (current): ~0.02× (40× slower - no parallelism, no kernels)
- Mamba-1 (CUDA): 1.0× (baseline with parallel scan)
- Mamba-2 (SSD): 2-8× (faster via matmul primitives)

**Memory Efficiency:**
- OSSM: Moderate (stores full output, no recomputation)
- Mamba-1: Excellent (selective recomputation)
- Mamba-2: Very good (efficient SSD formulation)

**Long Sequence Performance:**
- OSSM: Untested (likely struggles >10k due to sequential scan)
- Mamba-1: Excellent (tested up to 1M tokens)
- Mamba-2: Excellent (faster than Mamba-1 at all lengths)

### Unique Advantages

**OSSM Advantages:**
1. **Interpretable oscillators**: Clear physical meaning (frequency, damping)
2. **Explicit periodicity**: Built-in bias for periodic patterns
3. **Phase space dynamics**: Rich 2D rotational behavior per oscillator
4. **Simplicity**: Conceptually straightforward (harmonic oscillators)
5. **Multi-scale potential**: Different oscillators for different frequencies

**Mamba Advantages:**
1. **Hardware optimization**: 40-100× faster in practice
2. **Full selectivity**: Adaptive Δ, B, C (not just Δt)
3. **Proven scaling**: State-of-the-art on language modeling benchmarks
4. **Long-range memory**: HiPPO initialization optimized for recall
5. **Production-ready**: Optimized implementation, extensive testing
6. **Mamba-2 simplicity**: SSD formulation easier to implement and faster

### When to Use Which?

**Use OSSM when:**
- You have strong periodic/oscillatory patterns (audio, circadian rhythms, seasonal data)
- You want interpretable frequency components
- You need explicit multi-scale temporal dynamics
- Working with small to medium sequences (<10k)
- Prototyping research ideas in Julia

**Use Mamba when:**
- You need state-of-the-art performance on language/general sequences
- You require extreme efficiency (long sequences >100k)
- You have GPU resources and need speed
- You want production-ready implementation
- Working with information-dense sequential data

### Hybrid Possibilities

Could combine OSSM's oscillatory dynamics with Mamba's efficiency:
```
OSSM-Mamba Hybrid:
1. Use Mamba's parallel scan algorithm for OSSM's oscillator updates
2. Add OSSM's rotational dynamics to Mamba's state transitions
3. Multi-resolution: Mamba for fast dynamics, OSSM oscillators for slow periodic components
4. Frequency-selective Mamba: Use OSSM's ω to initialize Mamba's A matrix structure
```

### References

- **Mamba-1**: Gu & Dao, ["Mamba: Linear-Time Sequence Modeling with Selective State Spaces"](https://arxiv.org/abs/2312.00752) (2023)
- **Mamba-2**: Dao & Gu, ["Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"](https://arxiv.org/abs/2405.21060) (2024)
- **Visual Guide**: [A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)
- **Implementation**: [GitHub - state-spaces/mamba](https://github.com/state-spaces/mamba)
- **S4 Foundation**: Gu et al., ["Efficiently Modeling Long Sequences with Structured State Spaces"](https://arxiv.org/abs/2111.00396) (2021)

---

## Proposed Architectural Improvements

### SWAttention Improvements

#### 1. **Add Causal Masking Support**
```julia
struct SWAttention <: Lux.AbstractLuxLayer
    # ... existing fields
    causal::Bool  # NEW: enable causal (autoregressive) attention
end
```
**Why**: Essential for autoregressive tasks (language modeling, time series prediction)

#### 2. **Implement Relative Position Bias**
```julia
struct SWAttention <: Lux.AbstractLuxLayer
    # ... existing fields
    use_relative_pos::Bool
    max_distance::Int
end

# In parameters:
relative_pos_bias::Array  # (2*max_distance + 1, number_of_heads)
```
**Why**: Position information crucial for sequence tasks; relative positions generalize better than absolute

#### 3. **Add Attention Dropout**
```julia
# In forward pass after normalized_sigmoids:
if training
    attention_weights = dropout(attention_weights, p=dropout_rate)
end
```
**Why**: Regularization; prevents overfitting to specific attention patterns

#### 4. **Configurable Temperature Learning**
```julia
struct SWAttention <: Lux.AbstractLuxLayer
    # ... existing fields
    learnable_temperature::Bool
end

# In parameters:
τ::Vector{Float32}  # per-head or global temperature
```
**Why**: Fixed τ=1.0 may be suboptimal; learned temperature can adapt to data

#### 5. **Multi-Query Attention (MQA) / Grouped-Query Attention (GQA)**
```julia
struct SWAttention <: Lux.AbstractLuxLayer
    # ... existing fields
    kv_heads::Int  # number of KV heads (< number_of_heads for MQA/GQA)
end
```
**Why**: Reduces KV cache size for inference; huge memory savings with minimal quality loss

---

### OSSM Improvements

#### 0. **Mamba-Inspired Enhancements** ⭐

Based on the comparison with Mamba, these improvements would bring OSSM closer to production-ready:

**a) Make B and C Selective (like Mamba):**
```julia
# Current: only Δt is selective
# Proposed: B, C also input-dependent
function oscillator_step(block, params, xt, ut)
    # Selective parameters
    Δt = softplus.(params.WΔ * ut .+ reshape(params.bΔ, :, 1))  # (H,)
    B_t = params.WB * ut  # (2H, 1) - NEW: input-dependent input mixing
    C_t = params.WC * ut  # (dim_out,) - NEW: input-dependent output mixing

    # State update with selective B
    x_next = apply_oscillation(block, xt, ρ, θ) + B_t .* ut

    # Output with selective C
    y = (C_t' .* params.C) * xt + params.D * ut
end
```
**Impact**: Full selectivity like Mamba; more expressive filtering

**b) Add 1D Convolution Before SSM:**
```julia
struct OSSM <: Lux.AbstractLuxLayer
    # ... existing fields
    conv::Lux.Conv  # 1D conv, kernel size 4
end

function (block::OSSM)(u, params, state)
    # Convolve first (local context)
    u_conv = block.conv(u, params.conv, state.conv)
    # Then SSM (global context)
    # ... rest of processing
end
```
**Impact**: Better local context modeling; proven in Mamba

**c) Parallel Scan Implementation:**
```julia
# Replace sequential foldl with parallel associative scan
# Requires: expressing oscillator update as binary associative operator
# Benefit: O(log T) depth vs O(T), much faster on GPU
# Challenge: Non-trivial for 2D rotation matrices (not just element-wise)
```
**Impact**: 10-40× speedup for long sequences (needs GPU kernels)

**d) HiPPO-Inspired Frequency Initialization:**
```julia
function initialize_frequencies_hippo(H::Int)
    # Initialize frequencies to cover spectrum like HiPPO
    # Low frequencies for long-term memory
    ω_low = range(0.01, 0.1, length=H÷3)
    ω_mid = range(0.1, 1.0, length=H÷3)
    ω_high = range(1.0, 10.0, length=H÷3)
    return vcat(ω_low, ω_mid, ω_high)
end
```
**Impact**: Better coverage of timescales; principled initialization

#### 1. **Flexible Skip Connection**
Currently requires `dim_in == dim_out`. Add projection:
```julia
struct OSSM <: Lux.AbstractLuxLayer
    # ... existing fields
    skip_proj::Union{Nothing, Lux.Dense}
end

function OSSM(dim_in, dim_out, H; use_skip_proj=nothing)
    skip_proj = if dim_in != dim_out
        use_skip_proj === false ? nothing : Lux.Dense(dim_in => dim_out)
    else
        nothing
    end
    # ...
end
```
**Why**: Removes dimension constraint; more flexible architecture composition

#### 2. **Multi-Scale Oscillators**
```julia
struct OSSM <: Lux.AbstractLuxLayer
    # ... existing fields
    frequency_bands::Vector{Tuple{Float32, Float32}}  # (ω_min, ω_max) per band
end
```
Initialize different oscillator groups with different frequency ranges:
- Low freq: ω ∈ [0.01, 0.1] - long-term patterns
- Mid freq: ω ∈ [0.1, 1.0] - medium-term
- High freq: ω ∈ [1.0, 10.0] - short-term

**Why**: Captures patterns at multiple timescales explicitly

#### 3. **Learnable Initial State**
```julia
# In parameters:
x0::Array{Float32, 2}  # (2H, 1) learnable initial state

# In initialstates:
(; oscillation_state = copy(params.x0))
```
**Why**: Better than zero initialization; can encode prior knowledge about typical dynamics

#### 4. **Residual Oscillator Connections**
```julia
# In apply_oscillation:
x_next = ρ .* rotate(x) + (1 .- ρ) .* x_identity + B * ut
```
Add identity skip within oscillators (not just around the whole OSSM)

**Why**: Helps gradient flow; prevents oscillators from collapsing

#### 5. **Normalization Layers**
```julia
struct OSSM <: Lux.AbstractLuxLayer
    # ... existing fields
    norm_input::Union{Nothing, Lux.LayerNorm}
    norm_output::Union{Nothing, Lux.LayerNorm}
end
```
**Why**: Stabilizes training; standard practice in modern architectures

---

## Code-Level Improvements

### Performance Optimizations

#### 1. **Preallocate Rotation Matrices** (OSSM)
Current code allocates rotation matrix per oscillator in comprehension:
```julia
# Current (allocates H rotation matrices):
cols = [ρi * [cos(θi) -sin(θi); sin(θi) cos(θi)] * xi for ...]

# Improved (vectorized):
function apply_oscillation_vectorized(block, x, ρ, θ)
    x_view = reshape(x, 2, :)  # (2, H)
    cos_θ = cos.(θ)'  # (1, H)
    sin_θ = sin.(θ)'  # (1, H)

    # Vectorized rotation
    x1, x2 = x_view[1, :], x_view[2, :]
    x1_new = ρ .* (cos_θ .* x1 - sin_θ .* x2)
    x2_new = ρ .* (sin_θ .* x1 + cos_θ .* x2)

    return vcat(x1_new', x2_new')[:, 1:1]  # reshape to (2H, 1)
end
```
**Impact**: Reduces allocations; ~2-3x faster for large H

#### 2. **Fused Attention Computation** (SWAttention)
```julia
# Current: Multiple intermediate allocations
# Improved: Use BLAS operations directly
function compute_attention(q, k, v, d_k)
    # q, k, v: (d_k, T)
    scores = BLAS.gemm('T', 'N', 1.0/√d_k, q, k)  # (T, T) - fused transpose
    # ... normalize ...
    output = BLAS.gemm('N', 'N', 1.0, v, weights)  # (d_k, T)
end
```
**Impact**: Fewer allocations; better cache locality

#### 3. **In-place Operations**
```julia
# In OSSM forward pass, reuse buffer:
function (block::OSSM)(u, params, state)
    # ...
    Y = similar(u, block.dim_out, T)

    # Current allocates g_in .* u
    # Improved:
    u_gated = similar(u)
    u_gated .= g_in .* u  # in-place

    # Similarly for final output:
    out = similar(Y)
    out .= g_mix .* Y .+ (oneT .- g_mix) .* u
end
```
**Impact**: Reduces GC pressure; important for large batches

### Code Quality Improvements

#### 1. **Add Input Validation**
```julia
function (block::SWAttention)(x, params, state)
    @assert size(x, 1) == block.dimension "Input dimension mismatch"
    @assert ndims(x) == 2 "Expected 2D input (dimension, T)"
    # ...
end

function (block::OSSM)(u, params, state)
    @assert size(u, 1) == block.dim_in "Input dimension mismatch"
    @assert ndims(u) == 2 "Expected 2D input (dim_in, T)"
    # ...
end
```
**Why**: Better error messages; easier debugging

#### 2. **Separate Concerns - Extract Helper Modules**
```julia
# Create src/utils/attention_ops.jl
module AttentionOps
    export normalized_sigmoids, compute_attention_scores
    # ... attention utilities
end

# Create src/utils/oscillator_ops.jl
module OscillatorOps
    export apply_oscillation, make_rotation_matrix
    # ... oscillator utilities
end
```
**Why**: Better organization; reusable components; easier testing

#### 3. **Add Type Stability Checks**
```julia
# Use @code_warntype to check type stability
# Add explicit type annotations where needed:

function oscillator_step(block, params, xt::Matrix{T}, ut::Matrix{T}) where T
    # ...
    Δt = NNlib.softplus.(WΔ * ut .+ reshape(bΔ, :, 1))::Matrix{T}
    # ...
end
```
**Why**: Type stability crucial for Julia performance

#### 4. **Configuration Struct Pattern**
```julia
@kwdef struct SWAttentionConfig
    dimension::Int
    number_of_heads::Int
    sequence_length::Int = 1024
    dropout::Float32 = 0.0f0
    causal::Bool = false
    use_relative_pos::Bool = false
    learnable_temperature::Bool = false
end

function SWAttention(config::SWAttentionConfig)
    # construct from config
end
```
**Why**: Easier to manage many hyperparameters; better for experiments

#### 5. **Add Comprehensive Tests**
```julia
# test/test_attention.jl
@testset "SWAttention" begin
    @testset "Dimension checks" begin
        # test dimension compatibility
    end

    @testset "Gradient flow" begin
        # test backpropagation works
    end

    @testset "Causality" begin
        # test causal masking if implemented
    end
end

# test/test_ossm.jl
@testset "OSSM" begin
    @testset "State evolution" begin
        # test state updates correctly
    end

    @testset "Oscillator stability" begin
        # test ρ ∈ (0, 1], no NaN/Inf
    end

    @testset "Selective stepping" begin
        # test Δt > 0 always
    end
end
```

#### 6. **Add Docstrings**
```julia
"""
    SWAttention(sequence_length, dimension, number_of_heads)

Sliding Window Attention with normalized sigmoid activation.

Uses temperature-scaled sigmoid normalization instead of softmax for computing
attention weights. Each attention head operates on dimension/number_of_heads features.

# Arguments
- `sequence_length::Int`: Maximum sequence length (stored but not enforced)
- `dimension::Int`: Embedding dimension (must be divisible by number_of_heads)
- `number_of_heads::Int`: Number of parallel attention heads

# Returns
- `SWAttention` layer instance

# Example
```julia
using Lux, Random
attn = SWAttention(1024, 128, 4)
rng = Random.default_rng()
ps = Lux.initialparameters(rng, attn)
st = Lux.initialstates(rng, attn)
x = randn(Float32, 128, 64)  # (dimension, batch_size)
y, _ = attn(x, ps, st)  # (128, 64)
```
"""
function SWAttention(sequence_length::Int, dimension::Int, number_of_heads::Int)
    # ...
end
```

---

## Integration Suggestions

### Combining SWAttention + OSSM

Create a hybrid architecture:
```julia
struct SambaBlock <: Lux.AbstractLuxLayer
    attention::SWAttention
    ssm::OSSM
    mix_gate::Lux.Dense
    norm1::Lux.LayerNorm
    norm2::Lux.LayerNorm
end

function (block::SambaBlock)(x, params, state)
    # Parallel paths:
    # 1. Attention path
    x_norm1 = block.norm1(x, params.norm1, state.norm1)
    attn_out, st_attn = block.attention(x_norm1, params.attention, state.attention)

    # 2. SSM path
    x_norm2 = block.norm2(x, params.norm2, state.norm2)
    ssm_out, st_ssm = block.ssm(x_norm2, params.ssm, state.ssm)

    # 3. Gated mixing
    gate = block.mix_gate(x, params.mix_gate, state.mix_gate)
    out = gate .* attn_out + (1 .- gate) .* ssm_out + x  # residual

    return out, (attention=st_attn, ssm=st_ssm, ...)
end
```

**Why**: Combines global attention with local SSM dynamics; attention for long-range, SSM for sequential

---

## Priority Recommendations

### Immediate (High Impact, Low Effort)
1. ✅ **Add skip projection to OSSM** - removes `dim_in == dim_out` constraint
2. ✅ **Vectorize oscillator rotation** - 2-3× speedup, straightforward
3. ✅ **Add input validation** - better error messages, easy to add
4. ✅ **Write basic tests** - ensure correctness, prevent regressions

### Short-Term (High Impact, Medium Effort)
5. 🎯 **Add 1D convolution to OSSM** - proven in Mamba, local context
6. 🎯 **Add normalization to OSSM** - training stability (RMSNorm or LayerNorm)
7. 🎯 **Add causal masking to SWAttention** - enables autoregressive tasks
8. 🎯 **Implement attention dropout** - regularization
9. 🎯 **Add docstrings** - usability, easier onboarding

### Medium-Term (Mamba-Inspired, Higher Effort)
10. 🔬 **Make B and C selective in OSSM** - full selectivity like Mamba
11. 🔬 **HiPPO frequency initialization** - principled multi-scale coverage
12. 🔬 **Learnable initial state** - better than zero init
13. 🔬 **Multi-scale oscillator groups** - explicit frequency bands

### Long-Term (Research & Optimization)
14. 🚀 **Parallel associative scan** - 10-40× speedup (needs GPU kernels)
15. 🚀 **Hybrid OSSM-Mamba architecture** - combine strengths
16. 🚀 **MQA/GQA attention variants** - efficient inference
17. 🚀 **Learnable temperature in attention** - adaptive scaling
18. 🚀 **SambaBlock** (Attention + OSSM fusion) - explore combinations

### Impact Summary

**Biggest Performance Gains:**
- Vectorize oscillator rotation: ~3× speedup (easy)
- Parallel scan: ~40× speedup (hard, needs CUDA)
- 1D convolution: better accuracy (medium)

**Biggest Capability Gains:**
- Selective B, C: matches Mamba expressiveness
- Causal masking: enables language modeling
- Skip projection: architectural flexibility

**Best Quick Wins (do first):**
1. Skip projection (30 min)
2. Vectorize oscillators (1 hour)
3. Input validation (30 min)
4. Basic tests (2 hours)
5. 1D convolution (1-2 hours)

---

## References & Inspirations

### Key Papers

**State Space Models:**
- **S4**: Gu et al., ["Efficiently Modeling Long Sequences with Structured State Spaces"](https://arxiv.org/abs/2111.00396) (ICLR 2022)
- **Mamba-1**: Gu & Dao, ["Mamba: Linear-Time Sequence Modeling with Selective State Spaces"](https://arxiv.org/abs/2312.00752) (2023)
- **Mamba-2**: Dao & Gu, ["Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"](https://arxiv.org/abs/2405.21060) (2024)

**Attention Mechanisms:**
- Vaswani et al., ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) (NeurIPS 2017)
- Katharopoulos et al., ["Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"](https://arxiv.org/abs/2006.16236) (ICML 2020)

### Educational Resources

- [A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state) - Excellent visual introduction
- [Mamba Explained | The Gradient](https://thegradient.pub/mamba-explained/) - In-depth technical explanation
- [State Space Duality (Mamba-2) | Goomba Lab](https://goombalab.github.io/blog/2024/mamba2-part1-model/) - Mamba-2 deep dive
- [What Is A Mamba Model? | IBM](https://www.ibm.com/think/topics/mamba-model) - High-level overview

### Implementations

- [GitHub: state-spaces/mamba](https://github.com/state-spaces/mamba) - Official Mamba implementation (PyTorch + CUDA)
- [Lux.jl Documentation](https://lux.csail.mit.edu/) - Julia deep learning framework used in this project

### Neuroscience & Physics Inspirations

**OSSM:**
- Oscillatory networks from computational neuroscience
- Coupled harmonic oscillators from classical mechanics
- Phase space dynamics and limit cycles

### Potential Applications

**OSSM-specific:**
- Time series forecasting (explicit periodicity via oscillators)
- Audio/speech processing (multi-frequency decomposition)
- Circadian rhythm modeling (biological oscillations)
- Seasonal pattern detection (economic, climate data)
- Signal processing (Fourier-like learnable basis)

**General (SSM + Attention):**
- Long sequence modeling (genomics, long-form text)
- Efficient transformers (linear complexity alternative)
- Multimodal learning (audio, video, text)

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
| Sequential OSSM | 30-35% | 7-10 sec/step |
| Parallel Scan | 80-90% | 0.5-1 sec/step |
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

The drafter is built from **SwammaDrafterBlock** layers with time conditioning for diffusion:

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

Each block uses GLU-style gating between LinearAttention and WavePDE:

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
│  Also outputs α_bias (unused  │           │
│  in Drafter - used in full    │           │
│  Swamma for branch mixing)   │           │
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
│ ATTENTION │ │ (Oscillator SSM)  │         │
│           │ │                   │         │
│ O(n) glob │ │ Sequential state  │         │
│ context   │ │ space model with  │         │
│           │ │ damped harmonic   │         │
│ Q,K,V,O   │ │ oscillators       │         │
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

#### 1. LinearAttention (O(n) Global Context)
- Provides **global context** across the sequence without O(n²) cost
- Uses linear attention mechanism (no softmax)
- Processes `path_a` from the GLU split

#### 2. WavePDE (Oscillatory State Space Model)
- **Diagonal Linear Oscillatory State Space** model
- Models temporal dependencies via damped harmonic oscillators
- Each oscillator has learnable frequency `ω` and damping `α`
- Update: `x_t = ρ·R(θ)·x_{t-1} + B·u_t` where:
  - `ρ = exp(-α·Δt)` (damping)
  - `θ = ω·Δt` (rotation angle)
- Provides **sequential memory** that complements global attention

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
