# OSSAMMA-NER Architecture Diagram

A comprehensive technical reference for the OssammaNER (Named Entity Recognition with Dual-Gated Oscillatory Attention) architecture.

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    OSSAMMA-NER ARCHITECTURE                                        ║
║                      (Named Entity Recognition with Dual-Gated Oscillatory Attention)              ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ DESIGN INTENT: Combine global sequence understanding (oscillators) with local precision (sliding    │
│ window attention) via learnable dual gating for optimal NER boundary detection.                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
LAYER 1: INPUT EMBEDDINGS
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    Token IDs                          Position IDs                    Time Embedding
    (seq_len, batch)                   (seq_len,)                      (fixed t=0.5)
         │                                  │                               │
         ▼                                  ▼                               ▼
┌─────────────────────┐          ┌─────────────────────┐       ┌─────────────────────────────┐
│  Token Embedding    │          │ Position Embedding  │       │   FixedTimeEmbedding        │
│  ─────────────────  │          │ ─────────────────   │       │   ─────────────────────     │
│  vocab=32K → dim    │          │  max_seq → dim      │       │   sinusoidal encoding       │
│  (32000, emb_dim)   │          │  (512, emb_dim)     │       │   sin(args) ⊕ cos(args)     │
└─────────────────────┘          └─────────────────────┘       │   → (time_dim, batch)       │
         │                                  │                   └─────────────────────────────┘
         │    (emb_dim, seq, batch)         │                               │
         └──────────────┬───────────────────┘                               │
                        │                                                   │
                        ▼                                                   │
              ┌─────────────────┐                                           │
              │    ⊕ (add)      │                                           │
              └─────────────────┘                                           │
                        │                                                   │
                        ▼                                                   ▼
        ┌───────────────────────────────────────────────────────────────────────────────┐
        │                                                                               │
        │   x: (emb_dim, seq_len, batch)              t: (time_dim, batch)              │
        │                                                                               │
        └───────────────────────────────────────┬───────────────────────────────────────┘
                                                │
                                                ▼

═══════════════════════════════════════════════════════════════════════════════════════════════════════
LAYER 2-N: OSSAMMA-NER BLOCKS (× number_of_layers)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            OSSAMMA-NER BLOCK (Dual Gating Architecture)                             │
│                                                                                                     │
│  ╔═════════════════════════════════════════════════════════════════════════════════════════════╗   │
│  ║ ABSTRACTION: Two parallel processing streams with bidirectional gating                      ║   │
│  ║   • GLU-Global: Captures long-range dependencies via oscillators (physics-based SSM)        ║   │
│  ║   • Local-Sharp: Captures precise local patterns via sliding window attention               ║   │
│  ║   • Dual Gates: GLU controls what Local sees (input gate) AND what it outputs (output gate) ║   │
│  ╚═════════════════════════════════════════════════════════════════════════════════════════════╝   │
│                                                                                                     │
│     ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│     │                        TIME-CONDITIONED LAYER NORM                                       │   │
│     │                                                                                          │   │
│     │   Input x: (emb_dim, seq, batch)     Time t: (time_dim, batch)                          │   │
│     │        │                                  │                                              │   │
│     │        ▼                                  │                                              │   │
│     │   ┌──────────┐                           │                                              │   │
│     │   │LayerNorm │                           │                                              │   │
│     │   └────┬─────┘                           │                                              │   │
│     │        │                                 ▼                                              │   │
│     │        │                    ┌──────────────────────────┐                                │   │
│     │        │                    │ ScaleProj: time→emb_dim  │──▶ scale = 1 + proj(t)        │   │
│     │        │                    │ ShiftProj: time→emb_dim  │──▶ shift = proj(t)            │   │
│     │        │                    │ AlphaBiasProj: time→1    │──▶ α_bias (for mixing)        │   │
│     │        │                    └──────────────────────────┘                                │   │
│     │        │                                 │                                              │   │
│     │        ▼                                 │                                              │   │
│     │   normalized = LN(x) * scale + shift    │                                              │   │
│     │   Output: (emb_dim, seq, batch)         │                                              │   │
│     └──────────────────────────────────────────┼──────────────────────────────────────────────┘   │
│                        │                       │                                                   │
│                        │                       │ α_bias flows to mixing stage                      │
│                        ▼                       │                                                   │
│     ┌──────────────────────────────────────────┼──────────────────────────────────────────────┐   │
│     │                                          │                                              │   │
│     │   ╔════════════════════════════════════════════════════════════════════════════════╗   │   │
│     │   ║            GLU-GLOBAL BRANCH (Must complete FIRST - gates depend on it)        ║   │   │
│     │   ╚════════════════════════════════════════════════════════════════════════════════╝   │   │
│     │                                          │                                              │   │
│     │                      normalized: (emb_dim, seq, batch)                                 │   │
│     │                                          │                                              │   │
│     │                                          ▼                                              │   │
│     │   ┌──────────────────────────────────────────────────────────────────────────────────┐ │   │
│     │   │              GLU PROJECTION: Dense(emb_dim → 2×emb_dim)                          │ │   │
│     │   │                                                                                  │ │   │
│     │   │    ┌────────────────┐    ┌────────────────┐                                     │ │   │
│     │   │    │    path_a      │    │    path_b      │  (split along dim 1)                │ │   │
│     │   │    │ (emb_dim,seq,B)│    │ (emb_dim,seq,B)│                                     │ │   │
│     │   │    └───────┬────────┘    └───────┬────────┘                                     │ │   │
│     │   └────────────┼─────────────────────┼──────────────────────────────────────────────┘ │   │
│     │                │                     │                                                 │   │
│     │                ▼                     ▼                                                 │   │
│     │   ┌────────────────────────┐   ┌────────────────────────────────────────────────────┐ │   │
│     │   │   LINEAR ATTENTION     │   │            DLinOSS (Oscillator SSM)                │ │   │
│     │   │   ──────────────────   │   │            ──────────────────────                  │ │   │
│     │   │                        │   │                                                    │ │   │
│     │   │   ┌──────────────────┐ │   │  PHYSICS MODEL: Damped harmonic oscillators       │ │   │
│     │   │   │ Q,K,V projections│ │   │                                                    │ │   │
│     │   │   │ (emb→emb each)   │ │   │    ẍ + 2α·ẋ + ω²·x = u                            │ │   │
│     │   │   └────────┬─────────┘ │   │                                                    │ │   │
│     │   │            │           │   │  Parameters (log-space for stability):             │ │   │
│     │   │   Multi-head split     │   │    • log_time_step: (state_dim,)                  │ │   │
│     │   │   head_dim = emb/heads │   │    • log_stiffness: (state_dim,) → ω²             │ │   │
│     │   │            │           │   │    • log_damping: (state_dim,) → α                │ │   │
│     │   │            ▼           │   │    • input_proj: (state_dim, emb_dim)              │ │   │
│     │   │   ┌──────────────────┐ │   │    • output_proj: (emb_dim, state_dim)            │ │   │
│     │   │   │ Time projection  │ │   │                                                    │ │   │
│     │   │   │ Q_t = Q + t_proj │ │   │  State: (2, state_dim) = [velocities; positions]  │ │   │
│     │   │   │ K_t = K + t_proj │ │   │                                                    │ │   │
│     │   │   └────────┬─────────┘ │   │  Evolution (implicit Euler):                       │ │   │
│     │   │            │           │   │    implicit_damp = 1/(1 + Δt·α)                    │ │   │
│     │   │   Feature extraction   │   │    v_new = damp·v - Δt·ω²·damp·x + Δt·damp·u      │ │   │
│     │   │   Dense→Dense per head │   │    x_new = x + Δt·v_new                            │ │   │
│     │   │            │           │   │                                                    │ │   │
│     │   │            ▼           │   │  ACTIVATION: softplus(output) before GLU gate     │ │   │
│     │   │   ┌──────────────────┐ │   │                                                    │ │   │
│     │   │   │ Pos embeddings   │ │   │  Dim flow:                                         │ │   │
│     │   │   │ cos + sin streams│ │   │    (emb_dim, seq, B) → proj → (state_dim, seq, B) │ │   │
│     │   │   │ + LayerNorm      │ │   │    → accumulate over seq → extract positions       │ │   │
│     │   │   └────────┬─────────┘ │   │    → output proj → (emb_dim, seq, B)              │ │   │
│     │   │            │           │   │                                                    │ │   │
│     │   │   ACTIVATION: softplus │   └─────────────────────────┬──────────────────────────┘ │   │
│     │   │   on feature maps      │                             │                            │   │
│     │   │            │           │                             │                            │   │
│     │   │   Linear attention:    │                             ▼                            │   │
│     │   │   out = (V·K^T)·Q      │                      ┌─────────────────┐                 │   │
│     │   │   O(n·d²) complexity   │                      │ σ(·) sigmoid    │                 │   │
│     │   │            │           │                      │ gate stream     │                 │   │
│     │   └────────────┼───────────┘                      └────────┬────────┘                 │   │
│     │                │                                           │                          │   │
│     │                │ attn_out                                  │ osc_gate                 │   │
│     │                │ (emb_dim, seq, B)                         │ (emb_dim, seq, B)        │   │
│     │                │                                           │                          │   │
│     │                └─────────────────┬─────────────────────────┘                          │   │
│     │                                  │                                                    │   │
│     │                                  ▼                                                    │   │
│     │                     ┌────────────────────────────┐                                    │   │
│     │                     │      GLU GATING            │                                    │   │
│     │                     │  glu = attn ⊙ σ(osc)       │                                    │   │
│     │                     └────────────┬───────────────┘                                    │   │
│     │                                  │                                                    │   │
│     │                                  ▼                                                    │   │
│     │                     ┌────────────────────────────┐                                    │   │
│     │                     │  GLU Output Projection     │                                    │   │
│     │                     │  Dense(emb_dim → emb_dim)  │                                    │   │
│     │                     └────────────┬───────────────┘                                    │   │
│     │                                  │                                                    │   │
│     │                                  ▼                                                    │   │
│     │                     ┌────────────────────────────┐                                    │   │
│     │                     │        GLU_out             │                                    │   │
│     │                     │   (emb_dim, seq, batch)    │                                    │   │
│     │                     └───────────┬┬┬──────────────┘                                    │   │
│     │                                 │││                                                   │   │
│     └─────────────────────────────────┼┼┼───────────────────────────────────────────────────┘   │
│                                       │││                                                       │
│                        ┌──────────────┘│└──────────────┐                                        │
│                        │               │               │                                        │
│                        │               │               │                                        │
│                        ▼               │               ▼                                        │
│     ┌────────────────────────────┐     │     ┌────────────────────────────┐                     │
│     │      INPUT GATE            │     │     │      OUTPUT GATE           │                     │
│     │   W_in: (emb_dim, emb_dim) │     │     │   W_out: (emb_dim, emb_dim)│                     │
│     │   no bias                  │     │     │   no bias                  │                     │
│     │                            │     │     │                            │                     │
│     │   gate_in = σ(W_in·GLU)    │     │     │   gate_out = σ(W_out·GLU)  │                     │
│     └───────────┬────────────────┘     │     └────────────┬───────────────┘                     │
│                 │                      │                  │                                     │
│                 │ (emb, seq, B)        │                  │ (emb, seq, B)                       │
│                 │                      │                  │                                     │
│     ┌───────────┼──────────────────────┼──────────────────┼───────────────────────────────────┐ │
│     │           │                      │                  │                                   │ │
│     │   ╔══════════════════════════════════════════════════════════════════════════════════╗ │ │
│     │   ║                    LOCAL-SHARP BRANCH (with dual gating)                         ║ │ │
│     │   ╚══════════════════════════════════════════════════════════════════════════════════╝ │ │
│     │           │                      │                  │                                   │ │
│     │           │                      │ normalized       │                                   │ │
│     │           │                      │ (emb,seq,B)      │                                   │ │
│     │           │                      │                  │                                   │ │
│     │           │                      ▼                  │                                   │ │
│     │           │         ┌────────────────────────────┐  │                                   │ │
│     │           └────────▶│      INPUT GATING          │  │                                   │ │
│     │                     │  gated_x = norm ⊙ gate_in  │  │                                   │ │
│     │                     │                            │  │                                   │ │
│     │                     │  INTENT: GLU suppresses    │  │                                   │ │
│     │                     │  irrelevant features       │  │                                   │ │
│     │                     │  BEFORE local processing   │  │                                   │ │
│     │                     └────────────┬───────────────┘  │                                   │ │
│     │                                  │                  │                                   │ │
│     │                                  ▼                  │                                   │ │
│     │                     ┌────────────────────────────────────────────────────────────────┐ │ │
│     │                     │              SLIDING WINDOW ATTENTION (SWAttention)            │ │ │
│     │                     │                                                                │ │ │
│     │                     │   Parameters:                                                  │ │ │
│     │                     │     • QueryProj: (emb_dim, emb_dim)                           │ │ │
│     │                     │     • KeyProj: (emb_dim, emb_dim)                             │ │ │
│     │                     │     • ValueProj: (emb_dim, emb_dim)                           │ │ │
│     │                     │     • OutputProj: (emb_dim, emb_dim)                          │ │ │
│     │                     │     • window_size: radius for local attention (default 5)     │ │ │
│     │                     │                                                                │ │ │
│     │                     │   ┌──────────────────────────────────────────────────────────┐│ │ │
│     │                     │   │ Multi-head attention with sliding window mask            ││ │ │
│     │                     │   │                                                          ││ │ │
│     │                     │   │   Q, K, V = projections(gated_x)                         ││ │ │
│     │                     │   │   reshape: (head_dim, seq, num_heads, batch)             ││ │ │
│     │                     │   │                                                          ││ │ │
│     │                     │   │   scores = K^T @ Q / √head_dim                           ││ │ │
│     │                     │   │                                                          ││ │ │
│     │                     │   │   ┌─────────────────────────────────────────────────┐   ││ │ │
│     │                     │   │   │  SLIDING WINDOW MASK                            │   ││ │ │
│     │                     │   │   │  mask[i,j] = |i-j| ≤ window_size ? 1 : -∞       │   ││ │ │
│     │                     │   │   │                                                 │   ││ │ │
│     │                     │   │   │  Example (window=2):                            │   ││ │ │
│     │                     │   │   │    [1 1 1 . . .]                                │   ││ │ │
│     │                     │   │   │    [1 1 1 1 . .]                                │   ││ │ │
│     │                     │   │   │    [1 1 1 1 1 .]                                │   ││ │ │
│     │                     │   │   │    [. 1 1 1 1 1]                                │   ││ │ │
│     │                     │   │   │    [. . 1 1 1 1]                                │   ││ │ │
│     │                     │   │   │    [. . . 1 1 1]                                │   ││ │ │
│     │                     │   │   └─────────────────────────────────────────────────┘   ││ │ │
│     │                     │   │                                                          ││ │ │
│     │                     │   │   ACTIVATION: sigsoftmax (not standard softmax)          ││ │ │
│     │                     │   │   sigsoftmax(x) = softmax(x + logsigmoid(x))             ││ │ │
│     │                     │   │                                                          ││ │ │
│     │                     │   │   INTENT: Combines benefits of softmax (normalization)   ││ │ │
│     │                     │   │   and sigmoid (bounded gradients, sparse attention)      ││ │ │
│     │                     │   │                                                          ││ │ │
│     │                     │   │   out = attn_weights @ V                                 ││ │ │
│     │                     │   │   merge heads → output projection                        ││ │ │
│     │                     │   └──────────────────────────────────────────────────────────┘│ │ │
│     │                     │                                                                │ │ │
│     │                     │   Output: (emb_dim, seq, batch)                               │ │ │
│     │                     └────────────────────────┬───────────────────────────────────────┘ │ │
│     │                                              │                                         │ │
│     │                                              │ local_out                               │ │
│     │                                              ▼                                         │ │
│     │                                 ┌─────────────────────────────┐◀────────────────────────┘ │
│     │                                 │      OUTPUT GATING          │                           │
│     │                                 │                             │                           │
│     │                                 │  local_final = local_out    │                           │
│     │                                 │              + gate_out     │                           │
│     │                                 │                ⊙ GLU_out    │                           │
│     │                                 │                             │                           │
│     │                                 │  INTENT: GLU injects global │                           │
│     │                                 │  context where Local needs  │                           │
│     │                                 │  help (entity continuation) │                           │
│     │                                 └──────────────┬──────────────┘                           │
│     │                                                │                                          │
│     └────────────────────────────────────────────────┼──────────────────────────────────────────┘
│                                                      │                                          │
│                                                      │ local_final (emb, seq, B)                │
│                                                      │                                          │
│     ┌────────────────────────────────────────────────┼──────────────────────────────────────────┐
│     │                      ADAPTIVE MIXING                                                      │
│     │                                                │                                          │
│     │   ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│     │   │  input_pooled = mean(normalized, dim=seq)  → (emb_dim, batch)                  │    │
│     │   │                                                                                 │    │
│     │   │  α_logits = AlphaProjection(input_pooled)  → (1, batch)                        │    │
│     │   │           Dense(emb_dim → 1)                                                    │    │
│     │   │                                                                                 │    │
│     │   │  α = σ(α_logits + α_bias)    ← α_bias from time-conditioned norm               │    │
│     │   │                                                                                 │    │
│     │   │  ACTIVATION: sigmoid → α ∈ (0, 1)                                              │    │
│     │   │                                                                                 │    │
│     │   │  mixed = α · GLU_out + (1 - α) · local_final                                   │    │
│     │   │                                                                                 │    │
│     │   │  INTENT: Learned per-sample blend of global vs local processing                │    │
│     │   │  Time embedding biases α for diffusion-based training (fixed for NER)          │    │
│     │   └─────────────────────────────────────────────────────────────────────────────────┘    │
│     │                                                │                                          │
│     └────────────────────────────────────────────────┼──────────────────────────────────────────┘
│                                                      │                                          │
│                                                      │ mixed (emb_dim, seq, batch)              │
│                                                      ▼                                          │
│     ┌─────────────────────────────────────────────────────────────────────────────────────────┐ │
│     │                      RESIDUAL + NORMALIZATION                                           │ │
│     │                                                                                         │ │
│     │   ┌───────────────────────────────────────────────────────────────────────────────┐    │ │
│     │   │                                                                               │    │ │
│     │   │  pre_dropout = residual + mixed        (residual = original input x)         │    │ │
│     │   │                                                                               │    │ │
│     │   │  post_dropout = Dropout(pre_dropout)   dropout_rate = 0.1 (default)          │    │ │
│     │   │                                                                               │    │ │
│     │   │  output = LayerNorm(post_dropout)      (emb_dim,) learned scale & bias       │    │ │
│     │   │                                                                               │    │ │
│     │   │  PATTERN: Pre-norm (before branches) + Post-norm (after residual)            │    │ │
│     │   │                                                                               │    │ │
│     │   └───────────────────────────────────────────────────────────────────────────────┘    │ │
│     │                                                                                         │ │
│     └─────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                      │                                          │
│                                                      ▼                                          │
│                                     Output: (emb_dim, seq_len, batch)                           │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                      │
                                            Repeat × N layers
                                                      │
                                                      ▼

═══════════════════════════════════════════════════════════════════════════════════════════════════════
LAYER N+1: CLASSIFICATION HEAD
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                               │
    │   Input: (emb_dim, seq_len, batch)                                                           │
    │                                                                                               │
    │                           ┌──────────────────────────┐                                        │
    │                           │      Dropout(0.1)        │                                        │
    │                           └────────────┬─────────────┘                                        │
    │                                        │                                                      │
    │                                        ▼                                                      │
    │                           ┌──────────────────────────┐                                        │
    │                           │      LayerNorm           │                                        │
    │                           │   (emb_dim,) γ, β        │                                        │
    │                           └────────────┬─────────────┘                                        │
    │                                        │                                                      │
    │                                        ▼                                                      │
    │                           ┌──────────────────────────┐                                        │
    │                           │      Dropout(0.1)        │                                        │
    │                           └────────────┬─────────────┘                                        │
    │                                        │                                                      │
    │                                        ▼                                                      │
    │                           ┌──────────────────────────┐                                        │
    │                           │   Dense(emb → num_labels)│                                        │
    │                           │   W: (num_labels, emb)   │                                        │
    │                           │   b: (num_labels,)       │                                        │
    │                           │   NO activation          │                                        │
    │                           │   (logits for softmax)   │                                        │
    │                           └────────────┬─────────────┘                                        │
    │                                        │                                                      │
    │                                        ▼                                                      │
    │                           Output: (num_labels, seq_len, batch)                               │
    │                                                                                               │
    └───────────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
LOSS & INFERENCE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                               │
    │   TRAINING LOSS: ner_cross_entropy                                                           │
    │   ─────────────────────────────────                                                          │
    │                                                                                               │
    │   logits: (num_labels, seq_len, batch)                                                       │
    │   labels: (seq_len, batch)  where -100 = ignore (padding/special tokens)                     │
    │                                                                                               │
    │   loss = mean( -log_softmax(logits)[label] )   over valid positions only                     │
    │                                                                                               │
    │   ─────────────────────────────────────────────────────────────────────────────────────────  │
    │                                                                                               │
    │   INFERENCE: argmax(softmax(logits), dim=1) → predicted label indices                        │
    │              Then map indices to label strings                                               │
    │                                                                                               │
    └───────────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
NER LABEL SCHEMA (19 Labels, BIO Encoding)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
    │  Index │ Label         │ Description                                                       │
    │────────┼───────────────┼───────────────────────────────────────────────────────────────────│
    │    1   │ O             │ Outside any entity                                                │
    │────────┼───────────────┼───────────────────────────────────────────────────────────────────│
    │   2-3  │ B/I-PERSON    │ Named individuals (Einstein, Marie Curie)                         │
    │   4-5  │ B/I-AGENCY    │ Organizations, companies, governments                             │
    │   6-7  │ B/I-PLACE     │ Locations, addresses, celestial bodies                            │
    │   8-9  │ B/I-ORGANISM  │ Animals, plants, microorganisms                                   │
    │  10-11 │ B/I-EVENT     │ Named events, wars, eras                                          │
    │  12-13 │ B/I-INSTRUMENT│ Tools, products, devices                                          │
    │  14-15 │ B/I-WORK      │ Books, papers, films, datasets                                    │
    │  16-17 │ B/I-DOMAIN    │ Sciences, methods, fields of study                                │
    │  18-19 │ B/I-MEASURE   │ Numbers, dates, monetary values                                   │
    └─────────────────────────────────────────────────────────────────────────────────────────────┘

    INTENT: 9 entity types optimized for RAG (Retrieval-Augmented Generation)
            High linking potential between documents

═══════════════════════════════════════════════════════════════════════════════════════════════════════
MODEL CONFIGURATIONS (TOML-based)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    Configuration files are located in: configs/ner_*.toml

    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  Profile    │ Config File          │ Use Case                    │ Est. Params             │
    │─────────────┼──────────────────────┼─────────────────────────────┼─────────────────────────│
    │  minimal    │ ner_minimal.toml     │ Smoke testing / CI/CD       │    ~1.5M                │
    │  dev        │ ner_dev.toml         │ Local development           │    ~35M                 │
    │  production │ ner_production.toml  │ Production deployment       │   ~150M                 │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘

    DETAILED CONFIGURATIONS:
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                              │
    │  MINIMAL (ner_minimal.toml) - Smoke Testing                                                  │
    │  ────────────────────────────────────────────────────────────────────────────────────────── │
    │    vocab_size:           1,000                                                               │
    │    max_sequence_length:  64                                                                  │
    │    embedding_dimension:  64                                                                  │
    │    number_of_heads:      2                                                                   │
    │    number_of_layers:     2                                                                   │
    │    time_dimension:       32                                                                  │
    │    state_dimension:      64                                                                  │
    │    window_size:          8                                                                   │
    │    use_crf:              false                                                               │
    │    device:               cpu                                                                 │
    │                                                                                              │
    │  DEV (ner_dev.toml) - Development & Experimentation                                          │
    │  ────────────────────────────────────────────────────────────────────────────────────────── │
    │    vocab_size:           32,000                                                              │
    │    max_sequence_length:  256                                                                 │
    │    embedding_dimension:  256                                                                 │
    │    number_of_heads:      4                                                                   │
    │    number_of_layers:     6                                                                   │
    │    time_dimension:       64                                                                  │
    │    state_dimension:      256                                                                 │
    │    window_size:          32                                                                  │
    │    use_crf:              true                                                                │
    │    device:               gpu                                                                 │
    │                                                                                              │
    │  PRODUCTION (ner_production.toml) - Full-Scale Deployment                                    │
    │  ────────────────────────────────────────────────────────────────────────────────────────── │
    │    vocab_size:           50,000                                                              │
    │    max_sequence_length:  512                                                                 │
    │    embedding_dimension:  1024                                                                │
    │    number_of_heads:      16                                                                  │
    │    number_of_layers:     10                                                                  │
    │    time_dimension:       256                                                                 │
    │    state_dimension:      1024                                                                │
    │    window_size:          64                                                                  │
    │    use_crf:              true                                                                │
    │    device:               gpu (optimized for RTX 5090 / A100)                                 │
    │                                                                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘

    USAGE:
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                              │
    │  # Load model from TOML config                                                               │
    │  model = OssammaNER("configs/ner_production.toml")                                          │
    │                                                                                              │
    │  # Or load config separately                                                                 │
    │  config = load_ner_config("configs/ner_dev.toml")                                           │
    │  print_config_summary(config)  # Shows config with estimated params                         │
    │  model = OssammaNER(config)                                                                  │
    │                                                                                              │
    │  # Load training hyperparameters                                                             │
    │  train_config = load_training_config("configs/ner_production.toml")                         │
    │                                                                                              │
    │  # Estimate parameters for custom config                                                     │
    │  params = estimate_parameters(config)                                                        │
    │                                                                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘

    TOML STRUCTURE:
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                              │
    │  [model]                           # Core architecture                                       │
    │  vocab_size = 50000                                                                          │
    │  max_sequence_length = 512                                                                   │
    │  embedding_dimension = 1024                                                                  │
    │  number_of_heads = 16                                                                        │
    │  number_of_layers = 10                                                                       │
    │  num_labels = 19                                                                             │
    │                                                                                              │
    │  [model.dimensions]                # Internal dimensions                                     │
    │  time_dimension = 256                                                                        │
    │  state_dimension = 1024                                                                      │
    │                                                                                              │
    │  [model.attention]                 # Attention settings                                      │
    │  window_size = 64                                                                            │
    │                                                                                              │
    │  [model.oscillator]                # DLinOSS physics parameters                              │
    │  min_frequency = 0.01                                                                        │
    │  max_frequency = 5.0                                                                         │
    │  default_time_step = 0.05                                                                    │
    │                                                                                              │
    │  [model.regularization]            # Training regularization                                 │
    │  dropout_rate = 0.1                                                                          │
    │  label_smoothing = 0.1                                                                       │
    │  use_crf = true                                                                              │
    │                                                                                              │
    │  [training]                        # Training hyperparameters                                │
    │  batch_size = 32                                                                             │
    │  gradient_accumulation_steps = 4                                                             │
    │  learning_rate = 2e-4                                                                        │
    │  warmup_steps = 2000                                                                         │
    │  total_steps = 100000                                                                        │
    │                                                                                              │
    │  [training.checkpoints]            # Checkpoint settings                                     │
    │  eval_every = 500                                                                            │
    │  save_every = 2000                                                                           │
    │                                                                                              │
    │  [data]                            # Data paths                                              │
    │  train_path = "data/ner/train.jsonl"                                                        │
    │  val_path = "data/ner/validation.jsonl"                                                     │
    │                                                                                              │
    │  [hardware]                        # Device settings                                         │
    │  device = "gpu"                                                                              │
    │  mixed_precision = true                                                                      │
    │                                                                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
ACTIVATION FUNCTION SUMMARY
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  Function    │ Formula                          │ Where Used                                │
    │──────────────┼──────────────────────────────────┼───────────────────────────────────────────│
    │  sigmoid     │ σ(x) = 1/(1+e^(-x))              │ All gates (GLU, input, output, α)         │
    │  sigsoftmax  │ softmax(x + log(σ(x)))           │ SWAttention weights                       │
    │  softplus    │ log(1 + e^x)                     │ DLinOSS output, LinearAttention features  │
    │  gelu        │ x·Φ(x)                           │ LinearAttention feature projections       │
    │  (none)      │ linear                           │ Classification head (raw logits)          │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
STATE MANAGEMENT (Lux.jl Convention)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                              │
    │   STATELESS LAYERS: SWAttention, Dense, LayerNorm                                           │
    │     → state = (;)  (empty NamedTuple)                                                       │
    │                                                                                              │
    │   STATEFUL LAYERS: DLinOSS                                                                  │
    │     → state = (oscillator_state = zeros(2, state_dim),)                                     │
    │     → Carries velocity/position between timesteps within sequence                           │
    │     → Reset to zeros at sequence start                                                      │
    │                                                                                              │
    │   All layers return: (output, new_state)                                                    │
    │                                                                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
KEY ARCHITECTURAL INNOVATIONS
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                              │
    │  1. DUAL GATING: GLU branch controls Local branch bidirectionally                           │
    │     • Input gate: Filters what Local sees (pre-attention gating)                            │
    │     • Output gate: Injects global context where needed (post-attention gating)              │
    │     • Intent: Entity boundaries need local precision, but continuation needs global context │
    │                                                                                              │
    │  2. PHYSICS-BASED SSM: DLinOSS uses damped harmonic oscillators                             │
    │     • Learnable frequency (ω), damping (α), time step (Δt)                                  │
    │     • Stable evolution via implicit Euler integration                                       │
    │     • Intent: Natural decay/resonance patterns for sequence modeling                        │
    │                                                                                              │
    │  3. SIGSOFTMAX ATTENTION: softmax(x + logsigmoid(x))                                        │
    │     • Combines normalization (softmax) with bounded gradients (sigmoid)                     │
    │     • Intent: More stable training, potentially sparser attention                           │
    │                                                                                              │
    │  4. TIME-CONDITIONED NORMALIZATION: For diffusion/denoising tasks                           │
    │     • Scale and shift modulated by time embedding                                           │
    │     • α_bias provides time-dependent mixing preference                                      │
    │     • Intent: Support masked diffusion pretraining (LLaDA-style)                            │
    │                                                                                              │
    │  5. GLU PARALLEL BRANCHES: Linear attention + Oscillators                                   │
    │     • Linear attention: O(n·d²) complexity for content stream                               │
    │     • Oscillators: O(n·d) complexity for gate stream                                        │
    │     • Intent: Efficient gating without quadratic attention for gate computation             │
    │                                                                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
DATA FLOW SUMMARY
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    Token IDs (seq, batch)
         │
         ▼
    Embedding Layer → (emb_dim, seq, batch)
         │
         │    + Position Embedding (broadcast add)
         │
         ▼
    ┌─────────────────────────────────────┐
    │       OssammaNERBlock × N           │
    │                                     │
    │  [Time-Norm → GLU-Global → Dual-    │
    │   Gated Local → Mix → Residual]     │
    │                                     │
    └─────────────────────────────────────┘
         │
         ▼
    Dropout → LayerNorm → Dropout → Dense
         │
         ▼
    Logits (num_labels, seq, batch)
         │
         ▼
    Cross-Entropy Loss / argmax Prediction

═══════════════════════════════════════════════════════════════════════════════════════════════════════
COMPLEXITY ANALYSIS
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                              │
    │  COMPONENT                    │ TIME COMPLEXITY      │ SPACE COMPLEXITY    │ NOTES          │
    │──────────────────────────────────────────────────────────────────────────────────────────────│
    │                                                                                              │
    │  Token Embedding              │ O(n)                 │ O(V × d)            │ Lookup only    │
    │  Position Embedding           │ O(n)                 │ O(L × d)            │ Lookup only    │
    │  Time Embedding               │ O(d_t)               │ O(d_t)              │ Sinusoidal     │
    │                                                                                              │
    │  ─────────────────────────────────────────────────────────────────────────────────────────── │
    │  PER OSSAMMA-NER BLOCK:                                                                      │
    │  ─────────────────────────────────────────────────────────────────────────────────────────── │
    │                                                                                              │
    │  TimeConditionedLayerNorm     │ O(n × d)             │ O(d)                │ Element-wise   │
    │                                                                                              │
    │  GLU Projection               │ O(n × d²)            │ O(d × 2d)           │ Dense          │
    │                                                                                              │
    │  LinearAttention              │ O(n × d² / h)        │ O(n × d)            │ NOT O(n²)!     │
    │    - Q,K,V projections        │ O(n × d²)            │                     │                │
    │    - Feature maps             │ O(n × d_h²)          │                     │                │
    │    - Context: V @ K^T         │ O(n × d_h²)          │ O(d_h × d_h)        │ Per head       │
    │    - Output: ctx @ Q          │ O(n × d_h²)          │                     │                │
    │                                                                                              │
    │  DLinOSS Oscillators          │ O(n × d_s)           │ O(d_s × d)          │ Sequential     │
    │    - Input projection         │ O(n × d × d_s)       │                     │ scan           │
    │    - State evolution          │ O(n × d_s)           │ O(d_s)              │                │
    │    - Output projection        │ O(n × d_s × d)       │                     │                │
    │                                                                                              │
    │  GLU Gating                   │ O(n × d)             │ O(1)                │ Element-wise   │
    │  GLU Output Proj              │ O(n × d²)            │ O(d × d)            │                │
    │                                                                                              │
    │  Input/Output Gates           │ O(n × d²) each       │ O(d × d) each       │ No bias        │
    │                                                                                              │
    │  SWAttention                  │ O(n × w × d)         │ O(n × w × h)        │ w = window     │
    │    - Q,K,V projections        │ O(n × d²)            │                     │                │
    │    - Masked attention         │ O(n × w × d_h)       │ O(n × w)            │ Sparse!        │
    │    - Output projection        │ O(n × d²)            │                     │                │
    │                                                                                              │
    │  Alpha Mixing                 │ O(n × d)             │ O(d)                │ Pool + dense   │
    │                                                                                              │
    │  Residual + LayerNorm         │ O(n × d)             │ O(d)                │                │
    │                                                                                              │
    │  ─────────────────────────────────────────────────────────────────────────────────────────── │
    │  CLASSIFICATION HEAD:                                                                        │
    │  ─────────────────────────────────────────────────────────────────────────────────────────── │
    │                                                                                              │
    │  LayerNorm + Dropout          │ O(n × d)             │ O(d)                │                │
    │  Dense (d → num_labels)       │ O(n × d × L)         │ O(d × L)            │ L = 19 labels  │
    │                                                                                              │
    │  ═══════════════════════════════════════════════════════════════════════════════════════════ │
    │                                                                                              │
    │  TOTAL PER BLOCK:             │ O(n × d² + n × w × d)│ O(d²)               │                │
    │                                                                                              │
    │  TOTAL MODEL (N blocks):      │ O(N × n × d²)        │ O(N × d²)           │                │
    │                                                                                              │
    │  ═══════════════════════════════════════════════════════════════════════════════════════════ │
    │                                                                                              │
    │  KEY INSIGHT: Linear attention + sliding window = sub-quadratic in sequence length          │
    │               Standard transformer: O(n² × d)                                                │
    │               OssammaNER:          O(n × d² + n × w × d) where w << n                       │
    │                                                                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  NOTATION:                                                                                   │
    │    n = sequence length       d = embedding dimension    d_h = head dimension (d/h)          │
    │    h = number of heads       d_s = state dimension      d_t = time dimension                │
    │    w = window size           V = vocabulary size        L = max sequence length             │
    │    N = number of layers                                                                      │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
PARAMETER COUNT BREAKDOWN (base model: d=512, h=8, N=6, d_s=512, d_t=128)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                              │
    │  EMBEDDINGS:                                                                                 │
    │  ────────────────────────────────────────────────────────────────────────────────────────── │
    │    Token Embedding:     32,000 × 512                           = 16,384,000                 │
    │    Position Embedding:  512 × 512                              =    262,144                 │
    │    Time Embedding:      (sinusoidal, no learnable params)      =          0                 │
    │                                                         Subtotal: 16,646,144                │
    │                                                                                              │
    │  PER OSSAMMA-NER BLOCK:                                                                      │
    │  ────────────────────────────────────────────────────────────────────────────────────────── │
    │                                                                                              │
    │    TimeConditionedLayerNorm:                                                                 │
    │      LayerNorm (γ, β):           512 × 2                       =      1,024                 │
    │      ScaleProjection:            128 × 512 + 512               =     66,048                 │
    │      ShiftProjection:            128 × 512 + 512               =     66,048                 │
    │      AlphaBiasProjection:        128 × 1 + 1                   =        129                 │
    │                                                                                              │
    │    GLU Branch:                                                                               │
    │      GluProjection:              512 × 1024 + 1024             =    525,312                 │
    │      GluOutputProjection:        512 × 512 + 512               =    262,656                 │
    │                                                                                              │
    │    LinearAttention:                                                                          │
    │      Q,K,V,O projections:        4 × (512 × 512 + 512)         =  1,050,624                 │
    │      QueryFeatureLinear:         64 × 128 + 128 × 64 + 128+64  =     16,576                 │
    │      KeyFeatureLinear:           64 × 128 + 128 × 64 + 128+64  =     16,576                 │
    │      TimeProjection:             128 × 64 + 64                 =      8,256                 │
    │      FeatureNorm:                64 × 2 × 4 streams            =        512                 │
    │      PositionEmbeddings:         (fixed sinusoidal)            =          0                 │
    │                                                                                              │
    │    DLinOSS:                                                                                  │
    │      log_time_step:              512                           =        512                 │
    │      log_stiffness:              512                           =        512                 │
    │      log_damping:                512                           =        512                 │
    │      input_projection:           512 × 512                     =    262,144                 │
    │      output_projection:          512 × 512                     =    262,144                 │
    │                                                                                              │
    │    Dual Gates:                                                                               │
    │      InputGate (no bias):        512 × 512                     =    262,144                 │
    │      OutputGate (no bias):       512 × 512                     =    262,144                 │
    │                                                                                              │
    │    SWAttention:                                                                              │
    │      Q,K,V,O projections:        4 × (512 × 512 + 512)         =  1,050,624                 │
    │                                                                                              │
    │    AlphaProjection:              512 × 1 + 1                   =        513                 │
    │    AttentionDropout:             (no params)                   =          0                 │
    │    OutputNorm (γ, β):            512 × 2                       =      1,024                 │
    │                                                                                              │
    │                                                    Block Subtotal: ~4,215,000               │
    │                                                                                              │
    │  CLASSIFICATION HEAD:                                                                        │
    │  ────────────────────────────────────────────────────────────────────────────────────────── │
    │    Dropout:                      (no params)                   =          0                 │
    │    LayerNorm (γ, β):             512 × 2                       =      1,024                 │
    │    Dropout:                      (no params)                   =          0                 │
    │    Dense (512 → 19):             512 × 19 + 19                 =      9,747                 │
    │                                                         Subtotal:     10,771                │
    │                                                                                              │
    │  ═══════════════════════════════════════════════════════════════════════════════════════════ │
    │                                                                                              │
    │  TOTAL (6 layers):                                                                           │
    │    Embeddings:                                                    16,646,144                │
    │    Blocks (6 × ~4.2M):                                           25,290,000                 │
    │    Classification Head:                                              10,771                 │
    │                                                                  ───────────                │
    │                                                        TOTAL:   ~42,000,000 (~42M)          │
    │                                                                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
GRADIENT FLOW ANALYSIS
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                              │
    │                                    ┌─────────────┐                                           │
    │                                    │    Loss     │                                           │
    │                                    └──────┬──────┘                                           │
    │                                           │                                                  │
    │                              ∂L/∂logits   │                                                  │
    │                                           ▼                                                  │
    │                           ┌───────────────────────────────┐                                  │
    │                           │    Classification Head        │                                  │
    │                           │    (direct gradient path)     │                                  │
    │                           └───────────────┬───────────────┘                                  │
    │                                           │                                                  │
    │     ┌─────────────────────────────────────┼─────────────────────────────────────┐           │
    │     │                      BLOCK N (top)  │                                     │           │
    │     │                                     ▼                                     │           │
    │     │              ┌──────────────────────────────────────────┐                 │           │
    │     │              │           OutputNorm                     │                 │           │
    │     │              └──────────────────┬───────────────────────┘                 │           │
    │     │                                 │                                         │           │
    │     │                    ┌────────────┴────────────┐                            │           │
    │     │                    │                         │                            │           │
    │     │                    ▼                         │                            │           │
    │     │        ┌───────────────────────┐            │  ◀── RESIDUAL CONNECTION   │           │
    │     │        │   Adaptive Mixing     │            │      (gradient highway)    │           │
    │     │        │   (α blend)           │            │                            │           │
    │     │        └───────┬───────────────┘            │                            │           │
    │     │                │                            │                            │           │
    │     │     ┌──────────┴──────────┐                 │                            │           │
    │     │     │                     │                 │                            │           │
    │     │     ▼                     ▼                 │                            │           │
    │     │  ┌──────────┐      ┌──────────────┐        │                            │           │
    │     │  │GLU-Global│◀────▶│ Local-Sharp  │        │                            │           │
    │     │  │ Branch   │      │ Branch       │        │                            │           │
    │     │  │          │      │              │        │                            │           │
    │     │  │ Gradients│      │ Gated grads  │        │                            │           │
    │     │  │ flow via │      │ flow via     │        │                            │           │
    │     │  │ α weight │      │ (1-α) weight │        │                            │           │
    │     │  │   AND    │      │   AND        │        │                            │           │
    │     │  │ both     │      │ input/output │        │                            │           │
    │     │  │ gates    │      │ gate paths   │        │                            │           │
    │     │  └────┬─────┘      └──────┬───────┘        │                            │           │
    │     │       │                   │                │                            │           │
    │     │       └─────────┬─────────┘                │                            │           │
    │     │                 │                          │                            │           │
    │     │                 ▼                          │                            │           │
    │     │        ┌────────────────────┐              │                            │           │
    │     │        │TimeConditionedNorm │              │                            │           │
    │     │        └────────┬───────────┘              │                            │           │
    │     │                 │                          │                            │           │
    │     │                 └──────────────────────────┘                            │           │
    │     │                                                                         │           │
    │     └─────────────────────────────────────────────────────────────────────────┘           │
    │                                           │                                               │
    │                                           │ (repeat for blocks N-1 ... 1)                 │
    │                                           ▼                                               │
    │                           ┌───────────────────────────────┐                               │
    │                           │      Embeddings               │                               │
    │                           └───────────────────────────────┘                               │
    │                                                                                           │
    │  ═════════════════════════════════════════════════════════════════════════════════════   │
    │                                                                                           │
    │  GRADIENT FLOW PROPERTIES:                                                               │
    │                                                                                           │
    │  1. RESIDUAL CONNECTIONS: Direct gradient path through all layers                        │
    │     → Prevents vanishing gradients in deep networks                                      │
    │                                                                                           │
    │  2. DUAL GATING GRADIENTS:                                                               │
    │     - GLU_out receives gradients from: α mixing, input_gate, output_gate                │
    │     - Local branch receives gradients gated by (1-α) AND output_gate                    │
    │     - Input gate gradients flow to GLU_out via normalized input                         │
    │                                                                                           │
    │  3. SIGSOFTMAX GRADIENTS:                                                                │
    │     - sigsoftmax(x) = softmax(x + logsigmoid(x))                                        │
    │     - ∂sigsoftmax/∂x has bounded magnitude (avoids exploding gradients)                 │
    │                                                                                           │
    │  4. OSCILLATOR GRADIENTS:                                                                │
    │     - DLinOSS uses implicit Euler → stable backward pass                                 │
    │     - Log-parameterization → always positive frequencies/damping                         │
    │     - Damping ensures bounded state → bounded gradients                                  │
    │                                                                                           │
    │  5. LAYERNORM PLACEMENT:                                                                 │
    │     - Pre-norm before branches → normalized inputs to both paths                        │
    │     - Post-norm after residual → normalized outputs to next layer                       │
    │     → Stabilizes gradient magnitudes throughout network                                  │
    │                                                                                           │
    └──────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
TRAINING PIPELINE ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                              │
    │  PHASE 1: DATA PREPARATION                                                                   │
    │  ─────────────────────────────────────────────────────────────────────────────────────────── │
    │                                                                                              │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
    │  │ OntoNotes   │  │ CoNLL-2003  │  │  Few-NERD   │  │  WNUT-17    │  │  SciERC     │        │
    │  │   5.0       │  │             │  │             │  │             │  │  BioNLP     │        │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
    │         │                │                │                │                │               │
    │         └────────────────┴────────────────┴────────────────┴────────────────┘               │
    │                                           │                                                  │
    │                                           ▼                                                  │
    │                          ┌─────────────────────────────────────┐                            │
    │                          │     UNIFIED CONLL LOADER            │                            │
    │                          │                                     │                            │
    │                          │  • Label mapping to 9-type schema   │                            │
    │                          │  • BIO encoding validation          │                            │
    │                          │  • Cross-dataset deduplication      │                            │
    │                          └─────────────────┬───────────────────┘                            │
    │                                            │                                                 │
    │                                            ▼                                                 │
    │                          ┌─────────────────────────────────────┐                            │
    │                          │     TOKENIZATION + ALIGNMENT        │                            │
    │                          │                                     │                            │
    │                          │  Token: "playing"  → ["play", "ing"]│                            │
    │                          │  Label: B-EVENT    → [B-EVENT, I-EVENT]                         │
    │                          │                                     │                            │
    │                          │  Special tokens get label -100      │                            │
    │                          │  (ignored in loss)                  │                            │
    │                          └─────────────────┬───────────────────┘                            │
    │                                            │                                                 │
    │                                            ▼                                                 │
    │                          ┌─────────────────────────────────────┐                            │
    │                          │     DATA AUGMENTATION               │                            │
    │                          │                                     │                            │
    │                          │  • Entity replacement (same type)   │                            │
    │                          │  • Mention dropout (20%)            │                            │
    │                          │  • Context shuffling                │                            │
    │                          │  • Synonym replacement              │                            │
    │                          └─────────────────────────────────────┘                            │
    │                                                                                              │
    │  ═══════════════════════════════════════════════════════════════════════════════════════════ │
    │                                                                                              │
    │  PHASE 2: PRETRAINING (Optional - LLaDA-style masked diffusion)                             │
    │  ─────────────────────────────────────────────────────────────────────────────────────────── │
    │                                                                                              │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────┐    │
    │  │                                                                                     │    │
    │  │   Corpus: Wikipedia + BookCorpus + OpenWebText + PubMed + arXiv                    │    │
    │  │                                                                                     │    │
    │  │   Task: Predict masked tokens given noisy sequence + diffusion time t              │    │
    │  │                                                                                     │    │
    │  │   ┌─────────────────────────────────────────────────────────────────────────────┐  │    │
    │  │   │  Original:  "The quick brown fox jumps over the lazy dog"                   │  │    │
    │  │   │  Masked:    "The [M]   brown [M]  jumps [M]   the lazy [M]"  (t=0.4)        │  │    │
    │  │   │  Predict:   Reconstruct original tokens                                     │  │    │
    │  │   └─────────────────────────────────────────────────────────────────────────────┘  │    │
    │  │                                                                                     │    │
    │  │   Time embedding t conditions the model on noise level                             │    │
    │  │   (This is why TimeConditionedLayerNorm exists)                                    │    │
    │  │                                                                                     │    │
    │  │   Training: 500K steps, batch=32, effective=128 (4× grad accum)                   │    │
    │  │                                                                                     │    │
    │  └─────────────────────────────────────────────────────────────────────────────────────┘    │
    │                                                                                              │
    │  ═══════════════════════════════════════════════════════════════════════════════════════════ │
    │                                                                                              │
    │  PHASE 3: NER FINE-TUNING                                                                    │
    │  ─────────────────────────────────────────────────────────────────────────────────────────── │
    │                                                                                              │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────┐    │
    │  │                                                                                     │    │
    │  │   Load pretrained weights (or train from scratch)                                  │    │
    │  │                                                                                     │    │
    │  │   Fixed time embedding t=0.5 (no diffusion during NER)                             │    │
    │  │                                                                                     │    │
    │  │   ┌──────────────────────────────────────────────────────────────────────────┐     │    │
    │  │   │                                                                          │     │    │
    │  │   │   PRIMARY LOSS: Cross-entropy over valid tokens                         │     │    │
    │  │   │                                                                          │     │    │
    │  │   │   L_ce = -Σ log P(y_true | x) / N_valid                                  │     │    │
    │  │   │                                                                          │     │    │
    │  │   └──────────────────────────────────────────────────────────────────────────┘     │    │
    │  │                                                                                     │    │
    │  │   ┌──────────────────────────────────────────────────────────────────────────┐     │    │
    │  │   │                                                                          │     │    │
    │  │   │   AUXILIARY LOSS (optional): Boundary detection (20% weight)            │     │    │
    │  │   │                                                                          │     │    │
    │  │   │   Binary classification: is this token a B-* tag?                       │     │    │
    │  │   │   L_boundary = BCE(boundary_pred, boundary_true)                        │     │    │
    │  │   │                                                                          │     │    │
    │  │   └──────────────────────────────────────────────────────────────────────────┘     │    │
    │  │                                                                                     │    │
    │  │   ┌──────────────────────────────────────────────────────────────────────────┐     │    │
    │  │   │                                                                          │     │    │
    │  │   │   OPTIONAL: CRF Layer for BIO constraint enforcement                    │     │    │
    │  │   │                                                                          │     │    │
    │  │   │   Transition matrix A: P(tag_t | tag_{t-1})                             │     │    │
    │  │   │   Prevents invalid sequences like: O → I-PERSON                         │     │    │
    │  │   │                                                                          │     │    │
    │  │   │   Viterbi decoding at inference time                                    │     │    │
    │  │   │                                                                          │     │    │
    │  │   └──────────────────────────────────────────────────────────────────────────┘     │    │
    │  │                                                                                     │    │
    │  │   Training config:                                                                  │    │
    │  │     • 20 epochs, early stopping on validation F1                                   │    │
    │  │     • Learning rate: 1e-4 with cosine decay                                        │    │
    │  │     • Warmup: 10% of steps                                                         │    │
    │  │     • Dropout: 0.1                                                                 │    │
    │  │     • Batch size: 32                                                               │    │
    │  │                                                                                     │    │
    │  └─────────────────────────────────────────────────────────────────────────────────────┘    │
    │                                                                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
INFERENCE PIPELINE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                              │
    │   Raw Text                                                                                   │
    │   "Albert Einstein won the Nobel Prize in Physics in 1921."                                 │
    │        │                                                                                     │
    │        ▼                                                                                     │
    │   ┌────────────────────────────────────────────────────────────────────────────────────┐    │
    │   │  TOKENIZATION                                                                      │    │
    │   │  [CLS] Albert Einstein won the Nobel Prize in Physics in 1921 . [SEP]             │    │
    │   │  [101, 7654, 8256, 1839, 1996, 9680, 3396, 1999, 5767, 1999, 5765, 1012, 102]     │    │
    │   └────────────────────────────────────────────────────────────────────────────────────┘    │
    │        │                                                                                     │
    │        ▼                                                                                     │
    │   ┌────────────────────────────────────────────────────────────────────────────────────┐    │
    │   │  MODEL FORWARD PASS                                                                │    │
    │   │  logits = model(token_ids, params, state)                                         │    │
    │   │  logits: (19, 13, 1)  # (num_labels, seq_len, batch)                              │    │
    │   └────────────────────────────────────────────────────────────────────────────────────┘    │
    │        │                                                                                     │
    │        ▼                                                                                     │
    │   ┌────────────────────────────────────────────────────────────────────────────────────┐    │
    │   │  DECODING                                                                          │    │
    │   │                                                                                    │    │
    │   │  Option A: Greedy argmax                                                          │    │
    │   │    labels = argmax(logits, dim=1)                                                 │    │
    │   │                                                                                    │    │
    │   │  Option B: Viterbi (if CRF trained)                                               │    │
    │   │    labels = viterbi_decode(logits, transition_matrix)                             │    │
    │   │                                                                                    │    │
    │   │  Predictions:                                                                      │    │
    │   │  [CLS] Albert    Einstein won the Nobel      Prize    in Physics   in 1921      . │    │
    │   │   O    B-PERSON  I-PERSON  O   O   B-EVENT   I-EVENT  O  B-DOMAIN  O  B-MEASURE O │    │
    │   └────────────────────────────────────────────────────────────────────────────────────┘    │
    │        │                                                                                     │
    │        ▼                                                                                     │
    │   ┌────────────────────────────────────────────────────────────────────────────────────┐    │
    │   │  ENTITY EXTRACTION (extract_entities function)                                     │    │
    │   │                                                                                    │    │
    │   │  Parse BIO tags to extract spans:                                                 │    │
    │   │                                                                                    │    │
    │   │  entities = [                                                                     │    │
    │   │    (text: "Albert Einstein", label: "PERSON",  start: 0,  end: 15),              │    │
    │   │    (text: "Nobel Prize",     label: "EVENT",   start: 24, end: 35),              │    │
    │   │    (text: "Physics",         label: "DOMAIN",  start: 39, end: 46),              │    │
    │   │    (text: "1921",            label: "MEASURE", start: 50, end: 54),              │    │
    │   │  ]                                                                                │    │
    │   │                                                                                    │    │
    │   └────────────────────────────────────────────────────────────────────────────────────┘    │
    │                                                                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
DESIGN RATIONALE & ABSTRACTIONS
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                              │
    │  ABSTRACTION HIERARCHY:                                                                      │
    │  ═══════════════════════════════════════════════════════════════════════════════════════════ │
    │                                                                                              │
    │  Level 4: OssammaNER                                                                         │
    │           │                                                                                  │
    │           │  "A complete NER system that combines embeddings,                               │
    │           │   processing blocks, and classification"                                        │
    │           │                                                                                  │
    │           ├── Embeddings (token + position + time)                                          │
    │           ├── N × OssammaNERBlock                                                           │
    │           └── ClassificationHead                                                            │
    │                                                                                              │
    │  Level 3: OssammaNERBlock                                                                    │
    │           │                                                                                  │
    │           │  "A processing unit that fuses global context with                              │
    │           │   local precision via bidirectional gating"                                     │
    │           │                                                                                  │
    │           ├── TimeConditionedLayerNorm                                                      │
    │           ├── GLU-Global Branch (LinearAttention + DLinOSS + GLU gate)                     │
    │           ├── Dual Gates (Input + Output)                                                   │
    │           ├── Local-Sharp Branch (SWAttention)                                              │
    │           └── Adaptive Mixing + Residual                                                    │
    │                                                                                              │
    │  Level 2: Core Components                                                                    │
    │           │                                                                                  │
    │           ├── LinearAttention: "Global context via O(n) attention"                         │
    │           ├── DLinOSS: "Temporal dynamics via physics-based oscillators"                   │
    │           ├── SWAttention: "Local precision via windowed attention"                        │
    │           └── TimeConditionedLayerNorm: "Diffusion-aware normalization"                    │
    │                                                                                              │
    │  Level 1: Lux Primitives                                                                     │
    │           │                                                                                  │
    │           ├── Dense, LayerNorm, Dropout, Embedding                                          │
    │           └── Activation functions (sigmoid, softplus, sigsoftmax)                          │
    │                                                                                              │
    │  ═══════════════════════════════════════════════════════════════════════════════════════════ │
    │                                                                                              │
    │  WHY THIS ARCHITECTURE?                                                                      │
    │  ═══════════════════════════════════════════════════════════════════════════════════════════ │
    │                                                                                              │
    │  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  PROBLEM: NER requires BOTH global understanding AND local boundary precision         │ │
    │  │                                                                                        │ │
    │  │  Example: "New York Times reported that Apple Inc. released iPhone 15"               │ │
    │  │                                                                                        │ │
    │  │  - "New York" could be place OR part of "New York Times" (organization)              │ │
    │  │  - Need global context to disambiguate                                                │ │
    │  │  - But "Times" boundary is local (space before "reported")                           │ │
    │  └────────────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                              │
    │  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  SOLUTION: Dual-gated parallel branches                                               │ │
    │  │                                                                                        │ │
    │  │  1. GLU-Global branch captures document-level semantics:                              │ │
    │  │     - LinearAttention: What topics/entities are in the document?                      │ │
    │  │     - DLinOSS: How do concepts flow/decay through the sequence?                      │ │
    │  │     - GLU gate: Which global features are relevant at each position?                 │ │
    │  │                                                                                        │ │
    │  │  2. Local-Sharp branch captures boundary patterns:                                    │ │
    │  │     - SWAttention: What's the local syntactic structure?                             │ │
    │  │     - Window size 5: Captures typical entity span length                             │ │
    │  │     - sigsoftmax: Sharp attention for precise boundaries                             │ │
    │  │                                                                                        │ │
    │  │  3. Dual gating allows Global to guide Local:                                        │ │
    │  │     - Input gate: "Don't look at these words, they're irrelevant"                   │ │
    │  │     - Output gate: "Here's context you missed, add it back"                         │ │
    │  │     - α mixing: Per-sample blend based on input complexity                           │ │
    │  └────────────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                              │
    │  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  WHY OSCILLATORS (DLinOSS)?                                                           │ │
    │  │                                                                                        │ │
    │  │  Traditional RNNs: Exponential decay/growth, hard to control                         │ │
    │  │  Transformers: All-to-all attention, computationally expensive                       │ │
    │  │                                                                                        │ │
    │  │  Oscillators offer:                                                                   │ │
    │  │  - Natural periodicity: Language has rhythm (sentences, paragraphs)                  │ │
    │  │  - Controllable decay: Damping determines how far context persists                   │ │
    │  │  - Efficient O(n): Linear scan, constant memory                                      │ │
    │  │  - Interpretable: Frequencies correspond to different timescales                     │ │
    │  │                                                                                        │ │
    │  │  For NER: Entity mentions "resonate" across the document                             │ │
    │  │  (multiple mentions of same entity reinforce each other)                              │ │
    │  └────────────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                              │
    │  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  WHY 9 ENTITY TYPES?                                                                  │ │
    │  │                                                                                        │ │
    │  │  Optimized for RAG (Retrieval-Augmented Generation):                                  │ │
    │  │                                                                                        │ │
    │  │  - PERSON, AGENCY, PLACE: Core knowledge graph nodes                                 │ │
    │  │  - ORGANISM: Scientific/biomedical documents                                          │ │
    │  │  - EVENT: Temporal reasoning and news                                                 │ │
    │  │  - INSTRUMENT, WORK: Technical documentation                                          │ │
    │  │  - DOMAIN: Academic papers, categorization                                            │ │
    │  │  - MEASURE: Quantitative queries, comparisons                                         │ │
    │  │                                                                                        │ │
    │  │  Each type enables different retrieval strategies:                                    │ │
    │  │  "Find papers by [PERSON] about [DOMAIN] from [MEASURE:year]"                        │ │
    │  └────────────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
FILE REFERENCE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                              │
    │  CONFIGURATION FILES:                                                                        │
    │  configs/ner_minimal.toml      Smoke testing config (~1.5M params)                          │
    │  configs/ner_dev.toml          Development config (~35M params)                             │
    │  configs/ner_production.toml   Production config (~150M params)                             │
    │                                                                                              │
    │  SOURCE FILES:                                                                               │
    │  src/NER.jl                    OssammaNER model, config loading, loss functions             │
    │    - NERConfig struct          Configuration with all hyperparameters                       │
    │    - load_ner_config()         Load config from TOML file                                   │
    │    - load_training_config()    Load training hyperparameters from TOML                      │
    │    - estimate_parameters()     Estimate model parameter count                               │
    │    - print_config_summary()    Display config with param estimate                           │
    │                                                                                              │
    │  src/Ossamma.jl                OssammaNERBlock (dual gating architecture)                   │
    │  src/Attention.jl              SWAttention (sliding window)                                 │
    │  src/linearAttention.jl        LinearAttentionLayer                                         │
    │  src/Dlinoss.jl                DLinOSS oscillator SSM                                       │
    │                                                                                              │
    │  DOCUMENTATION:                                                                              │
    │  docs/NER_TRAINING_PLAN.md     Complete training pipeline specification                     │
    │  docs/OSSAMMA_NER_ARCHITECTURE.md  This file                                                │
    │                                                                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
```
