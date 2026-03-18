# Reasoning Diffusion Design — PRIME + ReasoningDrafter

## Summary

Phase 3a of the reasoning drafter pipeline should use the **LLaDA PRIME diffusion objective**
instead of next-token prediction. The drafter learns to unmask reasoning traces through
iterative deliberation, not predict words left-to-right. The WavePDE dynamics naturally
implement deliberation within each denoising step, while the PRIME partial masking
schedule orchestrates the progressive commitment across steps.

## Motivation

### Why diffusion, not autoregressive

Autoregressive generation forces serial commitment — decide token 1 before token 2.
This is wrong for reasoning, where you often know the conclusion before the intermediate
steps, or know step 3 before step 2.

PRIME diffusion allows **confidence-ordered commitment**: decide the easy parts first
regardless of position, then resolve the hard parts informed by the easy ones. Each
denoising step runs the full ReasoningDrafter backbone, so the wave dynamics deliberate
over a progressively-resolved reasoning trace.

### Why PRIME partial masking, not binary

Vanilla diffusion: tokens are either 100% masked or 100% revealed. PRIME: tokens are
partially masked at the sub-token level, creating a continuum of certainty.

For the WavePDE, this means:
- Positions at 90% mask: very uncertain → low damping, keep oscillating between alternatives
- Positions at 50% mask: partially resolved → moderate damping, narrowing deliberation
- Positions at 10% mask: nearly committed → high damping, almost fixed boundary condition

The VQ codebook in RuleConditionedWavePDE sees hidden states that implicitly encode the
mask pattern (masked sub-tokens carry the mask embedding, revealed sub-tokens carry content).
Codebook entries will emergently cluster by:
- **Mask density** — what phase of deliberation are we in
- **Semantic situation** — what type of reasoning is needed here

The 512-entry codebook gives room for this factorization without us imposing it.

### Why this fits the WavePDE decision story

The wave equation implements deliberation: propagation explores alternatives, interference
selects consistent options, damping commits. In the diffusion setting:

1. **Early steps (high mask ratio):** sparse signal, waves propagate far through uncertain
   medium. The model makes coarse structural decisions — "this is a 3-step proof,"
   "the answer is negative." Low damping everywhere.

2. **Middle steps:** partially resolved trace acts as boundary conditions for wave
   propagation. Wavefronts reflect off committed tokens, creating interference patterns
   that resolve the remaining positions. Codebook selects mid-deliberation strategies.

3. **Late steps (low mask ratio):** most positions fixed. Wave dynamics are highly
   constrained. Remaining decisions are forced by reflected wavefronts. High damping
   near committed tokens.

The VQ codebook learns to modulate c(x) and γ(x) differently at each phase — effectively
learning **how to deliberate at different levels of certainty**.

## Architecture

### Block compatibility

`LLaDAModel` currently uses `SwammaBlock` as its block type. `ReasoningDrafterBlock` has
the same interface: `(hidden, time_emb) → (hidden, state)`. The model should accept
either block type.

Proposed change: parameterize `LLaDAModel` on block type, or create a thin
`ReasoningLLaDA` constructor that wires `ReasoningDrafterBlock`s into `LLaDAModel`.

```
ReasoningLLaDA = LLaDAModel(
    blocks = [ReasoningDrafterBlock(...) for _ in 1:num_layers],
    # PRIME sub-token machinery from LLaDA
    # Granite tokenizer for proper subword tokens
)
```

### What LLaDA provides that ReasoningDrafter needs

| Component | Source | Purpose |
|---|---|---|
| SubtokenEmbeddings | LLaDA | PRIME sub-token decomposition |
| prime_code_table | LLaDA | Token ↔ sub-token mapping |
| apply_subtoken_mask | LLaDA | Partial masking at sub-token level |
| sample_mask_ratio | LLaDA | Diffusion schedule sampling |
| diffusion_loss | Training.jl | Sample t, mask, forward, CE on masked |
| unmask_subtoken_step | LLaDA | Confidence-based progressive unmasking |
| generate | LLaDA | Iterative denoising loop |
| TimeEmbedding | LLaDA | Diffusion timestep conditioning |

### What ReasoningDrafterBlock provides that LLaDA doesn't have

| Component | Source | Purpose |
|---|---|---|
| RuleConditionedWavePDE | ReasoningDrafter | VQ situation → modulated wave dynamics |
| AlgebraicCircuitLayer | ReasoningDrafter | SPN consistency verification |
| Adapter headers | ReasoningDrafter | Domain transfer without full retrain |
| EMA codebook | RuleConditionedWavePDE | Non-gradient codebook learning |

### Forward pass (per denoising step)

```
input: partially masked sub-token sequence + mask_ratio t

1. SubtokenEmbeddings(masked_subtokens)     → (d, seq, batch)
2. + PositionEmbedding                      → (d, seq, batch)
3. TimeEmbedding(t)                         → (t_dim, batch)  [mask ratio as time input]
4. For each ReasoningDrafterBlock:
   a. RMSNorm
   b. RuleConditionedWavePDE:
      - Encode hidden → VQ code (situation at this mask density)
      - Retrieve rule → modulate c(x), γ(x)
      - Leapfrog PDE integration (deliberation)
      - Gate + residual
   c. GLU(LinearAttention ⊙ sigmoid(WavePDE gate))
   d. AlgebraicCircuit (consistency check)
   e. LayerNorm
5. OutputHead → logits over vocab
6. diffusion_loss: CE on masked positions only
```

Key: the **mask_ratio t feeds into the TimeEmbedding**, which conditions the
time-conditioned LayerNorm at each block. The blocks know what diffusion step they're
in, which modulates normalization statistics. Combined with the VQ codebook seeing
the mask pattern in hidden states, the model has two channels of mask-density
information: explicit (time embedding) and implicit (hidden state content).

## Training

### Objective

```
L = E_t [ E_mask [ CE(predict masked sub-tokens | partially masked input, t) ] ]
```

where t ~ sample_mask_ratio(schedule) and mask ~ Bernoulli(t) at sub-token level.

This is exactly `diffusion_loss` from `Training.jl`, adapted to use `ReasoningDrafterBlock`s
instead of `SwammaBlock`s.

### Freeze strategy (same as current Phase 3a plan)

| Component | Status |
|---|---|
| WavePDE backbone (speed, damping, norms, gates) | **FROZEN** from Phase 1 |
| Adapter headers (Encoder, RuleBank, Circuit, GateBias) | **Train 1x** |
| VQ Codebook | **Train 0.1x** + EMA |
| LinearAttention | **Train 0.1x** |
| Circuit OutputWeight | **Train 0.1x** |
| TokenEmbedding / SubtokenEmbeddings | **Train 1x** (reinit for Granite vocab) |
| OutputHead | **Train 1x** (reinit for Granite vocab) |

### Tokenizer

**Granite tokenizer** (via HFTokenizer, already in codebase). Required because:
- PRIME sub-token decomposition operates on proper subword tokens
- Phase 3b distillation matches Granite's token space
- Must be consistent from Phase 3a onward

### Data

Reasoning datasets already downloaded (12,111 examples):
- GSM8K (7,473) — arithmetic chains: premise → steps → answer
- ReClor (4,638) — argumentation: context → question → reasoning → conclusion

Format: each example is a complete reasoning trace tokenized with Granite tokenizer,
fed whole into diffusion_loss. No left-to-right constraint, no teacher forcing on
token order. The model learns to reconstruct reasoning traces from partial information.

### Schedule

- Mask ratio schedule: cosine (more samples near t=0 and t=1, which are the
  "almost done" and "just starting" phases where the codebook can learn the
  most about deliberation strategy)
- Epochs: 10 over reasoning data
- Batch size: 32 (same memory profile as Phase 1)
- Apply `apply_reasoning_drafter_ema_codebook!` after each gradient step

## Phase 3b: Verifier-Judged Unmasking

With the diffusion framing, Phase 3b changes from "match Granite's token distribution"
to "learn to unmask in a way Granite approves."

1. **Drafter** starts from fully masked reasoning trace
2. Runs a few denoising steps → produces partially resolved trace
3. **Granite** evaluates: for each unmasked position, does the token match what
   Granite would predict given the full context?
4. **Loss**: KL divergence between drafter's logits and Granite's logits, but only
   on positions the drafter chose to unmask (high confidence positions)

This trains the drafter to be **calibrated** — unmask only when confident, and be
correct when it does unmask. The acceptance rate metric becomes: "what fraction of
the drafter's unmasking decisions does Granite endorse?"

## Speculative Execution (TiDAR Integration)

The inference loop becomes:

```
1. Start with masked reasoning trace (or partially resolved from prior round)
2. Drafter runs K denoising steps → proposes unmasked positions
3. Verifier (Granite) checks proposed unmaskings in parallel
4. Accept correct unmaskings, re-mask incorrect ones
5. Repeat until fully resolved
```

This is iterative speculative execution with rollback. The drafter proposes its most
confident decisions (confidence-ordered unmasking). The verifier filters. The drafter
refines. The wave dynamics make the drafter good at calibrating confidence — high
damping at a position means "I've deliberated enough here, I'm confident."

## Implementation Steps

### Step 1: Make LLaDAModel accept ReasoningDrafterBlocks

- Parameterize block type in `LLaDAModel` constructor
- Or create `ReasoningLLaDA` thin wrapper
- Verify forward pass works with ReasoningDrafterBlock interface

### Step 2: Wire Phase 2 surgery into LLaDA

- Load Phase 1 chess checkpoint (ReasoningDrafter params)
- Initialize LLaDA PRIME machinery (SubtokenEmbeddings, code_table) for Granite vocab
- Freeze backbone, add adapters
- Save as Phase 2 surgery checkpoint

### Step 3: Phase 3a training script

- New script: `scripts/train_reasoning_diffusion.jl`
- Uses `diffusion_loss` with cosine mask schedule
- Granite tokenizer for input
- Freeze strategy as above
- EMA codebook after each step
- Checkpoint overwrite every 300 steps

### Step 4: Evaluate codebook emergence

- After Phase 3a, inspect what the 512 VQ codes learned
- Cluster analysis: do codes separate by mask density? By reasoning type?
- Visualize c(x) and γ(x) modulation per code
- This validates (or refutes) the hypothesis that the codebook learns
  mask-density-aware deliberation strategies

### Step 5: Phase 3b verifier distillation

- Load Granite via NativeTeacherLM
- Run Granite on reasoning prompts, collect logits
- Train drafter with verifier-judged unmasking loss
- Measure acceptance rate

## Open Questions

- **Sub-token length for Granite:** the PRIME sub-token decomposition factor
  affects how granular the partial masking is. Granite vocab is 49,160 tokens.
  Current PRIME uses base-256 decomposition (2-3 sub-tokens per token). Is this
  the right granularity for reasoning?

- **Integration steps vs denoising steps:** currently Phase 1 uses 4 PDE integration
  steps per forward pass. In diffusion with e.g. 20 denoising steps, the total
  deliberation is 4×20 = 80 integration steps over the full generation. Should
  integration steps increase in later denoising steps (more deliberation when
  close to commitment)?

- **Codebook size:** 512 entries may be too few if the codebook must jointly represent
  mask density + reasoning type + difficulty. Consider 1024 or 2048 for Phase 3a,
  with warmup from the 512-entry Phase 1 codebook.

- **Mask schedule interaction with curriculum:** should early training use uniform
  schedule (broad exposure) and later training use cosine (focus on edge cases)?
  Or fixed cosine throughout?

## References

- Vejendla (2025). [Wave-PDE Nets](https://arxiv.org/abs/2510.04304). PRICAI 2025.
- LLaDA / PRIME partial masking: implemented in `src/LLaDA.jl`, `src/Training.jl`
- Granite tokenizer: `src/HFTokenizer.jl`, target `ibm-granite/granite-4.0-micro`
- ReasoningDrafter architecture: `src/ReasoningDrafter.jl`
- RuleConditionedWavePDE: `src/RuleConditionedWavePDE.jl`
- Phase 1 checkpoint: `checkpoints/reasoning_drafter/phase1/`
- Existing freeze strategy: `docs/SPARK_REASONING_RUNBOOK.md`
