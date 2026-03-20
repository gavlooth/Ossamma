# Reasoning Drafter — NVIDIA Spark GB10 Runbook

Target: NVIDIA Spark GB10, 130GB unified memory, CUDA 13.0, aarch64

## Quick Start

```bash
# 1. Setup environment
cd Swamma
julia --project=. -e 'using Pkg; Pkg.instantiate()'
pip install datasets   # for reasoning dataset download

# 2. Fast sanity check
./scripts/launch_reasoning_pipeline.sh --smoke

# 3. Practical bounded validation path
./scripts/launch_reasoning_pipeline.sh --bounded-medium

# 4. Full pipeline
./scripts/launch_reasoning_pipeline.sh --all
```

## Recommended Modes

Use the pipeline modes according to what you are trying to validate:

```bash
# Minimal end-to-end smoke
./scripts/launch_reasoning_pipeline.sh --smoke

# Recommended day-to-day bounded validation
./scripts/launch_reasoning_pipeline.sh --bounded-medium

# Phase-specific bounded validation
./scripts/launch_reasoning_pipeline.sh --phase 3a --bounded-medium
./scripts/launch_reasoning_pipeline.sh --phase 3b --bounded-medium

# Full long-run pipeline
./scripts/launch_reasoning_pipeline.sh --all
```

Current bounded-medium defaults:

| Mode | Default checkpoint root | Profile |
|---|---|---|
| `--smoke` | `checkpoints/reasoning_drafter_smoke/` | tiny sanity path |
| `--bounded-medium` | `checkpoints/reasoning_drafter_medium/` | practical bounded validation |
| `--all` | `checkpoints/reasoning_drafter/` | full pipeline |

Bounded-medium is the best current default for local validation on this machine:

- Phase 1 uses the compact smoke-compatible model and runs `3` optimizer steps.
- Phase 2 performs surgery into the bounded `132`-token vocabulary.
- Phase 3a runs `64` reasoning examples at `batch_size=2`, `max_steps=40`.
- Phase 3b runs the same `64`-example bounded slice at `batch_size=2`, `max_steps=40`.

## Phase-by-Phase

### Phase 0: Data Preparation

The pipeline downloads data automatically, but you can do it manually:

```bash
# Chess data (~19GB compressed, ~30GB uncompressed)
# Takes 1-2 hours depending on connection
./scripts/download_lichess_evals.sh

# Reasoning datasets (~1GB total, downloads from HuggingFace)
# Requires: pip install datasets
./scripts/download_reasoning_datasets.sh

# Datasets downloaded to data/reasoning/:
#   LogicNLI       — 16,000 logical entailment (premise→hypothesis)
#   GSM8K          —  7,473 arithmetic chain-of-thought
#   ReClor         —  4,638 argumentation (law exam reasoning)
#   ARC-Challenge  —  1,119 science reasoning with multi-hop
#   bAbI-deduction —  1,000 syllogistic deduction
#   bAbI-induction —  1,000 inductive reasoning
#   Total: 31,230 examples
```

### Phase 1: Chess Pre-Training

Learns constrained reasoning from Stockfish evaluations.

```bash
julia --project=. scripts/train_chess_reasoning.jl \
  --data data/chess/lichess_db_eval.jsonl \
  --max-positions 10000000 \
  --checkpoint-dir checkpoints/reasoning_drafter/phase1

# Expected: ~18M params, batch_size=64
# Monitor: move accuracy, eval MSE
# Output: checkpoints/reasoning_drafter/phase1/best.jld2
```

Bounded-medium Phase 1 is intentionally much smaller and uses the smoke-compatible model shape:

```bash
./scripts/launch_reasoning_pipeline.sh --bounded-medium

# Phase 1 portion:
#   data/chess/smoke.jsonl
#   max_positions=128
#   max_steps=3
#   compact 64-dim / 2-layer drafter
```

### Phase 2: Transfer Surgery

Freezes reasoning backbone, adds adapters, swaps vocab for Granite.

```bash
julia --project=. scripts/transfer_surgery.jl \
  --input checkpoints/reasoning_drafter/phase1/best.jld2 \
  --output checkpoints/reasoning_drafter/phase2/surgery.jld2 \
  --target-vocab 49160

# This is fast (no training, just checkpoint manipulation)
# Output: checkpoints/reasoning_drafter/phase2/surgery.jld2
```

Bounded-medium uses `--target-vocab 132` so the medium Phase 3a/3b validation path stays lightweight and internally compatible.

### Phase 3a: Language Fine-Tuning

Trains adapters on reasoning datasets while backbone stays frozen.

```bash
julia --project=. scripts/train_reasoning_language.jl \
  --checkpoint checkpoints/reasoning_drafter/phase2/surgery.jld2 \
  --data-dir data/reasoning \
  --output-dir checkpoints/reasoning_drafter/phase3a \
  --epochs 10

# 31,230 examples, batch_size=32 → ~976 steps/epoch → ~9,760 total steps
# Only ~650K params are trainable (adapters + thawed components)
# Monitor: next-token loss on reasoning text
```

Recommended bounded-medium Phase 3a:

```bash
./scripts/launch_reasoning_pipeline.sh --phase 3a --bounded-medium

# Uses:
#   64 examples total
#   batch_size=2
#   max_seq_length=64
#   max_steps=40
#   checkpoint_every=4
#
# Current measured profile on this GB10 host:
#   best_loss ≈ 4.1658
#   wall time ≈ 106s for 40 steps
#   peak process GPU memory ≈ 348 MiB
```

### Phase 3b: Granite Distillation

Matches drafter to Granite's output distribution.

```bash
julia --project=. scripts/distill_granite.jl \
  --drafter-checkpoint checkpoints/reasoning_drafter/phase3a/best.jld2 \
  --granite-model ibm-granite/granite-4.0-micro \
  --data-dir data/reasoning \
  --output-dir checkpoints/reasoning_drafter/phase3b \
  --epochs 5

# Requires: pip install safetensors torch (for Granite weight loading)
# Granite model downloaded automatically from HuggingFace
# Monitor: KL divergence, decreasing = good
```

Recommended bounded-medium Phase 3b:

```bash
./scripts/launch_reasoning_pipeline.sh --phase 3b --bounded-medium

# Uses:
#   64 examples total
#   batch_size=2
#   max_seq_length=64
#   max_steps=40
#   checkpoint_every=4
#
# Current measured profile on this GB10 host:
#   best_loss ≈ 4.8578 when run phase-only from the smoke-root Phase 3a checkpoint
#   peak process GPU memory ≈ 16.3 GiB
```

## Architecture

```
ReasoningDrafterBlock (per layer):
  RMSNorm
    → RuleConditionedWavePDE (VQ situation → modulated wave dynamics)
    → GLU(LinAttn content ⊙ sigmoid(WavePDE gate))
    → Residual
    → AlgebraicCircuit (SPN consistency check)
    → LayerNorm
```

### Freeze Strategy

| Component | Phase 1 | Phase 3a/3b |
|---|---|---|
| SpeedModWeight, DampingModWeight | Train | **FROZEN** |
| log_wave_speed, log_damping | Train | **FROZEN** |
| GluProjection, WavePDE gate | Train | **FROZEN** |
| Norms | Train | **FROZEN** |
| SumLogWeights, ComposeLogWeights | Train | **FROZEN** |
| Encoder, RuleBank | Train | **FROZEN** |
| LeafWeights | Train | **FROZEN** |
| Gate weights | Train | **FROZEN** |
| *EncoderHeader* | N/A | **Train 1x** |
| *RuleBankHeader* | N/A | **Train 1x** |
| *CircuitLeafHeader* | N/A | **Train 1x** |
| *GateBiasShifts* | N/A | **Train 1x** |
| Codebook | Train | Train 0.1x |
| LinearAttention | Train | Train 0.1x |
| Circuit OutputWeight | Train | Train 0.1x |
| TokenEmbedding | Train | **Train 1x** (reinit) |
| OutputHead | Train | **Train 1x** (reinit) |

## CUDA/GPU Rules

- **NEVER** use `try/catch` inside training loops (CUDA.jl #2197)
- **NEVER** use `@info "msg $var"` in loops (implicit try/catch)
- Use `println()` for all logging in training loops
- Set `grads = nothing` after `Optimisers.update` to free AD tape
- Use `GC.gc(false)` before gradient passes (not `GC.gc(true)` — too slow per step)

## Troubleshooting

```bash
# Check GPU
julia --project=. -e 'using CUDA; println(CUDA.versioninfo())'

# Test model loads
julia --project=. -e 'using Swamma; using Swamma.ReasoningDrafterMod; println("OK")'

# Test Granite loads (requires PyCall + safetensors)
julia --project=. -e '
using Swamma.NativeTeacherLM
config = granite_config_from_hf("ibm-granite/granite-4.0-micro")
println(config)
'
```

## Disk Space Requirements

| Data | Size |
|---|---|
| Lichess eval DB (compressed) | ~19 GB |
| Lichess eval DB (uncompressed) | ~30 GB |
| Reasoning datasets | ~500 MB |
| Granite model weights | ~2-8 GB |
| Checkpoints (all phases) | ~2 GB |
| **Total** | **~55 GB** |
