# Reasoning Drafter — TODO

Target verifier: **Granite** (via NativeTeacherLM, already in codebase)

---

## Immediate Stabilization Blockers

These items block Phase 3a from being a reliable training path and should be
completed before more reasoning runs are scheduled.

- [x] Make `ReasoningDrafter` differentiable under Zygote
  - Replaced the mutable `Vector{Any}` block-state accumulation in `src/ReasoningDrafter.jl`
  - Kept state reconstruction purely functional so `Zygote.withgradient` now works through the drafter
  - Added a regression test that computes a gradient through a small `ReasoningDrafter`
- [x] Fix overlength sequence handling in `ReasoningDrafter`
  - Chosen policy: fail fast with `ArgumentError` when `seq_len > config.max_sequence_length`
  - Aligned `scripts/train_reasoning_language.jl` dataset `max_seq_length` with `config.max_sequence_length` via clamping
  - Added a regression test covering `seq_len > config.max_sequence_length`
- [x] Wire `RuleConditionedWavePDE` codebook learning into training
  - Added a reusable `apply_reasoning_drafter_ema_codebook!` helper for post-step EMA application
  - Chosen update rule: hybrid, with normal optimizer updates plus explicit EMA codebook refresh after each training step
  - Wired EMA application into Phase 1, Phase 3a, and Phase 3b training loops
  - Preserved Phase 1 drafter state and checkpointed it so EMA statistics survive resume
  - Added regression tests showing active codebook values change after EMA application
- [x] Add a minimal Phase 3a trainability smoke test
  - Added `test/test_reasoning_trainability.jl`
  - Runs `2` optimizer steps on a tiny synthetic reasoning batch
  - Verified: forward pass succeeds, gradient step succeeds, loss is finite, parameters move, and checkpoint save succeeds
  - Added the smoke gate to `test/runtests.jl`
  - Use this smoke test as the gate before launching larger reasoning jobs

## Phase 1: Chess Pre-Training

- [ ] Wait for Lichess eval DB download to finish (~19GB zst → ~30GB jsonl)
- [ ] Decompress: `zstd -d data/chess/lichess_db_eval.jsonl.zst`
- [ ] Create smoke sample: `head -100000 data/chess/lichess_db_eval.jsonl > data/chess/sample_100k.jsonl`
- [ ] Smoke test training: `julia --project=. scripts/train_chess_reasoning.jl --data data/chess/sample_100k.jsonl --steps 200`
- [ ] Monitor: codebook utilization, move accuracy, eval MSE
- [ ] Full training run (~10M positions, 10 epochs)
- [ ] Save Phase 1 checkpoint: `checkpoints/chess_reasoning/phase1_final.jld2`
- [ ] Validate: codebook diversity (how many of 512 codes are used?), wave dynamics diversity

## Phase 2: Transfer Surgery

- [ ] Load Phase 1 checkpoint
- [ ] Freeze reasoning backbone:
  - SpeedModWeight, DampingModWeight
  - log_wave_speed, log_damping
  - GluProjection, WavePDE gate, all Norms
  - SumLogWeights, ComposeLogWeights
  - Encoder, RuleBank, LeafWeights, Gate weights
- [ ] Add adapter headers (`use_adapters=true`):
  - EncoderHeader (cd×cd), RuleBankHeader (cd×cd), GateBiasShift
  - CircuitLeafHeader (dim×dim), CircuitGateBiasShift
- [ ] Reinitialize TokenEmbedding and OutputHead for Granite vocab (49160 tokens)
- [ ] Verify: forward pass produces finite logits with new vocab
- [ ] Save surgery checkpoint: `checkpoints/chess_reasoning/phase2_surgery.jld2`

## Phase 3a: Language Fine-Tuning

Data ready in `data/reasoning/`:
- [x] GSM8K — 7,473 arithmetic reasoning chains
- [x] ReClor — 4,638 argumentation examples
- [x] LogicNLI — 16,000 logical entailment
- [x] ARC-Challenge — 1,119 science reasoning
- [x] bAbI-deduction — 1,000 syllogistic
- [x] bAbI-induction — 1,000 inductive
- **Total: 31,230 examples**

- [ ] Build Phase 3a training script (`scripts/train_reasoning_language.jl`)
  - Char-level tokenizer (vocab=132) for reasoning datasets
  - OR: use Granite tokenizer via HFTokenizer for proper subword tokens
  - Adapter params at 1x LR, codebook/leaves at 0.1x LR, embeddings at 1x LR
  - All backbone params frozen
- [ ] Decide tokenizer: char-level (simple, already built) vs Granite tokenizer (better transfer to Phase 3b)
  - **Recommendation: Granite tokenizer** — if Phase 3b uses Granite, the embeddings need to match
- [ ] Train on reasoning mix (equal weight per dataset, ~10 epochs)
- [ ] Monitor: codebook recluster rate, adapter weight norms, reasoning task accuracy
- [ ] Save Phase 3a checkpoint

## Phase 3b: Granite Distillation

- [ ] Load Granite model via `load_granite_model()` (NativeTeacherLM)
  - Target: `ibm-granite/granite-4.0-micro` or similar small Granite
  - Requires: HF model download, safetensors + PyCall
- [ ] Generate reasoning traces: run Granite on reasoning prompts, collect (prompt, logits) pairs
  - Use GSM8K + ReClor + LogicNLI prompts
  - Save verifier logits for each position
- [ ] Build distillation training script (`scripts/distill_reasoning_drafter.jl`)
  - Loss: KL(drafter_logits || granite_logits) on reasoning traces
  - Same freeze strategy as 3a (adapters + codebook/leaves trainable, backbone frozen)
  - Granite tokenizer required (must match verifier's token space)
- [ ] Train until acceptance rate plateaus
- [ ] Measure acceptance rate on held-out reasoning prompts

## Evaluation

- [ ] Acceptance rate: draft K tokens, verify with Granite, measure accept/reject ratio
- [ ] Compare vs random-init drafter (same Phase 3b distillation, no chess pre-training)
- [ ] Compare vs plain SwammaDrafter (no RuleConditionedWavePDE, no AlgebraicCircuit)
- [ ] Ablation: freeze backbone vs thaw everything during Phase 3b
- [ ] Per-dataset breakdown: acceptance rate on GSM8K vs ReClor vs LogicNLI

## Integration with TiDAR

- [ ] Wire ReasoningDrafter into existing `draft_tokens` / `verify_and_accept` pipeline
- [ ] Benchmark: tokens/second with reasoning drafter vs existing SwammaDrafter
- [ ] End-to-end: prompt → draft → verify → accept/reject → output

## Open Questions

- Granite vocab size (49160) means embeddings dominate param count (~25M). Consider tied embeddings.
- Char-level tokenizer for Phase 3a loses information vs Granite subword. May need to go straight to Granite tokenizer.
- Chess board is 64 squares but Granite sequences can be 1024+ tokens. The `max_sequence_length` needs to change between Phase 1 (64) and Phase 3 (512+).
- EMA codebook update: call `apply_rc_ema_codebook!` after each gradient step in all phases.
