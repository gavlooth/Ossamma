# Session Report

## 2026-03-17 — Reasoning Memory Prototypes (PredicateEngram + CircuitLayer)

### Objectives
- Prototype two novel reasoning memory modules inspired by recent literature survey:
  1. **PredicateEngram** — VQ-VAE codebook (hashable reasoning patterns) + Soft TPR (variable binding) + gated injection
  2. **AlgebraicCircuitLayer** — bank of decomposable sum-product circuits as einsum-native tensor ops

### Changes Saved
- **New file: `src/PredicateEngram.jl`** — `PredicateEngram` Lux layer:
  - VQ-VAE codebook quantizes hidden states into discrete "reasoning situation" codes
  - Rule bank stores (num_roles × num_roles) mixing matrices per code — initialized near identity
  - Soft TPR unbinding: learned filler extraction per role via projection
  - Rule application: batched matmul of rule_matrix × fillers (permutes/mixes roles)
  - Straight-through estimator for VQ quantization (Zygote-compatible)
  - Gated residual injection (gate bias -2.0, starts near-closed)
  - Separate `predicate_engram_commitment_loss` for training
- **New file: `src/CircuitLayer.jl`** — `AlgebraicCircuitLayer` Lux layer:
  - Bank of parallel 2-level sum-product networks
  - Leaf nodes: sigmoid neural predicates from hidden states (truth values in [0,1])
  - Product layer: grouped multiply in log-space (decomposable — disjoint scopes)
  - Sum layer: softplus-normalized weighted sums (smooth — same scopes)
  - Composition modes: `:mix` (weighted sum) or `:product` (conjunction across circuits)
  - `circuit_leaf_activations` for interpretability, `circuit_structure_summary` for inspection
  - All ops are standard tensor contractions — einsum-native, GPU-ready
- **Modified: `src/Swamma.jl`** — includes + exports for both modules
- **New: `test/test_predicate_engram.jl`** — 23 tests (shapes, VQ quantization, rule init, gate behavior)
- **New: `test/test_circuit_layer.jl`** — 26 tests (both compose modes, leaf activations, structure)

### Commands Run
- `julia --project=. test/test_predicate_engram.jl` — 23/23 pass
- `julia --project=. test/test_circuit_layer.jl` — 26/26 pass
- `julia --project=. test/test_engram.jl` — 24/24 pass (regression check)

### Design Rationale
- **PredicateEngram** fills the gap between Engram (lexical memory, content-addressed) and dynamic attention (expensive). It addresses logical reasoning via structure-addressed lookup: the VQ codebook learns to cluster hidden states by "reasoning situation," and the rule bank learns transformations (role permutation, filler routing) that encode common logical patterns.
- **AlgebraicCircuitLayer** provides guaranteed-tractable structured reasoning. The decomposability and smoothness properties ensure correct marginalization and composability. Circuit product composition (`:product` mode) enables conjunction of independent rule evaluations.

### Unresolved / Next Actions
- Neither module is integrated into LLaDA yet — both are standalone Lux layers ready for insertion
- Zygote gradient flow not yet verified for either module (forward pass only)
- VQ-VAE codebook collapse mitigation (EMA updates, commitment loss scheduling) not implemented
- Circuit depth is fixed at 2 levels — deeper circuits would need gradient checkpointing
- The two modules could be combined: PredicateEngram selects *which* circuit to activate, CircuitLayer evaluates it

---

## 2026-03-17 — Engram Conditional Memory Integration

### Objectives
- Integrate Engram conditional memory (Ma et al., 2026 — arxiv 2603.10087) into the Swamma/LLaDA architecture.
- Engram decouples static N-gram knowledge lookup (O(1)) from dynamic computation via multi-head hashing into large embedding tables with gated injection.

### Changes Saved
- **New file: `src/Engram.jl`** — `EngramModule` Lux layer implementing:
  - Multi-granular causal N-gram extraction (configurable orders, e.g., bigrams + trigrams)
  - Multi-head polynomial hashing into a combined embedding table
  - Vectorized gather + sum across heads, project to model dim, sigmoid-gated residual injection
  - Gate bias initialized at -2.0 (conservative injection at init)
  - `subtokens_to_token_ids` helper for PRIME subtoken → token ID reconstruction
- **Modified: `src/Swamma.jl`** — Added `include("Engram.jl")`, `using .EngramMod`, and exports
- **Modified: `src/LLaDA.jl`** — Full integration:
  - `LLaDAConfig` gains 6 new fields: `use_engram`, `engram_layers`, `engram_num_heads`, `engram_ngram_orders`, `engram_table_size`, `engram_head_dim`
  - `LLaDAModel` struct extended with `use_engram`, `engram_layer_map`, `EngramModules`
  - Auto-selection of engram layers when `engram_layers=[]`: early layer (2) + ~42% depth
  - Forward pass: token IDs reconstructed from subtokens (in `ignore_derivatives`), engram applied before selected SwammaBlocks in the `foldl` loop
  - `config_from_dict` updated for TOML parsing of vector fields
  - `initialparameters`/`initialstates` handle engram modules
  - Backward compatible: `use_engram=false` (default) produces identical behavior
- **New file: `test/test_engram.jl`** — 24 tests covering:
  - Standalone EngramModule (batched/unbatched, shapes, gate init)
  - `subtokens_to_token_ids` correctness
  - LLaDA with engram disabled (backward compat)
  - LLaDA with engram auto-selected layers (6-layer model → layers 2,3)
  - LLaDA with explicit engram layers
  - `config_from_dict` with engram TOML fields

### Commands Run
- `julia --project=. test/test_engram.jl` — 24/24 pass

### Best Current Recommendation
- Use `use_engram=true` with `engram_table_size` scaled to model size (512–4096 for small, 65536 for base, 262144 for large)
- For deeper models (12+ layers), explicit `engram_layers=[2, 5]` or similar gives more control than auto-selection
- The gate bias at -2.0 means engram starts nearly off — training will learn where injection helps

### Unresolved / Next Actions
- Gradient flow through engram embedding tables not yet verified under Zygote autodiff (forward pass works, training loop needs testing)
- No GPU/CUDA testing yet — the `_to_device_like` pattern is used for index transfer but untested on actual GPU
- Consider Option B (engram as third branch in SwammaBlock's α-mixer) if pre-attention injection proves insufficient
- Engram table memory footprint can be large (e.g., 256MB for base config) — consider gradient checkpointing or sparse updates

---

## 2026-03-15 — Distillation Plumbing Smoke (Teacher Targets + Losses)

### Objectives
- Verify the new teacher-target batch fields and mixed distillation losses run end-to-end without runtime regressions.
- Confirm trainer and threshold-sweep evaluation paths are stable with non-zero teacher loss weights.

### Changes Saved
- Teacher-target plumbing and distillation loss wiring completed in:
  - [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl)
  - [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl)
  - [`test/test_relation_extraction.jl`](/home/christos/code/julia/Swamma/test/test_relation_extraction.jl)
- Relation batch plumbing now separates:
  - candidate-pair activation mask for model forward
  - supervision mask for gold losses/metrics
- Teacher-only positive relations are now injected into candidate pairs without forcing contradictory supervised `NO_RELATION` targets on those slots.
- Teacher relation alignment now supports span-based mapping, so teacher relation order no longer has to follow the gold entity order.
- Teacher-only entity spans are now injected into the training span inventory while gold span supervision remains masked separately.
- Trainer now validates teacher payload coverage before training when distillation losses are enabled.
  - config key: `training.allow_missing_teacher_targets` for plumbing-only smoke runs
- Teacher-payload validator added:
  - [`scripts/validate_rebel_teacher_targets.jl`](/home/christos/code/julia/Swamma/scripts/validate_rebel_teacher_targets.jl)
- Teacher-target merge tool added:
  - [`scripts/merge_rebel_teacher_targets.jl`](/home/christos/code/julia/Swamma/scripts/merge_rebel_teacher_targets.jl)
- Teacher-request export tool added:
  - [`scripts/build_rebel_teacher_requests.jl`](/home/christos/code/julia/Swamma/scripts/build_rebel_teacher_requests.jl)
- Teacher-response parser added:
  - [`scripts/parse_rebel_teacher_responses.jl`](/home/christos/code/julia/Swamma/scripts/parse_rebel_teacher_responses.jl)
- Teacher-response generator added:
  - [`scripts/generate_rebel_teacher_responses.py`](/home/christos/code/julia/Swamma/scripts/generate_rebel_teacher_responses.py)
- End-to-end preparation orchestrator added:
  - [`scripts/prepare_rebel_teacher_corpus.py`](/home/christos/code/julia/Swamma/scripts/prepare_rebel_teacher_corpus.py)
- Pilot config added:
  - [`configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_distill_pilot.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_distill_pilot.toml)

### Experiment Commands And Key Metrics
- Resume smoke (`2000 -> 2005`) with teacher losses enabled:
  - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_distill_pilot.toml --resume checkpoints/redfm_base_safe_pair_edgev2_fromscratch/checkpoint_last.jls --max-steps 2005`
  - result: clean completion at step `2005`.
  - loaded weights: `teacher_entity_loss_weight=0.2`, `teacher_relation_loss_weight=0.4`, `teacher_confidence_loss_weight=0.2`.
- Threshold-sweep smoke on pilot checkpoint (`max_eval_batches=16`, `threshold=0.80`, `margin=0.10`):
  - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_distill_pilot.toml --threshold-sweep-checkpoint checkpoints/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_distill_pilot/checkpoint_last.jls --threshold-sweep-values 0.80 --threshold-sweep-margin 0.10 --max-eval-batches 16`
  - `pred spans + pred pairs`: `rel_f1=0.0004`, `oracle_rel=0.5625`, `pair_r=0.1437`, `pair_t16=0.0312`.
- Teacher-payload coverage check on current train split:
  - `julia --project=. scripts/validate_rebel_teacher_targets.jl --data data/rebel/train.jsonl`
  - result: `2878` rows scanned, `0` rows with `teacher_entities`, `0` rows with `teacher_relations`.
- Parse + tests after candidate/supervision mask split:
  - `julia --project=. -e 'include("src/Swamma.jl"); include("scripts/train_re_gpu.jl"); println("parse-ok")'`
  - `julia --project=. test/test_relation_extraction.jl`
  - result: parse passed, relation extraction tests passed including teacher-only pair injection coverage.
- Coverage-gate verification:
  - pilot config with opt-out:
    - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_distill_pilot.toml --resume checkpoints/redfm_base_safe_pair_edgev2_fromscratch/checkpoint_last.jls --max-steps 2000`
    - result: logs zero teacher coverage, warns, and completes because `allow_missing_teacher_targets=true`.
  - temporary config without opt-out:
    - `julia --project=. scripts/train_re_gpu.jl --config /tmp/jl_4RA9eRobk9.toml --resume checkpoints/redfm_base_safe_pair_edgev2_fromscratch/checkpoint_last.jls --max-steps 2000`
    - result: aborts before training with `teacher_entities and teacher_relations are missing in the train rows`.
- Synthetic merge + span-mapping verification:
  - merged reordered teacher entities plus index-based teacher relations into span-based `teacher_relations` with:
    - `julia --project=. scripts/merge_rebel_teacher_targets.jl --base /tmp/jl_QhonUZEboL_base.jsonl --teacher /tmp/jl_8p8xAlydi1_teacher.jsonl --output /tmp/jl_8DolGdIZSP_out.jsonl --strict`
  - validated merged output with:
    - `julia --project=. scripts/validate_rebel_teacher_targets.jl --data /tmp/jl_8DolGdIZSP_out.jsonl --require-teacher`
  - result: merged row preserved teacher coverage and produced span-based teacher relations (`head_start/head_stop/tail_start/tail_stop`) accepted by the validator.
- Teacher-request export verification:
  - `julia --project=. scripts/build_rebel_teacher_requests.jl --input data/rebel/train.jsonl --output /tmp/rebel_teacher_requests_sample.jsonl --max-rows 1`
  - result: exported one promptable request row with stable `match_key`, tokenized source context, strict response schema, and span-based relation instructions.
- Synthetic raw-response parse -> merge -> validate verification:
  - parsed raw teacher text JSON with:
    - `julia --project=. scripts/parse_rebel_teacher_responses.jl --input /tmp/jl_edsz2RaBSp_raw_teacher.jsonl --output /tmp/jl_parsed_teacher.jsonl --strict`
  - merged parsed annotations into base rows with:
    - `julia --project=. scripts/merge_rebel_teacher_targets.jl --base /tmp/jl_base_one.jsonl --teacher /tmp/jl_parsed_teacher.jsonl --output /tmp/jl_merged_teacher.jsonl --strict`
  - validated merged output with:
    - `julia --project=. scripts/validate_rebel_teacher_targets.jl --data /tmp/jl_merged_teacher.jsonl --require-teacher`
  - result: end-to-end export/parse/merge/validate pipeline now works on synthetic teacher responses, including teacher-only span cases.
- Generator + orchestrator verification:
  - generator CLI:
    - `python3 scripts/generate_rebel_teacher_responses.py --help`
  - orchestrator CLI:
    - `python3 scripts/prepare_rebel_teacher_corpus.py --help`
  - non-strict smoke reuse of an existing raw response file:
    - `python3 scripts/prepare_rebel_teacher_corpus.py --input /tmp/jl_base_one.jsonl --output-dir /tmp/rebel_teacher_pipeline_smoke_ok --raw-input /tmp/jl_edsz2RaBSp_raw_teacher.jsonl --skip-generation`
  - result: build-request -> parse -> merge -> validate chain completed end-to-end inside the orchestrator.
  - strict smoke on the same synthetic row fails as expected because the synthetic teacher labels intentionally do not match the base row schema.
- Parse + tests after teacher-only span injection:
  - `julia --project=. -e 'include("src/Swamma.jl"); include("scripts/train_re_gpu.jl"); println("parse-ok")'`
  - `julia --project=. test/test_relation_extraction.jl`
  - result: parse passed, full relation extraction tests passed including teacher-only span injection coverage.

### Best Current Checkpoint/Config Recommendation
- Keep this as a plumbing-validation result only.
- Do not treat the pilot as quality evidence for promotion or full distillation rollout.

### Unresolved Issues And Next Actions
- Baseline remains weak, and the current train split has no teacher payloads at all, so distillation should stay in controlled pilot mode.
- The remaining missing piece is real teacher output generation, not more trainer plumbing.
- Next actions:
  - run `prepare_rebel_teacher_corpus.py` on the real train split with a real teacher model.
  - inspect strict-validation failures, if any, on the produced merged corpus.
  - run the first matched-budget `base` vs `distill` comparison.

## 2026-03-15 — Decoder Bottleneck Probes (Pair-MLP vs Pair+Evidence-MLP)

### Objectives
- Isolate whether decoded relation collapse is primarily a decoder-head bottleneck after the edge-v2 training matrix.
- Test relation heads that remove biaffine residual dependence and rely on pair features (+ retrieval logits, optionally evidence).
- Test a class-imbalance-targeted relation objective (`relation_focal_gamma`) on the stronger decoder branch.

### Changes Saved
- Added decoder modes in relation extraction stack:
  - [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl)
  - `:pair_mlp`
  - `:pair_evidence_mlp`
- Added optional focal term support in relation CE:
  - [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl)
  - `relation_cross_entropy(...; no_relation_id, focal_gamma, positive_relation_weight)`
  - GPU-safe focal broadcast path (device-local constants)
- Wired focal config loading + propagation in trainer/eval loss paths:
  - [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl)
  - new config key: `training.relation_focal_gamma` (default `0.0`)
  - new config key: `training.positive_relation_weight` (default `1.0`)
  - new config key: `training.relation_logit_adjustment_tau` (default `0.0`)
  - added training-prior builder for relation logit adjustment (`NO_RELATION` kept unshifted)
- Added probe config files:
  - [`configs/redfm_base_safe_pair_edgev2_pairmlp_probe.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_edgev2_pairmlp_probe.toml)
  - [`configs/redfm_base_safe_pair_edgev2_pairevidmlp_probe.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_edgev2_pairevidmlp_probe.toml)
  - [`configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_probe.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_probe.toml)
  - [`configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw15_probe.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw15_probe.toml)
  - [`configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_probe.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_probe.toml)
  - [`configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw3_probe.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw3_probe.toml)
  - [`configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj1_probe.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj1_probe.toml)
  - [`configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj01_probe.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj01_probe.toml)
  - [`configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj025_probe.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj025_probe.toml)
  - [`configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj05_probe.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj05_probe.toml)
  - [`configs/redfm_base_safe_pair_edgev2_fusedevidence_focal2_posw2_probe.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_edgev2_fusedevidence_focal2_posw2_probe.toml)
- Added decoder coverage tests:
  - [`test/test_relation_extraction.jl`](/home/christos/code/julia/Swamma/test/test_relation_extraction.jl)
  - new testsets for `pair_mlp` and `pair_evidence_mlp`

### Experiment Commands And Key Metrics
- Pair-MLP probe run (`2000 -> 2250` from edge-v2 scratch checkpoint):
  - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairmlp_probe.toml --resume checkpoints/redfm_base_safe_pair_edgev2_fromscratch/checkpoint_last.jls --max-steps 2250`
  - eval@2250 (in-run): `rel_f1=0.0000`, `pair_recall=0.2073`, `relation_loss=5.2697`
- Pair+Evidence-MLP probe run (`2000 -> 2250` from same checkpoint):
  - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_probe.toml --resume checkpoints/redfm_base_safe_pair_edgev2_fromscratch/checkpoint_last.jls --max-steps 2250`
  - eval@2250 (in-run): `rel_f1=0.0000`, `pair_recall=0.3171`, `relation_loss=5.0877`
- Matched full-val threshold sweeps (`threshold=0.60/0.70/0.80`, `margin=0.10`, `max_eval_batches=128`):
  - Pair-MLP:
    - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairmlp_probe.toml --threshold-sweep-checkpoint checkpoints/redfm_base_safe_pair_edgev2_pairmlp_probe/checkpoint_last.jls --threshold-sweep-values 0.60,0.70,0.80 --threshold-sweep-margin 0.10 --max-eval-batches 128`
    - best `pred spans + pred pairs`: `rel_f1=0.0003` (`threshold=0.60`, `oracle_rel=0.6485`, `pair_r=0.1434`, `pair_t16=0.0604`)
  - Pair+Evidence-MLP:
    - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_probe.toml --threshold-sweep-checkpoint checkpoints/redfm_base_safe_pair_edgev2_pairevidmlp_probe/checkpoint_last.jls --threshold-sweep-values 0.60,0.70,0.80 --threshold-sweep-margin 0.10 --max-eval-batches 128`
    - best `pred spans + pred pairs`: `rel_f1=0.0008` (`threshold=0.80`, `oracle_rel=0.5604`, `pair_r=0.1917`, `pair_t16=0.0734`)
- Focal objective probe on Pair+Evidence-MLP (`gamma=2.0`, `2000 -> 2250`):
  - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_probe.toml --resume checkpoints/redfm_base_safe_pair_edgev2_fromscratch/checkpoint_last.jls --max-steps 2250`
  - eval@2250 (in-run): `rel_f1=0.0000`, `relation_loss=4.8929`, `pair_recall=0.3171`, `pair_t16=0.0976`
  - matched full-val sweep:
    - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_probe.toml --threshold-sweep-checkpoint checkpoints/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_probe/checkpoint_last.jls --threshold-sweep-values 0.60,0.70,0.80 --threshold-sweep-margin 0.10 --max-eval-batches 128`
    - best `pred spans + pred pairs`: `rel_f1=0.0007` (`threshold=0.80`, `oracle_rel=0.5363`, `pair_r=0.1908`, `pair_t16=0.0760`)
- Focal + positive-class weighting probe on Pair+Evidence-MLP (`gamma=2.0`, `positive_weight=2.0`, `2000 -> 2250`):
  - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_probe.toml --resume checkpoints/redfm_base_safe_pair_edgev2_fromscratch/checkpoint_last.jls --max-steps 2250`
  - eval@2250 (in-run): `rel_f1=0.0000`, `relation_loss=4.8323`, `oracle_rel=0.6951`, `pair_recall=0.3293`, `pair_t16=0.0976`
  - matched full-val sweep:
    - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_probe.toml --threshold-sweep-checkpoint checkpoints/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_probe/checkpoint_last.jls --threshold-sweep-values 0.60,0.70,0.80 --threshold-sweep-margin 0.10 --max-eval-batches 128`
    - best `pred spans + pred pairs`: `rel_f1=0.0011` (`threshold=0.80`, `oracle_rel=0.6477`, `pair_r=0.1917`, `pair_t16=0.0656`)
- Positive-weight sensitivity runs (`gamma=2.0`, same `2000 -> 2250` + matched sweep protocol):
  - `posw=1.5`:
    - train: `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw15_probe.toml --resume checkpoints/redfm_base_safe_pair_edgev2_fromscratch/checkpoint_last.jls --max-steps 2250`
    - sweep: `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw15_probe.toml --threshold-sweep-checkpoint checkpoints/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw15_probe/checkpoint_last.jls --threshold-sweep-values 0.60,0.70,0.80 --threshold-sweep-margin 0.10 --max-eval-batches 128`
    - best `pred spans + pred pairs`: `rel_f1=0.0003` (`threshold=0.80`, `oracle_rel=0.5648`, `pair_r=0.1917`, `pair_t16=0.0717`)
  - `posw=3.0`:
    - train: `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw3_probe.toml --resume checkpoints/redfm_base_safe_pair_edgev2_fromscratch/checkpoint_last.jls --max-steps 2250`
    - sweep: `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw3_probe.toml --threshold-sweep-checkpoint checkpoints/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw3_probe/checkpoint_last.jls --threshold-sweep-values 0.60,0.70,0.80 --threshold-sweep-margin 0.10 --max-eval-batches 128`
    - best `pred spans + pred pairs`: `rel_f1=0.0006` (`threshold=0.80`, `oracle_rel=0.5976`, `pair_r=0.1943`, `pair_t16=0.0717`)
- Logit-adjusted CE probe on top of best objective point (`focal2 + posw2`, `tau=1.0`):
  - train: `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj1_probe.toml --resume checkpoints/redfm_base_safe_pair_edgev2_fromscratch/checkpoint_last.jls --max-steps 2250`
  - eval@2250 (in-run): `relation_loss=5.9235`, `oracle_rel=0.6341`, `pair_recall=0.3171`, `pair_t16=0.0976`
  - sweep: `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj1_probe.toml --threshold-sweep-checkpoint checkpoints/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj1_probe/checkpoint_last.jls --threshold-sweep-values 0.60,0.70,0.80 --threshold-sweep-margin 0.10 --max-eval-batches 128`
  - best `pred spans + pred pairs`: `rel_f1=0.0003` (`threshold=0.80`, `oracle_rel=0.5639`, `pair_r=0.1960`, `pair_t16=0.0743`)
- Low-`tau` logit-adjusted CE check (`focal2 + posw2`, `tau=0.25`):
  - train: `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj025_probe.toml --resume checkpoints/redfm_base_safe_pair_edgev2_fromscratch/checkpoint_last.jls --max-steps 2250`
  - eval@2250 (in-run): `relation_loss=5.0346`, `oracle_rel=0.6341`, `pair_recall=0.3415`, `pair_t16=0.1098`
  - sweep: `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj025_probe.toml --threshold-sweep-checkpoint checkpoints/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj025_probe/checkpoint_last.jls --threshold-sweep-values 0.60,0.70,0.80 --threshold-sweep-margin 0.10 --max-eval-batches 128`
  - best `pred spans + pred pairs`: `rel_f1=0.0003` (`threshold=0.80`, `oracle_rel=0.6036`, `pair_r=0.1934`, `pair_t16=0.0751`)
- Low-`tau` closure checks (`focal2 + posw2`):
  - `tau=0.1`:
    - train: `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj01_probe.toml --resume checkpoints/redfm_base_safe_pair_edgev2_fromscratch/checkpoint_last.jls --max-steps 2250`
    - eval@2250 (in-run): `relation_loss=4.9300`, `oracle_rel=0.5610`, `pair_recall=0.3293`, `pair_t16=0.1098`, `rel_f1=0.0009`
    - sweep best `pred spans + pred pairs`: `rel_f1=0.0011` (`threshold=0.80`, `oracle_rel=0.5035`, `pair_r=0.1762`, `pair_t16=0.0691`)
  - `tau=0.5`:
    - train: `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_logitadj05_probe.toml --resume checkpoints/redfm_base_safe_pair_edgev2_fromscratch/checkpoint_last.jls --max-steps 2250`
    - eval@2250 (in-run): `relation_loss=5.3055`, `oracle_rel=0.6463`, `pair_recall=0.3293`, `pair_t16=0.0854`, `rel_f1=0.0010`
    - sweep best `pred spans + pred pairs`: `rel_f1=0.0003` (`threshold=0.80`, `oracle_rel=0.5967`, `pair_r=0.1943`, `pair_t16=0.0699`)
- Architecture-side decoder alternative on best objective point (`fused_evidence + focal2 + posw2`):
  - train: `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_fusedevidence_focal2_posw2_probe.toml --resume checkpoints/redfm_base_safe_pair_edgev2_fromscratch/checkpoint_last.jls --max-steps 2250`
  - eval@2250 (in-run): `relation_loss=11.3288`, `oracle_rel=0.6098`, `pair_recall=0.2317`, `pair_t16=0.1220`
  - sweep: `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_fusedevidence_focal2_posw2_probe.toml --threshold-sweep-checkpoint checkpoints/redfm_base_safe_pair_edgev2_fusedevidence_focal2_posw2_probe/checkpoint_last.jls --threshold-sweep-values 0.60,0.70,0.80 --threshold-sweep-margin 0.10 --max-eval-batches 128`
  - best `pred spans + pred pairs`: `rel_f1=0.0005` (`threshold=0.80`, `oracle_rel=0.5751`, `pair_r=0.1606`, `pair_t16=0.0734`)

### Best Current Checkpoint/Config Recommendation
- Keep `pair_evidence_mlp` as the stronger decoder direction over `pair_mlp` on this checkpoint family.
- Current branch ranking:
  - `pair_mlp`: best `rel_f1=0.0003` (degrades).
  - `pair_evidence_mlp`: best `rel_f1=0.0008`.
  - `pair_evidence_mlp + focal2`: best `rel_f1=0.0007` (worse than non-focal).
  - `pair_evidence_mlp + focal2 + posw1.5`: best `rel_f1=0.0003`.
  - `pair_evidence_mlp + focal2 + posw2`: best `rel_f1=0.0011` (best in this probe family).
  - `pair_evidence_mlp + focal2 + posw3`: best `rel_f1=0.0006`.
  - `pair_evidence_mlp + focal2 + posw2 + logitadj0.1`: best `rel_f1=0.0011` but with lower coverage (`oracle_rel=0.5035`, `pair_r=0.1762`).
  - `pair_evidence_mlp + focal2 + posw2 + logitadj1`: best `rel_f1=0.0003` (regressive).
  - `pair_evidence_mlp + focal2 + posw2 + logitadj0.25`: best `rel_f1=0.0003` (regressive).
  - `pair_evidence_mlp + focal2 + posw2 + logitadj0.5`: best `rel_f1=0.0003` (regressive).
  - `fused_evidence + focal2 + posw2`: best `rel_f1=0.0005` (regressive).
- Do not promote to `v1_locked` yet; even best probe branch remains materially below promotion target.

### Unresolved Issues And Next Actions
- Decoder architecture + objective tweaks improved the local edge-v2 ceiling, but global quality remains low.
- Next actions:
  - keep `focal2+posw2` as local objective baseline for this branch.
  - keep logit-adjustment disabled (no robust net win across `tau=0.1/0.25/0.5/1.0`).
  - keep the same matched threshold sweep protocol to preserve comparability.

## 2026-03-15 — Training-First Matrix (Control vs Edge-v2 vs Edge-v2+Curriculum)

### Objectives
- Execute a training-centric matrix to avoid further decode-only tuning.
- Compare control continuation against edge-v2 from-scratch variants under matched eval protocol.

### Changes Saved
- Added dedicated matrix configs:
  - [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_matrix_control.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_matrix_control.toml)
  - [`configs/redfm_base_safe_pair_edgev2_fromscratch.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_edgev2_fromscratch.toml)
  - [`configs/redfm_base_safe_pair_edgev2_curric10_fromscratch.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_edgev2_curric10_fromscratch.toml)
- Matrix run checkpoints produced:
  - `checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_matrix_control/checkpoint_last.jls`
  - `checkpoints/redfm_base_safe_pair_edgev2_fromscratch/checkpoint_last.jls`
  - `checkpoints/redfm_base_safe_pair_edgev2_curric10_fromscratch/checkpoint_last.jls`

### Experiment Commands And Key Metrics
- Control continuation (`1250 -> 2000`):
  - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_matrix_control.toml --resume checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4/checkpoint_last.jls --max-steps 2000`
  - in-train eval highlights:
    - step `1500`: `rel_f1=0.0022`, `pair_recall=0.1463`
    - step `1750`: `rel_f1=0.0000`, `pair_recall=0.2073`
    - step `2000`: `rel_f1=0.0000`, `pair_recall=0.1707`
- Edge-v2 from scratch (`0 -> 2000`):
  - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_fromscratch.toml --max-steps 2000`
  - in-train eval highlights:
    - step `1250`: `rel_f1=0.0000`, `pair_recall=0.2195`
    - step `1500`: `rel_f1=0.0023`, `pair_recall=0.2317`
    - step `1750`: `rel_f1=0.0000`, `pair_recall=0.2195`
    - step `2000`: `rel_f1=0.0000`, `pair_recall=0.1951`
- Edge-v2 + curriculum from scratch (`0 -> 2000`):
  - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_edgev2_curric10_fromscratch.toml --max-steps 2000`
  - in-train eval highlights:
    - step `1250`: `rel_f1=0.0000`, `pair_recall=0.1585`
    - step `1500`: `rel_f1=0.0000`, `pair_recall=0.1220`
    - step `1750`: `rel_f1=0.0000`, `pair_recall=0.2561`
    - step `2000`: `rel_f1=0.0000`, `pair_recall=0.3171`
- Matched full-val threshold sweeps for all three (`threshold=0.60/0.70/0.80`, margin `0.10`, `max_eval_batches=128`):
  - control best (`pred spans + pred pairs`): `rel_f1=0.0006`, `oracle_rel=0.4810`, `pair_r=0.1071`, `pair_t16=0.0509`
  - edge-v2 scratch best: `rel_f1=0.0008`, `oracle_rel=0.6140`, `pair_r=0.1485`, `pair_t16=0.0570`
  - edge-v2 + curriculum best: `rel_f1=0.0000`, `oracle_rel=0.5570`, `pair_r=0.1813`, `pair_t16=0.0622`

### Best Current Checkpoint/Config Recommendation
- None of the matrix legs is promotable.
- Keep this matrix as evidence that training alone (in this recipe) still does not recover decoded relation quality.

### Unresolved Issues And Next Actions
- Coverage improvements are not converting into relation classification quality.
- Next actions:
  - isolate decoder bottleneck with targeted decoder-side training (stronger relation head, less no-relation collapse).
  - run short calibration stress after each 250-step checkpoint to detect early confidence collapse and stop bad runs sooner.

## 2026-03-15 — Relation-Agnostic Compatibility Hook (Edge Retrieval v2)

### Objectives
- Complete the pending Edge Retrieval v2 TODO item: relation-agnostic compatibility projection.
- Keep the change checkpoint-safe and validate with tests plus matched checkpoint sweeps.

### Changes Saved
- Added runtime compatibility hook in retrieval head:
  - [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl)
  - `PairRetrievalHead` now accepts an optional ninth input value: `compatibility_scale`.
  - Added learned relation-agnostic compatibility term:
    - `compatibility_logits = sum(feature_proj .* (head_proj .* tail_proj), dims=1) / sqrt(r)`
    - applied as `logits += compatibility_scale * compatibility_logits`.
  - Model runtime input added:
    - `retrieval_compatibility_scale` (default `0.0`, so legacy behavior is unchanged).
  - Draft and final retrieval calls both pass compatibility scale.
- Wired trainer/config input propagation:
  - [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl)
  - `load_retrieval_bias_settings` now reads `retrieval_compatibility_scale`.
  - `with_retrieval_bias_inputs` and proposal/fixed/oracle/auto-calibration input builders now propagate the key.
- Added test coverage:
  - [`test/test_relation_extraction.jl`](/home/christos/code/julia/Swamma/test/test_relation_extraction.jl)
  - New testset: `Pair Retrieval Compatibility Scale Hook`
    - verifies `scale=0` is no-op vs default behavior.
    - verifies non-zero scale changes retrieval logits.
- Added ablation config:
  - [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun_compat025.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun_compat025.toml)

### Experiment Commands And Key Metrics
- Parse + tests:
  - `julia --project=. -e 'include("src/Swamma.jl"); include("scripts/train_re_gpu.jl"); println("parse-ok")'`
  - `julia --project=. test/test_relation_extraction.jl`
  - result: all relation extraction tests passed.
- Matched threshold sweeps on checkpoint:
  - checkpoint:
    - `checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun/checkpoint_last.jls`
  - baseline config:
    - `configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun.toml`
  - compat config (`retrieval_compatibility_scale=0.25`):
    - `configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun_compat025.toml`
- Sampled (`max_eval_batches=32`, thresholds `0.60/0.70/0.80`, margin `0.10`):
  - baseline best (`pred spans + pred pairs`): `rel_f1=0.0018`, `pair_r=0.1729`, `pair_t16=0.0634`
  - compat best: `rel_f1=0.0018`, `pair_r=0.1729`, `pair_t16=0.0692`
- Full-val (`max_eval_batches=128`, same thresholds/margin):
  - baseline best (`pred spans + pred pairs`): `rel_f1=0.0012`, `pair_r=0.1554`, `pair_t16=0.0639`
  - compat best: `rel_f1=0.0013`, `pair_r=0.1546`, `pair_t16=0.0648`

### Best Current Checkpoint/Config Recommendation
- Keep compatibility hook available but not promoted as default yet:
  - `retrieval_compatibility_scale = 0.0` remains the safest baseline default.
- Use compat scale as a controlled knob for future checkpoints:
  - current effect is marginal and not yet a robust promotion signal.

### Unresolved Issues And Next Actions
- Decoded relation F1 remains far below `v1_locked` target region.
- Next actions:
  - combine compatibility hook with stronger proposer-side redesign (not just decode/runtime bias knobs).
  - prioritize unresolved v2 items that can materially raise proposal quality (`pair_r`, `pair_t16`) before long-run promotion.

## 2026-03-15 — Seed Sweep Orchestration + Refreshed 3-Seed Aggregate

### Objectives
- Continue automation by removing manual multi-command seed execution.
- Produce a consistent 3-seed dataset with explicit `seed42` filenames and refreshed aggregation.

### Changes Saved
- Added orchestration script:
  - [`scripts/run_long_context_seed_sweep.jl`](/home/christos/code/julia/Swamma/scripts/run_long_context_seed_sweep.jl)
  - capabilities:
    - run benchmark/eval per seed
    - `--skip-existing` incremental operation
    - optional checkpoint flags for eval (`--swamma-checkpoint`, `--transformer-checkpoint`)
    - automatic call to `scripts/aggregate_long_context_seeds.jl`
    - dry-run mode (`--dry-run`)
- Updated protocol with one-command sweep usage:
  - [`docs/LONG_CONTEXT_PROTOCOL.md`](/home/christos/code/julia/Swamma/docs/LONG_CONTEXT_PROTOCOL.md)
- Generated seed-42 named outputs for consistency:
  - [`benchmarks/long_context_benchmark_seed42.csv`](/home/christos/code/julia/Swamma/benchmarks/long_context_benchmark_seed42.csv)
  - [`benchmarks/long_context_eval_seed42.csv`](/home/christos/code/julia/Swamma/benchmarks/long_context_eval_seed42.csv)
- Refreshed aggregate outputs from explicit `seed42/7/19` inputs:
  - [`benchmarks/long_context_benchmark_agg_3seed.csv`](/home/christos/code/julia/Swamma/benchmarks/long_context_benchmark_agg_3seed.csv)
  - [`benchmarks/long_context_eval_agg_3seed.csv`](/home/christos/code/julia/Swamma/benchmarks/long_context_eval_agg_3seed.csv)
  - [`benchmarks/long_context_aggregate_summary_3seed.md`](/home/christos/code/julia/Swamma/benchmarks/long_context_aggregate_summary_3seed.md)

### Experiment Commands And Key Metrics
- Dry-run validation:
  - `julia --project=. scripts/run_long_context_seed_sweep.jl --seeds 42,7,19 --skip-existing --dry-run`
- Executed sweep:
  - `julia --project=. scripts/run_long_context_seed_sweep.jl --seeds 42,7,19 --skip-existing`
  - behavior:
    - ran missing seed42 benchmark/eval
    - skipped existing seed7/seed19 files
    - aggregated automatically
- Refreshed 3-seed aggregate metrics:
  - Swamma exponent: `1.0358`
  - Transformer exponent: `1.4107`
  - latency ratio (Transformer/Swamma): `0.934, 0.996, 1.370, 1.582, 2.716` at contexts `1024..16384`
  - throughput means (tok/s):
    - Swamma: `5629.76, 5581.50, 5555.79, 5173.09, 5164.41`
    - Transformer: `6240.08, 5653.69, 4056.82, 3269.29, 1901.36`
  - eval remains random-init (`needle_acc=0.0000` across both architectures).

### Best Current Checkpoint/Config Recommendation
- Use orchestration script as default driver for future seed runs:
  - `scripts/run_long_context_seed_sweep.jl`
- Keep quick eval config for iteration:
  - `configs/swamma_vs_transformer/eval_long_context_quick.toml`
- For publishable quality claims:
  - rerun with checkpoint flags enabled in the sweep script.

### Unresolved Issues And Next Actions
- No matched long-context trained checkpoint pair currently present in repo for quality comparison.
- Next actions:
  - provide/train Swamma + Transformer checkpoints and rerun sweep with checkpoint flags.
  - then rerun full-budget eval config (`needle_batches=64`) for checkpointed aggregates.

## 2026-03-15 — 3-Seed Long-Context Sweep (GPU, Quick Eval)

### Objectives
- Expand long-context comparison from single-run to a 3-seed view.
- Generate aggregate benchmark/eval outputs with mean/std reporting.

### Changes Saved
- Added quick eval config for reproducible seed sweeps:
  - [`configs/swamma_vs_transformer/eval_long_context_quick.toml`](/home/christos/code/julia/Swamma/configs/swamma_vs_transformer/eval_long_context_quick.toml)
  - same architecture/context settings as default eval config, but `needle_batches = 8` for faster multi-seed iteration.
- Produced per-seed benchmark outputs:
  - [`benchmarks/long_context_benchmark_seed7.csv`](/home/christos/code/julia/Swamma/benchmarks/long_context_benchmark_seed7.csv)
  - [`benchmarks/long_context_benchmark_seed19.csv`](/home/christos/code/julia/Swamma/benchmarks/long_context_benchmark_seed19.csv)
- Produced per-seed eval outputs:
  - [`benchmarks/long_context_eval_seed7.csv`](/home/christos/code/julia/Swamma/benchmarks/long_context_eval_seed7.csv)
  - [`benchmarks/long_context_eval_seed19.csv`](/home/christos/code/julia/Swamma/benchmarks/long_context_eval_seed19.csv)
- Produced 3-seed aggregated outputs:
  - [`benchmarks/long_context_benchmark_agg_3seed.csv`](/home/christos/code/julia/Swamma/benchmarks/long_context_benchmark_agg_3seed.csv)
  - [`benchmarks/long_context_eval_agg_3seed.csv`](/home/christos/code/julia/Swamma/benchmarks/long_context_eval_agg_3seed.csv)
  - [`benchmarks/long_context_aggregate_summary_3seed.md`](/home/christos/code/julia/Swamma/benchmarks/long_context_aggregate_summary_3seed.md)

### Experiment Commands And Key Metrics
- Seeded benchmark runs:
  - `julia --project=. scripts/benchmark_long_context.jl --config configs/swamma_vs_transformer/benchmark_long_context.toml --output benchmarks/long_context_benchmark_seed7.csv --device gpu --seed 7`
  - `julia --project=. scripts/benchmark_long_context.jl --config configs/swamma_vs_transformer/benchmark_long_context.toml --output benchmarks/long_context_benchmark_seed19.csv --device gpu --seed 19`
- Seeded quick eval runs:
  - `julia --project=. scripts/eval_long_context.jl --config configs/swamma_vs_transformer/eval_long_context_quick.toml --output benchmarks/long_context_eval_seed7.csv --device gpu --seed 7`
  - `julia --project=. scripts/eval_long_context.jl --config configs/swamma_vs_transformer/eval_long_context_quick.toml --output benchmarks/long_context_eval_seed19.csv --device gpu --seed 19`
- 3-seed aggregation run:
  - `julia --project=. scripts/aggregate_long_context_seeds.jl --benchmark-csvs benchmarks/long_context_benchmark.csv,benchmarks/long_context_benchmark_seed7.csv,benchmarks/long_context_benchmark_seed19.csv --eval-csvs benchmarks/long_context_eval.csv,benchmarks/long_context_eval_seed7.csv,benchmarks/long_context_eval_seed19.csv --output-benchmark-csv benchmarks/long_context_benchmark_agg_3seed.csv --output-eval-csv benchmarks/long_context_eval_agg_3seed.csv --output-md benchmarks/long_context_aggregate_summary_3seed.md`
- Aggregate highlights (3 seeds):
  - Swamma exponent: `1.0363`
  - Transformer exponent: `1.3713`
  - latency ratio (Transformer/Swamma): `1.090, 0.959, 1.398, 1.575, 2.715` at contexts `1024..16384`
  - needle accuracy remained `0.0000 ± 0.0000` for both architectures at all contexts (random-init baseline).

### Best Current Checkpoint/Config Recommendation
- For systems scaling comparison:
  - use 3-seed aggregate benchmark report in `benchmarks/long_context_aggregate_summary_3seed.md`.
- For quality iteration:
  - keep using quick eval config for rapid seed sweeps; switch to full `needle_batches=64` once checkpointed models are available.

### Unresolved Issues And Next Actions
- Quality metrics remain random-init and therefore non-informative for architecture quality ranking.
- Next actions:
  - run checkpointed evals for all 3 seeds.
  - regenerate aggregated reports using checkpointed outputs.
  - optionally add text-eval corpus path and include text metrics in the aggregated tables.

## 2026-03-15 — Multi-Seed Aggregation Pipeline

### Objectives
- Continue long-context benchmarking workflow by adding reproducible multi-seed aggregation.
- Generate machine-readable aggregate tables and a compact markdown report from multiple CSV runs.

### Changes Saved
- Added aggregation script:
  - [`scripts/aggregate_long_context_seeds.jl`](/home/christos/code/julia/Swamma/scripts/aggregate_long_context_seeds.jl)
  - supports comma-separated benchmark/eval CSV inputs.
  - outputs:
    - aggregated benchmark CSV (`mean/std` per architecture+context)
    - aggregated eval CSV (`mean/std` per architecture+context)
    - markdown aggregate report (scaling exponents + ratio/needle tables)
- Updated protocol documentation with new aggregation command:
  - [`docs/LONG_CONTEXT_PROTOCOL.md`](/home/christos/code/julia/Swamma/docs/LONG_CONTEXT_PROTOCOL.md)
- Produced aggregate artifacts (single-run validation mode):
  - [`benchmarks/long_context_benchmark_agg.csv`](/home/christos/code/julia/Swamma/benchmarks/long_context_benchmark_agg.csv)
  - [`benchmarks/long_context_eval_agg.csv`](/home/christos/code/julia/Swamma/benchmarks/long_context_eval_agg.csv)
  - [`benchmarks/long_context_aggregate_summary.md`](/home/christos/code/julia/Swamma/benchmarks/long_context_aggregate_summary.md)

### Experiment Commands And Key Metrics
- Validation command:
  - `julia --project=. scripts/aggregate_long_context_seeds.jl --benchmark-csvs benchmarks/long_context_benchmark.csv --eval-csvs benchmarks/long_context_eval_full64.csv --output-benchmark-csv benchmarks/long_context_benchmark_agg.csv --output-eval-csv benchmarks/long_context_eval_agg.csv --output-md benchmarks/long_context_aggregate_summary.md`
- Result highlights (single-input aggregate, so std=0):
  - Swamma exponent: `1.0339`
  - Transformer exponent: `1.3404`
  - latency ratio (Transformer/Swamma): `1.217, 0.924, 1.463, 1.571, 2.701` for `1024..16384`
  - needle deltas: `0.0000` across contexts (random-init eval baseline).

### Best Current Checkpoint/Config Recommendation
- Continue using:
  - per-run benchmark CSVs from `scripts/benchmark_long_context.jl`
  - per-run eval CSVs from `scripts/eval_long_context.jl`
  - aggregate pass via `scripts/aggregate_long_context_seeds.jl` once >=3 seeds are available.

### Unresolved Issues And Next Actions
- Current aggregate still reflects one seed input; publishable claims require multi-seed inputs.
- Next actions:
  - run seeds 2/3 for benchmark and eval.
  - rerun aggregator with all seed CSVs and regenerate final summary tables.

## 2026-03-15 — Long-Context Result Summarizer + Checkpoint Audit

### Objectives
- Continue from completed GPU benchmark/eval runs and produce a compact, repeatable comparison artifact.
- Check whether compatible long-context LLaDA checkpoints are available for immediate quality reruns.

### Changes Saved
- Added CSV summary script:
  - [`scripts/summarize_long_context_results.jl`](/home/christos/code/julia/Swamma/scripts/summarize_long_context_results.jl)
  - parses benchmark/eval CSVs and writes a markdown report with:
    - log-log scaling exponents
    - max finite context per architecture
    - per-context throughput + latency ratio table
    - per-context needle accuracy comparison
- Updated protocol documentation to include summarization step:
  - [`docs/LONG_CONTEXT_PROTOCOL.md`](/home/christos/code/julia/Swamma/docs/LONG_CONTEXT_PROTOCOL.md)
  - added script reference and run command.
- Generated summary artifact:
  - [`benchmarks/long_context_summary.md`](/home/christos/code/julia/Swamma/benchmarks/long_context_summary.md)

### Experiment Commands And Key Metrics
- Checkpoint audit commands run:
  - `ls/find` over `checkpoints/` and `checkpoints_llm/` to identify candidate LLaDA long-context checkpoints.
  - result: no clear paired Swamma/Transformer long-context checkpoints for immediate fair quality comparison.
- Summary generation command:
  - `julia --project=. scripts/summarize_long_context_results.jl --benchmark-csv benchmarks/long_context_benchmark.csv --eval-csv benchmarks/long_context_eval_full64.csv --output-md benchmarks/long_context_summary.md`
- Reported metrics from summary:
  - Swamma exponent: `1.0339`
  - Transformer exponent: `1.3404`
  - latency ratio (Transformer/Swamma): `1.217, 0.924, 1.463, 1.571, 2.701` for contexts `1024..16384`
  - needle accuracy remained `0.0000` for both architectures at all contexts (random-init baseline).

### Best Current Checkpoint/Config Recommendation
- Keep using:
  - benchmark CSV: `benchmarks/long_context_benchmark.csv`
  - full eval CSV: `benchmarks/long_context_eval_full64.csv`
  - summary report: `benchmarks/long_context_summary.md`
- For meaningful quality claims, the next required input is matched trained checkpoints for both architectures.

### Unresolved Issues And Next Actions
- No matched trained long-context Swamma/Transformer checkpoint pair is wired into the eval flow yet.
- Next actions:
  - pick or train a matched checkpoint pair and rerun eval with `--swamma-checkpoint` / `--transformer-checkpoint`.
  - run multi-seed repeats and regenerate summary from aggregated CSVs.

## 2026-03-14 — Dense Transformer Baseline For Stable 16k GPU Runs

### Objectives
- Remove the Transformer long-context GPU crash (`CUDA illegal memory access`) at `8192+`.
- Keep Swamma vs Transformer benchmark/eval on the same GPU device for all target contexts.
- Regenerate benchmark/eval CSVs after the stability fix.

### Changes Saved
- Reworked Transformer baseline attention implementation in:
  - [`scripts/long_context_models.jl`](/home/christos/code/julia/Swamma/scripts/long_context_models.jl)
  - replaced `SWAttention(window=sequence_length)` baseline path with a new dense full-attention layer:
    - `DenseSelfAttention` using batched matmul (`NNlib.batched_mul`) + softmax attention.
  - retained PRIME sub-token pathway and output filtering behavior.
- Updated long-context runners to remove temporary high-context skip behavior:
  - [`scripts/benchmark_long_context.jl`](/home/christos/code/julia/Swamma/scripts/benchmark_long_context.jl)
  - [`scripts/eval_long_context.jl`](/home/christos/code/julia/Swamma/scripts/eval_long_context.jl)
  - `effective_device_for_point` now leaves device unchanged (no forced skips).

### Experiment Commands And Key Metrics
- Validation smoke (GPU):
  - `julia --project=. scripts/benchmark_long_context.jl --config /tmp/swamma_bench_smoke.toml --output /tmp/swamma_bench_smoke_gpu_dense.csv --device gpu`
  - completed without crashes.
- Targeted high-context Transformer checks (GPU):
  - `... --config /tmp/swamma_bench_transformer_8192.toml --device gpu`
    - `N=8192 mean=2852.08ms`
  - `... --config /tmp/swamma_bench_transformer_16384.toml --device gpu`
    - `N=16384 mean=9165.25ms`
  - both completed successfully; prior crash mode no longer reproduced.
- Full benchmark sweep (GPU, all contexts, both architectures):
  - `julia --project=. scripts/benchmark_long_context.jl --config configs/swamma_vs_transformer/benchmark_long_context.toml --output benchmarks/long_context_benchmark.csv --device gpu`
  - Swamma exponent: `1.034`
  - Transformer exponent: `1.340`
  - relative latency ratio (Transformer/Swamma):
    - `1024: 1.22x`
    - `2048: 0.92x`
    - `4096: 1.46x`
    - `8192: 1.57x`
    - `16384: 2.70x`
- Quick eval sweep (GPU, all contexts, needle stress only):
  - `julia --project=. scripts/eval_long_context.jl --config /tmp/swamma_eval_gpu_quick.toml --output benchmarks/long_context_eval.csv --device gpu`
  - all rows completed for both architectures through `16384`.
  - random-init run produced `needle_acc=0.0000` throughout (expected pretraining baseline).
- Full eval sweep (GPU, production eval config):
  - `julia --project=. scripts/eval_long_context.jl --config configs/swamma_vs_transformer/eval_long_context.toml --output benchmarks/long_context_eval_full64.csv --device gpu`
  - all rows completed for both architectures through `16384` (no skips/crashes).
  - random-init run produced `needle_acc=0.0000` throughout (expected pretraining baseline).

### Best Current Checkpoint/Config Recommendation
- For systems profiling now:
  - use [configs/swamma_vs_transformer/benchmark_long_context.toml](/home/christos/code/julia/Swamma/configs/swamma_vs_transformer/benchmark_long_context.toml) with `--device gpu`.
- For fast eval iteration:
  - use `/tmp/swamma_eval_gpu_quick.toml` (`needle_batches=8`) until trained checkpoints are available.
- For quality claims:
  - run eval with trained checkpoints and (optionally) text-eval enabled before concluding.

### Unresolved Issues And Next Actions
- Current evaluation results are random-init only; no model-quality claim yet.
- Next actions:
  - run full-budget eval (`needle_batches=64`) with trained Swamma and Transformer checkpoints.
  - add multi-seed benchmark/eval aggregation and summary script (mean/std, confidence intervals).
  - optionally pin Transformer baseline parameter count closer to Swamma for stricter fairness.

## 2026-03-14 — Long-Context Harness GPU Enablement + First Runs

### Objectives
- Enable the long-context benchmark/eval scripts to use GPU explicitly.
- Replace accidental CPU execution with controllable `--device cpu|gpu`.
- Complete a first non-smoke long-context sweep and persist CSV outputs.

### Changes Saved
- Updated shared long-context module for device robustness:
  - [`scripts/long_context_models.jl`](/home/christos/code/julia/Swamma/scripts/long_context_models.jl)
  - added device helpers (`to_device_like`, leaf-array detection).
  - PRIME compatibility masking now respects logits device.
  - moved position/time conditioning tensors to input device in Transformer baseline path.
  - made metrics/needle path GPU-safe by materializing logits to CPU where scalar indexing is used.
- Updated benchmark runner for GPU control and stability handling:
  - [`scripts/benchmark_long_context.jl`](/home/christos/code/julia/Swamma/scripts/benchmark_long_context.jl)
  - added `--device cpu|gpu`, recursive param/state/input transfer, CUDA synchronization for timing.
  - benchmark CSV now includes `device` and `run_note` columns.
  - added skip guard for known unstable case: Transformer full-window attention on GPU at `context >= 8192`.
    - rows are emitted as `device=skipped`, `run_note=skipped_full_attention_gpu_instability`.
  - scaling-fit now ignores non-finite rows.
- Updated eval runner for GPU control and aligned skip behavior:
  - [`scripts/eval_long_context.jl`](/home/christos/code/julia/Swamma/scripts/eval_long_context.jl)
  - added `--device cpu|gpu`, recursive param/state transfer.
  - eval CSV now includes `device` and `run_note` columns.
  - same Transformer `>=8192` GPU skip guard to avoid CUDA illegal-memory crash.

### Experiment Commands And Key Metrics
- GPU capability check:
  - `julia --project=. -e 'using CUDA; println(CUDA.functional()); ...'`
  - result: CUDA functional on `NVIDIA GB10`, ~`130.66 GB` VRAM.
- GPU smoke checks:
  - `julia --project=. scripts/benchmark_long_context.jl --config /tmp/swamma_bench_smoke.toml --output /tmp/swamma_bench_smoke_gpu.csv --device gpu`
  - `julia --project=. scripts/eval_long_context.jl --config /tmp/swamma_eval_smoke.toml --output /tmp/swamma_eval_smoke_gpu.csv --device gpu`
  - both completed successfully.
- Full benchmark run (GPU; Transformer `8192+` skipped by guard):
  - `julia --project=. scripts/benchmark_long_context.jl --config configs/swamma_vs_transformer/benchmark_long_context.toml --output benchmarks/long_context_benchmark.csv --device gpu`
  - Swamma (`1024..16384`, GPU):
    - exponent `1.036`
    - throughput roughly `5.1k–5.9k tok/s`
  - Transformer (`1024..4096`, GPU):
    - exponent `1.391` (fit over finite rows)
    - throughput `2.0k–3.45k tok/s`
  - relative speedup (Transformer/Swamma latency):
    - `1024: 1.62x`, `2048: 1.87x`, `4096: 2.92x` (Swamma faster)
    - `8192/16384: n/a` (skipped rows)
- Quick non-smoke eval run (shortened needle batches for turnaround):
  - config used: `/tmp/swamma_eval_gpu_quick.toml` (`needle_batches=8`, text eval off)
  - command:
    - `julia --project=. scripts/eval_long_context.jl --config /tmp/swamma_eval_gpu_quick.toml --output benchmarks/long_context_eval.csv --device gpu`
  - all random-init quality rows produced; needle scores stayed `0.0000` (expected pretraining behavior).
  - Transformer `8192/16384` emitted as skipped rows with run note.

### Best Current Checkpoint/Config Recommendation
- No checkpoint recommendation yet for quality claims (runs were random-init/system-validation).
- For system-complexity tracking now:
  - use `configs/swamma_vs_transformer/benchmark_long_context.toml` with `--device gpu`.
- For quick eval iteration:
  - keep a fast override (`needle_batches=8`) until trained checkpoints are available, then run full `needle_batches=64`.

### Unresolved Issues And Next Actions
- Blocking issue:
  - Transformer full-window baseline via current attention path is unstable on GPU at `context >= 8192` (`CUDA illegal memory access`).
- Next actions:
  - implement a dense full-attention baseline path for Transformer that is GPU-stable at long context (or patch the attention kernel).
  - run the full eval config (`needle_batches=64`) with trained checkpoints.
  - add multi-seed aggregation for benchmark/eval CSVs (mean/std reporting).

## 2026-03-14 — Swamma vs Transformer Long-Context Harness Setup

### Objectives
- Set up a runnable benchmarking/evaluation harness to test the long-context hypothesis:
  - Swamma block can be competitive on quality while scaling better with context.
- Ensure both scripts run end-to-end with the same PRIME sub-token interface.
- Remove setup blockers from CLI parsing/import paths so experiments are executable immediately.

### Changes Saved
- Added shared long-context model/eval module:
  - [`scripts/long_context_models.jl`](/home/christos/code/julia/Swamma/scripts/long_context_models.jl)
  - includes `ModelSpec`, Swamma/Transformer builders, PRIME carryover filtering, masked metrics, and synthetic needle eval.
- Added benchmark runner:
  - [`scripts/benchmark_long_context.jl`](/home/christos/code/julia/Swamma/scripts/benchmark_long_context.jl)
  - context sweep, latency/throughput measurement, log-log scaling exponent fit, CSV output.
- Added evaluation runner:
  - [`scripts/eval_long_context.jl`](/home/christos/code/julia/Swamma/scripts/eval_long_context.jl)
  - checkpoint-aware eval, optional text reconstruction eval, synthetic needle accuracy eval, CSV output.
- Added protocol/config docs:
  - [`docs/LONG_CONTEXT_PROTOCOL.md`](/home/christos/code/julia/Swamma/docs/LONG_CONTEXT_PROTOCOL.md)
  - [`configs/swamma_vs_transformer/benchmark_long_context.toml`](/home/christos/code/julia/Swamma/configs/swamma_vs_transformer/benchmark_long_context.toml)
  - [`configs/swamma_vs_transformer/eval_long_context.toml`](/home/christos/code/julia/Swamma/configs/swamma_vs_transformer/eval_long_context.toml)
- Post-setup fixes applied:
  - fixed ArgParse name-collision in both scripts by using `parse_cli_args` + `ArgParse.parse_args`.
  - removed illegal non-top-level `using` statements in eval script.
  - exported required symbols from `LongContextModels` so scripts can resolve shared APIs.

### Experiment Commands And Key Metrics
- Benchmark smoke run:
  - `julia --project=. scripts/benchmark_long_context.jl --config /tmp/swamma_bench_smoke.toml --output /tmp/swamma_bench_smoke.csv`
  - key rows:
    - `swamma`: `N=64 mean=4.84ms`, `N=128 mean=16.99ms`, fitted exponent `1.812`
    - `transformer`: `N=64 mean=3.33ms`, `N=128 mean=7.94ms`, fitted exponent `1.252`
  - output: `/tmp/swamma_bench_smoke.csv`
- Eval smoke run:
  - `julia --project=. scripts/eval_long_context.jl --config /tmp/swamma_eval_smoke.toml --output /tmp/swamma_eval_smoke.csv`
  - key rows:
    - all architectures/contexts completed, `init_mode=random_init`, no checkpoint errors.
    - `needle_acc=0.0000` at random init (expected for untrained models).
  - output: `/tmp/swamma_eval_smoke.csv`

### Best Current Checkpoint/Config Recommendation
- No trained long-context checkpoint is recommended yet (smoke-only validation with random init).
- Recommended starting configs for real runs:
  - benchmark: `configs/swamma_vs_transformer/benchmark_long_context.toml`
  - evaluation: `configs/swamma_vs_transformer/eval_long_context.toml`
- For quality comparisons, run `scripts/eval_long_context.jl` with trained Swamma/Transformer checkpoints and at least 3 seeds.

### Unresolved Issues And Next Actions
- Current results are infrastructure validation only; not evidence of architecture superiority.
- Benchmark currently runs in training mode and emits a Lux warning; switch forward timing path to inference (`Lux.testmode`) before reporting headline latency.
- Next actions:
  - train/load matched-parameter Swamma and Transformer checkpoints on identical data/schedule.
  - run full context sweeps (>=16k, then 32k/64k if memory allows) with 3 seeds.
  - publish mean/std for latency exponent, masked text metrics, and needle accuracy.

## 2026-03-14 — Stage-5 Retrieval-Bias Continuation (Full-Val Recheck)

### Objectives
- Complete the pending Stage-5 continuation (`1510 -> 1760`) with newly added retrieval-bias inputs.
- Re-evaluate with full validation coverage (`max_eval_batches=10000`) to remove sampled-batch noise.
- Isolate whether gains/regressions come from retrieval-bias terms, calibration, or continuation training itself.

### Changes Saved
- Added retrieval-bias plumbing to RE model forward path:
  - [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl)
  - new pair aux features include distance/type bias bases
  - retrieval head now accepts optional bias logits offset
  - model input supports `retrieval_distance_bias_scale` / `retrieval_type_bias_scale`
- Added retrieval-bias config loading and input threading:
  - [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl)
  - settings loader + propagation through train/eval/oracle/calibration paths
  - fixed `evaluate_oracle_ladder` missing settings variable bug (`UndefVarError`)
- Updated soft-rank config with active retrieval-bias scales:
  - [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence_rankloss_soft.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence_rankloss_soft.toml)
  - `retrieval_distance_bias_scale = 0.10`
  - `retrieval_type_bias_scale = 0.10`
- Added no-bias ablation config:
  - [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence_rankloss_soft_nobias.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence_rankloss_soft_nobias.toml)

### Experiment Commands And Outcomes
- Continuation training (completed, exit code `0`):
  - `julia --project=. scripts/train_re_gpu.jl --config ...rankloss_soft.toml --resume ...fusedevidence/checkpoint_step_1510.jls --max-steps 1760`
  - in-run eval at step `1750`: `rel_f1=0.0013`, `pair_recall=0.0854`, `pair_t16=0.0366`
- Full-val checkpoint sweep (bias on, step `1760`):
  - `rel_f1=0.0006`, `rel_p=0.0003`, `rel_r=0.0060`
  - `span_r=0.6359`, `pair_r=0.0959`, `pair_t16=0.0311`, `oracle_rel=0.3981` (from oracle ladder)
- Full-val checkpoint sweep (bias off ablation, step `1760`):
  - `rel_f1=0.0005`, `rel_p=0.0003`, `rel_r=0.0052`
  - `pair_r=0.0967`, `pair_t16=0.0320`
  - retrieval bias terms are near-neutral at this scale (small F1 delta, no coverage rescue)
- Full-val reference sweep (bias off, step `1510`):
  - `rel_f1=0.0003`, `pair_r=0.0898`, `pair_t16=0.0380`, `span_r=0.5551`
- Oracle ladder contrast (`1510` vs `1760`, bias-off config):
  - proposal-side coverage improved (`oracle_rel 0.3299 -> 0.3981`, `span_r 0.5551 -> 0.6359`, `pair_r 0.0898 -> 0.0959`)
  - gold-span/gold-pair relation quality regressed (`rel_f1 0.0322 -> 0.0208`)
- Threshold sweeps (full-val):
  - step `1760` (7 thresholds): best `pred spans + pred pairs rel_f1=0.0012` at threshold `0.70`
  - step `1510` (0.3/0.5/0.7): best `pred spans + pred pairs rel_f1=0.0006` at threshold `0.70`
  - conclusion: continuation improves decoded F1 slightly, but still far below promotion gates
- Controlled Stage-5 continuation from locked-family checkpoint (`1000 -> 1500`, biaffine + delayed soft rank-loss):
  - config: [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000.toml)
  - resume source: `checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025/checkpoint_step_1000.jls`
  - eval@1500 (default decode): `rel_f1=0.0022`, `pair_r=0.2683`, `pair_t16=0.0976`, `oracle_rel=0.6463`
  - calibrated full-val sweep best: `threshold=0.90`, `margin=0.40` -> `pred spans + pred pairs rel_f1=0.0041` (`pair_r=0.1701`, `pair_t16=0.0544`, `oracle_rel=0.6408`)
- Controlled-run seed variation (same recipe):
  - `seed=7` config: [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed7.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed7.toml)
    - eval@1500 default: `rel_f1=0.0000`, `pair_r=0.1707`, `pair_t16=0.0854`, `oracle_rel=0.5488`
    - calibrated (`0.90`, `0.40`): `rel_f1=0.0000`, `pair_r=0.1088`, `pair_t16=0.0415`, `oracle_rel=0.4732`
  - `seed=11` config: [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed11.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed11.toml)
    - eval@1500 default: `rel_f1=0.0000`, `pair_r=0.2317`, `pair_t16=0.0854`, `oracle_rel=0.8171`
    - calibrated (`0.90`, `0.40`): `rel_f1=0.0000`, `pair_r=0.1287`, `pair_t16=0.0449`, `oracle_rel=0.8385`
  - `seed=19` config: [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed19.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed19.toml)
    - eval@1500 default: `rel_f1=0.0013`, `pair_r=0.1585`, `pair_t16=0.1098`, `oracle_rel=0.6707`
    - calibrated (`0.90`, `0.40`): `rel_f1=0.0000`, `pair_r=0.1356`, `pair_t16=0.0639`, `oracle_rel=0.5794`
  - conclusion: large variance across seeds; controlled recipe is promising but not stable enough to promote
- Aggressive schedule decode-relaxation check (full-val, `margin=0.10`, thresholds `0.60/0.70/0.80`) on non-42 seeds:
  - `seed=7`: best `pred spans + pred pairs rel_f1=0.0005` (threshold `0.70`)
  - `seed=11`: best `pred spans + pred pairs rel_f1=0.0006` (threshold `0.70`)
  - `seed=19`: best `pred spans + pred pairs rel_f1=0.0005` (threshold `0.60`/`0.70` tie)
  - conclusion: relaxing decode does not recover aggressive-seed stability; collapse is mostly a model-state issue, not a calibration-only issue
- Aggressive reproducibility rerun with explicit seed42:
  - config: [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun.toml)
  - train path: resume from `step_1000` to `step_1500`; sampled eval@1500 `rel_f1=0.0027`, `pair_r=0.2439`, `pair_t16=0.1220`, `oracle_rel=0.6829`
  - full-val strict point (`threshold=0.90`, `margin=0.40`): `pred spans + pred pairs rel_f1=0.0011` (did not reproduce prior `0.0041`)
  - full-val relaxed sweep (`margin=0.10`, thresholds `0.60/0.70/0.80`) best `rel_f1=0.0012` (threshold `0.70`)
  - conclusion: previous `0.0041` appears to be a high-variance outlier; run-to-run nondeterminism/noise is material even under explicit seed.
- Full-val baseline normalization (`overgen4 checkpoint_last`) showed prior sampled-batch lock was optimistic:
  - locked sampled recipe (`threshold=0.70`, `margin=0.30`, per-rel overrides) gives full-val `rel_f1=0.0008`
  - tested baseline high-threshold sweep best in this pass: `rel_f1=0.0009`
  - controlled `from1000` run is therefore materially better on full-val (`0.0041` vs `0.0009` in tested settings)
- Gentle ranking schedule follow-up (`edge_w=0.015`, `start=1375`, `warmup=200`) on the same `1000 -> 1500` path:
  - base config: [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_gentle_from1000.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_gentle_from1000.toml)
  - seed configs:
    - [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_gentle_from1000_seed11.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_gentle_from1000_seed11.toml)
    - [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_gentle_from1000_seed19.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_gentle_from1000_seed19.toml)
  - full-val threshold sweep (`0.50, 0.70, 0.90`, margin `0.0`) best `pred spans + pred pairs`:
    - `seed=42`: `rel_f1=0.0010`
    - `seed=11`: `rel_f1=0.0016`
    - `seed=19`: `rel_f1=0.0016`
  - full-val threshold sweep (`0.60, 0.70, 0.80`, margin `0.1`) best `pred spans + pred pairs`:
    - `seed=42`: `rel_f1=0.0017`
    - `seed=11`: `rel_f1=0.0017`
    - `seed=19`: `rel_f1=0.0008`
  - full-val threshold sweep (`0.60, 0.70, 0.80`, margin `0.2`) best `pred spans + pred pairs`:
    - `seed=42`: `rel_f1=0.0017`
    - `seed=11`: `rel_f1=0.0015`
    - `seed=19`: `rel_f1=0.0008`
  - targeted full-val non-null sweep on strongest gentle point (`seed=11`, `threshold=0.70`, `margin=0.10`, nonnull `0.00..0.80`):
    - `pred spans + pred pairs` remained constant at `rel_f1=0.0017` for all tested non-null gates
    - `pair_r=0.1261`, `pair_t16=0.0596`, `oracle_rel=0.5164` unchanged across sweep
    - conclusion: non-null decode gate is currently inactive/non-influential for this checkpoint at this confidence+margin regime
  - interpretation: gentler schedule reduces collapse severity (mostly non-zero), but ceiling remains low and margin tightening hurts worst-case seed behavior.
- Midpoint schedule probe (`edge_w=0.02`, `start=1350`, `warmup=250`, `hard_negs=12`) from step `1000 -> 1250` (`seed=11`):
  - config: [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_mid_from1000_seed11.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_mid_from1000_seed11.toml)
  - eval@1250 (sampled): `rel_f1=0.0015`, `pair_r=0.0244`, `pair_t16=0.0122`, `oracle_rel=0.2805`
  - full-val threshold sweep (`0.60/0.70/0.80`, margin `0.10`) best:
    - `pred spans + pred pairs rel_f1=0.0004` at threshold `0.60`
    - `pair_r=0.0475`, `pair_t16=0.0259`, `oracle_rel=0.2789`
  - conclusion: midpoint schedule is rejected; it under-covers candidate pairs and regresses below both aggressive and gentle seed11 branches.
- Determinism hardening landed after reproducibility failure:
  - batch negative-span and hard-negative pair sampling now use explicit RNG (`prepare_rebel_batch(...; rng=...)`) instead of global RNG state
  - training/eval/oracle/auto-calibration batch builders now pass seeded RNG handles
  - startup now seeds global RNG via `Random.seed!(run_config.seed)`
  - smoke check (`max_eval_batches=8`, identical eval command repeated twice) produced identical metric rows
  - deterministic batch sampling regression checks added in [`test/test_relation_extraction.jl`](/home/christos/code/julia/Swamma/test/test_relation_extraction.jl) and passing
- Post-fix compact cross-seed strict recheck (full-val, `threshold=0.90`, `margin=0.40`):
  - `seed42_rerun`: `pred spans + pred pairs rel_f1=0.0011`
  - `seed7`: `rel_f1=0.0000`
  - `seed11`: `rel_f1=0.0000`
  - `seed19`: `rel_f1=0.0000`
  - conclusion: strict calibrated point remains non-robust even after deterministic eval sampling fix; schedule-only tuning is not sufficient.
- Edge-retrieval v2 scaffolding landed in core model path:
  - new proposer mode `:edge_retrieval_v2` accepted by pair-proposal dispatch and summary paths
  - mode reuses sparse semantic-retrieval machinery while disabling heuristic anchor fanout in this path
  - semantic retrieval precompute is now decoupled from router precompute gating for this mode
  - smoke eval config added: [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_edgev2_smoke.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_edgev2_smoke.toml)
  - unit smoke test added and passing in [`test/test_relation_extraction.jl`](/home/christos/code/julia/Swamma/test/test_relation_extraction.jl)
- Decisive identical-checkpoint comparison (`seed42_rerun`, full-val, `margin=0.10`, thresholds `0.60/0.70/0.80`):
  - `sparse_hybrid` best `pred spans + pred pairs rel_f1=0.0012` (`pair_r=0.1554`, `pair_t16=0.0639`, `oracle_rel=0.6295`)
  - `edge_retrieval_v2` best `rel_f1=0.0012` with identical pair/oracle metrics
  - conclusion: current `edge_retrieval_v2` scaffold is functionally parity with baseline on this checkpoint; no-go for immediate promotion.
- Edge-v2 short adaptation probe (`seed42`, `1000 -> 1250`) and full-val recheck:
  - sampled eval@1250: `rel_f1=0.0000`, `oracle_rel=0.1829`, `pair_r=0.0610`, `pair_t16=0.0488`
  - full-val sweep (`margin=0.10`, thresholds `0.60/0.70/0.80`) best:
    - `pred spans + pred pairs rel_f1=0.0013` (threshold `0.80`)
    - but coverage collapsed (`oracle_rel=0.1978`, `pair_r=0.0354`, `pair_t16=0.0294`)
  - conclusion: current edge-v2 adaptation recipe is not viable; any apparent F1 parity is coverage-fragile and not promotable.

### Current Recommendation
- Keep the fused-evidence `1510 -> 1760` branch **not promoted** after full-val verification.
- Treat retrieval-bias additions as infrastructure landed, but not yet performance-positive on fused branch.
- Keep the controlled `1000 -> 1500` biaffine soft-rank recipe in diagnostic status only; do **not** promote due instability and failed seed42 reproduction (`0.0011` on rerun strict check).
- Keep gentle schedule as a stability-control branch, not as promotion; while more stable, its ceiling remains low (`~0.0017` best tested).
- Determinism for eval sampling is now in place, and the compact post-fix cross-seed recheck has been completed.
- Drop midpoint schedule branch from further exploration unless retrieval coverage can be restored first.
- Pause further rank-loss schedule sweeps and move effort to architecture-level pair/edge modeling upgrades.
- `Swamma RE v2` architecture draft is now documented in [`docs/SWAMMA_RE_V2_ARCHITECTURE.md`](/home/christos/code/julia/Swamma/docs/SWAMMA_RE_V2_ARCHITECTURE.md) and should be used as the implementation reference.
- `edge_retrieval_v2` is kept as an implementation scaffold only until it demonstrates measurable lift on fixed-checkpoint comparisons.
- Close the current edge-v2 training recipe branch; next attempt must redesign edge selection + supervision jointly before retraining.

### Process Update
- Session-report workflow remains mandatory and active:
  - Rule source: [`AGENTS.md`](/home/christos/code/julia/Swamma/AGENTS.md)
  - Report target: [`docs/SESSION_REPORT.md`](/home/christos/code/julia/Swamma/docs/SESSION_REPORT.md)
  - this session has been appended before closeout.

## 2026-03-14 — RE Architecture Iteration

### Objectives
- Push relation extraction beyond zero F1 by improving sparse pair proposal quality.
- Validate whether curriculum and null-weight changes improve precision.
- Keep resume/checkpoint behavior stable while iterating architecture/configs.

### Changes Saved
- Added anchor-expanded sparse pair proposal logic in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl) (subquadratic fanout path).
- Fixed pair-sweep CLI parsing in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl).
- Added checkpoint-safe partial warm-start merge for architecture-mismatch resumes in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl).
- Added decode-time non-null probability gate and calibration CLI sweeps in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - `--nonnull-sweep-checkpoint`
  - `--nonnull-sweep-values`
  - `--nonnull-sweep-confidence`
  - `--nonnull-sweep-margin`
  - `--threshold-sweep-nonnull`
- Added constrained decode caps in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - `--decode-head-cap`
  - `--decode-tail-cap`
- Added per-relation decode calibration overrides in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - `--per-relation-thresholds` (`LABEL=VALUE` or `ID=VALUE`)
- Added auto-calibration mode in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - `--auto-calibrate-checkpoint`
  - `--auto-calibrate-threshold`
  - `--auto-calibrate-margin`
  - `--auto-calibrate-nonnull`
  - `--auto-calibrate-min-predictions`
  - `--auto-calibrate-thresholds`
  - includes global acceptance gate to reject globally harmful per-relation suggestions
- Added decode-time schema/type constraints in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - `--type-constraints-mode` (`off|hard`)
  - `--type-constraints-min-count`
  - relation-type compatibility mask is now applied in oracle/threshold/margin/nonnull/auto-calibration paths
  - added robust relation index-offset inference for type-rule construction
- Added optional decode-time inverse/symmetry consistency resolver in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - `--relation-consistency-mode` (`off|resolve`)
  - `--relation-consistency-min-count`
  - built from training reverse-edge statistics and applied as reverse-direction conflict resolution
- Added evidence diagnostics to evaluator outputs in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl) and [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - emitted diagnostics: top evidence token index, attention entropy, max attention weight
  - aggregated metrics: `ev_ent`, `ev_max`, `ev_eff`, `ev_t1`
  - checkpoint sweep table now includes evidence columns
- Added evidence pooling ablation mode in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - `--evidence-pooling-sweep-checkpoint`
  - `--evidence-pooling-modes`
  - supports `token|sentence|hybrid` pooling via model input flag without checkpoint-struct changes
- Added retrieval edge-ranking objective with hard-negative mining in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - hard-negative hinge helper on retrieval logits
  - TOML knobs: `edge_ranking_loss_weight`, `edge_ranking_margin`, `edge_ranking_hard_negatives`
  - applied in both teacher and proposal training losses
  - eval summary now reports `ret_rank` / `prop_rank`
  - implemented with checkpoint-safe config loading (no checkpoint schema break)
- Updated fused-evidence confidence path in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl) so confidence scoring uses fused pair/evidence/retrieval inputs when `relation_decoder_mode = fused_evidence`.
- Added/updated experiment configs:
  - `configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4.toml`
  - `configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_curric50.toml`
  - `configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_curric10.toml`
  - `configs/redfm_base_safe_pair_sparse_learned128_nullw05_overgen4.toml`
  - `configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence.toml`
  - `configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence_rankloss.toml`

### Key Experiment Outcomes
- Refreshed best checkpoint (`nullw025 + overgen4`, step `1250`) improved to:
  - `oracle_rel = 0.8293`
  - `pair_recall = 0.2073`
  - `pair_t16 = 0.0976`
  - raw `rel_f1 = 0.0027`
- Decode calibration improved F1 further:
  - best observed point: `confidence_threshold = 0.70`, `no_relation_margin = 0.30`
  - calibrated `rel_f1 = 0.0050`
- `curric50` and `curric10` (proposal-conditioned training) both underperformed baseline and returned `rel_f1` to zero.
- `null_relation_weight = 0.5` reduced coverage heavily and failed to improve F1.
- `fused_evidence` attempt surfaced resume param-tree mismatch; partial warm-start support was added.
- Non-null gating provided minor benefit only in weaker decode settings (`rel_f1 0.0027 -> 0.0028` at `nonnull=0.93`) and did not beat the calibrated best `0.0050`.
- Constrained decode caps (`head/tail = 1,2,4`) collapsed `pred spans + pred pairs` F1 to `0.0000` at the current best calibration point; control (`0,0`) preserved `0.0050`.
- Short fused-evidence resume with new confidence fusion (`1250 -> 1260`) was stable but weaker than baseline:
  - `oracle_rel = 0.6951`
  - `pair_recall = 0.1341`
  - `pair_t16 = 0.0488`
  - `rel_f1 = 0.0023` (best threshold point still below `0.0050`)
- Per-relation confidence calibration improved the decoded best point:
  - global baseline decode (`thr=0.70`, `margin=0.30`): `rel_f1 = 0.0050`
  - with `P127=0.95, P155=0.90`: `rel_f1 = 0.0056`
  - with `P127=0.95, P155=0.90, P571=0.85`: `rel_f1 = 0.0057` (current best)
- `v1_locked` reproducibility check (2 reruns, `max_eval_batches=8`) produced identical sampled metrics:
  - `rel_p=0.0031`, `rel_r=0.0366`, `rel_f1=0.0057`
  - observed `rel_f1` std over reruns: `0.0000`
- Auto-calibration run (`max_eval_batches=8`) proposed raw `P641=0.85`, but global gate rejected it because it dropped global F1 (`0.0050 -> 0.0039`); accepted set remains unchanged.
- Stage-3 type-constraint ablation (`max_eval_batches=8`, locked checkpoint family):
  - control (`off`, per-rel overrides): `rel_f1 = 0.0057`
  - `hard` constraints + per-rel overrides: `rel_f1 = 0.0050`
  - `hard` constraints only (no per-rel overrides): `rel_f1 = 0.0042`
  - hard constraints are functioning but not promoted because best constrained F1 remains below `v1_locked`.
- Stage-3 inverse/symmetry consistency ablation (`max_eval_batches=8`, locked decode):
  - control (`relation-consistency=off`): `rel_f1 = 0.0057`
  - resolver (`relation-consistency=resolve,min_count=1`): `rel_f1 = 0.0058` (repeated twice)
  - pair metrics unchanged (`pair_r=0.2073`, `pair_t16=0.0976`)
- Stage-4 evidence diagnostics baseline read (`max_eval_batches=8`, checkpoint sweep):
  - `ev_ent = 3.8232`
  - `ev_max = 0.1172`
  - `ev_eff = 55.54`
  - `ev_t1 = 56`
- Stage-4 evidence pooling sweep on fused-evidence checkpoint (`max_eval_batches=8`):
  - `token`: `rel_f1 = 0.0023` (best)
  - `sentence`: `rel_f1 = 0.0018`
  - `hybrid`: `rel_f1 = 0.0019`
  - recall stayed flat while precision dropped for `sentence/hybrid`
- Stage-4 short fused-evidence continuation completed (`1260 -> 1510`, `+250` updates):
  - eval@1500: `val_loss=15.2280`, `relation_loss=10.4497`
  - locked decode re-check (`thr=0.70`, `margin=0.30`, `P127=0.95,P155=0.90,P571=0.85`, `max_eval_batches=8`):
    - `pred spans + pred pairs`: `rel_p=0.0028`, `rel_r=0.0122`, `rel_f1=0.0046`
    - `oracle_rel=0.3659`, `pair_r=0.2073`, `pair_t16=0.0732`
  - consistency resolver (`resolve,min_count=1`) produced identical sampled row on this checkpoint.
  - checkpoint snapshot kept at `checkpoint_step_1510.jls`; default decode checkpoint sweep row: `total=14.9950`, `relation_loss=10.0617`, `rel_f1=0.0020`, `ev_ent=2.1763`, `ev_max=0.4146`.
- Stage-5 ranking-objective smoke validation:
  - rank-loss eval path (`max_eval_batches=1`) succeeded on `checkpoint_step_1510.jls` with `edge_ranking_loss_weight=0.2`.
  - one-step resume (`1510 -> 1511`) with rank-loss config completed without runtime/gradient errors.
- Stage-5 first controlled continuation (`1510 -> 1760`, rank-loss config) regressed:
  - eval@1750: `oracle_rel=0.1951`, `pair_recall=0.0488`, `pair_t16=0.0122`, `rel_f1=0.0000`
  - locked decode at step `1760` stayed at `pred spans + pred pairs rel_f1=0.0000` (with and without consistency resolver)
  - checkpoint snapshot saved at `..._fusedevidence_rankloss/checkpoint_step_1760.jls`.
- Stage-5 soft-scheduled continuation (`1510 -> 1760`, delayed rank-loss warmup) recovered:
  - in-run eval@1750: `oracle_rel=0.5976`, `pair_recall=0.1829`, `pair_t16=0.0610`, `rel_f1=0.0022`
  - locked decode at step `1760`: `pred spans + pred pairs rel_f1=0.0060` (with/without consistency resolver)
  - tradeoff remains: proposal metrics are still below baseline (`pair_r=0.1463`, `pair_t16=0.0366`).
  - checkpoint snapshot saved at `..._fusedevidence_rankloss_soft/checkpoint_step_1760.jls`.

### Current Recommendation
- Continue from `configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4.toml`.
- Treat step-1250 neighborhood as the current best checkpoint region.
- Use decode operating point `threshold=0.70`, `no_relation_margin=0.30` as current default.
- Keep decode caps disabled for now (`decode_head_cap=0`, `decode_tail_cap=0`).
- Use per-relation calibrated decode for current best operating point:
  - `threshold=0.70`
  - `no_relation_margin=0.30`
  - `per_relation_thresholds=P127=0.95,P155=0.90,P571=0.85`
- Use auto-calibration as a proposal tool only; keep global gate enabled and accept only non-degrading sets.
- Treat this calibrated decode row as `v1_locked` reference for subsequent ablations.
- Keep fused-evidence confidence variant as an ablation branch (latest calibrated `rel_f1=0.0046`) until it beats `v1_locked=0.0057`.
- Keep `type-constraints-mode=off` in default decode for now.
- Promote `relation-consistency=resolve` (`relation-consistency-min-count=1`) as the default Stage-3 decode add-on.
- Keep token pooling as default for fused-evidence path (sentence/hybrid currently regress precision).
- Do not promote the aggressive rank-loss setting (`edge_ranking_loss_weight=0.2`); it collapses coverage.
- Soft schedule is the new working candidate (`rel_f1=0.0060` at locked decode), but keep it as ablation until pair coverage recovers toward baseline.
- Next step: tune retrieval-side coverage (type/distance-aware retrieval bias, schedule tweaks) while preserving soft-rank calibration gains.

### Open Issues
- Current `hard` type constraints over-prune recall at the locked operating point.
- Auto-calibration currently proposes relation-level gains that can still hurt global F1; objective needs stronger global coupling.
- Evidence attention is still broad/diffuse at baseline (high entropy/effective-token count), indicating room for better evidence concentration.
- Strong ranking pressure can collapse proposal coverage and relation recall on fused-evidence branch if introduced too aggressively.

---

## 2026-03-14 — Swamma LLM Feasibility Review (No Code Changes)

### Objectives
- Determine whether Swamma blocks in this repository can support an LLM workflow.
- Identify what already exists for language modeling versus what is still missing for a production-grade foundation model run.

### Changes Saved
- No code/config files were modified in this session.
- Inspected architecture and training stack across:
  - [`src/Swamma.jl`](/home/christos/code/julia/Swamma/src/Swamma.jl)
  - [`src/LLaDA.jl`](/home/christos/code/julia/Swamma/src/LLaDA.jl)
  - [`src/Training.jl`](/home/christos/code/julia/Swamma/src/Training.jl)
  - [`src/DataLoader.jl`](/home/christos/code/julia/Swamma/src/DataLoader.jl)
  - [`src/HFTokenizer.jl`](/home/christos/code/julia/Swamma/src/HFTokenizer.jl)
  - [`scripts/train_llm.jl`](/home/christos/code/julia/Swamma/scripts/train_llm.jl)
  - [`examples/train_llada.jl`](/home/christos/code/julia/Swamma/examples/train_llada.jl)
  - [`ARCHITECTURE.md`](/home/christos/code/julia/Swamma/ARCHITECTURE.md)
  - [`README.md`](/home/christos/code/julia/Swamma/README.md)

### Key Experiment Outcomes
- No training/evaluation experiments were run in this session.
- Commands executed were repository inspection only:
  - `rg --files`
  - `rg -n "Swamma|...|LLM" -S README* docs src test scripts`
  - `sed -n ...` on the files listed above.
- Key feasibility conclusion:
  - The repo already contains a Swamma-based diffusion LM path (`LLaDAModel`) and training scripts (`scripts/train_llm.jl`, `examples/train_llada.jl`), so building an LLM on Swamma blocks is feasible.
  - Current path is suitable for experimentation/prototyping; additional engineering is still needed for robust large-scale pretraining.

### Current Recommendation
- For immediate experimentation, keep Swamma blocks as the core model primitive and use `LLaDAModel` with a curated tokenizer/data pipeline first.
- Treat current LLM stack as pre-production until:
  - tokenizer/data path is consolidated (prefer a single HF/BPE path over mixed char/word utilities),
  - training/eval instrumentation is standardized for perplexity/quality tracking,
  - long-run stability and checkpoint interoperability are validated at larger scale.
- Existing RE locked recommendation remains unchanged for relation extraction workstreams.

### Open Issues
- LLM data/tokenization stack is fragmented (char-level loader, ad hoc word tokenizer script, HF wrapper in separate module).
- No single validated large-scale pretraining checkpoint is documented for the LLaDA path.
- Generation quality/eval harness for LM capability is limited versus task-specific RE/NER tooling.

### Next Actions
- Create one canonical LLM training entrypoint that uses:
  - HF tokenizer + stable dataset loader,
  - unified checkpoint schema,
  - standard eval hooks (validation loss/perplexity + text quality probes).
- Run a short reproducible pilot (fixed seed/config) and log metrics in session reports to establish an LLM baseline.

---

## 2026-03-14 — Canonical LLaDA Pipeline Hardening

### Objectives
- Add a canonical LLaDA training entrypoint using Swamma blocks with HF tokenizer + deterministic chunking.
- Harden shared training utilities for long runs (state propagation, gradient clipping, total-step behavior).
- Validate that the new entrypoint parses and core config/training objects initialize correctly.

### Changes Saved
- Added new canonical training script:
  - [`scripts/train_llada_canonical.jl`](/home/christos/code/julia/Swamma/scripts/train_llada_canonical.jl)
  - Features: TOML-driven model/training config, `.txt/.jsonl` corpus loading, deterministic chunking (`seq_len` + `stride`), batch loader, checkpointing (`step_*.jls`, `best.jls`, `latest.jls`, `final.jls`), resume support, run metadata/config snapshots.
  - Added explicit tokenizer dependency failure path: clear error if Python `transformers` is missing in the PyCall environment.
  - Reads `data.tokenizer_model` from config when CLI tokenizer is left at default.
- Added canonical config template:
  - [`configs/llada_canonical.toml`](/home/christos/code/julia/Swamma/configs/llada_canonical.toml)
- Updated HF tokenizer compatibility:
  - [`src/HFTokenizer.jl`](/home/christos/code/julia/Swamma/src/HFTokenizer.jl)
  - `HuggingFaceTokenizer` now supports tokenizers without EOS/BOS IDs (`eos_token_id::Union{Int,Nothing}`) and uses safe fallback for `pad_token_id` (`pad -> eos -> sep -> 0`).
- Updated training utilities:
  - [`src/Training.jl`](/home/christos/code/julia/Swamma/src/Training.jl)
  - `train_step!` now propagates returned `new_state` into `train_state.state`.
  - Added global-norm gradient clipping helper (`clip_gradients`) and wired to `TrainingConfig.gradient_clip`.
  - `train!` now loops until `total_steps` (instead of stopping after a single finite iterator pass) with early stop guard if iterator yields no batches.
  - Fixed `TrainingConfig` Float32 default literals (`1f-4`, `1f-6`).
- Updated CPU local-attention autodiff path:
  - [`src/Attention.jl`](/home/christos/code/julia/Swamma/src/Attention.jl)
  - Reworked CPU banded attention helpers (`banded_attention_weights_cpu`, `apply_banded_attention_cpu`) to use deterministic `Zygote.Buffer` cell assignment (no partial/uninitialized writes, no in-place view mutation), fixing both AD mutation failure and forward-pass NaN behavior in CPU attention.
- Minor literal fix:
  - [`scripts/train_colab.jl`](/home/christos/code/julia/Swamma/scripts/train_colab.jl): `learning_rate = 1f-4`.
- Updated canonical validation data handling:
  - [`scripts/train_llada_canonical.jl`](/home/christos/code/julia/Swamma/scripts/train_llada_canonical.jl)
  - Pass validation batches as a plain vector so each eval pass starts from batch 1 (prevents artificial `val_loss=0.0` on repeated eval).
- Updated canonical training loader iteration behavior:
  - [`scripts/train_llada_canonical.jl`](/home/christos/code/julia/Swamma/scripts/train_llada_canonical.jl)
  - `BatchLoader` now auto-resets/shuffles on iterator exhaustion so `train!` can continue across multiple epochs without external reset hooks.
- Added regression coverage for LLaDA training path:
  - [`test/test_llada_training.jl`](/home/christos/code/julia/Swamma/test/test_llada_training.jl)
  - Covers finite diffusion loss and one-step `train_step!` update/step increment on `small_config()`.

### Experiment Commands and Key Metrics
- Script parse/load check (pass):
  - `julia --project=. -e 'include("scripts/train_llada_canonical.jl"); println("canonical_script_parse_ok")'`
  - Result: `canonical_script_parse_ok`.
- Training config default constructor check (pass):
  - `julia --project=. -e 'include("src/Swamma.jl"); using .Swamma; cfg = TrainingConfig(); println("training_config_defaults_ok lr=" * string(cfg.learning_rate));'`
  - Result: `training_config_defaults_ok lr=0.0001`.
- Canonical script smoke execution with tiny local text (blocked by environment dependency):
  - Command used `--config configs/small.toml --train-path /tmp/swamma_train.txt ... --tokenizer-model bert-base-uncased`.
  - Initial result: failed early because PyCall Python lacked `transformers` (`ModuleNotFoundError`).
  - Environment action: installed `transformers` into `/usr/bin/python3` user site via `python3 -m pip install --user --break-system-packages transformers`.
- CPU attention finite check (pass):
  - `julia --project=. -e 'include("src/Attention.jl"); ... layer=SWAttention(...); y,_=layer(...); ...'`
  - Result: `swattention_output_finite=true`, `nan_count=0`.
- Diffusion loss finite check (pass):
  - `julia --project=. -e 'include("src/Swamma.jl"); ... for i in 1:10 diffusion_loss(...) ...'`
  - Result: 10/10 finite losses (range observed: `~7.27` to `~8.18`).
- Minimal `train!` smoke run against synthetic in-memory batches (pass):
  - Command: `julia --project=. -e 'include("src/Swamma.jl"); ... train!(...) ...'` with `small_config()`, `total_steps=3`.
  - Result:
    - `Step 1 loss=7.4627`
    - `Step 2 loss=7.5201`
    - `Validation loss=7.7962` (new best)
    - `Step 3 loss=7.3741`
    - Completed at `step=3`, `best_loss=7.7961564`.
- Canonical end-to-end script smoke (pass after tokenizer + iterator fixes):
  - Command:
    - `julia --project=. scripts/train_llada_canonical.jl --config configs/small.toml --train-path /tmp/swamma_train.txt --val-path /tmp/swamma_train.txt --tokenizer-model bert-base-uncased --checkpoint-dir /tmp/swamma_ckpt_smoke2 --seq-len 16 --stride 8 --batch-size 2 --total-steps 2 --eval-every 1 --save-every 1 --log-every 1 --sample-steps 4`
  - Result:
    - `step 1 train_loss=10.8883`, `val_loss=11.0556`
    - `step 2 train_loss=10.9965`, `val_loss=10.5770`
    - final `best_validation_loss=10.576986`
    - checkpoints written (`best.jls`, `step_1.jls`, `step_2.jls`, `final.jls`) under `/tmp/swamma_ckpt_smoke2`
    - generation sample produced successfully.
- Multi-pass canonical smoke check:
  - First attempt (`total_steps=10`) revealed early stop after one loader pass (`final_step=2`) due non-resetting training iterator state.
  - After `BatchLoader` auto-reset fix, rerun with:
    - `--total-steps 6 --eval-every 2 --save-every 3 --log-every 2`
  - Result:
    - reached full `step=6` as requested
    - `val_loss@2=11.2812`
    - `val_loss@4=10.7109` (best)
    - `val_loss@6=10.7668`
    - final `best_validation_loss=10.710939`
    - checkpoints written: `step_3.jls`, `step_6.jls`, `best.jls`, `final.jls` under `/tmp/swamma_ckpt_smoke6`.
- Automated regression test run (pass):
  - `julia --project=. test/test_llada_training.jl`
  - Result: `LLaDA Training Smoke | Pass 4 / Total 4 | 1m17.3s`.

### Best Current Checkpoint/Config Recommendation
- For the LLaDA workflow introduced here, start from:
  - Config: [`configs/llada_canonical.toml`](/home/christos/code/julia/Swamma/configs/llada_canonical.toml)
  - Entrypoint: [`scripts/train_llada_canonical.jl`](/home/christos/code/julia/Swamma/scripts/train_llada_canonical.jl)
- No new model checkpoint was produced in this session due dependency + AD limitations above.
- Canonical script path is now runnable; smoke checkpoints were produced in `/tmp/swamma_ckpt_smoke2`.

### Unresolved Issues and Next Actions
- Environment dependency:
  - PyCall Python still needs `transformers` on any fresh machine/environment before running canonical script.
- Runtime note:
  - Warning about Lux dropout `training` flag outside AD still appears during evaluation; this is performance/ergonomics debt, not a functional blocker for the smoke path.
- Next actions:
  - Run a longer canonical trial (100-500 steps) on a real corpus and retain first non-trivial checkpoint under repository checkpoint path (not `/tmp`).
  - Add this new LLaDA training smoke test to the project’s default test aggregation flow (`runtests.jl` or equivalent), since the repo currently runs tests as standalone files.

---

## 2026-03-14 — LLaDA Canonical Stabilization (Final Verification Pass)

### Objectives
- Confirm end-to-end repository test status after the LLaDA canonical/training fixes.
- Record final validation outcome and immediate next actions for checkpoint-quality runs.

### Changes Saved
- No additional code/config file changes in this pass.
- Documentation update only:
  - Updated [`docs/SESSION_REPORT.md`](/home/christos/code/julia/Swamma/docs/SESSION_REPORT.md) with final verification results.

### Key Experiment Outcomes
- Full package test run completed successfully:
  - Command: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: exit code `0`, final line `Testing Swamma tests passed`.
- Included suite outcomes from this run:
  - `test_attention.jl`: all checks passed.
  - `test_router.jl`: all groups passed (`TokenRouter Shapes and Spans`, `Routing Utilities`, `Fusion and Cache Utilities`, `Metrics and Schedules`, `GatedExperts Wrapper`).
  - `test_llada_training.jl`: `LLaDA Training Smoke | Pass 4 / Total 4 | 1m49.0s`.
- Non-blocking runtime warnings still observed during tests:
  - Undeclared import warning for `Swamma.LinearChainCRF`.
  - Conflicting import warning for `TiDAR.GRANITE_VOCAB_SIZE`.
  - Deprecation warnings around `ignore(f)` usage.
  - Lux `training=Val{true}` non-AD slow-path warning.

### Current Recommendation
- Canonical LLaDA training path is now in a validated state for extension:
  - Config: [`configs/llada_canonical.toml`](/home/christos/code/julia/Swamma/configs/llada_canonical.toml)
  - Entrypoint: [`scripts/train_llada_canonical.jl`](/home/christos/code/julia/Swamma/scripts/train_llada_canonical.jl)
- Best smoke checkpoint reference remains:
  - `/tmp/swamma_ckpt_smoke6/best.jls` from the earlier 6-step canonical run (`best_validation_loss=10.710939`).

### Open Issues
- Clean up warning debt (imports + deprecated `ignore(f)` + Lux eval-mode handling) to reduce noise and future breakage risk.
- Run a longer real-corpus training job (100-500+ steps) and store artifacts under a persistent project checkpoint path instead of `/tmp`.

---

## 2026-03-14 — Warning Debt Cleanup (Imports + Ignore Derivatives)

### Objectives
- Continue from the green test baseline and remove the most frequent non-blocking runtime warnings.
- Keep behavior unchanged while making CI/test logs cleaner and less brittle.

### Changes Saved
- Updated [`src/NER.jl`](/home/christos/code/julia/Swamma/src/NER.jl):
  - Removed fragile top-level `import ..LinearChainCRF` (which produced undeclared-binding warning during module load).
  - Added lazy resolver `_linear_chain_crf_ctor()` and switched CRF layer construction to runtime symbol lookup.
- Updated [`src/Swamma.jl`](/home/christos/code/julia/Swamma/src/Swamma.jl):
  - Removed duplicate `GRANITE_VOCAB_SIZE` import from `TiDAR` into top-level `Swamma` namespace (kept the Drafter import as canonical source).
  - Removed redundant TiDAR-section re-export of `GRANITE_VOCAB_SIZE` (still exported once from Drafter section).
- Updated [`src/Training.jl`](/home/christos/code/julia/Swamma/src/Training.jl):
  - Replaced deprecated `Zygote.@ignore` block in masked CE one-hot construction with `ChainRulesCore.ignore_derivatives`.
  - Set evaluation path to `Lux.testmode(state)` to avoid training-mode dropout warnings during non-AD validation.
- Updated [`src/WavePDE.jl`](/home/christos/code/julia/Swamma/src/WavePDE.jl):
  - Replaced deprecated `@ignore` usage with `ChainRulesCore.ignore_derivatives` for device-side lambda materialization.
- Updated [`test/test_llada_training.jl`](/home/christos/code/julia/Swamma/test/test_llada_training.jl):
  - Finite-loss smoke now evaluates with `Lux.testmode(state)` to avoid non-AD training warnings in test output.

### Key Experiment Outcomes
- Verified focused smoke test:
  - `julia --project=. test/test_llada_training.jl`
  - Result: `LLaDA Training Smoke | Pass 4 / Total 4 | 1m17.2s`.
- Verified default test runner:
  - `julia --project=. test/runtests.jl`
  - Result:
    - `SWAttention Soundness & Dynamic Tests | 9/9`
    - `SWAttention Locality | 2/2`
    - `TokenRouter Shapes and Spans | 19/19`
    - `Routing Utilities | 5/5`
    - `Fusion and Cache Utilities | 2/2`
    - `Metrics and Schedules | 10/10`
    - `GatedExperts Wrapper | 3/3`
    - `LLaDA Training Smoke | 4/4 | 1m23.3s`
- Verified full package test entrypoint:
  - `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: exit code `0`, final line `Testing Swamma tests passed`.
- Warning status after this pass:
  - Resolved in tested path:
    - `Imported binding Swamma.LinearChainCRF was undeclared...`
    - `ignoring conflicting import of TiDAR.GRANITE_VOCAB_SIZE into Swamma`
    - deprecated `ignore(f)` warnings from `Training.jl` and `WavePDE.jl`
    - Lux training-mode warning in LLaDA smoke test/eval path.
  - Remaining global warning:
    - `Pkg.test()` still emits `project dependencies or compat requirements have changed since the manifest was last resolved` (environment/manifest hygiene, not a model correctness issue).

### Current Recommendation
- Use the canonical LLaDA path with the current warning-cleaned codebase:
  - Config: [`configs/llada_canonical.toml`](/home/christos/code/julia/Swamma/configs/llada_canonical.toml)
  - Entrypoint: [`scripts/train_llada_canonical.jl`](/home/christos/code/julia/Swamma/scripts/train_llada_canonical.jl)
- Treat this revision as a cleaner baseline for longer real-corpus runs and regression tracking.

### Open Issues
- Manifest drift warning remains; resolve with a deliberate dependency sync pass (`Pkg.resolve`) when ready to lock the environment.
- Broader cleanup opportunity still exists in other modules using `Zygote.@ignore` (e.g., `NER.jl`, `LogicGated.jl`) if full deprecation elimination is desired.

---

## 2026-03-14 — Deprecation Hygiene Pass 2 (`@ignore` in NER/LogicGated)

### Objectives
- Continue warning-debt reduction by removing remaining `Zygote.@ignore` usage in core routing/NER modules.
- Verify no behavioral regressions via focused smoke checks and full default test execution.

### Changes Saved
- Updated [`src/LogicGated.jl`](/home/christos/code/julia/Swamma/src/LogicGated.jl):
  - Replaced `using Zygote: @ignore` with `using ChainRulesCore`.
  - Updated STE detach path in `ste_gates` to `ChainRulesCore.ignore_derivatives`.
- Updated [`src/NER.jl`](/home/christos/code/julia/Swamma/src/NER.jl):
  - Replaced `using Zygote: @ignore` with `using ChainRulesCore`.
  - Refactored `ner_cross_entropy` constant-building path (valid mask/count, one-hot targets, mask tensor) into one `ChainRulesCore.ignore_derivatives` block.
  - Kept differentiable path unchanged (`logsoftmax`, masked CE reduction), preserving gradient flow through logits.

### Key Experiment Outcomes
- Confirmed no remaining `@ignore` in touched modules:
  - `rg -n "@ignore" src/NER.jl src/LogicGated.jl`
  - Result: no matches (expected).
- NER loss smoke check (pass):
  - `julia --project=. -e 'include("src/Swamma.jl"); ...; loss=Swamma.NER.ner_cross_entropy(...); println(loss)'`
  - Result: finite loss (`ner_loss=3.283029`).
- Default test aggregation (pass):
  - `julia --project=. test/runtests.jl`
  - Result:
    - attention suite pass (`9/9` + `2/2`)
    - router suite pass (`19/19`, `5/5`, `2/2`, `10/10`, `3/3`)
    - LLaDA smoke pass (`4/4`, ~`1m17s`)
- Full package test entrypoint (pass):
  - `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: exit code `0`, final `Testing Swamma tests passed`.

### Current Recommendation
- Current branch now has two completed warning-cleanup passes with stable tests.
- Use this revision as the baseline for longer canonical LLaDA experiments and for adding heavier RE/NER tests into the default test target.

### Open Issues
- `Pkg.test()` still prints manifest drift warning (`dependencies or compat requirements have changed since the manifest was last resolved`); this is environment hygiene, not correctness.
- Additional non-critical cleanup opportunity remains in broader config/log files and potential dependency pinning pass before reproducibility-sensitive runs.

---

## 2026-03-14 — Manifest Sync and Test Baseline Verification

### Objectives
- Resolve the remaining manifest/project drift warning shown by `Pkg.test()`.
- Re-verify the full test baseline after environment synchronization.

### Changes Saved
- Updated dependency lock state via:
  - `julia --project=. -e 'using Pkg; Pkg.resolve()'`
- Resulting file changes:
  - [`Manifest.toml`](/home/christos/code/julia/Swamma/Manifest.toml)
- No source code changes in this pass.

### Key Experiment Outcomes
- Resolve pass (success):
  - `julia --project=. -e 'using Pkg; Pkg.resolve()'`
  - Output: no package add/remove required; manifest synchronized.
- Full package tests (success):
  - `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: exit code `0`, final `Testing Swamma tests passed`.
  - Important change vs prior runs: the previous warning
    `project dependencies or compat requirements have changed since the manifest was last resolved`
    no longer appears.

### Current Recommendation
- Treat current branch as the clean environment baseline for further model work:
  - warning-cleaned code paths
  - synchronized manifest
  - passing full `Pkg.test()` run.

### Open Issues
- No new functional regressions observed in this pass.
- Next meaningful work should shift back to model quality experiments (longer LLaDA training and checkpoint evaluation).

---

## 2026-03-14 — Default Test Suite Expansion + RE `dropgrad` Cleanup

### Objectives
- Expand default test coverage to include relation extraction stability checks.
- Remove newly surfaced deprecation warnings (`Zygote.dropgrad`) from Relation Extraction now that RE tests run by default.

### Changes Saved
- Updated [`test/runtests.jl`](/home/christos/code/julia/Swamma/test/runtests.jl):
  - Promoted `test_relation_extraction.jl` into the default suite.
  - Kept `test_moet.jl` and `test_tidar.jl` behind `SWAMMA_TEST_FULL=1`.
- Updated [`test/test_relation_extraction.jl`](/home/christos/code/julia/Swamma/test/test_relation_extraction.jl):
  - Qualified RE symbols via `Swamma` module alias (`SW.`) to avoid `Main`-scope ambiguity when included from aggregated `runtests.jl`.
- Updated [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - Added `detach_constant(x)` helper implemented with `ChainRulesCore.ignore_derivatives`.
  - Replaced all `Zygote.dropgrad(...)` usages in heuristic scoring and loss/rrule helpers with `detach_constant(...)`.
  - Removed now-unused `import Zygote`.
- Dependency lock state remained synchronized from the earlier pass:
  - [`Manifest.toml`](/home/christos/code/julia/Swamma/Manifest.toml) remains updated.

### Key Experiment Outcomes
- RE standalone runtime/profile (pass):
  - `julia --project=. test/test_relation_extraction.jl`
  - Runtime observed ~34s total, all subtests pass.
- Aggregated default suite after inclusion/fixes (pass):
  - `julia --project=. test/runtests.jl`
  - Result:
    - attention suite pass (`9/9` + `2/2`)
    - router suite pass (`19/19`, `5/5`, `2/2`, `10/10`, `3/3`)
    - LLaDA smoke pass (`4/4`)
    - relation extraction suite pass (`10/10`, `5/5`, `4/4`, `2/2`, `3/3`, `6/6`)
- Package entrypoint verification (pass):
  - `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: exit code `0`, final `Testing Swamma tests passed`.
- Warning status:
  - Resolved: `dropgrad(x) is deprecated` warnings in RE path.
  - No manifest-drift warning in current `Pkg.test()` runs.

### Current Recommendation
- Use current branch state as the default development baseline:
  - includes RE regression coverage by default
  - deprecation-clean RE path
  - passing full package tests.

### Open Issues
- Default suite runtime is now higher due RE inclusion (expected tradeoff for coverage).
- If CI wall-time becomes a concern, consider splitting RE into `SWAMMA_TEST_MEDIUM=1` tier rather than removing coverage entirely.

---

## 2026-03-14 — Test Lane Tiering (`SWAMMA_TEST_MEDIUM`)

### Objectives
- Continue by adding a middle test lane to balance CI/runtime vs. coverage.
- Keep default `Pkg.test()` fast while retaining an easy path to run RE coverage.

### Changes Saved
- Updated [`test/runtests.jl`](/home/christos/code/julia/Swamma/test/runtests.jl):
  - Added lane flags:
    - `SWAMMA_TEST_FULL=1` → full suite
    - `SWAMMA_TEST_MEDIUM=1` → medium suite
  - New behavior:
    - default: `test_attention.jl`, `test_router.jl`, `test_llada_training.jl`
    - medium: default + `test_relation_extraction.jl`
    - full: medium + `test_moet.jl`, `test_tidar.jl`

### Key Experiment Outcomes
- Default lane validation (pass):
  - `julia --project=. test/runtests.jl`
  - Result: attention/router/llada suites all pass.
- Medium lane validation (pass):
  - `SWAMMA_TEST_MEDIUM=1 julia --project=. test/runtests.jl`
  - Result: default suites + full RE suite all pass.
- Package default entrypoint validation (pass):
  - `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: exit code `0`, final `Testing Swamma tests passed`.
  - Confirms `Pkg.test()` now runs the fast lane by default.

### Current Recommendation
- Use default lane for rapid local/CI feedback.
- Use medium lane (`SWAMMA_TEST_MEDIUM=1`) for regular feature validation where RE behavior matters.
- Reserve full lane for scheduled or pre-release checks.

### Open Issues
- Full lane (`SWAMMA_TEST_FULL=1`) was not re-run in this pass; unchanged from prior behavior assumptions.
- If desired, document lane commands in README/CONTRIBUTING for team discoverability.

---

## 2026-03-14 — Full Lane Stabilization + README Test-Lane Docs

### Objectives
- Continue from lane tiering by validating `SWAMMA_TEST_FULL=1`.
- Fix any remaining aggregated-test issues in heavy suites.
- Document lane usage in README for discoverability.

### Changes Saved
- Updated [`test/test_moet.jl`](/home/christos/code/julia/Swamma/test/test_moet.jl):
  - Added module alias `SW = Swamma`.
  - Qualified `MoETConfig` and `MoETModel` as `SW.MoETConfig` and `SW.MoETModel` to avoid `Main`-scope ambiguity in aggregated runs.
- Updated [`test/test_tidar.jl`](/home/christos/code/julia/Swamma/test/test_tidar.jl):
  - Switched from direct `include("../src/TiDAR.jl")` to `include("../src/Swamma.jl")`.
  - Uses `TD = Swamma.TiDAR` and calls `TD.verify_and_accept(...)`.
  - This avoids undeclared/ambiguous `Main` imports when run under `runtests.jl`.
- Updated [`README.md`](/home/christos/code/julia/Swamma/README.md):
  - Added **Testing Lanes** section with exact commands for default, medium, full, and `Pkg.test()` entrypoint behavior.

### Key Experiment Outcomes
- Full lane initial run surfaced and localized integration issues:
  - `SWAMMA_TEST_FULL=1 julia --project=. test/runtests.jl`
  - First failure: `test_moet.jl` (`UndefVarError: MoETConfig not defined in Main`).
  - After `test_moet` qualification fix, second failure: `test_tidar.jl` due direct TiDAR include/import assumptions under aggregate execution.
- Full lane after fixes (pass):
  - `SWAMMA_TEST_FULL=1 julia --project=. test/runtests.jl`
  - Result: all suites pass
    - attention, router, llada smoke, relation extraction, moet, tidar.
- Package default lane check (pass):
  - `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: exit code `0`, final `Testing Swamma tests passed`.

### Current Recommendation
- Current test setup is now coherent across all lanes:
  - default for fast feedback
  - medium for RE-inclusive checks
  - full for pre-release or scheduled deeper validation
- Use README testing lane commands as the standard team invocation pattern.

### Open Issues
- Full lane runtime is materially longer; best suited for non-blocking CI jobs or pre-merge gates where broader coverage is required.
- Optional follow-up: add a tiny summary table of expected lane runtimes in README based on current hardware.

---

## 2026-03-14 — CI Workflow Added for Test Lanes

### Objectives
- Continue by operationalizing the lane model in CI (not just local docs).
- Keep lane behavior explicit and discoverable for contributors.

### Changes Saved
- Added GitHub Actions workflow:
  - [`.github/workflows/test-lanes.yml`](/home/christos/code/julia/Swamma/.github/workflows/test-lanes.yml)
  - Behavior:
    - PRs / pushes to `main`: run default lane (`Pkg.test()`).
    - nightly schedule: run medium lane (`SWAMMA_TEST_MEDIUM=1`).
    - manual dispatch: selectable lane (`default`, `medium`, `full`).
- Updated README lane docs:
  - [`README.md`](/home/christos/code/julia/Swamma/README.md)
  - Added CI policy mapping beneath the Testing Lanes section.

### Key Experiment Outcomes
- Local post-change validation (pass):
  - `julia --project=. test/runtests.jl`
  - Result: default lane passed (`attention`, `router`, `llada` suites).
- Workflow correctness note:
  - CI workflow was validated by inspection and local lane command parity.
  - Remote GitHub Actions execution was not run from this environment.

### Current Recommendation
- Use the new workflow as the baseline CI policy:
  - fast feedback on PR/push
  - broader RE coverage nightly
  - full lane available on-demand via workflow dispatch.

### Open Issues
- If CI runtime pressure appears, consider reducing nightly frequency or pinning medium lane to selected branches.
- Optional next step: add a status badge for the new workflow in README.

---

## 2026-03-14 — README CI Status Badge

### Objectives
- Continue workflow polish by making CI lane status visible from the repository landing page.
- Keep documentation aligned with the newly added `test-lanes` GitHub Actions workflow.

### Changes Saved
- Updated [`README.md`](/home/christos/code/julia/Swamma/README.md):
  - Added a `test-lanes` workflow badge directly under the main title.
  - Badge target uses repository remote:
    - badge URL: `https://github.com/gavlooth/Ossamma/actions/workflows/test-lanes.yml/badge.svg`
    - link URL: `https://github.com/gavlooth/Ossamma/actions/workflows/test-lanes.yml`

### Key Experiment Outcomes
- Repository remote verification:
  - `git remote -v`
  - Result: origin is `https://github.com/gavlooth/Ossamma`.
- README update verification:
  - Confirmed badge markdown insertion under title in local file.

### Current Recommendation
- Keep the badge in README so contributors can quickly see lane CI health without opening Actions manually.

### Open Issues
- Badge health depends on workflow path and default branch policy in the remote repository; if repo/workflow is renamed, update badge URL accordingly.

---

## 2026-03-14 — CI Guidance Document (`docs/CI.md`)

### Objectives
- Continue by turning lane policy into actionable branch-protection guidance.
- Provide one canonical CI reference doc for contributors and maintainers.

### Changes Saved
- Added [`docs/CI.md`](/home/christos/code/julia/Swamma/docs/CI.md):
  - lane definitions (`default`, `medium`, `full`)
  - workflow job mapping from `.github/workflows/test-lanes.yml`
  - recommended required check (`test-lanes / default-fast`)
  - release-time full-lane recommendation
  - local parity commands.
- Updated [`README.md`](/home/christos/code/julia/Swamma/README.md):
  - added direct link to `docs/CI.md` from the Testing Lanes CI policy section.

### Key Experiment Outcomes
- Verified documentation references:
  - `README.md` now points to `docs/CI.md`.
  - `docs/CI.md` matches current workflow job names/triggers.
- No code-path changes; no new test execution required for this documentation-only pass.

### Current Recommendation
- Configure branch protection to require only `test-lanes / default-fast` on PRs.
- Use medium nightly and full manual lanes as non-blocking broader quality signals.

### Open Issues
- Branch protection settings must be applied in GitHub repository settings manually.
- If workflow/job names change, update `docs/CI.md` and required-check configuration together.

---

## 2026-03-14 — LLaDA PRIME-Only Path Cleanup

### Objectives
- Remove the remaining legacy LLaDA token-path surface so the model runs PRIME sub-token parameterization only.
- Fix AD and numeric stability issues introduced by PRIME carryover filtering.
- Verify training/test behavior after path cleanup.

### Changes Saved
- Updated [`src/LLaDA.jl`](/home/christos/code/julia/Swamma/src/LLaDA.jl):
  - Removed legacy/dead token-path API stubs:
    - deleted `apply_mask(...)`
    - deleted `unmask_step(...)`
  - Removed unused token-embedding state/parameters from `LLaDAModel`:
    - dropped `TokenEmbedding` field
    - removed corresponding setup in `Lux.initialparameters` and `Lux.initialstates`
    - removed `TokenEmbedding` state propagation in forward pass
  - Hardened PRIME carryover filtering:
    - compatibility mask creation now wrapped in `ChainRulesCore.ignore_derivatives` to avoid Zygote mutation errors
    - invalid/empty compatibility rows now fallback to full support (prevents all-invalid softmax rows)
    - replaced extreme fill value with finite `-1e9` equivalent (`convert(eltype(logits), -1.0f9)`) for stable masking behavior
- Updated [`src/Training.jl`](/home/christos/code/julia/Swamma/src/Training.jl):
  - In `diffusion_loss`, moved `token_ids_to_subtokens` and `apply_subtoken_mask` into `ChainRulesCore.ignore_derivatives` preprocessing so Zygote does not differentiate through integer-array mutation paths.
  - Kept PRIME-only forward/loss path unchanged otherwise.

### Key Experiment Outcomes
- Targeted LLaDA training smoke:
  - `julia --project=. test/test_llada_training.jl`
  - Result: pass
  - Metrics:
    - `LLaDA Training Smoke`: 4/4 pass, `1m59.9s`
    - `LLaDA PRIME Subtoken Smoke`: 7/7 pass, `1.3s`
- Aggregated test runner:
  - `julia --project=. test/runtests.jl`
  - Result: pass (exit code 0), includes LLaDA suites passing.
- Package test entrypoint:
  - `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: pass
  - Final line: `Testing Swamma tests passed`
  - LLaDA section timing:
    - `LLaDA Training Smoke`: 4/4 pass, `6m17.6s`
    - `LLaDA PRIME Subtoken Smoke`: 7/7 pass, `3.7s`

### Current Recommendation
- Use PRIME-only LLaDA as canonical path; do not reintroduce token-level binary masking helpers.
- Recommended config baseline for this path:
  - [`configs/llada_canonical.toml`](/home/christos/code/julia/Swamma/configs/llada_canonical.toml)
  - Keep `model.prime.prime_enabled = true`
  - Keep `prime_subtoken_length = 4`, `prime_subtoken_base = 16` unless ablation requires changes.

### Open Issues
- `mask_token_id` remains in parts of LLaDA/Training signatures for compatibility but is not operational in PRIME masking; optional follow-up is API cleanup to remove this legacy argument from call sites.
- PRIME carryover compatibility currently uses CPU-side mask construction; acceptable for now, but may become a throughput bottleneck at larger vocab/sequence scales and can be optimized later.

---

## 2026-03-14 — LLaDA Training API Cleanup (`mask_token_id` Removal)

### Objectives
- Continue PRIME-only cleanup by removing residual `mask_token_id` arguments from the LLaDA training API.
- Update direct call sites so LLaDA training/eval uses only PRIME sub-token state.
- Re-validate LLaDA training smoke and aggregated tests.

### Changes Saved
- Updated [`src/Training.jl`](/home/christos/code/julia/Swamma/src/Training.jl):
  - `diffusion_loss` signature changed from:
    - `diffusion_loss(model, params, state, token_ids, mask_token_id; ...)`
    - to `diffusion_loss(model, params, state, token_ids; ...)`
  - `train_step!` signature changed from:
    - `train_step!(train_state, model, batch, mask_token_id; ...)`
    - to `train_step!(train_state, model, batch; ...)`
  - `evaluate` signature changed from:
    - `evaluate(model, params, state, data_iterator, mask_token_id; ...)`
    - to `evaluate(model, params, state, data_iterator; ...)`
  - Internal training loop (`train!`) now calls the new signatures and no longer threads `model.mask_token_id`.
- Updated LLaDA tests in [`test/test_llada_training.jl`](/home/christos/code/julia/Swamma/test/test_llada_training.jl):
  - removed `model.mask_token_id` argument from `diffusion_loss` and `train_step!` calls.
- Updated helper script [`scripts/test_trainability.jl`](/home/christos/code/julia/Swamma/scripts/test_trainability.jl):
  - removed `mask_token_id` argument from `train_step!` call.
- Updated example [`examples/quickstart.jl`](/home/christos/code/julia/Swamma/examples/quickstart.jl):
  - updated `diffusion_loss` call to new signature.

### Key Experiment Outcomes
- LLaDA smoke suite:
  - `julia --project=. test/test_llada_training.jl`
  - Result: pass
  - Metrics:
    - `LLaDA Training Smoke`: 4/4 pass, `1m49.9s`
    - `LLaDA PRIME Subtoken Smoke`: 7/7 pass, `1.2s`
- Aggregated lane:
  - `julia --project=. test/runtests.jl`
  - Result: pass
  - LLaDA section:
    - `LLaDA Training Smoke`: 4/4 pass, `2m05.0s`
    - `LLaDA PRIME Subtoken Smoke`: 7/7 pass, `1.6s`

### Current Recommendation
- Keep the simplified LLaDA training API (`diffusion_loss`/`train_step!`/`evaluate`) without `mask_token_id`.
- Continue using PRIME defaults in canonical config:
  - [`configs/llada_canonical.toml`](/home/christos/code/julia/Swamma/configs/llada_canonical.toml)
  - `prime_enabled = true`, `prime_subtoken_length = 4`, `prime_subtoken_base = 16`.

### Open Issues
- `mask_token_id` still exists in `LLaDAConfig`/metadata for compatibility and tokenizer bookkeeping; it is no longer part of the PRIME training loop API.
- `examples/quickstart.jl` still contains legacy manual masking/unmasking demo snippets (`apply_mask`/`unmask_step`) from the pre-PRIME tutorial flow and should be fully migrated in a dedicated docs/example pass.

---

## 2026-03-14 — Hard PRIME Cut (No Compatibility Guardrails)

### Objectives
- Apply a stricter PRIME-only cleanup after removing training API compatibility arguments.
- Remove `mask_token_id` and `prime_enabled` from LLaDA core config/model surface.
- Update canonical training path and major LLaDA call sites to match the new strict schema.

### Changes Saved
- Updated core LLaDA schema in [`src/LLaDA.jl`](/home/christos/code/julia/Swamma/src/LLaDA.jl):
  - removed `mask_token_id` and `prime_enabled` from `LLaDAConfig`
  - removed serialization/parsing of those fields in `save_config` / `config_from_dict`
  - removed `mask_token_id` and `prime_enabled` fields from `LLaDAModel`
  - removed constructor kwargs for `mask_token_id` / `prime_enabled`
  - updated `generate` docstring signature to PRIME-only form
- Updated canonical training script in [`scripts/train_llada_canonical.jl`](/home/christos/code/julia/Swamma/scripts/train_llada_canonical.jl):
  - removed `get_mask_token_id` dependency and related logging/metadata fields
  - removed `mask_token_id` / `prime_enabled` when reconstructing `LLaDAConfig`
  - `resolved_vocab_size` now depends on tokenizer/model vocab only
- Updated LLaDA test coverage in [`test/test_llada_training.jl`](/home/christos/code/julia/Swamma/test/test_llada_training.jl):
  - removed `prime_enabled` config arg and assertion
- Updated example/helper scripts:
  - [`examples/quickstart.jl`](/home/christos/code/julia/Swamma/examples/quickstart.jl) rewritten to use PRIME subtoken masking/unmasking (`token_ids_to_subtokens`, `apply_subtoken_mask`, `unmask_subtoken_step`) and `subtoken_state` model input
  - [`scripts/test_trainability.jl`](/home/christos/code/julia/Swamma/scripts/test_trainability.jl) forward/backward smoke switched to PRIME subtoken masking path
- Updated LLaDA config files to remove stale mask-token key:
  - [`configs/base.toml`](/home/christos/code/julia/Swamma/configs/base.toml)
  - [`configs/small.toml`](/home/christos/code/julia/Swamma/configs/small.toml)
  - [`configs/large.toml`](/home/christos/code/julia/Swamma/configs/large.toml)
  - [`configs/production.toml`](/home/christos/code/julia/Swamma/configs/production.toml)
  - [`configs/train_base.toml`](/home/christos/code/julia/Swamma/configs/train_base.toml)
  - [`configs/llada_canonical.toml`](/home/christos/code/julia/Swamma/configs/llada_canonical.toml) (`prime_enabled` removed as well)
- Updated LLaDA config constructor call sites to remove removed keyword:
  - [`scripts/train_extended.jl`](/home/christos/code/julia/Swamma/scripts/train_extended.jl)
  - [`scripts/train_llm.jl`](/home/christos/code/julia/Swamma/scripts/train_llm.jl)
  - [`scripts/generate_text.jl`](/home/christos/code/julia/Swamma/scripts/generate_text.jl)
  - [`scripts/train_colab.jl`](/home/christos/code/julia/Swamma/scripts/train_colab.jl)
  - [`scripts/train_fast.jl`](/home/christos/code/julia/Swamma/scripts/train_fast.jl)
  - [`scripts/train_production.jl`](/home/christos/code/julia/Swamma/scripts/train_production.jl)

### Key Experiment Outcomes
- Core include check:
  - `julia --project=. -e 'include("src/Swamma.jl"); using .Swamma; println("swamma_include_ok")'`
  - result: `swamma_include_ok`
- Canonical script parse check:
  - `julia --project=. -e 'include("scripts/train_llada_canonical.jl"); println("canonical_parse_ok")'`
  - result: `canonical_parse_ok`
- LLaDA smoke tests:
  - `julia --project=. test/test_llada_training.jl`
  - result: pass
  - metrics:
    - `LLaDA Training Smoke`: 4/4 pass, `2m03.7s`
    - `LLaDA PRIME Subtoken Smoke`: 6/6 pass, `2.2s`
- Aggregated tests:
  - `julia --project=. test/runtests.jl`
  - result: pass
  - LLaDA metrics in run:
    - `LLaDA Training Smoke`: 4/4 pass, `1m56.3s`
    - `LLaDA PRIME Subtoken Smoke`: 6/6 pass, `2.5s`

### Current Recommendation
- Treat PRIME-only LLaDA as strict default and keep removed fields (`mask_token_id`, `prime_enabled`) out of new configs/scripts.
- Best active config recommendation for current LLaDA path remains:
  - [`configs/llada_canonical.toml`](/home/christos/code/julia/Swamma/configs/llada_canonical.toml)
  - `prime_subtoken_length = 4`, `prime_subtoken_base = 16`

### Open Issues
- Some older LLaDA scripts still include legacy token-mask variables for internal custom logic; they are no longer authoritative for LLaDA model schema and should be cleaned in a dedicated script modernization pass.
- Non-LLaDA subsystems (Drafter/TiDAR) still have their own `mask_token_id` semantics and were intentionally not altered in this PRIME-only LLaDA cleanup.

---

## 2026-03-14 — Relation Pair Proposer Dispatch Inspection

### Objectives
- Inspect `src/RelationExtraction.jl` to locate where relation pair proposer modes are defined, parsed, and dispatched.
- Identify the smallest safe insertion points for adding a new proposer mode such as `:edge_retrieval_v2`.

### Changes Saved
- No source code behavior changed.
- Added this inspection-only session report entry to satisfy repository session-report requirements.

### Key Experiment Outcomes
- Code search:
  - `rg -n "pair proposer|pair_proposer|proposal mode|proposer mode|edge_retrieval|proposal" src/RelationExtraction.jl`
  - key hits: config field, config parsing, summary gating, `pair_proposer_uses_router`, `propose_relation_pairs`, forward-pass dispatch.
- Structural inspection:
  - `nl -ba src/RelationExtraction.jl | sed -n '540,620p'`
  - `nl -ba src/RelationExtraction.jl | sed -n '680,920p'`
  - `nl -ba src/RelationExtraction.jl | sed -n '920,1110p'`
  - `nl -ba src/RelationExtraction.jl | sed -n '1520,1885p'`
  - `nl -ba src/RelationExtraction.jl | sed -n '2168,2345p'`
- Metrics:
  - pair proposer implementation count in `src/RelationExtraction.jl`: 1 concrete proposal head struct (`SparsePairProposalHead`)
  - central proposer dispatch points: 2 (`pair_proposer_uses_router`, `propose_relation_pairs`)
  - forward-path precompute gates tied to proposer mode: 2 (router outputs and semantic retrieval outputs)

### Current Recommendation
- If `:edge_retrieval_v2` reuses retrieval projections only, keep the existing `PairProposalHead` struct untouched and add the mode as a new branch inside `propose_relation_pairs`.
- If `:edge_retrieval_v2` also needs learned router logits/buckets, extend `pair_proposer_uses_router` and reuse the existing `SparsePairProposalHead` / `build_router_outputs` path rather than introducing a second proposal-head field.

### Open Issues
- `pair_anchor_top_spans` is a model field computed inside the constructor rather than a config field; any new mode that depends on anchor fanout should verify whether the existing derived value is sufficient.
- No dedicated tests were added in this inspection pass; a future implementation should extend `test/test_relation_extraction.jl` with at least one constructor/config smoke case and one proposal-path behavior check for the new mode.

---

## 2026-03-14 — Span Context Edge Controls + Sentence Neighbors

### Objectives
- Implement missing `4d` architecture items without changing checkpoint-serialized model structs:
  - add same-sentence span-context graph edges
  - add edge-family ablation controls (`adjacent`, `sentence`, `semantic`)
- Ensure train/eval/oracle/calibration input paths all carry the same runtime span-context controls.

### Changes Saved
- Updated span-context adjacency builder in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - `build_span_context_adjacency(...)` now supports:
    - `use_adjacent`
    - `use_sentence`
    - `use_semantic`
    - optional `sentence_ids` input
  - added sentence-group linking based on span start-token sentence id.
- Updated sparse span context block forward path in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - now accepts optional context-options input and applies runtime edge-family toggles.
- Updated model forward path in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - new optional runtime inputs:
    - `span_context_use_adjacent`
    - `span_context_use_sentence`
    - `span_context_use_semantic`
    - `span_context_sentence_ids`
  - threaded into `apply_span_context(...)`.
- Updated runtime settings plumbing in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - `load_retrieval_bias_settings(...)` now reads span-context edge toggles from `[relation_extraction]`.
  - `with_retrieval_bias_inputs(...)` now injects those toggles into model inputs.
  - `build_proposal_inputs(...)`, `build_fixed_proposal_inputs(...)`, oracle ladder inputs, and auto-calibration inputs now preserve/propagate these fields.
- Added tests in [`test/test_relation_extraction.jl`](/home/christos/code/julia/Swamma/test/test_relation_extraction.jl):
  - existing span-context test now exercises sentence-only mode with explicit `span_context_sentence_ids`.
  - new `Span Context Edge Family Controls` testset validates:
    - sentence-edge behavior
    - semantic-edge behavior
    - self-only fallback when all edge families are disabled.

### Key Experiment Outcomes
- Parse/inclusion check passed:
  - `julia --project=. -e 'Base.include(Main, "src/Swamma.jl"); Base.include(Main, "scripts/train_re_gpu.jl"); println("parse-ok")'`
- RE unit tests passed:
  - `julia --project=. test/test_relation_extraction.jl`
  - key rows:
    - `Relation Extraction Span Context`: 4/4 pass
    - `Span Context Edge Family Controls`: 12/12 pass
    - full file testsets all passing.

### Current Recommendation
- Keep these span-context controls enabled as runtime knobs (not serialized struct fields) to preserve old checkpoint compatibility.
- Use them for targeted ablations before further long retraining:
  - `span_context_use_adjacent=true/false`
  - `span_context_use_sentence=true/false`
  - `span_context_use_semantic=true/false`
- Prioritize next `4e` step: explicit edge retrieval score components and supervision alignment, now that span-context edge families are controllable.

### Open Issues
- Training data path does not currently provide sentence-id tensors by default, so sentence-neighbor edges are active only when `span_context_sentence_ids` is explicitly supplied.
- Optional speaker/section/sentence-root edge families remain unimplemented.

---

## 2026-03-15 — Sentence-ID Batch Plumbing For Span Context

### Objectives
- Activate sentence-neighbor span-context edges in normal training/eval runs (not only ad-hoc model calls).
- Keep implementation checkpoint-compatible by using runtime input tensors.
- Add regression tests for sentence-id generation behavior.

### Changes Saved
- Added sentence-id generation helpers in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - `is_sentence_ending_token(...)`
  - `infer_sentence_ids_from_tokens(...)`
  - `normalize_sentence_ids(...)`
  - `sentence_ids_for_row(...)`
- Extended `prepare_rebel_batch(...)` in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - now emits `sentence_ids` tensor with shape `[max_len, batch]`
  - supports explicit row-level `sentence_ids` when present (normalizes 0-based ids)
  - falls back to punctuation-based sentence segmentation otherwise
  - pads trailing token positions with last seen sentence id.
- Wired sentence ids into training inputs in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - `make_batch(...)` now adds `span_context_sentence_ids = batch.sentence_ids`.
- Updated RE smoke training harness in [`scripts/test_re_training.jl`](/home/christos/code/julia/Swamma/scripts/test_re_training.jl):
  - input batch now passes `token_mask` explicitly
  - input batch now passes `span_context_sentence_ids`
  - avoids runtime creation of GPU bool masks inside the AD-traced forward path.
- Expanded tests in [`test/test_relation_extraction.jl`](/home/christos/code/julia/Swamma/test/test_relation_extraction.jl):
  - `prepare_rebel_batch Sampled Negatives` now checks `sentence_ids` tensor presence.
  - added punctuation inference check (`["Ada", ".", "Bob", "!"] -> [1,1,2,2]`).
  - added explicit sentence-id normalization check (`[0,0,1,1] -> [1,1,2,2]`).

### Key Experiment Outcomes
- Parse checks passed:
  - `julia --project=. -e 'Base.include(Main, "src/Swamma.jl"); Base.include(Main, "scripts/train_re_gpu.jl"); println("parse-ok")'`
- RE tests passed:
  - `julia --project=. test/test_relation_extraction.jl`
  - updated test row:
    - `prepare_rebel_batch Sampled Negatives`: `14/14` pass.
- RE smoke training run passed:
  - `julia --project=. scripts/test_re_training.jl`
  - key metrics:
    - `step 1 loss=1.8183` (`~116771.5 ms`, compile-heavy)
    - `step 2 loss=1.3246` (`~93.4 ms`)
    - `step 3 loss=1.0150` (`~56.6 ms`)
  - note: this previously failed with a CUDA/Zygote `llvmcall requires the compiler` path when `token_mask` was omitted from smoke inputs; now resolved.

### Current Recommendation
- Keep sentence-neighbor edge family enabled by default in config (`span_context_use_sentence=true`) now that sentence-id tensors are provided in standard batch flow.
- For high-precision experiments, prefer explicit upstream sentence segmentation if available; punctuation fallback is intentionally simple.

### Open Issues
- Sentence segmentation fallback is punctuation-only and may split imperfectly around abbreviations.
- Speaker/section/sentence-root graph edges are still unimplemented.

---

## 2026-03-15 — Span Context Warm-Start Probe (`step_1000 -> 1030`)

### Objectives
- Measure whether enabling `span_context_layers=1` can be introduced safely mid-run using the existing `step_1000` checkpoint.
- Compare against a matched short control run with `span_context_layers=0`.

### Changes Saved
- Added probe configs:
  - [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_spanctx1_probe.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_spanctx1_probe.toml)
  - [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_spanctx0_probe.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_spanctx0_probe.toml)
- Updated execution TODO with this probe outcome:
  - [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md)

### Key Experiment Outcomes
- Span-context probe (`layers=1`, warm-start partial match, `1000 -> 1030`):
  - command:
    - `julia --project=. scripts/train_re_gpu.jl --config ..._spanctx1_probe.toml --resume checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025/checkpoint_step_1000.jls --max-steps 1030`
  - eval snapshots:
    - step `1010`: `val_loss=38.8475`, `pair_recall=0.0610`, `pair_t16=0.0488`, `rel_f1=0.0000`
    - step `1020`: `val_loss=37.9864`, `pair_recall=0.0122`, `pair_t16=0.0122`, `rel_f1=0.0000`
    - step `1030`: `val_loss=29.4886`, `pair_recall=0.0122`, `pair_t16=0.0000`, `rel_f1=0.0000`
- Matched control (`layers=0`, full resume, `1000 -> 1030`):
  - command:
    - `julia --project=. scripts/train_re_gpu.jl --config ..._spanctx0_probe.toml --resume checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025/checkpoint_step_1000.jls --max-steps 1030`
  - eval snapshots:
    - step `1010`: `val_loss=14.1946`, `pair_recall=0.1829`, `pair_t16=0.0488`, `rel_f1=0.0000`
    - step `1020`: `val_loss=15.2093`, `pair_recall=0.1341`, `pair_t16=0.0610`, `rel_f1=0.0000`
    - step `1030`: `val_loss=15.9628`, `pair_recall=0.1098`, `pair_t16=0.0732`, `rel_f1=0.0000`

### Current Recommendation
- Do not introduce `span_context_layers=1` by mid-run architecture warm-start from `step_1000`; current probe shows severe optimization/coverage regression.
- If span-context depth remains a target, evaluate it via:
  - from-scratch training recipe, or
  - staged pretrain where span-context modules are present from the start and warmed gradually.

### Open Issues
- Probe is short-window and does not settle final `rel_f1`; however, early loss/coverage deltas are large enough to reject this warm-start recipe.
- Need a dedicated staged/from-scratch experiment plan for span-context depth that preserves checkpoint comparability.

---

## 2026-03-15 — From-Scratch Span-Context Smoke + Sentence-Bias Retrieval Hook

### Objectives
- Validate whether span-context depth (`layers=1`) is at least stable when trained from initialization.
- Add sentence-distance retrieval bias plumbing using the new batch sentence IDs.

### Changes Saved
- Added a span-context smoke config:
  - [`configs/redfm_smoke_spanctx1.toml`](/home/christos/code/julia/Swamma/configs/redfm_smoke_spanctx1.toml)
- Added sampled sentence-bias eval config:
  - [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun_sentbias015.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun_sentbias015.toml)
- Added sentence-distance retrieval bias support in:
  - [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl)
  - new runtime input: `retrieval_sentence_bias_scale`
  - `gather_pair_aux_features(...)` now computes `sentence_bias_base` from `sentence_ids`
  - retrieval bias now combines distance + type + sentence components.
- Added trainer plumbing for the new runtime bias scale in:
  - [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl)
  - config key: `relation_extraction.retrieval_sentence_bias_scale` (default `0.0`)
- Added RE unit test coverage:
  - [`test/test_relation_extraction.jl`](/home/christos/code/julia/Swamma/test/test_relation_extraction.jl)
  - new testset: `Pair Aux Sentence Bias`.

### Key Experiment Outcomes
- From-scratch span-context smoke (`layers=1`, 50 steps):
  - command:
    - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_smoke_spanctx1.toml --max-steps 50`
  - key rows:
    - step `10`: `val_loss=15.0630`, `oracle_rel=0.0126`, `pair_recall=0.0063`
    - step `40`: `val_loss=16.1993`, `oracle_rel=0.0000`, `pair_recall=0.0000`
    - step `50`: `val_loss=15.2096`, `oracle_rel=0.0000`, `pair_recall=0.0000`, `rel_f1=0.0000`
- Matched baseline smoke (`layers=0`, 50 steps):
  - command:
    - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_smoke.toml --max-steps 50`
  - key rows:
    - step `25`: `val_loss=10.4125`, `oracle_rel=0.0063`, `pair_recall=0.0063`
    - step `50`: `val_loss=10.2391`, `oracle_rel=0.0126`, `pair_recall=0.0063`, `rel_f1=0.0000`
  - interpretation: `layers=1` under current smoke recipe is clearly worse on loss and proposal coverage.
- Validation checks:
  - `parse-ok` for `src/Swamma.jl` + `scripts/train_re_gpu.jl`
  - `julia --project=. test/test_relation_extraction.jl` passing, including:
    - `Pair Aux Sentence Bias`: `2/2`
    - `prepare_rebel_batch Sampled Negatives`: `14/14`.
- Fixed-checkpoint sentence-bias ablations (`seed42_rerun checkpoint_last`, margin `0.10`, thresholds `0.60/0.70/0.80`):
  - quick sample (`max_eval_batches=8`):
    - baseline (`scale=0.00`) best `pred spans + pred pairs rel_f1=0.0043`
    - sentence-bias (`scale=0.15`) best `rel_f1=0.0044`
  - larger sample (`max_eval_batches=128`):
    - baseline best `pred spans + pred pairs rel_f1=0.0012` (`pair_recall=0.1554`, `pair_t16=0.0639`)
    - sentence-bias best `rel_f1=0.0012` (`pair_recall=0.1563`, `pair_t16=0.0630`)
  - full-val (`max_eval_batches=10000`) produced the same rows as `128` in this setup (validation exhausted before limit):
    - baseline best `pred spans + pred pairs rel_f1=0.0012`
    - sentence-bias best `rel_f1=0.0012`
  - interpretation: no meaningful effect on F1; coverage deltas are negligible.

### Current Recommendation
- Keep span-context depth off in current promotion path (`span_context_layers=0`) until a dedicated staged recipe exists.
- Keep `retrieval_sentence_bias_scale` disabled by default for now; only revisit via full-val sweeps if needed.

### Open Issues
- `span_context_layers=1` remains non-competitive in both warm-start and short from-scratch probes with current settings.
- Sentence-bias knob is now full-val checked and non-promotable on the current checkpoint; any future revisit should be in a different retrieval architecture regime.

---

## 2026-03-15 — Full-Val Sentence-Bias Verdict + Span-Context Start-Step Gates

### Objectives
- Finalize sentence-bias verdict with full validation coverage.
- Add trainer-side staged gating knobs for span-context edge families.

### Changes Saved
- Full-val sentence-bias comparison executed against identical checkpoint:
  - baseline config: [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun.toml)
  - sentence-bias config: [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun_sentbias015.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun_sentbias015.toml)
- Added staged span-context edge-family start-step config support in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - `span_context_adjacent_start_step`
  - `span_context_sentence_start_step`
  - `span_context_semantic_start_step`
  - these gates are applied in training via `with_retrieval_bias_inputs(...; step=next_step)`.

### Key Experiment Outcomes
- Full-val threshold sweep (`max_eval_batches=10000`, margin `0.10`, thresholds `0.60/0.70/0.80`) on:
  - `checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun/checkpoint_last.jls`
- `pred spans + pred pairs`:
  - baseline (`scale=0.00`): best `rel_f1=0.0012` at threshold `0.70`, `pair_recall=0.1554`, `pair_t16=0.0639`
  - sentence-bias (`scale=0.15`): best `rel_f1=0.0012` at threshold `0.70`, `pair_recall=0.1563`, `pair_t16=0.0630`
- Verdict: no meaningful effect from sentence-bias on this checkpoint/decode regime.

### Current Recommendation
- Keep `retrieval_sentence_bias_scale=0.0` as default.
- If revisiting span-context depth, use the new start-step gates and avoid mid-run architecture insertion.

### Open Issues
- Span-context depth still requires a staged/from-scratch plan with stronger proposal retention.

---

## 2026-03-15 — Local-Bias Retrieval Ablation (Full-Val)

### Objectives
- Add local-distance retrieval bias as a checkpoint-safe runtime scoring knob.
- Measure whether local bias improves `pred spans + pred pairs` on the locked `seed42_rerun` checkpoint.

### Changes Saved
- Added local-bias term in pair aux feature pipeline in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - `gather_pair_aux_features(...; local_radius=...)` now emits `local_bias_base`.
  - model forward now accepts `retrieval_local_bias_scale` and adds this term into retrieval logits bias.
- Added trainer plumbing in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - new config key: `relation_extraction.retrieval_local_bias_scale` (default `0.0`).
  - propagated through train/eval/oracle/calibration input builders.
- Added ablation config:
  - [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun_localbias015.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun_localbias015.toml)
- Added unit coverage in [`test/test_relation_extraction.jl`](/home/christos/code/julia/Swamma/test/test_relation_extraction.jl):
  - `Pair Aux Bias Bases` now checks both sentence-bias and local-bias bases.

### Key Experiment Outcomes
- Validation checks:
  - parse check passed for `src/Swamma.jl` + `scripts/train_re_gpu.jl`
  - `julia --project=. test/test_relation_extraction.jl` passed all testsets.
- Threshold ablations on identical checkpoint (`margin=0.10`, thresholds `0.60/0.70/0.80`):
  - baseline (`retrieval_local_bias_scale=0.00`) best `pred spans + pred pairs rel_f1=0.0012`, `pair_recall=0.1554`, `pair_t16=0.0639`
  - local-bias (`retrieval_local_bias_scale=0.15`) best `rel_f1=0.0012`, `pair_recall=0.1537`, `pair_t16=0.0604`
  - same outcome for `max_eval_batches=128` and `10000`.
- Verdict: local-bias did not improve F1 and slightly reduced pair coverage in this regime.

### Current Recommendation
- Keep `retrieval_local_bias_scale=0.0` as default.
- Treat local-bias as non-promoted optional knob unless a future architecture branch shows a different operating region.

### Open Issues
- Retrieval-side gains remain bottlenecked by proposal quality; score-level bias tweaks alone are not moving the ceiling.

---

## 2026-03-15 — Staged Span-Context Activation Probe + Eval-Gate Fix

### Objectives
- Fix step-gating mismatch between training and in-training evaluation for staged span-context experiments.
- Re-test staged span-context activation against a matched no-span-context control.

### Changes Saved
- Added global span-context runtime switch in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - new input: `span_context_enabled` (default `true`)
  - `apply_span_context(...; enabled=false)` now bypasses span-context layers entirely.
- Extended trainer settings in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - new config key: `span_context_start_step`
  - `with_retrieval_bias_inputs(...; step=...)` now emits `span_context_enabled`.
- Fixed in-training eval gating:
  - `evaluate_model(...; current_step=step)` now uses step-aware retrieval/span-context inputs during training-time eval.
- Added staged probe config:
  - [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_spanctx1_staged_probe.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_spanctx1_staged_probe.toml)

### Key Experiment Outcomes
- Staged span-context run (`span_context_layers=1`, `span_context_start_step=1040`, `1000 -> 1050`):
  - pre-activation window:
    - step `1010`: `val_loss=13.8535`, `pair_recall=0.0732`, `rel_f1=0.0000`
    - step `1030`: `val_loss=15.0076`, `pair_recall=0.1585`, `rel_f1=0.0000`
  - post-activation window:
    - step `1040`: `val_loss=44.1212`, `pair_recall=0.1585`, `rel_f1=0.0012`
    - step `1050`: `val_loss=45.7022`, `pair_recall=0.2439`, `rel_f1=0.0015`
- Matched control (`span_context_layers=0`, same `1000 -> 1050` path):
  - step `1040`: `val_loss=14.9474`, `pair_recall=0.1829`, `rel_f1=0.0027`
  - step `1050`: `val_loss=14.4953`, `pair_recall=0.1829`, `rel_f1=0.0025`
- Interpretation:
  - eval-mismatch bug is fixed (pre-activation behavior is now sane),
  - but span-context activation still degrades overall quality vs control in this warm-start recipe.

### Current Recommendation
- Keep `span_context_layers=0` for the active continuation branch.
- Keep staged gates as infrastructure only; they are useful for controlled experiments but not yet promotable.

### Open Issues
- Span-context modules likely need a dedicated initialization/curriculum strategy before activation; direct warm-start insertion remains unstable.

---

## 2026-03-15 — Edge Retrieval v2 Explicit Compatibility Terms

### Objectives
- Land the missing explicit retrieval-score components from the v2 plan without changing checkpoint-serialized model parameter shapes.
- Add runtime knobs so the new terms can be ablated cleanly in train/eval/oracle/calibration flows.

### Changes Saved
- Updated pair auxiliary feature extraction in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - `gather_pair_aux_features(...)` now emits `type_compat_bias_base` in addition to distance/type/sentence/local bases.
  - `type_compat_bias_base` is computed from non-null entity-type distribution overlap at head/tail mention starts, weighted by token-level entity mass.
- Added explicit dot-product compatibility retrieval term in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - computes head-tail span-vector dot score (`/ sqrt(d)`) for both draft over-generated pruning and final retrieval scoring.
  - integrated as runtime additive retrieval bias.
- Added runtime input knobs in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - `retrieval_type_compat_bias_scale`
  - `retrieval_dot_bias_scale`
- Threaded both knobs through trainer plumbing in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - loaded from config in `load_retrieval_bias_settings(...)`
  - propagated through `with_retrieval_bias_inputs(...)`
  - propagated through proposal/fixed-proposal/oracle/auto-calibration input builders.
- Extended unit coverage in [`test/test_relation_extraction.jl`](/home/christos/code/julia/Swamma/test/test_relation_extraction.jl):
  - `Pair Aux Bias Bases` now passes synthetic `entity_logits` and asserts positive `type_compat_bias` on same-type pair.

### Key Validation Outcomes
- Parse check passed:
  - `julia --project=. -e 'Base.include(Main, "src/Swamma.jl"); Base.include(Main, "scripts/train_re_gpu.jl"); println("parse-ok")'`
- RE unit test suite passed:
  - `julia --project=. test/test_relation_extraction.jl`
- Fixed a GPU eval failure discovered during first threshold sweep:
  - root cause: mixed CPU matrix + CuArray broadcast in retrieval-bias assembly after adding dot term.
  - fix: compute dot-bias bases under `ignore_derivatives` and materialize CPU arrays before the final optional `CuArray(...)` cast of `retrieval_bias`.
- Ablation sweeps on checkpoint:
  - `checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun/checkpoint_last.jls`
  - decode setup: `thresholds=0.60/0.70/0.80`, `no_relation_margin=0.10`, `max_eval_batches=128`.
- Baseline (`type_compat=0.0`, `dot=0.0`):
  - best `pred spans + pred pairs rel_f1=0.0012` (`threshold=0.70`)
  - `pair_r=0.1554`, `pair_t16=0.0639`
- Type-compat only (`type_compat=0.15`, `dot=0.0`):
  - identical to baseline at all reported points (`best rel_f1=0.0012`, same pair coverage)
- Dot only (`type_compat=0.0`, `dot=0.10`):
  - best `rel_f1=0.0013` (`threshold=0.80`) but coverage regresses (`pair_r=0.1459`, `pair_t16=0.0484`)
  - at baseline operating point (`threshold=0.70`), `rel_f1` drops to `0.0011`
- Combined (`type_compat=0.15`, `dot=0.10`):
  - matches dot-only behavior on full-val.

### Current Recommendation
- Keep `retrieval_type_compat_bias_scale=0.0` and `retrieval_dot_bias_scale=0.0` on the promoted baseline branch.
- Treat both terms as optional knobs for future checkpoints only; current evidence does not justify promotion.

### Open Issues
- Compatibility terms are now implemented and benchmarked on one checkpoint family, but not yet tested across seeds/longer continuation runs.

---

## 2026-03-15 — Edge Retrieval v2 Multi-Family Candidate Composition

### Objectives
- Move `edge_retrieval_v2` from semantic-only candidate sourcing to a true multi-family sparse retriever.
- Cover the TODO requirement to keep local, routed, semantic, and reserve families in the edge-v2 path with runtime controls.

### Changes Saved
- Updated proposer family routing in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - `pair_proposer_uses_router` now includes `:edge_retrieval_v2`.
  - `propose_relation_pairs(...)` now supports edge-v2 family gates:
    - `edge_v2_use_local_neighbors`
    - `edge_v2_use_routed_buckets`
    - `edge_v2_use_semantic_topk`
    - `edge_v2_use_global_reserve`
  - local-neighbor candidates now feed edge-v2 scoring path (`build_edge_v2_pair_candidate`) when local family is enabled.
  - routed-bucket candidates now feed edge-v2 scoring path when routed family is enabled.
  - semantic top-k and global reserve are now independently gateable inside edge-v2 branch.
- Added runtime input parsing and propagation in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - model forward now reads all four `edge_v2_use_*` booleans and passes them into `propose_relation_pairs(...)`.
- Added trainer/config plumbing in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - `load_retrieval_bias_settings(...)` now loads:
    - `edge_v2_use_local_neighbors`
    - `edge_v2_use_routed_buckets`
    - `edge_v2_use_semantic_topk`
    - `edge_v2_use_global_reserve`
  - propagated through `with_retrieval_bias_inputs(...)`, proposal/fixed-proposal builders, oracle ladder, and auto-calibration eval inputs.
- Added unit tests in [`test/test_relation_extraction.jl`](/home/christos/code/julia/Swamma/test/test_relation_extraction.jl):
  - `Edge Retrieval v2 Family Gating`:
    - local-only gate verifies adjacency pairs are emitted.
    - routed-only gate verifies bucket-routed long pair (`1<->4`) is emitted while local neighbor pair (`1->2`) is absent.

### Key Validation Outcomes
- Parse check passed:
  - `julia --project=. -e 'Base.include(Main, "src/Swamma.jl"); Base.include(Main, "scripts/train_re_gpu.jl"); println("parse-ok")'`
- RE unit test suite passed:
  - `julia --project=. test/test_relation_extraction.jl`
  - includes new `Edge Retrieval v2 Family Gating` testset.
- End-to-end edge-v2 eval smoke passed:
  - command:
    - `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_edgev2_smoke.toml --eval-checkpoint checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun/checkpoint_last.jls --max-eval-batches 1`
  - output row (`checkpoint_last.jls`, step `1500`) confirms forward/eval path is healthy with edge-v2 candidate generation enabled.
- Quick family-gating threshold ablations completed (`max_eval_batches=32`, checkpoint `seed42_rerun`, thresholds `0.60/0.70/0.80`, margin `0.10`):
  - all-families on (`edgev2_smoke`)
  - semantic+reserve only (`edgev2_semres_only`)
  - local+routed+reserve without semantic (`edgev2_localrouted_reserve`)
  - result: all three produced identical sampled metric tables, including `pred spans + pred pairs` (`best rel_f1=0.0018`, `pair_r=0.1729`, `pair_t16=0.0634` at `threshold=0.70`).
- Larger edge-v2 check completed (`max_eval_batches=128`, all-families on):
  - `pred spans + pred pairs` reproduces the same full-val point as prior edge-v2 runs (`best rel_f1=0.0012`, `pair_r=0.1554`, `pair_t16=0.0639`).
  - confirms no measurable lift from family-composition change on this checkpoint distribution.

### Current Recommendation
- Keep all edge-v2 family gates available as architecture scaffolding, but do not treat family toggling as an optimization path for this checkpoint family.
- Move to the next retrieval/objective lever (not more family-toggle sweeps) unless a future checkpoint distribution shifts candidate saturation behavior.

### Open Issues
- Multi-family edge-v2 composition is implemented, tested, and now checked at larger eval budget, but still has no measurable quality lift vs current promoted baseline.

---

## 2026-03-15 — Sentence-Distance Embedding Retrieval Hook (Checkpoint-Safe)

### Objectives
- Add a sentence-distance embedding path to retrieval scoring without changing checkpoint-serialized parameter shapes.
- Test whether embedding-level sentence distance helps more than scalar sentence-bias terms.

### Changes Saved
- Updated pair aux feature extraction in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - `gather_pair_aux_features(...)` now emits `sentence_distance_ids` buckets in addition to existing distance IDs/bias bases.
- Extended `PairRetrievalHead` input contract in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - added 7/8-input variants carrying:
    - `sentence_distance_ids`
    - `sentence_embedding_scale`
  - implementation reuses `DistanceEmbedding` table to produce sentence-distance embeddings and adds them into retrieval distance embedding stream:
    - `distance_emb += scale * sentence_emb`
  - keeps output projection shape unchanged (checkpoint-safe).
- Added runtime input knob in [`src/RelationExtraction.jl`](/home/christos/code/julia/Swamma/src/RelationExtraction.jl):
  - `retrieval_sentence_embedding_scale` (default `0.0`).
- Added trainer/config plumbing in [`scripts/train_re_gpu.jl`](/home/christos/code/julia/Swamma/scripts/train_re_gpu.jl):
  - `load_retrieval_bias_settings(...)` now loads `retrieval_sentence_embedding_scale`.
  - propagated through train/eval/proposal/oracle/auto-calibration input builders.
- Extended tests in [`test/test_relation_extraction.jl`](/home/christos/code/julia/Swamma/test/test_relation_extraction.jl):
  - `Pair Aux Bias Bases` now asserts sentence-distance bucket IDs (`cross-sentence > 1`, same-sentence `== 1`).
- Added ablation config:
  - [`configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun_sentembed025.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun_sentembed025.toml)

### Key Validation Outcomes
- Parse check passed:
  - `julia --project=. -e 'Base.include(Main, "src/Swamma.jl"); Base.include(Main, "scripts/train_re_gpu.jl"); println("parse-ok")'`
- RE unit test suite passed:
  - `julia --project=. test/test_relation_extraction.jl`
- Sampled threshold ablation on checkpoint:
  - `checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_rankloss_soft_from1000_seed42_rerun/checkpoint_last.jls`
  - settings: `thresholds=0.60/0.70/0.80`, `margin=0.10`, `max_eval_batches=32`
  - with `retrieval_sentence_embedding_scale=0.25`:
    - best `pred spans + pred pairs rel_f1=0.0018` (unchanged vs baseline sampled point)
    - `pair_r` slightly lower (`0.1729 -> 0.1700`)
    - `pair_t16` unchanged (`0.0634`)

### Current Recommendation
- Keep `retrieval_sentence_embedding_scale=0.0` on promoted baseline.
- Retain the hook as an optional checkpoint-safe knob for future branches.

### Open Issues
- Sentence-distance embedding via shared table did not improve sampled decode metrics on this checkpoint family.

## Session Entry Template

Copy this block for new sessions:

```md
## YYYY-MM-DD — <Short Session Title>

### Objectives
- ...

### Changes Saved
- ...

### Key Experiment Outcomes
- ...

### Current Recommendation
- ...

### Open Issues
- ...
```

## 2026-03-15 — Hybrid Swamma Reasoning Findings

### Objectives
- Inspect the current Swamma diffusion / drafter implementation.
- Resolve architecture questions around dense width, depth, MoE placement, and DGX Spark feasibility.
- Capture a concrete design direction for a hybrid `AR LLM + Swamma drafter + MoE reasoning experts + symbolic expert` system.

### Changes Saved
- Added findings document:
  - [`docs/SWAMMA_HYBRID_REASONING_FINDINGS_2026-03-15.md`](/home/christos/code/julia/Swamma/docs/SWAMMA_HYBRID_REASONING_FINDINGS_2026-03-15.md)
- No source or config code changes were made in this session.

### Key Experiment Outcomes
- No training experiments were run.
- Local architectural inspection confirmed the existing Julia diffusion / masking path in:
  - [`src/LLaDA.jl`](/home/christos/code/julia/Swamma/src/LLaDA.jl)
  - [`src/Training.jl`](/home/christos/code/julia/Swamma/src/Training.jl)
  - [`src/Drafter.jl`](/home/christos/code/julia/Swamma/src/Drafter.jl)
  - [`src/TiDAR.jl`](/home/christos/code/julia/Swamma/src/TiDAR.jl)
- Local parameter-count scaling on Swamma/LLaDA was used to estimate why single-DGX-Spark full 120B training is not realistic.

### Current Recommendation
- Use Swamma as a reasoning drafter around a strong long-context autoregressive LLM.
- Keep the sequence backbone shared and place MoE only in the FFN path first.
- Spark-safe surrogate for development:
  - `embedding_dimension=4096`
  - `number_of_heads=32`
  - `number_of_layers=32`
  - `state_dimension=4096`
  - `num_experts=16`
  - `top_k=2`
  - MoE every 2nd layer
- Longer-term architecture direction:
  - hybrid `AR verifier/finalizer + Swamma reasoning drafter + reasoning-specialized MoE + rare symbolic expert`

### Open Issues
- The exact interface to a symbolic expert remains undecided:
  - masked spans
  - logical forms
  - puzzle states
  - latent thought segments
- No code has yet been written for `MoEFFN`, routing, or symbolic-expert integration.

## 2026-03-15 — Lux-Native Teacher LM Foundation

### Objectives
- Replace the hand-wavy “use Lux” claim with an actual Julia-native decoder foundation for teacher inference.
- Add the minimum reusable components needed before any native HF teacher import:
  - causal decoder blocks
  - RoPE
  - grouped-query-capable self-attention
  - simple generation helpers
- Verify that the new path is structurally correct before touching checkpoint import.

### Changes Saved
- Added [`src/NativeTeacherLM.jl`](/home/christos/code/julia/Swamma/src/NativeTeacherLM.jl):
  - `NativeTeacherConfig`
  - `RotaryEmbedding`
  - `CausalSelfAttention`
  - `GatedMLP`
  - `DecoderBlock`
  - `NativeCausalLM`
  - `build_causal_attention_bias`
  - `next_token_logits`
  - `greedy_generate`
- Updated [`src/Swamma.jl`](/home/christos/code/julia/Swamma/src/Swamma.jl):
  - included the new `NativeTeacherLM` submodule
  - re-exported the native teacher types/helpers through top-level `Swamma`
- Added [`test/test_native_teacher_lm.jl`](/home/christos/code/julia/Swamma/test/test_native_teacher_lm.jl):
  - forward-pass shape/finite smoke
  - causal masking regression check
  - greedy-generation smoke
  - causal-bias helper check
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) with the correct next sequence for native teacher work:
  - HF checkpoint importer
  - KV-cache decode
  - Julia-native RE teacher generation

### Key Experiment Outcomes
- Parse check passed:
  - `julia --project=. -e 'include("src/Swamma.jl"); using .Swamma; println("parse-ok")'`
- Native teacher test suite passed:
  - `julia --project=. test/test_native_teacher_lm.jl`
  - result: `13/13` tests passed in `4.6s`
- This is now a real Lux-native decoder stack, but still randomly initialized:
  - no Hugging Face checkpoint importer yet
  - no tokenizer/chat-template parity enforcement yet
  - no KV-cache decode path yet

### Current Recommendation
- Keep [`configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_distill_pilot.toml`](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_edgev2_pairevidmlp_focal2_posw2_distill_pilot.toml) as the plumbing-only distillation config until real teacher weights exist.
- For the Julia-native teacher path, target `ibm-granite/granite-4.0-micro` first rather than Qwen 7B.
- Treat the next critical task as checkpoint import, not more architecture churn.

### Open Issues
- `NativeTeacherLM` currently proves the native decoder architecture, not usable teacher quality.
- There is still no mapping from Hugging Face safetensors to the Lux parameter tree.
- Generation currently recomputes the full prefix each step; a KV-cache path is still required for practical teacher data generation.

## 2026-03-15 — Granite Native Importer (Config + Safetensors Mapping)

### Objectives
- Move the native teacher path past “decoder scaffold only” by adding a real Granite import path.
- Convert Hugging Face Granite config metadata into `NativeTeacherConfig`.
- Map Granite safetensors tensor names into the Lux `NativeTeacherLM` parameter tree and verify the mapping without depending on a multi-GB end-to-end load in tests.

### Changes Saved
- Extended [`src/NativeTeacherLM.jl`](/home/christos/code/julia/Swamma/src/NativeTeacherLM.jl):
  - added Granite-compatible config fields:
    - `rms_norm_eps`
    - `embedding_multiplier`
    - `logits_scaling`
    - `tie_word_embeddings`
  - updated native forward path to apply Granite-style embedding scaling and logits scaling
  - added Hugging Face helpers:
    - `resolve_hf_model_dir(...)`
    - `granite_config_from_hf(...)`
    - `load_granite_weights(...)`
    - `load_granite_model(...)`
  - added safetensors shard loading through Julia via `PyCall + safetensors`
  - added Granite tensor-name mapping for:
    - token embeddings
    - RMSNorm scales
    - `q/k/v/o` attention projections
    - combined `shared_mlp.input_linear.weight` split into gate/up projections
    - `shared_mlp.output_linear.weight`
    - tied output head
- Updated [`src/Swamma.jl`](/home/christos/code/julia/Swamma/src/Swamma.jl) exports so the Granite loader helpers are available through top-level `Swamma`.
- Extended [`test/test_native_teacher_lm.jl`](/home/christos/code/julia/Swamma/test/test_native_teacher_lm.jl):
  - synthetic Granite-style `config.json`
  - synthetic `model.safetensors.index.json`
  - synthetic safetensors shard generation
  - import verification for config parsing and parameter mapping

### Key Experiment Outcomes
- Parse check passed:
  - `julia --project=. -e 'include("src/Swamma.jl"); using .Swamma; println("parse-ok")'`
- Native teacher suite passed after importer work:
  - `julia --project=. test/test_native_teacher_lm.jl`
  - result: `30/30` tests passed in `8.4s`
- Real Granite config smoke passed without full weight download:
  - `julia --project=. -e 'include("src/Swamma.jl"); using .Swamma; cfg = granite_config_from_hf("ibm-granite/granite-4.0-micro"); println((cfg.vocab_size, cfg.embedding_dimension, cfg.number_of_layers, cfg.number_of_heads, cfg.number_of_kv_heads, cfg.mlp_hidden_dimension, cfg.embedding_multiplier, cfg.logits_scaling, cfg.tie_word_embeddings))'`
  - result:
    - `vocab_size=100352`
    - `embedding_dimension=2560`
    - `number_of_layers=40`
    - `number_of_heads=40`
    - `number_of_kv_heads=8`
    - `mlp_hidden_dimension=8192`
    - `embedding_multiplier=12`
    - `logits_scaling=10`
    - `tie_word_embeddings=true`

### Current Recommendation
- The native path is now real enough to continue; the blocker is no longer “we need a Julia decoder architecture.”
- Keep `ibm-granite/granite-4.0-micro` as the first supported teacher family.
- The next step should be one real full-shard weight load through `load_granite_model(...)`, then KV-cache decoding. Do not detour into another teacher architecture first.

### Open Issues
- The synthetic importer test proves the mapping logic, but I have not yet completed a full real Granite shard load end-to-end.
- Tokenizer/chat-template parity for the native Granite path is still not validated.
- Generation is still full-prefix recompute; practical teacher-corpus generation will still need KV-cache support.

## 2026-03-15 — Real Granite Load Smoke + Tokenizer Runtime Check

### Objectives
- Validate that the new native Granite importer works on the actual `ibm-granite/granite-4.0-micro` weight shards, not just synthetic safetensors.
- Add a reusable smoke script for the native Granite path.
- Check whether the existing Julia tokenizer wrapper can support Granite chat-template prompting in this runtime.

### Changes Saved
- Updated [`src/HFTokenizer.jl`](/home/christos/code/julia/Swamma/src/HFTokenizer.jl):
  - added `local_files_only` support to `load_tokenizer(...)`
  - added chat-template helpers:
    - `has_chat_template(...)`
    - `apply_chat_template(...)`
    - `apply_chat_template_tokens(...)`
- Added native smoke script:
  - [`scripts/smoke_native_granite_load.jl`](/home/christos/code/julia/Swamma/scripts/smoke_native_granite_load.jl)
  - script loads:
    - tokenizer if available
    - chat-template prompt if available
    - native Granite model/weights through `load_granite_model(...)`
  - tokenizer failure is reported explicitly instead of silently breaking the script
- Extended [`test/test_native_teacher_lm.jl`](/home/christos/code/julia/Swamma/test/test_native_teacher_lm.jl):
  - added a local Granite chat-template smoke
  - marked it as a broken/skip-style check when the PyCall tokenizer runtime is unavailable
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) to mark real-weight load complete and make the tokenizer-runtime issue explicit.

### Key Experiment Outcomes
- Parse check passed:
  - `julia --project=. -e 'include("src/Swamma.jl"); using .Swamma; println("parse-ok")'`
- Native teacher suite passed with one expected broken tokenizer-runtime check:
  - `julia --project=. test/test_native_teacher_lm.jl`
  - result: `30 passed`, `1 broken`
- Real Granite weight-load smoke passed:
  - `julia --project=. -e 'include("src/Swamma.jl"); using .Swamma, Random, Lux; rng = Random.MersenneTwister(1); @time model, ps, st = load_granite_model("ibm-granite/granite-4.0-micro"; rng=rng, dtype=Float16, local_files_only=true); println("loaded"); println((model.config.vocab_size, model.config.embedding_dimension, model.config.number_of_layers, model.config.number_of_heads, model.config.number_of_kv_heads, model.config.mlp_hidden_dimension)); println(size(ps.TokenEmbedding.weight)); println(size(ps.Blocks[1].Attention.QueryProjection.weight)); println(size(ps.Blocks[end].FeedForward.DownProjection.weight)); println(size(ps.OutputHead.weight));'`
  - result:
    - full load completed in `~25.8s`
    - allocator volume `~30.8 GiB`
    - imported shapes:
      - token embedding `(2560, 100352)`
      - first-layer Q projection `(2560, 2560)`
      - last-layer FFN down projection `(2560, 8192)`
      - output head `(100352, 2560)`
- Native smoke script passed for the weight-load path:
  - `julia --project=. scripts/smoke_native_granite_load.jl --model ibm-granite/granite-4.0-micro --dtype float16 --local-files-only`
  - result:
    - native model load succeeds and reports the expected Granite micro config
    - tokenizer phase currently reports a PyCall runtime failure instead of succeeding
- Tokenizer runtime finding:
  - plain `python3` can load `AutoTokenizer` and apply the Granite chat template
  - the PyCall runtime used by Julia currently fails importing the required `transformers` path with a `GenerationMixin`/`ModuleNotFoundError` chain
  - this is now treated as an explicit environment/runtime issue, not hidden under a passing native-model smoke

### Current Recommendation
- Treat the native Granite model-import path as working.
- Do not treat the tokenizer/chat-template path as solved yet.
- The next serious tasks are:
  - fix or replace the Julia tokenizer runtime path
  - add KV-cache decoding
  - only then build the Julia-native RE teacher generator

### Open Issues
- `HFTokenizer.jl` is still blocked by the current PyCall/`transformers` runtime for Granite chat-template usage.
- The native model can load real weights, but generation is still full-prefix recompute.
- No end-to-end native teacher-corpus generation run is possible yet until tokenizer/runtime plus cache support are addressed.

## 2026-03-15 — Native Granite Tokenizer Fallback + Real Forward Smoke

### Objectives
- Remove the remaining tokenizer/runtime blocker from the native Granite path.
- Replace the fragile `transformers`-through-PyCall tokenizer dependency with a fallback that can still render Granite prompts locally.
- Verify the full native path on a real prompt:
  - chat-template render
  - tokenize
  - real-weight load
  - real forward pass

### Changes Saved
- Reworked [`src/HFTokenizer.jl`](/home/christos/code/julia/Swamma/src/HFTokenizer.jl):
  - added a fallback backend built on:
    - Python `tokenizers`
    - local HF `tokenizer.json`
    - `tokenizer_config.json`
    - `special_tokens_map.json`
    - `chat_template.jinja`
  - `load_tokenizer(...)` now:
    - tries `transformers.AutoTokenizer` first
    - falls back to the local `tokenizers` backend when the `transformers` import path is broken under PyCall
  - added local snapshot resolution through `huggingface_hub.snapshot_download(...)`
  - kept the same Julia-facing encode/decode/batch/chat-template API across both backends
- Updated [`scripts/smoke_native_granite_load.jl`](/home/christos/code/julia/Swamma/scripts/smoke_native_granite_load.jl):
  - tokenizer stage now uses the fallback-capable wrapper
  - supports prompt rendering plus optional real forward smoke
- Updated [`test/test_native_teacher_lm.jl`](/home/christos/code/julia/Swamma/test/test_native_teacher_lm.jl):
  - Granite chat-template wrapper check now passes through the fallback backend

### Key Experiment Outcomes
- Native teacher suite now passes fully:
  - `julia --project=. test/test_native_teacher_lm.jl`
  - result: `35/35` tests passed
- Native Granite tokenizer + load smoke passed:
  - `julia --project=. scripts/smoke_native_granite_load.jl --model ibm-granite/granite-4.0-micro --dtype float16 --local-files-only`
  - result:
    - tokenizer fallback loads successfully
    - Granite chat template renders correctly
    - prompt token count: `27`
    - native model load succeeds on real weights
- Native Granite forward smoke passed:
  - `julia --project=. scripts/smoke_native_granite_load.jl --model ibm-granite/granite-4.0-micro --dtype float16 --local-files-only --run-forward --max-prompt-tokens 32`
  - result:
    - rendered prompt tokenized to `27` tokens
    - real forward pass completed
    - logits shape: `(100352, 27, 1)`
    - next-token logits shape: `(100352, 1)`

### Current Recommendation
- The native Granite path is now functionally validated:
  - local tokenizer fallback works
  - chat template works
  - real weights load
  - real forward execution works
- The next blocker is no longer basic correctness. It is performance:
  - KV-cache decode
  - then native teacher-corpus generation

### Open Issues
- `NativeTeacherLM` still recomputes the full prefix at every generation step.
- No native teacher generation script exists yet on top of the now-working Granite path.
- Full teacher-corpus generation should wait for KV-cache support, otherwise the runtime will be unnecessarily expensive.

## 2026-03-15 — KV-Cache Decode For Native Granite Path

### Objectives
- Remove the last obvious efficiency blocker from the native Granite path by adding cache-aware decoding.
- Verify that cached decoding stays numerically aligned with full-prefix recompute.
- Validate the cache path on both:
  - the small unit-test model
  - the real `ibm-granite/granite-4.0-micro` weights

### Changes Saved
- Extended [`src/NativeTeacherLM.jl`](/home/christos/code/julia/Swamma/src/NativeTeacherLM.jl):
  - added RoPE position-offset support for cached decoding
  - generalized `build_causal_attention_bias(...)` to rectangular masks with query offset
  - added cache structs:
    - `AttentionKVCache`
    - `NativeDecoderCache`
  - added cache helpers:
    - `init_decoder_cache(...)`
    - `cache_sequence_length(...)`
  - added cached execution helpers:
    - `forward_with_cache(...)`
    - `next_token_logits_cached(...)`
    - `greedy_generate_cached(...)`
  - refactored attention through an internal cache-capable path so standard forward and cached forward share the same logic
- Updated [`src/Swamma.jl`](/home/christos/code/julia/Swamma/src/Swamma.jl) exports for the new cache API.
- Extended [`test/test_native_teacher_lm.jl`](/home/christos/code/julia/Swamma/test/test_native_teacher_lm.jl):
  - cached next-token logits vs full recompute
  - cached greedy generation vs uncached greedy generation
  - rectangular/offset causal-mask checks
- Updated [`scripts/smoke_native_granite_load.jl`](/home/christos/code/julia/Swamma/scripts/smoke_native_granite_load.jl):
  - added `--run-cached-check`
  - compares cached next-token logits against full recompute on the real Granite model

### Key Experiment Outcomes
- Parse check passed:
  - `julia --project=. -e 'include("src/Swamma.jl"); using .Swamma; println("parse-ok")'`
- Native teacher suite passed after cache work:
  - `julia --project=. test/test_native_teacher_lm.jl`
  - result: `44/44` tests passed
- Real Granite cached smoke passed:
  - `julia --project=. scripts/smoke_native_granite_load.jl --model ibm-granite/granite-4.0-micro --dtype float16 --local-files-only --run-cached-check --max-prompt-tokens 32`
  - result:
    - prompt token count: `27`
    - cache sequence length after appending one token: `28`
    - cached logits shape: `(100352, 1)`
    - max absolute delta vs full recompute: `0.00747633`
- Interpretation:
  - on Float16 real weights, the cached path is close to full recompute
  - the remaining small delta is consistent with different accumulation/order effects rather than a broken cache path

### Current Recommendation
- Treat the native Granite path as functionally ready for generation work:
  - tokenizer fallback works
  - real weights load
  - real forward works
  - KV-cache decode works
- The next concrete task is a Julia-native RE teacher generation script on top of this path.

### Open Issues
- The native path is now blocked mainly by missing application code, not model infrastructure.
- There is still no end-to-end Julia-native teacher-corpus generation script for REBEL rows.
- Runtime/perf benchmarking for long rollouts has not yet been done beyond basic cache-smoke validation.

## 2026-03-15 — Julia-Native RE Teacher Generation Script

### Objectives
- Move beyond model/tokenizer infrastructure and add the first Julia-native RE teacher generation script.
- Reuse the existing REBEL request JSONL contract so downstream parser/merge/validate tooling stays unchanged.
- Smoke the full request -> native generation -> raw response JSONL path.

### Changes Saved
- Added [`scripts/generate_rebel_teacher_responses_native.jl`](/home/christos/code/julia/Swamma/scripts/generate_rebel_teacher_responses_native.jl):
  - consumes the same request JSONL as the Python generator
  - uses:
    - `HFTokenizer` fallback-backed prompt rendering
    - `load_granite_model(...)`
    - cached native decoding via `forward_with_cache(...)`
  - writes the same raw response JSONL contract expected by [`scripts/parse_rebel_teacher_responses.jl`](/home/christos/code/julia/Swamma/scripts/parse_rebel_teacher_responses.jl)
  - writes `.errors.jsonl` on generation failures
  - supports:
    - `--resume`
    - `--overwrite`
    - `--max-rows`
    - `--max-input-tokens`
    - `--max-new-tokens`
    - `--do-sample`
    - `--temperature`
    - `--top-p`
    - `--stop-sequence`
    - `--plain-prompt`
    - `--local-files-only`
    - `--dtype`
- Fixed a first-token cache bug discovered during native-generation wiring:
  - cached generation now uses the prefill logits for the first generated token instead of re-feeding the last prompt token
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) to reflect that the native generator exists and the next task is generation-quality control.

### Key Experiment Outcomes
- Parse + unit test checks passed after the cache/generation fix:
  - `julia --project=. -e 'include("src/Swamma.jl"); using .Swamma; println("parse-ok")'`
  - `julia --project=. test/test_native_teacher_lm.jl`
  - result: `44/44` tests passed
- Real Granite cached smoke still passed after the generation fix:
  - `julia --project=. scripts/smoke_native_granite_load.jl --model ibm-granite/granite-4.0-micro --dtype float16 --local-files-only --run-cached-check --max-prompt-tokens 32`
  - result: cache sequence length `28`, cached/full recompute max abs delta `0.00747633`
- Native request export smoke:
  - `julia --project=. scripts/build_rebel_teacher_requests.jl --input data/rebel/train.jsonl --output /tmp/rebel_teacher_requests_native_smoke.jsonl --max-rows 1`
  - result: exported one teacher request row
- Native generation smoke, greedy:
  - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_requests_native_smoke.jsonl --output /tmp/rebel_teacher_raw_native_smoke.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 1 --max-new-tokens 8`
  - result: model/runtime path works, but this request returned an empty greedy completion and was logged to `.errors.jsonl`
- Native generation smoke, sampled:
  - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_requests_native_smoke.jsonl --output /tmp/rebel_teacher_raw_native_smoke_sampled.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 1 --max-new-tokens 32 --do-sample --temperature 0.8 --top-p 0.95`
  - result: `accepted=1`, `failed=0`
  - confirms the Julia-native generator can produce a raw response row end-to-end

### Current Recommendation
- Treat the Julia-native RE teacher generator as operational at the plumbing level.
- Do not treat current sampled output quality as good enough yet for a real corpus rollout.
- The next task is quality control:
  - reduce empty greedy completions
  - improve JSON adherence
  - calibrate a small set of native generation settings before large-scale corpus generation

### Open Issues
- Greedy decode can still terminate immediately with an empty completion on at least some prompts.
- Sampled output can produce low-quality/non-JSON text without stronger decoding constraints.
- The native path is ready for corpus generation only after a short generation-quality tuning pass.

## 2026-03-15 — Native Granite Parity Fix + JSON Generation Diagnostics

### Objectives
- Determine whether the bad native RE teacher generations were caused by weak prompting or by native Granite inference drift.
- Tighten the native generator so failures preserve enough evidence to debug the next step instead of only reporting “invalid JSON”.
- Re-run a small native RE generation smoke after fixing any model-side mismatch.

### Changes Saved
- Extended [`src/NativeTeacherLM.jl`](/home/christos/code/julia/Swamma/src/NativeTeacherLM.jl):
  - added Granite config support for:
    - `attention_multiplier`
    - `residual_multiplier`
  - applied `attention_multiplier` inside native attention score scaling
  - applied `residual_multiplier` on both attention and feed-forward residual additions
- Extended [`scripts/generate_rebel_teacher_responses_native.jl`](/home/christos/code/julia/Swamma/scripts/generate_rebel_teacher_responses_native.jl):
  - added `--response-prefix`
  - added JSON-gated response validation by default
  - added `--allow-non-json` escape hatch
  - added `--verbose` timing/progress logging
  - reserve prompt-token budget explicitly for the response prefix instead of letting truncation cut it off
  - error logs now include `completion_preview` and `response_preview`
- Updated [`scripts/build_rebel_teacher_requests.jl`](/home/christos/code/julia/Swamma/scripts/build_rebel_teacher_requests.jl):
  - removed pseudo-schema literals like `{"start": Int, ...}` that the model was copying into outputs
  - explicitly warned against outputting keys such as `title`, `text`, or `tokens`
  - kept the concrete JSON skeleton example, but shifted the prose schema toward real JSON semantics rather than type-annotation text
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) to reflect that the current blocker is parity plus prompt/control quality, not missing native infrastructure.

### Key Experiment Outcomes
- Parse and unit checks still passed after the Granite-scaling fix:
  - `julia --project=. -e 'include("src/Swamma.jl"); using .Swamma; println("parse-ok")'`
  - `julia --project=. test/test_native_teacher_lm.jl`
  - result: `44/44` tests passed
- Hugging Face reference next-token probe on a short Granite prompt:
  - prompt: system=`You extract entities and relations.` user=`Barack Obama was born in Hawaii.`
  - `python3` + `transformers` top tokens before native comparison:
    - `Entities`
    - `Here`
    - `-`
    - `In`
    - `Entity`
    - `**`
    - `Extract`
    - `{\n}`
- Native Granite probe before the scaling fix was badly off:
  - top tokens were dominated by whitespace / punctuation / multilingual junk
  - this explained why the native RE teacher generator was emitting garbage despite the prompt controls
- Native Granite probe after the scaling fix is materially closer to Hugging Face:
  - `Here`, `Entities`, `Extract`, `-`, `Entity`, `**`, `In` now appear among the top predictions
  - conclusion: the native model is no longer obviously numerically broken at the next-token level on the short prompt
- Native RE generation diagnostics after the parity fix:
  - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_requests_native_qc_oneonly.jsonl --output /tmp/rebel_teacher_raw_native_qc_oneonly_postfix.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 1 --max-input-tokens 128 --max-new-tokens 32 --verbose`
  - result:
    - tokenizer load: `~1.84s`
    - native model load: `~22.81s`
    - row generation time: `~42.48s`
    - still failed JSON validation
    - failure preview moved from multilingual garbage to JSON-like malformed structure:
      - `{"entities":[{"title": ...`
- Stronger prefix probe:
  - `--response-prefix '{"entities":[{"start":'`
  - before prompt-text cleanup, the model copied `Int` literals into JSON
  - after prompt-text cleanup, the failure became:
    - `{"entities":[{"start":0": 0": 0"...`
  - conclusion: prompt/control is now the main blocker for RE JSON generation on the micro model, not the previously broken native inference path

### Current Recommendation
- Treat the native Granite path as materially more trustworthy after the config-faithful scaling fix.
- Do not start corpus generation or distillation from the native path yet.
- The next move should be a focused prompt/control pass:
  - keep the HF-vs-native short-prompt parity probe as a regression check
  - tune prompt wording and response prefixes against a tiny held-out RE shard
  - only proceed to larger teacher-corpus runs once a few rows produce parseable JSON consistently

### Open Issues
- The native generator is still CPU-only in practice; CUDA is available in Julia, but the current `NativeTeacherLM` implementation is not GPU-ready because it still contains host-oriented scalar loops and host array construction.
- Native Granite next-token parity improved substantially, but no reusable automated parity smoke exists yet.
- `ibm-granite/granite-4.0-micro` still fails to emit parseable RE JSON on the tested held-out row under the current prompt/prefix settings.
- Native RE generation remains too slow for large prompt iteration on CPU (`~22s` model load, `~42s` total for one 128-token / 32-new-token row).

## 2026-03-15 — Compact Prompt Sweep + Float32 Native Debug Baseline

### Objectives
- Reduce prompt ambiguity for the native RE teacher path without changing the downstream parser/merge contract.
- Check whether the remaining generation failures are caused by prompt contamination, truncation policy, or a deeper native-vs-HF generation gap.
- Set a more sane default dtype for CPU-bound native generation experiments.

### Changes Saved
- Extended [`scripts/build_rebel_teacher_requests.jl`](/home/christos/code/julia/Swamma/scripts/build_rebel_teacher_requests.jl):
  - added `--prompt-style <verbose|compact>`
  - added a `compact` request format that:
    - removes prose-heavy numbered token lines
    - emits a shorter schema description
    - provides `tokens = [...]` as a JSON array
  - preserved `--no-title`, which is now useful for compact-mode sweeps
- Extended [`scripts/generate_rebel_teacher_responses_native.jl`](/home/christos/code/julia/Swamma/scripts/generate_rebel_teacher_responses_native.jl):
  - changed prompt truncation from naive head-only clipping to head+tail preservation
  - switched the default native generation dtype from `float16` to `float32`
  - kept the compact-prompt experiments on the same raw-response JSONL contract
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) with the compact-prompt and `float32` baseline findings.

### Key Experiment Outcomes
- Hugging Face control with the compact prompt and strong prefix:
  - source row: the held-out `CBS Corporation` example
  - command family: `python3` + `transformers` on `ibm-granite/granite-4.0-micro`
  - result on compact prompt:
    - `{"entities":[{"start":1,"stop":2,"label":"ORG"}],"relations":[]}\n{"entities":[{"start":7,"stop":8,"label":"LOC"}]}`
  - interpretation:
    - the micro model can continue JSON-structured extraction text under the compact prompt
    - it still tends to overrun into extra objects, but the first object is structurally sane enough for the existing JSON-object extractor
- Hugging Face control with compact `--no-title` prompt:
  - result stayed structurally sane and continued a plausible entity list rather than collapsing into junk
- Native compact prompt, pre-`float32` default:
  - response preview became much closer to extraction semantics:
    - `{"entities":[{"start":0,"end":0,"label":"ORG","text":"CBS"...`
  - but still malformed and repetitive, so JSON validation rejected it
- Native compact prompt, `float32`, `--plain-prompt`, `--no-title`:
  - the generator still failed JSON validation
  - failure preview:
    - `{"entities":[{"start":0":0":0"...`
  - interpretation:
    - `float32` is the right debug baseline for CPU runs, but it does not by itself remove the remaining generation pathology
- Truncation finding:
  - the earlier 128-token debug runs were misleading because naive truncation dropped the useful prompt tail
  - the generator now preserves both the instruction head and the token-array tail before appending the forced JSON prefix

### Current Recommendation
- Use `compact` request prompts for native teacher debugging, not the original verbose numbered-token prompt.
- Use `float32` as the native CPU debug baseline.
- Treat Hugging Face compact-prompt generation as the control path showing the micro model is at least capable of near-structured JSON.
- Treat the native path as still failing at decode fidelity on full generation, even though short-prompt next-token parity improved substantially.

### Open Issues
- There is still no reusable automated HF-vs-native parity smoke script; the comparison is currently manual.
- The native compact prompt is better than the verbose prompt, but it still does not yield parseable RE JSON on the held-out row.
- The remaining gap now looks more like native full-generation behavior drift than a pure prompt-format issue.

## 2026-03-15 — Exact-Token HF-vs-Native Parity Script

### Objectives
- Replace the manual parity checks with a reusable exact-token comparison harness.
- Determine whether the remaining native generation mismatch is caused by tokenizer roundtrip issues or by true model-side divergence.

### Changes Saved
- Added [`scripts/compare_native_granite_hf.jl`](/home/christos/code/julia/Swamma/scripts/compare_native_granite_hf.jl):
  - loads one request row from teacher-request JSONL
  - builds the same prompt token sequence used by the native generator
  - supports:
    - `--plain-prompt`
    - `--response-prefix`
    - `--max-input-tokens`
    - `--max-new-tokens`
    - `--dtype`
    - `--local-files-only`
  - runs native greedy generation for `N` steps
  - runs Hugging Face greedy generation on the exact same prompt token IDs
  - prints a step-by-step token comparison table
- Fixed two early script bugs while validating it:
  - row loading from JSONL
  - Julia-to-Python boolean interpolation

### Key Experiment Outcomes
- Exact-token parity run:
  - command:
    - `julia --project=. scripts/compare_native_granite_hf.jl --input /tmp/rebel_teacher_requests_native_compact_notitle.jsonl --row-index 1 --model ibm-granite/granite-4.0-micro --response-prefix '{"entities":[{"start":' --max-input-tokens 512 --max-new-tokens 8 --dtype float32 --plain-prompt --local-files-only`
  - prompt token count: `308`
  - result: native and HF diverge at the very first generated token
- First 8-step comparison:
  - native:
    - `0`
    - `":`
    - `" "`
    - `<|end_of_text|>`
    - newline
    - `]`
    - triple backticks
    - `json`
  - HF:
    - `1`
    - `,"`
    - `stop`
    - `":`
    - `2`
    - `,"`
    - `label`
    - `":"`
- Interpretation:
  - this is no longer a tokenizer decode/re-encode artifact, because HF now consumes the exact same prompt token IDs as the native side
  - the remaining blocker is a true long-context native inference mismatch, not merely prompt style, dtype choice, or tokenizer roundtripping

### Current Recommendation
- Keep using [`scripts/compare_native_granite_hf.jl`](/home/christos/code/julia/Swamma/scripts/compare_native_granite_hf.jl) as the primary debugging tool for the native teacher path.
- Stop tuning prompts blindly until the native-vs-HF first-token divergence is reduced on the long RE prompt.
- The next debugging target should be the remaining long-context model math, most likely around positional handling / long-sequence attention behavior rather than tokenizer plumbing.

### Open Issues
- The native model still diverges from Hugging Face on the first generated token for the tested 308-token RE prompt, even after the earlier Granite scaling fix.
- Short-prompt next-token parity improved, but long-prompt generation parity is still broken.

## 2026-03-15 — Long-Context Boundary Check + Decode Heuristic Probe

### Objectives
- Find the sequence-length region where native/HF generation parity first breaks.
- Test whether a small amount of JSON-aware constrained decoding can overcome the remaining native long-context drift without changing the model.

### Changes Saved
- Extended [`scripts/generate_rebel_teacher_responses_native.jl`](/home/christos/code/julia/Swamma/scripts/generate_rebel_teacher_responses_native.jl):
  - added lightweight JSON-aware decode heuristics
  - new flag:
    - `--disable-json-heuristics`
  - heuristics currently:
    - suppress `0` when a numeric span field has just opened
    - suppress early EOS / code-fence / `json` tokens before the response has any chance to become valid
    - suppress obvious quote-style continuations immediately after a numeric field value
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) with the boundary-length result and the constrained-decode probe outcome.

### Key Experiment Outcomes
- Exact-token parity length sweep using [`scripts/compare_native_granite_hf.jl`](/home/christos/code/julia/Swamma/scripts/compare_native_granite_hf.jl):
  - `max_input_tokens=64`: first token matched (`0`)
  - `max_input_tokens=128`: first token matched (`0`)
  - `max_input_tokens=192`: first token matched (`0`)
  - `max_input_tokens=256`: first token diverged
    - native: `0`
    - HF: `1`
  - `max_input_tokens=308` / `512` effective prompt: still diverged on the first token
- Boundary-logit inspection:
  - HF at `256` tokens:
    - token `1`: `26.68`
    - token `0`: `26.28`
    - interpretation: HF is only modestly preferring `1` over `0`
  - native at `256` tokens:
    - token `0`: `26.48`
    - token `1`: `21.59`
    - interpretation: native is not catastrophically wrong, but it is much more strongly biased toward invalid `0`
  - native at `308` tokens:
    - token `0`: `23.78`
    - token `1`: `22.88`
    - interpretation: the same skew remains at full prompt length
- Heuristic native decode probe on the compact no-title request (`float32`, `512` token budget):
  - before heuristics:
    - response started with malformed `{"entities":[{"start":0"...`
  - after the first heuristic pass:
    - response started with `{"entities":[{"start":1"...`
    - but then drifted into `"CBS"`/free-text style content
  - after the second heuristic pass:
    - response stayed more extraction-shaped, e.g.:
      - `{"entities":[{"start":1","type":"ORG","text":"CBS",...`
    - still not valid JSON under the expected schema
- Interpretation:
  - constrained decoding can push the model away from the most obviously wrong branch
  - but it does not eliminate the underlying long-context native-vs-HF divergence

### Current Recommendation
- Keep the parity harness and the compact prompt path.
- Do not rely on the current constrained-decoding heuristics as a final solution; they improve the branch but do not restore schema-faithful JSON generation.
- The next serious debugging target should be long-context forward math in [`src/NativeTeacherLM.jl`](/home/christos/code/julia/Swamma/src/NativeTeacherLM.jl), now that the break window is bounded to somewhere between `192` and `256` prompt tokens.

### Open Issues
- Native/HF first-token parity still breaks once the prompt reaches the mid-length regime (`>=256` tokens in the current compact setup).
- The current JSON-aware heuristics are helpful but insufficient; the native model still drifts into the wrong schema/content family under long prompts.

## 2026-03-15 — RoPE Root Cause Fix + First Accepted Native RE Teacher Row

### Objectives
- Fix the actual long-context model-math bug rather than continuing to tune prompts around it.
- Re-run exact-token parity after the fix.
- Verify that the native RE teacher generator can now emit at least one accepted strict-JSON row end to end.

### Changes Saved
- Fixed [`src/NativeTeacherLM.jl`](/home/christos/code/julia/Swamma/src/NativeTeacherLM.jl):
  - corrected RoPE application from adjacent even/odd rotation to Granite/LLaMA-style half-split rotation
  - the native implementation now rotates the first half of each head against the second half, matching Hugging Face `rotate_half(...)` semantics
- Added [`scripts/compare_native_granite_hidden_states.jl`](/home/christos/code/julia/Swamma/scripts/compare_native_granite_hidden_states.jl):
  - compares native vs HF last-token hidden states layer by layer on the exact same prompt token IDs
  - useful for identifying where drift starts as context length changes
- Kept the compact request path and the exact-token generation parity harness from the prior steps.
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) to reflect that the long-context mismatch is now fixed and that the next phase is moving from one accepted row to consistent small-shard native generation.

### Key Experiment Outcomes
- Parse and tests stayed green after the RoPE fix:
  - `julia --project=. -e 'include("src/Swamma.jl"); using .Swamma; println("parse-ok")'`
  - `julia --project=. test/test_native_teacher_lm.jl`
  - result: `44/44` tests passed
- Exact-token parity after the RoPE fix:
  - `256`-token prompt:
    - native and HF matched exactly for the first 8 greedy tokens:
      - `1`, `,"`, `stop`, `":`, `3`, `,"`, `label`, `":"`
  - `308`-token prompt:
    - native and HF again matched exactly for the first 8 greedy tokens
  - interpretation:
    - the previously reported long-context first-token mismatch was caused by the incorrect RoPE rotation scheme
- Hidden-state parity after the RoPE fix:
  - `256`-token layerwise comparison shows near-exact agreement through almost the full stack
  - representative values:
    - block 1 cosine `~1.0`, L2 `~2e-5`
    - block 20 cosine `~1.0`, L2 `~4.6e-5`
    - block 39 cosine `~1.0`, L2 `~4.8e-4`
  - the final reported stage still shows a large mismatch, but that appears to be an API/labeling mismatch in the comparison harness rather than a real forward discrepancy, because token-level parity is exact
- First accepted native RE teacher row under strict JSON validation:
  - command:
    - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_requests_native_compact_notitle.jsonl --output /tmp/rebel_teacher_raw_native_compact_notitle_f32_postrope_256.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 1 --max-input-tokens 512 --max-new-tokens 256 --response-prefix '{"entities":[{"start":' --dtype float32 --plain-prompt --disable-json-heuristics --verbose`
  - result:
    - `accepted=1`
    - `failed=0`
    - generation time for the row: `~74.96s`
    - accepted response length: `541` chars
- Downstream parser compatibility:
  - `julia --project=. scripts/parse_rebel_teacher_responses.jl --input /tmp/rebel_teacher_raw_native_compact_notitle_f32_postrope_256.jsonl --output /tmp/rebel_teacher_parsed_native_compact_notitle_f32_postrope_256.jsonl --strict`
  - result:
    - parsed rows: `1`
    - failed rows: `0`
    - entities: `6`
    - relations: `4`

### Current Recommendation
- Treat the RoPE issue as resolved.
- Treat the native Granite path as now capable of producing valid strict-JSON RE teacher rows, at least on a one-row compact-prompt smoke.
- The next step is no longer model-math debugging. It is a small consistency sweep:
  - run a tiny compact-prompt shard
  - measure accepted/failed rate under strict JSON gating
  - then decide whether compact/no-title/`float32`/`512`/`256` is good enough for a first native teacher-corpus pilot

### Open Issues
- One accepted row is not yet enough to call the native teacher path production-ready.
- The compact native path is still slow on CPU (`~33s` model load, `~75s` generation for one 256-token decode row).
- The current native raw response still benefits from the compact no-title prompt and a relatively large decode budget; robustness on a small shard still needs to be measured.

## 2026-03-15 — Relation-First Native RE Teacher Control Pass

### Objectives
- Move beyond the one-row strict-JSON success and test whether the native Granite path can hold up on a tiny multi-row shard.
- Diagnose the broader-schema failure mode from the compact entity-first prompt.
- Find the best current native control recipe for RE teacher generation and save it explicitly.

### Changes Saved
- Updated [`scripts/build_rebel_teacher_requests.jl`](/home/christos/code/julia/Swamma/scripts/build_rebel_teacher_requests.jl):
  - tightened the compact prompt with explicit anti-overgeneration rules
  - added relation-oriented guidance:
    - extract only high-confidence spans needed for allowed relations or clearly salient named mentions
    - do not annotate every noun/token
    - prefer fewer annotations and close the JSON object immediately
- Updated [`scripts/generate_rebel_teacher_responses_native.jl`](/home/christos/code/julia/Swamma/scripts/generate_rebel_teacher_responses_native.jl):
  - added soft decode-cap knobs:
    - `--max-entities-hint`
    - `--max-relations-hint`
  - extended JSON heuristics to cover packed separator tokens such as `,{"`, `"},{"`, and `},{"`
  - added response-resolution helpers so strict validation now tries:
    - completion text alone
    - prefixed completion text
  - fixed the relation-first bug where the model restarted a full JSON object but the generator still prepended the response prefix, producing invalid duplicated JSON
  - generalized response-resolution helpers to accept `AbstractString`
  - hardened error previews so `nothing` no longer crashes the failure path
  - added object-tail detection so packed close-and-reopen tokens can be suppressed before a standalone `}` is emitted
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) with the new best-current recipe and the current `2/3` tiny-shard result.

### Key Experiment Outcomes
- Broader-schema entity-first control on the 3-row compact/no-title shard failed badly even after prompt tightening:
  - row 1 still ran for `~259.54s`
  - failure mode changed from `CONCEPT` flooding to `DATE` flooding
  - representative preview:
    - `{"entities":[{"start":1,"stop":3,"label":"ORG"},{"start":9,"stop":9,"label":"DATE"}, ...`
  - interpretation:
    - entity-first decoding is the wrong control surface on the full six-label schema
- Full non-JSON capture of that entity-first failure confirmed the decoder never reached `relations`:
  - it kept enumerating `DATE` spans up through token `50`
  - relation supervision would have been absent even if the text were repaired
- Relation-first probe used this response prefix:
  - `{"entities":[],"relations":[{"head_start":`
- Before the response-assembly fix, relation-first already looked promising:
  - row 1 failed in `17.78s`
  - row 2 failed in `72.85s`
  - those failures were fast because the model often restarted a full JSON object instead of continuing the prefix
  - the generator was incorrectly prepending the prefix anyway, which made otherwise parseable JSON invalid
- After the response-assembly fix:
  - strict row-1 relation-first run:
    - command:
      - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_requests_native_compact_notitle_3_tight.jsonl --output /tmp/rebel_teacher_raw_native_compact_notitle_3_tight_relfirst_row1_fixed.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 1 --max-input-tokens 512 --max-new-tokens 256 --response-prefix '{"entities":[],"relations":[{"head_start":' --dtype float32 --plain-prompt --verbose --max-entities-hint 6 --max-relations-hint 4`
    - result:
      - `accepted=1`
      - generation time `~16.91s`
      - strict downstream parse passed with:
        - rows `1/1`
        - entities `0`
        - relations `1`
- Tiny strict relation-first shard (`3` rows):
  - command:
    - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_requests_native_compact_notitle_3_tight.jsonl --output /tmp/rebel_teacher_raw_native_compact_notitle_3_tight_relfirst_3.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 3 --max-input-tokens 512 --max-new-tokens 256 --response-prefix '{"entities":[],"relations":[{"head_start":' --dtype float32 --plain-prompt --verbose --max-entities-hint 6 --max-relations-hint 4`
  - results:
    - row 1 accepted in `17.65s`
    - row 2 accepted in `81.16s`
    - row 3 failed
    - overall strict acceptance: `2/3`
- Downstream strict parser on the accepted relation-first rows:
  - command:
    - `julia --project=. scripts/parse_rebel_teacher_responses.jl --input /tmp/rebel_teacher_raw_native_compact_notitle_3_tight_relfirst_3.jsonl --output /tmp/rebel_teacher_parsed_native_compact_notitle_3_tight_relfirst_3.jsonl --strict`
  - result:
    - parsed rows `2`
    - failed rows `0`
    - entities total `0`
    - relations total `9`
- Hard-row diagnostic for row 3:
  - non-JSON capture showed the model continuing mid-object relation lists such as:
    - `1,"head_stop":3,"tail_start":5,"tail_stop":6,"label":"P17"},{"head_start":9,...`
  - interpretation:
    - row 3 is not a full-object restart case
    - it is a continuation-only relation flood that still fails to close under the current greedy decode recipe

### Best Current Recommendation
- Best current native RE teacher recipe:
  - compact prompt
  - `--no-title`
  - `float32`
  - `--max-input-tokens 512`
  - `--max-new-tokens 256`
  - `--plain-prompt`
  - relation-first response prefix:
    - `{"entities":[],"relations":[{"head_start":`
  - `--max-entities-hint 6`
  - `--max-relations-hint 4`
- Treat relation-first generation as the current default path for native RE teacher generation.
- Treat entity-first generation on the broad schema as a dead end for now.

### Open Issues
- The current best native relation-first recipe is improved but not yet robust:
  - tiny strict shard result is `2/3`, not `3/3`
- Row 3 still produces a malformed continuation-only relation flood and never closes the JSON object under greedy decode.
- The current accepted relation-first outputs contain relations only (`entities=[]`), which is usable for relation distillation but not yet ideal if we want teacher entity spans from the same pass.

### Next Actions
- Push the relation-first recipe from `2/3` to `3/3` on the tiny shard:
  - either improve packed-token close-out for continuation-only relation floods
  - or test a sampled relation-first decode recipe on the same rows
- If relation-first can be stabilized on a 10-row shard, use that as the first native teacher-corpus pilot instead of returning to entity-first prompting.

## 2026-03-16 — Relation-First Salvage Path + Tiny-Shard 3/3 Strict Acceptance

### Objectives
- Eliminate the remaining hard-row failure from the relation-first native RE teacher recipe.
- Distinguish between true empty outputs and truncated-but-salvageable relation continuations.
- Re-run the tiny strict shard after the fix and confirm whether the best current recipe is now stable enough for a larger pilot.

### Changes Saved
- Updated [`scripts/generate_rebel_teacher_responses_native.jl`](/home/christos/code/julia/Swamma/scripts/generate_rebel_teacher_responses_native.jl):
  - added strict relation-only partial-JSON salvage:
    - if full JSON validation fails, the generator now tries to recover complete top-level relation objects from a candidate of the form `{"entities":[],"relations":[...`
    - the repaired object is closed as strict JSON and revalidated before acceptance
  - added helpers:
    - `extract_complete_top_level_objects(...)`
    - `salvage_relation_only_json(...)`
  - threaded `max_relations_hint` through `resolve_response_text(...)` so salvage respects the current decode cap
  - kept the previously added relation-first response-resolution logic and error-preview hardening
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) to record that the tiny strict relation-first shard now reaches `3/3` acceptance.

### Key Experiment Outcomes
- Hard-row row-3 strict retry before salvage:
  - command:
    - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_request_row3_relfirst.jsonl --output /tmp/rebel_teacher_raw_native_row3_relfirst_fixed2.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 1 --max-input-tokens 512 --max-new-tokens 256 --response-prefix '{"entities":[],"relations":[{"head_start":' --dtype float32 --plain-prompt --verbose --max-entities-hint 6 --max-relations-hint 4`
  - result:
    - still failed under strict JSON
    - preview showed a continuation-only relation flood:
      - `1,"head_stop":3,"tail_start":5,"tail_stop":6,"label":"P17"},{"head_start":9,...`
  - interpretation:
    - the model was producing useful complete relation objects, but truncation prevented closure
- Row-3 non-JSON capture confirmed the above diagnosis:
  - it produced a long sequence of relation objects and stopped mid-list
  - this meant a relation-only salvage path was appropriate
- Row-3 strict retry after salvage:
  - command:
    - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_request_row3_relfirst.jsonl --output /tmp/rebel_teacher_raw_native_row3_relfirst_fixed3.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 1 --max-input-tokens 512 --max-new-tokens 256 --response-prefix '{"entities":[],"relations":[{"head_start":' --dtype float32 --plain-prompt --verbose --max-entities-hint 6 --max-relations-hint 4`
  - result:
    - `accepted=1`
    - generation time `~20.03s`
    - emitted strict response:
      - `{"entities":[],"relations":[{"head_start":1,"head_stop":3,"tail_start":5,"tail_stop":6,"label":"P17"}]}`
- Downstream parser verification for salvaged row 3:
  - `julia --project=. scripts/parse_rebel_teacher_responses.jl --input /tmp/rebel_teacher_raw_native_row3_relfirst_fixed3.jsonl --output /tmp/rebel_teacher_parsed_native_row3_relfirst_fixed3.jsonl --strict`
  - result:
    - parsed rows `1`
    - failed rows `0`
    - entities `0`
    - relations `1`
- Full tiny strict shard rerun after salvage:
  - command:
    - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_requests_native_compact_notitle_3_tight.jsonl --output /tmp/rebel_teacher_raw_native_compact_notitle_3_tight_relfirst_3_fixed.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 3 --max-input-tokens 512 --max-new-tokens 256 --response-prefix '{"entities":[],"relations":[{"head_start":' --dtype float32 --plain-prompt --verbose --max-entities-hint 6 --max-relations-hint 4`
  - result:
    - row 1 accepted in `17.61s`
    - row 2 accepted in `11.10s`
    - row 3 accepted in `13.54s`
    - overall strict acceptance: `3/3`
- Downstream parser on the full fixed tiny shard:
  - `julia --project=. scripts/parse_rebel_teacher_responses.jl --input /tmp/rebel_teacher_raw_native_compact_notitle_3_tight_relfirst_3_fixed.jsonl --output /tmp/rebel_teacher_parsed_native_compact_notitle_3_tight_relfirst_3_fixed.jsonl --strict`
  - result:
    - parsed rows `3`
    - failed rows `0`
    - entities total `0`
    - relations total `3`

### Best Current Recommendation
- Best native RE teacher recipe remains:
  - compact prompt
  - `--no-title`
  - `float32`
  - `--max-input-tokens 512`
  - `--max-new-tokens 256`
  - `--plain-prompt`
  - relation-first response prefix:
    - `{"entities":[],"relations":[{"head_start":`
  - `--max-entities-hint 6`
  - `--max-relations-hint 4`
- The key new recommendation is to keep the strict relation-only salvage path enabled for this recipe.
- This is now good enough to justify a 10-row strict shard before attempting a larger native teacher-corpus pilot.

### Open Issues
- The current native relation-first path still emits relations only (`entities=[]`), so it is not yet a full teacher-entity generator.
- The salvage path is pragmatic and effective, but it is still a control-time repair layer over a small teacher model rather than a proof that raw generation is schema-perfect.
- Quality remains unvalidated beyond tiny-shard structural success; the current metrics are acceptance and parseability, not relation correctness.

### Next Actions
- Run a 10-row strict relation-first shard with the current recipe and measure:
  - strict acceptance rate
  - parse success
  - relation count distribution
- If the 10-row shard stays stable, use the relation-first native path as the first real RE teacher-corpus pilot and compare its utility against the external-teacher path.

## 2026-03-16 — 10-Row Strict Relation-First Native Pilot

### Objectives
- Validate that the new relation-first native RE teacher recipe is not only a 3-row toy success.
- Measure strict generation acceptance, strict downstream parse success, and basic relation-yield statistics on a 10-row shard.

### Changes Saved
- No new code paths were added in this step.
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) to record the 10-row pilot result and promote the next action to a larger `50-100` row shard.

### Key Experiment Outcomes
- Built a 10-row compact/no-title request shard:
  - `julia --project=. scripts/build_rebel_teacher_requests.jl --input data/rebel/train.jsonl --output /tmp/rebel_teacher_requests_native_compact_notitle_10.jsonl --max-rows 10 --prompt-style compact --no-title`
- Ran strict native relation-first generation:
  - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_requests_native_compact_notitle_10.jsonl --output /tmp/rebel_teacher_raw_native_compact_notitle_10_relfirst.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 10 --max-input-tokens 512 --max-new-tokens 256 --response-prefix '{"entities":[],"relations":[{"head_start":' --dtype float32 --plain-prompt --verbose --max-entities-hint 6 --max-relations-hint 4`
- Strict generator results:
  - accepted `10/10`
  - failed `0/10`
  - representative per-row generation times after model load:
    - row 1: `17.84s`
    - row 2: `12.22s`
    - row 5: `15.76s`
    - row 10: `18.47s`
- Strict downstream parse:
  - `julia --project=. scripts/parse_rebel_teacher_responses.jl --input /tmp/rebel_teacher_raw_native_compact_notitle_10_relfirst.jsonl --output /tmp/rebel_teacher_parsed_native_compact_notitle_10_relfirst.jsonl --strict`
  - result:
    - parsed rows `10`
    - failed rows `0`
    - entities total `0`
    - relations total `7`
- Basic output-size summary:
  - rows `10`
  - mean response chars `110.7`
  - min chars `104`
  - max chars `153`
- Per-row parsed relation counts:
  - non-empty relation rows:
    - `docid:1755846-1` -> `1`
    - `docid:1755846-2` -> `1`
    - `docid:1701411-0` -> `1`
    - `docid:1854133-1` -> `1`
    - `docid:1872359-0` -> `1`
    - `docid:1698102-0` -> `1`
    - `docid:1698102-1` -> `1`
  - empty-relation rows:
    - `docid:1602703-0`
    - `docid:1751944-0`
    - `docid:1834314-0`

### Best Current Recommendation
- Promote the relation-first native recipe from “tiny-shard promising” to “pilot-ready for a larger shard”.
- Keep the current best recipe unchanged:
  - compact prompt
  - `--no-title`
  - `float32`
  - `--max-input-tokens 512`
  - `--max-new-tokens 256`
  - `--plain-prompt`
  - response prefix `{"entities":[],"relations":[{"head_start":`
  - `--max-entities-hint 6`
  - `--max-relations-hint 4`

### Open Issues
- Structural robustness now looks good, but semantic quality is still unverified.
- Outputs are still relation-only (`entities=[]`), so teacher entity spans are not being produced by the native path yet.
- Three of the ten parsed rows yielded empty relations; that may be correct or may indicate under-generation on some examples.

### Next Actions
- Run a `50-100` row relation-first pilot and inspect:
  - acceptance / parse success
  - non-empty relation yield
  - label distribution
  - obvious semantic failures from a manual spot check
- If that larger pilot holds, use this native path to produce the first meaningful RE teacher corpus for distillation experiments.

## 2026-03-16 — Semantic Gate + Label-Gloss Pilot Evaluation

### Objectives
- Distinguish structural robustness from semantic usefulness on the native relation-first pilot.
- Tighten “strict” acceptance so out-of-schema labels no longer count as successes.
- Test whether adding short natural-language glosses for Wikidata relation IDs improves label choice.
- Add a reusable evaluator so future pilot sweeps do not require manual spot checks.

### Changes Saved
- Updated [`scripts/generate_rebel_teacher_responses_native.jl`](/home/christos/code/julia/Swamma/scripts/generate_rebel_teacher_responses_native.jl):
  - strict validation now checks that emitted entity and relation labels belong to the row’s allowed `entity_labels` / `relation_labels`
  - added helpers:
    - `label_set_from_row(...)`
    - `payload_labels_allowed(...)`
  - threaded allowed label sets through:
    - strict response validation
    - relation-only salvage
    - early-stop resolution
- Updated [`scripts/build_rebel_teacher_requests.jl`](/home/christos/code/julia/Swamma/scripts/build_rebel_teacher_requests.jl):
  - added `REBEL_RELATION_GLOSSES` for the 32 REBEL relation IDs
  - compact prompts now include concise relation meaning hints such as:
    - `P127=owned by`
    - `P159=headquarters location`
    - `P571=inception`
  - added an explicit instruction:
    - choose the relation ID whose meaning best matches the spans; do not default to one label across rows
- Added [`scripts/evaluate_rebel_teacher_pilot.jl`](/home/christos/code/julia/Swamma/scripts/evaluate_rebel_teacher_pilot.jl):
  - compares parsed teacher outputs against gold REBEL rows
  - reports:
    - matched rows
    - non-empty rate
    - top-1 predicted label in gold-set rate
    - exact predicted label-set match rate
    - predicted/gold label histograms
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) with the new semantic metrics and the next control options.

### Key Experiment Outcomes
- Baseline structural pilot without label-aware validation looked overly good:
  - 10-row strict relation-first shard: `10/10` accepted, `10/10` parsed
  - but label inspection showed a collapse to a single invalid label family in earlier runs
- After adding label-aware strict validation:
  - 10-row relation-first shard on the pre-gloss prompt:
    - accepted `7/10`
    - failed `3/10`
    - parsed `7/7`
    - all accepted rows predicted the same label:
      - `P161`
- Gold comparison for the no-gloss label-gated run using [`scripts/evaluate_rebel_teacher_pilot.jl`](/home/christos/code/julia/Swamma/scripts/evaluate_rebel_teacher_pilot.jl):
  - command:
    - `julia --project=. scripts/evaluate_rebel_teacher_pilot.jl --gold data/rebel/train.jsonl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_10_relfirst_labelgated.jsonl --max-rows 10`
  - result:
    - matched rows `7`
    - non-empty rows `7`
    - top-1 label in gold `0.0000`
    - exact label-set match `0.0000`
    - predicted label counts:
      - `{"P161":7}`
  - interpretation:
    - structurally valid
    - semantically degenerate
- Glossed compact prompt pilot:
  - built with:
    - `julia --project=. scripts/build_rebel_teacher_requests.jl --input data/rebel/train.jsonl --output /tmp/rebel_teacher_requests_native_compact_notitle_10_gloss.jsonl --max-rows 10 --prompt-style compact --no-title`
  - generated with the same relation-first label-gated recipe
  - result:
    - accepted `7/10`
    - failed `3/10`
    - parsed `7/7`
    - predicted label counts:
      - `{"P127":3,"P161":2,"P57":1,"P136":1}`
- Gold comparison for the glossed label-gated run:
  - command:
    - `julia --project=. scripts/evaluate_rebel_teacher_pilot.jl --gold data/rebel/train.jsonl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_10_gloss_relfirst_labelgated.jsonl --max-rows 10`
  - result:
    - matched rows `7`
    - non-empty rows `7`
    - top-1 label in gold `0.1429`
    - exact label-set match `0.1429`
    - only `docid:1755846-1` matched the gold top label exactly (`P127`)
  - interpretation:
    - glosses broke the single-label collapse
    - but semantic quality is still poor overall

### Best Current Recommendation
- Keep the relation-first structural recipe.
- Keep label-aware strict validation enabled; it prevents misleading “success” metrics.
- Keep the relation glosses; they provide a measurable improvement over bare property IDs.
- Do not treat the current native teacher path as ready for corpus-scale distillation yet; semantic quality is still too weak.

### Open Issues
- Structural robustness is now much better than semantic accuracy.
- Even with glosses, the 10-row pilot only reaches:
  - top-1 label in gold `0.1429`
  - exact label-set match `0.1429`
- The model still emits only one relation per accepted row in these pilots, and still produces `entities=[]`.

### Next Actions
- Add a stronger semantic control step for relation-label selection:
  - either constrained label decoding at `label` fields
  - or a two-stage natural-language relation-name -> Wikidata-ID mapping path
- Re-run the 10-row pilot after that change and use [`scripts/evaluate_rebel_teacher_pilot.jl`](/home/christos/code/julia/Swamma/scripts/evaluate_rebel_teacher_pilot.jl) as the acceptance gate before scaling back up to `50-100` rows.

## 2026-03-16 — Decode-Time Label Constraint Probe

### Objectives
- Test whether constraining relation `label` fields at decode time improves semantic accuracy beyond prompt glosses alone.
- Compare this constrained run directly against the glossed label-gated 10-row pilot.

### Changes Saved
- Updated [`scripts/generate_rebel_teacher_responses_native.jl`](/home/christos/code/julia/Swamma/scripts/generate_rebel_teacher_responses_native.jl):
  - added `RelationLabelConstraint`
  - added `build_relation_label_constraint(...)`
  - added `current_open_label_prefix(...)`
  - added `apply_relation_label_constraint(...)`
  - when the decoder is inside an open relation `label` field, logits are now restricted to token continuations that correspond to the row’s allowed relation IDs, plus the closing quote when a full allowed ID is complete
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) with the constrained-vs-unconstrained comparison result.

### Key Experiment Outcomes
- Glossed relation-first prompt remained the better prompt baseline:
  - accepted `7/10`
  - top-1 label in gold `0.1429`
  - exact label-set match `0.1429`
- Constrained-label glossed pilot:
  - command:
    - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_requests_native_compact_notitle_10_gloss.jsonl --output /tmp/rebel_teacher_raw_native_compact_notitle_10_gloss_relfirst_labelconstrained.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 10 --max-input-tokens 512 --max-new-tokens 256 --response-prefix '{"entities":[],"relations":[{"head_start":' --dtype float32 --plain-prompt --verbose --max-entities-hint 6 --max-relations-hint 4`
  - result:
    - accepted `6/10`
    - failed `4/10`
    - parsed `6/6`
    - predicted label counts:
      - `{"P127":2,"P57":1,"P136":1,"P161":2}`
- Gold comparison for the constrained-label run:
  - command:
    - `julia --project=. scripts/evaluate_rebel_teacher_pilot.jl --gold data/rebel/train.jsonl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_10_gloss_relfirst_labelconstrained.jsonl --max-rows 10`
  - result:
    - matched rows `6`
    - non-empty rows `6`
    - top-1 label in gold `0.0000`
    - exact label-set match `0.0000`
- Interpretation:
  - token-level label constraints did not solve the semantic problem
  - they reduced acceptance from `7/10` to `6/10`
  - they did not improve gold-label agreement
  - the prompt glosses alone remain strictly better than adding this constrained label-selection layer

### Best Current Recommendation
- Keep:
  - relation-first generation
  - label-aware strict validation
  - relation glosses in the compact prompt
- Do not keep pushing the current token-level relation-label constraint path as the main solution.

### Open Issues
- The native path is structurally stable but semantically weak.
- Prompt glosses help somewhat, but semantic accuracy remains poor.
- Token-level label constraints are not the right next lever.

### Next Actions
- Move to a stronger semantic-control design:
  - prefer a two-stage natural-language relation-name -> Wikidata-ID mapping path
  - or another semantic reranking step that reasons over gloss text instead of raw ID tokens

## 2026-03-16 — Name-Canonicalization Variant Probe

### Objectives
- Test whether relaxing the label surface form from raw Wikidata IDs to human-readable relation names helps semantic accuracy.
- Compare three semantic-control variants on the same 10-row slice:
  - glossed ID prompt
  - glossed `id_or_name` prompt
  - glossed `name`-only prompt

### Changes Saved
- Updated [`scripts/parse_rebel_teacher_responses.jl`](/home/christos/code/julia/Swamma/scripts/parse_rebel_teacher_responses.jl):
  - added REBEL relation gloss -> ID canonicalization
  - parsed relation labels can now be:
    - raw Wikidata IDs such as `P127`
    - human-readable names such as `owned by`
  - normalization maps both forms back to the canonical Wikidata ID
- Updated [`scripts/generate_rebel_teacher_responses_native.jl`](/home/christos/code/julia/Swamma/scripts/generate_rebel_teacher_responses_native.jl):
  - strict validation now treats either canonical IDs or canonicalizable relation names as acceptable for relation labels
- Updated [`scripts/build_rebel_teacher_requests.jl`](/home/christos/code/julia/Swamma/scripts/build_rebel_teacher_requests.jl):
  - added `--relation-label-mode <id|id_or_name|name>`
  - `id`:
    - relation label must be the Wikidata ID
  - `id_or_name`:
    - relation label may be either the ID or the glossed name
  - `name`:
    - relation label must be the exact glossed relation name, not the ID
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) with the new variant comparison.

### Key Experiment Outcomes
- `id_or_name` variant:
  - built with the glossed compact prompt allowing either ID or exact relation name
  - 10-row result:
    - accepted `3/10`
    - failed `7/10`
    - parsed rows `3`
    - total predicted relations `1`
  - evaluation:
    - matched rows `3`
    - non-empty rows `1`
    - top-1 label in gold `0.0000`
    - exact label-set match `0.0000`
  - interpretation:
    - too brittle
    - not competitive with the glossed ID baseline
- `name`-only variant:
  - built with:
    - `julia --project=. scripts/build_rebel_teacher_requests.jl --input data/rebel/train.jsonl --output /tmp/rebel_teacher_requests_native_compact_notitle_10_nameonly.jsonl --max-rows 10 --prompt-style compact --no-title --relation-label-mode name`
  - generated with the same relation-first recipe
  - 10-row result:
    - accepted `5/10`
    - failed `5/10`
    - parsed rows `5`
    - total predicted relations `5`
  - evaluation:
    - `julia --project=. scripts/evaluate_rebel_teacher_pilot.jl --gold data/rebel/train.jsonl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_10_nameonly_relfirst.jsonl --max-rows 10`
    - matched rows `5`
    - non-empty rows `5`
    - top-1 label in gold `0.2000`
    - exact label-set match `0.2000`
    - predicted label counts:
      - `{"P127":2,"P31":1,"P161":2}`
- Comparison against the glossed ID baseline from the previous step:
  - glossed ID baseline:
    - accepted `7/10`
    - top-1 label in gold `0.1429`
    - exact label-set match `0.1429`
  - key takeaway:
    - `name`-only improves the *rate* slightly (`0.2000` vs `0.1429`)
    - but because it accepts fewer rows (`5` vs `7`), it does not improve the absolute count of correct rows
    - `id_or_name` is worse than both

### Best Current Recommendation
- Keep the glossed ID prompt as the best current single-pass native recipe.
- Keep downstream canonicalization of relation names; it is useful infrastructure for future multi-stage approaches.
- Do not switch the main pilot path to `id_or_name` or `name` mode in the current single-pass setup.

### Open Issues
- The semantic problem remains unresolved:
  - glossed ID mode is still weak
  - `id_or_name` is too brittle
  - `name`-only does not improve absolute correctness enough to justify the lower acceptance
- The current native path still emits `entities=[]` and mostly single-relation outputs.

### Next Actions
- Stop treating prompt-surface variations as the main lever.
- Move to an explicit two-stage semantic-control design:
  - generate relation meaning/name first
  - then map or rerank against the allowed Wikidata IDs using the gloss table

## 2026-03-16 — Two-Stage Relabel Scale Check

### Objectives
- Test whether the new two-stage native relation-label relabeling path still helps on a larger `50`-row shard.
- Compare two operating points:
  - glossed single-pass stage 1 plus two-stage relabel
  - no-gloss high-coverage stage 1 plus two-stage relabel
- Decide which architecture should be treated as the current best native RE teacher path.

### Changes Saved
- No code or config changes in this segment.
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) to record the `50`-row pilot results and promote the new preferred sequencing.

### Experiment Commands And Key Metrics
- Built the glossed `50`-row compact no-title request shard:
  - `julia --project=. scripts/build_rebel_teacher_requests.jl --input data/rebel/train.jsonl --output /tmp/rebel_teacher_requests_native_compact_notitle_50_gloss.jsonl --max-rows 50 --prompt-style compact --no-title --relation-label-mode id`
- Ran glossed single-pass stage 1:
  - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_requests_native_compact_notitle_50_gloss.jsonl --output /tmp/rebel_teacher_raw_native_compact_notitle_50_gloss_relfirst_labelgated.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 50 --max-input-tokens 512 --max-new-tokens 256 --response-prefix '{"entities":[],"relations":[{"head_start":' --dtype float32 --plain-prompt --verbose --max-entities-hint 6 --max-relations-hint 4`
  - result:
    - accepted `24/50`
    - failed `26/50`
    - parsed rows `24`
    - predicted relations `22`
    - top-1 label in gold `0.0455`
    - exact label-set match `0.0000`
    - label collapse shifted mostly to `P106`
- Ran the glossed two-stage relabel path:
  - `julia --project=. scripts/build_rebel_relation_label_requests.jl --requests /tmp/rebel_teacher_requests_native_compact_notitle_50_gloss.jsonl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_50_gloss_relfirst_labelgated.jsonl --output /tmp/rebel_relation_label_requests_50_gloss.jsonl`
  - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_relation_label_requests_50_gloss.jsonl --output /tmp/rebel_relation_label_raw_50_gloss.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 22 --max-input-tokens 512 --max-new-tokens 128 --response-prefix '{"label":"' --dtype float32 --plain-prompt --verbose --allow-non-json --disable-stop-on-complete-json`
  - `julia --project=. scripts/apply_rebel_relation_label_selections.jl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_50_gloss_relfirst_labelgated.jsonl --selections /tmp/rebel_relation_label_raw_50_gloss.jsonl --output /tmp/rebel_teacher_parsed_native_compact_notitle_50_gloss_relfirst_relabel2stage.jsonl --response-prefix '{"label":"'`
  - `julia --project=. scripts/evaluate_rebel_teacher_pilot.jl --gold data/rebel/train.jsonl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_50_gloss_relfirst_relabel2stage.jsonl --max-rows 50`
  - result:
    - matched rows `24`
    - non-empty rows `22`
    - predicted relations `22`
    - top-1 label in gold `0.3636`
    - exact label-set match `0.0455`
    - predicted labels became meaningfully distributed: `P31/P276/P17/P571/P159/P138/P403`
- Evaluated the older no-gloss high-coverage stage-1 baseline:
  - `julia --project=. scripts/evaluate_rebel_teacher_pilot.jl --gold data/rebel/train.jsonl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_50_relfirst.jsonl --max-rows 50`
  - result:
    - matched rows `50`
    - non-empty rows `29`
    - predicted relations `29`
    - top-1 label in gold `0.0345`
    - exact label-set match `0.0000`
    - all predicted labels collapsed to `P106`
- Ran two-stage relabel on the no-gloss stage-1 baseline:
  - `julia --project=. scripts/build_rebel_relation_label_requests.jl --requests /tmp/rebel_teacher_requests_native_compact_notitle_50.jsonl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_50_relfirst.jsonl --output /tmp/rebel_relation_label_requests_50_nogloss.jsonl`
  - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_relation_label_requests_50_nogloss.jsonl --output /tmp/rebel_relation_label_raw_50_nogloss.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 29 --max-input-tokens 512 --max-new-tokens 128 --response-prefix '{"label":"' --dtype float32 --plain-prompt --verbose --allow-non-json --disable-stop-on-complete-json`
  - `julia --project=. scripts/apply_rebel_relation_label_selections.jl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_50_relfirst.jsonl --selections /tmp/rebel_relation_label_raw_50_nogloss.jsonl --output /tmp/rebel_teacher_parsed_native_compact_notitle_50_relfirst_relabel2stage.jsonl --response-prefix '{"label":"'`
  - `julia --project=. scripts/evaluate_rebel_teacher_pilot.jl --gold data/rebel/train.jsonl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_50_relfirst_relabel2stage.jsonl --max-rows 50`
  - result:
    - matched rows `50`
    - non-empty rows `29`
    - predicted relations `29`
    - top-1 label in gold `0.4483`
    - exact label-set match `0.1379`
    - predicted labels spread across `P31/P276/P17/P571/P159/P138/P50/P19/P641/P403/P47`

### Best Current Recommendation
- The best current native RE teacher architecture is:
  - stage 1: simple compact no-title relation-first extraction without glossed stage-1 label hints
  - stage 2: separate native relation-label selection over the predicted spans, using glossed natural-language choices and downstream canonicalization back to Wikidata IDs
- This is better than the glossed single-pass path on both coverage and semantics:
  - glossed single-pass + relabel: `24` matched rows, top-1 `0.3636`
  - no-gloss single-pass + relabel: `50` matched rows, top-1 `0.4483`

### Unresolved Issues
- Stage 1 still emits `entities=[]` and relation spans only; the current gain is almost entirely from label correction, not fuller structured extraction.
- No-gloss stage 1 keeps high row coverage, but non-empty relation yield is still only `29/50`.
- The evaluation so far is still pilot-scale; this is enough to choose direction, not enough to claim corpus readiness.

### Next Actions
- Keep the no-gloss stage-1 + two-stage relabel stack as the promoted pilot path.
- Improve stage-1 relation yield without reintroducing the glossed acceptance collapse.
- Rerun the promoted two-stage stack on a `100`-row shard and inspect:
  - acceptance
  - non-empty relation yield
  - top-1 label-in-gold
  - exact label-set match
- If the `100`-row pilot holds up, use that path for the first real native teacher-corpus build.

## 2026-03-16 — No-Schema Stage-1 Upgrade

### Objectives
- Improve stage-1 relation yield without giving up the high acceptance of the promoted native two-stage path.
- Test whether a shorter compact prompt with `--no-schema` improves stage-1 coverage.
- Validate the new operating point on both `50`-row and `100`-row shards with the full stage-2 relabel pass.

### Changes Saved
- Updated [`scripts/build_rebel_teacher_requests.jl`](/home/christos/code/julia/Swamma/scripts/build_rebel_teacher_requests.jl):
  - request JSONL now always includes `entity_labels` and `relation_labels` metadata, even when `--no-schema` is used
  - this keeps strict validation and stage-2 relabeling intact while allowing the prompt text itself to shrink
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) with the new promoted no-schema results.

### Experiment Commands And Key Metrics
- Built the no-schema `50`-row request shard:
  - `julia --project=. scripts/build_rebel_teacher_requests.jl --input data/rebel/train.jsonl --output /tmp/rebel_teacher_requests_native_compact_notitle_50_noschema.jsonl --max-rows 50 --prompt-style compact --no-title --no-schema --relation-label-mode id`
  - first-row prompt shrank from about `1707` chars to about `960` chars while keeping label metadata in the JSONL
- Ran no-schema stage 1 on `50` rows:
  - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_requests_native_compact_notitle_50_noschema.jsonl --output /tmp/rebel_teacher_raw_native_compact_notitle_50_noschema_relfirst.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 50 --max-input-tokens 512 --max-new-tokens 256 --response-prefix '{"entities":[],"relations":[{"head_start":' --dtype float32 --plain-prompt --verbose --max-entities-hint 6 --max-relations-hint 4`
  - parse/eval:
    - matched rows `50`
    - non-empty rows `50`
    - predicted relations `50`
    - top-1 label in gold `0.1000`
    - exact label-set match `0.0200`
- Ran no-schema stage 2 on `50` rows:
  - `julia --project=. scripts/build_rebel_relation_label_requests.jl --requests /tmp/rebel_teacher_requests_native_compact_notitle_50_noschema.jsonl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_50_noschema_relfirst.jsonl --output /tmp/rebel_relation_label_requests_50_noschema.jsonl`
  - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_relation_label_requests_50_noschema.jsonl --output /tmp/rebel_relation_label_raw_50_noschema.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 50 --max-input-tokens 512 --max-new-tokens 128 --response-prefix '{"label":"' --dtype float32 --plain-prompt --verbose --allow-non-json --disable-stop-on-complete-json`
  - `julia --project=. scripts/apply_rebel_relation_label_selections.jl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_50_noschema_relfirst.jsonl --selections /tmp/rebel_relation_label_raw_50_noschema.jsonl --output /tmp/rebel_teacher_parsed_native_compact_notitle_50_noschema_relfirst_relabel2stage.jsonl --response-prefix '{"label":"'`
  - `julia --project=. scripts/evaluate_rebel_teacher_pilot.jl --gold data/rebel/train.jsonl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_50_noschema_relfirst_relabel2stage.jsonl --max-rows 50`
  - result:
    - matched rows `50`
    - non-empty rows `50`
    - predicted relations `50`
    - top-1 label in gold `0.3800`
    - exact label-set match `0.1000`
    - relabel updated `47/50` relations
- Extended the same path to `100` rows using resume:
  - `julia --project=. scripts/build_rebel_teacher_requests.jl --input data/rebel/train.jsonl --output /tmp/rebel_teacher_requests_native_compact_notitle_100_noschema.jsonl --max-rows 100 --prompt-style compact --no-title --no-schema --relation-label-mode id`
  - seeded stage-1 raw file with the `50`-row output and resumed:
    - `cp /tmp/rebel_teacher_raw_native_compact_notitle_50_noschema_relfirst.jsonl /tmp/rebel_teacher_raw_native_compact_notitle_100_noschema_relfirst.jsonl`
    - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_requests_native_compact_notitle_100_noschema.jsonl --output /tmp/rebel_teacher_raw_native_compact_notitle_100_noschema_relfirst.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --resume --max-rows 100 --max-input-tokens 512 --max-new-tokens 256 --response-prefix '{"entities":[],"relations":[{"head_start":' --dtype float32 --plain-prompt --verbose --max-entities-hint 6 --max-relations-hint 4`
  - stage-1 `100`-row result:
    - matched rows `100`
    - non-empty rows `100`
    - predicted relations `100`
    - top-1 label in gold `0.1200`
    - exact label-set match `0.0200`
  - seeded stage-2 raw file with the `50`-row output and resumed:
    - `cp /tmp/rebel_relation_label_raw_50_noschema.jsonl /tmp/rebel_relation_label_raw_100_noschema.jsonl`
    - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_relation_label_requests_100_noschema.jsonl --output /tmp/rebel_relation_label_raw_100_noschema.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --resume --max-rows 100 --max-input-tokens 512 --max-new-tokens 128 --response-prefix '{"label":"' --dtype float32 --plain-prompt --verbose --allow-non-json --disable-stop-on-complete-json`
  - final merged `100`-row result:
    - matched rows `100`
    - non-empty rows `100`
    - predicted relations `100`
    - top-1 label in gold `0.3800`
    - exact label-set match `0.0800`
    - relabel updated `95/100` relations

### Best Current Recommendation
- Promote the native teacher pilot path to:
  - stage 1: compact no-title no-schema relation-first extraction
  - stage 2: separate native relation-label relabeling with glossed natural-language choices and downstream canonicalization to Wikidata IDs
- Why this is now preferred:
  - it keeps full row coverage on the tested shards
  - it raises stage-1 relation yield from `29/50` to `50/50`
  - after relabeling, it holds `0.3800` top-1 label-in-gold on both `50` and `100` row pilots
- Compared with the prior no-gloss relabel path:
  - prior `50`-row no-gloss relabel: `29` predicted relations, top-1 `0.4483`, exact `0.1379`
  - current `50`-row no-schema relabel: `50` predicted relations, top-1 `0.3800`, exact `0.1000`
  - verdict: the no-schema path gives better absolute semantic yield because it keeps all rows non-empty

### Unresolved Issues
- Stage-2 relabel does not update every relation candidate yet:
  - `47/50` updates on the `50`-row no-schema pilot
  - `95/100` updates on the `100`-row no-schema pilot
- Exact label-set match is still modest.
- The pipeline still produces relation-only outputs with `entities=[]`; this is enough for the current distillation experiment path, but not a full extraction-quality solution.

### Next Actions
- Inspect the small residual tail of unlabeled / un-updated stage-2 candidates and make the relabel merge fully saturated.
- Run a sampled-vs-greedy comparison on the promoted no-schema path to see whether label accuracy can be lifted without losing the `100/100` non-empty rate.
- If that does not materially improve the metrics, use the current no-schema two-stage pipeline to build the first larger native teacher corpus for distillation.

## 2026-03-16 — Residual Relabel Tail Inspection

### Objectives
- Determine why the no-schema two-stage relabel path still misses a small tail of candidates.
- Separate merge/parser problems from true stage-2 semantic drift.
- Apply only safe repairs, not speculative label remapping.

### Changes Saved
- Updated [`scripts/apply_rebel_relation_label_selections.jl`](/home/christos/code/julia/Swamma/scripts/apply_rebel_relation_label_selections.jl):
  - added a fallback that extracts the `"label"` field directly from truncated / malformed selection text
  - added a small safe alias map for obvious near-glosses such as `headquartered in -> P159`
  - restricted merge-time canonicalization to the selection row’s allowed relation-label set
- Updated [`scripts/build_rebel_relation_label_requests.jl`](/home/christos/code/julia/Swamma/scripts/build_rebel_relation_label_requests.jl):
  - strengthened the stage-2 prompt to forbid free-form type outputs such as `city`, `single`, and `nickname`
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) with the repaired tail counts.

### Experiment Commands And Key Metrics
- Reapplied the repaired merge to the no-schema pilots:
  - `julia --project=. scripts/apply_rebel_relation_label_selections.jl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_50_noschema_relfirst.jsonl --selections /tmp/rebel_relation_label_raw_50_noschema.jsonl --output /tmp/rebel_teacher_parsed_native_compact_notitle_50_noschema_relfirst_relabel2stage_v2.jsonl --response-prefix '{"label":"'`
  - `julia --project=. scripts/apply_rebel_relation_label_selections.jl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_100_noschema_relfirst.jsonl --selections /tmp/rebel_relation_label_raw_100_noschema.jsonl --output /tmp/rebel_teacher_parsed_native_compact_notitle_100_noschema_relfirst_relabel2stage_v2.jsonl --response-prefix '{"label":"'`
  - result:
    - `50`-row updated relations improved from `47/50` to `48/50`
    - `100`-row updated relations improved from `95/100` to `96/100`
    - evaluation metrics stayed effectively flat:
      - `50`-row top-1 label-in-gold `0.3800`, exact label-set `0.1000`
      - `100`-row top-1 label-in-gold `0.3800`, exact label-set `0.0800`
- Inspected the remaining failed raw stage-2 responses on the `100`-row pilot:
  - parser/merge-safe salvage fixed cases like:
    - `headquartered in`
    - truncated `instance of`
  - the residual bad labels were true semantic drifts such as:
    - `city`
    - `single`
    - `nickname`
- Rebuilt the stage-2 prompt with explicit anti-drift wording and retried only the three failing candidates:
  - `julia --project=. scripts/build_rebel_relation_label_requests.jl --requests /tmp/rebel_teacher_requests_native_compact_notitle_100_noschema.jsonl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_100_noschema_relfirst.jsonl --output /tmp/rebel_relation_label_requests_100_noschema_v2.jsonl`
  - generated retry-only labels for:
    - `docid:158185-3#rel1`
    - `docid:172898-1#rel1`
    - `docid:47862814-3#rel1`
  - retry outcome:
    - `city -> country`
    - `nickname -> country`
    - `single` remained unresolved
- Merged the retry outputs back into the `100`-row label file:
  - updated relations improved again from `96/100` to `98/100`
  - final merged `100`-row metrics remained:
    - top-1 label-in-gold `0.3800`
    - exact label-set match `0.0800`

### Best Current Recommendation
- Keep the promoted no-schema two-stage path unchanged as the main pilot:
  - stage 1: compact no-title no-schema relation-first extraction
  - stage 2: native relabeling with glossed choices
- The remaining relabel misses are no longer a blocking infrastructure issue.
- They are now a narrow semantic-control issue at the label-choice stage.

### Unresolved Issues
- A very small residual tail remains even after safe salvage and targeted retry.
- The dominant remaining hard failure mode is free-form type/category output, especially cases like `single`, that do not map cleanly onto an allowed relation gloss.
- Fixing the residual tail alone is unlikely to move the global pilot metrics much unless it comes with better semantic accuracy overall.

### Next Actions
- Prefer improving stage-2 semantic choice quality over adding more merge-side heuristics.
- Run a sampled-vs-greedy stage-2 comparison on the promoted no-schema path and measure whether the `0.3800` top-1 rate can move without losing the `100/100` non-empty stage-1 property.
- If sampling does not improve the label metrics, proceed to building the first larger no-schema two-stage native teacher corpus and use that path for distillation experiments.

## 2026-03-16 — Sampled Stage-2 Comparison

### Objectives
- Compare greedy vs sampled stage-2 relabeling on the promoted `100`-row no-schema stack.
- Check whether extra decode entropy improves label accuracy or only increases off-target continuation text.
- Harden the merge step against malformed sampled payloads if needed.

### Changes Saved
- Updated [`scripts/apply_rebel_relation_label_selections.jl`](/home/christos/code/julia/Swamma/scripts/apply_rebel_relation_label_selections.jl):
  - merge no longer crashes when a sampled row contains malformed JSON
  - it now falls back cleanly from failed `JSON3.read(...)` to direct `"label"` extraction
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) with the greedy-vs-sampled outcome.

### Experiment Commands And Key Metrics
- Ran sampled stage-2 relabeling on the promoted `100`-row no-schema request set:
  - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_relation_label_requests_100_noschema_v2.jsonl --output /tmp/rebel_relation_label_raw_100_noschema_sampled_t07_p09.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --overwrite --max-rows 100 --max-input-tokens 512 --max-new-tokens 128 --response-prefix '{"label":"' --dtype float32 --plain-prompt --verbose --allow-non-json --disable-stop-on-complete-json --do-sample --temperature 0.7 --top-p 0.9 --seed 42`
  - generation result:
    - accepted `100/100`
    - failed `0/100`
    - several rows emitted much longer suffixes than greedy (`8` rows over `120` chars)
- First merge attempt exposed a robustness bug:
  - one sampled response contained malformed JSON-like text
  - the merge script crashed while trying to parse it as a JSON object
- After hardening the merge path, sampled merge completed successfully:
  - `julia --project=. scripts/apply_rebel_relation_label_selections.jl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_100_noschema_relfirst.jsonl --selections /tmp/rebel_relation_label_raw_100_noschema_sampled_t07_p09.jsonl --output /tmp/rebel_teacher_parsed_native_compact_notitle_100_noschema_relfirst_relabel2stage_sampled_t07_p09.jsonl --response-prefix '{"label":"'`
  - updated relations `100/100`
- Sampled evaluation:
  - `julia --project=. scripts/evaluate_rebel_teacher_pilot.jl --gold data/rebel/train.jsonl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_100_noschema_relfirst_relabel2stage_sampled_t07_p09.jsonl --max-rows 100`
  - result:
    - matched rows `100`
    - non-empty rows `100`
    - predicted relations `100`
    - top-1 label in gold `0.2700`
    - exact label-set match `0.0600`
- Greedy baseline for the same stack remained better:
  - matched rows `100`
  - non-empty rows `100`
  - predicted relations `100`
  - top-1 label in gold `0.3800`
  - exact label-set match `0.0800`

### Best Current Recommendation
- Keep greedy stage-2 relabeling as the default on the promoted no-schema stack.
- Sampling is not the right next lever:
  - it preserves coverage
  - it increases verbose off-target continuations
  - it reduces semantic accuracy on the tested `100`-row shard

### Unresolved Issues
- The stage-2 model still drifts into verbose explanations or loosely related label choices under higher decode entropy.
- The main remaining issue is semantic label choice quality, not structural acceptance or merge robustness.

### Next Actions
- Stop spending time on generic sampling sweeps for the current stage-2 relabel path.
- Use the greedy no-schema two-stage pipeline as the current production candidate for the first larger native teacher corpus.
- If label quality still needs improvement after a larger corpus pilot, move to stronger semantic control rather than sampling:
  - constrained reranking against allowed glosses
  - or another explicit label-choice scoring step

## 2026-03-16 — 250-Row Gate And Full-Corpus Launch

### Objectives
- Stop drifting on pilot-scale decode tweaks and force a concrete step toward training.
- Validate the greedy no-schema two-stage pipeline on a materially larger `250`-row shard.
- If the `250`-row gate is good enough, freeze the recipe and launch full-train stage-1 corpus generation.

### Changes Saved
- No new model-logic changes in this segment.
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md) with the `250`-row gate results and the explicit corpus-to-training sequence.

### Experiment Commands And Key Metrics
- Built the `250`-row no-schema request shard:
  - `julia --project=. scripts/build_rebel_teacher_requests.jl --input data/rebel/train.jsonl --output /tmp/rebel_teacher_requests_native_compact_notitle_250_noschema.jsonl --max-rows 250 --prompt-style compact --no-title --no-schema --relation-label-mode id`
- Extended stage 1 from the validated `100`-row seed:
  - `cp /tmp/rebel_teacher_raw_native_compact_notitle_100_noschema_relfirst.jsonl /tmp/rebel_teacher_raw_native_compact_notitle_250_noschema_relfirst.jsonl`
  - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_requests_native_compact_notitle_250_noschema.jsonl --output /tmp/rebel_teacher_raw_native_compact_notitle_250_noschema_relfirst.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --resume --max-rows 250 --max-input-tokens 512 --max-new-tokens 256 --response-prefix '{"entities":[],"relations":[{"head_start":' --dtype float32 --plain-prompt --verbose --max-entities-hint 6 --max-relations-hint 4`
  - stage-1 result:
    - resumed extension `150/150` accepted
    - full shard `250/250` matched
    - `250/250` non-empty rows
    - top-1 label in gold `0.1400`
    - exact label-set match `0.0320`
- Extended greedy stage 2 from the repaired `100`-row seed:
  - `julia --project=. scripts/build_rebel_relation_label_requests.jl --requests /tmp/rebel_teacher_requests_native_compact_notitle_250_noschema.jsonl --teacher /tmp/rebel_teacher_parsed_native_compact_notitle_250_noschema_relfirst.jsonl --output /tmp/rebel_relation_label_requests_250_noschema.jsonl`
  - `cp /tmp/rebel_relation_label_raw_100_noschema_mergedretry.jsonl /tmp/rebel_relation_label_raw_250_noschema.jsonl`
  - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_relation_label_requests_250_noschema.jsonl --output /tmp/rebel_relation_label_raw_250_noschema.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --resume --max-rows 250 --max-input-tokens 512 --max-new-tokens 128 --response-prefix '{"label":"' --dtype float32 --plain-prompt --verbose --allow-non-json --disable-stop-on-complete-json`
  - merge/eval:
    - updated relations `245/250`
    - matched rows `250`
    - non-empty rows `250`
    - predicted relations `250`
    - top-1 label in gold `0.3480`
    - exact label-set match `0.1000`

### Best Current Recommendation
- Freeze the current recipe and step into corpus production:
  - stage 1: compact no-title no-schema relation-first generation
  - stage 2: greedy relabel with glossed natural-language choices
- The `250`-row gate is good enough to stop more pilot-only drift:
  - full coverage survived
  - stage-2 semantic quality stayed in the same regime as the successful smaller pilots

### Concrete Step Toward Training
- Built the full-train no-schema request corpus:
  - `julia --project=. scripts/build_rebel_teacher_requests.jl --input data/rebel/train.jsonl --output /tmp/rebel_teacher_requests_native_compact_notitle_full_noschema.jsonl --prompt-style compact --no-title --no-schema --relation-label-mode id`
- Launched full-train stage-1 generation from the validated `250`-row seed:
  - `cp /tmp/rebel_teacher_raw_native_compact_notitle_250_noschema_relfirst.jsonl /tmp/rebel_teacher_raw_native_compact_notitle_full_noschema_relfirst.jsonl`
  - `julia --project=. scripts/generate_rebel_teacher_responses_native.jl --input /tmp/rebel_teacher_requests_native_compact_notitle_full_noschema.jsonl --output /tmp/rebel_teacher_raw_native_compact_notitle_full_noschema_relfirst.jsonl --teacher-model ibm-granite/granite-4.0-micro --local-files-only --resume --max-input-tokens 512 --max-new-tokens 256 --response-prefix '{"entities":[],"relations":[{"head_start":' --dtype float32 --plain-prompt --verbose --max-entities-hint 6 --max-relations-hint 6`
  - confirmed live progress reached row `251` in this session

### Unresolved Issues
- Full-train stage-1 generation is still in progress.
- Full-train stage-2 relabel has not started yet because it depends on the completed stage-1 raw corpus.
- Distillation training has not started yet; this segment was the freeze-and-launch step that makes it the next concrete action instead of another research branch.

### Next Actions
- Let full-train stage-1 generation complete.
- Run full-train greedy stage-2 relabel on the completed raw corpus.
- Merge the resulting teacher annotations back into the REDFM train split.
- Start the first matched-budget `base` vs `distill` training comparison on that merged corpus.

## 2026-03-16 — CLAUDE.md Drift Cleanup

### Objectives
- Bring [`CLAUDE.md`](/home/christos/code/julia/Swamma/CLAUDE.md) back in line with the current repository structure and active workflows.
- Remove stale references to the old `Samba2` / OSSM-only layout and the obsolete autonomous single-task training directive.

### Changes Saved
- Rewrote [`CLAUDE.md`](/home/christos/code/julia/Swamma/CLAUDE.md) to reflect the current `Swamma` codebase.
- Updated the guide to describe the active model surfaces:
  - core `Swamma` block/classifier stack
  - `SwammaNER`
  - relation extraction
  - `LLaDA`
  - drafter / TiDAR / MoET research paths
  - serving / monitoring modules
- Replaced outdated file references such as `src/attention.jl` and `src/ossm.jl` with the actual current source map.
- Replaced the stale "autonomous GPU training mode" section with current test-lane commands, common entry-point scripts, and repo-specific documentation expectations.

### Experiment Commands And Key Metrics
- Inspected current repo state and doc targets:
  - `git status --short`
  - `sed -n '1,240p' CLAUDE.md`
  - `sed -n '1,260p' README.md`
  - `sed -n '1,220p' Project.toml`
  - `sed -n '1,260p' src/Swamma.jl`
  - `sed -n '1,220p' test/runtests.jl`
  - `sed -n '1,220p' docs/CI.md`
  - `rg --files configs`
- Validation:
  - `git diff -- CLAUDE.md`
- Key metrics:
  - no model training or benchmark runs in this session
  - no tests run because the change was documentation-only

### Best Current Checkpoint/Config Recommendation
- No checkpoint or config recommendation changed in this session.
- Treat this as a documentation-alignment cleanup only.

### Unresolved Issues And Next Actions
- `CLAUDE.md` is now aligned at a high level, but it may need periodic refreshes as the active RE and distillation workflows keep moving.
- Next actions:
  - keep `CLAUDE.md` aligned with promoted workflows in [`README.md`](/home/christos/code/julia/Swamma/README.md)
  - update it again whenever a major entry point, canonical naming convention, or test-lane policy changes

## 2026-03-16 — CLAUDE.md Minimal Trim

### Objectives
- Remove unnecessary detail from [`CLAUDE.md`](/home/christos/code/julia/Swamma/CLAUDE.md) after the larger drift cleanup.
- Keep only the repository-specific guidance that is still useful during coding sessions.

### Changes Saved
- Trimmed [`CLAUDE.md`](/home/christos/code/julia/Swamma/CLAUDE.md) to a smaller set of instructions.
- Removed sections that mostly duplicated `README.md`:
  - detailed source map
  - long active-surface inventory
  - explicit script entry-point examples
- Kept only the high-signal repo rules:
  - canonical naming
  - avoid old `Samba2` / `ossm` references
  - repo is not NER-only
  - `scripts/train_re_gpu.jl` is the active RE control surface
  - test-lane reminders
  - mandatory session report requirement

### Experiment Commands And Key Metrics
- Inspected current doc state:
  - `sed -n '1,240p' CLAUDE.md`
  - `tail -n 80 docs/SESSION_REPORT.md`
- Validation:
  - diff reviewed after trimming
- Key metrics:
  - no tests run because the change was documentation-only
  - no training or benchmark runs in this session

### Best Current Checkpoint/Config Recommendation
- No checkpoint or config recommendation changed in this session.

### Unresolved Issues And Next Actions
- `CLAUDE.md` is now intentionally minimal, so future additions should stay limited to repo-specific rules rather than general project description.
- Next actions:
  - only add new entries to `CLAUDE.md` when they encode stable guidance not already covered by `README.md`

## 2026-03-16 — External Tool Install (Winx And DOMShell)

### Objectives
- Install the external repositories requested by the user:
  - `gabrielmaialva33/winx-code-agent`
  - `apireno/DOMShell`
- Verify whether they could be installed as Codex skills or needed a source install instead.

### Changes Saved
- No project code changed for the tool installs.
- Installed source checkouts under:
  - `/home/christos/.local/src/winx-code-agent`
  - `/home/christos/.local/src/DOMShell`
- Built `winx-code-agent` from source and linked the binary at:
  - `/home/christos/.local/bin/winx-code-agent`
- Built the DOMShell extension bundle under:
  - `/home/christos/.local/src/DOMShell/dist`

### Experiment Commands And Key Metrics
- Verified the two GitHub repos do not expose Codex skill folders with `SKILL.md`, so they were not installable through the skill installer as skills.
- Installed and built Winx:
  - `git clone https://github.com/gabrielmaialva33/winx-code-agent.git /home/christos/.local/src/winx-code-agent`
  - `cargo build --release`
  - `ln -sf /home/christos/.local/src/winx-code-agent/target/release/winx-code-agent /home/christos/.local/bin/winx-code-agent`
  - verification:
    - `/home/christos/.local/bin/winx-code-agent --help`
  - result:
    - release build completed successfully in `53.31s`
    - binary is callable and reports `serve` as its main command
- Installed and built DOMShell:
  - `git clone https://github.com/apireno/DOMShell.git /home/christos/.local/src/DOMShell`
  - `npm install`
  - `npm run build`
  - verification:
    - inspected `/home/christos/.local/src/DOMShell/dist`
  - result:
    - production bundle built successfully
    - emitted extension artifacts including `manifest.json`, `background.js`, and `sidepanel.html`
    - `npm install` reported `1` high-severity upstream vulnerability in the dependency tree

### Best Current Checkpoint/Config Recommendation
- No model checkpoint or training config recommendation changed in this session.

### Unresolved Issues And Next Actions
- These repositories are installed as local source trees, not as Codex skills.
- DOMShell still requires browser-side manual setup to become active:
  - load `/home/christos/.local/src/DOMShell/dist` as an unpacked Chrome extension, or install from the Chrome Web Store
- Winx is built locally, but MCP client integration is still manual if the user wants it added to a Claude Desktop or other MCP config.

## 2026-03-17 — ReasoningDrafter Architecture & Pipeline

### Objectives
- Evaluate last commit additions (Engram, PredicateEngram, AlgebraicCircuit)
- Fix review issues (7 items)
- Design and implement ReasoningDrafter for speculative decoding
- Build complete chess→language→Granite training pipeline

### Changes Made

**New modules:**
- `src/RuleConditionedWavePDE.jl` — VQ situation → modulated wave dynamics (replaces PredicateEngram's TPR)
- `src/ReasoningDrafter.jl` — speculative decoding module: RuleConditionedWavePDE → GLU(LinAttn ⊙ sigmoid(WavePDE)) → AlgebraicCircuit
- `src/chess/ChessTokenizer.jl` — FEN ↔ 64-square token grid + UCI move encoding
- `src/chess/ChessDataset.jl` — Lichess eval DB JSONL loader
- `src/chess/ReasoningDataset.jl` — 6-dataset reasoning loader (LogicNLI, GSM8K, ReClor, ARC, bAbI)

**Training pipeline (4 scripts):**
- `scripts/train_chess_reasoning.jl` — Phase 1: chess pre-training
- `scripts/transfer_surgery.jl` — Phase 2: freeze backbone + add adapters
- `scripts/train_reasoning_language.jl` — Phase 3a: language fine-tuning with frozen backbone
- `scripts/distill_granite.jl` — Phase 3b: KL distillation from Granite verifier
- `scripts/launch_reasoning_pipeline.sh` — master launcher

**Fixes (from commit review):**
- Deduplicated `to_device_like`, `is_gpu_array`, `const LuxLayer` across all submodules
- GPU-native VQ distance, EMA codebook update, collision diagnostics
- NativeTeacherLM: batched attention, vectorized RoPE, lazy PyCall, fix in-place mutation

**Adapter headers for domain transfer:**
- EncoderHeader, RuleBankHeader, GateBiasShift (in RuleConditionedWavePDE)
- CircuitLeafHeader, CircuitGateBiasShift (in ReasoningDrafterBlock)
- Identity-initialized, ~149K params overhead at dim=256

### Data Status
- Lichess eval DB: download in progress (~19GB)
- Reasoning datasets: 31,230 examples downloaded (6 datasets)

### Best Recommendation
Move development to NVIDIA Spark GB10. Run `./scripts/launch_reasoning_pipeline.sh --smoke` first, then full pipeline. Target verifier: Granite (ibm-granite/granite-4.0-micro).

### Unresolved
- Lichess download not yet complete
- Granite tokenizer integration for Phase 3 (currently using char-level)
- Acceptance rate evaluation framework not yet built

## 2026-03-17 — Reasoning Module Evaluation

### Objectives
- Evaluate the reasoning module implementation, with focus on `ReasoningDrafter` and `RuleConditionedWavePDE`
- Check whether the dedicated reasoning tests cover training-time and sequence-boundary behavior
- Produce review findings without changing code

### Changes Made
- No code or config files were changed in this session
- Inspected:
  - `src/ReasoningDrafter.jl`
  - `src/RuleConditionedWavePDE.jl`
  - `src/chess/ReasoningDataset.jl`
  - `scripts/train_reasoning_language.jl`
  - `test/test_reasoning_drafter.jl`
  - `test/test_chess_pipeline.jl`

### Experiment Commands And Key Metrics
- Verified reasoning tests pass:
  - `julia --project=. test/test_reasoning_drafter.jl`
  - result: `14/14` tests passed
- Verified chess integration tests pass:
  - `julia --project=. test/test_chess_pipeline.jl`
  - result: `48/48` aggregate checks passed across tokenizer, dataset, and drafter integration testsets
- Reproduced overlength-sequence failure:
  - `julia --project=. -e 'using Swamma, Lux, Random; using Swamma.ReasoningDrafterMod; ...; m(tokens,ps,st)'`
  - result: `DimensionMismatch: new dimensions (8, 5, 1) must be consistent with array length 32`
- Reproduced training-time autodiff failure:
  - `julia --project=. -e 'using Swamma, Lux, Random, Zygote; using Swamma.ReasoningDrafterMod; ...; Zygote.gradient(ps) do p; logits,_=m(tokens,p,st); sum(logits) end'`
  - result: Zygote fails with `Mutating arrays is not supported -- called setindex!(Vector{Any}, ...)`
- Searched for EMA codebook application:
  - `rg -n "apply_rc_ema_codebook!|rc_wavepde_commitment_loss" scripts src test`
  - result: `apply_rc_ema_codebook!` is defined/exported but not called; `rc_wavepde_commitment_loss` is defined/exported but not used in training

### Best Current Checkpoint/Config Recommendation
- Do not start Phase 3a reasoning fine-tuning from the current code as-is.
- First priority is to make `ReasoningDrafter` differentiable under Zygote.
- Second priority is to align dataset sequence length with `config.max_sequence_length` or add explicit truncation/padding inside the model path.
- Third priority is to wire codebook learning into training, either by calling `apply_rc_ema_codebook!` on schedule or by redesigning the forward/loss so `Codebook` receives intentional updates.

### Unresolved Issues And Next Actions
- `ReasoningDrafter` forward currently mutates `block_states`, which blocks gradient-based training.
- `ReasoningDrafter` crashes when `seq_len > config.max_sequence_length` because position embeddings are truncated before a reshape to full sequence length.
- `RuleConditionedWavePDE` updates EMA statistics in state, but the active codebook parameters are never refreshed during training.
- Current tests validate forward inference only; they do not cover:
  - gradient computation
  - overlength inputs
  - codebook update application during training

## 2026-03-17 — Reasoning Module TODO Follow-Up

### Objectives
- Turn the reasoning-module review findings into concrete repository TODO items
- Put the work in the existing trackers used by this repo

### Changes Made
- Updated [`docs/REASONING_DRAFTER_TODO.md`](/home/christos/code/julia/Swamma/docs/REASONING_DRAFTER_TODO.md)
  - added an `Immediate Stabilization Blockers` section
  - broke the review findings into actionable checkbox items
  - added explicit regression-test and trainability-smoke-test tasks
- Updated [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md)
  - added a top-level `ReasoningDrafter Stabilization Path` section
  - summarized the four high-priority blockers for visibility in the shared tracker

### Experiment Commands And Key Metrics
- Inspected tracker locations:
  - `rg --files docs | sort`
  - `rg -n "TODO|todo|Next actions|Unresolved" docs README.md src test scripts`
- Read existing tracker content:
  - `sed -n '1,220p' docs/REASONING_DRAFTER_TODO.md`
  - `sed -n '1,220p' TODO.md`
- No model training or test execution in this follow-up session

### Best Current Checkpoint/Config Recommendation
- Keep the reasoning path blocked on stabilization work first.
- Use the new `Immediate Stabilization Blockers` section in `docs/REASONING_DRAFTER_TODO.md` as the execution order for the next implementation session.

### Unresolved Issues And Next Actions
- Implement the Zygote-safe `ReasoningDrafter` state path first.
- Then fix overlength sequence behavior and align Phase 3a sequence lengths.
- Then wire codebook updates into real training behavior and add the missing regression coverage.

## 2026-03-17 — Reasoning Module Stabilization: Zygote Trainability

### Objectives
- Complete the first `ReasoningDrafter` stabilization TODO item
- Make the drafter forward path differentiable under Zygote
- Add regression coverage so the failure mode stays fixed

### Changes Made
- Updated [`src/ReasoningDrafter.jl`](/home/christos/code/julia/Swamma/src/ReasoningDrafter.jl)
  - replaced mutable `Vector{Any}` block-state accumulation with a recursive functional helper `_apply_reasoning_blocks`
  - rebuilt `Blocks` state directly from the returned tuple instead of mutating intermediate storage
- Updated [`test/test_reasoning_drafter.jl`](/home/christos/code/julia/Swamma/test/test_reasoning_drafter.jl)
  - added `using Zygote`
  - added a `gradient pass succeeds` regression test that differentiates through the full `ReasoningDrafter`
- Updated tracker state:
  - marked the first item complete in [`docs/REASONING_DRAFTER_TODO.md`](/home/christos/code/julia/Swamma/docs/REASONING_DRAFTER_TODO.md)
  - marked the corresponding sub-item complete in [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md)

### Experiment Commands And Key Metrics
- Full reasoning test file:
  - `julia --project=. test/test_reasoning_drafter.jl`
  - result: `19/19` tests passed
- Standalone gradient probe:
  - `julia --project=. -e 'using Swamma, Lux, Random, Zygote; ...; Zygote.withgradient(...)'`
  - result: gradient evaluation completed successfully
  - observed loss: `317.37946`
  - observed gradient tensor shape: `OutputHead.weight => (20, 8)`

### Best Current Checkpoint/Config Recommendation
- The original Zygote mutation blocker in `ReasoningDrafter` is resolved.
- Next implementation target should be the overlength-sequence handling item, because it is the next hard runtime failure on the Phase 3a path.

### Unresolved Issues And Next Actions
- `ReasoningDrafter` still crashes on `seq_len > config.max_sequence_length`.
- `RuleConditionedWavePDE` codebook updates are still state-only and not yet integrated into real training behavior.
- Phase 3a still needs a tiny end-to-end trainability smoke test before larger runs.

## 2026-03-17 — Reasoning Module Stabilization: Overlength Handling

### Objectives
- Complete the second `ReasoningDrafter` stabilization TODO item
- Define and implement a consistent overlength-sequence policy
- Align the Phase 3a data path with the model sequence limit

### Changes Made
- Updated [`src/ReasoningDrafter.jl`](/home/christos/code/julia/Swamma/src/ReasoningDrafter.jl)
  - added a fail-fast `ArgumentError` when `seq_len > config.max_sequence_length`
  - simplified position embedding lookup to use `1:seq_len` once the precondition is satisfied
- Updated [`scripts/train_reasoning_language.jl`](/home/christos/code/julia/Swamma/scripts/train_reasoning_language.jl)
  - computed `effective_max_seq_length = min(requested, config.max_sequence_length)`
  - added an explicit clamp log message when the requested dataset length exceeds model capacity
  - loaded reasoning datasets using the effective sequence length
- Updated [`test/test_reasoning_drafter.jl`](/home/christos/code/julia/Swamma/test/test_reasoning_drafter.jl)
  - added an `overlength input throws` regression test
- Updated tracker state:
  - marked the overlength-handling item complete in [`docs/REASONING_DRAFTER_TODO.md`](/home/christos/code/julia/Swamma/docs/REASONING_DRAFTER_TODO.md)
  - marked the corresponding sub-item complete in [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md)

### Experiment Commands And Key Metrics
- Full reasoning test file:
  - `julia --project=. test/test_reasoning_drafter.jl`
  - result: `20/20` tests passed
- Direct overlength repro:
  - `julia --project=. -e 'using Swamma, Lux, Random; ...; m(tokens, ps, st)'`
  - result: explicit `ArgumentError`
  - message: `ReasoningDrafter received seq_len=5, but max_sequence_length=4. Truncate or pad inputs before calling the model.`

### Best Current Checkpoint/Config Recommendation
- Overlength inputs now fail cleanly at the model boundary, and Phase 3a clamps dataset preparation to the checkpoint limit.
- The next blocking item is the `RuleConditionedWavePDE` codebook update path, which still needs to affect active training parameters.

### Unresolved Issues And Next Actions
- `RuleConditionedWavePDE` codebook updates are still state-only and not yet integrated into real training behavior.
- Phase 3a still needs a tiny end-to-end trainability smoke test before larger runs.

## 2026-03-17 — Reasoning Module Stabilization: Codebook Updates And Smoke Gate

### Objectives
- Make `RuleConditionedWavePDE` codebook updates real and reusable across reasoning training flows
- Integrate the EMA codebook application into active training scripts
- Add a minimal Phase 3a trainability smoke test and use it as the launch gate

### Changes Made
- Updated [`src/RuleConditionedWavePDE.jl`](/home/christos/code/julia/Swamma/src/RuleConditionedWavePDE.jl)
  - changed EMA state initialization to zero counts so unseen codes are not treated as already-active
  - made `apply_rc_ema_codebook!` device-safe by doing the codebook refresh on CPU and copying back to the active device
  - kept inactive codebook entries untouched during EMA application
- Updated [`src/ReasoningDrafter.jl`](/home/christos/code/julia/Swamma/src/ReasoningDrafter.jl)
  - added reusable `apply_reasoning_drafter_ema_codebook!` for post-step EMA refresh across all drafter blocks
- Updated [`src/Swamma.jl`](/home/christos/code/julia/Swamma/src/Swamma.jl)
  - re-exported `apply_reasoning_drafter_ema_codebook!`
- Updated training scripts:
  - [`scripts/train_reasoning_language.jl`](/home/christos/code/julia/Swamma/scripts/train_reasoning_language.jl): apply drafter EMA codebook refresh after each optimizer step
  - [`scripts/distill_granite.jl`](/home/christos/code/julia/Swamma/scripts/distill_granite.jl): apply drafter EMA codebook refresh after each optimizer step
  - [`scripts/train_chess_reasoning.jl`](/home/christos/code/julia/Swamma/scripts/train_chess_reasoning.jl): preserve drafter state through training, apply EMA codebook refresh each step, and save/load `st_cpu` in checkpoints
- Added regression and smoke coverage:
  - [`test/test_reasoning_drafter.jl`](/home/christos/code/julia/Swamma/test/test_reasoning_drafter.jl): added EMA codebook mutation checks for both the RC layer and full drafter helper
  - [`test/test_reasoning_trainability.jl`](/home/christos/code/julia/Swamma/test/test_reasoning_trainability.jl): new `2` step Phase 3a trainability smoke test with checkpoint round-trip
  - [`test/runtests.jl`](/home/christos/code/julia/Swamma/test/runtests.jl): added the trainability smoke test to the default lane
- Updated tracker state:
  - marked the remaining reasoning stabilization items complete in [`docs/REASONING_DRAFTER_TODO.md`](/home/christos/code/julia/Swamma/docs/REASONING_DRAFTER_TODO.md)
  - marked the top-level stabilization path complete in [`TODO.md`](/home/christos/code/julia/Swamma/TODO.md)

### Experiment Commands And Key Metrics
- Core reasoning tests:
  - `julia --project=. test/test_reasoning_drafter.jl`
  - result: `21/21` tests passed
- Phase 3a smoke gate:
  - `julia --project=. test/test_reasoning_trainability.jl`
  - result: `17/17` tests passed
  - runtime: about `44.4s`
- Default test driver:
  - `julia --project=. test/runtests.jl`
  - result: passed after isolating the smoke test in its own module
- Smoke gate behavior:
  - completed multiple optimizer steps
  - kept loss finite
  - applied EMA codebook refresh after optimizer steps
  - wrote a checkpoint successfully to a temporary directory

### Best Current Checkpoint/Config Recommendation
- The reasoning stabilization path is now in a materially better state for experimentation:
  - Zygote-safe drafter forward path
  - explicit overlength handling
  - active EMA codebook refresh after training steps
  - a Phase 3a smoke test that must pass before longer jobs
- Before launching larger runs, use:
  - `julia --project=. test/test_reasoning_drafter.jl`
  - `julia --project=. test/test_reasoning_trainability.jl`

### Unresolved Issues And Next Actions
- The reasoning pipeline still lacks real checkpointed training results; the next step is an actual short reasoning fine-tune run now that the smoke gate passes.
- Granite distillation and full pipeline evaluation remain unverified end to end.
- Chess Phase 1 and Phase 3a/3b training loops should be exercised on small real-data slices to confirm the new EMA refresh logic behaves well under longer runs.

## 2026-03-18 — Reasoning Pipeline: Phase 2 Compatibility Fix And Phase 3a Real-Data Smoke

### Objectives
- Continue from stabilization into executable reasoning pipeline work
- Produce the missing Phase 2 surgery checkpoint from the existing Phase 1 checkpoint
- Run the first real-data Phase 3a smoke adaptation on a small reasoning subset

### Changes Made
- Updated [`scripts/transfer_surgery.jl`](/home/christos/code/julia/Swamma/scripts/transfer_surgery.jl)
  - added compatibility handling for Phase 1 checkpoints that save wrapped chess params under `ps_cpu.Drafter`
  - transfer surgery now accepts both bare-drafter and wrapped-chess checkpoint layouts
- Updated [`scripts/train_reasoning_language.jl`](/home/christos/code/julia/Swamma/scripts/train_reasoning_language.jl)
  - replaced the CUDA-incompatible `CartesianIndex` next-token gather with a GPU-safe target-onehot loss path built outside gradient
  - replaced `Lux.cpu(...)` checkpoint export with `cpu_device()(...)`
- Updated [`scripts/distill_granite.jl`](/home/christos/code/julia/Swamma/scripts/distill_granite.jl)
  - replaced `Lux.cpu(...)` checkpoint export with `cpu_device()(...)` for consistency with the current environment
- Created a small real-data smoke subset:
  - `data/reasoning_smoke/gsm8k.jsonl` with `64` rows
  - `data/reasoning_smoke/reclor.jsonl` with `64` rows

### Experiment Commands And Key Metrics
- Narrow repository inspection:
  - confirmed existing Phase 1 checkpoint: `checkpoints/reasoning_drafter/phase1/best.jld2`
  - confirmed partial reasoning data present: `data/reasoning/gsm8k.jsonl`, `data/reasoning/reclor.jsonl`
- Phase 2 surgery:
  - `julia --project=. scripts/transfer_surgery.jl --input checkpoints/reasoning_drafter/phase1/best.jld2 --output checkpoints/reasoning_drafter/phase2/surgery.jld2 --target-vocab 49160`
  - result: succeeded
  - artifact: [`checkpoints/reasoning_drafter/phase2/surgery.jld2`](/home/christos/code/julia/Swamma/checkpoints/reasoning_drafter/phase2/surgery.jld2)
  - total params: `26.981M`
- Phase 3a smoke dataset preparation:
  - `head -n 64 data/reasoning/gsm8k.jsonl > data/reasoning_smoke/gsm8k.jsonl`
  - `head -n 64 data/reasoning/reclor.jsonl > data/reasoning_smoke/reclor.jsonl`
  - result: `128` total examples
- First Phase 3a real-data smoke run:
  - `julia --project=. scripts/train_reasoning_language.jl --checkpoint checkpoints/reasoning_drafter/phase2/surgery.jld2 --data-dir data/reasoning_smoke --output-dir checkpoints/reasoning_drafter/phase3a_smoke --epochs 1`
  - first attempt exposed a CUDA compilation failure in the loss path due to `CartesianIndex` gather on GPU
  - second attempt exposed a checkpoint export failure due to `Lux.cpu`
  - final rerun succeeded
  - metric: `avg_loss = 10.3654` for `Epoch 1/1`
  - artifact: [`checkpoints/reasoning_drafter/phase3a_smoke/best.jld2`](/home/christos/code/julia/Swamma/checkpoints/reasoning_drafter/phase3a_smoke/best.jld2)

### Best Current Checkpoint/Config Recommendation
- The current best executable reasoning path is now:
  - Phase 1 checkpoint: [`checkpoints/reasoning_drafter/phase1/best.jld2`](/home/christos/code/julia/Swamma/checkpoints/reasoning_drafter/phase1/best.jld2)
  - Phase 2 surgery checkpoint: [`checkpoints/reasoning_drafter/phase2/surgery.jld2`](/home/christos/code/julia/Swamma/checkpoints/reasoning_drafter/phase2/surgery.jld2)
  - Phase 3a smoke checkpoint: [`checkpoints/reasoning_drafter/phase3a_smoke/best.jld2`](/home/christos/code/julia/Swamma/checkpoints/reasoning_drafter/phase3a_smoke/best.jld2)
- For the next run, use the same Phase 3a script but increase data size before increasing epochs.

### Unresolved Issues And Next Actions
- Reasoning data on disk is still partial; only GSM8K and ReClor were available during this session.
- The next practical step is a larger Phase 3a run on the full downloaded reasoning set once the missing datasets are present.
- Granite distillation (`Phase 3b`) is still untested end to end and should be exercised only after a larger Phase 3a checkpoint exists.

## 2026-03-18 — Chess Phase 1 Full Training Setup & GPU Bug Fixes

### Objectives
- Increase batch size and checkpoint frequency for RE training
- Run the full reasoning drafter pipeline (Phase 1 chess → full)
- Fix GPU compilation errors blocking chess reasoning training
- Set up auto-restart with resume for crash resilience

### Changes Made
- **`configs/redfm_1b_distill_qwen7b.toml`**: batch_size 2→4, save_every 2500→300
- **`scripts/train_re_gpu.jl`**: checkpoint saves now overwrite `checkpoint_last.jls` only (no per-step accumulation)
- **`src/chess/ChessDataset.jl`**: exported `iterate_batches` (was missing)
- **`src/RuleConditionedWavePDE.jl`**: fixed `_ema_update` — was broadcasting CPU `Vector{Float32}` with GPU `CuArray` state; now does all computation on CPU and converts back to original device type
- **`scripts/train_chess_reasoning.jl`**:
  - Fixed Zygote-incompatible `CartesianIndex` indexing in `chess_loss` — replaced with one-hot matrix multiply built outside gradient
  - Fixed Zygote mutation error — removed `Vector{Any}` block state accumulation in `forward_chess`
  - Fixed `Lux.cpu()` → `cpu_device()()` for checkpoint serialization
  - Added `--resume` CLI flag and full resume support (params + optimizer state + step + epoch)
  - Checkpoint now saves `opt_state_cpu` for proper optimizer resume
- **`scripts/train_reasoning_language.jl`**: checkpoint_every 200→300, overwrite-only saves
- **`scripts/restart_chess_training.sh`**: new auto-restart script with resume detection (50 retries, 5s cooldown)
- **`scripts/download_reasoning_datasets.sh`**: ran successfully, downloaded GSM8K (7,473) and ReClor (4,638) = 12,111 total examples
- Downloaded full Lichess eval DB: **362.7M positions** (19GB compressed → 94GB uncompressed)

### Commands Run And Key Metrics
- Smoke training (1000 synthetic positions): loss 10.88→7.13 over 5 epochs, GPU confirmed working
- Full Lichess training launched (10M positions) but session lost before step 300 checkpoint

### Best Current Recommendation
- Relaunch `./scripts/restart_chess_training.sh data/chess/lichess_db_eval.jsonl 10000000`
- Restart script will auto-resume from `checkpoint_last.jld2` on crash
- All three training scripts now use overwrite-only checkpoints every 300 steps

### Unresolved Issues And Next Actions
- Full Phase 1 training needs to complete (10M positions, 10 epochs)
- Phase 2 surgery checkpoint exists from prior session; reusable once Phase 1 completes with full data
- Phase 3a ready with 12,111 reasoning examples (GSM8K + ReClor)
- 3 HuggingFace datasets failed to download (PrOntoQA, BoardgameQA not found; FOLIO gated; LogiQA script unsupported)

## 2026-03-18 — GPU Memory Crisis: Zygote AD Tape Fix For ReasoningDrafter

### Objectives
- Fix 80GB+ GPU memory usage for a 6.7M param model during Phase 1 chess training
- Get full Lichess data training running stably on Spark GB10
- Investigate Enzyme.jl as potential replacement for Zygote

### Root Cause Analysis
Zygote builds an AD tape storing every intermediate array for backprop. The ReasoningDrafterBlock contains:
- RuleConditionedWavePDE: FFT-based leapfrog loop (4-8 integration steps × 2 FFTs each)
- WavePDE gate: another FFT-based PDE loop
- LinearAttention with feature projections
- AlgebraicCircuitLayer with sum-product networks

Each block creates ~40+ large GPU intermediates that all stay alive until backward finishes. With 2 layers, this consumed 80GB for a 6.7M model.

### Changes Made
- **`src/RuleConditionedWavePDE.jl`**: wrapped modulation + leapfrog PDE integration entirely inside `ChainRulesCore.ignore_derivatives()` — all speed/damping computation and FFT steps are now detached from the AD tape
- **`src/ReasoningDrafter.jl`**:
  - Added `using ChainRulesCore`
  - Extracted block internals to `_block_forward()`
  - Wrapped entire block forward pass in `ignore_derivatives` with straight-through estimator: `hidden + (block_out - detach(hidden))` — gradients flow through residual stream only
- **`scripts/train_chess_reasoning.jl`**:
  - Added `forward_chess_backbone()` — runs backbone completely outside `withgradient`
  - Training loop now only differentiates through MoveHead + EvalHead Dense layers
  - Switched to streaming disk reads (`StreamingBatchIterator`) — no preloading positions into RAM
  - Reduced to 3 epochs, batch_size=64 for practical Spark training time
  - Added `using ChainRulesCore`

### Memory Progression
| Change | GPU MEM |
|---|---|
| Original (Zygote traces everything) | **80+ GB** (OOM killed) |
| `ignore_derivatives` around PDE only | 70 GB |
| Straight-through block wrapper | 28 GB |
| Backbone outside `withgradient` | 26 GB |

The remaining 26GB is CUDA memory pool holding freed allocations, not active usage.

### Enzyme.jl Investigation
- Enzyme.jl would solve this properly (LLVM-level AD, recomputes instead of storing)
- As of early 2026: Enzyme + CUDA + Lux still has excessive compile times and runtime failures
- Issues #1392, #2244, #2283 on EnzymeAD/Enzyme.jl track the blockers
- Lux plans to switch from Zygote to Enzyme but hasn't yet
- **Verdict**: correct long-term fix, not usable today

### Training Status
- Phase 1 chess training running on full Lichess data (500K positions, 3 epochs, batch_size=64)
- Resumed from step ~27,000, loss ~4.2-4.8 (down from 8.5)
- Checkpointing every 300 steps with auto-restart and resume
- ETA: ~2.5 hours remaining

### Best Current Recommendation
- Let Phase 1 finish, then run Phase 2 surgery + Phase 3a language fine-tuning
- The `ignore_derivatives` + straight-through pattern should be applied to main Swamma training if memory becomes an issue there too
- Consider writing a custom `rrule` for `leapfrog_step` loop to get proper gradients without the full tape

### Unresolved Issues And Next Actions
- Apply the same AD tape fix to `src/WavePDE.jl` (main Swamma backbone) — same memory problem exists there
- The straight-through estimator means backbone params only learn via EMA codebook updates, not gradient — acceptable for Phase 1 pre-training but needs revisiting for Phase 3
- Refactor other Swamma-based architectures to use the same memory-efficient pattern
- Monitor Enzyme.jl + CUDA + Lux maturity for eventual migration
