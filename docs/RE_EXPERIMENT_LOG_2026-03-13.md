# RE Experiment Log — 2026-03-13

## Scope

Focus: relation-side architecture and training objective above stable `SwammaBlock` encoder.

Primary target: improve `pred spans + pred pairs` relation quality while keeping pair proposal subquadratic.

## Changes Implemented

1. `scripts/train_re_gpu.jl`
- Added checkpoint-safe `null_relation_weight` loading from TOML (without changing serialized run-config layout).
- Added pair-sweep tooling and fixed CLI parser support for:
  - `--pair-sweep-checkpoint`
  - `--pair-sweep-budgets`
  - `--pair-sweep-overgenerate`
- Added partial warm-start resume for architecture changes:
  - merge only shape-compatible params/state from checkpoint
  - reset optimizer state on partial match
- Added decoding-margin controls and sweeps:
  - `--margin-sweep-checkpoint`
  - `--margin-sweep-values`
  - `--threshold-sweep-margin`

2. `src/RelationExtraction.jl`
- Added anchor-expanded pair proposal path with subquadratic complexity:
  - Top anchor spans (derived from proposer settings) connect to semantic/top-score fanout.
  - Candidate generation remains sparse (`O(N·k)` family), not exhaustive `O(N²)`.
- Kept old checkpoint compatibility by avoiding serialized `RelationExtractionConfig` field-layout changes.

3. New configs
- `configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4.toml`
- `configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_curric50.toml`
- `configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_curric10.toml`
- `configs/redfm_base_safe_pair_sparse_learned128_nullw05_overgen4.toml`
- `configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence.toml`

## Core Results

### A) Resume sanity + null weight

`config: redfm_base_safe_pair_sparse_learned128_nullw025.toml`  
`resume: checkpoints/redfm_base_safe_pair_sparse_learned128/checkpoint_best.jls`  
`step: 1000`

- `oracle_rel = 0.4878`
- `pair_recall = 0.0732`
- `pair_t16 = 0.0366`
- `rel_f1 = 0.0000`

### B) Pair sweep after anchor-expanded proposer (same checkpoint family)

`config: redfm_base_safe_pair_sparse_learned128_nullw025.toml`  
`checkpoint: .../redfm_base_safe_pair_sparse_learned128_nullw025/checkpoint_last.jls`

- Best row: `pairs=192, overgen=4`
  - `pair_r = 0.2195`
  - `pair_t16 = 0.0366`
  - `r/exh = 0.4500`

This is a clear pair-recall lift versus earlier sweeps on the same checkpoint family.

### C) Overgen-4 training extension (best run of the day)

`config: redfm_base_safe_pair_sparse_learned128_nullw025_overgen4.toml`  
`resume from step 1000 -> step 1250`

Eval at step 1250:
- `val_loss = 14.2400`
- `ment_p/r/f1 = 0.0564 / 0.6619 / 0.1039`
- `oracle_rel = 0.6707`
- `pair_recall = 0.2317`
- `pair_t16 = 0.1341`
- `rel_p = 0.0006`
- `rel_r = 0.0122`
- `rel_f1 = 0.0012` (first non-zero in this run family)

### D) Overgen-4 continuation to step 1500

Same config/run, eval at step 1500:
- `oracle_rel = 0.6707` (flat)
- `pair_recall = 0.2073` (down)
- `pair_t16 = 0.0854` (down)
- `rel_f1 = 0.0010` (down)

Step 1250 is better than step 1500 for this run.

### E) Curriculum test (rejected)

`config: redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_curric50.toml`  
`proposal_train_probability=0.5`, `proposal_loss_weight=0.5`  
`resume from step 1000 -> step 1250`

Eval at step 1250:
- `ment_r = 0.3209`
- `oracle_rel = 0.1220`
- `pair_recall = 0.0488`
- `rel_f1 = 0.0000`

This setting collapsed retrieval/coverage and is not viable.

### F) Mild curriculum test (also rejected)

`config: redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_curric10.toml`  
`proposal_train_probability=0.1`, `proposal_loss_weight=0.25`  
`resume from step 1000 -> step 1250`

Eval at step 1250:
- `ment_r = 0.5043`
- `oracle_rel = 0.3415`
- `pair_recall = 0.1098`
- `pair_t16 = 0.0610`
- `rel_f1 = 0.0000`

This is less catastrophic than `curric50`, but still materially worse than non-curriculum `overgen4`.

### G) Null-weight rebalancing test (rejected)

`config: redfm_base_safe_pair_sparse_learned128_nullw05_overgen4.toml`  
`null_relation_weight=0.5`  
`resume from step 1000 -> step 1250`

Eval at step 1250:
- `val_loss = 13.5643`
- `oracle_rel = 0.3293`
- `pair_recall = 0.0366`
- `pair_t16 = 0.0122`
- `rel_f1 = 0.0000`

Raising null weight improved loss terms but collapsed candidate coverage and zeroed end-task F1.

### H) Fused evidence decoder resume attempt

`config: redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence.toml`  
`resume from biaffine checkpoint at step 1000`

Observed failure:
- checkpoint params for `RelationHead` did not match fused-evidence parameter tree
- immediate training error on backward pass

### I) Resume compatibility patch after H

Trainer now supports partial warm-start for architecture changes:
- merges only matching parameter/state subtrees
- keeps newly introduced params from fresh init
- resets optimizer state when partial match occurs

Verified that same-architecture resume still works after this patch.

### J) Refreshed overgen-4 rerun (new best training checkpoint in this session)

`config: redfm_base_safe_pair_sparse_learned128_nullw025_overgen4.toml`  
`resume from step 1000 -> step 1250`

Eval at step 1250:
- `val_loss = 13.8310`
- `ment_p/r/f1 = 0.0679 / 0.7966 / 0.1251`
- `oracle_rel = 0.8293`
- `pair_recall = 0.2073`
- `pair_t16 = 0.0976`
- `rel_p = 0.0014`
- `rel_r = 0.0366`
- `rel_f1 = 0.0027`

This checkpoint improves relation F1 over earlier `0.0012`/`0.0010` runs despite slightly lower pair recall than the earlier step-1250 snapshot.

### K) Decoding calibration around NO_RELATION margin (new best decoded F1)

Using checkpoint from section J:
- Base decoded point (`threshold=0.50`, `margin=0.00`): `rel_f1 = 0.0027`
- Best found point (`threshold=0.70`, `margin=0.30`): `rel_f1 = 0.0050`

So decoding-time calibration alone produced ~1.85x F1 gain on `pred spans + pred pairs`.

### L) Non-null probability gate calibration (new decode primitive, limited gain)

Code change:
- Added decode-time gate on softmax non-null probability:
  - keep prediction only if `1 - p(NO_RELATION) >= nonnull_threshold`
- Added CLI diagnostics:
  - `--nonnull-sweep-checkpoint`
  - `--nonnull-sweep-values`
  - `--nonnull-sweep-confidence`
  - `--nonnull-sweep-margin`
  - `--threshold-sweep-nonnull`

Results on checkpoint from section J:

1. At current best calibrated point (`confidence=0.70`, `margin=0.30`):
- `nonnull = 0.80 / 0.90 / 0.93` all kept `rel_f1 = 0.0050` (no gain)
- `nonnull = 0.95` regressed to `rel_f1 = 0.0019`

2. At uncalibrated point (`confidence=0.50`, `margin=0.00`):
- baseline neighborhood was `rel_f1 = 0.0027`
- best observed with non-null gate: `nonnull = 0.93` gave `rel_f1 = 0.0028` (minor gain)
- too strict (`>= 0.97`) collapsed recall/F1 to zero

Conclusion:
- Non-null gating is useful as a calibration control surface, but it did not beat the existing best decoded point (`0.0050`).
- Current best remains threshold+margin tuning (`0.70`, `0.30`) without aggressive non-null filtering.

### M) Constrained decoding via per-head/per-tail caps (implemented, currently harmful)

Code change:
- Added decode caps in relation-set construction:
  - `max_relations_per_head`
  - `max_relations_per_tail`
- Added CLI controls:
  - `--decode-head-cap`
  - `--decode-tail-cap`
- Caps are now applied consistently in oracle ladder / threshold sweep / margin sweep / non-null sweep modes.

Control and cap tests at the current best calibration point (`threshold=0.70`, `margin=0.30`, `nonnull=0.0`):
- `cap head=0, tail=0` (control): `pred spans + pred pairs rel_f1 = 0.0050` (baseline preserved)
- `cap head=1, tail=1`: `rel_f1 = 0.0000`
- `cap head=2, tail=2`: `rel_f1 = 0.0000`
- `cap head=4, tail=4`: `rel_f1 = 0.0000`

Conclusion:
- Cap-based constrained decoding is functional but currently over-prunes true positives in this run family.
- Keep caps disabled (`0, 0`) by default for now.

### N) Fused-evidence confidence head conditioning (implemented, short resume test)

Code change:
- In `relation_decoder_mode = fused_evidence`, confidence head now consumes fused inputs:
  - `pair_features`
  - `evidence_summary`
  - `retrieval_logits`
- Non-fused modes keep prior confidence path for checkpoint compatibility.

Run:
- `config: configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence.toml`
- warm-start resume from baseline checkpoint:
  - `checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4/checkpoint_last.jls`
- short continuation: `step 1250 -> 1260`

Eval at fused-evidence step `1260` (default decode):
- `oracle_rel = 0.6951`
- `pair_recall = 0.1341`
- `pair_t16 = 0.0488`
- `rel_p = 0.0012`
- `rel_r = 0.0244`
- `rel_f1 = 0.0023`

Threshold sweep (`margin=0.0`) on same checkpoint:
- `threshold=0.50`: `rel_f1 = 0.0023` (best among tested)
- `threshold=0.70`: `rel_f1 = 0.0000`
- `threshold=0.90`: `rel_f1 = 0.0000`

Conclusion:
- This fused-evidence confidence redesign is trainable/stable with warm-start, but currently underperforms the calibrated biaffine baseline (`rel_f1 = 0.0050`).
- Keep as an ablation branch, not as the promoted mainline.

### O) Per-relation confidence threshold calibration (implemented, improved best F1)

Code change:
- Added per-relation confidence threshold overrides in decode:
  - CLI: `--per-relation-thresholds`
  - format: `LABEL=VALUE` or `ID=VALUE`, comma-separated
- Applied across oracle/threshold/margin/nonnull sweep paths.

Validation results on baseline checkpoint (same decode base: `threshold=0.70`, `margin=0.30`, caps off):

1. Previous global-best decoded point:
- `rel_f1 = 0.0050`

2. Per-relation override test:
- `P127=0.95, P155=0.90` -> `rel_f1 = 0.0056`

3. Expanded override:
- `P127=0.95, P155=0.90, P571=0.85` -> `rel_f1 = 0.0057` (new best)

4. Over-regularized example (rejected):
- adding `P641=0.85` regressed to `rel_f1 = 0.0046`

Conclusion:
- Per-relation confidence calibration provides a meaningful decode-time gain over global-only tuning.
- Current best decode recipe:
  - global: `threshold=0.70`, `no_relation_margin=0.30`
  - per-relation: `P127=0.95, P155=0.90, P571=0.85`
  - constrained caps disabled (`0,0`)

### P) `v1_locked` reproducibility reruns (Stage-1 baseline freeze)

Locked eval recipe:
- `config = configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4.toml`
- `checkpoint = checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4/checkpoint_last.jls`
- decode:
  - `threshold=0.70`
  - `no_relation_margin=0.30`
  - `per_relation_thresholds=P127=0.95,P155=0.90,P571=0.85`
  - `decode_head_cap=0`, `decode_tail_cap=0`
- `max_eval_batches=8`

Repro run #1 (`pred spans + pred pairs`):
- `rel_p = 0.0031`
- `rel_r = 0.0366`
- `rel_f1 = 0.0057`
- `oracle_rel = 0.8293`
- `pair_recall = 0.2073`
- `pair_t16 = 0.0976`

Repro run #2 (`pred spans + pred pairs`):
- `rel_p = 0.0031`
- `rel_r = 0.0366`
- `rel_f1 = 0.0057`
- `oracle_rel = 0.8293`
- `pair_recall = 0.2073`
- `pair_t16 = 0.0976`

Observed variance over two runs:
- mean `rel_f1 = 0.0057`
- std `rel_f1 = 0.0000`
- range `rel_f1 = [0.0057, 0.0057]`

Frozen reference row (`v1_locked`):
- `pred spans + pred pairs`: `rel_p=0.0031`, `rel_r=0.0366`, `rel_f1=0.0057`, `oracle_rel=0.8293`, `pair_r=0.2073`, `pair_t16=0.0976`

### Q) Auto-calibration helper with global acceptance gate (implemented)

Code change:
- Added `--auto-calibrate-checkpoint` mode in `scripts/train_re_gpu.jl`.
- Added per-relation threshold proposal search from validation TP/FP confidence histograms.
- Added strict global acceptance gate:
  - raw per-relation suggestions are rejected if global `pred spans + pred pairs rel_f1` drops below baseline.

Run (`max_eval_batches=8`, base decode `threshold=0.70`, `margin=0.30`, no per-relation seed overrides):
- Raw suggested override: `P641=0.85`
- Baseline global: `rel_f1 = 0.0050`
- Raw calibrated global: `rel_f1 = 0.0039` (worse)
- Gate decision: reject raw suggestion, keep accepted overrides as `none`
- Accepted global: `rel_f1 = 0.0050` (unchanged)

Conclusion:
- Auto-calibration infrastructure is now in place and safe by construction.
- It can emit candidate per-relation thresholds, but only globally non-degrading sets are accepted.
- Current best remains manual calibrated `v1_locked` recipe (`rel_f1 = 0.0057`).

### R) Decode-time schema/type constraints (Stage 3, implemented + ablated)

Code change:
- Added decode-time type constraints in `scripts/train_re_gpu.jl`:
  - CLI:
    - `--type-constraints-mode` (`off|hard`)
    - `--type-constraints-min-count`
  - Constraints are computed from training rows as relation -> allowed `(head_type, tail_type)` pairs.
  - Decoding now optionally masks relation predictions whose inferred span-type pair is invalid for that relation.
- Applied consistently across:
  - `oracle-ladder`
  - `threshold-sweep`
  - `margin-sweep`
  - `nonnull-sweep`
  - `auto-calibration`
- Added robust relation index-offset inference for constraint construction to avoid mixed 0/1-based entity-index misalignment.

Stage-3 ablations (`max_eval_batches=8`, checkpoint/config from `v1_locked`):

1. Control (`type-constraints=off`, per-rel overrides on):
- decode: `threshold=0.70`, `margin=0.30`, `P127=0.95,P155=0.90,P571=0.85`
- `pred spans + pred pairs`: `rel_p=0.0031`, `rel_r=0.0366`, `rel_f1=0.0057`

2. Constraints + per-rel overrides (`type-constraints=hard,min_count=1`):
- same decode settings as control + hard constraints
- active rule summary: `types=13`, `relations=32`, `rules=1380`
- `pred spans + pred pairs`: `rel_p=0.0031`, `rel_r=0.0122`, `rel_f1=0.0050`

3. Constraints only (`type-constraints=hard,min_count=1`, no per-rel overrides):
- decode: `threshold=0.70`, `margin=0.30`, no per-relation overrides
- `pred spans + pred pairs`: `rel_p=0.0025`, `rel_r=0.0122`, `rel_f1=0.0042`

4. Reference without constraints and without per-rel overrides:
- decode: `threshold=0.70`, `margin=0.30`, no per-relation overrides
- `pred spans + pred pairs`: `rel_p=0.0027`, `rel_r=0.0366`, `rel_f1=0.0050`

Conclusion:
- Type constraints are working and measurable, but in current `hard` form they trade too much recall for precision in this checkpoint family.
- Best hard-type-constrained result (`0.0050`) does not beat `v1_locked` (`0.0057`), so hard type masking is not promoted.
- Keep `type-constraints-mode=off` for default decode until a softer type-constraint strategy is added.

### S) Optional inverse/symmetry relation consistency (Stage 3 follow-up, implemented + promoted)

Code change:
- Added decode-time relation consistency controls in `scripts/train_re_gpu.jl`:
  - `--relation-consistency-mode` (`off|resolve`)
  - `--relation-consistency-min-count`
- Built rules from training-data reverse-edge statistics:
  - symmetric relations (`r <-> r`)
  - mutual inverse relation pairs (`r1 <-> r2`)
- Applied as a decode-time conflict resolver on reverse-direction predicted pairs:
  - if reverse predictions violate learned symmetry/inverse rule, keep higher-confidence edge and drop lower-confidence conflict.
- Integrated across oracle/threshold/margin/nonnull/auto-calibration decode paths.

Ablation at locked decode point (`threshold=0.70`, `margin=0.30`, `P127=0.95,P155=0.90,P571=0.85`, `max_eval_batches=8`):

1. Baseline (`relation-consistency=off`):
- `pred spans + pred pairs`: `rel_p=0.0031`, `rel_r=0.0366`, `rel_f1=0.0057`

2. Consistency resolver (`relation-consistency=resolve,min_count=1`):
- inferred rule summary: `symmetric=0`, `inverse_pairs=3`
- run #1: `rel_p=0.0031`, `rel_r=0.0366`, `rel_f1=0.0058`
- run #2: `rel_p=0.0031`, `rel_r=0.0366`, `rel_f1=0.0058`
- pair metrics unchanged: `pair_r=0.2073`, `pair_t16=0.0976`

Conclusion:
- This resolver gives a small but repeatable gain over `v1_locked` without pair-recall collapse.
- Promote `relation-consistency=resolve,min_count=1` as the default Stage-3 decode addition.
- Keep hard type masking disabled for default decode.

### T) Evidence diagnostics in evaluator (Stage 4 kickoff)

Code change:
- Added optional evidence diagnostics emission from `PairEvidenceSelectorHead` and model outputs:
  - `evidence_top_token_index`
  - `evidence_attention_entropy`
  - `evidence_attention_max_weight`
- Added evaluator-side aggregation in `scripts/train_re_gpu.jl`:
  - mean evidence entropy (`ev_ent`)
  - mean max attention weight / concentration proxy (`ev_max`)
  - mean effective-token count `exp(entropy)` (`ev_eff`)
  - most frequent top evidence token index (`ev_t1`)
- Added these columns to checkpoint sweep output and eval summary formatting.

Validation run (`max_eval_batches=8`, checkpoint `checkpoint_last.jls`, config `...nullw025_overgen4.toml`):
- `ev_ent = 3.8232`
- `ev_max = 0.1172`
- `ev_eff = 55.54`
- `ev_t1 = 56`

Interpretation:
- Evidence mass is currently diffuse (high entropy/effective-token count, low max weight), which matches the current low precision regime and supports the next planned evidence-pooling ablations.

### U) Evidence pooling ablation (`token` vs `sentence` vs `hybrid`)

Code change:
- Added optional evidence pooling mode control to model/eval inputs:
  - `evidence_pooling_mode = :token | :sentence | :hybrid`
- Added sweep mode in trainer CLI:
  - `--evidence-pooling-sweep-checkpoint`
  - `--evidence-pooling-modes`

Important scope note:
- For non-fused decoder checkpoints (`relation_decoder_mode=:biaffine`), pooling mode does not affect relation logits (evidence summary is not consumed by that decoder path).
- For fused-evidence checkpoints, pooling mode is causal and changes relation outputs.

Fused-evidence sweep run:
- `config = configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence.toml`
- `checkpoint = checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence/checkpoint_last.jls`
- `max_eval_batches = 8`

Results (`pred spans + pred pairs`):
- `token`: `rel_p=0.0012`, `rel_r=0.0244`, `rel_f1=0.0023` (best)
- `sentence`: `rel_p=0.0009`, `rel_r=0.0244`, `rel_f1=0.0018`
- `hybrid`: `rel_p=0.0010`, `rel_r=0.0244`, `rel_f1=0.0019`

Conclusion:
- For the current fused-evidence checkpoint, token-level pooling remains best; sentence/hybrid reduce precision/F1.
- Keep token pooling as default before launching the short Stage-4 resume window.

### V) Stage-4 short fused-evidence continuation (`+250` steps)

Run:
- `config = configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence.toml`
- `resume = checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence/checkpoint_last.jls`
- checkpoint step before run: `1260`
- command target: `--max-steps 1510` (`+250` updates)

Training outcome:
- run completed successfully at step `1510`
- checkpoint snapshot saved: `checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence/checkpoint_step_1510.jls`
- eval snapshot at step `1500`:
  - `val_loss=15.2280`
  - `relation_loss=10.4497`
  - `oracle_rel=0.3537`
  - `pair_recall=0.1829`
  - `pair_t16=0.0610`
  - `rel_p=0.0006`, `rel_r=0.0122`, `rel_f1=0.0011`
  - evidence diagnostics: `ev_ent=2.434`, `ev_max=0.361`, `ev_eff=16.2`

Locked decode re-check at new checkpoint (`max_eval_batches=8`):
- decode: `threshold=0.70`, `margin=0.30`, `P127=0.95,P155=0.90,P571=0.85`
- `relation-consistency=off`:
  - `pred spans + pred pairs`: `rel_p=0.0028`, `rel_r=0.0122`, `rel_f1=0.0046`
  - `oracle_rel=0.3659`, `pair_r=0.2073`, `pair_t16=0.0732`
- `relation-consistency=resolve,min_count=1`:
  - identical sampled row (`rel_f1=0.0046`)

Checkpoint sweep row at step `1510` (default decode in `--eval-checkpoint`, `max_eval_batches=8`):
- `total=14.9950`, `relation_loss=10.0617`, `prop_total=8.0979`
- `pair_r=0.2073`, `pair_t16=0.0732`
- `rel_p=0.0010`, `rel_r=0.0244`, `rel_f1=0.0020`
- evidence diagnostics: `ev_ent=2.1763`, `ev_max=0.4146`, `ev_eff=12.76`, `ev_t1=2`

Conclusion:
- The short continuation materially improved fused-branch calibrated F1 versus the prior fused baseline (`0.0023 -> 0.0046`) and increased precision.
- The branch still does not beat `v1_locked` (`0.0057`), so it remains an ablation branch and is not promoted.

### W) Stage-5 objective upgrade: retrieval edge ranking + hard negatives (implemented)

Code change:
- Added retrieval ranking objective in `scripts/train_re_gpu.jl`:
  - `retrieval_hard_negative_ranking_loss(...)`
  - hinge ranking term over pair retrieval logits with top-`k` hardest negatives per batch item
  - objective: `max(0, margin - pos + neg)` averaged across positives and selected hard negatives
- Added training config knobs (loaded from TOML `[training]`):
  - `edge_ranking_loss_weight` (default `0.0`)
  - `edge_ranking_margin` (default `0.2`)
  - `edge_ranking_hard_negatives` (default `16`)
- Integrated into both:
  - teacher-forced training loss (`relation_loss`)
  - proposal-conditioned loss (`proposal_training_loss`)
- Added eval visibility:
  - `ret_rank` and `prop_rank` fields in eval summary string
- Backward-compatibility note:
  - ranking settings are loaded from TOML at runtime (not embedded in `RETrainingRunConfig`) to keep old checkpoints deserializable.

Smoke validation:
- Checkpoint eval command (rank-loss config, `max_eval_batches=1`) completed successfully:
  - `config = configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence_rankloss.toml`
  - `checkpoint = .../checkpoint_step_1510.jls`
  - row: `total=11.4361`, `rel=6.9227`, `prop_tot=8.0431`, `rel_f1=0.0086`
- One-step resume smoke (`1510 -> 1511`) with rank-loss config completed:
  - log confirms active settings: `Edge rank wt=0.2`, `margin=0.2`, `hard negs=16`
  - completed without runtime/gradient errors.

### X) Stage-5 controlled continuation with rank-loss (`1510 -> 1760`)

Run:
- `config = configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence_rankloss.toml`
- `resume = checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence/checkpoint_step_1510.jls`
- target: `--max-steps 1760` (`+250` updates)
- completed successfully at step `1760`
- checkpoint snapshot saved: `checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence_rankloss/checkpoint_step_1760.jls`

In-run eval snapshot at step `1750`:
- `val_loss=16.0479`
- `relation_loss=10.8218`
- `ret_rank=0.1548`, `prop_rank=0.0415`
- `oracle_rel=0.1951`
- `pair_recall=0.0488`
- `pair_t16=0.0122`
- `rel_p=0.0000`, `rel_r=0.0000`, `rel_f1=0.0000`
- evidence concentration increased strongly (`ev_ent=0.772`, `ev_max=0.746`, `ev_eff=2.9`)

Locked decode re-check at step `1760` (`threshold=0.70`, `margin=0.30`, `P127=0.95,P155=0.90,P571=0.85`, `max_eval_batches=8`):
- `relation-consistency=off`:
  - `pred spans + pred pairs`: `rel_p=0.0000`, `rel_r=0.0000`, `rel_f1=0.0000`
  - `oracle_rel=0.2317`, `pair_r=0.0732`, `pair_t16=0.0488`
- `relation-consistency=resolve,min_count=1`:
  - identical sampled row (`rel_f1=0.0000`)

Conclusion:
- This first rank-loss continuation is a regression versus both prior fused checkpoint (`rel_f1=0.0046`) and `v1_locked` (`0.0057`).
- Current `edge_ranking_loss_weight=0.2` setting appears too aggressive for this branch; do not promote.

### Y) Stage-5 soft-scheduled rank-loss continuation (`1510 -> 1760`)

Code/config adjustment:
- Added step schedule controls in `scripts/train_re_gpu.jl`:
  - `edge_ranking_start_step`
  - `edge_ranking_warmup_steps`
  - per-step effective weight now ramps from `0` to configured max after `start_step`.
- Added config:
  - `configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence_rankloss_soft.toml`
  - settings used: `weight=0.05`, `start_step=1650`, `warmup_steps=150`, `margin=0.2`, `hard_negatives=16`

Run:
- resumed from `.../checkpoint_step_1510.jls`
- trained to `step 1760` (`+250`)
- checkpoint snapshot saved:
  - `checkpoints/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4_fusedevidence_rankloss_soft/checkpoint_step_1760.jls`

In-run eval at step `1750`:
- `val_loss=16.0197`
- `relation_loss=11.0854`
- `ret_rank=0.0247`, `prop_rank=0.0256`
- `oracle_rel=0.5976`
- `pair_recall=0.1829`
- `pair_t16=0.0610`
- `rel_p=0.0011`, `rel_r=0.0244`, `rel_f1=0.0022`

Locked decode re-check at step `1760` (`threshold=0.70`, `margin=0.30`, `P127=0.95,P155=0.90,P571=0.85`, `max_eval_batches=8`):
- `relation-consistency=off`:
  - `pred spans + pred pairs`: `rel_p=0.0040`, `rel_r=0.0122`, `rel_f1=0.0060`
  - `oracle_rel=0.5000`, `pair_r=0.1463`, `pair_t16=0.0366`
- `relation-consistency=resolve,min_count=1`:
  - identical sampled row (`rel_f1=0.0060`)

Checkpoint sweep row (`--eval-checkpoint`, default decode, `max_eval_batches=8`):
- `total=15.3312`, `relation_loss=10.3041`, `prop_total=8.0640`
- `pair_r=0.1463`, `pair_t16=0.0366`
- `rel_f1=0.0012`
- evidence diagnostics: `ev_ent=1.0200`, `ev_max=0.6879`, `ev_eff=3.78`, `ev_t1=49`

Conclusion:
- Soft schedule fixes the catastrophic collapse seen in the aggressive rank-loss run.
- At locked decode, calibrated `rel_f1=0.0060` beats `v1_locked=0.0057`, but proposal coverage metrics remain materially below baseline (`pair_r`/`pair_t16`), so promotion gate is still not passed.

## Interpretation

1. Main bottleneck moved
- We now have significantly better relation opportunity (`oracle_rel`) and pair coverage.
- End-task F1 remains low due precision collapse in predicted-span/predicted-pair decoding.

2. Effective architecture move
- Sparse anchor-expanded proposal was the highest-impact structural change today.
- It improved coverage without reverting to exhaustive pairing.

3. Objective warning
- Aggressive proposal-conditioned curriculum at this stage destabilizes mention/pair coverage.
- Even mild curriculum settings currently underperform the non-curriculum baseline.
- Increasing null weight to `0.5` from this checkpoint family also harms retrieval coverage.

## Recommended Baseline To Continue

Use:
- `config: configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4.toml`
- best observed checkpoint region: around step `1250`
- best decoded operating point currently: `threshold=0.70`, `no_relation_margin=0.30`
- best calibrated per-relation operating point: `threshold=0.70`, `margin=0.30`, `P127=0.95,P155=0.90,P571=0.85`
- promoted consistency add-on: `relation-consistency=resolve`, `relation-consistency-min-count=1`
- `v1_locked` reproducibility status: verified on 2/2 reruns with identical sampled eval outputs
- auto-calibration helper is available; current raw suggestion (`P641=0.85`) is rejected by global F1 gate
- optional non-null filter: keep at `<= 0.93`; higher values degrade recall
- keep constrained decode caps disabled for now: `decode_head_cap=0`, `decode_tail_cap=0`
- fused-evidence confidence variant currently below baseline; do not promote without clear F1 gain

Next work should target precision/calibration under predicted candidates, not raw pair recall alone.
