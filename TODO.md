# TODO — SWAMMA Rollout + Legacy Removal

## ReasoningDrafter Stabilization Path

- [x] Unblock `ReasoningDrafter` training before any new reasoning runs:
  - [x] remove the mutable block-state accumulation that breaks Zygote gradients
  - [x] fix `seq_len > max_sequence_length` handling in the model and Phase 3a data path
  - [x] make `RuleConditionedWavePDE` codebook updates real, not state-only bookkeeping
  - [x] add a tiny Phase 3a trainability smoke test that must pass before longer jobs

## LLaDA PRIME 3B Distillation Path

- [x] Create the `~3B` PRIME student config in [`configs/llada_prime_3b.toml`](/home/christos/code/julia/Swamma/configs/llada_prime_3b.toml)
- [x] Record the `~5B` stretch config in [`configs/llada_prime_5b.toml`](/home/christos/code/julia/Swamma/configs/llada_prime_5b.toml)
- [x] Write the full rollout checklist in [`docs/LLADA_PRIME_3B_DISTILLATION_TODO.md`](/home/christos/code/julia/Swamma/docs/LLADA_PRIME_3B_DISTILLATION_TODO.md)
- [ ] Execute the checklist to completion before promoting a new PRIME baseline

## REDFM English Training Path

Goal: train the Swamma relation extractor on English REDFM first, without adding multilingual scope yet.

### 0. Dataset And Naming Lock

- [x] Standardize naming on `REDFM` rather than `REBEL` for the current supervised pipeline.
- [x] Create config files:
  - [x] `configs/redfm_smoke.toml`
  - [x] `configs/redfm_base.toml`
- [x] Add downloader/converter:
  - [x] `scripts/download_redfm.py`
- [x] Download English REDFM into:
  - [x] `data/rebel/train.jsonl`
  - [x] `data/rebel/validation.jsonl`
  - [x] `data/rebel/test.jsonl`

### 1. Trainability Gate

- [x] Add synthetic GPU smoke test:
  - [x] `scripts/test_re_training.jl`
- [x] Add real REDFM GPU trainer:
  - [x] `scripts/train_re_gpu.jl`
- [x] Verify actual-data smoke run with `configs/redfm_smoke.toml`
  - [x] first real update completes
  - [x] optimizer update completes
  - [x] checkpoint save completes

### 2. Short REDFM Smoke Run

- [x] Run `configs/redfm_smoke.toml` for `50-200` update steps.
- [x] Confirm:
  - [x] no scalar indexing errors
  - [x] no CUDA device exceptions
  - [x] loss trends downward over the run
  - [x] `checkpoint_last.jls` is resumable
- [x] Record:
  - [x] first-step compile time: `~124.3s` on resumed step `51`
  - [x] steady-state ms/update: `~51.6 ms` median / `~64.5 ms` mean over steps `100-200`
  - [x] steady-state tok/s: `~19.8k` median / `~18.4k` mean over steps `100-200`
  - [x] GPU memory usage: `~0.25-0.60 GiB` observed via `nvidia-smi` spot checks during resumed runs

### 3. Baseline REDFM Training

- [ ] Run `configs/redfm_base.toml` for a short baseline slice first:
  - [ ] `--max-steps 10`
  - [ ] `--max-steps 100`
- [ ] If stable, launch the full English baseline run with:
  - [ ] `julia --project=. scripts/train_re_gpu.jl --config configs/redfm_base.toml`
- [ ] Keep scope fixed to English REDFM only for this phase.
- [ ] Save checkpoints under:
  - [ ] `checkpoints/redfm_base/`

### 4. Baseline Evaluation

- [ ] Add a simple REDFM evaluation script or mode for:
  - [ ] entity loss / boundary loss
  - [ ] relation loss
  - [ ] confidence loss
- [ ] Add task metrics on validation/test:
  - [ ] entity span precision / recall / F1
  - [ ] relation precision / recall / F1
- [ ] Decide the first acceptance gate:
  - [ ] training is numerically stable
  - [ ] validation loss is improving
  - [ ] relation F1 is above trivial baseline

### 4a. Architecture Experiment Record

- [x] Write experiment log for the March 12, 2026 RE ablation session:
  - [x] [docs/RE_EXPERIMENT_LOG_2026-03-12.md](/home/christos/code/julia/Swamma/docs/RE_EXPERIMENT_LOG_2026-03-12.md)
- [x] Run next relation-side tests from the earlier conclusions:
  - [x] `candidate_only + biaffine_rank = 128`
  - [x] `candidate_only + pair_neighbor_radius = 12`
- [x] Add richer pair-proposal diagnostics:
  - [x] gold-pair hit rate before relation head
  - [x] rank of matched gold pairs among retained pairs
  - [x] missed-pair distance distribution
- [x] Rerun `candidate_only` under the richer diagnostics
- [x] Rerun plain baseline under the richer diagnostics for apples-to-apples comparison
- [x] Modify pair proposal ranking directly instead of more encoder churn
- [x] Expose pair proposer mode / local-global quota cleanly and tune it against baseline checkpoints
- [x] Run fresh short baseline with the best proposer candidate (`hybrid12`)
- [x] Explain whether `hybrid12` still helps under proposal-conditioned relation/confidence loss
- [x] Compare one neighboring proposer config against `hybrid12` under the proposal-conditioned evaluator
- [x] Decide whether to launch a longer run from `redfm_base_safe_pair_hybrid12.toml`
- [x] Add a learned sparse routed pair proposer on top of span representations
  - [x] add pair-router config fields to `RelationExtractionConfig`
  - [x] add `SparsePairProposalHead`
  - [x] add `:sparse` / `:sparse_hybrid` proposer modes
  - [x] route spans through learned buckets before pair truncation
  - [x] keep local-neighbor and small global-reserve pair paths available
- [x] Add a dedicated sparse proposer config:
  - [x] `configs/redfm_base_safe_pair_sparse_hybrid12.toml`
- [x] Add tests for sparse routed auto-proposal
- [x] Run fresh `100`-step sparse proposer vs `hybrid12` comparison
- [x] Decide whether sparse proposer beats `hybrid12` strongly enough to continue
- [x] Resume sparse proposer run to `1000` steps and record scheduled evals
- [x] Promote sparse proposer to the new main relation-side baseline
- [x] Continue sparse proposer run to `5000` steps and save checkpoints
- [ ] If needed, rerun short baseline vs `candidate_only` comparisons again to measure nondeterminism explicitly

### 4a1. Active Execution Plan (Sequenced, 2026-03-14)

Goal: maximize `pred spans + pred pairs` relation F1 quickly and safely before costly retraining.

Current locked baseline:
- Checkpoint family: `configs/redfm_base_safe_pair_sparse_learned128_nullw025_overgen4.toml`
- Best observed region: around step `1250`
- Current best decoded point: `rel_f1 = 0.0057` with:
  - global `threshold = 0.70`
  - global `no_relation_margin = 0.30`
  - per-relation thresholds `P127=0.95, P155=0.90, P571=0.85`
  - decode caps disabled (`decode_head_cap=0`, `decode_tail_cap=0`)

Execution sequence (strict order):

- [x] **Stage 1 — Baseline Lock + Repro**
  - [x] lock best checkpoint family and decode recipe in experiment log
  - [x] add a one-command reproducible eval recipe in README/docs
  - [x] rerun the locked recipe twice to quantify variance (`max_eval_batches=8`)
  - [x] freeze this as `v1_locked` reference row in checkpoint sweep docs

- [x] **Stage 2 — Decode Hardening (No Retrain)**
  - [x] global threshold sweep
  - [x] NO_RELATION margin sweep
  - [x] non-null probability sweep
  - [x] per-relation threshold overrides
  - [x] constrained decode caps (implemented and tested; currently harmful)
  - [x] add auto-calibration helper that searches per-relation thresholds from validation confusion
  - [x] keep only monotonic precision gains; reject any setting dropping F1 below locked baseline

- [x] **Stage 3 — Schema/Type Constraints (No Retrain)**
  - [x] implement invalid type-pair masking at decode (entity-type compatibility matrix)
  - [x] add optional inverse/symmetry relation consistency constraints
  - [x] run ablation: constraints only vs constraints + per-relation thresholds
  - [x] promote only if calibrated F1 beats locked baseline with no pair recall collapse (promoted `relation-consistency=resolve,min_count=1`: `rel_f1=0.0058` > `v1_locked=0.0057`; keep hard type-mask off)

- [ ] **Stage 4 — Evidence Quality Upgrade (Light Retrain)**
  - [x] add evidence diagnostics (`top tokens`, entropy/concentration) to evaluator
  - [x] test sentence-level or hybrid evidence pooling ablation
  - [x] rerun short resume window (`~250` steps) from locked checkpoint (`1260 -> 1510` on fused-evidence branch; eval@1500 `val_loss=15.2280`, `relation_loss=10.4497`)
  - [ ] keep branch only if it improves both calibrated F1 and precision (latest fused branch improves vs prior fused run to `rel_f1=0.0046`, but still below `v1_locked=0.0057`)

- [ ] **Stage 5 — Retrieval/Objective Upgrade (Heavier Retrain)**
  - [x] add edge ranking loss + hard negatives near decision boundary (`edge_ranking_loss_weight`, `edge_ranking_margin`, `edge_ranking_hard_negatives`; hard-neg hinge on top retrieval negatives)
  - [x] add type/distance-aware retrieval bias terms (distance/type scalar biases injected into pair-retrieval logits via input scales)
  - [x] run first controlled rank-loss continuation window (`1510 -> 1760`, fused-evidence branch) and record result (`pred spans + pred pairs rel_f1=0.0000`, not promoted)
  - [x] run soft-scheduled rank-loss continuation window (`1510 -> 1760`, delayed warmup) and record tradeoff (`max_eval_batches=8` looked promising, but full-val follow-up is still weak)
  - [x] run full-val checkpoint sweeps (`max_eval_batches=10000`) for `step_1510` vs `step_1760` to remove sampling noise
  - [x] run retrieval-bias ablation (`retrieval_distance_bias_scale=0.0`, `retrieval_type_bias_scale=0.0`) on `step_1760`
  - [x] run full threshold sweeps to test calibration sensitivity (`step_1760` 7-threshold full sweep, `step_1510` 3-threshold full sweep)
  - [x] record current verdict: continuation improves proposal coverage (`span_r 0.5551 -> 0.6359`, `oracle_rel 0.3299 -> 0.3981`) but decoded relation F1 remains very low (`best rel_f1 0.0012` on full-val), so not promotable
  - [x] run controlled `1000 -> 1500` continuation from locked checkpoint (new biaffine soft-rank config; eval@1500 `pair_r=0.2683`, `pair_t16=0.0976`, default `rel_f1=0.0022`)
  - [x] calibrate controlled-run checkpoint on full-val (best found: `threshold=0.90`, `margin=0.40`, `pred spans + pred pairs rel_f1=0.0041`)
  - [x] normalize against full-val baseline (`overgen4 checkpoint_last` best in tested sweep: `rel_f1=0.0009`), confirming controlled run is currently better under full-val decode
  - [x] run multi-seed variation check of controlled run (`seed=7,11,19`):
    - `seed=7`: calibrated (`0.90,0.40`) `rel_f1=0.0000`
    - `seed=11`: calibrated (`0.90,0.40`) `rel_f1=0.0000`
    - `seed=19`: calibrated (`0.90,0.40`) `rel_f1=0.0000` (default decode still non-zero: `rel_f1=0.0013`)
    - conclusion: high variance remains; do not promote recipe yet despite best-seed win (`seed=42 rel_f1=0.0041`)
  - [x] test gentler ranking schedule (`edge_w=0.015`, start `1375`, warmup `200`) across seeds (`42,11,19`) and re-sweep decode (`threshold=0.50,0.70,0.90`, `margin=0.0`)
  - [x] gentle-schedule outcome:
    - `seed=42`: best `rel_f1=0.0010`
    - `seed=11`: best `rel_f1=0.0016`
    - `seed=19`: best `rel_f1=0.0016`
    - consistent non-zero behavior, but gains remain small and proposal coverage can still dip on harder seeds
  - [x] close margin-band check for gentle schedule (`margin=0.0/0.1/0.2`, full-val):
    - `seed=42`: `0.0010 / 0.0017 / 0.0017`
    - `seed=11`: `0.0016 / 0.0017 / 0.0015`
    - `seed=19`: `0.0016 / 0.0008 / 0.0008`
    - verdict: no robust cross-seed gain; keep gentle as stability-control ablation, not promotion path
  - [x] run targeted non-null sweep on best gentle point (`seed=11`, `thr=0.70`, `margin=0.10`, nonnull `0.00..0.80`):
    - `pred spans + pred pairs rel_f1` stayed flat at `0.0017` for all non-null values
    - proposal metrics unchanged (`oracle_rel=0.5164`, `pair_r=0.1261`, `pair_t16=0.0596`)
    - verdict: non-null gate is not currently the bottleneck for this branch
  - [x] run aggressive-schedule decode-relaxation check on non-42 seeds (full-val, `margin=0.10`, `threshold=0.60/0.70/0.80`):
    - `seed=7`: best `rel_f1=0.0005`
    - `seed=11`: best `rel_f1=0.0006`
    - `seed=19`: best `rel_f1=0.0005`
    - verdict: decode relaxation does not fix aggressive variance collapse; issue is mostly model-state stability
  - [x] run midpoint schedule probe (`edge_w=0.02`, `start=1350`, `warmup=250`, `hard_negs=12`) on `seed=11` (`1000 -> 1250`):
    - sampled eval@1250: `rel_f1=0.0015`, `pair_r=0.0244`, `pair_t16=0.0122`, `oracle_rel=0.2805`
    - full-val sweep (`margin=0.10`, `threshold=0.60/0.70/0.80`) best `rel_f1=0.0004`
    - verdict: reject midpoint schedule; retrieval/pair coverage regresses too hard
  - [x] run aggressive reproducibility probe with explicit `seed=42` (fresh run dir, `1000 -> 1500`):
    - sampled eval@1500: `rel_f1=0.0027`, `pair_r=0.2439`, `pair_t16=0.1220`, `oracle_rel=0.6829`
    - strict full-val check (`threshold=0.90`, `margin=0.40`): `rel_f1=0.0011` (fails to reproduce prior `0.0041`)
    - relaxed full-val sweep (`margin=0.10`, `threshold=0.60/0.70/0.80`) best `rel_f1=0.0012`
    - verdict: prior `0.0041` likely high-variance outlier; run-to-run nondeterminism is a blocking issue
  - [x] add determinism hardening for RE experiments before next schedule search:
    - route `prepare_rebel_batch` negative-mention and hard-negative pair sampling through explicit RNG instead of global RNG
    - pass seeded RNG through training and eval `make_batch` paths (`train/eval/oracle/auto-calibration`)
    - seed global RNG at startup (`Random.seed!(run_config.seed)`)
    - smoke validation: two identical sampled eval runs now produce identical metric rows
  - [x] run post-fix compact cross-seed strict recheck (`threshold=0.90`, `margin=0.40`, full-val):
    - `seed42_rerun`: `rel_f1=0.0011`
    - `seed7`: `rel_f1=0.0000`
    - `seed11`: `rel_f1=0.0000`
    - `seed19`: `rel_f1=0.0000`
    - verdict: strict operating point remains non-robust; schedule-only tuning is exhausted for now
  - [x] run decisive mode comparison on identical checkpoint (`seed42_rerun`, full-val, `margin=0.10`, `threshold=0.60/0.70/0.80`):
    - `sparse_hybrid` best `rel_f1=0.0012` (`pair_r=0.1554`, `pair_t16=0.0639`, `oracle_rel=0.6295`)
    - `edge_retrieval_v2` best `rel_f1=0.0012` (same pair/oracle metrics)
    - verdict: no measurable lift yet; keep `edge_retrieval_v2` as scaffold branch, not promotion candidate
  - [x] run short adaptation probe with `edge_retrieval_v2` (`seed42`, `1000 -> 1250`) and re-evaluate full-val:
    - sampled eval@1250: `rel_f1=0.0000`, `oracle_rel=0.1829`, `pair_r=0.0610`, `pair_t16=0.0488`
    - full-val sweep (`margin=0.10`, `threshold=0.60/0.70/0.80`) best `rel_f1=0.0013` but with severe coverage collapse (`oracle_rel=0.1978`, `pair_r=0.0354`, `pair_t16=0.0294`)
    - verdict: hard no-go for current edge-v2 adaptation recipe; do not continue this branch without redesigning edge selection/supervision
  - [ ] promote only if proposal metrics (`oracle_rel`, `pair_recall`, `pair_t16`) stay competitive and calibrated F1 improves

- [ ] **Stage 6 — Long Run + Freeze**
  - [ ] launch longer run only after Stage 2-5 pass gates
  - [ ] checkpoint sweep + calibrated decode report table
  - [ ] freeze `v2_candidate` only if repeated eval beats `v1_locked`

### 4b. Swamma RE v2 Redesign Gate

Goal: replace the current diagnostic-grade RE scaffold with a serious span-graph relation extractor above the stable `SwammaBlock` encoder.

- [x] Write `Swamma RE v2` architecture note:
  - [x] document module graph from tokens -> spans -> edges -> evidence -> relations
  - [x] define tensor shapes for each stage
  - [x] define complexity target for each stage
  - [x] state what stays from the current stack vs what is removed
  - [x] save draft doc: `docs/SWAMMA_RE_V2_ARCHITECTURE.md`
- [ ] Freeze the current sparse proposer stack as the `v1` baseline:
  - [ ] keep `configs/redfm_base_safe_pair_sparse_hybrid12.toml`
  - [ ] keep `checkpoint_step_1250.jls` / `checkpoint_step_5000.jls` as reference checkpoints
  - [ ] record the best observed `v1` validation numbers in docs
- [ ] Decide `v2` acceptance gates before implementation:
  - [ ] better proposal-conditioned relation loss than `v1`
  - [ ] better pair recall and pair@16 than `v1`
  - [ ] non-zero relation F1 on the sampled validation evaluator
  - [ ] no unacceptable throughput collapse vs `v1`

### 4c. Mention Proposal v2

- [x] Replace heuristic span ranking as the primary mention scorer
- [x] Add configurable mention score modes: `heuristic`, `learned`, `hybrid`
- [x] Blend learned mentionness with boundary/entity heuristic during proposal
- [x] Keep hybrid mention scoring as the active baseline after learned-only regression
- [ ] Add learned span proposal modules:
  - [ ] `StartHead`: `LayerNorm -> Dense(d -> 1)`
  - [ ] `EndHead`: `LayerNorm -> Dense(d -> 1)`
  - [ ] `TypeHead`: `LayerNorm -> Dense(d -> num_entity_labels)`
  - [ ] `SpanWidthEmbedding`
  - [x] `SpanMentionHead`
- [ ] Add learned span composition:
  - [ ] compose `[start ; end ; mean ; width_emb]`
  - [ ] project to one span vector
  - [x] emit mentionness logit
  - [ ] emit coarse entity-type logits
- [ ] Add coarse-to-fine mention pruning:
  - [ ] pre-prune top token starts
  - [ ] pre-prune top token ends
  - [ ] only enumerate spans from retained boundaries
  - [x] keep top `max_candidate_spans` by learned mentionness
- [x] Keep old heuristic span proposer only as an ablation path
- [ ] Add mention-level metrics:
  - [x] mention recall@K
  - [x] mention precision / recall / F1
  - [x] oracle relation coverage from retained mentions

### 4d. Span Context Graph v2

- [x] Add sparse span-to-span context layers after mention pruning
- [ ] Define sparse span graph edges:
  - [x] adjacent mentions by document order
  - [x] same-sentence mention neighbors
  - [x] routed semantic neighbors from span embeddings
  - [ ] optional speaker / section / sentence-root neighbors if data supports them
- [ ] Implement a span-context block:
  - [x] span query/key/value projections
  - [x] sparse attention or message passing over graph edges
  - [x] residual + norm + FFN on span nodes
- [x] Support `0 / 1 / 2` span-context layers via config
- [x] Add ablation flags to disable each edge family independently
- [x] Run short warm-start probe for `span_context_layers=1` from `step_1000` with sentence-edge inputs and compare against matched `span_context_layers=0` control:
  - [x] `span_context_layers=1` (`1000 -> 1030`): severe regression (`val_loss 38.8 -> 29.5`, `pair_recall 0.0610 -> 0.0122`, `rel_f1 0.0000`)
  - [x] matched control `span_context_layers=0` (`1000 -> 1030`): stable (`val_loss ~14-16`, `pair_recall 0.183/0.134/0.110`, still `rel_f1 0.0000`)
  - [x] verdict: do not introduce span-context depth via mid-run architecture warm-start; revisit with from-scratch or staged pretrain recipe
- [x] Add staged edge-family gating support in trainer inputs for span-context graph:
  - [x] config keys: `span_context_adjacent_start_step`, `span_context_sentence_start_step`, `span_context_semantic_start_step`
  - [x] training loop now applies these gates via `with_retrieval_bias_inputs(...; step=next_step)` without changing checkpoint-serialized model structs
- [x] Add global span-context activation gate for staged warm-start (`span_context_start_step`) and ensure training-time eval respects current step gate:
  - [x] `span_context_enabled` runtime input added and threaded through proposal/eval/oracle paths
  - [x] `evaluate_model(...; current_step=step)` now uses step-aware runtime gating during in-training eval
  - [x] staged probe (`span_context_start_step=1040`, `1000 -> 1050`) outcome:
    - pre-activation (`<=1030`) avoids immediate collapse and keeps moderate losses
    - post-activation (`1040+`) regresses sharply (`val_loss ~44-46`) and underperforms matched `span_context_layers=0` control (`val_loss ~14-15`, `rel_f1 ~0.0025-0.0027`)
  - [x] verdict: staged gate fixes eval-mismatch bug but span-context activation remains non-promotable in current recipe

### 4e. Edge Retrieval v2

- [x] Add learned pair-retrieval head over sparse candidate pairs
- [x] Over-generate sparse candidate pairs and rerank them by learned retrieval score
- [x] Add retrieval BCE supervision on candidate relation pairs
- [ ] Replace heuristic pair proposal with a learned edge retriever as the primary path
- [x] Add `edge_retrieval_v2` proposer-mode scaffold in `RelationExtraction.jl`:
  - [x] mode accepted by pair-proposer dispatch and summary printer
  - [x] semantic precompute wiring enabled for the new mode (decoupled from router precompute gate)
  - [x] heuristic anchor fanout disabled for `edge_retrieval_v2` path
  - [x] smoke eval command runs successfully (`max_eval_batches=1`)
  - [x] unit smoke test added and passing (`Relation Extraction Edge Retrieval v2 Proposal`)
- [ ] Add learned retrieval projections:
  - [x] head query projection (via `PairRetrievalHead.HeadProjection`)
  - [x] tail key projection (via `PairRetrievalHead.TailProjection`)
  - [x] relation-agnostic compatibility projection (runtime projected-compat term via `retrieval_compatibility_scale`)
  - [x] pair distance embedding (via `PairRetrievalHead.DistanceEmbedding`)
  - [x] sentence-distance embedding (shared-table runtime path via `retrieval_sentence_embedding_scale`)
- [x] Keep sparse retrieval families:
  - [x] local neighbors (edge-v2 runtime gate: `edge_v2_use_local_neighbors`)
  - [x] routed buckets (edge-v2 runtime gate: `edge_v2_use_routed_buckets`)
  - [x] semantic top-k matches (edge-v2 runtime gate: `edge_v2_use_semantic_topk`)
  - [x] small global reserve (edge-v2 runtime gate: `edge_v2_use_global_reserve`)
  - [x] anchor-expanded hub pairs (top-`k` anchors x all spans, `O(N·k)`)
- [x] Run quick edge-v2 family-gating ablation (`max_eval_batches=32`) on `seed42_rerun` checkpoint:
  - [x] all-families / semantic+reserve-only / local+routed+reserve-no-semantic produced identical sampled metrics
  - [x] no immediate retrieval/F1 lift observed from family toggles on this checkpoint (keep as structural scaffold, not promotion signal)
- [x] Run larger edge-v2 all-families check (`max_eval_batches=128`) on `seed42_rerun` checkpoint:
  - [x] metrics remain identical to prior edge-v2 point (`best rel_f1=0.0012`, `pair_r=0.1554`, `pair_t16=0.0639`)
  - [x] verdict: multi-family composition is landed and test-covered, but not a promotion signal on current checkpoint family
- [ ] Implement explicit edge retrieval score:
  - [x] dot-product compatibility (runtime bias term over head-tail span-vector dot product; `retrieval_dot_bias_scale`)
  - [x] local bias (runtime retrieval bias scale over local token distance window)
  - [x] type-compatibility bias (runtime bias term from non-null entity-type overlap; `retrieval_type_compat_bias_scale`)
  - [x] sentence / distance bias (runtime retrieval bias scales over pair distance bucket + sentence-distance bucket)
- [x] Add local-distance retrieval bias runtime hook (`retrieval_local_bias_scale`) and run threshold ablations on `seed42_rerun` checkpoint:
  - [x] sampled (`max_eval_batches=128`): best `pred spans + pred pairs rel_f1=0.0012` (unchanged), coverage slightly lower (`pair_r 0.1554 -> 0.1537`, `pair_t16 0.0639 -> 0.0604`)
  - [x] full-val (`max_eval_batches=10000`): matches sampled outcome; no meaningful F1 lift
  - [x] verdict: non-promotable on current checkpoint/decode regime; keep as optional knob
- [x] Add sentence-distance retrieval bias runtime hook (`retrieval_sentence_bias_scale`) and run sampled threshold ablation on `seed42_rerun` checkpoint:
  - [x] baseline (`scale=0.00`, `margin=0.10`, thresholds `0.60/0.70/0.80`, `max_eval_batches=8`) best `pred spans + pred pairs rel_f1=0.0043`
  - [x] sentence-bias (`scale=0.15`) best `pred spans + pred pairs rel_f1=0.0044`
  - [x] larger sampled check (`max_eval_batches=128`) confirms no meaningful change: baseline best `0.0012` vs sentence-bias best `0.0012` (pair coverage delta ~`+0.0009` recall)
  - [x] full-val check (`max_eval_batches=10000`) matches the larger sampled result exactly: no meaningful F1 change (`0.0012` vs `0.0012`)
  - [x] verdict: no material lift in sampled or full-val checks; keep as optional non-promoted knob
- [x] Add type-compat + dot-compat retrieval bias runtime hooks and run threshold ablations on `seed42_rerun` checkpoint:
  - [x] `type_compat=0.15` full-val (`max_eval_batches=128`) is identical to baseline (`best rel_f1=0.0012`, `pair_r=0.1554`, `pair_t16=0.0639`)
  - [x] `dot=0.10` full-val (`max_eval_batches=128`) slightly shifts threshold tradeoff but reduces coverage (`pair_r 0.1554 -> 0.1459`, `pair_t16 0.0639 -> 0.0484`) with no robust F1 lift (best `0.0013`)
  - [x] combined (`type_compat=0.15`, `dot=0.10`) matches dot-only behavior on full-val
  - [x] verdict: keep both scales at `0.0` for the promoted baseline; retain knobs for future checkpoints only
- [x] Add sentence-distance embedding runtime hook (`retrieval_sentence_embedding_scale`) using shared retrieval distance embedding table:
  - [x] checkpoint-safe implementation (no parameter shape change; reuse `PairRetrievalHead.DistanceEmbedding`)
  - [x] sampled threshold sweep (`max_eval_batches=32`, scale `0.25`) shows no meaningful lift vs baseline (`best rel_f1` unchanged at `0.0018`, slight pair recall dip `0.1729 -> 0.1700`)
  - [x] verdict: keep default `0.0` on promoted baseline; retain as optional knob
- [x] Add relation-agnostic compatibility runtime hook (`retrieval_compatibility_scale`) from learned retrieval projections:
  - [x] checkpoint-safe implementation (no parameter/state shape change; compatibility score from projected `feature ⊙ (head ⊙ tail)`)
  - [x] sampled threshold sweep (`max_eval_batches=32`, scale `0.25`) keeps best `rel_f1` unchanged (`0.0018`) with small `pair_t16` increase (`0.0634 -> 0.0692`)
  - [x] full-val threshold sweep (`max_eval_batches=128`) shows only marginal shift (`best rel_f1 0.0012 -> 0.0013`, `pair_r 0.1554 -> 0.1546`, `pair_t16 0.0639 -> 0.0648`)
  - [x] verdict: keep as optional knob; improvement is too small to promote as a baseline change
- [x] Add retrieval supervision:
  - [x] gold edge ranking loss
  - [x] hard-negative edge mining
  - [x] pair recall@K / pair@8 / pair@16 / pair@32 metrics
  - [x] matched-pair rank tracking
- [x] Keep current sparse proposer as a fallback ablation path

### 4f. Pair Evidence Selector v2

- [x] Add pair-specific evidence retrieval from token or sentence states
- [ ] Decide evidence granularity:
  - [x] token-level evidence selector
  - [ ] sentence-level evidence selector
  - [ ] hybrid token + sentence evidence
- [ ] Implement pair seed features:
  - [x] `[head ; tail ; abs_diff ; product]`
  - [ ] distance embedding
  - [ ] sentence-offset embedding
  - [ ] coarse entity-type pair features
- [ ] Implement pair-to-evidence attention:
  - [x] cross-attention from pair seed to token states
  - [x] token-mask-aware weighted pooling
  - [ ] top-m evidence token retention or weighted pooling
  - [ ] optional sentence pooling before evidence selection
- [ ] Add evidence diagnostics:
  - [x] top evidence token positions
  - [ ] top evidence sentence ids
  - [x] evidence entropy / concentration
- [x] Add dedicated evidence-v2 experiment config:
  - [x] `configs/redfm_base_safe_pair_sparse_hybrid12_evidence_v2.toml`
- [ ] Promote evidence-v2 over the structured-retrieval baseline only if it improves both:
  - [ ] step-250 total / relation loss
  - [ ] proposal-conditioned coverage and retrieval (`oracle_rel`, `pair_recall`, `pair_t16`)

### 4g. Relation Decoder v2

- [x] Replace biaffine-only relation scoring with a fused decoder
- [x] Keep biaffine as an auxiliary feature, not the full decoder
- [ ] Add fused relation features:
  - [x] biaffine compatibility scores
  - [x] pair seed MLP features
  - [x] evidence summary features
  - [ ] type-compatibility features
  - [ ] distance / sentence features
- [ ] Implement a fused relation classifier:
  - [x] feature concatenation or additive residual fusion
  - [x] relation MLP head
  - [x] confidence head conditioned on fused features
- [ ] Add decoder ablations:
  - [x] biaffine only
  - [ ] pair MLP only
  - [ ] pair MLP + evidence
  - [x] gated residual fused decoder
- [x] Add dedicated decoder-v2 experiment config:
  - [x] `configs/redfm_base_safe_pair_sparse_hybrid12_decoder_v2.toml`
- [ ] Promote decoder-v2 over the structured-retrieval baseline only if it improves both:
  - [ ] step-250 total / relation loss
  - [ ] proposal-conditioned retrieval metrics (`oracle_rel`, `pair_recall`, `pair_t16`)
  - [ ] keep if calibrated `pred spans + pred pairs rel_f1` beats current `0.0050` checkpoint baseline

### 4h. Training Objective v2

- [ ] Reduce train/infer mismatch in RE training
- [ ] Add staged training modes:
  - [ ] teacher-forced gold spans / gold pairs
  - [ ] predicted spans + gold pairs
  - [x] predicted spans + predicted pairs
  - [x] mixed scheduled-sampling curriculum
- [ ] Add new losses:
  - [ ] token boundary loss
  - [ ] token type loss
  - [x] span mentionness loss
  - [x] edge retrieval ranking loss
  - [ ] relation classification loss
  - [ ] confidence / calibration loss
- [ ] Add hard-negative sampling upgrades:
  - [ ] mention negatives near gold spans
  - [ ] edge negatives sharing one gold endpoint
  - [ ] same-type but wrong-relation negatives
  - [ ] same-sentence distractor negatives
- [ ] Add loss-weight config surface for all new objectives
- [x] Add curriculum config surface for switching from teacher-forced to predicted candidates
- [x] Add dedicated curriculum-v2 experiment config:
  - [x] `configs/redfm_base_safe_pair_sparse_hybrid12_curriculum_v2.toml`
- [ ] Promote curriculum-v2 over the structured-retrieval baseline only if it improves both:
  - [ ] proposal-conditioned relation / confidence loss without collapsing retrieval coverage
  - [ ] end-task sampled validation (`oracle_rel`, `pair_recall`, `rel_f1`)
- [x] Run proposal-conditioned curriculum sanity from step `1000 -> 1250` (`overgen4_curric50`) and reject if coverage collapses
- [x] Run mild curriculum sanity (`overgen4_curric10`) and reject if it still underperforms non-curriculum baseline

### 4i. Decoding And Calibration v2

- [x] Add constrained relation decoding instead of independent thresholding only
- [ ] Add schema-aware constraints:
  - [x] invalid type-pair masking
  - [x] optional symmetry / inverse-relation handling
  - [x] confidence threshold calibration per relation
- [ ] Add validation-time calibration sweep:
  - [x] global confidence threshold
  - [x] global non-null probability threshold
  - [x] per-relation confidence threshold
  - [x] max-relations-per-head and max-relations-per-tail caps
- [ ] Report calibrated relation precision / recall / F1 separately from raw logits

### 4j. Evaluation And Benchmarking v2

- [ ] Extend validation logging beyond loss-only summaries
- [ ] Add mention metrics:
  - [ ] span precision / recall / F1
  - [ ] mention recall@K
- [ ] Add retrieval metrics:
  - [ ] pair recall@K
  - [ ] pair@8 / pair@16 / pair@32
  - [ ] mean matched-pair rank
- [ ] Add evidence metrics:
  - [ ] evidence token recall if labels exist
  - [ ] evidence sentence recall if labels can be approximated
- [ ] Add end-task relation metrics:
  - [ ] micro precision / recall / F1
  - [ ] macro precision / recall / F1
  - [ ] no-relation vs positive-relation calibration
- [ ] Add checkpoint sweep table for `v2` runs
- [x] Fix CLI parser coverage for sweep diagnostics:
  - [x] `--pair-sweep-checkpoint`
  - [x] `--pair-sweep-budgets`
  - [x] `--pair-sweep-overgenerate`
  - [x] `--nonnull-sweep-checkpoint`
  - [x] `--nonnull-sweep-values`
  - [x] `--threshold-sweep-nonnull`
  - [x] `--decode-head-cap`
  - [x] `--decode-tail-cap`
  - [x] `--per-relation-thresholds`
- [ ] Compare `v2` against:
  - [ ] plain local proposer baseline
  - [ ] `hybrid12`
  - [ ] sparse routed `v1`

### 4k. Implementation Order

- [x] Phase 0: baseline lock + reproducible calibrated eval (`v1_locked`)
- [x] Phase 1: decode-only calibration (global + margin + non-null + per-relation)
- [x] Phase 2: schema/type constraints at decode
- [x] Phase 3: evidence diagnostics + evidence selector ablations
- [ ] Phase 4: retrieval/objective upgrades with controlled resumes
- [ ] Phase 5: long-run promotion and freeze only if repeated calibrated F1 improves over `v1_locked`

### 5. Operational Cleanup Before Long Run

- [ ] Separate compile-heavy warmup from steady-state measurement in the RE training workflow.
- [x] Add resume command to README or docs for REDFM training.
- [x] Add a note describing expected first-step compile time vs later step time.
- [ ] Verify checkpoint resume on `redfm_smoke` and `redfm_base`.
- [x] Add mandatory end-of-session report workflow (`AGENTS.md` + `docs/SESSION_REPORT.md`)

### 6. Distillation Preparation

- [ ] Freeze the supervised English REDFM baseline first.
- [ ] Only after baseline is stable, add teacher annotation generation for:
  - [ ] REBEL
  - [ ] mREBEL if needed later
- [x] Extend training rows to optionally carry teacher targets:
  - [x] teacher entity targets
  - [x] teacher relation targets
  - [x] teacher confidence targets
- [x] Add mixed supervised + distillation loss to `scripts/train_re_gpu.jl`.
- [x] Add distillation pilot config with non-zero teacher weights and run short resume smoke (`2000 -> 2005`).
- [x] Validate eval path on pilot checkpoint with threshold-sweep smoke (`max_eval_batches=16`).
- [x] Add teacher-payload validator script and confirm current REBEL train split has zero teacher payload coverage.
- [x] Add teacher-target merge script that writes span-based `teacher_relations` into REBEL rows.
- [x] Add teacher-request export script for REBEL rows with span-based response schema.
- [x] Add a lightweight pilot-evaluation script for parsed REBEL teacher outputs vs gold labels.
- [x] Add parser for raw teacher response text -> normalized REBEL teacher annotation rows.
- [x] Add teacher-data generation script for REBEL request JSONL -> raw teacher responses.
- [x] Add one-command REBEL teacher-corpus preparation orchestrator.
- [x] Add coverage gate so distillation runs fail fast when teacher payloads are missing.
- [x] Inject teacher-only positive relations into candidate pairs without forcing contradictory supervised `NO_RELATION` labels.
- [x] Switch teacher relation alignment to span-based mapping so teacher entity order no longer needs to match gold entity order.
- [x] Inject teacher-only entity spans into the training candidate spans while keeping gold span supervision masked separately.
- [x] Add a Lux-native causal teacher foundation module (`src/NativeTeacherLM.jl`) with:
  - [x] RoPE
  - [x] grouped-query-capable causal self-attention
  - [x] RMSNorm pre-norm decoder blocks
  - [x] greedy generation helpers
  - [x] focused forward/masking/generation tests
- [x] Add a native Hugging Face config + checkpoint importer for the first supported teacher family:
  - [x] pick `ibm-granite/granite-4.0-micro` as the first target
  - [x] map Granite HF tensor names to `NativeTeacherLM` parameter structure
  - [x] add synthetic safetensors import coverage tests
  - [x] smoke the real Granite HF config through the Julia loader path
- [x] Run one full real-weight Granite micro load through `load_granite_model(...)` on local/downloaded shards.
- [x] Validate tokenizer/chat-template compatibility against the chosen model family.
  - [x] replace the brittle PyCall `transformers` tokenizer path with a local fallback based on `tokenizers` + `chat_template.jinja`
  - [x] verify the native Granite smoke script renders the prompt correctly
  - [x] verify a real forward pass runs on the rendered prompt
- [x] Add KV-cache-aware decoding path for `NativeTeacherLM` so teacher generation is not full-prefix recompute.
  - [x] add per-layer attention KV cache structs
  - [x] add cached prefill / next-token helpers
  - [x] add cached greedy generation helper
  - [x] verify cached logits against full recompute on unit tests and real Granite smoke
- [x] Add a Julia-native RE teacher generation script on top of `NativeTeacherLM`.
  - [x] consume the same REBEL request JSONL format as the Python generator
  - [x] emit the same raw-response JSONL contract for the parser/merge pipeline
  - [x] smoke one request end-to-end with the native Granite path
- [ ] Improve native teacher response quality / control.
  - [x] add JSON-gated native generation with response-prefix support, verbose timing, and error previews
  - [x] compare native Granite next-token behavior against Hugging Face on a short prompt
  - [x] fix Granite-specific inference drift by matching `attention_multiplier` and `residual_multiplier`
  - [x] add a compact RE teacher-request prompt style for smaller, less ambiguous native generation prompts
  - [x] switch the native generator default dtype to `float32` for CPU-bound runs
  - [ ] reduce empty or malformed completions under greedy decode on real RE prompts
  - [x] add a reusable HF-vs-native parity smoke for exact prompt-token comparisons
  - [x] use the parity smoke to isolate the remaining long-context native generation mismatch
  - [x] identify the first-token parity break window (`192` tokens still matches, `256` tokens diverges)
  - [x] fix the long-context RoPE mismatch in `NativeTeacherLM` (Granite uses half-split rotation, not adjacent even/odd pairs)
  - [x] test lightweight JSON-aware decode heuristics (`0` suppression at numeric field starts, early EOS/code-fence suppression)
  - [x] confirm exact-token parity is restored at `256` and `308` prompt tokens after the RoPE fix
  - [x] produce one accepted native RE teacher row under strict JSON validation
  - [x] identify that entity-first decoding is the wrong control surface on the broader label schema (runaway entity/date enumeration)
  - [x] tighten the compact RE prompt toward relation-first, high-confidence, low-overgeneration extraction
  - [x] switch the best native control recipe to a relation-first response prefix:
    - [x] `{"entities":[],"relations":[{"head_start":`
  - [x] fix response assembly so strict validation accepts completions that restart the full JSON object instead of always re-prepending the response prefix
  - [x] verify a tiny strict relation-first shard can accept on more than one row
    - [x] initial 3-row compact/no-title shard: `2/3` accepted under strict JSON
    - [x] final 3-row compact/no-title shard: `3/3` accepted under strict JSON
    - [x] final accepted rows parsed downstream with `3` total relations and `0` entities
  - [x] raise the tiny strict relation-first shard from `2/3` to `3/3`
    - [x] add relation-only partial-JSON salvage for continuation-style relation floods
    - [x] verify the hard row accepts under strict JSON after salvage
  - [x] run a 10-row strict relation-first shard with the current best recipe
    - [x] strict generation acceptance: `10/10`
    - [x] strict downstream parse success: `10/10`
    - [x] parsed relation count total: `7`
  - [x] tighten strict native validation so accepted rows must also use allowed entity / relation labels
  - [x] add compact prompt relation glosses for Wikidata property IDs (`P127=owned by`, etc.)
  - [x] compare 10-row label-gated pilots before vs after relation glosses
    - [x] no-gloss label-gated pilot: `7/10` accepted, top-1 label-in-gold `0.000`, collapsed to `P161`
    - [x] glossed label-gated pilot: `7/10` accepted, top-1 label-in-gold `0.143`, predicted labels spread across `P127/P161/P57/P136`
  - [x] test decode-time relation-label token constraints on the glossed 10-row pilot
    - [x] constrained glossed pilot: `6/10` accepted, top-1 label-in-gold `0.000`
    - [x] conclusion: token-level label constraints reduce acceptance without improving semantic quality
  - [x] test natural-language relation-label variants with downstream canonicalization
    - [x] `id_or_name` prompt: `3/10` accepted, top-1 label-in-gold `0.000`
    - [x] `name`-only prompt: `5/10` accepted, top-1 label-in-gold `0.200`
    - [x] conclusion: naming variants are more brittle on acceptance and do not improve absolute correct-row count over the glossed ID prompt
  - [x] run a larger native relation-first pilot shard (`50-100` rows) and inspect non-empty relation yield / label mix
    - [x] glossed single-pass 50-row pilot: `24/50` accepted, `22` predicted relations, top-1 label-in-gold `0.0455`
    - [x] no-gloss single-pass 50-row pilot: `50/50` matched, `29` predicted relations, top-1 label-in-gold `0.0345`, full `P106` collapse
  - [x] add a stronger semantic control step for relation-label choice
    - [x] either constrained label decoding at `label` fields
    - [x] or a two-stage natural-language relation-name -> Wikidata-ID mapping path
    - [x] prefer an explicit two-stage mapping/reranking path; simple token constraints and inline name variants were not enough
    - [x] glossed stage1 + two-stage relabel on 50 rows: `24/50` matched, `22` predicted relations, top-1 label-in-gold `0.3636`, exact label-set match `0.0455`
    - [x] no-gloss stage1 + two-stage relabel on 50 rows: `50/50` matched, `29` predicted relations, top-1 label-in-gold `0.4483`, exact label-set match `0.1379`
    - [x] no-schema stage1 + single-pass 50-row pilot: `50/50` matched, `50` predicted relations, top-1 label-in-gold `0.1000`, exact label-set match `0.0200`
    - [x] no-schema stage1 + two-stage relabel on 50 rows: `50/50` matched, `50` predicted relations, top-1 label-in-gold `0.3800`, exact label-set match `0.1000`
    - [x] no-schema stage1 + single-pass 100-row pilot: `100/100` matched, `100` predicted relations, top-1 label-in-gold `0.1200`, exact label-set match `0.0200`
    - [x] no-schema stage1 + two-stage relabel on 100 rows: `100/100` matched, `100` predicted relations, top-1 label-in-gold `0.3800`, exact label-set match `0.0800`
    - [x] no-schema stage1 + single-pass 250-row pilot: `250/250` matched, `250` predicted relations, top-1 label-in-gold `0.1400`, exact label-set match `0.0320`
    - [x] no-schema stage1 + two-stage relabel on 250 rows: `250/250` matched, `250` predicted relations, top-1 label-in-gold `0.3480`, exact label-set match `0.1000`
    - [x] conclusion: the best current architecture is compact no-title no-schema relation-first stage 1 plus semantic stage-2 relabeling; glossed single-pass prompting is no longer the main path
  - [x] improve stage-1 non-empty relation yield without sacrificing the no-gloss stage-1 acceptance advantage
    - [x] test whether stage-1 relation span quality can be improved with lighter prompt edits that do not reintroduce the glossed acceptance collapse
    - [x] rerun the two-stage stack on a `100`-row shard once the stage-1 operating point is locked
  - [ ] investigate why stage-2 relabel updates miss a small residual tail (`48/50` on the repaired 50-row no-schema pilot, `98/100` on the repaired 100-row no-schema pilot)
    - [x] add safe merge-side salvage for truncated `"label"` fields and near-gloss aliases such as `headquartered in -> P159`
    - [x] tighten the stage-2 prompt against free-form type answers like `city`, `single`, and `nickname`
    - [x] confirm the remaining failures are semantic drift, not key mismatch or JSON parser failure
    - [ ] inspect the remaining hard case(s), currently dominated by outputs like `single` that still refuse to map to an allowed relation gloss
  - [ ] inspect whether the no-schema stage-1 relation-rate gain is worth the slight drop in exact label-set match versus the smaller 50-row no-gloss relabel pilot
  - [x] compare greedy vs sampled settings on a small held-out request set after the prompt/parity fixes
    - [x] stage-2 sampled relabel on the promoted 100-row no-schema stack (`temperature=0.7`, `top_p=0.9`) updated `100/100` relations after merge robustness fixes
    - [x] sampled semantic quality regressed versus greedy:
      - [x] greedy: top-1 label-in-gold `0.3800`, exact label-set `0.0800`
      - [x] sampled: top-1 label-in-gold `0.2700`, exact label-set `0.0600`
    - [x] conclusion: keep greedy stage-2 relabel as the default; sampling increases verbose off-target continuations without improving label choice
- [ ] Run first controlled distillation comparison (`base` vs `distill`) at matched step budget.
  - [x] freeze the native teacher-corpus recipe for the first larger build:
    - [x] stage 1: compact no-title no-schema relation-first generation
    - [x] stage 2: greedy relabel with glossed choices
  - [x] launch full-train stage-1 teacher generation from the validated `250`-row seed
  - [ ] wait for full-train stage-1 generation to finish, then run full-train stage-2 relabel
  - [ ] merge the completed full-train teacher corpus back into REDFM JSONL
  - [ ] run the first matched-budget distillation training comparison on the merged corpus

### 7. Explicit Non-Goals For This Phase

- [ ] Do not add more languages yet.
- [ ] Do not add SREDFM yet.
- [ ] Do not start full teacher distillation before the English supervised baseline is stable (plumbing smoke only for now).

## 0. Naming And Legacy Purge (Breaking)

- [x] Rename top-level module from `Ossamma` to `Swamma`.
- [x] Rename primary file `src/Ossamma.jl` to `src/Swamma.jl` and update all `include` / `using` paths.
- [x] Rename public API symbols to `Swamma*` and remove `Ossamma*` exports.
- [x] Rename terminology in code/config/docs from `Oscillator*` to `WaveGate*` where it refers to WavePDE gating.
- [x] Remove legacy aliases, exports, and the removed `ossm.jl` code path from the package graph.
- [x] Update scripts/tests/docs to import and reference `Swamma*` APIs.
- [x] Remove stale docs/files that used the old `Ossamma` naming.

## 1. mHC Core Implementation (Paper-Faithful Path)

- [ ] Add `src/HyperConnect.jl` with:
  - [ ] Sinkhorn-Knopp projection to doubly-stochastic matrices
  - [ ] `ManifoldHyperConnection` layer
  - [ ] Non-negative normalized `H_pre` / `H_post`
  - [ ] Near-identity initialization
- [ ] Add optional input-dependent parameterization (`dynamic=true`) per paper.
- [ ] Add unit tests for mHC invariants:
  - [ ] row sums ~1, col sums ~1
  - [ ] non-negativity
  - [ ] stability under composition (`H_res` products)

## 2. mHC Integration Into SWAMMA Blocks

- [ ] Integrate mHC into `SwammaBlock` residual pathways:
  - [ ] sublayer residual
  - [ ] FFN residual
- [ ] Integrate mHC into deep/drafter variants:
  - [ ] `SwammaBlockDeep`
  - [ ] `SwammaDrafterBlock`
  - [ ] `SwammaDrafterBlockDeep`
- [ ] Add config flags:
  - [ ] `residual_mode = plain | mhc`
  - [ ] `mhc_mode = paper | proposal`
  - [ ] `mhc_streams`
  - [ ] `mhc_sinkhorn_iters`
  - [ ] `mhc_input_dependent`

## 3. Ablation Matrix (Required)

- [ ] Baseline: plain residuals (current SWAMMA).
- [ ] mHC paper-like: `n=4`, input-dependent enabled.
- [ ] mHC proposal-like: `n=2`, static coefficients.
- [ ] Partial integration ablations:
  - [ ] mHC on sublayer residual only
  - [ ] mHC on FFN residual only
  - [ ] mHC on both residuals
- [ ] Sinkhorn iteration sweep: `5 / 10 / 20`.

## 4. Metrics And Acceptance Gates

- [ ] Stability metrics:
  - [ ] no loss spikes/divergence
  - [ ] gradient norm sanity
  - [ ] NaN/Inf-free training
- [ ] Quality metrics:
  - [ ] NER F1 / token accuracy
  - [ ] RE task entity-relation F1
- [ ] Cost metrics:
  - [ ] throughput (tokens/s)
  - [ ] memory footprint
  - [ ] overhead vs baseline
- [ ] Accept rollout only if stability improves without unacceptable throughput loss.

## 5. Inference / Export Parity

- [ ] Ensure ONNX export path mirrors mHC residual behavior for selected mode.
- [ ] Add parity tests (Julia vs exported runtime) for one forward pass per mode.

## 6. Cleanup Completion Criteria

- [ ] `rg -n "\bSwamma\b|\bSwamma[A-Za-z_]+" src scripts test` returns zero code-level hits (except migration notes if intentionally retained).
- [ ] `rg -n "OscillatorLayer|OscillatorNorm" src` returns only intentional internal storage names or is fully renamed.
- [ ] All tests and smoke scripts pass under `Swamma*` imports only.

## 7. Zygote AD Tape Memory Fix — Cross-Architecture Refactor

Zygote stores every intermediate array in the forward pass for backprop. FFT-based WavePDE
leapfrog loops create ~40+ intermediates per block that stay alive until backward finishes.
A 6.7M param ReasoningDrafter consumed 80GB; a 1B Swamma RE model likely has the same issue.

Pattern: `ignore_derivatives` + straight-through residual for blocks, custom `rrule` for PDE loop.

### Already Done

- [x] `src/RuleConditionedWavePDE.jl` — PDE integration + modulation fully detached from AD tape
- [x] `src/ReasoningDrafter.jl` — block forward wrapped with straight-through estimator
- [x] `scripts/train_chess_reasoning.jl` — backbone runs outside `withgradient`, only heads traced

### Remaining

- [ ] **`src/WavePDE.jl`** — main Swamma backbone WavePDE has same leapfrog loop, same tape blowup
  - Wrap integration loop + softplus param computation in `ignore_derivatives`
  - Test with `scripts/train_re_gpu.jl` — verify memory drops
- [ ] **`src/Swamma.jl` SwammaBlock** — apply straight-through block wrapper
  - Contains: LinearAttention + WavePDE gate + SWAttention + FFN
  - Extract `_block_forward()`, wrap in `ignore_derivatives`, straight-through residual
- [ ] **`src/Drafter.jl`** — existing SwammaDrafter (non-reasoning variant)
  - Profile memory; apply same pattern if excessive
- [ ] **`src/LLaDA.jl`** — uses SwammaBlocks internally; inherits fix once SwammaBlock is wrapped
  - Verify no extra tape blowup from masking/denoising loop
- [ ] **`src/MoET.jl`** — MoE routing is cheap but per-expert block forward is the concern
  - Apply block wrapper to expert blocks
- [ ] **Custom `rrule` for leapfrog PDE loop** — proper long-term fix
  - Adjoint-mode PDE integration (reverse leapfrog) instead of storing all intermediates
  - O(1) memory per integration step with full gradient signal
  - Unblocks Phase 3 fine-tuning where backbone gradients matter
- [ ] **Monitor Enzyme.jl + CUDA + Lux maturity**
  - Track: EnzymeAD/Enzyme.jl #1392, #2244, #2283
  - Once stable, replaces all manual `ignore_derivatives` wrappers
  - Lux plans to switch from Zygote to Enzyme as default backend
