# RE Experiment Log — 2026-03-12

This note records the RE architecture and config experiments run during the March 12, 2026 debugging session, along with the current conclusions.

## Baseline Context

- Task: relation extraction on the current REDFM/REBEL English pipeline.
- Backbone reference: plain `SwammaBlock` encoder.
- Main diagnosis outcome so far:
  - runtime stability is no longer the main issue
  - the dominant bottleneck is downstream of the encoder
  - proposal quality improved in several ablations, but relation scoring and pair selection remained weak

## Long Safe Run

- Config lineage: `redfm_base_safe`
- Best observed validation during long resumed training:
  - step `2500`: `val_loss 12.6371`
  - step `6000`: `val_loss 13.7025`
- Later behavior:
  - quality collapse started around step `13956`
  - step `14000`: `val_loss 29.4208`
  - recovery later reached step `16500`: `val_loss 18.7277`
- Conclusion:
  - training/runtime path is stable
  - best quality was early
  - later degradation looks like optimization/generalization failure, not a CUDA/runtime fault

## Diagnostic Checkpoint Sweep

Sampled 8-batch evaluation over selected baseline-safe checkpoints:

| Step | Total | Entity | Boundary | Relation | Conf | Span Recall | Pair Recall | Rel F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | `18.6281` | `4.2264` | `0.2685` | `13.6848` | `0.4484` | `0.1519` | `0.0244` | `0.0000` |
| 6000 | `14.1207` | `4.2566` | `0.2405` | `8.7086` | `0.9150` | `0.2407` | `0.0244` | `0.0000` |
| 13000 | `15.5336` | `4.1533` | `0.3253` | `9.3085` | `1.7466` | `0.3926` | `0.0488` | `0.0000` |
| 14000 | `29.3103` | `3.8500` | `0.3734` | `24.4379` | `0.6491` | `0.0344` | `0.0000` | `0.0000` |
| 16000 | `21.2691` | `3.6014` | `0.3594` | `16.8589` | `0.4495` | `0.0745` | `0.0000` | `0.0000` |
| 21000 | `36.2899` | `3.5122` | `0.3578` | `32.0295` | `0.3903` | `0.0946` | `0.0000` | `0.0000` |

Conclusion:

- the backbone is not the first failure
- relation loss is the dominant failure term in bad checkpoints
- pair recall is the hardest bottleneck
- end-to-end relation F1 stayed at zero on the sampled proposal-based evaluation

## Short-Run Baselines

### Plain Safe Baseline

| Step | Val loss | Entity | Boundary | Relation | Conf | Span recall | Pair recall | Rel F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | `17.1478` | `3.9367` | `0.2951` | `12.4686` | `0.4474` | `0.0831` | `0.0122` | `0.0000` |
| 2000 | `18.3441` | `4.2264` | `0.2685` | `13.3639` | `0.4854` | `0.1519` | `0.0244` | `0.0000` |

Conclusion:

- plain `SwammaBlock` remains the most robust encoder reference
- proposal recall is weak, but relation loss is still much better than most modified encoder variants

### Plain Safe Baseline With SwammaBlock OutputProjection

Config: `redfm_base_safe_outproj`

| Step | Val loss | Entity | Boundary | Relation | Conf | Span recall | Pair recall | Rel F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | `19.2257` | `2.7024` | `0.3188` | `15.6563` | `0.5481` | `0.0344` | `0.0000` | `0.0000` |
| 500 | `25.4019` | `2.8239` | `0.3059` | `21.6641` | `0.6080` | `0.2063` | `0.0000` | `0.0000` |
| 750 | `21.9994` | `2.2638` | `0.2873` | `18.9392` | `0.5092` | `0.2751` | `0.0000` | `0.0000` |
| 1000 | `24.3749` | `2.1782` | `0.2691` | `21.4251` | `0.5026` | `0.3897` | `0.0122` | `0.0000` |

Conclusion:

- adding a post-mix output projection to the main `SwammaBlock` improved proposal-side behavior
- it hurt relation loss enough to make the total result clearly worse
- this should remain an ablation flag, not the default encoder

## Candidate / Head Ablations

### Ergonomic Candidate Expansion

Config: `redfm_base_ergonomic`

- changes:
  - `window_size = 64`
  - `max_candidate_spans = 96`
  - `max_candidate_pairs = 256`
  - `pair_neighbor_radius = 8`

| Step | Val loss | Relation | Span recall | Pair recall | Rel F1 |
|---:|---:|---:|---:|---:|---:|
| 1000 | `19.2377` | `14.7839` | `0.2006` | `0.0488` | `0.0000` |
| 2000 | `21.1941` | `16.2317` | `0.1805` | `0.0000` | `0.0000` |

Conclusion:

- candidate coverage improved
- wider attention window made the relation side worse
- this was not a keeper config

### Candidate-Only Expansion

Config: `redfm_base_candidate_only`

- changes:
  - baseline `window_size = 24`
  - `max_candidate_spans = 96`
  - `max_candidate_pairs = 256`
  - `pair_neighbor_radius = 8`

| Step | Val loss | Relation | Span recall | Pair recall | Rel F1 |
|---:|---:|---:|---:|---:|---:|
| 1000 | `12.6606` | `8.7076` | `0.2264` | `0.0366` | `0.0000` |
| 2000 | `16.6614` | `12.8076` | `0.2722` | `0.0488` | `0.0000` |

Conclusion:

- this was the best-balanced nontrivial ablation
- more candidate coverage helped
- keeping the small local window was important
- this was the best branch to continue from for downstream head work under the earlier coarse evaluator

### Candidate-Only Rerun Under Rich Pair Diagnostics

Config: `redfm_base_candidate_only`

This rerun was done after adding pair-ranking diagnostics so the branch could be compared under the same richer evaluator as the later experiments.

| Step | Val loss | Entity | Boundary | Relation | Conf | Span recall | Pair recall | Pair@16 | Mean pair rank | Miss short | Miss medium | Miss long | Rel F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | `34.4323` | `3.0887` | `0.3317` | `30.5174` | `0.4946` | `0.2264` | `0.0122` | `0.0000` | `112.0` | `0.4125` | `0.3375` | `0.2500` | `0.0000` |
| 500 | `43.5031` | `3.3234` | `0.3075` | `39.3389` | `0.5333` | `0.3037` | `0.0366` | `0.0000` | `151.3` | `0.4125` | `0.3375` | `0.2500` | `0.0000` |
| 1000 | `28.6945` | `3.1544` | `0.2758` | `24.8103` | `0.4540` | `0.1862` | `0.0244` | `0.0000` | `130.0` | `0.4125` | `0.3375` | `0.2500` | `0.0000` |

Conclusion:

- the richer evaluator confirms the pair-proposal bottleneck directly
- gold pairs are almost never surfacing in the top `16` retained pairs
- even when a gold pair is retained, it lands very late in the proposal list
- missed pairs are distributed across short, medium, and long distances, so this is not mainly a long-context failure
- this rerun was much worse than the earlier `candidate_only` short run, which means short-horizon comparisons have significant variance and should not be treated as settled from a single run

### Plain Baseline Rerun Under Rich Pair Diagnostics

Config: `redfm_base_safe`

This rerun was used as the fresh apples-to-apples reference against the richer `candidate_only` rerun above.

| Step | Val loss | Entity | Boundary | Relation | Conf | Span recall | Pair recall | Pair@16 | Mean pair rank | Miss short | Miss medium | Miss long | Rel F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | `13.7606` | `3.6066` | `0.3236` | `9.3060` | `0.5243` | `0.2206` | `0.0000` | `0.0000` | `NaN` | `0.4125` | `0.3500` | `0.2375` | `0.0000` |
| 1000 | `28.1326` | `3.5883` | `0.2689` | `23.7951` | `0.4803` | `0.2980` | `0.0244` | `0.0122` | `11.0` | `0.4250` | `0.3250` | `0.2500` | `0.0000` |

Conclusion:

- the plain baseline is still bad on end-to-end RE at this horizon
- but in the fresh apples-to-apples reruns it beat `candidate_only` on the key relation-side metrics at step `1000`
- both runs had the same pair recall by step `1000`, but the baseline ranked matched gold pairs far earlier:
  - baseline: `pair_rank = 11.0`
  - candidate_only rerun: `pair_rank = 130.0`
- this makes pair ranking look more important than raw candidate-budget expansion

### Candidate-Only + Wider Pair Neighborhood

Config: `redfm_base_candidate_radius12`

| Step | Val loss | Entity | Boundary | Relation | Conf | Span recall | Pair recall | Rel F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | `20.3401` | `3.0964` | `0.3173` | `16.4013` | `0.5251` | `0.0573` | `0.0000` | `0.0000` |
| 500 | `24.3393` | `2.7612` | `0.2960` | `20.7758` | `0.5063` | `0.2407` | `0.0244` | `0.0000` |
| 750 | `25.6076` | `3.1024` | `0.2922` | `21.6682` | `0.5448` | `0.0860` | `0.0000` | `0.0000` |
| 1000 | `21.0508` | `3.5346` | `0.2773` | `16.7679` | `0.4711` | `0.2264` | `0.0366` | `0.0000` |

Conclusion:

- `pair_neighbor_radius = 12` did not improve on `candidate_only`
- pair recall ended up matching the original `candidate_only` run by step `1000`, not beating it
- relation loss and total val loss remained clearly worse than `candidate_only`
- this should not replace the current best branch
- comparable 8-batch checkpoint reevaluation at step `1000`: `total 21.7493`, `entity 3.5346`, `boundary 0.2773`, `relation 17.4641`, `confidence 0.4734`, `span_recall 0.2264`, `pair_recall 0.0366`

## Refinement / Interleaving Experiments

### Candidate + 2 Post-Encoder Local-Wave Refinement Blocks

Config: `redfm_base_candidate_refine2`

| Step | Val loss | Relation | Span recall | Pair recall | Rel F1 |
|---:|---:|---:|---:|---:|---:|
| 250 | `25.1258` | n/a | `0.3954` | `0.0000` | `0.0000` |
| 500 | `26.4951` | n/a | `0.4842` | `0.0366` | `0.0000` |
| 750 | `27.3837` | n/a | `0.5415` | `0.0000` | `0.0000` |
| 1000 | `31.1751` | `25.8002` | `0.5559` | `0.0000` | `0.0000` |

Conclusion:

- very strong span refiner
- harmful for downstream relation scoring and pair selection
- not a productive direction in the tested placement

### Interleaved Mixed Local+Wave With mHC

Config: `redfm_base_interleaved_mhc`

| Step | Val loss | Relation | Span recall | Pair recall | Rel F1 |
|---:|---:|---:|---:|---:|---:|
| 250 | `24.1939` | `19.8525` | `0.1748` | `0.0366` | `0.0000` |
| 500 | `20.2577` | `15.5713` | `0.3467` | `0.0244` | `0.0000` |
| 750 | `26.2623` | `22.0425` | `0.5645` | `0.0244` | `0.0000` |
| 1000 | `23.3296` | `18.8123` | `0.5759` | `0.0488` | `0.0000` |

Conclusion:

- strong span improvement
- relation side still much worse than baseline/candidate-only
- extra encoder structure did not solve the real bottleneck

### Interleaved Wave-Only With mHC

Config: `redfm_base_interleaved_waveonly_mhc`

| Step | Val loss | Relation | Span recall | Pair recall | Rel F1 |
|---:|---:|---:|---:|---:|---:|
| 250 | `22.9435` | `17.4807` | `0.1261` | `0.0244` | `0.0003` |
| 500 | `24.7011` | `19.8470` | `0.1318` | `0.0244` | `0.0000` |
| 750 | `28.2281` | `22.7301` | `0.1175` | `0.0244` | `0.0000` |
| 1000 | `30.9893` | `26.2163` | `0.3295` | `0.0610` | `0.0000` |

Conclusion:

- removing local attention did not fix the interleaved design
- wave-only was worse overall

### Interleaved Local-Only With mHC

Config: `redfm_base_interleaved_localonly_mhc`

| Step | Val loss | Relation | Span recall | Pair recall | Rel F1 |
|---:|---:|---:|---:|---:|---:|
| 1000 | `15.9325` | `11.9566` | `0.6046` | `0.0244` | `0.0000` |

Conclusion:

- best interleaved variant
- still worse than `candidate_only` on the main objective
- suggests local refinement helps spans but not enough for end-to-end RE

### Interleaved Linear+Window With mHC

Config: `redfm_base_interleaved_linearwindow_mhc`

| Step | Val loss | Relation | Span recall | Pair recall | Rel F1 |
|---:|---:|---:|---:|---:|---:|
| 1000 | `40.3644` | `35.9935` | `0.5817` | `0.0488` | `0.0000` |

Conclusion:

- replacing wave with linear attention inside the interleaved blocks was very bad without a post-mix projection

### Interleaved Linear+Window With OutputProjection

Config: `redfm_base_interleaved_linearwindow_outproj_mhc`

| Step | Val loss | Relation | Span recall | Pair recall | Rel F1 |
|---:|---:|---:|---:|---:|---:|
| 250 | `27.5957` | `22.8426` | `0.3295` | `0.0122` | `0.0000` |
| 500 | `26.4403` | `21.1456` | `0.4613` | `0.0244` | `0.0000` |
| 750 | `31.8158` | `26.7134` | `0.5501` | `0.0244` | `0.0000` |
| 1000 | `32.8128` | `27.5121` | `0.4499` | `0.0366` | `0.0000` |

Conclusion:

- the post-mix output projection improved this family materially
- even improved, it remained much worse than the simpler baseline paths

## Current Conclusions

1. Keep the encoder as plain `SwammaBlock`.
2. Keep the main block output projection disabled by default.
3. Stop spending cycles on encoder surgery for now.
4. The best architecture/config direction found so far is still the plain `SwammaBlock` family, but with a better pair proposer policy rather than more encoder changes.
5. The dominant remaining bottleneck is relation-side:
   - pair proposal remains weak
   - relation loss dominates bad checkpoints
   - relation F1 remains at zero on the current sampled proposal-based eval
6. New pair diagnostics indicate that the problem is not just long-range context:
   - fresh step-`1000` reevaluations showed `pair_top16_recall = 0.0122` for both `candidate_radius12` and `candidate_rank128`
   - mean matched-pair rank was late in the proposal list:
     - `candidate_radius12`: `18.0`
     - `candidate_rank128`: `43.5`
   - missed-pair distance shares were spread across buckets rather than dominated by long pairs:
     - `candidate_radius12`: short `0.4304`, medium `0.3291`, long `0.2405`
     - `candidate_rank128`: short `0.4359`, medium `0.3333`, long `0.2308`
   - interpretation: the pair proposer is mostly a ranking/selection problem, not simply a lack of broad receptive field
7. The fresh `candidate_only` rerun reinforces the same ranking story:
   - step `1000`: `pair_recall = 0.0244`, `pair_top16_recall = 0.0000`, `matched_pair_rank_mean = 130.0`
   - interpretation: the branch may still be directionally useful, but the proposal stage is too unstable and poorly ranked to declare it a robust winner from one early run
8. The fresh baseline rerun sharpens the comparison:
   - step `1000`: `pair_recall = 0.0244`, `pair_top16_recall = 0.0122`, `matched_pair_rank_mean = 11.0`
   - interpretation: candidate expansion did not buy a better ranking story in the current apples-to-apples comparison; the main problem is how pairs are ranked, not only how many are generated
9. A direct proposer change is the first thing that moved retrieval without retraining:
   - hybrid local+global pair ranking improved the plain baseline step-`1000` checkpoint from `pair_recall 0.0244` to `0.0732`
   - this points to pair-proposal policy as a more promising lever than more encoder or head churn
10. A fresh short training rerun with `hybrid12` is the first clear overall win over the fresh plain baseline:
   - plain baseline rerun step-`1000`: `total 28.1326`, `relation 23.7951`
   - `hybrid12` rerun step-`1000`: `total 16.3240`, `relation 11.8049`
   - this makes pair-proposer policy the new primary optimization path

## Relation-Head Follow-Up

### Candidate-Only + Stronger Relation Head

Config: `redfm_base_candidate_rank128`

| Step | Val loss | Entity | Boundary | Relation | Conf | Span recall | Pair recall | Rel F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | `26.3068` | `3.2233` | `0.3335` | `22.2689` | `0.4811` | `0.3266` | `0.0000` | `0.0000` |
| 500 | `29.8759` | `3.3104` | `0.3064` | `25.7929` | `0.4662` | `0.4613` | `0.0366` | `0.0000` |
| 750 | `31.5591` | `2.9595` | `0.2831` | `27.7130` | `0.6035` | `0.4642` | `0.0366` | `0.0000` |
| 1000 | `30.8091` | `3.7629` | `0.2781` | `26.3317` | `0.4365` | `0.2120` | `0.0488` | `0.0000` |

Conclusion:

- `biaffine_rank = 128` was clearly worse than `candidate_only`
- the larger relation head increased optimization difficulty and relation loss
- pair recall eventually matched the better short-run candidates, but not enough to offset the quality drop
- this should not be carried forward as the next default

## Pair-Proposer Ranking Follow-Up

### Hybrid Local + Global Pair Ranking

Code change: [RelationExtraction.jl](/home/christos/code/julia/Swamma/src/RelationExtraction.jl)

Change:

- kept the local-neighbor candidate generation
- added a global pool of directed pairs among the top-scoring spans
- ranked the full candidate pool by explicit pair score before truncation

This was evaluated on existing step-`1000` checkpoints without retraining.

| Checkpoint | Total | Relation | Span recall | Pair recall | Pair@16 | Mean pair rank |
|---|---:|---:|---:|---:|---:|---:|
| `redfm_base_safe` old proposer | `28.1326` | `23.7951` | `0.2980` | `0.0244` | `0.0122` | `11.0` |
| `redfm_base_safe` hybrid proposer | `28.2293` | `23.8823` | `0.2980` | `0.0732` | `0.0366` | `36.7` |
| `redfm_base_candidate_only` old proposer | `28.6945` | `24.8103` | `0.1862` | `0.0244` | `0.0000` | `130.0` |
| `redfm_base_candidate_only` hybrid proposer | `28.8645` | `25.0402` | `0.1862` | `0.0122` | `0.0000` | `153.0` |

Conclusion:

- this is the first direct proposer change that produced a real retrieval gain on held-out data without encoder changes or retraining
- it helped the plain baseline checkpoint substantially:
  - `pair_recall` tripled from `0.0244` to `0.0732`
  - `pair_top16_recall` tripled from `0.0122` to `0.0366`
- it hurt the `candidate_only` checkpoint, which suggests the global top-span pair pool is only useful when span scores are relatively well behaved
- direct pair-proposal logic is now the highest-leverage part of the stack

### Baseline Pair-Proposer Mode Matrix

Checkpoint: fresh `redfm_base_safe` step `1000`

| Proposer config | Total | Relation | Pair recall | Pair@16 | Mean pair rank |
|---|---:|---:|---:|---:|---:|
| `local` | `23.2377` | `18.8964` | `0.0244` | `0.0122` | `22.0` |
| `global10` | `29.9535` | `25.5919` | `0.0732` | `0.0366` | `32.0` |
| `hybrid8` | `27.4449` | `23.0754` | `0.0732` | `0.0366` | `36.3` |
| `hybrid12` | `26.8755` | `22.5465` | `0.0854` | `0.0366` | `44.4` |

Conclusion:

- `global10` and the hybrid modes improve pair retrieval versus local-only
- `hybrid12` gave the best retrieval on the fixed baseline checkpoint
- `local` still gave the best loss on the old fixed checkpoint, so retrieval gain alone was not enough to choose the new default

### Fresh Short Run With `hybrid12`

Config: [redfm_base_safe_pair_hybrid12.toml](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_hybrid12.toml)

Fresh `1000`-step training run plus 8-batch checkpoint reevaluation:

| Run | Total | Entity | Boundary | Relation | Conf | Span recall | Pair recall | Pair@16 | Mean pair rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain baseline rerun | `28.1326` | `3.5883` | `0.2689` | `23.7951` | `0.4803` | `0.2980` | `0.0244` | `0.0122` | `11.0` |
| `hybrid12` rerun | `16.3240` | `3.7174` | `0.2846` | `11.8049` | `0.5171` | `0.2120` | `0.0244` | `0.0000` | `20.0` |

Conclusion:

- the trained `hybrid12` run is the first short rerun that clearly beats the fresh plain baseline on the main validation objective
- the improvement came mostly from much lower relation loss
- pair recall did not improve at this horizon, so the gain is not simply “more retrieved pairs”; it is the interaction between the proposer policy and the rest of the inference path
- this is now the strongest working direction found in the session

### Proposal-Conditioned Reevaluation Of Plain Baseline vs `hybrid12`

The evaluator was then extended to add proposal-conditioned relation and confidence losses. This was necessary because proposer policy does not affect the original teacher-forced gold-pair loss directly.

8-batch checkpoint reevaluation at step `1000`:

| Run | Total | Entity | Boundary | Relation | Conf | Proposal Rel | Proposal Conf | Proposal Total | Span Recall | Pair Recall | Pair@16 | Mean Pair Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain baseline rerun | `28.1230` | `3.5883` | `0.2689` | `23.8302` | `0.4357` | `4.8289` | `0.6816` | `9.3678` | `0.2980` | `0.0244` | `0.0122` | `22.0` |
| `hybrid12` rerun | `15.9524` | `3.7174` | `0.2846` | `11.3650` | `0.5854` | `3.9470` | `0.7200` | `8.6690` | `0.2120` | `0.0244` | `0.0000` | `20.0` |

Conclusion:

- the `hybrid12` win survives the corrected metric
- even on proposal-conditioned scoring, `hybrid12` lowers relation loss relative to the fresh plain baseline
- the gain still does not come from higher raw pair recall at step `1000`
- the effect looks more like better pair-set quality once proposed than simple retrieval expansion
- proposer policy is now the strongest confirmed lever, and `hybrid12` remains the current working baseline for that direction

### Neighboring Proposer Check: `hybrid8`

Config: [redfm_base_safe_pair_hybrid8.toml](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_hybrid8.toml)

Fresh `1000`-step training run plus 8-batch checkpoint reevaluation:

| Run | Total | Entity | Boundary | Relation | Conf | Proposal Rel | Proposal Conf | Proposal Total | Span Recall | Pair Recall | Pair@16 | Mean Pair Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hybrid12` rerun | `15.9524` | `3.7174` | `0.2846` | `11.3650` | `0.5854` | `3.9470` | `0.7200` | `8.6690` | `0.2120` | `0.0244` | `0.0000` | `20.0` |
| `hybrid8` rerun | `16.8117` | `3.7829` | `0.2903` | `12.2899` | `0.4485` | `4.8503` | `0.7181` | `9.6416` | `0.2493` | `0.0122` | `0.0000` | `25.0` |

Conclusion:

- `hybrid8` is a clean step backward from `hybrid12`
- the loss gap is driven mostly by worse proposal-conditioned relation loss
- raw pair recall is also worse for `hybrid8` at this horizon
- this is enough to stop nearby proposer churn and use `hybrid12` as the next long-run candidate

## Next Tests

These are the next tests to run in order:

1. stop encoder changes and keep plain `SwammaBlock` as the architecture reference
2. keep working on pair proposal ranking directly:
   - treat `hybrid12` as the current working proposer baseline
   - keep `hybrid12` as the current working proposer baseline
   - only reopen proposer tuning if the longer run underperforms
3. launch the longer run from [redfm_base_safe_pair_hybrid12.toml](/home/christos/code/julia/Swamma/configs/redfm_base_safe_pair_hybrid12.toml)
4. rerun baseline vs `candidate_only` again only if needed to check nondeterminism, not as the primary next step

Not recommended right now:

- more encoder interleaving
- more `WavePDE` ablations
- more local-window/refinement variants
- wider backbone
- deeper backbone
- enabling `SwammaBlock` output projection as the baseline default
- enabling `biaffine_rank = 128` as the current default
- enabling `pair_neighbor_radius = 12` as the current default
