# Session Report

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
