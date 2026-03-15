# Long-Context Evaluation Protocol (Swamma vs Transformer)

This protocol tests two claims:

1. Swamma is competitive or better on quality.
2. Swamma scales better with longer context windows.

## Fairness Rules

Use the same for both models:

- tokenizer
- training data mixture
- number of train tokens
- optimizer and LR schedule
- parameter budget (or active parameter budget)
- evaluation datasets and seeds

Do not claim an architecture-only win if objectives differ (e.g., AR vs diffusion).

## Scripts

- Complexity/system benchmark: `scripts/benchmark_long_context.jl`
- Quality/context evaluation: `scripts/eval_long_context.jl`
- Result summarizer: `scripts/summarize_long_context_results.jl`
- Multi-seed aggregator: `scripts/aggregate_long_context_seeds.jl`
- Shared model builders: `scripts/long_context_models.jl`

## Configs

- `configs/swamma_vs_transformer/benchmark_long_context.toml`
- `configs/swamma_vs_transformer/eval_long_context.toml`

## Metrics

## Complexity Track

Reported per context length and architecture:

- forward latency (ms)
- forward throughput (tokens/s)
- parameter count
- fitted log-log runtime exponent over context lengths

Outputs CSV:

- `benchmarks/long_context_benchmark.csv`

## Quality Track

Reported per context length and architecture:

- masked reconstruction loss / ppl / accuracy
- positional masked accuracy (`early`, `middle`, `late` thirds)
- synthetic long-range needle retrieval accuracy (`needle_acc`)

Outputs CSV:

- `benchmarks/long_context_eval.csv`

## Run Commands

### 1) Complexity sweep

```bash
julia --project=. scripts/benchmark_long_context.jl \
  --config configs/swamma_vs_transformer/benchmark_long_context.toml \
  --output benchmarks/long_context_benchmark.csv
```

### 2) Quality sweep (synthetic needle only)

```bash
julia --project=. scripts/eval_long_context.jl \
  --config configs/swamma_vs_transformer/eval_long_context.toml \
  --output benchmarks/long_context_eval.csv
```

### 3) Quality sweep with checkpoints

```bash
julia --project=. scripts/eval_long_context.jl \
  --config configs/swamma_vs_transformer/eval_long_context.toml \
  --swamma-checkpoint checkpoints/llada_canonical/best.jls \
  --transformer-checkpoint checkpoints/transformer_baseline/best.jls \
  --output benchmarks/long_context_eval.csv
```

### 4) Quality sweep with text corpus

Edit `configs/swamma_vs_transformer/eval_long_context.toml`:

- set `run_text_eval = true`
- set `text_path = "..."`

Then run the command from step (2) or (3).

### 5) Summarize benchmark + eval CSVs

```bash
julia --project=. scripts/summarize_long_context_results.jl \
  --benchmark-csv benchmarks/long_context_benchmark.csv \
  --eval-csv benchmarks/long_context_eval_full64.csv \
  --output-md benchmarks/long_context_summary.md
```

### 6) Aggregate multiple seeds (mean/std)

Provide comma-separated CSV paths from different seeds:

```bash
julia --project=. scripts/aggregate_long_context_seeds.jl \
  --benchmark-csvs benchmarks/seed1_benchmark.csv,benchmarks/seed2_benchmark.csv,benchmarks/seed3_benchmark.csv \
  --eval-csvs benchmarks/seed1_eval.csv,benchmarks/seed2_eval.csv,benchmarks/seed3_eval.csv \
  --output-benchmark-csv benchmarks/long_context_benchmark_agg.csv \
  --output-eval-csv benchmarks/long_context_eval_agg.csv \
  --output-md benchmarks/long_context_aggregate_summary.md
```

### 7) One-command seed sweep (run + aggregate)

```bash
julia --project=. scripts/run_long_context_seed_sweep.jl \
  --seeds 42,7,19 \
  --device gpu \
  --benchmark-config configs/swamma_vs_transformer/benchmark_long_context.toml \
  --eval-config configs/swamma_vs_transformer/eval_long_context_quick.toml \
  --skip-existing
```

Optional checkpointed eval:

```bash
julia --project=. scripts/run_long_context_seed_sweep.jl \
  --seeds 42,7,19 \
  --device gpu \
  --eval-config configs/swamma_vs_transformer/eval_long_context_quick.toml \
  --swamma-checkpoint checkpoints/llada_canonical/checkpoint_best.jls \
  --transformer-checkpoint checkpoints/transformer_baseline/checkpoint_best.jls \
  --skip-existing
```

## Baseline Decision Criteria (example)

Treat this as a pass only if all hold at target context (e.g., 128k):

1. Swamma text quality >= Transformer text quality - 1%.
2. Swamma needle accuracy >= Transformer needle accuracy - 1%.
3. Swamma forward latency <= 0.65x Transformer latency.
4. Swamma log-log runtime exponent < Transformer exponent.

## Notes

- If checkpoint load fails for a context size, the eval script falls back to random init for that row and records the checkpoint error in CSV.
- Synthetic needle metric is architecture-stress only; it does not replace real long-context downstream tasks.
- For publishable claims, run at least 3 seeds and report mean ± std.
