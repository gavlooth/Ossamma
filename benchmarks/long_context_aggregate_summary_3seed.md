# Long-Context Multi-Seed Aggregate

Generated: 2026-03-15T00:36:13.561 UTC

## Inputs
- benchmark CSVs:
  - `benchmarks/long_context_benchmark_seed42.csv`
  - `benchmarks/long_context_benchmark_seed7.csv`
  - `benchmarks/long_context_benchmark_seed19.csv`
- eval CSVs:
  - `benchmarks/long_context_eval_seed42.csv`
  - `benchmarks/long_context_eval_seed7.csv`
  - `benchmarks/long_context_eval_seed19.csv`

## Scaling Exponents (From Aggregated Means)
- Swamma (`swamma`): 1.0358
- Transformer (`transformer`): 1.4107

## Throughput + Latency Ratio
| Context | Swamma tok/s (mean ± std) | Transformer tok/s (mean ± std) | Latency ratio (Transformer / Swamma) |
|---:|---:|---:|---:|
| 1024 | 5629.76 ± 47.82 | 6240.08 ± 1352.69 | 0.934 |
| 2048 | 5581.50 ± 47.26 | 5653.69 ± 653.79 | 0.996 |
| 4096 | 5555.79 ± 23.04 | 4056.82 ± 65.09 | 1.370 |
| 8192 | 5173.09 ± 38.91 | 3269.29 ± 37.49 | 1.582 |
| 16384 | 5164.41 ± 18.57 | 1901.36 ± 16.45 | 2.716 |

## Needle Accuracy
| Context | Swamma needle_acc (mean ± std) | Transformer needle_acc (mean ± std) | Delta (Swamma - Transformer) |
|---:|---:|---:|---:|
| 1024 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| 2048 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| 4096 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| 8192 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| 16384 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
