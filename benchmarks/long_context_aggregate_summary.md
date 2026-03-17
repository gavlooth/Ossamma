# Long-Context Multi-Seed Aggregate

Generated: 2026-03-14T23:52:54.155 UTC

## Inputs
- benchmark CSVs:
  - `benchmarks/long_context_benchmark.csv`
- eval CSVs:
  - `benchmarks/long_context_eval_full64.csv`

## Scaling Exponents (From Aggregated Means)
- Swamma (`swamma`): 1.0339
- Transformer (`transformer`): 1.3404

## Throughput + Latency Ratio
| Context | Swamma tok/s (mean ± std) | Transformer tok/s (mean ± std) | Latency ratio (Transformer / Swamma) |
|---:|---:|---:|---:|
| 1024 | 5580.13 ± 0.00 | 4584.63 ± 0.00 | 1.217 |
| 2048 | 5547.78 ± 0.00 | 6001.60 ± 0.00 | 0.924 |
| 4096 | 5519.84 ± 0.00 | 3773.39 ± 0.00 | 1.463 |
| 8192 | 5129.79 ± 0.00 | 3265.55 ± 0.00 | 1.571 |
| 16384 | 5160.14 ± 0.00 | 1910.58 ± 0.00 | 2.701 |

## Needle Accuracy
| Context | Swamma needle_acc (mean ± std) | Transformer needle_acc (mean ± std) | Delta (Swamma - Transformer) |
|---:|---:|---:|---:|
| 1024 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| 2048 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| 4096 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| 8192 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| 16384 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
