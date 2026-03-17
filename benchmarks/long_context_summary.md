# Long-Context Summary

Generated: 2026-03-14T23:44:00.423 UTC

## Inputs
- benchmark: `benchmarks/long_context_benchmark.csv`
- eval: `benchmarks/long_context_eval_full64.csv`
- swamma arch label: `swamma`
- transformer arch label: `transformer`

## Scaling Exponents
- Swamma log-log time exponent: 1.0339
- Transformer log-log time exponent: 1.3404

## Max Finite Benchmark Context
- Swamma: 16384
- Transformer: 16384

## Benchmark Table
| Context | Swamma tok/s | Transformer tok/s | Latency ratio (Transformer / Swamma) |
|---:|---:|---:|---:|
| 1024 | 5580.13 | 4584.63 | 1.217 |
| 2048 | 5547.78 | 6001.60 | 0.924 |
| 4096 | 5519.84 | 3773.39 | 1.463 |
| 8192 | 5129.79 | 3265.55 | 1.571 |
| 16384 | 5160.14 | 1910.58 | 2.701 |

## Needle Accuracy Table
| Context | Swamma needle_acc | Transformer needle_acc | Delta (Swamma - Transformer) |
|---:|---:|---:|---:|
| 1024 | 0.0000 | 0.0000 | 0.0000 |
| 2048 | 0.0000 | 0.0000 | 0.0000 |
| 4096 | 0.0000 | 0.0000 | 0.0000 |
| 8192 | 0.0000 | 0.0000 | 0.0000 |
| 16384 | 0.0000 | 0.0000 | 0.0000 |
