# Parallelization Strategy for Swamma

This note reflects the current `WavePDE`-based architecture.

## Current Reality

The old parallel-scan story applied to the removed legacy oscillator layers. That is no longer the active path in this repo.

The current `SwammaBlock` uses:

- `LinearAttention` for the global content path
- `WavePDELayer` for the structured gate path
- `SWAttention` for the local path

## What Is Already Parallel

`WavePDELayer` computes its spatial Laplacian spectrally with FFTs, so the expensive spatial operator is already parallel over the state dimension.

That means the main parallelism levers now are:

- batch parallelism
- data parallel training across GPUs
- sequence chunking for memory control
- mixed precision (`bf16`)
- activation checkpointing at the model or training-loop level

## What Is Still Sequential

The time integration inside `WavePDELayer` still advances one sequence step at a time. That is the remaining recurrence in the gate path.

In practice, the cost profile is better than the removed legacy oscillator path because:

- the spatial operator is FFT-based
- the gate is only one branch of the block
- the rest of the block remains matrix-heavy and GPU-friendly

## Recommendation

For the current codebase, prioritize:

1. multi-GPU data parallelism
2. `bf16` training
3. activation checkpointing
4. deep-but-narrow encoder ablations
5. profiling the `WavePDE` gate before designing any new recurrence-specific optimization
