# CLAUDE.md

This file gives Claude Code repository-specific guidance for working in `Swamma`.

## Keep In Mind

- Canonical naming in this repo: `SWAMMA` / Wave-PDE-first
- Main module is `src/Swamma.jl`.
- This repository is no longer the older `Samba2` / `attention.jl` / `ossm.jl` layout. Do not reference those names or paths in edits.
- The repo is not NER-only. Active surfaces include `SwammaNER`, relation extraction, `LLaDA`, and related research/training utilities.
- Treat `scripts/train_re_gpu.jl` as the active relation-extraction control surface for training, eval, and sweep workflows.
- Keep terminology aligned with `README.md` and current source names.
- Do not hardcode a single "current training target" unless the user explicitly asks for one.

## Working Baseline

```bash
julia --project=.
```

Do not assume every task should run the full suite. Match the lane to the files touched.

```bash
# Default lane
julia --project=. test/runtests.jl

# Medium lane
SWAMMA_TEST_MEDIUM=1 julia --project=. test/runtests.jl

# Full lane
SWAMMA_TEST_FULL=1 julia --project=. test/runtests.jl
```

Prefer the specific config or checkpoint already promoted by nearby docs, scripts, or experiments rather than inventing a new "default" in documentation.

## CUDA/GPU Training Rules

- **NEVER** use `try/catch` inside GPU training loops. Julia's GC keeps CUDA allocations alive across try/catch frames, causing silent OOM kills (CUDA.jl #2197).
- **NEVER** use `@info "msg $var"` (with interpolation) in training loops — it introduces an implicit try/catch. Use `println()` instead.
- After `Zygote.withgradient`, set `grads = nothing` to help GC free the AD tape.
- Use `GC.gc(true); CUDA.reclaim()` before each gradient pass for large models.
- Target hardware: NVIDIA Spark GB10, 130GB unified memory, CUDA 13.0, aarch64.

## Mandatory Session Report

Before ending any coding session in this repository, append a dated entry to `docs/SESSION_REPORT.md` summarizing:

- objectives attempted
- code or config changes made
- commands run and key outcomes
- best current recommendation
- unresolved issues and next actions

If no code changed, still add a short inspection-only entry.
