# CI Policy

This document defines how test lanes are expected to run in CI and how to
configure branch protections around them.

## Lanes

- default
  - command: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - scope: fast sanity checks (`attention`, `router`, `llada`)
  - expected usage: required on pull requests and pushes
- medium
  - command: `SWAMMA_TEST_MEDIUM=1 julia --project=. test/runtests.jl`
  - scope: default + relation extraction suite
  - expected usage: nightly scheduled checks
- full
  - command: `SWAMMA_TEST_FULL=1 julia --project=. test/runtests.jl`
  - scope: medium + `moet` + `tidar`
  - expected usage: manual pre-release verification

## Reasoning Pipeline Scope

The reasoning drafter pipeline is intentionally **not** part of the GitHub
Actions test lanes at the moment.

Reasons:

- bounded-medium and full reasoning runs are GPU-oriented and tuned for the
  Spark GB10 host, not for generic GitHub-hosted runners
- Phase 3b depends on Granite teacher execution and local model-cache behavior
- the goal of CI is fast merge protection, while reasoning pipeline validation
  is a heavier systems check

Current operational split:

- CI lanes:
  - `default`, `medium`, `full`
  - focus on repository-wide correctness and fast regression detection
- local reasoning validation:
  - `./scripts/launch_reasoning_pipeline.sh --smoke`
  - `./scripts/launch_reasoning_pipeline.sh --bounded-medium`
  - phase-specific bounded-medium runs for `3a` and `3b`

Treat bounded-medium reasoning runs as manual/local validation, not as branch
protection gates.

## GitHub Actions Mapping

Workflow file: `.github/workflows/test-lanes.yml`

- `default-fast` job
  - trigger: `pull_request`, `push` to `main`
  - lane: default
- `medium-nightly` job
  - trigger: `schedule` (`0 3 * * *`)
  - lane: medium
- `lane-dispatch` job
  - trigger: `workflow_dispatch`
  - lane: user-selected `default`, `medium`, or `full`

## Branch Protection Recommendation

For the main development branch:

- Require status check: `test-lanes / default-fast`.
- Do not require nightly/manual jobs (`medium-nightly`, `lane-dispatch`) for merge.
- Keep `default-fast` as the single hard gate to avoid blocking merges on long suites.

For release branches/tags:

- Run manual dispatch with `lane=full` before cut.
- Treat full-lane failures as release blockers.

## Local Parity Commands

Use these commands locally before opening PRs or triggering release jobs:

```bash
# Fast lane
julia --project=. test/runtests.jl

# Medium lane
SWAMMA_TEST_MEDIUM=1 julia --project=. test/runtests.jl

# Full lane
SWAMMA_TEST_FULL=1 julia --project=. test/runtests.jl
```

For reasoning-pipeline validation on the Spark host:

```bash
# Minimal reasoning sanity
./scripts/launch_reasoning_pipeline.sh --smoke

# Recommended day-to-day reasoning validation
./scripts/launch_reasoning_pipeline.sh --bounded-medium

# Phase-specific bounded-medium validation
./scripts/launch_reasoning_pipeline.sh --phase 3a --bounded-medium
./scripts/launch_reasoning_pipeline.sh --phase 3b --bounded-medium
```
