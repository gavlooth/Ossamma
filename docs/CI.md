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
