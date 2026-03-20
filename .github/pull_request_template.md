# Summary

Briefly describe what changed and why.

# Validation

- [ ] I ran the appropriate local test lane:
  - default: `julia --project=. test/runtests.jl`
  - medium: `SWAMMA_TEST_MEDIUM=1 julia --project=. test/runtests.jl`
  - full: `SWAMMA_TEST_FULL=1 julia --project=. test/runtests.jl`
- [ ] If this change touches the reasoning drafter pipeline, I ran the practical bounded validation path:
  - `./scripts/launch_reasoning_pipeline.sh --bounded-medium`
  - or a justified phase-specific equivalent:
    - `./scripts/launch_reasoning_pipeline.sh --phase 3a --bounded-medium`
    - `./scripts/launch_reasoning_pipeline.sh --phase 3b --bounded-medium`
- [ ] I updated `docs/SESSION_REPORT.md` for this coding session.

# Notes

- Mention any skipped validation, environment constraints, or follow-up work.
