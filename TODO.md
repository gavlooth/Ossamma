# TODO — SWAMMA Rollout + Legacy Removal

## 0. Naming And Legacy Purge (Breaking)

- [x] Rename top-level module from `Ossamma` to `Swamma`.
- [x] Rename primary file `src/Ossamma.jl` to `src/Swamma.jl` and update all `include` / `using` paths.
- [x] Rename public API symbols to `Swamma*` and remove `Ossamma*` exports.
- [x] Rename terminology in code/config/docs from `Oscillator*` to `WaveGate*` where it refers to WavePDE gating.
- [x] Remove legacy aliases, exports, and the removed `ossm.jl` code path from the package graph.
- [x] Update scripts/tests/docs to import and reference `Swamma*` APIs.
- [x] Remove stale docs/files that used the old `Ossamma` naming.

## 1. mHC Core Implementation (Paper-Faithful Path)

- [ ] Add `src/HyperConnect.jl` with:
  - [ ] Sinkhorn-Knopp projection to doubly-stochastic matrices
  - [ ] `ManifoldHyperConnection` layer
  - [ ] Non-negative normalized `H_pre` / `H_post`
  - [ ] Near-identity initialization
- [ ] Add optional input-dependent parameterization (`dynamic=true`) per paper.
- [ ] Add unit tests for mHC invariants:
  - [ ] row sums ~1, col sums ~1
  - [ ] non-negativity
  - [ ] stability under composition (`H_res` products)

## 2. mHC Integration Into SWAMMA Blocks

- [ ] Integrate mHC into `SwammaBlock` residual pathways:
  - [ ] sublayer residual
  - [ ] FFN residual
- [ ] Integrate mHC into deep/drafter variants:
  - [ ] `SwammaBlockDeep`
  - [ ] `SwammaDrafterBlock`
  - [ ] `SwammaDrafterBlockDeep`
- [ ] Add config flags:
  - [ ] `residual_mode = plain | mhc`
  - [ ] `mhc_mode = paper | proposal`
  - [ ] `mhc_streams`
  - [ ] `mhc_sinkhorn_iters`
  - [ ] `mhc_input_dependent`

## 3. Ablation Matrix (Required)

- [ ] Baseline: plain residuals (current SWAMMA).
- [ ] mHC paper-like: `n=4`, input-dependent enabled.
- [ ] mHC proposal-like: `n=2`, static coefficients.
- [ ] Partial integration ablations:
  - [ ] mHC on sublayer residual only
  - [ ] mHC on FFN residual only
  - [ ] mHC on both residuals
- [ ] Sinkhorn iteration sweep: `5 / 10 / 20`.

## 4. Metrics And Acceptance Gates

- [ ] Stability metrics:
  - [ ] no loss spikes/divergence
  - [ ] gradient norm sanity
  - [ ] NaN/Inf-free training
- [ ] Quality metrics:
  - [ ] NER F1 / token accuracy
  - [ ] RE task entity-relation F1
- [ ] Cost metrics:
  - [ ] throughput (tokens/s)
  - [ ] memory footprint
  - [ ] overhead vs baseline
- [ ] Accept rollout only if stability improves without unacceptable throughput loss.

## 5. Inference / Export Parity

- [ ] Ensure ONNX export path mirrors mHC residual behavior for selected mode.
- [ ] Add parity tests (Julia vs exported runtime) for one forward pass per mode.

## 6. Cleanup Completion Criteria

- [ ] `rg -n "\bSwamma\b|\bSwamma[A-Za-z_]+" src scripts test` returns zero code-level hits (except migration notes if intentionally retained).
- [ ] `rg -n "OscillatorLayer|OscillatorNorm" src` returns only intentional internal storage names or is fully renamed.
- [ ] All tests and smoke scripts pass under `Swamma*` imports only.
