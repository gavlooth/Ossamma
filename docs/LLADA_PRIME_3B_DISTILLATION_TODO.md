# LLaDA PRIME 3B Distillation Rollout TODO

Goal: train a meaningful `~3B` PRIME/LLaDA student on this machine using a suitable teacher, from planning through final checkpoint promotion.

Scope lock:
- Student architecture: [`configs/llada_prime_3b.toml`](/home/christos/code/julia/Swamma/configs/llada_prime_3b.toml)
- Student size: `2,953,666,880` parameters in the current implementation
- Primary training path: offline sequence distillation first
- Stretch target after success: [`configs/llada_prime_5b.toml`](/home/christos/code/julia/Swamma/configs/llada_prime_5b.toml)

Teacher recommendation:
- Primary teacher candidate: `Qwen/Qwen2.5-7B-Instruct`
- Fallback teacher candidate: `ibm-granite/granite-3.3-8b-instruct`
- Reason for lock: `7B-8B` is meaningfully stronger than the `~3B` student while still realistic for local or semi-local inference serving

Important execution rule:
- Do not block the first full run on token-logit distillation.
- Complete the first end-to-end rollout with offline teacher-text distillation.
- Treat online logit/KL distillation as a second-phase improvement only if the first sequence-distilled run is stable.

## 0. Already Done

- [x] Lock the first meaningful student target to [`configs/llada_prime_3b.toml`](/home/christos/code/julia/Swamma/configs/llada_prime_3b.toml)
- [x] Record the stretch student target in [`configs/llada_prime_5b.toml`](/home/christos/code/julia/Swamma/configs/llada_prime_5b.toml)
- [x] Verify the exact `~3B` student parameter count in the current implementation
- [x] Verify the canonical PRIME trainer exists in [`scripts/train_llada_canonical.jl`](/home/christos/code/julia/Swamma/scripts/train_llada_canonical.jl)
- [x] Verify the current canonical trainer does not yet implement teacher-aware distillation logic

## 1. Teacher Lock

- [x] Lock the primary teacher for the first full run
- [ ] Confirm the teacher license and usage terms are acceptable for the intended training data/output usage
- [ ] Lock the teacher runtime mode:
  - [ ] local quantized inference
  - [ ] remote API/server inference
- [x] Lock the fallback teacher in case throughput or quality is insufficient
- [ ] Record the teacher revision/hash used for reproducibility
- [x] Record the teacher decoding defaults for dataset generation:
  - [x] temperature
  - [x] top-p
  - [x] max-new-tokens
  - [x] stop sequences

## 2. Distillation Strategy Lock

- [x] Freeze the first rollout on offline sequence distillation
- [x] Explicitly defer online logit/KL distillation until after the first complete student run
- [x] Define what the teacher will generate:
  - [x] plain continuation corpus
  - [ ] instruction-response corpus
  - [ ] reasoning-heavy synthetic corpus
- [x] Freeze the first task mix for the teacher corpus
- [x] Decide whether the first run is:
  - [x] pure teacher-text training
  - [ ] mixed raw-corpus + teacher-text training
- [x] Lock the initial mixture ratio for the first full run

## 3. Data Sources And Prompt Set

- [x] Define the raw base corpus for the student
- [x] Define the prompt sources used to elicit distilled teacher outputs
- [ ] Split prompt sources into:
  - [ ] train
  - [ ] validation
  - [ ] final holdout review
- [x] Add a repo-tracked prompt manifest for the distillation run
- [x] Add a repo-tracked sampling manifest for teacher generation settings
- [ ] Ensure no prompt leakage between training and held-out review
- [ ] Ensure prompt mixture is not dominated by one narrow task family

## 4. Teacher-Corpus Generation Tooling

- [x] Add a teacher-corpus generator script for PRIME/LLaDA distillation
  - [x] [`scripts/generate_llada_teacher_corpus.py`](/home/christos/code/julia/Swamma/scripts/generate_llada_teacher_corpus.py)
- [x] Support the locked primary teacher in that generator
- [x] Support resume/retry without regenerating completed rows
- [x] Write outputs as JSONL with at least:
  - [x] `prompt`
  - [x] `teacher_text`
  - [x] `teacher_model`
  - [x] `teacher_revision`
  - [x] `generation_config`
  - [x] `source_split`
  - [x] `source_task`
- [x] Add deterministic sharding so large generation jobs can be resumed cleanly
- [x] Add failure logging for timeouts, empty outputs, and malformed rows
- [x] Add optional prompt hashing to detect duplicates

## 5. Teacher-Corpus Validation

- [x] Add a validator for the generated teacher corpus
  - [x] [`scripts/validate_llada_teacher_corpus.py`](/home/christos/code/julia/Swamma/scripts/validate_llada_teacher_corpus.py)
- [x] Reject empty outputs
- [x] Reject repeated boilerplate rows
- [x] Reject rows below a minimum character/token length
- [x] Detect exact duplicates
- [x] Detect near-duplicates above the chosen threshold
- [x] Record corpus stats:
  - [x] row count
  - [x] token length distribution
  - [x] split balance
  - [x] task/source balance
- [x] Produce cleaned train/val JSONL outputs for student training

## 6. Student-Training Data Packaging

- [x] Convert the cleaned teacher corpus into the format expected by [`scripts/train_llada_canonical.jl`](/home/christos/code/julia/Swamma/scripts/train_llada_canonical.jl)
  - [x] [`scripts/package_llada_distill_corpus.py`](/home/christos/code/julia/Swamma/scripts/package_llada_distill_corpus.py)
- [ ] Prepare a student train file
- [ ] Prepare a student validation file
- [ ] If using mixed training, prepare the mixed raw+teacher train file
- [ ] Keep the held-out review set out of the training mixture
- [x] Save the exact train/val manifests under the run directory

## 7. Canonical Trainer Readiness

- [x] Verify the `~3B` config still parses after any related config changes
- [x] Add a reproducible rollout runner for corpus prep + prep-only gating
  - [x] [`scripts/run_llada_distill_rollout.py`](/home/christos/code/julia/Swamma/scripts/run_llada_distill_rollout.py)
- [x] Verify tokenizer loading for the chosen student tokenizer
- [x] Verify enough train chunks are produced at `seq_len=2048`
- [x] Verify enough validation chunks are produced at `seq_len=2048`
- [ ] Verify checkpoint write/resume still works with the chosen data volume
- [x] Add run metadata fields for:
  - [x] teacher model
  - [x] corpus manifest
  - [x] distillation mode
  - [x] raw/teacher mixture ratio

Current local gate result:
- Packed-stream prep gate on `data/rebel/train.jsonl` + `data/rebel/validation.jsonl` with `ibm-granite/granite-4.0-micro` at `seq_len=2048` produced `162` train chunks and `22` val chunks.

## 8. Smoke Gates Before Large Training

- [x] Run a parser/config smoke on the `~3B` config
- [x] Run a tiny end-to-end smoke using a tiny distilled corpus
- [ ] Run a short GPU smoke on the `~3B` config with:
  - [ ] `seq_len=512`
  - [ ] `batch_size=1`
  - [ ] very small step budget
- [ ] Run a second smoke at `seq_len=1024`
- [ ] Only then run the first `seq_len=2048` smoke
- [ ] Record:
  - [ ] compile/startup time
  - [ ] memory use
  - [ ] steps/sec or tokens/sec
  - [ ] checkpoint success
  - [ ] resume success

Current real-teacher pilot state:
- A completed `Qwen/Qwen2.5-7B-Instruct` slice exists at `data/distill/llada_prime_3b_qwen_slice16` with `16` accepted teacher rows.
- Pure teacher-only packaging of that slice is still too small for `seq_len=2048` validation on its own.
- A mixed raw+teacher packaging of that slice (`15` teacher-train rows + `1` teacher-val row, repeated `64x`, plus REBEL raw text) produced `245` train chunks and `27` val chunks at `seq_len=2048`.

## 9. Pilot Distillation Run

- [ ] Launch the first pilot run on the `~3B` student
- [ ] Keep the pilot short enough to fail fast if memory or throughput is bad
- [ ] Save checkpoints in a dedicated directory
- [ ] Log training loss and validation loss on schedule
- [ ] Save at least one qualitative generation sample per eval window
- [ ] Compare the pilot against a matched no-teacher baseline
- [ ] Decide whether teacher-text distillation is helping before committing to the long run

## 10. Pilot Acceptance Gates

- [ ] Numerical stability holds
- [ ] No NaN/Inf failures
- [ ] No checkpoint corruption
- [ ] Resume works from the latest checkpoint
- [ ] Validation loss is not materially worse than the no-teacher baseline
- [ ] Qualitative samples are clearly better than the no-teacher baseline on held-out prompts
- [ ] Throughput is slow but operationally sustainable on this machine

## 11. Full Run Launch

- [ ] Freeze the final training corpus manifest
- [ ] Freeze the final teacher identity and generation settings
- [ ] Freeze the final student config
- [ ] Freeze the final checkpoint directory naming scheme
- [ ] Launch the full `~3B` student run
- [ ] Keep only one canonical full run active at a time on this machine
- [ ] Monitor:
  - [ ] GPU memory
  - [ ] swap pressure
  - [ ] checkpoint cadence
  - [ ] validation trend
  - [ ] generation quality drift

## 12. Mid-Run Operations

- [ ] Periodically test resume from the latest checkpoint
- [ ] Periodically inspect generation samples on a fixed held-out prompt set
- [ ] Abort early if validation and samples both clearly regress
- [ ] Do not change tokenizer or corpus definitions mid-run
- [ ] Do not silently change teacher settings after the run starts
- [ ] Record any manual interventions in a run log

## 13. Training Completion

- [ ] Reach the planned total step budget or an explicit early-stop decision
- [ ] Save final checkpoint
- [ ] Save best checkpoint
- [ ] Save final run config and metadata
- [ ] Save final corpus manifest
- [ ] Save final prompt review set
- [ ] Save representative generation samples from:
  - [ ] early run
  - [ ] middle run
  - [ ] final run

## 14. Final Evaluation

- [ ] Run held-out generation review on the fixed prompt set
- [ ] Compare the final student against:
  - [ ] the no-teacher baseline
  - [ ] the pilot checkpoint
  - [ ] the teacher on a small review subset
- [ ] Review coherence, instruction-following, and reasoning quality manually
- [ ] Record failure modes:
  - [ ] repetition
  - [ ] shallow answers
  - [ ] hallucination
  - [ ] formatting instability
- [ ] Decide whether the final checkpoint is promotable

## 15. Promotion Criteria

- [ ] The final student is materially better than the no-teacher baseline
- [ ] Training was stable enough to reproduce operationally
- [ ] The run metadata is complete enough to reproduce the experiment
- [ ] The best checkpoint is clearly identified
- [ ] The checkpoint is worth keeping as the new PRIME baseline for this machine

## 16. Optional Phase 2 After First Completion

- [ ] Only after the first full sequence-distilled run completes, decide whether to add online teacher-aware distillation
- [ ] If yes, add teacher fields/config surface to [`scripts/train_llada_canonical.jl`](/home/christos/code/julia/Swamma/scripts/train_llada_canonical.jl)
- [ ] Add teacher loader/runtime management
- [ ] Add teacher-aware loss terms
- [ ] Gate online distillation behind explicit config flags
- [ ] Run a tiny online-distillation smoke before any larger follow-up
