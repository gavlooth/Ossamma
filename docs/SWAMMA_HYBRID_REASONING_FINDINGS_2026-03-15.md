# Swamma Hybrid Reasoning Findings

Date: 2026-03-15

## Scope

This document summarizes the findings from an architecture discussion around:

- the existing Julia diffusion / masking implementation in Swamma and Ossamma
- Qwen hidden-size and depth scaling reference points
- dense-width versus depth tradeoffs for Swamma
- what fits on a single NVIDIA DGX Spark
- using Swamma as a reasoning drafter rather than a full standalone LLM
- introducing MoE as reasoning-specialized FFN experts
- adding a rare-gated symbolic expert path

## 1. Existing Diffusion / Partial Masking Implementation

The repository already contains a Julia implementation of a diffusion-style LLM path.

Key locations:

- [`src/LLaDA.jl`](/home/christos/code/julia/Swamma/src/LLaDA.jl)
- [`src/Training.jl`](/home/christos/code/julia/Swamma/src/Training.jl)
- [`src/Drafter.jl`](/home/christos/code/julia/Swamma/src/Drafter.jl)
- [`src/TiDAR.jl`](/home/christos/code/julia/Swamma/src/TiDAR.jl)

Observed implementation points:

- `apply_subtoken_mask(...)` performs random partial masking at the sub-token level.
- `sample_mask_ratio(...)` supports the diffusion-style mask-ratio schedule.
- `unmask_subtoken_step(...)` reveals masked sub-tokens progressively based on confidence.
- `generate(...)` performs iterative denoising from a masked state.
- `diffusion_loss(...)` computes loss only on masked positions.

This confirms that the codebase is not only conceptually diffusion-oriented; the core masking and denoising path is already implemented in Julia.

## 2. Qwen Reference Dimensions

Current Qwen config values were checked from Hugging Face model configs on 2026-03-15.

Dense Qwen references:

- `Qwen2.5-7B`: hidden size `3584`, layers `28`
- `Qwen2.5-14B`: hidden size `5120`, layers `48`
- `Qwen2.5-32B`: hidden size `5120`, layers `64`
- `Qwen2.5-72B`: hidden size `8192`, layers `80`
- `Qwen3-8B`: hidden size `4096`, layers `36`
- `Qwen3-14B`: hidden size `5120`, layers `40`
- `Qwen3-32B`: hidden size `5120`, layers `64`

MoE Qwen references:

- `Qwen3-30B-A3B`: hidden size `2048`, layers `48`, experts `128`
- `Qwen3-235B-A22B`: hidden size `4096`, layers `94`, experts `128`

Interpretation:

- dense Qwen uses width aggressively at large scale
- MoE Qwen does not simply widen the dense trunk further
- MoE scaling is consistent with a narrower shared trunk plus larger sparse capacity

## 3. Width vs Depth for Swamma

The main Swamma advantage is on sequence scaling, not on dense projection scaling.

Roughly:

- vanilla dense transformer layer: `O(L d^2 + L^2 d)`
- Swamma-like layer: removes the bad `L^2 d` full-attention term, but still pays substantial `L d^2`

This matters because the current `SwammaBlock` is not light. It contains:

- a global branch with `LinearAttention`
- a `WavePDE` gate path
- a local `SWAttention` branch
- alpha mixing
- FFN

Relevant code:

- [`src/Swamma.jl:287`](/home/christos/code/julia/Swamma/src/Swamma.jl:287)
- [`src/Swamma.jl:340`](/home/christos/code/julia/Swamma/src/Swamma.jl:340)

Conclusion:

- Swamma should not spend its advantage by pushing width too high too early.
- For large-scale designs, it is more coherent to keep width in a Qwen-like band and spend budget on depth, context, and sparse reasoning capacity.

## 4. MoE Placement: What to Expertize

Important architectural distinction:

- bad first design: route between whole `SwammaBlock` experts
- better design: keep the Swamma sequence backbone shared and expertize only the FFN sublayer

In other words:

```text
good:
shared Swamma block -> router -> FFN experts -> combine

bad:
router -> full SwammaBlock expert A/B/C -> combine
```

Why the whole-block approach is undesirable:

- every expert duplicates the global branch
- every expert duplicates the local branch
- every expert duplicates `WavePDE`
- routing becomes much harder to stabilize
- parameter growth is too aggressive

Natural insertion point in current code:

- `FFN::Union{SwiGLU, Nothing}` in [`src/Swamma.jl:336`](/home/christos/code/julia/Swamma/src/Swamma.jl:336)

Proposed conceptual replacement:

```julia
FFN::Union{SwiGLU, MoEFFN, Nothing}
```

## 5. Shared Layers Are Still Active

MoE does not make the shared backbone free.

If a model has `N` shared Swamma layers:

- all `N` shared layers are active for every token
- only the expertized FFN part is sparse

So active parameters are still:

```text
active params = shared backbone params + selected expert params
```

not:

```text
active params = only selected expert params
```

This is the key reason not to overbuild the shared trunk on Spark-scale hardware.

## 6. DGX Spark Feasibility

Official DGX Spark references checked on 2026-03-15:

- hardware overview: <https://docs.nvidia.com/dgx/dgx-spark/hardware.html>
- system overview: <https://docs.nvidia.com/dgx/dgx-spark/system-overview.html>
- NVIDIA fine-tuning playbook: <https://build.nvidia.com/spark/pytorch-fine-tune>
- product page: <https://www.nvidia.com/en-us/project-digits/>

Useful hardware signals:

- `128 GB` unified memory
- `273 GB/s` memory bandwidth
- marketed for local development and fine-tuning workflows, not for end-to-end 120B pretraining on a single box

Using local parameter-count measurements on the actual Swamma/LLaDA code, the shared dense backbone alone scales roughly like this:

- `d=6144`, `64` shared layers: about `32B` params
- `d=6144`, `80` shared layers: about `40B` params
- `d=6144`, `96` shared layers: about `49B` params
- `d=8192`, `64` shared layers: about `58B` params
- `d=8192`, `80` shared layers: about `72B` params

Implication:

- single DGX Spark is appropriate for architecture development and scaled-down training
- it is not appropriate for full 120B training from scratch
- PEFT / LoRA / QLoRA of large pretrained models is a separate question and is much more realistic than full training

## 7. Recommended Spark-Safe Surrogate

Recommended current development target for single-DGX-Spark experimentation:

```text
embedding_dimension = 4096
number_of_heads = 32
head_dim = 128
number_of_layers = 32
state_dimension = 4096
num_experts = 16
top_k = 2
MoE every 2nd layer
FFN-only MoE
```

Why this surrogate:

- preserves the intended architecture pattern
- is large enough to expose routing and stability issues
- is much more realistic on Spark than any 120B-class full training design

Paired future large-scale direction:

```text
d_model = 6144
shared layers = 64-80
num_experts = 16-32
top_k = 2
MoE every 2nd or 3rd layer
```

## 8. Swamma as a Reasoning Drafter

The more practical direction is not to train a standalone Swamma foundation model first.

Instead:

- use a strong long-context autoregressive LLM as the language backbone
- use Swamma as a parallel reasoning drafter
- let the AR model verify, accept, reject, or finalize the draft

Conceptual flow:

```text
context
-> AR long-context LLM
-> Swamma drafts reasoning tokens or masked refinements in parallel
-> AR LLM verifies / edits / finalizes
```

This aligns well with the existing drafter / verifier path:

- [`src/Drafter.jl`](/home/christos/code/julia/Swamma/src/Drafter.jl)
- [`src/TiDAR.jl`](/home/christos/code/julia/Swamma/src/TiDAR.jl)

## 9. Training Target for Swamma Drafter

If Swamma becomes the drafter, then it should be trained toward reasoning-specialized drafting rather than full fluency.

Promising data/task families:

- logical puzzles
- formal or semi-formal argumentation
- contradiction repair
- entailment / consistency tasks
- proof-style chain construction
- planning decomposition

This makes Swamma useful as:

- a fast parallel reasoner
- a structured refinement engine
- a drafter of candidate reasoning spans or next-step tokens

while the AR LLM remains responsible for:

- fluency
- broad language competence
- final answer rendering

## 10. Reasoning Experts, Not Just Capacity Experts

MoE becomes more meaningful if experts specialize by reasoning mode rather than being treated only as parameter-scaling devices.

Potential expert modes:

- deductive logic
- contradiction detection and repair
- argument structure / rebuttal
- arithmetic or symbolic pattern completion
- causal reasoning
- planning decomposition

This leads to a stronger interpretation of MoE:

- not just “more FFNs”
- but “mixture of reasoning processes”

## 11. Symbolic Tool Expert

The most anorthodox but coherent extension is to add a rare-gated symbolic expert beside the dense neural experts.

Important nuance:

- this symbolic expert is not a normal FFN expert
- it is a special routed mechanism
- it should be span-level or state-level, not single-token inner-loop by default

Candidate symbolic expert types:

- theorem prover
- SAT / SMT solver
- symbolic planner
- algebra engine
- HVM / SupGen-style symbolic synthesis runtime

This is directly related to the Higher Order Company / HVM / SupGen line of thought:

- HVM is positioned as a massively parallel symbolic runtime
- SupGen is taken as evidence that higher-order symbolic computation can be competitive in symbolic domains

Relevant external references:

- <https://higherorderco.com/>
- <https://gist.github.com/VictorTaelin/5d42e97684ff81f7e507aa5c149b2a64>

The coherent architecture becomes:

```text
AR LLM
-> Swamma reasoning drafter
-> router
   -> dense neural reasoning experts
   -> rare symbolic HVM/SupGen-style expert
-> merge
-> AR LLM verifier/finalizer
```

## 12. Central Open Design Question

If the symbolic expert path is pursued, the most important design question is the interface:

- What exact structure is passed to the symbolic expert?

Possible representations:

- masked token spans
- normalized logical forms
- puzzle states
- small AST-like reasoning structures
- compact latent thought segments

This interface decision is more important than the exact expert count.

## 13. Current Recommendation

Near-term recommendation:

1. Use Swamma as a reasoning drafter around a strong long-context AR LLM.
2. Keep the Swamma sequence backbone shared.
3. Add FFN-only MoE first.
4. Train the drafter on logic, argumentation, contradiction repair, and planning-style tasks.
5. Add a rare-gated symbolic expert only after the FFN-MoE drafter is stable.

This path preserves the architectural novelty while keeping the development plan realistic.
