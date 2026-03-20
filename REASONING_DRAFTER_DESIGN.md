# ReasoningDrafter Design Notes

## Corrected Architecture

The current `ReasoningDrafter` design is organized into three stages:

```text
[ INPUT h ]
    |
[ SHARED VQ-VAE ] -> coarse opcode
    |
[ 4 WAVE-PDEs ] -> propagated constraints (global preprocessor)
    |
[ PROPOSER BLOCK x N ]
    each block:
      norm
      gated WavePDE + LinearAttention fusion
      proposal refinement
      residual
    |
[ DYNAMIC ROLE BINDING ] -> fine opcode
    |
[ 4 PREDICATE HEADS ] -> logical verification
    |
[ ALGEBRAIC CIRCUIT ] -> consistency check
    |
[ VETO GATE ]
    |
[ + RESIDUAL ]
    |
[ OUTPUT h ]
```

## Stage 1: Global Front End

The front end is a shared structural preprocessor, not the proposer itself.

It does three jobs:

1. `Shared VQ-VAE` assigns a coarse opcode that represents the current reasoning regime.
2. Four `Wave-PDE` heads propagate constraints under different structural biases.
3. The fused output becomes the context field consumed by proposer blocks.

The front-end `Wave-PDE`s are global preprocessors. They shape the battlefield before proposal formation.

## Stage 2: Proposer Blocks

Each proposer block replaces a classical attention layer with a gated combination of wave-style propagation and associative retrieval.

```text
[ Norm ]
    |
[ Wave branch ] --------+
                        |
[ LinearAttention ] ----+--> [ gated fusion / GLU ] -> proposal
```

The intent is:

- `WavePDE` contributes propagated structure.
- `LinearAttention` contributes global associative recall.
- The gate/GLU decides how much of each branch to expose to the next layer.

This block is the model's proposal engine. It is not the final auditor.

## Stage 3: Audit Tail

After the proposer stack, the model performs explicit reasoning verification:

1. `Dynamic Role Binding` re-quantizes the proposal into a finer opcode.
2. `Predicate Heads` evaluate the candidate structure.
3. `Algebraic Circuit` checks consistency across the candidate conclusions.
4. `Veto Gate` suppresses proposals that fail the audit.
5. The residual path preserves the original hidden state when the audit is inconclusive.

## Design Rules

- Keep the 4-head Wave-PDE front end global and shared.
- Keep the proposer blocks lightweight and stackable.
- Use gated WavePDE + LinearAttention fusion inside proposer blocks instead of bare attention.
- Keep role binding, predicates, circuit, and veto out of the proposer path so they remain explicit audit stages.
- Gate the proposal update, not the entire hidden state, so residual learning stays stable.

## Practical Reading

This architecture is best read as:

```text
global structural preprocessing
-> repeated proposal refinement
-> explicit logical audit
-> vetoed residual update
```

That is the canonical interpretation going forward.
