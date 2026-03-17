# Reasoning Drafter: Vision & Training Strategy

## Core Idea

Pre-train the ReasoningDrafter's structured modules (AlgebraicCircuit + PredicateEngram) on deterministic game data (chess/Stockfish) to learn the **hidden features of logical reasoning** — constraint satisfaction, sequential consequence, valid inference under rules. Then fine-tune as a speculative drafter for an AR LLM verifier.

## Why Chess as Logic Pre-Training

Chess is not the target domain. It is a **massive corpus of pure constrained logic** with exact ground truth:

- Billions of positions with exact evaluations (Stockfish centipawn scores)
- Purely logical — no ambiguity, no noise, no style
- Sequential — move chains are causal consequence chains
- Constraint-heavy — legality, tactics, forced lines
- Features that make a position "tactically sharp" or "positionally won" are structurally analogous to features that make an argument "logically sound" or "well-supported"

## What Each Module Learns from Chess

### AlgebraicCircuit (SPN bank)

- **Leaf predicates** learn to detect board-level features (analogous to detecting premise truth values in argumentation)
- **Product nodes** learn conjunctive patterns — "these conditions must hold simultaneously" (piece coordination, tactical motifs)
- **Sum nodes** learn weighted alternatives — "any of these strategic plans is viable" (disjunctive reasoning)
- **SPN guarantees** mean the circuit's output is a calibrated probability — it approximates Stockfish's evaluation as a tractable probabilistic model

These features transfer because constraint satisfaction has the same structure whether the domain is chess or natural language reasoning.

### PredicateEngram (VQ-VAE + TPR)

The 4 abstract reasoning roles map onto:

| Role | Chess interpretation | General reasoning interpretation |
|------|---------------------|----------------------------------|
| Role 1 | Current position / state | Premise / current state |
| Role 2 | Candidate move / operator | Applied rule / inference step |
| Role 3 | Resulting position / state | Conclusion / resulting state |
| Role 4 | Positional constraints / context | Background constraints / context |

The VQ codebook (512 codes) learns **discrete reasoning situations** from chess (opening theory, tactical motifs, endgame patterns) that transfer to general reasoning patterns (modus ponens, case analysis, proof by contradiction). The rule mixing matrices learn how roles transform under each pattern.

### GLU Core (LinAttn + WavePDE)

- **LinearAttention**: Mixes information across sequence positions (propagates context)
- **WavePDE gate**: Frequency-selective gating — damps high-frequency noise (syntactic variation), amplifies low-frequency structure (logical/strategic patterns)
- The GLU's multiplicative gating provides the nonlinearity; no FFN needed since PredicateEngram and AlgebraicCircuit already provide per-position nonlinear transforms

## Architecture (implemented in `src/ReasoningDrafter.jl`)

Per block (no FFN, no local window attention):
```
RMSNorm → PredicateEngram → GLU(LinAttn content ⊙ sigmoid(WavePDE gate)) → Residual → AlgebraicCircuit → LayerNorm
```

- 2-3 blocks total
- ~790K params per block (at dim=256)
- ~18M params total (dominated by vocab embeddings at 32K vocab)

## Training Pipeline

### Phase 1: Chess Logic Pre-Training

- **Data**: Stockfish-evaluated positions (e.g., Lichess database, CCRL)
- **Task**: Tokenized board state → next best move prediction
- **What learns**: Circuit leaves learn constraint features; PredicateEngram codebook learns reasoning patterns; WavePDE learns temporal dynamics of move sequences
- **Ground truth**: Stockfish best move + evaluation score

### Phase 2: Verifier Distillation

- **Data**: AR verifier's reasoning traces (chain-of-thought, proofs, code)
- **Task**: Match verifier's output distribution via KL distillation
- **What adapts**: Pre-trained logic features adapt to natural language token space; codebook entries refine from chess motifs to language reasoning patterns

### Phase 3: Speculative Decoding Deployment

- Drafter generates K candidate tokens autoregressively
- Verifier (full AR LLM) accepts/rejects in parallel
- Circuit's calibrated constraint features improve acceptance rate on reasoning-heavy tokens

## Key Design Decisions

- **No SWAttention**: Local window is redundant at speculation-length sequences (8-64 tokens)
- **No SwiGLU FFN**: GLU + PredicateEngram + AlgebraicCircuit provide 3 nonlinearities per block; FFN would be a 4th (redundant)
- **Bidirectional WavePDE**: OK for AR inference since drafter re-forwards full sequence each step
- **Gating on all structured modules**: PredicateEngram and AlgebraicCircuit both have sigmoid gates initialized at -2.0, so they start near-identity and only activate when training signal says they help

## Open Questions

- **Board tokenization**: How to represent chess positions as token sequences for the drafter's embedding layer
- **Codebook size**: 512 codes may be too few for chess's combinatorial complexity; may need 2048+
- **Role count**: 4 roles is a hypothesis; chess might need more (pieces have many relational dimensions)
- **Transfer gap**: How well do features learned from exact game logic transfer to fuzzy natural language reasoning?
- **Training curriculum**: Should Phase 1 include other deterministic games (Go, checkers) for diversity?
