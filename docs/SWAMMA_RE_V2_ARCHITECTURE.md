# Swamma RE v2 Architecture (Draft 1)

Date: 2026-03-14  
Status: Design target for replacing diagnostic-grade RE v1 scaffold

## 1) Objective

Build a relation extractor that is:
- subquadratic in mention count (`N`) during candidate construction and scoring
- stable across seeds/reruns
- explicit about pair coverage vs precision tradeoff
- compatible with the existing stable `SwammaBlock` token backbone

Target asymptotic direction:
- avoid `O(N^2)` all-pairs scoring
- prefer `O(N log N + Nk)` retrieval-style edge generation with bounded `k`

## 2) Module Graph (Tokens -> Spans -> Edges -> Evidence -> Relations)

1. Token Encoder (`SwammaBlock` stack)
- input: token ids + mask
- output: contextual token states `H` of shape `[B, T, D]`

2. Mention Proposal v2 (coarse-to-fine)
- boundary heads produce start/end logits over tokens
- span composer builds span vectors from boundary-pooled token features
- mention scorer ranks spans and keeps top `K` per example
- output:
  - span list `S` (indices)
  - span embeddings `Z` of shape `[B, K, Ds]`

3. Span Context Graph v2 (sparse message passing)
- construct sparse span graph (local neighbors + semantic neighbors + sentence links)
- run `L` sparse context layers over span nodes
- output: contextualized span embeddings `Z'` `[B, K, Ds]`

4. Edge Retrieval v2 (subquadratic)
- generate a compact edge candidate set by approximate nearest-neighbor retrieval on span keys
- optional typed routing prior (entity-type compatibility and distance buckets)
- keep top `k` outgoing candidates per span
- output edge list `E` with size `|E| <= B * K * k`

5. Evidence Finder v2 (edge-conditioned evidence pooling)
- for each candidate edge `(i,j)`, attend over token states and/or sentence chunks conditioned on `(z'_i, z'_j)`
- produce evidence vector and concentration diagnostics
- output: edge evidence `U` `[B, |E|, De]`

6. Relation Scorer v2
- score each edge for:
  - relation label logits over `R` classes
  - confidence / calibration score
  - optional retrieval-rank auxiliary target
- decode with global threshold + relation priors + structural constraints

## 3) Tensor Shapes (Canonical)

Assume:
- batch `B`
- tokens `T`
- token dim `D`
- kept spans `K`
- candidate edges `M` where `M = K * k` (bounded)
- relation classes `R`

Shapes:
- token states: `H = [B, T, D]`
- span embeddings: `Z = [B, K, Ds]`
- contextualized spans: `Z' = [B, K, Ds]`
- edge index tensor: `I = [B, M, 2]`
- edge features: `F = [B, M, Df]`
- evidence vectors: `U = [B, M, De]`
- relation logits: `L_rel = [B, M, R]`
- confidence logits: `L_conf = [B, M, 1]`

## 4) Complexity Targets

Token encoding:
- `O(B * T * D * L_token)` (unchanged backbone cost)

Mention proposal:
- coarse boundary pass: `O(B * T * D)`
- span composition over limited width window `W`: `O(B * T * W * D)` then top-`K` pruning

Span context graph:
- sparse edges `E_span ~ K * k_span`
- cost `O(B * E_span * Ds)`

Edge retrieval:
- key/index build: `O(B * K * log K)` (or amortized via ANN structure)
- retrieval: `O(B * K * k * log K)` (or `O(B * K * k)` approximate)
- no full `K^2` relation matrix

Edge scoring:
- `O(B * M * (Df + De + R))`, where `M = K * k`

## 5) Keep vs Remove (from current stack)

Keep:
- `SwammaBlock` token backbone
- sparse candidate-pair mindset
- decode-time type/consistency constraint hooks
- evidence diagnostics interface

Remove / Replace:
- dense or quasi-dense pair scoring paths that trend to `K^2`
- unstable schedule-heavy tuning as primary improvement lever
- stochastic batch-negative paths without explicit RNG plumbing

Refactor:
- current pair proposer -> explicit retrieval module with bounded outgoing degree
- current relation head -> edge-centric scorer with calibrated confidence head

## 6) Training Objective v2

Primary losses:
- mention detection / span quality loss
- relation classification loss on candidate edges
- confidence calibration loss (ECE-aware proxy or focal calibration variant)

Auxiliary losses:
- retrieval ranking loss for edge proposal quality
- optional evidence alignment / concentration regularizer

Gating policy:
- do not promote models using only one lucky seed or one decode point
- require repeated deterministic eval pass and cross-seed compact checks

## 7) Acceptance Gates (v2 vs v1)

Must improve together:
- relation F1 on full validation under fixed decode protocol
- pair recall and pair@16 (coverage quality)
- stability (small variance across repeated runs with fixed seed config)

Operational constraints:
- throughput degradation must stay within acceptable bound (define per run)
- GPU memory footprint must remain trainable at current batch size or with known accumulation fallback

## 8) Immediate Implementation Sequence

1. Formalize `EdgeRetrievalV2` interface and integrate into `RelationExtraction.jl`.
2. Add deterministic retrieval candidate generation and logging.
3. Implement edge-conditioned evidence pooling block behind a config flag.
4. Add edge scorer with explicit confidence head and calibration stats.
5. Run short deterministic smoke runs, then compact cross-seed checks.
6. Only then launch long continuation runs.
