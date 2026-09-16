# M3 — Exemplar Bank Evaluation

## Goal

Verify that the Claim Exemplar Bank provides useful semantic coverage before M4 candidate generation is enabled.

## Retrieval Benchmark

Each held-out probe is embedded with:

sentence-transformers/all-MiniLM-L6-v2

The system retrieves the nearest exemplars using cosine similarity.

Measure:

- Recall@1
- Recall@3
- Recall@5
- Mean Reciprocal Rank (MRR)
- wrong-value retrieval rate
- cross-value confusion rate

## Leakage Checks

Before evaluation:

- no probe may exist in the exemplar bank
- no near-duplicate probe/exemplar pair
- no evaluation-specific wording may have been deliberately copied
- normalized-text duplicates must be absent

## Confusion Analysis

Record the most common incorrect retrievals.

Pay particular attention to semantically close values that could cause dangerous candidate generation.

## Reproducibility

The benchmark must be deterministic:

- fixed exemplar bank version
- fixed probe set
- fixed model
- fixed normalization
- fixed retrieval procedure
- fixed top-k values

## M3 Boundary

M3 only evaluates exemplar retrieval quality.

It does NOT:

- modify Phase 1 claim extraction
- change incident matching
- change contradiction logic
- create final semantic claims
- introduce NLI
- introduce an LLM
- alter production decisions

## Output

M3 produces:

1. frozen exemplar bank
2. leakage audit
3. retrieval benchmark results
4. confusion analysis
5. bank version

M4 uses these results to design semantic candidate generation and deterministic adjudication.