# M3 — Retrieval Benchmark

## Purpose

Evaluate whether the Claim Exemplar Bank provides useful semantic coverage before M4 candidate generation.

## Probe Set

Use a held-out probe set that is separate from:

- the exemplar bank
- the frozen Phase 1 evaluation corpus

Each probe contains:

- `probe_id`
- `text`
- `expected_claim_type`
- `expected_value`

Do not reuse or paraphrase evaluation examples when creating probes.

## Retrieval Procedure

For each probe:

1. Normalize the text using the M2 TextNormalizer.
2. Generate an embedding using `all-MiniLM-L6-v2`.
3. Compare against all exemplar embeddings using cosine similarity.
4. Rank exemplars by descending similarity.
5. Record the top-k results.

Use:

- `k = 1`
- `k = 3`
- `k = 5`

## Metrics

### Recall@1

Correct `(claim_type, value)` appears in the top 1 result.

### Recall@3

Correct `(claim_type, value)` appears in the top 3 results.

### Recall@5

Correct `(claim_type, value)` appears in the top 5 results.

### Mean Reciprocal Rank

Measure the reciprocal rank of the first correct exemplar.

### Wrong-Value Retrieval Rate

Percentage of probes where the highest-ranked exemplar belongs to the wrong canonical value.

### Cross-Value Confusion

Record which canonical values are most frequently confused.

## Reproducibility

Freeze:

- model: `sentence-transformers/all-MiniLM-L6-v2`
- embedding dimension: 384
- normalization procedure
- exemplar-bank version
- probe-set version
- cosine-similarity implementation
- top-k values

The benchmark must produce identical results across repeated runs.

## Leakage Protection

A probe must not:

- exist verbatim in the exemplar bank
- normalize to an exemplar sentence
- be a deliberate paraphrase of an exemplar
- come from the frozen evaluation corpus

Run leakage checks before benchmarking.

## M3 Boundary

This benchmark evaluates retrieval quality only.

It does not:

- modify Phase 1 extraction
- create production claims
- change incident matching
- change contradiction logic
- perform deterministic adjudication

Those decisions belong to M4.

## Output

The benchmark should produce:

- Recall@1
- Recall@3
- Recall@5
- MRR
- wrong-value retrieval rate
- confusion analysis
- reproducibility result