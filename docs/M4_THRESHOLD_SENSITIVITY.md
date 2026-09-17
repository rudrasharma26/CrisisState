# M4 — Threshold Sensitivity

## Goal

Select and document semantic candidate thresholds without tuning against the frozen M0 evaluation corpus or the held-out M3 retrieval benchmark.

## Data Separation

Three datasets remain separate:

1. M0 frozen evaluation corpus
2. M3 retrieval benchmark probes
3. M4 threshold-calibration set

Only the M4 calibration set may be used to select candidate thresholds.

## Threshold Sweep

Evaluate a deterministic range of cosine-similarity thresholds.

For each threshold record:

- accepted candidates
- rejected candidates
- correct semantic candidates
- incorrect candidates
- unresolved cases
- false semantic acceptances
- rejection rate

Also evaluate the similarity margin between the first and second admissible candidates.

## Selection Principle

Do not optimize for semantic accuracy alone.

Prefer a threshold configuration that:

- rejects weak semantic matches
- preserves useful candidate recall
- minimizes incorrect semantic claims
- preserves deterministic behaviour
- does not weaken Phase 1 behaviour

## Frozen Constraints

Threshold selection must not:

- modify Phase 1 lexical extraction
- modify incident matching gates
- modify contradiction logic
- use M0 evaluation examples for tuning
- use M3 benchmark probes for tuning

## Output

Record:

- selected threshold
- margin rule
- calibration-set version
- threshold-sweep results
- rationale
- final configuration version

## Provisional M4 Candidate Gate

Based on the 100-probe calibration sweep, the provisional candidate-generation gate is:

- cosine similarity >= 0.67
- top-1 vs top-2 similarity margin >= 0.04

Calibration result:

- Accepted: 41/100
- Correct accepted: 33/100
- Wrong accepted: 8/100
- Correct coverage: 33%
- Accepted accuracy: 80.49%

This is a candidate-generation gate only.

It is NOT the final semantic adjudication threshold.

M4 deterministic adjudication must apply additional claim-type compatibility,
lexical evidence, ambiguity checks, and rejection rules before creating a
semantic claim.

The gate remains subject to M4 evaluation against the frozen Phase 1 baseline.