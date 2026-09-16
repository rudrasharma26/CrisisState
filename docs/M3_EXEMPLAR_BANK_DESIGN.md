# M3 — Claim Exemplar Bank Design

## Purpose

Build a curated semantic reference bank for future candidate generation.

The exemplar bank is a retrieval resource, not a classifier and not a source of final semantic decisions.

## 1. Exemplar Structure

M3 uses the existing M1 `ClaimExemplar` model without modification.

Each exemplar contains:

- `id`
- `claim_type`
- `value`
- `text`
- `source`
- `version`
- `created_at`

M3 does not introduce additional exemplar fields.

The exemplar bank is a curated semantic reference set, not a classifier.

## 2. Coverage

The frozen Phase 1 vocabulary contains 20 canonical values:

- FLOODED: YES, NO
- ROAD_ACCESS: PASSABLE, IMPASSABLE
- WATER_LEVEL: LOW, MODERATE, HIGH, CRITICAL
- PEOPLE_TRAPPED: YES, NO
- BUILDING_DAMAGE: NONE, DAMAGED, COLLAPSED
- EVACUATION: ORDERED, NOT_ORDERED, IN_PROGRESS, COMPLETED
- TRAFFIC_STATUS: MOVING, SLOW, STOPPED

Target: 10 hand-authored exemplars per canonical value.

Total target: 200 exemplars.

## 3. Exemplar Count

Default target:

**10 exemplars per canonical value**

Use up to **15** for values with higher linguistic ambiguity.

The bank should prioritize semantic and linguistic diversity rather than superficial rewrites.

## 4. Exemplar Writing Rules

Each exemplar should:

- represent one primary claim
- clearly map to one canonical value
- resemble realistic crisis-report language
- vary syntax, vocabulary, length, and phrasing
- include both short and natural report-style formulations
- avoid unnecessary incident-specific details
- avoid near-duplicates

Do not create exemplars by rewriting evaluation examples.

M3 must not introduce M5/M6 semantic behavior prematurely.

## 5. Evaluation Leakage Prevention

The exemplar bank must be created independently of the frozen evaluation corpus.

Controls:

- never copy evaluation sentences into the bank
- never rewrite evaluation sentences into exemplars
- avoid near-duplicate exemplar/evaluation pairs
- maintain separate exemplar and evaluation data
- run text and normalized-text duplicate checks
- run a similarity-based leakage audit before freezing M3

Evaluation examples remain frozen and must not be used for exemplar construction.

## 6. Later Embedding Usage

M3 is designed for use with:

`sentence-transformers/all-MiniLM-L6-v2`

Embedding flow:

Exemplar text
→ TextNormalizer
→ 384-dimensional embedding
→ SQLite EmbeddingCache

At M4:

Report/clause
→ TextNormalizer
→ embedding
→ cosine similarity
→ top-k exemplar candidates
→ deterministic adjudication

Embeddings generate candidates only.

They do not directly create or finalize claims.

## 7. M3 Evaluation

M3 is evaluated independently before semantic candidate generation becomes production behavior.

Measure:

- Recall@1
- Recall@3
- Recall@5
- Mean Reciprocal Rank (MRR)
- wrong-value retrieval rate
- cross-value confusion

Also record similarity distributions and major confusion pairs.

M3 acceptance requires:

1. a frozen exemplar bank
2. successful leakage audit
3. reproducible retrieval evaluation
4. documented retrieval/confusion results

M4 uses these results when defining candidate thresholds and deterministic adjudication rules.

## 8. Frozen Principle

The exemplar bank supports:

**semantic candidate generation → deterministic adjudication**

It does not replace Phase 1 lexical extraction or become an independent decision-maker.