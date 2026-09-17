# M4 — Semantic Candidate Generation and Deterministic Adjudication

## Goal

Use semantic similarity to generate candidate claim interpretations while keeping the final decision deterministic and auditable.

M4 does not replace Phase 1 extraction. It extends it.

## Pipeline

Report
→ clause/span segmentation
→ Phase 1 lexical extraction
→ semantic candidate generation
→ candidate filtering
→ deterministic adjudication
→ normalized Claim

## 1. Semantic Candidate Generation

For each eligible text span:

1. Normalize the span using the M2 TextNormalizer.
2. Generate a 384-dimensional embedding with `all-MiniLM-L6-v2`.
3. Search the frozen Claim Exemplar Bank using cosine similarity.
4. Retrieve the top **5** exemplars.
5. Preserve:
   - exemplar ID
   - claim type
   - value
   - similarity score
   - rank

Top-5 retrieval is used because M3 achieved **100% Recall@5** on the held-out 60-probe benchmark.

Semantic retrieval only produces candidates.

It does not create the final claim.

## 2. Candidate Filtering

Candidates are filtered before adjudication.

### When Phase 1 identifies a claim type

Prefer candidates belonging to that same claim type.

Cross-type semantic matches are rejected from final adjudication.

Example:

`ROAD_ACCESS:IMPASSABLE`

must not be replaced by:

`TRAFFIC_STATUS:STOPPED`

merely because their embeddings are similar.

### When Phase 1 finds no usable claim

Candidates from different claim types may remain temporarily.

The system must require deterministic evidence that one interpretation is sufficiently distinguishable.

If candidates remain ambiguous, do not force a claim.

Create an `UnresolvedSpan`.

## 3. Similarity Threshold

M4 must not hard-code a threshold arbitrarily.

Thresholds must be selected from the M3 held-out retrieval results using a deterministic threshold-sensitivity analysis.

The frozen Phase 1 evaluation corpus must not be used to tune these thresholds.

The selected threshold must be versioned and recorded.

## 4. Deterministic Adjudication

Adjudication follows this precedence:

1. Valid Phase 1 lexical extraction
2. Compatible semantic candidate
3. Deterministic similarity/ambiguity rules
4. Unresolved result when evidence is insufficient

Semantic similarity cannot silently override a valid lexical result.

When multiple semantic candidates remain:

- prefer the highest admissible similarity
- require the configured similarity threshold
- apply the configured margin/ambiguity rule
- reject the interpretation when the decision remains ambiguous

No random selection.

No LLM decision.

No NLI decision.

## 5. Tie and Ambiguity Handling

If competing candidates are effectively indistinguishable under the configured rules:

```text
Decision = UNRESOLVED