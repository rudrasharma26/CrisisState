"""Deterministic semantic candidate adjudication for CrisisState M4."""

from __future__ import annotations

from typing import Any


class SemanticAdjudicator:
    """Turn semantic candidates into a deterministic decision.

    Semantic similarity proposes.
    This component decides whether the proposal is admissible.

    It never overrides a valid Phase 1 lexical claim.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.67,
        margin_threshold: float = 0.04,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")

        if not 0.0 <= margin_threshold <= 1.0:
            raise ValueError("margin_threshold must be between 0 and 1")

        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold

    def adjudicate(
        self,
        text: str,
        semantic_result: dict[str, Any],
        lexical_claim: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a deterministic semantic adjudication decision."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text must not be empty")

        if not isinstance(semantic_result, dict):
            raise TypeError("semantic_result must be a dictionary")

        # Phase 1 lexical extraction always takes precedence.
        if lexical_claim is not None:
            return {
                "status": "LEXICAL_PRESERVED",
                "decision_source": "LEXICAL",
                "claim_type": lexical_claim["claim_type"],
                "value": lexical_claim["value"],
                "semantic_candidates": semantic_result.get(
                    "candidates",
                    [],
                ),
                "rule_applied": "PRESERVE_PHASE1_LEXICAL_CLAIM",
            }

        status = semantic_result.get("status")

        if status != "CANDIDATE_AVAILABLE":
            return {
                "status": "UNRESOLVED",
                "decision_source": "NONE",
                "claim_type": None,
                "value": None,
                "semantic_candidates": semantic_result.get(
                    "candidates",
                    [],
                ),
                "rule_applied": (
                    f"SEMANTIC_RESULT_{status or 'MISSING'}"
                ),
            }

        candidates = semantic_result.get("candidates", [])

        if not candidates:
            return {
                "status": "UNRESOLVED",
                "decision_source": "NONE",
                "claim_type": None,
                "value": None,
                "semantic_candidates": [],
                "rule_applied": "NO_ADMISSIBLE_CANDIDATES",
            }

        top = candidates[0]

        score = float(top["similarity_score"])
        margin = float(semantic_result.get("margin", 0.0))

        if score < self.similarity_threshold:
            return {
                "status": "UNRESOLVED",
                "decision_source": "SEMANTIC_REJECTED",
                "claim_type": None,
                "value": None,
                "semantic_candidates": candidates,
                "similarity_score": score,
                "margin": margin,
                "rule_applied": "BELOW_SIMILARITY_THRESHOLD",
            }

        if margin < self.margin_threshold:
            return {
                "status": "UNRESOLVED",
                "decision_source": "SEMANTIC_REJECTED",
                "claim_type": None,
                "value": None,
                "semantic_candidates": candidates,
                "similarity_score": score,
                "margin": margin,
                "rule_applied": "INSUFFICIENT_SIMILARITY_MARGIN",
            }

        return {
            "status": "ACCEPTED",
            "decision_source": "SEMANTIC",
            "claim_type": top["claim_type"],
            "value": top["value"],
            "matched_exemplar_id": top["exemplar_id"],
            "similarity_score": score,
            "candidate_values": [
                {
                    "claim_type": candidate["claim_type"],
                    "value": candidate["value"],
                    "score": candidate["similarity_score"],
                    "exemplar_id": candidate["exemplar_id"],
                }
                for candidate in candidates
            ],
            "semantic_candidates": candidates,
            "rule_applied": "TOP_CANDIDATE_PASSED_DETERMINISTIC_GATE",
        }