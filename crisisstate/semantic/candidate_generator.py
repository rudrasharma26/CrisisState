"""Semantic candidate generation for CrisisState M4."""

from __future__ import annotations

from typing import Any

from crisisstate.semantic.exemplar_search import ExemplarSearch


class SemanticCandidateGenerator:
    """Generate filtered semantic claim candidates.

    This component proposes candidates only.
    It does not create final claims.
    """

    DEFAULT_TOP_K = 5
    DEFAULT_SIMILARITY_THRESHOLD = 0.67
    DEFAULT_MARGIN_THRESHOLD = 0.04

    def __init__(
        self,
        exemplar_search: ExemplarSearch,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")

        if not 0.0 <= margin_threshold <= 1.0:
            raise ValueError("margin_threshold must be between 0 and 1")

        self.exemplar_search = exemplar_search
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold

    def generate(
        self,
        text: str,
        preferred_claim_type: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        """Generate semantic candidates for a text span."""

        results = self.exemplar_search.search(
            text,
            top_k=top_k,
        )

        if not results:
            return {
                "text": text,
                "candidates": [],
                "status": "NO_CANDIDATES",
                "reason": "NO_EXEMPLARS_RETURNED",
            }

        compatible = results

        if preferred_claim_type is not None:
            compatible = [
                result
                for result in results
                if result["claim_type"] == preferred_claim_type
            ]

        thresholded = [
            result
            for result in compatible
            if result["similarity_score"] >= self.similarity_threshold
        ]

        if not thresholded:
            return {
                "text": text,
                "candidates": [],
                "status": "REJECTED",
                "reason": "BELOW_SIMILARITY_THRESHOLD",
            }

        top_candidate = thresholded[0]

        second_candidate = (
            thresholded[1]
            if len(thresholded) > 1
            else None
        )

        margin = (
            top_candidate["similarity_score"]
            - second_candidate["similarity_score"]
            if second_candidate is not None
            else 1.0
        )

        if margin < self.margin_threshold:
            return {
                "text": text,
                "candidates": thresholded,
                "status": "AMBIGUOUS",
                "reason": "INSUFFICIENT_SIMILARITY_MARGIN",
                "top_score": top_candidate["similarity_score"],
                "second_score": (
                    second_candidate["similarity_score"]
                    if second_candidate is not None
                    else None
                ),
                "margin": margin,
            }

        return {
            "text": text,
            "candidates": thresholded,
            "status": "CANDIDATE_AVAILABLE",
            "reason": "PASSED_SEMANTIC_GATE",
            "top_score": top_candidate["similarity_score"],
            "second_score": (
                second_candidate["similarity_score"]
                if second_candidate is not None
                else None
            ),
            "margin": margin,
        }