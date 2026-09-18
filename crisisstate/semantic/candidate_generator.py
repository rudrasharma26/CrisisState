"""Semantic candidate generation for CrisisState M4."""

from __future__ import annotations

from typing import Any, Dict, List

from crisisstate.semantic.exemplar_search import ExemplarSearch


class SemanticCandidateGenerator:
    """Generate deterministic semantic candidates.

    Semantic retrieval proposes exemplar matches.
    This component collapses duplicate exemplars into canonical
    claim-value candidates and applies the semantic gate.
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
            raise ValueError(
                "similarity_threshold must be between 0 and 1"
            )

        if not 0.0 <= margin_threshold <= 1.0:
            raise ValueError(
                "margin_threshold must be between 0 and 1"
            )

        self.exemplar_search = exemplar_search
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold

    @staticmethod
    def _collapse_canonical_candidates(
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep only the strongest exemplar for each canonical value."""
        best_by_value: Dict[tuple[str, str], Dict[str, Any]] = {}

        for result in results:
            key = (
                result["claim_type"],
                result["value"],
            )

            current = best_by_value.get(key)

            if (
                current is None
                or result["similarity_score"]
                > current["similarity_score"]
            ):
                best_by_value[key] = dict(result)

        candidates = sorted(
            best_by_value.values(),
            key=lambda item: (
                -item["similarity_score"],
                item["claim_type"],
                item["value"],
                item["exemplar_id"],
            ),
        )

        for rank, candidate in enumerate(candidates, start=1):
            candidate["canonical_rank"] = rank

        return candidates

    def generate(
        self,
        text: str,
        preferred_claim_type: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        """Generate semantically ranked canonical candidates."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

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

        candidates = self._collapse_canonical_candidates(
            compatible
        )

        thresholded = [
            candidate
            for candidate in candidates
            if candidate["similarity_score"]
            >= self.similarity_threshold
        ]

        if not thresholded:
            return {
                "text": text,
                "candidates": [],
                "status": "REJECTED",
                "reason": "BELOW_SIMILARITY_THRESHOLD",
            }

        top_candidate = thresholded[0]

        if len(thresholded) == 1:
            second_score = None
            margin = 1.0
        else:
            second_score = thresholded[1]["similarity_score"]
            margin = (
                top_candidate["similarity_score"]
                - second_score
            )

        if margin < self.margin_threshold:
            return {
                "text": text,
                "candidates": thresholded,
                "status": "AMBIGUOUS",
                "reason": "INSUFFICIENT_SIMILARITY_MARGIN",
                "top_score": top_candidate["similarity_score"],
                "second_score": second_score,
                "margin": margin,
            }

        return {
            "text": text,
            "candidates": thresholded,
            "status": "CANDIDATE_AVAILABLE",
            "reason": "PASSED_SEMANTIC_GATE",
            "top_score": top_candidate["similarity_score"],
            "second_score": second_score,
            "margin": margin,
        }