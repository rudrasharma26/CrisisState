"""Adapter between available exemplar-search APIs and M4."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


class ClaimExemplarSearchAdapter:
    """Expose one stable search(text, top_k) interface for M4."""

    def __init__(
        self,
        search_engine: Any,
        exemplars: Sequence[Dict[str, Any]],
    ) -> None:
        if not exemplars:
            raise ValueError("exemplars must not be empty")

        self.search_engine = search_engine
        self.exemplars = [dict(item) for item in exemplars]

    def search(
        self,
        text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        # Support both search APIs used during M2 development:
        # 1. search(text, candidates, k=...)
        # 2. search(text, top_k=...)
        try:
            raw_results = self.search_engine.search(
                text,
                self.exemplars,
                k=top_k,
            )
        except TypeError:
            raw_results = self.search_engine.search(
                text,
                top_k=top_k,
            )

        results: List[Dict[str, Any]] = []

        for rank, result in enumerate(raw_results, start=1):
            exemplar_id = result.get(
                "id",
                result.get("exemplar_id"),
            )

            score = result.get(
                "score",
                result.get("similarity_score"),
            )

            if exemplar_id is None:
                raise ValueError("Search result has no exemplar ID")

            if score is None:
                raise ValueError(
                    "Search result has no similarity score"
                )

            results.append(
                {
                    "rank": rank,
                    "exemplar_id": exemplar_id,
                    "claim_type": result["claim_type"],
                    "value": result["value"],
                    "text": result["text"],
                    "similarity_score": float(score),
                }
            )

        return results