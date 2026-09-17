"""Semantic search over the CrisisState Claim Exemplar Bank."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from crisisstate.semantic.embedding_cache import EmbeddingCache
from crisisstate.semantic.embedding_service import EmbeddingService
from crisisstate.semantic.text_normalizer import TextNormalizer


class ExemplarSearch:
    """Retrieve semantically similar claim exemplars.

    This component only generates candidates.
    It does not perform deterministic adjudication.
    """

    def __init__(
        self,
        exemplars: Sequence[Dict[str, Any]],
        embedding_service: EmbeddingService,
        embedding_cache: EmbeddingCache,
        text_normalizer: TextNormalizer | None = None,
    ) -> None:
        self.exemplars = list(exemplars)
        self.embedding_service = embedding_service
        self.embedding_cache = embedding_cache
        self.text_normalizer = text_normalizer or TextNormalizer()

        self._validate_exemplars()

        self._embedding_matrix: np.ndarray | None = None
        self._indexed_exemplar_ids: List[str] = []

    def _validate_exemplars(self) -> None:
        if not self.exemplars:
            raise ValueError("exemplars must not be empty")

        ids = []

        for exemplar in self.exemplars:
            required = {"id", "claim_type", "value", "text"}

            if not required.issubset(exemplar):
                missing = required - set(exemplar)
                raise ValueError(
                    f"Exemplar is missing required fields: {sorted(missing)}"
                )

            if not isinstance(exemplar["text"], str) or not exemplar["text"].strip():
                raise ValueError(
                    f"Exemplar {exemplar['id']} has invalid text"
                )

            ids.append(exemplar["id"])

        if len(ids) != len(set(ids)):
            raise ValueError("Exemplar IDs must be unique")

    def _ensure_index(self) -> None:
        """Ensure all exemplar embeddings are available in memory."""
        exemplar_ids = [exemplar["id"] for exemplar in self.exemplars]

        if (
            self._embedding_matrix is not None
            and self._indexed_exemplar_ids == exemplar_ids
        ):
            return

        normalized_texts = [
            self.text_normalizer.normalize(exemplar["text"])
            for exemplar in self.exemplars
        ]

        embeddings: List[np.ndarray | None] = [None] * len(self.exemplars)
        missing_texts: List[str] = []
        missing_positions: List[int] = []

        for index, normalized_text in enumerate(normalized_texts):
            cached = self.embedding_cache.get(
                self.embedding_service.model_name,
                normalized_text,
            )

            if cached is None:
                missing_texts.append(normalized_text)
                missing_positions.append(index)
            else:
                embeddings[index] = cached

        if missing_texts:
            generated = self.embedding_service.embed_many(missing_texts)

            for position, embedding in zip(missing_positions, generated):
                normalized_text = normalized_texts[position]

                self.embedding_cache.put(
                    self.embedding_service.model_name,
                    normalized_text,
                    embedding,
                )

                embeddings[position] = embedding

        if any(embedding is None for embedding in embeddings):
            raise RuntimeError("Failed to build complete exemplar embedding index")

        self._embedding_matrix = np.vstack(
            [embedding for embedding in embeddings if embedding is not None]
        ).astype(np.float32)

        self._indexed_exemplar_ids = exemplar_ids

    def search(
        self,
        text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the top-k semantically similar exemplars."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        self._ensure_index()

        normalized_text = self.text_normalizer.normalize(text)

        query_embedding = self.embedding_cache.get(
            self.embedding_service.model_name,
            normalized_text,
        )

        if query_embedding is None:
            query_embedding = self.embedding_service.embed(normalized_text)

            self.embedding_cache.put(
                self.embedding_service.model_name,
                normalized_text,
                query_embedding,
            )

        assert self._embedding_matrix is not None

        scores = self._embedding_matrix @ query_embedding

        limit = min(top_k, len(self.exemplars))
        ranked_indices = np.argsort(-scores, kind="stable")[:limit]

        results: List[Dict[str, Any]] = []

        for rank, index in enumerate(ranked_indices, start=1):
            exemplar = self.exemplars[int(index)]

            results.append(
                {
                    "rank": rank,
                    "exemplar_id": exemplar["id"],
                    "claim_type": exemplar["claim_type"],
                    "value": exemplar["value"],
                    "text": exemplar["text"],
                    "similarity_score": float(scores[index]),
                }
            )

        return results