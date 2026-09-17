"""Local embedding service for CrisisState."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Lazy-loaded CPU-first embedding service.

    Production model:
    sentence-transformers/all-MiniLM-L6-v2

    Embeddings are returned as float32 NumPy arrays and L2-normalized.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EXPECTED_DIMENSION = 384
    DEFAULT_DEVICE = "cpu"
    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model_factory: Callable[..., SentenceTransformer] | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")

        if not device.strip():
            raise ValueError("device must not be empty")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model_factory = model_factory or SentenceTransformer
        self._model: SentenceTransformer | None = None

    @property
    def model_loaded(self) -> bool:
        """Return whether the embedding model has been loaded."""
        return self._model is not None

    def _get_model(self) -> SentenceTransformer:
        """Load the model exactly once, on first use."""
        if self._model is None:
            self._model = self._model_factory(
                self.model_name,
                device=self.device,
            )

        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Embed one text string into a normalized 384-dimensional vector."""
        self._validate_text(text)

        embeddings = self.embed_many([text])
        return embeddings[0]

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        """Embed multiple text strings into a normalized matrix."""
        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a list or tuple of strings")

        for text in texts:
            self._validate_text(text)

        if not texts:
            return np.empty((0, self.EXPECTED_DIMENSION), dtype=np.float32)

        model = self._get_model()

        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        array = np.asarray(embeddings, dtype=np.float32)

        if array.ndim != 2:
            raise ValueError(
                f"Expected a 2D embedding matrix, got shape {array.shape}"
            )

        if array.shape[1] != self.EXPECTED_DIMENSION:
            raise ValueError(
                f"Expected {self.EXPECTED_DIMENSION}-dimensional embeddings, "
                f"got {array.shape[1]}"
            )

        return array

    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text must not be empty")