"""Embedding infrastructure for CrisisState — Phase 2, Milestone M2.

This module provides three cooperating components:

    TextNormalizer   — deterministic, semantics-preserving text normalization
    EmbeddingService — lazy model loading, batch embedding, cosine utilities
    EmbeddingCache   — SQLite-backed, SHA-256-keyed embedding cache
    ExemplarSearch   — reusable ranked-similarity search over candidate texts

CRITICAL: This module is infrastructure only.
It is NOT imported by any Phase 1 production module (pipeline, extractor,
matcher, contradiction, attention).  No embedding is used to make any
production decision in M2.  Phase 1 behavior is fully unaffected by the
existence of this file.

Model
-----
    sentence-transformers/all-MiniLM-L6-v2
    Output dimensionality: 384
    Embedding type: float32 L2-normalised dense vector

CPU Determinism
---------------
    torch.set_num_threads(1) is applied at service initialisation to minimise
    intra-op parallelism non-determinism on CPU.  Results are reproducible
    within NumPy floating-point tolerance (~1e-6) across repeated runs on the
    same hardware and OS.  Exact bit-equality is not guaranteed across
    different PyTorch/BLAS builds.  Tests use atol=1e-5 / rtol=1e-5.
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

#: Pinned model identifier — never silently substituted.
MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"

#: Output vector dimensionality for all-MiniLM-L6-v2.
VECTOR_DIM: int = 384

#: Separator used inside the cache key hash input.
_KEY_SEP: str = "||"


# ── TextNormalizer ────────────────────────────────────────────────────────────


class TextNormalizer:
    """Deterministic, semantics-preserving text normalization.

    Performs only safe transformations that improve embedding consistency
    without altering semantic content.  Specifically, negation, qualifiers,
    temporal expressions, and quantities are intentionally preserved so that
    the following examples remain distinguishable after normalisation:

        "road is blocked"
        "road is not blocked"
        "road may be blocked"
        "only emergency vehicles can pass"

    Transformations applied:
        1. Unicode NFC normalization (canonical decomposition + composition)
        2. Whitespace normalization (tabs, newlines → spaces; collapse runs)
        3. Lowercase (for embedding-model stability; MiniLM is uncased)
        4. Strip leading/trailing whitespace

    Transformations deliberately NOT applied:
        - Stopword removal
        - Stemming / lemmatization
        - Punctuation removal
        - Expansion of abbreviations
    """

    def normalize(self, text: str) -> str:
        """Return the canonically normalised form of *text*.

        Args:
            text: Raw input string.  May be empty.

        Returns:
            Normalised string.  Empty input returns an empty string.
        """
        if not text:
            return ""
        # 1. Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)
        # 2. Collapse all whitespace variants into a single space
        text = " ".join(text.split())
        # 3. Lowercase
        text = text.lower()
        # 4. Strip (redundant after step 2 but explicit for clarity)
        text = text.strip()
        return text


# ── EmbeddingService ──────────────────────────────────────────────────────────


class EmbeddingService:
    """Provides text embedding using a pinned sentence-transformers model.

    The SentenceTransformer model is loaded lazily — only on the first call
    to :meth:`embed` or :meth:`embed_batch`.  Importing this module does not
    trigger a model download or GPU/CPU allocation.

    Embeddings are L2-normalised float32 vectors of shape ``(VECTOR_DIM,)``
    (384 for ``all-MiniLM-L6-v2``).  L2 normalisation means that cosine
    similarity reduces to a simple dot product, which simplifies downstream
    ranking.

    CPU determinism is improved by calling ``torch.set_num_threads(1)`` once
    during initialisation, suppressing intra-op thread pool randomness.

    Args:
        model_id: Model identifier to load.  Defaults to the pinned
            ``MODEL_ID`` constant.  Changing this value will cause the cache
            to produce separate entries (different key prefix) so stale
            vectors are never silently reused.
        normalizer: Text normalizer instance.  If None, a default
            :class:`TextNormalizer` is created.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        normalizer: Optional[TextNormalizer] = None,
    ) -> None:
        self.model_id = model_id
        self.vector_dim = VECTOR_DIM
        self.normalizer = normalizer or TextNormalizer()
        self._model: Any = None  # loaded lazily

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the sentence-transformers model (once)."""
        if self._model is not None:
            return
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers and torch are required for M2 embedding "
                "infrastructure.  Install them with:\n"
                "    pip install sentence-transformers torch"
            ) from exc

        logger.info("Loading embedding model: %s", self.model_id)
        # Restrict intra-op threads for CPU reproducibility
        torch.set_num_threads(1)
        self._model = SentenceTransformer(self.model_id)
        logger.info("Embedding model loaded. dim=%d", VECTOR_DIM)

    def _normalize_text(self, text: str) -> str:
        return self.normalizer.normalize(text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Input text.  May be empty — returns a zero vector.

        Returns:
            L2-normalised float32 vector of shape ``(VECTOR_DIM,)``.
        """
        self._load_model()
        normalized = self._normalize_text(text)
        if not normalized:
            return np.zeros(VECTOR_DIM, dtype=np.float32)
        vec = self._model.encode(
            [normalized],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0].astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings in one forward pass.

        Args:
            texts: List of input texts.  Empty strings produce zero vectors.

        Returns:
            Float32 array of shape ``(len(texts), VECTOR_DIM)``.
            Row order matches *texts* order.
        """
        if not texts:
            return np.zeros((0, VECTOR_DIM), dtype=np.float32)
        self._load_model()
        # Track which indices are empty to avoid encoding them
        non_empty_indices = [i for i, t in enumerate(texts) if self._normalize_text(t)]
        result = np.zeros((len(texts), VECTOR_DIM), dtype=np.float32)
        if not non_empty_indices:
            return result
        normalized_texts = [self._normalize_text(texts[i]) for i in non_empty_indices]
        vecs = self._model.encode(
            normalized_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        ).astype(np.float32)
        for out_idx, src_idx in enumerate(non_empty_indices):
            result[src_idx] = vecs[out_idx]
        return result

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Return the cosine similarity between two vectors.

        For L2-normalised vectors this is equivalent to their dot product.
        Both vectors are re-normalised inside this function to guard against
        non-normalised inputs.

        Args:
            a: Vector of shape ``(d,)``.
            b: Vector of shape ``(d,)``.

        Returns:
            Scalar in ``[-1.0, 1.0]``.  Returns ``0.0`` if either vector
            has zero norm.
        """
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def top_k(
        query: np.ndarray,
        candidates: List[np.ndarray],
        k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Return the top-*k* most similar candidate vectors.

        Args:
            query: Query vector of shape ``(d,)``.
            candidates: List of candidate vectors, each of shape ``(d,)``.
            k: Maximum number of results to return.

        Returns:
            List of ``(index, score)`` tuples sorted by descending similarity.
            Length is ``min(k, len(candidates))``.
        """
        if not candidates:
            return []
        scores: List[Tuple[int, float]] = []
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return [(i, 0.0) for i in range(min(k, len(candidates)))]
        q = query / query_norm
        for i, c in enumerate(candidates):
            c_norm = float(np.linalg.norm(c))
            if c_norm == 0.0:
                scores.append((i, 0.0))
            else:
                scores.append((i, float(np.dot(q, c / c_norm))))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[: k]

    def _get_model(self) -> Any:
        """Helper to trigger lazy load for testing/inspection."""
        self._load_model()
        return self._model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Convenience top-level wrapper for :meth:`EmbeddingService.cosine_similarity`."""
    return EmbeddingService.cosine_similarity(a, b)



# ── EmbeddingCache ────────────────────────────────────────────────────────────


class EmbeddingCache:
    """SQLite-backed embedding cache keyed by (model_id, normalised_text).

    Cache key
    ---------
    SHA-256( model_id + ``||`` + normalised_text )

    This design means:
    - Same text with same model → same key → cached vector returned.
    - Different text → different key → cache miss.
    - Different model_id → different key prefix → stale vectors never reused.

    Storage
    -------
    Reuses the existing CrisisState SQLite ``Database`` infrastructure.
    The ``embedding_cache`` table is created by the M2 schema migration.
    An in-memory ``Database(":memory:")`` is also fully supported.

    Args:
        db: ``Database`` instance to use.  If None, creates a fresh
            in-memory database (useful for testing).
        model_id: Model identifier used as part of the cache key.
        normalizer: TextNormalizer instance used to canonicalize text before
            hashing.
    """

    def __init__(
        self,
        db: Any,  # crisisstate.storage.database.Database
        model_id: str = MODEL_ID,
        normalizer: Optional[TextNormalizer] = None,
    ) -> None:
        self.db = db
        self.model_id = model_id
        self.normalizer = normalizer or TextNormalizer()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Key computation
    # ------------------------------------------------------------------

    def cache_key(self, text: str) -> str:
        """Compute the deterministic cache key for *text*.

        Args:
            text: Raw (un-normalised) input text.

        Returns:
            64-character lowercase hex SHA-256 digest.
        """
        normalized = self.normalizer.normalize(text)
        payload = self.model_id + _KEY_SEP + normalized
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Cache operations
    # ------------------------------------------------------------------

    def get(self, text: str) -> Optional[np.ndarray]:
        """Retrieve a cached embedding for *text*, or None on cache miss.

        Args:
            text: Input text (will be normalised internally).

        Returns:
            Float32 ndarray of shape ``(VECTOR_DIM,)`` or ``None``.
        """
        key = self.cache_key(text)
        conn = self.db.get_connection()
        cursor = conn.execute(
            "SELECT vector FROM embedding_cache WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row is None:
            self._misses += 1
            return None
        self._hits += 1
        vector_list = json.loads(row["vector"])
        return np.array(vector_list, dtype=np.float32)

    def put(self, text: str, vector: np.ndarray) -> None:
        """Store an embedding in the cache.

        Existing entries for the same key are silently replaced.

        Args:
            text: Input text (will be normalised for storage).
            vector: Float32 ndarray of shape ``(VECTOR_DIM,)``.
        """
        key = self.cache_key(text)
        normalized = self.normalizer.normalize(text)
        vector_json = json.dumps(vector.tolist())
        created_at = datetime.now(timezone.utc).isoformat()
        conn = self.db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embedding_cache
                (key, model_id, normalized_text, vector, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, self.model_id, normalized, vector_json, created_at),
            )

    def get_stats(self) -> Dict[str, int]:
        """Return cache hit/miss statistics and total count for this model."""
        conn = self.db.get_connection()
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM embedding_cache WHERE model_id = ?",
            (self.model_id,),
        )
        row = cursor.fetchone()
        total = row["count"] if row else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_cached": total,
        }

    def clear_model(self) -> int:
        """Delete all cache entries for this model_id.

        Returns:
            Number of rows deleted.
        """
        conn = self.db.get_connection()
        with conn:
            cursor = conn.execute(
                "DELETE FROM embedding_cache WHERE model_id = ?", (self.model_id,)
            )
        return cursor.rowcount


# ── ExemplarSearch ────────────────────────────────────────────────────────────


class ExemplarSearch:
    """Ranked similarity search over a small collection of candidate texts.

    This class is the interface through which future milestones will query the
    ClaimExemplar bank.  In M2 the exemplar bank is empty; the search
    machinery is fully implemented and tested.

    Important: This class DOES NOT connect to Claim extraction, incident
    matching, or any other production pipeline component.  It is invoked only
    by tests and future M3+ semantic extraction logic.

    Args:
        service: :class:`EmbeddingService` to use for embedding.
        cache: Optional :class:`EmbeddingCache`.  If provided, both query
            and candidate embeddings are looked up / stored in the cache.
        k: Default number of top results to return.
    """

    def __init__(
        self,
        service: EmbeddingService,
        cache: Optional[EmbeddingCache] = None,
        k: int = 5,
    ) -> None:
        self.service = service
        self.cache = cache
        self.k = k

    def _get_embedding(self, text: str) -> np.ndarray:
        """Return embedding for *text*, consulting cache if available."""
        if self.cache is not None:
            cached = self.cache.get(text)
            if cached is not None:
                return cached
        vec = self.service.embed(text)
        if self.cache is not None:
            self.cache.put(text, vec)
        return vec

    def search(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-*k* candidates ranked by cosine similarity to *query_text*.

        Args:
            query_text: Text to search for.
            candidates: List of dicts, each containing at minimum:
                - ``"id"`` (str): unique identifier
                - ``"text"`` (str): text to compare against
                Additional keys are preserved in the output.
            k: Number of top results.  Defaults to ``self.k``.

        Returns:
            List of result dicts (copy of candidate dict + ``"score"`` key),
            sorted by descending similarity.  Empty list if *candidates* is
            empty or *query_text* is empty.
        """
        k = k or self.k
        if not candidates:
            return []
        if not query_text or not query_text.strip():
            return [dict(c, score=0.0) for c in candidates[:k]]

        query_vec = self._get_embedding(query_text)
        candidate_vecs = [self._get_embedding(c["text"]) for c in candidates]

        ranked = self.service.top_k(query_vec, candidate_vecs, k=k)
        results: List[Dict[str, Any]] = []
        for idx, score in ranked:
            result = dict(candidates[idx])
            result["score"] = round(float(score), 6)
            results.append(result)
        return results
