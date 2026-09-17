"""SQLite-backed embedding cache for CrisisState."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class EmbeddingCache:
    """Persistent cache for deterministic text embeddings.

    Cache identity is based on:
    - model name
    - normalized text

    Embeddings are stored as float32 BLOBs in SQLite.
    """

    EXPECTED_DIMENSION = 384

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    dtype TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(model_name, normalized_text)
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embedding_cache_lookup
                ON embedding_cache(model_name, normalized_text)
                """
            )

    @staticmethod
    def _make_key(model_name: str, normalized_text: str) -> str:
        payload = f"{model_name}\0{normalized_text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(
        self,
        model_name: str,
        normalized_text: str,
    ) -> np.ndarray | None:
        """Return a cached embedding, or None on a cache miss."""
        self._validate_inputs(model_name, normalized_text)

        cache_key = self._make_key(model_name, normalized_text)

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT dimension, dtype, embedding
                FROM embedding_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()

        if row is None:
            return None

        if row["dtype"] != "float32":
            raise ValueError(
                f"Unsupported cached dtype: {row['dtype']}"
            )

        dimension = int(row["dimension"])

        if dimension != self.EXPECTED_DIMENSION:
            raise ValueError(
                f"Unsupported cached dimension: {dimension}"
            )

        embedding = np.frombuffer(
            row["embedding"],
            dtype=np.float32,
        ).copy()

        if embedding.shape != (dimension,):
            raise ValueError(
                f"Cached embedding has invalid shape: {embedding.shape}"
            )

        return embedding

    def put(
        self,
        model_name: str,
        normalized_text: str,
        embedding: np.ndarray,
    ) -> None:
        """Store or replace an embedding in the cache."""
        self._validate_inputs(model_name, normalized_text)

        array = np.asarray(embedding, dtype=np.float32)

        if array.shape != (self.EXPECTED_DIMENSION,):
            raise ValueError(
                f"Expected embedding shape "
                f"({self.EXPECTED_DIMENSION},), got {array.shape}"
            )

        cache_key = self._make_key(model_name, normalized_text)
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO embedding_cache (
                    cache_key,
                    model_name,
                    normalized_text,
                    dimension,
                    dtype,
                    embedding,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    embedding = excluded.embedding,
                    dimension = excluded.dimension,
                    dtype = excluded.dtype,
                    created_at = excluded.created_at
                """,
                (
                    cache_key,
                    model_name,
                    normalized_text,
                    self.EXPECTED_DIMENSION,
                    "float32",
                    array.tobytes(),
                    created_at,
                ),
            )

    def delete(
        self,
        model_name: str,
        normalized_text: str,
    ) -> bool:
        """Delete one cached embedding.

        Returns True when an entry was deleted.
        """
        self._validate_inputs(model_name, normalized_text)

        cache_key = self._make_key(model_name, normalized_text)

        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM embedding_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            )

        return cursor.rowcount > 0

    def clear(self) -> None:
        """Remove all cached embeddings."""
        with self._connect() as connection:
            connection.execute("DELETE FROM embedding_cache")

    def count(self) -> int:
        """Return the number of cached embeddings."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM embedding_cache"
            ).fetchone()

        return int(row["count"])

    @staticmethod
    def _validate_inputs(
        model_name: str,
        normalized_text: str,
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string")

        if not model_name.strip():
            raise ValueError("model_name must not be empty")

        if not isinstance(normalized_text, str):
            raise TypeError("normalized_text must be a string")

        if not normalized_text.strip():
            raise ValueError("normalized_text must not be empty")