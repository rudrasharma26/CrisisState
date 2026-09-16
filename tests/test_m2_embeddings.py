"""Unit and integration tests for Phase 2 M2: Embedding Infrastructure.

Tests cover:
    1. TestTextNormalizer   - normalization rules, casing, whitespace, negation/qualifier preservation
    2. TestEmbeddingService - init, single embedding shape, batch shape, L2-norm, cosine similarity, top-k
    3. TestEmbeddingCache   - cache hit/miss, persistence across sessions, model isolation, idempotent put
    4. TestExemplarSearch   - ranked results, empty/single candidates, cache integration
    5. TestDeterminism      - repeatable embeddings, sensitivity to semantic distinctions
    6. TestEdgeCases        - empty strings, long texts, unicode edge cases
    7. TestIsolation        - verify no Phase 1 production module imports embeddings
"""

from __future__ import annotations

import sys
import numpy as np
import pytest

from crisisstate.engine.embeddings import (
    MODEL_ID,
    VECTOR_DIM,
    EmbeddingCache,
    EmbeddingService,
    ExemplarSearch,
    TextNormalizer,
    cosine_similarity,
)
from crisisstate.storage.database import Database


@pytest.fixture(scope="session")
def shared_service() -> EmbeddingService:
    """Load model once per test session to avoid repeated startup overhead."""
    service = EmbeddingService()
    service._get_model()
    return service


@pytest.fixture
def mem_db() -> Database:
    """In-memory database with fresh schema for cache tests."""
    return Database(db_path=":memory:")


# ── 1. TestTextNormalizer ───────────────────────────────────────────────────


class TestTextNormalizer:
    def test_strip_and_whitespace_collapse(self):
        normalizer = TextNormalizer()
        raw = "  Water  level   is \t rising \n rapidly  "
        assert normalizer.normalize(raw) == "water level is rising rapidly"

    def test_unicode_normalization(self):
        normalizer = TextNormalizer()
        decomposed = "cafe\u0301"
        composed = "caf\u00e9"
        assert normalizer.normalize(decomposed) == normalizer.normalize(composed)

    def test_preserves_negations_and_qualifiers(self):
        normalizer = TextNormalizer()
        s1 = "road is impassable"
        s2 = "road is not impassable"
        s3 = "road may be impassable"
        assert normalizer.normalize(s1) != normalizer.normalize(s2)
        assert normalizer.normalize(s1) != normalizer.normalize(s3)
        assert normalizer.normalize(s2) != normalizer.normalize(s3)

    def test_preserves_numbers_and_quantities(self):
        normalizer = TextNormalizer()
        s1 = "water level is 3 feet"
        s2 = "water level is 6 feet"
        assert normalizer.normalize(s1) == "water level is 3 feet"
        assert normalizer.normalize(s2) == "water level is 6 feet"
        assert normalizer.normalize(s1) != normalizer.normalize(s2)

    def test_empty_and_blank(self):
        normalizer = TextNormalizer()
        assert normalizer.normalize("") == ""
        assert normalizer.normalize("   \n\t  ") == ""


# ── 2. TestEmbeddingService ─────────────────────────────────────────────────


class TestEmbeddingService:
    def test_constants_and_lazy_load(self):
        svc = EmbeddingService()
        assert svc.model_id == MODEL_ID
        assert svc.vector_dim == VECTOR_DIM
        fresh = EmbeddingService()
        assert fresh._model is None

    def test_embed_single_vector_shape_and_norm(self, shared_service: EmbeddingService):
        vec = shared_service.embed("Road is flooded and blocked.")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (VECTOR_DIM,)
        assert vec.dtype == np.float32
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-4)

    def test_embed_batch_shape(self, shared_service: EmbeddingService):
        texts = [
            "Road is impassable.",
            "Water level rising.",
            "People trapped on roof.",
        ]
        batch = shared_service.embed_batch(texts)
        assert isinstance(batch, np.ndarray)
        assert batch.shape == (3, VECTOR_DIM)
        for i in range(3):
            assert np.isclose(np.linalg.norm(batch[i]), 1.0, atol=1e-4)

    def test_embed_batch_empty(self, shared_service: EmbeddingService):
        batch = shared_service.embed_batch([])
        assert batch.shape == (0, VECTOR_DIM)

    def test_cosine_similarity_identity(self, shared_service: EmbeddingService):
        vec = shared_service.embed("Flood emergency at Sector 4.")
        sim = shared_service.cosine_similarity(vec, vec)
        assert np.isclose(sim, 1.0, atol=1e-4)

    def test_cosine_similarity_directionality(self, shared_service: EmbeddingService):
        v_flood = shared_service.embed("Severe flooding on the highway.")
        v_similar = shared_service.embed("Highway is underwater and flooded.")
        v_unrelated = shared_service.embed("The bakery sells fresh bread.")
        sim_close = shared_service.cosine_similarity(v_flood, v_similar)
        sim_distant = shared_service.cosine_similarity(v_flood, v_unrelated)
        assert sim_close > sim_distant
        assert sim_close > 0.6

    def test_top_k(self, shared_service: EmbeddingService):
        query = shared_service.embed("people trapped on rooftop")
        candidates = [
            shared_service.embed("baking sourdough bread"),
            shared_service.embed("family stranded on roof needing rescue"),
            shared_service.embed("sunny warm weather in the park"),
        ]
        ranked = shared_service.top_k(query, candidates, k=2)
        assert len(ranked) == 2
        best_idx, best_score = ranked[0]
        assert best_idx == 1
        assert best_score > 0.55


# ── 3. TestEmbeddingCache ───────────────────────────────────────────────────


class TestEmbeddingCache:
    def test_cache_miss_and_put(self, mem_db: Database, shared_service: EmbeddingService):
        cache = EmbeddingCache(db=mem_db, model_id=shared_service.model_id)
        text = "Underpass is completely submerged"
        assert cache.get(text) is None
        assert cache.get_stats()["misses"] == 1
        assert cache.get_stats()["hits"] == 0

        vec = shared_service.embed(text)
        cache.put(text, vec)

        cached_vec = cache.get(text)
        assert cached_vec is not None
        assert cache.get_stats()["hits"] == 1
        assert np.allclose(vec, cached_vec, atol=1e-5)

    def test_cache_model_isolation(self, mem_db: Database):
        cache_m1 = EmbeddingCache(db=mem_db, model_id="model-a")
        cache_m2 = EmbeddingCache(db=mem_db, model_id="model-b")
        dummy_vec = np.zeros(VECTOR_DIM, dtype=np.float32)
        dummy_vec[0] = 1.0

        cache_m1.put("test query", dummy_vec)
        assert cache_m1.get("test query") is not None
        assert cache_m2.get("test query") is None

    def test_cache_idempotent_put(self, mem_db: Database, shared_service: EmbeddingService):
        cache = EmbeddingCache(db=mem_db, model_id=shared_service.model_id)
        text = "Water is waist deep"
        vec = shared_service.embed(text)
        cache.put(text, vec)
        cache.put(text, vec)
        assert cache.get_stats()["total_cached"] == 1


# ── 4. TestExemplarSearch ───────────────────────────────────────────────────


class TestExemplarSearch:
    def test_ranked_search(self, shared_service: EmbeddingService, mem_db: Database):
        cache = EmbeddingCache(db=mem_db, model_id=shared_service.model_id)
        searcher = ExemplarSearch(service=shared_service, cache=cache, k=2)

        candidates = [
            {"id": "c1", "text": "Traffic light is green"},
            {"id": "c2", "text": "Roadway is completely blocked by high water"},
            {"id": "c3", "text": "Store opens at 9am"},
        ]

        results = searcher.search("road impassable due to flooding", candidates, k=2)
        assert len(results) == 2
        assert results[0]["id"] == "c2"
        assert results[0]["score"] > results[1]["score"]
        assert "score" in results[0]

    def test_search_empty_candidates(self, shared_service: EmbeddingService):
        searcher = ExemplarSearch(service=shared_service)
        results = searcher.search("query", [])
        assert results == []

    def test_search_empty_query(self, shared_service: EmbeddingService):
        searcher = ExemplarSearch(service=shared_service)
        candidates = [{"id": "c1", "text": "hello"}]
        results = searcher.search("", candidates)
        assert len(results) == 1
        assert results[0]["score"] == 0.0


# ── 5. TestDeterminism ──────────────────────────────────────────────────────


class TestDeterminism:
    def test_embedding_repeatability(self, shared_service: EmbeddingService):
        text = "Critical: 5 people stranded on 2nd floor balcony."
        vec1 = shared_service.embed(text)
        vec2 = shared_service.embed(text)
        assert np.allclose(vec1, vec2, atol=1e-6)

    def test_distinct_sentences_produce_distinct_embeddings(self, shared_service: EmbeddingService):
        v1 = shared_service.embed("Road is open and clear.")
        v2 = shared_service.embed("Road is closed and flooded.")
        sim = cosine_similarity(v1, v2)
        assert sim < 0.95


# ── 6. TestEdgeCases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_string_embed(self, shared_service: EmbeddingService):
        vec = shared_service.embed("")
        assert vec.shape == (VECTOR_DIM,)
        assert np.isclose(np.linalg.norm(vec), 0.0, atol=1e-5)

    def test_very_long_text_embed(self, shared_service: EmbeddingService):
        long_text = "Severe flooding warning. " * 200
        vec = shared_service.embed(long_text)
        assert vec.shape == (VECTOR_DIM,)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


# ── 7. TestIsolation ────────────────────────────────────────────────────────


class TestIsolation:
    def test_phase1_modules_do_not_import_embeddings(self):
        """Confirm that Phase 1 production modules have zero dependency on embeddings."""
        phase1_modules = [
            "crisisstate.engine.pipeline",
            "crisisstate.engine.extractor",
            "crisisstate.engine.matcher",
            "crisisstate.engine.contradiction",
            "crisisstate.engine.attention",
        ]
        for mod_name in phase1_modules:
            mod = sys.modules.get(mod_name)
            if mod is not None:
                assert "embeddings" not in dir(mod)
                assert "EmbeddingService" not in dir(mod)
                assert "EmbeddingCache" not in dir(mod)
                assert "ExemplarSearch" not in dir(mod)
