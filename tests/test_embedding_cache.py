import numpy as np
import pytest

from crisisstate.semantic.embedding_cache import EmbeddingCache


MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def make_embedding(index: int = 0) -> np.ndarray:
    embedding = np.zeros(384, dtype=np.float32)
    embedding[index] = 1.0
    return embedding


def test_cache_miss_returns_none(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    assert cache.get(MODEL, "road is blocked") is None
    assert cache.count() == 0


def test_put_and_get_round_trip(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    embedding = make_embedding(3)

    cache.put(MODEL, "road is blocked", embedding)

    result = cache.get(MODEL, "road is blocked")

    assert result is not None
    assert result.dtype == np.float32
    assert result.shape == (384,)
    assert np.array_equal(result, embedding)
    assert cache.count() == 1


def test_same_text_and_model_is_replaced(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    first = make_embedding(1)
    second = make_embedding(2)

    cache.put(MODEL, "road is blocked", first)
    cache.put(MODEL, "road is blocked", second)

    result = cache.get(MODEL, "road is blocked")

    assert result is not None
    assert np.array_equal(result, second)
    assert cache.count() == 1


def test_different_models_do_not_collide(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    model_a = "model-a"
    model_b = "model-b"

    embedding_a = make_embedding(4)
    embedding_b = make_embedding(5)

    cache.put(model_a, "road is blocked", embedding_a)
    cache.put(model_b, "road is blocked", embedding_b)

    assert np.array_equal(
        cache.get(model_a, "road is blocked"),
        embedding_a,
    )

    assert np.array_equal(
        cache.get(model_b, "road is blocked"),
        embedding_b,
    )

    assert cache.count() == 2


def test_delete_existing_entry(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    cache.put(MODEL, "road is blocked", make_embedding(6))

    assert cache.delete(MODEL, "road is blocked") is True
    assert cache.get(MODEL, "road is blocked") is None
    assert cache.count() == 0


def test_delete_missing_entry_returns_false(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    assert cache.delete(MODEL, "missing text") is False


def test_rejects_wrong_embedding_dimension(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    invalid = np.zeros(128, dtype=np.float32)

    with pytest.raises(ValueError):
        cache.put(MODEL, "road is blocked", invalid)


def test_cache_persists_across_instances(tmp_path):
    db_path = tmp_path / "embeddings.db"

    first_cache = EmbeddingCache(db_path)
    embedding = make_embedding(7)

    first_cache.put(MODEL, "flood water", embedding)

    second_cache = EmbeddingCache(db_path)

    result = second_cache.get(MODEL, "flood water")

    assert result is not None
    assert np.array_equal(result, embedding)
    assert second_cache.count() == 1