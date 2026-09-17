import numpy as np
import pytest

from crisisstate.semantic.embedding_service import EmbeddingService


class FakeSentenceTransformer:
    """Deterministic fake model for unit testing."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device

    def encode(
        self,
        texts,
        batch_size,
        convert_to_numpy,
        normalize_embeddings,
        show_progress_bar,
    ):
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        assert show_progress_bar is False

        rows = []

        for text in texts:
            if text == "road is blocked":
                vector = np.zeros(384, dtype=np.float32)
                vector[0] = 1.0
            elif text == "flood water":
                vector = np.zeros(384, dtype=np.float32)
                vector[1] = 1.0
            else:
                vector = np.zeros(384, dtype=np.float32)
                vector[2] = 1.0

            rows.append(vector)

        return np.vstack(rows)


def test_model_is_lazy_loaded():
    calls = []

    def factory(model_name, device):
        calls.append((model_name, device))
        return FakeSentenceTransformer(model_name, device)

    service = EmbeddingService(model_factory=factory)

    assert service.model_loaded is False
    assert calls == []

    service.embed("road is blocked")

    assert service.model_loaded is True
    assert len(calls) == 1
    assert calls[0][0] == EmbeddingService.DEFAULT_MODEL_NAME
    assert calls[0][1] == "cpu"


def test_model_is_loaded_only_once():
    calls = []

    def factory(model_name, device):
        calls.append((model_name, device))
        return FakeSentenceTransformer(model_name, device)

    service = EmbeddingService(model_factory=factory)

    service.embed("road is blocked")
    service.embed("flood water")

    assert len(calls) == 1


def test_single_embedding_shape_dtype_and_normalization():
    service = EmbeddingService(
        model_factory=FakeSentenceTransformer,
    )

    embedding = service.embed("road is blocked")

    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(embedding), 1.0)


def test_batch_embedding_shape():
    service = EmbeddingService(
        model_factory=FakeSentenceTransformer,
    )

    embeddings = service.embed_many(
        [
            "road is blocked",
            "flood water",
        ]
    )

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32


def test_empty_batch_returns_empty_matrix():
    service = EmbeddingService(
        model_factory=FakeSentenceTransformer,
    )

    embeddings = service.embed_many([])

    assert embeddings.shape == (0, 384)
    assert embeddings.dtype == np.float32
    assert service.model_loaded is False


def test_rejects_non_string_input():
    service = EmbeddingService(
        model_factory=FakeSentenceTransformer,
    )

    with pytest.raises(TypeError):
        service.embed(None)


def test_rejects_empty_text():
    service = EmbeddingService(
        model_factory=FakeSentenceTransformer,
    )

    with pytest.raises(ValueError):
        service.embed("   ")


def test_rejects_invalid_batch_input():
    service = EmbeddingService(
        model_factory=FakeSentenceTransformer,
    )

    with pytest.raises(TypeError):
        service.embed_many("road is blocked")