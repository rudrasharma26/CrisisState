import numpy as np
import pytest

from crisisstate.semantic.embedding_cache import EmbeddingCache
from crisisstate.semantic.embedding_service import EmbeddingService
from crisisstate.semantic.exemplar_search import ExemplarSearch


class FakeSentenceTransformer:
    """Small deterministic fake embedding model."""

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
        vectors = []

        mapping = {
            "road blocked": 0,
            "road open": 1,
            "flood water": 2,
            "traffic stopped": 3,
        }

        for text in texts:
            vector = np.zeros(384, dtype=np.float32)
            vector[mapping.get(text, 0)] = 1.0
            vectors.append(vector)

        return np.vstack(vectors)


@pytest.fixture
def embedding_service():
    return EmbeddingService(
        model_factory=FakeSentenceTransformer,
    )


@pytest.fixture
def exemplars():
    return [
        {
            "id": "exm_1",
            "claim_type": "ROAD_ACCESS",
            "value": "IMPASSABLE",
            "text": "road blocked",
        },
        {
            "id": "exm_2",
            "claim_type": "ROAD_ACCESS",
            "value": "PASSABLE",
            "text": "road open",
        },
        {
            "id": "exm_3",
            "claim_type": "FLOODED",
            "value": "YES",
            "text": "flood water",
        },
        {
            "id": "exm_4",
            "claim_type": "TRAFFIC_STATUS",
            "value": "STOPPED",
            "text": "traffic stopped",
        },
    ]


def test_search_returns_ranked_results(
    tmp_path,
    embedding_service,
    exemplars,
):
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    search = ExemplarSearch(
        exemplars,
        embedding_service,
        cache,
    )

    results = search.search("road blocked", top_k=3)

    assert len(results) == 3
    assert results[0]["exemplar_id"] == "exm_1"
    assert results[0]["similarity_score"] == pytest.approx(1.0)
    assert [result["rank"] for result in results] == [1, 2, 3]


def test_search_returns_descending_scores(
    tmp_path,
    embedding_service,
    exemplars,
):
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    search = ExemplarSearch(
        exemplars,
        embedding_service,
        cache,
    )

    results = search.search("road blocked", top_k=4)

    scores = [result["similarity_score"] for result in results]

    assert scores == sorted(scores, reverse=True)


def test_top_k_cannot_exceed_exemplar_count(
    tmp_path,
    embedding_service,
    exemplars,
):
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    search = ExemplarSearch(
        exemplars,
        embedding_service,
        cache,
    )

    results = search.search("road blocked", top_k=100)

    assert len(results) == len(exemplars)


def test_query_embedding_is_cached(
    tmp_path,
    embedding_service,
    exemplars,
):
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    search = ExemplarSearch(
        exemplars,
        embedding_service,
        cache,
    )

    search.search("road blocked", top_k=2)

    first_count = cache.count()

    search.search("road blocked", top_k=2)

    second_count = cache.count()

    assert first_count == second_count


def test_exemplar_embeddings_are_cached(
    tmp_path,
    embedding_service,
    exemplars,
):
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    search = ExemplarSearch(
        exemplars,
        embedding_service,
        cache,
    )

    assert cache.count() == 0

    search.search("road blocked query", top_k=1)

    # Four exemplar embeddings + one distinct query embedding.
    assert cache.count() == len(exemplars) + 1

def test_rejects_invalid_query(
    tmp_path,
    embedding_service,
    exemplars,
):
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    search = ExemplarSearch(
        exemplars,
        embedding_service,
        cache,
    )

    with pytest.raises(TypeError):
        search.search(None)

    with pytest.raises(ValueError):
        search.search("   ")


def test_rejects_invalid_top_k(
    tmp_path,
    embedding_service,
    exemplars,
):
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    search = ExemplarSearch(
        exemplars,
        embedding_service,
        cache,
    )

    with pytest.raises(ValueError):
        search.search("road blocked", top_k=0)


def test_empty_exemplar_bank_is_rejected(tmp_path, embedding_service):
    cache = EmbeddingCache(tmp_path / "embeddings.db")

    with pytest.raises(ValueError):
        ExemplarSearch(
            [],
            embedding_service,
            cache,
        )