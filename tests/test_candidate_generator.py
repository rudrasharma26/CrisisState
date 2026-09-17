import pytest

from crisisstate.semantic.candidate_generator import SemanticCandidateGenerator


class FakeSearch:
    def __init__(self, results):
        self.results = results

    def search(self, text, top_k=5):
        return self.results[:top_k]


def candidate(
    rank,
    exemplar_id,
    claim_type,
    value,
    score,
):
    return {
        "rank": rank,
        "exemplar_id": exemplar_id,
        "claim_type": claim_type,
        "value": value,
        "text": f"{value} example",
        "similarity_score": score,
    }


def test_candidate_passes_semantic_gate():
    search = FakeSearch(
        [
            candidate(1, "exm_1", "ROAD_ACCESS", "IMPASSABLE", 0.80),
            candidate(2, "exm_2", "ROAD_ACCESS", "IMPASSABLE", 0.70),
        ]
    )

    generator = SemanticCandidateGenerator(search)

    result = generator.generate("road blocked")

    assert result["status"] == "CANDIDATE_AVAILABLE"
    assert result["reason"] == "PASSED_SEMANTIC_GATE"
    assert result["margin"] == pytest.approx(0.10)
    assert len(result["candidates"]) == 2


def test_below_threshold_is_rejected():
    search = FakeSearch(
        [
            candidate(1, "exm_1", "ROAD_ACCESS", "IMPASSABLE", 0.60),
        ]
    )

    generator = SemanticCandidateGenerator(search)

    result = generator.generate("road blocked")

    assert result["status"] == "REJECTED"
    assert result["reason"] == "BELOW_SIMILARITY_THRESHOLD"


def test_small_margin_is_ambiguous():
    search = FakeSearch(
        [
            candidate(1, "exm_1", "ROAD_ACCESS", "IMPASSABLE", 0.80),
            candidate(2, "exm_2", "ROAD_ACCESS", "PASSABLE", 0.78),
        ]
    )

    generator = SemanticCandidateGenerator(search)

    result = generator.generate("road status")

    assert result["status"] == "AMBIGUOUS"
    assert result["reason"] == "INSUFFICIENT_SIMILARITY_MARGIN"
    assert result["margin"] == pytest.approx(0.02)


def test_preferred_claim_type_filters_candidates():
    search = FakeSearch(
        [
            candidate(1, "exm_1", "TRAFFIC_STATUS", "STOPPED", 0.90),
            candidate(2, "exm_2", "ROAD_ACCESS", "IMPASSABLE", 0.82),
        ]
    )

    generator = SemanticCandidateGenerator(search)

    result = generator.generate(
        "road blocked",
        preferred_claim_type="ROAD_ACCESS",
    )

    assert result["status"] == "CANDIDATE_AVAILABLE"
    assert all(
        item["claim_type"] == "ROAD_ACCESS"
        for item in result["candidates"]
    )


def test_cross_type_candidates_can_be_rejected_by_preferred_type():
    search = FakeSearch(
        [
            candidate(1, "exm_1", "TRAFFIC_STATUS", "STOPPED", 0.90),
        ]
    )

    generator = SemanticCandidateGenerator(search)

    result = generator.generate(
        "road blocked",
        preferred_claim_type="ROAD_ACCESS",
    )

    assert result["status"] == "REJECTED"
    assert result["reason"] == "BELOW_SIMILARITY_THRESHOLD"


def test_no_results():
    search = FakeSearch([])

    generator = SemanticCandidateGenerator(search)

    result = generator.generate("unknown")

    assert result["status"] == "NO_CANDIDATES"
    assert result["reason"] == "NO_EXEMPLARS_RETURNED"


def test_invalid_thresholds():
    search = FakeSearch([])

    with pytest.raises(ValueError):
        SemanticCandidateGenerator(
            search,
            similarity_threshold=1.1,
        )

    with pytest.raises(ValueError):
        SemanticCandidateGenerator(
            search,
            margin_threshold=-0.1,
        )