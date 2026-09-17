import pytest

from crisisstate.semantic.adjudicator import SemanticAdjudicator


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


def test_accepts_valid_semantic_candidate():
    adjudicator = SemanticAdjudicator()

    semantic_result = {
        "status": "CANDIDATE_AVAILABLE",
        "margin": 0.10,
        "candidates": [
            candidate(
                1,
                "exm_1",
                "ROAD_ACCESS",
                "IMPASSABLE",
                0.80,
            ),
            candidate(
                2,
                "exm_2",
                "ROAD_ACCESS",
                "IMPASSABLE",
                0.70,
            ),
        ],
    }

    result = adjudicator.adjudicate(
        "road blocked",
        semantic_result,
    )

    assert result["status"] == "ACCEPTED"
    assert result["decision_source"] == "SEMANTIC"
    assert result["claim_type"] == "ROAD_ACCESS"
    assert result["value"] == "IMPASSABLE"
    assert result["matched_exemplar_id"] == "exm_1"
    assert result["similarity_score"] == pytest.approx(0.80)


def test_preserves_phase1_lexical_claim():
    adjudicator = SemanticAdjudicator()

    semantic_result = {
        "status": "CANDIDATE_AVAILABLE",
        "margin": 0.30,
        "candidates": [
            candidate(
                1,
                "exm_1",
                "ROAD_ACCESS",
                "PASSABLE",
                0.90,
            )
        ],
    }

    lexical_claim = {
        "claim_type": "ROAD_ACCESS",
        "value": "IMPASSABLE",
    }

    result = adjudicator.adjudicate(
        "road blocked",
        semantic_result,
        lexical_claim=lexical_claim,
    )

    assert result["status"] == "LEXICAL_PRESERVED"
    assert result["decision_source"] == "LEXICAL"
    assert result["claim_type"] == "ROAD_ACCESS"
    assert result["value"] == "IMPASSABLE"


def test_unresolved_when_candidate_generation_rejects():
    adjudicator = SemanticAdjudicator()

    semantic_result = {
        "status": "REJECTED",
        "reason": "BELOW_SIMILARITY_THRESHOLD",
        "candidates": [],
    }

    result = adjudicator.adjudicate(
        "uncertain text",
        semantic_result,
    )

    assert result["status"] == "UNRESOLVED"
    assert result["claim_type"] is None
    assert result["value"] is None


def test_unresolved_when_margin_is_insufficient():
    adjudicator = SemanticAdjudicator()

    semantic_result = {
        "status": "CANDIDATE_AVAILABLE",
        "margin": 0.01,
        "candidates": [
            candidate(
                1,
                "exm_1",
                "ROAD_ACCESS",
                "IMPASSABLE",
                0.80,
            ),
            candidate(
                2,
                "exm_2",
                "ROAD_ACCESS",
                "PASSABLE",
                0.79,
            ),
        ],
    }

    result = adjudicator.adjudicate(
        "road status",
        semantic_result,
    )

    assert result["status"] == "UNRESOLVED"
    assert result["rule_applied"] == "INSUFFICIENT_SIMILARITY_MARGIN"


def test_unresolved_when_score_is_below_gate():
    adjudicator = SemanticAdjudicator(
        similarity_threshold=0.67,
    )

    semantic_result = {
        "status": "CANDIDATE_AVAILABLE",
        "margin": 0.20,
        "candidates": [
            candidate(
                1,
                "exm_1",
                "ROAD_ACCESS",
                "IMPASSABLE",
                0.65,
            )
        ],
    }

    result = adjudicator.adjudicate(
        "road blocked",
        semantic_result,
    )

    assert result["status"] == "UNRESOLVED"
    assert result["rule_applied"] == "BELOW_SIMILARITY_THRESHOLD"


def test_audit_information_is_preserved():
    adjudicator = SemanticAdjudicator()

    candidates = [
        candidate(
            1,
            "exm_1",
            "ROAD_ACCESS",
            "IMPASSABLE",
            0.80,
        ),
        candidate(
            2,
            "exm_2",
            "ROAD_ACCESS",
            "PASSABLE",
            0.68,
        ),
    ]

    semantic_result = {
        "status": "CANDIDATE_AVAILABLE",
        "margin": 0.12,
        "candidates": candidates,
    }

    result = adjudicator.adjudicate(
        "vehicles cannot pass",
        semantic_result,
    )

    assert len(result["candidate_values"]) == 2
    assert result["candidate_values"][0]["exemplar_id"] == "exm_1"
    assert result["candidate_values"][0]["score"] == pytest.approx(0.80)
    assert result["rule_applied"] == (
        "TOP_CANDIDATE_PASSED_DETERMINISTIC_GATE"
    )


def test_rejects_invalid_input():
    adjudicator = SemanticAdjudicator()

    with pytest.raises(TypeError):
        adjudicator.adjudicate(
            None,
            {},
        )

    with pytest.raises(ValueError):
        adjudicator.adjudicate(
            "   ",
            {},
        )