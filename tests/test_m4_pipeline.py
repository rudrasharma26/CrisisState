from crisisstate.domain.models import Report
from crisisstate.semantic.adjudicator import SemanticAdjudicator
from crisisstate.semantic.candidate_generator import (
    SemanticCandidateGenerator,
)
from crisisstate.semantic.m4_pipeline import (
    M4SemanticPipeline,
)
from crisisstate.semantic.search_adapter import (
    ClaimExemplarSearchAdapter,
)
import pytest

class FakeSearchEngine:
    def search(self, text, candidates, k=5):
        if "unusable" in text.lower():
            return [
                {
                    "id": "exm_impassable",
                    "claim_type": "ROAD_ACCESS",
                    "value": "IMPASSABLE",
                    "text": "road cannot be used",
                    "score": 0.80,
                },
                {
                    "id": "exm_passable",
                    "claim_type": "ROAD_ACCESS",
                    "value": "PASSABLE",
                    "text": "road can be used",
                    "score": 0.70,
                },
            ]

        if "uncertain route" in text.lower():
            return [
                {
                    "id": "exm_impassable",
                    "claim_type": "ROAD_ACCESS",
                    "value": "IMPASSABLE",
                    "text": "road cannot be used",
                    "score": 0.80,
                },
                {
                    "id": "exm_passable",
                    "claim_type": "ROAD_ACCESS",
                    "value": "PASSABLE",
                    "text": "road can be used",
                    "score": 0.78,
                },
            ]

        if "stopped traffic" in text.lower():
            return [
                {
                    "id": "stop_1",
                    "claim_type": "TRAFFIC_STATUS",
                    "value": "STOPPED",
                    "text": "traffic stopped",
                    "score": 0.81,
                },
                {
                    "id": "stop_2",
                    "claim_type": "TRAFFIC_STATUS",
                    "value": "STOPPED",
                    "text": "vehicles halted",
                    "score": 0.78,
                },
                {
                    "id": "move_1",
                    "claim_type": "TRAFFIC_STATUS",
                    "value": "MOVING",
                    "text": "traffic moving",
                    "score": 0.76,
                },
            ]

        return []


def build_pipeline():
    exemplars = [
        {
            "id": "exm_impassable",
            "claim_type": "ROAD_ACCESS",
            "value": "IMPASSABLE",
            "text": "road cannot be used",
        },
        {
            "id": "exm_passable",
            "claim_type": "ROAD_ACCESS",
            "value": "PASSABLE",
            "text": "road can be used",
        },
    ]

    search = ClaimExemplarSearchAdapter(
        FakeSearchEngine(),
        exemplars,
    )

    generator = SemanticCandidateGenerator(
        search,
        similarity_threshold=0.67,
        margin_threshold=0.04,
    )

    adjudicator = SemanticAdjudicator(
        similarity_threshold=0.67,
        margin_threshold=0.04,
    )

    return M4SemanticPipeline(
        generator,
        adjudicator,
    )


def test_semantic_claim_is_created():
    pipeline = build_pipeline()

    report = Report(
        id="rep_1",
        text="The route is unusable for vehicles.",
    )

    result = pipeline.process_report(report)

    assert len(result.claims) == 1
    assert (
        result.claims[0].claim_type.value
        == "ROAD_ACCESS"
    )
    assert result.claims[0].value == "IMPASSABLE"
    assert (
        result.claims[0].extraction_method.value
        == "SEMANTIC"
    )


def test_phase1_lexical_claim_is_preserved_once():
    pipeline = build_pipeline()

    report = Report(
        id="rep_2",
        text="Main Road is blocked. Main Road is blocked.",
    )

    result = pipeline.process_report(report)

    lexical_claims = [
        claim
        for claim in result.claims
        if claim.extraction_method.value == "LEXICAL"
    ]

    assert len(lexical_claims) == 1
    assert lexical_claims[0].value == "IMPASSABLE"


def test_ambiguous_span_is_unresolved():
    pipeline = build_pipeline()

    report = Report(
        id="rep_3",
        text="uncertain route",
    )

    result = pipeline.process_report(report)

    assert len(result.claims) == 0
    assert len(result.unresolved_spans) == 1
    assert (
        result.unresolved_spans[0].rejection_reason
        == "SEMANTIC_RESULT_AMBIGUOUS"
    )


def test_same_value_exemplars_do_not_create_false_ambiguity():
    class TrafficPipeline:
        pass

    search = ClaimExemplarSearchAdapter(
        FakeSearchEngine(),
        [
            {
                "id": "stop_1",
                "claim_type": "TRAFFIC_STATUS",
                "value": "STOPPED",
                "text": "traffic stopped",
            },
            {
                "id": "stop_2",
                "claim_type": "TRAFFIC_STATUS",
                "value": "STOPPED",
                "text": "vehicles halted",
            },
            {
                "id": "move_1",
                "claim_type": "TRAFFIC_STATUS",
                "value": "MOVING",
                "text": "traffic moving",
            },
        ],
    )

    generator = SemanticCandidateGenerator(
        search,
        similarity_threshold=0.67,
        margin_threshold=0.04,
    )

    result = generator.generate(
        "stopped traffic"
    )

    assert result["status"] == "CANDIDATE_AVAILABLE"
    assert result["candidates"][0]["value"] == "STOPPED"
    assert result["margin"] == pytest.approx(0.05)


def test_lexical_claim_type_prevents_semantic_duplicate():
    pipeline = build_pipeline()

    report = Report(
        id="rep_4",
        text="Main Road is blocked. The route is unusable.",
    )

    result = pipeline.process_report(report)

    road_claims = [
        claim
        for claim in result.claims
        if claim.claim_type.value == "ROAD_ACCESS"
    ]

    assert len(road_claims) == 1
    assert road_claims[0].extraction_method.value == "LEXICAL"