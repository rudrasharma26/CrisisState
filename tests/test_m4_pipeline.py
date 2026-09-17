from crisisstate.domain.models import Report
from crisisstate.semantic.adjudicator import SemanticAdjudicator
from crisisstate.semantic.candidate_generator import SemanticCandidateGenerator
from crisisstate.semantic.m4_pipeline import M4SemanticPipeline
from crisisstate.semantic.search_adapter import ClaimExemplarSearchAdapter


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
            "value": "road can be used",
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

    return M4SemanticPipeline(generator, adjudicator)


def test_semantic_claim_is_created_for_unresolved_lexical_span():
    pipeline = build_pipeline()

    report = Report(
        id="rep_1",
        text="The route is unusable for vehicles.",
    )

    result = pipeline.process_report(report)

    assert len(result.claims) == 1
    assert result.claims[0].claim_type.value == "ROAD_ACCESS"
    assert result.claims[0].value == "IMPASSABLE"
    assert result.claims[0].extraction_method.value == "SEMANTIC"
    assert result.claims[0].matched_exemplar_id == "exm_impassable"


def test_phase1_lexical_claim_is_preserved():
    pipeline = build_pipeline()

    report = Report(
        id="rep_2",
        text="Main Road is blocked.",
    )

    result = pipeline.process_report(report)

    assert len(result.claims) == 1
    assert result.claims[0].claim_type.value == "ROAD_ACCESS"
    assert result.claims[0].value == "IMPASSABLE"
    assert result.claims[0].extraction_method.value == "LEXICAL"


def test_ambiguous_semantic_span_becomes_unresolved():
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

def test_audit_trail_is_generated():
    pipeline = build_pipeline()

    report = Report(
        id="rep_4",
        text="The route is unusable for vehicles.",
    )

    result = pipeline.process_report(report)

    assert len(result.audit_trail) == 1
    audit = result.audit_trail[0]

    assert "source_span" in audit
    assert "semantic_candidates" in audit
    assert "adjudication" in audit