"""Tests for Phase 2 M1: Schema Extensions.

Verifies that:
  - New Claim fields have correct Phase 1 defaults.
  - ExtractionMethod, ValueModality, and other new enums are importable and correct.
  - ClaimExemplar, UnresolvedSpan, IncidentMatch, ClaimStateEntry can be
    created with expected defaults.
  - EvidenceLink new fields default correctly.
  - Incident.claim_state_entries defaults to an empty list.
  - All new models round-trip correctly through the SQLite repository.
  - API serialisation remains backward-compatible (no fields removed).
  - Phase 1 extraction populates extraction_method=LEXICAL and
    extraction_confidence from the rule confidence.
"""

import pytest
from datetime import datetime, timezone

from crisisstate.domain.models import (
    Claim,
    ClaimExemplar,
    ClaimStateEntry,
    EvidenceLink,
    Incident,
    IncidentMatch,
    IncidentStateSnapshot,
    Report,
    UnresolvedSpan,
)
from crisisstate.domain.vocabulary import (
    AttentionLevel,
    ClaimStateStatus,
    ClaimStatus,
    ClaimType,
    EvidenceType,
    ExemplarSource,
    ExtractionMethod,
    MatchReason,
    SeverityLevel,
    ValueModality,
)
from crisisstate.engine.extractor import extract_claims
from crisisstate.storage.database import Database
from crisisstate.storage.repository import CrisisRepository


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_repo() -> CrisisRepository:
    """Return a fresh in-memory repository."""
    return CrisisRepository(Database(":memory:"))


# ── 1. Claim defaults and new fields ─────────────────────────────────────────

class TestClaimM1Fields:
    def test_extraction_method_defaults_to_lexical(self):
        claim = Claim(report_id="r1", claim_type=ClaimType.ROAD_ACCESS, subject="Main Road", value="IMPASSABLE")
        assert claim.extraction_method == ExtractionMethod.LEXICAL

    def test_value_modality_defaults_to_asserted(self):
        claim = Claim(report_id="r1", claim_type=ClaimType.FLOODED, subject="Main Road", value="YES")
        assert claim.value_modality == ValueModality.ASSERTED

    def test_extraction_confidence_defaults_to_none(self):
        claim = Claim(report_id="r1", claim_type=ClaimType.WATER_LEVEL, subject="Main Road", value="HIGH")
        assert claim.extraction_confidence is None

    def test_matched_exemplar_id_defaults_to_none(self):
        claim = Claim(report_id="r1", claim_type=ClaimType.TRAFFIC_STATUS, subject="Main Road", value="STOPPED")
        assert claim.matched_exemplar_id is None

    def test_similarity_score_defaults_to_none(self):
        claim = Claim(report_id="r1", claim_type=ClaimType.EVACUATION, subject="Riverbank Colony", value="ORDERED")
        assert claim.similarity_score is None

    def test_source_span_defaults_to_none(self):
        claim = Claim(report_id="r1", claim_type=ClaimType.PEOPLE_TRAPPED, subject="Main Road", value="YES")
        assert claim.source_span is None

    def test_qualifier_defaults_to_none(self):
        claim = Claim(report_id="r1", claim_type=ClaimType.ROAD_ACCESS, subject="Main Road", value="PASSABLE")
        assert claim.qualifier is None

    def test_candidate_values_defaults_to_empty_list(self):
        claim = Claim(report_id="r1", claim_type=ClaimType.ROAD_ACCESS, subject="Main Road", value="PASSABLE")
        assert claim.candidate_values == []

    def test_set_non_default_m1_fields(self):
        claim = Claim(
            report_id="r1",
            claim_type=ClaimType.ROAD_ACCESS,
            subject="5th Avenue",
            value="IMPASSABLE",
            extraction_method=ExtractionMethod.SEMANTIC,
            extraction_confidence=0.87,
            matched_exemplar_id="exm_abc123",
            similarity_score=0.87,
            source_span={"start": 0, "end": 22, "text": "road is blocked"},
            value_modality=ValueModality.NEGATED,
            qualifier={"permitted_scope": "EMERGENCY_VEHICLES_ONLY"},
            candidate_values=["IMPASSABLE", "PASSABLE"],
        )
        assert claim.extraction_method == ExtractionMethod.SEMANTIC
        assert claim.extraction_confidence == pytest.approx(0.87)
        assert claim.matched_exemplar_id == "exm_abc123"
        assert claim.similarity_score == pytest.approx(0.87)
        assert claim.source_span["start"] == 0
        assert claim.value_modality == ValueModality.NEGATED
        assert claim.qualifier["permitted_scope"] == "EMERGENCY_VEHICLES_ONLY"
        assert "IMPASSABLE" in claim.candidate_values

    def test_phase1_claim_still_compatible(self):
        """Existing Phase 1 call-sites that do not pass M1 args still work."""
        claim = Claim(
            report_id="rep_legacy",
            claim_type=ClaimType.BUILDING_DAMAGE,
            subject="Market Square",
            value="COLLAPSED",
        )
        assert claim.status == ClaimStatus.ACTIVE
        assert claim.confidence == 1.0
        assert claim.extraction_method == ExtractionMethod.LEXICAL
        assert claim.value_modality == ValueModality.ASSERTED


# ── 2. New enum values ────────────────────────────────────────────────────────

class TestNewEnums:
    def test_extraction_method_members(self):
        assert {e.value for e in ExtractionMethod} == {"LEXICAL", "SEMANTIC", "NLI_CONFIRMED", "HYBRID"}

    def test_value_modality_members(self):
        assert {e.value for e in ValueModality} == {"ASSERTED", "QUALIFIED", "NEGATED", "AMBIGUOUS"}

    def test_exemplar_source_members(self):
        assert {e.value for e in ExemplarSource} == {"HAND_AUTHORED", "CORPUS_DERIVED"}

    def test_match_reason_members(self):
        assert {e.value for e in MatchReason} == {"SPATIAL", "TEMPORAL", "ENTITY", "HYBRID", "SEMANTIC_TIEBREAK"}

    def test_claim_state_status_members(self):
        assert {e.value for e in ClaimStateStatus} == {
            "SUPPORTED", "WEAKLY_SUPPORTED", "CONFLICTING", "UNRESOLVED", "SUPERSEDED"
        }


# ── 3. ClaimExemplar ─────────────────────────────────────────────────────────

class TestClaimExemplar:
    def test_creation_with_defaults(self):
        ex = ClaimExemplar(
            claim_type=ClaimType.ROAD_ACCESS,
            value="IMPASSABLE",
            text="Road is completely blocked by flood water.",
        )
        assert ex.id.startswith("exm_")
        assert ex.source == ExemplarSource.HAND_AUTHORED
        assert ex.version == "v1"
        assert isinstance(ex.created_at, datetime)

    def test_creation_corpus_derived(self):
        ex = ClaimExemplar(
            claim_type=ClaimType.WATER_LEVEL,
            value="CRITICAL",
            text="Water has reached chest height.",
            source=ExemplarSource.CORPUS_DERIVED,
            version="v2",
        )
        assert ex.source == ExemplarSource.CORPUS_DERIVED
        assert ex.version == "v2"

    def test_repository_round_trip(self):
        repo = make_repo()
        ex = ClaimExemplar(
            claim_type=ClaimType.PEOPLE_TRAPPED,
            value="YES",
            text="People are stranded in the colony.",
        )
        repo.save_exemplar(ex)
        results = repo.list_exemplars(claim_type=ClaimType.PEOPLE_TRAPPED, value="YES")
        assert len(results) == 1
        assert results[0].text == ex.text
        assert results[0].source == ExemplarSource.HAND_AUTHORED

    def test_list_exemplars_filtered(self):
        repo = make_repo()
        ex1 = ClaimExemplar(claim_type=ClaimType.ROAD_ACCESS, value="IMPASSABLE", text="txt1")
        ex2 = ClaimExemplar(claim_type=ClaimType.ROAD_ACCESS, value="PASSABLE", text="txt2")
        ex3 = ClaimExemplar(claim_type=ClaimType.FLOODED, value="YES", text="txt3")
        for e in [ex1, ex2, ex3]:
            repo.save_exemplar(e)
        road_results = repo.list_exemplars(claim_type=ClaimType.ROAD_ACCESS)
        assert len(road_results) == 2
        all_results = repo.list_exemplars()
        assert len(all_results) == 3


# ── 4. UnresolvedSpan ────────────────────────────────────────────────────────

class TestUnresolvedSpan:
    def test_creation_with_defaults(self):
        span = UnresolvedSpan(
            report_id="rep_001",
            span_text="vehicles cannot get through",
        )
        assert span.id.startswith("usp_")
        assert span.top_candidates == []
        assert span.rejection_reason is None
        assert isinstance(span.created_at, datetime)

    def test_creation_with_candidates(self):
        span = UnresolvedSpan(
            report_id="rep_002",
            span_text="marooned residents",
            top_candidates=[
                {"claim_type": "PEOPLE_TRAPPED", "value": "YES", "score": 0.72},
                {"claim_type": "EVACUATION", "value": "NOT_ORDERED", "score": 0.41},
            ],
            rejection_reason="BELOW_THRESHOLD",
        )
        assert len(span.top_candidates) == 2
        assert span.rejection_reason == "BELOW_THRESHOLD"

    def test_repository_round_trip(self):
        repo = make_repo()
        span = UnresolvedSpan(
            report_id="rep_003",
            span_text="the road may be blocked",
            top_candidates=[{"claim_type": "ROAD_ACCESS", "value": "IMPASSABLE", "score": 0.55}],
            rejection_reason="AMBIGUOUS",
        )
        repo.save_unresolved_span(span)
        results = repo.list_unresolved_spans_for_report("rep_003")
        assert len(results) == 1
        assert results[0].span_text == "the road may be blocked"
        assert results[0].rejection_reason == "AMBIGUOUS"
        assert results[0].top_candidates[0]["score"] == 0.55

    def test_phase1_never_creates_unresolved_spans(self):
        """Phase 1 pipeline should not generate any unresolved spans."""
        from crisisstate.engine.pipeline import IncidentPipeline
        repo = make_repo()
        pipeline = IncidentPipeline(repo)
        report = Report(
            text="Main Road is impassable due to severe flooding.",
            latitude=28.6139,
            longitude=77.2090,
            metadata={"location": "Main Road"},
        )
        pipeline.process_report(report)
        spans = repo.list_unresolved_spans_for_report(report.id)
        assert spans == []


# ── 5. IncidentMatch ─────────────────────────────────────────────────────────

class TestIncidentMatch:
    def test_creation_with_defaults(self):
        match = IncidentMatch(
            report_id="rep_001",
            incident_id="inc_001",
            match_reason=MatchReason.ENTITY,
        )
        assert match.id.startswith("imt_")
        assert match.gate_results == {}
        assert match.semantic_score is None
        assert isinstance(match.created_at, datetime)

    def test_creation_with_gate_results(self):
        match = IncidentMatch(
            report_id="rep_002",
            incident_id="inc_002",
            match_reason=MatchReason.HYBRID,
            gate_results={
                "entity_match": True,
                "spatial_distance_km": 0.15,
                "time_diff_hours": 0.5,
            },
            semantic_score=None,
        )
        assert match.gate_results["entity_match"] is True
        assert match.gate_results["spatial_distance_km"] == pytest.approx(0.15)

    def test_all_match_reasons_valid(self):
        for reason in MatchReason:
            m = IncidentMatch(
                report_id="r1",
                incident_id="i1",
                match_reason=reason,
            )
            assert m.match_reason == reason

    def test_repository_round_trip(self):
        repo = make_repo()
        # Need incident and report to exist for FK if enforced; use soft insert
        match = IncidentMatch(
            report_id="rep_test",
            incident_id="inc_test",
            match_reason=MatchReason.SPATIAL,
            gate_results={"spatial_distance_km": 0.8, "time_diff_hours": 1.2},
        )
        repo.save_incident_match(match)
        results = repo.list_matches_for_report("rep_test")
        assert len(results) == 1
        assert results[0].match_reason == MatchReason.SPATIAL
        assert results[0].gate_results["spatial_distance_km"] == pytest.approx(0.8)


# ── 6. EvidenceLink new fields ────────────────────────────────────────────────

class TestEvidenceLinkM1Fields:
    def test_strength_components_defaults_to_empty_dict(self):
        link = EvidenceLink(
            report_id="r1",
            claim_id="c1",
            incident_id="i1",
            link_type=EvidenceType.SUPPORTS,
            reason="corroborates",
        )
        assert link.strength_components == {}

    def test_strength_total_defaults_to_none(self):
        link = EvidenceLink(
            report_id="r1",
            claim_id="c1",
            incident_id="i1",
            link_type=EvidenceType.CONTRADICTS,
            reason="conflicts",
        )
        assert link.strength_total is None

    def test_independence_group_id_defaults_to_none(self):
        link = EvidenceLink(
            report_id="r1",
            claim_id="c1",
            incident_id="i1",
            link_type=EvidenceType.NEUTRAL,
            reason="neutral",
        )
        assert link.independence_group_id is None

    def test_set_strength_fields(self):
        link = EvidenceLink(
            report_id="r1",
            claim_id="c1",
            incident_id="i1",
            link_type=EvidenceType.SUPPORTS,
            reason="corroborates",
            strength_components={"source_reliability": 0.9, "lexical_match": 1.0},
            strength_total=0.95,
            independence_group_id="grp_news_sources",
        )
        assert link.strength_total == pytest.approx(0.95)
        assert link.independence_group_id == "grp_news_sources"


# ── 7. Incident.claim_state_entries ──────────────────────────────────────────

class TestIncidentClaimStateEntries:
    def test_claim_state_entries_defaults_to_empty(self):
        inc = Incident(title="Test", location_name="Main Road")
        assert inc.claim_state_entries == []

    def test_claim_state_entry_creation(self):
        entry = ClaimStateEntry(
            incident_id="inc_001",
            claim_type=ClaimType.ROAD_ACCESS,
            subject="Main Road",
            status=ClaimStateStatus.CONFLICTING,
            dominant_value=None,
            competing_values=["PASSABLE", "IMPASSABLE"],
            supporting_claim_ids=["c1"],
            conflicting_claim_ids=["c2", "c3"],
        )
        assert entry.id.startswith("cse_")
        assert entry.status == ClaimStateStatus.CONFLICTING
        assert "PASSABLE" in entry.competing_values

    def test_incident_round_trip_with_claim_state_entries(self):
        repo = make_repo()
        inc = Incident(title="Flood", location_name="5th Avenue")
        inc.claim_state_entries = [
            {"incident_id": inc.id, "claim_type": "ROAD_ACCESS", "subject": "5th Avenue", "status": "CONFLICTING"}
        ]
        repo.save_incident(inc)
        loaded = repo.get_incident(inc.id)
        assert loaded is not None
        assert len(loaded.claim_state_entries) == 1
        assert loaded.claim_state_entries[0]["status"] == "CONFLICTING"


# ── 8. Extractor populates M1 fields ─────────────────────────────────────────

class TestExtractorM1Behaviour:
    def test_extraction_method_is_lexical(self):
        report = Report(
            text="Main Road is impassable due to severe flooding.",
            metadata={"location": "Main Road"},
        )
        claims = extract_claims(report)
        assert len(claims) > 0
        for c in claims:
            assert c.extraction_method == ExtractionMethod.LEXICAL

    def test_extraction_confidence_populated_from_rule(self):
        report = Report(
            text="Main Road is impassable due to severe flooding.",
            metadata={"location": "Main Road"},
        )
        claims = extract_claims(report)
        road_claims = [c for c in claims if c.claim_type == ClaimType.ROAD_ACCESS]
        assert len(road_claims) == 1
        # IMPASSABLE rule has confidence 0.95
        assert road_claims[0].extraction_confidence == pytest.approx(0.95)

    def test_value_modality_defaults_asserted_from_extractor(self):
        report = Report(
            text="People are trapped on Bridge Road.",
            metadata={"location": "Bridge Road"},
        )
        claims = extract_claims(report)
        for c in claims:
            assert c.value_modality == ValueModality.ASSERTED

    def test_no_candidate_values_in_phase1(self):
        report = Report(
            text="Traffic is completely stopped near Market Square.",
            metadata={"location": "Market Square"},
        )
        claims = extract_claims(report)
        for c in claims:
            assert c.candidate_values == []


# ── 9. Repository full round-trip for new Claim fields ───────────────────────

class TestClaimRepositoryM1RoundTrip:
    def test_save_and_load_claim_with_m1_fields(self):
        repo = make_repo()
        claim = Claim(
            report_id="rep_001",
            claim_type=ClaimType.ROAD_ACCESS,
            subject="Main Road",
            value="IMPASSABLE",
            extraction_method=ExtractionMethod.LEXICAL,
            extraction_confidence=0.95,
            value_modality=ValueModality.ASSERTED,
            candidate_values=[],
        )
        claim.incident_id = "inc_dummy"
        repo.save_claim(claim)
        loaded = repo.get_claim(claim.id)
        assert loaded is not None
        assert loaded.extraction_method == ExtractionMethod.LEXICAL
        assert loaded.extraction_confidence == pytest.approx(0.95)
        assert loaded.value_modality == ValueModality.ASSERTED
        assert loaded.matched_exemplar_id is None
        assert loaded.candidate_values == []

    def test_save_and_load_evidence_link_with_m1_fields(self):
        repo = make_repo()
        # incident needed for evidence_link FK lookup
        inc = Incident(title="Test", location_name="Main Road")
        repo.save_incident(inc)

        link = EvidenceLink(
            report_id="rep_001",
            claim_id="clm_001",
            incident_id=inc.id,
            link_type=EvidenceType.CONTRADICTS,
            reason="conflicting access state",
            strength_components={"source_weight": 0.8},
            strength_total=0.8,
            independence_group_id="grp_field_team",
        )
        repo.save_evidence_link(link)
        links = repo.list_evidence_for_incident(inc.id)
        assert len(links) == 1
        loaded = links[0]
        assert loaded.strength_components == {"source_weight": 0.8}
        assert loaded.strength_total == pytest.approx(0.8)
        assert loaded.independence_group_id == "grp_field_team"


# ── 10. Backward-compatible API serialisation ─────────────────────────────────

class TestAPISerializationCompatibility:
    def test_claim_serialisation_has_all_phase1_fields(self):
        claim = Claim(
            report_id="r1",
            claim_type=ClaimType.ROAD_ACCESS,
            subject="Main Road",
            value="IMPASSABLE",
        )
        data = claim.model_dump(mode="json")
        # Phase 1 fields must still be present
        for field in ("id", "incident_id", "report_id", "claim_type", "subject",
                      "value", "timestamp", "status", "confidence",
                      "supporting_evidence", "contradicting_evidence"):
            assert field in data, f"Missing Phase 1 field: {field}"

    def test_evidence_link_serialisation_has_all_phase1_fields(self):
        link = EvidenceLink(
            report_id="r1",
            claim_id="c1",
            incident_id="i1",
            link_type=EvidenceType.SUPPORTS,
            reason="corroborates",
        )
        data = link.model_dump(mode="json")
        for field in ("id", "report_id", "claim_id", "incident_id",
                      "link_type", "confidence", "reason", "created_at"):
            assert field in data, f"Missing Phase 1 field: {field}"

    def test_incident_serialisation_has_all_phase1_fields(self):
        inc = Incident(title="Test Flood", location_name="Market Square")
        data = inc.model_dump(mode="json")
        for field in ("id", "title", "location_name", "latitude", "longitude",
                      "created_at", "updated_at", "current_severity",
                      "current_attention_level", "attention_factors",
                      "current_confidence", "linked_report_ids", "claim_ids",
                      "state_summary"):
            assert field in data, f"Missing Phase 1 field: {field}"
        # M1 field also present
        assert "claim_state_entries" in data
