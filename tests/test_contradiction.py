"""Unit tests for deterministic contradiction and compatibility detection."""

from datetime import datetime, timedelta
from crisisstate.domain.models import Claim, Report
from crisisstate.domain.vocabulary import (
    ClaimStatus,
    ClaimType,
    EvidenceType,
    are_values_contradictory,
    are_values_supporting,
)
from crisisstate.engine.contradiction import ContradictionEngine


def test_vocabulary_contradiction_helpers():
    assert are_values_contradictory(ClaimType.ROAD_ACCESS, "PASSABLE", "IMPASSABLE")
    assert are_values_contradictory(ClaimType.ROAD_ACCESS, "IMPASSABLE", "PASSABLE")
    assert not are_values_contradictory(ClaimType.ROAD_ACCESS, "PASSABLE", "PASSABLE")

    assert are_values_contradictory(ClaimType.WATER_LEVEL, "LOW", "CRITICAL")
    assert are_values_contradictory(ClaimType.WATER_LEVEL, "MODERATE", "CRITICAL")
    assert not are_values_contradictory(ClaimType.WATER_LEVEL, "LOW", "MODERATE")

    assert are_values_supporting(ClaimType.ROAD_ACCESS, "PASSABLE", "passable")
    assert not are_values_supporting(ClaimType.ROAD_ACCESS, "PASSABLE", "IMPASSABLE")


def test_contradiction_engine_flags_conflict_within_window():
    engine = ContradictionEngine(contradiction_window_hours=2.0)
    t0 = datetime(2026, 9, 15, 8, 0)

    old_claim = Claim(
        id="clm_old",
        incident_id="inc_001",
        report_id="rep_old",
        claim_type=ClaimType.ROAD_ACCESS,
        subject="Main Road",
        value="IMPASSABLE",
        timestamp=t0,
    )

    rep_new = Report(
        id="rep_new",
        text="Main Road is passable.",
        timestamp=t0 + timedelta(minutes=20),
    )
    new_claim = Claim(
        id="clm_new",
        incident_id="inc_001",
        report_id=rep_new.id,
        claim_type=ClaimType.ROAD_ACCESS,
        subject="Main Road",
        value="PASSABLE",
        timestamp=rep_new.timestamp,
    )

    links = engine.evaluate_claim(new_claim, [old_claim], "inc_001", rep_new)

    assert len(links) == 1
    assert links[0].link_type == EvidenceType.CONTRADICTS
    assert new_claim.status == ClaimStatus.CONTRADICTED
    assert old_claim.status == ClaimStatus.CONTRADICTED
    assert "rep_new" in old_claim.contradicting_evidence
    assert "rep_old" in new_claim.contradicting_evidence


def test_contradiction_engine_supersedes_after_window():
    engine = ContradictionEngine(contradiction_window_hours=2.0)
    t0 = datetime(2026, 9, 15, 8, 0)

    old_claim = Claim(
        id="clm_morning",
        incident_id="inc_002",
        report_id="rep_morning",
        claim_type=ClaimType.FLOODED,
        subject="Sector 4 Underpass",
        value="YES",
        timestamp=t0,
    )

    # Report 3 hours later stating water is drained
    t_later = t0 + timedelta(hours=3)
    rep_later = Report(
        id="rep_afternoon",
        text="Sector 4 Underpass water drained and clear.",
        timestamp=t_later,
    )
    new_claim = Claim(
        id="clm_afternoon",
        incident_id="inc_002",
        report_id=rep_later.id,
        claim_type=ClaimType.FLOODED,
        subject="Sector 4 Underpass",
        value="NO",
        timestamp=t_later,
    )

    links = engine.evaluate_claim(new_claim, [old_claim], "inc_002", rep_later)

    assert len(links) == 1
    assert links[0].link_type == EvidenceType.NEUTRAL
    assert old_claim.status == ClaimStatus.SUPERSEDED
    assert new_claim.status == ClaimStatus.ACTIVE
