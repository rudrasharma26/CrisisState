"""Unit tests for domain models."""

from datetime import datetime
from crisisstate.domain.models import (
    Claim,
    EvidenceLink,
    Incident,
    IncidentStateSnapshot,
    Report,
)
from crisisstate.domain.vocabulary import (
    AttentionLevel,
    ClaimStatus,
    ClaimType,
    EvidenceType,
    SeverityLevel,
)


def test_report_creation_and_defaults():
    rep = Report(text="Flood waters rising on Main Road.")
    assert rep.id.startswith("rep_")
    assert rep.text == "Flood waters rising on Main Road."
    assert rep.source_type == "citizen"
    assert isinstance(rep.timestamp, datetime)
    assert rep.metadata == {}


def test_claim_creation():
    claim = Claim(
        report_id="rep_123",
        claim_type=ClaimType.ROAD_ACCESS,
        subject="Main Road",
        value="IMPASSABLE",
    )
    assert claim.id.startswith("clm_")
    assert claim.status == ClaimStatus.ACTIVE
    assert claim.value == "IMPASSABLE"
    assert claim.confidence == 1.0


def test_evidence_link_creation():
    link = EvidenceLink(
        report_id="rep_001",
        claim_id="clm_001",
        incident_id="inc_001",
        link_type=EvidenceType.CONTRADICTS,
        reason="Report states road is passable, contradicting earlier closure.",
    )
    assert link.id.startswith("evd_")
    assert link.link_type == EvidenceType.CONTRADICTS
    assert "contradicting" in link.reason


def test_incident_snapshot_creation():
    snap = IncidentStateSnapshot(
        incident_id="inc_001",
        changed_field="ROAD_ACCESS:Main Road",
        previous_value="PASSABLE",
        new_value="CONFLICTING (IMPASSABLE/PASSABLE)",
        reason="Conflicting reports received",
    )
    assert snap.id.startswith("snp_")
    assert snap.changed_field == "ROAD_ACCESS:Main Road"
    assert snap.previous_value == "PASSABLE"


def test_incident_initialization():
    inc = Incident(
        title="Urban Flooding at Sector 4",
        location_name="Sector 4 Underpass",
        latitude=28.65,
        longitude=77.25,
    )
    assert inc.id.startswith("inc_")
    assert inc.current_severity == SeverityLevel.LOW
    assert inc.current_attention_level == AttentionLevel.LOW
