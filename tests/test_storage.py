"""Unit tests for SQLite storage layer."""

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
    ClaimType,
    EvidenceType,
    SeverityLevel,
)
from crisisstate.storage.database import Database
from crisisstate.storage.repository import CrisisRepository


def test_sqlite_repository_full_crud():
    db = Database(":memory:")
    repo = CrisisRepository(db)

    # 1. Report
    rep = Report(id="r1", text="Flooding on Main Road", source_type="sensor")
    repo.save_report(rep)
    fetched_rep = repo.get_report("r1")
    assert fetched_rep is not None
    assert fetched_rep.text == "Flooding on Main Road"

    # 2. Incident
    inc = Incident(
        id="i1",
        title="Main Road Inundation",
        location_name="Main Road",
        current_severity=SeverityLevel.HIGH,
        current_attention_level=AttentionLevel.HIGH,
    )
    repo.save_incident(inc)
    fetched_inc = repo.get_incident("i1")
    assert fetched_inc is not None
    assert fetched_inc.title == "Main Road Inundation"
    assert fetched_inc.current_severity == SeverityLevel.HIGH

    # 3. Claim
    clm = Claim(
        id="c1",
        incident_id="i1",
        report_id="r1",
        claim_type=ClaimType.ROAD_ACCESS,
        subject="Main Road",
        value="IMPASSABLE",
    )
    repo.save_claim(clm)
    claims = repo.list_claims_for_incident("i1")
    assert len(claims) == 1
    assert claims[0].value == "IMPASSABLE"

    # 4. Evidence Link
    evd = EvidenceLink(
        id="e1",
        report_id="r1",
        claim_id="c1",
        incident_id="i1",
        link_type=EvidenceType.CONTRADICTS,
        reason="Disagrees with previous report",
    )
    repo.save_evidence_link(evd)
    links = repo.list_evidence_for_incident("i1")
    assert len(links) == 1
    assert links[0].link_type == EvidenceType.CONTRADICTS

    # 5. Snapshots
    snp = IncidentStateSnapshot(
        id="s1",
        incident_id="i1",
        changed_field="SEVERITY",
        previous_value="LOW",
        new_value="HIGH",
        reason="Water level rose quickly",
    )
    repo.save_snapshot(snp)
    snaps = repo.list_snapshots_for_incident("i1")
    assert len(snaps) == 1
    assert snaps[0].changed_field == "SEVERITY"
