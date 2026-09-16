"""Unit tests for evidence linking and explanation."""

from datetime import datetime
from crisisstate.domain.models import Claim, Report
from crisisstate.domain.vocabulary import ClaimType, EvidenceType
from crisisstate.engine.contradiction import ContradictionEngine


def test_corroboration_evidence_linking():
    engine = ContradictionEngine()
    t = datetime(2026, 9, 15, 8, 0)

    claim1 = Claim(
        id="clm_1",
        incident_id="inc_01",
        report_id="rep_1",
        claim_type=ClaimType.TRAFFIC_STATUS,
        subject="Main Road",
        value="STOPPED",
        timestamp=t,
    )

    rep2 = Report(id="rep_2", text="Traffic completely halted on Main Road.", timestamp=t)
    claim2 = Claim(
        id="clm_2",
        incident_id="inc_01",
        report_id=rep2.id,
        claim_type=ClaimType.TRAFFIC_STATUS,
        subject="Main Road",
        value="STOPPED",
        timestamp=t,
    )

    links = engine.evaluate_claim(claim2, [claim1], "inc_01", rep2)

    assert len(links) == 1
    link = links[0]
    assert link.link_type == EvidenceType.SUPPORTS
    assert "corroborates" in link.reason
    assert "rep_2" in claim1.supporting_evidence
    assert "rep_1" in claim2.supporting_evidence
