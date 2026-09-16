"""Unit tests for transparent attention scoring and factor breakdown."""

from datetime import datetime
from crisisstate.domain.models import Claim, Incident
from crisisstate.domain.vocabulary import (
    AttentionLevel,
    ClaimStatus,
    ClaimType,
    SeverityLevel,
)
from crisisstate.engine.attention import AttentionScorer


def test_attention_scorer_low_baseline():
    scorer = AttentionScorer()
    now = datetime(2026, 9, 15, 8, 0)
    inc = Incident(
        title="Test Inc",
        location_name="Main Road",
        created_at=now,
        updated_at=now,
        linked_report_ids=["r1"],
    )
    claim = Claim(
        report_id="r1",
        claim_type=ClaimType.WATER_LEVEL,
        subject="Main Road",
        value="LOW",
        timestamp=now,
    )

    att, sev, factors, score = scorer.compute_attention(inc, [claim], now)

    assert att in [AttentionLevel.LOW, AttentionLevel.MEDIUM]
    assert sev in [SeverityLevel.INFORMATIONAL, SeverityLevel.LOW]
    assert "severity" in factors
    assert "conflict" in factors
    assert factors["conflict"] == 0.0


def test_attention_scorer_critical_severity_on_trapped_people():
    scorer = AttentionScorer()
    now = datetime(2026, 9, 15, 8, 0)
    inc = Incident(
        title="Test Inc",
        location_name="Main Road",
        created_at=now,
        updated_at=now,
        linked_report_ids=["r1", "r2"],
    )
    claim1 = Claim(
        report_id="r1",
        claim_type=ClaimType.PEOPLE_TRAPPED,
        subject="Main Road",
        value="YES",
        timestamp=now,
    )

    att, sev, factors, score = scorer.compute_attention(inc, [claim1], now)

    assert sev == SeverityLevel.CRITICAL
    assert att == AttentionLevel.CRITICAL
    assert factors["severity"] == 1.0


def test_attention_scorer_escalates_on_conflict():
    scorer = AttentionScorer()
    now = datetime(2026, 9, 15, 8, 0)
    inc = Incident(
        title="Test Inc",
        location_name="Main Road",
        created_at=now,
        updated_at=now,
        linked_report_ids=["r1", "r2"],
    )
    claim1 = Claim(
        report_id="r1",
        claim_type=ClaimType.ROAD_ACCESS,
        subject="Main Road",
        value="PASSABLE",
        timestamp=now,
        status=ClaimStatus.CONTRADICTED,
    )
    claim2 = Claim(
        report_id="r2",
        claim_type=ClaimType.ROAD_ACCESS,
        subject="Main Road",
        value="IMPASSABLE",
        timestamp=now,
        status=ClaimStatus.CONTRADICTED,
    )

    att, sev, factors, score = scorer.compute_attention(inc, [claim1, claim2], now)

    assert factors["conflict"] > 0.5
    assert factors["uncertainty"] > 0.5
