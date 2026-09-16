"""End-to-End Acceptance Test for CrisisState Synthetic Scenario Replay."""

from crisisstate.replay.runner import ReplayRunner, load_reports_from_file
from crisisstate.storage.database import Database
from crisisstate.storage.repository import CrisisRepository


def test_acceptance_complete_scenario_replay():
    """
    Acceptance criteria:
    1. Loads the synthetic report sequence
    2. Processes reports in chronological order
    3. Creates incidents (2 distinct geographic clusters)
    4. Extracts claims
    5. Identifies at least one explicit contradiction
    6. Records evidence links
    7. Produces updated incident state
    8. Produces incident-state history (timeline snapshots)
    """
    db = Database(":memory:")
    repo = CrisisRepository(db)
    runner = ReplayRunner(repository=repo)

    reports = load_reports_from_file()
    assert len(reports) == 9

    summary = runner.run(reports=reports, verbose=False)

    # 1. Check counts
    assert summary["reports_processed"] == 9
    assert summary["incidents_created"] == 2
    assert summary["claims_extracted"] >= 10
    assert summary["contradictions_detected"] >= 1

    # 2. Check Incidents
    incidents = repo.list_incidents()
    assert len(incidents) == 2

    # Find Main Road incident and Sector 4 incident
    main_road_inc = next(i for i in incidents if "Main Road" in i.location_name)
    sec4_inc = next(i for i in incidents if "Sector 4 Underpass" in i.location_name)

    assert len(main_road_inc.linked_report_ids) == 5
    assert len(sec4_inc.linked_report_ids) == 4

    # 3. Main Road contradiction verification
    main_claims = repo.list_claims_for_incident(main_road_inc.id)
    contradicted_claims = [c for c in main_claims if c.status.value == "CONTRADICTED"]
    assert len(contradicted_claims) >= 2

    # Check that ROAD_ACCESS is conflicting on Main Road
    assert "ROAD_ACCESS:Main Road" in main_road_inc.state_summary
    assert "CONFLICTING" in main_road_inc.state_summary["ROAD_ACCESS:Main Road"]

    # Check evidence links for Main Road
    evidence_links = repo.list_evidence_for_incident(main_road_inc.id)
    contradiction_links = [e for e in evidence_links if e.link_type.value == "CONTRADICTS"]
    assert len(contradiction_links) >= 1
    assert any("ROAD_ACCESS" in e.reason for e in contradiction_links)

    # 4. Check Timeline History / Snapshots
    main_snapshots = repo.list_snapshots_for_incident(main_road_inc.id)
    assert len(main_snapshots) >= 5

    # Check snapshot fields
    changed_fields = {s.changed_field for s in main_snapshots}
    assert "ROAD_ACCESS:Main Road" in changed_fields
    assert "SEVERITY" in changed_fields
    assert "ATTENTION_LEVEL" in changed_fields

    # Check Severity escalation to CRITICAL on Main Road due to trapped people
    assert main_road_inc.current_severity.value == "CRITICAL"
    assert main_road_inc.current_attention_level.value == "CRITICAL"
    assert main_road_inc.attention_factors["severity"] == 1.0

    # 5. Sector 4 Underpass state update verification
    # rep_008 & rep_009 updated FLOODED to NO and TRAFFIC_STATUS to MOVING
    sec4_snapshots = repo.list_snapshots_for_incident(sec4_inc.id)
    assert len(sec4_snapshots) >= 4
    assert sec4_inc.state_summary.get("FLOODED:Sector 4 Underpass") == "NO"
    assert sec4_inc.state_summary.get("TRAFFIC_STATUS:Sector 4 Underpass") == "MOVING"
