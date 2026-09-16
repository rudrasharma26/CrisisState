"""Unit tests for incident state updates and snapshot tracking."""

from datetime import datetime, timedelta
from crisisstate.domain.models import Report
from crisisstate.engine.pipeline import IncidentPipeline
from crisisstate.storage.database import Database
from crisisstate.storage.repository import CrisisRepository


def test_pipeline_tracks_snapshots_on_state_change():
    db = Database(":memory:")
    repo = CrisisRepository(db)
    pipeline = IncidentPipeline(repo)

    t0 = datetime(2026, 9, 15, 8, 0)

    # First report establishes initial state
    rep1 = Report(
        id="rep_test_1",
        text="Water is building up near Main Road. Vehicles can still pass.",
        timestamp=t0,
    )
    res1 = pipeline.process_report(rep1)
    assert len(res1.new_snapshots) > 0
    snap_fields = [s.changed_field for s in res1.new_snapshots]
    assert "ROAD_ACCESS:Main Road" in snap_fields
    assert "SEVERITY" in snap_fields

    # Second report brings contradiction
    rep2 = Report(
        id="rep_test_2",
        text="Main Road is impassable due to heavy flood water.",
        timestamp=t0 + timedelta(minutes=15),
    )
    res2 = pipeline.process_report(rep2)
    assert res2.contradiction_detected is True

    # Check that snapshot recorded transition to CONFLICTING
    conflict_snaps = [
        s for s in res2.new_snapshots if s.changed_field == "ROAD_ACCESS:Main Road"
    ]
    assert len(conflict_snaps) == 1
    assert "CONFLICTING" in conflict_snaps[0].new_value
    assert conflict_snaps[0].previous_value == "PASSABLE"

    # Verify snapshots persisted in repo
    all_snaps = repo.list_snapshots_for_incident(res2.incident.id)
    assert len(all_snaps) >= 3
