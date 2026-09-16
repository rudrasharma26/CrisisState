"""Unit tests for incident matching and clustering."""

from datetime import datetime, timedelta
from crisisstate.domain.models import Incident, Report
from crisisstate.engine.matcher import IncidentMatcher, haversine_distance_km


def test_haversine_calculation():
    # Distance between ~Connaught Place and Sector 4 area (~5.6 km)
    dist = haversine_distance_km(28.6139, 77.2090, 28.6500, 77.2500)
    assert 5.0 < dist < 7.0


def test_matcher_associates_same_location():
    matcher = IncidentMatcher(max_distance_km=2.0)
    t0 = datetime(2026, 9, 15, 8, 0)
    existing_inc = Incident(
        id="inc_main",
        title="Flooding at Main Road",
        location_name="Main Road",
        latitude=28.6139,
        longitude=77.2090,
        created_at=t0,
        updated_at=t0,
    )

    # Incoming report 15 mins later, nearby coordinates
    rep = Report(
        text="Main Road has standing water.",
        timestamp=t0 + timedelta(minutes=15),
        latitude=28.6141,
        longitude=77.2092,
    )

    matched_inc, is_new = matcher.match_or_create(rep, [existing_inc])
    assert is_new is False
    assert matched_inc.id == "inc_main"


def test_matcher_creates_new_incident_for_distant_location():
    matcher = IncidentMatcher(max_distance_km=2.0)
    t0 = datetime(2026, 9, 15, 8, 0)
    existing_inc = Incident(
        id="inc_main",
        title="Flooding at Main Road",
        location_name="Main Road",
        latitude=28.6139,
        longitude=77.2090,
        created_at=t0,
        updated_at=t0,
    )

    # Sector 4 Underpass is > 5km away and different entity
    rep = Report(
        text="Sector 4 Underpass is submerged.",
        timestamp=t0 + timedelta(minutes=20),
        latitude=28.6500,
        longitude=77.2500,
    )

    matched_inc, is_new = matcher.match_or_create(rep, [existing_inc])
    assert is_new is True
    assert "Sector 4 Underpass" in matched_inc.location_name
