"""Unit tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from crisisstate.app.main import app
from crisisstate.app.api.routes import get_pipeline, get_repository
from crisisstate.engine.pipeline import IncidentPipeline
from crisisstate.storage.database import Database
from crisisstate.storage.repository import CrisisRepository


@pytest.fixture(autouse=True)
def setup_test_api():
    test_db = Database(":memory:")
    test_repo = CrisisRepository(test_db)
    test_pipeline = IncidentPipeline(test_repo)
    app.dependency_overrides[get_repository] = lambda: test_repo
    app.dependency_overrides[get_pipeline] = lambda: test_pipeline
    yield
    app.dependency_overrides.clear()


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "system": "CrisisState"}


def test_api_replay_and_query_endpoints():
    client = TestClient(app)

    # Trigger replay
    replay_resp = client.post("/api/replay")
    assert replay_resp.status_code == 200
    data = replay_resp.json()
    assert data["reports_processed"] == 9
    assert data["incidents_created"] == 2

    # Query reports
    reports_resp = client.get("/api/reports")
    assert reports_resp.status_code == 200
    assert len(reports_resp.json()) == 9

    # Query incidents
    inc_resp = client.get("/api/incidents")
    assert inc_resp.status_code == 200
    incidents = inc_resp.json()
    assert len(incidents) >= 2

    # Query incident detail
    first_id = incidents[0]["id"]
    detail_resp = client.get(f"/api/incidents/{first_id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert "incident" in detail_data
    assert "claims" in detail_data
    assert "evidence_links" in detail_data

    # Query incident timeline
    timeline_resp = client.get(f"/api/incidents/{first_id}/timeline")
    assert timeline_resp.status_code == 200
    timeline = timeline_resp.json()
    assert len(timeline) > 0


def test_ingest_single_report_endpoint():
    client = TestClient(app)
    payload = {
        "text": "Market Square is flooded and road is impassable.",
        "source_type": "citizen",
        "source_id": "app_user_42",
        "latitude": 28.7000,
        "longitude": 77.3000,
        "metadata": {"location": "Market Square"},
    }
    resp = client.post("/api/reports", json=payload)
    assert resp.status_code == 200
    res = resp.json()
    assert res["incident_title"] == "Urban Flooding at Market Square"
    assert res["is_new_incident"] is True
    assert len(res["claims"]) >= 1
