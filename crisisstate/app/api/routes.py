"""FastAPI REST API routes for CrisisState."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from crisisstate.domain.models import Incident, Report
from crisisstate.engine.pipeline import IncidentPipeline
from crisisstate.replay.runner import ReplayRunner
from crisisstate.storage.database import Database
from crisisstate.storage.repository import CrisisRepository


# Shared database and repository instance for API
_db = Database("crisisstate.db")
_repo = CrisisRepository(_db)
_pipeline = IncidentPipeline(_repo)


def get_repository() -> CrisisRepository:
    return _repo


def get_pipeline() -> IncidentPipeline:
    return _pipeline


router = APIRouter(prefix="/api")


class ReportIngestRequest(BaseModel):
    id: Optional[str] = None
    text: str
    source_type: str = "citizen"
    source_id: str = "web_input"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


@router.post("/reports", response_model=Dict[str, Any])
def ingest_report(
    payload: ReportIngestRequest,
    pipeline: IncidentPipeline = Depends(get_pipeline),
):
    """Ingest a single crisis report through the 10-step intelligence pipeline."""
    kwargs = {
        "text": payload.text,
        "source_type": payload.source_type,
        "source_id": payload.source_id,
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "metadata": payload.metadata,
    }
    if payload.id:
        kwargs["id"] = payload.id

    report = Report(**kwargs)
    result = pipeline.process_report(report)
    return result.to_dict()


@router.get("/reports", response_model=List[Dict[str, Any]])
def list_reports(repo: CrisisRepository = Depends(get_repository)):
    """List all ingested reports chronologically."""
    reports = repo.list_reports()
    return [r.model_dump(mode="json") for r in reports]


@router.get("/incidents", response_model=List[Dict[str, Any]])
def list_incidents(repo: CrisisRepository = Depends(get_repository)):
    """List all tracked incidents with current status."""
    incidents = repo.list_incidents()
    return [inc.model_dump(mode="json") for inc in incidents]


@router.get("/incidents/{incident_id}", response_model=Dict[str, Any])
def get_incident_detail(
    incident_id: str,
    repo: CrisisRepository = Depends(get_repository),
):
    """Get full details of a specific incident including claims and evidence."""
    inc = repo.get_incident(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found")

    claims = repo.list_claims_for_incident(incident_id)
    evidence = repo.list_evidence_for_incident(incident_id)

    return {
        "incident": inc.model_dump(mode="json"),
        "claims": [c.model_dump(mode="json") for c in claims],
        "evidence_links": [e.model_dump(mode="json") for e in evidence],
    }


@router.get("/incidents/{incident_id}/timeline", response_model=List[Dict[str, Any]])
def get_incident_timeline(
    incident_id: str,
    repo: CrisisRepository = Depends(get_repository),
):
    """Retrieve the chronological state change snapshots for an incident."""
    inc = repo.get_incident(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found")

    snapshots = repo.list_snapshots_for_incident(incident_id)
    return [s.model_dump(mode="json") for s in snapshots]


@router.post("/replay", response_model=Dict[str, Any])
def trigger_replay(repo: CrisisRepository = Depends(get_repository)):
    """Replay the synthetic flooding scenario and return final incident states."""
    runner = ReplayRunner(repository=repo)
    summary = runner.run(verbose=False)
    return summary
