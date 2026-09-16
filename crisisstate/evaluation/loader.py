"""Dataset loader and validator for the CrisisState evaluation corpus."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from crisisstate.evaluation.schemas import (
    GoldClaim,
    GoldContradiction,
    GoldIncident,
    GoldReport,
)


class EvaluationDataset:
    """Encapsulates a loaded and validated evaluation corpus."""

    def __init__(
        self,
        reports: List[GoldReport],
        incidents: List[GoldIncident],
        claims: List[GoldClaim],
        contradictions: List[GoldContradiction],
        dataset_dir: Path,
    ):
        self.reports = sorted(reports, key=lambda r: r.timestamp)
        self.incidents = incidents
        self.claims = claims
        self.contradictions = contradictions
        self.dataset_dir = dataset_dir

        self.reports_by_id: Dict[str, GoldReport] = {r.id: r for r in reports}
        self.incidents_by_id: Dict[str, GoldIncident] = {i.id: i for i in incidents}
        self.claims_by_id: Dict[str, GoldClaim] = {c.id: c for c in claims}

    def validate_integrity(self) -> List[str]:
        """Perform cross-reference validation and return any error messages."""
        errors: List[str] = []

        # Check that reports point to valid incidents
        for r in self.reports:
            if r.gold_incident_id not in self.incidents_by_id:
                errors.append(
                    f"Report '{r.id}' references unknown gold incident '{r.gold_incident_id}'"
                )

        # Check that contradictions reference valid claims and reports
        for c in self.contradictions:
            if c.claim_id_a not in self.claims_by_id:
                errors.append(f"Contradiction references unknown claim_id_a '{c.claim_id_a}'")
            if c.claim_id_b not in self.claims_by_id:
                errors.append(f"Contradiction references unknown claim_id_b '{c.claim_id_b}'")
            if c.report_id_a not in self.reports_by_id:
                errors.append(f"Contradiction references unknown report_id_a '{c.report_id_a}'")
            if c.report_id_b not in self.reports_by_id:
                errors.append(f"Contradiction references unknown report_id_b '{c.report_id_b}'")

        return errors


def load_evaluation_dataset(dataset_dir: Path) -> EvaluationDataset:
    """
    Load and parse all evaluation files from the specified directory.
    Validates each record against Pydantic schemas.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {dataset_dir}")

    reports_file = dataset_path / "reports.jsonl"
    incidents_file = dataset_path / "incidents.jsonl"
    claims_file = dataset_path / "claims.jsonl"
    contradictions_file = dataset_path / "contradictions.jsonl"

    incidents: List[GoldIncident] = []
    if incidents_file.exists():
        with open(incidents_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                incidents.append(GoldIncident(**data))

    reports: List[GoldReport] = []
    if reports_file.exists():
        with open(reports_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                reports.append(GoldReport(**data))

    claims: List[GoldClaim] = []
    if claims_file.exists():
        with open(claims_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                claims.append(GoldClaim(**data))
    else:
        # Extract from reports if claims.jsonl not separate
        for r in reports:
            claims.extend(r.gold_claims)

    contradictions: List[GoldContradiction] = []
    if contradictions_file.exists():
        with open(contradictions_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                contradictions.append(GoldContradiction(**data))

    dataset = EvaluationDataset(
        reports=reports,
        incidents=incidents,
        claims=claims,
        contradictions=contradictions,
        dataset_dir=dataset_path,
    )

    errors = dataset.validate_integrity()
    if errors:
        raise ValueError(f"Evaluation dataset validation failed with errors: {errors}")

    return dataset
