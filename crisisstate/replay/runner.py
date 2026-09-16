"""Replay runner for sequential processing of crisis report streams."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from crisisstate.data import SYNTHETIC_DATA_PATH
from crisisstate.domain.models import Incident, Report
from crisisstate.engine.pipeline import IncidentPipeline, PipelineResult
from crisisstate.storage.database import Database
from crisisstate.storage.repository import CrisisRepository


def load_reports_from_file(file_path: Optional[Path] = None) -> List[Report]:
    """Load and parse reports from a JSON file."""
    path = file_path or SYNTHETIC_DATA_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    reports = []
    for item in raw_data:
        ts = item.get("timestamp")
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if isinstance(ts, str) else datetime.utcnow()
        reports.append(
            Report(
                id=item["id"],
                text=item["text"],
                timestamp=dt,
                source_type=item.get("source_type", "citizen"),
                source_id=item.get("source_id", "anon"),
                latitude=item.get("latitude"),
                longitude=item.get("longitude"),
                metadata=item.get("metadata", {}),
            )
        )
    # Ensure sorted by chronological timestamp
    return sorted(reports, key=lambda r: r.timestamp)


class ReplayRunner:
    """Executes a chronological replay of crisis reports."""

    def __init__(self, repository: Optional[CrisisRepository] = None):
        if repository is None:
            db = Database(":memory:")
            self.repository = CrisisRepository(db)
        else:
            self.repository = repository
        self.pipeline = IncidentPipeline(self.repository)

    def run(
        self,
        reports: Optional[List[Report]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Process report stream and return execution metrics and final incident states."""
        if reports is None:
            reports = load_reports_from_file()

        results: List[PipelineResult] = []
        contradiction_count = 0
        total_claims_extracted = 0

        if verbose:
            print("=" * 80)
            print(" CRISISSTATE REPLAY ENGINE - URBAN FLOODING INCIDENT INTELLIGENCE")
            print("=" * 80)
            print(f"Total reports to process: {len(reports)}\n")

        for idx, report in enumerate(reports, start=1):
            res = self.pipeline.process_report(report)
            results.append(res)
            total_claims_extracted += len(res.extracted_claims)

            if res.contradiction_detected:
                contradiction_count += 1

            if verbose:
                self._print_step(idx, res)

        incidents = self.repository.list_incidents()

        if verbose:
            self._print_summary(incidents, results, contradiction_count)

        return {
            "reports_processed": len(reports),
            "incidents_created": len(incidents),
            "claims_extracted": total_claims_extracted,
            "contradictions_detected": contradiction_count,
            "incidents": [
                {
                    "id": inc.id,
                    "title": inc.title,
                    "location": inc.location_name,
                    "severity": inc.current_severity.value,
                    "attention_level": inc.current_attention_level.value,
                    "attention_factors": inc.attention_factors,
                    "confidence": inc.current_confidence,
                    "state_summary": inc.state_summary,
                    "total_reports": len(inc.linked_report_ids),
                    "total_claims": len(inc.claim_ids),
                }
                for inc in incidents
            ],
        }

    def _print_step(self, step: int, res: PipelineResult) -> None:
        rep = res.report
        inc = res.incident
        print(f"--- [STEP {step:02d}] Ingesting Report: {rep.id} ({rep.timestamp.isoformat()}) ---")
        print(f"  Source: {rep.source_type} [{rep.source_id}] | Location: {inc.location_name}")
        print(f"  Text: \"{rep.text}\"")
        action = "Created NEW Incident" if res.is_new_incident else "Matched Existing Incident"
        print(f"  Incident Association: [{inc.id}] {inc.title} ({action})")

        # Claims
        if res.extracted_claims:
            claims_str = ", ".join(
                f"{c.claim_type.value}={c.value} (status: {c.status.value})"
                for c in res.extracted_claims
            )
            print(f"  Extracted Claims: {claims_str}")
        else:
            print("  Extracted Claims: None")

        # Evidence & Contradictions
        if res.evidence_links:
            for link in res.evidence_links:
                prefix = "  [!] CONTRADICTION" if link.link_type.value == "CONTRADICTS" else "  [+] CORROBORATION"
                print(f"{prefix}: {link.reason}")

        # Snapshots / State updates
        if res.new_snapshots:
            print("  State Changes / Snapshots:")
            for s in res.new_snapshots:
                prev = s.previous_value or "None"
                print(f"    * {s.changed_field}: {prev} -> {s.new_value} | Reason: {s.reason}")

        # Current Attention
        att = inc.current_attention_level.value
        sev = inc.current_severity.value
        f = inc.attention_factors
        print(
            f"  Current Status: Severity={sev} | Attention={att} | "
            f"Factors: [sev={f.get('severity')}, conflict={f.get('conflict')}, "
            f"recency={f.get('recency')}, unc={f.get('uncertainty')}]"
        )
        print()

    def _print_summary(
        self,
        incidents: List[Incident],
        results: List[PipelineResult],
        contradiction_count: int,
    ) -> None:
        print("=" * 80)
        print(" REPLAY SUMMARY & EVOLVING INCIDENT INTELLIGENCE STATE")
        print("=" * 80)
        print(f"Total Incidents Identified: {len(incidents)}")
        print(f"Total Contradictions Detected: {contradiction_count}")
        print("-" * 80)

        for inc in incidents:
            print(f"Incident: {inc.title} (ID: {inc.id})")
            print(f"  Location: {inc.location_name} (Coordinates: {inc.latitude}, {inc.longitude})")
            print(f"  Severity: {inc.current_severity.value}")
            print(f"  Attention Level: {inc.current_attention_level.value}")
            print(f"  Attention Factors: {json.dumps(inc.attention_factors, indent=4)}")
            print(f"  Confidence Score: {inc.current_confidence}")
            print(f"  Linked Reports: {len(inc.linked_report_ids)} {inc.linked_report_ids}")
            print("  Known State Summary:")
            for k, v in inc.state_summary.items():
                print(f"    - {k}: {v}")

            snapshots = self.repository.list_snapshots_for_incident(inc.id)
            print(f"  Timeline History ({len(snapshots)} state transitions):")
            for idx, s in enumerate(snapshots, start=1):
                prev = s.previous_value or "INIT"
                ref = f" (Ref: {s.evidence_reference})" if s.evidence_reference else ""
                print(f"    [{idx:02d}] {s.timestamp.strftime('%H:%M:%S')} - {s.changed_field}: {prev} -> {s.new_value}{ref}")
                print(f"         Reason: {s.reason}")
            print()
