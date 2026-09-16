"""10-Step Uncertainty-Aware Incident Processing Pipeline."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from crisisstate.domain.models import (
    Claim,
    EvidenceLink,
    Incident,
    IncidentStateSnapshot,
    Report,
    ensure_utc,
)
from crisisstate.domain.vocabulary import (
    AttentionLevel,
    ClaimStatus,
    ClaimType,
    EvidenceType,
    SeverityLevel,
)
from crisisstate.engine.attention import AttentionScorer
from crisisstate.engine.contradiction import ContradictionEngine
from crisisstate.engine.extractor import extract_claims
from crisisstate.engine.matcher import IncidentMatcher
from crisisstate.storage.repository import CrisisRepository


class PipelineResult:
    """Encapsulates the complete traceable output of processing a single report."""

    def __init__(
        self,
        report: Report,
        incident: Incident,
        is_new_incident: bool,
        extracted_claims: List[Claim],
        evidence_links: List[EvidenceLink],
        new_snapshots: List[IncidentStateSnapshot],
        contradiction_detected: bool,
    ):
        self.report = report
        self.incident = incident
        self.is_new_incident = is_new_incident
        self.extracted_claims = extracted_claims
        self.evidence_links = evidence_links
        self.new_snapshots = new_snapshots
        self.contradiction_detected = contradiction_detected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report.id,
            "incident_id": self.incident.id,
            "incident_title": self.incident.title,
            "is_new_incident": self.is_new_incident,
            "claims_extracted_count": len(self.extracted_claims),
            "claims": [
                {
                    "id": c.id,
                    "claim_type": c.claim_type.value,
                    "subject": c.subject,
                    "value": c.value,
                    "status": c.status.value,
                    "supporting_evidence": c.supporting_evidence,
                    "contradicting_evidence": c.contradicting_evidence,
                }
                for c in self.extracted_claims
            ],
            "contradiction_detected": self.contradiction_detected,
            "evidence_links": [
                {
                    "id": ev.id,
                    "link_type": ev.link_type.value,
                    "claim_id": ev.claim_id,
                    "report_id": ev.report_id,
                    "reason": ev.reason,
                }
                for ev in self.evidence_links
            ],
            "incident_state": {
                "severity": self.incident.current_severity.value,
                "attention_level": self.incident.current_attention_level.value,
                "attention_factors": self.incident.attention_factors,
                "state_summary": self.incident.state_summary,
            },
            "snapshots_generated": [
                {
                    "changed_field": s.changed_field,
                    "previous_value": s.previous_value,
                    "new_value": s.new_value,
                    "reason": s.reason,
                    "evidence_reference": s.evidence_reference,
                }
                for s in self.new_snapshots
            ],
        }


class IncidentPipeline:
    """
    Modular engine executing the 10-step incident intelligence pipeline.
    """

    def __init__(self, repository: CrisisRepository):
        self.repo = repository
        self.matcher = IncidentMatcher()
        self.contradiction_engine = ContradictionEngine()
        self.scorer = AttentionScorer()

    def process_report(self, report: Report) -> PipelineResult:
        """Execute the full 10-step pipeline on an incoming report."""

        # Step 1: Ingest report & Step 2: Normalize basic metadata
        normalized_report = self._normalize_report(report)
        self.repo.save_report(normalized_report)

        # Step 3: Determine incident association (match existing or create new)
        active_incidents = self.repo.list_incidents()
        incident, is_new = self.matcher.match_or_create(
            normalized_report, active_incidents
        )

        # Step 4: Extract claims
        new_claims = extract_claims(normalized_report)

        # Step 5: Link claims to the incident
        for claim in new_claims:
            claim.incident_id = incident.id

        # Step 6 & 7: Compare new claims against existing claims and detect contradictions
        existing_claims = self.repo.list_claims_for_incident(incident.id)
        all_evidence_links: List[EvidenceLink] = []
        contradiction_found = False

        for new_claim in new_claims:
            links = self.contradiction_engine.evaluate_claim(
                new_claim,
                existing_claims,
                incident.id,
                normalized_report,
            )
            for link in links:
                if link.link_type == EvidenceType.CONTRADICTS:
                    contradiction_found = True
                self.repo.save_evidence_link(link)
                all_evidence_links.append(link)

            # Persist the new claim
            self.repo.save_claim(new_claim)

        # Save any updated statuses on existing claims
        for old_claim in existing_claims:
            self.repo.save_claim(old_claim)

        # Step 8: Attach supporting / contradicting evidence & update incident links
        if normalized_report.id not in incident.linked_report_ids:
            incident.linked_report_ids.append(normalized_report.id)
        for c in new_claims:
            if c.id not in incident.claim_ids:
                incident.claim_ids.append(c.id)

        # Step 9: Update incident state and record snapshots
        all_incident_claims = self.repo.list_claims_for_incident(incident.id)
        snapshots = self._update_incident_state(
            incident,
            all_incident_claims,
            normalized_report,
            all_evidence_links,
            is_new,
        )

        # Persist snapshots
        for s in snapshots:
            self.repo.save_snapshot(s)

        # Persist updated incident
        self.repo.save_incident(incident)

        # Step 10: Produce traceable representation
        return PipelineResult(
            report=normalized_report,
            incident=incident,
            is_new_incident=is_new,
            extracted_claims=new_claims,
            evidence_links=all_evidence_links,
            new_snapshots=snapshots,
            contradiction_detected=contradiction_found,
        )

    def _normalize_report(self, report: Report) -> Report:
        """Normalize metadata and text."""
        cleaned_text = " ".join(report.text.strip().split())
        cleaned_source = report.source_type.strip().lower()
        return Report(
            id=report.id,
            text=cleaned_text,
            timestamp=ensure_utc(report.timestamp),
            source_type=cleaned_source,
            source_id=report.source_id.strip(),
            latitude=round(report.latitude, 6) if report.latitude is not None else None,
            longitude=round(report.longitude, 6) if report.longitude is not None else None,
            metadata=report.metadata or {},
        )

    def _update_incident_state(
        self,
        incident: Incident,
        claims: List[Claim],
        latest_report: Report,
        new_evidence_links: List[EvidenceLink],
        is_new: bool,
    ) -> List[IncidentStateSnapshot]:
        """Recalculates state summary, severity, attention score and produces state change snapshots."""
        snapshots: List[IncidentStateSnapshot] = []
        old_severity = incident.current_severity
        old_attention = incident.current_attention_level
        old_state_summary = dict(incident.state_summary)

        # Group claims by (claim_type, subject)
        grouped_claims: Dict[Tuple[str, str], List[Claim]] = {}
        for c in claims:
            key = (c.claim_type.value, c.subject)
            grouped_claims.setdefault(key, []).append(c)

        new_state_summary: Dict[str, str] = {}
        state_reasons: Dict[str, str] = {}

        for (ctype, subj), c_list in grouped_claims.items():
            summary_key = f"{ctype}:{subj}"
            # Consider only non-superseded claims for active state
            active_claims = [c for c in c_list if c.status != ClaimStatus.SUPERSEDED]
            if not active_claims:
                active_claims = c_list

            contradicted = [c for c in active_claims if c.status == ClaimStatus.CONTRADICTED]

            if contradicted:
                # We have active contradiction
                unique_values = sorted(list({c.value for c in active_claims}))
                new_state_summary[summary_key] = f"CONFLICTING ({'/'.join(unique_values)})"
                contr_reps = sorted(list({c.report_id for c in contradicted}))
                state_reasons[summary_key] = (
                    f"Conflicting claims across reports [{', '.join(contr_reps)}] for {ctype} on {subj}"
                )
            else:
                # Latest claim determines value
                latest_c = sorted(active_claims, key=lambda x: x.timestamp)[-1]
                new_state_summary[summary_key] = latest_c.value
                state_reasons[summary_key] = (
                    f"Report {latest_c.report_id} established {ctype}={latest_c.value}"
                )

        incident.state_summary = new_state_summary

        # Active claims for attention computation
        non_superseded = [c for c in claims if c.status != ClaimStatus.SUPERSEDED]
        active_eval_claims = non_superseded if non_superseded else claims

        # Attention and Severity computation
        att_level, sev_level, factors, comp_score = self.scorer.compute_attention(
            incident=incident,
            claims=active_eval_claims,
            reference_time=latest_report.timestamp,
        )

        incident.current_severity = sev_level
        incident.current_attention_level = att_level
        incident.attention_factors = factors
        incident.current_confidence = round(1.0 - factors["uncertainty"], 3)
        incident.updated_at = latest_report.timestamp

        # Detect diffs for snapshots
        # 1. State summary diffs
        for key, new_val in new_state_summary.items():
            prev_val = old_state_summary.get(key)
            if prev_val != new_val:
                snapshots.append(
                    IncidentStateSnapshot(
                        incident_id=incident.id,
                        timestamp=latest_report.timestamp,
                        changed_field=key,
                        previous_value=prev_val,
                        new_value=new_val,
                        reason=state_reasons.get(key, f"Updated by Report {latest_report.id}"),
                        evidence_reference=latest_report.id,
                    )
                )

        # 2. Severity change snapshot
        if is_new or old_severity != sev_level:
            snapshots.append(
                IncidentStateSnapshot(
                    incident_id=incident.id,
                    timestamp=latest_report.timestamp,
                    changed_field="SEVERITY",
                    previous_value=None if is_new else old_severity.value,
                    new_value=sev_level.value,
                    reason=f"Severity assessed as {sev_level.value} from active incident claims",
                    evidence_reference=latest_report.id,
                )
            )

        # 3. Attention level change snapshot
        if is_new or old_attention != att_level:
            snapshots.append(
                IncidentStateSnapshot(
                    incident_id=incident.id,
                    timestamp=latest_report.timestamp,
                    changed_field="ATTENTION_LEVEL",
                    previous_value=None if is_new else old_attention.value,
                    new_value=att_level.value,
                    reason=(
                        f"Attention score shifted to {att_level.value} "
                        f"(factors: sev={factors['severity']}, conflict={factors['conflict']}, recency={factors['recency']})"
                    ),
                    evidence_reference=latest_report.id,
                )
            )

        return snapshots
