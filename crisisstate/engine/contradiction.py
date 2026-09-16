"""Deterministic contradiction detection and evidence linking engine."""

from typing import List, Tuple
from crisisstate.domain.models import Claim, EvidenceLink, Report, ensure_utc
from crisisstate.domain.vocabulary import (
    ClaimStatus,
    EvidenceType,
    are_values_contradictory,
    are_values_supporting,
)


class ContradictionEngine:
    """Evaluates claims for contradictions and produces traceable evidence links."""

    def __init__(self, contradiction_window_hours: float = 2.0):
        self.contradiction_window_hours = contradiction_window_hours

    def evaluate_claim(
        self,
        new_claim: Claim,
        existing_claims: List[Claim],
        incident_id: str,
        report: Report,
    ) -> List[EvidenceLink]:
        """
        Compares new_claim against existing claims of the incident.
        Creates EvidenceLink entries for supports, contradictions, or state updates,
        and updates claim statuses accordingly.
        """
        generated_links: List[EvidenceLink] = []

        for old_claim in existing_claims:
            # Must refer to the same incident and claim type
            if old_claim.claim_type != new_claim.claim_type:
                continue

            # Must refer to the same or closely related subject/entity
            if old_claim.subject.strip().lower() != new_claim.subject.strip().lower():
                continue

            time_diff_hours = abs((ensure_utc(new_claim.timestamp) - ensure_utc(old_claim.timestamp)).total_seconds()) / 3600.0

            # 1. Check for Incompatible Values
            if are_values_contradictory(
                new_claim.claim_type, new_claim.value, old_claim.value
            ):
                if time_diff_hours <= self.contradiction_window_hours:
                    # Contemporaneous contradiction
                    reason = (
                        f"Report {report.id} claims {new_claim.claim_type.value}='{new_claim.value}' "
                        f"on {new_claim.subject}, which directly contradicts earlier contemporaneous claim "
                        f"{old_claim.id} ('{old_claim.value}') from Report {old_claim.report_id} "
                        f"within the active time window ({time_diff_hours:.1f}h apart)."
                    )

                    link = EvidenceLink(
                        report_id=report.id,
                        claim_id=old_claim.id,
                        incident_id=incident_id,
                        link_type=EvidenceType.CONTRADICTS,
                        confidence=min(new_claim.confidence, old_claim.confidence),
                        reason=reason,
                        created_at=report.timestamp,
                    )
                    generated_links.append(link)

                    # Update statuses and evidence references
                    new_claim.status = ClaimStatus.CONTRADICTED
                    old_claim.status = ClaimStatus.CONTRADICTED

                    if report.id not in old_claim.contradicting_evidence:
                        old_claim.contradicting_evidence.append(report.id)
                    if old_claim.report_id not in new_claim.contradicting_evidence:
                        new_claim.contradicting_evidence.append(old_claim.report_id)
                else:
                    # Temporal state progression (supersedes older claim)
                    reason = (
                        f"Report {report.id} updates {new_claim.claim_type.value} from '{old_claim.value}' "
                        f"to '{new_claim.value}' on {new_claim.subject} after {time_diff_hours:.1f} hours, "
                        f"superseding earlier claim {old_claim.id}."
                    )
                    link = EvidenceLink(
                        report_id=report.id,
                        claim_id=old_claim.id,
                        incident_id=incident_id,
                        link_type=EvidenceType.NEUTRAL,
                        confidence=new_claim.confidence,
                        reason=reason,
                        created_at=report.timestamp,
                    )
                    generated_links.append(link)
                    old_claim.status = ClaimStatus.SUPERSEDED
                    new_claim.status = ClaimStatus.ACTIVE

            # 2. Check for Corroboration / Support
            elif are_values_supporting(
                new_claim.claim_type, new_claim.value, old_claim.value
            ):
                reason = (
                    f"Report {report.id} corroborates {new_claim.claim_type.value}='{new_claim.value}' "
                    f"on {new_claim.subject}, supporting earlier claim {old_claim.id} from Report {old_claim.report_id}."
                )

                link = EvidenceLink(
                    report_id=report.id,
                    claim_id=old_claim.id,
                    incident_id=incident_id,
                    link_type=EvidenceType.SUPPORTS,
                    confidence=max(new_claim.confidence, old_claim.confidence),
                    reason=reason,
                    created_at=report.timestamp,
                )
                generated_links.append(link)

                if report.id not in old_claim.supporting_evidence:
                    old_claim.supporting_evidence.append(report.id)
                if old_claim.report_id not in new_claim.supporting_evidence:
                    new_claim.supporting_evidence.append(old_claim.report_id)

        return generated_links
