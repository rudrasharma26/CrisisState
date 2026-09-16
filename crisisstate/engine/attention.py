"""Transparent attention scoring mechanism and severity evaluator."""

from datetime import datetime
from typing import Dict, List, Tuple
from crisisstate.domain.models import Claim, Incident, ensure_utc
from crisisstate.domain.vocabulary import (
    AttentionLevel,
    ClaimStatus,
    ClaimType,
    SeverityLevel,
)


SEVERITY_WEIGHTS: Dict[Tuple[ClaimType, str], float] = {
    (ClaimType.PEOPLE_TRAPPED, "YES"): 1.0,
    (ClaimType.BUILDING_DAMAGE, "COLLAPSED"): 0.95,
    (ClaimType.WATER_LEVEL, "CRITICAL"): 0.90,
    (ClaimType.ROAD_ACCESS, "IMPASSABLE"): 0.75,
    (ClaimType.WATER_LEVEL, "HIGH"): 0.70,
    (ClaimType.BUILDING_DAMAGE, "DAMAGED"): 0.65,
    (ClaimType.TRAFFIC_STATUS, "STOPPED"): 0.50,
    (ClaimType.WATER_LEVEL, "MODERATE"): 0.40,
    (ClaimType.FLOODED, "YES"): 0.40,
    (ClaimType.ROAD_ACCESS, "PASSABLE"): 0.20,
    (ClaimType.WATER_LEVEL, "LOW"): 0.15,
    (ClaimType.FLOODED, "NO"): 0.05,
    (ClaimType.PEOPLE_TRAPPED, "NO"): 0.05,
}


class AttentionScorer:
    """Computes transparent attention score and interpretable factors for an incident."""

    def compute_attention(
        self,
        incident: Incident,
        claims: List[Claim],
        reference_time: datetime,
    ) -> Tuple[AttentionLevel, SeverityLevel, Dict[str, float], float]:
        """
        Calculates:
        1. attention_level (LOW, MEDIUM, HIGH, CRITICAL)
        2. severity_level (INFORMATIONAL, LOW, MODERATE, HIGH, CRITICAL)
        3. factor breakdown (severity, conflict, recency, new_evidence, uncertainty)
        4. overall composite score (0.0 to 1.0)
        """
        if not claims:
            return (
                AttentionLevel.LOW,
                SeverityLevel.LOW,
                {
                    "severity": 0.1,
                    "conflict": 0.0,
                    "recency": 0.5,
                    "new_evidence": 0.1,
                    "uncertainty": 0.5,
                },
                0.2,
            )

        # 1. Severity Factor
        max_sev = 0.1
        for claim in claims:
            w = SEVERITY_WEIGHTS.get((claim.claim_type, claim.value), 0.2)
            if w > max_sev:
                max_sev = w
        severity_factor = round(max_sev, 3)

        # Determine SeverityLevel enum
        if severity_factor >= 0.9:
            sev_level = SeverityLevel.CRITICAL
        elif severity_factor >= 0.7:
            sev_level = SeverityLevel.HIGH
        elif severity_factor >= 0.4:
            sev_level = SeverityLevel.MODERATE
        elif severity_factor >= 0.2:
            sev_level = SeverityLevel.LOW
        else:
            sev_level = SeverityLevel.INFORMATIONAL

        # 2. Conflict Factor
        contradicted_count = sum(1 for c in claims if c.status == ClaimStatus.CONTRADICTED)
        conflict_factor = round(
            min(1.0, (contradicted_count / max(1, len(claims))) * 1.5) if contradicted_count > 0 else 0.0,
            3,
        )

        # 3. Recency Factor
        time_elapsed_hours = max(
            0.0, (ensure_utc(reference_time) - ensure_utc(incident.updated_at)).total_seconds() / 3600.0
        )
        # Recency decays over 12 hours
        recency_factor = round(max(0.1, 1.0 - (time_elapsed_hours / 12.0)), 3)

        # 4. New Evidence / Volume Factor
        evidence_volume = len(incident.linked_report_ids)
        evidence_factor = round(min(1.0, evidence_volume / 5.0), 3)

        # 5. Uncertainty Factor
        # Uncertainty is high when conflicts exist or evidence is low/unverified
        if conflict_factor > 0:
            uncertainty_factor = round(min(1.0, 0.4 + (conflict_factor * 0.6)), 3)
        else:
            # Low conflict: uncertainty decreases as more reports corroborate
            uncertainty_factor = round(max(0.1, 0.6 - (evidence_factor * 0.4)), 3)

        # Composite Attention Score
        composite_score = round(
            (0.35 * severity_factor)
            + (0.25 * conflict_factor)
            + (0.15 * recency_factor)
            + (0.15 * uncertainty_factor)
            + (0.10 * evidence_factor),
            3,
        )

        if composite_score >= 0.75 or severity_factor >= 0.95:
            att_level = AttentionLevel.CRITICAL
        elif composite_score >= 0.55 or conflict_factor >= 0.5:
            att_level = AttentionLevel.HIGH
        elif composite_score >= 0.35:
            att_level = AttentionLevel.MEDIUM
        else:
            att_level = AttentionLevel.LOW

        factors = {
            "severity": severity_factor,
            "conflict": conflict_factor,
            "recency": recency_factor,
            "new_evidence": evidence_factor,
            "uncertainty": uncertainty_factor,
        }

        return att_level, sev_level, factors, composite_score
