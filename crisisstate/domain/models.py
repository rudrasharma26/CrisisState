"""Domain entities and Pydantic models for CrisisState.

Phase 1 models (Report, Claim, EvidenceLink, IncidentStateSnapshot, Incident)
are unchanged in behaviour.  M1 adds new optional fields to Claim and
EvidenceLink with safe defaults so that all existing code paths continue to
work without modification.

New models introduced in M1 (ClaimExemplar, UnresolvedSpan, IncidentMatch,
ClaimStateEntry) are purely structural and not yet wired into the Phase 1
pipeline.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from crisisstate.domain.vocabulary import (
    AttentionLevel,
    ClaimStateStatus,
    ClaimStatus,
    ClaimType,
    EvidenceType,
    ExemplarSource,
    ExtractionMethod,
    MatchReason,
    SeverityLevel,
    ValueModality,
)


def ensure_utc(dt: Optional[datetime] = None) -> datetime:
    """Ensure datetime is timezone-aware in UTC."""
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ── Existing Phase 1 models ──────────────────────────────────────────────────

class Report(BaseModel):
    id: str = Field(default_factory=lambda: f"rep_{uuid.uuid4().hex[:8]}")
    text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_type: str = "citizen"
    source_id: str = "anonymous"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    """A single atomic proposition extracted from a report.

    Phase 1 fields are unchanged.  M1 adds optional semantic-layer fields
    that all default to values consistent with Phase 1 lexical extraction:

    * extraction_method  → ExtractionMethod.LEXICAL
    * extraction_confidence → mirrors the existing `confidence` value
    * matched_exemplar_id   → None
    * similarity_score      → None
    * source_span           → None
    * value_modality        → ValueModality.ASSERTED
    * qualifier             → None
    * candidate_values      → []
    """

    id: str = Field(default_factory=lambda: f"clm_{uuid.uuid4().hex[:8]}")
    incident_id: Optional[str] = None
    report_id: str
    claim_type: ClaimType
    subject: str  # Entity / landmark / road
    value: str    # e.g. "PASSABLE", "IMPASSABLE", "HIGH"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: ClaimStatus = ClaimStatus.ACTIVE
    confidence: float = 1.0
    supporting_evidence: List[str] = Field(default_factory=list)   # Report IDs or EvidenceLink IDs
    contradicting_evidence: List[str] = Field(default_factory=list)

    # ── M1: Semantic-layer preparation fields ──────────────────────────────
    # These fields have defaults that preserve exact Phase 1 behaviour.
    # No Phase 1 code reads or writes them; they are populated below.

    extraction_method: ExtractionMethod = ExtractionMethod.LEXICAL
    """How the claim was extracted.  Phase 1 always sets LEXICAL."""

    extraction_confidence: Optional[float] = None
    """Confidence specific to the extraction step.
    Phase 1 populates this from the pattern-rule confidence weight."""

    matched_exemplar_id: Optional[str] = None
    """ID of the ClaimExemplar that triggered this claim.  None in Phase 1."""

    similarity_score: Optional[float] = None
    """Semantic similarity score against the matched exemplar.  None in Phase 1."""

    source_span: Optional[Dict[str, Any]] = None
    """Character-offset span in the source report text.
    Example: {"start": 12, "end": 28, "text": "road is blocked"}
    None in Phase 1."""

    value_modality: ValueModality = ValueModality.ASSERTED
    """Epistemic modality of the extracted value.  Defaults to ASSERTED in Phase 1.
    Modality detection is reserved for M2+."""

    qualifier: Optional[Dict[str, Any]] = None
    """Structured qualifier for scoped claims.
    Example: {"permitted_scope": "EMERGENCY_VEHICLES_ONLY"}
    None in Phase 1."""

    candidate_values: List[str] = Field(default_factory=list)
    """Competing semantic candidate values before adjudication.
    Empty list in Phase 1; populated in M3+."""


class EvidenceLink(BaseModel):
    """A typed relationship between a report and a claim.

    Phase 1 fields are unchanged.  M1 adds three optional fields for future
    multi-source corroboration strength modelling:

    * strength_components → {}
    * strength_total      → None
    * independence_group_id → None
    """

    id: str = Field(default_factory=lambda: f"evd_{uuid.uuid4().hex[:8]}")
    report_id: str
    claim_id: str
    incident_id: str
    link_type: EvidenceType
    confidence: float = 1.0
    reason: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── M1: Evidence strength preparation fields ───────────────────────────

    strength_components: Dict[str, float] = Field(default_factory=dict)
    """Named components that will contribute to the total evidence strength.
    Empty in Phase 1; populated in M3+."""

    strength_total: Optional[float] = None
    """Composite evidence strength score [0, 1].  None in Phase 1."""

    independence_group_id: Optional[str] = None
    """Groups corroborating links that share a common information source,
    used to penalise correlated evidence.  None in Phase 1."""


class IncidentStateSnapshot(BaseModel):
    id: str = Field(default_factory=lambda: f"snp_{uuid.uuid4().hex[:8]}")
    incident_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    changed_field: str  # e.g. "ROAD_ACCESS:Main Road" or "SEVERITY"
    previous_value: Optional[str] = None
    new_value: str
    reason: str
    evidence_reference: Optional[str] = None  # Report or Evidence link ID


class Incident(BaseModel):
    id: str = Field(default_factory=lambda: f"inc_{uuid.uuid4().hex[:8]}")
    title: str
    location_name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    current_severity: SeverityLevel = SeverityLevel.LOW
    current_attention_level: AttentionLevel = AttentionLevel.LOW
    attention_factors: Dict[str, float] = Field(default_factory=dict)
    current_confidence: float = 0.5
    linked_report_ids: List[str] = Field(default_factory=list)
    claim_ids: List[str] = Field(default_factory=list)
    state_summary: Dict[str, str] = Field(default_factory=dict)
    # "ROAD_ACCESS:Main Road" -> "CONFLICTING" | "PASSABLE"

    # ── M1: Structured per-(incident, claim_type) state ───────────────────
    # claim_state_entries is a preparation field for M2+ structured state.
    # Phase 1 does not populate it; the existing state_summary dict remains
    # the authoritative state representation until explicitly migrated.
    claim_state_entries: List[Dict[str, Any]] = Field(default_factory=list)
    """Structured ClaimStateEntry objects (serialised as dicts).
    Empty in Phase 1; populated in future milestones."""


# ── M1: New domain models (not wired into Phase 1 pipeline) ─────────────────

class ClaimExemplar(BaseModel):
    """A canonical text exemplar that typifies a (claim_type, value) pair.

    Used by the future semantic extraction layer (M3) to find similar
    expressions in incoming reports.  No embeddings are stored here yet.

    Source may be HAND_AUTHORED (from evaluation corpus) or CORPUS_DERIVED.
    """

    id: str = Field(default_factory=lambda: f"exm_{uuid.uuid4().hex[:8]}")
    claim_type: ClaimType
    value: str
    text: str
    source: ExemplarSource = ExemplarSource.HAND_AUTHORED
    version: str = "v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UnresolvedSpan(BaseModel):
    """A text span from a report that could not be resolved to a canonical claim.

    Phase 1 never creates UnresolvedSpan objects.  The model is defined here
    so that future semantic extraction stages can surface ambiguous text spans
    for human review or downstream resolution.
    """

    id: str = Field(default_factory=lambda: f"usp_{uuid.uuid4().hex[:8]}")
    report_id: str
    span_text: str
    top_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    """Top candidate (claim_type, value) pairs with scores, e.g.:
    [{"claim_type": "ROAD_ACCESS", "value": "IMPASSABLE", "score": 0.72}, ...]"""
    rejection_reason: Optional[str] = None
    """Why the span could not be resolved (e.g. "BELOW_THRESHOLD", "AMBIGUOUS")."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IncidentMatch(BaseModel):
    """Diagnostic record capturing the outcome of a report-to-incident matching decision.

    Phase 1 matching produces ENTITY / SPATIAL / TEMPORAL / HYBRID decisions
    but does not persist them.  This model makes matching outcomes traceable
    and supports future semantic tiebreak scenarios.
    """

    id: str = Field(default_factory=lambda: f"imt_{uuid.uuid4().hex[:8]}")
    report_id: str
    incident_id: str
    match_reason: MatchReason
    gate_results: Dict[str, Any] = Field(default_factory=dict)
    """Breakdown of individual gate outcomes, e.g.:
    {"entity_match": true, "spatial_distance_km": 0.12, "time_diff_hours": 0.25}"""
    semantic_score: Optional[float] = None
    """Semantic similarity score used for tiebreaking.  None in Phase 1."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ClaimStateEntry(BaseModel):
    """Structured state representation for one (incident_id, claim_type, subject) tuple.

    Replaces the opaque string in state_summary in future milestones.
    Phase 1 does not write this model; it is defined here for M2+ migration.
    """

    id: str = Field(default_factory=lambda: f"cse_{uuid.uuid4().hex[:8]}")
    incident_id: str
    claim_type: ClaimType
    subject: str
    status: ClaimStateStatus
    dominant_value: Optional[str] = None
    competing_values: List[str] = Field(default_factory=list)
    supporting_claim_ids: List[str] = Field(default_factory=list)
    conflicting_claim_ids: List[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
