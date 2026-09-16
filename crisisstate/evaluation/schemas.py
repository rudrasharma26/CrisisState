"""Pydantic schemas for the CrisisState evaluation harness."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from crisisstate.domain.vocabulary import ClaimType, EvidenceType


class GoldModality(str, Enum):
    """Anticipated modality values for claims."""
    ASSERTED = "ASSERTED"
    QUALIFIED = "QUALIFIED"
    NEGATED = "NEGATED"
    AMBIGUOUS = "AMBIGUOUS"


class TestCategory(str, Enum):
    """Test case categories for evaluation corpus."""
    __test__ = False
    EXACT_LEXICAL = "EXACT_LEXICAL"
    PARAPHRASE = "PARAPHRASE"
    NEGATION = "NEGATION"
    HEDGING = "HEDGING"
    QUALIFIED_SCOPED = "QUALIFIED_SCOPED"
    GEO_CLOSE_DISTINCT = "GEO_CLOSE_DISTINCT"
    SEMANTIC_SIMILAR_DISTINCT = "SEMANTIC_SIMILAR_DISTINCT"
    CONTRADICTION = "CONTRADICTION"
    TEMPORAL_SUPERSESSION = "TEMPORAL_SUPERSESSION"


class GoldIncident(BaseModel):
    """Gold standard incident definition."""
    id: str
    title: str
    location_name: str
    latitude: float
    longitude: float
    description: Optional[str] = None


class GoldClaim(BaseModel):
    """Gold standard claim annotation."""
    id: str
    report_id: str
    claim_type: ClaimType
    subject: str
    value: str
    modality: GoldModality = GoldModality.ASSERTED
    confidence: float = 1.0


class GoldContradiction(BaseModel):
    """Gold standard pair of contradicting claims or reports."""
    claim_id_a: str
    claim_id_b: str
    report_id_a: str
    report_id_b: str
    incident_id: str
    relationship: str = "CONTRADICTS"  # "CONTRADICTS" or "SUPERSEDES"
    reason: str


class GoldReport(BaseModel):
    """Gold standard evaluation report."""
    id: str
    text: str
    timestamp: datetime
    source_type: str = "citizen"
    source_id: str = "eval_source"
    location: str
    latitude: float
    longitude: float
    gold_incident_id: str
    category: TestCategory
    gold_claims: List[GoldClaim] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Metric and Summary Schemas
class MetricScore(BaseModel):
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


class IncidentMatchingMetrics(BaseModel):
    false_merge_rate: float = 0.0
    missed_merge_rate: float = 0.0
    total_distinct_pairs: int = 0
    total_same_pairs: int = 0


class PerformanceMetrics(BaseModel):
    total_runtime_seconds: float = 0.0
    avg_latency_per_report_ms: float = 0.0


class EvaluationSummary(BaseModel):
    system_version: str
    dataset_version: str
    total_reports: int
    category_counts: Dict[str, int]
    claim_extraction: MetricScore
    claim_normalization_accuracy: float
    modality_accuracy: float
    incident_matching: IncidentMatchingMetrics
    contradiction_detection: MetricScore
    determinism: str  # "PASS" or "FAIL"
    performance: PerformanceMetrics
    generated_at: datetime = Field(default_factory=datetime.utcnow)
