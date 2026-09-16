"""Domain vocabulary, claim types, allowed values, and contradiction matrix for CrisisState."""

from enum import Enum
from typing import Dict, Set, Tuple


class ClaimType(str, Enum):
    FLOODED = "FLOODED"
    ROAD_ACCESS = "ROAD_ACCESS"
    WATER_LEVEL = "WATER_LEVEL"
    PEOPLE_TRAPPED = "PEOPLE_TRAPPED"
    BUILDING_DAMAGE = "BUILDING_DAMAGE"
    EVACUATION = "EVACUATION"
    TRAFFIC_STATUS = "TRAFFIC_STATUS"


class ClaimStatus(str, Enum):
    ACTIVE = "ACTIVE"
    CONTRADICTED = "CONTRADICTED"
    SUPERSEDED = "SUPERSEDED"


class EvidenceType(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"


class AttentionLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SeverityLevel(str, Enum):
    INFORMATIONAL = "INFORMATIONAL"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ── M1: New enumerations for the future semantic layer ──────────────────────

class ExtractionMethod(str, Enum):
    """How a claim was extracted.

    Phase 1 always uses LEXICAL. SEMANTIC, NLI_CONFIRMED, and HYBRID are
    reserved for later milestones and must not alter Phase 1 behaviour.
    """
    LEXICAL = "LEXICAL"
    SEMANTIC = "SEMANTIC"
    NLI_CONFIRMED = "NLI_CONFIRMED"
    HYBRID = "HYBRID"


class ValueModality(str, Enum):
    """Epistemic modality of a claim value.

    Phase 1 defaults every extracted claim to ASSERTED.
    Detection of QUALIFIED / NEGATED / AMBIGUOUS is reserved for M2+.
    """
    ASSERTED = "ASSERTED"
    QUALIFIED = "QUALIFIED"
    NEGATED = "NEGATED"
    AMBIGUOUS = "AMBIGUOUS"


class ExemplarSource(str, Enum):
    """Provenance of a ClaimExemplar entry."""
    HAND_AUTHORED = "HAND_AUTHORED"
    CORPUS_DERIVED = "CORPUS_DERIVED"


class MatchReason(str, Enum):
    """Deterministic reason code for an incident match decision."""
    SPATIAL = "SPATIAL"
    TEMPORAL = "TEMPORAL"
    ENTITY = "ENTITY"
    HYBRID = "HYBRID"
    SEMANTIC_TIEBREAK = "SEMANTIC_TIEBREAK"


class ClaimStateStatus(str, Enum):
    """Per-(incident, claim_type) state representation.

    Replaces the plain-string state_summary values in future milestones.
    Phase 1 state_summary continues to use its existing string format.
    """
    SUPPORTED = "SUPPORTED"
    WEAKLY_SUPPORTED = "WEAKLY_SUPPORTED"
    CONFLICTING = "CONFLICTING"
    UNRESOLVED = "UNRESOLVED"
    SUPERSEDED = "SUPERSEDED"


# ── Existing vocabulary tables (unchanged) ──────────────────────────────────

# Allowed values per claim type
ALLOWED_CLAIM_VALUES: Dict[ClaimType, Set[str]] = {
    ClaimType.FLOODED: {"YES", "NO"},
    ClaimType.ROAD_ACCESS: {"PASSABLE", "IMPASSABLE"},
    ClaimType.WATER_LEVEL: {"LOW", "MODERATE", "HIGH", "CRITICAL"},
    ClaimType.PEOPLE_TRAPPED: {"YES", "NO"},
    ClaimType.BUILDING_DAMAGE: {"NONE", "DAMAGED", "COLLAPSED"},
    ClaimType.EVACUATION: {"ORDERED", "NOT_ORDERED", "IN_PROGRESS", "COMPLETED"},
    ClaimType.TRAFFIC_STATUS: {"MOVING", "SLOW", "STOPPED"},
}

# Explicit pairwise contradiction definitions
# If (val_a, val_b) is in the set for the claim type, they are incompatible.
CONTRADICTION_PAIRS: Dict[ClaimType, Set[Tuple[str, str]]] = {
    ClaimType.FLOODED: {
        ("YES", "NO"),
        ("NO", "YES"),
    },
    ClaimType.ROAD_ACCESS: {
        ("PASSABLE", "IMPASSABLE"),
        ("IMPASSABLE", "PASSABLE"),
    },
    ClaimType.WATER_LEVEL: {
        ("LOW", "HIGH"),
        ("HIGH", "LOW"),
        ("LOW", "CRITICAL"),
        ("CRITICAL", "LOW"),
        ("MODERATE", "CRITICAL"),
        ("CRITICAL", "MODERATE"),
    },
    ClaimType.PEOPLE_TRAPPED: {
        ("YES", "NO"),
        ("NO", "YES"),
    },
    ClaimType.BUILDING_DAMAGE: {
        ("NONE", "DAMAGED"),
        ("DAMAGED", "NONE"),
        ("NONE", "COLLAPSED"),
        ("COLLAPSED", "NONE"),
    },
    ClaimType.EVACUATION: {
        ("NOT_ORDERED", "ORDERED"),
        ("ORDERED", "NOT_ORDERED"),
        ("NOT_ORDERED", "IN_PROGRESS"),
        ("IN_PROGRESS", "NOT_ORDERED"),
    },
    ClaimType.TRAFFIC_STATUS: {
        ("MOVING", "STOPPED"),
        ("STOPPED", "MOVING"),
    },
}


def are_values_contradictory(claim_type: ClaimType, val_a: str, val_b: str) -> bool:
    """Check if two values for the given claim type directly contradict each other."""
    val_a = val_a.strip().upper()
    val_b = val_b.strip().upper()
    if val_a == val_b:
        return False
    pairs = CONTRADICTION_PAIRS.get(claim_type, set())
    return (val_a, val_b) in pairs


def are_values_supporting(claim_type: ClaimType, val_a: str, val_b: str) -> bool:
    """Check if two values for the given claim type support each other."""
    val_a = val_a.strip().upper()
    val_b = val_b.strip().upper()
    return val_a == val_b
