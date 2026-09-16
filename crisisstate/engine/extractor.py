"""Deterministic claim and entity extraction for urban flooding reports."""

import re
from typing import List, Optional, Tuple
from crisisstate.domain.models import Claim, Report
from crisisstate.domain.vocabulary import ClaimType, ExtractionMethod


# Known landmark entities or generic patterns for urban flooding
KNOWN_ENTITIES = [
    "Main Road",
    "Sector 4 Underpass",
    "Riverbank Colony",
    "Market Square",
    "Bridge Road",
    "5th Avenue",
    "Central Bus Stand",
]

ENTITY_REGEX = re.compile(
    r"\b(Main Road|Sector 4 Underpass|Riverbank Colony|Market Square|Bridge Road|5th Avenue|Central Bus Stand|"
    r"(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z0-9][a-zA-Z0-9]+)?\s+(?:Road|Street|Avenue|Underpass|Bridge|Colony|Sector|Lane|Highway)))\b",
    re.IGNORECASE,
)

# Pattern definitions per claim type: (ClaimType, value, regex_pattern, confidence)
EXTRACTION_RULES: List[Tuple[ClaimType, str, re.Pattern, float]] = [
    # ROAD_ACCESS
    (
        ClaimType.ROAD_ACCESS,
        "IMPASSABLE",
        re.compile(
            r"\b(impassable|cannot cross|cannot pass|blocked|closed to traffic|unable to pass|cut off|submerged road|no access|cannot drive through)\b",
            re.IGNORECASE,
        ),
        0.95,
    ),
    (
        ClaimType.ROAD_ACCESS,
        "PASSABLE",
        re.compile(
            r"\b(can still pass|passable|clear to drive|vehicles can pass|open for traffic|open to traffic|accessible|can pass through|can pass)\b",
            re.IGNORECASE,
        ),
        0.95,
    ),
    # WATER_LEVEL
    (
        ClaimType.WATER_LEVEL,
        "CRITICAL",
        re.compile(
            r"\b(critical water level|chest deep|waist-deep|overflowing banks|danger level|water reached roof|submerged completely)\b",
            re.IGNORECASE,
        ),
        0.95,
    ),
    (
        ClaimType.WATER_LEVEL,
        "HIGH",
        re.compile(
            r"\b(water is rising rapidly|rising fast|knee deep|deep water|severe flooding|high water|water rising fast)\b",
            re.IGNORECASE,
        ),
        0.90,
    ),
    (
        ClaimType.WATER_LEVEL,
        "MODERATE",
        re.compile(
            r"\b(water is building up|difficult to cross|water on road|water accumulation|moderate water|ankle deep|rising)\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        ClaimType.WATER_LEVEL,
        "LOW",
        re.compile(
            r"\b(low water|puddles|minor water|water level is low|shallow water|water receded|water receding)\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    # PEOPLE_TRAPPED
    (
        ClaimType.PEOPLE_TRAPPED,
        "YES",
        re.compile(
            r"\b(people are trapped|residents stranded|people stranded|stranded|people stuck|need rescue|trapped inside|people trapped)\b",
            re.IGNORECASE,
        ),
        0.95,
    ),
    (
        ClaimType.PEOPLE_TRAPPED,
        "NO",
        re.compile(
            r"\b(no one trapped|residents safe|all evacuated|no people trapped|everyone safe|everyone rescued)\b",
            re.IGNORECASE,
        ),
        0.90,
    ),
    # BUILDING_DAMAGE
    (
        ClaimType.BUILDING_DAMAGE,
        "COLLAPSED",
        re.compile(
            r"\b(building collapsed|wall collapsed|structure collapsed|house collapsed)\b",
            re.IGNORECASE,
        ),
        0.95,
    ),
    (
        ClaimType.BUILDING_DAMAGE,
        "DAMAGED",
        re.compile(
            r"\b(building damaged|structural damage|damage to homes|homes damaged|roof damaged|cracked walls|damage nearby)\b",
            re.IGNORECASE,
        ),
        0.90,
    ),
    (
        ClaimType.BUILDING_DAMAGE,
        "NONE",
        re.compile(
            r"\b(no building damage|structures intact|no structural damage|no damage to buildings)\b",
            re.IGNORECASE,
        ),
        0.90,
    ),
    # TRAFFIC_STATUS
    (
        ClaimType.TRAFFIC_STATUS,
        "STOPPED",
        re.compile(
            r"\b(traffic is completely stopped|traffic stopped|gridlock|standstill|completely stopped|traffic stalled)\b",
            re.IGNORECASE,
        ),
        0.95,
    ),
    (
        ClaimType.TRAFFIC_STATUS,
        "SLOW",
        re.compile(
            r"\b(traffic is slow|heavy traffic|crawling traffic|moving slowly|slow traffic)\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        ClaimType.TRAFFIC_STATUS,
        "MOVING",
        re.compile(
            r"\b(traffic is moving|traffic flowing|traffic normal|moving freely|traffic resumed|moving)\b",
            re.IGNORECASE,
        ),
        0.90,
    ),
    # FLOODED
    (
        ClaimType.FLOODED,
        "NO",
        re.compile(
            r"\b(not flooded|no flood|water cleared|drained|road is dry|cleared of water|flood receded)\b",
            re.IGNORECASE,
        ),
        0.90,
    ),
    (
        ClaimType.FLOODED,
        "YES",
        re.compile(
            r"\b(is flooded|flooding reported|flooded|under water|waterlogging|inundated)\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    # EVACUATION
    (
        ClaimType.EVACUATION,
        "ORDERED",
        re.compile(
            r"\b(evacuation ordered|evacuate immediately|evacuation announced|order to evacuate)\b",
            re.IGNORECASE,
        ),
        0.95,
    ),
    (
        ClaimType.EVACUATION,
        "COMPLETED",
        re.compile(
            r"\b(evacuation completed|area evacuated|everyone moved out)\b",
            re.IGNORECASE,
        ),
        0.95,
    ),
    (
        ClaimType.EVACUATION,
        "NOT_ORDERED",
        re.compile(
            r"\b(no evacuation|evacuation not required|not ordered to evacuate)\b",
            re.IGNORECASE,
        ),
        0.90,
    ),
]


def extract_entity(report: Report) -> str:
    """Extract primary subject entity from report text or metadata."""
    # Check metadata first
    if "location" in report.metadata and report.metadata["location"]:
        return str(report.metadata["location"]).strip()
    if "entity" in report.metadata and report.metadata["entity"]:
        return str(report.metadata["entity"]).strip()

    # Match in text against known entities first (case-insensitive)
    for entity in KNOWN_ENTITIES:
        pattern = rf"\b{re.escape(entity)}\b"
        if re.search(pattern, report.text, re.IGNORECASE):
            return entity

    # Match general regex pattern
    match = ENTITY_REGEX.search(report.text)
    if match:
        return match.group(1).strip()

    return "General Urban Area"


def extract_claims(report: Report) -> List[Claim]:
    """Deterministically extract claims from a report based on keyword/pattern rules."""
    subject = extract_entity(report)
    claims: List[Claim] = []
    seen_types = set()

    # Evaluate rules in order
    for claim_type, value, pattern, conf in EXTRACTION_RULES:
        # Avoid multiple claims for the same claim_type from a single report sentence unless distinct
        if claim_type in seen_types:
            continue
        if pattern.search(report.text):
            claims.append(
                Claim(
                    report_id=report.id,
                    claim_type=claim_type,
                    subject=subject,
                    value=value,
                    timestamp=report.timestamp,
                    confidence=conf,
                    supporting_evidence=[report.id],
                    contradicting_evidence=[],
                    # M1: populate lexical extraction provenance
                    extraction_method=ExtractionMethod.LEXICAL,
                    extraction_confidence=conf,
                )
            )
            seen_types.add(claim_type)

    return claims
