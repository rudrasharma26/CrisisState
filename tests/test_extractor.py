"""Unit tests for deterministic entity and claim extraction."""

from crisisstate.domain.models import Report
from crisisstate.domain.vocabulary import ClaimType
from crisisstate.engine.extractor import extract_claims, extract_entity


def test_extract_entity_from_text_known_and_pattern():
    rep1 = Report(text="Water is rising rapidly near Main Road.")
    assert extract_entity(rep1) == "Main Road"

    rep2 = Report(text="Sector 4 Underpass has blocked vehicles.")
    assert extract_entity(rep2) == "Sector 4 Underpass"

    rep3 = Report(text="Heavy flooding at 5th Avenue.")
    assert extract_entity(rep3) == "5th Avenue"


def test_extract_entity_from_metadata():
    rep = Report(
        text="Water everywhere downtown!",
        metadata={"location": "Riverbank Colony"},
    )
    assert extract_entity(rep) == "Riverbank Colony"


def test_extract_claims_road_access_passable():
    rep = Report(text="Vehicles can still pass through Main Road safely.")
    claims = extract_claims(rep)
    assert len(claims) >= 1
    c = next(cl for cl in claims if cl.claim_type == ClaimType.ROAD_ACCESS)
    assert c.value == "PASSABLE"
    assert c.subject == "Main Road"


def test_extract_claims_road_access_impassable():
    rep = Report(text="Main Road is impassable due to deep water.")
    claims = extract_claims(rep)
    c = next(cl for cl in claims if cl.claim_type == ClaimType.ROAD_ACCESS)
    assert c.value == "IMPASSABLE"


def test_extract_claims_trapped_and_critical_water():
    rep = Report(
        text="Critical water level near Main Road, people are trapped inside shops."
    )
    claims = extract_claims(rep)
    types = {cl.claim_type: cl.value for cl in claims}
    assert types.get(ClaimType.WATER_LEVEL) == "CRITICAL"
    assert types.get(ClaimType.PEOPLE_TRAPPED) == "YES"


def test_extract_claims_building_damage():
    rep = Report(text="Sector 4 Underpass has building damaged nearby.")
    claims = extract_claims(rep)
    c = next(cl for cl in claims if cl.claim_type == ClaimType.BUILDING_DAMAGE)
    assert c.value == "DAMAGED"
