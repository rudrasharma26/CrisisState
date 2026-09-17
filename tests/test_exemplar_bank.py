import json
import re
from collections import Counter
from pathlib import Path

from crisisstate.domain.models import ClaimExemplar
from crisisstate.domain.vocabulary import ALLOWED_CLAIM_VALUES


ROOT = Path(__file__).resolve().parents[1]
BANK_PATH = ROOT / "data" / "exemplars" / "claim_exemplars.json"
EVAL_DIR = ROOT / "data" / "evaluation"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_bank():
    with BANK_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_eval_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from collect_eval_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from collect_eval_strings(item)


def test_exemplar_bank_schema_and_values():
    data = load_bank()

    assert len(data) == 200
    assert len({item["id"] for item in data}) == 200

    for item in data:
        exemplar = ClaimExemplar.model_validate(item)

        allowed = ALLOWED_CLAIM_VALUES[exemplar.claim_type]
        assert exemplar.value in allowed
        assert exemplar.source.value == "HAND_AUTHORED"
        assert exemplar.version == "v1"
        assert exemplar.text.strip()


def test_exemplar_coverage():
    data = load_bank()

    counts = Counter(
        (item["claim_type"], item["value"])
        for item in data
    )

    expected_pairs = {
        (claim_type.value, value)
        for claim_type, values in ALLOWED_CLAIM_VALUES.items()
        for value in values
    }

    assert set(counts) == expected_pairs
    assert all(count == 10 for count in counts.values())


def test_exemplar_texts_are_unique():
    data = load_bank()

    normalized = [normalize(item["text"]) for item in data]

    assert len(normalized) == len(set(normalized))


def test_no_exact_or_normalized_evaluation_leakage():
    eval_strings = []

    for path in EVAL_DIR.glob("*.json"):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        eval_strings.extend(collect_eval_strings(payload))

    raw_eval = set(eval_strings)
    normalized_eval = {normalize(text) for text in eval_strings}

    for item in load_bank():
        assert item["text"] not in raw_eval
        assert normalize(item["text"]) not in normalized_eval