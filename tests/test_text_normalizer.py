from crisisstate.semantic.text_normalizer import TextNormalizer


def test_normalize_lowercases_text():
    normalizer = TextNormalizer()

    assert normalizer.normalize("ROAD IS BLOCKED") == "road is blocked"


def test_normalize_strips_outer_whitespace():
    normalizer = TextNormalizer()

    assert normalizer.normalize("   road is blocked   ") == "road is blocked"


def test_normalize_collapses_whitespace():
    normalizer = TextNormalizer()

    assert normalizer.normalize("road   is\tblocked\nnow") == "road is blocked now"


def test_normalize_empty_string():
    normalizer = TextNormalizer()

    assert normalizer.normalize("") == ""


def test_normalize_rejects_non_string():
    normalizer = TextNormalizer()

    try:
        normalizer.normalize(None)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError for non-string input")