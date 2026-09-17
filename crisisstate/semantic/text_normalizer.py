"""Deterministic text normalization for CrisisState semantic processing."""

from __future__ import annotations

import re


class TextNormalizer:
    """Normalize text deterministically before semantic processing."""

    _WHITESPACE_RE = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        """Return a deterministic normalized representation of text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        return self._WHITESPACE_RE.sub(" ", text.strip().lower())