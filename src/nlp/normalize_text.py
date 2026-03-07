"""Rule-based text normalization for ASR output."""

from __future__ import annotations

import re
from typing import Any

from src.utils.config import DEFAULT_REMOVE_FILLERS, FILLER_WORDS

TOKEN_PATTERN = re.compile(r"[a-zA-Z']+")


def _collapse_whitespace(text: str) -> str:
    """Replace repeated whitespace with single spaces."""
    return " ".join(text.split())


def _tokenize_basic(text: str) -> list[str]:
    """Tokenize text with a simple regex baseline."""
    return TOKEN_PATTERN.findall(text)


def normalize_text(text: str, remove_fillers: bool = DEFAULT_REMOVE_FILLERS) -> dict[str, Any]:
    """Normalize ASR text for downstream gloss conversion.

    Steps:
    1) whitespace cleanup
    2) lowercase normalization
    3) optional filler-word removal
    4) simple tokenization
    """
    cleaned = _collapse_whitespace((text or "").strip().lower())

    tokens = _tokenize_basic(cleaned)

    if remove_fillers:
        tokens = [tok for tok in tokens if tok not in FILLER_WORDS]

    clean_text = " ".join(tokens)

    return {
        "clean_text": clean_text,
        "tokens": tokens,
    }
