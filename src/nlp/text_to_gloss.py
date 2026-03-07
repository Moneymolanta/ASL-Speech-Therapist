"""Deprecated legacy wrapper.

The main translation path is now model-based in `src/models/inference.py`.
This wrapper is retained only for compatibility with older scripts.
"""

from __future__ import annotations

from src.asl.fallback_rules import fallback_text_to_gloss


def text_to_gloss(tokens: list[str]) -> dict[str, object]:
    """Compatibility wrapper around debug fallback rules."""
    return fallback_text_to_gloss(tokens)
