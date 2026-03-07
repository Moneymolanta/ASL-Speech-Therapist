"""Debug-only fallback rules for English->gloss conversion.

This module is intentionally not the primary translation path.
Use it for sanity checks and comparison against the learned model.
"""

from __future__ import annotations

from src.utils.config import GLOSS_STOPWORDS


def fallback_text_to_gloss(tokens: list[str]) -> dict[str, object]:
    """Simple rule-based fallback used only for debugging/baselines."""
    filtered = [token for token in tokens if token not in GLOSS_STOPWORDS]
    gloss_tokens = [token.upper() for token in filtered]
    return {
        "predicted_gloss_tokens": gloss_tokens,
        "predicted_gloss_text": " ".join(gloss_tokens),
        "used_fallback": True,
    }
