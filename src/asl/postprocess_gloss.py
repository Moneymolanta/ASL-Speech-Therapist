"""Postprocessing helpers for model-predicted gloss sequences."""

from __future__ import annotations

SPECIAL_TOKENS = {"<pad>", "<bos>", "<eos>", "<unk>"}


def clean_gloss_tokens(tokens: list[str]) -> list[str]:
    """Remove special tokens and normalize to uppercase gloss tokens."""
    cleaned: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.lower() in SPECIAL_TOKENS:
            continue
        cleaned.append(token.upper())
    return cleaned


def to_gloss_text(tokens: list[str]) -> str:
    """Join cleaned gloss tokens into a readable gloss string."""
    return " ".join(tokens)
