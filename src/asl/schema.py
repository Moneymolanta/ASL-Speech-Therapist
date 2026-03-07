"""Schema definitions for ASL prediction outputs.

The project now treats English->ASL as a learned sequence prediction task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ASLPrediction:
    """Structured model output for ASL gloss prediction."""

    clean_text: str
    predicted_gloss_tokens: list[str]
    predicted_gloss_text: str
    model_name: str
    used_fallback: bool = False
    empty_after_postprocess: bool = False
    debug_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a JSON-serializable dictionary."""
        payload = asdict(self)
        debug_info = payload.pop("debug_info", None)
        if debug_info:
            payload.update(debug_info)
        return payload
