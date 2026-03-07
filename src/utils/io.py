"""Small file I/O helpers shared across scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: str | Path) -> None:
    """Create parent directory for a file path if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(path: str | Path, payload: Any) -> None:
    """Save a JSON-serializable payload with UTF-8 encoding."""
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Load JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
