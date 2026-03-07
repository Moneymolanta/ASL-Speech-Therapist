"""Dataset cleanup utility for paired English/ASL gloss data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_paired_records
from src.nlp.normalize_text import normalize_text


def preprocess_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize English text and standardize gloss capitalization."""
    cleaned_records: list[dict[str, str]] = []
    for record in records:
        norm = normalize_text(record["english"], remove_fillers=False)
        clean_english = norm["clean_text"]
        clean_gloss = " ".join(record["gloss"].strip().upper().split())

        if clean_english and clean_gloss:
            cleaned_records.append({"english": clean_english, "gloss": clean_gloss})

    return cleaned_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess paired English/ASL gloss dataset.")
    parser.add_argument("--input", required=True, help="Input dataset (.json, .jsonl, .csv)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    records = load_paired_records(args.input)
    cleaned = preprocess_records(records)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(cleaned, handle, indent=2, ensure_ascii=False)

    print(f"Saved {len(cleaned)} preprocessed records to {output_path}")


if __name__ == "__main__":
    main()
