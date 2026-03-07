"""Dataset utilities for English->ASL gloss paired data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.models.tokenizer_utils import SimpleWhitespaceTokenizer, Vocab


def load_paired_records(path: str) -> list[dict[str, str]]:
    """Load paired `english`/`gloss` records from JSON, JSONL, or CSV."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = file_path.suffix.lower()
    records: list[dict[str, str]] = []

    if suffix == ".json":
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("data", [])
        if not isinstance(payload, list):
            raise ValueError("JSON dataset must be a list of {english, gloss} records.")
        records = [
            {"english": str(item["english"]), "gloss": str(item["gloss"])}
            for item in payload
            if "english" in item and "gloss" in item
        ]

    elif suffix == ".jsonl":
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "english" in item and "gloss" in item:
                    records.append({"english": str(item["english"]), "gloss": str(item["gloss"])})

    elif suffix == ".csv":
        with file_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if "english" in row and "gloss" in row:
                    records.append({"english": str(row["english"]), "gloss": str(row["gloss"])})
    else:
        raise ValueError("Unsupported dataset format. Use .json, .jsonl, or .csv")

    if not records:
        raise ValueError("No usable paired records found in dataset.")

    return records


class EnglishASLDataset(Dataset):
    """PyTorch dataset for seq2seq training with teacher forcing targets."""

    def __init__(
        self,
        records: list[dict[str, str]],
        src_tokenizer: SimpleWhitespaceTokenizer,
        tgt_tokenizer: SimpleWhitespaceTokenizer,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_src_len: int = 64,
        max_tgt_len: int = 64,
    ):
        self.records = records
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        english = record["english"]
        gloss = record["gloss"]

        src_tokens = self.src_tokenizer.tokenize(english)[: self.max_src_len - 2]
        tgt_tokens = self.tgt_tokenizer.tokenize(gloss)[: self.max_tgt_len - 2]

        src_ids = [self.src_vocab.bos_idx] + self.src_vocab.encode(src_tokens) + [self.src_vocab.eos_idx]
        tgt_ids = [self.tgt_vocab.bos_idx] + self.tgt_vocab.encode(tgt_tokens) + [self.tgt_vocab.eos_idx]

        tgt_input_ids = tgt_ids[:-1]
        tgt_output_ids = tgt_ids[1:]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_input_ids": torch.tensor(tgt_input_ids, dtype=torch.long),
            "tgt_output_ids": torch.tensor(tgt_output_ids, dtype=torch.long),
            "english": english,
            "gloss": gloss,
        }
