"""Tokenizer and vocabulary utilities for English->ASL sequence modeling."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


@dataclass
class SimpleWhitespaceTokenizer:
    """Minimal tokenizer for educational seq2seq baselines."""

    lowercase: bool = True

    def tokenize(self, text: str) -> list[str]:
        processed = (text or "").strip()
        if self.lowercase:
            processed = processed.lower()
        return [token for token in processed.split() if token]


class Vocab:
    """Integer vocabulary with save/load helpers."""

    def __init__(self, stoi: dict[str, int]):
        self.stoi = dict(stoi)
        self.itos = {index: token for token, index in self.stoi.items()}

    @classmethod
    def build(cls, token_sequences: list[list[str]], min_freq: int = 1) -> "Vocab":
        counter: Counter[str] = Counter()
        for sequence in token_sequences:
            counter.update(sequence)

        stoi: dict[str, int] = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        next_idx = len(stoi)

        for token, count in sorted(counter.items()):
            if count < min_freq:
                continue
            if token in stoi:
                continue
            stoi[token] = next_idx
            next_idx += 1

        return cls(stoi=stoi)

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.stoi[EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.stoi)

    def token_to_id(self, token: str) -> int:
        return self.stoi.get(token, self.unk_idx)

    def id_to_token(self, index: int) -> str:
        return self.itos.get(index, UNK_TOKEN)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id(token) for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token(i) for i in ids]

    def to_dict(self) -> dict[str, object]:
        return {"stoi": self.stoi}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Vocab":
        stoi = payload["stoi"]
        assert isinstance(stoi, dict)
        return cls(stoi=stoi)

    def save_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "Vocab":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)


def build_vocabs(
    records: list[dict[str, str]],
    src_tokenizer: SimpleWhitespaceTokenizer,
    tgt_tokenizer: SimpleWhitespaceTokenizer,
    min_freq: int = 1,
) -> tuple[Vocab, Vocab]:
    """Build source (English) and target (ASL gloss) vocabularies."""
    src_sequences = [src_tokenizer.tokenize(record["english"]) for record in records]
    tgt_sequences = [tgt_tokenizer.tokenize(record["gloss"]) for record in records]

    src_vocab = Vocab.build(src_sequences, min_freq=min_freq)
    tgt_vocab = Vocab.build(tgt_sequences, min_freq=min_freq)
    return src_vocab, tgt_vocab
