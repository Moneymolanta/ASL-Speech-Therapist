"""Collate utilities for batching variable-length translation sequences."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


class TranslationCollator:
    """Pads source and target sequences to the longest item in each batch."""

    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:
        src = [item["src_ids"] for item in batch]
        tgt_input = [item["tgt_input_ids"] for item in batch]
        tgt_output = [item["tgt_output_ids"] for item in batch]

        src_padded = pad_sequence(src, batch_first=True, padding_value=self.pad_idx)
        tgt_input_padded = pad_sequence(tgt_input, batch_first=True, padding_value=self.pad_idx)
        tgt_output_padded = pad_sequence(tgt_output, batch_first=True, padding_value=self.pad_idx)

        return {
            "src_ids": src_padded,
            "tgt_input_ids": tgt_input_padded,
            "tgt_output_ids": tgt_output_padded,
            "english": [item["english"] for item in batch],
            "gloss": [item["gloss"] for item in batch],
        }
