"""Loss functions for English->ASL sequence training."""

from __future__ import annotations

import torch
from torch import nn


def seq2seq_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Cross-entropy over vocabulary, ignoring padding tokens.

    Args:
        logits: [batch, tgt_len, vocab_size]
        targets: [batch, tgt_len]
    """
    vocab_size = logits.size(-1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    return loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
