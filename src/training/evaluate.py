"""Evaluation utilities for English->ASL model training."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.training.losses import seq2seq_cross_entropy


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    pad_idx: int,
) -> dict[str, float]:
    """Run validation loop and compute loss/token accuracy."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            src_ids = batch["src_ids"].to(device)
            tgt_input_ids = batch["tgt_input_ids"].to(device)
            tgt_output_ids = batch["tgt_output_ids"].to(device)

            logits = model(src_ids=src_ids, tgt_input_ids=tgt_input_ids)
            loss = seq2seq_cross_entropy(logits, tgt_output_ids, pad_idx=pad_idx)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            non_pad_mask = tgt_output_ids.ne(pad_idx)
            total_tokens += int(non_pad_mask.sum().item())
            correct_tokens += int(((predictions == tgt_output_ids) & non_pad_mask).sum().item())

    avg_loss = total_loss / max(len(dataloader), 1)
    token_accuracy = correct_tokens / max(total_tokens, 1)

    return {
        "val_loss": avg_loss,
        "val_token_accuracy": token_accuracy,
    }
