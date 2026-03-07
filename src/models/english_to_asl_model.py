"""Simple Transformer seq2seq model for English->ASL gloss prediction.

This is intentionally educational and lightweight for a student project baseline.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class EnglishToASLTransformer(nn.Module):
    """Transformer encoder-decoder for text-to-gloss sequence generation."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        d_model: int = 64,
        nhead: int = 2,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # batch_first=True avoids shape warnings and keeps tensors [batch, seq, hidden].
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def _make_tgt_mask(self, tgt_len: int, device: torch.device) -> torch.Tensor:
        # Bool mask expected by nn.Transformer: True entries are masked.
        return torch.triu(torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, src_ids: torch.Tensor, tgt_input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for teacher-forcing training.

        Args:
            src_ids: [batch, src_len]
            tgt_input_ids: [batch, tgt_len]

        Returns:
            logits: [batch, tgt_len, tgt_vocab_size]
        """
        src_key_padding_mask = src_ids.eq(self.src_pad_idx)
        tgt_key_padding_mask = tgt_input_ids.eq(self.tgt_pad_idx)
        tgt_mask = self._make_tgt_mask(tgt_input_ids.size(1), src_ids.device)

        src_embed = self.src_embedding(src_ids) * math.sqrt(self.d_model)
        tgt_embed = self.tgt_embedding(tgt_input_ids) * math.sqrt(self.d_model)

        src_embed = self.positional_encoding(src_embed)
        tgt_embed = self.positional_encoding(tgt_embed)

        hidden = self.transformer(
            src=src_embed,
            tgt=tgt_embed,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_projection(hidden)

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 32,
    ) -> torch.Tensor:
        """Greedy decoding for inference.

        Args:
            src_ids: [batch, src_len]

        Returns:
            generated token ids: [batch, <=max_len]
        """
        self.eval()
        batch_size = src_ids.size(0)
        generated = torch.full(
            (batch_size, 1),
            fill_value=bos_idx,
            dtype=torch.long,
            device=src_ids.device,
        )

        for _ in range(max_len - 1):
            logits = self.forward(src_ids=src_ids, tgt_input_ids=generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if torch.all(next_token.squeeze(1).eq(eos_idx)):
                break

        return generated
