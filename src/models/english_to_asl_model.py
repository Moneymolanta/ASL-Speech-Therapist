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
        beam_width: int = 1,
    ) -> torch.Tensor:
        """Decode output tokens using greedy or beam search.

        Args:
            src_ids: [batch, src_len]
            bos_idx: Beginning-of-sequence token index.
            eos_idx: End-of-sequence token index.
            max_len: Maximum output length.
            beam_width: Beam size. 1 = greedy, >1 = beam search.

        Returns:
            generated token ids: [batch, <=max_len]
        """
        self.eval()
        if beam_width <= 1:
            return self._greedy_decode(src_ids, bos_idx, eos_idx, max_len)
        return self._beam_search(src_ids, bos_idx, eos_idx, max_len, beam_width)

    def _greedy_decode(
        self,
        src_ids: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int,
    ) -> torch.Tensor:
        """Original greedy decoding."""
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

    def _beam_search(
        self,
        src_ids: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int,
        beam_width: int,
    ) -> torch.Tensor:
        """Beam search decoding for improved translation quality.

        Maintains `beam_width` candidate sequences at each step, selecting
        the top-scoring beams based on cumulative log-probability.
        Only supports batch_size=1 for simplicity.
        """
        device = src_ids.device

        # Each beam: (log_prob, token_ids_list)
        beams: list[tuple[float, list[int]]] = [(0.0, [bos_idx])]
        completed: list[tuple[float, list[int]]] = []

        for _ in range(max_len - 1):
            if not beams:
                break

            all_candidates: list[tuple[float, list[int]]] = []

            for score, seq in beams:
                if seq[-1] == eos_idx:
                    completed.append((score, seq))
                    continue

                tgt_tensor = torch.tensor([seq], dtype=torch.long, device=device)
                logits = self.forward(src_ids=src_ids, tgt_input_ids=tgt_tensor)
                log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)

                top_log_probs, top_indices = torch.topk(log_probs, beam_width)

                for log_p, idx in zip(top_log_probs.tolist(), top_indices.tolist()):
                    new_seq = seq + [idx]
                    all_candidates.append((score + log_p, new_seq))

            if not all_candidates:
                break

            # Keep top beam_width candidates
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:beam_width]

        # Add remaining beams to completed
        completed.extend(beams)

        if not completed:
            return torch.tensor([[bos_idx]], dtype=torch.long, device=device)

        # Length-normalized scoring to avoid bias toward short sequences
        best_score = float("-inf")
        best_seq = [bos_idx]
        for score, seq in completed:
            length_penalty = ((5.0 + len(seq)) / 6.0) ** 0.6
            normalized = score / length_penalty
            if normalized > best_score:
                best_score = normalized
                best_seq = seq

        return torch.tensor([best_seq], dtype=torch.long, device=device)
