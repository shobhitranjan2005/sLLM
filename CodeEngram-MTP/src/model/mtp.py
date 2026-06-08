"""Multi-Token Prediction (MTP) modules.

MTP adds extra prediction heads on top of the Transformer hidden states.
The normal LM head predicts token t+1. An MTP head can predict token t+2,
which gives the model a stronger training signal for future code structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MTPHead(nn.Module):
    """A lightweight future-token prediction head.

    The input is the final Transformer hidden state with shape:
        [batch, seq_len, hidden_size]

    The output is vocabulary logits with shape:
        [batch, seq_len, vocab_size]

    In Phase 3A we keep this intentionally simple: one linear projection.
    Later we can make this deeper if ablation shows it is useful.
    """

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, seq_len, hidden_size].")
        return self.proj(hidden_states)


class MTPModule(nn.Module):
    """Container for one or more MTP heads.

    For our prototype we start with only one head:
        head 0 predicts t+2

    Optional future extension:
        head 1 predicts t+3
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_heads: int = 1) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be at least 1 when MTP is enabled.")

        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [MTPHead(hidden_size=hidden_size, vocab_size=vocab_size) for _ in range(num_heads)]
        )

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        return [head(hidden_states) for head in self.heads]
