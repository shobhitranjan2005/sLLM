"""Base Transformer backbone for CodeEngram-MTP.

This file contains only the normal decoder Transformer blocks.
The project-specific modules, MTP and Engram, are added in later phases.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.attention import CausalSelfAttention


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm used in many modern LLMs."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network.

    It uses two input projections: one for values and one for the gate.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(x))


class GELUFeedForward(nn.Module):
    """Classic GELU feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm decoder Transformer block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.0,
        activation: str = "swiglu",
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.ffn_norm = RMSNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads, dropout=dropout)

        if activation.lower() == "swiglu":
            self.ffn = SwiGLUFeedForward(hidden_size, intermediate_size, dropout=dropout)
        elif activation.lower() == "gelu":
            self.ffn = GELUFeedForward(hidden_size, intermediate_size, dropout=dropout)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
