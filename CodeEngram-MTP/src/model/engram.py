"""Engram memory modules for CodeEngram-MTP.

Phase 4A added the standalone hash-based memory table.
Phase 4B adds the injection gate that lets memory vectors enter the
Transformer residual stream in a controlled way.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EngramConfig:
    """Configuration for the hash-based Engram memory."""

    memory_slots: int
    memory_dim: int
    ngram_size: int = 3
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.memory_slots <= 0:
            raise ValueError("memory_slots must be positive.")
        if self.memory_dim <= 0:
            raise ValueError("memory_dim must be positive.")
        if self.ngram_size <= 0:
            raise ValueError("ngram_size must be positive.")


class EngramMemory(nn.Module):
    """Hash-based n-gram memory table.

    The module converts token n-grams into stable hash IDs, then retrieves
    trainable memory vectors. For a sequence of length S, it returns one memory
    vector per token position.
    """

    def __init__(self, config: EngramConfig) -> None:
        super().__init__()
        self.config = config
        self.memory = nn.Embedding(config.memory_slots, config.memory_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.memory.weight, mean=0.0, std=0.02)

    def hash_ngrams(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return memory slot IDs with shape [batch, seq_len].

        Each position receives the hash of the n-gram ending at that position.
        Early positions are left-padded with pad_token_id.
        """

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len].")

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        n = self.config.ngram_size

        padded = torch.full(
            (batch_size, n - 1 + seq_len),
            fill_value=self.config.pad_token_id,
            dtype=input_ids.dtype,
            device=device,
        )
        padded[:, n - 1 :] = input_ids

        # Deterministic polynomial rolling hash. Constants are fixed so the same
        # token n-gram maps to the same memory slot across runs.
        hash_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        base = 257
        modulus = 2_147_483_647

        for offset in range(n):
            token_slice = padded[:, offset : offset + seq_len].long()
            hash_ids = (hash_ids * base + token_slice + 1) % modulus

        return hash_ids % self.config.memory_slots

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return memory vectors and memory slot IDs.

        Returns:
            memory_vectors: Tensor with shape [batch, seq_len, memory_dim]
            memory_ids: Tensor with shape [batch, seq_len]
        """

        memory_ids = self.hash_ngrams(input_ids)
        memory_vectors = self.memory(memory_ids)
        return memory_vectors, memory_ids

    def count_parameters(self) -> int:
        return self.memory.weight.numel()


class EngramInjectionGate(nn.Module):
    """Inject Engram memory into the Transformer hidden state.

    Formula:
        hidden = hidden + sigmoid(W_g hidden) * W_m(memory_vector)

    The scalar gate is learned per token position. Gate values close to 0 mean
    the model ignores memory; values close to 1 mean memory strongly influences
    the residual stream.
    """

    def __init__(self, hidden_size: int, memory_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.memory_proj = nn.Linear(memory_dim, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.memory_proj.weight, mean=0.0, std=0.02)
        # Start conservative: sigmoid(-2) ≈ 0.12, so Engram helps gently at first.
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def forward(
        self,
        hidden: torch.Tensor,
        memory_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, seq_len, hidden_size].")
        if memory_vectors.ndim != 3:
            raise ValueError("memory_vectors must have shape [batch, seq_len, memory_dim].")
        if hidden.shape[:2] != memory_vectors.shape[:2]:
            raise ValueError("hidden and memory_vectors must share batch/sequence dimensions.")

        projected_memory = self.memory_proj(memory_vectors)
        gate = torch.sigmoid(self.gate_proj(hidden))
        injected = hidden + self.dropout(gate * projected_memory)
        return injected, gate
