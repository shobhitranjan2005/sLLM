"""Main CodeEngram model class.

This class supports the staged architecture:
base decoder Transformer, optional MTP heads, and optional Engram memory injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.model.transformer import RMSNorm, TransformerBlock
from src.model.mtp import MTPModule
from src.model.engram import EngramConfig, EngramInjectionGate, EngramMemory
from src.utils.losses import mtp_cross_entropy, next_token_cross_entropy


@dataclass
class CodeEngramConfig:
    vocab_size: int = 32_000
    max_seq_len: int = 1024
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 2048
    activation: str = "swiglu"
    dropout: float = 0.1
    tie_word_embeddings: bool = True

    mtp_enabled: bool = False
    mtp_num_heads: int = 1
    mtp_t2_loss_weight: float = 0.3
    mtp_t3_loss_weight: float = 0.1

    engram_enabled: bool = False
    engram_memory_slots: int = 50_000
    engram_memory_dim: int = 768
    engram_ngram_size: int = 3
    engram_injection_layer: int = 4
    engram_gate_regularization_weight: float = 0.01

    @classmethod
    def from_yaml_dict(cls, config: dict[str, Any]) -> "CodeEngramConfig":
        model_cfg = config["model"]
        mtp_cfg = config.get("mtp", {})
        engram_cfg = config.get("engram", {})
        return cls(
            vocab_size=int(model_cfg["vocab_size"]),
            max_seq_len=int(model_cfg["max_seq_len"]),
            hidden_size=int(model_cfg["hidden_size"]),
            num_layers=int(model_cfg["num_layers"]),
            num_attention_heads=int(model_cfg["num_attention_heads"]),
            intermediate_size=int(model_cfg["intermediate_size"]),
            activation=str(model_cfg.get("activation", "swiglu")),
            dropout=float(model_cfg.get("dropout", 0.0)),
            tie_word_embeddings=bool(model_cfg.get("tie_word_embeddings", True)),
            mtp_enabled=bool(mtp_cfg.get("enabled", False)),
            mtp_num_heads=int(mtp_cfg.get("num_heads", 1)),
            mtp_t2_loss_weight=float(mtp_cfg.get("t2_loss_weight", 0.3)),
            mtp_t3_loss_weight=float(mtp_cfg.get("t3_loss_weight", 0.1)),
            engram_enabled=bool(engram_cfg.get("enabled", False)),
            engram_memory_slots=int(engram_cfg.get("memory_slots", 50_000)),
            engram_memory_dim=int(engram_cfg.get("memory_dim", model_cfg["hidden_size"])),
            engram_ngram_size=int(engram_cfg.get("ngram_size", 3)),
            engram_injection_layer=int(engram_cfg.get("injection_layer", 4)),
            engram_gate_regularization_weight=float(
                engram_cfg.get("gate_regularization_weight", 0.01)
            ),
        )


class CodeEngramForCausalLM(nn.Module):
    """Base decoder-only Transformer language model."""

    def __init__(self, config: CodeEngramConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.mtp: MTPModule | None = None
        if config.mtp_enabled:
            self.mtp = MTPModule(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                num_heads=config.mtp_num_heads,
            )

        self.engram: EngramMemory | None = None
        self.engram_gate: EngramInjectionGate | None = None
        if config.engram_enabled:
            self.engram = EngramMemory(
                EngramConfig(
                    memory_slots=config.engram_memory_slots,
                    memory_dim=config.engram_memory_dim,
                    ngram_size=config.engram_ngram_size,
                    pad_token_id=0,
                )
            )
            self.engram_gate = EngramInjectionGate(
                hidden_size=config.hidden_size,
                memory_dim=config.engram_memory_dim,
                dropout=config.dropout,
            )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # self.apply initializes all Linear layers. Re-apply the Engram gate
        # initialization so the memory gate starts conservative.
        if self.engram_gate is not None:
            self.engram_gate.reset_parameters()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len].")

        _, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}."
            )

        hidden = self.token_embedding(input_ids)
        hidden = self.dropout(hidden)

        engram_memory_ids = None
        engram_gate_values = None
        engram_reg_loss = None
        engram_memory_vectors = None

        if self.engram is not None:
            engram_memory_vectors, engram_memory_ids = self.engram(input_ids)

        # injection_layer is treated as 1-based: 4 means after block 4.
        injection_index = max(0, min(self.config.engram_injection_layer, len(self.blocks)))

        for block_index, block in enumerate(self.blocks, start=1):
            hidden = block(hidden)
            if (
                self.engram_gate is not None
                and engram_memory_vectors is not None
                and block_index == injection_index
            ):
                hidden, engram_gate_values = self.engram_gate(hidden, engram_memory_vectors)

        # If injection_layer was 0, inject before final norm after all blocks are skipped above.
        if (
            self.engram_gate is not None
            and engram_memory_vectors is not None
            and injection_index == 0
        ):
            hidden, engram_gate_values = self.engram_gate(hidden, engram_memory_vectors)

        if engram_gate_values is not None:
            engram_reg_loss = engram_gate_values.mean()

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)

        lm_loss = None
        mtp_loss = None
        loss = None

        mtp_logits: list[torch.Tensor] | None = None
        if self.mtp is not None:
            mtp_logits = self.mtp(hidden)

        if labels is not None:
            lm_loss = next_token_cross_entropy(logits, labels)
            loss = lm_loss

            if mtp_logits is not None:
                mtp_losses: list[torch.Tensor] = []
                for head_index, head_logits in enumerate(mtp_logits):
                    future_offset = head_index + 2
                    if labels.size(1) > future_offset:
                        mtp_losses.append(mtp_cross_entropy(head_logits, labels, future_offset))

                if mtp_losses:
                    mtp_loss = sum(mtp_losses) / len(mtp_losses)
                    loss = loss + self.config.mtp_t2_loss_weight * mtp_loss

            if engram_reg_loss is not None:
                loss = loss + self.config.engram_gate_regularization_weight * engram_reg_loss

        return {
            "logits": logits,
            "loss": loss,
            "lm_loss": lm_loss,
            "mtp_logits": mtp_logits,
            "mtp_loss": mtp_loss,
            "engram_memory_ids": engram_memory_ids,
            "engram_gate_values": engram_gate_values,
            "engram_reg_loss": engram_reg_loss,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
