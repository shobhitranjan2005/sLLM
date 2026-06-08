"""Ablation utilities for comparing CodeEngram-MTP variants.

The goal is to compare:
A. Base Transformer
B. Base + MTP
C. Base + Engram
D. Base + MTP + Engram

This early version runs tiny CPU-safe forward-pass metrics. After real training,
we will extend it to load separate checkpoints and run generated-code metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

import torch

from src.model.codeengram_model import CodeEngramConfig, CodeEngramForCausalLM


@dataclass
class AblationRow:
    variant: str
    params: int
    loss: float
    lm_loss: float
    mtp_loss: float | None
    gate_mean: float | None


def make_variant_config(base: CodeEngramConfig, variant: str) -> CodeEngramConfig:
    """Create a config copy for one ablation variant."""
    cfg = CodeEngramConfig(**asdict(base))
    cfg.mtp_enabled = "MTP" in variant
    cfg.engram_enabled = "Engram" in variant
    if cfg.engram_enabled and cfg.engram_memory_dim != cfg.hidden_size:
        cfg.engram_memory_dim = cfg.hidden_size
    return cfg


def run_forward_ablation(
    base_config: CodeEngramConfig,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    variants: Iterable[str] | None = None,
) -> List[AblationRow]:
    """Run one batch through each architecture variant and report losses."""
    if variants is None:
        variants = [
            "Base",
            "Base + MTP",
            "Base + Engram",
            "Base + MTP + Engram",
        ]

    rows: List[AblationRow] = []

    for variant in variants:
        torch.manual_seed(42)
        cfg = make_variant_config(base_config, variant)
        model = CodeEngramForCausalLM(cfg)
        model.eval()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)

        loss = outputs["loss"]
        lm_loss = outputs["lm_loss"]
        mtp_loss = outputs.get("mtp_loss")
        gate_values = outputs.get("engram_gate_values")

        if loss is None or lm_loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Invalid loss for variant: {variant}")

        gate_mean = None
        if gate_values is not None:
            gate_mean = float(gate_values.mean().item())

        rows.append(
            AblationRow(
                variant=variant,
                params=model.count_parameters(),
                loss=float(loss.item()),
                lm_loss=float(lm_loss.item()),
                mtp_loss=float(mtp_loss.item()) if mtp_loss is not None else None,
                gate_mean=gate_mean,
            )
        )

    return rows


def format_ablation_table(rows: Iterable[AblationRow]) -> str:
    """Return a simple markdown table for README/logging."""
    lines = [
        "| Variant | Params | Loss | LM Loss | MTP Loss | Gate Mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        mtp = "-" if row.mtp_loss is None else f"{row.mtp_loss:.4f}"
        gate = "-" if row.gate_mean is None else f"{row.gate_mean:.4f}"
        lines.append(
            f"| {row.variant} | {row.params:,} | {row.loss:.4f} | {row.lm_loss:.4f} | {mtp} | {gate} |"
        )
    return "\n".join(lines)
