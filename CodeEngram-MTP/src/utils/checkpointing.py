"""Checkpoint utilities for CodeEngram-MTP.

This module keeps saving/loading logic in one place so training and inference do
not duplicate fragile checkpoint code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.model.codeengram_model import CodeEngramConfig, CodeEngramForCausalLM


def save_training_checkpoint(
    path: str | Path,
    model: CodeEngramForCausalLM,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model weights, config, and small metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": model.config.__dict__,
        "step": step,
    }
    if extra:
        payload.update(extra)

    torch.save(payload, path)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[CodeEngramForCausalLM, dict[str, Any]]:
    """Load a CodeEngram model and checkpoint metadata."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = CodeEngramConfig(**checkpoint["config"])
    model = CodeEngramForCausalLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint
