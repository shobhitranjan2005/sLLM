"""Engram memory evaluation helpers.

This module checks whether Engram gate activity can be inspected and whether a
small factual-recall benchmark is wired correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

import torch


@dataclass
class MemoryEvalResult:
    total_prompts: int
    gate_mean: float
    gate_max: float
    memory_vectors_seen: bool


def evaluate_engram_gate_activity(model: Any, tokenizer: Any, prompts: Iterable[str], device: str = "cpu") -> MemoryEvalResult:
    """Run prompts through an Engram-enabled model and summarize gate activity."""
    model.eval()
    model.to(device)

    gate_values: List[torch.Tensor] = []
    memory_seen = False
    count = 0

    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids)
            gates = outputs.get("engram_gate_values")
            memory_ids = outputs.get("engram_memory_ids")
            count += 1
            if gates is not None:
                gate_values.append(gates.detach().cpu())
            if memory_ids is not None:
                memory_seen = True

    if gate_values:
        merged = torch.cat([g.reshape(-1) for g in gate_values])
        gate_mean = float(merged.mean().item())
        gate_max = float(merged.max().item())
    else:
        gate_mean = 0.0
        gate_max = 0.0

    return MemoryEvalResult(
        total_prompts=count,
        gate_mean=gate_mean,
        gate_max=gate_max,
        memory_vectors_seen=memory_seen,
    )
