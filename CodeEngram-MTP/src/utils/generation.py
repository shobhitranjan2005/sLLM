"""Text generation helpers for the base model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_token_ids(
    model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Autoregressively generate token IDs from a model."""

    model.eval()
    generated = input_ids

    for _ in range(max_new_tokens):
        context = generated[:, -model.config.max_seq_len :]
        logits = model(context)["logits"][:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                cutoff = values[:, [-1]]
                logits = torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break

    return generated


@torch.no_grad()
def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str | torch.device = "cpu",
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> str:
    """Encode a prompt, generate token IDs, and decode to readable text."""

    encoded = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)

    output_ids = generate_token_ids(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
