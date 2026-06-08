"""Reusable loss functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def shifted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    shift: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy where each position predicts a future token.

    shift=1: token t predicts token t+1, the normal LM objective.
    shift=2: token t predicts token t+2, the first MTP objective.
    shift=3: token t predicts token t+3, optional later MTP objective.
    """

    if shift < 1:
        raise ValueError("shift must be >= 1.")
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq_len, vocab_size].")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, seq_len].")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must have matching batch/sequence dimensions.")

    if logits.size(1) <= shift:
        raise ValueError("sequence length must be larger than shift.")

    shift_logits = logits[:, :-shift, :].contiguous()
    shift_labels = labels[:, shift:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def next_token_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Normal next-token language-modeling loss: token t predicts token t+1."""

    return shifted_cross_entropy(logits=logits, labels=labels, shift=1)


def mtp_cross_entropy(mtp_logits: torch.Tensor, labels: torch.Tensor, future_offset: int) -> torch.Tensor:
    """Future-token prediction loss for an MTP head.

    future_offset=2 means token t predicts token t+2.
    """

    return shifted_cross_entropy(logits=mtp_logits, labels=labels, shift=future_offset)
