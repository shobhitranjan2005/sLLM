from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.train_mtp import train_one_mtp_run


CHECKPOINT_PATH = ROOT / "checkpoints" / "mtp_tiny_check.pt"
TRAIN_PATH = ROOT / "data" / "processed" / "tiny_train.jsonl"


def main() -> None:
    stats = train_one_mtp_run(
        train_path=TRAIN_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        tiny=True,
        max_steps=3,
        batch_size=2,
        learning_rate=3e-4,
    )

    assert CHECKPOINT_PATH.exists(), "checkpoint was not created"
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    assert checkpoint["step"] == 3
    assert checkpoint["phase"] == "3B-MTP"
    assert checkpoint["config"]["mtp_enabled"] is True
    assert stats["first_loss"] > 0
    assert stats["last_loss"] > 0
    assert stats["last_lm_loss"] > 0
    assert stats["last_mtp_loss"] > 0

    print("MTP training check passed.")
    print(f"First total loss: {stats['first_loss']:.4f}")
    print(f"Last total loss: {stats['last_loss']:.4f}")
    print(f"Last LM loss: {stats['last_lm_loss']:.4f}")
    print(f"Last MTP loss: {stats['last_mtp_loss']:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
