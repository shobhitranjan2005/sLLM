from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.train_engram import train_one_joint_run


CHECKPOINT_PATH = ROOT / "checkpoints" / "joint_tiny_check.pt"
TRAIN_PATH = ROOT / "data" / "processed" / "tiny_train.jsonl"


def main() -> None:
    stats = train_one_joint_run(
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
    assert checkpoint["phase"] == "5-JOINT-MTP-ENGRAM"
    assert checkpoint["config"]["mtp_enabled"] is True
    assert checkpoint["config"]["engram_enabled"] is True
    assert checkpoint["config"]["engram_memory_dim"] == checkpoint["config"]["hidden_size"]
    assert stats["first_loss"] > 0
    assert stats["last_loss"] > 0
    assert stats["last_lm_loss"] > 0
    assert stats["last_mtp_loss"] > 0
    assert stats["last_gate_reg"] >= 0
    assert 0.0 <= stats["last_gate_mean"] <= 1.0

    print("Joint MTP + Engram training check passed.")
    print(f"First total loss: {stats['first_loss']:.4f}")
    print(f"Last total loss: {stats['last_loss']:.4f}")
    print(f"Last LM loss: {stats['last_lm_loss']:.4f}")
    print(f"Last MTP loss: {stats['last_mtp_loss']:.4f}")
    print(f"Last gate regularization: {stats['last_gate_reg']:.4f}")
    print(f"Last gate mean: {stats['last_gate_mean']:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
