"""Check Phase 8 GRPO-style training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.train_grpo import train_one_grpo_run


def main() -> None:
    checkpoint_path = ROOT / "checkpoints" / "check_grpo_tiny.pt"
    stats = train_one_grpo_run(
        train_path=ROOT / "data" / "processed" / "tiny_train.jsonl",
        checkpoint_path=checkpoint_path,
        tiny=True,
        max_steps=2,
        group_size=3,
        learning_rate=1e-5,
        max_new_tokens=2,
    )

    if not checkpoint_path.exists():
        raise RuntimeError("GRPO checkpoint was not created.")
    if stats["steps"] != 2.0:
        raise RuntimeError("GRPO did not run the expected number of steps.")
    if not all(key in stats for key in ["loss", "mean_reward", "max_reward", "reward_spread"]):
        raise RuntimeError("GRPO stats are incomplete.")

    print("GRPO check passed.")
    print(f"Last loss: {stats['loss']:.4f}")
    print(f"Mean reward: {stats['mean_reward']:.4f}")
    print(f"Reward spread: {stats['reward_spread']:.4f}")


if __name__ == "__main__":
    main()
