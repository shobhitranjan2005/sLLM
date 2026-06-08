"""Supervised fine-tuning for the base CodeEngram Transformer.

Phase 2B keeps this intentionally small and readable. The default command runs a
CPU-safe tiny training check on the toy dataset. The full 120M config can be used
later when real data and GPU compute are ready.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.codeengram_model import CodeEngramConfig, CodeEngramForCausalLM
from src.tokenizer.code_tokenizer import SimpleCodeTokenizer
from src.utils.dataset import CodeJsonlDataset, causal_lm_collate_fn
from src.utils.checkpointing import save_training_checkpoint


CONFIG_PATH = ROOT / "configs" / "model_120m.yaml"
TOKENIZER_PATH = ROOT / "data" / "processed" / "tokenizer" / "tokenizer.json"
DEFAULT_TRAIN_PATH = ROOT / "data" / "processed" / "tiny_train.jsonl"
DEFAULT_CHECKPOINT_PATH = ROOT / "checkpoints" / "sft_tiny.pt"


def load_yaml_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_config(raw_config: dict[str, Any], tokenizer_size: int, tiny: bool) -> CodeEngramConfig:
    config = CodeEngramConfig.from_yaml_dict(raw_config)
    config.vocab_size = tokenizer_size

    if tiny:
        # Small enough to run quickly on CPU for pipeline verification.
        config.max_seq_len = 128
        config.hidden_size = 128
        config.num_layers = 2
        config.num_attention_heads = 4
        config.intermediate_size = 256
        config.dropout = 0.0

    return config


def train_one_run(
    train_path: Path,
    checkpoint_path: Path,
    tiny: bool = True,
    max_steps: int = 10,
    batch_size: int = 2,
    learning_rate: float = 3e-4,
) -> dict[str, float]:
    raw_config = load_yaml_config()
    tokenizer = SimpleCodeTokenizer.load(TOKENIZER_PATH)
    config = build_config(raw_config, tokenizer_size=len(tokenizer), tiny=tiny)

    dataset = CodeJsonlDataset(train_path, tokenizer=tokenizer, max_length=config.max_seq_len)
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: causal_lm_collate_fn(batch, tokenizer.pad_token_id),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not tiny else "cpu")
    model = CodeEngramForCausalLM(config).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    first_loss: float | None = None
    last_loss: float | None = None
    step = 0

    while step < max_steps:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Training loss is not finite.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            step += 1
            loss_value = float(loss.item())
            if first_loss is None:
                first_loss = loss_value
            last_loss = loss_value

            print(f"step {step:04d} | loss {loss_value:.4f}")
            if step >= max_steps:
                break

    save_training_checkpoint(
        checkpoint_path,
        model=model,
        step=step,
        extra={
            "first_loss": first_loss,
            "last_loss": last_loss,
        },
    )

    assert first_loss is not None and last_loss is not None
    return {
        "first_loss": first_loss,
        "last_loss": last_loss,
        "steps": float(step),
        "parameters": float(model.count_parameters()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the full model config. Default is tiny CPU-safe mode.",
    )
    args = parser.parse_args()

    stats = train_one_run(
        train_path=args.train_path,
        checkpoint_path=args.checkpoint_path,
        tiny=not args.full,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    print("SFT training run complete.")
    print(f"First loss: {stats['first_loss']:.4f}")
    print(f"Last loss: {stats['last_loss']:.4f}")
    print(f"Steps: {int(stats['steps'])}")
    print(f"Parameters: {int(stats['parameters']):,}")
    print(f"Checkpoint saved to: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
