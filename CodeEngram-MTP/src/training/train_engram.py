"""Joint fine-tuning for CodeEngram-MTP.

Phase 5 trains the combined architecture:
    Base Transformer + MTP head + Engram memory gate

Objective:
    loss = lm_loss + 0.3 * mtp_loss + 0.01 * gate_regularization

The default path is intentionally tiny/CPU-safe. Use --full only later on a real GPU.
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
from src.utils.checkpointing import save_training_checkpoint
from src.utils.dataset import CodeJsonlDataset, causal_lm_collate_fn


CONFIG_PATH = ROOT / "configs" / "model_120m.yaml"
TOKENIZER_PATH = ROOT / "data" / "processed" / "tokenizer" / "tokenizer.json"
DEFAULT_TRAIN_PATH = ROOT / "data" / "processed" / "tiny_train.jsonl"
DEFAULT_CHECKPOINT_PATH = ROOT / "checkpoints" / "joint_tiny.pt"


def load_yaml_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_joint_config(raw_config: dict[str, Any], tokenizer_size: int, tiny: bool) -> CodeEngramConfig:
    """Build config with both MTP and Engram enabled."""
    config = CodeEngramConfig.from_yaml_dict(raw_config)
    config.vocab_size = tokenizer_size

    # Phase 5 enables both prototype innovations together.
    config.mtp_enabled = True
    config.mtp_num_heads = 1
    config.engram_enabled = True

    if tiny:
        # CPU-safe debug model. This is not the final 120M training setup.
        config.max_seq_len = 64
        config.hidden_size = 64
        config.num_layers = 2
        config.num_attention_heads = 4
        config.intermediate_size = 128
        config.dropout = 0.0
        config.engram_memory_slots = 128
        config.engram_memory_dim = 64
        config.engram_ngram_size = 3
        config.engram_injection_layer = 2
        config.engram_gate_regularization_weight = 0.01

    if config.engram_memory_dim != config.hidden_size:
        # Our current gate supports projection, but matching dimensions keeps early
        # experiments simpler and easier to debug.
        config.engram_memory_dim = config.hidden_size

    return config


def train_one_joint_run(
    train_path: Path,
    checkpoint_path: Path,
    tiny: bool = True,
    max_steps: int = 10,
    batch_size: int = 2,
    learning_rate: float = 3e-4,
) -> dict[str, float]:
    raw_config = load_yaml_config()
    tokenizer = SimpleCodeTokenizer.load(TOKENIZER_PATH)
    config = build_joint_config(raw_config, tokenizer_size=len(tokenizer), tiny=tiny)

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
    last_lm_loss: float | None = None
    last_mtp_loss: float | None = None
    last_gate_reg: float | None = None
    last_gate_mean: float | None = None
    step = 0

    while step < max_steps:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            lm_loss = outputs["lm_loss"]
            mtp_loss = outputs["mtp_loss"]
            gate_reg = outputs["engram_reg_loss"]
            gate_values = outputs["engram_gate_values"]

            if loss is None or lm_loss is None or mtp_loss is None or gate_reg is None:
                raise RuntimeError("Expected LM, MTP, and Engram losses, but at least one was missing.")
            if gate_values is None:
                raise RuntimeError("Expected Engram gate values, but they were missing.")
            if not torch.isfinite(loss) or not torch.isfinite(lm_loss) or not torch.isfinite(mtp_loss):
                raise RuntimeError("Joint training loss is not finite.")
            if not torch.isfinite(gate_reg):
                raise RuntimeError("Engram gate regularization loss is not finite.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            step += 1
            loss_value = float(loss.item())
            if first_loss is None:
                first_loss = loss_value
            last_loss = loss_value
            last_lm_loss = float(lm_loss.item())
            last_mtp_loss = float(mtp_loss.item())
            last_gate_reg = float(gate_reg.item())
            last_gate_mean = float(gate_values.mean().item())

            print(
                f"step {step:04d} | total {loss_value:.4f} | "
                f"lm {last_lm_loss:.4f} | mtp {last_mtp_loss:.4f} | "
                f"gate_reg {last_gate_reg:.4f} | gate_mean {last_gate_mean:.4f}"
            )
            if step >= max_steps:
                break

    save_training_checkpoint(
        checkpoint_path,
        model=model,
        step=step,
        extra={
            "phase": "5-JOINT-MTP-ENGRAM",
            "first_loss": first_loss,
            "last_loss": last_loss,
            "last_lm_loss": last_lm_loss,
            "last_mtp_loss": last_mtp_loss,
            "last_gate_reg": last_gate_reg,
            "last_gate_mean": last_gate_mean,
            "mtp_loss_weight": config.mtp_t2_loss_weight,
            "gate_regularization_weight": config.engram_gate_regularization_weight,
        },
    )

    assert first_loss is not None
    assert last_loss is not None
    assert last_lm_loss is not None
    assert last_mtp_loss is not None
    assert last_gate_reg is not None
    assert last_gate_mean is not None
    return {
        "first_loss": first_loss,
        "last_loss": last_loss,
        "last_lm_loss": last_lm_loss,
        "last_mtp_loss": last_mtp_loss,
        "last_gate_reg": last_gate_reg,
        "last_gate_mean": last_gate_mean,
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

    stats = train_one_joint_run(
        train_path=args.train_path,
        checkpoint_path=args.checkpoint_path,
        tiny=not args.full,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    print("Joint training run complete.")
    print(f"First total loss: {stats['first_loss']:.4f}")
    print(f"Last total loss: {stats['last_loss']:.4f}")
    print(f"Last LM loss: {stats['last_lm_loss']:.4f}")
    print(f"Last MTP loss: {stats['last_mtp_loss']:.4f}")
    print(f"Last gate regularization: {stats['last_gate_reg']:.4f}")
    print(f"Last gate mean: {stats['last_gate_mean']:.4f}")
    print(f"Steps: {int(stats['steps'])}")
    print(f"Parameters: {int(stats['parameters']):,}")
    print(f"Checkpoint saved to: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
