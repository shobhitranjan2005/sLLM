"""Final lightweight CLI demo for CodeEngram-MTP.

This script is intentionally simple: it loads a checkpoint if one exists and
runs text generation from a prompt. If no checkpoint exists, it can build a tiny
random model with --random-init so the CLI path can still be tested.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.codeengram_model import CodeEngramConfig, CodeEngramForCausalLM
from src.tokenizer.code_tokenizer import SimpleCodeTokenizer
from src.utils.checkpointing import load_model_from_checkpoint
from src.utils.generation import generate_text

TOKENIZER_PATH = ROOT / "data" / "processed" / "tokenizer" / "tokenizer.json"
DEFAULT_CHECKPOINTS = [
    ROOT / "checkpoints" / "joint_tiny.pt",
    ROOT / "checkpoints" / "mtp_tiny.pt",
    ROOT / "checkpoints" / "sft_tiny.pt",
    ROOT / "checkpoints" / "check_grpo_tiny.pt",
]


def build_tiny_random_model(vocab_size: int, device: torch.device) -> CodeEngramForCausalLM:
    """Build a CPU/GPU-safe random model for smoke-testing the demo command."""
    config = CodeEngramConfig(
        vocab_size=vocab_size,
        max_seq_len=128,
        hidden_size=96,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=192,
        dropout=0.0,
        mtp_enabled=True,
        mtp_num_heads=1,
        engram_enabled=True,
        engram_memory_slots=256,
        engram_memory_dim=96,
        engram_ngram_size=3,
        engram_injection_layer=1,
    )
    model = CodeEngramForCausalLM(config).to(device)
    model.eval()
    return model


def find_default_checkpoint() -> Path | None:
    for path in DEFAULT_CHECKPOINTS:
        if path.exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small CodeEngram-MTP demo.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="### Problem\nWrite a Python function for binary search.",
        help="Prompt passed to the model.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to load.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Use a tiny random model if no checkpoint is available. Output will not be meaningful.",
    )
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tokenizer = SimpleCodeTokenizer.load(TOKENIZER_PATH)

    checkpoint_path = args.checkpoint or find_default_checkpoint()
    checkpoint_step = "random-init"

    if checkpoint_path is not None and checkpoint_path.exists():
        model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
        checkpoint_step = checkpoint.get("step", "unknown")
        source = str(checkpoint_path)
    elif args.random_init:
        model = build_tiny_random_model(vocab_size=len(tokenizer), device=device)
        source = "tiny random model"
    else:
        raise FileNotFoundError(
            "No checkpoint found. Run a training check first, for example:\n"
            "  python scripts/check_training.py\n"
            "or use --random-init for a smoke test."
        )

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("CodeEngram-MTP Demo")
    print(f"Device: {device}")
    print(f"Model source: {source}")
    print(f"Checkpoint step: {checkpoint_step}")
    print("\nPrompt:\n")
    print(args.prompt)
    print("\nGenerated output:\n")
    print(output)


if __name__ == "__main__":
    main()
