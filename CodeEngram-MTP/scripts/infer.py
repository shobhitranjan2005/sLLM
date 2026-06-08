"""Run text generation from a saved CodeEngram checkpoint.

This is a demo/inference script, not an evaluation script. The tiny checkpoint
will produce weak/random text because it trains for only a few steps; the goal of
Phase 2C is to prove that checkpoint loading and generation work end-to-end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tokenizer.code_tokenizer import SimpleCodeTokenizer
from src.utils.checkpointing import load_model_from_checkpoint
from src.utils.generation import generate_text

TOKENIZER_PATH = ROOT / "data" / "processed" / "tokenizer" / "tokenizer.json"
DEFAULT_CHECKPOINT_PATH = ROOT / "checkpoints" / "sft_tiny.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--prompt", type=str, default="### Problem\nWrite a function to add two numbers.")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tokenizer = SimpleCodeTokenizer.load(TOKENIZER_PATH)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)

    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
    print("\nGenerated text:\n")
    print(text)


if __name__ == "__main__":
    main()
