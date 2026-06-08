"""End-to-end inference check.

This script trains a tiny checkpoint if needed, loads it back, and generates a
few tokens. It proves that training -> checkpoint -> inference works.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tokenizer.code_tokenizer import SimpleCodeTokenizer
from src.training.train_sft import train_one_run
from src.utils.checkpointing import load_model_from_checkpoint
from src.utils.generation import generate_text

CHECKPOINT_PATH = ROOT / "checkpoints" / "sft_tiny_infer_check.pt"
TRAIN_PATH = ROOT / "data" / "processed" / "tiny_train.jsonl"
TOKENIZER_PATH = ROOT / "data" / "processed" / "tokenizer" / "tokenizer.json"


def main() -> None:
    train_one_run(
        train_path=TRAIN_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        tiny=True,
        max_steps=2,
        batch_size=2,
        learning_rate=3e-4,
    )

    tokenizer = SimpleCodeTokenizer.load(TOKENIZER_PATH)
    model, checkpoint = load_model_from_checkpoint(CHECKPOINT_PATH, device="cpu")

    prompt = "### Problem\nWrite a function to add two numbers."
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device="cpu",
        max_new_tokens=8,
        temperature=0.8,
        top_k=20,
    )

    if not isinstance(generated, str) or len(generated.strip()) == 0:
        raise RuntimeError("Inference generated empty text.")

    print("Inference check passed.")
    print(f"Checkpoint step: {checkpoint.get('step')}")
    print("Generated preview:")
    print(generated[:300])


if __name__ == "__main__":
    main()
