"""Train the early dependency-free tokenizer on JSONL training data."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tokenizer.code_tokenizer import SimpleCodeTokenizer
from src.utils.dataset import iter_training_texts


def train_tokenizer(train_path: str, output_path: str, vocab_size: int = 512) -> SimpleCodeTokenizer:
    tokenizer = SimpleCodeTokenizer.train_from_texts(
        iter_training_texts(train_path),
        vocab_size=vocab_size,
    )
    tokenizer.save(output_path)
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the simple CodeEngram tokenizer.")
    parser.add_argument("--train-path", default="data/processed/tiny_train.jsonl")
    parser.add_argument("--output-path", default="data/processed/tokenizer/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=512)
    args = parser.parse_args()

    tokenizer = train_tokenizer(args.train_path, args.output_path, args.vocab_size)
    print(f"Tokenizer saved to: {Path(args.output_path)}")
    print(f"Vocab size: {len(tokenizer)}")


if __name__ == "__main__":
    main()
