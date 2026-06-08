from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.codeengram_model import CodeEngramConfig, CodeEngramForCausalLM
from src.tokenizer.code_tokenizer import SimpleCodeTokenizer
CONFIG_PATH = ROOT / "configs" / "model_120m.yaml"
TOKENIZER_PATH = ROOT / "data" / "processed" / "tokenizer" / "tokenizer.json"


def load_config(use_full_config: bool) -> CodeEngramConfig:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    config = CodeEngramConfig.from_yaml_dict(raw_config)

    if not use_full_config:
        # Tiny sanity-check mode keeps the test fast on CPU.
        tokenizer = SimpleCodeTokenizer.load(str(TOKENIZER_PATH))
        config.vocab_size = len(tokenizer)
        config.max_seq_len = 64
        config.hidden_size = 128
        config.num_layers = 2
        config.num_attention_heads = 4
        config.intermediate_size = 256
        config.dropout = 0.0

    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Instantiate the full configured model instead of the tiny CPU sanity-check model.",
    )
    args = parser.parse_args()

    config = load_config(use_full_config=args.full)
    model = CodeEngramForCausalLM(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, min(16, config.max_seq_len)))
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    logits = outputs["logits"]
    loss = outputs["loss"]

    assert logits.shape == (2, input_ids.shape[1], config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)

    print("Base Transformer check passed.")
    print(f"Mode: {'full config' if args.full else 'tiny sanity check'}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Parameter count: {model.count_parameters():,}")


if __name__ == "__main__":
    main()
