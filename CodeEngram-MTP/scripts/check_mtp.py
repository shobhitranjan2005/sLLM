from __future__ import annotations

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


def build_tiny_mtp_config() -> CodeEngramConfig:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    tokenizer = SimpleCodeTokenizer.load(str(TOKENIZER_PATH))
    config = CodeEngramConfig.from_yaml_dict(raw_config)

    # Tiny CPU-safe model for MTP shape/loss verification.
    config.vocab_size = len(tokenizer)
    config.max_seq_len = 64
    config.hidden_size = 128
    config.num_layers = 2
    config.num_attention_heads = 4
    config.intermediate_size = 256
    config.dropout = 0.0

    config.mtp_enabled = True
    config.mtp_num_heads = 1
    config.mtp_t2_loss_weight = 0.3

    return config


def main() -> None:
    config = build_tiny_mtp_config()
    model = CodeEngramForCausalLM(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    logits = outputs["logits"]
    mtp_logits = outputs["mtp_logits"]
    loss = outputs["loss"]
    lm_loss = outputs["lm_loss"]
    mtp_loss = outputs["mtp_loss"]

    assert logits.shape == (2, 16, config.vocab_size)
    assert isinstance(mtp_logits, list)
    assert len(mtp_logits) == 1
    assert mtp_logits[0].shape == (2, 16, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    assert lm_loss is not None and torch.isfinite(lm_loss)
    assert mtp_loss is not None and torch.isfinite(mtp_loss)

    print("MTP architecture check passed.")
    print(f"LM logits shape: {tuple(logits.shape)}")
    print(f"MTP t+2 logits shape: {tuple(mtp_logits[0].shape)}")
    print(f"LM loss: {lm_loss.item():.4f}")
    print(f"MTP loss: {mtp_loss.item():.4f}")
    print(f"Combined loss: {loss.item():.4f}")
    print(f"Parameter count: {model.count_parameters():,}")


if __name__ == "__main__":
    main()
