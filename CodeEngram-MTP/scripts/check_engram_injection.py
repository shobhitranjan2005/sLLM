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


def build_tiny_engram_config() -> CodeEngramConfig:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    tokenizer = SimpleCodeTokenizer.load(str(TOKENIZER_PATH))
    config = CodeEngramConfig.from_yaml_dict(raw_config)
    config.vocab_size = len(tokenizer)
    config.max_seq_len = 64
    config.hidden_size = 128
    config.num_layers = 4
    config.num_attention_heads = 4
    config.intermediate_size = 256
    config.dropout = 0.0

    config.mtp_enabled = False
    config.engram_enabled = True
    config.engram_memory_slots = 256
    config.engram_memory_dim = 128
    config.engram_ngram_size = 3
    config.engram_injection_layer = 2
    config.engram_gate_regularization_weight = 0.01
    return config


def main() -> None:
    config = build_tiny_engram_config()
    model = CodeEngramForCausalLM(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 16), dtype=torch.long)
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    logits = outputs["logits"]
    loss = outputs["loss"]
    memory_ids = outputs["engram_memory_ids"]
    gate_values = outputs["engram_gate_values"]
    reg_loss = outputs["engram_reg_loss"]

    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    assert memory_ids is not None and memory_ids.shape == input_ids.shape
    assert gate_values is not None and gate_values.shape == (2, 16, 1)
    assert reg_loss is not None and torch.isfinite(reg_loss)
    assert gate_values.min().item() >= 0.0
    assert gate_values.max().item() <= 1.0

    print("Engram injection check passed.")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Memory IDs shape: {tuple(memory_ids.shape)}")
    print(f"Gate shape: {tuple(gate_values.shape)}")
    print(f"Gate mean: {gate_values.mean().item():.4f}")
    print(f"Engram regularization loss: {reg_loss.item():.4f}")
    print(f"Total loss: {loss.item():.4f}")
    print(f"Parameter count: {model.count_parameters():,}")


if __name__ == "__main__":
    main()
