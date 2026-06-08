from __future__ import annotations

import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.ablation import format_ablation_table, run_forward_ablation
from src.evaluation.eval_code import evaluate_reference_jsonl
from src.evaluation.eval_memory import evaluate_engram_gate_activity
from src.model.codeengram_model import CodeEngramConfig, CodeEngramForCausalLM
from src.tokenizer.code_tokenizer import SimpleCodeTokenizer
from src.utils.dataset import CodeJsonlDataset, causal_lm_collate_fn

TOKENIZER_PATH = ROOT / "data" / "processed" / "tokenizer" / "tokenizer.json"
VAL_PATH = ROOT / "data" / "processed" / "tiny_val.jsonl"


def build_tiny_base_config(vocab_size: int) -> CodeEngramConfig:
    return CodeEngramConfig(
        vocab_size=vocab_size,
        max_seq_len=64,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        dropout=0.0,
        mtp_num_heads=1,
        engram_memory_slots=128,
        engram_memory_dim=64,
        engram_ngram_size=3,
        engram_injection_layer=2,
    )


def main() -> None:
    tokenizer = SimpleCodeTokenizer.load(TOKENIZER_PATH)

    code_result = evaluate_reference_jsonl(str(VAL_PATH))
    assert code_result.total == 1
    assert code_result.syntax_valid == 1
    assert code_result.unit_test_passed == 1

    dataset = CodeJsonlDataset(VAL_PATH, tokenizer=tokenizer, max_length=64)
    batch = causal_lm_collate_fn([dataset[0]], tokenizer.pad_token_id)
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    base_config = build_tiny_base_config(vocab_size=len(tokenizer))
    rows = run_forward_ablation(base_config, input_ids=input_ids, labels=labels)
    assert len(rows) == 4
    assert any(row.mtp_loss is not None for row in rows)
    assert any(row.gate_mean is not None for row in rows)

    engram_config = build_tiny_base_config(vocab_size=len(tokenizer))
    engram_config.engram_enabled = True
    model = CodeEngramForCausalLM(engram_config)
    memory_result = evaluate_engram_gate_activity(
        model,
        tokenizer,
        prompts=["binary search time complexity", "BFS uses queue"],
    )
    assert memory_result.total_prompts == 2
    assert memory_result.memory_vectors_seen is True
    assert memory_result.gate_mean >= 0.0

    print("Evaluation check passed.")
    print(f"Syntax valid rate: {code_result.syntax_valid_rate:.2f}")
    print(f"Unit-test pass rate: {code_result.unit_test_pass_rate:.2f}")
    print(format_ablation_table(rows))
    print(f"Engram gate mean: {memory_result.gate_mean:.4f}")


if __name__ == "__main__":
    main()
