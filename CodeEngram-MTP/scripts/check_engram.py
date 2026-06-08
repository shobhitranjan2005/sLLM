from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.engram import EngramConfig, EngramMemory
from src.tokenizer.code_tokenizer import SimpleCodeTokenizer

TOKENIZER_PATH = ROOT / "data" / "processed" / "tokenizer" / "tokenizer.json"
FACTS_PATH = ROOT / "data" / "engram_facts" / "dsa_facts.jsonl"


def main() -> None:
    tokenizer = SimpleCodeTokenizer.load(str(TOKENIZER_PATH))

    config = EngramConfig(
        memory_slots=128,
        memory_dim=64,
        ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
    )
    engram = EngramMemory(config)
    engram.eval()

    texts = [
        "binary search uses low high pointers",
        "bfs uses queue and dfs uses stack",
    ]
    encoded = [tokenizer.encode(text, add_special_tokens=True)[:12] for text in texts]
    max_len = max(len(x) for x in encoded)
    input_ids = torch.full((len(encoded), max_len), tokenizer.pad_token_id, dtype=torch.long)

    for row, ids in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    with torch.no_grad():
        memory_vectors, memory_ids = engram(input_ids)

    assert memory_ids.shape == input_ids.shape
    assert memory_vectors.shape == (input_ids.size(0), input_ids.size(1), config.memory_dim)
    assert memory_ids.min().item() >= 0
    assert memory_ids.max().item() < config.memory_slots
    assert torch.isfinite(memory_vectors).all()
    assert FACTS_PATH.exists()

    repeated_vectors, repeated_ids = engram(input_ids)
    assert torch.equal(memory_ids, repeated_ids)
    assert torch.allclose(memory_vectors, repeated_vectors)

    print("Engram memory table check passed.")
    print(f"Input IDs shape: {tuple(input_ids.shape)}")
    print(f"Memory IDs shape: {tuple(memory_ids.shape)}")
    print(f"Memory vectors shape: {tuple(memory_vectors.shape)}")
    print(f"Memory slots: {config.memory_slots}")
    print(f"Memory dim: {config.memory_dim}")
    print(f"N-gram size: {config.ngram_size}")
    print(f"Engram parameters: {engram.count_parameters():,}")
    print(f"Fact file: {FACTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
