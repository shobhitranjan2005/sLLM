"""Dataset utilities for CodeEngram-MTP.

This module loads JSONL samples in the project training format, converts each
sample into a single instruction-style text block, and provides a small collator
for causal language-model training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch


REQUIRED_FIELDS = [
    "problem",
    "reasoning",
    "algorithm",
    "code",
    "complexity",
    "tests",
]


@dataclass
class CodeSample:
    """One clean Python DSA training sample."""

    problem: str
    reasoning: str
    algorithm: str
    code: str
    complexity: Dict[str, str]
    tests: List[str]

    @classmethod
    def from_dict(cls, item: Dict[str, Any]) -> "CodeSample":
        missing = [field for field in REQUIRED_FIELDS if field not in item]
        if missing:
            raise ValueError(f"Sample is missing required fields: {missing}")

        complexity = item["complexity"]
        if not isinstance(complexity, dict):
            raise TypeError("complexity must be a dictionary with time and space")

        if "time" not in complexity or "space" not in complexity:
            raise ValueError("complexity must contain both 'time' and 'space'")

        tests = item["tests"]
        if not isinstance(tests, list):
            raise TypeError("tests must be a list of test strings")

        return cls(
            problem=str(item["problem"]),
            reasoning=str(item["reasoning"]),
            algorithm=str(item["algorithm"]),
            code=str(item["code"]),
            complexity={
                "time": str(complexity["time"]),
                "space": str(complexity["space"]),
            },
            tests=[str(test) for test in tests],
        )

    def to_training_text(self) -> str:
        """Convert the structured sample to the text format used for LM training."""
        tests_text = "\n".join(self.tests)

        return (
            "### Problem\n"
            f"{self.problem}\n\n"
            "### Reasoning\n"
            f"{self.reasoning}\n\n"
            "### Algorithm\n"
            f"{self.algorithm}\n\n"
            "### Code\n"
            f"{self.code}\n\n"
            "### Complexity\n"
            f"Time: {self.complexity['time']}\n"
            f"Space: {self.complexity['space']}\n\n"
            "### Tests\n"
            f"{tests_text}"
        )


def load_jsonl(path: str | Path) -> List[CodeSample]:
    """Load a JSONL file into a list of CodeSample objects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    samples: List[CodeSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                samples.append(CodeSample.from_dict(item))
            except Exception as exc:
                raise ValueError(f"Invalid sample at line {line_number}: {exc}") from exc

    return samples


class CodeJsonlDataset(torch.utils.data.Dataset):
    """JSONL dataset wrapper for raw text or token IDs.

    If tokenizer is None, __getitem__ returns raw training text.
    If tokenizer is provided, __getitem__ returns a dict with input_ids and labels.
    """

    def __init__(self, path: str | Path, tokenizer: Optional[Any] = None, max_length: int = 1024):
        self.samples = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        text = self.samples[index].to_training_text()

        if self.tokenizer is None:
            return text

        input_ids = self.tokenizer.encode(text, add_special_tokens=True)[: self.max_length]
        return {
            "input_ids": input_ids,
            "labels": input_ids.copy(),
        }


def causal_lm_collate_fn(batch: List[Dict[str, List[int]]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Pad a batch for causal LM training.

    input_ids are padded with pad_token_id.
    labels are padded with -100 so the loss ignores padded positions.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids: List[List[int]] = []
    labels: List[List[int]] = []

    for item in batch:
        ids = list(item["input_ids"])
        lab = list(item["labels"])
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        labels.append(lab + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def iter_training_texts(path: str | Path) -> Iterable[str]:
    """Yield formatted training texts one by one."""
    for sample in load_jsonl(path):
        yield sample.to_training_text()
