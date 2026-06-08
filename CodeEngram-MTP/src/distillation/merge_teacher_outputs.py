"""Merge reasoning-teacher and code-teacher outputs into final training JSONL."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

REQUIRED_FINAL_KEYS = {"problem", "reasoning", "algorithm", "code", "complexity", "tests"}


def read_jsonl_map(path: str | Path) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "id" not in record:
                raise ValueError(f"Missing id at {path}:{line_no}")
            items[record["id"]] = record
    return items


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_code_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def validate_final_sample(sample: dict[str, Any]) -> None:
    missing = REQUIRED_FINAL_KEYS - set(sample)
    if missing:
        raise ValueError(f"Final sample missing keys: {sorted(missing)}")
    if not isinstance(sample["complexity"], dict):
        raise ValueError("complexity must be a dictionary")
    if "time" not in sample["complexity"] or "space" not in sample["complexity"]:
        raise ValueError("complexity must contain time and space")
    if not isinstance(sample["tests"], list):
        raise ValueError("tests must be a list")
    if not validate_code_syntax(sample["code"]):
        raise ValueError(f"Code syntax invalid for sample: {sample.get('id', '<unknown>')}")


def merge(reasoning_path: str | Path, code_path: str | Path) -> list[dict[str, Any]]:
    reasoning = read_jsonl_map(reasoning_path)
    code = read_jsonl_map(code_path)
    common_ids = sorted(set(reasoning) & set(code))
    if not common_ids:
        raise ValueError("No overlapping IDs between reasoning and code teacher outputs")

    merged: list[dict[str, Any]] = []
    for item_id in common_ids:
        r = reasoning[item_id]
        c = code[item_id]
        sample = {
            "id": item_id,
            "problem": r.get("problem") or c.get("problem"),
            "reasoning": r["reasoning"],
            "algorithm": r["algorithm"],
            "code": c["code"],
            "complexity": r["complexity"],
            "tests": c["tests"],
            "metadata": {
                "reasoning_teacher": r.get("teacher_model", "unknown"),
                "code_teacher": c.get("teacher_model", "unknown"),
                "distillation_stage": "phase7",
            },
        }
        validate_final_sample(sample)
        merged.append(sample)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning", default="data/teacher_outputs/reasoning_mock.jsonl")
    parser.add_argument("--code", default="data/teacher_outputs/code_mock.jsonl")
    parser.add_argument("--output", default="data/processed/distilled_train.jsonl")
    args = parser.parse_args()

    records = merge(args.reasoning, args.code)
    write_jsonl(args.output, records)
    print(f"Merged {len(records)} distilled samples into {args.output}")


if __name__ == "__main__":
    main()
