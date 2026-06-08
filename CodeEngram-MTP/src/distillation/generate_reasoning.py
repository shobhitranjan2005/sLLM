"""Generate teacher reasoning traces for raw programming problems.

This script has two modes:
1. Mock mode: deterministic local output for pipeline testing.
2. API mode: OpenAI-compatible API call for real teacher distillation.

Example mock run:
    python src/distillation/generate_reasoning.py \
        --input data/raw/tiny_problems.jsonl \
        --output data/teacher_outputs/reasoning_mock.jsonl \
        --mock

Example API run:
    python src/distillation/generate_reasoning.py \
        --input data/raw/problems.jsonl \
        --output data/teacher_outputs/reasoning.jsonl \
        --api-base-url https://integrate.api.nvidia.com/v1 \
        --api-key-env NVIDIA_API_KEY \
        --model deepseek-ai/deepseek-r1-distill-qwen-32b
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "id" not in item or "problem" not in item:
                raise ValueError(f"Missing id/problem at {path}:{line_no}")
            records.append(item)
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def mock_reasoning(problem: dict[str, Any]) -> dict[str, Any]:
    title = problem.get("title", problem["id"])
    tags = set(problem.get("tags", []))

    if "hash-map" in tags:
        algorithm = "Use a hash map to store values already seen and look for the complement target - x."
        reasoning = (
            "For each number, the needed partner is target minus the current number. "
            "If that complement was seen earlier, the answer is found immediately."
        )
        time, space = "O(n)", "O(n)"
    elif "binary-search" in tags:
        algorithm = "Use low and high pointers and repeatedly discard half of the sorted search space."
        reasoning = (
            "Because the array is sorted, comparing the middle element with target tells us "
            "which half can still contain the answer."
        )
        time, space = "O(log n)", "O(1)"
    elif "stack" in tags:
        algorithm = "Use a stack to track opening brackets and match each closing bracket with the latest opener."
        reasoning = (
            "Bracket validity depends on nesting order. A stack preserves the most recent unmatched opener, "
            "so every closer can be checked against it."
        )
        time, space = "O(n)", "O(n)"
    else:
        algorithm = "Choose a direct algorithm based on the constraints and data structure requirements."
        reasoning = f"Analyze the constraints of {title}, identify edge cases, then implement the simplest correct approach."
        time, space = "O(n)", "O(n)"

    return {
        "id": problem["id"],
        "problem": problem["problem"],
        "reasoning": reasoning,
        "algorithm": algorithm,
        "complexity": {"time": time, "space": space},
        "teacher_model": "mock-reasoning-teacher",
    }


def build_prompt(problem: dict[str, Any]) -> str:
    return f"""You are a reasoning teacher for Python DSA problems.
Return STRICT JSON with keys: reasoning, algorithm, complexity.
complexity must have keys: time, space.

Problem ID: {problem['id']}
Problem:
{problem['problem']}
"""


def api_reasoning(problem: dict[str, Any], api_base_url: str, api_key_env: str, model: str) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai to use API mode: pip install openai") from exc

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} is not set")

    client = OpenAI(base_url=api_base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only valid JSON. No markdown."},
            {"role": "user", "content": build_prompt(problem)},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    return {
        "id": problem["id"],
        "problem": problem["problem"],
        "reasoning": parsed["reasoning"],
        "algorithm": parsed["algorithm"],
        "complexity": parsed["complexity"],
        "teacher_model": model,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/tiny_problems.jsonl")
    parser.add_argument("--output", default="data/teacher_outputs/reasoning_mock.jsonl")
    parser.add_argument("--mock", action="store_true", help="Use local deterministic mock teacher output")
    parser.add_argument("--api-base-url", default="https://integrate.api.nvidia.com/v1")
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--model", default="deepseek-ai/deepseek-r1-distill-qwen-32b")
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between API calls")
    args = parser.parse_args()

    problems = read_jsonl(args.input)
    outputs: list[dict[str, Any]] = []

    for problem in problems:
        if args.mock:
            outputs.append(mock_reasoning(problem))
        else:
            outputs.append(api_reasoning(problem, args.api_base_url, args.api_key_env, args.model))
            if args.sleep > 0:
                time.sleep(args.sleep)

    write_jsonl(args.output, outputs)
    print(f"Wrote {len(outputs)} reasoning records to {args.output}")


if __name__ == "__main__":
    main()
