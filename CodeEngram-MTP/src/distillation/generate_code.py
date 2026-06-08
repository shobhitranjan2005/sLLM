"""Generate teacher code and tests for raw programming problems.

Supports mock mode for safe local checks and API mode for real teacher distillation.
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


def mock_code(problem: dict[str, Any]) -> dict[str, Any]:
    tags = set(problem.get("tags", []))

    if "hash-map" in tags:
        code = """def solve(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        need = target - x\n        if need in seen:\n            return [seen[need], i]\n        seen[x] = i\n    return []\n"""
        tests = [
            "assert solve([2, 7, 11, 15], 9) == [0, 1]",
            "assert solve([3, 2, 4], 6) == [1, 2]",
        ]
    elif "binary-search" in tags:
        code = """def solve(nums, target):\n    left, right = 0, len(nums) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if nums[mid] == target:\n            return mid\n        if nums[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n"""
        tests = [
            "assert solve([-1, 0, 3, 5, 9, 12], 9) == 4",
            "assert solve([-1, 0, 3, 5, 9, 12], 2) == -1",
        ]
    elif "stack" in tags:
        code = """def solve(s):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for ch in s:\n        if ch in pairs.values():\n            stack.append(ch)\n        elif ch in pairs:\n            if not stack or stack[-1] != pairs[ch]:\n                return False\n            stack.pop()\n    return not stack\n"""
        tests = [
            "assert solve('()[]{}') is True",
            "assert solve('(]') is False",
            "assert solve('([{}])') is True",
        ]
    else:
        code = """def solve(*args, **kwargs):\n    raise NotImplementedError('Teacher code needed')\n"""
        tests = []

    return {
        "id": problem["id"],
        "problem": problem["problem"],
        "code": code,
        "tests": tests,
        "teacher_model": "mock-code-teacher",
    }


def build_prompt(problem: dict[str, Any]) -> str:
    return f"""You are a Python coding teacher.
Return STRICT JSON with keys: code, tests.
- code must define a function named solve.
- tests must be a list of Python assert statements.
- no markdown.

Problem ID: {problem['id']}
Problem:
{problem['problem']}
"""


def api_code(problem: dict[str, Any], api_base_url: str, api_key_env: str, model: str) -> dict[str, Any]:
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
        "code": parsed["code"],
        "tests": parsed["tests"],
        "teacher_model": model,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/tiny_problems.jsonl")
    parser.add_argument("--output", default="data/teacher_outputs/code_mock.jsonl")
    parser.add_argument("--mock", action="store_true", help="Use local deterministic mock teacher output")
    parser.add_argument("--api-base-url", default="https://integrate.api.nvidia.com/v1")
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--model", default="qwen/qwen2.5-coder-32b-instruct")
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between API calls")
    args = parser.parse_args()

    problems = read_jsonl(args.input)
    outputs: list[dict[str, Any]] = []

    for problem in problems:
        if args.mock:
            outputs.append(mock_code(problem))
        else:
            outputs.append(api_code(problem, args.api_base_url, args.api_key_env, args.model))
            if args.sleep > 0:
                time.sleep(args.sleep)

    write_jsonl(args.output, outputs)
    print(f"Wrote {len(outputs)} code records to {args.output}")


if __name__ == "__main__":
    main()
