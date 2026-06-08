from pathlib import Path
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data/raw/tiny_problems.jsonl"
    reasoning_path = root / "data/teacher_outputs/reasoning_mock.jsonl"
    code_path = root / "data/teacher_outputs/code_mock.jsonl"
    merged_path = root / "data/processed/distilled_train.jsonl"

    assert raw_path.exists(), f"Missing {raw_path}"

    run([
        sys.executable,
        "src/distillation/generate_reasoning.py",
        "--input",
        str(raw_path),
        "--output",
        str(reasoning_path),
        "--mock",
    ])
    run([
        sys.executable,
        "src/distillation/generate_code.py",
        "--input",
        str(raw_path),
        "--output",
        str(code_path),
        "--mock",
    ])
    run([
        sys.executable,
        "src/distillation/merge_teacher_outputs.py",
        "--reasoning",
        str(reasoning_path),
        "--code",
        str(code_path),
        "--output",
        str(merged_path),
    ])

    lines = [line for line in merged_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3, f"Expected 3 merged samples, got {len(lines)}"
    print("Distillation pipeline check passed.")
    print(f"Merged samples: {len(lines)}")


if __name__ == "__main__":
    main()
