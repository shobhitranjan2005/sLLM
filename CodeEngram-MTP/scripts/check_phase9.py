from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    ".gitignore",
    "LICENSE",
    "PROJECT_REPORT.md",
    "scripts/demo.py",
    "scripts/run_all_checks.py",
    "README.md",
]


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        raise RuntimeError(f"Missing Phase 9 files: {missing}")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/demo.py",
            "--random-init",
            "--cpu",
            "--prompt",
            "### Problem\nReturn the maximum of two numbers.",
            "--max-new-tokens",
            "3",
            "--temperature",
            "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Demo smoke test failed.")
    if "CodeEngram-MTP Demo" not in result.stdout:
        raise RuntimeError("Demo output header missing.")

    print("Phase 9 check passed.")
    print("Demo CLI smoke test passed.")


if __name__ == "__main__":
    main()
