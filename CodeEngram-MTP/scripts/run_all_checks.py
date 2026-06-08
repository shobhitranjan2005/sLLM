"""Run project sanity checks from one command.

Default mode runs all important checks. Use --quick for lightweight checks only.
The runner executes scripts in-process to avoid repeated PyTorch process-start overhead
on small CPU machines.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUICK_CHECKS = [
    "scripts/check_project.py",
    "scripts/check_dataset.py",
    "scripts/check_tokenizer.py",
    "scripts/check_model.py",
    "scripts/check_mtp.py",
    "scripts/check_engram.py",
    "scripts/check_engram_injection.py",
    "scripts/check_evaluation.py",
    "scripts/check_phase9.py",
]

FULL_EXTRA_CHECKS = [
    "scripts/check_training.py",
    "scripts/check_inference.py",
    "scripts/check_mtp_training.py",
    "scripts/check_joint_training.py",
    "scripts/check_distillation.py",
    "scripts/check_grpo.py",
]


def run_check(script: str) -> None:
    print(f"\n=== Running {script} ===", flush=True)
    old_argv = sys.argv[:]
    try:
        sys.argv = [script]
        runpy.run_path(str(ROOT / script), run_name="__main__")
    finally:
        sys.argv = old_argv


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip training-style checks.")
    args = parser.parse_args()

    checks = QUICK_CHECKS if args.quick else QUICK_CHECKS + FULL_EXTRA_CHECKS
    for script in checks:
        run_check(script)

    print("\nAll selected checks passed.")


if __name__ == "__main__":
    main()
