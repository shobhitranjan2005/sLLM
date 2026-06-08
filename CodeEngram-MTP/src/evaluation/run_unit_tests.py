"""Safe unit-test runner for generated Python code.

This module is intentionally small and conservative. It runs candidate code in a
separate Python process with a timeout so evaluation bugs do not crash the main
training/evaluation loop.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class UnitTestResult:
    passed: bool
    total_tests: int
    passed_tests: int
    stdout: str
    stderr: str
    returncode: int


def build_test_file(code: str, tests: Iterable[str]) -> str:
    """Combine candidate code and assert-style tests into one Python file."""
    tests_list = list(tests)
    test_block = "\n".join(tests_list)
    return (
        "# Auto-generated temporary test file for CodeEngram-MTP evaluation.\n"
        + code
        + "\n\n"
        + test_block
        + "\n\nprint('ALL_TESTS_PASSED')\n"
    )


def run_python_unit_tests(code: str, tests: Iterable[str], timeout_seconds: float = 5.0) -> UnitTestResult:
    """Run code + tests in a subprocess and return a structured result."""
    tests_list: List[str] = list(tests)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "candidate_test.py"
        path.write_text(build_test_file(code, tests_list), encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return UnitTestResult(
                passed=False,
                total_tests=len(tests_list),
                passed_tests=0,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + "\nTIMEOUT",
                returncode=-1,
            )

    passed = completed.returncode == 0 and "ALL_TESTS_PASSED" in completed.stdout
    return UnitTestResult(
        passed=passed,
        total_tests=len(tests_list),
        passed_tests=len(tests_list) if passed else 0,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )
