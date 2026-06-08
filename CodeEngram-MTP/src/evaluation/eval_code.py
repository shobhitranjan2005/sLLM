"""Code-quality evaluation helpers.

These metrics are simple but useful for a coding LLM prototype:
- Python syntax validity
- Unit-test pass rate when tests are available
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable, List

from src.evaluation.run_unit_tests import run_python_unit_tests
from src.utils.dataset import CodeSample, load_jsonl


@dataclass
class CodeEvalResult:
    total: int
    syntax_valid: int
    unit_test_passed: int

    @property
    def syntax_valid_rate(self) -> float:
        return self.syntax_valid / self.total if self.total else 0.0

    @property
    def unit_test_pass_rate(self) -> float:
        return self.unit_test_passed / self.total if self.total else 0.0


def is_valid_python(code: str) -> bool:
    """Return True if code parses as Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def evaluate_reference_solutions(samples: Iterable[CodeSample]) -> CodeEvalResult:
    """Evaluate dataset reference code against its own tests.

    This is a data-quality check, not a model-quality check. Later, generated
    model code will be evaluated with the same primitives.
    """
    total = 0
    syntax_valid = 0
    unit_test_passed = 0

    for sample in samples:
        total += 1
        if is_valid_python(sample.code):
            syntax_valid += 1
        result = run_python_unit_tests(sample.code, sample.tests)
        if result.passed:
            unit_test_passed += 1

    return CodeEvalResult(total=total, syntax_valid=syntax_valid, unit_test_passed=unit_test_passed)


def evaluate_reference_jsonl(path: str) -> CodeEvalResult:
    return evaluate_reference_solutions(load_jsonl(path))
