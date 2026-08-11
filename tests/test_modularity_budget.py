from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_modularity_budget.py"
SPEC = importlib.util.spec_from_file_location("check_modularity_budget", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
budget = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = budget
SPEC.loader.exec_module(budget)


PROFILE = {
    "module_warning_lines": 8,
    "module_limit_lines": 12,
    "function_warning_lines": 4,
    "function_limit_lines": 8,
    "main_limit_lines": 6,
}


def _function(lines: int, name: str = "repair") -> str:
    body = "\n".join(f"    value_{index} = {index}" for index in range(lines - 1))
    return f"def {name}():\n{body}\n"


def _errors(current: str, baseline: str | None):
    return [
        finding
        for finding in budget.evaluate_file("src/task.py", current, baseline, PROFILE)
        if finding.level == "ERROR"
    ]


def test_inherited_oversized_code_is_grandfathered_without_growth():
    inherited = _function(14)
    assert not _errors(inherited, inherited)


def test_inherited_oversized_code_cannot_grow():
    assert _errors(_function(15), _function(14))


def test_new_oversized_module_is_rejected():
    assert _errors("\n".join(f"value_{index} = {index}" for index in range(13)), None)


def test_main_has_the_stricter_orchestration_limit():
    errors = _errors(_function(7, name="main"), None)
    assert any("limit=6" in finding.message for finding in errors)


def test_small_extracted_strategy_module_passes():
    assert not _errors(_function(4), None)
