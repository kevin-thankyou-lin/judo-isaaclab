#!/usr/bin/env python3
"""Fail when Python modules or functions exceed a ratcheted size budget."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Finding:
    level: str
    path: str
    message: str


def _line_count(text: str | None) -> int:
    return 0 if text is None else len(text.splitlines())


def _within_ratchet(current: int, baseline: int, limit: int) -> bool:
    """Permit a legacy violation only when it does not grow."""
    return current <= limit or (baseline > limit and current <= baseline)


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.symbols: dict[str, tuple[str, int, int]] = {}

    def _visit_symbol(self, node: ast.AST, kind: str) -> None:
        name = getattr(node, "name")
        qualified = ".".join([*self.stack, name])
        end_lineno = getattr(node, "end_lineno", node.lineno)
        self.symbols[qualified] = (kind, node.lineno, end_lineno - node.lineno + 1)
        self.stack.append(name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_symbol(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_symbol(node, "function")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._visit_symbol(node, "class")


def _symbols(text: str | None, path: str) -> tuple[dict[str, tuple[str, int, int]], str | None]:
    if text is None:
        return {}, None
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        return {}, f"cannot parse Python: {exc.msg} at line {exc.lineno}"
    visitor = _SymbolVisitor()
    visitor.visit(tree)
    return visitor.symbols, None


def evaluate_file(
    path: str,
    current_text: str,
    baseline_text: str | None,
    profile: dict[str, Any],
) -> list[Finding]:
    findings: list[Finding] = []
    current_lines = _line_count(current_text)
    baseline_lines = _line_count(baseline_text)
    module_warning = int(profile["module_warning_lines"])
    module_limit = int(profile["module_limit_lines"])

    if not _within_ratchet(current_lines, baseline_lines, module_limit):
        findings.append(
            Finding(
                "ERROR",
                path,
                f"module has {current_lines} lines; limit={module_limit}, "
                f"baseline={baseline_lines}; extract a coherent responsibility",
            )
        )
    elif current_lines > module_warning and current_lines != baseline_lines:
        disposition = "grandfathered without growth" if baseline_lines > module_limit else "warning"
        findings.append(
            Finding(
                "WARN",
                path,
                f"module has {current_lines} lines (warning={module_warning}; {disposition})",
            )
        )

    current_symbols, parse_error = _symbols(current_text, path)
    baseline_symbols, _ = _symbols(baseline_text, path)
    if parse_error:
        findings.append(Finding("ERROR", path, parse_error))
        return findings

    function_warning = int(profile["function_warning_lines"])
    function_limit = int(profile["function_limit_lines"])
    main_limit = profile.get("main_limit_lines")
    for qualified, (kind, lineno, current_size) in sorted(current_symbols.items()):
        if kind != "function":
            continue
        baseline_size = baseline_symbols.get(qualified, (kind, 0, 0))[2]
        is_main = qualified == "main"
        limit = int(main_limit) if is_main and main_limit is not None else function_limit
        warning = min(function_warning, limit)
        symbol_path = f"{path}:{lineno} ({qualified})"
        if not _within_ratchet(current_size, baseline_size, limit):
            findings.append(
                Finding(
                    "ERROR",
                    symbol_path,
                    f"function has {current_size} lines; limit={limit}, baseline={baseline_size}; split it",
                )
            )
        elif current_size > warning and current_size != baseline_size:
            disposition = "grandfathered without growth" if baseline_size > limit else "warning"
            findings.append(
                Finding(
                    "WARN",
                    symbol_path,
                    f"function has {current_size} lines (warning={warning}; {disposition})",
                )
            )
    return findings


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _baseline_text(repo: Path, baseline_ref: str, path: str) -> str | None:
    result = _git(repo, "show", f"{baseline_ref}:{path}", check=False)
    return result.stdout if result.returncode == 0 else None


def _profile_for(path: str, profiles: list[dict[str, Any]]) -> dict[str, Any] | None:
    parts = Path(path).parts
    for profile in profiles:
        if parts and parts[0] in profile["roots"]:
            return profile
    return None


def _current_python_files(repo: Path, profiles: list[dict[str, Any]]) -> list[Path]:
    roots = {root for profile in profiles for root in profile["roots"]}
    paths: set[Path] = set()
    for root in roots:
        root_path = repo / root
        if root_path.exists():
            paths.update(path for path in root_path.rglob("*.py") if path.is_file())
    return sorted(paths)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/modularity_budget.json",
        help="budget JSON relative to the repository root",
    )
    parser.add_argument("--baseline-ref", help="override the pinned baseline revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel").stdout.strip())
    config_path = repo / args.config
    config = json.loads(config_path.read_text())
    baseline_ref = args.baseline_ref or config["baseline_ref"]
    profiles = config["profiles"]

    resolved = _git(repo, "rev-parse", "--verify", f"{baseline_ref}^{{commit}}", check=False)
    if resolved.returncode != 0:
        print(f"ERROR {config_path}: baseline revision is unavailable: {baseline_ref}")
        return 2

    findings: list[Finding] = []
    production_delta = 0
    for current_path in _current_python_files(repo, profiles):
        relative = current_path.relative_to(repo).as_posix()
        profile = _profile_for(relative, profiles)
        if profile is None:
            continue
        current_text = current_path.read_text()
        baseline_text = _baseline_text(repo, baseline_ref, relative)
        findings.extend(evaluate_file(relative, current_text, baseline_text, profile))
        if profile["name"] == "production":
            production_delta += _line_count(current_text) - _line_count(baseline_text)

    large_change_warning = int(config["large_change_warning_lines"])
    if production_delta > large_change_warning:
        findings.append(
            Finding(
                "WARN",
                "production total",
                f"net growth is {production_delta} lines from {baseline_ref[:12]} "
                f"(review threshold={large_change_warning}); document the decomposition",
            )
        )

    for finding in sorted(findings, key=lambda item: (item.level != "ERROR", item.path)):
        print(f"{finding.level} {finding.path}: {finding.message}")
    errors = sum(finding.level == "ERROR" for finding in findings)
    warnings = sum(finding.level == "WARN" for finding in findings)
    print(
        f"modularity budget: {'FAIL' if errors else 'PASS'}; "
        f"errors={errors}; warnings={warnings}; baseline={baseline_ref}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
