"""Bind one immutable perturbation case to the opt-in physical PutPot runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def adapted_command(case_json: str | Path, command: list[str]) -> list[str]:
    """Append the runner-owned case flag exactly once."""

    if not command:
        raise ValueError("perturbation adapter requires a runner command")
    if "--quality-config-json" not in command:
        raise ValueError("perturbation adapter requires explicit quality mode")
    if "--quality-perturbation-case-json" in command:
        raise ValueError("runner command already owns a perturbation case")
    path = Path(case_json).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"perturbation case is missing: {path}")
    return [*command, "--quality-perturbation-case-json", str(path)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    return subprocess.run(adapted_command(args.case_json, command)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
