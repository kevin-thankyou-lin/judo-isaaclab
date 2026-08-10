"""Run one rollout command with durable log and exact return-code bookkeeping."""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ValueError("guarded rollout requires a command after --")
    log_path = Path(args.log).resolve()
    status_path = Path(args.status_json).resolve()
    existing = [str(path) for path in (log_path, status_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite guarded rollout artifacts: {existing}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc)
    with open(log_path, "xb") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for block in iter(process.stdout.readline, b""):
            log.write(block)
            log.flush()
            sys.stdout.buffer.write(block)
            sys.stdout.buffer.flush()
        returncode = int(process.wait())
        os.fsync(log.fileno())
    finished = datetime.datetime.now(datetime.timezone.utc)
    status = {
        "schema_version": 1,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "command": command,
        "returncode": returncode,
        "log": str(log_path),
        "bookkeeping_complete": True,
    }
    with open(status_path, "xb") as stream:
        stream.write((json.dumps(status, indent=2, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print("PUTPOT_GUARDED_ROLLOUT=" + json.dumps(status, sort_keys=True))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
