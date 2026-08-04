"""Write a durable timing receipt after an owning Isaac process exits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any


def _process_is_live(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.is_file():
        return False
    try:
        fields = stat.read_text(encoding="utf-8").split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return len(fields) > 2 and fields[2] != "Z"


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--started-monotonic", type=float, required=True)
    receipts = parser.add_mutually_exclusive_group(required=True)
    receipts.add_argument("--receipt-jsonl")
    receipts.add_argument("--receipt-json")
    parser.add_argument("--payload-json", required=True)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)

    deadline = time.monotonic() + args.timeout_s
    while _process_is_live(args.pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    finished = time.monotonic()
    payload = json.loads(args.payload_json)
    payload["shutdown"] = {
        **payload.get("shutdown", {}),
        "shutdown_s": max(0.0, finished - args.started_monotonic),
        "completion": (
            "process_exit_observed"
            if not _process_is_live(args.pid)
            else "shutdown_monitor_timeout"
        ),
        "monitor_pid": os.getpid(),
    }
    if isinstance(payload.get("phase_timings_s"), dict):
        payload["phase_timings_s"]["shutdown"] = payload["shutdown"]["shutdown_s"]
    if args.receipt_jsonl:
        _append_jsonl(Path(args.receipt_jsonl), payload)
    else:
        _write_json(Path(args.receipt_json), payload)


if __name__ == "__main__":
    main()
