"""Run same-pair PutPot retries in one persistent Isaac process.

The request manifest is a JSON list.  Every item contains an ``argv`` list for
``run_putpot_skill_program.main`` plus immutable ``log``, ``pair``, and
``code_head`` receipts.  The worker stops at a code/capability boundary or when
a diagnostic result warrants a fresh fully rendered process.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Iterator

from judo_isaaclab.putpot_runtime import (
    diagnostic_classification,
    ensure_fresh_output_paths,
    render_recommendation,
    timing_accounting,
)
import run_putpot_skill_program as putpot


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _git_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], text=True
    )


def _process_started_monotonic() -> float:
    """Return Linux process start time on the same clock as time.monotonic()."""

    stat = Path("/proc/self/stat").read_text(encoding="utf-8")
    fields_after_comm = stat[stat.rfind(")") + 2 :].split()
    start_ticks = int(fields_after_comm[19])  # proc(5) field 22; list starts at 3.
    return start_ticks / os.sysconf("SC_CLK_TCK")


@contextmanager
def _attempt_log(path: Path) -> Iterator[None]:
    """Redirect Python and native fd output to one immutable attempt log."""

    ensure_fresh_output_paths([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    with open(path, "x", encoding="utf-8") as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def _append_receipt(path: Path, value: dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    worker_started_monotonic = _process_started_monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-manifest", required=True)
    parser.add_argument("--receipt-jsonl", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.request_manifest)
    receipt_path = Path(args.receipt_jsonl)
    ensure_fresh_output_paths([receipt_path])
    requests = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(requests, list) or not requests:
        raise ValueError("persistent worker request manifest must be a nonempty list")
    startup_head = _git_head()
    startup_status = _git_status()
    if startup_status:
        raise RuntimeError(
            "persistent PutPot worker requires a clean committed code boundary"
        )
    worker_pid = os.getpid()
    worker_bootstrap_import_and_validation_s = (
        time.monotonic() - worker_started_monotonic
    )
    completed = 0
    try:
        for index, request in enumerate(requests, start=1):
            expected_head = request["code_head"]
            current_head = _git_head()
            current_status = _git_status()
            if (
                current_head != startup_head
                or current_head != expected_head
                or current_status != startup_status
            ):
                _append_receipt(
                    receipt_path,
                    {
                        "type": "worker_boundary",
                        "pid": worker_pid,
                        "startup_head": startup_head,
                        "current_head": current_head,
                        "expected_head": expected_head,
                        "startup_status": startup_status,
                        "current_status": current_status,
                        "reason": "code_head_changed",
                    },
                )
                break
            attempt_started_monotonic = time.monotonic()
            started = datetime.now(timezone.utc).isoformat()
            result_path = Path(request["result_json"])
            error = None
            with _attempt_log(Path(request["log"])):
                print("PERSISTENT_WORKER_REQUEST=" + json.dumps(request, sort_keys=True))
                try:
                    putpot.main([*request["argv"], "--persistent-session"])
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
            result = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.is_file()
                else None
            )
            classification = diagnostic_classification(result, error)
            recommendation = render_recommendation(result, error)
            attempt_wall_time_s = time.monotonic() - attempt_started_monotonic
            runtime_receipt = putpot._LAST_ATTEMPT_RUNTIME_RECEIPT
            phase_timings = (
                runtime_receipt.get("phase_timings_s", {})
                if isinstance(runtime_receipt, dict)
                else {}
            )
            completed += 1
            receipt = {
                "type": "attempt",
                "request_index": index,
                "pair": request["pair"],
                "pid": worker_pid,
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "code_head": current_head,
                "result_json": str(result_path.resolve()),
                "result_present": result is not None,
                "error": error,
                "diagnostic_classification": classification,
                "render_recommendation": recommendation,
                "timing_accounting": timing_accounting(
                    attempt_wall_time_s, phase_timings
                ),
                "worker_bootstrap_import_and_validation_s": (
                    worker_bootstrap_import_and_validation_s if index == 1 else 0.0
                ),
                "runtime": runtime_receipt,
            }
            _append_receipt(receipt_path, receipt)
            if (
                recommendation is not None
                or classification
                == "deterministic_controller_or_config_exception"
            ):
                break
    finally:
        runtime = putpot._PERSISTENT_RUNTIME
        summary = {
            "type": "worker_summary",
            "pid": worker_pid,
            "code_head": startup_head,
            "completed_attempts": completed,
            "worker_started_monotonic": worker_started_monotonic,
            "worker_bootstrap_import_and_validation_s": (
                worker_bootstrap_import_and_validation_s
            ),
            "worker_wall_time_before_shutdown_s": (
                time.monotonic() - worker_started_monotonic
            ),
            "shutdown": (
                None
                if runtime is None
                else {
                    "runtime_key": runtime["key"],
                    "attempts": runtime["attempts"],
                }
            ),
        }
        if runtime is None:
            _append_receipt(receipt_path, summary)
        else:
            shutdown_started = time.monotonic()
            subprocess.Popen(
                [
                    sys.executable,
                    str(
                        Path(__file__).resolve().parents[1]
                        / "src/judo_isaaclab/shutdown_monitor.py"
                    ),
                    "--pid",
                    str(worker_pid),
                    "--started-monotonic",
                    repr(shutdown_started),
                    "--receipt-jsonl",
                    str(receipt_path),
                    "--payload-json",
                    json.dumps(summary, sort_keys=True),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            putpot.close_persistent_runtime()


if __name__ == "__main__":
    main()
