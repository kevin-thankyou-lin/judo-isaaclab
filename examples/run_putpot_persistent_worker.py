"""Run interactive PutPot diagnose-to-repair requests in one Isaac process.

The supervisor appends one request to ``request_jsonl`` and waits for its
durable receipt/ack before diagnosing and appending a revised program spec.
The worker never executes prebuilt identical requests.  Python/code boundaries
still require restart; validated program-spec changes are reloaded per request.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from typing import Iterator

from judo_isaaclab.putpot_runtime import (
    append_jsonl,
    diagnostic_classification,
    ensure_fresh_output_paths,
    read_jsonl,
    render_recommendation,
    timing_accounting,
    failed_stage_program_parameter_observations,
    validate_material_spec_revision,
    validate_same_spec_retry,
)
from judo_isaaclab.putpot_program_spec import load_program_spec
from judo_isaaclab.putpot_controller_protocol import sha256_file
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
    append_jsonl(path, value)


def main() -> None:
    worker_started_monotonic = _process_started_monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-jsonl", required=True)
    parser.add_argument("--receipt-jsonl", required=True)
    parser.add_argument("--poll-interval-s", type=float, default=0.1)
    args = parser.parse_args()

    request_path = Path(args.request_jsonl)
    receipt_path = Path(args.receipt_jsonl)
    ensure_fresh_output_paths([receipt_path])
    if args.poll_interval_s <= 0.0:
        raise ValueError("poll interval must be positive")
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.touch(exist_ok=True)
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
    consumed_queue_entries = 0
    previous_attempt_receipt = None

    def terminate_at_boundary(_signum, _frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate_at_boundary)
    _append_receipt(
        receipt_path,
        {
            "type": "worker_ready",
            "pid": worker_pid,
            "code_head": startup_head,
            "request_jsonl": str(request_path.resolve()),
            "max_diagnose_repair_cycles": 4,
        },
    )
    try:
        while True:
            queue = read_jsonl(request_path)
            if consumed_queue_entries >= len(queue):
                time.sleep(args.poll_interval_s)
                continue
            request = queue[consumed_queue_entries]
            consumed_queue_entries += 1
            request_type = request.get("type", "attempt")
            if request_type == "shutdown":
                _append_receipt(
                    receipt_path,
                    {
                        "type": "worker_boundary",
                        "pid": worker_pid,
                        "code_head": startup_head,
                        "request_id": request.get("request_id"),
                        "reason": "supervisor_shutdown",
                    },
                )
                break
            if request_type != "attempt":
                _append_receipt(
                    receipt_path,
                    {
                        "type": "request_rejected",
                        "pid": worker_pid,
                        "request_id": request.get("request_id"),
                        "reason": f"unknown_request_type:{request_type}",
                    },
                )
                continue
            if completed >= 4:
                _append_receipt(
                    receipt_path,
                    {
                        "type": "request_rejected",
                        "pid": worker_pid,
                        "request_id": request.get("request_id"),
                        "reason": "four_diagnose_repair_cycle_limit",
                    },
                )
                continue
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
            try:
                program_spec = load_program_spec(request["program_spec_json"])
                if program_spec.sha256 != request["program_spec_sha256"]:
                    raise ValueError(
                        "request program-spec hash does not match file bytes"
                    )
                controller_plugin_sha256 = sha256_file(
                    request["controller_plugin_py"]
                )
                if (
                    controller_plugin_sha256
                    != request["controller_plugin_sha256"]
                ):
                    raise ValueError(
                        "request controller-plugin hash does not match file bytes"
                    )
                previous_plugin_sha256 = (
                    None
                    if previous_attempt_receipt is None
                    else previous_attempt_receipt.get(
                        "controller_plugin", {}
                    ).get("sha256")
                )
                plugin_changed = (
                    previous_plugin_sha256 is not None
                    and previous_plugin_sha256 != controller_plugin_sha256
                )
                if not plugin_changed:
                    validate_same_spec_retry(
                        previous_attempt_receipt,
                        program_spec.sha256,
                        request.get("ambiguity_reason"),
                    )
                if (
                    previous_attempt_receipt is None
                    or previous_attempt_receipt["program_spec"]["sha256"]
                    != program_spec.sha256
                ):
                    validate_material_spec_revision(
                        previous_attempt_receipt,
                        program_spec.sha256,
                        program_spec.parameters,
                    )
            except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
                _append_receipt(
                    receipt_path,
                    {
                        "type": "request_rejected",
                        "pid": worker_pid,
                        "request_id": request.get("request_id"),
                        "reason": f"{type(exc).__name__}: {exc}",
                    },
                )
                continue
            index = completed + 1
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
            runtime_spec = (
                runtime_receipt.get("program_spec")
                if isinstance(runtime_receipt, dict)
                else None
            )
            result_spec = (
                result.get("provenance", {}).get("program_spec")
                if isinstance(result, dict)
                else None
            )
            runtime_controller = (
                runtime_receipt.get("controller_plugin")
                if isinstance(runtime_receipt, dict)
                else None
            )
            result_controller = (
                result.get("provenance", {}).get("controller_plugin")
                if isinstance(result, dict)
                else None
            )
            observed_hashes = {
                value.get("sha256")
                for value in (runtime_spec, result_spec)
                if isinstance(value, dict)
            }
            missing_hash_receipt = not isinstance(runtime_spec, dict) or (
                isinstance(result, dict) and not isinstance(result_spec, dict)
            )
            if missing_hash_receipt or observed_hashes != {program_spec.sha256}:
                error = (
                    "ValueError: missing or mismatched program-spec hash across "
                    "request, runtime, and result receipts"
                )
                classification = diagnostic_classification(result, error)
                recommendation = render_recommendation(result, error)
            observed_controller_hashes = {
                value.get("sha256")
                for value in (runtime_controller, result_controller)
                if isinstance(value, dict)
            }
            missing_controller_receipt = not isinstance(
                runtime_controller, dict
            ) or (
                isinstance(result, dict)
                and not isinstance(result_controller, dict)
            )
            if (
                missing_controller_receipt
                or observed_controller_hashes != {controller_plugin_sha256}
            ):
                error = (
                    "ValueError: missing or mismatched controller-plugin hash "
                    "across request, runtime, and result receipts"
                )
                classification = diagnostic_classification(result, error)
                recommendation = render_recommendation(result, error)
            failed_stage = None
            failed_stage_observations: dict[str, dict[str, object]] = {}
            try:
                failed_stage, failed_stage_observations = (
                    failed_stage_program_parameter_observations(
                        result,
                        program_spec.parameters,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                error = f"ValueError: invalid failed-stage parameter receipt: {exc}"
                classification = diagnostic_classification(result, error)
                recommendation = render_recommendation(result, error)
            if (
                isinstance(result, dict)
                and failed_stage == "bimanual_handle_grasp"
                and program_spec.parameters[
                    "receiving_jaw_close_horizon_steps"
                ]
                > 0
                and "receiving_jaw_close_horizon_steps"
                not in failed_stage_observations
            ):
                error = (
                    "ValueError: receiving-jaw close horizon was not observed "
                    "at the failed bimanual acquisition stage"
                )
                classification = diagnostic_classification(result, error)
                recommendation = render_recommendation(result, error)
            phase_timings = (
                runtime_receipt.get("phase_timings_s", {})
                if isinstance(runtime_receipt, dict)
                else {}
            )
            completed += 1
            receipt = {
                "type": "attempt",
                "request_index": index,
                "request_id": request.get("request_id"),
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
                "program_spec": program_spec.receipt(),
                "controller_plugin": runtime_controller,
                "observed_controller_plugin_hashes": sorted(
                    observed_controller_hashes
                ),
                "observed_program_spec_hashes": sorted(observed_hashes),
                "failed_stage": failed_stage,
                "failed_stage_program_parameter_observations": (
                    failed_stage_observations
                ),
                "ambiguity_reason": request.get("ambiguity_reason"),
                "acknowledged": True,
            }
            _append_receipt(receipt_path, receipt)
            previous_attempt_receipt = receipt
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
