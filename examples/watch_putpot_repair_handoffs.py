"""Wake a tmux-hosted repair agent when a PutPot handoff is left idle.

The remote campaign and persistent Isaac worker deliberately stop at durable
interactive boundaries. This watchdog closes the orchestration gap between a
written worker receipt and the next coding-agent turn without weakening the
append-only queue safeguards: it never submits a program spec itself.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from typing import Any


REMOTE_SNAPSHOT_SCRIPT = r'''
import json
from pathlib import Path
import sys


def read_jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def process_argv():
    values = []
    for child in Path("/proc").iterdir():
        if not child.name.isdigit():
            continue
        try:
            argv = [
                value.decode("utf-8", errors="replace")
                for value in (child / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if argv:
            values.append((int(child.name), argv))
    return values


def inspect_epoch(receipt_path, *, active_worker_pid=None):
    epoch = receipt_path.parent
    session_path = epoch / "interactive_session.json"
    request_path = epoch / "requests.jsonl"
    if not session_path.is_file() or not request_path.is_file():
        return None
    session = json.loads(session_path.read_text())
    requests = read_jsonl(request_path)
    receipts = read_jsonl(receipt_path)
    attempt_requests = [row for row in requests if row.get("type") == "attempt"]
    attempt_receipts = [row for row in receipts if row.get("type") == "attempt"]
    shutdown_requested = any(row.get("type") == "shutdown" for row in requests)
    summary = next(
        (row for row in reversed(receipts) if row.get("type") == "worker_summary"),
        None,
    )
    latest = attempt_receipts[-1] if attempt_receipts else None
    action = None
    fingerprint = None
    ready_since = receipt_path.stat().st_mtime
    if (
        active_worker_pid is not None
        and latest is not None
        and len(attempt_requests) == len(attempt_receipts)
        and not shutdown_requested
        and len(attempt_requests) < int(session["attempt_limit"])
        and latest.get("render_recommendation") is None
    ):
        action = "diagnose_and_repair"
        fingerprint = latest["request_id"]
        finished = latest.get("finished_at")
        if finished:
            try:
                ready_since = __import__("datetime").datetime.fromisoformat(finished).timestamp()
            except ValueError:
                pass
    elif active_worker_pid is None and summary is not None:
        action = "rotate_after_visit"
        fingerprint = f"{session['repair_epoch']}:worker_summary"
    if action is None:
        return None
    return {
        "action": action,
        "fingerprint": fingerprint,
        "pair": session["pair"],
        "session_json": str(session_path),
        "attempts_completed": len(attempt_receipts),
        "attempt_limit": int(session["attempt_limit"]),
        "latest_result_json": latest.get("result_json") if latest else None,
        "latest_program_spec_sha256": (
            latest.get("program_spec", {}).get("sha256") if latest else None
        ),
        "ready_since_epoch_s": ready_since,
        "worker_pid": active_worker_pid,
    }


root = Path(sys.argv[1])
ledger = json.loads((root / "semantic_repair_ledger.json").read_text())
summary = ledger.get("summary", {})
processes = process_argv()
workers = []
coordinators = []
for pid, argv in processes:
    joined = " ".join(argv)
    if "run_semantic_repair_campaign.py" in joined and "--task putpot" in joined:
        coordinators.append(pid)
    if not any(value.endswith("run_putpot_persistent_worker.py") for value in argv):
        continue
    if "--receipt-jsonl" not in argv:
        continue
    receipt_path = Path(argv[argv.index("--receipt-jsonl") + 1])
    workers.append((pid, receipt_path))

boundary = None
for pid, receipt_path in workers:
    candidate = inspect_epoch(receipt_path, active_worker_pid=pid)
    if candidate is not None:
        boundary = candidate
        break

if boundary is None and not workers and not coordinators:
    candidates = list((root / "putpot").glob(
        "*/semantic_repair/repair_epochs/*/worker_receipts.jsonl"
    ))
    if candidates:
        newest = max(candidates, key=lambda path: path.stat().st_mtime)
        boundary = inspect_epoch(newest)

print(json.dumps({
    "ledger_summary": summary,
    "active_worker_pids": [pid for pid, _ in workers],
    "active_coordinator_pids": coordinators,
    "boundary": boundary,
}, sort_keys=True))
'''


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _snapshot(
    ssh: list[str], results_root: str, *, timeout_seconds: float
) -> dict[str, Any]:
    completed = subprocess.run(
        [*ssh, "python3", "-", results_root],
        input=REMOTE_SNAPSHOT_SCRIPT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=timeout_seconds,
    )
    return json.loads(completed.stdout)


def _prompt(boundary: dict[str, Any]) -> str:
    if boundary["action"] == "diagnose_and_repair":
        return (
            "Continue the PutPot campaign from the durable interactive boundary. "
            f"Pair {boundary['pair']} has acknowledged "
            f"{boundary['attempts_completed']}/{boundary['attempt_limit']} cycles; "
            f"inspect {boundary['latest_result_json']} and its trace, diagnose the "
            "physics failure, then submit exactly one materially revised program "
            f"spec through {boundary['session_json']}. Do not blind-repeat or "
            "restart Isaac. Recheck the queue first and do nothing duplicate if "
            "another turn already handled this receipt."
        )
    return (
        "Continue the terminal PutPot campaign. The previous asset visit has "
        f"closed at {boundary['session_json']}; reconcile its receipts and ledger, "
        "then rotate to the next pending asset under the four-cycle policy. Recheck "
        "live processes first and do not launch a duplicate simulator."
    )


def _should_wake(
    boundary: dict[str, Any],
    state: dict[str, Any],
    *,
    now: float,
    grace_seconds: float,
    repeat_seconds: float,
    max_wakes: int,
    rotation_grace_seconds: float | None = None,
) -> bool:
    effective_grace = (
        rotation_grace_seconds
        if boundary["action"] == "rotate_after_visit"
        and rotation_grace_seconds is not None
        else grace_seconds
    )
    if now - float(boundary["ready_since_epoch_s"]) < effective_grace:
        return False
    prior = state.get("boundaries", {}).get(boundary["fingerprint"], {})
    wake_count = int(prior.get("wake_count", 0))
    if wake_count >= max_wakes:
        return False
    last_wake = prior.get("last_wake_epoch_s")
    return last_wake is None or now - float(last_wake) >= repeat_seconds


def _wake_tmux(target: str, prompt: str) -> None:
    subprocess.run(
        ["tmux", "display-message", "-pt", target, "#{pane_id}"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(["tmux", "send-keys", "-t", target, "-l", "--", prompt], check=True)
    # Codex renders pasted input asynchronously. Submitting in the immediately
    # following process can race that paste and leave a complete prompt sitting
    # unsent at the input boundary.
    time.sleep(0.25)
    subprocess.run(["tmux", "send-keys", "-t", target, "C-m"], check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ssh-command",
        required=True,
        help="quoted SSH command prefix, for example 'ssh -p 47698 host'",
    )
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--agent-tmux", required=True)
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--ssh-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--grace-seconds", type=float, default=600.0)
    parser.add_argument("--rotation-grace-seconds", type=float, default=30.0)
    parser.add_argument("--repeat-seconds", type=float, default=1200.0)
    parser.add_argument("--max-wakes", type=int, default=2)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.grace_seconds < 0
        or args.rotation_grace_seconds < 0
        or args.repeat_seconds <= 0
        or args.max_wakes < 1
        or args.ssh_timeout_seconds <= 0
    ):
        raise ValueError("invalid watchdog timing or wake limit")

    state_path = Path(args.state_json)
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"schema_version": 1, "boundaries": {}}
    )
    consecutive_snapshot_failures = 0
    while True:
        try:
            snapshot = _snapshot(
                shlex.split(args.ssh_command),
                args.results_root,
                timeout_seconds=args.ssh_timeout_seconds,
            )
            consecutive_snapshot_failures = 0
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
        ) as error:
            consecutive_snapshot_failures += 1
            print(
                "PUTPOT_HANDOFF_WATCHDOG_SNAPSHOT_ERROR="
                + json.dumps(
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "consecutive_failures": consecutive_snapshot_failures,
                        "error": f"{type(error).__name__}: {error}",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if args.once:
                raise
            time.sleep(args.poll_seconds)
            continue
        summary = snapshot.get("ledger_summary", {})
        if summary.get("accepted") == summary.get("total") and summary.get("total"):
            print("PUTPOT_HANDOFF_WATCHDOG_COMPLETE=" + json.dumps(summary, sort_keys=True))
            return
        boundary = snapshot.get("boundary")
        now = time.time()
        if boundary and _should_wake(
            boundary,
            state,
            now=now,
            grace_seconds=args.grace_seconds,
            repeat_seconds=args.repeat_seconds,
            max_wakes=args.max_wakes,
            rotation_grace_seconds=args.rotation_grace_seconds,
        ):
            prompt = _prompt(boundary)
            if not args.dry_run:
                _wake_tmux(args.agent_tmux, prompt)
            entry = state.setdefault("boundaries", {}).setdefault(
                boundary["fingerprint"], {}
            )
            entry.update(
                {
                    "action": boundary["action"],
                    "pair": boundary["pair"],
                    "last_wake_epoch_s": now,
                    "last_wake_at": datetime.now(timezone.utc).isoformat(),
                    "wake_count": int(entry.get("wake_count", 0)) + 1,
                    "dry_run": args.dry_run,
                }
            )
            _atomic_json(state_path, state)
            print(
                "PUTPOT_HANDOFF_WATCHDOG_WAKE="
                + json.dumps({**boundary, "dry_run": args.dry_run}, sort_keys=True),
                flush=True,
            )
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
