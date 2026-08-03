"""Run and conflict-safely publish the deterministic PutMarker repair lane.

Simulator output is first written below an isolated staging root.  A successful
pair is copied to one immutable pair-owned attempt directory and merged into
the latest authoritative ledger while holding a lock.  The merge asserts that
every non-target pair is byte-for-byte unchanged.  A failed pair is diagnosed
and terminates the lane before another simulator launch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_semantic_repair_campaign import _diagnosis, _strict_semantic_success
from run_three_task_asset_campaign import (
    _atomic_json,
    _command,
    _load,
    _task_success,
    enumerate_pairs,
    validate_asset_inventory,
    validate_demo,
)

def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _putmarker_task(config: dict[str, Any]) -> dict[str, Any]:
    values = [task for task in config["tasks"] if task["name"] == "putmarker"]
    if len(values) != 1:
        raise RuntimeError("campaign config must contain exactly one putmarker task")
    return values[0]


def _next_attempt(root: Path) -> Path:
    numbers = []
    for path in root.glob("attempt_*"):
        suffix = path.name.rsplit("_", 1)[-1]
        if suffix.isdigit():
            numbers.append(int(suffix))
    return root / f"attempt_{max(numbers, default=0) + 1:03d}"


def _actual_cpu_receipt(result: dict[str, Any]) -> dict[str, Any]:
    receipt = result.get("protocol", {}).get("actual_device_receipt", {})
    actual = receipt.get("actual", {})
    observed = [actual.get("manager_environment"), actual.get("simulation_context")]
    observed.extend(actual.get("action_tensors", []))
    if not (
        receipt.get("requested") == "cpu"
        and receipt.get("expected") == "cpu"
        and receipt.get("matched") is True
        and observed
        and all(value == "cpu" for value in observed)
    ):
        raise RuntimeError(f"CPU physics/action receipt failed: {receipt}")
    return receipt


def _artifact_receipt(
    attempt_root: Path, result: dict[str, Any], assets: dict[str, str]
) -> dict[str, Any]:
    source = {
        "source": "putmarker_repair_lane",
        "result_path": str(attempt_root / "skill_result.json"),
        "result": result,
        "trace_path": str(attempt_root / "skill_trace.npz"),
    }
    if not _strict_semantic_success(source):
        raise RuntimeError("strict semantic success contract was not satisfied")
    if not _task_success(result):
        raise RuntimeError("coded PutMarker task success is false")
    protocol = result.get("protocol", {})
    if not (
        protocol.get("controller")
        == "semantic_keyframe_joint_spline_with_cartesian_dls"
        and protocol.get("candidate_sampling") is False
        and protocol.get("scene_resets") == 1
        and protocol.get("inter_stage_resets") == 0
        and protocol.get("teleports_after_reset") == 0
        and protocol.get("grasp_assistance")
        in {
            "none",
            "task_config:right=friction",
            "task_config:right=fixed_joint(link_2)",
        }
    ):
        raise RuntimeError(f"deterministic continuity contract failed: {protocol}")
    device = _actual_cpu_receipt(result)
    video = result.get("video", {})
    if not (
        video.get("codec") == "h264"
        and int(video.get("size_bytes", 0)) > 0
        and int(video.get("frame_count", 0)) > 0
        and video.get("full_decode_returncode") == 0
    ):
        raise RuntimeError(f"video receipt failed: {video}")
    required = {
        "result": attempt_root / "skill_result.json",
        "trace": attempt_root / "skill_trace.npz",
        "video": attempt_root / "skill.mp4",
        "demonstration": attempt_root / "skill_demo.hdf5",
        "log": attempt_root / "skill.log",
    }
    missing = [str(path) for path in required.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing or empty accepted artifacts: {missing}")
    demonstration = validate_demo(required["demonstration"], assets)
    if demonstration["actions"] <= 0:
        raise RuntimeError("accepted HDF5 contains no actions")
    return {
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in required.items()
        },
        "demonstration": demonstration,
        "actual_device_receipt": device,
        "terminal": result["terminal"],
        "stage_success_trace": result["stage_success_trace"],
    }


def _run_pair(
    task: dict[str, Any],
    record: dict[str, Any],
    *,
    staging_root: Path,
    python: str,
    gear_repo: str,
    workflow_id: str,
) -> tuple[dict[str, Any], Path]:
    attempt_root = _next_attempt(staging_root / record["pair_id"])
    attempt_root.mkdir(parents=True, exist_ok=False)
    command = _command(
        task,
        python=python,
        gear_repo=gear_repo,
        target=record["dataset"],
        mode="skill",
        output=attempt_root,
        source_keyframes=None,
        direct_replay_result=None,
    )
    command.extend(
        ["--draw-coordinate-axes", "--camera-width", "320", "--camera-height", "240"]
    )
    log_path = attempt_root / "skill.log"
    started = datetime.now(timezone.utc).isoformat()
    with open(log_path, "w", encoding="utf-8") as stream:
        stream.write("COMMAND=" + json.dumps(command) + "\n")
        stream.write(f"WORKFLOW_ID={workflow_id}\n")
        stream.write(f"HOST={socket.gethostname()}\n")
        stream.flush()
        runner = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
        stream.write(f"RUNNER_EXIT={runner.returncode}\n")
        stream.flush()
        checker_command = [
            python,
            str(REPO_ROOT / "examples/check_putmarker_run.py"),
            "--log",
            str(log_path),
            "--result-json",
            str(attempt_root / "skill_result.json"),
            "--trace-npz",
            str(attempt_root / "skill_trace.npz"),
            "--video",
            str(attempt_root / "skill.mp4"),
        ]
        checker = subprocess.run(
            checker_command, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
        stream.write(f"CHECK_EXIT={checker.returncode}\n")
    result_path = attempt_root / "skill_result.json"
    result = _load(result_path) if result_path.is_file() else {}
    attempt = {
        "lane_attempt": attempt_root.name,
        "code_head": _git_head(),
        "workflow_id": workflow_id,
        "host": socket.gethostname(),
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "runner_returncode": runner.returncode,
        "checker_returncode": checker.returncode,
        "staging_root": str(attempt_root.resolve()),
        "status": "failed",
    }
    try:
        if runner.returncode != 0 or checker.returncode != 0:
            raise RuntimeError(
                f"runner/checker returned {runner.returncode}/{checker.returncode}"
            )
        attempt.update(_artifact_receipt(attempt_root, result, record["assets"]))
        attempt["status"] = "accepted"
    except Exception as error:
        attempt["failure"] = str(error)
        if result:
            try:
                attempt["diagnosis"] = _diagnosis(
                    "putmarker",
                    {
                        "source": "putmarker_repair_lane",
                        "result_path": str(result_path),
                        "result": result,
                        "trace_path": str(attempt_root / "skill_trace.npz"),
                    },
                )
            except Exception as diagnosis_error:
                attempt["diagnosis_error"] = str(diagnosis_error)
    _atomic_json(attempt_root / "pair_receipt.json", attempt)
    return attempt, attempt_root


def _summary(pairs: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in pairs.values():
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return {
        "total": len(pairs),
        "accepted": counts.get("accepted", 0),
        "pending": counts.get("pending", 0),
        "semantic_success_artifact_pending": counts.get(
            "semantic_success_artifact_pending", 0
        ),
        "awaiting_semantic_audit": counts.get("awaiting_semantic_audit", 0),
        "status_counts": counts,
    }


def _next_published_attempt(pair_root: Path, lane_id: str) -> Path:
    repair_root = pair_root / "semantic_repair"
    repair_root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in repair_root.glob("attempt_*"):
        suffix = path.name.rsplit("_", 1)[-1]
        if suffix.isdigit():
            numbers.append(int(suffix))
    return repair_root / f"attempt_{lane_id}_{max(numbers, default=0) + 1:03d}"


def _copy_immutable(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.publishing-{os.getpid()}")
    if temporary.exists() or destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, temporary)
    for source_path in source.iterdir():
        copied = temporary / source_path.name
        if source_path.is_file() and _sha256(source_path) != _sha256(copied):
            raise RuntimeError(f"published artifact hash mismatch: {source_path}")
    os.replace(temporary, destination)


def _merge_accepted_pair(
    *,
    ledger_path: Path,
    output_root: Path,
    key: str,
    attempt: dict[str, Any],
    attempt_root: Path,
    lane_id: str,
) -> dict[str, Any]:
    lock_path = ledger_path.with_name(ledger_path.name + ".putmarker-repair.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = _load(ledger_path)
        if key not in current.get("pairs", {}):
            raise RuntimeError(f"authoritative ledger no longer contains {key}")
        current_record = current["pairs"][key]
        if current_record.get("task") != "putmarker":
            raise RuntimeError(f"refusing non-PutMarker merge: {key}")
        if current_record.get("status") == "accepted":
            return {
                "status": "already_accepted",
                "accepted_source": current_record.get("accepted_source"),
                "ledger_sha256": _sha256(ledger_path),
            }
        before_sha = _sha256(ledger_path)
        untouched = {
            name: value for name, value in current["pairs"].items() if name != key
        }
        untouched_digest = _json_digest(untouched)
        pair_root = output_root / "putmarker" / current_record["pair_id"]
        published_root = _next_published_attempt(pair_root, lane_id)
        _copy_immutable(attempt_root, published_root)
        published_demo = validate_demo(
            published_root / "skill_demo.hdf5", current_record["assets"]
        )
        published_result = _load(published_root / "skill_result.json")
        _artifact_receipt(published_root, published_result, current_record["assets"])
        published_attempt = {
            **attempt,
            "attempt": published_root.name,
            "published_root": str(published_root.resolve()),
            "result": str((published_root / "skill_result.json").resolve()),
            "video": str((published_root / "skill.mp4").resolve()),
            "trace": str((published_root / "skill_trace.npz").resolve()),
            "demonstration": published_demo,
            "ledger_before_sha256": before_sha,
        }
        updated_record = dict(current_record)
        updated_record.update(
            {
                "status": "accepted",
                "accepted_source": lane_id,
                "result": published_attempt["result"],
                "video": published_attempt["video"],
                "demonstration": published_demo,
                "attempts": [
                    *current_record.get("attempts", []),
                    published_attempt,
                ],
                "repair_lane": {
                    "lane_id": lane_id,
                    "workflow_id": attempt["workflow_id"],
                    "code_head": attempt["code_head"],
                    "actual_device_receipt": attempt["actual_device_receipt"],
                },
            }
        )
        current["pairs"][key] = updated_record
        if _json_digest(
            {name: value for name, value in current["pairs"].items() if name != key}
        ) != untouched_digest:
            raise RuntimeError("non-target pair changed while preparing merge")
        current["summary"] = _summary(current["pairs"])
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        # Optimistic drift check supplements the lock because older campaign
        # writers predate this lock.  Never replace a ledger we did not read.
        if _sha256(ledger_path) != before_sha:
            raise RuntimeError("authoritative ledger changed during pair merge")
        _atomic_json(ledger_path, current)
        merged = _load(ledger_path)
        if _json_digest(
            {name: value for name, value in merged["pairs"].items() if name != key}
        ) != untouched_digest:
            raise RuntimeError("post-merge non-target pair digest changed")
        if merged["pairs"][key].get("accepted_source") != lane_id:
            raise RuntimeError("post-merge pair receipt was not retained")
        after_sha = _sha256(ledger_path)
        merge_receipt = {
            "status": "merged",
            "pair": key,
            "ledger_before_sha256": before_sha,
            "ledger_after_sha256": after_sha,
            "untouched_pairs_sha256": untouched_digest,
            "published_root": str(published_root.resolve()),
            "summary": merged["summary"],
        }
        _atomic_json(attempt_root / "merge_receipt.json", merge_receipt)
        return merge_receipt


def _putmarker_counts(ledger: dict[str, Any]) -> dict[str, int]:
    values = [value for value in ledger["pairs"].values() if value["task"] == "putmarker"]
    return {
        "total": len(values),
        "accepted": sum(value["status"] == "accepted" for value in values),
        "pending": sum(value["status"] != "accepted" for value in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--gear-repo", required=True)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--lane-id", default="putmarker_repair_lane_20260803")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pair", action="append")
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args()

    config = _load(Path(args.config))
    task = _putmarker_task(config)
    inventory = validate_asset_inventory(task, enumerate_pairs(task))
    ledger_path = Path(args.ledger)
    initial = _load(ledger_path)
    chosen = [
        key
        for key, value in initial["pairs"].items()
        if value["task"] == "putmarker"
        and value["status"] != "accepted"
        and (not args.pair or value["pair_id"] in args.pair or key in args.pair)
    ]
    if args.max_pairs is not None:
        chosen = chosen[: args.max_pairs]
    staging_root = Path(args.staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    lane_path = staging_root / "lane_ledger.json"
    lane = _load(lane_path) if lane_path.is_file() else {
        "schema_version": 1,
        "lane_id": args.lane_id,
        "workflow_id": args.workflow_id,
        "code_head": _git_head(),
        "host": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "authoritative_ledger_initial_sha256": _sha256(ledger_path),
        "inventory": inventory,
        "chosen_pairs": chosen,
        "pairs": {},
    }
    _atomic_json(lane_path, lane)
    print(
        "PUTMARKER_REPAIR_LANE_START="
        + json.dumps(
            {"chosen": len(chosen), "counts": _putmarker_counts(initial), "lane": lane},
            sort_keys=True,
        ),
        flush=True,
    )
    for key in chosen:
        latest = _load(ledger_path)
        record = latest["pairs"][key]
        if record["status"] == "accepted":
            lane["pairs"][key] = {"status": "accepted_elsewhere"}
            _atomic_json(lane_path, lane)
            print(f"PUTMARKER_REPAIR_SKIP_ACCEPTED={key}", flush=True)
            continue
        print(f"PUTMARKER_REPAIR_START={key}", flush=True)
        attempt, attempt_root = _run_pair(
            task,
            record,
            staging_root=staging_root,
            python=args.python,
            gear_repo=args.gear_repo,
            workflow_id=args.workflow_id,
        )
        lane["pairs"][key] = attempt
        _atomic_json(lane_path, lane)
        print(
            "PUTMARKER_REPAIR_ATTEMPT="
            + json.dumps({"pair": key, **attempt}, sort_keys=True),
            flush=True,
        )
        if attempt["status"] != "accepted":
            print(f"PUTMARKER_REPAIR_STOP_ON_FAILURE={key}", flush=True)
            raise SystemExit(2)
        merge = _merge_accepted_pair(
            ledger_path=ledger_path,
            output_root=Path(args.output_root),
            key=key,
            attempt=attempt,
            attempt_root=attempt_root,
            lane_id=args.lane_id,
        )
        lane["pairs"][key]["merge"] = merge
        _atomic_json(lane_path, lane)
        print("PUTMARKER_REPAIR_MERGE=" + json.dumps(merge, sort_keys=True), flush=True)

    final = _load(ledger_path)
    counts = _putmarker_counts(final)
    lane["finished_at"] = datetime.now(timezone.utc).isoformat()
    lane["final_counts"] = counts
    lane["authoritative_ledger_final_sha256"] = _sha256(ledger_path)
    _atomic_json(lane_path, lane)
    print("PUTMARKER_REPAIR_FINAL=" + json.dumps(counts, sort_keys=True), flush=True)
    if counts != {"total": 40, "accepted": 40, "pending": 0}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
