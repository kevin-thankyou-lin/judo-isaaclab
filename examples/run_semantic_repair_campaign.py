"""Analyze and resumably repair every deterministic semantic campaign pair.

This runner never overwrites the primary campaign or comparison-audit artifacts.
Each new execution is written to a numbered ``semantic_repair/attempt_*``
directory and recorded atomically before the next pair is started.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from run_three_task_asset_campaign import (
    _atomic_json,
    _command,
    _load,
    _task_success,
    enumerate_pairs,
    validate_asset_inventory,
    validate_demo,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _trace(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _result_record(result_path: Path, source: str) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    result = _load(result_path)
    trace_value = result.get("provenance", {}).get("trace", {}).get("path")
    trace_path = Path(trace_value) if trace_value else result_path.with_name("skill_trace.npz")
    return {
        "source": source,
        "result_path": str(result_path.resolve()),
        "result": result,
        "trace_path": str(trace_path),
    }


def _semantic_sources(
    task_root: Path, pair_id: str, primary: dict[str, Any], audit: dict[str, Any]
) -> list[dict[str, Any]]:
    pair_root = task_root / pair_id
    values = []
    if primary.get("method") == "deterministic_semantic_skill":
        path = Path(primary.get("result", pair_root / "skill_result.json"))
        record = _result_record(path, "primary_campaign")
        if record:
            values.append(record)
    audit_path = Path(
        audit.get("result", pair_root / "semantic_audit" / "skill_result.json")
    )
    record = _result_record(audit_path, "replay_success_semantic_audit")
    if record:
        values.append(record)
    repair_root = pair_root / "semantic_repair"
    for path in sorted(repair_root.glob("attempt_*/skill_result.json")):
        record = _result_record(path, "semantic_repair")
        if record:
            values.append(record)
    return values


def _strict_semantic_success(source: dict[str, Any]) -> bool:
    result = source["result"]
    return bool(
        result.get("mode") == "skill"
        and result.get("status") == "passed"
        and _task_success(result)
        and all(result.get("acceptance_checks", {}).values())
        and result.get("provenance", {}).get("demonstration")
    )


def _preserved_primary_success(primary: dict[str, Any], result: dict[str, Any]) -> bool:
    """Accept an already-certified deterministic primary artifact as immutable."""

    return bool(
        primary.get("status") == "accepted"
        and primary.get("method") != "source_action_replay"
        and primary.get("demonstration")
        and _task_success(result)
    )


def _semantic_motion_success(source: dict[str, Any]) -> bool:
    from run_replay_success_semantic_audit import semantic_acceptance_satisfied

    return semantic_acceptance_satisfied(source["result"])


def _validate_demo_receipt(
    demonstration: dict[str, Any], assets: dict[str, str]
) -> dict[str, Any]:
    actual = validate_demo(demonstration["path"], assets)
    if demonstration.get("sha256") not in (None, actual["sha256"]):
        raise RuntimeError(
            f"demonstration hash does not match its ledger: {demonstration['path']}"
        )
    if demonstration.get("actions") not in (None, actual["actions"]):
        raise RuntimeError(
            f"demonstration action count does not match its ledger: {demonstration['path']}"
        )
    return actual


def _diagnosis(task: str, source: dict[str, Any]) -> dict[str, Any]:
    from judo_isaaclab.semantic_repair import diagnose_semantic_failure

    result = source["result"]
    trace = _trace(Path(source["trace_path"]))
    value = diagnose_semantic_failure(task, result, trace)
    diagnosis = {
        "first_failed_stage": value.first_failed_stage,
        "reason": value.reason,
        "signed_residuals": value.signed_residuals,
        "visual_frame": value.visual_frame,
    }
    video = result.get("video") or {}
    fps = float(result.get("protocol", {}).get("control_rate_hz", 30))
    diagnosis["visual_evidence"] = {
        "video": video.get("path"),
        "video_sha256": video.get("sha256"),
        "frame": diagnosis["visual_frame"],
        "time_s": diagnosis["visual_frame"] / fps,
    }
    diagnosis["source"] = source["source"]
    diagnosis["result"] = source["result_path"]
    diagnosis["trace"] = source["trace_path"]
    return diagnosis


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _taxonomy(records: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    total = 0
    for key, record in records.items():
        task_groups = grouped.setdefault(record["task"], {})
        for diagnosis in record.get("diagnoses", []):
            total += 1
            stage = diagnosis["first_failed_stage"]
            group = task_groups.setdefault(
                stage, {"failure_records": 0, "pairs": [], "sources": []}
            )
            group["failure_records"] += 1
            if key not in group["pairs"]:
                group["pairs"].append(key)
            source = diagnosis.get("source")
            if source and source not in group["sources"]:
                group["sources"].append(source)
    return {
        "diagnosed_failure_records": total,
        "by_task_and_first_failed_stage": grouped,
    }


def _stop_after_attempt(stop_on_failure: bool, record: dict[str, Any]) -> bool:
    """Stop a sequential lane before the next pair after one failed repair."""

    return bool(stop_on_failure and record.get("status") != "accepted")


def refresh_ledger(
    config: dict[str, Any], output_root: Path, ledger_path: Path
) -> dict[str, Any]:
    existing = _load(ledger_path) if ledger_path.is_file() else {}
    old_pairs = existing.get("pairs", {})
    records: dict[str, Any] = {}
    for task in config["tasks"]:
        pairs = enumerate_pairs(task)
        validate_asset_inventory(task, pairs)
        task_root = output_root / task["name"]
        primary = _load(task_root / "ledger.json")
        audit_path = task_root / "semantic_audit_ledger.json"
        audit_pairs = _load(audit_path).get("pairs", {}) if audit_path.is_file() else {}
        for pair in pairs:
            pair_id = pair["pair_id"]
            key = f"{task['name']}:{pair_id}"
            previous = old_pairs.get(key, {})
            primary_entry = primary.get("pairs", {}).get(pair_id, {})
            sources = _semantic_sources(
                task_root,
                pair_id,
                primary_entry,
                audit_pairs.get(pair_id, {}),
            )
            primary_result_path = Path(
                primary_entry.get("result", task_root / pair_id / "skill_result.json")
            )
            primary_result = (
                _load(primary_result_path) if primary_result_path.is_file() else {}
            )
            preserved_primary = _preserved_primary_success(
                primary_entry, primary_result
            )
            successes = [source for source in sources if _strict_semantic_success(source)]
            motion_successes = [
                source
                for source in sources
                if not _strict_semantic_success(source) and _semantic_motion_success(source)
            ]
            failures = [
                source
                for source in sources
                if not _strict_semantic_success(source)
                and not _semantic_motion_success(source)
            ]
            audit_motion_successes = [
                source
                for source in motion_successes
                if source["source"] == "replay_success_semantic_audit"
            ]
            primary_demo = primary_entry.get("demonstration")
            combined_audit_success = bool(
                audit_motion_successes
                and primary_entry.get("status") == "accepted"
                and primary_demo
            )
            accepted = bool(successes or preserved_primary or combined_audit_success)
            record = {
                "task": task["name"],
                "pair_id": pair_id,
                "dataset": pair["dataset"],
                "assets": pair["assets"],
                "status": (
                    "accepted"
                    if accepted
                    else "semantic_success_artifact_pending"
                    if motion_successes
                    else "pending"
                ),
                "attempts": previous.get("attempts", []),
                "diagnoses": [_diagnosis(task["name"], source) for source in failures],
            }
            if successes:
                selected = successes[-1]
                result = selected["result"]
                demonstration = _validate_demo_receipt(
                    result["provenance"]["demonstration"], pair["assets"]
                )
                record.update(
                    {
                        "accepted_source": selected["source"],
                        "result": selected["result_path"],
                        "video": result.get("video", {}).get("path"),
                        "demonstration": demonstration,
                    }
                )
            elif preserved_primary:
                demonstration = _validate_demo_receipt(primary_demo, pair["assets"])
                record.update(
                    {
                        "accepted_source": "preserved_primary_deterministic",
                        "result": str(primary_result_path.resolve()),
                        "video": primary_entry.get("video"),
                        "demonstration": demonstration,
                    }
                )
            elif combined_audit_success:
                selected = audit_motion_successes[-1]
                demonstration = _validate_demo_receipt(primary_demo, pair["assets"])
                record.update(
                    {
                        "accepted_source": "semantic_audit_with_preserved_primary_demo",
                        "result": selected["result_path"],
                        "semantic_result": selected["result_path"],
                        "video": selected["result"].get("video", {}).get("path"),
                        "demonstration": demonstration,
                        "demonstration_source": str(primary_result_path.resolve()),
                        "semantic_acceptance_excluded_controls": [
                            "direct_source_action_replay_failed"
                        ],
                    }
                )
            elif motion_successes:
                selected = motion_successes[-1]
                record["semantic_success_evidence"] = {
                    "source": selected["source"],
                    "result": selected["result_path"],
                    "video": selected["result"].get("video", {}).get("path"),
                    "excluded_controls": ["direct_source_action_replay_failed"],
                    "artifact_gap": "strict semantic demo was not written",
                }
            elif not sources:
                record["status"] = "awaiting_semantic_audit"
            records[key] = record
    status_counts: dict[str, int] = {}
    for record in records.values():
        status = record["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    ledger = {
        "schema_version": 1,
        "campaign": str(output_root.resolve()),
        "policy": (
            "one geometry-conditioned deterministic semantic program per task; "
            "numbered immutable attempts; one heavy Isaac process"
        ),
        "code_head": _git_head(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pairs": records,
        "taxonomy": _taxonomy(records),
        "summary": {
            "total": len(records),
            "accepted": status_counts.get("accepted", 0),
            "pending": status_counts.get("pending", 0),
            "semantic_success_artifact_pending": status_counts.get(
                "semantic_success_artifact_pending", 0
            ),
            "awaiting_semantic_audit": status_counts.get("awaiting_semantic_audit", 0),
            "status_counts": status_counts,
        },
    }
    _atomic_json(ledger_path, ledger)
    return ledger


def _next_attempt(pair_root: Path) -> Path:
    repair_root = pair_root / "semantic_repair"
    numbers = [int(path.name.split("_")[-1]) for path in repair_root.glob("attempt_*")]
    return repair_root / f"attempt_{max(numbers, default=0) + 1:03d}"


def _run_pair(
    task: dict[str, Any],
    record: dict[str, Any],
    *,
    output_root: Path,
    python: str,
    gear_repo: str,
) -> dict[str, Any]:
    pair_root = output_root / task["name"] / record["pair_id"]
    attempt_root = _next_attempt(pair_root)
    attempt_root.mkdir(parents=True, exist_ok=False)
    source_keyframes = (
        output_root / task["name"] / "source_keyframes.json"
        if task.get("needs_keyframes")
        else None
    )
    command = _command(
        task,
        python=python,
        gear_repo=gear_repo,
        target=record["dataset"],
        mode="skill",
        output=attempt_root,
        source_keyframes=source_keyframes,
        # Repair is a universal semantic-generation contract.  A successful
        # source replay is evidence, not a reason to reject a valid skill demo.
        direct_replay_result=None,
    )
    started = datetime.now(timezone.utc).isoformat()
    with open(attempt_root / "skill.log", "w", encoding="utf-8") as stream:
        stream.write("COMMAND=" + json.dumps(command) + "\n")
        stream.flush()
        returncode = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=False
        ).returncode
    result_path = attempt_root / "skill_result.json"
    result = _load(result_path) if result_path.is_file() else {}
    attempt = {
        "attempt": attempt_root.name,
        "code_head": _git_head(),
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "returncode": returncode,
        "command": command,
        "result": str(result_path.resolve()),
        "video": str((attempt_root / "skill.mp4").resolve()),
        "status": "failed",
    }
    if returncode == 0 and _strict_semantic_success(
        {
            "source": "semantic_repair",
            "result_path": str(result_path),
            "result": result,
            "trace_path": str(attempt_root / "skill_trace.npz"),
        }
    ):
        demonstration = result["provenance"]["demonstration"]
        attempt["demonstration"] = validate_demo(demonstration["path"], record["assets"])
        attempt["status"] = "accepted"
    return attempt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gear-repo", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ledger")
    parser.add_argument("--task", action="append")
    parser.add_argument("--pair", action="append")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    config = _load(args.config)
    output_root = Path(args.output_root)
    ledger_path = Path(args.ledger) if args.ledger else output_root / "semantic_repair_ledger.json"
    ledger = refresh_ledger(config, output_root, ledger_path)
    print("SEMANTIC_REPAIR_SUMMARY=" + json.dumps(ledger["summary"], sort_keys=True))
    if args.analyze_only:
        return
    if ledger["summary"]["awaiting_semantic_audit"]:
        raise RuntimeError("semantic audit is incomplete; refusing to launch repair Isaac")

    selected_tasks = set(args.task or [])
    selected_pairs = set(args.pair or [])
    task_map = {task["name"]: task for task in config["tasks"]}
    pending = [
        (key, record)
        for key, record in ledger["pairs"].items()
        if record["status"] != "accepted"
        and (not selected_tasks or record["task"] in selected_tasks)
        and (not selected_pairs or record["pair_id"] in selected_pairs or key in selected_pairs)
    ]
    if args.max_pairs is not None:
        pending = pending[: args.max_pairs]
    for key, record in pending:
        print(f"SEMANTIC_REPAIR_START={key}", flush=True)
        attempt = _run_pair(
            task_map[record["task"]],
            record,
            output_root=output_root,
            python=args.python,
            gear_repo=args.gear_repo,
        )
        record["attempts"].append(attempt)
        _atomic_json(ledger_path, ledger)
        print(
            "SEMANTIC_REPAIR_ATTEMPT="
            + json.dumps({"pair": key, **attempt}, sort_keys=True),
            flush=True,
        )
        ledger = refresh_ledger(config, output_root, ledger_path)
        if _stop_after_attempt(args.stop_on_failure, ledger["pairs"][key]):
            print(f"SEMANTIC_REPAIR_STOP_ON_FAILURE={key}", flush=True)
            break

    summary = ledger["summary"]
    print(f"SEMANTIC_REPAIR_FINAL={summary['accepted']}/{summary['total']}", flush=True)
    if summary["accepted"] != summary["total"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
