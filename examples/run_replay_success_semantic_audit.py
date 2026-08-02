"""Run the deterministic semantic skill on pairs already accepted by replay.

This is a comparison audit: it never mutates the primary campaign ledger or
overwrites the accepted replay artifacts.  Results live below each pair's
``semantic_audit`` directory and in a separate per-task audit ledger.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from run_three_task_asset_campaign import (
    _atomic_json,
    _command,
    _load,
    _run,
    _sha256,
    _task_success,
    enumerate_pairs,
    validate_asset_inventory,
    validate_demo,
)


def reusable_semantic_result(
    result: dict[str, Any], target_dataset: str
) -> bool:
    """Verify terminal semantic evidence without requiring task success."""
    if result.get("mode") != "skill" or result.get("status") not in {
        "passed",
        "failed",
    }:
        return False
    provenance = result.get("provenance", {})
    if provenance.get("target_dataset", {}).get("sha256") != _sha256(
        target_dataset
    ):
        return False
    for artifact in (result.get("video"), provenance.get("trace")):
        if not artifact or not os.path.isfile(artifact.get("path", "")):
            return False
        if artifact.get("sha256") != _sha256(artifact["path"]):
            return False
    required = (
        "fully_decodable",
        "h264_nonempty",
        "one_reset",
        "real_target_assets",
        "zero_inter_stage_resets",
    )
    checks = result.get("acceptance_checks", {})
    return all(checks.get(name) is True for name in required)


def select_replay_success_pairs(
    pairs: list[dict[str, Any]], ledger: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return only pairs accepted by unmodified source-action replay."""
    records = ledger.get("pairs", {})
    return [
        pair
        for pair in pairs
        if records.get(pair["pair_id"], {}).get("status") == "accepted"
        and records[pair["pair_id"]].get("method") == "source_action_replay"
    ]


def run_task_audit(
    task: dict[str, Any],
    *,
    python: str,
    gear_repo: str,
    output_root: Path,
    dry_run: bool,
    max_pairs: int | None,
) -> dict[str, Any]:
    pairs = enumerate_pairs(task)
    validate_asset_inventory(task, pairs)
    task_root = output_root / task["name"]
    campaign_ledger = _load(task_root / "ledger.json")
    selected = select_replay_success_pairs(pairs, campaign_ledger)
    if max_pairs is not None:
        selected = selected[:max_pairs]
    print(
        "SEMANTIC_AUDIT_SELECTION="
        + json.dumps(
            {"task": task["name"], "pairs": len(selected)}, sort_keys=True
        ),
        flush=True,
    )

    ledger_path = task_root / "semantic_audit_ledger.json"
    ledger = _load(ledger_path) if ledger_path.is_file() else {
        "schema_version": 1,
        "task": task["name"],
        "selection": "primary campaign source_action_replay successes",
        "pairs": {},
    }
    source_keyframes = (
        task_root / "source_keyframes.json" if task.get("needs_keyframes") else None
    )
    if source_keyframes is not None and not source_keyframes.is_file():
        raise RuntimeError(f"missing source keyframes: {source_keyframes}")

    for pair in selected:
        pair_id = pair["pair_id"]
        existing = ledger["pairs"].get(pair_id, {})
        if existing.get("status") == "accepted":
            validate_demo(existing["demonstration"]["path"], pair["assets"])
            print(f"SEMANTIC_AUDIT_RESUME_ACCEPTED={task['name']}:{pair_id}")
            continue
        if existing.get("status") == "semantic_failed":
            existing_result = _load(existing["result"])
            if reusable_semantic_result(existing_result, pair["dataset"]):
                print(
                    f"SEMANTIC_AUDIT_RESUME_FAILED={task['name']}:{pair_id}"
                )
                continue

        pair_root = task_root / pair_id
        audit_root = pair_root / "semantic_audit"
        audit_root.mkdir(parents=True, exist_ok=True)
        result_path = audit_root / "skill_result.json"
        command = _command(
            task,
            python=python,
            gear_repo=gear_repo,
            target=pair["dataset"],
            mode="skill",
            output=audit_root,
            source_keyframes=source_keyframes,
            direct_replay_result=pair_root / "replay_result.json",
        )
        returncode = _run(command, audit_root / "skill.log", dry_run=dry_run)
        if dry_run:
            continue

        result = _load(result_path) if result_path.is_file() else {}
        if returncode != 0 or not result:
            record = {
                "status": "infrastructure_failed",
                "returncode": returncode,
                "result": str(result_path.resolve()),
            }
        elif result.get("status") == "passed" and _task_success(result):
            demo = validate_demo(audit_root / "skill_demo.hdf5", pair["assets"])
            record = {
                "status": "accepted",
                "result": str(result_path.resolve()),
                "video": str((audit_root / "skill.mp4").resolve()),
                "demonstration": demo,
            }
        else:
            record = {
                "status": "semantic_failed",
                "returncode": returncode,
                "result": str(result_path.resolve()),
                "video": str((audit_root / "skill.mp4").resolve()),
            }
        record.update(
            {
                "dataset": pair["dataset"],
                "assets": pair["assets"],
                "baseline_method": "source_action_replay",
            }
        )
        ledger["pairs"][pair_id] = record
        ledger["summary"] = {
            "selected": len(selected),
            "completed": sum(
                value.get("status") in {"accepted", "semantic_failed"}
                for value in ledger["pairs"].values()
            ),
            "accepted": sum(
                value.get("status") == "accepted"
                for value in ledger["pairs"].values()
            ),
            "semantic_failed": sum(
                value.get("status") == "semantic_failed"
                for value in ledger["pairs"].values()
            ),
        }
        _atomic_json(ledger_path, ledger)
        print(
            "SEMANTIC_AUDIT_PAIR="
            + json.dumps(
                {"task": task["name"], "pair": pair_id, **record},
                sort_keys=True,
            ),
            flush=True,
        )
        if record["status"] == "infrastructure_failed":
            raise RuntimeError(f"{task['name']}:{pair_id}: infrastructure failure")
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gear-repo", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--task", action="append")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = _load(args.config)
    selected_names = set(args.task or [])
    tasks = [
        task
        for task in config["tasks"]
        if not selected_names or task["name"] in selected_names
    ]
    unknown = selected_names - {task["name"] for task in tasks}
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    ledgers = [
        run_task_audit(
            task,
            python=args.python,
            gear_repo=args.gear_repo,
            output_root=Path(args.output_root),
            dry_run=args.dry_run,
            max_pairs=args.max_pairs,
        )
        for task in tasks
    ]
    if args.dry_run:
        print("SEMANTIC_AUDIT_DRY_RUN_PASSED")
        return
    accepted = sum(
        ledger.get("summary", {}).get("accepted", 0) for ledger in ledgers
    )
    selected = sum(
        ledger.get("summary", {}).get("selected", 0) for ledger in ledgers
    )
    print(f"SEMANTIC_AUDIT_FINAL={accepted}/{selected}", flush=True)


if __name__ == "__main__":
    main()
