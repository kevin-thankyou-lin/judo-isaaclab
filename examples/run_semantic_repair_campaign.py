"""Analyze and resumably repair every deterministic semantic campaign pair.

This runner never overwrites the primary campaign or comparison-audit artifacts.
Each new execution is written to a numbered ``semantic_repair/attempt_*``
directory and recorded atomically before the next pair is started.
PutPot round-robin visits are fail-closed and capped at four interactive
diagnose-to-repair cycles before the asset must rotate to hard-case review.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from run_three_task_asset_campaign import (
    _atomic_json,
    _command,
    _load,
    _sha256,
    _task_success,
    enumerate_pairs,
    validate_asset_inventory,
    validate_demo,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PUTPOT_MAX_FRESH_ATTEMPTS_PER_ASSET_VISIT = 4


def _validate_fresh_attempts_per_asset_visit(
    value: int, selected_tasks: set[str]
) -> int:
    if value < 1:
        raise ValueError("fresh attempts per asset visit must be positive")
    if (
        not selected_tasks or "putpot" in selected_tasks
    ) and value > PUTPOT_MAX_FRESH_ATTEMPTS_PER_ASSET_VISIT:
        raise ValueError(
            "PutPot asset visits are capped at four diagnose-to-repair cycles"
        )
    return value


def _trace(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _published_artifact_receipt(
    result_path: Path,
    artifact: dict[str, Any] | None,
    sibling_name: str,
) -> tuple[dict[str, Any] | None, bool]:
    """Rebase inaccessible provenance to a hash-identical published sibling."""

    if not artifact or not artifact.get("path"):
        return artifact, False
    original = Path(artifact["path"])
    if original.is_file():
        return artifact, False
    sibling = result_path.with_name(sibling_name)
    expected_hash = artifact.get("sha256")
    if (
        not sibling.is_file()
        or expected_hash is None
        or _sha256(sibling) != expected_hash
    ):
        return artifact, False
    published = dict(artifact)
    published["path"] = str(sibling.resolve())
    return published, True


def _result_record(result_path: Path, source: str) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    result = _load(result_path)
    provenance = result.get("provenance", {})
    video, video_rebased = _published_artifact_receipt(
        result_path, result.get("video"), "skill.mp4"
    )
    trace, trace_rebased = _published_artifact_receipt(
        result_path, provenance.get("trace"), "skill_trace.npz"
    )
    demonstration, demo_rebased = _published_artifact_receipt(
        result_path, provenance.get("demonstration"), "skill_demo.hdf5"
    )
    trace_value = (trace or {}).get("path")
    trace_path = Path(trace_value) if trace_value else result_path.with_name("skill_trace.npz")
    return {
        "source": source,
        "result_path": str(result_path.resolve()),
        "result": result,
        "trace_path": str(trace_path),
        "video_path": (video or {}).get("path"),
        "demonstration": demonstration,
        "published_artifacts_rebased": bool(
            video_rebased or trace_rebased or demo_rebased
        ),
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


def _requires_putpot_bimanual_transport(
    task: str | None, source: dict[str, Any]
) -> bool:
    """Identify target-direct PutPot repairs governed by the transport gate."""

    return task == "putpot" and source.get("source") == "semantic_repair"


def _putpot_bimanual_transport_satisfied(
    task: str | None, source: dict[str, Any]
) -> bool:
    return bool(
        not _requires_putpot_bimanual_transport(task, source)
        or source["result"].get("checks", {}).get(
            "bimanual_transport_completed"
        )
        is True
    )


def _strict_semantic_success(
    source: dict[str, Any], task: str | None = None
) -> bool:
    from judo_isaaclab.putpot_runtime import full_render_required_for_merge

    result = source["result"]
    return bool(
        result.get("mode") == "skill"
        and result.get("status") == "passed"
        and _task_success(result)
        and all(result.get("acceptance_checks", {}).values())
        and _putpot_bimanual_transport_satisfied(task, source)
        and result.get("provenance", {}).get("demonstration")
        and (task != "putpot" or full_render_required_for_merge(result))
    )


def _preserved_primary_success(primary: dict[str, Any], result: dict[str, Any]) -> bool:
    """Accept an already-certified deterministic primary artifact as immutable."""

    return bool(
        primary.get("status") == "accepted"
        and primary.get("method") != "source_action_replay"
        and primary.get("demonstration")
        and _task_success(result)
    )


def _semantic_motion_success(
    source: dict[str, Any], task: str | None = None
) -> bool:
    from run_replay_success_semantic_audit import semantic_acceptance_satisfied

    return bool(
        _putpot_bimanual_transport_satisfied(task, source)
        and semantic_acceptance_satisfied(source["result"])
    )


def _validate_demo_receipt(
    demonstration: dict[str, Any],
    assets: dict[str, str],
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_path = demonstration["path"]
    if not Path(selected_path).is_file() and fallback is not None:
        if demonstration.get("sha256") not in (None, fallback.get("sha256")):
            raise RuntimeError(
                "preserved demonstration hash differs from inaccessible provenance"
            )
        if demonstration.get("actions") not in (None, fallback.get("actions")):
            raise RuntimeError(
                "preserved demonstration action count differs from inaccessible provenance"
            )
        selected_path = fallback["path"]
    actual = validate_demo(selected_path, assets)
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


def _pending_status(previous: dict[str, Any]) -> str:
    """Preserve an operator-classified hard case across ledger refreshes."""

    return (
        "hard_case_pending"
        if previous.get("status") == "hard_case_pending"
        else "pending"
    )


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
            successes = [
                source
                for source in sources
                if _strict_semantic_success(source, task["name"])
            ]
            motion_successes = [
                source
                for source in sources
                if not _strict_semantic_success(source, task["name"])
                and _semantic_motion_success(source, task["name"])
            ]
            failures = [
                source
                for source in sources
                if not _strict_semantic_success(source, task["name"])
                and not _semantic_motion_success(source, task["name"])
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
            pending_status = _pending_status(previous)
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
                    else pending_status
                ),
                "attempts": previous.get("attempts", []),
                "diagnoses": [_diagnosis(task["name"], source) for source in failures],
            }
            if successes:
                selected = successes[-1]
                result = selected["result"]
                demonstration = _validate_demo_receipt(
                    selected.get("demonstration")
                    or result["provenance"]["demonstration"],
                    pair["assets"],
                    fallback=previous.get("demonstration"),
                )
                record.update(
                    {
                        "accepted_source": selected["source"],
                        "result": selected["result_path"],
                        "video": selected.get("video_path")
                        or result.get("video", {}).get("path"),
                        "demonstration": demonstration,
                    }
                )
                if selected.get("published_artifacts_rebased"):
                    record["trace"] = selected["trace_path"]
            elif preserved_primary:
                demonstration = _validate_demo_receipt(
                    primary_demo,
                    pair["assets"],
                    fallback=previous.get("demonstration"),
                )
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
                demonstration = _validate_demo_receipt(
                    primary_demo,
                    pair["assets"],
                    fallback=previous.get("demonstration"),
                )
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
            "target-direct geometry-conditioned deterministic semantic program per task; "
            "source demonstration supplies phase/contact intent but source semantic success "
            "is not required; reanchor contact milestones; separate task and motion gates; "
            "numbered immutable attempts; preserve hash-verified successes; stop on first "
            "failed pair; PutPot round-robin visits capped at four diagnose-to-repair cycles; "
            "one heavy Isaac process"
        ),
        "code_head": _git_head(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pairs": records,
        "taxonomy": _taxonomy(records),
        "summary": {
            "total": len(records),
            "accepted": status_counts.get("accepted", 0),
            "pending": (
                status_counts.get("pending", 0)
                + status_counts.get("hard_case_pending", 0)
            ),
            "hard_case_pending": status_counts.get("hard_case_pending", 0),
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


def _attempt_number(path: Path) -> int:
    return int(path.name.split("_")[-1])


def _run_pair(
    task: dict[str, Any],
    record: dict[str, Any],
    *,
    output_root: Path,
    python: str,
    gear_repo: str,
    repair_epoch: str | None = None,
    repair_epoch_attempt: int | None = None,
    program_spec_json: str | Path | None = None,
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
    if task["name"] == "putpot":
        selected_spec = (
            Path(program_spec_json)
            if program_spec_json is not None
            else REPO_ROOT / "configs/putpot_semantic_program_v4.json"
        )
        command.extend(["--program-spec-json", str(selected_spec.resolve())])
    if repair_epoch is not None:
        command.extend(
            [
                "--runtime-receipt-json",
                str(attempt_root / "skill_runtime.json"),
                "--lifetime-attempt-number",
                str(_attempt_number(attempt_root)),
                "--repair-epoch",
                repair_epoch,
                "--repair-epoch-attempt",
                str(repair_epoch_attempt),
                "--repair-epoch-attempt-limit",
                str(PUTPOT_MAX_FRESH_ATTEMPTS_PER_ASSET_VISIT),
            ]
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
        "runtime_receipt": (
            str((attempt_root / "skill_runtime.json").resolve())
            if repair_epoch is not None
            else None
        ),
        "status": "failed",
        "lifetime_attempt": _attempt_number(attempt_root),
        "repair_epoch": repair_epoch,
        "repair_epoch_attempt": repair_epoch_attempt,
        "repair_epoch_attempt_limit": (
            PUTPOT_MAX_FRESH_ATTEMPTS_PER_ASSET_VISIT
            if repair_epoch is not None
            else None
        ),
    }
    if returncode == 0 and _strict_semantic_success(
        {
            "source": "semantic_repair",
            "result_path": str(result_path),
            "result": result,
            "trace_path": str(attempt_root / "skill_trace.npz"),
        },
        task["name"],
    ):
        demonstration = result["provenance"]["demonstration"]
        attempt["demonstration"] = validate_demo(demonstration["path"], record["assets"])
        attempt["status"] = "accepted"
    return attempt


def _without_render(command: list[str]) -> list[str]:
    """Remove camera/video arguments for a true diagnostic worker request."""

    result: list[str] = []
    index = 0
    while index < len(command):
        value = command[index]
        if value == "--render":
            index += 1
            continue
        if value == "--video":
            index += 2
            continue
        result.append(value)
        index += 1
    return result


def _putpot_worker_visit(
    task: dict[str, Any],
    record: dict[str, Any],
    *,
    output_root: Path,
    python: str,
    gear_repo: str,
    attempt_limit: int,
    repair_epoch: str,
    initial_program_spec_json: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run an interactive spec-revision queue in one no-camera Isaac worker."""

    from judo_isaaclab.putpot_queue import (
        static_worker_argv,
        submit_program_request,
        submit_shutdown,
    )
    from judo_isaaclab.putpot_runtime import read_jsonl

    pair_root = output_root / task["name"] / record["pair_id"]
    repair_root = pair_root / "semantic_repair"
    first_root = _next_attempt(pair_root)
    first_number = _attempt_number(first_root)
    source_keyframes = output_root / task["name"] / "source_keyframes.json"
    code_head = _git_head()
    epoch_root = repair_root / "repair_epochs" / repair_epoch
    epoch_root.mkdir(parents=True, exist_ok=False)
    request_path = epoch_root / "requests.jsonl"
    receipt_path = epoch_root / "worker_receipts.jsonl"
    with open(request_path, "x", encoding="utf-8"):
        pass
    prototype_command = _command(
        task,
        python=python,
        gear_repo=gear_repo,
        target=record["dataset"],
        mode="skill",
        output=first_root,
        source_keyframes=source_keyframes,
        direct_replay_result=None,
    )
    session_path = epoch_root / "interactive_session.json"
    session = {
        "schema_version": 1,
        "pair": record["pair_id"],
        "code_head": code_head,
        "repair_root": str(repair_root.resolve()),
        "epoch_root": str(epoch_root.resolve()),
        "repair_epoch": repair_epoch,
        "first_lifetime_attempt": first_number,
        "attempt_limit": attempt_limit,
        "request_jsonl": str(request_path.resolve()),
        "receipt_jsonl": str(receipt_path.resolve()),
        "static_argv": static_worker_argv(
            _without_render(prototype_command)[2:]
        ),
    }
    _atomic_json(session_path, session)
    worker_command = [
        python,
        str((REPO_ROOT / "examples/run_putpot_persistent_worker.py").resolve()),
        "--request-jsonl",
        str(request_path),
        "--receipt-jsonl",
        str(receipt_path),
    ]
    default_spec = (
        Path(initial_program_spec_json)
        if initial_program_spec_json is not None
        else REPO_ROOT / "configs/putpot_semantic_program_v4.json"
    )
    attempts: list[dict[str, Any]] = []
    consumed_receipts = 0
    worker_returncode = None
    with open(epoch_root / "worker.log", "x", encoding="utf-8") as stream:
        worker = subprocess.Popen(
            worker_command,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        submit_program_request(session_path, default_spec)
        render_receipt = None
        while True:
            receipts = read_jsonl(receipt_path)
            while consumed_receipts < len(receipts):
                receipt = receipts[consumed_receipts]
                consumed_receipts += 1
                receipt_type = receipt.get("type")
                if receipt_type == "attempt":
                    requests = {
                        value["request_id"]: value
                        for value in read_jsonl(request_path)
                        if value.get("type") == "attempt"
                    }
                    request = requests[receipt["request_id"]]
                    result_path = Path(request["result_json"])
                    attempts.append(
                        {
                            "attempt": result_path.parent.name,
                            "code_head": receipt["code_head"],
                            "started_at": receipt["started_at"],
                            "finished_at": receipt["finished_at"],
                            "returncode": 0 if receipt["error"] is None else 1,
                            "command": worker_command,
                            "worker_request": request,
                            "worker_receipt": receipt,
                            "result": str(result_path.resolve()),
                            "video": None,
                            "runtime_receipt": str(
                                (result_path.parent / "skill_runtime.json").resolve()
                            ),
                            "status": (
                                "render_candidate"
                                if receipt["render_recommendation"]
                                else "diagnostic_failed"
                            ),
                            "lifetime_attempt": request["lifetime_attempt"],
                            "repair_epoch": repair_epoch,
                            "repair_epoch_attempt": request[
                                "repair_epoch_attempt"
                            ],
                            "repair_epoch_attempt_limit": attempt_limit,
                            "program_spec": receipt["program_spec"],
                        }
                    )
                    print(
                        "PUTPOT_INTERACTIVE_ACK="
                        + json.dumps(
                            {
                                "pair": record["pair_id"],
                                "session_json": str(session_path.resolve()),
                                "request_id": receipt["request_id"],
                                "program_spec_sha256": receipt[
                                    "program_spec"
                                ]["sha256"],
                                "diagnostic_classification": receipt[
                                    "diagnostic_classification"
                                ],
                                "render_recommendation": receipt[
                                    "render_recommendation"
                                ],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    if receipt["render_recommendation"] is not None:
                        render_receipt = receipt
                        submit_shutdown(
                            session_path, reason="separate_render_handoff"
                        )
                    elif len(attempts) >= attempt_limit:
                        submit_shutdown(
                            session_path, reason="repair_cycle_limit"
                        )
                    else:
                        print(
                            "PUTPOT_INTERACTIVE_REPAIR_REQUIRED="
                            + json.dumps(
                                {
                                    "pair": record["pair_id"],
                                    "session_json": str(session_path.resolve()),
                                    "prior_result": receipt["result_json"],
                                    "program_spec_sha256": receipt[
                                        "program_spec"
                                    ]["sha256"],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                elif receipt_type == "worker_boundary":
                    worker_returncode = worker.wait(timeout=120)
                    break
            if worker_returncode is not None:
                break
            polled = worker.poll()
            if polled is not None:
                worker_returncode = polled
                break
            time.sleep(0.1)

    if render_receipt is not None:
        render_epoch = repair_epoch + "-render"
        rendered = _run_pair(
            task,
            record,
            output_root=output_root,
            python=python,
            gear_repo=gear_repo,
            repair_epoch=render_epoch,
            repair_epoch_attempt=1,
            program_spec_json=render_receipt["program_spec"]["path"],
        )
        rendered["render_reason"] = render_receipt["render_recommendation"]
        rendered["diagnostic_repair_epoch"] = repair_epoch
        rendered["program_spec"] = render_receipt["program_spec"]
        attempts.append(rendered)
    if worker_returncode != 0 and not attempts:
        raise RuntimeError(
            f"persistent PutPot worker failed before producing receipts: {epoch_root}"
        )
    return attempts


def _record_completed_visit_attempts(
    config: dict[str, Any],
    output_root: Path,
    ledger_path: Path,
    ledger: dict[str, Any],
    key: str,
    visit_attempts: list[dict[str, Any]],
    visit_attempt_limit: int,
) -> dict[str, Any]:
    """Atomically record every completed receipt, even after acceptance is found.

    A PutPot worker may finish several diagnostics and a separate rendered run
    before the campaign refreshes the ledger.  Artifact discovery can therefore
    make the pair accepted while earlier completed receipts are still being
    appended.  Acceptance stops future physics, never receipt bookkeeping.
    """

    for visit_attempt, attempt in enumerate(visit_attempts, 1):
        attempt["asset_visit_attempt"] = visit_attempt
        attempt["asset_visit_attempt_limit"] = visit_attempt_limit
        ledger["pairs"][key]["attempts"].append(attempt)
        _atomic_json(ledger_path, ledger)
        print(
            "SEMANTIC_REPAIR_ATTEMPT="
            + json.dumps({"pair": key, **attempt}, sort_keys=True),
            flush=True,
        )
        ledger = refresh_ledger(config, output_root, ledger_path)
    return ledger


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
    parser.add_argument("--fresh-attempts-per-asset-visit", type=int, default=4)
    parser.add_argument("--putpot-initial-program-spec-json")
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
    visit_attempt_limit = _validate_fresh_attempts_per_asset_visit(
        args.fresh_attempts_per_asset_visit, selected_tasks
    )
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
        if record["task"] == "putpot":
            repair_epoch = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                + "-"
                + record["pair_id"]
            )
            visit_attempts = _putpot_worker_visit(
                task_map[record["task"]],
                record,
                output_root=output_root,
                python=args.python,
                gear_repo=args.gear_repo,
                attempt_limit=visit_attempt_limit,
                repair_epoch=repair_epoch,
                initial_program_spec_json=args.putpot_initial_program_spec_json,
            )
        else:
            visit_attempts = []
            for visit_attempt in range(1, visit_attempt_limit + 1):
                attempt = _run_pair(
                    task_map[record["task"]],
                    record,
                    output_root=output_root,
                    python=args.python,
                    gear_repo=args.gear_repo,
                )
                attempt["asset_visit_attempt"] = visit_attempt
                attempt["asset_visit_attempt_limit"] = visit_attempt_limit
                visit_attempts.append(attempt)
                if attempt["status"] == "accepted":
                    break
        ledger = _record_completed_visit_attempts(
            config,
            output_root,
            ledger_path,
            ledger,
            key,
            visit_attempts,
            visit_attempt_limit,
        )
        if _stop_after_attempt(args.stop_on_failure, ledger["pairs"][key]):
            print(f"SEMANTIC_REPAIR_STOP_ON_FAILURE={key}", flush=True)
            break

    summary = ledger["summary"]
    print(f"SEMANTIC_REPAIR_FINAL={summary['accepted']}/{summary['total']}", flush=True)
    if summary["accepted"] != summary["total"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
