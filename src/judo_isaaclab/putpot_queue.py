"""Append-only supervisor queue for interactive PutPot program revisions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any

from .putpot_program_spec import load_program_spec
from .putpot_controller_protocol import sha256_file
from .putpot_runtime import (
    append_jsonl,
    read_jsonl,
    validate_material_spec_revision,
    validate_same_spec_retry,
)


_DYNAMIC_VALUE_FLAGS = {
    "--video",
    "--trace-npz",
    "--result-json",
    "--demo-hdf5",
    "--runtime-receipt-json",
    "--lifetime-attempt-number",
    "--repair-epoch",
    "--repair-epoch-attempt",
    "--repair-epoch-attempt-limit",
    "--program-spec-json",
    "--controller-plugin-py",
    "--controller-plugin-sha256",
    "--controller-plugin-log",
}


def static_worker_argv(argv: list[str]) -> list[str]:
    """Remove per-attempt outputs, identity, and program-spec arguments."""

    result: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] in _DYNAMIC_VALUE_FLAGS:
            index += 2
            continue
        result.append(argv[index])
        index += 1
    return result


def _attempt_receipts(session: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        value
        for value in read_jsonl(session["receipt_jsonl"])
        if value.get("type") == "attempt"
    ]


def submit_program_request(
    session_json: str | os.PathLike[str],
    program_spec_json: str | os.PathLike[str],
    *,
    controller_plugin_py: str | os.PathLike[str] | None = None,
    ambiguity_reason: str | None = None,
) -> dict[str, Any]:
    """Append one immutable spec and reloadable Python controller revision."""

    session_path = Path(session_json)
    session = json.loads(session_path.read_text(encoding="utf-8"))
    queue = read_jsonl(session["request_jsonl"])
    queued_attempts = [value for value in queue if value.get("type") == "attempt"]
    receipts = _attempt_receipts(session)
    if len(queued_attempts) != len(receipts):
        raise RuntimeError("previous PutPot request has not been acknowledged")
    cycle = len(queued_attempts) + 1
    if cycle > int(session["attempt_limit"]):
        raise ValueError("PutPot diagnose-to-repair cycle limit exceeded")

    source_spec = load_program_spec(program_spec_json)
    previous = receipts[-1] if receipts else None
    selected_plugin = Path(
        controller_plugin_py
        if controller_plugin_py is not None
        else (
            previous["controller_plugin"]["path"]
            if previous is not None and previous.get("controller_plugin")
            else session["initial_controller_plugin_py"]
        )
    )
    source_plugin_sha256 = sha256_file(selected_plugin)
    previous_plugin_sha256 = (
        None
        if previous is None
        else previous.get("controller_plugin", {}).get("sha256")
    )
    plugin_changed = (
        previous_plugin_sha256 is not None
        and previous_plugin_sha256 != source_plugin_sha256
    )
    if not plugin_changed:
        validate_same_spec_retry(previous, source_spec.sha256, ambiguity_reason)
    if previous is None or previous["program_spec"]["sha256"] != source_spec.sha256:
        validate_material_spec_revision(
            previous,
            source_spec.sha256,
            source_spec.parameters,
        )

    lifetime = int(session["first_lifetime_attempt"]) + cycle - 1
    attempt_root = Path(session["repair_root"]) / f"attempt_{lifetime:03d}"
    attempt_root.mkdir(parents=True, exist_ok=False)
    spec_root = Path(session["epoch_root"]) / "program_specs"
    spec_root.mkdir(parents=True, exist_ok=True)
    immutable_spec = spec_root / f"spec_{cycle:03d}.json"
    with open(program_spec_json, "rb") as source, open(immutable_spec, "xb") as target:
        shutil.copyfileobj(source, target)
        target.flush()
        os.fsync(target.fileno())
    spec = load_program_spec(immutable_spec)
    if spec.sha256 != source_spec.sha256:
        raise RuntimeError("immutable PutPot program-spec copy hash changed")
    plugin_root = Path(session["epoch_root"]) / "controller_plugins"
    plugin_root.mkdir(parents=True, exist_ok=True)
    immutable_plugin = plugin_root / f"controller_{cycle:03d}.py"
    with open(selected_plugin, "rb") as source, open(immutable_plugin, "xb") as target:
        shutil.copyfileobj(source, target)
        target.flush()
        os.fsync(target.fileno())
    plugin_sha256 = sha256_file(immutable_plugin)
    if plugin_sha256 != source_plugin_sha256:
        raise RuntimeError("immutable PutPot controller-plugin hash changed")

    argv = list(session["static_argv"])
    video_path = None
    if "--render" in argv:
        video_path = str(attempt_root / "skill.mp4")
        argv.extend(["--video", video_path])
    argv.extend(
        [
            "--trace-npz",
            str(attempt_root / "skill_trace.npz"),
            "--result-json",
            str(attempt_root / "skill_result.json"),
            "--demo-hdf5",
            str(attempt_root / "skill_demo.hdf5"),
            "--runtime-receipt-json",
            str(attempt_root / "skill_runtime.json"),
            "--lifetime-attempt-number",
            str(lifetime),
            "--repair-epoch",
            session["repair_epoch"],
            "--repair-epoch-attempt",
            str(cycle),
            "--repair-epoch-attempt-limit",
            str(session["attempt_limit"]),
            "--program-spec-json",
            str(immutable_spec.resolve()),
            "--controller-plugin-py",
            str(immutable_plugin.resolve()),
            "--controller-plugin-sha256",
            plugin_sha256,
            "--controller-plugin-log",
            str(attempt_root / "controller_plugin.log"),
        ]
    )
    request = {
        "type": "attempt",
        "request_id": f"{session['repair_epoch']}:{cycle}",
        "pair": session["pair"],
        "code_head": session["code_head"],
        "argv": argv,
        "log": str(attempt_root / "skill.log"),
        "result_json": str(attempt_root / "skill_result.json"),
        "video": video_path,
        "lifetime_attempt": lifetime,
        "repair_epoch": session["repair_epoch"],
        "repair_epoch_attempt": cycle,
        "program_spec_json": str(immutable_spec.resolve()),
        "program_spec_sha256": spec.sha256,
        "controller_plugin_py": str(immutable_plugin.resolve()),
        "controller_plugin_sha256": plugin_sha256,
        "ambiguity_reason": ambiguity_reason,
    }
    append_jsonl(session["request_jsonl"], request)
    return request


def submit_shutdown(
    session_json: str | os.PathLike[str], *, reason: str
) -> dict[str, Any]:
    """Ask the waiting worker to close Isaac at an acknowledged boundary."""

    if not reason.strip():
        raise ValueError("shutdown reason must be nonempty")
    session = json.loads(Path(session_json).read_text(encoding="utf-8"))
    request = {
        "type": "shutdown",
        "request_id": f"{session['repair_epoch']}:shutdown",
        "reason": reason,
    }
    append_jsonl(session["request_jsonl"], request)
    return request
