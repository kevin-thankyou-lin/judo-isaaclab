"""Fail-closed evidence ledger for coding-agent asset adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable


PROOF_PHASES = (
    "target_replay",
    "target_skill",
    "final_render",
)

OPTIONAL_PHASES = ("source_replay", "source_skill")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {name: _expand_environment(item) for name, item in value.items()}
    return value


def _digest(path: str | Path) -> str:
    digest = sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for name in path.split("."):
        if not isinstance(current, dict) or name not in current:
            raise KeyError(path)
        current = current[name]
    return current


def _first_path(value: dict[str, Any], paths: Iterable[str], default: Any = None) -> Any:
    for path in paths:
        try:
            return _read_path(value, path)
        except KeyError:
            continue
    return default


@dataclass(frozen=True)
class EvidenceContract:
    """Task-specific schema interpreted by the generic harness."""

    task_name: str
    success_check: str
    source: dict[str, Any]
    target_search: dict[str, Any]
    stages: tuple[str, ...]
    result_paths: dict[str, Any]
    required_protocol_checks: tuple[str, ...] = ()
    thresholds: dict[str, float] = field(default_factory=dict)
    semantic_frames: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceContract":
        value = _expand_environment(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )
        value.pop("schema_version", None)
        value["stages"] = tuple(value["stages"])
        value["required_protocol_checks"] = tuple(
            value.get("required_protocol_checks", ())
        )
        contract = cls(**value)
        contract.validate()
        return contract

    def validate(self) -> None:
        if not self.task_name or not self.success_check:
            raise ValueError("task_name and success_check are required")
        if not self.stages or len(set(self.stages)) != len(self.stages):
            raise ValueError("stages must be a nonempty unique sequence")
        if "dataset" not in self.source or "assets" not in self.source:
            raise ValueError("source dataset and assets are required")
        if not self.target_search.get("require_replay_failure", False):
            raise ValueError("target_search must require replay failure")
        if not self.result_paths.get("task_success"):
            raise ValueError("result_paths.task_success is required")
        stage_paths = self.result_paths.get("stages", {})
        missing = [name for name in self.stages if not stage_paths.get(name)]
        if missing:
            raise ValueError(f"missing result paths for stages: {missing}")


@dataclass(frozen=True)
class AttemptEvaluation:
    accepted: bool
    task_success: bool
    stages: dict[str, bool]
    diagnosis: str
    recommendation: str
    metrics: dict[str, Any]
    checks: dict[str, bool]


def _diagnose(
    contract: EvidenceContract,
    phase: str,
    task_success: bool,
    stages: dict[str, bool],
    metrics: dict[str, Any],
    protocol_ok: bool,
) -> tuple[str, str]:
    if not protocol_ok:
        return (
            "infrastructure_or_protocol",
            "Repair missing artifacts/provenance/protocol checks before changing motion code.",
        )
    if phase == "target_replay" and task_success:
        return (
            "target_too_easy",
            "Reject this target and select one where source replay genuinely fails.",
        )
    if task_success:
        return "passed", "Promote this attempt without changing the skill."

    failed_stage = next((name for name in contract.stages if not stages[name]), None)
    if failed_stage is None:
        return (
            "finalization_or_stability",
            "Inspect terminal stability, release, and the final coded success predicate.",
        )
    if failed_stage == contract.stages[0]:
        return (
            "grasp_or_pick",
            "Inspect object-relative grasp frames, contact formation, and lift retention.",
        )

    tracking_error = metrics.get("eef_tracking_error_m")
    tracking_limit = contract.thresholds.get("eef_tracking_error_m", 0.02)
    if tracking_error is not None and float(tracking_error) > tracking_limit:
        return (
            "reachability",
            "Use target-relative Cartesian IK or a new waypoint; preserve other stages.",
        )
    if "open" in failed_stage or "insert" in failed_stage:
        return (
            "interaction_path_or_grasp_transform",
            "Preserve the contact transform and move along the target articulation/insertion axis.",
        )
    return (
        "placement_release_support",
        "Refine support-frame clearance, release timing, and the stable-support window.",
    )


def evaluate_result(
    contract: EvidenceContract,
    phase: str,
    result: dict[str, Any],
    *,
    returncode: int = 0,
    video_exists: bool = False,
) -> AttemptEvaluation:
    """Normalize one task result and evaluate its phase-specific gate."""
    if phase not in (*PROOF_PHASES, *OPTIONAL_PHASES):
        raise ValueError(f"unknown phase: {phase}")
    task_success = bool(
        _first_path(result, contract.result_paths["task_success"], False)
    )
    stages = {
        name: bool(
            _first_path(result, contract.result_paths["stages"][name], False)
        )
        for name in contract.stages
    }
    metrics = {
        name: _first_path(result, paths)
        for name, paths in contract.result_paths.get("metrics", {}).items()
    }
    checks = {
        name: bool(
            _first_path(
                result,
                (
                    f"checks.{name}",
                    f"acceptance_checks.{name}",
                ),
                False,
            )
        )
        for name in contract.required_protocol_checks
    }
    protocol_ok = returncode == 0 and all(checks.values())
    if phase == "final_render":
        protocol_ok = protocol_ok and video_exists

    expected_success = phase in {"source_skill", "target_skill", "final_render"}
    expected_failure = phase == "target_replay"
    accepted = protocol_ok and (
        (expected_success and task_success)
        or (expected_failure and not task_success)
        or phase == "source_replay"
    )
    diagnosis, recommendation = _diagnose(
        contract, phase, task_success, stages, metrics, protocol_ok
    )
    return AttemptEvaluation(
        accepted=accepted,
        task_success=task_success,
        stages=stages,
        diagnosis=diagnosis,
        recommendation=recommendation,
        metrics=metrics,
        checks=checks,
    )


class EvidenceLedger:
    """Append-only task evidence with atomic persistence and proof checks."""

    def __init__(self, path: str | Path, contract_path: str | Path):
        self.path = Path(path)
        self.contract_path = Path(contract_path).resolve()
        self.contract = EvidenceContract.load(self.contract_path)
        if self.path.exists():
            self.value = json.loads(self.path.read_text(encoding="utf-8"))
            if self.value["contract_sha256"] != _digest(self.contract_path):
                raise ValueError("contract changed after ledger creation")
        else:
            self.value = {
                "schema_version": 1,
                "status": "running",
                "task_name": self.contract.task_name,
                "contract": str(self.contract_path),
                "contract_sha256": _digest(self.contract_path),
                "attempts": [],
            }
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def add_attempt(
        self,
        *,
        phase: str,
        result_path: str | Path,
        log_path: str | Path,
        returncode: int,
        revision: str,
        source_id: str,
        target_id: str | None = None,
        trace_path: str | Path | None = None,
        video_path: str | Path | None = None,
        command: list[str] | None = None,
    ) -> dict[str, Any]:
        result_path = Path(result_path)
        log_path = Path(log_path)
        if not result_path.is_file() or not log_path.is_file():
            raise FileNotFoundError("result and log must exist before ingestion")
        trace = Path(trace_path) if trace_path else None
        video = Path(video_path) if video_path else None
        result = json.loads(result_path.read_text(encoding="utf-8"))
        evaluation = evaluate_result(
            self.contract,
            phase,
            result,
            returncode=returncode,
            video_exists=bool(video and video.is_file() and video.stat().st_size),
        )
        artifacts: dict[str, Any] = {
            "result": {"path": str(result_path), "sha256": _digest(result_path)},
            "log": {"path": str(log_path), "sha256": _digest(log_path)},
        }
        for name, path in (("trace", trace), ("video", video)):
            if path is not None:
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(f"{name} artifact is missing or empty")
                artifacts[name] = {"path": str(path), "sha256": _digest(path)}
        record = {
            "index": len(self.value["attempts"]),
            "phase": phase,
            "accepted": evaluation.accepted,
            "returncode": returncode,
            "revision": revision,
            "source_id": source_id,
            "target_id": target_id,
            "task_success": evaluation.task_success,
            "stages": evaluation.stages,
            "metrics": evaluation.metrics,
            "checks": evaluation.checks,
            "diagnosis": evaluation.diagnosis,
            "recommendation": evaluation.recommendation,
            "command": command or [],
            "artifacts": artifacts,
        }
        self.value["attempts"].append(record)
        proof = self.proof_status()
        self.value["status"] = "complete" if proof["complete"] else "running"
        self.save()
        return record

    def latest_accepted(self, phase: str) -> dict[str, Any] | None:
        return next(
            (
                attempt
                for attempt in reversed(self.value["attempts"])
                if attempt["phase"] == phase and attempt["accepted"]
            ),
            None,
        )

    def proof_status(self) -> dict[str, Any]:
        attempts = {phase: self.latest_accepted(phase) for phase in PROOF_PHASES}
        missing = [phase for phase, attempt in attempts.items() if attempt is None]
        if missing:
            return {"complete": False, "missing": missing}
        target_ids = {attempts[phase]["target_id"] for phase in PROOF_PHASES}
        source_ids = {attempts[phase]["source_id"] for phase in PROOF_PHASES}
        revisions = {
            attempts[phase]["revision"] for phase in ("target_skill", "final_render")
        }
        complete = len(target_ids) == len(source_ids) == len(revisions) == 1
        return {
            "complete": complete,
            "missing": [],
            "same_target": len(target_ids) == 1,
            "same_source": len(source_ids) == 1,
            "render_matches_skill_revision": len(revisions) == 1,
        }


def execute_attempt(
    ledger: EvidenceLedger,
    *,
    phase: str,
    command: list[str],
    result_path: str | Path,
    log_path: str | Path,
    revision: str,
    source_id: str,
    target_id: str | None = None,
    trace_path: str | Path | None = None,
    video_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run exactly one foreground simulator command and ingest its evidence."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("COMMAND=" + json.dumps(command) + "\n")
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    return ledger.add_attempt(
        phase=phase,
        result_path=result_path,
        log_path=log_path,
        returncode=completed.returncode,
        revision=revision,
        source_id=source_id,
        target_id=target_id,
        trace_path=trace_path,
        video_path=video_path,
        command=command,
    )
