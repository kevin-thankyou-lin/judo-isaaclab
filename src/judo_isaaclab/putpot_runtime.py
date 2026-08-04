"""Persistent PutPot worker contracts that do not depend on Isaac imports."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any, Iterator


PHASE_NAMES = (
    "app_startup",
    "asset_env_load",
    "reset",
    "trajectory_build",
    "rollout",
    "render_encode",
    "trace_demo",
    "validation_decode_hash",
    "shutdown",
)


@dataclass
class PhaseTimers:
    """Accumulate explicit wall-clock phase durations for one attempt."""

    seconds: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in PHASE_NAMES}
    )

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if name not in self.seconds:
            raise ValueError(f"unknown PutPot timing phase: {name}")
        started = time.monotonic()
        try:
            yield
        finally:
            self.seconds[name] += time.monotonic() - started

    def add(self, name: str, seconds: float) -> None:
        if name not in self.seconds:
            raise ValueError(f"unknown PutPot timing phase: {name}")
        if seconds < 0.0:
            raise ValueError("phase duration must be nonnegative")
        self.seconds[name] += float(seconds)

    def receipt(self) -> dict[str, float]:
        return {name: float(self.seconds[name]) for name in PHASE_NAMES}


@dataclass(frozen=True)
class AttemptIdentity:
    """Separate immutable artifact numbering from one repair epoch visit."""

    lifetime_attempt: int
    repair_epoch: str
    repair_epoch_attempt: int
    repair_epoch_attempt_limit: int = 4

    def __post_init__(self) -> None:
        if self.lifetime_attempt < 1:
            raise ValueError("lifetime attempt must be positive")
        if not self.repair_epoch:
            raise ValueError("repair epoch must be nonempty")
        if self.repair_epoch_attempt < 1:
            raise ValueError("repair epoch attempt must be positive")
        if self.repair_epoch_attempt_limit != 4:
            raise ValueError("PutPot repair epochs must contain exactly four attempts")
        if self.repair_epoch_attempt > self.repair_epoch_attempt_limit:
            raise ValueError("repair epoch attempt exceeds the four-attempt limit")

    def receipt(self) -> dict[str, Any]:
        return {
            "lifetime_attempt": self.lifetime_attempt,
            "repair_epoch": self.repair_epoch,
            "repair_epoch_attempt": self.repair_epoch_attempt,
            "repair_epoch_attempt_limit": self.repair_epoch_attempt_limit,
        }


def ensure_fresh_output_paths(paths: list[str | os.PathLike[str] | None]) -> None:
    """Fail closed instead of overwriting any numbered attempt artifact."""

    existing = [str(Path(path)) for path in paths if path and Path(path).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite attempt artifacts: {existing}")


def diagnostic_classification(
    result: dict[str, Any] | None, error: str | None = None
) -> str:
    """Classify a camera-free attempt from result and exception provenance."""

    deterministic_prefixes = (
        "AssertionError:",
        "FileNotFoundError:",
        "KeyError:",
        "TypeError:",
        "ValueError:",
    )
    if error and error.startswith(deterministic_prefixes):
        return "deterministic_controller_or_config_exception"

    if not result:
        return "ambiguous_failure"
    checks = result.get("checks")
    if not isinstance(checks, dict):
        return "ambiguous_failure"
    render_only = {"h264_nonempty", "fully_decodable"}
    required = {key: value for key, value in checks.items() if key not in render_only}
    if required and all(value is True for value in required.values()):
        return "final_acceptance_candidate"
    if not result.get("stage_success_trace") or not result.get("provenance", {}).get(
        "trace"
    ):
        return "ambiguous_failure"
    return "diagnosed_physics_failure"


def render_recommendation(
    result: dict[str, Any] | None, error: str | None = None
) -> str | None:
    """Select only ambiguous failures and strict non-video candidates to render."""

    classification = diagnostic_classification(result, error)
    if classification in {"ambiguous_failure", "final_acceptance_candidate"}:
        return classification
    return None


def full_render_required_for_merge(result: dict[str, Any]) -> bool:
    """Require actual decoded camera evidence on every mergeable result."""

    video = result.get("video")
    checks = result.get("checks", {})
    return bool(
        video
        and checks.get("h264_nonempty") is True
        and checks.get("fully_decodable") is True
    )


@contextmanager
def without_scene_camera_sensors(scene_cfg_type: type[Any]) -> Iterator[None]:
    """Temporarily make a Gear scene builder omit every camera sensor config.

    Gear builds scene sensors independently from observation terms.  AppLauncher
    can therefore disable cameras only after this environment-level policy has
    replaced the freshly rebuilt sensor configs with ``None``.  The class method
    is restored immediately after environment construction; the constructed
    scene keeps its camera-free config.
    """

    original = scene_cfg_type.build_from_spec

    def build_without_cameras(scene: Any, spec: Any) -> Any:
        result = original(scene, spec)
        for name in spec.camera_names:
            setattr(scene, f"{name}_camera", None)
        return result

    scene_cfg_type.build_from_spec = build_without_cameras
    try:
        yield
    finally:
        scene_cfg_type.build_from_spec = original
