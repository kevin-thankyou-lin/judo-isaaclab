"""Persistent PutPot worker contracts that do not depend on Isaac imports."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator, Mapping


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


def timing_accounting(
    attempt_wall_time_s: float, phase_timings_s: Mapping[str, float]
) -> dict[str, float]:
    """Expose named-phase coverage without absorbing uninstrumented overhead."""

    wall = float(attempt_wall_time_s)
    if wall < 0.0:
        raise ValueError("attempt wall time must be nonnegative")
    named_phase_sum = sum(float(phase_timings_s.get(name, 0.0)) for name in PHASE_NAMES)
    return {
        "attempt_wall_time_s": wall,
        "named_phase_sum_s": named_phase_sum,
        # Do not clamp this value: a negative result is evidence of overlapping
        # phase timers and must remain visible instead of being hidden.
        "unattributed_time_s": wall - named_phase_sum,
    }


def instantiated_scene_sensor_inventory(scene: Any) -> dict[str, Any]:
    """Inventory camera sensor *instances*, independent of scene config warnings."""

    sensors = getattr(scene, "sensors", None)
    if sensors is None:
        sensors = getattr(scene, "_sensors", None)
    if not isinstance(sensors, Mapping):
        raise TypeError("Isaac scene does not expose an instantiated sensor mapping")

    def is_camera_sensor(sensor: Any) -> bool:
        candidates = [type(sensor)]
        cfg = getattr(sensor, "cfg", None)
        if cfg is not None:
            candidates.append(type(cfg))
        for candidate in candidates:
            for base in candidate.__mro__:
                qualified = f"{base.__module__}.{base.__name__}".lower()
                if "camera" in qualified:
                    return True
        return False

    all_names = sorted(str(name) for name in sensors)
    camera_names = sorted(
        str(name) for name, sensor in sensors.items() if is_camera_sensor(sensor)
    )
    return {
        "instantiated_scene_sensor_names": all_names,
        "instantiated_scene_sensor_count": len(all_names),
        "instantiated_scene_camera_sensor_names": camera_names,
        "instantiated_scene_camera_sensor_count": len(camera_names),
    }


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
        if not 1 <= self.repair_epoch_attempt_limit <= 4:
            raise ValueError("PutPot repair epochs are capped at four cycles")
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


def append_jsonl(path: str | os.PathLike[str], value: Mapping[str, Any]) -> None:
    """Durably append one complete JSON object to an interactive queue."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read only complete newline-terminated objects from an append-only queue."""

    target = Path(path)
    if not target.is_file():
        return []
    text = target.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    result: list[dict[str, Any]] = []
    for line in lines:
        if not line.endswith("\n") or not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("interactive PutPot queue entries must be JSON objects")
        result.append(value)
    return result


def validate_same_spec_retry(
    previous_receipt: Mapping[str, Any] | None,
    program_spec_sha256: str,
    ambiguity_reason: str | None,
) -> None:
    """Fail closed on identical-spec retries without proven ambiguity."""

    if previous_receipt is None:
        return
    previous_spec = previous_receipt.get("program_spec", {})
    previous_hash = (
        previous_spec.get("sha256") if isinstance(previous_spec, Mapping) else None
    )
    if previous_hash != program_spec_sha256:
        return
    classification = previous_receipt.get("diagnostic_classification")
    if classification != "ambiguous_failure" or not (
        isinstance(ambiguity_reason, str) and ambiguity_reason.strip()
    ):
        raise ValueError(
            "same-spec PutPot retry requires prior ambiguous_failure receipt "
            "and an explicit ambiguity reason"
        )


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
