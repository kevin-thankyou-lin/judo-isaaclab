"""Versioned task-adaptation bundles and simulator trial evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StageSpec:
    """One executable subtask in a continuous task strategy."""

    name: str
    target_name: str
    start_state: int
    target_state: int
    horizon: int
    parameters: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("stage name must be nonempty")
        if self.start_state < 0 or self.horizon < 1:
            raise ValueError(f"invalid stage interval for {self.name}")
        if not (
            self.start_state
            < self.target_state
            <= self.start_state + self.horizon
        ):
            raise ValueError(
                f"target_state for {self.name} must lie inside its horizon"
            )


@dataclass(frozen=True)
class TaskAdaptationBundle:
    """Immutable inputs plus an editable semantic strategy."""

    task_name: str
    dataset: str
    episode: str
    objects_root: str
    source_assets: dict[str, str]
    target_assets: dict[str, str]
    success_check: str
    checkpoint_state: int
    correspondences: dict[str, Any]
    stages: tuple[StageSpec, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "TaskAdaptationBundle":
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream)
        stages = tuple(StageSpec(**stage) for stage in value.pop("stages"))
        bundle = cls(stages=stages, **value)
        bundle.validate()
        return bundle

    def validate(self) -> None:
        if not self.task_name or not self.success_check:
            raise ValueError("task_name and success_check are required")
        if self.checkpoint_state < 0:
            raise ValueError("checkpoint_state must be nonnegative")
        if set(self.source_assets) != set(self.target_assets):
            raise ValueError("source and target asset categories must match")
        if self.source_assets == self.target_assets:
            raise ValueError("target assets must differ from source assets")
        previous_end = None
        for stage in self.stages:
            stage.validate()
            if previous_end is not None and stage.start_state != previous_end:
                raise ValueError("stages must be physically contiguous")
            previous_end = stage.start_state + stage.horizon


@dataclass(frozen=True)
class TrialEvidence:
    """The small, stable contract consumed by the orchestration loop."""

    status: str
    reached: bool
    metrics: dict[str, Any]
    result_path: str
    controls_path: str

    @classmethod
    def from_result(
        cls,
        result_path: str | Path,
        controls_path: str | Path,
    ) -> "TrialEvidence":
        with open(result_path, encoding="utf-8") as stream:
            result = json.load(stream)
        best = result.get("repeat_evaluation", {}).get("best_sample", {})
        return cls(
            status=str(result.get("status", "unknown")),
            reached=bool(result.get("best_sample_reached_keyframe", False)),
            metrics=best,
            result_path=str(result_path),
            controls_path=str(controls_path),
        )


def corrected_insert_offset(
    current: list[float] | tuple[float, float, float],
    evidence: TrialEvidence,
    *,
    gain: float = 0.5,
    maximum_step: float = 0.01,
) -> list[float]:
    """Use signed branch-frame error to propose one bounded correction."""
    vector = evidence.metrics.get("keyframe_position_error_vector_m_mean")
    if vector is None:
        return list(current)
    return [
        float(value - max(-maximum_step, min(maximum_step, gain * error)))
        for value, error in zip(current, vector)
    ]


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float32,
    )


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.linalg.norm(quaternion)
    twice_cross = 2.0 * np.cross(quaternion[1:], vector)
    return (
        vector
        + quaternion[0] * twice_cross
        + np.cross(quaternion[1:], twice_cross)
    )


def _pose_inverse(pose: np.ndarray) -> np.ndarray:
    result = np.empty(7, dtype=np.float32)
    result[3:] = pose[3:] / np.linalg.norm(pose[3:])
    result[4:] *= -1.0
    result[:3] = _quat_rotate(result[3:], -pose[:3])
    return result


def _pose_compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(7, dtype=np.float32)
    result[:3] = left[:3] + _quat_rotate(left[3:], right[:3])
    result[3:] = _quat_multiply(left[3:], right[3:])
    result[3:] /= np.linalg.norm(result[3:])
    return result


def asset_relative_grasp_pose(
    source_eef_pose: Any,
    source_object_pose: Any,
    target_object_pose: Any,
    source_object_size: Any,
    target_object_size: Any,
    *,
    scale_limits: tuple[float, float] = (0.5, 2.0),
    scale_axes: tuple[bool, bool, bool] = (True, True, False),
    source_contact_pose: Any | None = None,
) -> np.ndarray:
    """Map a demonstrated EEF grasp into a live target-object frame.

    Asset instances share canonical object axes. The demonstrated local EEF
    orientation is preserved. When a demonstrated pinch/contact pose is
    supplied, its object-relative position scales with the asset while the
    contact-to-wrist transform remains rigid. The fallback scales only the
    wrist's lateral object offsets because its vertical component includes the
    fixed wrist-to-fingertip standoff. No source joint path is retained.
    """
    source_eef_pose = np.asarray(source_eef_pose, dtype=np.float32)
    source_object_pose = np.asarray(source_object_pose, dtype=np.float32)
    target_object_pose = np.asarray(target_object_pose, dtype=np.float32)
    source_object_size = np.asarray(source_object_size, dtype=np.float32)
    target_object_size = np.asarray(target_object_size, dtype=np.float32)
    for name, value, shape in (
        ("source_eef_pose", source_eef_pose, (7,)),
        ("source_object_pose", source_object_pose, (7,)),
        ("target_object_pose", target_object_pose, (7,)),
        ("source_object_size", source_object_size, (3,)),
        ("target_object_size", target_object_size, (3,)),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if np.any(source_object_size <= 0.0) or np.any(target_object_size <= 0.0):
        raise ValueError("object dimensions must be positive")
    minimum, maximum = scale_limits
    if not 0.0 < minimum <= maximum:
        raise ValueError("scale_limits must satisfy 0 < minimum <= maximum")
    if len(scale_axes) != 3:
        raise ValueError("scale_axes must contain three booleans")

    scale = np.clip(target_object_size / source_object_size, minimum, maximum)
    if source_contact_pose is not None:
        source_contact_pose = np.asarray(source_contact_pose, dtype=np.float32)
        if source_contact_pose.shape != (7,):
            raise ValueError(
                "source_contact_pose must have shape (7,), got "
                f"{source_contact_pose.shape}"
            )
        local_contact = _pose_compose(
            _pose_inverse(source_object_pose), source_contact_pose
        )
        local_contact[:3] *= scale
        target_contact = _pose_compose(target_object_pose, local_contact)
        contact_to_eef = _pose_compose(
            _pose_inverse(source_contact_pose), source_eef_pose
        )
        return _pose_compose(target_contact, contact_to_eef)

    local_eef = _pose_compose(_pose_inverse(source_object_pose), source_eef_pose)
    local_eef[:3] *= np.where(np.asarray(scale_axes, dtype=bool), scale, 1.0)
    return _pose_compose(target_object_pose, local_eef)
