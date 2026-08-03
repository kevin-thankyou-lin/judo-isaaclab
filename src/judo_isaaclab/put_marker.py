"""Deterministic Cartesian skills for ``PutMarkerInDrawer``.

The module is deliberately simulator-agnostic.  It turns a small set of
semantic poses into one continuous nominal trajectory; an IsaacLab runner can
track the trajectory with Jacobian IK without sampling candidate actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _pose(value: Any, name: str = "pose") -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {value.shape}")
    norm = np.linalg.norm(value[3:])
    if norm < 1.0e-8:
        raise ValueError(f"{name} has a zero quaternion")
    result = value.copy()
    result[3:] /= norm
    return result


def quaternion_multiply(left: Any, right: Any) -> np.ndarray:
    """Multiply scalar-first quaternions."""
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def quaternion_rotate(quaternion: Any, vector: Any) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    vector = np.asarray(vector, dtype=np.float64)
    twice_cross = 2.0 * np.cross(quaternion[1:], vector)
    return vector + quaternion[0] * twice_cross + np.cross(quaternion[1:], twice_cross)


def compose_pose(left: Any, right: Any) -> np.ndarray:
    left = _pose(left, "left")
    right = _pose(right, "right")
    result = np.empty(7, dtype=np.float64)
    result[:3] = left[:3] + quaternion_rotate(left[3:], right[:3])
    result[3:] = quaternion_multiply(left[3:], right[3:])
    result[3:] /= np.linalg.norm(result[3:])
    return result


def inverse_pose(value: Any) -> np.ndarray:
    value = _pose(value)
    result = np.empty(7, dtype=np.float64)
    result[3:] = value[3:] * np.asarray([1.0, -1.0, -1.0, -1.0])
    result[:3] = quaternion_rotate(result[3:], -value[:3])
    return result


def transfer_pose(
    value: Any,
    source_frame: Any,
    target_frame: Any,
    *,
    local_position_scale: Any = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Transfer a pose through corresponding semantic frames."""
    local = compose_pose(inverse_pose(source_frame), value)
    scale = np.asarray(local_position_scale, dtype=np.float64)
    if scale.shape != (3,) or np.any(scale <= 0.0):
        raise ValueError("local_position_scale must contain three positive values")
    local[:3] *= scale
    return compose_pose(target_frame, local)


def pose_from_matrix(matrix: Any) -> np.ndarray:
    """Convert a homogeneous matrix to ``xyz + wxyz``."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"matrix must have shape (4, 4), got {matrix.shape}")
    rotation = matrix[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[2, 1] - rotation[1, 2]) / s,
                    0.25 * s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                ]
            )
        elif axis == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[0, 2] - rotation[2, 0]) / s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    0.25 * s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                ]
            )
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[1, 0] - rotation[0, 1]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                    0.25 * s,
                ]
            )
    result = np.concatenate((matrix[:3, 3], quaternion))
    return _pose(result)


def _slerp(left: np.ndarray, right: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = left[None] + fraction[:, None] * (right - left)[None]
        return result / np.linalg.norm(result, axis=1, keepdims=True)
    angle = np.arccos(dot)
    denominator = np.sin(angle)
    return (
        np.sin((1.0 - fraction) * angle)[:, None] * left[None]
        + np.sin(fraction * angle)[:, None] * right[None]
    ) / denominator


def interpolate_poses(start: Any, target: Any, steps: int) -> np.ndarray:
    """Quintic Cartesian interpolation including the target, excluding the start."""
    if steps < 1:
        raise ValueError("steps must be positive")
    start = _pose(start, "start")
    target = _pose(target, "target")
    fraction = np.linspace(1.0 / steps, 1.0, steps)
    smooth = fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)
    result = np.empty((steps, 7), dtype=np.float64)
    result[:, :3] = start[:3] + smooth[:, None] * (target[:3] - start[:3])
    result[:, 3:] = _slerp(start[3:], target[3:], smooth)
    return result


@dataclass(frozen=True)
class DrawerGeometry:
    """Semantic geometry for one selected drawer in cabinet-local coordinates."""

    root_pose: np.ndarray
    slide_axis_local: np.ndarray
    joint_origin_local: np.ndarray
    handle_point_local: np.ndarray
    cavity_center_local: np.ndarray
    lower_limit_m: float
    upper_limit_m: float
    cavity_size: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_pose", _pose(self.root_pose, "root_pose"))
        for name in (
            "slide_axis_local",
            "joint_origin_local",
            "handle_point_local",
            "cavity_center_local",
            "cavity_size",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,)")
            object.__setattr__(self, name, value)
        axis = self.slide_axis_local / np.linalg.norm(self.slide_axis_local)
        object.__setattr__(self, "slide_axis_local", axis)
        if not self.upper_limit_m > self.lower_limit_m:
            raise ValueError("drawer upper limit must exceed lower limit")
        if np.any(self.cavity_size <= 0.0):
            raise ValueError("cavity dimensions must be positive")

    def drawer_frame(self, joint_position_m: float) -> np.ndarray:
        joint_position_m = float(np.clip(joint_position_m, self.lower_limit_m, self.upper_limit_m))
        local = np.asarray([*self.cavity_center_local, 1.0, 0.0, 0.0, 0.0])
        local[:3] += self.slide_axis_local * joint_position_m
        return compose_pose(self.root_pose, local)

    def handle_frame(self, joint_position_m: float) -> np.ndarray:
        local = np.asarray([*self.handle_point_local, 1.0, 0.0, 0.0, 0.0])
        local[:3] += self.slide_axis_local * joint_position_m
        return compose_pose(self.root_pose, local)

    def corresponding_joint_position(self, source: "DrawerGeometry", source_position_m: float) -> float:
        fraction = (source_position_m - source.lower_limit_m) / (
            source.upper_limit_m - source.lower_limit_m
        )
        return self.lower_limit_m + fraction * (self.upper_limit_m - self.lower_limit_m)

    def transfer_drawer_pose(
        self,
        source: "DrawerGeometry",
        value: Any,
        source_joint_position_m: float,
    ) -> np.ndarray:
        target_joint = self.corresponding_joint_position(source, source_joint_position_m)
        scale = self.cavity_size / source.cavity_size
        return transfer_pose(
            value,
            source.drawer_frame(source_joint_position_m),
            self.drawer_frame(target_joint),
            local_position_scale=scale,
        )

    def transfer_handle_pose(
        self,
        source: "DrawerGeometry",
        value: Any,
        source_joint_position_m: float,
    ) -> np.ndarray:
        target_joint = self.corresponding_joint_position(source, source_joint_position_m)
        return transfer_pose(
            value,
            source.handle_frame(source_joint_position_m),
            self.handle_frame(target_joint),
        )


def reanchor_marker_placement(
    trajectory: "SkillTrajectory",
    intended_collision_clear_marker_pose: Any,
    intended_cavity_marker_pose: Any,
    observed_marker_pose: Any,
    observed_left_pose: Any,
) -> "SkillTrajectory":
    """Preserve the observed grasp while following drawer-local marker poses."""

    required = (
        "drawer_open",
        "marker_collision_clear",
        "marker_cavity",
        "marker_release",
        "marker_settle",
        "left_withdraw",
    )
    missing = [name for name in required if name not in trajectory.waypoint_steps]
    if missing:
        raise ValueError(f"placement trajectory is missing waypoints: {missing}")
    steps = trajectory.waypoint_steps
    contact = compose_pose(inverse_pose(observed_marker_pose), observed_left_pose)
    corrected_clear = compose_pose(intended_collision_clear_marker_pose, contact)
    corrected_cavity = compose_pose(intended_cavity_marker_pose, contact)
    left = np.asarray(trajectory.left_poses, dtype=np.float64).copy()
    start = steps["drawer_open"] + 1
    clear_end = steps["marker_collision_clear"]
    cavity_end = steps["marker_cavity"]
    settle_end = steps["marker_settle"]
    withdraw_end = steps["left_withdraw"]
    left[start : clear_end + 1] = interpolate_poses(
        observed_left_pose, corrected_clear, clear_end - start + 1
    )
    left[clear_end + 1 : cavity_end + 1] = interpolate_poses(
        corrected_clear, corrected_cavity, cavity_end - clear_end
    )
    left[cavity_end + 1 : settle_end + 1] = corrected_cavity
    left[settle_end + 1 : withdraw_end + 1] = interpolate_poses(
        corrected_cavity,
        trajectory.left_poses[withdraw_end],
        withdraw_end - settle_end,
    )
    return SkillTrajectory(
        left_poses=left,
        right_poses=trajectory.right_poses.copy(),
        grippers=trajectory.grippers.copy(),
        stage_names=trajectory.stage_names,
        waypoint_steps=dict(trajectory.waypoint_steps),
    )


def center_marker_over_cavity(marker_pose: Any, cavity_frame: Any) -> np.ndarray:
    """Center a transferred marker pose on the measured drawer support region."""

    local = compose_pose(inverse_pose(cavity_frame), marker_pose)
    local[:2] = 0.0
    return compose_pose(cavity_frame, local)


def geometry_conditioned_drawer_open_position(
    geometry: DrawerGeometry,
    requested_m: float,
    *,
    coded_threshold_m: float = 0.05,
    threshold_margin_m: float = 0.025,
    limit_margin_m: float = 0.02,
) -> float:
    """Choose one deterministic working position from measured joint limits.

    The target remains safely inside the authored upper limit while maintaining
    a positive margin above the immutable coded opening threshold.
    """

    required = float(coded_threshold_m + threshold_margin_m)
    upper = float(geometry.upper_limit_m - limit_margin_m)
    if upper < required:
        raise ValueError(
            "drawer joint has no safe opening interval above the coded threshold"
        )
    return float(np.clip(max(float(requested_m), required), geometry.lower_limit_m, upper))


def lift_handle_pull_pose(
    pose: Any, cabinet_root_pose: Any, lift_m: float
) -> np.ndarray:
    """Bias the pull endpoint along the cabinet's semantic up axis.

    A small upward hook keeps closing fingers engaged with high drawer pulls
    as the arm approaches its inner-workspace limit.  Expressing the offset in
    the cabinet frame also preserves the meaning for rotated assets.
    """
    result = _pose(pose, "pose").copy()
    root = _pose(cabinet_root_pose, "cabinet_root_pose")
    if not np.isfinite(lift_m) or lift_m < 0.0:
        raise ValueError("lift_m must be finite and nonnegative")
    result[:3] += quaternion_rotate(root[3:], [0.0, 0.0, 1.0]) * float(lift_m)
    return result


@dataclass(frozen=True)
class SkillWaypoint:
    name: str
    stage: str
    steps: int
    left_pose: np.ndarray
    right_pose: np.ndarray
    left_gripper: float
    right_gripper: float

    def __post_init__(self) -> None:
        if not self.name or not self.stage or self.steps < 1:
            raise ValueError("waypoint name, stage, and positive steps are required")
        object.__setattr__(self, "left_pose", _pose(self.left_pose, "left_pose"))
        object.__setattr__(self, "right_pose", _pose(self.right_pose, "right_pose"))


@dataclass(frozen=True)
class SkillTrajectory:
    left_poses: np.ndarray
    right_poses: np.ndarray
    grippers: np.ndarray
    stage_names: tuple[str, ...]
    waypoint_steps: dict[str, int]

    @property
    def steps(self) -> int:
        return len(self.left_poses)


class PutMarkerSkillProgram:
    """Builder for one uninterrupted, programmed nominal rollout."""

    def __init__(
        self,
        left_start: Any,
        right_start: Any,
        *,
        left_gripper_open: float = -0.0475,
        right_gripper_open: float = -0.0475,
    ) -> None:
        self._left = _pose(left_start, "left_start")
        self._right = _pose(right_start, "right_start")
        self._left_gripper = float(left_gripper_open)
        self._right_gripper = float(right_gripper_open)
        self._initial_left = self._left.copy()
        self._initial_right = self._right.copy()
        self._initial_left_gripper = self._left_gripper
        self._initial_right_gripper = self._right_gripper
        self._waypoints: list[SkillWaypoint] = []

    def _append(
        self,
        name: str,
        stage: str,
        steps: int,
        *,
        left_pose: Any | None = None,
        right_pose: Any | None = None,
        left_gripper: float | None = None,
        right_gripper: float | None = None,
    ) -> None:
        if left_pose is not None:
            self._left = _pose(left_pose, "left_pose")
        if right_pose is not None:
            self._right = _pose(right_pose, "right_pose")
        if left_gripper is not None:
            self._left_gripper = float(left_gripper)
        if right_gripper is not None:
            self._right_gripper = float(right_gripper)
        self._waypoints.append(
            SkillWaypoint(
                name=name,
                stage=stage,
                steps=steps,
                left_pose=self._left,
                right_pose=self._right,
                left_gripper=self._left_gripper,
                right_gripper=self._right_gripper,
            )
        )

    def grasp_marker(
        self,
        pregrasp_pose: Any,
        grasp_pose: Any,
        lift_pose: Any,
        *,
        approach_steps: int,
        close_steps: int,
        lift_steps: int,
        closed: float = 0.0,
    ) -> None:
        self._append("marker_pregrasp", "grasp_marker", approach_steps, left_pose=pregrasp_pose)
        self._append(
            "marker_grasp",
            "grasp_marker",
            close_steps,
            left_pose=grasp_pose,
            left_gripper=closed,
        )
        self._append("marker_lift", "grasp_marker", lift_steps, left_pose=lift_pose)

    def open_drawer(
        self,
        hold_marker_pose: Any,
        handle_pregrasp_pose: Any,
        handle_grasp_pose: Any,
        handle_open_pose: Any,
        *,
        hold_steps: int,
        approach_steps: int,
        close_steps: int,
        pull_steps: int,
        closed: float = 0.0,
    ) -> None:
        self._append("marker_clear_hold", "open_drawer", hold_steps, left_pose=hold_marker_pose)
        self._append(
            "handle_pregrasp",
            "open_drawer",
            approach_steps,
            right_pose=handle_pregrasp_pose,
        )
        self._append(
            "handle_grasp",
            "open_drawer",
            close_steps,
            right_pose=handle_grasp_pose,
            right_gripper=closed,
        )
        self._append("drawer_open", "open_drawer", pull_steps, right_pose=handle_open_pose)

    def place_marker_in_drawer(
        self,
        collision_clear_pose: Any,
        cavity_pose: Any,
        *,
        transit_steps: int,
        lower_steps: int,
    ) -> None:
        self._append(
            "marker_collision_clear",
            "place_marker_in_drawer",
            transit_steps,
            left_pose=collision_clear_pose,
        )
        self._append(
            "marker_cavity",
            "place_marker_in_drawer",
            lower_steps,
            left_pose=cavity_pose,
        )

    def release_marker(
        self,
        release_pose: Any,
        withdraw_pose: Any,
        *,
        release_steps: int,
        settle_steps: int,
        withdraw_steps: int,
        opened: float = -0.0475,
    ) -> None:
        self._append(
            "marker_release",
            "release_marker",
            release_steps,
            left_pose=release_pose,
            left_gripper=opened,
        )
        self._append("marker_settle", "release_marker", settle_steps)
        self._append(
            "left_withdraw",
            "release_marker",
            withdraw_steps,
            left_pose=withdraw_pose,
        )

    def close_drawer(
        self,
        handle_closed_pose: Any,
        right_withdraw_pose: Any,
        *,
        push_steps: int,
        release_steps: int,
        opened: float = -0.0475,
    ) -> None:
        self._append("drawer_closed", "close_drawer", push_steps, right_pose=handle_closed_pose)
        self._append(
            "handle_release",
            "close_drawer",
            release_steps,
            right_pose=right_withdraw_pose,
            right_gripper=opened,
        )

    def build(self) -> SkillTrajectory:
        if not self._waypoints:
            raise ValueError("skill program has no waypoints")
        left_start = self._initial_left
        right_start = self._initial_right
        left_parts = []
        right_parts = []
        gripper_parts = []
        stages: list[str] = []
        waypoint_steps: dict[str, int] = {}
        cursor = 0
        left = left_start
        right = right_start
        left_gripper = self._initial_left_gripper
        right_gripper = self._initial_right_gripper
        for waypoint in self._waypoints:
            left_parts.append(interpolate_poses(left, waypoint.left_pose, waypoint.steps))
            right_parts.append(interpolate_poses(right, waypoint.right_pose, waypoint.steps))
            alpha = np.linspace(1.0 / waypoint.steps, 1.0, waypoint.steps)
            smooth = alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)
            grippers = np.empty((waypoint.steps, 2), dtype=np.float64)
            grippers[:, 0] = left_gripper + smooth * (waypoint.left_gripper - left_gripper)
            grippers[:, 1] = right_gripper + smooth * (waypoint.right_gripper - right_gripper)
            gripper_parts.append(grippers)
            stages.extend([waypoint.stage] * waypoint.steps)
            cursor += waypoint.steps
            waypoint_steps[waypoint.name] = cursor - 1
            left = waypoint.left_pose
            right = waypoint.right_pose
            left_gripper = waypoint.left_gripper
            right_gripper = waypoint.right_gripper
        return SkillTrajectory(
            left_poses=np.concatenate(left_parts),
            right_poses=np.concatenate(right_parts),
            grippers=np.concatenate(gripper_parts),
            stage_names=tuple(stages),
            waypoint_steps=waypoint_steps,
        )
