"""Deterministic semantic frames and primitives for ``PutPotOnCooktop``.

The module has no IsaacLab dependency.  It represents the pot bottom and
cooktop top as corresponding support frames and builds one continuous nominal
bimanual rollout from sparse, simulator-backed end-effector keyframes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .put_marker import (
    SkillTrajectory,
    SkillWaypoint,
    _pose,
    compose_pose,
    interpolate_poses,
    quaternion_rotate,
    transfer_pose,
)


CENTERED_ON_COOKTOP_TOLERANCE_M = 0.03


def cooktop_center_error_m(pot_pose: Any, cooktop_pose: Any) -> float:
    """Return planar root-center error for the pot and cooktop."""
    pot = _pose(pot_pose, "pot_pose")
    cooktop = _pose(cooktop_pose, "cooktop_pose")
    return float(np.linalg.norm(pot[:2] - cooktop[:2]))


def reanchor_centered_support(
    trajectory: SkillTrajectory,
    center_correction_xy: Any,
    observed_left_pose: Any,
    observed_right_pose: Any,
) -> SkillTrajectory:
    """Apply observed pot-to-cooktop center error to the support path only."""
    correction = np.asarray(center_correction_xy, dtype=np.float64)
    if correction.shape != (2,) or not np.all(np.isfinite(correction)):
        raise ValueError("center_correction_xy must contain two finite values")
    required = (
        "support_align",
        "support_lower",
        "pot_unload",
        "pot_release",
        "bimanual_withdraw",
    )
    missing = [name for name in required if name not in trajectory.waypoint_steps]
    if missing:
        raise ValueError(f"support trajectory is missing waypoints: {missing}")
    steps = trajectory.waypoint_steps
    start = steps["support_align"] + 1
    lower_end = steps["support_lower"]
    release_end = steps["pot_release"]
    withdraw_end = steps["bimanual_withdraw"]
    left = np.asarray(trajectory.left_poses, dtype=np.float64).copy()
    right = np.asarray(trajectory.right_poses, dtype=np.float64).copy()
    left_lower = left[lower_end].copy()
    right_lower = right[lower_end].copy()
    left_lower[:2] += correction
    right_lower[:2] += correction
    left[start : lower_end + 1] = interpolate_poses(
        observed_left_pose, left_lower, lower_end - start + 1
    )
    right[start : lower_end + 1] = interpolate_poses(
        observed_right_pose, right_lower, lower_end - start + 1
    )
    left[lower_end + 1 : release_end + 1] = left_lower
    right[lower_end + 1 : release_end + 1] = right_lower
    left[release_end + 1 : withdraw_end + 1] = interpolate_poses(
        left_lower,
        trajectory.left_poses[withdraw_end],
        withdraw_end - release_end,
    )
    right[release_end + 1 : withdraw_end + 1] = interpolate_poses(
        right_lower,
        trajectory.right_poses[withdraw_end],
        withdraw_end - release_end,
    )
    return SkillTrajectory(
        left_poses=left,
        right_poses=right,
        grippers=trajectory.grippers.copy(),
        stage_names=trajectory.stage_names,
        waypoint_steps=dict(trajectory.waypoint_steps),
    )


def reanchor_centered_unload(
    trajectory: SkillTrajectory,
    center_correction_xy: Any,
    observed_left_pose: Any,
    observed_right_pose: Any,
) -> SkillTrajectory:
    """Correct residual center error while both handles remain grasped."""
    correction = np.asarray(center_correction_xy, dtype=np.float64)
    if correction.shape != (2,) or not np.all(np.isfinite(correction)):
        raise ValueError("center_correction_xy must contain two finite values")
    steps = trajectory.waypoint_steps
    required = ("support_lower", "pot_unload", "pot_release", "bimanual_withdraw")
    missing = [name for name in required if name not in steps]
    if missing:
        raise ValueError(f"unload trajectory is missing waypoints: {missing}")
    start = steps["support_lower"] + 1
    unload_end = steps["pot_unload"]
    release_end = steps["pot_release"]
    withdraw_end = steps["bimanual_withdraw"]
    left = np.asarray(trajectory.left_poses, dtype=np.float64).copy()
    right = np.asarray(trajectory.right_poses, dtype=np.float64).copy()
    left_unload = left[unload_end].copy()
    right_unload = right[unload_end].copy()
    left_unload[:2] += correction
    right_unload[:2] += correction
    left[start : unload_end + 1] = interpolate_poses(
        observed_left_pose, left_unload, unload_end - start + 1
    )
    right[start : unload_end + 1] = interpolate_poses(
        observed_right_pose, right_unload, unload_end - start + 1
    )
    left[unload_end + 1 : release_end + 1] = left_unload
    right[unload_end + 1 : release_end + 1] = right_unload
    left[release_end + 1 : withdraw_end + 1] = interpolate_poses(
        left_unload,
        trajectory.left_poses[withdraw_end],
        withdraw_end - release_end,
    )
    right[release_end + 1 : withdraw_end + 1] = interpolate_poses(
        right_unload,
        trajectory.right_poses[withdraw_end],
        withdraw_end - release_end,
    )
    return SkillTrajectory(
        left_poses=left,
        right_poses=right,
        grippers=trajectory.grippers.copy(),
        stage_names=trajectory.stage_names,
        waypoint_steps=dict(trajectory.waypoint_steps),
    )


def reanchor_centered_release(
    trajectory: SkillTrajectory,
    center_correction_xy: Any,
    observed_left_pose: Any,
    observed_right_pose: Any,
) -> SkillTrajectory:
    """Use the final measured center residual during continuous release."""
    correction = np.asarray(center_correction_xy, dtype=np.float64)
    if correction.shape != (2,) or not np.all(np.isfinite(correction)):
        raise ValueError("center_correction_xy must contain two finite values")
    steps = trajectory.waypoint_steps
    required = ("pot_unload", "pot_release", "bimanual_withdraw")
    missing = [name for name in required if name not in steps]
    if missing:
        raise ValueError(f"release trajectory is missing waypoints: {missing}")
    start = steps["pot_unload"] + 1
    release_end = steps["pot_release"]
    withdraw_end = steps["bimanual_withdraw"]
    left = np.asarray(trajectory.left_poses, dtype=np.float64).copy()
    right = np.asarray(trajectory.right_poses, dtype=np.float64).copy()
    left_release = left[release_end].copy()
    right_release = right[release_end].copy()
    left_release[:2] += correction
    right_release[:2] += correction
    left[start : release_end + 1] = interpolate_poses(
        observed_left_pose, left_release, release_end - start + 1
    )
    right[start : release_end + 1] = interpolate_poses(
        observed_right_pose, right_release, release_end - start + 1
    )
    left[release_end + 1 : withdraw_end + 1] = interpolate_poses(
        left_release,
        trajectory.left_poses[withdraw_end],
        withdraw_end - release_end,
    )
    right[release_end + 1 : withdraw_end + 1] = interpolate_poses(
        right_release,
        trajectory.right_poses[withdraw_end],
        withdraw_end - release_end,
    )
    return SkillTrajectory(
        left_poses=left,
        right_poses=right,
        grippers=trajectory.grippers.copy(),
        stage_names=trajectory.stage_names,
        waypoint_steps=dict(trajectory.waypoint_steps),
    )


@dataclass(frozen=True)
class RigidSupportGeometry:
    """Root pose, axis-aligned local size, and semantic horizontal supports."""

    root_pose: np.ndarray
    size: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_pose", _pose(self.root_pose, "root_pose"))
        size = np.asarray(self.size, dtype=np.float64)
        if size.shape != (3,) or np.any(size <= 0.0):
            raise ValueError("size must contain three positive values")
        object.__setattr__(self, "size", size)

    def frame_at_local_z(self, local_z: float) -> np.ndarray:
        local = np.asarray([0.0, 0.0, float(local_z), 1.0, 0.0, 0.0, 0.0])
        return compose_pose(self.root_pose, local)

    @property
    def bottom_frame(self) -> np.ndarray:
        """Pot-bottom support plane centered on the root's local -Z face."""
        return self.frame_at_local_z(-0.5 * self.size[2])

    @property
    def top_frame(self) -> np.ndarray:
        """Cooktop-top support plane centered on the root's local +Z face."""
        return self.frame_at_local_z(0.5 * self.size[2])

    def transfer_pose_from(
        self,
        source: "RigidSupportGeometry",
        value: Any,
        *,
        scale_local_position: bool = True,
    ) -> np.ndarray:
        scale = self.size / source.size if scale_local_position else np.ones(3)
        return transfer_pose(
            value,
            source.root_pose,
            self.root_pose,
            local_position_scale=scale,
        )


def support_aligned_pot_pose(
    pot: RigidSupportGeometry,
    cooktop: RigidSupportGeometry,
    *,
    xy_offset_local: Any = (0.0, 0.0),
    clearance_m: float = 0.0,
) -> np.ndarray:
    """Place the pot bottom on the cooktop top while preserving upright pot yaw."""
    offset = np.asarray(xy_offset_local, dtype=np.float64)
    if offset.shape != (2,):
        raise ValueError("xy_offset_local must have shape (2,)")
    if clearance_m < 0.0:
        raise ValueError("clearance_m must be nonnegative")
    result = pot.root_pose.copy()
    translated = quaternion_rotate(
        cooktop.root_pose[3:], np.asarray([offset[0], offset[1], 0.0])
    )
    result[:2] = cooktop.root_pose[:2] + translated[:2]
    result[2] = cooktop.top_frame[2] + 0.5 * pot.size[2] + float(clearance_m)
    return result


class PutPotSkillProgram:
    """Builder for one uninterrupted bimanual grasp/place/release rollout."""

    def __init__(
        self,
        left_start: Any,
        right_start: Any,
        *,
        opened: float = -0.0475,
    ) -> None:
        self._left = _pose(left_start, "left_start")
        self._right = _pose(right_start, "right_start")
        self._left_gripper = float(opened)
        self._right_gripper = float(opened)
        self._initial_left = self._left.copy()
        self._initial_right = self._right.copy()
        self._initial_grippers = (self._left_gripper, self._right_gripper)
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

    def bimanual_handle_grasp(
        self,
        left_pregrasp: Any,
        right_pregrasp: Any,
        left_grasp: Any,
        right_grasp: Any,
        *,
        approach_steps: int,
        left_close_steps: int,
        right_close_steps: int,
        closed: float = 0.0,
    ) -> None:
        self._append(
            "bimanual_pregrasp",
            "bimanual_handle_grasp",
            approach_steps,
            left_pose=left_pregrasp,
            right_pose=right_pregrasp,
        )
        self._append(
            "left_handle_grasp",
            "bimanual_handle_grasp",
            left_close_steps,
            left_pose=left_grasp,
            left_gripper=closed,
        )
        self._append(
            "right_handle_grasp",
            "bimanual_handle_grasp",
            right_close_steps,
            right_pose=right_grasp,
            right_gripper=closed,
        )

    def lift_and_transport(
        self,
        left_lift: Any,
        right_lift: Any,
        left_transport: Any,
        right_transport: Any,
        left_align: Any,
        right_align: Any,
        *,
        lift_steps: int,
        transport_steps: int,
        align_steps: int,
    ) -> None:
        self._append(
            "pot_lift",
            "lift_transport",
            lift_steps,
            left_pose=left_lift,
            right_pose=right_lift,
        )
        self._append(
            "pot_transport",
            "lift_transport",
            transport_steps,
            left_pose=left_transport,
            right_pose=right_transport,
        )
        self._append(
            "support_align",
            "support_alignment",
            align_steps,
            left_pose=left_align,
            right_pose=right_align,
        )

    def unload_release_and_settle(
        self,
        left_lower: Any,
        right_lower: Any,
        left_withdraw: Any,
        right_withdraw: Any,
        *,
        lower_steps: int,
        unload_steps: int,
        release_steps: int,
        withdraw_steps: int,
        settle_steps: int,
        opened: float = -0.0475,
    ) -> None:
        self._append(
            "support_lower",
            "support_alignment",
            lower_steps,
            left_pose=left_lower,
            right_pose=right_lower,
        )
        self._append("pot_unload", "unload_release", unload_steps)
        self._append(
            "pot_release",
            "unload_release",
            release_steps,
            left_gripper=opened,
            right_gripper=opened,
        )
        self._append(
            "bimanual_withdraw",
            "stable_settle",
            withdraw_steps,
            left_pose=left_withdraw,
            right_pose=right_withdraw,
        )
        self._append("stable_settle", "stable_settle", settle_steps)

    def build(self) -> SkillTrajectory:
        if not self._waypoints:
            raise ValueError("skill program has no waypoints")
        left = self._initial_left
        right = self._initial_right
        left_gripper, right_gripper = self._initial_grippers
        left_parts = []
        right_parts = []
        gripper_parts = []
        stage_names: list[str] = []
        waypoint_steps: dict[str, int] = {}
        cursor = 0
        for waypoint in self._waypoints:
            left_parts.append(interpolate_poses(left, waypoint.left_pose, waypoint.steps))
            right_parts.append(interpolate_poses(right, waypoint.right_pose, waypoint.steps))
            fraction = np.linspace(1.0 / waypoint.steps, 1.0, waypoint.steps)
            smooth = fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)
            grippers = np.empty((waypoint.steps, 2), dtype=np.float64)
            grippers[:, 0] = left_gripper + smooth * (waypoint.left_gripper - left_gripper)
            grippers[:, 1] = right_gripper + smooth * (waypoint.right_gripper - right_gripper)
            gripper_parts.append(grippers)
            stage_names.extend([waypoint.stage] * waypoint.steps)
            cursor += waypoint.steps
            waypoint_steps[waypoint.name] = cursor - 1
            left, right = waypoint.left_pose, waypoint.right_pose
            left_gripper, right_gripper = waypoint.left_gripper, waypoint.right_gripper
        return SkillTrajectory(
            left_poses=np.concatenate(left_parts),
            right_poses=np.concatenate(right_parts),
            grippers=np.concatenate(gripper_parts),
            stage_names=tuple(stage_names),
            waypoint_steps=waypoint_steps,
        )
