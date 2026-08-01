"""Deterministic semantic frames and primitives for ``HangMugOnTree``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .put_marker import (
    SkillTrajectory,
    SkillWaypoint,
    _pose,
    interpolate_poses,
    transfer_pose,
)


@dataclass(frozen=True)
class RigidAssetGeometry:
    """Root pose and local axis-aligned size for a rigid task asset."""

    root_pose: np.ndarray
    size: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_pose", _pose(self.root_pose, "root_pose"))
        size = np.asarray(self.size, dtype=np.float64)
        if size.shape != (3,) or np.any(size <= 0.0):
            raise ValueError("size must contain three positive values")
        object.__setattr__(self, "size", size)

    def transfer_pose_from(
        self,
        source: "RigidAssetGeometry",
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


class HangMugSkillProgram:
    """Build one uninterrupted grasp, handover, insert, and release rollout."""

    def __init__(
        self, left_start: Any, right_start: Any, *, opened: float = -0.0475
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
        if steps <= 0:
            raise ValueError("steps must be positive")
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

    def semantic_left_grasp(
        self,
        pregrasp: Any,
        grasp: Any,
        lift: Any,
        *,
        approach_steps: int,
        close_steps: int,
        lift_steps: int,
        closed: float = 0.0,
    ) -> None:
        self._append(
            "left_pregrasp",
            "semantic_left_grasp",
            approach_steps,
            left_pose=pregrasp,
        )
        self._append(
            "left_grasp",
            "semantic_left_grasp",
            close_steps,
            left_pose=grasp,
            left_gripper=closed,
        )
        self._append(
            "left_lift", "semantic_left_grasp", lift_steps, left_pose=lift
        )

    def physical_handover(
        self,
        left_anchor: Any,
        right_pregrasp: Any,
        right_grasp: Any,
        left_release: Any,
        *,
        approach_steps: int,
        close_steps: int,
        release_steps: int,
        closed: float = 0.0,
        opened: float = -0.0475,
    ) -> None:
        self._append(
            "handover_pregrasp",
            "physical_handover",
            approach_steps,
            left_pose=left_anchor,
            right_pose=right_pregrasp,
        )
        self._append(
            "right_grasp",
            "physical_handover",
            close_steps,
            right_pose=right_grasp,
            right_gripper=closed,
        )
        self._append(
            "left_release",
            "physical_handover",
            release_steps,
            left_pose=left_release,
            left_gripper=opened,
        )

    def handle_to_branch_insert(
        self,
        right_transport: Any,
        right_approach: Any,
        right_insert: Any,
        *,
        transport_steps: int,
        approach_steps: int,
        insert_steps: int,
    ) -> None:
        self._append(
            "tree_transport",
            "handle_to_branch_insertion",
            transport_steps,
            right_pose=right_transport,
        )
        self._append(
            "branch_approach",
            "handle_to_branch_insertion",
            approach_steps,
            right_pose=right_approach,
        )
        self._append(
            "branch_insert",
            "handle_to_branch_insertion",
            insert_steps,
            right_pose=right_insert,
        )

    def release_and_support(
        self,
        right_unload: Any,
        right_settle: Any,
        *,
        unload_steps: int,
        release_steps: int,
        settle_steps: int,
        opened: float = -0.0475,
    ) -> None:
        self._append(
            "branch_unload",
            "release_support",
            unload_steps,
            right_pose=right_unload,
        )
        self._append(
            "right_release",
            "release_support",
            release_steps,
            right_gripper=opened,
        )
        self._append(
            "stable_support",
            "stable_settle",
            settle_steps,
            right_pose=right_settle,
        )

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
            left_parts.append(
                interpolate_poses(left, waypoint.left_pose, waypoint.steps)
            )
            right_parts.append(
                interpolate_poses(right, waypoint.right_pose, waypoint.steps)
            )
            fraction = np.linspace(1.0 / waypoint.steps, 1.0, waypoint.steps)
            smooth = fraction**3 * (
                10.0 - 15.0 * fraction + 6.0 * fraction**2
            )
            grippers = np.empty((waypoint.steps, 2), dtype=np.float64)
            grippers[:, 0] = left_gripper + smooth * (
                waypoint.left_gripper - left_gripper
            )
            grippers[:, 1] = right_gripper + smooth * (
                waypoint.right_gripper - right_gripper
            )
            gripper_parts.append(grippers)
            stage_names.extend([waypoint.stage] * waypoint.steps)
            cursor += waypoint.steps
            waypoint_steps[waypoint.name] = cursor - 1
            left, right = waypoint.left_pose, waypoint.right_pose
            left_gripper, right_gripper = (
                waypoint.left_gripper,
                waypoint.right_gripper,
            )
        return SkillTrajectory(
            left_poses=np.concatenate(left_parts),
            right_poses=np.concatenate(right_parts),
            grippers=np.concatenate(gripper_parts),
            stage_names=tuple(stage_names),
            waypoint_steps=waypoint_steps,
        )
