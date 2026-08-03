"""Deterministic semantic frames and primitives for ``HangMugOnTree``."""

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
    inverse_pose,
    pose_from_matrix,
    quaternion_rotate,
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


def geometry_conditioned_hang_pose(
    source_mug_pose: Any,
    source_tree_pose: Any,
    source_parts: Any,
    target_parts: Any,
    source_branches: Any,
    target_tree_pose: Any,
    target_branches: Any,
) -> tuple[np.ndarray, Any, Any]:
    """Map a verified handle-on-branch relationship through measured parts."""

    from .semantic_parts import closest_branch, corresponding_branch

    source_handle_world = compose_pose(
        source_mug_pose, source_parts.handle_hole_frame
    )
    source_handle_tree_local = compose_pose(
        inverse_pose(source_tree_pose), source_handle_world
    )
    source_branch = closest_branch(source_branches, source_handle_tree_local[:3])
    target_branch = corresponding_branch(source_branch, target_branches)
    source_branch_world = compose_pose(source_tree_pose, source_branch.frame)
    target_branch_world = compose_pose(target_tree_pose, target_branch.frame)
    target_handle_world = transfer_pose(
        source_handle_world,
        source_branch_world,
        target_branch_world,
        local_position_scale=(
            target_branch.length_m / source_branch.length_m,
            target_parts.handle_outer_size[1]
            / source_parts.handle_outer_size[1],
            target_parts.handle_outer_size[2]
            / source_parts.handle_outer_size[2],
        ),
    )
    # The source demonstration proposes the supported branch and resolves the
    # handle roll/sign, but its branch-relative translation and residual axis
    # error are not a target clearance contract.  A shallow target handle can
    # otherwise place the branch against the rim even when the nominal root
    # pose looks plausible.  Project the transferred handle x axis onto the
    # plane normal to the authored branch tangent, then rebuild a right-handed
    # handle frame whose y (hole) axis is exactly that tangent.  The projection
    # selects the closest roll to the semantic source without an asset ID or a
    # sampled candidate.
    branch_tangent_world = quaternion_rotate(
        target_branch_world[3:], [1.0, 0.0, 0.0]
    )
    transferred_handle_x = quaternion_rotate(
        target_handle_world[3:], [1.0, 0.0, 0.0]
    )
    handle_x = transferred_handle_x - (
        np.dot(transferred_handle_x, branch_tangent_world) * branch_tangent_world
    )
    handle_x_norm = np.linalg.norm(handle_x)
    if handle_x_norm < 1.0e-8:
        transferred_handle_z = quaternion_rotate(
            target_handle_world[3:], [0.0, 0.0, 1.0]
        )
        handle_x = np.cross(branch_tangent_world, transferred_handle_z)
        handle_x_norm = np.linalg.norm(handle_x)
    if handle_x_norm < 1.0e-8:
        raise ValueError("cannot resolve handle roll around the target branch")
    handle_x /= handle_x_norm
    handle_z = np.cross(handle_x, branch_tangent_world)
    aligned_handle_matrix = np.eye(4, dtype=np.float64)
    aligned_handle_matrix[:3, :3] = np.column_stack(
        (handle_x, branch_tangent_world, handle_z)
    )
    target_handle_world[3:] = pose_from_matrix(aligned_handle_matrix)[3:]
    # Seat the authored target handle-hole center at the middle of the detected
    # branch segment after aligning the opening axis.  The extraction frame is
    # intentionally near the tip for branch identification and approach, but
    # a long branch with a narrow handle can slide free from that shallow
    # location after release.  The segment midpoint is a geometry-only support
    # location with branch material on both sides; it uses no asset ID or
    # sampled candidate.
    target_support_local = target_branch.frame.copy()
    target_support_local[:3] = 0.5 * (
        target_branch.inner_point + target_branch.tip_point
    )
    target_support_world = compose_pose(target_tree_pose, target_support_local)
    target_handle_world[:3] = target_support_world[:3]
    return (
        compose_pose(target_handle_world, inverse_pose(target_parts.handle_hole_frame)),
        source_branch,
        target_branch,
    )


def reanchor_physical_handover(
    trajectory: SkillTrajectory,
    nominal_mug_pose: Any,
    observed_mug_pose: Any,
    observed_right_pose: Any,
) -> SkillTrajectory:
    """Reanchor only the handover path to the mug pose observed after lift.

    The mug can rotate inside a frictional left grasp, especially when its
    geometry changes.  This deterministic feedback update preserves the
    semantic right-contact transform while leaving pick and support targets
    unchanged.
    """

    required = (
        "left_lift",
        "handover_pregrasp",
        "right_grasp",
        "left_release",
        "tree_transport",
    )
    missing = [name for name in required if name not in trajectory.waypoint_steps]
    if missing:
        raise ValueError(f"handover trajectory is missing waypoints: {missing}")
    steps = trajectory.waypoint_steps
    start = steps["left_lift"] + 1
    pregrasp_end = steps["handover_pregrasp"]
    grasp_end = steps["right_grasp"]
    release_end = steps["left_release"]
    transport_end = steps["tree_transport"]
    right = np.asarray(trajectory.right_poses, dtype=np.float64).copy()
    corrected_pregrasp = transfer_pose(
        right[pregrasp_end], nominal_mug_pose, observed_mug_pose
    )
    corrected_grasp = transfer_pose(
        right[grasp_end], nominal_mug_pose, observed_mug_pose
    )
    right[start : pregrasp_end + 1] = interpolate_poses(
        observed_right_pose, corrected_pregrasp, pregrasp_end - start + 1
    )
    right[pregrasp_end + 1 : grasp_end + 1] = interpolate_poses(
        corrected_pregrasp, corrected_grasp, grasp_end - pregrasp_end
    )
    right[grasp_end + 1 : release_end + 1] = corrected_grasp
    right[release_end + 1 : transport_end + 1] = interpolate_poses(
        corrected_grasp,
        trajectory.right_poses[transport_end],
        transport_end - release_end,
    )
    return SkillTrajectory(
        left_poses=trajectory.left_poses.copy(),
        right_poses=right,
        grippers=trajectory.grippers.copy(),
        stage_names=trajectory.stage_names,
        waypoint_steps=dict(trajectory.waypoint_steps),
    )


def ensure_pick_latch_clearance(
    handover_body_pose: Any,
    initial_body_pose: Any,
    body_height_m: float,
    *,
    pick_threshold_m: float = 0.05,
) -> np.ndarray:
    """Keep a frictionally carried mug safely above the coded pick threshold."""

    handover = _pose(handover_body_pose, "handover_body_pose").copy()
    initial = _pose(initial_body_pose, "initial_body_pose")
    if body_height_m <= 0.0:
        raise ValueError("body_height_m must be positive")
    handover[2] = max(
        handover[2], initial[2] + float(pick_threshold_m) + float(body_height_m)
    )
    return handover


def reanchor_right_grasp_from_observed_mug(
    trajectory: SkillTrajectory,
    nominal_right_contact: Any,
    observed_mug_pose: Any,
    observed_right_pose: Any,
) -> SkillTrajectory:
    """Recompute the close path from the mug observed at handover pregrasp."""

    required = ("handover_pregrasp", "right_grasp", "left_release")
    missing = [name for name in required if name not in trajectory.waypoint_steps]
    if missing:
        raise ValueError(f"handover trajectory is missing waypoints: {missing}")
    steps = trajectory.waypoint_steps
    start = steps["handover_pregrasp"] + 1
    grasp_end = steps["right_grasp"]
    release_end = steps["left_release"]
    nominal_contact = _pose(nominal_right_contact, "nominal_right_contact")
    corrected_grasp = compose_pose(observed_mug_pose, nominal_contact)
    right = np.asarray(trajectory.right_poses, dtype=np.float64).copy()
    right[start : grasp_end + 1] = interpolate_poses(
        observed_right_pose, corrected_grasp, grasp_end - start + 1
    )
    right[grasp_end + 1 : release_end + 1] = corrected_grasp
    return SkillTrajectory(
        left_poses=trajectory.left_poses.copy(),
        right_poses=right,
        grippers=trajectory.grippers.copy(),
        stage_names=trajectory.stage_names,
        waypoint_steps=dict(trajectory.waypoint_steps),
    )


def reanchor_branch_transport_contact(
    trajectory: SkillTrajectory,
    planned_right_contact: Any,
    observed_mug_pose: Any,
    observed_right_pose: Any,
    *,
    completed_waypoint: str = "left_release",
) -> SkillTrajectory:
    """Reanchor future transport to the currently observed right contact."""

    if completed_waypoint not in trajectory.waypoint_steps:
        raise ValueError(
            f"trajectory is missing {completed_waypoint} waypoint"
        )
    planned_contact = _pose(planned_right_contact, "planned_right_contact")
    observed_contact = compose_pose(
        inverse_pose(observed_mug_pose), observed_right_pose
    )
    right = np.asarray(trajectory.right_poses, dtype=np.float64).copy()
    start = trajectory.waypoint_steps[completed_waypoint] + 1
    for index in range(start, len(right)):
        intended_mug = compose_pose(right[index], inverse_pose(planned_contact))
        right[index] = compose_pose(intended_mug, observed_contact)
    return SkillTrajectory(
        left_poses=trajectory.left_poses.copy(),
        right_poses=right,
        grippers=trajectory.grippers.copy(),
        stage_names=trajectory.stage_names,
        waypoint_steps=dict(trajectory.waypoint_steps),
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
        left_observer: Any | None = None,
    ) -> None:
        """Transport and insert while the left wrist observes the branch.

        When supplied, ``left_observer`` is reached during transport and then
        held through branch alignment, insertion, release, and settling.  This
        keeps the target branch visible without adding a stop or changing the
        right-arm insertion trajectory.
        """
        self._append(
            "tree_transport",
            "handle_to_branch_insertion",
            transport_steps,
            left_pose=left_observer,
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
