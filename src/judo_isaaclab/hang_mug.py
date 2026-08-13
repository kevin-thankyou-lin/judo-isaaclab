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
    *,
    branch_support_fraction: float = 0.5,
    branch_roll_offset_rad: float = 0.0,
    target_branch_rank: int | None = None,
) -> tuple[np.ndarray, Any, Any]:
    """Map a verified handle-on-branch relationship through measured parts."""

    branch_support_fraction = float(branch_support_fraction)
    if (
        not np.isfinite(branch_support_fraction)
        or not 0.25 <= branch_support_fraction <= 0.75
    ):
        raise ValueError("branch support fraction must be finite and in [0.25, 0.75]")
    branch_roll_offset_rad = float(branch_roll_offset_rad)
    if not np.isfinite(branch_roll_offset_rad) or abs(branch_roll_offset_rad) > np.pi / 2:
        raise ValueError("branch roll offset must be finite and within 90 degrees")

    from .semantic_parts import closest_branch, corresponding_branch

    source_handle_world = compose_pose(
        source_mug_pose, source_parts.handle_hole_frame
    )
    source_handle_tree_local = compose_pose(
        inverse_pose(source_tree_pose), source_handle_world
    )
    source_branch = closest_branch(source_branches, source_handle_tree_local[:3])
    target_values = tuple(target_branches)
    if target_branch_rank is None:
        target_branch = corresponding_branch(source_branch, target_values)
    else:
        if (
            isinstance(target_branch_rank, bool)
            or not isinstance(target_branch_rank, int)
            or not 0 <= target_branch_rank < len(target_values)
        ):
            raise ValueError("target branch rank is outside the inferred branch set")
        target_branch = target_values[target_branch_rank]
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
    if branch_roll_offset_rad:
        from .put_marker import quaternion_multiply

        half = 0.5 * branch_roll_offset_rad
        target_handle_world[3:] = quaternion_multiply(
            target_handle_world[3:],
            np.asarray([np.cos(half), 0.0, np.sin(half), 0.0]),
        )
    # Seat the authored target handle-hole center at a bounded fraction of the
    # detected branch segment after aligning the opening axis.  The midpoint is
    # the default; preserved rollout evidence may propose a deeper data-only
    # fraction when release slides toward the tip.
    target_support_local = target_branch.frame.copy()
    target_support_local[:3] = (
        target_branch.inner_point
        + branch_support_fraction
        * (target_branch.tip_point - target_branch.inner_point)
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
    observed_left_pose: Any,
    observed_right_pose: Any,
) -> SkillTrajectory:
    """Reanchor only the handover path to the mug pose observed after lift.

    The mug can rotate inside a frictional left grasp, especially when its
    geometry changes.  This deterministic feedback update preserves the
    semantic right-contact transform while holding the observed left anchor
    through receiver closure.  The authored left-release transform and all
    support targets remain unchanged.
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
    orient_end = steps.get("handover_orient_clear", pregrasp_end)
    settle_end = steps.get("right_grasp_settle", orient_end)
    grasp_end = steps["right_grasp"]
    release_end = steps["left_release"]
    lift_end = steps.get("handover_receiver_lift", release_end)
    hold_end = steps.get("handover_confirm", release_end)
    transport_end = steps["tree_transport"]
    left = np.asarray(trajectory.left_poses, dtype=np.float64).copy()
    right = np.asarray(trajectory.right_poses, dtype=np.float64).copy()
    observed_left = _pose(observed_left_pose, "observed_left_pose")
    corrected_release = transfer_pose(
        left[release_end], left[pregrasp_end], observed_left
    )
    left[start : grasp_end + 1] = observed_left
    left[grasp_end + 1 : release_end + 1] = interpolate_poses(
        observed_left, corrected_release, release_end - grasp_end
    )
    left[release_end + 1 : hold_end + 1] = corrected_release
    corrected_pregrasp = transfer_pose(
        right[pregrasp_end], nominal_mug_pose, observed_mug_pose
    )
    corrected_grasp = transfer_pose(
        right[grasp_end], nominal_mug_pose, observed_mug_pose
    )
    right[start : pregrasp_end + 1] = interpolate_poses(
        observed_right_pose, corrected_pregrasp, pregrasp_end - start + 1
    )
    if "handover_orient_clear" in steps:
        corrected_orient = transfer_pose(
            right[orient_end], nominal_mug_pose, observed_mug_pose
        )
        right[pregrasp_end + 1 : orient_end + 1] = interpolate_poses(
            corrected_pregrasp, corrected_orient, orient_end - pregrasp_end
        )
        right[orient_end + 1 : settle_end + 1] = interpolate_poses(
            corrected_orient, corrected_grasp, settle_end - orient_end
        )
        right[settle_end + 1 : grasp_end + 1] = corrected_grasp
    else:
        right[pregrasp_end + 1 : grasp_end + 1] = interpolate_poses(
            corrected_pregrasp, corrected_grasp, grasp_end - pregrasp_end
        )
    corrected_lift_right = transfer_pose(
        right[lift_end], right[release_end], corrected_grasp
    )
    right[grasp_end + 1 : release_end + 1] = corrected_grasp
    if lift_end > release_end:
        right[release_end + 1 : lift_end + 1] = interpolate_poses(
            corrected_grasp, corrected_lift_right, lift_end - release_end
        )
    right[lift_end + 1 : hold_end + 1] = corrected_lift_right
    right[hold_end + 1 : transport_end + 1] = interpolate_poses(
        corrected_lift_right,
        trajectory.right_poses[transport_end],
        transport_end - hold_end,
    )
    return SkillTrajectory(
        left_poses=left,
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


def transfer_handover_contact_by_handle_frame(
    source_mug_pose: Any,
    target_mug_pose: Any,
    source_handle_hole_frame: Any,
    target_handle_hole_frame: Any,
    source_right_eef_pose: Any,
) -> np.ndarray:
    """Preserve the demonstrated receiver pose in the authored handle frame."""

    source_handle = compose_pose(source_mug_pose, source_handle_hole_frame)
    target_handle = compose_pose(target_mug_pose, target_handle_hole_frame)
    return transfer_pose(source_right_eef_pose, source_handle, target_handle)


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
    approach_end = steps.get("right_grasp_settle", steps["right_grasp"])
    grasp_end = steps["right_grasp"]
    release_end = steps.get("handover_confirm", steps["left_release"])
    left_release_end = steps["left_release"]
    lift_end = steps.get("handover_receiver_lift", left_release_end)
    nominal_contact = _pose(nominal_right_contact, "nominal_right_contact")
    corrected_grasp = compose_pose(observed_mug_pose, nominal_contact)
    right = np.asarray(trajectory.right_poses, dtype=np.float64).copy()
    if "handover_orient_clear" in steps:
        orient_end = steps["handover_orient_clear"]
        corrected_clear = transfer_pose(
            right[orient_end], right[grasp_end], corrected_grasp
        )
        right[start : orient_end + 1] = interpolate_poses(
            observed_right_pose, corrected_clear, orient_end - start + 1
        )
        right[orient_end + 1 : approach_end + 1] = interpolate_poses(
            corrected_clear, corrected_grasp, approach_end - orient_end
        )
    else:
        right[start : approach_end + 1] = interpolate_poses(
            observed_right_pose, corrected_grasp, approach_end - start + 1
        )
    right[approach_end + 1 : grasp_end + 1] = corrected_grasp
    corrected_lift = transfer_pose(
        right[lift_end], right[left_release_end], corrected_grasp
    )
    right[grasp_end + 1 : left_release_end + 1] = corrected_grasp
    if lift_end > left_release_end:
        right[left_release_end + 1 : lift_end + 1] = interpolate_poses(
            corrected_grasp, corrected_lift, lift_end - left_release_end
        )
    right[lift_end + 1 : release_end + 1] = corrected_lift
    return SkillTrajectory(
        left_poses=trajectory.left_poses.copy(),
        right_poses=right,
        grippers=trajectory.grippers.copy(),
        stage_names=trajectory.stage_names,
        waypoint_steps=dict(trajectory.waypoint_steps),
    )


def reanchor_handover_contact_acquire(
    trajectory: SkillTrajectory,
    nominal_right_contact: Any,
    observed_mug_pose: Any,
    observed_left_pose: Any,
    observed_right_pose: Any,
    *,
    desired_contact_local_bias_m: Any = (0.0, 0.0, 0.0),
    maximum_translation_m: float = 0.04,
    maximum_rotation_error_rad: float = 0.12,
) -> tuple[SkillTrajectory, dict[str, Any]]:
    """Move the left-held mug into a closed, stationary receiver.

    The correction is the live world-space residual between the demonstrated
    mug-relative receiver contact and the observed right wrist.  It changes no
    orientation, controller setting, or later semantic target.
    """

    required = (
        "right_grasp",
        "handover_contact_acquire",
        "left_release",
    )
    missing = [name for name in required if name not in trajectory.waypoint_steps]
    if missing:
        raise ValueError(f"handover trajectory is missing waypoints: {missing}")
    limit = float(maximum_translation_m)
    if not np.isfinite(limit) or not 0.0 < limit <= 0.04:
        raise ValueError("maximum contact-acquire translation must be in (0, 0.04] m")
    rotation_limit = float(maximum_rotation_error_rad)
    if not np.isfinite(rotation_limit) or not 0.0 < rotation_limit <= 0.2:
        raise ValueError("maximum contact-acquire rotation error must be in (0, 0.2] rad")
    nominal_contact = _pose(nominal_right_contact, "nominal_right_contact")
    mug = _pose(observed_mug_pose, "observed_mug_pose")
    observed_left = _pose(observed_left_pose, "observed_left_pose")
    observed_right = _pose(observed_right_pose, "observed_right_pose")
    local_bias = np.asarray(desired_contact_local_bias_m, dtype=np.float64)
    if (
        local_bias.shape != (3,)
        or not np.all(np.isfinite(local_bias))
        or np.linalg.norm(local_bias) > 0.005
    ):
        raise ValueError("contact-acquire local bias must be three finite values within 5 mm")
    world_bias = quaternion_rotate(mug[3:], local_bias)
    desired_right = compose_pose(mug, nominal_contact)
    desired_right[:3] += world_bias
    translation = observed_right[:3] - desired_right[:3]
    norm = float(np.linalg.norm(translation))
    rotation_error = compose_pose(inverse_pose(desired_right), observed_right)
    rotation_error_rad = float(
        2.0 * np.arccos(np.clip(abs(rotation_error[3]), 0.0, 1.0))
    )
    if norm > limit:
        raise RuntimeError(
            f"live handover contact residual {norm:.6f} m exceeds {limit:.6f} m; "
            f"world_translation_m={translation.tolist()}; "
            f"desired_right_contact_world={desired_right.tolist()}; "
            f"observed_right_eef_world={observed_right.tolist()}; "
            f"rotation_error_rad={rotation_error_rad:.9f}"
        )
    if rotation_error_rad > rotation_limit:
        raise RuntimeError(
            "live handover contact rotation error "
            f"{rotation_error_rad:.6f} rad exceeds {rotation_limit:.6f} rad"
        )

    steps = trajectory.waypoint_steps
    grasp_end = steps["right_grasp"]
    acquire_end = steps["handover_contact_acquire"]
    release_end = steps["left_release"]
    if not grasp_end < acquire_end < release_end:
        raise ValueError("contact acquisition must lie between grasp and release")
    left = np.asarray(trajectory.left_poses, dtype=np.float64).copy()
    right = np.asarray(trajectory.right_poses, dtype=np.float64).copy()
    corrected_left = observed_left.copy()
    corrected_left[:3] += translation
    left[grasp_end + 1 : acquire_end + 1] = interpolate_poses(
        observed_left, corrected_left, acquire_end - grasp_end
    )
    right[grasp_end + 1 : acquire_end + 1] = observed_right
    corrected_release = trajectory.left_poses[release_end].copy()
    corrected_release[:3] += translation
    left[acquire_end + 1 : release_end + 1] = interpolate_poses(
        corrected_left, corrected_release, release_end - acquire_end
    )
    right[acquire_end + 1 : release_end + 1] = observed_right

    lift_end = steps.get("handover_receiver_lift", release_end)
    confirm_end = steps.get("handover_confirm", lift_end)
    corrected_lift = transfer_pose(
        trajectory.right_poses[lift_end],
        trajectory.right_poses[release_end],
        observed_right,
    )
    if lift_end > release_end:
        right[release_end + 1 : lift_end + 1] = interpolate_poses(
            observed_right, corrected_lift, lift_end - release_end
        )
    right[lift_end + 1 : confirm_end + 1] = corrected_lift
    left[release_end + 1 : confirm_end + 1] = corrected_release
    receipt = {
        "strategy": "translate_left_held_mug_into_stationary_closed_receiver",
        "desired_right_contact_world": desired_right.tolist(),
        "desired_contact_local_bias_m": local_bias.tolist(),
        "desired_contact_world_bias_m": world_bias.tolist(),
        "observed_right_eef_world": observed_right.tolist(),
        "world_translation_m": translation.tolist(),
        "translation_norm_m": norm,
        "maximum_translation_m": limit,
        "rotation_error_rad": rotation_error_rad,
        "maximum_rotation_error_rad": rotation_limit,
        "orientation_unchanged": True,
        "acquire_steps": acquire_end - grasp_end,
    }
    return (
        SkillTrajectory(
            left_poses=left,
            right_poses=right,
            grippers=trajectory.grippers.copy(),
            stage_names=trajectory.stage_names,
            waypoint_steps=dict(trajectory.waypoint_steps),
        ),
        receipt,
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
    if (
        completed_waypoint == "left_release"
        and "handover_confirm" in trajectory.waypoint_steps
    ):
        raise ValueError("branch transport cannot reanchor before handover confirmation")
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
        receiver_lift: Any | None = None,
        receiver_lift_steps: int = 0,
        right_orient_clear: Any | None = None,
        orient_steps: int = 0,
        approach_steps: int,
        close_steps: int,
        release_steps: int,
        contact_settle_steps: int = 0,
        contact_acquire_steps: int = 0,
        confirm_steps: int = 0,
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
        if orient_steps < 0:
            raise ValueError("orient_steps must be nonnegative")
        if bool(orient_steps) != (right_orient_clear is not None):
            raise ValueError("clear orientation target and steps must be selected together")
        if orient_steps and not contact_settle_steps:
            raise ValueError("clear orientation requires an open descent segment")
        if orient_steps:
            self._append(
                "handover_orient_clear",
                "physical_handover",
                orient_steps,
                right_pose=right_orient_clear,
            )
        if contact_settle_steps < 0:
            raise ValueError("contact_settle_steps must be nonnegative")
        if contact_settle_steps:
            self._append(
                "right_grasp_settle",
                "physical_handover",
                contact_settle_steps,
                right_pose=right_grasp,
            )
        self._append(
            "right_grasp",
            "physical_handover",
            close_steps,
            right_pose=right_grasp,
            right_gripper=closed,
        )
        if contact_acquire_steps < 0:
            raise ValueError("contact_acquire_steps must be nonnegative")
        if contact_acquire_steps:
            self._append(
                "handover_contact_acquire",
                "physical_handover",
                contact_acquire_steps,
            )
        self._append(
            "left_release",
            "physical_handover",
            release_steps,
            left_pose=left_release,
            left_gripper=opened,
        )
        if receiver_lift_steps < 0:
            raise ValueError("receiver_lift_steps must be nonnegative")
        if receiver_lift_steps:
            if receiver_lift is None:
                raise ValueError("receiver lift requires a right-wrist target")
            self._append(
                "handover_receiver_lift",
                "physical_handover",
                receiver_lift_steps,
                right_pose=receiver_lift,
            )
        if confirm_steps < 0:
            raise ValueError("confirm_steps must be nonnegative")
        if confirm_steps:
            self._append(
                "handover_confirm", "physical_handover", confirm_steps
            )

    def handle_to_branch_insert(
        self,
        right_transport: Any,
        right_approach: Any,
        right_insert: Any,
        *,
        transport_steps: int,
        right_orient_clear: Any | None = None,
        orient_steps: int = 0,
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
        if orient_steps < 0:
            raise ValueError("orient_steps must be nonnegative")
        if bool(orient_steps) != (right_orient_clear is not None):
            raise ValueError("clear branch orientation target and steps must match")
        if orient_steps:
            self._append(
                "branch_orient_clear",
                "handle_to_branch_insertion",
                orient_steps,
                right_pose=right_orient_clear,
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
