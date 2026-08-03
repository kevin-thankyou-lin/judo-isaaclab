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
    _slerp,
    compose_pose,
    inverse_pose,
    interpolate_poses,
    quaternion_multiply,
    quaternion_rotate,
    transfer_pose,
)


CENTERED_ON_COOKTOP_TOLERANCE_M = 0.03
TRANSPORT_PLANNING_MARGIN_M = 1.0e-4
CONTACT_FEEDBACK_HORIZON_STEPS = 10
HANDLE_PAD_DEPTH_MARGIN_M = 0.003
YAM_LEFT_FINGER_PIVOT_LOCAL_M = np.asarray([-0.045060, 0.024000, 0.054560])
YAM_FINGER_SEPARATION_LOCAL_M = np.asarray([0.090120, -0.048000, 0.0])
HANDLE_PAD_RELATIVE_DEPTH_M = 0.003


def _linear_contact_feedback_poses(start: Any, target: Any, steps: int) -> np.ndarray:
    """Close a measured contact residual over a fixed feedback horizon.

    Contact feedback is recomputed every controller step.  A quintic profile
    would restart at zero velocity on every update and defer almost the entire
    correction to the last frame.  Closing over a short fixed horizon also
    tracks a handle that moves when the opposite gripper contacts the pot.
    """

    if steps < 1:
        raise ValueError("contact feedback steps must be positive")
    start_pose = _pose(start, "start")
    target_pose = _pose(target, "target")
    horizon = min(CONTACT_FEEDBACK_HORIZON_STEPS, steps)
    fraction = np.linspace(1.0 / horizon, 1.0, horizon)
    prefix = np.empty((horizon, 7), dtype=np.float64)
    prefix[:, :3] = (
        start_pose[:3]
        + fraction[:, None] * (target_pose[:3] - start_pose[:3])
    )
    prefix[:, 3:] = _slerp(start_pose[3:], target_pose[3:], fraction)
    if horizon == steps:
        return prefix
    return np.concatenate(
        (prefix, np.broadcast_to(target_pose, (steps - horizon, 7)))
    )


def handle_axial_contact_scale(
    source_handle_size: Any, target_handle_size: Any, handle_axis: int
) -> np.ndarray:
    """Scale only outward wrist reach with measured handle-axis extent."""

    source = np.asarray(source_handle_size, dtype=np.float64)
    target = np.asarray(target_handle_size, dtype=np.float64)
    if source.shape != (3,) or target.shape != (3,) or np.any(source <= 0.0) or np.any(target <= 0.0):
        raise ValueError("handle sizes must contain three positive values")
    if handle_axis not in (0, 1):
        raise ValueError("handle_axis must be horizontal")
    scale = np.ones(3, dtype=np.float64)
    scale[handle_axis] = target[handle_axis] / source[handle_axis]
    return scale


def transfer_handle_pose_preserving_surface_clearance(
    value: Any,
    source_frame: Any,
    target_frame: Any,
    source_handle_size: Any,
    target_handle_size: Any,
    handle_axis: int,
) -> np.ndarray:
    """Transfer a wrist pose while preserving local handle-surface clearance."""

    source = np.asarray(source_handle_size, dtype=np.float64)
    target = np.asarray(target_handle_size, dtype=np.float64)
    if source.shape != (3,) or target.shape != (3,) or np.any(source <= 0.0) or np.any(target <= 0.0):
        raise ValueError("handle sizes must contain three positive values")
    if handle_axis not in (0, 1):
        raise ValueError("handle_axis must be horizontal")
    local = compose_pose(inverse_pose(source_frame), value)
    local[handle_axis] *= target[handle_axis] / source[handle_axis]
    for axis in range(3):
        if axis == handle_axis:
            continue
        sign = -1.0 if local[axis] < 0.0 else 1.0
        surface_clearance = abs(local[axis]) - 0.5 * source[axis]
        local[axis] = sign * (0.5 * target[axis] + surface_clearance)
    return compose_pose(target_frame, local)


def seat_handle_inside_finger_pads(grasp_pose: Any, depth_m: float) -> np.ndarray:
    """Move a wrist opposite the YAM pad tip-to-base axis to deepen contact."""

    grasp = _pose(grasp_pose, "grasp_pose")
    if not np.isfinite(depth_m) or depth_m < 0.0:
        raise ValueError("depth_m must be finite and nonnegative")
    result = grasp.copy()
    result[:3] += float(depth_m) * quaternion_rotate(
        result[3:], np.asarray([0.0, 0.0, 1.0])
    )
    return result


def balance_handle_contact_across_finger_pads(
    grasp_pose: Any,
    relative_depth_m: float = HANDLE_PAD_RELATIVE_DEPTH_M,
) -> np.ndarray:
    """Tilt about the left-finger pivot to deepen only the right-pad contact."""

    grasp = _pose(grasp_pose, "grasp_pose")
    separation = YAM_FINGER_SEPARATION_LOCAL_M
    span = float(np.linalg.norm(separation))
    if not np.isfinite(relative_depth_m) or not 0.0 <= relative_depth_m < span:
        raise ValueError("relative_depth_m must be finite and smaller than finger span")
    axis = np.asarray([separation[1], -separation[0], 0.0], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = float(np.arcsin(relative_depth_m / span))
    delta = np.concatenate(
        ([np.cos(0.5 * angle)], axis * np.sin(0.5 * angle))
    )
    result = grasp.copy()
    pivot_world = grasp[:3] + quaternion_rotate(
        grasp[3:], YAM_LEFT_FINGER_PIVOT_LOCAL_M
    )
    result[3:] = quaternion_multiply(grasp[3:], delta)
    result[3:] /= np.linalg.norm(result[3:])
    result[:3] = pivot_world - quaternion_rotate(
        result[3:], YAM_LEFT_FINGER_PIVOT_LOCAL_M
    )
    return result


@dataclass(frozen=True)
class SmoothBimanualTransport:
    """One sampled pot path and the two rigidly attached handle paths."""

    pot_poses: np.ndarray
    left_poses: np.ndarray
    right_poses: np.ndarray
    minimum_cooktop_clearance_m: float
    cooktop_overlap_samples: int


def _minimum_jerk_fraction(steps: int) -> np.ndarray:
    if steps < 1:
        raise ValueError("steps must be positive")
    fraction = np.linspace(1.0 / steps, 1.0, steps)
    return fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)


def _cooktop_overlap_mask(
    pot_positions: np.ndarray,
    pot_size: Any,
    cooktop: "RigidSupportGeometry",
) -> np.ndarray:
    size = np.asarray(pot_size, dtype=np.float64)
    if size.shape != (3,) or np.any(size <= 0.0):
        raise ValueError("pot_size must contain three positive values")
    inverse_rotation = cooktop.root_pose[3:] * np.asarray([1.0, -1.0, -1.0, -1.0])
    local_xy = np.asarray(
        [
            quaternion_rotate(inverse_rotation, position - cooktop.root_pose[:3])[:2]
            for position in np.asarray(pot_positions, dtype=np.float64)
        ]
    )
    combined_half_extent = 0.5 * (size[:2] + cooktop.size[:2])
    return np.all(np.abs(local_xy) <= combined_half_extent[None], axis=1)


def minimum_cooktop_clearance_m(
    pot_poses: Any,
    pot_size: Any,
    cooktop: "RigidSupportGeometry",
) -> float:
    """Measure minimum pot-bottom clearance while swept footprints overlap."""
    poses = np.asarray(pot_poses, dtype=np.float64)
    size = np.asarray(pot_size, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError("pot_poses must have shape (N, 7)")
    overlap = _cooktop_overlap_mask(poses[:, :3], size, cooktop)
    if not np.any(overlap):
        return float("inf")
    clearance = poses[:, 2] - 0.5 * size[2] - cooktop.top_frame[2]
    return float(np.min(clearance[overlap]))


def smooth_collision_aware_bimanual_transport(
    start_pot_pose: Any,
    target_pot_pose: Any,
    left_contact_local: Any,
    right_contact_local: Any,
    pot_size: Any,
    cooktop: "RigidSupportGeometry",
    *,
    steps: int,
    collision_clearance_m: float,
) -> SmoothBimanualTransport:
    """Build one minimum-jerk bimanual sweep that clears the cooktop.

    The planar motion and orientation use a single quintic time law.  A C2
    vertical bump is added only when the swept pot footprint would otherwise
    intersect the cooktop top plane.  Both wrist poses are composed from the
    same pot path, so their handle transforms stay rigid throughout.
    """
    if steps < 8:
        raise ValueError("smooth transport requires at least eight steps")
    if collision_clearance_m < 0.0:
        raise ValueError("collision_clearance_m must be nonnegative")
    start = _pose(start_pot_pose, "start_pot_pose")
    target = _pose(target_pot_pose, "target_pot_pose")
    left_contact = _pose(left_contact_local, "left_contact_local")
    right_contact = _pose(right_contact_local, "right_contact_local")
    size = np.asarray(pot_size, dtype=np.float64)
    pot_poses = interpolate_poses(start, target, steps)
    overlap = _cooktop_overlap_mask(pot_poses[:, :3], size, cooktop)
    required_bottom_z = cooktop.top_frame[2] + float(collision_clearance_m)
    base_bottom_z = pot_poses[:, 2] - 0.5 * size[2]
    required_lift = np.zeros(steps, dtype=np.float64)
    required_lift[overlap] = np.maximum(
        0.0, required_bottom_z - base_bottom_z[overlap]
    )
    if required_lift[-1] > 1.0e-9:
        raise ValueError("transport target does not clear the cooktop")
    positive = np.flatnonzero(required_lift > 1.0e-9)
    lift_profile = np.zeros(steps, dtype=np.float64)
    if len(positive):
        first = int(positive[0])
        last = int(positive[-1])
        if first == 0:
            raise ValueError(
                "pot starts inside the cooktop footprint below transport clearance"
            )
        rise = _minimum_jerk_fraction(first + 1)
        lift_profile[: first + 1] = rise
        lift_profile[first : last + 1] = 1.0
        fall_steps = steps - last - 1
        if fall_steps:
            lift_profile[last + 1 :] = 1.0 - _minimum_jerk_fraction(fall_steps)
        pot_poses[:, 2] += (
            float(np.max(required_lift)) + TRANSPORT_PLANNING_MARGIN_M
        ) * lift_profile
    minimum_clearance = minimum_cooktop_clearance_m(pot_poses, size, cooktop)
    if minimum_clearance + 1.0e-9 < collision_clearance_m:
        raise AssertionError("constructed transport violates cooktop clearance")
    left = np.asarray([compose_pose(pose, left_contact) for pose in pot_poses])
    right = np.asarray([compose_pose(pose, right_contact) for pose in pot_poses])
    return SmoothBimanualTransport(
        pot_poses=pot_poses,
        left_poses=left,
        right_poses=right,
        minimum_cooktop_clearance_m=minimum_clearance,
        cooktop_overlap_samples=int(np.count_nonzero(overlap)),
    )


def cartesian_smoothness_metrics(
    left_poses: Any,
    right_poses: Any,
    *,
    control_rate_hz: float = 30.0,
) -> dict[str, float | int]:
    """Return deterministic finite-difference metrics for a bimanual path."""
    left = np.asarray(left_poses, dtype=np.float64)
    right = np.asarray(right_poses, dtype=np.float64)
    if left.ndim != 2 or right.shape != left.shape or left.shape[1] != 7:
        raise ValueError("left_poses and right_poses must have matching (N, 7) shapes")
    if len(left) < 8 or control_rate_hz <= 0.0:
        raise ValueError("smoothness metrics require at least eight poses and positive rate")
    position = np.concatenate((left[:, :3], right[:, :3]), axis=1)
    velocity = np.diff(position, axis=0) * control_rate_hz
    acceleration = np.diff(velocity, axis=0) * control_rate_hz
    jerk = np.diff(acceleration, axis=0) * control_rate_hz
    speed = np.linalg.norm(velocity, axis=1)
    peak_speed = float(np.max(speed))
    trim = max(2, int(np.ceil(0.08 * len(speed))))
    interior = speed[trim:-trim] if len(speed) > 2 * trim else speed
    stop_threshold = max(1.0e-6, 0.05 * peak_speed)
    return {
        "path_length_m": float(np.sum(speed) / control_rate_hz),
        "peak_speed_mps": peak_speed,
        "peak_acceleration_mps2": float(np.max(np.linalg.norm(acceleration, axis=1))),
        "peak_jerk_mps3": float(np.max(np.linalg.norm(jerk, axis=1))),
        "internal_stop_count": int(np.count_nonzero(interior <= stop_threshold)),
        "maximum_step_m": float(np.max(speed) / control_rate_hz),
    }


def cooktop_center_error_m(pot_pose: Any, cooktop_pose: Any) -> float:
    """Return planar root-center error for the pot and cooktop."""
    pot = _pose(pot_pose, "pot_pose")
    cooktop = _pose(cooktop_pose, "cooktop_pose")
    return float(np.linalg.norm(pot[:2] - cooktop[:2]))


def reanchor_second_handle_grasp(
    trajectory: SkillTrajectory,
    nominal_pot_pose: Any,
    observed_pot_pose: Any,
    observed_left_pose: Any,
    observed_right_pose: Any,
) -> SkillTrajectory:
    """Track the second handle after the first physical grasp moves the pot."""

    steps = trajectory.waypoint_steps
    required = ("left_handle_grasp", "right_handle_grasp")
    missing = [name for name in required if name not in steps]
    if missing:
        raise ValueError(f"handle trajectory is missing waypoints: {missing}")
    start = steps["left_handle_grasp"] + 1
    end = steps["right_handle_grasp"]
    right_target = transfer_pose(
        trajectory.right_poses[end], nominal_pot_pose, observed_pot_pose
    )
    left = trajectory.left_poses.copy()
    right = trajectory.right_poses.copy()
    left[start : end + 1] = observed_left_pose
    right[start : end + 1] = interpolate_poses(
        observed_right_pose, right_target, end - start + 1
    )
    return SkillTrajectory(
        left_poses=left,
        right_poses=right,
        grippers=trajectory.grippers.copy(),
        stage_names=trajectory.stage_names,
        waypoint_steps=dict(trajectory.waypoint_steps),
    )


def track_bimanual_handle_targets(
    trajectory: SkillTrajectory,
    current_step: int,
    observed_pot_pose: Any,
    observed_left_pose: Any,
    observed_right_pose: Any,
    left_contact_local: Any,
    right_contact_local: Any,
    *,
    left_contact_latched: bool = False,
    right_contact_latched: bool = False,
) -> SkillTrajectory:
    """Track both fixed local handle contacts during simultaneous closing.

    Either handle may latch first on a new asset.  Keep an already latched
    wrist at its measured contact and smoothly chase the other handle in the
    observed pot frame instead of assuming that the left contact always leads.
    """

    steps = trajectory.waypoint_steps
    left_end = steps.get("left_handle_grasp")
    right_end = steps.get("right_handle_grasp")
    if left_end is None or right_end is None:
        raise ValueError("handle trajectory is missing grasp waypoints")
    if current_step < left_end or current_step >= right_end:
        raise ValueError("current_step is outside the second-handle approach")
    start = current_step + 1
    remaining = right_end - current_step
    left = trajectory.left_poses.copy()
    right = trajectory.right_poses.copy()
    if left_contact_latched and right_contact_latched:
        left[start : right_end + 1] = _pose(
            observed_left_pose, "observed_left_pose"
        )
        right[start : right_end + 1] = _pose(
            observed_right_pose, "observed_right_pose"
        )
    else:
        left_target = compose_pose(observed_pot_pose, left_contact_local)
        right_target = compose_pose(observed_pot_pose, right_contact_local)
        left[start : right_end + 1] = (
            _pose(observed_left_pose, "observed_left_pose")
            if left_contact_latched
            else _linear_contact_feedback_poses(
                observed_left_pose, left_target, remaining
            )
        )
        right[start : right_end + 1] = (
            _pose(observed_right_pose, "observed_right_pose")
            if right_contact_latched
            else _linear_contact_feedback_poses(
                observed_right_pose, right_target, remaining
            )
        )
    return SkillTrajectory(
        left_poses=left,
        right_poses=right,
        grippers=trajectory.grippers.copy(),
        stage_names=trajectory.stage_names,
        waypoint_steps=dict(trajectory.waypoint_steps),
    )


def reanchor_bimanual_transport_from_observation(
    trajectory: SkillTrajectory,
    observed_pot_pose: Any,
    observed_left_pose: Any,
    observed_right_pose: Any,
    final_pot_pose: Any,
    pot_size: Any,
    cooktop: "RigidSupportGeometry",
    *,
    transport_clearance_m: float,
    collision_clearance_m: float,
) -> tuple[SkillTrajectory, SmoothBimanualTransport]:
    """Rebuild the transport tail from measured contacts after both closes."""

    steps = trajectory.waypoint_steps
    required = (
        "right_handle_grasp",
        "smooth_transport",
        "support_lower",
        "pot_release",
        "bimanual_withdraw",
    )
    missing = [name for name in required if name not in steps]
    if missing:
        raise ValueError(f"transport trajectory is missing waypoints: {missing}")
    pot_pose = _pose(observed_pot_pose, "observed_pot_pose")
    final_pose = _pose(final_pot_pose, "final_pot_pose")
    left_pose = _pose(observed_left_pose, "observed_left_pose")
    right_pose = _pose(observed_right_pose, "observed_right_pose")
    left_contact = compose_pose(inverse_pose(pot_pose), left_pose)
    right_contact = compose_pose(inverse_pose(pot_pose), right_pose)
    transport_target = final_pose.copy()
    transport_target[2] += (
        max(float(transport_clearance_m), float(collision_clearance_m))
        + TRANSPORT_PLANNING_MARGIN_M
    )
    start = steps["right_handle_grasp"] + 1
    transport_end = steps["smooth_transport"]
    transport = smooth_collision_aware_bimanual_transport(
        pot_pose,
        transport_target,
        left_contact,
        right_contact,
        pot_size,
        cooktop,
        steps=transport_end - start + 1,
        collision_clearance_m=collision_clearance_m,
    )
    left = trajectory.left_poses.copy()
    right = trajectory.right_poses.copy()
    left[start : transport_end + 1] = transport.left_poses
    right[start : transport_end + 1] = transport.right_poses
    lower_end = steps["support_lower"]
    release_end = steps["pot_release"]
    withdraw_end = steps["bimanual_withdraw"]
    left_lower = compose_pose(final_pose, left_contact)
    right_lower = compose_pose(final_pose, right_contact)
    left[transport_end + 1 : lower_end + 1] = interpolate_poses(
        transport.left_poses[-1], left_lower, lower_end - transport_end
    )
    right[transport_end + 1 : lower_end + 1] = interpolate_poses(
        transport.right_poses[-1], right_lower, lower_end - transport_end
    )
    left[lower_end + 1 : release_end + 1] = left_lower
    right[lower_end + 1 : release_end + 1] = right_lower
    left_withdraw = left_lower.copy()
    right_withdraw = right_lower.copy()
    left_withdraw[:3] += [0.0, 0.08, 0.12]
    right_withdraw[:3] += [0.0, -0.08, 0.12]
    left[release_end + 1 : withdraw_end + 1] = interpolate_poses(
        left_lower, left_withdraw, withdraw_end - release_end
    )
    right[release_end + 1 : withdraw_end + 1] = interpolate_poses(
        right_lower, right_withdraw, withdraw_end - release_end
    )
    left[withdraw_end + 1 :] = left_withdraw
    right[withdraw_end + 1 :] = right_withdraw
    return (
        SkillTrajectory(
            left_poses=left,
            right_poses=right,
            grippers=trajectory.grippers.copy(),
            stage_names=trajectory.stage_names,
            waypoint_steps=dict(trajectory.waypoint_steps),
        ),
        transport,
    )


def reanchor_centered_lowering(
    trajectory: SkillTrajectory,
    center_correction_xy: Any,
    observed_left_pose: Any,
    observed_right_pose: Any,
) -> SkillTrajectory:
    """Correct tracking residual only during the short centered lowering."""
    correction = np.asarray(center_correction_xy, dtype=np.float64)
    if correction.shape != (2,) or not np.all(np.isfinite(correction)):
        raise ValueError("center_correction_xy must contain two finite values")
    steps = trajectory.waypoint_steps
    required = ("smooth_transport", "support_lower", "pot_release", "bimanual_withdraw")
    missing = [name for name in required if name not in steps]
    if missing:
        raise ValueError(f"smooth trajectory is missing waypoints: {missing}")
    start = steps["smooth_transport"] + 1
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
        left_lower, trajectory.left_poses[withdraw_end], withdraw_end - release_end
    )
    right[release_end + 1 : withdraw_end + 1] = interpolate_poses(
        right_lower, trajectory.right_poses[withdraw_end], withdraw_end - release_end
    )
    return SkillTrajectory(
        left_poses=left,
        right_poses=right,
        grippers=trajectory.grippers.copy(),
        stage_names=trajectory.stage_names,
        waypoint_steps=dict(trajectory.waypoint_steps),
    )


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
    anchor_name = (
        "center_slide"
        if "center_slide" in steps
        else "pot_unload"
        if "pot_unload" in steps
        else "support_lower"
    )
    required = (anchor_name, "pot_release", "bimanual_withdraw")
    missing = [name for name in required if name not in steps]
    if missing:
        raise ValueError(f"release trajectory is missing waypoints: {missing}")
    start = steps[anchor_name] + 1
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


def reanchor_supported_center_slide(
    trajectory: SkillTrajectory,
    observed_pot_pose: Any,
    observed_cooktop_pose: Any,
    observed_right_pose: Any,
) -> SkillTrajectory:
    """Preserve observed right contact while sliding the supported pot to center."""
    steps = trajectory.waypoint_steps
    required = ("left_unload_release", "center_slide", "pot_release", "bimanual_withdraw")
    missing = [name for name in required if name not in steps]
    if missing:
        raise ValueError(f"center-slide trajectory is missing waypoints: {missing}")
    pot_pose = _pose(observed_pot_pose, "observed_pot_pose")
    cooktop_pose = _pose(observed_cooktop_pose, "observed_cooktop_pose")
    right_pose = _pose(observed_right_pose, "observed_right_pose")
    right_contact = compose_pose(inverse_pose(pot_pose), right_pose)
    centered_pot_pose = pot_pose.copy()
    centered_pot_pose[:2] = cooktop_pose[:2]
    right_center = compose_pose(centered_pot_pose, right_contact)
    start = steps["left_unload_release"] + 1
    slide_end = steps["center_slide"]
    release_end = steps["pot_release"]
    withdraw_end = steps["bimanual_withdraw"]
    right = np.asarray(trajectory.right_poses, dtype=np.float64).copy()
    right[start : slide_end + 1] = interpolate_poses(
        right_pose, right_center, slide_end - start + 1
    )
    right[slide_end + 1 : release_end + 1] = right_center
    right[release_end + 1 : withdraw_end + 1] = interpolate_poses(
        right_center,
        trajectory.right_poses[withdraw_end],
        withdraw_end - release_end,
    )
    return SkillTrajectory(
        left_poses=trajectory.left_poses.copy(),
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


@dataclass(frozen=True)
class _SampledSkillSegment:
    name: str
    stage: str
    left_poses: np.ndarray
    right_poses: np.ndarray
    grippers: np.ndarray

    def __post_init__(self) -> None:
        left = np.asarray(self.left_poses, dtype=np.float64)
        right = np.asarray(self.right_poses, dtype=np.float64)
        grippers = np.asarray(self.grippers, dtype=np.float64)
        if not self.name or not self.stage or left.ndim != 2 or left.shape[1] != 7:
            raise ValueError("sampled segment needs a name, stage, and (N, 7) poses")
        if right.shape != left.shape or grippers.shape != (len(left), 2) or len(left) < 1:
            raise ValueError("sampled segment arrays have incompatible shapes")
        object.__setattr__(self, "left_poses", left)
        object.__setattr__(self, "right_poses", right)
        object.__setattr__(self, "grippers", grippers)


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
        self._segments: list[SkillWaypoint | _SampledSkillSegment] = []

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
        self._segments.append(
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

    def _append_sampled(
        self,
        name: str,
        stage: str,
        left_poses: Any,
        right_poses: Any,
    ) -> None:
        left = np.asarray(left_poses, dtype=np.float64)
        right = np.asarray(right_poses, dtype=np.float64)
        grippers = np.tile(
            np.asarray([self._left_gripper, self._right_gripper], dtype=np.float64),
            (len(left), 1),
        )
        segment = _SampledSkillSegment(name, stage, left, right, grippers)
        self._segments.append(segment)
        self._left = segment.left_poses[-1].copy()
        self._right = segment.right_poses[-1].copy()

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
        simultaneous: bool = False,
    ) -> None:
        self._append(
            "bimanual_pregrasp",
            "bimanual_handle_grasp",
            approach_steps,
            left_pose=left_pregrasp,
            right_pose=right_pregrasp,
        )
        if simultaneous:
            self._append(
                "left_handle_grasp",
                "bimanual_handle_grasp",
                left_close_steps,
                left_pose=left_grasp,
                right_pose=right_grasp,
                left_gripper=closed,
                right_gripper=closed,
            )
            self._append(
                "right_handle_grasp",
                "bimanual_handle_grasp",
                right_close_steps,
            )
        else:
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

    def smooth_bimanual_transport_to_center(
        self,
        start_pot_pose: Any,
        target_pot_pose: Any,
        left_contact_local: Any,
        right_contact_local: Any,
        pot_size: Any,
        cooktop: RigidSupportGeometry,
        *,
        steps: int,
        collision_clearance_m: float,
    ) -> SmoothBimanualTransport:
        """Append one collision-checked bimanual sweep to the centered target."""
        transport = smooth_collision_aware_bimanual_transport(
            start_pot_pose,
            target_pot_pose,
            left_contact_local,
            right_contact_local,
            pot_size,
            cooktop,
            steps=steps,
            collision_clearance_m=collision_clearance_m,
        )
        self._append_sampled(
            "smooth_transport",
            "smooth_bimanual_transport",
            transport.left_poses,
            transport.right_poses,
        )
        return transport

    def short_lower_release_and_settle(
        self,
        left_lower: Any,
        right_lower: Any,
        left_withdraw: Any,
        right_withdraw: Any,
        *,
        lower_steps: int,
        release_steps: int,
        withdraw_steps: int,
        settle_steps: int,
        opened: float = -0.0475,
    ) -> None:
        """Lower at center, release both handles, withdraw, and settle."""
        self._append(
            "support_lower",
            "support_alignment",
            lower_steps,
            left_pose=left_lower,
            right_pose=right_lower,
        )
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

    def supported_center_slide_and_settle(
        self,
        left_lower: Any,
        right_lower: Any,
        left_withdraw: Any,
        right_center: Any,
        right_withdraw: Any,
        *,
        lower_steps: int,
        left_release_steps: int,
        center_steps: int,
        right_release_steps: int,
        withdraw_steps: int,
        settle_steps: int,
        opened: float = -0.0475,
    ) -> None:
        """Stage on support, release the limiting arm, then slide to center."""
        self._append(
            "support_lower",
            "support_alignment",
            lower_steps,
            left_pose=left_lower,
            right_pose=right_lower,
        )
        self._append(
            "left_unload_release",
            "unload_release",
            left_release_steps,
            left_pose=left_withdraw,
            left_gripper=opened,
        )
        self._append(
            "center_slide",
            "support_alignment",
            center_steps,
            right_pose=right_center,
        )
        self._append(
            "pot_release",
            "unload_release",
            right_release_steps,
            right_gripper=opened,
        )
        self._append(
            "bimanual_withdraw",
            "stable_settle",
            withdraw_steps,
            right_pose=right_withdraw,
        )
        self._append("stable_settle", "stable_settle", settle_steps)

    def build(self) -> SkillTrajectory:
        if not self._segments:
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
        for segment in self._segments:
            if isinstance(segment, SkillWaypoint):
                left_part = interpolate_poses(left, segment.left_pose, segment.steps)
                right_part = interpolate_poses(right, segment.right_pose, segment.steps)
                smooth = _minimum_jerk_fraction(segment.steps)
                grippers = np.empty((segment.steps, 2), dtype=np.float64)
                grippers[:, 0] = left_gripper + smooth * (segment.left_gripper - left_gripper)
                grippers[:, 1] = right_gripper + smooth * (segment.right_gripper - right_gripper)
                left, right = segment.left_pose, segment.right_pose
                left_gripper, right_gripper = segment.left_gripper, segment.right_gripper
                steps = segment.steps
            else:
                left_part = segment.left_poses
                right_part = segment.right_poses
                grippers = segment.grippers
                left, right = left_part[-1], right_part[-1]
                left_gripper, right_gripper = grippers[-1]
                steps = len(left_part)
            left_parts.append(left_part)
            right_parts.append(right_part)
            gripper_parts.append(grippers)
            stage_names.extend([segment.stage] * steps)
            cursor += steps
            waypoint_steps[segment.name] = cursor - 1
        return SkillTrajectory(
            left_poses=np.concatenate(left_parts),
            right_poses=np.concatenate(right_parts),
            grippers=np.concatenate(gripper_parts),
            stage_names=tuple(stage_names),
            waypoint_steps=waypoint_steps,
        )
