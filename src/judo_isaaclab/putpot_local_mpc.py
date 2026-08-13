"""Deterministic handle-local receding-horizon control for PutPot acquisition.

This controller is deliberately small.  It projects one observed active-wrist
SE(3) increment and one jaw increment onto fixed bounds, executes only the first
element of a short deterministic plan, and replans after the next observation.
It never samples candidate rollouts and has no IsaacLab dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .put_marker import inverse_pose, quaternion_multiply, quaternion_rotate


PRECLOSURE_GEOMETRIC_EXTRA_RECENTER_STEPS = 2
PRECLOSURE_GEOMETRIC_BUDGET_EXHAUSTION_TOLERANCE_M = 1.0e-6


@dataclass(frozen=True)
class HandleLocalMpcConfig:
    horizon_steps: int = 15
    maximum_translation_step_m: float = 0.004
    maximum_rotation_step_rad: float = 0.08
    maximum_jaw_step: float = 0.004
    closed_jaw_command: float = 0.0
    minimum_force_n: float = 1.0
    minimum_pad_fraction_margin: float = 0.10
    maximum_pre_peer_pot_motion_m: float = 0.003
    robust_latch_steps: int = 15
    source_prior_initial_weight: float = 0.25
    source_prior_decay_steps: int = 5
    closure_position_tolerance_m: float = 0.010
    closure_rotation_tolerance_rad: float = 0.20
    physical_contact_threshold_n: float = 0.1
    depth_guard_transverse_tolerance_m: float = 0.010
    depth_guard_release_steps: int = 3
    maximum_contact_recenter_step_m: float = 0.001
    maximum_contact_recenter_total_m: float = 0.012

    def __post_init__(self) -> None:
        if not 10 <= self.horizon_steps <= 20:
            raise ValueError("handle-local MPC horizon must be in [10, 20]")
        positive = (
            self.maximum_translation_step_m,
            self.maximum_rotation_step_rad,
            self.maximum_jaw_step,
            self.minimum_force_n,
            self.maximum_pre_peer_pot_motion_m,
            self.closure_position_tolerance_m,
            self.closure_rotation_tolerance_rad,
            self.physical_contact_threshold_n,
            self.depth_guard_transverse_tolerance_m,
            self.maximum_contact_recenter_step_m,
            self.maximum_contact_recenter_total_m,
        )
        if not np.all(np.isfinite(positive)) or any(value <= 0.0 for value in positive):
            raise ValueError("handle-local MPC positive bounds must be finite")
        if not 0.0 <= self.minimum_pad_fraction_margin <= 0.5:
            raise ValueError("pad-fraction margin must be in [0, 0.5]")
        if (
            self.robust_latch_steps < 1
            or self.source_prior_decay_steps < 1
            or self.depth_guard_release_steps < 1
        ):
            raise ValueError(
                "latch, source-prior decay, and depth-guard release steps must be positive"
            )
        if not 0.0 <= self.source_prior_initial_weight < 1.0:
            raise ValueError("source prior must be a bounded warm-start weight")


@dataclass(frozen=True)
class HandleLocalMpcCommand:
    wrist_target_pose: np.ndarray
    jaw_command: float
    robust_streak: int
    robust_latch_ready: bool
    depth_guard_alignment_streak: int
    depth_guard_released: bool
    contact_recenter_total_m: float
    closure_committed: bool
    fail_closed: bool
    fail_reason: str | None
    frame_receipt: dict[str, Any]


def handle_local_mpc_active(
    *,
    enabled: bool,
    peer_latched: bool,
    step: int,
    peer_latch_step: int,
    grasp_complete_step: int,
) -> bool:
    """Select only the post-peer acquisition window without changing other stages."""

    if min(step, peer_latch_step, grasp_complete_step) < 0:
        raise ValueError("contact-window steps must be nonnegative")
    return bool(
        enabled
        and peer_latched
        and peer_latch_step < step <= grasp_complete_step
    )


def handle_local_bootstrap_active(
    *,
    enabled: bool,
    latch_ready: bool,
    step: int,
    bootstrap_start_step: int,
    grasp_complete_step: int,
) -> bool:
    """Select a bounded active-arm bootstrap before the peer acquisition."""

    if min(step, bootstrap_start_step, grasp_complete_step) < 0:
        raise ValueError("bootstrap steps must be nonnegative")
    if bootstrap_start_step > grasp_complete_step:
        raise ValueError("bootstrap must start before grasp completion")
    return bool(
        enabled
        and not latch_ready
        and bootstrap_start_step <= step <= grasp_complete_step
    )


def contact_window_joint_nominal_weight(
    *, active: bool, non_contact_weight: float = 1.0
) -> float:
    """Zero the source-joint anchor only while local contact control is active."""

    if not np.isfinite(non_contact_weight) or not 0.0 <= non_contact_weight <= 1.0:
        raise ValueError("non-contact joint nominal weight must be in [0, 1]")
    return 0.0 if active else float(non_contact_weight)


def source_prior_weight(step: int, config: HandleLocalMpcConfig) -> float:
    """Decay the source pose from a warm start to zero deterministic authority."""

    if step < 0:
        raise ValueError("contact-window step must be nonnegative")
    fraction = max(0.0, 1.0 - step / float(config.source_prior_decay_steps))
    return float(config.source_prior_initial_weight * fraction)


def _pose(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite pose")
    norm = float(np.linalg.norm(result[3:]))
    if norm <= 1.0e-9:
        raise ValueError(f"{name} quaternion must be nonzero")
    result = result.copy()
    result[3:] /= norm
    return result


def _vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape {shape} with finite values")
    return result.copy()


def _clip_norm(value: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value.copy() if norm <= maximum else value * (maximum / norm)


def _unit(value: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-9:
        raise ValueError(f"{name} must be nonzero")
    return value / norm


def realized_contact_recenter_displacement_m(
    previous_wrist_position_m: Any,
    current_wrist_position_m: Any,
    preceding_translation_world_m: Any,
) -> float:
    """Measure the realized positive motion from one recenter command.

    The controller budget is a physical swept-motion bound, so it must not be
    charged for Cartesian motion that the joint-space IK did not realize.  Cap
    the measured projection by the preceding bounded command to reject
    unrelated settling or overshoot.
    """

    previous = _vector(previous_wrist_position_m, (3,), "previous wrist position")
    current = _vector(current_wrist_position_m, (3,), "current wrist position")
    command = _vector(
        preceding_translation_world_m,
        (3,),
        "preceding recenter translation",
    )
    command_norm = float(np.linalg.norm(command))
    if command_norm <= 1.0e-12:
        return 0.0
    positive_projection = max(
        0.0,
        float(np.dot(current - previous, command / command_norm)),
    )
    return min(positive_projection, command_norm)


def _axis_angle(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    inverse_actual = actual * np.asarray([1.0, -1.0, -1.0, -1.0])
    delta = quaternion_multiply(target, inverse_actual)
    if delta[0] < 0.0:
        delta = -delta
    vector_norm = float(np.linalg.norm(delta[1:]))
    if vector_norm <= 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, float(delta[0]))
    return angle * delta[1:] / vector_norm


def _apply_axis_angle(pose: np.ndarray, increment: np.ndarray) -> np.ndarray:
    result = pose.copy()
    angle = float(np.linalg.norm(increment))
    if angle <= 1.0e-12:
        return result
    axis = increment / angle
    delta = np.concatenate(([np.cos(0.5 * angle)], axis * np.sin(0.5 * angle)))
    result[3:] = quaternion_multiply(delta, result[3:])
    result[3:] /= np.linalg.norm(result[3:])
    return result


def _pad_margin_ok(forces: np.ndarray, fractions: np.ndarray, config: HandleLocalMpcConfig) -> bool:
    contacting = forces >= config.physical_contact_threshold_n
    if not np.any(contacting):
        return True
    selected = fractions[contacting]
    return bool(
        np.all(np.isfinite(selected))
        and np.all(selected >= config.minimum_pad_fraction_margin - 1.0e-12)
        and np.all(
            selected <= 1.0 - config.minimum_pad_fraction_margin + 1.0e-12
        )
    )


def _robust_arm(forces: np.ndarray, fractions: np.ndarray, config: HandleLocalMpcConfig) -> bool:
    margin = np.minimum(fractions, 1.0 - fractions)
    return bool(
        np.all(forces >= config.minimum_force_n)
        and np.all(np.isfinite(margin))
        and np.all(margin >= config.minimum_pad_fraction_margin - 1.0e-12)
    )


def handle_local_mpc_frame_receipt_complete(receipt: dict[str, Any]) -> bool:
    """Machine-check the per-frame geometry/control receipt used by diagnostics."""

    required = {
        "schema_version",
        "controller",
        "contact_window_step",
        "horizon_steps",
        "source_prior_weight",
        "joint_nominal_weight",
        "observed_frames",
        "signed_residuals",
        "planned_controls",
        "executed_control",
        "hard_constraints",
        "contact_frame_guard",
        "contact_fraction_recenter",
        "closure",
        "latch",
        "fail_closed",
        "fail_reason",
    }
    if set(receipt) != required or receipt.get("schema_version") != 1:
        return False
    frames = receipt.get("observed_frames", {})
    if set(frames) != {
        "pot_body",
        "active_handle_contact",
        "active_wrist",
        "object_relative_wrist_prior",
        "object_relative_jaw_axis_prior",
        "object_relative_pad_depth_axis_prior",
        "active_pad_centers",
        "active_pad_axes",
        "jaw_midpoint",
        "jaw_axis",
        "mean_pad_depth_axis",
    }:
        return False
    residuals = receipt.get("signed_residuals", {})
    controls = receipt.get("executed_control", {})
    constraints = receipt.get("hard_constraints", {})
    contact_frame_guard = receipt.get("contact_frame_guard", {})
    contact_fraction_recenter = receipt.get("contact_fraction_recenter", {})
    closure = receipt.get("closure", {})
    latch = receipt.get("latch", {})
    return bool(
        set(residuals)
        == {
            "jaw_midpoint_to_handle_world_m",
            "jaw_midpoint_to_handle_local_m",
            "source_warm_start_world_m",
            "wrist_axis_angle_world_rad",
            "jaw_axis_alignment_world_rad",
            "pad_axis_alignment_world_rad",
            "pad_fraction_to_center",
            "force_to_minimum_n",
        }
        and set(controls)
        == {
            "translation_world_m",
            "rotation_axis_angle_world_rad",
            "jaw_increment",
            "wrist_target_pose",
            "jaw_command",
        }
        and set(constraints)
        == {
            "pre_peer_pot_motion_m",
            "pre_peer_pot_motion_within_limit",
            "active_contact_pad_margin_valid",
            "peer_contact_pad_margin_valid",
            "translation_step_within_bound",
            "rotation_step_within_bound",
            "jaw_step_within_bound",
        }
        and set(contact_frame_guard)
        == {
            "enabled",
            "active",
            "depth_axis_source",
            "depth_axis_world",
            "physical_contact_observed",
            "interior_single_pad_transverse_intercept_enabled",
            "interior_single_pad_transverse_intercept_active",
            "rotation_held_during_interior_single_pad_intercept",
            "transverse_tolerance_m",
            "transverse_residual_world_m",
            "transverse_residual_norm_m",
            "signed_depth_residual_m",
            "suppressed_depth_control_world_m",
            "transverse_aligned",
            "alignment_consecutive_frames",
            "release_required_consecutive_frames",
            "released",
        }
        and set(contact_fraction_recenter)
        == {
            "budget_accounting",
            "enabled",
            "active",
            "surface_tangent_enabled",
            "surface_tangent_axis_valid",
            "surface_tangent_axis_world",
            "guarded_depth_completion_enabled",
            "guarded_depth_completion_active",
            "guarded_depth_completion_translation_world_m",
            "preserve_transverse_centering",
            "preserve_bounded_closure",
            "bounded_closure_priority_active",
            "preclosure_geometric_prestage_enabled",
            "preclosure_geometric_prestage_active",
            "preclosure_geometric_uses_raw_pad_axis",
            "finger_tip_to_base_axis_world",
            "pad_fraction_axis_extent_m",
            "contact_fraction_delta",
            "pre_release_margin_protection_enabled",
            "pre_release_margin_protection_active",
            "protected_pad_fraction_margin",
            "acceptance_pad_fraction_margin",
            "requested_translation_m",
            "executed_translation_m",
            "budgeted_axial_translation_world_m",
            "retained_transverse_translation_world_m",
            "executed_handle_normal_component_m",
            "world_command_budget_m",
            "world_command_norm_m",
            "total_translation_m",
            "maximum_step_m",
            "base_maximum_total_m",
            "maximum_total_m",
            "effective_maximum_total_m",
            "preclosure_geometric_extra_budget_m",
            "preclosure_geometric_maximum_total_m",
            "preclosure_geometric_budget_exhaustion_tolerance_m",
        }
        and set(closure)
        == {
            "commit_enabled",
            "was_committed",
            "initial_alignment_satisfied",
            "interior_single_pad_trigger_enabled",
            "interior_single_pad_triggered",
            "interior_single_pad_closure_hold_active",
            "wrist_frozen_for_interior_single_pad_closure",
            "transverse_aligned_two_pad_trigger_enabled",
            "transverse_aligned_two_pad_triggered",
            "transverse_aligned_two_pad_closure_hold_active",
            "wrist_frozen_for_transverse_aligned_two_pad_closure",
            "loaded_pad_pivot_closure_enabled",
            "loaded_pad_pivot_closure_active",
            "loaded_pad_pivot_index",
            "loaded_pad_pivot_translation_world_m",
            "loaded_pad_pivot_translation_norm_m",
            "loaded_pad_pivot_jaw_scale",
            "dual_force_pad_margin_pivot_enabled",
            "dual_force_pad_margin_pivot_active",
            "dual_force_pad_margin_pivot_geometry_feasible",
            "dual_force_pad_margin_pivot_strong_index",
            "dual_force_pad_margin_pivot_weak_index",
            "dual_force_pad_margin_pivot_target_fraction",
            "dual_force_pad_margin_pivot_predicted_fraction",
            "dual_force_pad_margin_pivot_translation_tracking_scale",
            "dual_force_pad_margin_pivot_uncompensated_translation_world_m",
            "dual_force_pad_margin_pivot_translation_world_m",
            "dual_force_pad_margin_pivot_rotation_axis_angle_world_rad",
            "dual_force_pad_margin_pivot_point_world_m",
            "pre_peer_motion_budgeted_closure_enabled",
            "pre_peer_motion_budgeted_closure_active",
            "pre_peer_motion_remaining_m",
            "pre_peer_motion_control_scale",
            "pre_peer_motion_unbudgeted_translation_world_m",
            "pre_peer_motion_unbudgeted_rotation_axis_angle_world_rad",
            "pre_peer_motion_unbudgeted_jaw_increment",
            "geometric_preseat_required",
            "geometric_preseat_satisfied",
            "geometric_preseat_finite_pad_count",
            "geometric_preseat_interior_pad_count",
            "increment_active",
            "committed",
            "closed_command_reached",
            "dual_force_backed",
            "paused_on_dual_force_backing",
        }
        and set(latch)
        == {
            "active_force_and_margin",
            "peer_force_and_margin",
            "peer_required",
            "robust_frame",
            "strict_four_pad_frame",
            "consecutive_frames",
            "required_consecutive_frames",
            "ready",
        }
    )


def handle_local_mpc_step(
    *,
    contact_window_step: int,
    observed_pot_pose: Any,
    observed_handle_contact_frame: Any,
    active_wrist_pose: Any,
    object_relative_wrist_prior: Any,
    object_relative_jaw_axis_prior: Any,
    object_relative_pad_depth_axis_prior: Any,
    source_warm_start_wrist_pose: Any,
    active_pad_centers_world: Any,
    active_pad_axes_world: Any,
    active_pad_fractions: Any,
    active_finger_forces_n: Any,
    peer_pad_fractions: Any,
    peer_finger_forces_n: Any,
    active_grasp: bool,
    peer_grasp: bool,
    pre_peer_pot_displacement_m: float,
    current_jaw_command: float,
    robust_streak: int,
    require_peer_latch: bool = True,
    depth_guarded_transverse_intercept: bool = False,
    depth_guard_use_handle_contact_normal: bool = False,
    depth_guard_alignment_streak: int = 0,
    depth_guard_released: bool = False,
    contact_fraction_recenter: bool = False,
    contact_recenter_use_handle_tangent: bool = False,
    contact_recenter_preserve_transverse_centering: bool = False,
    contact_recenter_preserve_bounded_closure: bool = False,
    allow_bounded_closure_commit: bool = False,
    closure_committed: bool = False,
    pause_committed_closure_on_dual_force_backing: bool = False,
    allow_interior_single_pad_closure: bool = False,
    allow_interior_single_pad_transverse_intercept: bool = False,
    allow_transverse_aligned_two_pad_closure: bool = False,
    transverse_aligned_closure_pivot_pad_index: int | None = None,
    allow_dual_force_pad_margin_pivot: bool = False,
    require_geometric_preseat_for_closure: bool = False,
    budget_committed_closure_by_pre_peer_motion: bool = False,
    active_pad_fraction_axis_extent_m: float = 0.0,
    contact_recenter_total_m: float = 0.0,
    config: HandleLocalMpcConfig = HandleLocalMpcConfig(),
) -> HandleLocalMpcCommand:
    """Plan and return the first bounded active-wrist and jaw control increment."""

    if min(
        contact_window_step,
        robust_streak,
        depth_guard_alignment_streak,
        contact_recenter_total_m,
    ) < 0:
        raise ValueError(
            "contact-window, robust, and depth-guard streaks must be nonnegative"
        )
    if transverse_aligned_closure_pivot_pad_index not in (None, 0, 1):
        raise ValueError("closure pivot pad index must be 0, 1, or None")
    if (
        transverse_aligned_closure_pivot_pad_index is not None
        and not allow_transverse_aligned_two_pad_closure
    ):
        raise ValueError(
            "closure pivot pad requires transverse-aligned two-pad closure"
        )
    pot = _pose(observed_pot_pose, "observed_pot_pose")
    handle = _pose(observed_handle_contact_frame, "observed_handle_contact_frame")
    wrist = _pose(active_wrist_pose, "active_wrist_pose")
    prior = _pose(object_relative_wrist_prior, "object_relative_wrist_prior")
    jaw_axis_prior_local = _unit(
        _vector(
            object_relative_jaw_axis_prior,
            (3,),
            "object_relative_jaw_axis_prior",
        ),
        "object_relative_jaw_axis_prior",
    )
    pad_axis_prior_local = _unit(
        _vector(
            object_relative_pad_depth_axis_prior,
            (3,),
            "object_relative_pad_depth_axis_prior",
        ),
        "object_relative_pad_depth_axis_prior",
    )
    warm = _pose(source_warm_start_wrist_pose, "source_warm_start_wrist_pose")
    centers = _vector(active_pad_centers_world, (2, 3), "active_pad_centers_world")
    axes = _vector(active_pad_axes_world, (2, 3), "active_pad_axes_world")
    fractions = np.asarray(active_pad_fractions, dtype=np.float64)
    forces = _vector(active_finger_forces_n, (2,), "active_finger_forces_n")
    peer_fractions = np.asarray(peer_pad_fractions, dtype=np.float64)
    peer_forces = _vector(peer_finger_forces_n, (2,), "peer_finger_forces_n")
    if fractions.shape != (2,) or peer_fractions.shape != (2,):
        raise ValueError("pad fractions must contain two values per gripper")
    contacting = forces >= config.physical_contact_threshold_n
    physical_contact_observed = bool(np.any(contacting))
    scalars = np.asarray(
        [pre_peer_pot_displacement_m, current_jaw_command], dtype=np.float64
    )
    if not np.all(np.isfinite(scalars)):
        raise ValueError("pot displacement and jaw command must be finite")
    if not np.isfinite(active_pad_fraction_axis_extent_m) or (
        (contact_fraction_recenter or allow_dual_force_pad_margin_pivot)
        and active_pad_fraction_axis_extent_m <= 0.0
    ):
        raise ValueError(
            "contact repair requires a positive finite pad-fraction axis extent"
        )

    jaw_midpoint = np.mean(centers, axis=0)
    jaw_axis = _unit(centers[1] - centers[0], "jaw closing line")
    mean_pad_axis = _unit(np.mean(axes, axis=0), "mean pad depth axis")
    handle_contact_normal = _unit(
        quaternion_rotate(handle[3:], np.asarray([0.0, 0.0, 1.0])),
        "handle contact normal",
    )
    # Preserve the demonstrated force-free corridor on the gripper's pad-depth
    # axis.  The target handle normal becomes authoritative only once physical
    # contact exists; enabling it earlier changed Pair 15's interior first
    # contact into an edge intersection.  Once a force-backed tangent recenter
    # has physically started, retain that handle frame across transient force
    # dropouts until guard release.  Otherwise the axis can snap back to the
    # pad frame for one step and turn a tangential command inward again.
    handle_contact_normal_latched = bool(
        depth_guard_use_handle_contact_normal
        and contact_fraction_recenter
        and contact_recenter_use_handle_tangent
        and contact_recenter_total_m > 0.0
        and not depth_guard_released
    )
    # A finite pad intersection immediately after force-backed recentering is
    # retained only for bounded tangent control during a sub-threshold force
    # dropout.  ``contacting`` remains unchanged and is still the sole source
    # for force-backed margin, latch, closure, and acceptance decisions.
    transient_geometric_contacting = bool(
        handle_contact_normal_latched
        and not physical_contact_observed
    ) & np.isfinite(fractions)
    control_contacting = contacting | transient_geometric_contacting
    control_contact_observed = bool(np.any(control_contacting))
    use_contact_normal_depth_axis = bool(
        depth_guard_use_handle_contact_normal
        and (physical_contact_observed or handle_contact_normal_latched)
    )
    depth_guard_axis = (
        handle_contact_normal if use_contact_normal_depth_axis else mean_pad_axis
    )
    desired_jaw_axis = _unit(
        quaternion_rotate(pot[3:], jaw_axis_prior_local), "desired jaw axis"
    )
    desired_pad_axis = _unit(
        quaternion_rotate(pot[3:], pad_axis_prior_local), "desired pad axis"
    )
    translation_world = handle[:3] - jaw_midpoint
    translation_local = quaternion_rotate(
        inverse_pose(handle)[3:], translation_world
    )
    warm_residual = warm[:3] - wrist[:3]
    wrist_rotation = _axis_angle(wrist[3:], prior[3:])
    jaw_alignment = np.cross(jaw_axis, desired_jaw_axis)
    pad_alignment = np.cross(mean_pad_axis, desired_pad_axis)
    rotation_residual = wrist_rotation + 0.25 * (jaw_alignment + pad_alignment)
    prior_weight = source_prior_weight(contact_window_step, config)
    blended_translation = (
        (1.0 - prior_weight) * translation_world + prior_weight * warm_residual
    )
    signed_depth_residual = float(np.dot(translation_world, depth_guard_axis))
    transverse_residual = (
        translation_world - signed_depth_residual * depth_guard_axis
    )
    active_margin_ok = _pad_margin_ok(forces, fractions, config)
    peer_margin_ok = _pad_margin_ok(peer_forces, peer_fractions, config)
    pot_motion_ok = bool(
        pre_peer_pot_displacement_m
        <= config.maximum_pre_peer_pot_motion_m + 1.0e-12
    )
    interior_single_pad_transverse_intercept_active = bool(
        allow_interior_single_pad_transverse_intercept
        and np.count_nonzero(contacting) == 1
        and active_margin_ok
        and pot_motion_ok
    )
    depth_guard_alignment_observation_valid = bool(
        not physical_contact_observed
        or interior_single_pad_transverse_intercept_active
    )
    transverse_residual_norm = float(np.linalg.norm(transverse_residual))
    transverse_aligned = bool(
        transverse_residual_norm <= config.depth_guard_transverse_tolerance_m
    )
    next_depth_guard_streak = (
        depth_guard_alignment_streak + 1
        if (
            depth_guarded_transverse_intercept
            and not depth_guard_released
            and depth_guard_alignment_observation_valid
            and transverse_aligned
        )
        else 0
    )
    next_depth_guard_released = bool(
        depth_guard_released
        or (
            depth_guarded_transverse_intercept
            and next_depth_guard_streak >= config.depth_guard_release_steps
        )
    )
    depth_guard_active = bool(
        depth_guarded_transverse_intercept
        and depth_guard_alignment_observation_valid
        and not next_depth_guard_released
    )
    suppressed_depth_control = np.zeros(3, dtype=np.float64)
    if depth_guard_active:
        blended_depth = float(np.dot(blended_translation, depth_guard_axis))
        suppressed_depth_control = blended_depth * depth_guard_axis
        blended_translation = blended_translation - suppressed_depth_control
    remaining = max(1, config.horizon_steps - contact_window_step)
    translation_increment = _clip_norm(
        blended_translation / remaining, config.maximum_translation_step_m
    )
    rotation_increment = _clip_norm(
        rotation_residual / remaining, config.maximum_rotation_step_rad
    )
    nominal_translation_increment = translation_increment.copy()
    nominal_rotation_increment = rotation_increment.copy()

    active_robust = _robust_arm(forces, fractions, config)
    peer_robust = _robust_arm(peer_forces, peer_fractions, config)
    strict_four_pad_frame = bool(
        active_grasp
        and peer_grasp
        and active_robust
        and peer_robust
        and pot_motion_ok
    )
    robust_frame = bool(
        active_grasp
        and active_robust
        and pot_motion_ok
        and (not require_peer_latch or (peer_grasp and peer_robust))
    )
    next_streak = robust_streak + 1 if robust_frame else 0
    preclosure_pose_aligned = bool(
        np.linalg.norm(translation_world) <= config.closure_position_tolerance_m
        and np.linalg.norm(rotation_residual)
        <= config.closure_rotation_tolerance_rad
    )
    finite_pad_intersections = np.isfinite(fractions)
    finite_interior_pad_intersections = bool(
        np.any(
            finite_pad_intersections
            & (fractions >= config.minimum_pad_fraction_margin)
            & (fractions <= 1.0 - config.minimum_pad_fraction_margin)
        )
    )
    preclosure_geometric_prestage_enabled = bool(
        allow_dual_force_pad_margin_pivot
        or require_geometric_preseat_for_closure
    )
    preclosure_geometric_extra_budget_m = float(
        PRECLOSURE_GEOMETRIC_EXTRA_RECENTER_STEPS
        * config.maximum_contact_recenter_step_m
        if preclosure_geometric_prestage_enabled
        else 0.0
    )
    preclosure_geometric_maximum_total_m = float(
        config.maximum_contact_recenter_total_m
        + preclosure_geometric_extra_budget_m
    )
    preclosure_geometric_prestage_eligible = bool(
        preclosure_geometric_prestage_enabled
        and not closure_committed
        and np.count_nonzero(finite_pad_intersections) == 1
        and (preclosure_pose_aligned or contact_recenter_total_m > 0.0)
        and pot_motion_ok
        and peer_margin_ok
    )
    # Attempt 53 first authorized closure while the only finite pad
    # intersection was still at fraction -0.0229 and the pot was unloaded.
    # Expose that geometric intersection to the existing bounded recenter only
    # once the ordinary closure pose gate is met.  A measured recenter response
    # then latches this open-jaw prestage even if the tangent move perturbs the
    # pose residual.
    if preclosure_geometric_prestage_eligible:
        control_contacting = control_contacting | finite_pad_intersections
        control_contact_observed = bool(np.any(control_contacting))
    preclosure_geometric_uses_raw_pad_axis = bool(
        preclosure_geometric_prestage_eligible
        and not physical_contact_observed
    )
    contact_fraction_axis_world = (
        _unit(
            np.mean(axes[control_contacting], axis=0),
            "contacting pad tip-to-base axis",
        )
        if control_contact_observed
        else mean_pad_axis
    )
    contact_fraction_handle_tangent = (
        contact_fraction_axis_world
        - float(np.dot(contact_fraction_axis_world, depth_guard_axis))
        * depth_guard_axis
    )
    contact_fraction_handle_tangent_norm = float(
        np.linalg.norm(contact_fraction_handle_tangent)
    )
    surface_tangent_axis_valid = bool(
        contact_fraction_handle_tangent_norm > 1.0e-9
    )
    contact_recenter_axis_world = (
        contact_fraction_axis_world
        if preclosure_geometric_uses_raw_pad_axis
        else (
            contact_fraction_handle_tangent
            / contact_fraction_handle_tangent_norm
            if contact_recenter_use_handle_tangent and surface_tangent_axis_valid
            else contact_fraction_axis_world
        )
    )
    pre_release_margin_protection_enabled = bool(
        contact_fraction_recenter
        and contact_recenter_use_handle_tangent
        and depth_guarded_transverse_intercept
        and not next_depth_guard_released
        and control_contact_observed
    )
    protected_pad_fraction_margin = config.minimum_pad_fraction_margin
    if (
        pre_release_margin_protection_enabled
        or preclosure_geometric_prestage_eligible
    ):
        # A full transverse command can consume pad-edge margin before its
        # effect is visible on the next observation.  Maintain one maximum
        # command of measured pad-fraction reserve while the contact-normal
        # depth guard is unreleased.  This is a control reserve only: the
        # unchanged acceptance margin remains authoritative.
        protected_pad_fraction_margin = min(
            0.5,
            config.minimum_pad_fraction_margin
            + config.maximum_translation_step_m
            / active_pad_fraction_axis_extent_m,
        )
    contact_fraction_delta = 0.0
    if control_contact_observed and np.all(
        np.isfinite(fractions[control_contacting])
    ):
        fraction_corrections = []
        for fraction in fractions[control_contacting]:
            if fraction < protected_pad_fraction_margin:
                fraction_corrections.append(
                    protected_pad_fraction_margin - float(fraction)
                )
            elif fraction > 1.0 - protected_pad_fraction_margin:
                fraction_corrections.append(
                    1.0 - protected_pad_fraction_margin - float(fraction)
                )
        if fraction_corrections:
            signs = np.sign(fraction_corrections)
            if np.all(signs == signs[0]):
                contact_fraction_delta = float(
                    max(fraction_corrections, key=abs)
                )
    requested_recenter_translation_m = float(
        contact_fraction_delta * active_pad_fraction_axis_extent_m
    )
    pre_release_margin_protection_active = bool(
        pre_release_margin_protection_enabled
        and contact_fraction_delta != 0.0
        and active_margin_ok
    )
    effective_maximum_recenter_total_m = float(
        preclosure_geometric_maximum_total_m
        if preclosure_geometric_prestage_eligible
        else config.maximum_contact_recenter_total_m
    )
    remaining_recenter_m = max(
        0.0,
        effective_maximum_recenter_total_m - contact_recenter_total_m,
    )
    if (
        preclosure_geometric_prestage_eligible
        and remaining_recenter_m
        <= PRECLOSURE_GEOMETRIC_BUDGET_EXHAUSTION_TOLERANCE_M
    ):
        # Float32 wrist observations left a 24.8 nm remainder after Attempt
        # 54 consumed its measured budget.  Treat a sub-micron remainder as
        # exhausted so the controller fails closed instead of emitting an
        # ineffective command through the rest of the acquisition window.
        remaining_recenter_m = 0.0
    executed_recenter_translation_m = float(
        np.clip(
            requested_recenter_translation_m,
            -min(config.maximum_contact_recenter_step_m, remaining_recenter_m),
            min(config.maximum_contact_recenter_step_m, remaining_recenter_m),
        )
    )
    contact_recenter_active = bool(
        contact_fraction_recenter
        and control_contact_observed
        and (
            not active_margin_ok
            or pre_release_margin_protection_active
            or preclosure_geometric_prestage_eligible
        )
        and contact_fraction_delta != 0.0
        and remaining_recenter_m > 0.0
        and pot_motion_ok
        and peer_margin_ok
        and (
            not contact_recenter_use_handle_tangent
            or surface_tangent_axis_valid
            or preclosure_geometric_uses_raw_pad_axis
        )
        and (
            not depth_guarded_transverse_intercept
            or next_depth_guard_released
            or (
                contact_recenter_use_handle_tangent
                and control_contact_observed
            )
        )
    )
    nominal_depth_completion_component_m = float(
        np.dot(nominal_translation_increment, depth_guard_axis)
    )
    guarded_depth_completion_active = bool(
        contact_fraction_recenter
        and contact_recenter_use_handle_tangent
        and physical_contact_observed
        and not active_margin_ok
        and contact_fraction_delta > 0.0
        and remaining_recenter_m <= 1.0e-12
        and signed_depth_residual * nominal_depth_completion_component_m > 0.0
        and pot_motion_ok
        and peer_margin_ok
        and (
            not depth_guarded_transverse_intercept
            or next_depth_guard_released
        )
    )
    preclosure_geometric_prestage_active = bool(
        preclosure_geometric_prestage_eligible and contact_recenter_active
    )
    dual_force_backed = bool(np.all(forces >= config.minimum_force_n))
    pad_edge_margins = np.minimum(fractions, 1.0 - fractions)
    dual_force_pad_margin_pivot_active = bool(
        allow_dual_force_pad_margin_pivot
        and closure_committed
        and dual_force_backed
        and np.all(np.isfinite(pad_edge_margins))
        and np.all(np.linalg.norm(axes, axis=1) > 1.0e-9)
        and np.count_nonzero(
            pad_edge_margins < config.minimum_pad_fraction_margin - 1.0e-12
        )
        == 1
        and np.count_nonzero(
            pad_edge_margins >= config.minimum_pad_fraction_margin - 1.0e-12
        )
        == 1
        and pot_motion_ok
        and peer_margin_ok
    )
    dual_force_pad_margin_pivot_geometry_feasible = False
    dual_force_pad_margin_pivot_strong_index = None
    dual_force_pad_margin_pivot_weak_index = None
    dual_force_pad_margin_pivot_target_fraction = None
    dual_force_pad_margin_pivot_predicted_fraction = None
    dual_force_pad_margin_pivot_translation_tracking_scale = 1.0
    dual_force_pad_margin_pivot_uncompensated_translation = np.zeros(
        3, dtype=np.float64
    )
    dual_force_pad_margin_pivot_translation = np.zeros(3, dtype=np.float64)
    dual_force_pad_margin_pivot_rotation = np.zeros(3, dtype=np.float64)
    dual_force_pad_margin_pivot_point = np.zeros(3, dtype=np.float64)
    if dual_force_pad_margin_pivot_active:
        # Attempt 22 is the pair-local force-backed baseline: its ordinary
        # bounded closure captured both pads while pad 0 was broad and pad 1
        # remained at fraction 0.0268.  Once that dual-force state exists,
        # rotate the wrist about the measured broad-pad contact instead of
        # translating both contacts together.  The rigid-contact prediction
        # moves only the weak contact toward a conservative 25%/75% interior
        # target, while the ordinary Cartesian and rotation step bounds cap
        # every command.  This path is opt-in and leaves all gains and quality
        # predicates unchanged.
        weak = int(np.argmin(pad_edge_margins))
        strong = 1 - weak
        target_fraction = 0.25 if fractions[weak] < 0.5 else 0.75
        fraction_direction = float(np.sign(target_fraction - fractions[weak]))
        unit_axes = axes / np.linalg.norm(axes, axis=1)[:, None]
        contacts = centers + (
            (fractions[:, None] - 0.5)
            * active_pad_fraction_axis_extent_m
            * unit_axes
        )
        pivot_point = contacts[strong]
        separation = contacts[weak] - pivot_point
        desired_motion = -fraction_direction * unit_axes[weak]
        rotation_axis = np.cross(separation, desired_motion)
        rotation_axis_norm = float(np.linalg.norm(rotation_axis))
        if rotation_axis_norm > 1.0e-9:
            rotation_axis /= rotation_axis_norm
            coefficient_cos = float(np.dot(desired_motion, separation))
            coefficient_sin = float(
                np.dot(desired_motion, np.cross(rotation_axis, separation))
            )
            required_motion_m = float(
                abs(target_fraction - fractions[weak])
                * active_pad_fraction_axis_extent_m
            )

            def predicted_motion(angle: float) -> float:
                return float(
                    coefficient_cos * (np.cos(angle) - 1.0)
                    + coefficient_sin * np.sin(angle)
                )

            maximum_pivot_rotation_rad = 0.35
            if predicted_motion(maximum_pivot_rotation_rad) >= required_motion_m:
                low, high = 0.0, maximum_pivot_rotation_rad
                for _ in range(64):
                    midpoint = 0.5 * (low + high)
                    if predicted_motion(midpoint) < required_motion_m:
                        low = midpoint
                    else:
                        high = midpoint
                step_angle = min(high, config.maximum_rotation_step_rad)

                def pivot_translation(angle: float) -> np.ndarray:
                    delta = np.concatenate(
                        (
                            [np.cos(0.5 * angle)],
                            rotation_axis * np.sin(0.5 * angle),
                        )
                    )
                    return (
                        pivot_point
                        + quaternion_rotate(delta, wrist[:3] - pivot_point)
                        - wrist[:3]
                    )

                if (
                    np.linalg.norm(pivot_translation(step_angle))
                    > config.maximum_translation_step_m
                ):
                    low, high = 0.0, step_angle
                    for _ in range(64):
                        midpoint = 0.5 * (low + high)
                        if (
                            np.linalg.norm(pivot_translation(midpoint))
                            <= config.maximum_translation_step_m
                        ):
                            low = midpoint
                        else:
                            high = midpoint
                    step_angle = low
                dual_force_pad_margin_pivot_geometry_feasible = True
                dual_force_pad_margin_pivot_strong_index = strong
                dual_force_pad_margin_pivot_weak_index = weak
                dual_force_pad_margin_pivot_target_fraction = target_fraction
                dual_force_pad_margin_pivot_predicted_fraction = float(
                    fractions[weak]
                    + fraction_direction
                    * predicted_motion(step_angle)
                    / active_pad_fraction_axis_extent_m
                )
                dual_force_pad_margin_pivot_uncompensated_translation = (
                    pivot_translation(step_angle)
                )
                # Attempt 52 measured the first compensated-pivot response:
                # translation realized 65.1% of its command while rotation
                # realized only 24.9%.  The resulting excess translation moved
                # the pot through the unchanged 3 mm guard.  Scale only this
                # pair-opted pivot translation by 40%, the bounded measured
                # rotation/translation tracking ratio, so the physical swept
                # motion follows the same strong-contact pivot.  Controller
                # and IK gains remain untouched.
                dual_force_pad_margin_pivot_translation_tracking_scale = 0.4
                dual_force_pad_margin_pivot_translation = (
                    dual_force_pad_margin_pivot_translation_tracking_scale
                    * dual_force_pad_margin_pivot_uncompensated_translation
                )
                dual_force_pad_margin_pivot_rotation = rotation_axis * step_angle
                dual_force_pad_margin_pivot_point = pivot_point.copy()
        dual_force_pad_margin_pivot_active = bool(
            dual_force_pad_margin_pivot_geometry_feasible
        )
    # ``contact_recenter_total_m`` is the realized positive axial wrist motion
    # accumulated by the runtime from the preceding commands.  Do not charge
    # this physical-motion budget for a command before its realization is
    # observed on the next control frame.
    next_contact_recenter_total_m = float(contact_recenter_total_m)
    fail_reason = None
    if not pot_motion_ok:
        fail_reason = "pre_peer_pot_motion_exceeded"
    elif (
        preclosure_geometric_prestage_eligible
        and contact_fraction_delta != 0.0
        and remaining_recenter_m <= 1.0e-12
    ):
        fail_reason = "preclosure_geometric_prestage_budget_exhausted"
    elif not active_margin_ok and not (
        contact_recenter_active
        or guarded_depth_completion_active
        or dual_force_pad_margin_pivot_active
    ):
        fail_reason = "active_contact_outside_pad_margin"
    elif not peer_margin_ok:
        fail_reason = "peer_contact_outside_pad_margin"
    fail_closed = fail_reason is not None
    interior_single_pad_triggered = bool(
        allow_interior_single_pad_closure
        and np.count_nonzero(contacting) == 1
        and active_margin_ok
        and pot_motion_ok
    )
    interior_single_pad_closure_hold_active = bool(
        allow_interior_single_pad_closure
        and (interior_single_pad_triggered or closure_committed)
        and not dual_force_backed
        and not fail_closed
    )
    transverse_aligned_two_pad_triggered = bool(
        allow_transverse_aligned_two_pad_closure
        and depth_guarded_transverse_intercept
        and next_depth_guard_released
        and transverse_aligned
        and np.count_nonzero(contacting) == 1
        and active_margin_ok
        and pot_motion_ok
    )
    transverse_aligned_two_pad_closure_hold_active = bool(
        allow_transverse_aligned_two_pad_closure
        and (transverse_aligned_two_pad_triggered or closure_committed)
        and not dual_force_backed
        and not fail_closed
    )

    retained_transverse_translation = np.zeros(3, dtype=np.float64)
    recenter_world_command_budget_m = config.maximum_translation_step_m
    if contact_recenter_active:
        # contact_pad_fraction is measured from finger tip (0) toward finger
        # base (1).  Translating the finger opposite its tip->base axis moves a
        # fixed world contact baseward in the finger frame, increasing a low
        # fraction; the sign reverses naturally for a high fraction.
        contact_recenter_translation = (
            -executed_recenter_translation_m * contact_recenter_axis_world
        )
        if contact_recenter_preserve_transverse_centering:
            if preclosure_geometric_prestage_eligible:
                # The geometric preseat is deliberately a single-axis move.
                # Retaining the nominal Cartesian step here would combine the
                # correction with the same approach motion that exposed the
                # edge-only intersection in Attempt 53.
                retained_transverse_translation = np.zeros(3, dtype=np.float64)
            elif contact_recenter_use_handle_tangent:
                handle_tangent_translation = translation_increment - float(
                    np.dot(translation_increment, depth_guard_axis)
                ) * depth_guard_axis
                retained_transverse_translation = (
                    handle_tangent_translation
                    - float(
                        np.dot(
                            handle_tangent_translation,
                            contact_recenter_axis_world,
                        )
                    )
                    * contact_recenter_axis_world
                )
            else:
                retained_transverse_translation = translation_increment - float(
                    np.dot(translation_increment, contact_fraction_axis_world)
                ) * contact_fraction_axis_world
        if contact_recenter_use_handle_tangent:
            recenter_world_command_budget_m = min(
                config.maximum_translation_step_m,
                remaining_recenter_m,
            )
        translation_increment = _clip_norm(
            contact_recenter_translation + retained_transverse_translation,
            recenter_world_command_budget_m,
        )
        rotation_increment = np.zeros(3, dtype=np.float64)
    guarded_depth_completion_translation = np.zeros(3, dtype=np.float64)
    if guarded_depth_completion_active:
        # The bounded tangent sweep can bring a low, force-backed fingertip
        # intersection to the handle opening without yet placing the pad's
        # unchanged 10% margin over the surface.  Once that existing 12 mm
        # budget is physically exhausted, finish the still-open approach only
        # along the live handle normal.  The ordinary 4 mm Cartesian bound and
        # 3 mm pre-peer pot-motion guard remain authoritative, and orientation
        # and jaw commands stay fixed until the original broad-contact pose
        # gate is satisfied.
        guarded_depth_completion_translation = (
            nominal_depth_completion_component_m * depth_guard_axis
        )
        translation_increment = guarded_depth_completion_translation.copy()
        rotation_increment = np.zeros(3, dtype=np.float64)
    if robust_frame or fail_closed:
        translation_increment = np.zeros(3, dtype=np.float64)
        rotation_increment = np.zeros(3, dtype=np.float64)
    if interior_single_pad_transverse_intercept_active and not fail_closed:
        # A lone interior pad can touch while the open jaw is still
        # transversely offset from the handle center.  Keep the existing
        # depth guard active so the commanded translation remains tangent to
        # the contact, and hold orientation so that the loaded pad is not
        # swept around the handle.  Closure remains governed by the unchanged
        # two-sided pose gate below.
        rotation_increment = np.zeros(3, dtype=np.float64)
    if (
        interior_single_pad_closure_hold_active
        or transverse_aligned_two_pad_closure_hold_active
    ):
        # A force-backed interior pad is stronger near-contact evidence than
        # the distant nominal contact-frame residual.  Keep that loaded pad
        # fixed while the existing bounded jaw stroke brings in its peer.
        translation_increment = np.zeros(3, dtype=np.float64)
        rotation_increment = np.zeros(3, dtype=np.float64)
    aligned_for_closure = bool(
        np.linalg.norm(translation_world) <= config.closure_position_tolerance_m
        and np.linalg.norm(rotation_residual) <= config.closure_rotation_tolerance_rad
    )
    closure_authorized = bool(
        (
            aligned_for_closure
            or (allow_bounded_closure_commit and closure_committed)
            or interior_single_pad_triggered
            or transverse_aligned_two_pad_triggered
        )
        and (
            not require_geometric_preseat_for_closure
            or finite_interior_pad_intersections
        )
    )
    pause_committed_closure = bool(
        pause_committed_closure_on_dual_force_backing
        and closure_committed
        and dual_force_backed
    )
    jaw_increment = (
        float(
            np.clip(
                (config.closed_jaw_command - current_jaw_command) / remaining,
                -config.maximum_jaw_step,
                config.maximum_jaw_step,
            )
        )
        if (
            closure_authorized
            and not pause_committed_closure
            and not fail_closed
            and not robust_frame
        )
        else 0.0
    )
    if contact_recenter_active and not contact_recenter_preserve_bounded_closure:
        jaw_increment = 0.0
    if preclosure_geometric_prestage_active:
        # This is an open-jaw correction.  Do not let the legacy
        # preserve-bounded-closure option override it: closing on the measured
        # edge was the earliest causal quality failure in Attempt 53.
        jaw_increment = 0.0
    bounded_closure_priority_active = bool(
        contact_recenter_active
        and contact_recenter_preserve_bounded_closure
        and jaw_increment != 0.0
        and not preclosure_geometric_prestage_active
    )
    next_closure_committed = bool(
        allow_bounded_closure_commit
        and (closure_committed or jaw_increment != 0.0)
    )
    if bounded_closure_priority_active:
        translation_increment = nominal_translation_increment
        rotation_increment = nominal_rotation_increment
    loaded_pad_pivot_closure_active = bool(
        transverse_aligned_closure_pivot_pad_index is not None
        and transverse_aligned_two_pad_closure_hold_active
        and jaw_increment != 0.0
        and not fail_closed
        and not dual_force_backed
    )
    loaded_pad_pivot_translation = np.zeros(3, dtype=np.float64)
    loaded_pad_pivot_jaw_scale = 1.0
    if loaded_pad_pivot_closure_active:
        # Pair 15 measured that a 4 mm Cartesian pivot command realizes only
        # 1.557 mm along the live jaw axis.  A simultaneous 4 mm jaw command
        # retracts the loaded finger 1.663 mm on the first frame and 2.769 mm
        # on the next, immediately dropping its force.  Across all four finite
        # geometric-contact responses the minimum measured wrist-to-finger
        # ratio is 0.441.  Retain the bounded 4 mm wrist pivot but scale only
        # this pair-opted jaw increment to 40%, keeping positive loaded-pad
        # preload while the peer pad closes.
        loaded_pad_pivot_jaw_scale = 0.4
        unscaled_jaw_increment = jaw_increment
        jaw_increment *= loaded_pad_pivot_jaw_scale
        pivot_sign = (
            1.0 if transverse_aligned_closure_pivot_pad_index == 1 else -1.0
        )
        loaded_pad_pivot_translation = (
            pivot_sign * unscaled_jaw_increment * jaw_axis
        )
        translation_increment = _clip_norm(
            loaded_pad_pivot_translation,
            config.maximum_translation_step_m,
        )
        rotation_increment = np.zeros(3, dtype=np.float64)
    if dual_force_pad_margin_pivot_active:
        translation_increment = dual_force_pad_margin_pivot_translation.copy()
        rotation_increment = dual_force_pad_margin_pivot_rotation.copy()
        jaw_increment = 0.0
    pre_peer_motion_remaining_m = max(
        0.0,
        config.maximum_pre_peer_pot_motion_m - pre_peer_pot_displacement_m,
    )
    pre_peer_motion_budgeted_closure_active = bool(
        budget_committed_closure_by_pre_peer_motion
        and closure_committed
        and np.count_nonzero(contacting) == 1
        and active_margin_ok
        and jaw_increment != 0.0
        and not fail_closed
    )
    pre_peer_motion_unbudgeted_translation = translation_increment.copy()
    pre_peer_motion_unbudgeted_rotation = rotation_increment.copy()
    pre_peer_motion_unbudgeted_jaw_increment = jaw_increment
    pre_peer_motion_control_scale = 1.0
    unbudgeted_translation_norm_m = float(
        np.linalg.norm(pre_peer_motion_unbudgeted_translation)
    )
    if (
        pre_peer_motion_budgeted_closure_active
        and unbudgeted_translation_norm_m > pre_peer_motion_remaining_m
    ):
        # Attempt 55 issued another full 4 mm Cartesian correction with only
        # 0.398 mm left under the unchanged 3 mm pre-peer object-motion guard.
        # The resulting observation reached dual force one frame too late,
        # after the pot had already moved 3.621 mm.  Conservatively treat the
        # remaining object-motion allowance as a prospective upper bound on
        # the simultaneous closure actuation.  Attempt 56 proved that scaling
        # only the wrist correction leaves the 4 mm jaw stroke free to move
        # the pot another 0.996 mm.  Scale translation, rotation, and jaw
        # together so the closure path is preserved, while the ordinary guard
        # still fails closed on the next measured observation.
        pre_peer_motion_control_scale = float(
            pre_peer_motion_remaining_m / unbudgeted_translation_norm_m
        )
        translation_increment *= pre_peer_motion_control_scale
        rotation_increment *= pre_peer_motion_control_scale
        jaw_increment *= pre_peer_motion_control_scale
    actual_recenter_translation_m = (
        max(
            0.0,
            -float(np.dot(translation_increment, contact_recenter_axis_world)),
        )
        if contact_recenter_active and not bounded_closure_priority_active
        else 0.0
    )
    reported_recenter_translation_m = (
        actual_recenter_translation_m
        if contact_recenter_use_handle_tangent
        else (
            executed_recenter_translation_m
            if contact_recenter_active and not bounded_closure_priority_active
            else 0.0
        )
    )
    target = wrist.copy()
    target[:3] += translation_increment
    target = _apply_axis_angle(target, rotation_increment)
    jaw_command = float(current_jaw_command + jaw_increment)

    translation_plan = np.repeat(translation_increment[None], config.horizon_steps, axis=0)
    rotation_plan = np.repeat(rotation_increment[None], config.horizon_steps, axis=0)
    jaw_plan = np.repeat(jaw_increment, config.horizon_steps)
    fraction_residual = [
        None if not np.isfinite(value) else float(0.5 - value) for value in fractions
    ]
    receipt = {
        "schema_version": 1,
        "controller": "deterministic_handle_local_receding_horizon_mpc_lite",
        "contact_window_step": int(contact_window_step),
        "horizon_steps": int(config.horizon_steps),
        "source_prior_weight": prior_weight,
        "joint_nominal_weight": 0.0,
        "observed_frames": {
            "pot_body": pot.tolist(),
            "active_handle_contact": handle.tolist(),
            "active_wrist": wrist.tolist(),
            "object_relative_wrist_prior": prior.tolist(),
            "object_relative_jaw_axis_prior": jaw_axis_prior_local.tolist(),
            "object_relative_pad_depth_axis_prior": pad_axis_prior_local.tolist(),
            "active_pad_centers": centers.tolist(),
            "active_pad_axes": axes.tolist(),
            "jaw_midpoint": jaw_midpoint.tolist(),
            "jaw_axis": jaw_axis.tolist(),
            "mean_pad_depth_axis": mean_pad_axis.tolist(),
        },
        "signed_residuals": {
            "jaw_midpoint_to_handle_world_m": translation_world.tolist(),
            "jaw_midpoint_to_handle_local_m": translation_local.tolist(),
            "source_warm_start_world_m": warm_residual.tolist(),
            "wrist_axis_angle_world_rad": wrist_rotation.tolist(),
            "jaw_axis_alignment_world_rad": jaw_alignment.tolist(),
            "pad_axis_alignment_world_rad": pad_alignment.tolist(),
            "pad_fraction_to_center": fraction_residual,
            "force_to_minimum_n": np.maximum(0.0, config.minimum_force_n - forces).tolist(),
        },
        "planned_controls": {
            "translation_world_m": translation_plan.tolist(),
            "rotation_axis_angle_world_rad": rotation_plan.tolist(),
            "jaw_increment": jaw_plan.tolist(),
        },
        "executed_control": {
            "translation_world_m": translation_increment.tolist(),
            "rotation_axis_angle_world_rad": rotation_increment.tolist(),
            "jaw_increment": jaw_increment,
            "wrist_target_pose": target.tolist(),
            "jaw_command": jaw_command,
        },
        "hard_constraints": {
            "pre_peer_pot_motion_m": float(pre_peer_pot_displacement_m),
            "pre_peer_pot_motion_within_limit": pot_motion_ok,
            "active_contact_pad_margin_valid": active_margin_ok,
            "peer_contact_pad_margin_valid": peer_margin_ok,
            "translation_step_within_bound": bool(
                np.linalg.norm(translation_increment)
                <= config.maximum_translation_step_m + 1.0e-12
            ),
            "rotation_step_within_bound": bool(
                np.linalg.norm(rotation_increment)
                <= config.maximum_rotation_step_rad + 1.0e-12
            ),
            "jaw_step_within_bound": bool(
                abs(jaw_increment) <= config.maximum_jaw_step + 1.0e-12
            ),
        },
        "contact_frame_guard": {
            "enabled": bool(depth_guarded_transverse_intercept),
            "active": depth_guard_active,
            "depth_axis_source": (
                "observed_handle_contact_normal"
                if physical_contact_observed and use_contact_normal_depth_axis
                else "latched_observed_handle_contact_normal"
                if handle_contact_normal_latched
                else "mean_pad_depth_axis"
            ),
            "depth_axis_world": depth_guard_axis.tolist(),
            "physical_contact_observed": physical_contact_observed,
            "interior_single_pad_transverse_intercept_enabled": bool(
                allow_interior_single_pad_transverse_intercept
            ),
            "interior_single_pad_transverse_intercept_active": (
                interior_single_pad_transverse_intercept_active
            ),
            "rotation_held_during_interior_single_pad_intercept": bool(
                interior_single_pad_transverse_intercept_active
                and np.allclose(rotation_increment, 0.0, atol=1.0e-12)
            ),
            "transverse_tolerance_m": config.depth_guard_transverse_tolerance_m,
            "transverse_residual_world_m": transverse_residual.tolist(),
            "transverse_residual_norm_m": float(
                np.linalg.norm(transverse_residual)
            ),
            "signed_depth_residual_m": signed_depth_residual,
            "suppressed_depth_control_world_m": (
                suppressed_depth_control.tolist()
            ),
            "transverse_aligned": transverse_aligned,
            "alignment_consecutive_frames": next_depth_guard_streak,
            "release_required_consecutive_frames": config.depth_guard_release_steps,
            "released": next_depth_guard_released,
        },
        "contact_fraction_recenter": {
            "budget_accounting": "measured_positive_axial_wrist_displacement",
            "enabled": bool(contact_fraction_recenter),
            "active": contact_recenter_active,
            "surface_tangent_enabled": bool(
                contact_recenter_use_handle_tangent
            ),
            "surface_tangent_axis_valid": surface_tangent_axis_valid,
            "surface_tangent_axis_world": (
                contact_recenter_axis_world.tolist()
            ),
            "guarded_depth_completion_enabled": bool(
                contact_recenter_use_handle_tangent
            ),
            "guarded_depth_completion_active": (
                guarded_depth_completion_active
            ),
            "guarded_depth_completion_translation_world_m": (
                guarded_depth_completion_translation.tolist()
            ),
            "preserve_transverse_centering": bool(
                contact_recenter_preserve_transverse_centering
            ),
            "preserve_bounded_closure": bool(
                contact_recenter_preserve_bounded_closure
            ),
            "bounded_closure_priority_active": bounded_closure_priority_active,
            "preclosure_geometric_prestage_enabled": (
                preclosure_geometric_prestage_enabled
            ),
            "preclosure_geometric_prestage_active": (
                preclosure_geometric_prestage_active
            ),
            "preclosure_geometric_uses_raw_pad_axis": (
                preclosure_geometric_uses_raw_pad_axis
            ),
            "finger_tip_to_base_axis_world": contact_fraction_axis_world.tolist(),
            "pad_fraction_axis_extent_m": float(
                active_pad_fraction_axis_extent_m
            ),
            "contact_fraction_delta": contact_fraction_delta,
            "pre_release_margin_protection_enabled": (
                pre_release_margin_protection_enabled
            ),
            "pre_release_margin_protection_active": (
                pre_release_margin_protection_active
            ),
            "protected_pad_fraction_margin": protected_pad_fraction_margin,
            "acceptance_pad_fraction_margin": (
                config.minimum_pad_fraction_margin
            ),
            "requested_translation_m": requested_recenter_translation_m,
            "executed_translation_m": (
                reported_recenter_translation_m
            ),
            "budgeted_axial_translation_world_m": (
                -actual_recenter_translation_m * contact_recenter_axis_world
            ).tolist(),
            "retained_transverse_translation_world_m": (
                retained_transverse_translation.tolist()
            ),
            "executed_handle_normal_component_m": float(
                np.dot(translation_increment, depth_guard_axis)
            ),
            "world_command_budget_m": recenter_world_command_budget_m,
            "world_command_norm_m": float(
                np.linalg.norm(translation_increment)
            ),
            "total_translation_m": next_contact_recenter_total_m,
            "maximum_step_m": config.maximum_contact_recenter_step_m,
            "base_maximum_total_m": (
                config.maximum_contact_recenter_total_m
            ),
            "maximum_total_m": config.maximum_contact_recenter_total_m,
            "effective_maximum_total_m": (
                effective_maximum_recenter_total_m
            ),
            "preclosure_geometric_extra_budget_m": (
                preclosure_geometric_extra_budget_m
            ),
            "preclosure_geometric_maximum_total_m": (
                preclosure_geometric_maximum_total_m
            ),
            "preclosure_geometric_budget_exhaustion_tolerance_m": (
                PRECLOSURE_GEOMETRIC_BUDGET_EXHAUSTION_TOLERANCE_M
            ),
        },
        "closure": {
            "commit_enabled": bool(allow_bounded_closure_commit),
            "was_committed": bool(closure_committed),
            "initial_alignment_satisfied": aligned_for_closure,
            "interior_single_pad_trigger_enabled": bool(
                allow_interior_single_pad_closure
            ),
            "interior_single_pad_triggered": interior_single_pad_triggered,
            "interior_single_pad_closure_hold_active": (
                interior_single_pad_closure_hold_active
            ),
            "wrist_frozen_for_interior_single_pad_closure": bool(
                interior_single_pad_closure_hold_active
                and np.allclose(translation_increment, 0.0, atol=1.0e-12)
                and np.allclose(rotation_increment, 0.0, atol=1.0e-12)
            ),
            "transverse_aligned_two_pad_trigger_enabled": bool(
                allow_transverse_aligned_two_pad_closure
            ),
            "transverse_aligned_two_pad_triggered": (
                transverse_aligned_two_pad_triggered
            ),
            "transverse_aligned_two_pad_closure_hold_active": (
                transverse_aligned_two_pad_closure_hold_active
            ),
            "wrist_frozen_for_transverse_aligned_two_pad_closure": bool(
                transverse_aligned_two_pad_closure_hold_active
                and np.allclose(translation_increment, 0.0, atol=1.0e-12)
                and np.allclose(rotation_increment, 0.0, atol=1.0e-12)
            ),
            "loaded_pad_pivot_closure_enabled": bool(
                transverse_aligned_closure_pivot_pad_index is not None
            ),
            "loaded_pad_pivot_closure_active": (
                loaded_pad_pivot_closure_active
            ),
            "loaded_pad_pivot_index": (
                transverse_aligned_closure_pivot_pad_index
            ),
            "loaded_pad_pivot_translation_world_m": (
                loaded_pad_pivot_translation.tolist()
            ),
            "loaded_pad_pivot_translation_norm_m": float(
                np.linalg.norm(loaded_pad_pivot_translation)
            ),
            "loaded_pad_pivot_jaw_scale": loaded_pad_pivot_jaw_scale,
            "dual_force_pad_margin_pivot_enabled": bool(
                allow_dual_force_pad_margin_pivot
            ),
            "dual_force_pad_margin_pivot_active": (
                dual_force_pad_margin_pivot_active
            ),
            "dual_force_pad_margin_pivot_geometry_feasible": (
                dual_force_pad_margin_pivot_geometry_feasible
            ),
            "dual_force_pad_margin_pivot_strong_index": (
                dual_force_pad_margin_pivot_strong_index
            ),
            "dual_force_pad_margin_pivot_weak_index": (
                dual_force_pad_margin_pivot_weak_index
            ),
            "dual_force_pad_margin_pivot_target_fraction": (
                dual_force_pad_margin_pivot_target_fraction
            ),
            "dual_force_pad_margin_pivot_predicted_fraction": (
                dual_force_pad_margin_pivot_predicted_fraction
            ),
            "dual_force_pad_margin_pivot_translation_tracking_scale": (
                dual_force_pad_margin_pivot_translation_tracking_scale
            ),
            "dual_force_pad_margin_pivot_uncompensated_translation_world_m": (
                dual_force_pad_margin_pivot_uncompensated_translation.tolist()
            ),
            "dual_force_pad_margin_pivot_translation_world_m": (
                dual_force_pad_margin_pivot_translation.tolist()
            ),
            "dual_force_pad_margin_pivot_rotation_axis_angle_world_rad": (
                dual_force_pad_margin_pivot_rotation.tolist()
            ),
            "dual_force_pad_margin_pivot_point_world_m": (
                dual_force_pad_margin_pivot_point.tolist()
            ),
            "pre_peer_motion_budgeted_closure_enabled": bool(
                budget_committed_closure_by_pre_peer_motion
            ),
            "pre_peer_motion_budgeted_closure_active": (
                pre_peer_motion_budgeted_closure_active
            ),
            "pre_peer_motion_remaining_m": pre_peer_motion_remaining_m,
            "pre_peer_motion_control_scale": pre_peer_motion_control_scale,
            "pre_peer_motion_unbudgeted_translation_world_m": (
                pre_peer_motion_unbudgeted_translation.tolist()
            ),
            "pre_peer_motion_unbudgeted_rotation_axis_angle_world_rad": (
                pre_peer_motion_unbudgeted_rotation.tolist()
            ),
            "pre_peer_motion_unbudgeted_jaw_increment": (
                pre_peer_motion_unbudgeted_jaw_increment
            ),
            "geometric_preseat_required": bool(
                require_geometric_preseat_for_closure
            ),
            "geometric_preseat_satisfied": (
                finite_interior_pad_intersections
            ),
            "geometric_preseat_finite_pad_count": int(
                np.count_nonzero(finite_pad_intersections)
            ),
            "geometric_preseat_interior_pad_count": int(
                np.count_nonzero(
                    finite_pad_intersections
                    & (fractions >= config.minimum_pad_fraction_margin)
                    & (fractions <= 1.0 - config.minimum_pad_fraction_margin)
                )
            ),
            "increment_active": bool(jaw_increment != 0.0),
            "committed": next_closure_committed,
            "closed_command_reached": bool(
                abs(jaw_command - config.closed_jaw_command) <= 1.0e-12
            ),
            "dual_force_backed": dual_force_backed,
            "paused_on_dual_force_backing": pause_committed_closure,
        },
        "latch": {
            "active_force_and_margin": active_robust,
            "peer_force_and_margin": peer_robust,
            "peer_required": bool(require_peer_latch),
            "robust_frame": robust_frame,
            "strict_four_pad_frame": strict_four_pad_frame,
            "consecutive_frames": next_streak,
            "required_consecutive_frames": config.robust_latch_steps,
            "ready": next_streak >= config.robust_latch_steps,
        },
        "fail_closed": fail_closed,
        "fail_reason": fail_reason,
    }
    if not handle_local_mpc_frame_receipt_complete(receipt):
        raise AssertionError("incomplete handle-local MPC frame receipt")
    return HandleLocalMpcCommand(
        wrist_target_pose=target,
        jaw_command=jaw_command,
        robust_streak=next_streak,
        robust_latch_ready=next_streak >= config.robust_latch_steps,
        depth_guard_alignment_streak=next_depth_guard_streak,
        depth_guard_released=next_depth_guard_released,
        contact_recenter_total_m=next_contact_recenter_total_m,
        closure_committed=next_closure_committed,
        fail_closed=fail_closed,
        fail_reason=fail_reason,
        frame_receipt=receipt,
    )


def handle_local_mpc_config_receipt(config: HandleLocalMpcConfig) -> dict[str, Any]:
    return {"schema_version": 1, **asdict(config)}
