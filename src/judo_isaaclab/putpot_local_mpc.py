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
        )
        if not np.all(np.isfinite(positive)) or any(value <= 0.0 for value in positive):
            raise ValueError("handle-local MPC positive bounds must be finite")
        if not 0.0 <= self.minimum_pad_fraction_margin <= 0.5:
            raise ValueError("pad-fraction margin must be in [0, 0.5]")
        if self.robust_latch_steps < 1 or self.source_prior_decay_steps < 1:
            raise ValueError("latch and source-prior decay steps must be positive")
        if not 0.0 <= self.source_prior_initial_weight < 1.0:
            raise ValueError("source prior must be a bounded warm-start weight")


@dataclass(frozen=True)
class HandleLocalMpcCommand:
    wrist_target_pose: np.ndarray
    jaw_command: float
    robust_streak: int
    robust_latch_ready: bool
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
    config: HandleLocalMpcConfig = HandleLocalMpcConfig(),
) -> HandleLocalMpcCommand:
    """Plan and return the first bounded active-wrist and jaw control increment."""

    if contact_window_step < 0 or robust_streak < 0:
        raise ValueError("contact-window step and robust streak must be nonnegative")
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
    scalars = np.asarray(
        [pre_peer_pot_displacement_m, current_jaw_command], dtype=np.float64
    )
    if not np.all(np.isfinite(scalars)):
        raise ValueError("pot displacement and jaw command must be finite")

    jaw_midpoint = np.mean(centers, axis=0)
    jaw_axis = _unit(centers[1] - centers[0], "jaw closing line")
    mean_pad_axis = _unit(np.mean(axes, axis=0), "mean pad depth axis")
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
    remaining = max(1, config.horizon_steps - contact_window_step)
    translation_increment = _clip_norm(
        blended_translation / remaining, config.maximum_translation_step_m
    )
    rotation_increment = _clip_norm(
        rotation_residual / remaining, config.maximum_rotation_step_rad
    )

    active_margin_ok = _pad_margin_ok(forces, fractions, config)
    peer_margin_ok = _pad_margin_ok(peer_forces, peer_fractions, config)
    pot_motion_ok = bool(
        pre_peer_pot_displacement_m
        <= config.maximum_pre_peer_pot_motion_m + 1.0e-12
    )
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
    fail_reason = None
    if not pot_motion_ok:
        fail_reason = "pre_peer_pot_motion_exceeded"
    elif not active_margin_ok:
        fail_reason = "active_contact_outside_pad_margin"
    elif not peer_margin_ok:
        fail_reason = "peer_contact_outside_pad_margin"
    fail_closed = fail_reason is not None

    if robust_frame or fail_closed:
        translation_increment = np.zeros(3, dtype=np.float64)
        rotation_increment = np.zeros(3, dtype=np.float64)
    aligned_for_closure = bool(
        np.linalg.norm(translation_world) <= config.closure_position_tolerance_m
        and np.linalg.norm(rotation_residual) <= config.closure_rotation_tolerance_rad
    )
    jaw_increment = (
        float(
            np.clip(
                (config.closed_jaw_command - current_jaw_command) / remaining,
                -config.maximum_jaw_step,
                config.maximum_jaw_step,
            )
        )
        if aligned_for_closure and not fail_closed and not robust_frame
        else 0.0
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
        fail_closed=fail_closed,
        fail_reason=fail_reason,
        frame_receipt=receipt,
    )


def handle_local_mpc_config_receipt(config: HandleLocalMpcConfig) -> dict[str, Any]:
    return {"schema_version": 1, **asdict(config)}
