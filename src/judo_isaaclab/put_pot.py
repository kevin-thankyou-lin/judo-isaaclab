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
TRANSPORT_CONTACT_REANCHOR_MIN_STEPS = 10
HANDLE_PAD_GEOMETRIC_MARGIN_M = 0.003
# PutPot012 attempt_011 finished its simultaneous close 7 mm shallow along
# the commanded local pad axis.  After switching to the local authored handle
# tangent, attempt_014 began transport at only 0.0006 normalized pad depth and
# drifted below the tip.  Its preceding 2 mm command produced +0.024 normalized
# response, so another 4 mm gives a measured transport margin while remaining
# well inside the 68 mm pad.
HANDLE_PAD_TRACKING_COMPENSATION_M = 0.013
HANDLE_PAD_DEPTH_MARGIN_M = (
    HANDLE_PAD_GEOMETRIC_MARGIN_M + HANDLE_PAD_TRACKING_COMPENSATION_M
)
YAM_LEFT_FINGER_PIVOT_LOCAL_M = np.asarray([-0.045060, 0.024000, 0.054560])
YAM_FINGER_SEPARATION_LOCAL_M = np.asarray([0.090120, -0.048000, 0.0])
YAM_RIGHT_FINGER_PIVOT_LOCAL_M = (
    YAM_LEFT_FINGER_PIVOT_LOCAL_M + YAM_FINGER_SEPARATION_LOCAL_M
)
YAM_FINGER_PAD_AXIS_LENGTH_M = float(np.linalg.norm([0.068, 0.0, -0.003]))
HANDLE_PAD_RELATIVE_DEPTH_M = 0.003
HANDLE_JAW_CENTERING_LIMIT_M = 0.040
# Pot020 attempt_005 tracked its jaw correction within 7 mm but left the
# missing finger near -0.04 along the pad axis.  Its authored positive depth
# imbalance is +37.5 mm at 49.4% retained transverse thickness, so it needs the
# same bounded 2 mm positive pivot extension as the thinner Pot019 handle.
THIN_HANDLE_BALANCE_RATIO = 0.50
# Pot020 attempts 002-003 showed that position symmetry at 49.4% retained
# transverse thickness changed a sustained one-finger contact into a complete
# close-window miss.  Keep the position mirror restricted to the 42.1%-retained
# Pot019 geometry where it produced a hash-verified bimanual success.
THIN_HANDLE_SYMMETRY_RATIO = 0.45
THIN_HANDLE_POSITIVE_BALANCE_EXTRA_M = 0.002
HALF_THIN_HANDLE_POSITIVE_BALANCE_LIMIT_M = 0.005
MISSING_FINGER_CONTACT_STEP_M = 0.001
MISSING_FINGER_JAW_AXIS_MIN_M = 0.050
# Pot020 attempt_004 reached the former 40 mm cap with the missing finger's
# contact at -0.043 of the authored 68.1 mm YAM pad axis: 2.93 mm tip-side.
# Add the measured residual plus a 2 mm on-pad margin.
MISSING_FINGER_CONTACT_LIMIT_M = 0.045
MISSING_FINGER_CONTACT_DELAY_STEPS = 10
MISSING_FINGER_PAD_DEPTH_STEP_M = 0.001
MISSING_FINGER_PAD_DEPTH_LIMIT_M = 0.010
MISSING_FINGER_PAD_TARGET_FRACTION = 0.02
SINGLE_FINGER_CONTACT_LATCH_FRACTION = 0.20
SINGLE_FINGER_CONTACT_LATCH_STEPS = CONTACT_FEEDBACK_HORIZON_STEPS
MISSING_FINGER_CONTACT_SETTLE_STEPS = (
    MISSING_FINGER_CONTACT_DELAY_STEPS
    + int(np.ceil(MISSING_FINGER_CONTACT_LIMIT_M / MISSING_FINGER_CONTACT_STEP_M))
    + int(np.ceil(MISSING_FINGER_PAD_DEPTH_LIMIT_M / MISSING_FINGER_PAD_DEPTH_STEP_M))
    + 15
)


def _linear_contact_feedback_poses(
    start: Any,
    target: Any,
    steps: int,
    *,
    horizon_steps: int = CONTACT_FEEDBACK_HORIZON_STEPS,
) -> np.ndarray:
    """Close a measured contact residual over a fixed feedback horizon.

    Contact feedback is recomputed every controller step.  A quintic profile
    would restart at zero velocity on every update and defer almost the entire
    correction to the last frame.  Closing over a short fixed horizon also
    tracks a handle that moves when the opposite gripper contacts the pot.
    """

    if steps < 1:
        raise ValueError("contact feedback steps must be positive")
    if horizon_steps < 1:
        raise ValueError("contact feedback horizon must be positive")
    start_pose = _pose(start, "start")
    target_pose = _pose(target, "target")
    horizon = min(horizon_steps, steps)
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


def transfer_handle_approach_orientation(
    pose: Any,
    source_root: Any,
    target_root: Any,
    source_contact_frame: Any,
    target_contact_frame: Any,
) -> np.ndarray:
    """Orient an approach through the same local frame as its grasp.

    A curved handle can place a pregrasp wrist nearest to a different authored
    segment than the eventual contact.  Reusing the grasp contact frame keeps
    the source approach-to-grasp rotation continuous after asset transfer.
    """

    value = _pose(pose, "pose")
    oriented = transfer_pose(
        value,
        compose_pose(source_root, source_contact_frame),
        compose_pose(target_root, target_contact_frame),
    )
    result = value.copy()
    result[3:] = oriented[3:]
    return result


def transfer_handle_pose_through_contact_frames(
    pose: Any,
    source_root: Any,
    target_root: Any,
    source_contact_frame: Any,
    target_contact_frame: Any,
) -> np.ndarray:
    """Preserve the measured object-to-gripper transform at a local handle part.

    Whole-handle bounds are useful for bootstrapping which authored collision
    segment corresponds to the demonstrated contact.  They are not a stable
    position frame for thin or strongly curved target handles.  Once the local
    source and target segments are identified, transfer the complete pose
    through those object-local frames so position and orientation use the same
    correspondence.
    """

    return transfer_pose(
        _pose(pose, "pose"),
        compose_pose(_pose(source_root, "source_root"), source_contact_frame),
        compose_pose(_pose(target_root, "target_root"), target_contact_frame),
    )


def expand_handle_pregrasp_clearance(
    pregrasp_pose: Any,
    grasp_pose: Any,
    source_handle_size: Any,
    target_handle_size: Any,
    handle_axis: int,
) -> np.ndarray:
    """Keep open fingers clear when the target handle is transversely smaller."""

    pregrasp = _pose(pregrasp_pose, "pregrasp_pose")
    grasp = _pose(grasp_pose, "grasp_pose")
    source = np.asarray(source_handle_size, dtype=np.float64)
    target = np.asarray(target_handle_size, dtype=np.float64)
    if source.shape != (3,) or target.shape != (3,) or np.any(source <= 0.0) or np.any(target <= 0.0):
        raise ValueError("handle sizes must contain three positive values")
    if handle_axis not in (0, 1):
        raise ValueError("handle_axis must be horizontal")
    approach = pregrasp[:3] - grasp[:3]
    distance = float(np.linalg.norm(approach))
    if distance <= 1.0e-9:
        raise ValueError("pregrasp and grasp positions must be distinct")
    transverse = [axis for axis in range(3) if axis != handle_axis]
    extra = 0.5 * max(
        0.0, max(float(source[axis] - target[axis]) for axis in transverse)
    )
    result = pregrasp.copy()
    result[:3] += extra * approach / distance
    return result


def geometry_conditioned_handle_pad_depth(
    source_handle_size: Any,
    target_handle_size: Any,
    handle_axis: int,
    pad_axis_in_handle_frame: Any,
    *,
    base_depth_m: float = HANDLE_PAD_DEPTH_MARGIN_M,
) -> float:
    """Compensate pad depth for a target handle's measured cross-section loss.

    Surface-clearance transfer preserves the wrist-to-handle boundary, but a
    thinner curved handle meets the finger closer to the pad tip.  Add the
    measured transverse shrink projected onto that wrist's pad axis to the
    already calibrated base depth.  The separate signed pivot correction
    supplies asymmetric second-pad contact without adding uniform baseward
    seating.  Larger target handles retain the common program unchanged.
    """

    source = np.asarray(source_handle_size, dtype=np.float64)
    target = np.asarray(target_handle_size, dtype=np.float64)
    if source.shape != (3,) or target.shape != (3,) or np.any(source <= 0.0) or np.any(target <= 0.0):
        raise ValueError("handle sizes must contain three positive values")
    if handle_axis not in (0, 1):
        raise ValueError("handle_axis must be horizontal")
    pad_axis = np.asarray(pad_axis_in_handle_frame, dtype=np.float64)
    if pad_axis.shape != (3,) or not np.all(np.isfinite(pad_axis)):
        raise ValueError("pad_axis_in_handle_frame must contain three finite values")
    norm = float(np.linalg.norm(pad_axis))
    if norm <= 1.0e-9:
        raise ValueError("pad_axis_in_handle_frame must be nonzero")
    pad_axis /= norm
    if not np.isfinite(base_depth_m) or base_depth_m < 0.0:
        raise ValueError("base_depth_m must be finite and nonnegative")
    transverse = [axis for axis in range(3) if axis != handle_axis]
    projected_loss = sum(
        abs(float(pad_axis[axis]))
        * max(0.0, float(source[axis] - target[axis]))
        for axis in transverse
    )
    maximum_loss = max(
        0.0, max(float(source[axis] - target[axis]) for axis in transverse)
    )
    cross_section_loss = 0.5 * (min(projected_loss, maximum_loss) + maximum_loss)
    return float(base_depth_m + cross_section_loss)


def geometry_conditioned_transport_steps(
    base_steps: int,
    source_handle_size: Any,
    target_handle_size: Any,
    handle_axis: int,
) -> int:
    """Slow continuous transport in proportion to handle cross-section loss."""

    source = np.asarray(source_handle_size, dtype=np.float64)
    target = np.asarray(target_handle_size, dtype=np.float64)
    if source.shape != (3,) or target.shape != (3,) or np.any(source <= 0.0) or np.any(target <= 0.0):
        raise ValueError("handle sizes must contain three positive values")
    if handle_axis not in (0, 1):
        raise ValueError("handle_axis must be horizontal")
    if base_steps < 1:
        raise ValueError("base_steps must be positive")
    transverse = [axis for axis in range(3) if axis != handle_axis]
    retained_ratio = min(
        1.0, min(float(target[axis] / source[axis]) for axis in transverse)
    )
    return int(np.ceil(base_steps / retained_ratio))


def support_boundary_staging_pose(
    initial_pot_pose: Any,
    centered_pot_pose: Any,
    cooktop: "RigidSupportGeometry",
    *,
    support_inset_m: float,
) -> np.ndarray:
    """Place the pot center just inside the measured cooktop support boundary."""

    initial = _pose(initial_pot_pose, "initial_pot_pose")
    centered = _pose(centered_pot_pose, "centered_pot_pose")
    if not np.isfinite(support_inset_m) or support_inset_m < 0.0:
        raise ValueError("support_inset_m must be finite and nonnegative")
    direction_world = initial[:2] - centered[:2]
    distance = float(np.linalg.norm(direction_world))
    if distance <= 1.0e-9:
        return centered.copy()
    direction_world /= distance
    direction_local = quaternion_rotate(
        inverse_pose(cooktop.root_pose)[3:],
        np.asarray([direction_world[0], direction_world[1], 0.0]),
    )[:2]
    limits = [
        0.5 * float(cooktop.size[axis]) / abs(float(direction_local[axis]))
        for axis in range(2)
        if abs(float(direction_local[axis])) > 1.0e-9
    ]
    boundary_distance = min(limits)
    staging_distance = min(distance, max(0.0, boundary_distance - support_inset_m))
    staged = centered.copy()
    staged[:2] += staging_distance * direction_world
    return staged


def geometry_conditioned_grasp_hold_steps(
    base_steps: int,
    source_handle_size: Any,
    target_handle_size: Any,
    handle_axis: int,
) -> int:
    """Give thinner measured handle cross-sections longer to form both contacts."""

    source = np.asarray(source_handle_size, dtype=np.float64)
    target = np.asarray(target_handle_size, dtype=np.float64)
    if source.shape != (3,) or target.shape != (3,) or np.any(source <= 0.0) or np.any(target <= 0.0):
        raise ValueError("handle sizes must contain three positive values")
    if handle_axis not in (0, 1):
        raise ValueError("handle_axis must be horizontal")
    if base_steps < 1:
        raise ValueError("base_steps must be positive")
    transverse = [axis for axis in range(3) if axis != handle_axis]
    retained_ratio = min(
        1.0, min(float(target[axis] / source[axis]) for axis in transverse)
    )
    scaled = int(np.ceil(base_steps / retained_ratio))
    if THIN_HANDLE_SYMMETRY_RATIO <= retained_ratio < THIN_HANDLE_BALANCE_RATIO:
        scaled += MISSING_FINGER_CONTACT_SETTLE_STEPS
    return scaled


def handle_jaw_center_offset_m(
    grasp_pose: Any,
    pot_root_pose: Any,
    handle_points_local: Any,
) -> float:
    """Measure the authored handle-span midpoint along the YAM jaw axis."""

    grasp = _pose(grasp_pose, "grasp_pose")
    root = _pose(pot_root_pose, "pot_root_pose")
    points = np.asarray(handle_points_local, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("handle_points_local must have shape (N, 3), N >= 4")
    if not np.all(np.isfinite(points)):
        raise ValueError("handle_points_local must be finite")
    inverse_grasp = inverse_pose(grasp)
    points_in_gripper = np.stack(
        [
            quaternion_rotate(
                inverse_grasp[3:],
                compose_pose(root, [*point, 1.0, 0.0, 0.0, 0.0])[:3]
                - grasp[:3],
            )
            for point in points
        ]
    )
    jaw_axis = YAM_FINGER_SEPARATION_LOCAL_M.copy()
    jaw_axis[2] = 0.0
    jaw_axis /= np.linalg.norm(jaw_axis)
    projections = points_in_gripper @ jaw_axis
    finger_center = 0.5 * (
        YAM_LEFT_FINGER_PIVOT_LOCAL_M + YAM_RIGHT_FINGER_PIVOT_LOCAL_M
    )
    return float(
        0.5 * (np.min(projections) + np.max(projections))
        - np.dot(finger_center, jaw_axis)
    )


def center_handle_between_finger_pads(
    grasp_pose: Any,
    offset_m: float,
    *,
    maximum_correction_m: float = HANDLE_JAW_CENTERING_LIMIT_M,
) -> np.ndarray:
    """Translate the wrist so the measured handle span is centered in its jaw."""

    grasp = _pose(grasp_pose, "grasp_pose")
    if not np.isfinite(offset_m):
        raise ValueError("offset_m must be finite")
    if not np.isfinite(maximum_correction_m) or maximum_correction_m < 0.0:
        raise ValueError("maximum_correction_m must be finite and nonnegative")
    correction = float(np.clip(offset_m, -maximum_correction_m, maximum_correction_m))
    jaw_axis = YAM_FINGER_SEPARATION_LOCAL_M.copy()
    jaw_axis[2] = 0.0
    jaw_axis /= np.linalg.norm(jaw_axis)
    result = grasp.copy()
    result[:3] += correction * quaternion_rotate(result[3:], jaw_axis)
    return result


def mirror_handle_position_in_receiving_jaw_frame(
    reference_pose: Any,
    reference_handle_frame: Any,
    receiving_pose: Any,
    receiving_handle_frame: Any,
    pot_root_pose: Any,
    receiving_handle_points_local: Any,
) -> tuple[np.ndarray, float]:
    """Mirror a proven position, then center it for the receiving arm's jaw.

    Mirroring the complete reference pose discards the receiving arm's useful
    orientation.  Mirroring only its position can instead move the authored
    handle away from the receiving jaw center because that orientation has a
    different jaw axis.  Re-measure and remove that signed residual after the
    positional transfer.
    """

    mirrored = transfer_pose(
        _pose(reference_pose, "reference_pose"),
        _pose(reference_handle_frame, "reference_handle_frame"),
        _pose(receiving_handle_frame, "receiving_handle_frame"),
    )
    result = _pose(receiving_pose, "receiving_pose")
    result[:3] = mirrored[:3]
    offset = handle_jaw_center_offset_m(
        result, pot_root_pose, receiving_handle_points_local
    )
    return center_handle_between_finger_pads(result, offset), offset


def single_finger_contact_observed(
    finger_forces_n: Any,
    pad_fractions: Any,
    *,
    contact_threshold_n: float = 0.1,
    minimum_pad_fraction: float = SINGLE_FINGER_CONTACT_LATCH_FRACTION,
) -> bool:
    """Return whether one measured contact lies inside the pad support region."""

    forces = np.asarray(finger_forces_n, dtype=np.float64)
    fractions = np.asarray(pad_fractions, dtype=np.float64)
    if forces.shape != (2,) or not np.all(np.isfinite(forces)):
        raise ValueError("finger_forces_n must contain two finite values")
    if fractions.shape != (2,):
        raise ValueError("pad_fractions must contain two values")
    if not np.isfinite(contact_threshold_n) or contact_threshold_n < 0.0:
        raise ValueError("contact_threshold_n must be finite and nonnegative")
    if not np.isfinite(minimum_pad_fraction) or not 0.0 <= minimum_pad_fraction < 0.5:
        raise ValueError("minimum_pad_fraction must be finite and in [0, 0.5)")
    contacting = forces >= contact_threshold_n
    if int(np.sum(contacting)) != 1:
        return False
    fraction = float(fractions[int(np.flatnonzero(contacting)[0])])
    return bool(
        np.isfinite(fraction)
        and minimum_pad_fraction <= fraction <= 1.0 - minimum_pad_fraction
    )


def preserve_loaded_contact_target(
    observed_pose: Any,
    commanded_pose: Any,
    *,
    maximum_position_residual_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reanchor a loaded contact without releasing measured compression."""

    observed = _pose(observed_pose, "observed_pose")
    commanded = _pose(commanded_pose, "commanded_pose")
    if (
        not np.isfinite(maximum_position_residual_m)
        or maximum_position_residual_m <= 0.0
    ):
        raise ValueError("maximum_position_residual_m must be finite and positive")
    residual = commanded[:3] - observed[:3]
    norm = float(np.linalg.norm(residual))
    if norm > maximum_position_residual_m:
        residual *= maximum_position_residual_m / norm
    result = observed.copy()
    result[:3] += residual
    result[3:] = commanded[3:]
    return result, residual


def reanchor_missing_finger_contact(
    contact_local: Any,
    observed_root_pose: Any,
    finger_forces_n: Any,
    finger_pad_centers_world: Any,
    signed_correction_m: float,
    *,
    contact_threshold_n: float = 0.1,
    step_m: float = MISSING_FINGER_CONTACT_STEP_M,
    limit_m: float = MISSING_FINGER_CONTACT_LIMIT_M,
) -> tuple[np.ndarray, float]:
    """Move a fixed contact frame toward the finger missing target contact.

    This is deterministic contact feedback, not candidate search: one
    contacting finger fixes the signed direction along the observed pad-center
    axis, while two or zero contacts leave the object-to-gripper frame intact.
    """

    local = _pose(contact_local, "contact_local")
    root = _pose(observed_root_pose, "observed_root_pose")
    forces = np.asarray(finger_forces_n, dtype=np.float64)
    centers = np.asarray(finger_pad_centers_world, dtype=np.float64)
    if forces.shape != (2,) or not np.all(np.isfinite(forces)):
        raise ValueError("finger_forces_n must contain two finite values")
    if centers.shape != (2, 3) or not np.all(np.isfinite(centers)):
        raise ValueError("finger_pad_centers_world must contain two finite points")
    if not np.isfinite(signed_correction_m):
        raise ValueError("signed_correction_m must be finite")
    if not np.isfinite(contact_threshold_n) or contact_threshold_n < 0.0:
        raise ValueError("contact_threshold_n must be finite and nonnegative")
    if not np.isfinite(step_m) or step_m <= 0.0:
        raise ValueError("step_m must be finite and positive")
    if not np.isfinite(limit_m) or limit_m < step_m:
        raise ValueError("limit_m must be finite and at least step_m")
    contacting = forces >= contact_threshold_n
    if int(np.sum(contacting)) != 1:
        return local.copy(), float(signed_correction_m)
    direction = 1.0 if contacting[1] else -1.0
    updated = float(
        np.clip(signed_correction_m + direction * step_m, -limit_m, limit_m)
    )
    delta = updated - float(signed_correction_m)
    world = compose_pose(root, local)
    jaw_axis = centers[1] - centers[0]
    jaw_norm = float(np.linalg.norm(jaw_axis))
    if jaw_norm < MISSING_FINGER_JAW_AXIS_MIN_M:
        # Pot020 attempt 011 showed that the two pad centers converge to only
        # 3.4 mm while the fingers close.  Their normalized difference is no
        # longer a physical jaw direction then and drove the fixed contact
        # target away from the handle.  Keep the last valid object-to-gripper
        # frame once the measured jaw aperture falls below 50 mm.
        return local.copy(), float(signed_correction_m)
    world[:3] += delta * jaw_axis / jaw_norm
    return compose_pose(inverse_pose(root), world), updated


def geometry_conditioned_right_first_close(
    source_handle_size: Any,
    target_handle_size: Any,
    handle_axis: int,
    predicted_imbalance_m: float,
) -> bool:
    """Stabilize a positive-imbalance handle from its proven peer.

    Pot020 attempts 001-011 consistently latched the right handle while the
    positive-imbalance left jaw still had only one pad in contact.  Attempt 012
    proved that closing the left jaw first merely pushes and rotates the free
    pot.  Close the proven right jaw first so its bimanual contact milestone
    stabilizes the pot for left seating.  The 45-50% retained transverse-
    thickness band captures Pot020.  Pot021 shows the same signed one-pad
    failure outside that band when its authored +48.8 mm pad imbalance exceeds
    the 40 mm jaw-centering correction reach.  Thinner Pot019 remains
    simultaneous because its +37.5 mm imbalance is inside that measured reach
    and has a hash-verified bimanual success.
    """

    source = np.asarray(source_handle_size, dtype=np.float64)
    target = np.asarray(target_handle_size, dtype=np.float64)
    if source.shape != (3,) or target.shape != (3,) or np.any(source <= 0.0) or np.any(target <= 0.0):
        raise ValueError("handle sizes must contain three positive values")
    if handle_axis not in (0, 1):
        raise ValueError("handle_axis must be horizontal")
    if not np.isfinite(predicted_imbalance_m):
        raise ValueError("predicted_imbalance_m must be finite")
    transverse = [axis for axis in range(3) if axis != handle_axis]
    retained_ratio = min(float(target[axis] / source[axis]) for axis in transverse)
    half_thickness_positive_imbalance = (
        THIN_HANDLE_SYMMETRY_RATIO
        <= retained_ratio
        < THIN_HANDLE_BALANCE_RATIO
        and predicted_imbalance_m > 0.0
    )
    beyond_centering_reach = (
        predicted_imbalance_m > HANDLE_JAW_CENTERING_LIMIT_M
    )
    return bool(half_thickness_positive_imbalance or beyond_centering_reach)


def geometry_conditioned_peer_contact_transfer(
    left_handle_size: Any,
    right_handle_size: Any,
    predicted_imbalance_m: float,
    *,
    relative_size_tolerance: float = 0.05,
) -> bool:
    """Use a proven peer contact for matching handles beyond jaw reach."""

    left = np.asarray(left_handle_size, dtype=np.float64)
    right = np.asarray(right_handle_size, dtype=np.float64)
    if left.shape != (3,) or right.shape != (3,):
        raise ValueError("handle sizes must contain three values")
    if np.any(left <= 0.0) or np.any(right <= 0.0):
        raise ValueError("handle sizes must be positive")
    if not np.isfinite(predicted_imbalance_m):
        raise ValueError("predicted_imbalance_m must be finite")
    if not np.isfinite(relative_size_tolerance) or not (
        0.0 <= relative_size_tolerance < 1.0
    ):
        raise ValueError("relative_size_tolerance must be in [0, 1)")
    relative_difference = np.max(
        np.abs(left - right) / np.maximum(left, right)
    )
    return bool(
        relative_difference <= relative_size_tolerance
        and predicted_imbalance_m > HANDLE_JAW_CENTERING_LIMIT_M
    )


def reanchor_authored_handle_in_observed_jaw(
    observed_root_pose: Any,
    observed_eef_pose: Any,
    finger_pad_centers_world: Any,
    handle_points_local: Any,
    approach_delta_local: Any,
) -> tuple[np.ndarray, float, float]:
    """Center authored handle points in the observed open jaw at a milestone.

    The result is a new object-local contact frame, not a world-space teleport.
    It is intended for the next controller segment after the peer grasp has
    physically stabilized the object.
    """

    root = _pose(observed_root_pose, "observed_root_pose")
    eef = _pose(observed_eef_pose, "observed_eef_pose")
    centers = np.asarray(finger_pad_centers_world, dtype=np.float64)
    points = np.asarray(handle_points_local, dtype=np.float64)
    approach = np.asarray(approach_delta_local, dtype=np.float64)
    if centers.shape != (2, 3) or not np.all(np.isfinite(centers)):
        raise ValueError("finger_pad_centers_world must contain two finite points")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("handle_points_local must have shape (N, 3), N >= 4")
    if not np.all(np.isfinite(points)):
        raise ValueError("handle_points_local must be finite")
    if approach.shape != (3,) or not np.all(np.isfinite(approach)):
        raise ValueError("approach_delta_local must contain three finite values")
    jaw_axis = centers[1] - centers[0]
    jaw_norm = float(np.linalg.norm(jaw_axis))
    if jaw_norm < MISSING_FINGER_JAW_AXIS_MIN_M:
        raise ValueError("observed jaw must be open for authored centering")
    jaw_axis /= jaw_norm
    jaw_midpoint = np.mean(centers, axis=0)
    points_world = np.stack(
        [root[:3] + quaternion_rotate(root[3:], point) for point in points]
    )
    projection = (points_world - jaw_midpoint) @ jaw_axis
    signed_residual = 0.5 * (float(np.min(projection)) + float(np.max(projection)))
    target = eef.copy()
    translation = signed_residual * jaw_axis + quaternion_rotate(
        root[3:], approach
    )
    target[:3] += translation
    return (
        compose_pose(inverse_pose(root), target),
        signed_residual,
        float(np.linalg.norm(translation)),
    )


def milestone_reanchor_within_authored_clearance(
    candidate_translation_m: float,
    regrasp_clearance_m: float,
    *,
    pad_depth_margin_m: float = HANDLE_PAD_DEPTH_MARGIN_M,
) -> bool:
    """Reject milestone targets that exceed measured collision clearance.

    The pad-depth margin permits the contact target to advance beyond the
    collision-free pregrasp standoff, while keeping a composed jaw/approach
    correction from crossing the handle by more than authored geometry allows.
    """

    values = np.asarray(
        [candidate_translation_m, regrasp_clearance_m, pad_depth_margin_m],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("milestone distances must be finite and nonnegative")
    return bool(
        candidate_translation_m
        <= regrasp_clearance_m + pad_depth_margin_m
    )


def reanchor_missing_finger_pad_depth(
    contact_local: Any,
    observed_root_pose: Any,
    finger_forces_n: Any,
    pad_fractions: Any,
    pad_axes_world: Any,
    depth_correction_m: float,
    *,
    contact_threshold_n: float = 0.1,
    step_m: float = MISSING_FINGER_PAD_DEPTH_STEP_M,
    limit_m: float = MISSING_FINGER_PAD_DEPTH_LIMIT_M,
    target_fraction: float = MISSING_FINGER_PAD_TARGET_FRACTION,
) -> tuple[np.ndarray, float]:
    """Seat a near-contact missing finger by its signed authored pad residual."""

    local = _pose(contact_local, "contact_local")
    root = _pose(observed_root_pose, "observed_root_pose")
    forces = np.asarray(finger_forces_n, dtype=np.float64)
    fractions = np.asarray(pad_fractions, dtype=np.float64)
    axes = np.asarray(pad_axes_world, dtype=np.float64)
    if forces.shape != (2,) or not np.all(np.isfinite(forces)):
        raise ValueError("finger_forces_n must contain two finite values")
    if fractions.shape != (2,):
        raise ValueError("pad_fractions must contain two values")
    if axes.shape != (2, 3) or not np.all(np.isfinite(axes)):
        raise ValueError("pad_axes_world must contain two finite 3D axes")
    if not np.isfinite(depth_correction_m) or depth_correction_m < 0.0:
        raise ValueError("depth_correction_m must be finite and nonnegative")
    if not np.isfinite(step_m) or step_m <= 0.0:
        raise ValueError("step_m must be finite and positive")
    if not np.isfinite(limit_m) or limit_m < step_m:
        raise ValueError("limit_m must be finite and at least step_m")
    if not np.isfinite(target_fraction) or target_fraction < 0.0:
        raise ValueError("target_fraction must be finite and nonnegative")
    contacting = forces >= contact_threshold_n
    if int(np.sum(contacting)) != 1:
        return local.copy(), float(depth_correction_m)
    missing = int(np.flatnonzero(~contacting)[0])
    fraction = float(fractions[missing])
    if not np.isfinite(fraction) or fraction >= target_fraction:
        return local.copy(), float(depth_correction_m)
    measured_residual = (target_fraction - fraction) * YAM_FINGER_PAD_AXIS_LENGTH_M
    delta = min(step_m, measured_residual, limit_m - depth_correction_m)
    if delta <= 0.0:
        return local.copy(), float(depth_correction_m)
    world = compose_pose(root, local)
    axis = axes[missing]
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-9:
        raise ValueError("missing finger pad axis must be nonzero")
    # Moving the finger opposite its tip-to-base axis moves a fixed object
    # contact from the tip toward the base, increasing the signed fraction.
    world[:3] -= delta * axis / norm
    return compose_pose(inverse_pose(root), world), float(depth_correction_m + delta)


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
    """Tilt about one finger pivot to deepen only the opposite pad contact.

    PutPot012 traces showed that pivoting about the authored left finger held
    the failing pad fraction unchanged while moving the already-valid pad.
    Positive depth preserves the right finger and deepens the left; negative
    depth preserves the left finger and deepens the right.
    """

    grasp = _pose(grasp_pose, "grasp_pose")
    magnitude = abs(float(relative_depth_m))
    span = float(np.linalg.norm(YAM_FINGER_SEPARATION_LOCAL_M))
    if not np.isfinite(relative_depth_m) or magnitude >= span:
        raise ValueError("relative_depth_m must be finite and smaller than finger span")
    if magnitude == 0.0:
        return grasp.copy()
    if relative_depth_m > 0.0:
        separation = -YAM_FINGER_SEPARATION_LOCAL_M
        pivot = YAM_RIGHT_FINGER_PIVOT_LOCAL_M
    else:
        separation = YAM_FINGER_SEPARATION_LOCAL_M
        pivot = YAM_LEFT_FINGER_PIVOT_LOCAL_M
    axis = np.asarray(
        [separation[1], -separation[0], 0.0], dtype=np.float64
    )
    axis /= np.linalg.norm(axis)
    angle = float(np.arcsin(magnitude / span))
    delta = np.concatenate(
        ([np.cos(0.5 * angle)], axis * np.sin(0.5 * angle))
    )
    result = grasp.copy()
    pivot_world = grasp[:3] + quaternion_rotate(
        grasp[3:], pivot
    )
    result[3:] = quaternion_multiply(grasp[3:], delta)
    result[3:] /= np.linalg.norm(result[3:])
    result[:3] = pivot_world - quaternion_rotate(
        result[3:], pivot
    )
    return result


def handle_finger_pad_depth_imbalance(
    grasp_pose: Any,
    pot_root_pose: Any,
    handle_points_local: Any,
) -> float:
    """Estimate finger-0 minus finger-1 pad depth from authored geometry."""

    grasp = _pose(grasp_pose, "grasp_pose")
    root = _pose(pot_root_pose, "pot_root_pose")
    points = np.asarray(handle_points_local, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("handle_points_local must have shape (N, 3), N >= 4")
    if not np.all(np.isfinite(points)):
        raise ValueError("handle_points_local must be finite")
    inverse_grasp = inverse_pose(grasp)
    points_in_gripper = np.stack(
        [
            quaternion_rotate(
                inverse_grasp[3:],
                compose_pose(root, [*point, 1.0, 0.0, 0.0, 0.0])[:3]
                - grasp[:3],
            )
            for point in points
        ]
    )

    def depth(pivot: np.ndarray) -> float:
        index = int(
            np.argmin(np.linalg.norm(points_in_gripper[:, :2] - pivot[:2], axis=1))
        )
        return float(points_in_gripper[index, 2] - pivot[2])

    return depth(YAM_LEFT_FINGER_PIVOT_LOCAL_M) - depth(
        YAM_RIGHT_FINGER_PIVOT_LOCAL_M
    )


def bounded_handle_pad_balance(
    predicted_imbalance_m: float,
    maximum_balance_m: float = HANDLE_PAD_RELATIVE_DEPTH_M,
) -> float:
    """Convert authored pad imbalance to the bounded pivot direction.

    PutPot016 attempt_008 applied the opposite sign and pushed the right
    second-pad fraction beyond 1.02.  Preserve the authored imbalance sign so
    the pivot moves contact away from that measured pad-base failure.
    """

    if not np.isfinite(predicted_imbalance_m):
        raise ValueError("predicted_imbalance_m must be finite")
    if not np.isfinite(maximum_balance_m) or maximum_balance_m < 0.0:
        raise ValueError("maximum_balance_m must be finite and nonnegative")
    return float(
        np.clip(predicted_imbalance_m, -maximum_balance_m, maximum_balance_m)
    )


def geometry_conditioned_handle_balance_limit(
    source_handle_size: Any,
    target_handle_size: Any,
    handle_axis: int,
    predicted_imbalance_m: float,
    *,
    base_limit_m: float = HANDLE_PAD_RELATIVE_DEPTH_M,
) -> float:
    """Add measured finger-0 pivot authority only for severely thinned handles.

    Pot019 attempt_004 and Pot020 attempts 005-009 held the opposite arm while
    this arm's finger 0 remained tip-side.  Preserve Pot019's proven 5 mm cap.
    In the separate 45-50% retained-thickness band, attempt 010 showed that a
    15 mm pivot removed contact entirely, so retain the proven 5 mm bound and
    handle lateral seating from the observed pad-center axis instead.
    """

    source = np.asarray(source_handle_size, dtype=np.float64)
    target = np.asarray(target_handle_size, dtype=np.float64)
    if source.shape != (3,) or target.shape != (3,) or np.any(source <= 0.0) or np.any(target <= 0.0):
        raise ValueError("handle sizes must contain three positive values")
    if handle_axis not in (0, 1):
        raise ValueError("handle_axis must be horizontal")
    if not np.isfinite(predicted_imbalance_m):
        raise ValueError("predicted_imbalance_m must be finite")
    if not np.isfinite(base_limit_m) or base_limit_m < 0.0:
        raise ValueError("base_limit_m must be finite and nonnegative")
    transverse = [axis for axis in range(3) if axis != handle_axis]
    retained_ratio = min(float(target[axis] / source[axis]) for axis in transverse)
    if (
        THIN_HANDLE_SYMMETRY_RATIO
        <= retained_ratio
        < THIN_HANDLE_BALANCE_RATIO
        and predicted_imbalance_m > 0.0
    ):
        return float(
            max(base_limit_m, HALF_THIN_HANDLE_POSITIVE_BALANCE_LIMIT_M)
        )
    extra = (
        THIN_HANDLE_POSITIVE_BALANCE_EXTRA_M
        if retained_ratio < THIN_HANDLE_SYMMETRY_RATIO
        and predicted_imbalance_m > 0.0
        else 0.0
    )
    return float(base_limit_m + extra)


def geometry_conditioned_target_handle_symmetry(
    source_negative_size: Any,
    source_positive_size: Any,
    target_negative_size: Any,
    target_positive_size: Any,
    handle_axis: int,
    *,
    symmetry_relative_tolerance: float = 0.02,
) -> bool:
    """Use one target-side contact relation for a thin measured-symmetric pair.

    The position-only mirror has its own measured thinning boundary.  It does
    not grant the extra finger-pivot authority controlled by
    ``THIN_HANDLE_BALANCE_RATIO``.
    """

    values = [
        np.asarray(value, dtype=np.float64)
        for value in (
            source_negative_size,
            source_positive_size,
            target_negative_size,
            target_positive_size,
        )
    ]
    if any(value.shape != (3,) or np.any(value <= 0.0) for value in values):
        raise ValueError("handle sizes must contain three positive values")
    if handle_axis not in (0, 1):
        raise ValueError("handle_axis must be horizontal")
    if not np.isfinite(symmetry_relative_tolerance) or symmetry_relative_tolerance < 0.0:
        raise ValueError("symmetry_relative_tolerance must be finite and nonnegative")
    source_negative, source_positive, target_negative, target_positive = values
    transverse = [axis for axis in range(3) if axis != handle_axis]
    retained_ratio = min(
        min(float(target_negative[axis] / source_negative[axis]) for axis in transverse),
        min(float(target_positive[axis] / source_positive[axis]) for axis in transverse),
    )
    scale = np.maximum(target_negative, target_positive)
    symmetric = np.all(
        np.abs(target_negative - target_positive)
        <= symmetry_relative_tolerance * scale
    )
    return bool(retained_ratio < THIN_HANDLE_SYMMETRY_RATIO and symmetric)


@dataclass(frozen=True)
class SmoothBimanualTransport:
    """One sampled pot path and the two rigidly attached handle paths."""

    pot_poses: np.ndarray
    left_poses: np.ndarray
    right_poses: np.ndarray
    minimum_cooktop_clearance_m: float
    cooktop_overlap_samples: int
    initial_clearance_recovery_m: float


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
    initial_clearance_recovery_m = 0.0
    if len(positive):
        first = int(positive[0])
        last = int(positive[-1])
        if first == 0:
            # A feedback reanchor may observe tracking drift after the pot has
            # already entered the footprint.  Make the very next commanded
            # sample a vertical clearance recovery, then continue the C2 fall.
            # The observed state itself is retained in the trace; no reset or
            # teleport is introduced.
            initial_clearance_recovery_m = float(np.max(required_lift))
            lift_profile[: last + 1] = 1.0
        else:
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
        initial_clearance_recovery_m=initial_clearance_recovery_m,
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


def maximum_bimanual_position_step_m(
    left_poses: Any, right_poses: Any
) -> float:
    """Measure the largest per-arm Cartesian position increment."""

    left = np.asarray(left_poses, dtype=np.float64)
    right = np.asarray(right_poses, dtype=np.float64)
    if left.ndim != 2 or right.shape != left.shape or left.shape[1] != 7:
        raise ValueError("left_poses and right_poses must have matching (N, 7) shapes")
    if len(left) < 2:
        return 0.0
    return float(
        max(
            np.max(np.linalg.norm(np.diff(left[:, :3], axis=0), axis=1)),
            np.max(np.linalg.norm(np.diff(right[:, :3], axis=0), axis=1)),
        )
    )


def transport_reanchor_position_step_limit_m(
    controller_limit_m: float, handle_retention_tolerance_m: float
) -> float:
    """Keep feedback steps within both controller and handle geometry limits."""

    values = np.asarray(
        [controller_limit_m, handle_retention_tolerance_m], dtype=np.float64
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("transport reanchor step limits must be positive and finite")
    # ``handle_retention_tolerance_m`` is already derived as half of the
    # narrowest measured handle cross-section.  Halving it again rejects
    # smooth observed-state tails that remain inside the authored handle
    # contact region, leaving the controller to chase a stale trajectory.
    return float(min(values[0], values[1]))


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
    required = (
        "bimanual_pregrasp",
        "left_handle_grasp",
        "right_handle_grasp",
    )
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
    right_first_close: bool = False,
    feedback_horizon_steps: int = CONTACT_FEEDBACK_HORIZON_STEPS,
) -> SkillTrajectory:
    """Track both fixed local handle contacts during simultaneous closing.

    Either handle may latch first on a new asset.  Keep an already latched
    wrist at its measured contact and smoothly chase the other handle in the
    observed pot frame instead of assuming that the left contact always leads.
    """

    steps = trajectory.waypoint_steps
    pregrasp_end = steps.get("bimanual_pregrasp")
    left_end = steps.get("left_handle_grasp")
    right_end = steps.get("right_handle_grasp")
    if pregrasp_end is None or left_end is None or right_end is None:
        raise ValueError("handle trajectory is missing grasp waypoints")
    grasp_end = max(left_end, right_end)
    if current_step < pregrasp_end or current_step >= grasp_end:
        raise ValueError("current_step is outside the contact-closing window")
    start = current_step + 1
    remaining = grasp_end - current_step
    left = trajectory.left_poses.copy()
    right = trajectory.right_poses.copy()
    if right_first_close and current_step < right_end:
        right_target = compose_pose(observed_pot_pose, right_contact_local)
        right_remaining = right_end - current_step
        right[start : right_end + 1] = _linear_contact_feedback_poses(
            observed_right_pose,
            right_target,
            right_remaining,
            horizon_steps=feedback_horizon_steps,
        )
        right[right_end + 1 : grasp_end + 1] = right_target
        # Keep the left arm at its authored pregrasp until the proven right
        # contact milestone stabilizes the free pot.
    elif left_contact_latched and right_contact_latched:
        left[start : grasp_end + 1] = _pose(
            observed_left_pose, "observed_left_pose"
        )
        right[start : grasp_end + 1] = _pose(
            observed_right_pose, "observed_right_pose"
        )
    else:
        left_target = compose_pose(observed_pot_pose, left_contact_local)
        right_target = compose_pose(observed_pot_pose, right_contact_local)
        left[start : grasp_end + 1] = (
            _pose(observed_left_pose, "observed_left_pose")
            if left_contact_latched
            else _linear_contact_feedback_poses(
                observed_left_pose,
                left_target,
                remaining,
                horizon_steps=feedback_horizon_steps,
            )
        )
        right[start : grasp_end + 1] = (
            _pose(observed_right_pose, "observed_right_pose")
            if right_contact_latched
            else _linear_contact_feedback_poses(
                observed_right_pose,
                right_target,
                remaining,
                horizon_steps=feedback_horizon_steps,
            )
        )
    return SkillTrajectory(
        left_poses=left,
        right_poses=right,
        grippers=trajectory.grippers.copy(),
        stage_names=trajectory.stage_names,
        waypoint_steps=dict(trajectory.waypoint_steps),
    )


def reanchor_bimanual_contact_hold(
    trajectory: SkillTrajectory,
    current_step: int,
    observed_pot_pose: Any,
    retained_left_contact_local: Any,
    retained_right_contact_local: Any,
) -> SkillTrajectory:
    """Follow retained object-local two-arm contacts before transport."""

    hold_end = trajectory.waypoint_steps.get("bimanual_contact_hold")
    if hold_end is None:
        raise ValueError("trajectory has no bimanual contact hold")
    if current_step < 0 or current_step >= hold_end:
        raise ValueError("current_step is outside the bimanual contact hold")
    start = current_step + 1
    left = trajectory.left_poses.copy()
    right = trajectory.right_poses.copy()
    left_target = compose_pose(
        _pose(observed_pot_pose, "observed_pot_pose"),
        _pose(retained_left_contact_local, "retained_left_contact_local"),
    )
    right_target = compose_pose(
        _pose(observed_pot_pose, "observed_pot_pose"),
        _pose(retained_right_contact_local, "retained_right_contact_local"),
    )
    left[start : hold_end + 1] = left_target
    right[start : hold_end + 1] = right_target
    return SkillTrajectory(
        left_poses=left,
        right_poses=right,
        grippers=trajectory.grippers.copy(),
        stage_names=trajectory.stage_names,
        waypoint_steps=dict(trajectory.waypoint_steps),
    )


def compensate_retained_contact_tracking(
    reference_contact_local: Any,
    observed_pot_pose: Any,
    observed_eef_pose: Any,
    *,
    correction_limit_m: float = HANDLE_PAD_GEOMETRIC_MARGIN_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Oppose measured object-local hold drift within a geometry margin."""

    reference = _pose(reference_contact_local, "reference_contact_local")
    root = _pose(observed_pot_pose, "observed_pot_pose")
    observed = _pose(observed_eef_pose, "observed_eef_pose")
    observed_local = compose_pose(inverse_pose(root), observed)
    correction = reference[:3] - observed_local[:3]
    norm = float(np.linalg.norm(correction))
    if not np.isfinite(correction_limit_m) or correction_limit_m < 0.0:
        raise ValueError("correction_limit_m must be finite and nonnegative")
    if norm > correction_limit_m and norm > 0.0:
        correction *= correction_limit_m / norm
    target = reference.copy()
    target[:3] += correction
    return target, correction


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
    current_step: int | None = None,
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
    grasp_end = max(steps["left_handle_grasp"], steps["right_handle_grasp"])
    start = grasp_end + 1 if current_step is None else int(current_step) + 1
    transport_end = steps["smooth_transport"]
    if start <= grasp_end or start > transport_end:
        raise ValueError("current transport reanchor step is outside the transport window")
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
    left_withdraw = left_lower.copy()
    right_withdraw = right_lower.copy()
    if "center_slide" in steps:
        left_release_end = steps["left_unload_release"]
        nominal_left_withdraw_delta = (
            trajectory.left_poses[left_release_end, :3]
            - trajectory.left_poses[lower_end, :3]
        )
        left_withdraw[:3] += nominal_left_withdraw_delta
        left[lower_end + 1 : left_release_end + 1] = interpolate_poses(
            left_lower,
            left_withdraw,
            left_release_end - lower_end,
        )
        left[left_release_end + 1 :] = left_withdraw
    else:
        left[lower_end + 1 : release_end + 1] = left_lower
        left_withdraw[:3] += [0.0, 0.08, 0.12]
    right[lower_end + 1 : release_end + 1] = right_lower
    right_withdraw[:3] += [0.0, -0.08, 0.12]
    if "center_slide" not in steps:
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


def transport_contact_reanchor_required(
    trajectory: SkillTrajectory,
    step: int,
    observed_pot_pose: Any,
    observed_left_pose: Any,
    observed_right_pose: Any,
    reference_left_contact_local: Any | None,
    reference_right_contact_local: Any | None,
    *,
    last_reanchor_step: int | None,
    tracking_tolerance_m: float,
    minimum_interval_steps: int = TRANSPORT_CONTACT_REANCHOR_MIN_STEPS,
) -> bool:
    """Return whether grasped transport has drifted beyond handle retention."""

    if not np.isfinite(tracking_tolerance_m) or tracking_tolerance_m <= 0.0:
        raise ValueError("tracking_tolerance_m must be positive and finite")
    if minimum_interval_steps < 1:
        raise ValueError("minimum_interval_steps must be positive")
    transport_end = trajectory.waypoint_steps.get("smooth_transport")
    if transport_end is None or step < 0 or transport_end - step < 8:
        return False
    if last_reanchor_step is None:
        return True
    if reference_left_contact_local is None or reference_right_contact_local is None:
        raise ValueError("accepted reanchor contact frames are required")
    if step - last_reanchor_step < minimum_interval_steps:
        return False
    pot_pose = _pose(observed_pot_pose, "observed_pot_pose")
    inverse_pot = inverse_pose(pot_pose)
    observed_left_contact = compose_pose(
        inverse_pot, _pose(observed_left_pose, "observed_left_pose")
    )
    observed_right_contact = compose_pose(
        inverse_pot, _pose(observed_right_pose, "observed_right_pose")
    )
    left_drift = np.linalg.norm(
        observed_left_contact[:3]
        - _pose(reference_left_contact_local, "reference_left_contact_local")[:3]
    )
    right_drift = np.linalg.norm(
        observed_right_contact[:3]
        - _pose(reference_right_contact_local, "reference_right_contact_local")[:3]
    )
    return bool(max(left_drift, right_drift) > tracking_tolerance_m)


def reanchor_centered_lowering(
    trajectory: SkillTrajectory,
    center_correction_xy: Any,
    observed_left_pose: Any,
    observed_right_pose: Any,
    *,
    vertical_correction_m: float | None = None,
) -> SkillTrajectory:
    """Translate measured contact poses to center during the short lowering."""
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
    left_lower = _pose(observed_left_pose, "observed_left_pose")
    right_lower = _pose(observed_right_pose, "observed_right_pose")
    left_lower[:2] += correction
    right_lower[:2] += correction
    if vertical_correction_m is None:
        vertical_correction_m = float(
            trajectory.left_poses[lower_end, 2]
            - trajectory.left_poses[steps["smooth_transport"], 2]
        )
    if not np.isfinite(vertical_correction_m):
        raise ValueError("vertical_correction_m must be finite")
    left_lower[2] += float(vertical_correction_m)
    right_lower[2] += float(vertical_correction_m)
    left[start : lower_end + 1] = interpolate_poses(
        observed_left_pose, left_lower, lower_end - start + 1
    )
    right[start : lower_end + 1] = interpolate_poses(
        observed_right_pose, right_lower, lower_end - start + 1
    )
    if "center_slide" in steps:
        left_release_end = steps["left_unload_release"]
        slide_end = steps["center_slide"]
        nominal_left_lower = trajectory.left_poses[lower_end]
        left_withdraw_delta = (
            trajectory.left_poses[left_release_end, :3]
            - nominal_left_lower[:3]
        )
        left_withdraw = left_lower.copy()
        left_withdraw[:3] += left_withdraw_delta
        left[lower_end + 1 : left_release_end + 1] = interpolate_poses(
            left_lower, left_withdraw, left_release_end - lower_end
        )
        left[left_release_end + 1 :] = left_withdraw
        right[lower_end + 1 : slide_end + 1] = right_lower
        return SkillTrajectory(
            left_poses=left,
            right_poses=right,
            grippers=trajectory.grippers.copy(),
            stage_names=trajectory.stage_names,
            waypoint_steps=dict(trajectory.waypoint_steps),
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
    left_release = _pose(observed_left_pose, "observed_left_pose")
    right_release = _pose(observed_right_pose, "observed_right_pose")
    if anchor_name != "center_slide":
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
    *,
    current_step: int | None = None,
    reference_right_contact_local: Any | None = None,
    contact_recovery_steps: int = CONTACT_FEEDBACK_HORIZON_STEPS,
    support_unload_m: float = 0.0,
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
    right_contact = (
        compose_pose(inverse_pose(pot_pose), right_pose)
        if reference_right_contact_local is None
        else _pose(
            reference_right_contact_local,
            "reference_right_contact_local",
        )
    )
    centered_pot_pose = pot_pose.copy()
    centered_pot_pose[:2] = cooktop_pose[:2]
    right_center = compose_pose(centered_pot_pose, right_contact)
    start = (
        steps["left_unload_release"] + 1
        if current_step is None
        else int(current_step) + 1
    )
    slide_end = steps["center_slide"]
    if start > slide_end:
        raise ValueError("current_step is outside the supported center slide")
    if not np.isfinite(support_unload_m) or support_unload_m < 0.0:
        raise ValueError("support_unload_m must be finite and nonnegative")
    release_end = steps["pot_release"]
    withdraw_end = steps["bimanual_withdraw"]
    right = np.asarray(trajectory.right_poses, dtype=np.float64).copy()
    nominal_withdraw_delta = (
        trajectory.right_poses[withdraw_end, :3]
        - trajectory.right_poses[slide_end, :3]
    )
    right_withdraw = right_center.copy()
    right_withdraw[:3] += nominal_withdraw_delta
    remaining = slide_end - start + 1
    if reference_right_contact_local is None:
        if support_unload_m > 0.0:
            lift_steps = min(
                CONTACT_FEEDBACK_HORIZON_STEPS,
                (remaining - 8) // 2,
            )
            if lift_steps < 1:
                raise ValueError("supported slide needs room for support unloading")
            lifted_pot_pose = pot_pose.copy()
            lifted_pot_pose[2] += float(support_unload_m)
            centered_lifted_pot_pose = centered_pot_pose.copy()
            centered_lifted_pot_pose[2] += float(support_unload_m)
            right_lift = compose_pose(lifted_pot_pose, right_contact)
            right_center_lift = compose_pose(
                centered_lifted_pot_pose, right_contact
            )
            translate_steps = remaining - 2 * lift_steps
            right[start : start + lift_steps] = interpolate_poses(
                right_pose, right_lift, lift_steps
            )
            right[
                start + lift_steps : start + lift_steps + translate_steps
            ] = interpolate_poses(
                right_lift, right_center_lift, translate_steps
            )
            right[start + lift_steps + translate_steps : slide_end + 1] = (
                interpolate_poses(right_center_lift, right_center, lift_steps)
            )
        else:
            right[start : slide_end + 1] = interpolate_poses(
                right_pose, right_center, remaining
            )
    else:
        if contact_recovery_steps < 1:
            raise ValueError("contact_recovery_steps must be positive")
        recovery_steps = min(int(contact_recovery_steps), remaining - 8)
        if recovery_steps < 1:
            raise ValueError("supported slide needs room for contact recovery")
        right_reseated = compose_pose(pot_pose, right_contact)
        right[start : start + recovery_steps] = interpolate_poses(
            right_pose, right_reseated, recovery_steps
        )
        right[start + recovery_steps : slide_end + 1] = interpolate_poses(
            right_reseated, right_center, remaining - recovery_steps
        )
    right[slide_end + 1 : release_end + 1] = right_center
    right[release_end + 1 : withdraw_end + 1] = interpolate_poses(
        right_center,
        right_withdraw,
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
        right_first: bool = False,
        contact_hold_steps: int = 0,
    ) -> None:
        if simultaneous and right_first:
            raise ValueError("simultaneous and right_first are mutually exclusive")
        if contact_hold_steps < 0:
            raise ValueError("contact_hold_steps must be nonnegative")
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
        elif right_first:
            self._append(
                "right_handle_grasp",
                "bimanual_handle_grasp",
                right_close_steps,
                right_pose=right_grasp,
                right_gripper=closed,
            )
            self._append(
                "left_handle_grasp",
                "bimanual_handle_grasp",
                left_close_steps,
                left_pose=left_grasp,
                left_gripper=closed,
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
        if contact_hold_steps:
            self._append(
                "bimanual_contact_hold",
                "bimanual_handle_grasp",
                contact_hold_steps,
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
