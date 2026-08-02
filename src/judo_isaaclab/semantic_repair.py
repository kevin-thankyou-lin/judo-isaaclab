"""Failure taxonomy and resumable state for deterministic semantic repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from judo_isaaclab.put_marker import inverse_pose, quaternion_rotate


@dataclass(frozen=True)
class FailureDiagnosis:
    first_failed_stage: str
    reason: str
    signed_residuals: dict[str, Any]
    visual_frame: int


def _local_offset(reference_pose: Any, world_position: Any) -> list[float]:
    reference = np.asarray(reference_pose, dtype=np.float64)
    position = np.asarray(world_position, dtype=np.float64)
    inverse = inverse_pose(reference)
    return quaternion_rotate(inverse[3:], position - reference[:3]).tolist()


def _tracking_residuals(
    trace: dict[str, np.ndarray] | None,
    *,
    index: int,
    reference_pose: Any,
) -> dict[str, list[float]]:
    if not trace or index < 0:
        return {}
    result = {}
    for arm in ("left", "right"):
        actual = trace.get(f"{arm}_eef_poses")
        desired = trace.get(f"desired_{arm}_eef_poses")
        if actual is None or desired is None or index >= len(actual) or index >= len(desired):
            continue
        result[f"{arm}_eef_position_error_local_m"] = _local_offset(
            reference_pose,
            np.asarray(actual[index, :3]) - np.asarray(desired[index, :3])
            + np.asarray(reference_pose[:3]),
        )
    return result


def diagnose_semantic_failure(
    task: str,
    result: dict[str, Any],
    trace: dict[str, np.ndarray] | None = None,
) -> FailureDiagnosis:
    """Classify the first missing coded stage and retain signed local residuals."""

    checks = result.get("checks", {})
    terminal = result.get("terminal", {})
    metrics = result.get("metrics", {})
    protocol = result.get("protocol", {})
    parameters = protocol.get("parameters", {})

    if task == "putpot":
        if not terminal.get("stage1", False):
            stage = "bimanual_handle_grasp"
            reason = "two contact-backed handle grasps did not latch the lifted-pot stage"
            frame = 199
        elif not checks.get("bimanual_transport_completed", False):
            stage = "bimanual_transport"
            reason = "the pot did not remain bimanually held through transport"
            frame = int(metrics.get("transport_plan", {}).get("end_step", 379))
        elif not checks.get("centered_on_cooktop", False):
            stage = "support_alignment"
            reason = "released pot center remained outside the coded cooktop support region"
            frame = int(protocol.get("steps", 1)) - 1
        else:
            stage = "stable_settle"
            reason = "centered support did not remain valid for the strict terminal window"
            frame = int(protocol.get("steps", 1)) - 1
        intended = result.get("semantic_frames", {}).get("intended_final_pot_pose")
        cooktop = terminal.get("cooktop_pose")
        pot = terminal.get("pot_pose")
        residuals: dict[str, Any] = {
            "right_grasp_frame_count": int(metrics.get("right_grasp_frames", 0)),
            "left_grasp_frame_count": int(metrics.get("left_grasp_frames", 0)),
            "center_margin_m": float(
                parameters.get("center_tolerance_m", 0.03)
                - terminal.get("center_error_m", float("inf"))
            ),
            "minimum_cooktop_clearance_m": float(
                metrics.get("transport_executed", {}).get(
                    "minimum_cooktop_clearance_m", float("nan")
                )
            ),
        }
        if cooktop and pot:
            residuals["pot_center_in_cooktop_frame_m"] = _local_offset(cooktop, pot[:3])
        if cooktop and pot and intended:
            residuals["terminal_minus_intended_in_cooktop_frame_m"] = _local_offset(
                cooktop,
                np.asarray(pot[:3]) - np.asarray(intended[:3]) + np.asarray(cooktop[:3]),
            )
        reference = pot or cooktop
        if reference:
            residuals.update(
                _tracking_residuals(trace, index=min(frame, 199), reference_pose=reference)
            )
        return FailureDiagnosis(stage, reason, residuals, frame)

    if task == "putmarker":
        if not terminal.get("stage1", False):
            stage = "marker_grasp"
            reason = "contact-backed marker lift did not latch"
            frame = 118
        elif not terminal.get("stage2", False):
            stage = "open_drawer"
            reason = "drawer motion did not remain beyond the coded 5 cm opening"
            frame = 418
        elif not terminal.get("stage3", False):
            stage = "marker_cavity_support"
            reason = "released marker did not stably remain in the drawer cavity"
            frame = 520
        else:
            stage = "close_drawer"
            reason = "a drawer joint remained outside the coded closed threshold"
            frame = int(protocol.get("steps", 1)) - 1
        cabinet = terminal.get("cabinet_pose")
        marker = terminal.get("marker_pose")
        target_geometry = result.get("geometry", {}).get("target", {})
        slide_axis = np.asarray(target_geometry.get("slide_axis_local", [1.0, 0.0, 0.0]))
        maximum_open = float(metrics.get("maximum_drawer_open_m", 0.0))
        residuals = {
            "drawer_open_residual_along_slide_axis_m": maximum_open - 0.05,
            "drawer_slide_axis_local": slide_axis.tolist(),
            "terminal_drawer_positions_m": terminal.get("drawer_joint_position", []),
            "right_handle_grasp_frame_count": int(
                metrics.get("right_handle_grasp_frames", 0)
            ),
        }
        if cabinet and marker:
            residuals["marker_center_in_cabinet_frame_m"] = _local_offset(
                cabinet, marker[:3]
            )
        if cabinet:
            residuals.update(
                _tracking_residuals(trace, index=min(frame, 418), reference_pose=cabinet)
            )
        return FailureDiagnosis(stage, reason, residuals, frame)

    if task == "hangmug":
        if not terminal.get("stage1", False):
            stage = "left_mug_grasp"
            reason = "left contact-backed mug lift did not latch"
            frame = 219
        elif not terminal.get("stage2", False):
            stage = "physical_handover"
            reason = "right contact did not support the released elevated mug"
            frame = 419
        else:
            stage = "handle_to_branch_support"
            reason = "released handle did not remain stably supported by the branch"
            frame = 739
        tree = terminal.get("tree_pose")
        mug = terminal.get("mug_pose")
        intended = result.get("semantic_frames", {}).get("intended_final_mug_pose")
        residuals = {
            "left_grasp_frame_count": int(metrics.get("left_grasp_frames", 0)),
            "right_grasp_frame_count": int(metrics.get("right_grasp_frames", 0)),
            "terminal_tree_xy_error_m": float(
                terminal.get("mug_tree_xy_error_m", float("nan"))
            ),
        }
        if tree and mug:
            residuals["mug_center_in_tree_frame_m"] = _local_offset(tree, mug[:3])
        if tree and mug and intended:
            residuals["terminal_minus_intended_in_tree_frame_m"] = _local_offset(
                tree,
                np.asarray(mug[:3]) - np.asarray(intended[:3]) + np.asarray(tree[:3]),
            )
        if tree:
            residuals.update(
                _tracking_residuals(trace, index=min(frame, 419), reference_pose=tree)
            )
        return FailureDiagnosis(stage, reason, residuals, frame)

    raise ValueError(f"unknown semantic task: {task}")
