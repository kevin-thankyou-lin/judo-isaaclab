"""Deterministic replay-prefix repair for HangMug final insertion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .hang_mug import (
    SkillTrajectory,
    geometry_conditioned_hang_pose,
)
from .put_marker import compose_pose, interpolate_poses, inverse_pose, transfer_pose


@dataclass(frozen=True)
class ReplayHangTail:
    """A hang-only trajectory derived from the live post-handover state."""

    trajectory: SkillTrajectory
    intended_final_mug_pose: np.ndarray
    right_contact_in_mug: np.ndarray
    source_branch: Any
    target_branch: Any
    planned_mug_poses: np.ndarray


def replay_prefix_steps(keyframes: dict[str, Any]) -> int:
    """Return the source-action count ending at the proven handover."""

    try:
        action_index = int(keyframes["frames"]["handover"]["action_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("source keyframes lack a handover action index") from error
    if action_index < 0:
        raise ValueError("handover action index must be nonnegative")
    return action_index + 1


def replay_tail_ready(sample: dict[str, Any]) -> bool:
    """Require a real right-only handover before departing from source actions."""

    return bool(
        sample.get("stage2")
        and sample.get("right_grasp")
        and not sample.get("left_grasp")
    )


def replay_tail_steps(keyframes: dict[str, Any]) -> int:
    """Return the source-relationship suffix horizon."""

    try:
        path_steps = len(keyframes["clean_insertion_path"]["mug_poses"])
    except (KeyError, TypeError) as error:
        raise ValueError("source keyframes lack a clean insertion path") from error
    if path_steps < 2:
        raise ValueError("clean insertion path must contain at least two poses")
    return 100 + path_steps - 1 + 40 + 40 + 60


def _source_relationship_mug_path(
    *,
    keyframes: dict[str, Any],
    source_parts: Any,
    target_parts: Any,
    source_branch: Any,
    target_branch: Any,
    target_tree_pose: np.ndarray,
    observed_mug: np.ndarray,
) -> np.ndarray:
    """Transfer the verified source insertion curve into the target branch frame."""

    frames = keyframes["frames"]
    source_tree_pose = np.asarray(
        frames["stable_settle"]["tree_pose"], dtype=np.float64
    )
    source_branch_world = compose_pose(source_tree_pose, source_branch.frame)
    target_branch_world = compose_pose(target_tree_pose, target_branch.frame)
    local_scale = (
        target_branch.length_m / source_branch.length_m,
        target_parts.handle_outer_size[1] / source_parts.handle_outer_size[1],
        target_parts.handle_outer_size[2] / source_parts.handle_outer_size[2],
    )
    mapped = []
    for source_mug_pose in keyframes["clean_insertion_path"]["mug_poses"]:
        source_handle = compose_pose(source_mug_pose, source_parts.handle_hole_frame)
        target_handle = transfer_pose(
            source_handle,
            source_branch_world,
            target_branch_world,
            local_position_scale=local_scale,
        )
        mapped.append(
            compose_pose(target_handle, inverse_pose(target_parts.handle_hole_frame))
        )
    bridge = interpolate_poses(observed_mug, mapped[0], 100)
    return np.concatenate((bridge, np.asarray(mapped[1:])), axis=0)


def _trajectory_from_mug_path(
    mug_path: np.ndarray,
    *,
    left_pose: np.ndarray,
    right_contact: np.ndarray,
) -> SkillTrajectory:
    """Build a right-held insertion and release without resampling the curve."""

    right_insert = np.asarray(
        [compose_pose(pose, right_contact) for pose in mug_path], dtype=np.float64
    )
    unload_steps, release_steps, settle_steps = 40, 40, 60
    hold = np.repeat(right_insert[-1:], unload_steps + release_steps + settle_steps, axis=0)
    right = np.concatenate((right_insert, hold), axis=0)
    left = np.repeat(left_pose[None], len(right), axis=0)
    grippers = np.empty((len(right), 2), dtype=np.float64)
    grippers[:, 0] = -0.0475
    grippers[:, 1] = 0.0
    release_start = len(right_insert) + unload_steps
    fraction = np.linspace(1.0 / release_steps, 1.0, release_steps)
    smooth = fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)
    grippers[release_start : release_start + release_steps, 1] = -0.0475 * smooth
    grippers[release_start + release_steps :, 1] = -0.0475
    bridge_end = 99
    insert_end = len(right_insert) - 1
    unload_end = insert_end + unload_steps
    release_end = unload_end + release_steps
    return SkillTrajectory(
        left_poses=left,
        right_poses=right,
        grippers=grippers,
        stage_names=(
            ("source_relationship_transport",) * (bridge_end + 1)
            + ("handle_to_branch_insertion",) * (insert_end - bridge_end)
            + ("release_support",) * (unload_steps + release_steps)
            + ("stable_settle",) * settle_steps
        ),
        waypoint_steps={
            "tree_transport": bridge_end,
            "branch_approach": bridge_end,
            "branch_insert": insert_end,
            "branch_unload": unload_end,
            "right_release": release_end,
            "stable_support": len(right) - 1,
        },
    )


def build_replay_hang_tail(
    *,
    keyframes: dict[str, Any],
    source_parts: Any,
    target_parts: Any,
    source_branches: Any,
    target_tree_pose: Any,
    target_branches: Any,
    observed_mug_pose: Any,
    left_eef_pose: Any,
    right_eef_pose: Any,
) -> ReplayHangTail:
    """Build only transport, insertion, and release from observed contact.

    The source action prefix owns pick and handover.  This suffix preserves the
    live mug-to-right-gripper contact frame and uses measured target handle and
    branch geometry to correct the final support pose without a reset or search.
    """

    observed_mug = np.asarray(observed_mug_pose, dtype=np.float64)
    left_start = np.asarray(left_eef_pose, dtype=np.float64)
    right_start = np.asarray(right_eef_pose, dtype=np.float64)
    if any(pose.shape != (7,) for pose in (observed_mug, left_start, right_start)):
        raise ValueError("observed mug and EEF poses must each have shape (7,)")
    frames = keyframes["frames"]
    _, source_branch, target_branch = geometry_conditioned_hang_pose(
        frames["stable_settle"]["mug_pose"],
        frames["stable_settle"]["tree_pose"],
        source_parts,
        target_parts,
        source_branches,
        target_tree_pose,
        target_branches,
    )
    right_contact = compose_pose(inverse_pose(observed_mug), right_start)
    planned_mug_poses = _source_relationship_mug_path(
        keyframes=keyframes,
        source_parts=source_parts,
        target_parts=target_parts,
        source_branch=source_branch,
        target_branch=target_branch,
        target_tree_pose=np.asarray(target_tree_pose, dtype=np.float64),
        observed_mug=observed_mug,
    )
    trajectory = _trajectory_from_mug_path(
        planned_mug_poses,
        left_pose=left_start,
        right_contact=right_contact,
    )
    if trajectory.steps != replay_tail_steps(keyframes):
        raise RuntimeError("replay hang tail horizon changed unexpectedly")
    return ReplayHangTail(
        trajectory=trajectory,
        intended_final_mug_pose=planned_mug_poses[-1],
        right_contact_in_mug=right_contact,
        source_branch=source_branch,
        target_branch=target_branch,
        planned_mug_poses=planned_mug_poses,
    )


def repeated_joint_nominal(
    source_actions: Any, prefix_steps: int, tail_steps: int
) -> np.ndarray:
    """Hold the last replayed joint target as the suffix IK null-space nominal."""

    actions = np.asarray(source_actions, dtype=np.float64)
    if actions.ndim != 2 or not 0 < prefix_steps <= len(actions):
        raise ValueError("prefix_steps must select an action from a 2-D action array")
    if tail_steps <= 0:
        raise ValueError("tail_steps must be positive")
    return np.repeat(actions[prefix_steps - 1 : prefix_steps], tail_steps, axis=0)
