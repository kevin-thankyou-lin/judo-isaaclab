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


REPLAY_HANG_UNLOAD_STEPS = 80
REPLAY_HANG_RELEASE_STEPS = 40
REPLAY_HANG_SETTLE_STEPS = 60


@dataclass(frozen=True)
class ReplayHangTail:
    """A hang-only trajectory derived from the live post-handover state."""

    trajectory: SkillTrajectory
    intended_final_mug_pose: np.ndarray
    right_contact_in_mug: np.ndarray
    source_branch: Any
    target_branch: Any
    planned_mug_poses: np.ndarray
    planning_tree_pose: np.ndarray
    support_alignment: dict[str, Any]
    unload_steps: int
    release_steps: int


def replay_prefix_steps(
    keyframes: dict[str, Any], *, latch_grace_steps: int = 0
) -> int:
    """Return the source-action count ending after a bounded handover grace."""

    try:
        action_index = int(keyframes["frames"]["handover"]["action_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("source keyframes lack a handover action index") from error
    if action_index < 0:
        raise ValueError("handover action index must be nonnegative")
    if latch_grace_steps < 0:
        raise ValueError("latch_grace_steps must be nonnegative")
    return action_index + 1 + latch_grace_steps


def replay_tail_ready(sample: dict[str, Any]) -> bool:
    """Require a real right-only handover before departing from source actions."""

    return bool(
        sample.get("stage2")
        and sample.get("right_grasp")
        and not sample.get("left_grasp")
    )


def replay_tail_steps(
    keyframes: dict[str, Any],
    *,
    unload_steps: int = REPLAY_HANG_UNLOAD_STEPS,
    release_steps: int = REPLAY_HANG_RELEASE_STEPS,
    insert_time_scale: int = 1,
) -> int:
    """Return the source-relationship suffix horizon."""

    try:
        path_steps = len(keyframes["clean_insertion_path"]["mug_poses"])
    except (KeyError, TypeError) as error:
        raise ValueError("source keyframes lack a clean insertion path") from error
    if path_steps < 2:
        raise ValueError("clean insertion path must contain at least two poses")
    if unload_steps < 0:
        raise ValueError("unload_steps must be nonnegative")
    if release_steps <= 0:
        raise ValueError("release_steps must be positive")
    if insert_time_scale < 1:
        raise ValueError("insert_time_scale must be positive")
    return (
        100
        + (path_steps - 1) * insert_time_scale
        + unload_steps
        + release_steps
        + REPLAY_HANG_SETTLE_STEPS
    )


def _source_relationship_mug_path(
    *,
    keyframes: dict[str, Any],
    source_parts: Any,
    target_parts: Any,
    source_branch: Any,
    target_branch: Any,
    target_tree_pose: np.ndarray,
    observed_mug: np.ndarray,
    supported_mug_pose: np.ndarray,
    insert_time_scale: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
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
    mapped = np.asarray(mapped, dtype=np.float64)
    supported = np.asarray(supported_mug_pose, dtype=np.float64)
    original_terminal = mapped[-1].copy()
    world_delta = compose_pose(supported, inverse_pose(original_terminal))
    identity = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    corrections = interpolate_poses(identity, world_delta, len(mapped))
    mapped = np.asarray(
        [compose_pose(correction, pose) for correction, pose in zip(corrections, mapped)]
    )
    if insert_time_scale < 1:
        raise ValueError("insert_time_scale must be positive")
    if insert_time_scale > 1:
        segments = [
            np.concatenate((
                start[None],
                interpolate_poses(start, end, insert_time_scale)[:-1],
            ))
            for start, end in zip(mapped[:-1], mapped[1:])
        ]
        mapped = np.concatenate((*segments, mapped[-1:]), axis=0)
    bridge = interpolate_poses(observed_mug, mapped[0], 100)
    quaternion_dot = abs(float(np.dot(original_terminal[3:], supported[3:])))
    receipt = {
        "method": "quintic_world_pose_alignment_to_branch_midpoint",
        "source_relationship_steps": len(mapped),
        "insert_time_scale": insert_time_scale,
        "original_terminal_pose": original_terminal.tolist(),
        "supported_terminal_pose": supported.tolist(),
        "terminal_translation_m": float(
            np.linalg.norm(supported[:3] - original_terminal[:3])
        ),
        "terminal_rotation_rad": float(
            2.0 * np.arccos(np.clip(quaternion_dot, -1.0, 1.0))
        ),
        "terminal_pose_exact": bool(np.allclose(mapped[-1], supported, atol=1.0e-12)),
    }
    return np.concatenate((bridge, mapped[1:]), axis=0), receipt


def _trajectory_from_mug_path(
    mug_path: np.ndarray,
    *,
    left_pose: np.ndarray,
    right_contact: np.ndarray,
    unload_steps: int = REPLAY_HANG_UNLOAD_STEPS,
    release_steps: int = REPLAY_HANG_RELEASE_STEPS,
) -> SkillTrajectory:
    """Build a right-held insertion and release without resampling the curve."""

    right_insert = np.asarray(
        [compose_pose(pose, right_contact) for pose in mug_path], dtype=np.float64
    )
    if unload_steps < 0:
        raise ValueError("unload_steps must be nonnegative")
    if release_steps <= 0:
        raise ValueError("release_steps must be positive")
    settle_steps = REPLAY_HANG_SETTLE_STEPS
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
    target_branch_policy: str = "source_corresponding",
    unload_steps: int = REPLAY_HANG_UNLOAD_STEPS,
    release_steps: int = REPLAY_HANG_RELEASE_STEPS,
    insert_time_scale: int = 1,
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
    source_branches = tuple(source_branches)
    target_branches = tuple(target_branches)
    frames = keyframes["frames"]
    supported_mug_pose, source_branch, target_branch = geometry_conditioned_hang_pose(
        frames["stable_settle"]["mug_pose"],
        frames["stable_settle"]["tree_pose"],
        source_parts,
        target_parts,
        source_branches,
        target_tree_pose,
        target_branches,
    )
    right_contact = compose_pose(inverse_pose(observed_mug), right_start)
    branch_selection = {
        "policy": target_branch_policy,
        "changed_from_source_correspondence": False,
    }
    if target_branch_policy == "nearest_eef":
        candidates = []
        corresponding = target_branch
        for index, candidate in enumerate(target_branches):
            candidate_mug, _, _ = geometry_conditioned_hang_pose(
                frames["stable_settle"]["mug_pose"],
                frames["stable_settle"]["tree_pose"],
                source_parts,
                target_parts,
                source_branches,
                target_tree_pose,
                target_branches,
                target_branch_override=candidate,
            )
            candidate_eef = compose_pose(candidate_mug, right_contact)
            candidates.append((
                float(np.linalg.norm(candidate_eef[:3] - right_start[:3])),
                index,
                candidate,
                candidate_mug,
                candidate_eef,
            ))
        _, _, target_branch, supported_mug_pose, selected_eef = min(candidates)
        branch_selection = {
            "policy": target_branch_policy,
            "method": "minimum_target_eef_translation_from_observed_handover",
            "changed_from_source_correspondence": target_branch is not corresponding,
            "corresponding_normalized_height": float(
                corresponding.normalized_height
            ),
            "selected_normalized_height": float(target_branch.normalized_height),
            "selected_target_eef_pose": selected_eef.tolist(),
            "candidate_translation_m": [row[0] for row in candidates],
            "selected_candidate_index": next(
                row[1] for row in candidates if row[2] is target_branch
            ),
        }
    elif target_branch_policy != "source_corresponding":
        raise ValueError(f"unsupported target branch policy: {target_branch_policy}")
    planned_mug_poses, support_alignment = _source_relationship_mug_path(
        keyframes=keyframes,
        source_parts=source_parts,
        target_parts=target_parts,
        source_branch=source_branch, target_branch=target_branch,
        target_tree_pose=np.asarray(target_tree_pose, dtype=np.float64),
        observed_mug=observed_mug,
        supported_mug_pose=supported_mug_pose,
        insert_time_scale=insert_time_scale,
    )
    support_alignment["target_branch_selection"] = branch_selection
    trajectory = _trajectory_from_mug_path(
        planned_mug_poses,
        left_pose=left_start,
        right_contact=right_contact,
        unload_steps=unload_steps,
        release_steps=release_steps,
    )
    if trajectory.steps != replay_tail_steps(
        keyframes, unload_steps=unload_steps, release_steps=release_steps,
        insert_time_scale=insert_time_scale,
    ):
        raise RuntimeError("replay hang tail horizon changed unexpectedly")
    return ReplayHangTail(
        trajectory=trajectory,
        intended_final_mug_pose=planned_mug_poses[-1],
        right_contact_in_mug=right_contact,
        source_branch=source_branch,
        target_branch=target_branch,
        planned_mug_poses=planned_mug_poses,
        planning_tree_pose=np.asarray(target_tree_pose, dtype=np.float64),
        support_alignment=support_alignment,
        unload_steps=unload_steps,
        release_steps=release_steps,
    )


def replace_replay_hang_tail_path(
    tail: ReplayHangTail, mug_poses: Any
) -> ReplayHangTail:
    """Rebuild a replay tail around a geometry-corrected mug path."""

    path = np.asarray(mug_poses, dtype=np.float64)
    if path.shape != tail.planned_mug_poses.shape:
        raise ValueError("replacement mug path must preserve the replay-tail horizon")
    trajectory = _trajectory_from_mug_path(
        path,
        left_pose=tail.trajectory.left_poses[0],
        right_contact=tail.right_contact_in_mug,
        unload_steps=tail.unload_steps,
        release_steps=tail.release_steps,
    )
    return ReplayHangTail(
        trajectory=trajectory,
        intended_final_mug_pose=path[-1],
        right_contact_in_mug=tail.right_contact_in_mug,
        source_branch=tail.source_branch,
        target_branch=tail.target_branch,
        planned_mug_poses=path,
        planning_tree_pose=tail.planning_tree_pose,
        support_alignment=tail.support_alignment,
        unload_steps=tail.unload_steps,
        release_steps=tail.release_steps,
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
