"""Deterministic replay-prefix repair for HangMug final insertion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .hang_mug import (
    HangMugSkillProgram,
    SkillTrajectory,
    geometry_conditioned_hang_pose,
)
from .put_marker import compose_pose, inverse_pose, quaternion_rotate


@dataclass(frozen=True)
class ReplayHangTail:
    """A hang-only trajectory derived from the live post-handover state."""

    trajectory: SkillTrajectory
    intended_final_mug_pose: np.ndarray
    right_contact_in_mug: np.ndarray
    source_branch: Any
    target_branch: Any


def replay_prefix_steps(keyframes: dict[str, Any]) -> int:
    """Return the source-action count ending at the held insertion keyframe."""

    try:
        action_index = int(keyframes["frames"]["inserted_held"]["action_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("source keyframes lack an inserted_held action index") from error
    if action_index < 0:
        raise ValueError("inserted_held action index must be nonnegative")
    return action_index + 1


def replay_tail_ready(sample: dict[str, Any]) -> bool:
    """Require a real right-only handover before departing from source actions."""

    return bool(
        sample.get("stage2")
        and sample.get("right_grasp")
        and not sample.get("left_grasp")
    )


def replay_tail_steps() -> int:
    """Fixed horizon of the deterministic geometry-conditioned suffix."""

    return 60 + 50 + 50 + 40 + 40 + 60


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
    insert_clearance_m: float,
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
    if insert_clearance_m <= 0.0:
        raise ValueError("insert_clearance_m must be positive")

    frames = keyframes["frames"]
    final_mug_pose, source_branch, target_branch = geometry_conditioned_hang_pose(
        frames["stable_settle"]["mug_pose"],
        frames["stable_settle"]["tree_pose"],
        source_parts,
        target_parts,
        source_branches,
        target_tree_pose,
        target_branches,
    )
    target_branch_world = compose_pose(target_tree_pose, target_branch.frame)
    branch_tangent = quaternion_rotate(
        target_branch_world[3:], np.asarray([1.0, 0.0, 0.0])
    )
    right_contact = compose_pose(inverse_pose(observed_mug), right_start)

    transport_mug = observed_mug.copy()
    transport_mug[:2] = 0.5 * (observed_mug[:2] + final_mug_pose[:2])
    transport_mug[2] = max(
        observed_mug[2], final_mug_pose[2] + float(insert_clearance_m)
    )
    approach_mug = final_mug_pose.copy()
    approach_mug[:3] += branch_tangent * float(insert_clearance_m)
    approach_mug[2] += 0.03

    program = HangMugSkillProgram(
        left_start,
        right_start,
        left_gripper=-0.0475,
        right_gripper=0.0,
    )
    program.handle_to_branch_insert(
        compose_pose(transport_mug, right_contact),
        compose_pose(approach_mug, right_contact),
        compose_pose(final_mug_pose, right_contact),
        transport_steps=60,
        approach_steps=50,
        insert_steps=50,
        left_observer=left_start,
    )
    right_insert = compose_pose(final_mug_pose, right_contact)
    program.release_and_support(
        right_insert,
        right_insert,
        unload_steps=40,
        release_steps=40,
        settle_steps=60,
    )
    trajectory = program.build()
    if trajectory.steps != replay_tail_steps():
        raise RuntimeError("replay hang tail horizon changed unexpectedly")
    return ReplayHangTail(
        trajectory=trajectory,
        intended_final_mug_pose=final_mug_pose,
        right_contact_in_mug=right_contact,
        source_branch=source_branch,
        target_branch=target_branch,
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
