"""Incrementally keep a HangMug insertion suffix in the observed tree frame."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from judo_isaaclab.put_marker import (
    SkillTrajectory,
    compose_pose,
    interpolate_poses,
    inverse_pose,
)


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    dot = abs(float(np.dot(left[3:], right[3:])))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


@dataclass
class ObservedTreePathTracker:
    """Co-move the held support suffix with observed tree motion.

    The insertion path itself is deliberately left unchanged.  Tracking a
    contact-induced tree displacement during approach creates a positive
    feedback loop: the wrist chases the displaced branch and can push the mug
    body into the tree.  Observations before ``branch_insert`` therefore update
    only the tracking baseline.  Once insertion completes, the handle is in the
    branch and the remaining held-support poses follow only subsequent,
    incremental tree motion.
    """

    planning_pose: np.ndarray
    observed_pose: np.ndarray | None = None
    updates: int = 0

    def __post_init__(self) -> None:
        self.reset(self.planning_pose)

    def reset(self, planning_pose: Any) -> None:
        pose = np.asarray(planning_pose, dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError("planning tree pose must have shape (7,)")
        self.planning_pose = pose.copy()
        self.observed_pose = pose.copy()
        self.updates = 0

    def _emit_receipt(
        self, trajectory: SkillTrajectory, step: int,
        *, current: np.ndarray, translation: float, rotation: float,
    ) -> None:
        insert = int(trajectory.waypoint_steps["branch_insert"])
        unload = int(trajectory.waypoint_steps["branch_unload"])
        if step not in {insert, (insert + unload) // 2, unload - 1}:
            return
        print(
            "HANGMUG_CONTINUOUS_TREE_TRACKING="
            + json.dumps(
                {
                    "cumulative_rotation_rad": _rotation_distance(
                        current, self.planning_pose
                    ),
                    "cumulative_translation_m": float(
                        np.linalg.norm(current[:3] - self.planning_pose[:3])
                    ),
                    "incremental_rotation_rad": rotation,
                    "incremental_translation_m": translation,
                    "trajectory_step": int(step),
                    "updates": self.updates,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def update(
        self,
        trajectory: SkillTrajectory | None,
        trajectory_step: int | None,
        sample: dict[str, Any],
    ) -> SkillTrajectory | None:
        current = np.asarray(sample["tree_pose"], dtype=np.float64)
        if trajectory is None or trajectory_step is None or not sample["right_grasp"]:
            return trajectory
        insert = int(trajectory.waypoint_steps["branch_insert"])
        unload = int(trajectory.waypoint_steps["branch_unload"])
        if trajectory_step < insert:
            self.observed_pose = current.copy()
            if trajectory_step == insert - 1:
                print(
                    "HANGMUG_TREE_TRACKING_BASELINE="
                    + json.dumps(
                        {
                            "cumulative_rotation_rad": _rotation_distance(
                                current, self.planning_pose
                            ),
                            "cumulative_translation_m": float(
                                np.linalg.norm(
                                    current[:3] - self.planning_pose[:3]
                                )
                            ),
                            "trajectory_step": int(trajectory_step),
                            "support_tracking_starts_at": insert,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return trajectory
        if not insert <= trajectory_step < unload:
            return trajectory
        previous = np.asarray(self.observed_pose, dtype=np.float64)
        self.observed_pose = current.copy()
        translation = float(np.linalg.norm(current[:3] - previous[:3]))
        rotation = _rotation_distance(current, previous)
        if translation <= 1.0e-6 and rotation <= 1.0e-5:
            self._emit_receipt(
                trajectory, trajectory_step, current=current,
                translation=translation, rotation=rotation,
            )
            return trajectory
        world_delta = compose_pose(current, inverse_pose(previous))
        right = np.asarray(trajectory.right_poses, dtype=np.float64).copy()
        start = trajectory_step + 1
        identity = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        held_steps = unload - start + 1
        if held_steps > 0:
            corrections = interpolate_poses(identity, world_delta, held_steps)
            right[start : unload + 1] = np.asarray(
                [
                    compose_pose(correction, pose)
                    for correction, pose in zip(
                        corrections, right[start : unload + 1]
                    )
                ],
                dtype=np.float64,
            )
        if unload + 1 < len(right):
            right[unload + 1 :] = np.asarray(
                [
                    compose_pose(world_delta, pose)
                    for pose in right[unload + 1 :]
                ],
                dtype=np.float64,
            )
        self.updates += 1
        self._emit_receipt(
            trajectory, trajectory_step, current=current,
            translation=translation, rotation=rotation,
        )
        return SkillTrajectory(
            left_poses=trajectory.left_poses.copy(),
            right_poses=right,
            grippers=trajectory.grippers.copy(),
            stage_names=trajectory.stage_names,
            waypoint_steps=dict(trajectory.waypoint_steps),
        )
