from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from hangmug_tree_tracking import ObservedTreePathTracker
from judo_isaaclab.put_marker import SkillTrajectory


def _pose(x=0.0, y=0.0, z=0.0):
    return np.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0])


def _trajectory() -> SkillTrajectory:
    poses = np.asarray([_pose(1.0) for _ in range(5)])
    return SkillTrajectory(
        left_poses=poses.copy(),
        right_poses=poses.copy(),
        grippers=np.zeros((5, 2)),
        stage_names=("insert",) * 5,
        waypoint_steps={
            "branch_approach": 0,
            "branch_insert": 3,
            "branch_unload": 4,
        },
    )


def test_tracker_comoves_only_unexecuted_suffix_with_full_tree_se3(capsys):
    tracker = ObservedTreePathTracker(_pose())
    yaw_90 = np.asarray([
        2.0, 3.0, 0.0, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)
    ])
    original = _trajectory()
    tracked = tracker.update(
        original,
        3,
        {"tree_pose": yaw_90.tolist(), "right_grasp": True},
    )
    assert tracked.right_poses[:4] == pytest.approx(original.right_poses[:4])
    assert tracked.right_poses[4, :3] == pytest.approx([2.0, 4.0, 0.0])
    assert abs(float(np.dot(tracked.right_poses[4, 3:], yaw_90[3:]))) == (
        pytest.approx(1.0)
    )
    assert "HANGMUG_CONTINUOUS_TREE_TRACKING=" in capsys.readouterr().out


def test_tracker_does_not_move_path_without_right_grasp():
    tracker = ObservedTreePathTracker(_pose())
    original = _trajectory()
    tracked = tracker.update(
        original,
        2,
        {"tree_pose": _pose(0.1).tolist(), "right_grasp": False},
    )
    assert tracked is original
    assert tracker.updates == 0
