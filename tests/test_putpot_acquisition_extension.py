import numpy as np
import pytest

from judo_isaaclab.put_marker import SkillTrajectory
from run_putpot_skill_program import _extend_handle_local_acquisition_window


def _trajectory() -> SkillTrajectory:
    poses = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return SkillTrajectory(
        left_poses=poses,
        right_poses=poses + np.asarray([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
        grippers=np.asarray([[-0.0475, -0.0475], [0.0, 0.0]]),
        stage_names=("approach", "contact_hold"),
        waypoint_steps={"pregrasp": 0, "bimanual_contact_hold": 1},
    )


def test_acquisition_extension_is_bounded_and_repeats_only_terminal_hold():
    trajectory = _trajectory()
    nominal = np.arange(28, dtype=np.float64).reshape(2, 14)
    extended, extended_nominal = _extend_handle_local_acquisition_window(
        trajectory, nominal, 3
    )

    assert trajectory.steps == 2
    assert extended.steps == 5
    np.testing.assert_array_equal(extended.left_poses[:2], trajectory.left_poses)
    np.testing.assert_array_equal(extended.right_poses[:2], trajectory.right_poses)
    np.testing.assert_array_equal(
        extended.left_poses[2:], np.repeat(trajectory.left_poses[-1][None], 3, axis=0)
    )
    np.testing.assert_array_equal(
        extended.grippers[2:], np.repeat(trajectory.grippers[-1][None], 3, axis=0)
    )
    np.testing.assert_array_equal(
        extended_nominal[2:], np.repeat(nominal[-1][None], 3, axis=0)
    )
    assert extended.stage_names[2:] == (
        "handle_local_acquisition_extension",
    ) * 3
    assert extended.waypoint_steps["handle_local_acquisition_extension"] == 4


def test_acquisition_extension_rejects_unbounded_or_repeated_windows():
    trajectory = _trajectory()
    nominal = np.zeros((2, 14), dtype=np.float64)
    with pytest.raises(ValueError, match=r"\[0, 120\]"):
        _extend_handle_local_acquisition_window(trajectory, nominal, 121)
    extended, extended_nominal = _extend_handle_local_acquisition_window(
        trajectory, nominal, 1
    )
    with pytest.raises(ValueError, match="already extended"):
        _extend_handle_local_acquisition_window(
            extended, extended_nominal, 1
        )


def test_zero_acquisition_extension_preserves_non_contact_behavior():
    trajectory = _trajectory()
    nominal = np.zeros((2, 14), dtype=np.float64)
    unchanged_trajectory, unchanged_nominal = _extend_handle_local_acquisition_window(
        trajectory, nominal, 0
    )
    assert unchanged_trajectory is trajectory
    assert unchanged_nominal is nominal
