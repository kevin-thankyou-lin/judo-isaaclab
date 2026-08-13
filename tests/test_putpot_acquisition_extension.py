import numpy as np
import pytest

from judo_isaaclab.put_marker import SkillTrajectory
from run_putpot_skill_program import (
    _assert_acquisition_only_stage,
    _extend_handle_local_acquisition_window,
    _finish_handle_local_acquisition_window_after_latch,
    _parser,
    _quality_environment_kwargs,
    _resolved_program_command,
)


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


def test_acquisition_extension_is_inserted_before_transport_suffix():
    trajectory = _trajectory()
    poses = np.concatenate(
        (
            trajectory.left_poses,
            np.asarray(
                [
                    [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ),
        )
    )
    trajectory = SkillTrajectory(
        left_poses=poses,
        right_poses=poses,
        grippers=np.asarray(
            [[-0.0475, -0.0475], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ),
        stage_names=("approach", "contact_hold", "smooth_transport", "release"),
        waypoint_steps={
            "pregrasp": 0,
            "bimanual_contact_hold": 1,
            "transport_end": 2,
            "release": 3,
        },
    )
    nominal = np.arange(56, dtype=np.float64).reshape(4, 14)

    extended, extended_nominal = _extend_handle_local_acquisition_window(
        trajectory, nominal, 2, acquisition_end_step=1
    )

    assert extended.stage_names == (
        "approach",
        "contact_hold",
        "handle_local_acquisition_extension",
        "handle_local_acquisition_extension",
        "smooth_transport",
        "release",
    )
    np.testing.assert_array_equal(extended.left_poses[2:4], poses[[1, 1]])
    np.testing.assert_array_equal(extended.left_poses[4:], poses[2:])
    np.testing.assert_array_equal(extended_nominal[2:4], nominal[[1, 1]])
    np.testing.assert_array_equal(extended_nominal[4:], nominal[2:])
    assert extended.waypoint_steps["bimanual_contact_hold"] == 1
    assert extended.waypoint_steps["handle_local_acquisition_extension"] == 3
    assert extended.waypoint_steps["transport_end"] == 4
    assert extended.waypoint_steps["release"] == 5

    finished, finished_nominal, removed = (
        _finish_handle_local_acquisition_window_after_latch(
            extended,
            extended_nominal,
            completion_step=2,
        )
    )
    assert removed == 1
    assert finished.stage_names == (
        "approach",
        "contact_hold",
        "handle_local_acquisition_extension",
        "smooth_transport",
        "release",
    )
    np.testing.assert_array_equal(finished.left_poses[3:], poses[2:])
    np.testing.assert_array_equal(finished_nominal[3:], nominal[2:])
    assert finished.waypoint_steps["handle_local_acquisition_extension"] == 2
    assert finished.waypoint_steps["transport_end"] == 3
    assert finished.waypoint_steps["release"] == 4


def test_four_pad_latch_before_extension_removes_all_unused_acquisition_holds():
    trajectory = _trajectory()
    nominal = np.arange(28, dtype=np.float64).reshape(2, 14)
    extended, extended_nominal = _extend_handle_local_acquisition_window(
        trajectory,
        nominal,
        3,
    )
    finished, finished_nominal, removed = (
        _finish_handle_local_acquisition_window_after_latch(
            extended,
            extended_nominal,
            completion_step=1,
        )
    )
    assert removed == 3
    assert finished.stage_names == trajectory.stage_names
    np.testing.assert_array_equal(finished.left_poses, trajectory.left_poses)
    np.testing.assert_array_equal(finished_nominal, nominal)
    assert finished.waypoint_steps["handle_local_acquisition_extension"] == 1


def test_fail_closed_base_command_routes_without_plugin_and_stage_guard_blocks_transport():
    held = {
        "stage": "bimanual_handle_grasp_fail_closed",
        "left_pose": np.arange(7, dtype=np.float64),
        "right_pose": np.arange(7, dtype=np.float64) + 10.0,
        "grippers": np.asarray([0.0, 0.0]),
    }
    assert _resolved_program_command(held, None) is held
    plugin = {"stage": "plugin", "kind": "cartesian_target"}
    assert _resolved_program_command(held, plugin) is plugin
    _assert_acquisition_only_stage(held["stage"])
    with pytest.raises(RuntimeError, match="forbidden stage"):
        _assert_acquisition_only_stage("smooth_bimanual_transport")
    with pytest.raises(RuntimeError, match="forbidden stage"):
        _assert_acquisition_only_stage("release_and_withdraw")


def test_quality_config_is_opt_in_and_legacy_parser_default_is_unchanged():
    required = [
        "--gear-repo", "gear",
        "--source-dataset", "source.hdf5",
        "--target-dataset", "target.hdf5",
        "--objects-root", "objects",
        "--mode", "replay",
        "--trace-npz", "trace.npz",
        "--result-json", "result.json",
    ]
    assert _parser(required).quality_config_json is None
    assert (
        _parser(required + ["--quality-config-json", "quality.json"]).quality_config_json
        == "quality.json"
    )


def test_quality_environment_disables_only_supported_manual_recorder():
    def supported(task_name, enable_manual_recorder=True):
        pass

    def legacy(task_name):
        pass

    assert _quality_environment_kwargs(supported, None) == {}
    assert _quality_environment_kwargs(supported, object()) == {
        "enable_manual_recorder": False
    }
    with pytest.raises(RuntimeError, match="paired Gear support"):
        _quality_environment_kwargs(legacy, object())
