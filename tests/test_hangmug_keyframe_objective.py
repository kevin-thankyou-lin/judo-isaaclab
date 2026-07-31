import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from hangmug_grasp_keyframe_mpc import (
    _apply_history_control_overrides,
    _correspond_pose_between_branches,
    _expand_control_corrections,
    _expand_task_space_program,
    _objective_components,
    _quat_to_matrix,
    _semantic_base_controls,
    _semantic_reference_trajectory,
    _subtask_reached,
    _task_program_knots,
    insert,
)
from render_hangmug_mpc_comparison import _quaternion_error


@pytest.fixture
def rollout_inputs():
    states = np.zeros((2, 2, 36), dtype=np.float32)
    reference = np.zeros((2, 36), dtype=np.float32)
    states[:, :, 3] = 1.0
    reference[:, 3] = 1.0
    states[:, :, 32] = 1.0
    reference[:, 32] = 1.0
    sensors = np.zeros((2, 2, 10), dtype=np.float32)
    controls = np.zeros((2, 2, 14), dtype=np.float32)
    nominal = np.zeros((2, 14), dtype=np.float32)
    return states, sensors, controls, reference, nominal


@pytest.mark.parametrize(
    ("target_name", "sensor_index"),
    (
        ("left_grasp", 2),
        ("right_grasp", 6),
        ("handover_latched", 8),
        ("hang_complete", 9),
    ),
)
def test_target_success_selects_expected_sensor(
    rollout_inputs, target_name, sensor_index
):
    states, sensors, controls, reference, nominal = rollout_inputs
    sensors[0, :, sensor_index] = 1.0
    if target_name == "left_grasp":
        sensors[0, :, 3] = 1.0
    elif target_name == "right_grasp":
        sensors[0, :, 2] = 1.0
        sensors[0, :, 7] = 1.0
    elif target_name == "handover_latched":
        sensors[0, :, 6] = 1.0

    result = _objective_components(
        states,
        sensors,
        controls,
        reference=reference,
        nominal=nominal,
        keyframe_offset=0,
        target_name=target_name,
    )

    assert result["keyframe_target_success"].tolist() == [True, False]
    assert result["rewards"][0] > result["rewards"][1]


def test_video_quaternion_error_is_sign_invariant():
    quaternion = np.array([0.5, 0.5, 0.5, 0.5])

    assert _quaternion_error(quaternion, -quaternion) == pytest.approx(0.0)


@pytest.mark.parametrize("target_name", ("tree_approach", "inserted_held"))
def test_tree_targets_require_latched_handover_and_right_grasp(
    rollout_inputs, target_name
):
    states, sensors, controls, reference, nominal = rollout_inputs
    sensors[0, :, 6] = 1.0
    sensors[0, :, 8] = 1.0
    sensors[1, :, 6] = 1.0

    result = _objective_components(
        states,
        sensors,
        controls,
        reference=reference,
        nominal=nominal,
        keyframe_offset=0,
        target_name=target_name,
    )

    assert result["keyframe_target_success"].tolist() == [True, False]
    assert result["rewards"][0] > result["rewards"][1]


def test_tree_relative_target_is_invariant_to_tree_translation_and_yaw(
    rollout_inputs,
):
    states, sensors, controls, reference, nominal = rollout_inputs
    reference[:, :3] = [1.0, 0.0, 0.0]
    yaw = np.sqrt(0.5)
    states[0, :, :3] = [2.0, 4.0, 0.0]
    states[0, :, 3:7] = [yaw, 0.0, 0.0, yaw]
    states[0, :, 29:32] = [2.0, 3.0, 0.0]
    states[0, :, 32:36] = [yaw, 0.0, 0.0, yaw]
    states[1, :, :3] = [2.1, 4.0, 0.0]
    states[1, :, 3:7] = [yaw, 0.0, 0.0, yaw]
    states[1, :, 29:32] = [2.0, 3.0, 0.0]
    states[1, :, 32:36] = [yaw, 0.0, 0.0, yaw]
    sensors[:, :, 6] = 1.0
    sensors[:, :, 8] = 1.0

    result = _objective_components(
        states,
        sensors,
        controls,
        reference=reference,
        nominal=nominal,
        keyframe_offset=0,
        target_name="inserted_held",
    )

    assert result["target_frame"] == "mug_relative_to_tree_root"
    assert result["keyframe_position_error_m"][0] == pytest.approx(
        0.0, abs=1.0e-6
    )
    assert result["keyframe_rotation_error_rad"][0] == pytest.approx(0.0)
    assert result["keyframe_position_error_m"][1] == pytest.approx(0.1)


def test_branch_correspondence_adapts_to_a_taller_branch(rollout_inputs):
    states, sensors, controls, reference, nominal = rollout_inputs
    source_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    target_points = source_points + [0.0, 0.0, 0.2]
    reference[:, :3] = [0.0, 0.0, 0.5]
    states[0, :, :3] = [0.0, 0.0, 0.7]
    states[1, :, :3] = [0.0, 0.0, 0.5]
    sensors[:, :, 6] = 1.0
    sensors[:, :, 8] = 1.0

    result = _objective_components(
        states,
        sensors,
        controls,
        reference=reference,
        nominal=nominal,
        keyframe_offset=0,
        target_name="inserted_held",
        source_branch_points=source_points,
        target_branch_points=target_points,
    )

    assert result["target_frame"] == "mug_relative_to_corresponded_branch"
    assert result["keyframe_position_error_m"][0] == pytest.approx(
        0.0, abs=1.0e-6
    )
    assert result["keyframe_position_error_m"][1] == pytest.approx(0.2)
    assert result["rewards"][0] - result["rewards"][1] > 10.0


def test_pose_correspondence_preserves_branch_relative_pose_and_orientation():
    source_points = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    target_points = np.array(
        [[0.0, 0.0, 0.2], [0.0, 1.0, 1.2], [1.0, 0.0, 0.2]]
    )
    tree_pose = np.array([0.5, -0.3, 0.8, 1.0, 0.0, 0.0, 0.0])
    pose = np.array([0.52, -0.2, 0.84, 1.0, 0.0, 0.0, 0.0])

    mapped = _correspond_pose_between_branches(
        pose,
        tree_pose,
        tree_pose,
        source_points,
        target_points,
    )

    source_origin, source_rotation = (
        source_points[0],
        np.stack(
            (
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ),
            axis=-1,
        ),
    )
    target_origin, target_rotation = (
        target_points[0],
        np.stack(
            (
                [0.0, np.sqrt(0.5), np.sqrt(0.5)],
                [1.0, 0.0, 0.0],
                [0.0, np.sqrt(0.5), -np.sqrt(0.5)],
            ),
            axis=-1,
        ),
    )
    source_coordinates = source_rotation.T @ (
        pose[:3] - tree_pose[:3] - source_origin
    )
    expected_position = (
        tree_pose[:3] + target_origin + target_rotation @ source_coordinates
    )

    assert mapped[:3] == pytest.approx(expected_position)
    mapped_rotation = _quat_to_matrix(mapped[3:7])
    expected_rotation = target_rotation @ source_rotation.T
    assert mapped_rotation == pytest.approx(expected_rotation, abs=1.0e-6)


def test_acceptance_depends_only_on_subtask_completion():
    group = {
        "count": 6,
        "acceptance_success_count": 6,
        "keyframe_position_error_m_max": 100.0,
        "keyframe_rotation_error_rad_max": np.pi,
    }

    assert _subtask_reached(group)
    group["acceptance_success_count"] = 5
    assert not _subtask_reached(group)


def test_hang_completion_rejects_latched_but_misaligned_mug(rollout_inputs):
    states, sensors, controls, reference, nominal = rollout_inputs
    states[1, :, :3] = [0.1, 0.0, 0.0]
    sensors[:, :, 8:10] = 1.0

    result = _objective_components(
        states,
        sensors,
        controls,
        reference=reference,
        nominal=nominal,
        keyframe_offset=1,
        target_name="hang_complete",
    )

    assert result["keyframe_target_success"].tolist() == [True, False]
    assert result["acceptance_window_fraction"].tolist() == [1.0, 0.0]


def test_smooth_corrections_interpolate_and_preserve_grippers():
    nominal = np.ones((5, 14), dtype=np.float32)
    knots = np.zeros((1, 2, 14), dtype=np.float32)
    knots[0, 1, 7:13] = 0.2
    knots[0, 1, :6] = 0.3
    knots[0, :, (6, 13)] = 1.0

    controls = _expand_control_corrections(
        knots,
        nominal,
        max_action_delta=0.1,
        right_arm_only=True,
    )

    assert controls.shape == (1, 5, 14)
    assert controls[0, :, :7] == pytest.approx(1.0)
    assert controls[0, :, 13] == pytest.approx(1.0)
    assert controls[0, :, 7:13].max() == pytest.approx(1.1)


def test_task_program_smoothly_interpolates_between_stage_offsets():
    knots = _task_program_knots(
        5,
        translation_start=[0.0, 0.0, 0.04],
        translation_goal=[0.0, 0.0, 0.12],
    )

    assert knots.shape == (5, 6)
    assert knots[0, :3] == pytest.approx([0.0, 0.0, 0.04])
    assert knots[-1, :3] == pytest.approx([0.0, 0.0, 0.12])
    assert np.diff(knots[:, 2]).min() >= 0.0
    assert knots[:, 3:] == pytest.approx(0.0)


def test_single_task_knot_uses_goal_offset():
    knots = _task_program_knots(
        1,
        translation_start=[0.0, 0.0, 0.0],
        translation_goal=[0.0, 0.0, 0.12],
    )

    assert knots[0, :3] == pytest.approx([0.0, 0.0, 0.12])


def test_task_program_preserves_base_actions_and_clips_search_delta():
    nominal = np.arange(56, dtype=np.float32).reshape(4, 14)
    program = _task_program_knots(
        2,
        translation_start=[0.0, 0.0, 0.0],
        translation_goal=[0.0, 0.0, 0.1],
    )
    candidates = np.stack((program, program))
    candidates[1, :, 0] += 1.0
    candidates[1, :, 3] += 1.0

    controls = _expand_task_space_program(
        candidates,
        nominal,
        program,
        max_translation_delta=0.02,
        max_rotation_delta=0.1,
    )

    assert controls.shape == (2, 4, 20)
    assert controls[:, :, :14] == pytest.approx(
        np.broadcast_to(nominal, (2, *nominal.shape))
    )
    assert controls[1, :, 14] - controls[0, :, 14] == pytest.approx(0.02)
    assert controls[1, :, 17] - controls[0, :, 17] == pytest.approx(0.1)


def test_semantic_reference_starts_from_live_pose_and_ends_at_keyframe():
    start = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    target = np.array(
        [0.3, -0.2, 0.1, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    )

    reference = _semantic_reference_trajectory(start, target, horizon=4)

    assert reference.shape == (4, 7)
    assert reference[-1] == pytest.approx(target)
    assert np.linalg.norm(reference[0, :3] - start[:3]) < np.linalg.norm(
        target[:3] - start[:3]
    )
    assert np.linalg.norm(reference[:, 3:7], axis=-1) == pytest.approx(1.0)


def test_insert_uses_branch_frame_offsets_and_holds_seated_pose():
    start = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    target = np.array([0.5, 0.2, 0.1, 1.0, 0.0, 0.0, 0.0])
    branch_rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    reference = insert(
        start,
        target,
        branch_rotation,
        horizon=10,
        approach_offset_branch=(0.1, 0.0, 0.0),
        seat_offset_branch=(0.0, 0.0, -0.02),
        approach_fraction=0.4,
        seat_fraction=0.8,
    )

    assert reference.shape == (10, 7)
    assert reference[3, :3] == pytest.approx([0.5, 0.3, 0.1])
    assert reference[7:, :3] == pytest.approx(
        np.broadcast_to([0.5, 0.2, 0.08], (3, 3))
    )
    assert reference[:, 3:7] == pytest.approx(
        np.broadcast_to([1.0, 0.0, 0.0, 0.0], (10, 4))
    )


def test_insert_rejects_invalid_phase_order():
    with pytest.raises(ValueError, match="0 < approach < seat < 1"):
        insert(
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            np.eye(3),
            horizon=4,
            approach_fraction=0.9,
            seat_fraction=0.8,
        )


@pytest.mark.parametrize(
    ("target_name", "expected_right_gripper"),
    (
        ("inserted_held", [13.0, 13.0, 13.0]),
        ("hang_complete", [13.0, 27.0, 41.0]),
    ),
)
def test_semantic_base_uses_only_hold_release_intent(
    target_name, expected_right_gripper
):
    nominal = np.arange(42, dtype=np.float32).reshape(3, 14)

    controls = _semantic_base_controls(nominal, target_name)

    assert controls[:, :7] == pytest.approx(
        np.broadcast_to(nominal[0, :7], (3, 7))
    )
    assert controls[:, 7:13] == pytest.approx(nominal[:, 7:13])
    assert controls[:, 13] == pytest.approx(expected_right_gripper)


def test_objective_accepts_task_space_control_reference(rollout_inputs):
    states, sensors, _, reference, nominal = rollout_inputs
    controls = np.zeros((2, 2, 20), dtype=np.float32)
    control_reference = np.zeros((2, 20), dtype=np.float32)

    result = _objective_components(
        states,
        sensors,
        controls,
        reference=reference,
        nominal=nominal,
        control_reference=control_reference,
        keyframe_offset=0,
        target_name="left_grasp",
    )

    assert result["action_delta_rms"] == pytest.approx(0.0)


def test_history_controls_replace_earlier_stage(tmp_path):
    path = tmp_path / "stage.npz"
    replacement = np.full((2, 14), 7.0, dtype=np.float32)
    np.savez_compressed(
        path,
        best_sample=replacement,
        start_state=np.int64(12),
    )

    result = _apply_history_control_overrides(
        np.zeros((6, 14), dtype=np.float32),
        checkpoint_state=10,
        start_state=16,
        controls_paths=[path],
    )

    assert result[:2] == pytest.approx(0.0)
    assert result[2:4] == pytest.approx(7.0)
    assert result[4:] == pytest.approx(0.0)
