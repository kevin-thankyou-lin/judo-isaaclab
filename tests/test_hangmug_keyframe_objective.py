import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from hangmug_grasp_keyframe_mpc import (
    _align_corresponding_handle_point,
    _apply_left_visibility_pose,
    _apply_history_control_overrides,
    _correspond_pose_between_branches,
    _expand_control_corrections,
    _expand_task_space_program,
    _eef_target_for_mug_target,
    _initialize_task_stage,
    _objective_components,
    _pose_compose,
    _pose_inverse,
    _quat_to_matrix,
    _semantic_base_controls,
    _semantic_reference_to_keyframe,
    _semantic_reference_trajectory,
    _subtask_reached,
    _task_program_knots,
    grasp,
    insert,
    release,
)
from render_hangmug_mpc_comparison import _quaternion_error


class _FillValue:
    def __init__(self):
        self.value = None

    def fill_(self, value):
        self.value = value


def test_initial_task_stage_restores_phase_latches_only():
    env = type(
        "Env",
        (),
        {
            name: _FillValue()
            for name in (
                "stage1_success",
                "stage2_success",
                "stage3_success",
                "_prev_stage1_success",
                "_prev_stage2_success",
                "_prev_stage3_success",
                "_stage2_reward_given",
                "_stage3_reward_given",
            )
        },
    )()

    _initialize_task_stage(env, 2)

    assert env.stage1_success.value is True
    assert env.stage2_success.value is True
    assert env.stage3_success.value is False
    assert env._stage2_reward_given.value is True


def test_handle_correspondence_shifts_mug_origin_to_align_landmarks():
    pose = np.asarray([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    source = np.asarray([0.06, 0.01, 0.02])
    target = np.asarray([0.04, -0.01, 0.01])

    aligned = _align_corresponding_handle_point(pose, source, target)

    np.testing.assert_allclose(aligned[:3] + target, pose[:3] + source)
    np.testing.assert_allclose(aligned[3:], pose[3:])


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


def test_handover_latch_without_right_grasp_is_not_success(rollout_inputs):
    states, sensors, controls, reference, nominal = rollout_inputs
    sensors[0, :, 8] = 1.0
    sensors[1, :, 8] = 1.0
    sensors[1, :, 6] = 1.0

    result = _objective_components(
        states,
        sensors,
        controls,
        reference=reference,
        nominal=nominal,
        keyframe_offset=0,
        target_name="handover_latched",
    )

    assert result["keyframe_target_success"].tolist() == [False, True]


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


def test_handle_support_allows_seating_depth_but_rejects_radial_error(
    rollout_inputs,
):
    states, sensors, controls, reference, nominal = rollout_inputs
    branch_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    states[0, :, :3] = [0.02, 0.0, 0.0]
    states[1, :, :3] = [0.02, 0.02, 0.0]
    sensors[:, :, 6] = 1.0
    sensors[:, :, 8] = 1.0

    result = _objective_components(
        states,
        sensors,
        controls,
        reference=reference,
        nominal=nominal,
        keyframe_offset=1,
        target_name="inserted_held",
        source_branch_points=branch_points,
        target_branch_points=branch_points,
        source_handle_point_mug=np.zeros(3),
        target_handle_point_mug=np.zeros(3),
    )

    assert result["target_frame"] == (
        "handle_hole_relative_to_corresponded_branch"
    )
    assert result["keyframe_position_error_m"].tolist() == pytest.approx(
        [0.0, 0.02]
    )
    assert result["keyframe_insertion_depth_supported"].tolist() == [
        True,
        True,
    ]
    assert result["keyframe_target_success"].tolist() == [True, False]


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


def test_eef_target_preserves_live_eef_to_mug_grasp_transform():
    yaw = np.sqrt(0.5)
    current_eef = np.array([0.1, -0.2, 0.3, yaw, 0.0, 0.0, yaw])
    eef_to_mug = np.array([0.04, 0.01, -0.02, 1.0, 0.0, 0.0, 0.0])
    current_mug = _pose_compose(current_eef, eef_to_mug)
    target_mug = np.array([0.8, 0.3, 0.6, yaw, 0.0, yaw, 0.0])

    target_eef = _eef_target_for_mug_target(
        current_eef, current_mug, target_mug
    )

    assert _pose_compose(target_eef, eef_to_mug) == pytest.approx(
        target_mug, abs=1.0e-6
    )
    identity = _pose_compose(target_eef, _pose_inverse(target_eef))
    assert identity == pytest.approx(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], abs=1.0e-6
    )


def test_inserted_held_rejects_orientation_mismatch_and_unstable_window(
    rollout_inputs,
):
    states, sensors, controls, reference, nominal = rollout_inputs
    sensors[:, :, 6] = 1.0
    sensors[:, :, 8] = 1.0
    angle = 0.3
    states[0, 1, 3:7] = [
        np.cos(angle / 2.0),
        0.0,
        0.0,
        np.sin(angle / 2.0),
    ]
    states[1, 0, :3] = [0.02, 0.0, 0.0]

    result = _objective_components(
        states,
        sensors,
        controls,
        reference=reference,
        nominal=nominal,
        keyframe_offset=1,
        target_name="inserted_held",
    )

    assert result["keyframe_target_success"].tolist() == [False, True]
    assert result["acceptance_window_fraction"].tolist() == [0.5, 0.5]


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


def test_hang_completion_uses_existing_stable_task_latch(rollout_inputs):
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

    assert result["keyframe_target_success"].tolist() == [True, True]
    assert result["acceptance_window_fraction"].tolist() == [1.0, 1.0]


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


def test_semantic_reference_reaches_acceptance_step_then_holds():
    start = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    target = np.asarray([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])

    reference = _semantic_reference_to_keyframe(
        start, target, horizon=6, keyframe_offset=2
    )

    assert reference.shape == (6, 7)
    assert reference[2:] == pytest.approx(
        np.broadcast_to(target, (4, 7))
    )


def test_grasp_approaches_outside_object_then_moves_inward():
    start = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    target = np.asarray([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    reference = grasp(
        start,
        target,
        np.eye(3),
        10,
        pregrasp_offset_object=[0.1, 0.0, 0.0],
        approach_fraction=0.5,
        contact_fraction=0.8,
    )

    assert reference[4, :3] == pytest.approx([0.6, 0.0, 0.0])
    assert reference[7:, :3] == pytest.approx(
        np.broadcast_to(target[:3], (3, 3))
    )
    assert reference[7:, 3:7] == pytest.approx(
        np.broadcast_to(target[3:7], (3, 4)), abs=1e-6
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


def test_insert_default_backs_out_then_seats_along_branch_tangent():
    target = np.array([0.2, -0.1, 0.5, 1.0, 0.0, 0.0, 0.0])
    reference = insert(
        np.array([0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]),
        target,
        np.eye(3),
        horizon=20,
    )

    assert reference[11, :3] == pytest.approx(
        target[:3] + [0.05, 0.0, 0.0]
    )
    assert reference[16:, :3] == pytest.approx(
        np.broadcast_to(target[:3], (4, 3))
    )
    # Direct tip alignment never uses the old under-branch clearance lane.
    assert reference[:, 2].min() >= 0.3
    assert reference[11, 1:3] == pytest.approx(target[1:3])


def test_insert_aligns_orientation_before_tangent_seating():
    target_angle = 0.4
    target = np.array(
        [
            0.2,
            -0.1,
            0.5,
            np.cos(target_angle / 2.0),
            0.0,
            0.0,
            np.sin(target_angle / 2.0),
        ]
    )
    reference = insert(
        np.array([0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]),
        target,
        np.eye(3),
        horizon=20,
        approach_fraction=0.6,
        seat_fraction=0.8,
    )

    assert reference[11, 3:7] == pytest.approx(target[3:7])
    assert reference[12:, 3:7] == pytest.approx(
        np.broadcast_to(target[3:7], (8, 4))
    )


def test_insert_clearance_waypoint_precedes_tip_approach():
    target = np.array([0.5, 0.2, 0.1, 1.0, 0.0, 0.0, 0.0])
    reference = insert(
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        target,
        np.eye(3),
        horizon=20,
        clearance_offset_branch=(-0.02, 0.03, 0.04),
        clearance_fraction=0.2,
        approach_offset_branch=(0.05, 0.0, 0.0),
        approach_fraction=0.6,
        seat_fraction=0.85,
    )

    assert reference[3, :3] == pytest.approx([0.48, 0.23, 0.14])
    assert reference[11, :3] == pytest.approx([0.55, 0.2, 0.1])
    assert reference[16:, :3] == pytest.approx(
        np.broadcast_to(target[:3], (4, 3))
    )


def test_insert_applies_calibrated_branch_frame_eef_offset():
    reference = insert(
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.2, 0.1, 1.0, 0.0, 0.0, 0.0]),
        np.eye(3),
        horizon=10,
        target_position_offset_branch=(0.01, -0.02, 0.03),
        target_rotation_offset_branch=(0.0, 0.0, 0.1),
        approach_fraction=0.4,
        seat_fraction=0.8,
    )

    assert reference[-1, :3] == pytest.approx([0.51, 0.18, 0.13])
    assert reference[-1, 3:7] == pytest.approx(
        [np.cos(0.05), 0.0, 0.0, np.sin(0.05)]
    )


def test_insert_uses_calibrated_clearance_orientation_waypoint():
    reference = insert(
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.2, 0.1, 1.0, 0.0, 0.0, 0.0]),
        np.eye(3),
        horizon=10,
        clearance_offset_branch=(-0.02, 0.03, 0.04),
        clearance_rotation_offset_branch=(0.0, 0.0, 0.2),
        clearance_fraction=0.2,
        approach_fraction=0.6,
        seat_fraction=0.8,
    )

    assert reference[1, 3:7] == pytest.approx(
        [np.cos(0.1), 0.0, 0.0, np.sin(0.1)]
    )
    assert reference[-1, 3:7] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0]
    )


def test_insert_rejects_invalid_phase_order():
    with pytest.raises(
        ValueError, match="0 < clearance < approach < seat < 1"
    ):
        insert(
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            np.eye(3),
            horizon=4,
            approach_fraction=0.9,
            seat_fraction=0.8,
        )


def test_release_smoothly_opens_then_holds():
    controls = np.zeros((10, 14), dtype=np.float32)

    released = release(
        controls,
        open_value=-0.0475,
        start_fraction=0.2,
        end_fraction=0.6,
    )

    assert released[:2, 13] == pytest.approx(0.0)
    assert np.diff(released[:, 13]).max() <= 0.0
    assert released[5:, 13] == pytest.approx(-0.0475)
    assert released[:, :13] == pytest.approx(controls[:, :13])


def test_release_rejects_invalid_phase_order():
    with pytest.raises(ValueError, match="0 <= start < end <= 1"):
        release(
            np.zeros((4, 14), dtype=np.float32),
            start_fraction=0.8,
            end_fraction=0.2,
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


def test_left_visibility_pose_reaches_source_target_then_holds():
    controls = np.zeros((10, 14), dtype=np.float32)
    controls[0, :7] = np.arange(7, dtype=np.float32)
    target = np.arange(7, dtype=np.float32) + 10.0

    result = _apply_left_visibility_pose(
        controls,
        target,
        reach_fraction=0.4,
    )

    assert result[3:, :7] == pytest.approx(
        np.broadcast_to(target, (7, 7))
    )
    assert result[0, :7] != pytest.approx(target)
    assert result[:, 7:] == pytest.approx(controls[:, 7:])


def test_left_visibility_pose_pulls_back_during_seating():
    controls = np.zeros((10, 14), dtype=np.float32)
    observer = np.ones(7, dtype=np.float32)
    retreat = np.full(7, 2.0, dtype=np.float32)

    result = _apply_left_visibility_pose(
        controls,
        observer,
        reach_fraction=0.3,
        retreat_left_action=retreat,
        retreat_start_fraction=0.6,
        retreat_end_fraction=0.9,
    )

    assert result[2:6, :7] == pytest.approx(
        np.broadcast_to(observer, (4, 7))
    )
    assert result[8:, :7] == pytest.approx(
        np.broadcast_to(retreat, (2, 7))
    )
    assert result[:, 7:] == pytest.approx(controls[:, 7:])


@pytest.mark.parametrize("fraction", (0.0, -0.1, 1.1))
def test_left_visibility_pose_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="reach fraction"):
        _apply_left_visibility_pose(
            np.zeros((2, 14), dtype=np.float32),
            np.zeros(7, dtype=np.float32),
            reach_fraction=fraction,
        )


def test_semantic_grasp_base_preserves_left_gripper_only():
    nominal = np.arange(42, dtype=np.float32).reshape(3, 14)

    controls = _semantic_base_controls(nominal, "left_grasp")

    assert controls[:, :6] == pytest.approx(nominal[:, :6])
    assert controls[:, 6] == pytest.approx(nominal[:, 6])
    assert controls[:, 7:] == pytest.approx(
        np.broadcast_to(nominal[0, 7:], (3, 7))
    )


@pytest.mark.parametrize("target_name", ("right_grasp", "handover_latched"))
def test_semantic_handover_preserves_demo_gripper_timing(target_name):
    nominal = np.arange(42, dtype=np.float32).reshape(3, 14)

    controls = _semantic_base_controls(nominal, target_name)

    if target_name == "handover_latched":
        assert controls[:, :6] == pytest.approx(nominal[:, :6])
    else:
        assert controls[:, :6] == pytest.approx(
            np.broadcast_to(nominal[0, :6], (3, 6))
        )
    if target_name == "right_grasp":
        assert controls[:, 6] == pytest.approx(
            np.broadcast_to(nominal[0, 6], 3)
        )
    else:
        assert controls[:, 6] == pytest.approx(nominal[:, 6])
    assert controls[:, 7:13] == pytest.approx(nominal[:, 7:13])
    assert controls[:, 13] == pytest.approx(nominal[:, 13])


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


def test_motion_quality_cost_prefers_smooth_right_joint_trajectory(
    rollout_inputs,
):
    states, sensors, controls, reference, nominal = rollout_inputs
    states = np.repeat(states[:, :1], 6, axis=1)
    sensors = np.repeat(sensors[:, :1], 6, axis=1)
    controls = np.repeat(controls[:, :1], 6, axis=1)
    reference = np.repeat(reference[:1], 6, axis=0)
    nominal = np.repeat(nominal[:1], 6, axis=0)
    smooth = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    jerky = np.asarray([0.0, 0.8, 0.2, 1.0, 0.4, 1.0], dtype=np.float32)
    states[0, :, 21:27] = smooth[:, None]
    states[1, :, 21:27] = jerky[:, None]

    result = _objective_components(
        states,
        sensors,
        controls,
        reference=reference,
        nominal=nominal,
        keyframe_offset=5,
        target_name="left_grasp",
        right_joint_path_weight=0.2,
        right_joint_accel_weight=1.0,
        right_joint_jerk_weight=0.5,
    )

    assert result["right_joint_path_l2"][0] < result["right_joint_path_l2"][1]
    assert result["right_joint_accel_l2"][0] < result[
        "right_joint_accel_l2"
    ][1]
    assert result["right_joint_jerk_l2"][0] < result["right_joint_jerk_l2"][1]
    assert result["rewards"][0] > result["rewards"][1]


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
