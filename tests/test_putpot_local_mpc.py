import copy

import numpy as np
import pytest

from judo_isaaclab.putpot_local_mpc import (
    HandleLocalMpcConfig,
    contact_window_joint_nominal_weight,
    handle_local_mpc_active,
    handle_local_mpc_frame_receipt_complete,
    handle_local_mpc_step,
    source_prior_weight,
)


def _pose(x=0.0, y=0.0, z=0.0):
    return np.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _inputs(**overrides):
    jaw_axis = np.asarray([0.090120, -0.048000, 0.0], dtype=np.float64)
    jaw_axis /= np.linalg.norm(jaw_axis)
    values = {
        "contact_window_step": 0,
        "observed_pot_pose": _pose(),
        "observed_handle_contact_frame": _pose(x=0.030),
        "active_wrist_pose": _pose(),
        "object_relative_wrist_prior": _pose(x=0.020),
        "object_relative_jaw_axis_prior": jaw_axis,
        "object_relative_pad_depth_axis_prior": [0.0, 0.0, 1.0],
        "source_warm_start_wrist_pose": _pose(x=0.025),
        "active_pad_centers_world": np.stack((-0.045 * jaw_axis, 0.045 * jaw_axis)),
        "active_pad_axes_world": [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        "active_pad_fractions": [np.nan, np.nan],
        "active_finger_forces_n": [0.0, 0.0],
        "peer_pad_fractions": [0.5, 0.5],
        "peer_finger_forces_n": [2.0, 2.0],
        "active_grasp": False,
        "peer_grasp": True,
        "pre_peer_pot_displacement_m": 0.002,
        "current_jaw_command": -0.0475,
        "robust_streak": 0,
    }
    values.update(overrides)
    return values


def test_horizon_is_short_and_every_planned_control_is_bounded():
    with pytest.raises(ValueError, match="horizon"):
        HandleLocalMpcConfig(horizon_steps=9)
    with pytest.raises(ValueError, match="horizon"):
        HandleLocalMpcConfig(horizon_steps=21)

    config = HandleLocalMpcConfig(horizon_steps=15)
    command = handle_local_mpc_step(**_inputs(), config=config)
    plan = command.frame_receipt["planned_controls"]
    assert np.asarray(plan["translation_world_m"]).shape == (15, 3)
    assert np.asarray(plan["rotation_axis_angle_world_rad"]).shape == (15, 3)
    assert np.asarray(plan["jaw_increment"]).shape == (15,)
    assert np.max(np.linalg.norm(plan["translation_world_m"], axis=1)) <= 0.004
    assert np.max(np.linalg.norm(plan["rotation_axis_angle_world_rad"], axis=1)) <= 0.08
    assert np.max(np.abs(plan["jaw_increment"])) <= 0.004


def test_source_prior_decays_to_zero_and_joint_nominal_is_zero_only_in_window():
    config = HandleLocalMpcConfig(source_prior_initial_weight=0.25, source_prior_decay_steps=5)
    assert source_prior_weight(0, config) == pytest.approx(0.25)
    assert source_prior_weight(3, config) == pytest.approx(0.10)
    assert source_prior_weight(5, config) == 0.0
    assert source_prior_weight(20, config) == 0.0
    assert contact_window_joint_nominal_weight(active=True, non_contact_weight=0.7) == 0.0
    assert contact_window_joint_nominal_weight(active=False, non_contact_weight=0.7) == 0.7


def test_hard_motion_and_pad_constraints_fail_closed_without_more_control():
    motion = handle_local_mpc_step(
        **_inputs(pre_peer_pot_displacement_m=0.00301)
    )
    assert motion.fail_closed
    assert motion.fail_reason == "pre_peer_pot_motion_exceeded"
    np.testing.assert_allclose(motion.wrist_target_pose, _pose())
    assert motion.jaw_command == pytest.approx(-0.0475)

    edge = handle_local_mpc_step(
        **_inputs(active_finger_forces_n=[2.0, 0.0], active_pad_fractions=[0.05, np.nan])
    )
    assert edge.fail_closed
    assert edge.fail_reason == "active_contact_outside_pad_margin"
    assert not edge.frame_receipt["hard_constraints"]["active_contact_pad_margin_valid"]


def test_strict_four_pad_latch_requires_fifteen_consecutive_margin_frames():
    streak = 0
    command = None
    for index in range(15):
        command = handle_local_mpc_step(
            **_inputs(
                contact_window_step=index,
                active_finger_forces_n=[1.0, 1.0],
                active_pad_fractions=[0.10, 0.90],
                active_grasp=True,
                robust_streak=streak,
            )
        )
        streak = command.robust_streak
        assert command.robust_latch_ready is (index == 14)
    assert command is not None
    assert command.frame_receipt["latch"]["strict_four_pad_frame"]
    assert command.frame_receipt["latch"]["consecutive_frames"] == 15


def test_local_mpc_output_is_deterministic_and_frame_receipt_is_complete():
    first = handle_local_mpc_step(**_inputs(contact_window_step=2))
    second = handle_local_mpc_step(**_inputs(contact_window_step=2))
    np.testing.assert_array_equal(first.wrist_target_pose, second.wrist_target_pose)
    assert first.jaw_command == second.jaw_command
    assert first.frame_receipt == second.frame_receipt
    assert handle_local_mpc_frame_receipt_complete(first.frame_receipt)
    incomplete = copy.deepcopy(first.frame_receipt)
    del incomplete["observed_frames"]["active_pad_axes"]
    assert not handle_local_mpc_frame_receipt_complete(incomplete)


def test_non_contact_activation_and_nominal_behavior_are_unchanged():
    assert not handle_local_mpc_active(
        enabled=False,
        peer_latched=True,
        step=151,
        peer_latch_step=149,
        grasp_complete_step=246,
    )
    assert not handle_local_mpc_active(
        enabled=True,
        peer_latched=False,
        step=151,
        peer_latch_step=149,
        grasp_complete_step=246,
    )
    assert not handle_local_mpc_active(
        enabled=True,
        peer_latched=True,
        step=149,
        peer_latch_step=149,
        grasp_complete_step=246,
    )
    assert handle_local_mpc_active(
        enabled=True,
        peer_latched=True,
        step=150,
        peer_latch_step=149,
        grasp_complete_step=246,
    )
    assert contact_window_joint_nominal_weight(active=False) == 1.0
