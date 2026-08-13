import copy

import numpy as np
import pytest

from judo_isaaclab.putpot_local_mpc import (
    HandleLocalMpcConfig,
    contact_window_joint_nominal_weight,
    handle_local_bootstrap_active,
    handle_local_mpc_active,
    handle_local_mpc_frame_receipt_complete,
    handle_local_mpc_step,
    realized_contact_recenter_displacement_m,
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


def test_contact_fraction_recenter_is_bounded_and_defers_margin_fail_close():
    command = handle_local_mpc_step(
        **_inputs(
            contact_window_step=20,
            active_finger_forces_n=[0.0, 2.0],
            active_pad_fractions=[np.nan, -0.046],
            active_pad_axes_world=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        contact_fraction_recenter=True,
        active_pad_fraction_axis_extent_m=0.068,
    )
    recenter = command.frame_receipt["contact_fraction_recenter"]
    assert not command.fail_closed
    assert recenter["active"]
    assert recenter["contact_fraction_delta"] == pytest.approx(0.146)
    assert recenter["requested_translation_m"] == pytest.approx(0.009928)
    assert recenter["executed_translation_m"] == pytest.approx(0.001)
    assert command.contact_recenter_total_m == pytest.approx(0.0)
    assert recenter["total_translation_m"] == pytest.approx(0.0)
    assert (
        recenter["budget_accounting"]
        == "measured_positive_axial_wrist_displacement"
    )
    assert np.linalg.norm(
        command.frame_receipt["executed_control"]["translation_world_m"]
    ) == pytest.approx(0.001)
    contact_pad_axis = np.asarray(
        recenter["finger_tip_to_base_axis_world"]
    )
    translation = np.asarray(
        command.frame_receipt["executed_control"]["translation_world_m"]
    )
    np.testing.assert_allclose(contact_pad_axis, [0.0, 0.0, 1.0])
    assert np.dot(translation, contact_pad_axis) == pytest.approx(-0.001)
    assert not recenter["preserve_transverse_centering"]
    assert not recenter["preserve_bounded_closure"]
    assert not recenter["bounded_closure_priority_active"]
    np.testing.assert_allclose(
        recenter["retained_transverse_translation_world_m"], 0.0
    )

    centered = handle_local_mpc_step(
        **_inputs(
            contact_window_step=20,
            observed_handle_contact_frame=_pose(x=0.003, y=0.002),
            active_finger_forces_n=[0.0, 2.0],
            active_pad_fractions=[np.nan, -0.046],
        ),
        contact_fraction_recenter=True,
        contact_recenter_preserve_transverse_centering=True,
        active_pad_fraction_axis_extent_m=0.068,
    )
    centered_recenter = centered.frame_receipt["contact_fraction_recenter"]
    centered_translation = np.asarray(
        centered.frame_receipt["executed_control"]["translation_world_m"]
    )
    assert centered_recenter["preserve_transverse_centering"]
    np.testing.assert_allclose(
        centered_recenter["retained_transverse_translation_world_m"],
        [0.003, 0.002, 0.0],
    )
    np.testing.assert_allclose(centered_translation, [0.003, 0.002, -0.001])
    assert np.linalg.norm(centered_translation) < 0.004
    assert not centered_recenter["preserve_bounded_closure"]
    assert not centered_recenter["bounded_closure_priority_active"]
    assert centered.frame_receipt["executed_control"]["jaw_increment"] == 0.0

    centered_closing = handle_local_mpc_step(
        **_inputs(
            contact_window_step=20,
            observed_handle_contact_frame=_pose(x=0.003, y=0.002),
            active_finger_forces_n=[0.0, 2.0],
            active_pad_fractions=[np.nan, -0.046],
        ),
        contact_fraction_recenter=True,
        contact_recenter_preserve_transverse_centering=True,
        contact_recenter_preserve_bounded_closure=True,
        active_pad_fraction_axis_extent_m=0.068,
    )
    closing_recenter = centered_closing.frame_receipt[
        "contact_fraction_recenter"
    ]
    assert closing_recenter["preserve_bounded_closure"]
    assert closing_recenter["bounded_closure_priority_active"]
    assert closing_recenter["executed_translation_m"] == 0.0
    np.testing.assert_allclose(
        centered_closing.frame_receipt["executed_control"][
            "translation_world_m"
        ],
        [0.003, 0.002, 0.0],
    )
    assert centered_closing.frame_receipt["executed_control"][
        "jaw_increment"
    ] == pytest.approx(0.004)
    assert centered_closing.jaw_command == pytest.approx(-0.0435)

    realized = realized_contact_recenter_displacement_m(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -0.0003],
        translation,
    )
    assert realized == pytest.approx(0.0003)
    assert realized_contact_recenter_displacement_m(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -0.002],
        translation,
    ) == pytest.approx(0.001)
    assert realized_contact_recenter_displacement_m(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0003],
        translation,
    ) == pytest.approx(0.0)

    observed = handle_local_mpc_step(
        **_inputs(
            contact_window_step=21,
            active_finger_forces_n=[0.0, 2.0],
            active_pad_fractions=[np.nan, -0.046],
        ),
        contact_fraction_recenter=True,
        active_pad_fraction_axis_extent_m=0.068,
        contact_recenter_total_m=realized,
    )
    assert not observed.fail_closed
    assert observed.contact_recenter_total_m == pytest.approx(0.0003)

    exhausted = handle_local_mpc_step(
        **_inputs(
            contact_window_step=21,
            active_finger_forces_n=[0.0, 2.0],
            active_pad_fractions=[np.nan, -0.046],
        ),
        contact_fraction_recenter=True,
        active_pad_fraction_axis_extent_m=0.068,
        contact_recenter_total_m=0.012,
    )
    assert exhausted.fail_closed
    assert exhausted.fail_reason == "active_contact_outside_pad_margin"


def test_bounded_closure_commit_preserves_initial_gate_and_all_hard_guards():
    outside_gate = handle_local_mpc_step(
        **_inputs(
            contact_window_step=20,
            observed_handle_contact_frame=_pose(x=0.011),
        ),
        allow_bounded_closure_commit=True,
    )
    assert outside_gate.jaw_command == pytest.approx(-0.0475)
    assert not outside_gate.closure_committed
    assert not outside_gate.frame_receipt["closure"][
        "initial_alignment_satisfied"
    ]

    initial = handle_local_mpc_step(
        **_inputs(
            contact_window_step=20,
            observed_handle_contact_frame=_pose(x=0.0099),
        ),
        allow_bounded_closure_commit=True,
    )
    assert initial.jaw_command == pytest.approx(-0.0435)
    assert initial.closure_committed
    assert initial.frame_receipt["closure"]["committed"]

    continued = handle_local_mpc_step(
        **_inputs(
            contact_window_step=21,
            observed_handle_contact_frame=_pose(x=0.011),
            current_jaw_command=initial.jaw_command,
        ),
        allow_bounded_closure_commit=True,
        closure_committed=initial.closure_committed,
    )
    assert not continued.frame_receipt["closure"][
        "initial_alignment_satisfied"
    ]
    assert continued.frame_receipt["closure"]["was_committed"]
    assert continued.jaw_command == pytest.approx(-0.0395)

    motion_guard = handle_local_mpc_step(
        **_inputs(
            observed_handle_contact_frame=_pose(x=0.011),
            current_jaw_command=continued.jaw_command,
            pre_peer_pot_displacement_m=0.00301,
        ),
        allow_bounded_closure_commit=True,
        closure_committed=continued.closure_committed,
    )
    assert motion_guard.fail_closed
    assert motion_guard.fail_reason == "pre_peer_pot_motion_exceeded"
    assert motion_guard.jaw_command == pytest.approx(continued.jaw_command)
    assert motion_guard.frame_receipt["executed_control"]["jaw_increment"] == 0.0

    margin_guard = handle_local_mpc_step(
        **_inputs(
            observed_handle_contact_frame=_pose(x=0.011),
            current_jaw_command=continued.jaw_command,
            active_finger_forces_n=[2.0, 0.0],
            active_pad_fractions=[0.05, np.nan],
        ),
        allow_bounded_closure_commit=True,
        closure_committed=continued.closure_committed,
    )
    assert margin_guard.fail_closed
    assert margin_guard.fail_reason == "active_contact_outside_pad_margin"
    assert margin_guard.jaw_command == pytest.approx(continued.jaw_command)


def test_interior_single_pad_event_freezes_wrist_and_starts_bounded_closure():
    triggered = handle_local_mpc_step(
        **_inputs(
            contact_window_step=22,
            observed_handle_contact_frame=_pose(x=0.030),
            active_finger_forces_n=[0.0, 4.5],
            active_pad_fractions=[np.nan, 0.196],
        ),
        allow_bounded_closure_commit=True,
        allow_interior_single_pad_closure=True,
    )
    closure = triggered.frame_receipt["closure"]
    assert not closure["initial_alignment_satisfied"]
    assert closure["interior_single_pad_trigger_enabled"]
    assert closure["interior_single_pad_triggered"]
    assert closure["interior_single_pad_closure_hold_active"]
    assert closure["wrist_frozen_for_interior_single_pad_closure"]
    np.testing.assert_allclose(triggered.wrist_target_pose, _pose())
    assert triggered.jaw_command == pytest.approx(-0.0435)
    assert triggered.closure_committed

    continued = handle_local_mpc_step(
        **_inputs(
            contact_window_step=23,
            observed_handle_contact_frame=_pose(x=0.030),
            current_jaw_command=triggered.jaw_command,
        ),
        allow_bounded_closure_commit=True,
        closure_committed=triggered.closure_committed,
        allow_interior_single_pad_closure=True,
    )
    assert not continued.frame_receipt["closure"][
        "interior_single_pad_triggered"
    ]
    assert continued.frame_receipt["closure"][
        "interior_single_pad_closure_hold_active"
    ]
    np.testing.assert_allclose(continued.wrist_target_pose, _pose())
    assert continued.jaw_command == pytest.approx(-0.0395)

    edge = handle_local_mpc_step(
        **_inputs(
            contact_window_step=22,
            observed_handle_contact_frame=_pose(x=0.030),
            active_finger_forces_n=[0.0, 4.5],
            active_pad_fractions=[np.nan, 0.05],
        ),
        allow_bounded_closure_commit=True,
        allow_interior_single_pad_closure=True,
    )
    assert edge.fail_closed
    assert edge.fail_reason == "active_contact_outside_pad_margin"
    assert edge.jaw_command == pytest.approx(-0.0475)


def test_single_pad_transverse_intercept_keeps_open_jaw_and_depth_guard():
    intercepted = handle_local_mpc_step(
        **_inputs(
            contact_window_step=22,
            observed_handle_contact_frame=_pose(x=0.030, z=-0.020),
            active_finger_forces_n=[0.0, 4.5],
            active_pad_fractions=[np.nan, 0.196],
        ),
        depth_guarded_transverse_intercept=True,
        allow_bounded_closure_commit=True,
        allow_interior_single_pad_transverse_intercept=True,
    )
    guard = intercepted.frame_receipt["contact_frame_guard"]
    control = intercepted.frame_receipt["executed_control"]
    assert guard["physical_contact_observed"]
    assert guard["interior_single_pad_transverse_intercept_enabled"]
    assert guard["interior_single_pad_transverse_intercept_active"]
    assert guard["active"]
    assert guard["rotation_held_during_interior_single_pad_intercept"]
    np.testing.assert_allclose(control["translation_world_m"], [0.004, 0.0, 0.0])
    np.testing.assert_allclose(control["rotation_axis_angle_world_rad"], 0.0)
    assert control["jaw_increment"] == 0.0
    assert intercepted.jaw_command == pytest.approx(-0.0475)
    assert not intercepted.closure_committed

    edge = handle_local_mpc_step(
        **_inputs(
            contact_window_step=22,
            observed_handle_contact_frame=_pose(x=0.030, z=-0.020),
            active_finger_forces_n=[0.0, 4.5],
            active_pad_fractions=[np.nan, 0.05],
        ),
        depth_guarded_transverse_intercept=True,
        allow_bounded_closure_commit=True,
        allow_interior_single_pad_transverse_intercept=True,
    )
    assert edge.fail_closed
    assert edge.fail_reason == "active_contact_outside_pad_margin"
    np.testing.assert_allclose(edge.wrist_target_pose, _pose())


def test_handle_normal_depth_guard_removes_inward_handle_motion():
    half_sqrt_two = np.sqrt(0.5)
    handle = np.asarray(
        [0.030, 0.0, -0.020, half_sqrt_two, 0.0, half_sqrt_two, 0.0]
    )
    corrected = handle_local_mpc_step(
        **_inputs(contact_window_step=22, observed_handle_contact_frame=handle),
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
    )
    guard = corrected.frame_receipt["contact_frame_guard"]
    control = np.asarray(
        corrected.frame_receipt["executed_control"]["translation_world_m"]
    )
    assert guard["depth_axis_source"] == "observed_handle_contact_normal"
    np.testing.assert_allclose(
        guard["depth_axis_world"], [1.0, 0.0, 0.0], atol=1.0e-12
    )
    assert np.dot(control, np.asarray(guard["depth_axis_world"])) == pytest.approx(0.0)
    np.testing.assert_allclose(control, [0.0, 0.0, -0.004], atol=1.0e-12)

    legacy = handle_local_mpc_step(
        **_inputs(contact_window_step=22, observed_handle_contact_frame=handle),
        depth_guarded_transverse_intercept=True,
    )
    legacy_guard = legacy.frame_receipt["contact_frame_guard"]
    assert legacy_guard["depth_axis_source"] == "mean_pad_depth_axis"
    np.testing.assert_allclose(
        legacy.frame_receipt["executed_control"]["translation_world_m"],
        [0.004, 0.0, 0.0],
    )


def test_handle_tangent_contact_recenter_removes_normal_motion_and_honors_budget():
    half_sqrt_two = np.sqrt(0.5)
    handle = np.asarray(
        [0.030, 0.0, -0.020, half_sqrt_two, 0.0, half_sqrt_two, 0.0]
    )
    values = _inputs(
        contact_window_step=22,
        observed_handle_contact_frame=handle,
        active_pad_axes_world=[[0.6, 0.0, 0.8], [0.6, 0.0, 0.8]],
        active_finger_forces_n=[0.0, 4.5],
        active_pad_fractions=[np.nan, 0.05],
    )
    corrected = handle_local_mpc_step(
        **values,
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
        depth_guard_released=True,
        contact_fraction_recenter=True,
        contact_recenter_use_handle_tangent=True,
        contact_recenter_preserve_transverse_centering=True,
        active_pad_fraction_axis_extent_m=0.068,
    )
    recenter = corrected.frame_receipt["contact_fraction_recenter"]
    control = np.asarray(
        corrected.frame_receipt["executed_control"]["translation_world_m"]
    )
    assert recenter["active"]
    assert recenter["surface_tangent_enabled"]
    assert recenter["surface_tangent_axis_valid"]
    np.testing.assert_allclose(
        recenter["surface_tangent_axis_world"], [0.0, 0.0, 1.0], atol=1.0e-12
    )
    np.testing.assert_allclose(control, [0.0, 0.0, -0.001], atol=1.0e-12)
    assert recenter["executed_handle_normal_component_m"] == pytest.approx(0.0)
    assert recenter["executed_translation_m"] == pytest.approx(0.001)
    assert recenter["world_command_norm_m"] == pytest.approx(0.001)

    budget_limited = handle_local_mpc_step(
        **values,
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
        depth_guard_released=True,
        contact_fraction_recenter=True,
        contact_recenter_use_handle_tangent=True,
        contact_recenter_preserve_transverse_centering=True,
        active_pad_fraction_axis_extent_m=0.068,
        contact_recenter_total_m=0.0116,
    )
    limited = budget_limited.frame_receipt["contact_fraction_recenter"]
    assert limited["world_command_budget_m"] == pytest.approx(0.0004)
    assert limited["world_command_norm_m"] == pytest.approx(0.0004)
    assert limited["executed_translation_m"] == pytest.approx(0.0004)


def test_handle_tangent_contact_recenter_fails_closed_for_degenerate_axis():
    half_sqrt_two = np.sqrt(0.5)
    handle = np.asarray(
        [0.030, 0.0, -0.020, half_sqrt_two, 0.0, half_sqrt_two, 0.0]
    )
    command = handle_local_mpc_step(
        **_inputs(
            contact_window_step=22,
            observed_handle_contact_frame=handle,
            active_pad_axes_world=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            active_finger_forces_n=[0.0, 4.5],
            active_pad_fractions=[np.nan, 0.05],
        ),
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
        depth_guard_released=True,
        contact_fraction_recenter=True,
        contact_recenter_use_handle_tangent=True,
        contact_recenter_preserve_transverse_centering=True,
        active_pad_fraction_axis_extent_m=0.068,
    )
    assert command.fail_closed
    assert command.fail_reason == "active_contact_outside_pad_margin"
    assert not command.frame_receipt["contact_fraction_recenter"][
        "surface_tangent_axis_valid"
    ]
    np.testing.assert_allclose(command.wrist_target_pose, _pose())


def test_pair_15_attempt_32_handle_normal_counterfactual_is_tangential():
    handle = np.asarray(
        [
            0.61187602806807,
            0.23259209326643707,
            0.8761085390776744,
            -0.6751483904549511,
            0.6651924157195416,
            0.25275053902989103,
            -0.19445016316627017,
        ]
    )
    values = _inputs(
        contact_window_step=22,
        observed_handle_contact_frame=handle,
        active_wrist_pose=np.asarray(
            [
                0.5336552262306213,
                0.3293636441230774,
                0.9304019808769226,
                0.3299423321962161,
                0.4734162923273498,
                0.8141761826339007,
                -0.06428230872982285,
            ]
        ),
        active_pad_centers_world=np.asarray(
            [
                [0.5682171583175659, 0.3191834092140198, 0.8169037699699402],
                [0.6023063659667969, 0.24947421252727509, 0.8754481077194214],
            ]
        ),
        active_pad_axes_world=np.asarray(
            [
                [-0.4906572103500366, 0.4487762451171875, 0.7468975186347961],
                [-0.46121275424957275, 0.38456183671951294, 0.799622118473053],
            ]
        ),
        active_pad_fractions=[np.nan, 0.19592289626598358],
        active_finger_forces_n=[0.0, 4.500335216522217],
        peer_pad_fractions=[np.nan, np.nan],
        peer_finger_forces_n=[0.0, 0.0],
        pre_peer_pot_displacement_m=0.0,
    )
    legacy = handle_local_mpc_step(
        **values,
        depth_guarded_transverse_intercept=True,
        allow_interior_single_pad_transverse_intercept=True,
    )
    corrected = handle_local_mpc_step(
        **values,
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
        allow_interior_single_pad_transverse_intercept=True,
    )
    normal = np.asarray(
        corrected.frame_receipt["contact_frame_guard"]["depth_axis_world"]
    )
    legacy_control = np.asarray(
        legacy.frame_receipt["executed_control"]["translation_world_m"]
    )
    corrected_control = np.asarray(
        corrected.frame_receipt["executed_control"]["translation_world_m"]
    )
    assert np.dot(legacy_control, normal) == pytest.approx(-0.0031620383037050317)
    assert np.dot(corrected_control, normal) == pytest.approx(0.0, abs=1.0e-12)
    assert np.linalg.norm(corrected_control) == pytest.approx(0.004)
    np.testing.assert_allclose(
        corrected_control,
        [-0.0010426901125651581, -0.0007217176380684674, 0.0037936685385072497],
    )


def test_committed_closure_pauses_only_while_both_pads_are_force_backed():
    paused = handle_local_mpc_step(
        **_inputs(
            contact_window_step=22,
            observed_handle_contact_frame=_pose(x=0.011),
            current_jaw_command=-0.0195,
            active_finger_forces_n=[2.0, 2.0],
            active_pad_fractions=[0.55, 0.05],
        ),
        contact_fraction_recenter=True,
        contact_recenter_preserve_transverse_centering=True,
        contact_recenter_preserve_bounded_closure=True,
        active_pad_fraction_axis_extent_m=0.068,
        allow_bounded_closure_commit=True,
        closure_committed=True,
        pause_committed_closure_on_dual_force_backing=True,
    )
    assert paused.closure_committed
    assert paused.jaw_command == pytest.approx(-0.0195)
    assert paused.frame_receipt["closure"]["dual_force_backed"]
    assert paused.frame_receipt["closure"][
        "paused_on_dual_force_backing"
    ]
    assert paused.frame_receipt["contact_fraction_recenter"]["active"]
    assert not paused.frame_receipt["contact_fraction_recenter"][
        "bounded_closure_priority_active"
    ]
    assert paused.frame_receipt["contact_fraction_recenter"][
        "executed_translation_m"
    ] == pytest.approx(0.001)

    resumed = handle_local_mpc_step(
        **_inputs(
            contact_window_step=23,
            observed_handle_contact_frame=_pose(x=0.011),
            current_jaw_command=paused.jaw_command,
            active_finger_forces_n=[0.0, 2.0],
            active_pad_fractions=[np.nan, 0.05],
        ),
        contact_fraction_recenter=True,
        contact_recenter_preserve_transverse_centering=True,
        contact_recenter_preserve_bounded_closure=True,
        active_pad_fraction_axis_extent_m=0.068,
        allow_bounded_closure_commit=True,
        closure_committed=paused.closure_committed,
        pause_committed_closure_on_dual_force_backing=True,
    )
    assert not resumed.frame_receipt["closure"]["dual_force_backed"]
    assert not resumed.frame_receipt["closure"][
        "paused_on_dual_force_backing"
    ]
    assert resumed.frame_receipt["executed_control"][
        "jaw_increment"
    ] == pytest.approx(0.004)


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


def test_peer_free_bootstrap_requires_robust_active_dual_pad_latch():
    streak = 0
    for index in range(15):
        command = handle_local_mpc_step(
            **_inputs(
                contact_window_step=index,
                active_finger_forces_n=[1.5, 1.5],
                active_pad_fractions=[0.25, 0.75],
                active_grasp=True,
                peer_finger_forces_n=[0.0, 0.0],
                peer_pad_fractions=[np.nan, np.nan],
                peer_grasp=False,
                robust_streak=streak,
            ),
            require_peer_latch=False,
        )
        streak = command.robust_streak
    assert command.robust_latch_ready
    assert command.frame_receipt["latch"]["robust_frame"]
    assert not command.frame_receipt["latch"]["strict_four_pad_frame"]
    assert not command.frame_receipt["latch"]["peer_required"]


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
    incomplete_latch = copy.deepcopy(first.frame_receipt)
    del incomplete_latch["latch"]["peer_required"]
    assert not handle_local_mpc_frame_receipt_complete(incomplete_latch)


def test_depth_guard_centers_transverse_contact_frame_before_inward_motion():
    guarded = handle_local_mpc_step(
        **_inputs(
            contact_window_step=20,
            observed_handle_contact_frame=_pose(x=0.100, z=-0.030),
        ),
        depth_guarded_transverse_intercept=True,
    )
    control = np.asarray(
        guarded.frame_receipt["executed_control"]["translation_world_m"]
    )
    guard = guarded.frame_receipt["contact_frame_guard"]
    assert guard["enabled"]
    assert guard["active"]
    assert guard["transverse_residual_norm_m"] == pytest.approx(0.100)
    assert guard["signed_depth_residual_m"] == pytest.approx(-0.030)
    assert np.dot(control, [0.0, 0.0, 1.0]) == pytest.approx(0.0, abs=1.0e-12)
    assert np.linalg.norm(control) == pytest.approx(0.004)

    centered = handle_local_mpc_step(
        **_inputs(
            contact_window_step=20,
            observed_handle_contact_frame=_pose(x=0.005, z=-0.030),
        ),
        depth_guarded_transverse_intercept=True,
        depth_guard_alignment_streak=2,
    )
    assert not centered.frame_receipt["contact_frame_guard"]["active"]
    assert centered.frame_receipt["contact_frame_guard"]["released"]
    assert centered.frame_receipt["executed_control"]["translation_world_m"][2] < 0.0

    departed_after_release = handle_local_mpc_step(
        **_inputs(
            contact_window_step=21,
            observed_handle_contact_frame=_pose(x=0.020, z=-0.030),
        ),
        depth_guarded_transverse_intercept=True,
        depth_guard_released=centered.depth_guard_released,
    )
    departed_guard = departed_after_release.frame_receipt["contact_frame_guard"]
    assert not departed_guard["active"]
    assert departed_guard["released"]
    assert departed_after_release.frame_receipt["executed_control"][
        "translation_world_m"
    ][2] < 0.0


def test_depth_guard_is_opt_in_and_preserves_default_controller_output():
    values = _inputs(
        contact_window_step=20,
        observed_handle_contact_frame=_pose(x=0.100, z=-0.030),
    )
    default = handle_local_mpc_step(**values)
    explicit_off = handle_local_mpc_step(
        **values, depth_guarded_transverse_intercept=False
    )
    np.testing.assert_array_equal(
        default.wrist_target_pose, explicit_off.wrist_target_pose
    )
    assert default.jaw_command == explicit_off.jaw_command
    assert default.frame_receipt == explicit_off.frame_receipt


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


def test_peer_free_bootstrap_activation_is_bounded_to_acquisition_window():
    assert not handle_local_bootstrap_active(
        enabled=False,
        latch_ready=False,
        step=110,
        bootstrap_start_step=110,
        grasp_complete_step=356,
    )
    assert not handle_local_bootstrap_active(
        enabled=True,
        latch_ready=False,
        step=109,
        bootstrap_start_step=110,
        grasp_complete_step=356,
    )
    assert handle_local_bootstrap_active(
        enabled=True,
        latch_ready=False,
        step=110,
        bootstrap_start_step=110,
        grasp_complete_step=356,
    )
    assert not handle_local_bootstrap_active(
        enabled=True,
        latch_ready=True,
        step=170,
        bootstrap_start_step=110,
        grasp_complete_step=356,
    )
    assert not handle_local_bootstrap_active(
        enabled=True,
        latch_ready=False,
        step=357,
        bootstrap_start_step=110,
        grasp_complete_step=356,
    )
