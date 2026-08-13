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
    np.testing.assert_allclose(
        centered_recenter["budgeted_axial_translation_world_m"],
        [0.0, 0.0, -0.001],
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
    np.testing.assert_allclose(
        closing_recenter["budgeted_axial_translation_world_m"], 0.0
    )
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

    # Retained orthogonal centering is governed by the unchanged Cartesian
    # step bound, not the 12 mm axial contact-recenter allowance.
    assert realized_contact_recenter_displacement_m(
        [0.0, 0.0, 0.0],
        [0.003, 0.002, -0.0003],
        centered_recenter["budgeted_axial_translation_world_m"],
    ) == pytest.approx(0.0003)

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


def test_transverse_aligned_two_pad_closure_waits_for_guard_release():
    half_sqrt_two = np.sqrt(0.5)
    handle = np.asarray(
        [0.030, 0.0, -0.005, half_sqrt_two, 0.0, half_sqrt_two, 0.0]
    )
    values = _inputs(
        contact_window_step=22,
        observed_handle_contact_frame=handle,
        active_finger_forces_n=[0.0, 4.5],
        active_pad_fractions=[np.nan, 0.474],
    )
    waiting = handle_local_mpc_step(
        **values,
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
        depth_guard_alignment_streak=1,
        allow_bounded_closure_commit=True,
        pause_committed_closure_on_dual_force_backing=True,
        allow_interior_single_pad_transverse_intercept=True,
        allow_transverse_aligned_two_pad_closure=True,
    )
    waiting_closure = waiting.frame_receipt["closure"]
    assert not waiting.depth_guard_released
    assert not waiting_closure["transverse_aligned_two_pad_triggered"]
    assert waiting.frame_receipt["executed_control"]["jaw_increment"] == 0.0
    assert not waiting.closure_committed

    triggered = handle_local_mpc_step(
        **values,
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
        depth_guard_alignment_streak=2,
        allow_bounded_closure_commit=True,
        pause_committed_closure_on_dual_force_backing=True,
        allow_interior_single_pad_transverse_intercept=True,
        allow_transverse_aligned_two_pad_closure=True,
    )
    closure = triggered.frame_receipt["closure"]
    assert triggered.depth_guard_released
    assert closure["transverse_aligned_two_pad_trigger_enabled"]
    assert closure["transverse_aligned_two_pad_triggered"]
    assert closure["transverse_aligned_two_pad_closure_hold_active"]
    assert closure["wrist_frozen_for_transverse_aligned_two_pad_closure"]
    assert not closure["interior_single_pad_triggered"]
    np.testing.assert_allclose(triggered.wrist_target_pose, _pose())
    assert triggered.jaw_command == pytest.approx(-0.0435)
    assert triggered.closure_committed

    motion_guard = handle_local_mpc_step(
        **{**values, "pre_peer_pot_displacement_m": 0.00301},
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
        depth_guard_alignment_streak=2,
        allow_bounded_closure_commit=True,
        pause_committed_closure_on_dual_force_backing=True,
        allow_interior_single_pad_transverse_intercept=True,
        allow_transverse_aligned_two_pad_closure=True,
    )
    assert motion_guard.fail_closed
    assert motion_guard.fail_reason == "pre_peer_pot_motion_exceeded"
    assert motion_guard.frame_receipt["executed_control"]["jaw_increment"] == 0.0


def test_transverse_aligned_closure_can_pivot_about_loaded_pad_one():
    half_sqrt_two = np.sqrt(0.5)
    handle = np.asarray(
        [0.030, 0.0, -0.005, half_sqrt_two, 0.0, half_sqrt_two, 0.0]
    )
    values = _inputs(
        contact_window_step=22,
        observed_handle_contact_frame=handle,
        active_finger_forces_n=[0.0, 4.5],
        active_pad_fractions=[np.nan, 0.474],
    )
    command = handle_local_mpc_step(
        **values,
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
        depth_guard_alignment_streak=2,
        allow_bounded_closure_commit=True,
        pause_committed_closure_on_dual_force_backing=True,
        allow_interior_single_pad_transverse_intercept=True,
        allow_transverse_aligned_two_pad_closure=True,
        transverse_aligned_closure_pivot_pad_index=1,
    )
    closure = command.frame_receipt["closure"]
    jaw_axis = np.asarray(
        command.frame_receipt["observed_frames"]["jaw_axis"]
    )
    expected_translation = 0.004 * jaw_axis
    assert command.closure_committed
    assert closure["loaded_pad_pivot_closure_enabled"]
    assert closure["loaded_pad_pivot_closure_active"]
    assert closure["loaded_pad_pivot_index"] == 1
    assert not closure["wrist_frozen_for_transverse_aligned_two_pad_closure"]
    np.testing.assert_allclose(
        closure["loaded_pad_pivot_translation_world_m"], expected_translation
    )
    assert closure["loaded_pad_pivot_translation_norm_m"] == pytest.approx(0.004)
    assert closure["loaded_pad_pivot_jaw_scale"] == pytest.approx(0.4)
    np.testing.assert_allclose(
        command.wrist_target_pose[:3], _pose()[:3] + expected_translation
    )
    assert command.jaw_command == pytest.approx(-0.0459)

    with pytest.raises(ValueError, match="requires transverse-aligned"):
        handle_local_mpc_step(
            **values,
            transverse_aligned_closure_pivot_pad_index=1,
        )


def test_dual_force_pad_margin_pivot_compensates_measured_tracking_ratio():
    wrist = np.asarray(
        [
            0.5640213,
            0.2785081,
            0.94189227,
            0.41531724,
            0.463308,
            0.7726191,
            -0.12616195,
        ],
        dtype=np.float64,
    )
    centers = np.asarray(
        [
            [0.6142322, 0.23112977, 0.85405874],
            [0.62944, 0.19547606, 0.9000379],
        ],
        dtype=np.float64,
    )
    axes = np.asarray(
        [
            [-0.5342989, 0.60616183, 0.58914596],
            [-0.5144022, 0.5522913, 0.65602213],
        ],
        dtype=np.float64,
    )
    fractions = np.asarray([0.56464046, 0.0268029], dtype=np.float64)
    command = handle_local_mpc_step(
        **_inputs(
            contact_window_step=27,
            active_wrist_pose=wrist,
            object_relative_wrist_prior=wrist,
            source_warm_start_wrist_pose=wrist,
            active_pad_centers_world=centers,
            active_pad_axes_world=axes,
            active_pad_fractions=fractions,
            active_finger_forces_n=[11.55667, 22.79371],
            active_grasp=True,
            current_jaw_command=-0.0195,
        ),
        require_peer_latch=False,
        contact_fraction_recenter=True,
        contact_recenter_preserve_bounded_closure=True,
        allow_bounded_closure_commit=True,
        closure_committed=True,
        pause_committed_closure_on_dual_force_backing=True,
        allow_dual_force_pad_margin_pivot=True,
        active_pad_fraction_axis_extent_m=0.06806614249944687,
    )
    closure = command.frame_receipt["closure"]
    assert not command.fail_closed
    assert closure["dual_force_pad_margin_pivot_enabled"]
    assert closure["dual_force_pad_margin_pivot_active"]
    assert closure["dual_force_pad_margin_pivot_geometry_feasible"]
    assert closure["dual_force_pad_margin_pivot_strong_index"] == 0
    assert closure["dual_force_pad_margin_pivot_weak_index"] == 1
    assert closure["dual_force_pad_margin_pivot_target_fraction"] == 0.25
    assert closure["dual_force_pad_margin_pivot_predicted_fraction"] > fractions[1]
    assert closure[
        "dual_force_pad_margin_pivot_translation_tracking_scale"
    ] == pytest.approx(0.4)
    assert np.linalg.norm(
        closure[
            "dual_force_pad_margin_pivot_uncompensated_translation_world_m"
        ]
    ) == pytest.approx(0.004)
    assert np.linalg.norm(
        closure["dual_force_pad_margin_pivot_translation_world_m"]
    ) == pytest.approx(0.0016)
    assert np.linalg.norm(
        closure["dual_force_pad_margin_pivot_rotation_axis_angle_world_rad"]
    ) < 0.08
    assert command.jaw_command == pytest.approx(-0.0195)
    assert handle_local_mpc_frame_receipt_complete(command.frame_receipt)


def test_pair_15_preseats_finite_edge_with_open_jaw_before_closure():
    values = _inputs(
        contact_window_step=20,
        observed_handle_contact_frame=_pose(),
        object_relative_wrist_prior=_pose(),
        source_warm_start_wrist_pose=_pose(),
        active_pad_fractions=[np.nan, -0.02293512225151062],
        active_finger_forces_n=[0.0, 0.0],
    )
    command = handle_local_mpc_step(
        **values,
        depth_guarded_transverse_intercept=True,
        contact_fraction_recenter=True,
        contact_recenter_use_handle_tangent=True,
        contact_recenter_preserve_transverse_centering=True,
        contact_recenter_preserve_bounded_closure=True,
        allow_bounded_closure_commit=True,
        allow_dual_force_pad_margin_pivot=True,
        depth_guard_released=True,
        active_pad_fraction_axis_extent_m=0.06806614249944687,
    )
    recenter = command.frame_receipt["contact_fraction_recenter"]
    control = command.frame_receipt["executed_control"]
    assert not command.fail_closed
    assert recenter["preclosure_geometric_prestage_enabled"]
    assert recenter["preclosure_geometric_prestage_active"]
    assert recenter["preclosure_geometric_uses_raw_pad_axis"]
    assert recenter["active"]
    assert recenter["protected_pad_fraction_margin"] == pytest.approx(
        0.1 + 0.004 / 0.06806614249944687
    )
    assert recenter["base_maximum_total_m"] == pytest.approx(0.012)
    assert recenter["preclosure_geometric_extra_budget_m"] == pytest.approx(0.002)
    assert recenter["preclosure_geometric_maximum_total_m"] == pytest.approx(
        0.014
    )
    assert recenter["maximum_total_m"] == pytest.approx(0.012)
    assert recenter["effective_maximum_total_m"] == pytest.approx(0.014)
    assert recenter["executed_translation_m"] == pytest.approx(0.001)
    np.testing.assert_allclose(control["translation_world_m"], [0.0, 0.0, -0.001])
    np.testing.assert_allclose(control["rotation_axis_angle_world_rad"], 0.0)
    assert control["jaw_increment"] == 0.0
    assert command.jaw_command == pytest.approx(-0.0475)
    assert not command.closure_committed
    assert not recenter["bounded_closure_priority_active"]
    assert handle_local_mpc_frame_receipt_complete(command.frame_receipt)

    unopted = handle_local_mpc_step(
        **values,
        depth_guarded_transverse_intercept=True,
        contact_fraction_recenter=True,
        contact_recenter_use_handle_tangent=True,
        contact_recenter_preserve_transverse_centering=True,
        contact_recenter_preserve_bounded_closure=True,
        allow_bounded_closure_commit=True,
        active_pad_fraction_axis_extent_m=0.06806614249944687,
    )
    assert not unopted.frame_receipt["contact_fraction_recenter"][
        "preclosure_geometric_prestage_enabled"
    ]
    assert unopted.frame_receipt["executed_control"]["jaw_increment"] == pytest.approx(
        0.004
    )


def test_pair_15_preclosure_prestage_fails_closed_at_float32_budget_floor():
    command = handle_local_mpc_step(
        **_inputs(
            contact_window_step=90,
            observed_handle_contact_frame=_pose(),
            object_relative_wrist_prior=_pose(),
            source_warm_start_wrist_pose=_pose(),
            active_pad_fractions=[np.nan, 0.14034],
            active_finger_forces_n=[0.0, 0.0],
        ),
        depth_guarded_transverse_intercept=True,
        depth_guard_released=True,
        contact_fraction_recenter=True,
        contact_recenter_use_handle_tangent=True,
        contact_recenter_preserve_transverse_centering=True,
        contact_recenter_preserve_bounded_closure=True,
        allow_bounded_closure_commit=True,
        allow_dual_force_pad_margin_pivot=True,
        active_pad_fraction_axis_extent_m=0.06806614249944687,
        contact_recenter_total_m=0.013999975,
    )
    recenter = command.frame_receipt["contact_fraction_recenter"]
    assert command.fail_closed
    assert command.fail_reason == "preclosure_geometric_prestage_budget_exhausted"
    assert not recenter["active"]
    assert recenter["maximum_total_m"] == pytest.approx(0.012)
    assert recenter["effective_maximum_total_m"] == pytest.approx(0.014)
    assert recenter["preclosure_geometric_budget_exhaustion_tolerance_m"] == (
        pytest.approx(1.0e-6)
    )
    np.testing.assert_allclose(
        command.frame_receipt["executed_control"]["translation_world_m"], 0.0
    )
    assert command.frame_receipt["executed_control"]["jaw_increment"] == 0.0
    assert handle_local_mpc_frame_receipt_complete(command.frame_receipt)


def test_handle_normal_depth_guard_removes_inward_handle_motion():
    half_sqrt_two = np.sqrt(0.5)
    handle = np.asarray(
        [0.030, 0.0, -0.020, half_sqrt_two, 0.0, half_sqrt_two, 0.0]
    )
    corrected = handle_local_mpc_step(
        **_inputs(
            contact_window_step=22,
            observed_handle_contact_frame=handle,
            active_finger_forces_n=[0.0, 4.5],
            active_pad_fractions=[np.nan, 0.196],
        ),
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
        allow_interior_single_pad_transverse_intercept=True,
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


def test_handle_normal_depth_guard_preserves_force_free_pad_axis_corridor():
    half_sqrt_two = np.sqrt(0.5)
    handle = np.asarray(
        [0.030, 0.0, -0.020, half_sqrt_two, 0.0, half_sqrt_two, 0.0]
    )
    force_free = handle_local_mpc_step(
        **_inputs(contact_window_step=22, observed_handle_contact_frame=handle),
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
    )
    guard = force_free.frame_receipt["contact_frame_guard"]
    assert not guard["physical_contact_observed"]
    assert guard["depth_axis_source"] == "mean_pad_depth_axis"
    np.testing.assert_allclose(guard["depth_axis_world"], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        force_free.frame_receipt["executed_control"]["translation_world_m"],
        [0.004, 0.0, 0.0],
    )


def test_handle_normal_depth_guard_stays_latched_across_contact_dropout():
    half_sqrt_two = np.sqrt(0.5)
    handle = np.asarray(
        [0.030, 0.0, -0.020, half_sqrt_two, 0.0, half_sqrt_two, 0.0]
    )
    command = handle_local_mpc_step(
        **_inputs(
            contact_window_step=36,
            observed_handle_contact_frame=handle,
            active_finger_forces_n=[0.0, 0.089],
            active_pad_fractions=[np.nan, 0.156],
        ),
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
        depth_guard_released=False,
        contact_fraction_recenter=True,
        contact_recenter_use_handle_tangent=True,
        contact_recenter_preserve_transverse_centering=True,
        active_pad_fraction_axis_extent_m=0.068,
        contact_recenter_total_m=0.003,
    )
    guard = command.frame_receipt["contact_frame_guard"]
    control = np.asarray(
        command.frame_receipt["executed_control"]["translation_world_m"]
    )
    assert not guard["physical_contact_observed"]
    assert guard["depth_axis_source"] == "latched_observed_handle_contact_normal"
    np.testing.assert_allclose(
        guard["depth_axis_world"], [1.0, 0.0, 0.0], atol=1.0e-12
    )
    assert np.dot(control, guard["depth_axis_world"]) == pytest.approx(
        0.0, abs=1.0e-12
    )
    recenter = command.frame_receipt["contact_fraction_recenter"]
    assert recenter["pre_release_margin_protection_active"]
    assert recenter["active"]
    assert recenter["world_command_norm_m"] < 0.004
    assert not command.fail_closed


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

    depth_completion = handle_local_mpc_step(
        **values,
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
        depth_guard_released=True,
        contact_fraction_recenter=True,
        contact_recenter_use_handle_tangent=True,
        contact_recenter_preserve_transverse_centering=True,
        active_pad_fraction_axis_extent_m=0.068,
        contact_recenter_total_m=0.012,
    )
    completion = depth_completion.frame_receipt["contact_fraction_recenter"]
    completion_control = np.asarray(
        depth_completion.frame_receipt["executed_control"]["translation_world_m"]
    )
    handle_normal = np.asarray(
        depth_completion.frame_receipt["contact_frame_guard"]["depth_axis_world"]
    )
    assert not depth_completion.fail_closed
    assert completion["guarded_depth_completion_enabled"]
    assert completion["guarded_depth_completion_active"]
    np.testing.assert_allclose(
        completion_control,
        completion["guarded_depth_completion_translation_world_m"],
    )
    np.testing.assert_allclose(
        completion_control - np.dot(completion_control, handle_normal) * handle_normal,
        0.0,
        atol=1.0e-12,
    )
    assert np.linalg.norm(completion_control) <= 0.004 + 1.0e-12
    np.testing.assert_allclose(
        depth_completion.frame_receipt["executed_control"][
            "rotation_axis_angle_world_rad"
        ],
        0.0,
    )
    assert depth_completion.frame_receipt["executed_control"]["jaw_increment"] == 0.0

    pot_motion_values = {**values, "pre_peer_pot_displacement_m": 0.0031}
    pot_motion_exceeded = handle_local_mpc_step(
        **pot_motion_values,
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
        depth_guard_released=True,
        contact_fraction_recenter=True,
        contact_recenter_use_handle_tangent=True,
        contact_recenter_preserve_transverse_centering=True,
        active_pad_fraction_axis_extent_m=0.068,
        contact_recenter_total_m=0.012,
    )
    assert pot_motion_exceeded.fail_closed
    assert pot_motion_exceeded.fail_reason == "pre_peer_pot_motion_exceeded"
    assert not pot_motion_exceeded.frame_receipt["contact_fraction_recenter"][
        "guarded_depth_completion_active"
    ]


def test_handle_tangent_recenter_can_protect_margin_before_depth_guard_release():
    half_sqrt_two = np.sqrt(0.5)
    handle = np.asarray(
        [0.030, 0.0, -0.020, half_sqrt_two, 0.0, half_sqrt_two, 0.0]
    )
    command = handle_local_mpc_step(
        **_inputs(
            contact_window_step=29,
            observed_handle_contact_frame=handle,
            active_pad_axes_world=[[0.6, 0.0, 0.8], [0.6, 0.0, 0.8]],
            active_finger_forces_n=[0.0, 2.38],
            active_pad_fractions=[np.nan, 0.094],
        ),
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
        depth_guard_released=False,
        contact_fraction_recenter=True,
        contact_recenter_use_handle_tangent=True,
        contact_recenter_preserve_transverse_centering=True,
        active_pad_fraction_axis_extent_m=0.068,
    )
    recenter = command.frame_receipt["contact_fraction_recenter"]
    normal = np.asarray(
        command.frame_receipt["contact_frame_guard"]["depth_axis_world"]
    )
    control = np.asarray(
        command.frame_receipt["executed_control"]["translation_world_m"]
    )
    assert not command.fail_closed
    assert not command.depth_guard_released
    assert recenter["active"]
    assert np.dot(control, normal) == pytest.approx(0.0, abs=1.0e-12)
    assert recenter["pre_release_margin_protection_enabled"]
    assert recenter["protected_pad_fraction_margin"] == pytest.approx(
        0.1 + 0.004 / 0.068
    )
    assert recenter["acceptance_pad_fraction_margin"] == pytest.approx(0.1)
    assert recenter["executed_translation_m"] == pytest.approx(0.001)
    assert command.frame_receipt["executed_control"]["jaw_increment"] == 0.0


def test_pre_release_margin_reserve_activates_before_acceptance_edge():
    half_sqrt_two = np.sqrt(0.5)
    handle = np.asarray(
        [0.030, 0.0, -0.020, half_sqrt_two, 0.0, half_sqrt_two, 0.0]
    )
    command = handle_local_mpc_step(
        **_inputs(
            contact_window_step=24,
            observed_handle_contact_frame=handle,
            active_pad_axes_world=[[0.6, 0.0, 0.8], [0.6, 0.0, 0.8]],
            active_finger_forces_n=[0.0, 2.2],
            active_pad_fractions=[np.nan, 0.15],
        ),
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
        depth_guard_released=False,
        contact_fraction_recenter=True,
        contact_recenter_use_handle_tangent=True,
        contact_recenter_preserve_transverse_centering=True,
        active_pad_fraction_axis_extent_m=0.068,
    )
    recenter = command.frame_receipt["contact_fraction_recenter"]
    assert command.frame_receipt["hard_constraints"][
        "active_contact_pad_margin_valid"
    ]
    assert recenter["pre_release_margin_protection_active"]
    assert recenter["active"]
    assert recenter["executed_translation_m"] == pytest.approx(0.0006)


def test_pair_15_attempt_34_exhausted_tangent_budget_completes_depth_open_jaw():
    command = handle_local_mpc_step(
        **_inputs(
            contact_window_step=85,
            observed_handle_contact_frame=[
                0.6225847730164783,
                0.22119748566733805,
                0.8636750852797689,
                -0.6751265205259744,
                0.6651734204239027,
                0.25282198785588494,
                -0.19449818636855173,
            ],
            active_wrist_pose=[
                0.5253683924674988,
                0.33023208379745483,
                0.941718339920044,
                0.33117954685149137,
                0.47353861946063636,
                0.8136624389727908,
                -0.06351943821701292,
            ],
            active_pad_centers_world=[
                [0.5602778196334839, 0.32006704807281494, 0.8283215165138245],
                [0.5941927433013916, 0.2503335773944855, 0.8869801163673401],
            ],
            active_pad_axes_world=[
                [-0.49295365810394287, 0.4487238824367523, 0.745415210723877],
                [-0.463673859834671, 0.38450437784194946, 0.798224925994873],
            ],
            active_pad_fractions=[np.nan, 0.00905468687415123],
            active_finger_forces_n=[0.0, 1.117085576057434],
            pre_peer_pot_displacement_m=4.2613424260753435e-05,
            current_jaw_command=-0.04749999940395355,
        ),
        depth_guarded_transverse_intercept=True,
        depth_guard_use_handle_contact_normal=True,
        depth_guard_released=True,
        contact_fraction_recenter=True,
        contact_recenter_use_handle_tangent=True,
        contact_recenter_preserve_transverse_centering=True,
        contact_recenter_preserve_bounded_closure=True,
        allow_bounded_closure_commit=True,
        pause_committed_closure_on_dual_force_backing=True,
        allow_interior_single_pad_transverse_intercept=True,
        active_pad_fraction_axis_extent_m=0.06806614249944687,
        contact_recenter_total_m=0.012,
    )
    receipt = command.frame_receipt["contact_fraction_recenter"]
    assert not command.fail_closed
    assert receipt["guarded_depth_completion_active"]
    np.testing.assert_allclose(
        command.frame_receipt["executed_control"]["translation_world_m"],
        [
            0.0023947056063658764,
            -0.0031915069011783036,
            5.087410576909503e-05,
        ],
    )
    assert command.frame_receipt["executed_control"]["jaw_increment"] == 0.0
    np.testing.assert_allclose(
        command.frame_receipt["executed_control"]["rotation_axis_angle_world_rad"],
        0.0,
    )


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


def test_pair_15_committed_single_pad_closure_respects_remaining_motion_budget():
    values = _inputs(
        contact_window_step=20,
        observed_handle_contact_frame=_pose(x=0.030),
        current_jaw_command=-0.0275,
        active_finger_forces_n=[0.0, 14.2],
        active_pad_fractions=[np.nan, 0.1275],
        pre_peer_pot_displacement_m=0.0026016814898068784,
    )
    legacy = handle_local_mpc_step(
        **values,
        allow_bounded_closure_commit=True,
        closure_committed=True,
    )
    legacy_control = legacy.frame_receipt["executed_control"]
    assert np.linalg.norm(legacy_control["translation_world_m"]) == pytest.approx(
        0.004
    )

    budgeted = handle_local_mpc_step(
        **values,
        allow_bounded_closure_commit=True,
        closure_committed=True,
        budget_committed_closure_by_pre_peer_motion=True,
    )
    closure = budgeted.frame_receipt["closure"]
    control = budgeted.frame_receipt["executed_control"]
    remaining_m = 0.003 - values["pre_peer_pot_displacement_m"]
    assert not budgeted.fail_closed
    assert closure["pre_peer_motion_budgeted_closure_enabled"]
    assert closure["pre_peer_motion_budgeted_closure_active"]
    assert closure["pre_peer_motion_remaining_m"] == pytest.approx(remaining_m)
    assert closure["pre_peer_motion_control_scale"] == pytest.approx(
        remaining_m / 0.004
    )
    assert np.linalg.norm(control["translation_world_m"]) == pytest.approx(
        remaining_m
    )
    assert closure["pre_peer_motion_unbudgeted_jaw_increment"] == pytest.approx(
        0.004
    )
    assert control["jaw_increment"] == pytest.approx(
        0.004 * closure["pre_peer_motion_control_scale"]
    )
    assert handle_local_mpc_frame_receipt_complete(budgeted.frame_receipt)

    unopted_closure = legacy.frame_receipt["closure"]
    assert not unopted_closure["pre_peer_motion_budgeted_closure_enabled"]
    assert not unopted_closure["pre_peer_motion_budgeted_closure_active"]
    assert unopted_closure["pre_peer_motion_control_scale"] == pytest.approx(1.0)


def test_pair_15_right_closure_requires_interior_geometric_preseat():
    aligned = _inputs(
        observed_handle_contact_frame=_pose(x=0.009),
        active_pad_fractions=[np.nan, np.nan],
        active_finger_forces_n=[0.0, 0.0],
        current_jaw_command=-0.0475,
    )
    legacy = handle_local_mpc_step(**aligned)
    assert legacy.frame_receipt["executed_control"]["jaw_increment"] > 0.0

    gated = handle_local_mpc_step(
        **aligned,
        contact_fraction_recenter=True,
        require_geometric_preseat_for_closure=True,
        active_pad_fraction_axis_extent_m=0.068,
    )
    closure = gated.frame_receipt["closure"]
    assert closure["geometric_preseat_required"]
    assert not closure["geometric_preseat_satisfied"]
    assert closure["geometric_preseat_finite_pad_count"] == 0
    assert closure["geometric_preseat_interior_pad_count"] == 0
    assert closure["geometric_preseat_prospective_pad_fractions"] == [None, None]
    assert not closure["geometric_preseat_prospective_broad_contact"]
    assert closure["geometric_preseat_source_wrist_target_active"]
    np.testing.assert_allclose(
        closure["geometric_preseat_source_wrist_residual_world_m"],
        [0.02, 0.0, 0.0],
    )
    assert gated.frame_receipt["executed_control"]["jaw_increment"] == 0.0
    assert gated.frame_receipt["executed_control"][
        "translation_world_m"
    ][0] == pytest.approx(0.02 / 15.0)

    prospective = handle_local_mpc_step(
        **{
            **aligned,
            "active_wrist_pose": _pose(x=0.018),
        },
        contact_fraction_recenter=True,
        require_geometric_preseat_for_closure=True,
        geometric_preseat_predicted_pad_fractions=[0.5286, 0.5287],
        active_pad_fraction_axis_extent_m=0.068,
    )
    prospective_closure = prospective.frame_receipt["closure"]
    assert prospective_closure["geometric_preseat_satisfied"]
    assert prospective_closure["geometric_preseat_source_wrist_aligned"]
    assert prospective_closure["geometric_preseat_prospective_broad_contact"]
    assert prospective_closure["geometric_preseat_prospective_closure_ready"]
    assert prospective_closure[
        "geometric_preseat_prospective_pad_fractions"
    ] == pytest.approx([0.5286, 0.5287])
    assert prospective.frame_receipt["executed_control"]["jaw_increment"] > 0.0
    assert handle_local_mpc_frame_receipt_complete(prospective.frame_receipt)

    one_predicted_pad = handle_local_mpc_step(
        **{
            **aligned,
            "active_wrist_pose": _pose(x=0.018),
        },
        contact_fraction_recenter=True,
        require_geometric_preseat_for_closure=True,
        geometric_preseat_predicted_pad_fractions=[0.5286, np.nan],
        active_pad_fraction_axis_extent_m=0.068,
    )
    assert not one_predicted_pad.frame_receipt["closure"][
        "geometric_preseat_prospective_closure_ready"
    ]
    assert one_predicted_pad.frame_receipt["executed_control"][
        "jaw_increment"
    ] == 0.0

    committed = handle_local_mpc_step(
        **{
            **aligned,
            "active_wrist_pose": _pose(x=0.018),
        },
        contact_fraction_recenter=True,
        contact_recenter_preserve_bounded_closure=True,
        allow_bounded_closure_commit=True,
        require_geometric_preseat_for_closure=True,
        geometric_preseat_predicted_pad_fractions=[0.5286, 0.5287],
        active_pad_fraction_axis_extent_m=0.068,
    )
    assert committed.closure_committed
    continued = handle_local_mpc_step(
        **{
            **aligned,
            "active_wrist_pose": _pose(x=0.019),
            "active_pad_fractions": [-0.02, np.nan],
            "active_finger_forces_n": [0.5, 0.0],
            "current_jaw_command": committed.jaw_command,
        },
        contact_fraction_recenter=True,
        contact_recenter_preserve_bounded_closure=True,
        allow_bounded_closure_commit=True,
        closure_committed=committed.closure_committed,
        require_geometric_preseat_for_closure=True,
        geometric_preseat_predicted_pad_fractions=[0.5286, 0.5287],
        active_pad_fraction_axis_extent_m=0.068,
    )
    continued_closure = continued.frame_receipt["closure"]
    assert continued_closure["was_committed"]
    assert continued_closure["geometric_preseat_satisfied"]
    assert continued_closure["geometric_preseat_source_wrist_target_active"]
    assert continued.frame_receipt["contact_fraction_recenter"][
        "bounded_closure_priority_active"
    ]
    assert continued.frame_receipt["executed_control"]["jaw_increment"] > 0.0
    assert handle_local_mpc_frame_receipt_complete(continued.frame_receipt)

    edge = handle_local_mpc_step(
        **{
            **aligned,
            "active_pad_fractions": [-0.0204, np.nan],
        },
        contact_fraction_recenter=True,
        require_geometric_preseat_for_closure=True,
        active_pad_fraction_axis_extent_m=0.068,
    )
    edge_recenter = edge.frame_receipt["contact_fraction_recenter"]
    assert edge_recenter["preclosure_geometric_prestage_enabled"]
    assert edge_recenter["preclosure_geometric_prestage_active"]
    assert edge.frame_receipt["executed_control"]["jaw_increment"] == 0.0
    assert np.linalg.norm(
        edge.frame_receipt["executed_control"]["translation_world_m"]
    ) == pytest.approx(0.001)

    interior = handle_local_mpc_step(
        **{
            **aligned,
            "active_pad_fractions": [0.16, np.nan],
        },
        contact_fraction_recenter=True,
        require_geometric_preseat_for_closure=True,
        active_pad_fraction_axis_extent_m=0.068,
    )
    interior_closure = interior.frame_receipt["closure"]
    assert interior_closure["geometric_preseat_satisfied"]
    assert interior_closure["geometric_preseat_interior_pad_count"] == 1
    assert interior.frame_receipt["executed_control"]["jaw_increment"] > 0.0
    assert handle_local_mpc_frame_receipt_complete(interior.frame_receipt)

    force_backed = handle_local_mpc_step(
        **{
            **aligned,
            "active_pad_fractions": [0.16, np.nan],
            "active_finger_forces_n": [2.0, 0.0],
        },
        contact_fraction_recenter=True,
        require_geometric_preseat_for_closure=True,
        active_pad_fraction_axis_extent_m=0.068,
    )
    assert not force_backed.frame_receipt["closure"][
        "geometric_preseat_source_wrist_target_active"
    ]


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
