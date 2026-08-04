import numpy as np
from types import SimpleNamespace
import pytest

from judo_isaaclab.put_marker import _slerp

from judo_isaaclab.put_pot import (
    CENTERED_ON_COOKTOP_TOLERANCE_M,
    CONTACT_FEEDBACK_HORIZON_STEPS,
    TRANSPORT_CONTACT_REANCHOR_MIN_STEPS,
    TWO_CONTACT_PAD_SUPPORT_LIMIT_M,
    HANDLE_PAD_DEPTH_MARGIN_M,
    LOADED_JAW_REACH_AVOIDANCE_FRACTION,
    HANDLE_PAD_GEOMETRIC_MARGIN_M,
    MISSING_FINGER_CONTACT_LIMIT_M,
    MEASURED_TARGET_LEFT_GRASP_ORIENTATION_LOCAL_WXYZ,
    SINGLE_FINGER_CONTACT_LATCH_STEPS,
    PutPotSkillProgram,
    RigidSupportGeometry,
    YAM_FINGER_SEPARATION_LOCAL_M,
    YAM_FINGER_PAD_AXIS_LENGTH_M,
    YAM_LEFT_FINGER_PIVOT_LOCAL_M,
    YAM_RIGHT_FINGER_PIVOT_LOCAL_M,
    apply_object_local_receiving_grasp_orientation,
    balance_handle_contact_across_finger_pads,
    bounded_handle_pad_balance,
    cartesian_smoothness_metrics,
    contact_budget_transport_steps,
    complete_peer_contact_transport_steps,
    center_handle_between_finger_pads,
    cooktop_center_error_m,
    expand_handle_pregrasp_clearance,
    finish_contact_hold_after_coded_pick,
    geometry_conditioned_grasp_hold_steps,
    geometry_conditioned_loaded_jaw_rotation_fraction,
    geometry_gated_milestone_reanchor,
    geometry_conditioned_handle_balance_limit,
    geometry_conditioned_handle_pad_depth,
    geometry_conditioned_peer_contact_transfer,
    geometry_conditioned_right_first_close,
    geometry_conditioned_target_handle_symmetry,
    geometry_conditioned_transport_steps,
    loaded_pick_height_for_support_clearance,
    geometry_conditioned_vertical_rise_fraction,
    advance_loaded_contact_hold_lift,
    remaining_contact_vertical_rise_fraction,
    handle_finger_pad_depth_imbalance,
    handle_jaw_center_offset_m,
    handle_axial_contact_scale,
    maximum_bimanual_position_step_m,
    milestone_reanchor_within_authored_clearance,
    mirror_handle_position_in_receiving_jaw_frame,
    orient_loaded_jaw_around_authored_handle,
    peer_contact_transfer_horizon_steps,
    peer_contact_gripper_reseat_distance_m,
    peer_contact_latch_supported,
    preserve_loaded_contact_target,
    reinforce_loaded_contact_for_motion,
    reanchor_authored_handle_in_observed_jaw,
    reanchor_bimanual_transport_from_observation,
    reanchor_bimanual_contact_hold,
    compensate_retained_contact_tracking,
    reanchor_missing_finger_contact,
    reanchor_missing_finger_pad_depth,
    reanchor_single_contact_pad_fraction,
    reanchor_two_contact_pad_support,
    retime_loaded_gripper_close_for_pad_reseat,
    reanchor_centered_support,
    reanchor_centered_unload,
    reanchor_handle_jaw_center_step,
    reanchor_centered_release,
    reanchor_centered_lowering,
    reanchor_second_handle_grasp,
    reanchor_supported_center_slide,
    seat_handle_inside_finger_pads,
    select_geometry_conditioned_milestone_reanchor,
    single_finger_contact_observed,
    single_contact_pad_base_residual_m,
    single_contact_pad_reseat_saturated,
    smooth_collision_aware_bimanual_transport,
    support_aligned_pot_pose,
    support_boundary_staging_pose,
    track_bimanual_handle_targets,
    track_loaded_pad_center_from_observation,
    track_retained_contact_from_observed_object,
    transfer_peer_contact_pose_in_receiving_jaw_frame,
    twist_jaw_away_from_limited_axis,
    twist_loaded_jaw_about_observed_contact,
    transfer_handle_approach_orientation,
    transfer_handle_pose_through_contact_frames,
    transfer_handle_pose_preserving_surface_clearance,
    transport_contact_reanchor_required,
    transport_reanchor_position_step_limit_m,
    _linear_contact_feedback_poses,
)

from run_putpot_skill_program import _build_center_repair, _sparse_joint_nominal


def test_center_repair_preserves_supported_prefix_and_releases_after_slide():
    sample = {
        "pot_pose": [0.10, 0.20, 0.80, 1.0, 0.0, 0.0, 0.0],
        "cooktop_pose": [0.16, 0.25, 0.73, 1.0, 0.0, 0.0, 0.0],
        "left_eef_pose": [0.3, 0.3, 1.0, 1.0, 0.0, 0.0, 0.0],
        "right_eef_pose": [0.0, 0.20, 0.85, 1.0, 0.0, 0.0, 0.0],
    }
    trajectory = _build_center_repair(
        sample,
        SimpleNamespace(center_repair_steps=6, release_steps=2, withdraw_steps=3, settle_steps=4),
    )
    assert trajectory.steps == 15
    assert trajectory.stage_names[:6] == ["supported_center_repair"] * 6
    assert np.all(trajectory.grippers[:6, 1] == 0.0)
    assert np.all(trajectory.grippers[6:, 1] < 0.0)
    np.testing.assert_allclose(
        trajectory.right_poses[5, :2] - np.asarray(sample["right_eef_pose"])[:2],
        np.asarray(sample["cooktop_pose"])[:2] - np.asarray(sample["pot_pose"])[:2],
        atol=1e-9,
    )


from judo_isaaclab.put_marker import (
    compose_pose,
    interpolate_poses,
    inverse_pose,
    quaternion_multiply,
    quaternion_rotate,
    transfer_pose,
)


def _pose(x=0.0, y=0.0, z=0.0):
    return np.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0])


def test_support_frames_and_alignment_match_exact_planes():
    pot = RigidSupportGeometry(_pose(0.2, 0.1, 0.8), [0.4, 0.3, 0.14])
    cooktop = RigidSupportGeometry(_pose(0.7, -0.3, 0.78), [0.3, 0.34, 0.06])
    aligned = support_aligned_pot_pose(
        pot, cooktop, xy_offset_local=(0.01, -0.02), clearance_m=0.005
    )
    placed = RigidSupportGeometry(aligned, pot.size)
    assert aligned[:2] == pytest.approx([0.71, -0.32])
    assert placed.bottom_frame[2] - cooktop.top_frame[2] == pytest.approx(0.005)


def test_default_support_alignment_targets_true_cooktop_center():
    pot = RigidSupportGeometry(_pose(0.2, 0.1, 0.8), [0.4, 0.3, 0.14])
    cooktop = RigidSupportGeometry(_pose(0.7, -0.3, 0.78), [0.3, 0.34, 0.06])
    aligned = support_aligned_pot_pose(pot, cooktop, clearance_m=0.005)
    assert aligned[:2] == pytest.approx(cooktop.root_pose[:2])
    assert cooktop_center_error_m(aligned, cooktop.root_pose) == pytest.approx(0.0)
    assert cooktop_center_error_m(
        _pose(0.731, -0.3, 0.8), cooktop.root_pose
    ) > CENTERED_ON_COOKTOP_TOLERANCE_M


def test_handle_contact_scales_only_measured_outward_reach():
    scale = handle_axial_contact_scale(
        [0.06, 0.08, 0.04], [0.045, 0.07, 0.02], 0
    )
    assert scale == pytest.approx([0.75, 1.0, 1.0])


def test_handle_transfer_preserves_measured_transverse_surface_clearance():
    source = _pose()
    target = _pose(x=0.4, y=-0.2, z=0.1)
    wrist = _pose(x=0.06, y=0.09, z=0.066)
    transferred = transfer_handle_pose_preserving_surface_clearance(
        wrist,
        source,
        target,
        [0.06, 0.086, 0.041],
        [0.048, 0.082, 0.034],
        0,
    )
    assert transferred[:3] == pytest.approx(
        [0.4 + 0.048, -0.2 + 0.088, 0.1 + 0.0625]
    )
    assert transferred[3:] == pytest.approx(wrist[3:])


def test_handle_approach_and_grasp_share_one_contact_frame_rotation():
    source_root = _pose()
    target_root = np.asarray([0.3, -0.2, 0.1, 0.92387953, 0.0, 0.0, 0.38268343])
    source_contact = _pose(x=0.15)
    target_contact = _pose(x=0.18)
    pregrasp = np.asarray([0.20, 0.04, 0.08, 1.0, 0.0, 0.0, 0.0])
    grasp = np.asarray([0.18, 0.02, 0.06, 0.99904822, 0.0, 0.0, 0.04361939])

    transferred_pregrasp = transfer_handle_approach_orientation(
        pregrasp, source_root, target_root, source_contact, target_contact
    )
    transferred_grasp = transfer_handle_approach_orientation(
        grasp, source_root, target_root, source_contact, target_contact
    )
    source_relative = quaternion_multiply(
        inverse_pose(pregrasp)[3:], grasp[3:]
    )
    target_relative = quaternion_multiply(
        inverse_pose(transferred_pregrasp)[3:], transferred_grasp[3:]
    )
    assert target_relative == pytest.approx(source_relative, abs=1.0e-7)


def test_handle_contact_transfer_preserves_complete_object_to_gripper_pose():
    source_root = _pose(x=0.1, y=-0.2, z=0.3)
    target_root = _pose(x=0.7, y=0.4, z=0.5)
    source_contact = _pose(x=-0.18, y=0.02, z=0.06)
    target_contact = _pose(x=-0.15, y=-0.03, z=0.04)
    source_gripper = compose_pose(
        compose_pose(source_root, source_contact),
        np.asarray([0.04, -0.07, 0.09, 0.92387953, 0.0, 0.0, 0.38268343]),
    )

    target_gripper = transfer_handle_pose_through_contact_frames(
        source_gripper,
        source_root,
        target_root,
        source_contact,
        target_contact,
    )
    source_relative = compose_pose(
        inverse_pose(compose_pose(source_root, source_contact)), source_gripper
    )
    target_relative = compose_pose(
        inverse_pose(compose_pose(target_root, target_contact)), target_gripper
    )
    assert target_relative == pytest.approx(source_relative, abs=1.0e-7)


def test_pregrasp_clearance_expands_by_measured_transverse_handle_shrink():
    grasp = _pose(x=0.20)
    pregrasp = _pose(x=0.20, y=0.03)
    expanded = expand_handle_pregrasp_clearance(
        pregrasp,
        grasp,
        source_handle_size=[0.06, 0.086, 0.041],
        target_handle_size=[0.07, 0.059, 0.028],
        handle_axis=0,
    )
    assert expanded[:3] == pytest.approx([0.20, 0.0435, 0.0])
    assert expanded[3:] == pytest.approx(pregrasp[3:])


def test_handle_contact_depth_moves_opposite_local_pad_tip_axis():
    grasp = _pose(x=0.07, y=0.02, z=0.04)
    deepened = seat_handle_inside_finger_pads(grasp, HANDLE_PAD_DEPTH_MARGIN_M)
    assert HANDLE_PAD_DEPTH_MARGIN_M == pytest.approx(0.016)
    assert deepened[:3] == pytest.approx([0.07, 0.02, 0.056])
    assert deepened[3:] == pytest.approx(grasp[3:])


def test_handle_pad_depth_adds_measured_transverse_cross_section_loss():
    depth = geometry_conditioned_handle_pad_depth(
        [0.0594, 0.0858, 0.0405],
        [0.0687, 0.0589, 0.0272],
        0,
        [0.0, 0.6, 0.8],
    )
    projected = 0.6 * 0.0269 + 0.8 * 0.0133
    assert depth == pytest.approx(0.016 + 0.5 * (projected + 0.0269))
    assert geometry_conditioned_handle_pad_depth(
        [0.06, 0.08, 0.03], [0.07, 0.09, 0.04], 0, [0.0, 0.8, 0.6]
    ) == pytest.approx(HANDLE_PAD_DEPTH_MARGIN_M)


def test_thin_handle_scales_contact_stabilization_duration():
    assert geometry_conditioned_grasp_hold_steps(
        30, [0.06, 0.08, 0.04], [0.07, 0.06, 0.016], 0
    ) == 75
    assert geometry_conditioned_grasp_hold_steps(
        30, [0.06, 0.08, 0.04], [0.07, 0.09, 0.05], 0
    ) == 30
    assert geometry_conditioned_grasp_hold_steps(
        30,
        [0.059453, 0.085784, 0.040560],
        [0.069219, 0.061195, 0.020012],
        0,
    ) == 141


def test_authored_handle_span_is_centered_along_measured_jaw_axis():
    grasp = _pose()
    points = np.asarray(
        [
            [-0.050, 0.010, 0.050],
            [-0.040, 0.010, 0.060],
            [0.010, -0.010, 0.050],
            [0.020, -0.010, 0.060],
        ]
    )
    offset = handle_jaw_center_offset_m(grasp, _pose(), points)
    centered = center_handle_between_finger_pads(grasp, offset)
    assert offset < 0.0
    assert handle_jaw_center_offset_m(centered, _pose(), points) == pytest.approx(
        0.0, abs=1.0e-9
    )


def test_transport_duration_scales_with_measured_handle_cross_section():
    assert geometry_conditioned_transport_steps(
        180,
        [0.0594, 0.0858, 0.0405],
        [0.0687, 0.0589, 0.0272],
        0,
    ) == 269
    assert geometry_conditioned_transport_steps(
        180, [0.06, 0.08, 0.03], [0.07, 0.09, 0.04], 0
    ) == 180
    assert geometry_conditioned_transport_steps(
        180,
        [0.059453, 0.085784, 0.040560],
        [0.069219, 0.061195, 0.020012],
        0,
    ) == 365
    assert geometry_conditioned_transport_steps(
        60,
        [0.059453, 0.085784, 0.040560],
        [0.069219, 0.061195, 0.020012],
        0,
    ) == 122
    # The strict accepted peer-contact control (PutPot031) also uses 120
    # smooth transport steps.  Preserve the complete minimum-jerk path while
    # capping only its sampling horizon to the loaded-contact budget.
    assert contact_budget_transport_steps(365, 80) == 120
    assert contact_budget_transport_steps(50, 80) == 50
    assert contact_budget_transport_steps(180, 0) == 180
    assert complete_peer_contact_transport_steps(365, 80) == 365
    assert complete_peer_contact_transport_steps(180, 0) == 180
    with pytest.raises(ValueError, match="planned_steps"):
        complete_peer_contact_transport_steps(0, 80)
    with pytest.raises(ValueError, match="retained_contact_steps"):
        complete_peer_contact_transport_steps(180, -1)
    assert peer_contact_transfer_horizon_steps(0.02) == 10
    assert peer_contact_transfer_horizon_steps(0.197) == 40
    assert peer_contact_transfer_horizon_steps(0.197, maximum_step_m=0.01) == 20
    with pytest.raises(ValueError, match="translation_m"):
        peer_contact_transfer_horizon_steps(-0.001)


def test_two_contact_pad_support_moves_shallow_contact_baseward_with_bound():
    contact = _pose(x=0.1, z=0.2)
    corrected, cumulative, residual = reanchor_two_contact_pad_support(
        contact,
        _pose(),
        [4.0, 5.0],
        [0.05, 0.30],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        0.0,
        step_m=0.001,
        target_fraction=0.20,
        correction_limit_m=0.0015,
    )
    assert residual > 0.0
    assert cumulative == pytest.approx(-0.001)
    assert corrected[:3] == pytest.approx([0.1, 0.0, 0.199])

    corrected_again, cumulative_again, _ = reanchor_two_contact_pad_support(
        corrected,
        _pose(),
        [4.0, 5.0],
        [0.05, 0.30],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        cumulative,
        step_m=0.001,
        target_fraction=0.20,
        correction_limit_m=0.0015,
    )
    assert cumulative_again == pytest.approx(-0.0015)
    assert corrected_again[2] == pytest.approx(0.1985)


def test_peer_contact_gripper_hold_includes_measured_jaw_centering_motion():
    assert peer_contact_gripper_reseat_distance_m(
        0.0, 0.028, position_locked=False
    ) == pytest.approx(0.028)
    assert peer_contact_gripper_reseat_distance_m(
        0.049, 0.027, position_locked=False
    ) == pytest.approx(0.049)
    assert peer_contact_gripper_reseat_distance_m(
        0.049, 0.027, position_locked=True
    ) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        peer_contact_gripper_reseat_distance_m(
            -0.001, 0.027, position_locked=False
        )


def test_two_contact_pad_support_default_authority_matches_pad_depth_margin():
    assert TWO_CONTACT_PAD_SUPPORT_LIMIT_M == pytest.approx(
        HANDLE_PAD_DEPTH_MARGIN_M
    )
    contact = _pose(z=0.2)
    cumulative = 0.0
    for _ in range(32):
        contact, cumulative, _ = reanchor_two_contact_pad_support(
            contact,
            _pose(),
            [4.0, 5.0],
            [-0.10, 0.30],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            cumulative,
        )
    assert cumulative == pytest.approx(-HANDLE_PAD_DEPTH_MARGIN_M)
    assert contact[2] == pytest.approx(0.2 - HANDLE_PAD_DEPTH_MARGIN_M)


def test_two_contact_pad_support_is_inactive_without_two_finite_contacts():
    contact = _pose(x=0.1, z=0.2)
    for forces, fractions in (([4.0, 0.0], [0.05, np.nan]), ([4.0, 5.0], [0.3, 0.4])):
        corrected, cumulative, residual = reanchor_two_contact_pad_support(
            contact,
            _pose(),
            forces,
            fractions,
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            0.0,
        )
        assert corrected == pytest.approx(contact)
        assert cumulative == pytest.approx(0.0)
        assert residual == pytest.approx(0.0)


def test_loaded_pick_height_acquires_support_clearance_before_transport():
    cooktop = RigidSupportGeometry(_pose(x=0.2, z=0.8), [0.4, 0.4, 0.1])
    overlapping_pot = _pose(x=0.0, z=0.82)

    height = loaded_pick_height_for_support_clearance(
        overlapping_pot,
        [0.2, 0.2, 0.2],
        cooktop,
        collision_clearance_m=0.025,
    )

    # Initial clearance is 0.82 - 0.10 - 0.85 = -0.13 m.
    assert height == pytest.approx(0.155)
    assert loaded_pick_height_for_support_clearance(
        _pose(x=-1.0, z=0.82),
        [0.2, 0.2, 0.2],
        cooktop,
        collision_clearance_m=0.025,
    ) == pytest.approx(0.05)


def test_support_staging_uses_authored_boundary_and_inset():
    cooktop = RigidSupportGeometry(_pose(), [0.4, 0.6, 0.1])
    staged = support_boundary_staging_pose(
        _pose(x=1.0),
        _pose(),
        cooktop,
        support_inset_m=0.006,
    )
    assert staged[:2] == pytest.approx([0.194, 0.0])
    assert staged[2:] == pytest.approx(_pose()[2:])
    assert 0.006 + HANDLE_PAD_GEOMETRIC_MARGIN_M == pytest.approx(0.009)


def test_handle_pad_balance_keeps_right_pivot_and_deepens_left_finger():
    grasp = _pose(x=0.07, y=0.02, z=0.04)
    balanced = balance_handle_contact_across_finger_pads(grasp)
    right_before = grasp[:3] + YAM_RIGHT_FINGER_PIVOT_LOCAL_M
    right_after = balanced[:3] + quaternion_rotate(
        balanced[3:], YAM_RIGHT_FINGER_PIVOT_LOCAL_M
    )
    left_before = grasp[:3] + YAM_LEFT_FINGER_PIVOT_LOCAL_M
    left_after = balanced[:3] + quaternion_rotate(
        balanced[3:], YAM_LEFT_FINGER_PIVOT_LOCAL_M
    )
    assert right_after == pytest.approx(right_before)
    assert left_after[2] - left_before[2] == pytest.approx(0.003, abs=2.0e-6)


def test_negative_handle_pad_balance_keeps_left_pivot_and_deepens_right_finger():
    grasp = _pose(x=0.07, y=0.02, z=0.04)
    balanced = balance_handle_contact_across_finger_pads(grasp, -0.003)
    left_before = grasp[:3] + YAM_LEFT_FINGER_PIVOT_LOCAL_M
    left_after = balanced[:3] + quaternion_rotate(
        balanced[3:], YAM_LEFT_FINGER_PIVOT_LOCAL_M
    )
    right_before = grasp[:3] + YAM_RIGHT_FINGER_PIVOT_LOCAL_M
    right_after = balanced[:3] + quaternion_rotate(
        balanced[3:], YAM_RIGHT_FINGER_PIVOT_LOCAL_M
    )
    assert left_after == pytest.approx(left_before)
    assert right_after[2] - right_before[2] == pytest.approx(0.003, abs=2.0e-6)


def test_handle_pad_depth_imbalance_uses_nearest_authored_points():
    points = np.asarray(
        [
            [-0.045, 0.024, 0.08],
            [-0.044, 0.024, 0.08],
            [0.045, -0.024, 0.11],
            [0.044, -0.024, 0.11],
        ]
    )
    assert handle_finger_pad_depth_imbalance(_pose(), _pose(), points) == pytest.approx(
        -0.03
    )


def test_handle_pad_balance_preserves_measured_imbalance_sign_and_cap():
    assert bounded_handle_pad_balance(0.053) == pytest.approx(0.003)
    assert bounded_handle_pad_balance(-0.026) == pytest.approx(-0.003)
    assert bounded_handle_pad_balance(0.001) == pytest.approx(0.001)


def test_severely_thin_positive_imbalance_gets_measured_extra_pivot_only():
    source = [0.0594, 0.0858, 0.0405]
    target = [0.0666, 0.0657, 0.0171]
    assert geometry_conditioned_handle_balance_limit(
        source, target, 0, 0.030
    ) == pytest.approx(0.005)
    assert geometry_conditioned_handle_balance_limit(
        source, target, 0, -0.038
    ) == pytest.approx(0.003)
    assert geometry_conditioned_handle_balance_limit(
        source, [0.0666, 0.0657, 0.030], 0, 0.030
    ) == pytest.approx(0.003)
    assert geometry_conditioned_handle_balance_limit(
        [0.059453, 0.085784, 0.040560],
        [0.069219, 0.061195, 0.020012],
        0,
        0.0375,
    ) == pytest.approx(0.005)


def test_single_contact_reseats_from_measured_pad_base_residual():
    contact = _pose()
    reseated, cumulative, residual = reanchor_single_contact_pad_fraction(
        contact,
        _pose(),
        [0.0, 2.0],
        [np.nan, 0.84],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        0.0,
    )
    assert residual == pytest.approx(
        (0.84 - 0.25) * YAM_FINGER_PAD_AXIS_LENGTH_M
    )
    assert cumulative == pytest.approx(0.001)
    assert reseated[:3] == pytest.approx([0.0, 0.001, 0.0])

    unchanged, cumulative, residual = reanchor_single_contact_pad_fraction(
        contact,
        _pose(),
        [0.0, 2.0],
        [np.nan, 0.22],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        0.0,
    )
    assert unchanged == pytest.approx(contact)
    assert cumulative == pytest.approx(0.0)
    assert residual == pytest.approx(0.0)


def test_loaded_jaw_twist_scales_only_from_retained_pad_base_residual():
    assert geometry_conditioned_loaded_jaw_rotation_fraction(
        [0.0, 8.0], [np.nan, 0.734]
    ) == pytest.approx(0.934)
    assert geometry_conditioned_loaded_jaw_rotation_fraction(
        [0.0, 8.0], [np.nan, 0.22]
    ) == pytest.approx(LOADED_JAW_REACH_AVOIDANCE_FRACTION)
    assert geometry_conditioned_loaded_jaw_rotation_fraction(
        [8.0, 8.0], [0.2, 0.8]
    ) == pytest.approx(LOADED_JAW_REACH_AVOIDANCE_FRACTION)
    assert geometry_conditioned_loaded_jaw_rotation_fraction(
        [0.0, 8.0], [np.nan, 0.734], jaw_center_residual_m=0.002
    ) == pytest.approx(0.0)
    assert geometry_conditioned_loaded_jaw_rotation_fraction(
        [0.0, 8.0], [np.nan, 0.734], jaw_center_residual_m=0.004
    ) > LOADED_JAW_REACH_AVOIDANCE_FRACTION


def test_pad_reseat_saturation_starts_close_only_after_authority_is_spent():
    limit = 0.04083968658057993
    assert not single_contact_pad_reseat_saturated(
        limit - 0.001, limit, 0.02933932340273547
    )
    assert single_contact_pad_reseat_saturated(
        limit, limit, 0.02933932340273547
    )
    assert not single_contact_pad_reseat_saturated(limit, limit, 0.0)


def test_loaded_gripper_close_holds_for_measured_reseat_then_closes():
    program = PutPotSkillProgram(_pose(), _pose(y=1.0))
    program.bimanual_handle_grasp(
        _pose(), _pose(y=1.0), _pose(x=0.1), _pose(x=0.1, y=1.0),
        approach_steps=2, left_close_steps=4, right_close_steps=4,
        right_first=True, contact_hold_steps=8,
    )
    trajectory = program.build()
    step = trajectory.waypoint_steps["left_handle_grasp"] - 2
    residual = single_contact_pad_base_residual_m(
        [0.0, 8.0], [np.nan, 0.734]
    )
    retimed, hold_steps = retime_loaded_gripper_close_for_pad_reseat(
        trajectory, step, residual, reseat_step_m=0.02
    )
    grasp_end = trajectory.waypoint_steps["bimanual_contact_hold"]
    assert hold_steps == 2
    assert retimed.grippers[step + 1 : step + 3, 0] == pytest.approx(
        trajectory.grippers[step, 0]
    )
    assert np.all(np.diff(retimed.grippers[step + 2 : grasp_end + 1, 0]) >= 0.0)
    assert retimed.grippers[grasp_end, 0] == pytest.approx(0.0)
    assert retimed.grippers[:, 1] == pytest.approx(trajectory.grippers[:, 1])
    assert retimed.left_poses == pytest.approx(trajectory.left_poses)


def test_loaded_gripper_close_without_reseat_uses_full_remaining_window():
    program = PutPotSkillProgram(_pose(), _pose(y=1.0))
    program.bimanual_handle_grasp(
        _pose(), _pose(y=1.0), _pose(x=0.1), _pose(x=0.1, y=1.0),
        approach_steps=2, left_close_steps=4, right_close_steps=4,
        right_first=True, contact_hold_steps=8,
    )
    trajectory = program.build()
    step = trajectory.waypoint_steps["left_handle_grasp"] - 2
    retimed, hold_steps = retime_loaded_gripper_close_for_pad_reseat(
        trajectory, step, 0.0
    )
    grasp_end = trajectory.waypoint_steps["bimanual_contact_hold"]
    expected = np.linspace(
        trajectory.grippers[step, 0], 0.0, grasp_end - step + 1
    )
    assert hold_steps == 0
    assert retimed.grippers[step : grasp_end + 1, 0] == pytest.approx(expected)
    assert np.all(np.diff(retimed.grippers[step : grasp_end + 1, 0]) >= 0.0)


def test_loaded_gripper_open_hold_scales_from_measured_peer_approach():
    program = PutPotSkillProgram(_pose(), _pose(y=1.0))
    program.bimanual_handle_grasp(
        _pose(), _pose(y=1.0), _pose(x=0.1), _pose(x=0.1, y=1.0),
        approach_steps=2, left_close_steps=4, right_close_steps=4,
        right_first=True, contact_hold_steps=100,
    )
    trajectory = program.build()
    step = trajectory.waypoint_steps["right_handle_grasp"]
    retimed, hold_steps = retime_loaded_gripper_close_for_pad_reseat(
        trajectory, step, 0.16975527504133892, reseat_step_m=0.002
    )
    grasp_end = trajectory.waypoint_steps["bimanual_contact_hold"]
    assert hold_steps == 85
    assert retimed.grippers[step : step + hold_steps + 1, 0] == pytest.approx(
        trajectory.grippers[step, 0]
    )
    assert retimed.grippers[grasp_end, 0] == pytest.approx(0.0)


def test_loaded_gripper_hold_covers_measured_jaw_centering_horizon():
    program = PutPotSkillProgram(_pose(), _pose(y=1.0))
    program.bimanual_handle_grasp(
        _pose(), _pose(y=1.0), _pose(x=0.1), _pose(x=0.1, y=1.0),
        approach_steps=2, left_close_steps=4, right_close_steps=4,
        right_first=True, contact_hold_steps=100,
    )
    trajectory = program.build()
    step = trajectory.waypoint_steps["right_handle_grasp"] + 10
    retimed, hold_steps = retime_loaded_gripper_close_for_pad_reseat(
        trajectory, step, 0.0828290903209271, reseat_step_m=0.002
    )
    assert hold_steps == 42
    assert retimed.grippers[step : step + hold_steps + 1, 0] == pytest.approx(
        trajectory.grippers[step, 0]
    )


def test_measured_receiving_orientation_is_reached_before_contact():
    program = PutPotSkillProgram(_pose(), _pose(y=1.0))
    program.bimanual_handle_grasp(
        _pose(), _pose(y=1.0), _pose(x=0.1), _pose(x=0.1, y=1.0),
        approach_steps=2, left_close_steps=4, right_close_steps=4,
        right_first=True, contact_hold_steps=100,
    )
    trajectory = program.build()
    orientation_local = MEASURED_TARGET_LEFT_GRASP_ORIENTATION_LOCAL_WXYZ[
        "pot_023"
    ]
    oriented = apply_object_local_receiving_grasp_orientation(
        trajectory, _pose(), orientation_local
    )
    pregrasp_end = trajectory.waypoint_steps["bimanual_pregrasp"]
    grasp_end = trajectory.waypoint_steps["bimanual_contact_hold"]
    assert oriented.left_poses[0, 3:] == pytest.approx(
        trajectory.left_poses[0, 3:]
    )
    assert oriented.left_poses[pregrasp_end, 3:] == pytest.approx(
        orientation_local
    )
    assert oriented.left_poses[grasp_end, 3:] == pytest.approx(
        orientation_local
    )
    assert oriented.left_poses[:, :3] == pytest.approx(
        trajectory.left_poses[:, :3]
    )
    jaw_local = YAM_FINGER_SEPARATION_LOCAL_M.copy()
    jaw_local[2] = 0.0
    jaw_local /= np.linalg.norm(jaw_local)
    jaw_in_pot = quaternion_rotate(orientation_local, jaw_local)
    assert jaw_in_pot == pytest.approx(
        [-0.98294378, 0.05256169, -0.17623507], abs=1.0e-7
    )


def test_high_reach_avoidance_twist_pivots_about_observed_contact():
    observed = _pose(0.1, 0.2, 0.3)
    loaded = _pose(0.2, 0.4, 0.6)
    loaded[3:] = [0.5, 0.5, 0.5, 0.5]
    pad_centers = [[0.12, 0.21, 0.34], [0.08, 0.19, 0.34]]
    twisted, angle, did_pivot = twist_loaded_jaw_about_observed_contact(
        observed,
        loaded,
        [0.0, 8.0],
        pad_centers,
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        0.934,
    )
    assert did_pivot
    assert np.isfinite(angle)
    pivot_local = quaternion_rotate(
        inverse_pose(observed)[3:],
        np.asarray(pad_centers[1]) - observed[:3],
    )
    preserved_center = twisted[:3] + quaternion_rotate(
        twisted[3:], pivot_local
    )
    assert preserved_center == pytest.approx(pad_centers[1])
    wrist_twisted, _, did_pivot = twist_loaded_jaw_about_observed_contact(
        observed,
        loaded,
        [0.0, 8.0],
        pad_centers,
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        LOADED_JAW_REACH_AVOIDANCE_FRACTION,
    )
    expected, _ = twist_jaw_away_from_limited_axis(
        loaded,
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        rotation_fraction=LOADED_JAW_REACH_AVOIDANCE_FRACTION,
    )
    assert not did_pivot
    assert wrist_twisted == pytest.approx(expected)


def test_loaded_jaw_orientation_centers_handle_around_contacting_pad():
    observed = _pose()
    loaded = _pose()
    handle_points = np.asarray(
        [
            [0.09, 0.09, -0.01],
            [0.11, 0.09, -0.01],
            [0.09, 0.11, 0.01],
            [0.11, 0.11, 0.01],
        ]
    )
    pad_centers = [[0.0, -0.04, 0.0], [0.0, 0.04, 0.0]]
    centered, angle, pivot_residual = (
        orient_loaded_jaw_around_authored_handle(
            observed,
            loaded,
            _pose(),
            handle_points,
            [0.0, 8.0],
            pad_centers,
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        )
    )
    assert abs(angle) > 0.1
    assert abs(pivot_residual) < abs(
        handle_jaw_center_offset_m(loaded, _pose(), handle_points)
    )
    assert handle_jaw_center_offset_m(
        centered, _pose(), handle_points
    ) == pytest.approx(0.0, abs=1.0e-12)


def test_loaded_pad_tracking_reanchors_only_next_left_wrist_target():
    program = PutPotSkillProgram(_pose(), _pose(y=1.0))
    program.bimanual_handle_grasp(
        _pose(), _pose(y=1.0), _pose(x=0.1), _pose(x=0.1, y=1.0),
        approach_steps=2, left_close_steps=4, right_close_steps=4,
        right_first=True, contact_hold_steps=8,
    )
    trajectory = program.build()
    step = trajectory.waypoint_steps["left_handle_grasp"] + 1
    observed = _pose(0.1, 0.2, 0.3)
    centers = np.asarray([[0.12, 0.21, 0.34], [0.08, 0.19, 0.34]])
    retained = _pose(0.3, 0.4, 0.5)
    retained[3:] = [0.5, 0.5, 0.5, 0.5]
    tracked, correction = track_loaded_pad_center_from_observation(
        trajectory, step, observed, [0.0, 8.0], centers, retained
    )
    pivot_local = quaternion_rotate(
        inverse_pose(observed)[3:], centers[1] - observed[:3]
    )
    preserved_center = tracked.left_poses[step + 1, :3] + quaternion_rotate(
        tracked.left_poses[step + 1, 3:], pivot_local
    )
    assert preserved_center == pytest.approx(centers[1])
    assert correction == pytest.approx(
        tracked.left_poses[step + 1, :3]
        - trajectory.left_poses[step + 1, :3]
    )
    assert tracked.right_poses == pytest.approx(trajectory.right_poses)
    assert tracked.left_poses[: step + 1] == pytest.approx(
        trajectory.left_poses[: step + 1]
    )


def test_loaded_pad_tracking_anticipates_next_jaw_close_step():
    program = PutPotSkillProgram(_pose(), _pose(y=1.0))
    program.bimanual_handle_grasp(
        _pose(), _pose(y=1.0), _pose(x=0.1), _pose(x=0.1, y=1.0),
        approach_steps=2, left_close_steps=4, right_close_steps=4,
        right_first=True, contact_hold_steps=8,
    )
    trajectory = program.build()
    step = trajectory.waypoint_steps["left_handle_grasp"] - 2
    observed = _pose(0.1, 0.2, 0.3)
    centers = np.asarray([[0.12, 0.21, 0.34], [0.08, 0.19, 0.34]])
    retained = _pose(0.3, 0.4, 0.5)
    retained[3:] = [0.5, 0.5, 0.5, 0.5]
    tracked, _ = track_loaded_pad_center_from_observation(
        trajectory, step, observed, [0.0, 8.0], centers, retained
    )
    pivot_local = quaternion_rotate(
        inverse_pose(observed)[3:], centers[1] - observed[:3]
    )
    projected_center = tracked.left_poses[step + 1, :3] + quaternion_rotate(
        tracked.left_poses[step + 1, 3:], pivot_local
    )
    closure_m = trajectory.grippers[step + 1, 0] - trajectory.grippers[step, 0]
    away_from_missing = (centers[1] - centers[0]) / np.linalg.norm(
        centers[1] - centers[0]
    )
    assert closure_m > 0.0
    assert projected_center == pytest.approx(
        centers[1] + closure_m * away_from_missing
    )


def test_only_thin_measured_symmetric_target_handles_share_contact_relation():
    source_negative = [0.0594, 0.0858, 0.0405]
    source_positive = [0.0594, 0.0858, 0.0405]
    target_negative = [0.066588, 0.065714, 0.017064]
    target_positive = [0.066586, 0.065714, 0.017060]
    assert geometry_conditioned_target_handle_symmetry(
        source_negative,
        source_positive,
        target_negative,
        target_positive,
        0,
    )
    # Pot020 attempts 002-003 showed that mirroring at 0.4934 retained
    # thickness moves the receiving jaw away from sustained contact.
    assert not geometry_conditioned_target_handle_symmetry(
        [0.059453, 0.085784, 0.040560],
        [0.059405, 0.085782, 0.040535],
        [0.069219, 0.061195, 0.020012],
        [0.069210, 0.061195, 0.020007],
        0,
    )
    assert not geometry_conditioned_target_handle_symmetry(
        source_negative,
        source_positive,
        [0.066, 0.066, 0.030],
        [0.066, 0.066, 0.030],
        0,
    )
    assert not geometry_conditioned_target_handle_symmetry(
        source_negative,
        source_positive,
        target_negative,
        [0.080, 0.050, 0.020],
        0,
    )


def test_target_symmetric_position_transfer_can_preserve_arm_orientation():
    right_handle = _pose(x=0.16)
    left_handle = np.asarray([-0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    right_pose = np.asarray([0.16, -0.04, 0.08, 0.92387953, 0.0, 0.0, 0.38268343])
    left_pose = np.asarray([-0.18, 0.03, 0.09, 0.70710678, 0.70710678, 0.0, 0.0])
    handle_points = np.asarray(
        [
            [-0.20, -0.03, -0.02],
            [-0.20, -0.03, 0.02],
            [-0.14, 0.01, -0.02],
            [-0.14, 0.01, 0.02],
        ]
    )
    mirrored = transfer_pose(right_pose, right_handle, left_handle)
    uncentered = left_pose.copy()
    uncentered[:3] = mirrored[:3]
    assert abs(handle_jaw_center_offset_m(uncentered, _pose(), handle_points)) > 1.0e-3

    result, offset = mirror_handle_position_in_receiving_jaw_frame(
        right_pose,
        right_handle,
        left_pose,
        left_handle,
        _pose(),
        handle_points,
    )

    assert offset != pytest.approx(0.0)
    assert result[3:] == pytest.approx(left_pose[3:])
    assert handle_jaw_center_offset_m(result, _pose(), handle_points) == pytest.approx(
        0.0, abs=1.0e-9
    )


def test_transport_contact_reanchor_uses_contact_feedback_horizon():
    assert TRANSPORT_CONTACT_REANCHOR_MIN_STEPS == CONTACT_FEEDBACK_HORIZON_STEPS


def test_observed_peer_contact_transfers_full_pose_before_receiving_jaw_centering():
    right_handle = _pose(x=0.16)
    left_handle = np.asarray([-0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    right_pose = np.asarray(
        [0.16, -0.04, 0.08, 0.92387953, 0.0, 0.0, 0.38268343]
    )
    handle_points = np.asarray(
        [
            [-0.20, -0.03, -0.02],
            [-0.20, -0.03, 0.02],
            [-0.14, 0.01, -0.02],
            [-0.14, 0.01, 0.02],
        ]
    )

    result, offset = transfer_peer_contact_pose_in_receiving_jaw_frame(
        right_pose,
        right_handle,
        left_handle,
        _pose(),
        handle_points,
    )
    transferred = transfer_pose(right_pose, right_handle, left_handle)

    assert offset != pytest.approx(0.0)
    assert result[3:] == pytest.approx(transferred[3:])
    assert handle_jaw_center_offset_m(result, _pose(), handle_points) == pytest.approx(
        0.0, abs=1.0e-9
    )


def test_full_signed_jaw_residual_can_recenter_latch_target():
    handle_points = np.asarray(
        [
            [-0.20, -0.03, -0.02],
            [-0.20, -0.03, 0.02],
            [-0.14, 0.01, -0.02],
            [-0.14, 0.01, 0.02],
        ]
    )
    grasp = _pose(0.08, -0.06, 0.04)
    residual = handle_jaw_center_offset_m(grasp, _pose(), handle_points)
    centered = center_handle_between_finger_pads(
        grasp, residual, maximum_correction_m=abs(residual)
    )
    assert abs(residual) > 0.04
    assert handle_jaw_center_offset_m(
        centered, _pose(), handle_points
    ) == pytest.approx(0.0, abs=1.0e-9)


def test_single_finger_contact_detection_is_exact_and_thresholded():
    assert SINGLE_FINGER_CONTACT_LATCH_STEPS == CONTACT_FEEDBACK_HORIZON_STEPS
    assert single_finger_contact_observed([0.0, 0.1], [np.nan, 0.2])
    assert single_finger_contact_observed([1.0, 0.0], [0.5, np.nan])
    assert not single_finger_contact_observed([0.0, 0.0], [np.nan, np.nan])
    assert not single_finger_contact_observed([0.1, 0.1], [0.5, 0.5])
    assert not single_finger_contact_observed([0.0, 2.0], [np.nan, -0.005])
    assert not single_finger_contact_observed([0.0, 2.0], [np.nan, 0.81])


def test_peer_contact_latch_uses_measured_contact_before_nominal_close():
    # PutPot023 attempt_023 first sustained this contact 89 controller steps
    # before the nominal close schedule.
    assert peer_contact_latch_supported(True, [0.0, 11.53], [np.nan, 0.801])
    assert not peer_contact_latch_supported(
        False, [0.0, 11.53], [np.nan, 0.801]
    )


def test_loaded_contact_reanchor_preserves_bounded_tracking_residual():
    observed = _pose()
    commanded = _pose(0.03, 0.04, 0.0)
    result, residual = preserve_loaded_contact_target(
        observed, commanded, maximum_position_residual_m=0.025
    )

    assert residual == pytest.approx([0.015, 0.02, 0.0])
    assert result[:3] == pytest.approx(residual)
    assert result[3:] == pytest.approx(commanded[3:])


def test_loaded_contact_residual_survives_object_local_hold_reanchor():
    root = _pose(0.4, -0.2, 0.8)
    observed = _pose(0.6, -0.1, 0.9)
    commanded = _pose(0.63, -0.06, 0.9)
    loaded, residual = preserve_loaded_contact_target(
        observed,
        commanded,
        maximum_position_residual_m=0.025,
    )
    reference_local = compose_pose(inverse_pose(root), loaded)
    retained_local, _ = compensate_retained_contact_tracking(
        reference_local,
        root,
        observed,
    )
    program = PutPotSkillProgram(_pose(), _pose(y=1.0))
    program.bimanual_handle_grasp(
        _pose(x=0.1),
        _pose(0.1, 1.0),
        _pose(x=0.2),
        _pose(0.2, 1.0),
        approach_steps=2,
        left_close_steps=2,
        right_close_steps=2,
        contact_hold_steps=2,
    )
    trajectory = program.build()
    hold = reanchor_bimanual_contact_hold(
        trajectory,
        trajectory.waypoint_steps["left_handle_grasp"],
        root,
        retained_local,
        compose_pose(inverse_pose(root), _pose(0.2, 1.0)),
    )
    first_hold = trajectory.waypoint_steps["left_handle_grasp"] + 1

    assert np.linalg.norm(residual) == pytest.approx(0.025)
    assert np.linalg.norm(hold.left_poses[first_hold, :3] - observed[:3]) > 0.025


def test_motion_preload_follows_loaded_residual_and_handle_geometry():
    observed = _pose(0.10, 0.20, 0.30)
    loaded = _pose(0.12, 0.20, 0.30)
    reinforced, preload = reinforce_loaded_contact_for_motion(
        loaded,
        observed,
        [0.057, 0.083, 0.0212],
        0,
    )
    assert preload == pytest.approx([0.009, 0.0, 0.0])
    assert reinforced[:3] == pytest.approx([0.109, 0.20, 0.30])
    assert reinforced[3:] == pytest.approx(loaded[3:])

    bounded, bounded_preload = reinforce_loaded_contact_for_motion(
        loaded,
        observed,
        [0.057, 0.008, 0.010],
        0,
    )
    assert np.linalg.norm(bounded_preload) == pytest.approx(0.004)
    assert bounded[0] == pytest.approx(0.104)


def test_motion_preload_rebases_stale_attempt024_contact_on_observation():
    # attempt_024 reached a physical bimanual grasp, but the contact authored
    # before closing was 8.7 cm from the measured object-local jaw at transport
    # entry.  Only the direction, not that stale displacement, may survive.
    observed = _pose()
    loaded = _pose(0.055869096, 0.007902598, 0.066114762)
    reinforced, preload = reinforce_loaded_contact_for_motion(
        loaded,
        observed,
        [0.057326, 0.083289, 0.021211],
        0,
    )

    assert np.linalg.norm(loaded[:3] - observed[:3]) > 0.08
    assert np.linalg.norm(preload) == pytest.approx(0.009)
    assert reinforced[:3] == pytest.approx(preload)
    assert np.linalg.norm(reinforced[:3] - observed[:3]) < 0.01


def test_loaded_transport_reference_opposes_drift_without_becoming_observation():
    root = _pose(0.4, -0.2, 0.8)
    loaded_reference = _pose(0.23, 0.12, 0.11)
    observed_local = loaded_reference.copy()
    observed_local[:3] -= [0.013, -0.022, -0.010]
    observed_world = compose_pose(root, observed_local)
    retained, correction = compensate_retained_contact_tracking(
        loaded_reference,
        root,
        observed_world,
    )

    assert np.linalg.norm(correction) == pytest.approx(HANDLE_PAD_GEOMETRIC_MARGIN_M)
    assert np.dot(correction, loaded_reference[:3] - observed_local[:3]) > 0.0
    assert retained[:3] == pytest.approx(loaded_reference[:3] + correction)
    assert retained[:3] != pytest.approx(observed_local[:3])


def test_jaw_twist_removes_limited_axis_without_changing_pad_tangent():
    pose = _pose()
    pads = np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    result, angle = twist_jaw_away_from_limited_axis(pose, pads)
    jaw = YAM_FINGER_SEPARATION_LOCAL_M.copy()
    jaw[2] = 0.0
    jaw /= np.linalg.norm(jaw)

    assert angle != pytest.approx(0.0)
    assert quaternion_rotate(result[3:], jaw)[0] == pytest.approx(0.0, abs=1e-9)
    assert quaternion_rotate(result[3:], [0.0, 1.0, 0.0]) == pytest.approx(
        [0.0, 1.0, 0.0], abs=1e-9
    )


def test_loaded_jaw_twist_uses_measured_partial_reach_avoidance():
    pose = _pose()
    pads = np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    full, full_angle = twist_jaw_away_from_limited_axis(pose, pads)
    partial, partial_angle = twist_jaw_away_from_limited_axis(
        pose,
        pads,
        rotation_fraction=LOADED_JAW_REACH_AVOIDANCE_FRACTION,
    )

    assert partial_angle == pytest.approx(
        LOADED_JAW_REACH_AVOIDANCE_FRACTION * full_angle
    )
    assert partial[3:] != pytest.approx(full[3:])
    with pytest.raises(ValueError, match="rotation_fraction"):
        twist_jaw_away_from_limited_axis(pose, pads, rotation_fraction=1.01)


def test_contact_recovery_remeasures_authored_jaw_residual_each_step():
    points = np.asarray(
        [[0.02, -0.01, 0.0], [0.02, 0.01, 0.0], [0.04, -0.01, 0.0], [0.04, 0.01, 0.0]]
    )
    contact, residual, applied = reanchor_handle_jaw_center_step(
        _pose(y=0.03), _pose(), points
    )
    assert residual > 0.001
    assert applied == pytest.approx(0.001)
    assert handle_jaw_center_offset_m(contact, _pose(), points) == pytest.approx(
        residual - applied
    )


def test_single_finger_contact_reanchors_toward_missing_finger_with_a_cap():
    contact = _pose()
    root = _pose()
    toward_finger_zero, signed = reanchor_missing_finger_contact(
        contact,
        root,
        [0.0, 2.0],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        0.0,
    )
    assert toward_finger_zero[:3] == pytest.approx([0.001, 0.0, 0.0])
    assert signed == pytest.approx(0.001)

    toward_finger_one, signed = reanchor_missing_finger_contact(
        contact,
        root,
        [2.0, 0.0],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        0.0,
    )
    assert toward_finger_one[:3] == pytest.approx([-0.001, 0.0, 0.0])
    assert signed == pytest.approx(-0.001)

    unchanged, signed = reanchor_missing_finger_contact(
        contact,
        root,
        [0.0, 2.0],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        MISSING_FINGER_CONTACT_LIMIT_M,
    )
    assert unchanged == pytest.approx(contact)
    assert signed == pytest.approx(MISSING_FINGER_CONTACT_LIMIT_M)

    collapsed, signed = reanchor_missing_finger_contact(
        contact,
        root,
        [0.0, 2.0],
        [[0.0, 0.0, 0.0], [0.0034, 0.0, 0.0]],
        0.012,
    )
    assert collapsed == pytest.approx(contact)
    assert signed == pytest.approx(0.012)

    closed_jaw, signed = reanchor_missing_finger_contact(
        contact,
        root,
        [2.0, 0.0],
        [[0.0, 0.0, 0.0], [0.0034, 0.0, 0.0]],
        0.0,
        gripper_pose_world=_pose(),
    )
    expected_axis = -YAM_FINGER_SEPARATION_LOCAL_M
    expected_axis /= np.linalg.norm(expected_axis)
    assert closed_jaw[:3] == pytest.approx(-0.001 * expected_axis)
    assert signed == pytest.approx(-0.001)


def test_positive_imbalance_half_thickness_handle_closes_proven_right_first():
    source = [0.059453, 0.085784, 0.040560]
    pot020 = [0.069219, 0.061195, 0.020012]
    pot019 = [0.0687, 0.0589, 0.0171]
    assert geometry_conditioned_right_first_close(source, pot020, 0, 0.0375)
    assert not geometry_conditioned_right_first_close(source, pot020, 0, -0.0375)
    assert not geometry_conditioned_right_first_close(source, pot019, 0, 0.0375)
    assert geometry_conditioned_right_first_close(
        source, [0.070, 0.065, 0.030], 0, 0.0488
    )
    assert not geometry_conditioned_right_first_close(
        source, [0.070, 0.065, 0.030], 0, -0.0488
    )


def test_vertical_rise_finishes_inside_measured_contact_window():
    fraction = geometry_conditioned_vertical_rise_fraction(425, 80)
    assert fraction == pytest.approx(80 / 425)
    assert geometry_conditioned_vertical_rise_fraction(180, 0) == 1.0
    assert geometry_conditioned_vertical_rise_fraction(1000, 80) == 0.15
    assert geometry_conditioned_vertical_rise_fraction(100, 80) == 0.30


def test_loaded_vertical_rise_uses_attempt026_remaining_contact_budget():
    # Both contacts latched at step 234 and transport began after step 296, so
    # 62 of the 80 geometry-conditioned retention steps were already consumed.
    # Finish in the first half and retain the second half for controller lag.
    fraction = remaining_contact_vertical_rise_fraction(288, 80, 62)
    assert fraction == pytest.approx(9 / 288)
    assert int(np.ceil(288 * fraction)) == 9
    assert remaining_contact_vertical_rise_fraction(288, 80, 100) == pytest.approx(
        8 / 288
    )
    assert remaining_contact_vertical_rise_fraction(180, 0, 0) == 1.0


def test_loaded_contact_hold_lifts_to_coded_pick_with_margin():
    initial = _pose(z=0.80)
    command, residual = advance_loaded_contact_hold_lift(
        initial, _pose(z=0.80), 0.0
    )
    assert command == 0.002
    assert residual == 0.002
    command, residual = advance_loaded_contact_hold_lift(
        initial, _pose(z=0.80), 0.024
    )
    assert command == pytest.approx(0.026)
    assert residual == pytest.approx(0.025)
    _, saturated_residual = advance_loaded_contact_hold_lift(
        initial,
        _pose(z=0.80),
        0.024,
        maximum_residual_m=0.05,
    )
    assert saturated_residual == pytest.approx(0.026)
    command, residual = advance_loaded_contact_hold_lift(
        initial, _pose(z=0.854), 0.054
    )
    assert command == pytest.approx(0.055)
    assert residual == pytest.approx(0.001)
    command, residual = advance_loaded_contact_hold_lift(
        initial, _pose(z=0.855), 0.055
    )
    assert command == pytest.approx(0.055)
    assert residual == pytest.approx(0.0)

    program = PutPotSkillProgram(_pose(), _pose(y=1.0))
    program.bimanual_handle_grasp(
        _pose(0.1), _pose(0.1, 1.0), _pose(0.2), _pose(0.2, 1.0),
        approach_steps=2, left_close_steps=2, right_close_steps=2,
        contact_hold_steps=3,
    )
    trajectory = program.build()
    step = trajectory.waypoint_steps["right_handle_grasp"]
    tilted_observation = _pose(z=0.80)
    tilted_observation[3:] = [np.cos(0.2), np.sin(0.2), 0.0, 0.0]
    lifted = reanchor_bimanual_contact_hold(
        trajectory,
        step,
        tilted_observation,
        _pose(0.2),
        _pose(0.2, 1.0),
        object_local_lift_residual_m=0.002,
        support_normal_world=[0.0, 0.0, 1.0],
        support_aligned_object_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
    )
    assert lifted.left_poses[step + 1, 2] == pytest.approx(0.802)
    assert lifted.right_poses[step + 1, 2] == pytest.approx(0.802)
    assert lifted.left_poses[step + 1, 3:] == pytest.approx([1.0, 0.0, 0.0, 0.0])

    gradual = reanchor_bimanual_contact_hold(
        trajectory,
        step,
        tilted_observation,
        _pose(0.2),
        _pose(0.2, 1.0),
        support_aligned_object_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        support_alignment_fraction=0.02,
    )
    expected = _slerp(
        tilted_observation[3:],
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.asarray([0.02]),
    )[0]
    assert gradual.left_poses[step + 1, 3:] == pytest.approx(expected)


def test_matching_handles_beyond_jaw_reach_transfer_proven_peer_contact():
    left = [0.057326, 0.083289, 0.021211]
    right = [0.057321, 0.083291, 0.021211]
    assert geometry_conditioned_peer_contact_transfer(left, right, 0.0488)
    assert not geometry_conditioned_peer_contact_transfer(left, right, 0.0375)
    assert not geometry_conditioned_peer_contact_transfer(
        left, [0.070, 0.060, 0.030], 0.0488
    )


def test_right_first_peer_grasp_holds_closed_contact_before_transport():
    program = PutPotSkillProgram(_pose(), _pose(y=1.0))
    program.bimanual_handle_grasp(
        _pose(0.1),
        _pose(0.1, 1.0),
        _pose(0.2),
        _pose(0.2, 1.0),
        approach_steps=2,
        left_close_steps=2,
        right_close_steps=2,
        right_first=True,
        contact_hold_steps=3,
    )
    trajectory = program.build()
    left_end = trajectory.waypoint_steps["left_handle_grasp"]
    hold_end = trajectory.waypoint_steps["bimanual_contact_hold"]
    assert left_end == 5
    assert hold_end == 8
    assert trajectory.left_poses[left_end + 1 : hold_end + 1] == pytest.approx(
        np.broadcast_to(_pose(0.2), (3, 7))
    )
    assert trajectory.grippers[left_end + 1 : hold_end + 1] == pytest.approx(
        np.zeros((3, 2))
    )
    tracked = track_bimanual_handle_targets(
        trajectory,
        left_end + 1,
        _pose(),
        _pose(0.2),
        _pose(0.2, 1.0),
        _pose(0.2),
        _pose(0.2, 1.0),
        right_first_close=True,
        feedback_horizon_steps=1,
    )
    assert tracked.left_poses[left_end + 2] == pytest.approx(_pose(0.2))
    observed_pot = _pose()
    observed_left = _pose(0.21, 0.01)
    observed_right = _pose(0.21, 1.01)
    reference_left = compose_pose(inverse_pose(observed_pot), observed_left)
    retained_left, signed_retention = compensate_retained_contact_tracking(
        reference_left,
        observed_pot,
        _pose(0.209, 0.009),
    )
    retained_right = compose_pose(inverse_pose(observed_pot), observed_right)
    latched = reanchor_bimanual_contact_hold(
        trajectory, left_end, observed_pot, retained_left, retained_right
    )
    assert latched.left_poses[left_end + 1 : hold_end + 1] == pytest.approx(
        np.broadcast_to(_pose(0.211, 0.011), (3, 7))
    )
    assert latched.right_poses[left_end + 1 : hold_end + 1] == pytest.approx(
        np.broadcast_to(observed_right, (3, 7))
    )
    assert signed_retention == pytest.approx([0.001, 0.001, 0.0])

    class Actions:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self.value

    keyframes = {
        "semantic_indices": {
            "left_pregrasp": 1,
            "right_pregrasp": 2,
            "left_handle_grasp": 3,
            "right_handle_grasp": 4,
            "pot_lift": 5,
            "pot_transport": 6,
            "support_align": 7,
            "support_lower": 8,
            "pot_release": 9,
            "stable_settle": 10,
        }
    }
    nominal = _sparse_joint_nominal(
        {"actions": Actions(np.zeros((11, 14)))}, trajectory, keyframes
    )
    assert nominal.shape == (trajectory.steps, 14)


def test_authored_handle_reanchors_to_observed_open_jaw_midpoint():
    points = np.asarray(
        [
            [0.020, -0.010, 0.0],
            [0.020, 0.010, 0.0],
            [0.040, -0.010, 0.0],
            [0.040, 0.010, 0.0],
        ]
    )
    local, residual, translation = reanchor_authored_handle_in_observed_jaw(
        _pose(),
        _pose(y=0.2),
        [[-0.050, 0.0, 0.0], [0.050, 0.0, 0.0]],
        points,
        [0.0, 0.0, -0.040],
    )
    assert residual == pytest.approx(0.030)
    assert translation == pytest.approx(0.050)
    assert local[:3] == pytest.approx([0.030, 0.2, -0.040])


def test_milestone_reanchor_rejects_motion_beyond_authored_clearance():
    assert milestone_reanchor_within_authored_clearance(0.080, 0.066)
    assert not milestone_reanchor_within_authored_clearance(0.1765, 0.0666)
    with pytest.raises(ValueError):
        milestone_reanchor_within_authored_clearance(-0.001, 0.066)


def test_nonpeer_milestone_cannot_bypass_authored_clearance():
    original = _pose(0.1, 0.2, 0.3)
    candidate = _pose(0.4, 0.5, 0.6)
    selected, accepted = geometry_gated_milestone_reanchor(
        original,
        candidate,
        0.16975527504133892,
        0.06798973344669754,
    )
    assert not accepted
    assert selected == pytest.approx(original)
    selected, accepted = geometry_gated_milestone_reanchor(
        original, candidate, 0.080, 0.06798973344669754
    )
    assert accepted
    assert selected == pytest.approx(candidate)


def test_geometry_conditioned_peer_milestone_preserves_centered_pose_whole():
    original = _pose(0.1, 0.2, 0.3)
    candidate = _pose(0.4, 0.5, 0.6)
    candidate[3:] = [0.5, 0.5, 0.5, 0.5]
    selected, accepted = select_geometry_conditioned_milestone_reanchor(
        original,
        candidate,
        0.16975527504133892,
        0.06798973344669754,
        peer_contact_transfer=True,
    )
    assert accepted
    assert selected == pytest.approx(candidate)
    assert selected[3:] != pytest.approx(original[3:])


def test_missing_finger_pad_residual_scales_deterministic_seating():
    contact = _pose()
    seated, depth = reanchor_missing_finger_pad_depth(
        contact,
        _pose(),
        [0.0, 3.0],
        [-0.043, 0.10],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        0.0,
    )
    assert seated[:3] == pytest.approx([0.0, 0.0, -0.001])
    assert depth == pytest.approx(0.001)

    unchanged, depth = reanchor_missing_finger_pad_depth(
        contact,
        _pose(),
        [0.0, 3.0],
        [np.nan, 0.10],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        0.0,
    )
    assert unchanged == pytest.approx(contact)
    assert depth == pytest.approx(0.0)


def test_contact_feedback_chases_a_moving_handle_before_close_window_ends():
    feedback = _linear_contact_feedback_poses(_pose(), _pose(x=0.06), 30)
    assert feedback[0, 0] == pytest.approx(
        0.06 / CONTACT_FEEDBACK_HORIZON_STEPS
    )
    assert feedback[CONTACT_FEEDBACK_HORIZON_STEPS - 1, 0] == pytest.approx(0.06)
    assert feedback[-1, 0] == pytest.approx(0.06)

    measured = _linear_contact_feedback_poses(
        _pose(), _pose(x=0.063), 30, horizon_steps=3
    )
    assert measured[0, 0] == pytest.approx(0.021)
    assert measured[2, 0] == pytest.approx(0.063)


def test_single_smooth_transport_preserves_contacts_and_clears_cooktop():
    pot_size = np.asarray([0.30, 0.28, 0.20])
    cooktop = RigidSupportGeometry(_pose(0.7, -0.3, 0.8), [0.36, 0.34, 0.10])
    start = _pose(0.05, 0.05, 0.78)
    target = _pose(0.7, -0.3, cooktop.top_frame[2] + 0.5 * pot_size[2] + 0.16)
    left_contact = _pose(0.0, 0.17, 0.02)
    right_contact = _pose(0.0, -0.17, 0.02)
    transport = smooth_collision_aware_bimanual_transport(
        start,
        target,
        left_contact,
        right_contact,
        pot_size,
        cooktop,
        steps=180,
        collision_clearance_m=0.025,
    )
    assert transport.minimum_cooktop_clearance_m >= 0.025 - 1.0e-9
    assert transport.cooktop_overlap_samples > 0
    for index in (0, 60, 120, 179):
        assert compose_pose(
            inverse_pose(transport.pot_poses[index]), transport.left_poses[index]
        ) == pytest.approx(left_contact)
        assert compose_pose(
            inverse_pose(transport.pot_poses[index]), transport.right_poses[index]
        ) == pytest.approx(right_contact)
    metrics = cartesian_smoothness_metrics(
        transport.left_poses, transport.right_poses
    )
    assert metrics["internal_stop_count"] == 0

    fast_rise = smooth_collision_aware_bimanual_transport(
        start,
        target,
        left_contact,
        right_contact,
        pot_size,
        cooktop,
        steps=180,
        collision_clearance_m=0.025,
        vertical_rise_fraction=0.20,
        frontload_horizontal_axis=0,
    )
    assert fast_rise.vertical_rise_steps == 36
    assert fast_rise.pot_poses[35, 2] == pytest.approx(target[2])
    assert fast_rise.pot_poses[35, 0] == pytest.approx(target[0])
    assert fast_rise.minimum_cooktop_clearance_m >= 0.025 - 1.0e-9
    with pytest.raises(ValueError, match="frontload_horizontal_axis"):
        smooth_collision_aware_bimanual_transport(
            start,
            target,
            left_contact,
            right_contact,
            pot_size,
            cooktop,
            steps=180,
            collision_clearance_m=0.025,
            frontload_horizontal_axis=2,
        )
    assert maximum_bimanual_position_step_m(
        transport.left_poses, transport.right_poses
    ) <= metrics["maximum_step_m"]


def test_bimanual_position_step_limit_is_measured_per_arm():
    left = np.asarray([_pose()] * 4 + [_pose(x=0.02)] * 4)
    right = np.asarray([_pose(y=1.0)] * 4 + [_pose(0.0, 1.02)] * 4)
    assert maximum_bimanual_position_step_m(left, right) == pytest.approx(0.02)
    assert cartesian_smoothness_metrics(left, right)["maximum_step_m"] > 0.02


def test_transport_reanchor_step_limit_respects_handle_retention_geometry():
    assert transport_reanchor_position_step_limit_m(0.025, 0.010) == pytest.approx(
        0.010
    )
    assert transport_reanchor_position_step_limit_m(0.008, 0.010) == pytest.approx(
        0.008
    )
    assert transport_reanchor_position_step_limit_m(0.004, 0.010) == pytest.approx(
        0.004
    )


def test_feedback_transport_recovers_clearance_on_first_commanded_sample():
    cooktop = RigidSupportGeometry(_pose(0.7, -0.3, 0.8), [0.36, 0.34, 0.10])
    pot_size = np.asarray([0.22, 0.22, 0.20])
    start = _pose(0.7, -0.18, 0.968)
    target = _pose(0.7, -0.3, 1.0)
    transport = smooth_collision_aware_bimanual_transport(
        start, target, _pose(0.0, 0.13), _pose(0.0, -0.13), pot_size, cooktop,
        steps=20, collision_clearance_m=0.025,
    )
    assert transport.initial_clearance_recovery_m == pytest.approx(0.007, abs=1.0e-4)
    assert transport.minimum_cooktop_clearance_m >= 0.025 - 1.0e-9
    assert transport.pot_poses[0, 2] - start[2] < 0.025


def test_clearance_lift_reaches_height_before_overlap_without_tall_arch():
    cooktop = RigidSupportGeometry(
        [0.7055, -0.3, 0.7843, 0.7071, 0.0, 0.0, -0.7071],
        [0.264, 0.3355, 0.0653],
    )
    transport = smooth_collision_aware_bimanual_transport(
        [0.7069, 0.0048, 0.8370, 0.3473, 0.0146, 0.0, -0.9376],
        [0.7055, -0.3, 0.9303, 0.5141, 0.0, 0.0, -0.8577],
        _pose(-0.20, -0.12, 0.12),
        _pose(0.23, -0.03, 0.11),
        [0.3362, 0.2345, 0.1646],
        cooktop,
        steps=180,
        collision_clearance_m=0.025,
    )

    assert transport.minimum_cooktop_clearance_m >= 0.025 - 1.0e-9
    assert np.max(transport.pot_poses[:, 2]) < 1.10


def test_single_transport_removes_segment_boundary_speed_dips():
    cooktop = RigidSupportGeometry(_pose(0.7, -0.3, 0.8), [0.36, 0.34, 0.10])
    pot_size = np.asarray([0.30, 0.28, 0.20])
    start = _pose(0.05, 0.05, 0.78)
    target = _pose(0.7, -0.3, 1.11)
    left_contact = _pose(0.0, 0.17, 0.02)
    right_contact = _pose(0.0, -0.17, 0.02)
    smooth = smooth_collision_aware_bimanual_transport(
        start, target, left_contact, right_contact, pot_size, cooktop,
        steps=180, collision_clearance_m=0.025,
    )
    lift = _pose(0.05, 0.05, 1.05)
    middle = _pose(0.38, -0.12, 1.11)
    segmented_pot = np.concatenate(
        (
            interpolate_poses(start, lift, 60),
            interpolate_poses(lift, middle, 60),
            interpolate_poses(middle, target, 60),
        )
    )
    segmented_left = np.asarray(
        [compose_pose(pose, left_contact) for pose in segmented_pot]
    )
    segmented_right = np.asarray(
        [compose_pose(pose, right_contact) for pose in segmented_pot]
    )
    smooth_metrics = cartesian_smoothness_metrics(
        smooth.left_poses, smooth.right_poses
    )
    segmented_metrics = cartesian_smoothness_metrics(
        segmented_left, segmented_right
    )
    assert smooth_metrics["internal_stop_count"] == 0
    assert segmented_metrics["internal_stop_count"] > 0
    assert smooth_metrics["peak_jerk_mps3"] < segmented_metrics["peak_jerk_mps3"]


def test_center_feedback_preserves_completed_smooth_transport():
    cooktop = RigidSupportGeometry(_pose(0.7, -0.3, 0.8), [0.36, 0.34, 0.10])
    pot_size = np.asarray([0.30, 0.28, 0.20])
    start = _pose(0.05, 0.05, 0.78)
    high_center = _pose(0.7, -0.3, 0.981)
    left_contact = _pose(0.0, 0.17, 0.02)
    right_contact = _pose(0.0, -0.17, 0.02)
    program = PutPotSkillProgram(_pose(), _pose(0.0, 1.0))
    program.bimanual_handle_grasp(
        _pose(0.1), _pose(0.1, 1.0),
        compose_pose(start, left_contact), compose_pose(start, right_contact),
        approach_steps=2, left_close_steps=2, right_close_steps=2,
    )
    program.smooth_bimanual_transport_to_center(
        start, high_center, left_contact, right_contact, pot_size, cooktop,
        steps=20, collision_clearance_m=0.025,
    )
    lower_left = compose_pose(_pose(0.7, -0.3, 0.956), left_contact)
    lower_right = compose_pose(_pose(0.7, -0.3, 0.956), right_contact)
    program.short_lower_release_and_settle(
        lower_left, lower_right, _pose(0.6, 0.0, 1.2), _pose(0.6, -0.4, 1.2),
        lower_steps=4, release_steps=2, withdraw_steps=3, settle_steps=3,
    )
    trajectory = program.build()
    transport_end = trajectory.waypoint_steps["smooth_transport"]
    original_left = trajectory.left_poses.copy()
    observed_left = trajectory.left_poses[transport_end].copy()
    observed_right = trajectory.right_poses[transport_end].copy()
    observed_left[:2] += [0.03, 0.04]
    observed_right[:2] += [-0.02, 0.01]
    corrected = reanchor_centered_lowering(
        trajectory, [0.01, -0.02],
        observed_left, observed_right,
        vertical_correction_m=-0.007,
    )
    assert corrected.left_poses[: transport_end + 1] == pytest.approx(
        original_left[: transport_end + 1]
    )
    lower_end = trajectory.waypoint_steps["support_lower"]
    assert corrected.left_poses[lower_end, :3] == pytest.approx(
        observed_left[:3] + [0.01, -0.02, -0.007]
    )
    assert corrected.left_poses[lower_end, 3:] == pytest.approx(
        observed_left[3:]
    )
    assert corrected.right_poses[lower_end, :2] == pytest.approx(
        observed_right[:2] + [0.01, -0.02]
    )


def test_handle_and_transport_feedback_follow_observed_pot_without_reset():
    cooktop = RigidSupportGeometry(_pose(0.7, -0.3, 0.8), [0.36, 0.34, 0.10])
    pot_size = np.asarray([0.30, 0.28, 0.20])
    nominal = _pose(0.05, 0.05, 0.78)
    observed = _pose(0.07, 0.04, 0.79)
    left_contact = _pose(0.0, 0.17, 0.02)
    right_contact = _pose(0.0, -0.17, 0.02)
    final = _pose(0.7, -0.3, 0.956)
    program = PutPotSkillProgram(_pose(), _pose(0.0, 1.0))
    program.bimanual_handle_grasp(
        _pose(0.1),
        _pose(0.1, 1.0),
        compose_pose(nominal, left_contact),
        compose_pose(nominal, right_contact),
        approach_steps=2,
        left_close_steps=2,
        right_close_steps=3,
    )
    program.smooth_bimanual_transport_to_center(
        nominal,
        _pose(0.7, -0.3, 1.081),
        left_contact,
        right_contact,
        pot_size,
        cooktop,
        steps=20,
        collision_clearance_m=0.025,
    )
    program.short_lower_release_and_settle(
        compose_pose(final, left_contact),
        compose_pose(final, right_contact),
        _pose(0.6, 0.0, 1.2),
        _pose(0.6, -0.4, 1.2),
        lower_steps=4,
        release_steps=2,
        withdraw_steps=3,
        settle_steps=3,
    )
    trajectory = program.build()
    observed_left = compose_pose(observed, left_contact)
    observed_right = compose_pose(observed, right_contact)
    grasp = reanchor_second_handle_grasp(
        trajectory, nominal, observed, observed_left, observed_right
    )
    right_end = trajectory.waypoint_steps["right_handle_grasp"]
    assert grasp.right_poses[right_end] == pytest.approx(observed_right)
    left_end = trajectory.waypoint_steps["left_handle_grasp"]
    tracked = track_bimanual_handle_targets(
        trajectory,
        left_end,
        observed,
        observed_left,
        trajectory.right_poses[left_end],
        left_contact,
        right_contact,
    )
    assert tracked.left_poses[left_end + 1] == pytest.approx(observed_left)
    assert tracked.right_poses[right_end] == pytest.approx(observed_right)

    pregrasp_end = trajectory.waypoint_steps["bimanual_pregrasp"]
    early = track_bimanual_handle_targets(
        trajectory,
        pregrasp_end,
        observed,
        trajectory.left_poses[pregrasp_end],
        trajectory.right_poses[pregrasp_end],
        left_contact,
        right_contact,
    )
    assert early.left_poses[right_end] == pytest.approx(observed_left)
    assert early.right_poses[right_end] == pytest.approx(observed_right)

    right_first_program = PutPotSkillProgram(_pose(), _pose(0.0, 1.0))
    right_first_program.bimanual_handle_grasp(
        _pose(0.1),
        _pose(0.1, 1.0),
        compose_pose(nominal, left_contact),
        compose_pose(nominal, right_contact),
        approach_steps=2,
        left_close_steps=3,
        right_close_steps=2,
        right_first=True,
    )
    right_first_trajectory = right_first_program.build()
    right_first_pregrasp = right_first_trajectory.waypoint_steps[
        "bimanual_pregrasp"
    ]
    right_first_end = right_first_trajectory.waypoint_steps["right_handle_grasp"]
    left_final_end = right_first_trajectory.waypoint_steps["left_handle_grasp"]
    right_first = track_bimanual_handle_targets(
        right_first_trajectory,
        right_first_pregrasp,
        observed,
        right_first_trajectory.left_poses[right_first_pregrasp],
        right_first_trajectory.right_poses[right_first_pregrasp],
        left_contact,
        right_contact,
        right_first_close=True,
    )
    assert right_first.left_poses[
        right_first_pregrasp + 1 : right_first_end + 1
    ] == pytest.approx(
        right_first_trajectory.left_poses[
            right_first_pregrasp + 1 : right_first_end + 1
        ]
    )
    assert right_first.right_poses[right_first_end] == pytest.approx(observed_right)
    assert right_first.right_poses[left_final_end] == pytest.approx(observed_right)

    latched = track_bimanual_handle_targets(
        trajectory,
        left_end,
        observed,
        observed_left,
        observed_right,
        left_contact,
        right_contact,
        left_contact_latched=True,
        right_contact_latched=True,
    )
    assert latched.left_poses[left_end + 1 : right_end + 1] == pytest.approx(
        np.broadcast_to(observed_left, (right_end - left_end, 7))
    )
    assert latched.right_poses[left_end + 1 : right_end + 1] == pytest.approx(
        np.broadcast_to(observed_right, (right_end - left_end, 7))
    )

    displaced_left = observed_left.copy()
    displaced_left[0] -= 0.06
    right_first = track_bimanual_handle_targets(
        trajectory,
        left_end,
        observed,
        displaced_left,
        observed_right,
        left_contact,
        right_contact,
        right_contact_latched=True,
    )
    assert right_first.right_poses[left_end + 1 : right_end + 1] == pytest.approx(
        np.broadcast_to(observed_right, (right_end - left_end, 7))
    )
    assert right_first.left_poses[left_end + 1, 0] > displaced_left[0]
    assert right_first.left_poses[left_end + 1, 0] < observed_left[0]
    assert right_first.left_poses[left_end + 1, 0] == pytest.approx(
        displaced_left[0]
        + (observed_left[0] - displaced_left[0]) / (right_end - left_end)
    )
    assert right_first.left_poses[right_end] == pytest.approx(observed_left)

    immediate = track_bimanual_handle_targets(
        trajectory,
        left_end,
        observed,
        displaced_left,
        observed_right,
        left_contact,
        right_contact,
        right_contact_latched=True,
        feedback_horizon_steps=1,
    )
    assert immediate.left_poses[left_end + 1] == pytest.approx(observed_left)

    corrected, transport = reanchor_bimanual_transport_from_observation(
        grasp,
        observed,
        observed_left,
        observed_right,
        final,
        pot_size,
        cooktop,
        transport_clearance_m=0.025,
        collision_clearance_m=0.025,
    )
    start = right_end + 1
    assert compose_pose(inverse_pose(transport.pot_poses[0]), corrected.left_poses[start]) == (
        pytest.approx(left_contact)
    )
    assert transport.minimum_cooktop_clearance_m >= 0.025 - 1.0e-9
    lower_end = trajectory.waypoint_steps["support_lower"]
    assert corrected.left_poses[lower_end] == pytest.approx(
        compose_pose(final, left_contact)
    )

    loaded_left_contact = left_contact.copy()
    loaded_left_contact[0] += 0.025
    loaded, loaded_transport = reanchor_bimanual_transport_from_observation(
        grasp,
        observed,
        observed_left,
        observed_right,
        final,
        pot_size,
        cooktop,
        transport_clearance_m=0.025,
        collision_clearance_m=0.025,
        left_contact_local=loaded_left_contact,
        right_contact_local=right_contact,
    )
    assert compose_pose(
        inverse_pose(loaded_transport.pot_poses[0]), loaded.left_poses[start]
    ) == pytest.approx(loaded_left_contact)
    assert loaded.left_poses[lower_end] == pytest.approx(
        compose_pose(final, loaded_left_contact)
    )

    late_step = start + 2
    late, late_transport = reanchor_bimanual_transport_from_observation(
        trajectory,
        observed,
        observed_left,
        observed_right,
        final,
        pot_size,
        cooktop,
        transport_clearance_m=0.025,
        collision_clearance_m=0.025,
        current_step=late_step,
    )
    assert compose_pose(
        inverse_pose(late_transport.pot_poses[0]),
        late.left_poses[late_step + 1],
    ) == pytest.approx(left_contact)


def test_transport_reanchor_repeats_only_after_measured_contact_drift():
    program = PutPotSkillProgram(_pose(), _pose(y=1.0))
    program.bimanual_handle_grasp(
        _pose(0.1), _pose(0.1, 1.0), _pose(0.2), _pose(0.2, 1.0),
        approach_steps=2, left_close_steps=2, right_close_steps=2,
    )
    program.smooth_bimanual_transport_to_center(
        _pose(0.0, 0.0, 0.8), _pose(0.5, 0.0, 1.0),
        _pose(0.0, 0.1), _pose(0.0, -0.1), [0.2, 0.2, 0.2],
        RigidSupportGeometry(_pose(0.5, 0.0, 0.8), [0.4, 0.4, 0.1]),
        steps=20, collision_clearance_m=0.025,
    )
    trajectory = program.build()
    step = trajectory.waypoint_steps["right_handle_grasp"] + 2
    observed_pot = _pose()
    observed_left = _pose(0.2)
    observed_right = _pose(0.2, 1.0)
    left_contact = compose_pose(inverse_pose(observed_pot), observed_left)
    right_contact = compose_pose(inverse_pose(observed_pot), observed_right)
    assert transport_contact_reanchor_required(
        trajectory, step, observed_pot, observed_left, observed_right, None, None,
        last_reanchor_step=None, tracking_tolerance_m=0.01,
    )
    drifted_right = observed_right.copy()
    drifted_right[1] += 0.012
    assert not transport_contact_reanchor_required(
        trajectory, step, observed_pot, observed_left, drifted_right,
        left_contact, right_contact,
        last_reanchor_step=step - 5, tracking_tolerance_m=0.01,
    )
    assert not transport_contact_reanchor_required(
        trajectory,
        step,
        observed_pot,
        observed_left,
        drifted_right,
        left_contact,
        right_contact,
        last_reanchor_step=step - 20,
        tracking_tolerance_m=0.01,
        minimum_interval_steps=80,
    )
    assert transport_contact_reanchor_required(
        trajectory, step, observed_pot, observed_left, drifted_right,
        left_contact, right_contact,
        last_reanchor_step=step - 10, tracking_tolerance_m=0.01,
    )
    loaded_left_reference = left_contact.copy()
    loaded_left_reference[0] += 0.028
    assert not transport_contact_reanchor_required(
        trajectory,
        step,
        observed_pot,
        observed_left,
        observed_right,
        loaded_left_reference,
        right_contact,
        last_reanchor_step=step - 10,
        tracking_tolerance_m=0.01,
        expected_left_tracking_residual_local=[0.028, 0.0, 0.0],
        expected_right_tracking_residual_local=[0.0, 0.0, 0.0],
    )
    with pytest.raises(ValueError, match="expected left tracking residual"):
        transport_contact_reanchor_required(
            trajectory,
            step,
            observed_pot,
            observed_left,
            observed_right,
            loaded_left_reference,
            right_contact,
            last_reanchor_step=step - 10,
            tracking_tolerance_m=0.01,
            expected_left_tracking_residual_local=[0.028, 0.0],
        )
    transport_end = trajectory.waypoint_steps["smooth_transport"]
    assert not transport_contact_reanchor_required(
        trajectory,
        transport_end - 7,
        observed_pot,
        observed_left,
        observed_right,
        left_contact,
        right_contact,
        last_reanchor_step=step,
        tracking_tolerance_m=0.01,
    )


def test_retained_contact_tracking_changes_only_next_left_transport_command():
    program = PutPotSkillProgram(_pose(), _pose(0.0, 1.0))
    program.bimanual_handle_grasp(
        _pose(0.1), _pose(0.1, 1.0), _pose(0.2), _pose(0.2, 1.0),
        approach_steps=2, left_close_steps=2, right_close_steps=2,
    )
    program.smooth_bimanual_transport_to_center(
        _pose(0.0, 0.0, 0.8), _pose(0.5, 0.0, 1.0),
        _pose(0.0, 0.1), _pose(0.0, -0.1), [0.2, 0.2, 0.2],
        RigidSupportGeometry(_pose(0.5, 0.0, 0.8), [0.4, 0.4, 0.1]),
        steps=20, collision_clearance_m=0.025,
    )
    trajectory = program.build()
    original_left = trajectory.left_poses.copy()
    original_right = trajectory.right_poses.copy()
    step = trajectory.waypoint_steps["right_handle_grasp"] + 3
    observed_pot = _pose(0.03, -0.02, 0.01)
    retained_left_contact = _pose(0.11, 0.04, 0.06)

    tracked = track_retained_contact_from_observed_object(
        trajectory, step, observed_pot, retained_left_contact
    )

    planned_delta = compose_pose(
        original_right[step + 1], inverse_pose(original_right[step])
    )
    assert tracked.left_poses[step + 1] == pytest.approx(
        compose_pose(
            planned_delta,
            compose_pose(observed_pot, retained_left_contact),
        )
    )
    assert tracked.left_poses[: step + 1] == pytest.approx(
        original_left[: step + 1]
    )
    assert tracked.left_poses[step + 2 :] == pytest.approx(
        original_left[step + 2 :]
    )
    assert tracked.right_poses == pytest.approx(original_right)
    with pytest.raises(ValueError, match="outside transport"):
        track_retained_contact_from_observed_object(
            trajectory,
            trajectory.waypoint_steps["smooth_transport"],
            observed_pot,
            retained_left_contact,
        )


def test_support_geometry_transfers_object_relative_handle_pose_with_scale():
    source = RigidSupportGeometry(_pose(1.0, 2.0, 3.0), [0.4, 0.2, 0.1])
    target = RigidSupportGeometry(_pose(4.0, 5.0, 6.0), [0.5, 0.4, 0.2])
    source_wrist = _pose(1.2, 2.1, 3.05)
    target_wrist = target.transfer_pose_from(source, source_wrist)
    assert target_wrist[:3] == pytest.approx([4.25, 5.2, 6.1])


def test_putpot_program_is_one_continuous_named_bimanual_rollout():
    program = PutPotSkillProgram(_pose(), _pose(0.0, 1.0, 0.0))
    program.bimanual_handle_grasp(
        _pose(0.1, 0.0, 0.0),
        _pose(0.1, 1.0, 0.0),
        _pose(0.2, 0.0, 0.0),
        _pose(0.2, 1.0, 0.0),
        approach_steps=4,
        left_close_steps=3,
        right_close_steps=3,
    )
    program.lift_and_transport(
        _pose(0.2, 0.0, 0.2),
        _pose(0.2, 1.0, 0.2),
        _pose(0.5, 0.0, 0.3),
        _pose(0.5, 1.0, 0.3),
        _pose(0.7, 0.0, 0.2),
        _pose(0.7, 1.0, 0.2),
        lift_steps=5,
        transport_steps=6,
        align_steps=4,
    )
    program.unload_release_and_settle(
        _pose(0.7, 0.0, 0.1),
        _pose(0.7, 1.0, 0.1),
        _pose(0.5, 0.0, 0.4),
        _pose(0.5, 1.0, 0.4),
        lower_steps=4,
        unload_steps=3,
        release_steps=3,
        withdraw_steps=4,
        settle_steps=5,
    )
    trajectory = program.build()
    assert trajectory.steps == 44
    assert trajectory.left_poses[-1] == pytest.approx(_pose(0.5, 0.0, 0.4))
    assert trajectory.right_poses[-1] == pytest.approx(_pose(0.5, 1.0, 0.4))
    assert trajectory.grippers[trajectory.waypoint_steps["right_handle_grasp"]] == pytest.approx([0.0, 0.0])
    assert trajectory.grippers[trajectory.waypoint_steps["pot_unload"]] == pytest.approx([0.0, 0.0])
    assert trajectory.grippers[trajectory.waypoint_steps["pot_release"]] == pytest.approx([-0.0475, -0.0475])
    assert set(trajectory.stage_names) == {
        "bimanual_handle_grasp",
        "lift_transport",
        "support_alignment",
        "unload_release",
        "stable_settle",
    }


def test_coded_pick_retime_removes_only_unused_contact_hold():
    program = PutPotSkillProgram(_pose(), _pose(y=1.0))
    program.bimanual_handle_grasp(
        _pose(0.1),
        _pose(0.1, 1.0),
        _pose(0.2),
        _pose(0.2, 1.0),
        approach_steps=2,
        left_close_steps=2,
        right_close_steps=2,
        contact_hold_steps=10,
    )
    program.smooth_bimanual_transport_to_center(
        _pose(z=0.2),
        _pose(x=0.3, z=0.2),
        _pose(0.2),
        _pose(0.2, 1.0),
        [0.2, 0.2, 0.2],
        RigidSupportGeometry(_pose(0.3), [0.5, 0.5, 0.1]),
        steps=8,
        collision_clearance_m=0.01,
    )
    trajectory = program.build()
    old_hold = trajectory.waypoint_steps["bimanual_contact_hold"]
    old_transport = trajectory.waypoint_steps["smooth_transport"]
    current = old_hold - 6

    retimed, removed = finish_contact_hold_after_coded_pick(
        trajectory, current
    )

    assert removed == 5
    assert retimed.waypoint_steps["bimanual_contact_hold"] == current + 1
    assert retimed.waypoint_steps["smooth_transport"] == old_transport - 5
    assert len(retimed.left_poses) == len(trajectory.left_poses)
    assert retimed.stage_names[-5:] == ("smooth_bimanual_transport",) * 5


def test_center_feedback_reanchors_only_support_lowering_and_release():
    program = PutPotSkillProgram(_pose(), _pose(0.0, 1.0, 0.0))
    program.bimanual_handle_grasp(
        _pose(0.1), _pose(0.1, 1.0), _pose(0.2), _pose(0.2, 1.0),
        approach_steps=2, left_close_steps=2, right_close_steps=2,
    )
    program.lift_and_transport(
        _pose(0.2, 0.0, 0.2), _pose(0.2, 1.0, 0.2),
        _pose(0.4, 0.0, 0.3), _pose(0.4, 1.0, 0.3),
        _pose(0.6, 0.0, 0.2), _pose(0.6, 1.0, 0.2),
        lift_steps=2, transport_steps=2, align_steps=2,
    )
    program.unload_release_and_settle(
        _pose(0.6, 0.0, 0.1), _pose(0.6, 1.0, 0.1),
        _pose(0.4, 0.0, 0.4), _pose(0.4, 1.0, 0.4),
        lower_steps=2, unload_steps=2, release_steps=2,
        withdraw_steps=2, settle_steps=2,
    )
    trajectory = program.build()
    original_left = trajectory.left_poses.copy()
    original_right = trajectory.right_poses.copy()
    align_end = trajectory.waypoint_steps["support_align"]
    lower_end = trajectory.waypoint_steps["support_lower"]
    release_end = trajectory.waypoint_steps["pot_release"]
    withdraw_end = trajectory.waypoint_steps["bimanual_withdraw"]
    adjusted = reanchor_centered_support(
        trajectory,
        [0.01, -0.02],
        original_left[align_end],
        original_right[align_end],
    )
    assert adjusted.left_poses[: align_end + 1] == pytest.approx(
        original_left[: align_end + 1]
    )
    assert adjusted.left_poses[lower_end, :2] == pytest.approx(
        original_left[lower_end, :2] + [0.01, -0.02]
    )
    assert adjusted.right_poses[release_end, :2] == pytest.approx(
        original_right[lower_end, :2] + [0.01, -0.02]
    )
    assert adjusted.left_poses[withdraw_end] == pytest.approx(
        original_left[withdraw_end]
    )
    assert adjusted.left_poses[withdraw_end + 1 :] == pytest.approx(
        original_left[withdraw_end + 1 :]
    )

    corrected = reanchor_centered_unload(
        adjusted,
        [0.005, 0.006],
        adjusted.left_poses[lower_end],
        adjusted.right_poses[lower_end],
    )
    unload_end = trajectory.waypoint_steps["pot_unload"]
    assert corrected.left_poses[unload_end, :2] == pytest.approx(
        adjusted.left_poses[unload_end, :2] + [0.005, 0.006]
    )
    assert corrected.right_poses[unload_end, :2] == pytest.approx(
        adjusted.right_poses[unload_end, :2] + [0.005, 0.006]
    )
    assert corrected.left_poses[withdraw_end] == pytest.approx(
        original_left[withdraw_end]
    )

    observed_release_left = corrected.left_poses[unload_end].copy()
    observed_release_right = corrected.right_poses[unload_end].copy()
    observed_release_left[:2] += [0.02, 0.01]
    observed_release_right[:2] += [-0.01, 0.015]
    release_corrected = reanchor_centered_release(
        corrected,
        [0.002, -0.003],
        observed_release_left,
        observed_release_right,
    )
    assert release_corrected.left_poses[release_end, :2] == pytest.approx(
        observed_release_left[:2] + [0.002, -0.003]
    )


def test_simultaneous_bimanual_close_reaches_both_handles_before_hold():
    program = PutPotSkillProgram(_pose(), _pose(y=1.0))
    program.bimanual_handle_grasp(
        _pose(0.1), _pose(0.1, 1.0), _pose(0.2), _pose(0.2, 1.0),
        approach_steps=2, left_close_steps=3, right_close_steps=2,
        simultaneous=True,
    )
    trajectory = program.build()
    first_close = trajectory.waypoint_steps["left_handle_grasp"]
    hold = trajectory.waypoint_steps["right_handle_grasp"]
    assert trajectory.grippers[first_close] == pytest.approx([0.0, 0.0])
    assert trajectory.left_poses[first_close: hold + 1] == pytest.approx(
        np.broadcast_to(_pose(0.2), (hold - first_close + 1, 7))
    )
    assert trajectory.right_poses[first_close: hold + 1] == pytest.approx(
        np.broadcast_to(_pose(0.2, 1.0), (hold - first_close + 1, 7))
    )


def test_supported_center_slide_releases_left_before_exact_center_motion():
    program = PutPotSkillProgram(_pose(), _pose(0.0, 1.0, 0.0))
    program.bimanual_handle_grasp(
        _pose(0.1),
        _pose(0.1, 1.0),
        _pose(0.2),
        _pose(0.2, 1.0),
        approach_steps=2,
        left_close_steps=2,
        right_close_steps=2,
    )
    program.smooth_bimanual_transport_to_center(
        _pose(),
        _pose(0.4),
        _pose(0.2),
        _pose(0.2, 1.0),
        [0.1, 0.1, 0.1],
        RigidSupportGeometry(_pose(10.0), [0.4, 0.4, 0.1]),
        steps=8,
        collision_clearance_m=0.025,
    )
    program.supported_center_slide_and_settle(
        _pose(0.6, 0.0, 0.1),
        _pose(0.6, 1.0, 0.1),
        _pose(0.5, 0.2, 0.3),
        _pose(0.7, 0.9, 0.1),
        _pose(0.5, 0.8, 0.3),
        lower_steps=2,
        left_release_steps=2,
        center_steps=12,
        right_release_steps=2,
        withdraw_steps=2,
        settle_steps=2,
    )
    trajectory = program.build()
    assert trajectory.grippers[
        trajectory.waypoint_steps["left_unload_release"]
    ] == pytest.approx([-0.0475, 0.0])
    assert trajectory.right_poses[
        trajectory.waypoint_steps["center_slide"]
    ] == pytest.approx(_pose(0.7, 0.9, 0.1))
    assert trajectory.grippers[
        trajectory.waypoint_steps["pot_release"]
    ] == pytest.approx([-0.0475, -0.0475])

    transport_reanchored, _ = reanchor_bimanual_transport_from_observation(
        trajectory,
        _pose(0.0, 0.0, 0.1),
        _pose(0.2, 0.0, 0.1),
        _pose(0.2, 1.0, 0.1),
        _pose(0.4, 0.0, 0.1),
        [0.1, 0.1, 0.1],
        RigidSupportGeometry(_pose(10.0), [0.4, 0.4, 0.1]),
        transport_clearance_m=0.025,
        collision_clearance_m=0.025,
    )
    lower_end = trajectory.waypoint_steps["support_lower"]
    left_release_end = trajectory.waypoint_steps["left_unload_release"]
    assert transport_reanchored.left_poses[left_release_end, :3] == pytest.approx(
        transport_reanchored.left_poses[lower_end, :3] + [-0.1, 0.2, 0.2]
    )
    assert transport_reanchored.left_poses[left_release_end + 1] == pytest.approx(
        transport_reanchored.left_poses[left_release_end]
    )

    unloaded = reanchor_supported_center_slide(
        trajectory,
        _pose(0.6, 0.1, 0.1),
        _pose(0.7, -0.2, 0.1),
        _pose(0.65, 0.9, 0.2),
        support_unload_m=0.006,
    )
    slide_start = trajectory.waypoint_steps["left_unload_release"] + 1
    slide_end = trajectory.waypoint_steps["center_slide"]
    assert np.max(unloaded.right_poses[slide_start : slide_end + 1, 2]) > 0.205
    assert unloaded.right_poses[slide_end, :2] == pytest.approx([0.75, 0.6])

    lowered = reanchor_centered_lowering(
        trajectory,
        [0.0, 0.0],
        _pose(0.55, 0.0, 0.1),
        _pose(0.55, 1.0, 0.1),
        vertical_correction_m=0.0,
    )
    left_release_end = trajectory.waypoint_steps["left_unload_release"]
    assert lowered.left_poses[left_release_end, :3] == pytest.approx(
        [0.45, 0.2, 0.3]
    )
    assert lowered.right_poses[left_release_end] == pytest.approx(
        _pose(0.55, 1.0, 0.1)
    )

    reanchored = reanchor_supported_center_slide(
        lowered,
        _pose(0.6, 0.1, 0.1),
        _pose(0.7, -0.2, 0.1),
        _pose(0.65, 0.9, 0.2),
    )
    assert reanchored.right_poses[slide_end, :2] == pytest.approx(
        [0.75, 0.6]
    )
    late = reanchor_supported_center_slide(
        reanchored,
        _pose(0.65, 0.0, 0.1),
        _pose(0.7, -0.2, 0.1),
        _pose(0.72, 0.84, 0.22),
        current_step=slide_end - 10,
        reference_right_contact_local=_pose(0.05, 0.8, 0.1),
        contact_recovery_steps=2,
    )
    assert late.right_poses[slide_end, :2] == pytest.approx([0.75, 0.6])
    assert late.right_poses[slide_end - 8] == pytest.approx(
        _pose(0.70, 0.8, 0.2)
    )
