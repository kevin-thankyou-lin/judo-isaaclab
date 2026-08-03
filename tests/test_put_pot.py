import numpy as np
from types import SimpleNamespace
import pytest

from judo_isaaclab.put_pot import (
    CENTERED_ON_COOKTOP_TOLERANCE_M,
    CONTACT_FEEDBACK_HORIZON_STEPS,
    HANDLE_PAD_DEPTH_MARGIN_M,
    HANDLE_PAD_GEOMETRIC_MARGIN_M,
    MISSING_FINGER_CONTACT_LIMIT_M,
    PutPotSkillProgram,
    RigidSupportGeometry,
    YAM_FINGER_SEPARATION_LOCAL_M,
    YAM_LEFT_FINGER_PIVOT_LOCAL_M,
    YAM_RIGHT_FINGER_PIVOT_LOCAL_M,
    balance_handle_contact_across_finger_pads,
    bounded_handle_pad_balance,
    cartesian_smoothness_metrics,
    center_handle_between_finger_pads,
    cooktop_center_error_m,
    expand_handle_pregrasp_clearance,
    geometry_conditioned_grasp_hold_steps,
    geometry_conditioned_handle_balance_limit,
    geometry_conditioned_handle_pad_depth,
    geometry_conditioned_peer_contact_transfer,
    geometry_conditioned_right_first_close,
    geometry_conditioned_target_handle_symmetry,
    geometry_conditioned_transport_steps,
    handle_finger_pad_depth_imbalance,
    handle_jaw_center_offset_m,
    handle_axial_contact_scale,
    maximum_bimanual_position_step_m,
    milestone_reanchor_within_authored_clearance,
    mirror_handle_position_in_receiving_jaw_frame,
    reanchor_authored_handle_in_observed_jaw,
    reanchor_bimanual_transport_from_observation,
    reanchor_bimanual_contact_hold,
    reanchor_missing_finger_contact,
    reanchor_missing_finger_pad_depth,
    reanchor_centered_support,
    reanchor_centered_unload,
    reanchor_centered_release,
    reanchor_centered_lowering,
    reanchor_second_handle_grasp,
    reanchor_supported_center_slide,
    seat_handle_inside_finger_pads,
    smooth_collision_aware_bimanual_transport,
    support_aligned_pot_pose,
    support_boundary_staging_pose,
    track_bimanual_handle_targets,
    transfer_handle_approach_orientation,
    transfer_handle_pose_through_contact_frames,
    transfer_handle_pose_preserving_surface_clearance,
    transport_contact_reanchor_required,
    transport_reanchor_position_step_limit_m,
    _linear_contact_feedback_poses,
)

from run_putpot_skill_program import _build_center_repair


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
    observed_left = _pose(0.21, 0.01)
    observed_right = _pose(0.21, 1.01)
    latched = reanchor_bimanual_contact_hold(
        trajectory, left_end, observed_left, observed_right
    )
    assert latched.left_poses[left_end + 1 : hold_end + 1] == pytest.approx(
        np.broadcast_to(observed_left, (3, 7))
    )
    assert latched.right_poses[left_end + 1 : hold_end + 1] == pytest.approx(
        np.broadcast_to(observed_right, (3, 7))
    )


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
    assert transport_contact_reanchor_required(
        trajectory, step, observed_pot, observed_left, drifted_right,
        left_contact, right_contact,
        last_reanchor_step=step - 10, tracking_tolerance_m=0.01,
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
