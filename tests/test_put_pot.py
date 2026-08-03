import numpy as np
from types import SimpleNamespace
import pytest

from judo_isaaclab.put_pot import (
    CENTERED_ON_COOKTOP_TOLERANCE_M,
    CONTACT_FEEDBACK_HORIZON_STEPS,
    PutPotSkillProgram,
    RigidSupportGeometry,
    cartesian_smoothness_metrics,
    cooktop_center_error_m,
    handle_axial_contact_scale,
    reanchor_bimanual_transport_from_observation,
    reanchor_centered_support,
    reanchor_centered_unload,
    reanchor_centered_release,
    reanchor_centered_lowering,
    reanchor_second_handle_grasp,
    reanchor_supported_center_slide,
    seat_handle_inside_finger_pads,
    smooth_collision_aware_bimanual_transport,
    support_aligned_pot_pose,
    track_bimanual_handle_targets,
    transfer_handle_pose_preserving_surface_clearance,
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


from judo_isaaclab.put_marker import compose_pose, interpolate_poses, inverse_pose


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


def test_handle_contact_depth_moves_opposite_local_pad_tip_axis():
    grasp = _pose(x=0.07, y=0.02, z=0.04)
    deepened = seat_handle_inside_finger_pads(grasp, 0.003)
    assert deepened[:3] == pytest.approx([0.07, 0.02, 0.043])
    assert deepened[3:] == pytest.approx(grasp[3:])


def test_contact_feedback_chases_a_moving_handle_before_close_window_ends():
    feedback = _linear_contact_feedback_poses(_pose(), _pose(x=0.06), 30)
    assert feedback[0, 0] == pytest.approx(
        0.06 / CONTACT_FEEDBACK_HORIZON_STEPS
    )
    assert feedback[CONTACT_FEEDBACK_HORIZON_STEPS - 1, 0] == pytest.approx(0.06)
    assert feedback[-1, 0] == pytest.approx(0.06)


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
    corrected = reanchor_centered_lowering(
        trajectory, [0.01, -0.02],
        trajectory.left_poses[transport_end], trajectory.right_poses[transport_end],
    )
    assert corrected.left_poses[: transport_end + 1] == pytest.approx(
        original_left[: transport_end + 1]
    )
    lower_end = trajectory.waypoint_steps["support_lower"]
    assert corrected.left_poses[lower_end, :2] == pytest.approx(
        trajectory.left_poses[lower_end, :2] + [0.01, -0.02]
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

    release_corrected = reanchor_centered_release(
        corrected,
        [0.002, -0.003],
        corrected.left_poses[unload_end],
        corrected.right_poses[unload_end],
    )
    assert release_corrected.left_poses[release_end, :2] == pytest.approx(
        corrected.left_poses[release_end, :2] + [0.002, -0.003]
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
    program.supported_center_slide_and_settle(
        _pose(0.6, 0.0, 0.1),
        _pose(0.6, 1.0, 0.1),
        _pose(0.5, 0.2, 0.3),
        _pose(0.7, 0.9, 0.1),
        _pose(0.5, 0.8, 0.3),
        lower_steps=2,
        left_release_steps=2,
        center_steps=3,
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

    reanchored = reanchor_supported_center_slide(
        trajectory,
        _pose(0.6, 0.1, 0.1),
        _pose(0.7, -0.2, 0.1),
        _pose(0.65, 0.9, 0.2),
    )
    slide_end = trajectory.waypoint_steps["center_slide"]
    assert reanchored.right_poses[slide_end, :2] == pytest.approx(
        [0.75, 0.6]
    )
