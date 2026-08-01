import numpy as np
import pytest

from judo_isaaclab.put_pot import (
    CENTERED_ON_COOKTOP_TOLERANCE_M,
    PutPotSkillProgram,
    RigidSupportGeometry,
    cooktop_center_error_m,
    reanchor_centered_support,
    reanchor_centered_unload,
    reanchor_centered_release,
    support_aligned_pot_pose,
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
