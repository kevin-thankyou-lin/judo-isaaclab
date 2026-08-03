import numpy as np
import pytest

from judo_isaaclab.put_marker import (
    DrawerGeometry,
    PutMarkerSkillProgram,
    compose_pose,
    center_marker_over_cavity,
    geometry_conditioned_drawer_open_position,
    interpolate_poses,
    inverse_pose,
    offset_handle_pull_pose,
    pose_from_matrix,
    quaternion_rotate,
    reanchor_marker_placement,
    retarget_drawer_local_pose,
    transfer_pose,
)


IDENTITY = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def _pose(x=0.0, y=0.0, z=0.0):
    value = IDENTITY.copy()
    value[:3] = [x, y, z]
    return value


def test_pose_helpers_compose_inverse_and_matrix_conversion():
    half = np.sqrt(0.5)
    transform = np.asarray([1.0, 2.0, 3.0, half, 0.0, 0.0, half])
    assert compose_pose(inverse_pose(transform), transform) == pytest.approx(IDENTITY)

    matrix = np.eye(4)
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    matrix[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    converted = pose_from_matrix(matrix)
    assert converted[:3] == pytest.approx([1.0, 2.0, 3.0])
    assert abs(converted[3]) == pytest.approx(half)
    assert abs(converted[6]) == pytest.approx(half)


def test_transfer_pose_uses_corresponding_frame_and_local_scale():
    source_frame = _pose(1.0, 0.0, 0.0)
    source_value = _pose(1.1, 0.2, 0.3)
    target_frame = _pose(3.0, 4.0, 5.0)
    transferred = transfer_pose(
        source_value,
        source_frame,
        target_frame,
        local_position_scale=(2.0, 0.5, 1.5),
    )
    assert transferred[:3] == pytest.approx([3.2, 4.1, 5.45])


def test_marker_pose_is_centered_in_measured_cavity_frame():
    cavity = _pose(0.8, -0.1, 0.9)
    marker = _pose(0.7, 0.05, 0.86)
    centered = center_marker_over_cavity(marker, cavity)
    assert centered[:3] == pytest.approx([0.8, -0.1, 0.86])
    supported = center_marker_over_cavity(marker, cavity, support_clearance_m=0.03)
    assert supported[:3] == pytest.approx([0.8, -0.1, 0.93])


def test_drawer_local_pose_retargets_to_observed_joint_frame():
    geometry = DrawerGeometry(
        root_pose=_pose(1.0, 2.0, 3.0),
        slide_axis_local=[1.0, 0.0, 0.0],
        joint_origin_local=[0.0, 0.0, 0.0],
        handle_point_local=[0.1, 0.0, 0.0],
        cavity_center_local=[0.0, 0.0, 0.0],
        lower_limit_m=0.0,
        upper_limit_m=0.2,
        cavity_size=[0.2, 0.3, 0.1],
    )
    corrected = retarget_drawer_local_pose(_pose(1.12, 2.02, 3.04), geometry, 0.10, 0.055)
    assert corrected == pytest.approx(_pose(1.075, 2.02, 3.04))


def test_drawer_geometry_preserves_open_fraction_and_semantic_offsets():
    source = DrawerGeometry(
        root_pose=_pose(1.0, 0.0, 0.0),
        slide_axis_local=[1.0, 0.0, 0.0],
        joint_origin_local=[0.01, 0.0, -0.1],
        handle_point_local=[0.16, 0.0, -0.1],
        cavity_center_local=[0.0, 0.0, -0.1],
        lower_limit_m=0.0,
        upper_limit_m=0.25,
        cavity_size=[0.20, 0.30, 0.10],
    )
    target = DrawerGeometry(
        root_pose=_pose(2.0, 1.0, 0.2),
        slide_axis_local=[1.0, 0.0, 0.0],
        joint_origin_local=[0.02, 0.0, -0.12],
        handle_point_local=[0.18, 0.0, -0.12],
        cavity_center_local=[0.01, 0.0, -0.12],
        lower_limit_m=0.0,
        upper_limit_m=0.30,
        cavity_size=[0.24, 0.24, 0.12],
    )
    source_joint = 0.125
    assert target.corresponding_joint_position(source, source_joint) == pytest.approx(0.15)

    source_handle = source.handle_frame(source_joint)
    source_wrist = source_handle.copy()
    source_wrist[:3] += [0.01, 0.02, 0.03]
    target_wrist = target.transfer_handle_pose(source, source_wrist, source_joint)
    target_handle = target.handle_frame(0.15)
    assert target_wrist[:3] - target_handle[:3] == pytest.approx([0.01, 0.02, 0.03])

    source_cavity = source.drawer_frame(source_joint)
    source_marker = source_cavity.copy()
    source_marker[:3] += [0.05, 0.04, 0.02]
    target_marker = target.transfer_drawer_pose(source, source_marker, source_joint)
    target_cavity = target.drawer_frame(0.15)
    assert target_marker[:3] - target_cavity[:3] == pytest.approx([0.06, 0.032, 0.024])


def test_geometry_conditioned_open_position_has_threshold_and_limit_margins():
    geometry = DrawerGeometry(
        root_pose=_pose(),
        slide_axis_local=[1.0, 0.0, 0.0],
        joint_origin_local=[0.0, 0.0, 0.0],
        handle_point_local=[0.1, 0.0, 0.0],
        cavity_center_local=[0.0, 0.0, 0.0],
        lower_limit_m=0.0,
        upper_limit_m=0.11,
        cavity_size=[0.2, 0.3, 0.1],
    )
    assert geometry_conditioned_drawer_open_position(geometry, 0.06) == pytest.approx(
        0.075
    )
    assert geometry_conditioned_drawer_open_position(geometry, 0.10) == pytest.approx(
        0.09
    )


def test_handle_pull_offset_uses_cabinet_semantic_up_axis():
    half = np.sqrt(0.5)
    root = np.asarray([0.0, 0.0, 0.0, half, half, 0.0, 0.0])
    offset = offset_handle_pull_pose(_pose(1.0, 2.0, 3.0), root, -0.025)
    assert offset[:3] == pytest.approx([1.0, 2.025, 3.0])
    with pytest.raises(ValueError, match="finite"):
        offset_handle_pull_pose(_pose(), root, float("nan"))


def test_quintic_pose_interpolation_is_continuous_and_hits_target():
    path = interpolate_poses(_pose(), _pose(0.2, -0.1, 0.3), 20)
    assert path.shape == (20, 7)
    assert path[-1] == pytest.approx(_pose(0.2, -0.1, 0.3))
    steps = np.diff(np.vstack((_pose()[:3], path[:, :3])), axis=0)
    assert np.linalg.norm(steps[0]) < np.linalg.norm(steps[9])
    assert np.linalg.norm(steps[-1]) < np.linalg.norm(steps[9])


def test_put_marker_skills_build_one_continuous_named_rollout():
    program = PutMarkerSkillProgram(_pose(), _pose(0.0, 1.0, 0.0))
    program.grasp_marker(
        _pose(0.1, 0.0, 0.0),
        _pose(0.2, 0.0, 0.0),
        _pose(0.2, 0.0, 0.2),
        approach_steps=4,
        close_steps=3,
        lift_steps=5,
    )
    program.open_drawer(
        _pose(0.2, 0.2, 0.3),
        _pose(0.0, 0.8, 0.0),
        _pose(0.1, 0.8, 0.0),
        _pose(0.3, 0.8, 0.0),
        hold_steps=3,
        approach_steps=4,
        close_steps=3,
        pull_steps=6,
        closed=-0.02,
        pull_closed=0.0,
    )
    program.place_marker_in_drawer(
        _pose(0.4, 0.2, 0.3),
        _pose(0.4, 0.2, 0.1),
        transit_steps=5,
        lower_steps=5,
    )
    program.release_marker(
        _pose(0.4, 0.2, 0.1),
        _pose(0.2, 0.2, 0.4),
        release_steps=3,
        settle_steps=4,
        withdraw_steps=5,
    )
    program.close_drawer(
        _pose(0.1, 0.8, 0.0),
        _pose(0.0, 1.0, 0.2),
        push_steps=6,
        release_steps=3,
    )
    trajectory = program.build()

    assert trajectory.steps == 59
    assert trajectory.left_poses[0, 0] > 0.0
    assert trajectory.left_poses[-1] == pytest.approx(_pose(0.2, 0.2, 0.4))
    assert trajectory.right_poses[-1] == pytest.approx(_pose(0.0, 1.0, 0.2))
    assert set(trajectory.stage_names) == {
        "grasp_marker",
        "open_drawer",
        "place_marker_in_drawer",
        "release_marker",
        "close_drawer",
    }
    release_step = trajectory.waypoint_steps["marker_release"]
    handle_grasp_step = trajectory.waypoint_steps["handle_grasp"]
    drawer_open_step = trajectory.waypoint_steps["drawer_open"]
    assert trajectory.grippers[handle_grasp_step, 1] == pytest.approx(-0.02)
    assert trajectory.grippers[drawer_open_step, 1] == pytest.approx(0.0)
    assert trajectory.grippers[release_step, 0] == pytest.approx(-0.0475)
    assert trajectory.grippers[-1, 1] == pytest.approx(-0.0475)

    observed_marker = _pose(0.2, 0.2, 0.3)
    observed_left = _pose(0.3, 0.2, 0.4)
    adjusted = reanchor_marker_placement(
        trajectory,
        _pose(0.5, -0.1, 0.4),
        _pose(0.6, -0.1, 0.2),
        observed_marker,
        observed_left,
    )
    cavity_end = trajectory.waypoint_steps["marker_cavity"]
    assert adjusted.left_poses[cavity_end] == pytest.approx(
        _pose(0.7, -0.1, 0.3)
    )
    drawer_open_end = trajectory.waypoint_steps["drawer_open"]
    assert adjusted.left_poses[: drawer_open_end + 1] == pytest.approx(
        trajectory.left_poses[: drawer_open_end + 1]
    )
