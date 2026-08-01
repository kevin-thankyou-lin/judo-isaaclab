import numpy as np
import trimesh

from judo_isaaclab.collision_screening import (
    object_path_collision_reports,
    object_path_clearance_reports,
    rigid_weld_eef_poses,
    rigid_weld_object_poses,
    select_robot_feasible_object_path,
    screen_object_paths,
    screen_rigid_weld_paths,
)


IDENTITY = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def test_rigid_weld_preserves_object_to_eef_transform():
    current_eef = IDENTITY.copy()
    current_object = IDENTITY.copy()
    current_object[:3] = [0.2, -0.1, 0.3]
    path = np.broadcast_to(IDENTITY, (3, 7)).copy()
    path[:, 0] = [0.1, 0.2, 0.3]

    result = rigid_weld_object_poses(path, current_eef, current_object)

    np.testing.assert_allclose(
        result[:, :3],
        path[:, :3] + current_object[:3],
        atol=1e-7,
    )

    recovered = rigid_weld_eef_poses(result, current_eef, current_object)
    np.testing.assert_allclose(recovered, path, atol=1e-7)


def test_screen_selects_shortest_clear_path_over_colliding_path():
    object_mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    tree_mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    tree_pose = IDENTITY.copy()
    tree_pose[:3] = [0.5, 0.0, 0.0]
    current_eef = IDENTITY.copy()
    current_object = IDENTITY.copy()
    current_object[:3] = [-0.2, 0.0, 0.0]
    colliding = np.broadcast_to(IDENTITY, (5, 7)).copy()
    colliding[:, 0] = np.linspace(0.0, 0.7, 5)
    clear = colliding.copy()
    clear[:, 1] = np.asarray([0.0, 0.2, 0.2, 0.2, 0.2])

    selected, reports = screen_rigid_weld_paths(
        [colliding, clear],
        current_eef_pose=current_eef,
        current_object_pose=current_object,
        tree_pose=tree_pose,
        object_mesh=object_mesh,
        tree_mesh=tree_mesh,
        preinsert_end_step=4,
        required_clearance_m=0.01,
        sample_stride=1,
        maximum_vertices=100,
    )

    assert selected == 1
    assert reports[0]["valid"] is False
    assert reports[1]["valid"] is True


def test_object_first_screening_uses_the_same_clearance_contract():
    object_mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    tree_mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    tree_pose = IDENTITY.copy()
    tree_pose[:3] = [0.5, 0.0, 0.0]
    colliding = np.broadcast_to(IDENTITY, (5, 7)).copy()
    colliding[:, 0] = np.linspace(-0.2, 0.5, 5)
    clear = colliding.copy()
    clear[:, 1] = np.asarray([0.0, 0.2, 0.2, 0.2, 0.2])

    selected, reports = screen_object_paths(
        [colliding, clear],
        tree_pose=tree_pose,
        object_mesh=object_mesh,
        tree_mesh=tree_mesh,
        preinsert_end_step=4,
        required_clearance_m=0.01,
        sample_stride=1,
        maximum_vertices=100,
    )

    assert selected == 1
    assert reports[0]["valid"] is False
    assert reports[1]["valid"] is True


def test_body_clearance_report_identifies_the_offending_step():
    body_mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    tree_mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    tree_pose = IDENTITY.copy()
    tree_pose[:3] = [0.5, 0.0, 0.0]
    path = np.broadcast_to(IDENTITY, (3, 7)).copy()
    path[:, 0] = [0.0, 0.25, 0.5]

    report = object_path_clearance_reports(
        path,
        tree_pose=tree_pose,
        object_mesh=body_mesh,
        tree_mesh=tree_mesh,
        required_clearance_m=0.01,
        sample_stride=1,
        maximum_vertices=100,
    )[0]

    assert report["valid"] is False
    assert report["minimum_clearance_step"] == 2
    assert report["semantic_surface"] == "object_body"
    assert report["allowed_tree_contact"] is False


def test_exact_body_collision_report_rejects_only_intersecting_path():
    body_mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    tree_mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    tree_pose = IDENTITY.copy()
    tree_pose[:3] = [0.5, 0.0, 0.0]
    colliding = np.broadcast_to(IDENTITY, (3, 7)).copy()
    colliding[:, 0] = [0.0, 0.25, 0.5]
    clear = colliding.copy()
    clear[:, 1] = 0.2

    collision, separated = object_path_collision_reports(
        [colliding, clear],
        tree_pose=tree_pose,
        object_mesh=body_mesh,
        tree_mesh=tree_mesh,
        sample_stride=1,
    )

    assert collision["valid"] is False
    assert collision["collision_steps"] == [2]
    assert collision["first_collision_step"] == 2
    assert separated["valid"] is True
    assert separated["collision_steps"] == []


def test_robot_feasible_selection_prefers_less_terminal_rotation():
    current_eef = IDENTITY.copy()
    current_object = IDENTITY.copy()
    paths = np.broadcast_to(IDENTITY, (2, 4, 7)).copy()
    paths[:, :, 0] = np.linspace(0.0, 0.1, 4)
    half_angle = 0.4 / 2.0
    paths[0, -1, 3:7] = [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)]

    selected, reports = select_robot_feasible_object_path(
        paths,
        current_eef_pose=current_eef,
        current_object_pose=current_object,
    )

    assert selected == 1
    assert reports[0]["eef_terminal_rotation_rad"] > 0.39
    assert reports[1]["eef_terminal_rotation_rad"] == 0.0
