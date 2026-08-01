import numpy as np
import trimesh

from judo_isaaclab.collision_screening import (
    rigid_weld_object_poses,
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
