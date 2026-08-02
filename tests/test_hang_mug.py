import numpy as np
import pytest

from judo_isaaclab.hang_mug import (
    HangMugSkillProgram,
    RigidAssetGeometry,
    reanchor_physical_handover,
)


def _pose(x=0.0, y=0.0, z=0.0):
    return np.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0])


def test_asset_geometry_scales_object_relative_semantic_frame():
    source = RigidAssetGeometry(_pose(1.0, 2.0, 3.0), [0.2, 0.1, 0.3])
    target = RigidAssetGeometry(_pose(4.0, 5.0, 6.0), [0.4, 0.15, 0.24])
    transferred = target.transfer_pose_from(source, _pose(1.05, 2.04, 3.1))
    assert transferred[:3] == pytest.approx([4.1, 5.06, 6.08])


def test_hangmug_program_is_one_continuous_named_rollout():
    program = HangMugSkillProgram(_pose(), _pose(0.0, -1.0, 0.0))
    left_observer = _pose(0.4, 0.3, 0.4)
    program.semantic_left_grasp(
        _pose(0.1),
        _pose(0.2),
        _pose(0.2, 0.0, 0.2),
        approach_steps=2,
        close_steps=2,
        lift_steps=2,
    )
    program.physical_handover(
        _pose(0.3, 0.0, 0.2),
        _pose(0.3, -0.8, 0.2),
        _pose(0.3, -0.7, 0.2),
        _pose(0.3, 0.2, 0.2),
        approach_steps=2,
        close_steps=2,
        release_steps=2,
    )
    program.handle_to_branch_insert(
        _pose(0.5, -0.6, 0.3),
        _pose(0.6, -0.5, 0.3),
        _pose(0.7, -0.5, 0.3),
        transport_steps=2,
        approach_steps=2,
        insert_steps=2,
        left_observer=left_observer,
    )
    program.release_and_support(
        _pose(0.71, -0.5, 0.3),
        _pose(0.71, -0.5, 0.3),
        unload_steps=2,
        release_steps=2,
        settle_steps=3,
    )
    trajectory = program.build()
    assert trajectory.steps == 25
    assert set(trajectory.stage_names) == {
        "semantic_left_grasp",
        "physical_handover",
        "handle_to_branch_insertion",
        "release_support",
        "stable_settle",
    }
    assert trajectory.grippers[
        trajectory.waypoint_steps["left_grasp"]
    ] == pytest.approx([0.0, -0.0475])
    assert trajectory.grippers[
        trajectory.waypoint_steps["right_grasp"]
    ] == pytest.approx([0.0, 0.0])
    assert trajectory.grippers[
        trajectory.waypoint_steps["left_release"]
    ] == pytest.approx([-0.0475, 0.0])
    assert trajectory.grippers[
        trajectory.waypoint_steps["right_release"]
    ] == pytest.approx([-0.0475, -0.0475])
    transport_end = trajectory.waypoint_steps["tree_transport"]
    assert trajectory.left_poses[transport_end] == pytest.approx(left_observer)
    assert trajectory.left_poses[transport_end:] == pytest.approx(
        np.broadcast_to(left_observer, trajectory.left_poses[transport_end:].shape)
    )


def test_handover_reanchor_changes_only_handover_and_transport_entry():
    program = HangMugSkillProgram(_pose(), _pose(0.0, -1.0, 0.0))
    program.semantic_left_grasp(
        _pose(0.1), _pose(0.2), _pose(0.3),
        approach_steps=2, close_steps=2, lift_steps=2,
    )
    program.physical_handover(
        _pose(0.3), _pose(0.4, -0.2), _pose(0.4, -0.1), _pose(0.3, 0.2),
        approach_steps=2, close_steps=2, release_steps=2,
    )
    program.handle_to_branch_insert(
        _pose(0.6, -0.1), _pose(0.7, -0.1), _pose(0.8, -0.1),
        transport_steps=2, approach_steps=2, insert_steps=2,
    )
    trajectory = program.build()
    original = trajectory.right_poses.copy()
    adjusted = reanchor_physical_handover(
        trajectory, _pose(0.3), _pose(0.3, 0.1), original[5]
    )
    lift_end = trajectory.waypoint_steps["left_lift"]
    grasp_end = trajectory.waypoint_steps["right_grasp"]
    transport_end = trajectory.waypoint_steps["tree_transport"]
    assert adjusted.right_poses[: lift_end + 1] == pytest.approx(
        original[: lift_end + 1]
    )
    assert adjusted.right_poses[grasp_end, :3] == pytest.approx(
        original[grasp_end, :3] + [0.0, 0.1, 0.0]
    )
    assert adjusted.right_poses[transport_end] == pytest.approx(
        original[transport_end]
    )
    assert adjusted.right_poses[transport_end + 1 :] == pytest.approx(
        original[transport_end + 1 :]
    )
    assert adjusted.left_poses == pytest.approx(trajectory.left_poses)
    assert adjusted.grippers == pytest.approx(trajectory.grippers)
