import numpy as np
import pytest

from judo_isaaclab.hang_mug import HangMugSkillProgram, RigidAssetGeometry


def _pose(x=0.0, y=0.0, z=0.0):
    return np.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0])


def test_asset_geometry_scales_object_relative_semantic_frame():
    source = RigidAssetGeometry(_pose(1.0, 2.0, 3.0), [0.2, 0.1, 0.3])
    target = RigidAssetGeometry(_pose(4.0, 5.0, 6.0), [0.4, 0.15, 0.24])
    transferred = target.transfer_pose_from(source, _pose(1.05, 2.04, 3.1))
    assert transferred[:3] == pytest.approx([4.1, 5.06, 6.08])


def test_hangmug_program_is_one_continuous_named_rollout():
    program = HangMugSkillProgram(_pose(), _pose(0.0, -1.0, 0.0))
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
