from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from judo_isaaclab.hang_mug import (
    HangMugSkillProgram,
    RigidAssetGeometry,
    ensure_pick_latch_clearance,
    geometry_conditioned_hang_pose,
    reanchor_branch_transport_contact,
    reanchor_physical_handover,
)
from judo_isaaclab.semantic_parts import BranchPart, MugParts
from judo_isaaclab.put_marker import compose_pose
from run_hangmug_skill_program import (
    _install_grasp_assist_config,
    _select_grasp_assist_config,
    _validate_datagen_grasp_assists,
)


def _pose(x=0.0, y=0.0, z=0.0):
    return np.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0])


def test_asset_geometry_scales_object_relative_semantic_frame():
    source = RigidAssetGeometry(_pose(1.0, 2.0, 3.0), [0.2, 0.1, 0.3])
    target = RigidAssetGeometry(_pose(4.0, 5.0, 6.0), [0.4, 0.15, 0.24])
    transferred = target.transfer_pose_from(source, _pose(1.05, 2.04, 3.1))
    assert transferred[:3] == pytest.approx([4.1, 5.06, 6.08])


def test_hang_pose_centers_target_handle_hole_on_authored_branch_support():
    source_parts = MugParts(
        body_frame=_pose(),
        body_size=np.asarray([0.2, 0.2, 0.3]),
        handle_hole_frame=_pose(0.1),
        handle_outer_size=np.asarray([0.08, 0.04, 0.06]),
        handle_thickness_m=0.01,
        handle_axis=0,
        handle_sign=1,
    )
    target_parts = MugParts(
        body_frame=_pose(),
        body_size=np.asarray([0.3, 0.3, 0.25]),
        handle_hole_frame=_pose(0.15),
        handle_outer_size=np.asarray([0.10, 0.08, 0.03]),
        handle_thickness_m=0.012,
        handle_axis=0,
        handle_sign=1,
    )

    def branch(x, z, length):
        return BranchPart(
            frame=_pose(x, 0.0, z),
            inner_point=np.asarray([x - 0.4, 0.0, z]),
            tip_point=np.asarray([x + 0.1, 0.0, z]),
            tangent=np.asarray([1.0, 0.0, 0.0]),
            length_m=length,
            radius_m=0.01,
            normalized_height=z / 2.0,
            azimuth_rad=0.0,
        )

    source_branch = branch(1.0, 1.0, 1.0)
    target_branch = branch(1.5, 1.5, 1.5)
    final, matched_source, matched_target = geometry_conditioned_hang_pose(
        _pose(1.0, 0.02, 1.03),
        _pose(),
        source_parts,
        target_parts,
        (branch(-1.0, 0.3, 0.8), source_branch),
        _pose(2.0, 3.0, 0.0),
        (branch(-1.0, 0.4, 0.9), target_branch),
    )

    assert matched_source is source_branch
    assert matched_target is target_branch
    assert final[:3] == pytest.approx([3.35, 3.0, 1.5])
    target_handle_world = compose_pose(final, target_parts.handle_hole_frame)
    target_branch_world = compose_pose(_pose(2.0, 3.0, 0.0), target_branch.frame)
    assert target_handle_world[:3] == pytest.approx(target_branch_world[:3])


def test_datagen_grasp_assist_validation_requires_canonical_mechanism():
    FixedJointGraspAssist = type("FixedJointGraspAssist", (), {})
    env = SimpleNamespace(grasp_assists={"left": FixedJointGraspAssist()})
    config = {"left": {"mechanism": "fixed_joint", "arm": "left_arm"}}
    assert _validate_datagen_grasp_assists(env, config) == (
        "task_config:left=fixed_joint"
    )
    with pytest.raises(RuntimeError, match="names differ"):
        _validate_datagen_grasp_assists(env, {"right": config["left"]})


def test_datagen_grasp_assist_mechanism_override():
    config = {
        "left": {
            "mechanism": "fixed_joint",
            "arm": "left_arm",
            "target": {"object": "mug"},
            "friction": {"high": 100.0, "low": 0.5},
        }
    }
    selected = _select_grasp_assist_config(config, "friction")
    manager_module = SimpleNamespace(GRASP_ASSIST_CONFIG=config)
    config_module = SimpleNamespace(GRASP_ASSIST_CONFIG=config)
    _install_grasp_assist_config(manager_module, config_module, selected)

    assert selected["left"]["mechanism"] == "friction"
    assert manager_module.GRASP_ASSIST_CONFIG["left"]["mechanism"] == "friction"
    assert config_module.GRASP_ASSIST_CONFIG["left"]["mechanism"] == "friction"
    assert config["left"]["mechanism"] == "fixed_joint"


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


def test_pick_clearance_uses_measured_mug_height():
    initial = _pose(0.0, 0.0, 0.80)
    shallow = _pose(0.2, 0.1, 0.86)
    adjusted = ensure_pick_latch_clearance(shallow, initial, 0.07)
    assert adjusted[:2] == pytest.approx(shallow[:2])
    assert adjusted[2] == pytest.approx(0.92)


def test_branch_transport_reanchors_observed_right_contact():
    program = HangMugSkillProgram(_pose(z=1), _pose(z=1))
    program.semantic_left_grasp(
        _pose(0.1, z=1), _pose(0.2, z=1), _pose(0.3, z=1),
        approach_steps=2, close_steps=2, lift_steps=2,
    )
    program.physical_handover(
        _pose(0.3, z=1), _pose(0.3, -0.1, 1), _pose(0.3, -0.2, 1),
        _pose(0.3, 0.1, 1), approach_steps=2, close_steps=2, release_steps=2,
    )
    program.handle_to_branch_insert(
        _pose(0.5, -0.2, 1.1), _pose(0.6, -0.3, 1.0),
        _pose(0.7, -0.4, 0.9), transport_steps=2, approach_steps=2,
        insert_steps=2,
    )
    trajectory = program.build()
    nominal_contact = _pose(0.05, -0.02, 0.03)
    observed_mug = _pose(0.4, 0.2, 0.8)
    observed_right = _pose(0.47, 0.16, 0.85)
    adjusted = reanchor_branch_transport_contact(
        trajectory, nominal_contact, observed_mug, observed_right
    )
    start = trajectory.waypoint_steps["left_release"] + 1
    from judo_isaaclab.put_marker import compose_pose, inverse_pose

    observed_contact = compose_pose(inverse_pose(observed_mug), observed_right)
    intended_mug = compose_pose(
        trajectory.right_poses[start], inverse_pose(nominal_contact)
    )
    assert adjusted.right_poses[start] == pytest.approx(
        compose_pose(intended_mug, observed_contact)
    )
    assert adjusted.right_poses[:start] == pytest.approx(
        trajectory.right_poses[:start]
    )
