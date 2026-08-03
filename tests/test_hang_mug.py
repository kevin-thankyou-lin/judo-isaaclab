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
    reanchor_right_grasp_from_observed_mug,
)
from judo_isaaclab.semantic_parts import BranchPart, MugParts
from judo_isaaclab.put_marker import compose_pose, quaternion_rotate
from run_hangmug_skill_program import (
    _add_right_handover_assist,
    _install_grasp_assist_config,
    _requires_observed_handover_reanchor,
    _schema_aware_success_acceptance,
    _select_grasp_assist_config,
    _update_authored_assist_releases,
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
    assert np.all(np.isfinite(final))
    target_handle_world = compose_pose(final, target_parts.handle_hole_frame)
    target_support = target_branch.frame.copy()
    target_support[:3] = 0.5 * (
        target_branch.inner_point + target_branch.tip_point
    )
    target_branch_world = compose_pose(_pose(2.0, 3.0, 0.0), target_support)
    assert target_handle_world[:3] == pytest.approx(target_branch_world[:3])
    handle_hole_axis = quaternion_rotate(
        target_handle_world[3:], [0.0, 1.0, 0.0]
    )
    branch_tangent = quaternion_rotate(
        target_branch_world[3:], [1.0, 0.0, 0.0]
    )
    assert handle_hole_axis == pytest.approx(branch_tangent)


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


def test_right_handover_assist_uses_zero_delay_contact_backed_joint():
    config = {
        "left": {
            "mechanism": "friction",
            "arm": "left_arm",
            "target": {"object": "mug"},
            "friction": {"high": 100.0, "low": 0.5},
        }
    }
    selected = _add_right_handover_assist(config)
    assert selected["right"] == {
        **config["left"],
        "arm": "right_arm",
        "mechanism": "fixed_joint",
        "grasp_delay_s": 0.0,
    }
    assert config.keys() == {"left"}


def test_authored_boundaries_release_both_grasp_assists():
    import torch

    class Assist:
        def __init__(self):
            self.calls = []

        def update(self, *, engage, disable):
            self.calls.append((engage.tolist(), disable.tolist()))

    left = Assist()
    right = Assist()
    env = SimpleNamespace(
        robot=SimpleNamespace(
            is_grasping=lambda: (
                torch.tensor([True]),
                torch.tensor([True]),
            )
        ),
        grasp_assists={"left": left, "right": right},
    )
    trajectory = SimpleNamespace(
        waypoint_steps={"left_release": 5, "branch_unload": 7}
    )

    _update_authored_assist_releases(env, trajectory, 4)
    assert left.calls == []
    assert right.calls[-1] == ([True], [False])

    _update_authored_assist_releases(env, trajectory, 5)
    assert left.calls[-1] == ([True], [True])
    assert right.calls[-1] == ([True], [False])

    _update_authored_assist_releases(env, trajectory, 8)
    assert left.calls[-1] == ([True], [True])
    assert right.calls[-1] == ([True], [True])


def test_replay_acceptance_omits_only_skill_driven_right_assist_check():
    checks = {
        "coded_task_success": True,
        "right_handover_observed": True,
        "stable_hang_window": True,
        "physics_device_cpu": True,
        "right_grasp_assist_engaged": False,
        "right_grasp_assist_released": True,
    }

    replay = _schema_aware_success_acceptance(checks, coded_skill=False)
    assert "right_grasp_assist_engaged" not in replay
    assert replay["right_handover_observed"] is True
    assert replay["stable_hang_window"] is True
    assert replay["physics_device_cpu"] is True

    skill = _schema_aware_success_acceptance(checks, coded_skill=True)
    assert skill["right_grasp_assist_engaged"] is False


def test_observed_handover_reanchor_is_geometry_conditioned_for_tall_mugs():
    assert _requires_observed_handover_reanchor(
        SimpleNamespace(body_size=np.asarray([0.08, 0.081, 0.107]))
    )
    assert not _requires_observed_handover_reanchor(
        SimpleNamespace(body_size=np.asarray([0.088, 0.090, 0.077]))
    )
    with pytest.raises(ValueError, match="three positive"):
        _requires_observed_handover_reanchor(
            SimpleNamespace(body_size=np.asarray([0.08, -0.01, 0.10]))
        )


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

    second_mug = _pose(0.55, -0.05, 0.95)
    second_right = _pose(0.64, -0.06, 1.01)
    second_planned_contact = compose_pose(
        inverse_pose(observed_mug), observed_right
    )
    readjusted = reanchor_branch_transport_contact(
        adjusted,
        second_planned_contact,
        second_mug,
        second_right,
        completed_waypoint="tree_transport",
    )
    second_start = trajectory.waypoint_steps["tree_transport"] + 1
    intended_mug = compose_pose(
        adjusted.right_poses[second_start], inverse_pose(second_planned_contact)
    )
    second_observed_contact = compose_pose(
        inverse_pose(second_mug), second_right
    )
    assert readjusted.right_poses[second_start] == pytest.approx(
        compose_pose(intended_mug, second_observed_contact)
    )
    assert readjusted.right_poses[:second_start] == pytest.approx(
        adjusted.right_poses[:second_start]
    )


def test_handover_pregrasp_reanchors_close_to_observed_mug():
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
    adjusted = reanchor_right_grasp_from_observed_mug(
        trajectory, nominal_contact, observed_mug, observed_right
    )
    start = trajectory.waypoint_steps["handover_pregrasp"] + 1
    grasp_end = trajectory.waypoint_steps["right_grasp"]
    release_end = trajectory.waypoint_steps["left_release"]
    corrected = compose_pose(observed_mug, nominal_contact)
    assert adjusted.right_poses[start - 1] == pytest.approx(
        trajectory.right_poses[start - 1]
    )
    assert adjusted.right_poses[grasp_end] == pytest.approx(corrected)
    assert adjusted.right_poses[grasp_end + 1 : release_end + 1] == pytest.approx(
        np.repeat(corrected[None], release_end - grasp_end, axis=0)
    )
