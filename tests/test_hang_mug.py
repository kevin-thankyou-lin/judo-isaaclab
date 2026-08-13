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
    reanchor_handover_contact_acquire,
    reanchor_physical_handover,
    reanchor_right_grasp_from_observed_mug,
    transfer_handover_contact_by_handle_frame,
)
from judo_isaaclab.semantic_parts import BranchPart, MugParts
from judo_isaaclab.put_marker import compose_pose, inverse_pose, quaternion_rotate
from run_hangmug_skill_program import (
    _branch_approach_mug_pose,
    _branch_reanchor_waypoints,
    _branch_support_seated_pose,
    PROVEN_CONTROL_DEFAULTS,
    _add_right_handover_assist,
    _array_sha256,
    _bounded_handover_offset,
    _direct_actions_exact,
    _install_grasp_assist_config,
    _handover_boundary_receipt,
    _handover_target_with_local_pitch,
    _handover_target_with_local_straddle,
    _handover_contact_acquire_guard_receipt,
    _handover_lift_guard_receipt,
    _pick_boundary_receipt,
    _independent_terminal_hang_receipt,
    _require_proven_control_defaults,
    _resolve_target_assets,
    _requires_observed_handover_reanchor,
    _require_reusable_pick_boundary,
    _source_pick_prefix_steps,
    _source_dataset_receipt,
    _sparse_joint_nominal,
    _schema_aware_success_acceptance,
    _semantic_stage_receipt,
    _select_grasp_assist_config,
    _support_preserving_target_state,
    _terminal_stability,
    _trajectory_after,
    _update_authored_assist_releases,
    _validate_datagen_grasp_assists,
)


def test_branch_approach_height_can_preserve_middle_branch_axis():
    final = _pose(0.5, -0.2, 0.9)
    branch = _pose()

    on_axis = _branch_approach_mug_pose(final, branch, 0.04, 0.0)
    raised = _branch_approach_mug_pose(final, branch, 0.04, 0.03)

    assert on_axis[:3] == pytest.approx([0.54, -0.2, 0.9])
    assert raised[:3] == pytest.approx([0.54, -0.2, 0.93])
    with pytest.raises(ValueError, match="branch approach height"):
        _branch_approach_mug_pose(final, branch, 0.04, 0.081)


def test_handover_contact_transfer_preserves_authored_handle_frame_relation():
    source_mug = _pose(0.4, 0.1, 0.9)
    target_mug = _pose(0.7, -0.2, 0.8)
    source_handle = _pose(0.03, 0.0, 0.01)
    target_handle = _pose(0.06, 0.0, -0.02)
    source_eef = compose_pose(
        compose_pose(source_mug, source_handle), _pose(-0.09, -0.02, 0.14)
    )

    transferred = transfer_handover_contact_by_handle_frame(
        source_mug, target_mug, source_handle, target_handle, source_eef
    )

    source_relation = compose_pose(
        inverse_pose(compose_pose(source_mug, source_handle)), source_eef
    )
    target_relation = compose_pose(
        inverse_pose(compose_pose(target_mug, target_handle)), transferred
    )
    np.testing.assert_allclose(target_relation, source_relation, atol=1.0e-9)


def test_semantic_stage_receipt_stops_at_first_incomplete_completed_stage(monkeypatch):
    monkeypatch.setattr(
        "dc_study.datagen.hang_mug_status.ORDERED_STAGES",
        ("pick", "handover", "alignment", "insertion_and_support", "release_and_hang"),
    )
    diagnostics = {
        "selected_branch": None,
        "failure_reason": None,
        "failure_step": None,
        "deepest_overlap_m": 0.0,
        "consecutive_overlap_steps": 0,
        "fallen": False,
    }
    base = {
        "pick": False, "handover": False, "alignment": False,
        "insertion_and_support": False, "release_and_hang": False,
        "task_success": False, "released": False, "stable": False,
        "contact_policy": True, "diagnostics": diagnostics,
    }
    statuses = [dict(base), {**base, "pick": True}, {**base, "pick": True, "alignment": True}]
    receipt = _semantic_stage_receipt(statuses)
    assert receipt["completed_stages"] == ["pick"]
    assert receipt["last_completed_stage"] == "pick"
    assert receipt["first_failed_stage"] == "handover"
    assert receipt["first_completed_steps"]["pick"] == 1
    assert receipt["first_completed_steps"]["alignment"] == 2


def _pose(x=0.0, y=0.0, z=0.0):
    return np.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0])


def _asset(tmp_path, relative, minimum):
    path = tmp_path / relative
    path.mkdir(parents=True)
    (path / "asset_size.json").write_text(
        __import__("json").dumps(
            {"min": minimum, "max": [0.1, 0.1, 0.1], "size": {"x": 0.2, "y": 0.2, "z": 0.2}}
        )
    )
    return str(path)


def test_task2_source_receipt_binds_actions_and_never_processed_actions(tmp_path):
    import h5py

    path = tmp_path / "source.hdf5"
    actions = np.arange(42, dtype=np.float32).reshape(3, 14)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.attrs["num_samples"] = 3
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("processed_actions", data=np.zeros((4, 14), dtype=np.float32))
        demo.create_dataset("states/rigid_object/mug/root_pose", data=np.zeros((4, 7)))
    receipt = _source_dataset_receipt(str(path), "demo_0", None)
    assert receipt["action_dataset"] == "actions"
    assert receipt["actions_sha256"] == _array_sha256(actions)
    assert receipt["processed_actions_shape"] == [4, 14]
    assert "not_executed" in receipt["processed_actions_role"]
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _source_dataset_receipt(str(path), "demo_0", "0" * 64)

    class TensorLike:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    assert _direct_actions_exact(list(actions), TensorLike(actions.copy()))
    changed = actions.copy()
    changed[1, 3] += 1
    assert not _direct_actions_exact(list(actions), TensorLike(changed))


def test_source_pick_prefix_is_exactly_aligned_and_physically_completed():
    keyframes = {
        "frames": {
            "right_pregrasp": {
                "sample_index": 393,
                "action_index": 392,
                "stage1": True,
                "stage2": False,
                "left_grasp": True,
            }
        }
    }
    assert _source_pick_prefix_steps(keyframes) == 393
    keyframes["frames"]["right_pregrasp"]["sample_index"] = 394
    with pytest.raises(ValueError, match="not aligned"):
        _source_pick_prefix_steps(keyframes)
    keyframes["frames"]["right_pregrasp"]["sample_index"] = 393
    keyframes["frames"]["right_pregrasp"]["stage1"] = False
    with pytest.raises(ValueError, match="completed Pick"):
        _source_pick_prefix_steps(keyframes)


def test_reusable_pick_boundary_requires_latch_and_only_left_assist():
    sample = {
        "stage1": True,
        "stage2": False,
        "grasp_assist_engaged": {"left": True, "right": False},
    }
    _require_reusable_pick_boundary(sample)
    for mutation in (
        {"stage1": False},
        {"stage2": True},
        {"grasp_assist_engaged": {"left": False, "right": False}},
        {"grasp_assist_engaged": {"left": True, "right": True}},
    ):
        changed = {**sample, **mutation}
        with pytest.raises(RuntimeError, match="completed Pick"):
            _require_reusable_pick_boundary(changed)


def test_handover_boundary_requires_latch_right_contact_and_assist_release():
    sample = {
        "step": 592,
        "stage2": True,
        "right_grasp": True,
        "grasp_assist_engaged": {"left": False, "right": True},
    }
    receipt = _handover_boundary_receipt(sample)
    assert receipt == {
        "stage": "handover",
        "checked_after_step": 592,
        "checks": {
            "stage2_latched": True,
            "right_contact_secure": True,
            "right_assist_secure": True,
            "left_assist_released": True,
        },
        "passed": True,
        "safe_to_continue": True,
    }
    for mutation, failed_check in (
        ({"stage2": False}, "stage2_latched"),
        ({"right_grasp": False}, "right_contact_secure"),
        (
            {"grasp_assist_engaged": {"left": False, "right": False}},
            "right_assist_secure",
        ),
        (
            {"grasp_assist_engaged": {"left": True, "right": True}},
            "left_assist_released",
        ),
    ):
        failed = _handover_boundary_receipt({**sample, **mutation})
        assert not failed["passed"]
        assert not failed["checks"][failed_check]
        assert failed["safe_to_continue"] is (failed_check == "stage2_latched")


def test_missing_stage2_alone_does_not_block_physically_secure_continuation():
    receipt = _handover_boundary_receipt(
        {
            "step": 499,
            "stage2": False,
            "right_grasp": True,
            "grasp_assist_engaged": {"left": False, "right": True},
        }
    )
    assert not receipt["passed"]
    assert receipt["safe_to_continue"]
    assert receipt["checks"]["stage2_latched"] is False


def test_handover_lift_guard_requires_entry_contact_then_assist_only():
    sample = {
        "step": 449,
        "stage1": True,
        "right_grasp": True,
        "grasp_assist_engaged": {"left": False, "right": True},
    }
    receipt = _handover_lift_guard_receipt(sample, phase="entry")
    assert receipt["passed"]
    assert receipt["checked_after_step"] == 449
    assert receipt["phase"] == "entry"
    assert receipt["diagnostics"] == {
        "right_contact_raw": True,
        "left_assist_engaged": False,
    }
    for mutation, failed_check in (
        ({"stage1": False}, "pick_latched"),
        ({"right_grasp": False}, "right_contact_secure"),
        (
            {"grasp_assist_engaged": {"left": False, "right": False}},
            "right_assist_secure",
        ),
    ):
        failed = _handover_lift_guard_receipt({**sample, **mutation}, phase="entry")
        assert not failed["passed"]
        assert not failed["checks"][failed_check]
    raw_flicker = _handover_lift_guard_receipt(
        {**sample, "right_grasp": False}, phase="lift_row"
    )
    assert raw_flicker["passed"]
    assert not raw_flicker["diagnostics"]["right_contact_raw"]
    assert "right_contact_secure" not in raw_flicker["checks"]


def test_task2_target_assets_are_same_index_and_use_source_state_template(tmp_path):
    root = tmp_path / "objects"
    mug = _asset(root, "MugHangable/mug_teacup_000001", [-0.1, -0.1, -0.03])
    tree = _asset(root, "ThreeLayerMugTree/mug_tree_000001", [-0.1, -0.1, -0.19])
    args = SimpleNamespace(
        target_mug_asset=mug,
        target_tree_asset=tree,
        target_dataset=None,
        source_dataset="source.hdf5",
        objects_root=str(root),
    )
    resolved, template = _resolve_target_assets(args, {})
    assert resolved == {"mug": mug, "mug_tree": tree}
    assert template == "source.hdf5"
    args.target_tree_asset = _asset(
        root, "ThreeLayerMugTree/mug_tree_000002", [-0.1, -0.1, -0.19]
    )
    with pytest.raises(ValueError, match="same index"):
        _resolve_target_assets(args, {})


def test_support_preserving_target_state_uses_authored_minimum(tmp_path):
    source_mug = _asset(tmp_path, "MugHangable/mug_teacup_000000", [-0.1, -0.1, -0.0575])
    target_mug = _asset(tmp_path, "MugHangable/mug_teacup_000001", [-0.1, -0.1, -0.0293])
    source_tree = _asset(tmp_path, "ThreeLayerMugTree/mug_tree_000000", [-0.1, -0.1, -0.1962])
    target_tree = _asset(tmp_path, "ThreeLayerMugTree/mug_tree_000001", [-0.1, -0.1, -0.1889])
    template = {
        "initial_state": {"rigid_object": {
            "mug": {"root_pose": np.asarray([_pose(0.7, 0.15, 0.8075)])},
            "mug_tree": {"root_pose": np.asarray([_pose(0.75, -0.3, 0.9462)])},
        }},
        "mug_pose": np.asarray([_pose(0.7, 0.15, 0.8075)]),
        "tree_pose": np.asarray([_pose(0.75, -0.3, 0.9462)]),
    }
    target, receipt = _support_preserving_target_state(
        template,
        {"mug": source_mug, "mug_tree": source_tree},
        {"mug": target_mug, "mug_tree": target_tree},
    )
    assert target["mug_pose"][0, :2] == pytest.approx([0.7, 0.15])
    assert target["mug_pose"][0, 2] == pytest.approx(0.7793)
    assert target["tree_pose"][0, 2] == pytest.approx(0.9389)
    assert receipt["mug"]["preserved_support_z_m"] == pytest.approx(0.75)
    assert template["mug_pose"][0, 2] == pytest.approx(0.8075)


def test_proven_control_defaults_are_fail_closed():
    args = SimpleNamespace(**PROVEN_CONTROL_DEFAULTS)
    _require_proven_control_defaults(args)
    args.damping = 0.046
    with pytest.raises(ValueError, match="changes the proven default"):
        _require_proven_control_defaults(args)


def test_terminal_stability_requires_thirty_current_physical_success_rows():
    row = {
        "task_success": True,
        "stage3": True,
        "hang_predicate_now": True,
        "left_grasp": False,
        "right_grasp": False,
    }
    assert _terminal_stability([dict(row) for _ in range(30)])["passed"]
    rows = [dict(row) for _ in range(30)]
    rows[-2]["hang_predicate_now"] = False
    assert not _terminal_stability(rows)["passed"]


def _durable_terminal_rows(*, stage2: bool = False):
    status = {
        "released": True,
        "stable": True,
        "contact_policy": True,
        "diagnostics": {
            "branch_engaged": True,
            "raw_conditions": {
                "insertion_support_candidate": True,
                "release_hang_candidate": True,
            },
            "completed_stage_latches": {
                "pick": True,
                "handover": stage2,
                "alignment": False,
                "insertion_and_support": False,
                "release_and_hang": False,
            },
        },
    }
    sample = {
        "stage1": True,
        "stage2": stage2,
        "stage3": False,
        "left_grasp": False,
        "right_grasp": False,
        "grasp_assist_engaged": {"left": False, "right": False},
    }
    return [dict(status) for _ in range(30)], [dict(sample) for _ in range(30)]


def test_genuine_terminal_hang_is_independently_accepted_with_stage2_false():
    statuses, samples = _durable_terminal_rows(stage2=False)
    receipt = _independent_terminal_hang_receipt(
        statuses,
        samples,
        {
            "explicit_env_reset_calls": 1,
            "initial_state_restores": 1,
            "resets_during_episode": 0,
        },
    )
    assert receipt["passed"]
    assert receipt["coded_stage_latches"] == {
        "stage1": True,
        "stage2": False,
        "stage3": False,
    }
    assert receipt["adapter_completed_stage_latches"]["handover"] is False


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        ({"released": False}, "adapter_released"),
        ({"stable": False}, "adapter_stable"),
        ({"contact_policy": False}, "bounded_contact_entire_rollout"),
        ({"diagnostics": {"branch_engaged": False}}, "branch_engaged"),
    ),
)
def test_independent_terminal_semantic_failure_still_rejects(mutation, failed_check):
    statuses, samples = _durable_terminal_rows()
    changed = dict(statuses[-1])
    if "diagnostics" in mutation:
        changed["diagnostics"] = {
            **changed["diagnostics"],
            **mutation["diagnostics"],
        }
    else:
        changed.update(mutation)
    statuses[-1] = changed
    receipt = _independent_terminal_hang_receipt(
        statuses,
        samples,
        {
            "explicit_env_reset_calls": 1,
            "initial_state_restores": 1,
            "resets_during_episode": 0,
        },
    )
    assert not receipt["passed"]
    assert not receipt["checks"][failed_check]


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

    deeper, _, _ = geometry_conditioned_hang_pose(
        _pose(1.0, 0.02, 1.03),
        _pose(),
        source_parts,
        target_parts,
        (branch(-1.0, 0.3, 0.8), source_branch),
        _pose(2.0, 3.0, 0.0),
        (branch(-1.0, 0.4, 0.9), target_branch),
        branch_support_fraction=0.35,
    )
    deeper_handle_world = compose_pose(deeper, target_parts.handle_hole_frame)
    deeper_support = target_branch.frame.copy()
    deeper_support[:3] = target_branch.inner_point + 0.35 * (
        target_branch.tip_point - target_branch.inner_point
    )
    deeper_branch_world = compose_pose(_pose(2.0, 3.0, 0.0), deeper_support)
    assert deeper_handle_world[:3] == pytest.approx(deeper_branch_world[:3])
    assert deeper_handle_world[3:] == pytest.approx(target_handle_world[3:])
    rolled, _, _ = geometry_conditioned_hang_pose(
        _pose(1.0, 0.02, 1.03),
        _pose(),
        source_parts,
        target_parts,
        (branch(-1.0, 0.3, 0.8), source_branch),
        _pose(2.0, 3.0, 0.0),
        (branch(-1.0, 0.4, 0.9), target_branch),
        branch_roll_offset_rad=np.pi / 6,
    )
    rolled_handle = compose_pose(rolled, target_parts.handle_hole_frame)
    assert rolled_handle[:3] == pytest.approx(target_handle_world[:3])
    assert quaternion_rotate(rolled_handle[3:], [0.0, 1.0, 0.0]) == pytest.approx(
        branch_tangent
    )
    assert quaternion_rotate(rolled_handle[3:], [1.0, 0.0, 0.0]) != pytest.approx(
        quaternion_rotate(target_handle_world[3:], [1.0, 0.0, 0.0])
    )
    alternate, _, alternate_target = geometry_conditioned_hang_pose(
        _pose(1.0, 0.02, 1.03),
        _pose(),
        source_parts,
        target_parts,
        (branch(-1.0, 0.3, 0.8), source_branch),
        _pose(2.0, 3.0, 0.0),
        (branch(-1.0, 0.4, 0.9), target_branch),
        target_branch_rank=0,
    )
    assert alternate_target is not target_branch
    alternate_handle = compose_pose(alternate, target_parts.handle_hole_frame)
    alternate_support = alternate_target.frame.copy()
    alternate_support[:3] = 0.5 * (
        alternate_target.inner_point + alternate_target.tip_point
    )
    assert alternate_handle[:3] == pytest.approx(
        compose_pose(_pose(2.0, 3.0, 0.0), alternate_support)[:3]
    )
    with pytest.raises(ValueError, match="outside the inferred branch set"):
        geometry_conditioned_hang_pose(
            _pose(), _pose(), source_parts, target_parts,
            (source_branch,), _pose(), (target_branch,), target_branch_rank=1,
        )
    with pytest.raises(ValueError, match="support fraction"):
        geometry_conditioned_hang_pose(
            _pose(), _pose(), source_parts, target_parts,
            (source_branch,), _pose(), (target_branch,),
            branch_support_fraction=0.24,
        )
    with pytest.raises(ValueError, match="roll offset"):
        geometry_conditioned_hang_pose(
            _pose(), _pose(), source_parts, target_parts,
            (source_branch,), _pose(), (target_branch,),
            branch_roll_offset_rad=np.pi,
        )


def test_branch_support_seating_changes_only_vertical_waypoint_translation():
    pose = _pose(0.7, -0.2, 0.96)
    seated = _branch_support_seated_pose(pose, 0.01)
    assert seated == pytest.approx([0.7, -0.2, 0.95, *pose[3:]])
    assert pose[2] == pytest.approx(0.96)
    with pytest.raises(ValueError, match="seat-down"):
        _branch_support_seated_pose(pose, 0.031)


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
        "all_stages_latched": True,
        "right_handover_observed": True,
        "stable_hang_window": True,
        "independent_terminal_hang": True,
        "handover_boundary_passed": True,
        "handover_safe_to_continue": True,
        "physics_device_cpu": True,
        "right_grasp_assist_engaged": False,
        "right_grasp_assist_released": True,
    }

    replay = _schema_aware_success_acceptance(checks, coded_skill=False)
    assert "right_grasp_assist_engaged" not in replay
    assert replay["right_handover_observed"] is True
    assert replay["stable_hang_window"] is True
    assert replay["coded_task_success"] is True
    assert replay["physics_device_cpu"] is True

    skill = _schema_aware_success_acceptance(checks, coded_skill=True)
    assert skill["right_grasp_assist_engaged"] is False
    assert skill["independent_terminal_hang"] is True
    assert skill["handover_safe_to_continue"] is True
    for diagnostic in (
        "coded_task_success",
        "all_stages_latched",
        "right_handover_observed",
        "stable_hang_window",
        "handover_boundary_passed",
    ):
        assert diagnostic not in skill


def test_observed_handover_reanchor_is_geometry_conditioned_for_tall_mugs():
    assert _requires_observed_handover_reanchor(
        SimpleNamespace(body_size=np.asarray([0.08, 0.081, 0.107]))
    )
    assert not _requires_observed_handover_reanchor(
        SimpleNamespace(body_size=np.asarray([0.088, 0.090, 0.077]))
    )
    assert _requires_observed_handover_reanchor(
        SimpleNamespace(body_size=np.asarray([0.088, 0.090, 0.077])),
        handle_frame_transfer=True,
    )
    with pytest.raises(ValueError, match="three positive"):
        _requires_observed_handover_reanchor(
            SimpleNamespace(body_size=np.asarray([0.08, -0.01, 0.10]))
        )


def test_handover_translation_offset_is_bounded_and_finite():
    assert _bounded_handover_offset([0.01, -0.02, 0.01]) == pytest.approx(
        [0.01, -0.02, 0.01]
    )
    with pytest.raises(ValueError, match="three finite"):
        _bounded_handover_offset([0.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="exceeds 4 cm"):
        _bounded_handover_offset([0.041, 0.0, 0.0])


def test_handover_local_pitch_rotates_only_receiver_orientation():
    pose = _pose(0.4, -0.1, 0.9)
    rotated = _handover_target_with_local_pitch(pose, np.pi / 4.0)
    np.testing.assert_allclose(rotated[:3], pose[:3], atol=0.0)
    np.testing.assert_allclose(
        quaternion_rotate(rotated[3:], [0.0, 0.0, 1.0]),
        [np.sqrt(0.5), 0.0, np.sqrt(0.5)],
        atol=1.0e-12,
    )
    with pytest.raises(ValueError, match="within 45 degrees"):
        _handover_target_with_local_pitch(pose, np.pi / 4.0 + 1.0e-6)


def test_handover_local_straddle_translates_only_along_oriented_closing_axis():
    pose = _handover_target_with_local_pitch(_pose(0.4, -0.1, 0.9), np.pi / 4.0)
    shifted = _handover_target_with_local_straddle(pose, -0.124)
    np.testing.assert_allclose(shifted[3:], pose[3:], atol=0.0)
    np.testing.assert_allclose(
        shifted[:3] - pose[:3],
        quaternion_rotate(pose[3:], [-0.124, 0.0, 0.0]),
        atol=1.0e-12,
    )
    with pytest.raises(ValueError, match="within 14 cm"):
        _handover_target_with_local_straddle(pose, -0.141)


def test_pitched_receiver_orients_clear_then_descends_open_before_close():
    clear = _pose(0.4, -0.1, 1.0)
    oriented_clear = _handover_target_with_local_pitch(clear, np.pi / 4.0)
    grasp = oriented_clear.copy()
    grasp[2] -= 0.08
    program = HangMugSkillProgram(_pose(), _pose(0.0, -0.5, 0.9))
    program.semantic_left_grasp(
        _pose(), _pose(), _pose(), approach_steps=1, close_steps=1, lift_steps=1
    )
    program.physical_handover(
        _pose(0.4, 0.1, 0.9),
        clear,
        grasp,
        _pose(0.4, 0.2, 0.9),
        right_orient_clear=oriented_clear,
        orient_steps=3,
        approach_steps=4,
        contact_settle_steps=4,
        close_steps=2,
        release_steps=2,
    )
    program.handle_to_branch_insert(
        grasp, grasp, grasp, transport_steps=1, approach_steps=1, insert_steps=1
    )
    trajectory = program.build()
    pre = trajectory.waypoint_steps["handover_pregrasp"]
    orient = trajectory.waypoint_steps["handover_orient_clear"]
    descend = trajectory.waypoint_steps["right_grasp_settle"]
    close = trajectory.waypoint_steps["right_grasp"]

    np.testing.assert_allclose(
        trajectory.right_poses[pre + 1 : orient + 1, :3],
        np.repeat(clear[None, :3], orient - pre, axis=0),
    )
    assert np.all(np.diff(trajectory.right_poses[orient + 1 : descend + 1, 2]) < 0)
    np.testing.assert_allclose(
        trajectory.right_poses[orient + 1 : descend + 1, 3:],
        np.repeat(oriented_clear[None, 3:], descend - orient, axis=0),
    )
    np.testing.assert_allclose(
        trajectory.grippers[: descend + 1, 1], -0.0475
    )
    assert trajectory.grippers[close, 1] == pytest.approx(0.0)

    nominal_mug = _pose(0.5, 0.0, 0.8)
    observed_mug = _pose(0.52, 0.0, 0.8)
    adjusted = reanchor_physical_handover(
        trajectory,
        nominal_mug,
        observed_mug,
        _pose(0.4, 0.1, 0.9),
        _pose(0.2, -0.2, 1.0),
    )
    np.testing.assert_allclose(
        adjusted.right_poses[orient, :3] - adjusted.right_poses[descend, :3],
        [0.0, 0.0, 0.08],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        adjusted.right_poses[orient, 3:], adjusted.right_poses[descend, 3:]
    )
    contact = compose_pose(inverse_pose(nominal_mug), grasp)
    readjusted = reanchor_right_grasp_from_observed_mug(
        adjusted, contact, _pose(0.53, 0.0, 0.8), adjusted.right_poses[pre]
    )
    np.testing.assert_allclose(
        readjusted.right_poses[orient, :3]
        - readjusted.right_poses[descend, :3],
        [0.0, 0.0, 0.08],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        readjusted.right_poses[orient, 3:], readjusted.right_poses[descend, 3:]
    )

    with pytest.raises(ValueError, match="open descent"):
        HangMugSkillProgram(_pose(), _pose()).physical_handover(
            _pose(), clear, grasp, _pose(), right_orient_clear=oriented_clear,
            orient_steps=2, approach_steps=2, close_steps=2, release_steps=2,
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
    insertion_start = trajectory.waypoint_steps["left_release"] + 1
    unload_end = trajectory.waypoint_steps["branch_unload"]
    insertion_grippers = trajectory.grippers[insertion_start : unload_end + 1]
    assert insertion_grippers[:, 0] == pytest.approx(-0.0475)
    assert insertion_grippers[:, 1] == pytest.approx(0.0)
    np.testing.assert_allclose(np.diff(insertion_grippers, axis=0), 0.0)
    release_start = unload_end + 1
    right_release = trajectory.grippers[release_start:, 1]
    assert np.all(np.diff(right_release) <= 0.0)
    assert np.flatnonzero(right_release < 0.0)[0] == 0
    transport_end = trajectory.waypoint_steps["tree_transport"]
    assert trajectory.left_poses[transport_end] == pytest.approx(left_observer)
    assert trajectory.left_poses[transport_end:] == pytest.approx(
        np.broadcast_to(left_observer, trajectory.left_poses[transport_end:].shape)
    )


def test_branch_receiver_orients_clear_then_approaches_without_rotation():
    transport = _pose(0.55, -0.1, 1.05)
    approach = _handover_target_with_local_pitch(
        _pose(0.7, -0.2, 0.98), np.pi / 4.0
    )
    orient = transport.copy()
    orient[3:] = approach[3:]
    insert = approach.copy()
    insert[:3] += [0.02, -0.05, -0.04]
    program = HangMugSkillProgram(_pose(), _pose(0.0, -0.5, 0.9))
    program.semantic_left_grasp(
        _pose(), _pose(), _pose(), approach_steps=1, close_steps=1, lift_steps=1
    )
    program.physical_handover(
        _pose(), _pose(), _pose(), _pose(),
        approach_steps=1, close_steps=1, release_steps=1,
    )
    program.handle_to_branch_insert(
        transport,
        approach,
        insert,
        transport_steps=2,
        right_orient_clear=orient,
        orient_steps=3,
        approach_steps=4,
        insert_steps=2,
    )
    program.release_and_support(
        insert, insert, unload_steps=1, release_steps=1, settle_steps=1
    )
    trajectory = program.build()
    transport_end = trajectory.waypoint_steps["tree_transport"]
    orient_end = trajectory.waypoint_steps["branch_orient_clear"]
    approach_end = trajectory.waypoint_steps["branch_approach"]
    np.testing.assert_allclose(
        trajectory.right_poses[transport_end + 1 : orient_end + 1, :3],
        np.repeat(transport[None, :3], orient_end - transport_end, axis=0),
    )
    np.testing.assert_allclose(
        trajectory.right_poses[orient_end + 1 : approach_end + 1, 3:],
        np.repeat(approach[None, 3:], approach_end - orient_end, axis=0),
    )
    assert np.all(
        trajectory.grippers[transport_end + 1 : approach_end + 1, 1] == 0.0
    )
    assert "branch_orient_clear" in _branch_reanchor_waypoints(trajectory)
    with pytest.raises(ValueError, match="target and steps must match"):
        HangMugSkillProgram(_pose(), _pose()).handle_to_branch_insert(
            transport, approach, insert, transport_steps=1, orient_steps=1,
            approach_steps=1, insert_steps=1,
        )

    legacy = HangMugSkillProgram(_pose(), _pose())
    legacy.handle_to_branch_insert(
        transport, approach, insert,
        transport_steps=1, approach_steps=1, insert_steps=1,
    )
    assert "branch_orient_clear" not in _branch_reanchor_waypoints(legacy.build())


def test_handover_reanchor_changes_only_handover_and_transport_entry():
    program = HangMugSkillProgram(_pose(), _pose(0.0, -1.0, 0.0))
    program.semantic_left_grasp(
        _pose(0.1), _pose(0.2), _pose(0.3),
        approach_steps=2, close_steps=2, lift_steps=2,
    )
    program.physical_handover(
        _pose(0.3), _pose(0.4, -0.2), _pose(0.4, -0.1), _pose(0.3, 0.2),
        approach_steps=2, close_steps=2, release_steps=2, confirm_steps=3,
    )
    program.handle_to_branch_insert(
        _pose(0.6, -0.1), _pose(0.7, -0.1), _pose(0.8, -0.1),
        transport_steps=2, approach_steps=2, insert_steps=2,
    )
    trajectory = program.build()
    original = trajectory.right_poses.copy()
    original_left = trajectory.left_poses.copy()
    observed_left = _pose(0.3, 0.05)
    adjusted = reanchor_physical_handover(
        trajectory, _pose(0.3), _pose(0.3, 0.1), observed_left, original[5]
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
    pregrasp_end = trajectory.waypoint_steps["handover_pregrasp"]
    release_end = trajectory.waypoint_steps["left_release"]
    confirm_end = trajectory.waypoint_steps["handover_confirm"]
    assert adjusted.left_poses[: lift_end + 1] == pytest.approx(
        original_left[: lift_end + 1]
    )
    assert adjusted.left_poses[lift_end + 1 : grasp_end + 1] == pytest.approx(
        np.repeat(observed_left[None], grasp_end - lift_end, axis=0)
    )
    expected_release = original_left[release_end].copy()
    expected_release[:3] += observed_left[:3] - original_left[pregrasp_end, :3]
    assert adjusted.left_poses[release_end] == pytest.approx(expected_release)
    assert adjusted.left_poses[release_end + 1 : confirm_end + 1] == pytest.approx(
        np.repeat(expected_release[None], confirm_end - release_end, axis=0)
    )
    assert adjusted.right_poses[grasp_end + 1 : confirm_end + 1] == pytest.approx(
        np.repeat(adjusted.right_poses[grasp_end][None], confirm_end - grasp_end, axis=0)
    )
    assert adjusted.grippers == pytest.approx(trajectory.grippers)


def test_pick_clearance_uses_measured_mug_height():
    initial = _pose(0.0, 0.0, 0.80)
    shallow = _pose(0.2, 0.1, 0.86)
    adjusted = ensure_pick_latch_clearance(shallow, initial, 0.07)
    assert adjusted[:2] == pytest.approx(shallow[:2])
    assert adjusted[2] == pytest.approx(0.92)
    with_margin = ensure_pick_latch_clearance(
        shallow, initial, 0.07, pick_threshold_m=0.06
    )
    assert with_margin[2] == pytest.approx(adjusted[2] + 0.01)


def test_pick_boundary_requires_latched_contact_backed_left_hold():
    sample = {
        "step": 219,
        "stage1": True,
        "left_grasp": True,
        "grasp_assist_engaged": {"left": True, "right": False},
    }
    assert _pick_boundary_receipt(sample)["passed"] is True
    for mutation in (
        {"stage1": False},
        {"left_grasp": False},
        {"grasp_assist_engaged": {"left": False, "right": False}},
        {"grasp_assist_engaged": {"left": True, "right": True}},
    ):
        receipt = _pick_boundary_receipt({**sample, **mutation})
        assert receipt["passed"] is False
        assert receipt["safe_to_continue"] is False


def test_branch_transport_reanchors_observed_right_contact():
    program = HangMugSkillProgram(_pose(z=1), _pose(z=1))
    program.semantic_left_grasp(
        _pose(0.1, z=1), _pose(0.2, z=1), _pose(0.3, z=1),
        approach_steps=2, close_steps=2, lift_steps=2,
    )
    program.physical_handover(
        _pose(0.3, z=1), _pose(0.3, -0.1, 1), _pose(0.3, -0.2, 1),
        _pose(0.3, 0.1, 1), approach_steps=2, close_steps=2, release_steps=2,
        confirm_steps=3,
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
        trajectory, nominal_contact, observed_mug, observed_right,
        completed_waypoint="handover_confirm",
    )
    start = trajectory.waypoint_steps["handover_confirm"] + 1
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
    release_end = trajectory.waypoint_steps["left_release"]
    confirm_end = trajectory.waypoint_steps["handover_confirm"]
    assert adjusted.right_poses[release_end + 1 : confirm_end + 1] == pytest.approx(
        np.repeat(trajectory.right_poses[release_end][None], 3, axis=0)
    )
    with pytest.raises(ValueError, match="before handover confirmation"):
        reanchor_branch_transport_contact(
            trajectory, nominal_contact, observed_mug, observed_right,
            completed_waypoint="left_release",
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


def test_handover_contact_settle_keeps_receiver_open_until_pose_is_reached():
    program = HangMugSkillProgram(_pose(z=1), _pose(z=1))
    program.semantic_left_grasp(
        _pose(z=1), _pose(z=1), _pose(z=1),
        approach_steps=1, close_steps=1, lift_steps=1,
    )
    grasp = _pose(0.3, -0.2, 1)
    release_left = _pose(0.2, z=1)
    lift_right = _pose(0.3, -0.2, 1.055)
    program.physical_handover(
        _pose(z=1), _pose(0.3, -0.1, 1), grasp, release_left,
        receiver_lift=lift_right,
        receiver_lift_steps=3,
        approach_steps=2, contact_settle_steps=3, close_steps=2,
        release_steps=2, confirm_steps=3,
    )
    trajectory = program.build()
    legacy = HangMugSkillProgram(_pose(z=1), _pose(z=1))
    legacy.semantic_left_grasp(
        _pose(z=1), _pose(z=1), _pose(z=1),
        approach_steps=1, close_steps=1, lift_steps=1,
    )
    legacy.physical_handover(
        _pose(z=1), _pose(0.3, -0.1, 1), grasp, release_left,
        approach_steps=2, contact_settle_steps=3, close_steps=2,
        release_steps=2, confirm_steps=3,
    )
    legacy_trajectory = legacy.build()
    settle_end = trajectory.waypoint_steps["right_grasp_settle"]
    grasp_end = trajectory.waypoint_steps["right_grasp"]
    lift_end = trajectory.waypoint_steps["handover_receiver_lift"]
    release_end = trajectory.waypoint_steps["left_release"]
    confirm_end = trajectory.waypoint_steps["handover_confirm"]
    assert trajectory.right_poses[settle_end] == pytest.approx(grasp)
    assert trajectory.right_poses[settle_end + 1 : grasp_end + 1] == pytest.approx(
        np.repeat(grasp[None], grasp_end - settle_end, axis=0)
    )
    assert trajectory.grippers[settle_end, 1] == pytest.approx(-0.0475)
    assert trajectory.grippers[grasp_end, 1] == pytest.approx(0.0)
    acquire = slice(grasp_end + 1, release_end + 1)
    legacy_release = slice(
        legacy_trajectory.waypoint_steps["right_grasp"] + 1,
        legacy_trajectory.waypoint_steps["left_release"] + 1,
    )
    assert trajectory.left_poses[acquire] == pytest.approx(
        legacy_trajectory.left_poses[legacy_release]
    )
    assert trajectory.right_poses[acquire] == pytest.approx(
        legacy_trajectory.right_poses[legacy_release]
    )
    assert trajectory.grippers[acquire] == pytest.approx(
        legacy_trajectory.grippers[legacy_release]
    )
    receiver_lift = slice(release_end + 1, lift_end + 1)
    assert trajectory.left_poses[receiver_lift] == pytest.approx(
        np.repeat(release_left[None], 3, axis=0)
    )
    assert np.all(np.diff(trajectory.right_poses[receiver_lift, 2]) > 0)
    assert trajectory.right_poses[lift_end] == pytest.approx(lift_right)
    assert trajectory.grippers[receiver_lift] == pytest.approx(
        np.repeat([[-0.0475, 0.0]], 3, axis=0)
    )
    assert trajectory.right_poses[lift_end + 1 : confirm_end + 1] == pytest.approx(
        np.repeat(lift_right[None], confirm_end - lift_end, axis=0)
    )
    assert trajectory.grippers[lift_end + 1, 0] < 0.0
    assert confirm_end - lift_end == 3
    assert trajectory.left_poses[release_end + 1 : confirm_end + 1] == pytest.approx(
        np.repeat(
            trajectory.left_poses[release_end][None],
            confirm_end - release_end,
            axis=0,
        )
    )
    assert trajectory.right_poses[lift_end + 1 : confirm_end + 1] == pytest.approx(
        np.repeat(trajectory.right_poses[lift_end][None], 3, axis=0)
    )
    assert trajectory.grippers[confirm_end] == pytest.approx([-0.0475, 0.0])

    observed_mug = _pose(0.4, 0.2, 0.8)
    corrected = compose_pose(observed_mug, _pose(0.05, -0.02, 0.03))
    adjusted = reanchor_right_grasp_from_observed_mug(
        trajectory, _pose(0.05, -0.02, 0.03), observed_mug, _pose(0.4, 0.0, 1)
    )
    assert adjusted.right_poses[settle_end] == pytest.approx(corrected)
    assert adjusted.right_poses[settle_end + 1 : grasp_end + 1] == pytest.approx(
        np.repeat(corrected[None], grasp_end - settle_end, axis=0)
    )
    expected_lift = corrected.copy()
    expected_lift[2] += 0.055
    assert adjusted.right_poses[lift_end] == pytest.approx(expected_lift)
    assert adjusted.right_poses[lift_end + 1 : confirm_end + 1] == pytest.approx(
        np.repeat(expected_lift[None], confirm_end - lift_end, axis=0)
    )
    class FakeActions:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def __array__(self, dtype=None):
            return np.asarray(self.value, dtype=dtype)

    source = {"actions": FakeActions(np.arange(20 * 14).reshape(20, 14))}
    indices = {
        "left_pregrasp": 1,
        "left_grasp": 2,
        "left_lift": 3,
        "right_pregrasp": 4,
        "dual_grasp": 5,
        "handover": 6,
        "tree_approach": 7,
        "inserted_held": 8,
        "release": 9,
        "stable_settle": 10,
    }
    nominal = _sparse_joint_nominal(
        source, trajectory, {"semantic_indices": indices}
    )
    assert nominal.shape == (trajectory.steps, 14)
    assert nominal[settle_end] == pytest.approx(source["actions"].value[5])

    suffix = _trajectory_after(trajectory, "left_lift")
    assert suffix.steps == trajectory.steps - (
        trajectory.waypoint_steps["left_lift"] + 1
    )
    assert "left_lift" not in suffix.waypoint_steps
    assert suffix.waypoint_steps["handover_pregrasp"] == 1
    assert suffix.grippers[0] == pytest.approx([0.0, -0.0475])
    continued = _sparse_joint_nominal(
        source,
        suffix,
        {"semantic_indices": indices},
        initial_action_index=7,
    )
    assert continued[0] == pytest.approx(
        0.5 * (source["actions"].value[7] + source["actions"].value[4])
    )


def test_contact_acquire_moves_held_mug_by_live_residual_before_release():
    program = HangMugSkillProgram(_pose(), _pose())
    program.semantic_left_grasp(
        _pose(), _pose(), _pose(), approach_steps=1, close_steps=1, lift_steps=1
    )
    program.physical_handover(
        _pose(0.4, 0.1, 0.8),
        _pose(0.3, -0.1, 0.9),
        _pose(0.4, -0.1, 0.9),
        _pose(0.4, 0.2, 0.8),
        approach_steps=2,
        close_steps=2,
        contact_acquire_steps=4,
        release_steps=3,
        confirm_steps=2,
    )
    trajectory = program.build()
    mug = _pose(0.5, 0.0, 0.8)
    nominal_contact = _pose(-0.1, -0.02, 0.1)
    desired_right = compose_pose(mug, nominal_contact)
    # Preserved Pair 6 Attempt 12 residual: 30.212 mm, safely below 35 mm.
    translation = np.asarray([0.0285902692, 0.0069545818, -0.0068566126])
    observed_right = desired_right.copy()
    observed_right[:3] += translation
    observed_left = _pose(0.4, 0.1, 0.8)

    adjusted, receipt = reanchor_handover_contact_acquire(
        trajectory, nominal_contact, mug, observed_left, observed_right
    )

    grasp_end = adjusted.waypoint_steps["right_grasp"]
    acquire_end = adjusted.waypoint_steps["handover_contact_acquire"]
    release_end = adjusted.waypoint_steps["left_release"]
    acquire_x = adjusted.left_poses[grasp_end + 1 : acquire_end + 1, 0]
    assert np.all(np.diff(acquire_x) > 0)
    np.testing.assert_allclose(
        adjusted.left_poses[acquire_end, :3],
        observed_left[:3] + translation,
    )
    np.testing.assert_allclose(
        adjusted.right_poses[grasp_end + 1 : release_end + 1],
        np.repeat(observed_right[None], release_end - grasp_end, axis=0),
    )
    assert adjusted.grippers[acquire_end] == pytest.approx([0.0, 0.0])
    assert adjusted.grippers[release_end, 0] < 0.0
    assert receipt["world_translation_m"] == pytest.approx(translation)
    assert receipt["translation_norm_m"] == pytest.approx(np.linalg.norm(translation))
    assert receipt["maximum_translation_m"] == pytest.approx(0.04)
    assert receipt["rotation_error_rad"] == pytest.approx(0.0)
    assert receipt["orientation_unchanged"] is True

    class Actions:
        def __init__(self):
            self.value = np.arange(20 * 14).reshape(20, 14)

        def detach(self):
            return self

        def cpu(self):
            return self

        def __array__(self, dtype=None):
            return np.asarray(self.value, dtype=dtype)

    indices = {
        "left_pregrasp": 1, "left_grasp": 2, "left_lift": 3,
        "right_pregrasp": 4, "dual_grasp": 5, "handover": 6,
        "tree_approach": 7, "inserted_held": 8, "release": 9,
        "stable_settle": 10,
    }
    actions = Actions()
    nominal = _sparse_joint_nominal(
        {"actions": actions}, trajectory, {"semantic_indices": indices}
    )
    assert nominal[acquire_end] == pytest.approx(actions.value[5])


def test_contact_acquire_rejects_unbounded_live_residual():
    program = HangMugSkillProgram(_pose(), _pose())
    program.physical_handover(
        _pose(), _pose(), _pose(), _pose(),
        approach_steps=1,
        close_steps=1,
        contact_acquire_steps=2,
        release_steps=1,
    )
    _, receipt = reanchor_handover_contact_acquire(
        program.build(), _pose(), _pose(), _pose(), _pose(0.035514)
    )
    assert receipt["translation_norm_m"] == pytest.approx(0.035514)
    with pytest.raises(RuntimeError, match="exceeds"):
        reanchor_handover_contact_acquire(
            program.build(), _pose(), _pose(), _pose(), _pose(0.040001)
        )


def test_contact_acquire_guard_requires_receiver_contact_only_at_completion():
    sample = {
        "step": 9,
        "stage1": True,
        "left_grasp": True,
        "right_grasp": False,
        "grasp_assist_engaged": {"left": True, "right": False},
    }
    assert _handover_contact_acquire_guard_receipt(sample, phase="entry")["passed"]
    assert _handover_contact_acquire_guard_receipt(sample, phase="row")["passed"]
    assert not _handover_contact_acquire_guard_receipt(
        sample, phase="completion"
    )["passed"]
    sample["right_grasp"] = True
    sample["grasp_assist_engaged"]["right"] = True
    sample["grasp_assist_engaged"]["left"] = False
    assert _handover_contact_acquire_guard_receipt(sample, phase="row")["passed"]
    assert _handover_contact_acquire_guard_receipt(
        sample, phase="completion"
    )["passed"]
    sample["right_grasp"] = False
    sample["grasp_assist_engaged"]["right"] = False
    assert not _handover_contact_acquire_guard_receipt(sample, phase="row")["passed"]


def test_receiver_lift_requires_target_and_follows_release():
    program = HangMugSkillProgram(_pose(), _pose())
    with pytest.raises(ValueError, match="requires a right-wrist target"):
        program.physical_handover(
            _pose(), _pose(), _pose(), _pose(),
            receiver_lift_steps=2,
            approach_steps=1, close_steps=1, release_steps=1,
        )
