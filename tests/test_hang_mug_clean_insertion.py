import numpy as np
import pytest

import judo_isaaclab.hang_mug_clean_insertion as clean_insertion
from judo_isaaclab.hang_mug_clean_insertion import (
    apply_branch_radial_clearance,
    repair_compensated_insertion_path,
)
from judo_isaaclab.put_marker import SkillTrajectory
from judo_isaaclab.semantic_parts import BranchPart


IDENTITY = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def test_branch_radial_clearance_is_bounded_and_preserves_final_pose():
    path = np.repeat(IDENTITY[None], 40, axis=0)
    path[:, :3] = [0.5, 0.02, 0.0]
    branch = BranchPart(
        frame=IDENTITY,
        inner_point=np.asarray([0.0, 0.0, 0.0]),
        tip_point=np.asarray([1.0, 0.0, 0.0]),
        tangent=np.asarray([1.0, 0.0, 0.0]),
        length_m=1.0,
        radius_m=0.01,
        normalized_height=0.5,
        azimuth_rad=0.0,
    )

    corrected, receipt = apply_branch_radial_clearance(
        path,
        tree_pose=IDENTITY,
        mug_body_frame=IDENTITY,
        mug_body_size=[0.08, 0.08, 0.10],
        target_branch=branch,
        collision_steps=[18, 19],
    )

    assert receipt["method"] == "single_pass_branch_radial_clearance"
    assert receipt["maximum_displacement_m"] == pytest.approx(0.01)
    assert receipt["preserves_final_pose"] is True
    np.testing.assert_allclose(corrected[0], path[0])
    np.testing.assert_allclose(corrected[-1], path[-1])
    np.testing.assert_allclose(corrected[18:20, 1], 0.03)
    np.testing.assert_allclose(corrected[:, 0], path[:, 0])
    np.testing.assert_allclose(corrected[:, 2], path[:, 2])


def test_branch_radial_clearance_rejects_empty_collision_window():
    branch = BranchPart(
        frame=IDENTITY,
        inner_point=np.asarray([0.0, 0.0, 0.0]),
        tip_point=np.asarray([1.0, 0.0, 0.0]),
        tangent=np.asarray([1.0, 0.0, 0.0]),
        length_m=1.0,
        radius_m=0.01,
        normalized_height=0.5,
        azimuth_rad=0.0,
    )
    with pytest.raises(ValueError, match="collision_steps"):
        apply_branch_radial_clearance(
            np.repeat(IDENTITY[None], 3, axis=0),
            tree_pose=IDENTITY,
            mug_body_frame=IDENTITY,
            mug_body_size=[0.08, 0.08, 0.10],
            target_branch=branch,
            collision_steps=[],
        )


def test_branch_radial_clearance_tapers_to_exact_terminal_support():
    path = np.repeat(IDENTITY[None], 40, axis=0)
    path[:, :3] = [0.5, 0.02, 0.0]
    branch = BranchPart(
        frame=IDENTITY,
        inner_point=np.asarray([0.0, 0.0, 0.0]),
        tip_point=np.asarray([1.0, 0.0, 0.0]),
        tangent=np.asarray([1.0, 0.0, 0.0]),
        length_m=1.0,
        radius_m=0.01,
        normalized_height=0.5,
        azimuth_rad=0.0,
    )

    corrected, receipt = apply_branch_radial_clearance(
        path,
        tree_pose=IDENTITY,
        mug_body_frame=IDENTITY,
        mug_body_size=[0.08, 0.08, 0.10],
        target_branch=branch,
        collision_steps=[35, 36],
    )

    assert receipt["correction_window"][-1] == len(path) - 1
    assert receipt["preserves_final_pose"] is True
    np.testing.assert_allclose(corrected[-1], path[-1], atol=1.0e-12)


def test_branch_radial_clearance_never_routes_downward():
    path = np.repeat(IDENTITY[None], 40, axis=0)
    path[:, :3] = [0.5, 0.01, -0.02]
    branch = BranchPart(
        frame=IDENTITY,
        inner_point=np.asarray([0.0, 0.0, 0.0]),
        tip_point=np.asarray([1.0, 0.0, 0.0]),
        tangent=np.asarray([1.0, 0.0, 0.0]),
        length_m=1.0,
        radius_m=0.01,
        normalized_height=0.5,
        azimuth_rad=0.0,
    )

    corrected, receipt = apply_branch_radial_clearance(
        path,
        tree_pose=IDENTITY,
        mug_body_frame=IDENTITY,
        mug_body_size=[0.06, 0.08, 0.10],
        target_branch=branch,
        collision_steps=[18, 19],
    )

    assert receipt["gravity_safe_horizontal_projection"] is True
    assert receipt["maximum_displacement_m"] == pytest.approx(0.04)
    assert receipt["raw_direction_world"][2] < 0.0
    assert receipt["direction_world"][2] == pytest.approx(0.0)
    assert np.min(corrected[:, 2] - path[:, 2]) == pytest.approx(0.0)
    np.testing.assert_allclose(corrected[-1], path[-1], atol=1.0e-12)


def test_compensated_insertion_is_rescreened_before_execution(monkeypatch):
    trajectory = SkillTrajectory(
        left_poses=np.repeat(IDENTITY[None], 4, axis=0),
        right_poses=np.repeat(IDENTITY[None], 4, axis=0),
        grippers=np.zeros((4, 2)),
        stage_names=("insert",) * 4,
        waypoint_steps={"branch_insert": 2},
    )
    receipts = iter(
        (
            {"passed": False, "collision_count": 1, "collision_steps": [1]},
            {"passed": True, "collision_count": 0, "collision_steps": []},
        )
    )
    monkeypatch.setattr(
        clean_insertion,
        "exact_body_collision_receipt",
        lambda *args, **kwargs: next(receipts),
    )

    correction_args = {}

    def correct(path, **kwargs):
        correction_args.update(kwargs)
        corrected = path.copy()
        corrected[0, 0] += 0.02
        return corrected, {
            "method": "test_correction",
            "source_collision_steps": [0],
            "correction_window": [0, 1],
        }

    monkeypatch.setattr(clean_insertion, "apply_branch_radial_clearance", correct)
    corrected, receipt = repair_compensated_insertion_path(
        trajectory,
        completed_step=0,
        right_contact_in_mug=IDENTITY,
        tree_pose=IDENTITY,
        mug_body_frame=IDENTITY,
        mug_body_size=[0.08, 0.08, 0.10],
        target_branch=object(),
        target_assets={"mug": "mug", "mug_tree": "tree"},
    )

    assert receipt["passed"] is True
    assert receipt["pre_correction_collision_count"] == 1
    assert receipt["observation_compensated_path"] is True
    assert receipt["geometry_correction"]["future_start_step"] == 1
    assert receipt["geometry_correction"]["source_collision_steps"] == [1]
    assert receipt["geometry_correction"]["correction_window"] == [1, 2]
    assert correction_args["collision_steps"] == [0]
    np.testing.assert_allclose(corrected.right_poses[0], trajectory.right_poses[0])
    assert corrected.right_poses[1, 0] == pytest.approx(0.02)
    np.testing.assert_allclose(corrected.right_poses[3], trajectory.right_poses[3])


def test_held_suffix_is_rescreened_without_rewriting_completed_steps(monkeypatch):
    trajectory = SkillTrajectory(
        left_poses=np.repeat(IDENTITY[None], 6, axis=0),
        right_poses=np.repeat(IDENTITY[None], 6, axis=0),
        grippers=np.zeros((6, 2)),
        stage_names=("insert",) * 6,
        waypoint_steps={"branch_insert": 2, "branch_unload": 5},
    )
    receipts = iter(
        (
            {"passed": True, "collision_count": 0, "collision_steps": []},
            {"passed": False, "collision_count": 1, "collision_steps": [4]},
            {"passed": True, "collision_count": 0, "collision_steps": []},
        )
    )
    monkeypatch.setattr(
        clean_insertion,
        "exact_body_collision_receipt",
        lambda *args, **kwargs: next(receipts),
    )

    def correct(path, **kwargs):
        corrected = path.copy()
        corrected[0, 0] += 0.02
        return corrected, {
            "method": "test_correction",
            "source_collision_steps": [0],
            "correction_window": [0, 1],
        }

    monkeypatch.setattr(clean_insertion, "apply_branch_radial_clearance", correct)
    corrected, receipt = repair_compensated_insertion_path(
        trajectory,
        completed_step=3,
        right_contact_in_mug=IDENTITY,
        tree_pose=IDENTITY,
        mug_body_frame=IDENTITY,
        mug_body_size=[0.08, 0.08, 0.10],
        target_branch=object(),
        target_assets={"mug": "mug", "mug_tree": "tree"},
        executed_mug_poses=np.repeat(IDENTITY[None], 4, axis=0),
    )

    assert receipt["passed"] is True
    assert receipt["audit_end_step"] == 5
    assert receipt["executed_prefix"]["passed"] is True
    assert receipt["geometry_correction"]["future_start_step"] == 4
    np.testing.assert_allclose(
        corrected.right_poses[:4], trajectory.right_poses[:4]
    )
    assert corrected.right_poses[4, 0] == pytest.approx(0.02)
    np.testing.assert_allclose(corrected.right_poses[5], trajectory.right_poses[5])


def test_held_suffix_rejects_observed_collision_before_future_repair(monkeypatch):
    trajectory = SkillTrajectory(
        left_poses=np.repeat(IDENTITY[None], 6, axis=0),
        right_poses=np.repeat(IDENTITY[None], 6, axis=0),
        grippers=np.zeros((6, 2)),
        stage_names=("insert",) * 6,
        waypoint_steps={"branch_insert": 2, "branch_unload": 5},
    )
    monkeypatch.setattr(
        clean_insertion,
        "exact_body_collision_receipt",
        lambda *args, **kwargs: {
            "passed": False,
            "collision_count": 1,
            "collision_steps": [2],
        },
    )
    with pytest.raises(RuntimeError, match="collided in execution"):
        repair_compensated_insertion_path(
            trajectory,
            completed_step=3,
            right_contact_in_mug=IDENTITY,
            tree_pose=IDENTITY,
            mug_body_frame=IDENTITY,
            mug_body_size=[0.08, 0.08, 0.10],
            target_branch=object(),
            target_assets={"mug": "mug", "mug_tree": "tree"},
            executed_mug_poses=np.repeat(IDENTITY[None], 4, axis=0),
        )
