import numpy as np
import pytest

from judo_isaaclab.hang_mug_clean_insertion import (
    apply_branch_radial_clearance,
)
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
