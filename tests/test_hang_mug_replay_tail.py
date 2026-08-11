import numpy as np
import pytest

from judo_isaaclab.hang_mug_replay_tail import (
    REPLAY_HANG_UNLOAD_STEPS,
    apply_branch_tip_support_clearance,
    build_replay_hang_tail,
    repeated_joint_nominal,
    replay_prefix_steps,
    replace_replay_hang_tail_path,
    replay_tail_ready,
    replay_tail_steps,
)
from judo_isaaclab.put_marker import compose_pose
from judo_isaaclab.semantic_parts import BranchPart, MugParts


def _pose(x=0.0, y=0.0, z=0.0):
    return np.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0])


def _parts(scale=1.0):
    return MugParts(
        body_frame=_pose(),
        body_size=np.asarray([0.2, 0.2, 0.3]) * scale,
        handle_hole_frame=_pose(0.1 * scale),
        handle_outer_size=np.asarray([0.08, 0.04, 0.06]) * scale,
        handle_thickness_m=0.01 * scale,
        handle_axis=0,
        handle_sign=1,
    )


def _branch(x, z, length):
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


def test_replay_prefix_ends_before_source_release_and_requires_live_handover():
    keyframes = {"frames": {"handover": {"action_index": 358}}}
    assert replay_prefix_steps(keyframes) == 359
    assert replay_prefix_steps(keyframes, latch_grace_steps=8) == 367
    with pytest.raises(ValueError, match="latch_grace_steps"):
        replay_prefix_steps(keyframes, latch_grace_steps=-1)
    assert replay_tail_ready(
        {"stage2": True, "right_grasp": True, "left_grasp": False}
    )
    assert not replay_tail_ready(
        {"stage2": True, "right_grasp": False, "left_grasp": False}
    )
    assert not replay_tail_ready(
        {"stage2": True, "right_grasp": True, "left_grasp": True}
    )


def test_joint_nominal_holds_last_byte_identical_prefix_action():
    actions = np.arange(42, dtype=np.float64).reshape(3, 14)
    nominal = repeated_joint_nominal(actions, prefix_steps=2, tail_steps=4)
    assert nominal.shape == (4, 14)
    assert np.array_equal(nominal, np.repeat(actions[1:2], 4, axis=0))


def test_branch_tip_support_clearance_retains_radius_beyond_handle():
    poses = np.repeat(_pose()[None], 11, axis=0)
    branch = _branch(1.0, 1.0, 0.08)

    corrected, receipt = apply_branch_tip_support_clearance(
        poses,
        tree_pose=_pose(),
        target_branch=branch,
        handle_axis_span_m=0.04,
        approach_step=5,
    )

    assert receipt["method"] == "geometry_bounded_branch_tip_support_clearance"
    assert receipt["displacement_m"] == pytest.approx(0.015)
    assert receipt["branch_tip_engagement_margin_m"] == pytest.approx(0.005)
    np.testing.assert_allclose(corrected[:6], poses[:6])
    assert corrected[-1, 0] == pytest.approx(0.015)
    assert receipt["terminal_pose_changed"] is True


def test_hang_tail_preserves_observed_contact_and_targets_measured_branch():
    source_branch = _branch(1.0, 1.0, 1.0)
    target_branch = _branch(1.5, 1.5, 1.5)
    source_parts = _parts()
    target_parts = _parts(1.2)
    observed_mug = _pose(0.5, -0.1, 1.0)
    observed_right = _pose(0.58, -0.12, 1.05)
    keyframes = {
        "frames": {
            "stable_settle": {
                "mug_pose": _pose(1.0, 0.02, 1.03).tolist(),
                "tree_pose": _pose().tolist(),
            }
        },
        "clean_insertion_path": {
            "mug_poses": [
                _pose(0.6, -0.08, 1.1).tolist(),
                _pose(0.8, -0.02, 1.05).tolist(),
                _pose(1.0, 0.02, 1.03).tolist(),
            ]
        },
    }

    tail = build_replay_hang_tail(
        keyframes=keyframes,
        source_parts=source_parts,
        target_parts=target_parts,
        source_branches=(_branch(-1.0, 0.3, 0.8), source_branch),
        target_tree_pose=_pose(2.0, 3.0, 0.0),
        target_branches=(_branch(-1.0, 0.4, 0.9), target_branch),
        observed_mug_pose=observed_mug,
        left_eef_pose=_pose(0.3, 0.2, 1.1),
        right_eef_pose=observed_right,
    )

    assert tail.trajectory.steps == replay_tail_steps(keyframes)
    reconstructed_right = compose_pose(observed_mug, tail.right_contact_in_mug)
    assert reconstructed_right == pytest.approx(observed_right)
    assert tail.intended_final_mug_pose == pytest.approx(
        tail.planned_mug_poses[-1]
    )
    assert tail.support_alignment["terminal_pose_exact"] is True
    target_handle = compose_pose(
        tail.intended_final_mug_pose, target_parts.handle_hole_frame
    )
    target_midpoint = 0.5 * (target_branch.inner_point + target_branch.tip_point)
    target_support = compose_pose(_pose(2.0, 3.0, 0.0), _pose(*target_midpoint))
    assert target_handle[:3] == pytest.approx(target_support[:3])
    assert tail.trajectory.stage_names[0] == "source_relationship_transport"
    assert tail.trajectory.stage_names[-1] == "stable_settle"
    assert tail.trajectory.grippers[0] == pytest.approx([-0.0475, 0.0])
    branch_unload = tail.trajectory.waypoint_steps["branch_unload"]
    assert (
        branch_unload - tail.trajectory.waypoint_steps["branch_insert"]
        == REPLAY_HANG_UNLOAD_STEPS
    )
    assert np.all(tail.trajectory.grippers[: branch_unload + 1, 1] == 0.0)

    corrected = tail.planned_mug_poses.copy()
    corrected[1, 2] += 0.01
    replaced = replace_replay_hang_tail_path(tail, corrected)
    assert replaced.trajectory.steps == tail.trajectory.steps
    assert replaced.intended_final_mug_pose == pytest.approx(corrected[-1])
    assert replaced.planned_mug_poses == pytest.approx(corrected)


def test_nearest_eef_branch_policy_is_explicit_and_deterministic():
    source_branch = _branch(1.0, 1.0, 1.0)
    corresponding = _branch(1.5, 1.5, 1.0)
    closer = _branch(0.5, 0.5, 1.0)
    keyframes = {
        "frames": {
            "stable_settle": {
                "mug_pose": _pose(1.0, 0.0, 1.0).tolist(),
                "tree_pose": _pose().tolist(),
            }
        },
        "clean_insertion_path": {
            "mug_poses": [
                _pose(0.8, 0.0, 1.1).tolist(),
                _pose(1.0, 0.0, 1.0).tolist(),
            ]
        },
    }

    tail = build_replay_hang_tail(
        keyframes=keyframes,
        source_parts=_parts(),
        target_parts=_parts(),
        source_branches=(source_branch,),
        target_tree_pose=_pose(),
        target_branches=(corresponding, closer),
        observed_mug_pose=_pose(),
        left_eef_pose=_pose(),
        right_eef_pose=_pose(0.45, 0.0, 0.5),
        target_branch_policy="nearest_eef",
    )

    selection = tail.support_alignment["target_branch_selection"]
    assert tail.target_branch is closer
    assert selection["policy"] == "nearest_eef"
    assert selection["changed_from_source_correspondence"] is True
    assert selection["selected_candidate_index"] == 1


def test_nearest_eef_rejects_task_height_infeasible_branch():
    source_branch = _branch(1.0, 1.0, 1.0)
    feasible = _branch(1.5, 1.5, 1.0)
    closer_but_too_low = _branch(0.5, 0.5, 1.0)
    keyframes = {
        "frames": {
            "stable_settle": {
                "mug_pose": _pose(1.0, 0.0, 1.0).tolist(),
                "tree_pose": _pose().tolist(),
            }
        },
        "clean_insertion_path": {
            "mug_poses": [
                _pose(0.8, 0.0, 1.1).tolist(),
                _pose(1.0, 0.0, 1.0).tolist(),
            ]
        },
    }

    tail = build_replay_hang_tail(
        keyframes=keyframes,
        source_parts=_parts(),
        target_parts=_parts(),
        source_branches=(source_branch,),
        target_tree_pose=_pose(),
        target_branches=(feasible, closer_but_too_low),
        observed_mug_pose=_pose(),
        left_eef_pose=_pose(),
        right_eef_pose=_pose(0.45, 0.0, 0.5),
        target_branch_policy="nearest_eef",
        minimum_supported_mug_z=0.8,
    )

    selection = tail.support_alignment["target_branch_selection"]
    assert tail.target_branch is feasible
    assert selection["selected_candidate_index"] == 0
    assert selection["candidate_task_height_feasible"] == [True, False]
    assert selection["minimum_supported_mug_z_m"] == pytest.approx(0.8)


def test_source_like_tail_releases_immediately_after_clean_insert():
    keyframes = {
        "frames": {
            "stable_settle": {
                "mug_pose": _pose(1.0, 0.0, 1.0).tolist(),
                "tree_pose": _pose().tolist(),
            }
        },
        "clean_insertion_path": {
            "mug_poses": [
                _pose(0.8, 0.0, 1.1).tolist(),
                _pose(1.0, 0.0, 1.0).tolist(),
            ]
        },
    }
    tail = build_replay_hang_tail(
        keyframes=keyframes,
        source_parts=_parts(),
        target_parts=_parts(),
        source_branches=(_branch(1.0, 1.0, 1.0),),
        target_tree_pose=_pose(),
        target_branches=(_branch(1.0, 1.0, 1.0),),
        observed_mug_pose=_pose(),
        left_eef_pose=_pose(),
        right_eef_pose=_pose(),
        unload_steps=0,
        release_steps=1,
    )

    insert = tail.trajectory.waypoint_steps["branch_insert"]
    assert tail.trajectory.waypoint_steps["branch_unload"] == insert
    assert tail.trajectory.waypoint_steps["right_release"] == insert + 1
    assert tail.trajectory.grippers[insert, 1] == 0.0
    assert tail.trajectory.grippers[insert + 1, 1] == pytest.approx(-0.0475)
    assert tail.trajectory.steps == replay_tail_steps(
        keyframes, unload_steps=0, release_steps=1
    )

    corrected = tail.planned_mug_poses.copy()
    corrected[-1, 2] += 0.01
    replaced = replace_replay_hang_tail_path(tail, corrected)
    assert replaced.trajectory.waypoint_steps == tail.trajectory.waypoint_steps
    assert replaced.unload_steps == 0
    assert replaced.release_steps == 1


def test_insertion_time_scale_preserves_geometry_and_densifies_only_insert():
    keyframes = {
        "frames": {"stable_settle": {
            "mug_pose": _pose(1.0, 0.0, 1.0).tolist(),
            "tree_pose": _pose().tolist(),
        }},
        "clean_insertion_path": {"mug_poses": [
            _pose(0.8, 0.0, 1.1).tolist(),
            _pose(0.9, 0.0, 1.05).tolist(),
            _pose(1.0, 0.0, 1.0).tolist(),
        ]},
    }
    def build(scale):
        return build_replay_hang_tail(
            keyframes=keyframes,
            source_parts=_parts(), target_parts=_parts(),
            source_branches=(_branch(1.0, 1.0, 1.0),),
            target_tree_pose=_pose(),
            target_branches=(_branch(1.0, 1.0, 1.0),),
            observed_mug_pose=_pose(), left_eef_pose=_pose(),
            right_eef_pose=_pose(), unload_steps=0, release_steps=1,
            insert_time_scale=scale,
        )

    baseline = build(1)
    dense = build(2)

    assert dense.planned_mug_poses[0] == pytest.approx(
        baseline.planned_mug_poses[0]
    )
    assert dense.planned_mug_poses[-1] == pytest.approx(
        baseline.planned_mug_poses[-1]
    )
    assert dense.trajectory.waypoint_steps["branch_approach"] == 99
    baseline_insert = baseline.trajectory.waypoint_steps["branch_insert"]
    assert dense.trajectory.waypoint_steps["branch_insert"] == (
        99 + 2 * (baseline_insert - 99)
    )
    assert dense.support_alignment["insert_time_scale"] == 2
    assert dense.trajectory.steps == replay_tail_steps(
        keyframes, unload_steps=0, release_steps=1, insert_time_scale=2
    )
