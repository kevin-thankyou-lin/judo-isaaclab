"""Execute one uninterrupted HangMug replay, hybrid repair, or full skill."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable

import numpy as np

from judo_isaaclab.semantic_execution import SemanticExecutionEvent


@dataclass
class HangMugRollout:
    samples: list[dict[str, Any]]
    actions: list[np.ndarray]
    mug_poses: list[Any]
    left_eef: list[Any]
    right_eef: list[Any]
    desired_left: list[Any]
    desired_right: list[Any]
    desired_steps: list[int]
    trajectory: Any
    joint_nominal: Any
    intended_final: Any
    nominal_right_contact: Any
    source_branch: Any
    target_branch: Any
    frame_stats: list[tuple[float, float]]
    planned_clean_insertion: dict[str, Any] | None


def _start_replay_tail(
    *, keyframes, source_parts, target_parts, source_branches,
    target_tree, target_branches, target_assets, sample, source_actions,
    repair_prefix_steps, args,
):
    """Build and fail-closed screen the source-relationship insertion tail."""

    from judo_isaaclab.hang_mug_clean_insertion import (
        apply_branch_radial_clearance,
        exact_body_collision_receipt,
    )
    from judo_isaaclab.hang_mug_replay_tail import (
        build_replay_hang_tail,
        repeated_joint_nominal,
        replace_replay_hang_tail_path,
        replay_tail_ready,
    )

    if not replay_tail_ready(sample):
        raise RuntimeError(
            "replay hang tail requires an observed right-only handover "
            f"at prefix step {repair_prefix_steps - 1}: {sample}"
        )
    tail = build_replay_hang_tail(
        keyframes=keyframes, source_parts=source_parts,
        target_parts=target_parts, source_branches=source_branches,
        target_tree_pose=target_tree.root_pose,
        target_branches=target_branches, observed_mug_pose=sample["mug_pose"],
        left_eef_pose=sample["left_eef_pose"],
        right_eef_pose=sample["right_eef_pose"],
    )
    receipt = None
    if args.require_clean_insertion:
        receipt = exact_body_collision_receipt(
            tail.planned_mug_poses, tree_pose=target_tree.root_pose,
            target_assets=target_assets,
        )
        receipt["support_alignment"] = tail.support_alignment
        print(
            "HANGMUG_PLANNED_CLEAN_INSERTION="
            + json.dumps(receipt, sort_keys=True), flush=True,
        )
        if not receipt["passed"]:
            corrected, correction = apply_branch_radial_clearance(
                tail.planned_mug_poses,
                tree_pose=target_tree.root_pose,
                mug_body_frame=target_parts.body_frame,
                mug_body_size=target_parts.body_size,
                target_branch=tail.target_branch,
                collision_steps=receipt["collision_steps"],
            )
            tail = replace_replay_hang_tail_path(tail, corrected)
            receipt = exact_body_collision_receipt(
                tail.planned_mug_poses, tree_pose=target_tree.root_pose,
                target_assets=target_assets,
            )
            receipt["support_alignment"] = tail.support_alignment
            receipt["geometry_correction"] = correction
            print(
                "HANGMUG_PLANNED_CLEAN_INSERTION_CORRECTED="
                + json.dumps(receipt, sort_keys=True), flush=True,
            )
            if not receipt["passed"]:
                raise RuntimeError(
                    "single-pass branch-radial correction did not clear the "
                    "target cup body from the tree"
                )
    joint_nominal = repeated_joint_nominal(
        source_actions.detach().cpu().numpy(),
        repair_prefix_steps,
        tail.trajectory.steps,
    )
    return tail, joint_nominal, receipt


def _record_feedback_collision_receipt(
    receipt, *, planned, trajectory, trajectory_step,
):
    """Attach and print the collision receipt for one feedback correction."""

    insert = trajectory.waypoint_steps["branch_insert"]
    unload = trajectory.waypoint_steps["branch_unload"]
    held_convergence = trajectory_step == (insert + unload) // 2
    receipt_key = (
        "held_convergence_clean_insertion"
        if held_convergence
        else "approach_feedback_clean_insertion"
    )
    planned[receipt_key] = receipt
    label = (
        "HANGMUG_HELD_CONVERGENCE_CLEAN_INSERTION="
        if held_convergence
        else "HANGMUG_APPROACH_FEEDBACK_CLEAN_INSERTION="
    )
    print(label + json.dumps(receipt, sort_keys=True), flush=True)


def _screen_or_reject_observation_feedback(
    trajectory, previous_trajectory, *, repair_kwargs, previous_repair_kwargs,
):
    """Keep the last exact-screened suffix when pose feedback is unsafe."""

    from judo_isaaclab.hang_mug_clean_insertion import (
        repair_compensated_insertion_path,
    )

    try:
        corrected, receipt = repair_compensated_insertion_path(
            trajectory, **repair_kwargs,
        )
        return corrected, receipt, 1.0
    except RuntimeError as error:
        if "collision-unsafe" not in str(error):
            raise
        retained, receipt = repair_compensated_insertion_path(
            previous_trajectory, **previous_repair_kwargs,
        )
        receipt = dict(receipt)
        receipt["rejected_observation_feedback"] = True
        receipt["rejected_reason"] = str(error)
        receipt["right_dls_gain"] = 2.0
        return retained, receipt, 2.0


def execute_hangmug_rollout(
    *,
    env: Any,
    source: dict[str, Any],
    keyframes: dict[str, Any] | None,
    source_parts: Any,
    target_parts: Any,
    source_branches: Any,
    target_tree: Any,
    target_branches: Any,
    target_assets: dict[str, str],
    trajectory: Any,
    joint_nominal: Any,
    intended_final: Any,
    nominal_handover_mug: Any,
    nominal_right_contact: Any,
    source_branch: Any,
    target_branch: Any,
    observed_handover_reanchor: bool,
    repair_prefix_steps: int | None,
    total_steps: int,
    args: Any,
    demo_recorder: Any,
    encoder: Any,
    samples: list[dict[str, Any]],
    protocol_recorder: Any,
    execution_hooks: Any,
    ik_action: Callable[..., Any],
    sample_environment: Callable[..., dict[str, Any]],
    update_assist_releases: Callable[..., None],
    render_frame: Callable[..., np.ndarray],
) -> HangMugRollout:
    """Execute source actions and optional target-direct Cartesian suffix."""

    actions = []
    mug_poses = []
    left_eef = []
    right_eef = []
    desired_left = []
    desired_right = []
    desired_steps = []
    frame_stats = []
    right_dls_gain = 1.0
    planned_clean_insertion = None
    milestones_by_step: dict[int, list[str]] = {}
    if trajectory is not None:
        for name, milestone_step in trajectory.waypoint_steps.items():
            milestones_by_step.setdefault(int(milestone_step), []).append(name)
    initial_stage = (
        "source_action_prefix"
        if repair_prefix_steps is not None
        else "direct_source_action_replay"
        if trajectory is None
        else trajectory.stage_names[0]
    )
    protocol_recorder.start_rollout()
    execution_hooks.emit(
        SemanticExecutionEvent(
            kind="rollout_start",
            step=0,
            stage=initial_stage,
            observation=samples[-1],
            metadata={"mode": args.mode},
        )
    )
    for step in range(total_steps):
        trajectory_step = None
        if repair_prefix_steps is not None and step < repair_prefix_steps:
            action = source["actions"][step : step + 1]
            stage = "source_action_prefix"
        elif repair_prefix_steps is not None:
            trajectory_step = step - repair_prefix_steps
            if trajectory is None:
                tail, joint_nominal, planned_clean_insertion = _start_replay_tail(
                    keyframes=keyframes, source_parts=source_parts,
                    target_parts=target_parts, source_branches=source_branches,
                    target_tree=target_tree, target_branches=target_branches,
                    target_assets=target_assets, sample=samples[-1],
                    source_actions=source["actions"],
                    repair_prefix_steps=repair_prefix_steps, args=args,
                )
                trajectory = tail.trajectory
                intended_final = tail.intended_final_mug_pose
                nominal_right_contact = tail.right_contact_in_mug
                source_branch = tail.source_branch
                target_branch = tail.target_branch
                for name, milestone_step in trajectory.waypoint_steps.items():
                    absolute_step = repair_prefix_steps + int(milestone_step)
                    milestones_by_step.setdefault(absolute_step, []).append(name)
            stage = trajectory.stage_names[trajectory_step]
            action = ik_action(
                env,
                trajectory.left_poses[trajectory_step],
                trajectory.right_poses[trajectory_step],
                trajectory.grippers[trajectory_step],
                joint_nominal[trajectory_step],
                args,
                right_dls_gain=right_dls_gain,
                integrate_left_ik=True,
                integrate_right_ik=True,
            )
            desired_steps.append(step)
            desired_left.append(trajectory.left_poses[trajectory_step])
            desired_right.append(trajectory.right_poses[trajectory_step])
        elif trajectory is None:
            action = source["actions"][step : step + 1]
            stage = "direct_source_action_replay"
        else:
            trajectory_step = step
            stage = trajectory.stage_names[step]
            integrate = step > trajectory.waypoint_steps["left_grasp"]
            action = ik_action(
                env,
                trajectory.left_poses[step],
                trajectory.right_poses[step],
                trajectory.grippers[step],
                joint_nominal[step],
                args,
                right_dls_gain=right_dls_gain,
                integrate_left_ik=integrate,
                integrate_right_ik=integrate,
            )
            desired_steps.append(step)
            desired_left.append(trajectory.left_poses[step])
            desired_right.append(trajectory.right_poses[step])
        execution_hooks.emit(
            SemanticExecutionEvent(kind="before_step", step=step, stage=stage)
        )
        observation, _, terminated, truncated, info = env.step(action)
        if (
            trajectory is not None
            and trajectory_step is not None
        ):
            update_assist_releases(env, trajectory, trajectory_step)
        sample = sample_environment(env, step, stage, info)
        protocol_recorder.record_step(
            step=step, stage=stage, terminated=terminated, truncated=truncated
        )
        for arm in ("left", "right"):
            protocol_recorder.record_contact_observation(
                f"{arm}_mug_grasp",
                step=step,
                active=sample[f"{arm}_grasp"],
                source="env.robot.is_grasping",
            )
        execution_hooks.emit(
            SemanticExecutionEvent(
                kind="after_step", step=step, stage=stage, observation=sample
            )
        )
        for milestone in milestones_by_step.get(step, ()):
            execution_hooks.emit(
                SemanticExecutionEvent(
                    kind="milestone",
                    step=step,
                    stage=stage,
                    milestone=milestone,
                    observation=sample,
                )
            )
        demo_recorder.append(
            action,
            env.scene.get_state(is_relative=False),
            observation=observation,
            semantic_observation=sample,
        )
        samples.append(sample)
        actions.append(action[0].detach().cpu().numpy())
        mug_poses.append(sample["mug_pose"])
        left_eef.append(sample["left_eef_pose"])
        right_eef.append(sample["right_eef_pose"])

        if trajectory is not None and trajectory_step is not None:
            branch_entry_clearance_m = None
            if target_branch is not None:
                branch_entry_clearance_m = (
                    0.5 * float(target_parts.handle_outer_size[2])
                    + float(target_branch.radius_m)
                )
            previous_trajectory = trajectory
            previous_right_contact = nominal_right_contact
            trajectory, nominal_right_contact, feedback_compensated = _reanchor_full_skill(
                trajectory,
                trajectory_step,
                sample,
                nominal_handover_mug,
                nominal_right_contact,
                intended_final,
                observed_handover_reanchor,
                branch_entry_clearance_m,
            )
            if feedback_compensated and args.require_clean_insertion:
                repair_kwargs = {
                    "completed_step": trajectory_step,
                    "right_contact_in_mug": nominal_right_contact,
                    "tree_pose": target_tree.root_pose,
                    "mug_body_frame": target_parts.body_frame,
                    "mug_body_size": target_parts.body_size,
                    "target_branch": target_branch,
                    "target_assets": target_assets,
                    "executed_mug_poses": mug_poses[
                        0 if repair_prefix_steps is None else repair_prefix_steps :
                    ],
                }
                previous_repair_kwargs = dict(repair_kwargs)
                previous_repair_kwargs["right_contact_in_mug"] = (
                    previous_right_contact
                )
                trajectory, feedback_receipt, gain = (
                    _screen_or_reject_observation_feedback(
                        trajectory,
                        previous_trajectory,
                        repair_kwargs=repair_kwargs,
                        previous_repair_kwargs=previous_repair_kwargs,
                    )
                )
                if gain > 1.0:
                    nominal_right_contact = previous_right_contact
                right_dls_gain = max(right_dls_gain, gain)
                _record_feedback_collision_receipt(
                    feedback_receipt, planned=planned_clean_insertion,
                    trajectory=trajectory, trajectory_step=trajectory_step,
                )
        if encoder is not None:
            frame = render_frame(env, sample)
            encoder.write(frame)
            frame_stats.append((float(frame.mean()), float(frame.std())))
        if (step + 1) % 50 == 0 or sample["task_success"]:
            keys = (
                "step", "program_stage", "stage1", "stage2", "stage3",
                "task_success", "left_grasp", "right_grasp",
                "grasp_assist_engaged", "mug_pose", "mug_tree_xy_error_m",
            )
            print(
                "HANGMUG_PROGRESS="
                + json.dumps({key: sample[key] for key in keys}, sort_keys=True),
                flush=True,
            )
    protocol_recorder.finish_rollout()
    execution_hooks.emit(
        SemanticExecutionEvent(
            kind="rollout_end",
            step=total_steps - 1,
            stage=samples[-1]["program_stage"],
            observation=samples[-1],
        )
    )
    return HangMugRollout(
        samples=samples, actions=actions, mug_poses=mug_poses,
        left_eef=left_eef, right_eef=right_eef,
        desired_left=desired_left, desired_right=desired_right,
        desired_steps=desired_steps, trajectory=trajectory,
        joint_nominal=joint_nominal, intended_final=intended_final,
        nominal_right_contact=nominal_right_contact,
        source_branch=source_branch, target_branch=target_branch,
        frame_stats=frame_stats, planned_clean_insertion=planned_clean_insertion,
    )


def _apply_held_convergence_feedback(
    trajectory, *, step, sample, intended_final,
):
    """Smoothly correct the remaining held suffix from a midpoint observation."""

    insert = trajectory.waypoint_steps.get("branch_insert")
    unload = trajectory.waypoint_steps.get("branch_unload")
    convergence_step = (
        (insert + unload) // 2
        if insert is not None and unload is not None
        else None
    )
    if (
        step != convergence_step
        or not sample["right_grasp"]
        or intended_final is None
    ):
        return trajectory, False

    from judo_isaaclab.hang_mug import compensate_low_branch_insert

    before = trajectory.right_poses.copy()
    trajectory = compensate_low_branch_insert(
        trajectory,
        intended_final,
        sample["mug_pose"],
        completed_step=step,
    )
    correction = trajectory.right_poses[-1, :3] - before[-1, :3]
    quaternion_dot = abs(float(np.dot(
        trajectory.right_poses[-1, 3:], before[-1, 3:]
    )))
    compensated = not np.allclose(trajectory.right_poses, before)
    print("HANGMUG_HELD_CONVERGENCE_COMPENSATION=" + json.dumps({
        "applied_translation_m": correction.tolist(),
        "applied_rotation_rad": float(
            2.0 * np.arccos(np.clip(quaternion_dot, -1.0, 1.0))
        ),
        "completed_step": int(step),
        "observed_mug_position_m": list(sample["mug_pose"][:3]),
        "intended_support_position_m": intended_final[:3].tolist(),
    }, sort_keys=True))
    return trajectory, compensated


def _reanchor_full_skill(
    trajectory,
    step,
    sample,
    nominal_handover_mug,
    nominal_right_contact,
    intended_final,
    observed_handover_reanchor,
    branch_entry_clearance_m=None,
):
    """Apply deterministic observed-contact feedback to a full semantic skill."""

    from judo_isaaclab.hang_mug import (
        reanchor_branch_transport_contact,
        reanchor_physical_handover,
        reanchor_right_grasp_from_observed_mug,
    )
    from judo_isaaclab.put_marker import compose_pose, inverse_pose

    feedback_compensated = False

    if (
        observed_handover_reanchor
        and step == trajectory.waypoint_steps["left_lift"]
        and sample["left_grasp"]
    ):
        trajectory = reanchor_physical_handover(
            trajectory,
            nominal_handover_mug,
            sample["mug_pose"],
            sample["left_eef_pose"],
            sample["right_eef_pose"],
        )
    if (
        observed_handover_reanchor
        and step == trajectory.waypoint_steps["handover_pregrasp"]
        and sample["left_grasp"]
    ):
        trajectory = reanchor_right_grasp_from_observed_mug(
            trajectory,
            nominal_right_contact,
            sample["mug_pose"],
            sample["right_eef_pose"],
        )
    names = tuple(
        name for name in (
            "left_release", "tree_transport", "branch_approach",
            "branch_insert", "branch_unload",
        ) if name in trajectory.waypoint_steps
    )
    completed_names = tuple(
        name for name in names if step == trajectory.waypoint_steps[name]
    )
    if completed_names and sample["right_grasp"]:
        for completed in completed_names:
            trajectory = reanchor_branch_transport_contact(
                trajectory,
                nominal_right_contact,
                sample["mug_pose"],
                sample["right_eef_pose"],
                completed_waypoint=completed,
            )
            nominal_right_contact = compose_pose(
                inverse_pose(sample["mug_pose"]), sample["right_eef_pose"]
            )
            if completed == "branch_approach" and intended_final is not None:
                from judo_isaaclab.hang_mug import compensate_low_branch_approach

                before = trajectory.right_poses.copy()
                trajectory = compensate_low_branch_approach(
                    trajectory,
                    intended_final,
                    sample["mug_pose"],
                    support_clearance_m=(
                        0.01
                        if branch_entry_clearance_m is None
                        else branch_entry_clearance_m
                    ),
                    maximum_vertical_correction_m=(
                        0.03
                        if branch_entry_clearance_m is None
                        else max(0.03, 1.5 * branch_entry_clearance_m)
                    ),
                )
                correction_z = float(
                    np.max(trajectory.right_poses[:, 2] - before[:, 2])
                )
                feedback_compensated = correction_z > 0.0
                print("HANGMUG_APPROACH_COMPENSATION=" + json.dumps({
                    "applied_vertical_m": correction_z,
                    "observed_mug_z_m": float(sample["mug_pose"][2]),
                    "support_clearance_m": (
                        0.01
                        if branch_entry_clearance_m is None
                        else branch_entry_clearance_m
                    ),
                    "intended_support_z_m": float(intended_final[2]),
                }, sort_keys=True))
            if completed == "branch_insert" and intended_final is not None:
                from judo_isaaclab.hang_mug import compensate_low_branch_insert

                before = trajectory.right_poses.copy()
                trajectory = compensate_low_branch_insert(
                    trajectory,
                    intended_final,
                    sample["mug_pose"],
                )
                feedback_compensated = feedback_compensated or not np.allclose(
                    trajectory.right_poses, before
                )
                correction = trajectory.right_poses[-1, :3] - before[-1, :3]
                quaternion_dot = abs(float(np.dot(
                    trajectory.right_poses[-1, 3:], before[-1, 3:]
                )))
                print("HANGMUG_INSERT_COMPENSATION=" + json.dumps({
                    "applied_translation_m": correction.tolist(),
                    "applied_rotation_rad": float(
                        2.0 * np.arccos(np.clip(quaternion_dot, -1.0, 1.0))
                    ),
                    "observed_mug_position_m": list(sample["mug_pose"][:3]),
                    "intended_support_position_m": intended_final[:3].tolist(),
                }, sort_keys=True))
    trajectory, held_compensated = _apply_held_convergence_feedback(
        trajectory, step=step, sample=sample, intended_final=intended_final,
    )
    feedback_compensated = feedback_compensated or held_compensated
    return trajectory, nominal_right_contact, feedback_compensated
