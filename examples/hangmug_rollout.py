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
                from judo_isaaclab.hang_mug_replay_tail import (
                    build_replay_hang_tail,
                    repeated_joint_nominal,
                    replay_tail_ready,
                )

                if not replay_tail_ready(samples[-1]):
                    raise RuntimeError(
                        "replay hang tail requires an observed right-only handover "
                        f"at prefix step {repair_prefix_steps - 1}: {samples[-1]}"
                    )
                tail = build_replay_hang_tail(
                    keyframes=keyframes,
                    source_parts=source_parts,
                    target_parts=target_parts,
                    source_branches=source_branches,
                    target_tree_pose=target_tree.root_pose,
                    target_branches=target_branches,
                    observed_mug_pose=samples[-1]["mug_pose"],
                    left_eef_pose=samples[-1]["left_eef_pose"],
                    right_eef_pose=samples[-1]["right_eef_pose"],
                    insert_clearance_m=args.insert_clearance_m,
                )
                trajectory = tail.trajectory
                intended_final = tail.intended_final_mug_pose
                nominal_right_contact = tail.right_contact_in_mug
                source_branch = tail.source_branch
                target_branch = tail.target_branch
                joint_nominal = repeated_joint_nominal(
                    source["actions"].detach().cpu().numpy(),
                    repair_prefix_steps,
                    trajectory.steps,
                )
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
        if trajectory is not None and trajectory_step is not None:
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
            trajectory, nominal_right_contact = _reanchor_full_skill(
                trajectory,
                trajectory_step,
                sample,
                nominal_handover_mug,
                nominal_right_contact,
                intended_final,
                observed_handover_reanchor,
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
        frame_stats=frame_stats,
    )


def _reanchor_full_skill(
    trajectory,
    step,
    sample,
    nominal_handover_mug,
    nominal_right_contact,
    intended_final,
    observed_handover_reanchor,
):
    """Apply deterministic observed-contact feedback to a full semantic skill."""

    from judo_isaaclab.hang_mug import (
        reanchor_branch_transport_contact,
        reanchor_physical_handover,
        reanchor_right_grasp_from_observed_mug,
    )
    from judo_isaaclab.put_marker import compose_pose, inverse_pose

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
    completed = next(
        (name for name in names if step == trajectory.waypoint_steps[name]), None
    )
    if completed is not None and sample["right_grasp"]:
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
        if completed == "branch_insert" and intended_final is not None:
            from judo_isaaclab.hang_mug import compensate_low_branch_insert

            trajectory = compensate_low_branch_insert(
                trajectory,
                intended_final,
                sample["mug_pose"],
            )
    return trajectory, nominal_right_contact
