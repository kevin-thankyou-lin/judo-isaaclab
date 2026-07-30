"""Validate a near-contact HangMug branch through the Judo IsaacLab backend."""

import argparse
import json
import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gear-repo",
        default=(
            "/home/linke/Projects/"
            "gear-dc-study-judo-hangmug-friction-20260730"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=(
            "/tmp/hangmug_source_demo/tasks_data/HangMugOnTree/"
            "teleop/hangmug_000.hdf5"
        ),
    )
    parser.add_argument(
        "--objects-root",
        default=(
            "/tmp/hangmug_source_demo/tasks_data/"
            "HangMugOnTree/objects"
        ),
    )
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--checkpoint-step", type=int, default=110)
    parser.add_argument("--branch-step", type=int, default=116)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--friction-high", type=float, default=30.0)
    parser.add_argument("--friction-low", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--result-json",
        default="/tmp/judo_isaaclab_hangmug_near_contact_smoke.json",
    )
    return parser.parse_args()


def _tensor_tree(group, index, device):
    import h5py
    import torch

    result = {}
    for name, value in group.items():
        if isinstance(value, h5py.Group):
            result[name] = _tensor_tree(value, index, device)
        else:
            result[name] = torch.as_tensor(
                value[index : index + 1],
                dtype=torch.float32,
                device=device,
            )
    return result


def _load_inputs(args, device):
    import h5py

    with h5py.File(args.dataset, "r") as handle:
        group = handle[f"data/{args.episode}"]
        checkpoint = _tensor_tree(
            group["states"], args.checkpoint_step, device
        )
        history = np.asarray(
            group["actions"][args.checkpoint_step : args.branch_step],
            dtype=np.float32,
        )
        nominal = np.asarray(
            group["actions"][args.branch_step : args.branch_step + args.horizon],
            dtype=np.float32,
        )
        reference_mug_pose = np.asarray(
            group["states/rigid_object/mug/root_pose"][
                args.branch_step + args.horizon
            ],
            dtype=np.float32,
        )
    return checkpoint, history, nominal, reference_mug_pose


def _encode_state(env):
    import torch

    state = env.scene.get_state(is_relative=True)
    return (
        torch.cat(
            (
                state["rigid_object"]["mug"]["root_pose"],
                state["rigid_object"]["mug"]["root_velocity"],
                state["articulation"]["left_arm"]["joint_position"],
                state["articulation"]["right_arm"]["joint_position"],
            ),
            dim=-1,
        )
        .detach()
        .cpu()
        .numpy()
    )


def _quaternion_angle(left, right):
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    if np.dot(left, right) < 0.0:
        right = -right
    return float(
        2.0
        * np.arctan2(
            np.linalg.norm(left - right), np.linalg.norm(left + right)
        )
    )


def _spread(states):
    maxima = {
        "mug_translation_m": 0.0,
        "mug_rotation_rad": 0.0,
        "mug_velocity_max_abs": 0.0,
        "left_q_max_abs": 0.0,
        "right_q_max_abs": 0.0,
    }
    for step in range(states.shape[1]):
        reference = states[0, step]
        for row in states[1:, step]:
            maxima["mug_translation_m"] = max(
                maxima["mug_translation_m"],
                float(np.linalg.norm(row[:3] - reference[:3])),
            )
            maxima["mug_rotation_rad"] = max(
                maxima["mug_rotation_rad"],
                _quaternion_angle(row[3:7], reference[3:7]),
            )
            maxima["mug_velocity_max_abs"] = max(
                maxima["mug_velocity_max_abs"],
                float(np.max(np.abs(row[7:13] - reference[7:13]))),
            )
            maxima["left_q_max_abs"] = max(
                maxima["left_q_max_abs"],
                float(np.max(np.abs(row[13:21] - reference[13:21]))),
            )
            maxima["right_q_max_abs"] = max(
                maxima["right_q_max_abs"],
                float(np.max(np.abs(row[21:29] - reference[21:29]))),
            )
    return maxima


def _event_sample(env, phase, step_index, action_step):
    import torch

    env_ids = torch.arange(env.num_envs, device=env.device)
    assist = env.grasp_assists["left"]
    end_effector = env.robot.arms["left_arm"].end_effector
    forces = [
        finger.contact_force("mug", env_ids).detach().cpu().numpy().tolist()
        for finger in end_effector.fingers
    ]
    grasp = (
        end_effector.is_grasping("mug", env_ids=env_ids)
        .detach()
        .cpu()
        .numpy()
        .astype(bool)
        .tolist()
    )
    engaged = (
        assist.engaged.detach().cpu().numpy().astype(bool).tolist()
    )
    arm = env.scene["left_arm"]
    link_index = arm.data.body_names.index(
        assist.end_effector.attach_link_name
    )
    gripper_position = (
        arm.data.body_link_pose_w[:, link_index, :3].detach().cpu().numpy()
    )
    mug_position = env.scene["mug"].data.root_pose_w[:, :3].detach().cpu().numpy()
    return {
        "phase": phase,
        "step_index": step_index,
        "action_step": action_step,
        "left_finger_forces_n": forces,
        "left_grasp": grasp,
        "assist_engaged": engaged,
        "mug_minus_gripper_world_z_m": (
            mug_position[:, 2] - gripper_position[:, 2]
        ).tolist(),
    }


def _first_event_by_clone(events, predicate, num_rollouts):
    result = []
    for env_index in range(num_rollouts):
        match = next(
            (
                event["action_step"]
                for event in events
                if event["action_step"] is not None
                and predicate(event, env_index)
            ),
            None,
        )
        result.append(match)
    return result


def _reference_divergence(states, reference_pose):
    terminal = states[:, -1, :7]
    translations = np.linalg.norm(terminal[:, :3] - reference_pose[:3], axis=1)
    rotations = np.asarray(
        [
            _quaternion_angle(row[3:7], reference_pose[3:7])
            for row in terminal
        ]
    )
    return {
        "translation_m": {
            "max": float(translations.max()),
            "mean": float(translations.mean()),
        },
        "rotation_rad": {
            "max": float(rotations.max()),
            "mean": float(rotations.mean()),
        },
    }


def main():
    args = _parser()
    if args.checkpoint_step >= args.branch_step:
        raise ValueError("checkpoint-step must precede branch-step")
    sys.path.insert(0, os.path.abspath(args.gear_repo))

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(
        {"headless": True, "device": "cpu", "enable_cameras": False}
    ).app
    runner = None
    try:
        from dc_study.planning import (
            RigidObjectMpcState,
            create_cpu_batched_planning_runner,
        )
        from judo_isaaclab import (
            BranchContext,
            HistoryConditionedIsaacLabBackend,
        )

        assets = {
            "mug": os.path.join(args.objects_root, "Mug", "mug_000"),
            "mug_tree": os.path.join(
                args.objects_root, "MugTree", "mug_tree_000"
            ),
        }
        assist_config = {
            "left": {
                "arm": "left_arm",
                "target": {"object": "mug"},
                "grasp_delay_s": 0.0,
                "mechanism": "friction",
                "friction": {
                    "high": args.friction_high,
                    "low": args.friction_low,
                },
            }
        }
        runner = create_cpu_batched_planning_runner(
            task_name="HangMugOnTree-v0",
            assets_instance_paths=assets,
            num_envs=args.num_rollouts,
            observation_modalities=["proprioception"],
            enable_cameras=False,
            grasp_assist_config=assist_config,
            objects_randomization=None,
            init_joint_pos_randomization=0.0,
            enable_gripper_grasp_clamp=False,
            planning_substep_contact_sensors=True,
        )
        runner.env.reset(warm_up=False, seed=args.seed)
        checkpoint, history, nominal, reference_pose = _load_inputs(
            args, runner.env.device
        )

        events = []

        def observe(env, phase, step_index):
            if phase == "reset":
                action_step = None
            elif phase == "history":
                action_step = args.checkpoint_step + step_index
            else:
                action_step = args.branch_step + step_index
            events.append(
                _event_sample(env, phase, step_index, action_step)
            )

        backend = HistoryConditionedIsaacLabBackend(
            runner,
            state_encoder=_encode_state,
            step_observer=observe,
        )
        backend.set_branch_context(
            BranchContext(
                checkpoint_state=checkpoint,
                action_history=history,
                rigid_object_states={
                    "mug": RigidObjectMpcState(False),
                    "mug_tree": RigidObjectMpcState(False),
                },
                is_relative=True,
            )
        )
        controls = np.repeat(
            nominal[None, :, :], args.num_rollouts, axis=0
        )
        states, sensors, _ = backend.rollout(np.empty(0), controls)

        reset_event = events[0]
        branch_event = events[len(history)]
        candidate_events = [
            event for event in events if event["phase"] == "candidate"
        ]
        contact_steps = _first_event_by_clone(
            events,
            lambda event, index: max(
                event["left_finger_forces_n"][0][index],
                event["left_finger_forces_n"][1][index],
            )
            >= 0.1,
            args.num_rollouts,
        )
        grasp_steps = _first_event_by_clone(
            events,
            lambda event, index: event["left_grasp"][index],
            args.num_rollouts,
        )
        assist_steps = _first_event_by_clone(
            events,
            lambda event, index: event["assist_engaged"][index],
            args.num_rollouts,
        )
        branch_relative_z = np.asarray(
            branch_event["mug_minus_gripper_world_z_m"]
        )
        max_relative_drop = max(
            float(
                np.max(
                    branch_relative_z
                    - np.asarray(
                        event["mug_minus_gripper_world_z_m"]
                    )
                )
            )
            for event in candidate_events
        )
        checkpoint_contact_free = all(
            max(
                reset_event["left_finger_forces_n"][0][index],
                reset_event["left_finger_forces_n"][1][index],
            )
            < 0.1
            and not reset_event["left_grasp"][index]
            and not reset_event["assist_engaged"][index]
            for index in range(args.num_rollouts)
        )
        branch_grasped = all(branch_event["left_grasp"]) and all(
            branch_event["assist_engaged"]
        )
        retained = all(
            all(event["left_grasp"]) and all(event["assist_engaged"])
            for event in candidate_events
        )
        spread = _spread(states)
        reference_divergence = _reference_divergence(
            states, reference_pose
        )
        passed = (
            states.shape == (args.num_rollouts, args.horizon, 29)
            and sensors.shape == (args.num_rollouts, args.horizon, 0)
            and checkpoint_contact_free
            and branch_grasped
            and retained
            and all(step is not None for step in contact_steps)
            and all(step is not None for step in grasp_steps)
            and all(step is not None for step in assist_steps)
            and all(np.isfinite(value) for value in spread.values())
        )
        result = {
            "status": "passed" if passed else "failed",
            "checkpoint_step": args.checkpoint_step,
            "branch_step": args.branch_step,
            "history_steps": history.shape[0],
            "horizon_steps": args.horizon,
            "num_rollouts": args.num_rollouts,
            "identical_controls": True,
            "checkpoint_contact_free": checkpoint_contact_free,
            "first_contact_action_step_by_clone": contact_steps,
            "first_grasp_action_step_by_clone": grasp_steps,
            "first_assist_action_step_by_clone": assist_steps,
            "branch_all_grasped_and_assisted": branch_grasped,
            "candidate_grasp_retained": retained,
            "max_candidate_relative_vertical_drop_m": max_relative_drop,
            "internal_max_spread": spread,
            "recorded_main_terminal_divergence": reference_divergence,
            "backend_diagnostics": vars(backend.last_diagnostics),
        }
        with open(args.result_json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print(
            "JUDO_ISAACLAB_HANGMUG_SMOKE="
            + json.dumps(result, sort_keys=True)
        )
        if not passed:
            raise RuntimeError(result)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if runner is not None:
            runner.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
