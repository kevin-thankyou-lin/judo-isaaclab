"""Real PutPot smoke for HistoryConditionedIsaacLabBackend.

This example expects the gear-dc-study planning branch and the officially
downloaded PutPot demo/assets used by that project.
"""

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
        default="/home/linke/Projects/gear-dc-study-judo-cpu-batched-20260729",
    )
    parser.add_argument(
        "--dataset",
        default=(
            "/tmp/putpot_canonical_dataset_20260730/tasks_data/"
            "PutPotOnCooktop/teleop/putpot_000.hdf5"
        ),
    )
    parser.add_argument(
        "--objects-root",
        default=(
            "/tmp/putpot_canonical_dataset_20260730/tasks_data/"
            "PutPotOnCooktop/objects"
        ),
    )
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--checkpoint-step", type=int, default=240)
    parser.add_argument("--branch-step", type=int, default=280)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--result-json",
        default="/tmp/judo_isaaclab_putpot_backend_smoke.json",
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
    return checkpoint, history, nominal


def _encode_state(env):
    import torch

    state = env.scene.get_state(is_relative=False)
    return (
        torch.cat(
            (
                state["rigid_object"]["pot"]["root_pose"],
                state["rigid_object"]["pot"]["root_velocity"],
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
    return 2.0 * np.arctan2(
        np.linalg.norm(left - right), np.linalg.norm(left + right)
    )


def _spread(states):
    maxima = {
        "pot_translation_m": 0.0,
        "pot_rotation_rad": 0.0,
        "pot_velocity_max_abs": 0.0,
        "left_q_max_abs": 0.0,
        "right_q_max_abs": 0.0,
    }
    for step in range(states.shape[1]):
        reference = states[0, step]
        for row in states[1:, step]:
            maxima["pot_translation_m"] = max(
                maxima["pot_translation_m"],
                float(np.linalg.norm(row[:3] - reference[:3])),
            )
            maxima["pot_rotation_rad"] = max(
                maxima["pot_rotation_rad"],
                float(_quaternion_angle(row[3:7], reference[3:7])),
            )
            maxima["pot_velocity_max_abs"] = max(
                maxima["pot_velocity_max_abs"],
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


def _grasp_counts(env):
    import torch

    env_ids = torch.arange(env.num_envs, device=env.device)
    return {
        side: int(
            env.robot.arms[f"{side}_arm"]
            .end_effector.is_grasping("pot", env_ids=env_ids)
            .sum()
            .item()
        )
        for side in ("left", "right")
    }


def main():
    args = _parser()
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
            "pot": os.path.join(args.objects_root, "Pot", "pot_000"),
            "cooktop": os.path.join(
                args.objects_root, "Cooktop", "cooktop_000"
            ),
        }
        runner = create_cpu_batched_planning_runner(
            task_name="PutPotOnCooktop-v0",
            assets_instance_paths=assets,
            num_envs=args.num_rollouts,
            observation_modalities=["proprioception"],
            enable_cameras=False,
            objects_randomization=None,
            init_joint_pos_randomization=0.0,
            enable_gripper_grasp_clamp=False,
            planning_substep_contact_sensors=True,
        )
        runner.env.reset(warm_up=False, seed=args.seed)
        checkpoint, history, nominal = _load_inputs(args, runner.env.device)
        backend = HistoryConditionedIsaacLabBackend(
            runner,
            state_encoder=_encode_state,
        )
        backend.set_branch_context(
            BranchContext(
                checkpoint_state=checkpoint,
                action_history=history,
                rigid_object_states={
                    "pot": RigidObjectMpcState(False),
                    "cooktop": RigidObjectMpcState(False),
                },
                is_relative=False,
            )
        )
        controls = np.repeat(
            nominal[None, :, :], args.num_rollouts, axis=0
        )
        states, sensors, _ = backend.rollout(np.empty(0), controls)
        grasp_counts = _grasp_counts(runner.env)
        spread = _spread(states)
        passed = (
            states.shape == (args.num_rollouts, args.horizon, 29)
            and sensors.shape == (args.num_rollouts, args.horizon, 0)
            and grasp_counts == {
                "left": args.num_rollouts,
                "right": args.num_rollouts,
            }
            and all(np.isfinite(value) for value in spread.values())
        )
        result = {
            "status": "passed" if passed else "failed",
            "num_rollouts": args.num_rollouts,
            "history_steps": history.shape[0],
            "horizon_steps": args.horizon,
            "identical_controls": True,
            "grasp_counts": grasp_counts,
            "internal_max_spread": spread,
            "backend_diagnostics": vars(backend.last_diagnostics),
        }
        with open(args.result_json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print("JUDO_ISAACLAB_PUTPOT_SMOKE=" + json.dumps(result, sort_keys=True))
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
