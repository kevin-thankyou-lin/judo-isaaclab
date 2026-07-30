"""Run multi-round Judo CEM toward the first HangMug grasp keyframe."""

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
    parser.add_argument("--start-state", type=int, default=110)
    parser.add_argument("--target-state", type=int, default=116)
    parser.add_argument("--target-name", default="left_grasp")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num-rollouts", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--num-elites", type=int, default=4)
    parser.add_argument("--duplicate-nominal", type=int, default=4)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=0.03)
    parser.add_argument("--max-action-delta", type=float, default=0.08)
    parser.add_argument("--friction-high", type=float, default=30.0)
    parser.add_argument("--friction-low", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--result-json",
        default="/tmp/judo_isaaclab_hangmug_grasp_keyframe_mpc.json",
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


def _state_row(group, index):
    return np.concatenate(
        (
            np.asarray(group["rigid_object/mug/root_pose"][index]),
            np.asarray(group["rigid_object/mug/root_velocity"][index]),
            np.asarray(group["articulation/left_arm/joint_position"][index]),
            np.asarray(group["articulation/right_arm/joint_position"][index]),
        )
    ).astype(np.float32)


def _load_demo(args, device):
    import h5py

    with h5py.File(args.dataset, "r") as handle:
        group = handle[f"data/{args.episode}"]
        states = group["states"]
        checkpoint = _tensor_tree(states, args.start_state, device)
        nominal = np.asarray(
            group["actions"][
                args.start_state : args.start_state + args.horizon
            ],
            dtype=np.float32,
        )
        reference = np.stack(
            [
                _state_row(states, index)
                for index in range(
                    args.start_state + 1,
                    args.start_state + args.horizon + 1,
                )
            ]
        )
    return checkpoint, nominal, reference


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


def _encode_sensors(env):
    import torch

    env_ids = torch.arange(env.num_envs, device=env.device)
    end_effector = env.robot.arms["left_arm"].end_effector
    forces = [
        finger.contact_force("mug", env_ids).reshape(-1, 1)
        for finger in end_effector.fingers
    ]
    grasp = end_effector.is_grasping(
        "mug", env_ids=env_ids
    ).float().reshape(-1, 1)
    assist = env.grasp_assists["left"].engaged.float().reshape(-1, 1)
    return (
        torch.cat((*forces, grasp, assist), dim=1)
        .detach()
        .cpu()
        .numpy()
    )


def _quaternion_error(actual, target):
    actual = actual / np.linalg.norm(actual, axis=-1, keepdims=True)
    target = target / np.linalg.norm(target, axis=-1, keepdims=True)
    dots = np.abs(np.sum(actual * target, axis=-1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def _objective_components(
    states,
    sensors,
    controls,
    *,
    reference,
    nominal,
    keyframe_offset,
):
    position_error = np.linalg.norm(
        states[:, :, :3] - reference[None, :, :3], axis=-1
    )
    rotation_error = _quaternion_error(
        states[:, :, 3:7], reference[None, :, 3:7]
    )
    left_joint_error = np.sqrt(
        np.mean(
            np.square(
                states[:, :, 13:21] - reference[None, :, 13:21]
            ),
            axis=-1,
        )
    )
    action_delta = np.sqrt(
        np.mean(
            np.square(controls - nominal[None, :, :]),
            axis=(1, 2),
        )
    )
    grasp = sensors[:, :, 2]
    assist = sensors[:, :, 3]
    post_keyframe = slice(keyframe_offset, None)
    rewards = (
        -80.0 * position_error[:, keyframe_offset]
        -4.0 * rotation_error[:, keyframe_offset]
        -1.0 * left_joint_error[:, keyframe_offset]
        -5.0 * position_error.mean(axis=1)
        -0.5 * rotation_error.mean(axis=1)
        -2.0 * action_delta
        +2.0 * grasp[:, keyframe_offset]
        +1.0 * assist[:, keyframe_offset]
        +1.0 * grasp[:, post_keyframe].mean(axis=1)
    )
    return {
        "rewards": rewards,
        "keyframe_position_error_m": position_error[:, keyframe_offset],
        "keyframe_rotation_error_rad": rotation_error[:, keyframe_offset],
        "keyframe_left_joint_rms_rad": left_joint_error[:, keyframe_offset],
        "action_delta_rms": action_delta,
        "keyframe_grasp": grasp[:, keyframe_offset] > 0.5,
        "keyframe_assist": assist[:, keyframe_offset] > 0.5,
        "post_keyframe_grasp_fraction": grasp[:, post_keyframe].mean(axis=1),
    }


def _group_summary(name, rows, components):
    rewards = components["rewards"][rows]
    position = components["keyframe_position_error_m"][rows]
    rotation = components["keyframe_rotation_error_rad"][rows]
    action_delta = components["action_delta_rms"][rows]
    grasp = components["keyframe_grasp"][rows]
    assist = components["keyframe_assist"][rows]
    retention = components["post_keyframe_grasp_fraction"][rows]
    return {
        "name": name,
        "count": int(len(rewards)),
        "reward_mean": float(rewards.mean()),
        "reward_max": float(rewards.max()),
        "keyframe_position_error_m_mean": float(position.mean()),
        "keyframe_position_error_m_max": float(position.max()),
        "keyframe_rotation_error_rad_mean": float(rotation.mean()),
        "keyframe_rotation_error_rad_max": float(rotation.max()),
        "action_delta_rms_mean": float(action_delta.mean()),
        "keyframe_grasp_count": int(grasp.sum()),
        "keyframe_assist_count": int(assist.sum()),
        "post_keyframe_grasp_fraction_mean": float(retention.mean()),
    }


def main():
    args = _parser()
    keyframe_offset = args.target_state - args.start_state - 1
    if not 0 <= keyframe_offset < args.horizon:
        raise ValueError("target-state must fall inside the rollout horizon")
    if args.num_rollouts < 3:
        raise ValueError("num-rollouts must be at least three")
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
        from judo.optimizers.cem import (
            CrossEntropyMethod,
            CrossEntropyMethodConfig,
        )
        from judo_isaaclab import (
            BranchContext,
            HistoryConditionedIsaacLabBackend,
            JudoIsaacLabMPC,
        )

        np.random.seed(args.seed)
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
        checkpoint, nominal, reference = _load_demo(
            args, runner.env.device
        )
        context = BranchContext(
            checkpoint_state=checkpoint,
            action_history=np.empty((0, nominal.shape[1]), dtype=np.float32),
            rigid_object_states={
                "mug": RigidObjectMpcState(False),
                "mug_tree": RigidObjectMpcState(False),
            },
            is_relative=True,
        )
        backend = HistoryConditionedIsaacLabBackend(
            runner,
            state_encoder=_encode_state,
            sensor_encoder=_encode_sensors,
        )
        optimizer = CrossEntropyMethod(
            CrossEntropyMethodConfig(
                num_rollouts=args.num_rollouts,
                num_nodes=args.horizon,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                num_elites=args.num_elites,
            ),
            nu=nominal.shape[1],
        )

        def expand(knots):
            controls = np.asarray(knots, dtype=np.float32).copy()
            delta = np.clip(
                controls - nominal[None, :, :],
                -args.max_action_delta,
                args.max_action_delta,
            )
            controls = nominal[None, :, :] + delta
            controls[:, :, (6, 13)] = nominal[None, :, (6, 13)]
            return controls

        iteration_summaries = []

        def objective(states, sensors, controls):
            components = _objective_components(
                states,
                sensors,
                controls,
                reference=reference,
                nominal=nominal,
                keyframe_offset=keyframe_offset,
            )
            iteration_summaries.append(
                {
                    "iteration": len(iteration_summaries),
                    "reward_max": float(components["rewards"].max()),
                    "reward_mean": float(components["rewards"].mean()),
                    "keyframe_grasp_count": int(
                        components["keyframe_grasp"].sum()
                    ),
                    "keyframe_assist_count": int(
                        components["keyframe_assist"].sum()
                    ),
                    "best_keyframe_position_error_m": float(
                        components["keyframe_position_error_m"][
                            np.argmax(components["rewards"])
                        ]
                    ),
                }
            )
            return components["rewards"]

        mpc = JudoIsaacLabMPC(
            optimizer,
            backend,
            objective,
            num_iterations=args.num_iterations,
            duplicate_nominal=args.duplicate_nominal,
            min_improvement=0.01,
            noise_std_multiplier=2.0,
            control_expander=expand,
        )
        plan = mpc.plan(context, nominal)

        best_sample = plan.best_sampled_knots
        group_size = args.num_rollouts // 3
        nominal_rows = slice(0, group_size)
        optimized_rows = slice(group_size, 2 * group_size)
        best_rows = slice(2 * group_size, args.num_rollouts)
        evaluation_knots = np.empty(
            (args.num_rollouts, args.horizon, nominal.shape[1]),
            dtype=np.float64,
        )
        evaluation_knots[nominal_rows] = nominal
        evaluation_knots[optimized_rows] = plan.optimized_knots
        evaluation_knots[best_rows] = best_sample
        evaluation_controls = expand(evaluation_knots)
        states, sensors, _ = backend.rollout(
            np.empty(0), evaluation_controls
        )
        evaluation = _objective_components(
            states,
            sensors,
            evaluation_controls,
            reference=reference,
            nominal=nominal,
            keyframe_offset=keyframe_offset,
        )
        groups = {
            "nominal": _group_summary(
                "nominal", nominal_rows, evaluation
            ),
            "optimized_mean": _group_summary(
                "optimized_mean", optimized_rows, evaluation
            ),
            "best_sample": _group_summary(
                "best_sample", best_rows, evaluation
            ),
        }
        best_group = groups["best_sample"]
        reached = (
            best_group["keyframe_grasp_count"] == best_group["count"]
            and best_group["keyframe_assist_count"] == best_group["count"]
            and best_group["keyframe_position_error_m_max"] <= 0.02
            and best_group["keyframe_rotation_error_rad_max"] <= 0.10
        )
        result = {
            "status": "passed" if reached else "failed",
            "source": {
                "episode": args.episode,
                "start_state": args.start_state,
                "target_state": args.target_state,
                "target_name": args.target_name,
                "horizon": args.horizon,
                "control_hz": 30,
            },
            "optimizer": {
                "type": "judo.optimizers.cem.CrossEntropyMethod",
                "num_rollouts": args.num_rollouts,
                "num_iterations": args.num_iterations,
                "num_elites": args.num_elites,
                "duplicate_nominal": args.duplicate_nominal,
                "sigma_min": args.sigma_min,
                "sigma_max": args.sigma_max,
                "max_action_delta": args.max_action_delta,
            },
            "iteration_summaries": iteration_summaries,
            "plan": {
                "accepted_update": bool(plan.accepted_update),
                "best_rollout": int(plan.best_rollout),
                "best_iteration": int(plan.best_iteration),
                "improvement": float(plan.improvement),
                "nominal_reward_mean": float(plan.nominal_reward_mean),
                "nominal_reward_std": float(plan.nominal_reward_std),
                "first_action_delta_l2": float(
                    np.linalg.norm(plan.action - nominal[0])
                ),
            },
            "repeat_evaluation": groups,
            "best_sample_reached_keyframe": reached,
        }
        with open(args.result_json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print(
            "JUDO_ISAACLAB_HANGMUG_KEYFRAME_MPC="
            + json.dumps(result, sort_keys=True)
        )
        if not reached:
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
