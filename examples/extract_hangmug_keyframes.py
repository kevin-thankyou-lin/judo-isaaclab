"""Extract simulator-verified phase keyframes from the canonical HangMug demo."""

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
    parser.add_argument("--start-state", type=int, default=84)
    parser.add_argument("--precontact-actions", type=int, default=6)
    parser.add_argument("--friction-high", type=float, default=30.0)
    parser.add_argument("--friction-low", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--result-json",
        default="/tmp/judo_isaaclab_hangmug_keyframes.json",
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


def _load_demo(args, device):
    import h5py
    import torch

    with h5py.File(args.dataset, "r") as handle:
        group = handle[f"data/{args.episode}"]
        state = _tensor_tree(group["states"], args.start_state, device)
        actions = torch.as_tensor(
            np.asarray(group["actions"], dtype=np.float32),
            device=device,
        )
        recorded_mug_poses = np.asarray(
            group["states/rigid_object/mug/root_pose"], dtype=np.float32
        )
    return state, actions, recorded_mug_poses


def _first(events, predicate):
    return next((event for event in events if predicate(event)), None)


def _sample(env, action_step):
    import torch

    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
    forces = {}
    grasps = {}
    for side in ("left", "right"):
        end_effector = env.robot.arms[f"{side}_arm"].end_effector
        forces[side] = [
            float(finger.contact_force("mug", env_ids)[0].item())
            for finger in end_effector.fingers
        ]
        grasps[side] = bool(
            end_effector.is_grasping("mug", env_ids=env_ids)[0].item()
        )
    origin = env.scene.env_origins[0]
    mug_position = env.scene["mug"].data.root_pos_w[0] - origin
    tree_position = env.scene["mug_tree"].data.root_pos_w[0] - origin
    return {
        "action_step": int(action_step),
        "state_index": int(action_step + 1),
        "left_contact": max(forces["left"]) >= 0.1,
        "right_contact": max(forces["right"]) >= 0.1,
        "left_grasp": grasps["left"],
        "right_grasp": grasps["right"],
        "assist_engaged": bool(
            env.grasp_assists["left"].engaged[0].item()
        ),
        "stage1_pick": bool(env.stage1_success[0].item()),
        "stage2_handover": bool(env.stage2_success[0].item()),
        "stage3_hang": bool(env.stage3_success[0].item()),
        "mug_position": mug_position.detach().cpu().tolist(),
        "tree_position": tree_position.detach().cpu().tolist(),
        "mug_tree_xy_distance_m": float(
            torch.linalg.vector_norm(
                mug_position[:2] - tree_position[:2]
            ).item()
        ),
        "hang_xy_tolerance_m": float(env.hang_xy_tolerance),
    }


def _keyframe(name, event, recorded_mug_poses):
    if event is None:
        return None
    state_index = event["state_index"]
    return {
        "name": name,
        "action_step": event["action_step"],
        "state_index": state_index,
        "time_s": event["state_index"] / 30.0,
        "sim_mug_position": event["mug_position"],
        "recorded_mug_pose": recorded_mug_poses[state_index].tolist(),
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
            num_envs=1,
            observation_modalities=["proprioception"],
            enable_cameras=False,
            grasp_assist_config=assist_config,
            objects_randomization=None,
            init_joint_pos_randomization=0.0,
            enable_gripper_grasp_clamp=False,
            planning_substep_contact_sensors=True,
        )
        env = runner.env
        env.reset(warm_up=False, seed=args.seed)
        state, actions, recorded_mug_poses = _load_demo(args, env.device)
        runner.reset(
            state,
            {
                "mug": RigidObjectMpcState(False),
                "mug_tree": RigidObjectMpcState(False),
            },
            is_relative=True,
        )

        events = []
        for action_step in range(args.start_state, int(actions.shape[0])):
            _, _, terminated, truncated, _ = env.step(
                actions[action_step : action_step + 1]
            )
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError(f"Unexpected done at action {action_step}")
            events.append(_sample(env, action_step))

        left_contact = _first(events, lambda item: item["left_contact"])
        left_grasp = _first(events, lambda item: item["left_grasp"])
        pick = _first(events, lambda item: item["stage1_pick"])
        right_contact = _first(events, lambda item: item["right_contact"])
        right_grasp = _first(events, lambda item: item["right_grasp"])
        handover = _first(events, lambda item: item["stage2_handover"])
        tree_approach = _first(
            events,
            lambda item: (
                item["stage2_handover"]
                and item["mug_tree_xy_distance_m"]
                <= item["hang_xy_tolerance_m"]
            ),
        )
        release = _first(
            events,
            lambda item: (
                tree_approach is not None
                and item["action_step"] >= tree_approach["action_step"]
                and not item["left_grasp"]
                and not item["right_grasp"]
            ),
        )
        hang = _first(events, lambda item: item["stage3_hang"])
        precontact_state = max(
            args.start_state,
            left_contact["state_index"] - args.precontact_actions
            if left_contact is not None
            else args.start_state,
        )
        keyframes = {
            "precontact_checkpoint": {
                "name": "precontact_checkpoint",
                "state_index": precontact_state,
                "time_s": precontact_state / 30.0,
                "recorded_mug_pose": recorded_mug_poses[
                    precontact_state
                ].tolist(),
            },
            "left_contact": _keyframe(
                "left_contact", left_contact, recorded_mug_poses
            ),
            "left_grasp": _keyframe(
                "left_grasp", left_grasp, recorded_mug_poses
            ),
            "pick_latched": _keyframe(
                "pick_latched", pick, recorded_mug_poses
            ),
            "right_contact": _keyframe(
                "right_contact", right_contact, recorded_mug_poses
            ),
            "right_grasp": _keyframe(
                "right_grasp", right_grasp, recorded_mug_poses
            ),
            "handover_latched": _keyframe(
                "handover_latched", handover, recorded_mug_poses
            ),
            "tree_approach": _keyframe(
                "tree_approach", tree_approach, recorded_mug_poses
            ),
            "both_released": _keyframe(
                "both_released", release, recorded_mug_poses
            ),
            "hang_latched": _keyframe(
                "hang_latched", hang, recorded_mug_poses
            ),
        }
        missing = [name for name, value in keyframes.items() if value is None]
        result = {
            "status": "passed" if not missing else "failed",
            "dataset": os.path.abspath(args.dataset),
            "episode": args.episode,
            "control_hz": 30,
            "source_start_state": args.start_state,
            "keyframes": keyframes,
            "missing_keyframes": missing,
        }
        with open(args.result_json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print(
            "JUDO_ISAACLAB_HANGMUG_KEYFRAMES="
            + json.dumps(result, sort_keys=True)
        )
        if missing:
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
