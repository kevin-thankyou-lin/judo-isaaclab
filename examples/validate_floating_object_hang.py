"""Validate an object-first HangMug path before asking the robot to track it."""

import argparse
import json
import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import render_hangmug_mpc_comparison as render


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gear-repo", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--objects-root", required=True)
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--controls-npz", required=True)
    parser.add_argument("--target-mug", required=True)
    parser.add_argument("--target-mug-tree", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--release-steps", type=int, default=120)
    parser.add_argument("--stability-steps", type=int, default=30)
    parser.add_argument("--release-open-value", type=float, default=-0.0475)
    parser.add_argument(
        "--release-pose",
        type=float,
        nargs=7,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help="Override the final floating-object pose used for release.",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def _renderer_args(args):
    return argparse.Namespace(
        gear_repo=args.gear_repo,
        dataset=args.dataset,
        objects_root=args.objects_root,
        episode=args.episode,
        target_mug=args.target_mug,
        target_mug_tree=args.target_mug_tree,
        controls_npz=args.controls_npz,
    )


def _relative_mug_pose(env):
    pose = env.scene["mug"].data.root_pose_w.detach().clone()
    pose[:, :3] -= env.scene.env_origins
    return pose


def main():
    args = _parser()
    if args.release_steps < args.stability_steps or args.stability_steps < 1:
        raise ValueError("release steps must cover the positive stability window")
    sys.path.insert(0, os.path.abspath(args.gear_repo))

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(
        {"headless": True, "device": "cpu", "enable_cameras": False}
    ).app
    runner = None
    try:
        import torch
        from dc_study.planning import (
            RigidObjectMpcState,
            create_cpu_batched_planning_runner,
        )

        controls = np.load(args.controls_npz)
        if (
            "reference_object_poses" not in controls.files
            or not controls["reference_object_poses"].size
        ):
            raise ValueError("controls artifact has no floating-object reference")
        object_path = np.asarray(
            controls["reference_object_poses"], dtype=np.float32
        )
        if args.release_pose is not None:
            object_path = object_path.copy()
            object_path[-1] = np.asarray(args.release_pose, dtype=np.float32)

        runner = create_cpu_batched_planning_runner(
            task_name="HangMugOnTree-v0",
            assets_instance_paths={
                "mug": args.target_mug,
                "mug_tree": args.target_mug_tree,
            },
            num_envs=1,
            observation_modalities=["proprioception"],
            enable_cameras=False,
            grasp_assist_config=None,
            objects_randomization=None,
            init_joint_pos_randomization=0.0,
            enable_gripper_grasp_clamp=False,
            planning_substep_contact_sensors=True,
            replicate_physics=True,
        )
        env = runner.env
        env.reset(warm_up=False, seed=args.seed)
        inputs = render._load_inputs(_renderer_args(args), env.device)
        runner.reset(
            inputs["checkpoint"],
            {
                "mug": RigidObjectMpcState(False),
                "mug_tree": RigidObjectMpcState(False),
            },
            is_relative=True,
        )
        render._initialize_task_stage(env, 2)
        for action in inputs["history"]:
            env.step(action.reshape(1, -1))

        mug = env.scene["mug"]
        env_ids = torch.arange(1, device=env.device)
        zero_velocity = torch.zeros((1, 6), device=env.device)
        tracking_error = []
        for relative_pose in object_path:
            pose = torch.as_tensor(
                relative_pose, dtype=torch.float32, device=env.device
            ).reshape(1, 7)
            pose[:, :3] += env.scene.env_origins
            mug.write_root_pose_to_sim(pose, env_ids=env_ids)
            mug.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
            env.scene.write_data_to_sim()
            env.sim.forward()
            actual = _relative_mug_pose(env)[0]
            tracking_error.append(
                float(
                    torch.linalg.vector_norm(
                        actual[:3]
                        - torch.as_tensor(
                            relative_pose[:3], device=env.device
                        )
                    )
                )
            )

        hold_action = inputs["candidate"][1, -1].detach().clone()
        hold_action[6] = args.release_open_value
        hold_action[13] = args.release_open_value
        stage3 = []
        speed = []
        for _ in range(args.release_steps):
            _, _, terminated, truncated, _ = env.step(
                hold_action.reshape(1, -1)
            )
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError("floating-object release produced a done signal")
            stage3.append(bool(env.stage3_success[0]))
            speed.append(
                float(torch.linalg.vector_norm(mug.data.root_vel_w[0, :3]))
            )

        terminal_pose = _relative_mug_pose(env)[0].detach().cpu().numpy()
        terminal_position_drift = float(
            np.linalg.norm(terminal_pose[:3] - object_path[-1, :3])
        )
        stable = all(stage3[-args.stability_steps :])
        result = {
            "status": "passed" if stable else "failed",
            "protocol": {
                "reference_mode": "floating_object",
                "kinematic_path_steps": int(len(object_path)),
                "release_steps": args.release_steps,
                "stability_steps": args.stability_steps,
                "robot_tracking_used": False,
                "target_mug": args.target_mug,
                "target_mug_tree": args.target_mug_tree,
                "release_pose_override": args.release_pose,
            },
            "kinematic_tracking_error_m_max": max(tracking_error),
            "stage3_stable": stable,
            "stage3_first_step": (
                next((index for index, value in enumerate(stage3) if value), None)
            ),
            "terminal_position_drift_m": terminal_position_drift,
            "terminal_speed_m_s": speed[-1],
            "release_speed_m_s_max": max(speed),
        }
        with open(args.result_json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print("FLOATING_OBJECT_HANG_FINAL=" + json.dumps(result, sort_keys=True))
        if not stable:
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
