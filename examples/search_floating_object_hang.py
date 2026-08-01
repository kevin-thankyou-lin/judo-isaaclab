"""Search a continuous floating-mug insertion whose cold release is supported."""

import argparse
import json
import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import render_hangmug_mpc_comparison as render
from hangmug_grasp_keyframe_mpc import (
    _branch_frame,
    _quat_to_matrix,
    insert,
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gear-repo", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--objects-root", required=True)
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--controls-npz", required=True)
    parser.add_argument("--target-mug", required=True)
    parser.add_argument("--target-mug-tree", required=True)
    parser.add_argument("--seed-pose", type=float, nargs=7, required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--num-envs", type=int, default=24)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--num-elites", type=int, default=5)
    parser.add_argument("--path-steps", type=int, default=120)
    parser.add_argument("--release-steps", type=int, default=90)
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


def _candidate_paths(start_pose, seed_pose, branch_rotation, parameters, horizon):
    """Convert branch-frame target/clearance parameters into mug SE(3) paths."""
    paths = []
    for parameter in np.asarray(parameters, dtype=np.float32):
        target_delta = parameter[:3]
        rotation_delta = parameter[3:6]
        clearance_radial = parameter[6:8]
        calibrated_target = seed_pose[:3] + branch_rotation @ target_delta
        start_offset = branch_rotation.T @ (start_pose[:3] - calibrated_target)
        clearance_x = max(float(start_offset[0]), 0.05)
        paths.append(
            insert(
                start_pose,
                seed_pose,
                branch_rotation,
                horizon,
                target_position_offset_branch=target_delta,
                target_rotation_offset_branch=rotation_delta,
                clearance_offset_branch=(
                    clearance_x,
                    float(clearance_radial[0]),
                    float(clearance_radial[1]),
                ),
                approach_offset_branch=(0.05, 0.0, 0.0),
                clearance_fraction=0.20,
                approach_fraction=0.65,
                seat_fraction=0.90,
            )
        )
    return np.asarray(paths, dtype=np.float32)


def _score(success, terminal_drift, peak_speed, terminal_height):
    """Prefer coded support; otherwise retain the least energetic release."""
    return (
        np.asarray(success, dtype=np.float32) * 1000.0
        - 100.0 * np.asarray(terminal_drift)
        - 2.0 * np.asarray(peak_speed)
        + 0.1 * np.asarray(terminal_height)
    )


def main():
    args = _parser()
    if args.num_elites >= args.num_envs:
        raise ValueError("num-elites must be smaller than num-envs")
    sys.path.insert(0, os.path.abspath(args.gear_repo))
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(
        {"headless": True, "device": "cpu", "enable_cameras": False}
    ).app
    runner = None
    try:
        import torch
        from dc_study.planning import RigidObjectMpcState, create_cpu_batched_planning_runner

        rng = np.random.default_rng(args.seed)
        runner = create_cpu_batched_planning_runner(
            task_name="HangMugOnTree-v0",
            assets_instance_paths={
                "mug": args.target_mug,
                "mug_tree": args.target_mug_tree,
            },
            num_envs=args.num_envs,
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
        seed_pose = np.asarray(args.seed_pose, dtype=np.float32)
        branch_local = _branch_frame(
            np.asarray(inputs["target_branch_points"], dtype=np.float32)
        )[1]
        tree_pose = inputs["checkpoint"]["rigid_object"]["mug_tree"]["root_pose"][0]
        branch_rotation = (
            _quat_to_matrix(tree_pose[3:7].detach().cpu().numpy()) @ branch_local
        ).astype(np.float32)

        # Parameters: target xyz, target rotation-vector xyz, clearance radial yz,
        # all in the matched branch frame. Candidate zero is always the exact seed.
        mean = np.zeros(8, dtype=np.float32)
        std = np.asarray(
            [0.020, 0.015, 0.015, 0.18, 0.18, 0.18, 0.030, 0.030],
            dtype=np.float32,
        )
        lower = np.asarray(
            [-0.050, -0.040, -0.040, -0.50, -0.50, -0.50, -0.060, -0.060]
        )
        upper = -lower
        iteration_results = []
        best = None
        best_path = None
        best_score = -np.inf
        env_ids = torch.arange(args.num_envs, device=env.device)
        zero_velocity = torch.zeros((args.num_envs, 6), device=env.device)

        for iteration in range(args.num_iterations):
            parameters = rng.normal(mean, std, size=(args.num_envs, 8)).astype(np.float32)
            parameters = np.clip(parameters, lower, upper)
            parameters[0] = 0.0
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
                env.step(action.reshape(1, -1).expand(args.num_envs, -1))
            start_pose = env.scene["mug"].data.root_pose_w[0].detach().cpu().numpy()
            start_pose[:3] -= env.scene.env_origins[0].detach().cpu().numpy()
            paths = _candidate_paths(
                start_pose, seed_pose, branch_rotation, parameters, args.path_steps
            )
            mug = env.scene["mug"]
            for step in range(args.path_steps):
                pose = torch.as_tensor(paths[:, step], device=env.device)
                pose = pose.clone()
                pose[:, :3] += env.scene.env_origins
                mug.write_root_pose_to_sim(pose, env_ids=env_ids)
                mug.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
                env.scene.write_data_to_sim()
                env.sim.forward()

            env.stage3_success.zero_()
            env._consecutive_hang_steps.zero_()
            env._stage3_anchor_mug_pos.zero_()
            initial_pose = mug.data.root_pose_w.detach().clone()
            hold = inputs["candidate"][0, 0].reshape(1, -1).expand(args.num_envs, -1).clone()
            hold[:, 6] = -0.0475
            hold[:, 13] = -0.0475
            peak_speed = torch.zeros(args.num_envs, device=env.device)
            for _ in range(args.release_steps):
                _, _, terminated, truncated, _ = env.step(hold)
                if bool(terminated.any()) or bool(truncated.any()):
                    raise RuntimeError("floating support search produced a done signal")
                peak_speed = torch.maximum(
                    peak_speed,
                    torch.linalg.vector_norm(mug.data.root_vel_w[:, :3], dim=-1),
                )
            terminal_pose = mug.data.root_pose_w.detach().clone()
            drift = torch.linalg.vector_norm(
                terminal_pose[:, :3] - initial_pose[:, :3], dim=-1
            )
            success = env.stage3_success.detach().cpu().numpy()
            scores = _score(
                success,
                drift.cpu().numpy(),
                peak_speed.cpu().numpy(),
                terminal_pose[:, 2].cpu().numpy(),
            )
            order = np.argsort(scores)[::-1]
            elites = parameters[order[: args.num_elites]]
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), np.asarray([0.002] * 3 + [0.03] * 3 + [0.003] * 2))
            winner = int(order[0])
            if float(scores[winner]) > best_score:
                best_score = float(scores[winner])
                best = parameters[winner].copy()
                best_path = paths[winner].copy()
            summary = {
                "iteration": iteration,
                "success_count": int(success.sum()),
                "best_score": float(scores[winner]),
                "best_terminal_drift_m": float(drift[winner]),
                "best_peak_speed_m_s": float(peak_speed[winner]),
                "best_parameters": parameters[winner].tolist(),
            }
            iteration_results.append(summary)
            print("FLOATING_SUPPORT_ITERATION=" + json.dumps(summary, sort_keys=True))
            if success.any():
                break

        result = {
            "status": "passed" if any(row["success_count"] for row in iteration_results) else "failed",
            "protocol": {
                "num_envs": args.num_envs,
                "iterations_run": len(iteration_results),
                "path_steps": args.path_steps,
                "release_steps": args.release_steps,
                "robot_tracking_used": False,
                "cold_reset_each_iteration": True,
            },
            "seed_pose": seed_pose.tolist(),
            "best_score": best_score,
            "best_parameters": best.tolist(),
            "iterations": iteration_results,
        }
        with open(args.result_json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        np.savez_compressed(
            args.output_npz,
            best_object_path=best_path,
            best_parameters=best,
            seed_pose=seed_pose,
            branch_rotation=branch_rotation,
        )
        print("FLOATING_SUPPORT_FINAL=" + json.dumps(result, sort_keys=True))
        if result["status"] != "passed":
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
