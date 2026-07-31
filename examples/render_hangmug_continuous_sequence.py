"""Render staged HangMug controls in one continuously stepped Isaac process."""

import argparse
import json
import os
import sys
import tempfile
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
    parser.add_argument("--controls-npz", nargs="+", required=True)
    parser.add_argument("--target-mug", required=True)
    parser.add_argument("--target-mug-tree", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=("top_camera", "left_wrist_camera", "right_wrist_camera"),
    )
    parser.add_argument("--draw-coordinate-axes", action="store_true")
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def _renderer_args(args, controls_path):
    return argparse.Namespace(
        gear_repo=args.gear_repo,
        dataset=args.dataset,
        objects_root=args.objects_root,
        episode=args.episode,
        target_mug=args.target_mug,
        target_mug_tree=args.target_mug_tree,
        controls_npz=controls_path,
        output=args.output,
        result_json=args.result_json,
        fps=args.fps,
        friction_high=30.0,
        friction_low=0.5,
        draw_coordinate_axes=args.draw_coordinate_axes,
        camera_names=args.camera_names,
        seed=args.seed,
    )


def _validate_sequence(stages):
    checkpoint = stages[0]["checkpoint_state"]
    expected_start = stages[0]["start_state"]
    expected_history = stages[0]["history"].detach().cpu().numpy()
    boundaries = []
    for index, stage in enumerate(stages):
        if stage["checkpoint_state"] != checkpoint:
            raise ValueError("all stages must use the same checkpoint")
        if stage["start_state"] != expected_start:
            raise ValueError(
                f"stage {index} starts at {stage['start_state']}, "
                f"expected {expected_start}"
            )
        actual_history = stage["history"].detach().cpu().numpy()
        if actual_history.shape != expected_history.shape or not np.allclose(
            actual_history, expected_history, atol=1.0e-6
        ):
            raise ValueError(
                f"stage {index} history does not match prior promoted controls"
            )
        horizon = int(stage["candidate"].shape[1])
        boundaries.append(
            {
                "name": stage["target_name"],
                "start_state": stage["start_state"],
                "target_state": stage["target_state"],
                "horizon": horizon,
            }
        )
        expected_history = np.concatenate(
            (
                expected_history,
                stage["candidate"][1].detach().cpu().numpy(),
            ),
            axis=0,
        )
        expected_start += horizon
    return boundaries


def main():
    args = _parser()
    sys.path.insert(0, os.path.abspath(args.gear_repo))

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(
        {"headless": True, "device": "cpu", "enable_cameras": True}
    ).app
    runner = None
    try:
        from dc_study.planning import (
            RigidObjectMpcState,
            create_cpu_batched_planning_runner,
        )

        runner = create_cpu_batched_planning_runner(
            task_name="HangMugOnTree-v0",
            assets_instance_paths={
                "mug": args.target_mug,
                "mug_tree": args.target_mug_tree,
            },
            num_envs=2,
            observation_modalities=["rgb", "proprioception"],
            enable_cameras=True,
            grasp_assist_config={
                "left": {
                    "arm": "left_arm",
                    "target": {"object": "mug"},
                    "grasp_delay_s": 0.0,
                    "mechanism": "friction",
                    "friction": {"high": 30.0, "low": 0.5},
                }
            },
            objects_randomization=None,
            init_joint_pos_randomization=0.0,
            enable_gripper_grasp_clamp=False,
            planning_substep_contact_sensors=True,
            camera_width=640,
            camera_height=480,
        )
        env = runner.env
        env.reset(warm_up=False, seed=args.seed)
        stages = [
            render._load_inputs(
                _renderer_args(args, controls_path), env.device
            )
            for controls_path in args.controls_npz
        ]
        for stage in stages:
            stage["camera_names"] = list(args.camera_names)
        boundaries = _validate_sequence(stages)

        coordinate_draw = None
        if args.draw_coordinate_axes:
            from isaacsim.core.utils.extensions import enable_extension

            if not enable_extension("isaacsim.util.debug_draw"):
                raise RuntimeError("could not enable debug draw")
            import isaacsim.util.debug_draw._debug_draw as debug_draw

            coordinate_draw = debug_draw.acquire_debug_draw_interface()

        first = stages[0]
        runner.reset(
            first["checkpoint"],
            {
                "mug": RigidObjectMpcState(False),
                "mug_tree": RigidObjectMpcState(False),
            },
            is_relative=True,
        )
        if first["initial_task_stage"]:
            render._initialize_task_stage(
                env, first["initial_task_stage"]
            )
        for action in first["history"]:
            env.step(action.reshape(1, -1).expand(2, -1))

        traces = []
        frame_stats = []
        stage_trace_ranges = []
        global_step = 0
        with render._StreamingEncoder(args.fps, args.output) as encoder:
            sample = render._sample(env, -1)
            traces.append(sample)
            frame, stats = render._compose_frame(
                env,
                sample,
                first["target_name"],
                inputs=first,
                coordinate_draw=coordinate_draw,
            )
            encoder.write(frame)
            frame_stats.append(stats)
            for stage in stages:
                trace_start = len(traces) - 1
                for local_step in range(stage["candidate"].shape[1]):
                    _, _, terminated, truncated, _ = env.step(
                        stage["candidate"][:, local_step]
                    )
                    if bool(terminated.any()) or bool(truncated.any()):
                        raise RuntimeError(
                            f"unexpected done at global step {global_step}"
                        )
                    sample = render._sample(env, global_step)
                    traces.append(sample)
                    frame, stats = render._compose_frame(
                        env,
                        sample,
                        stage["target_name"],
                        inputs=stage,
                        coordinate_draw=coordinate_draw,
                    )
                    encoder.write(frame)
                    frame_stats.append(stats)
                    global_step += 1
                stage_trace_ranges.append(
                    {
                        "name": stage["target_name"],
                        "trace_start": trace_start,
                        "trace_end": len(traces) - 1,
                    }
                )

        stage_acceptance = []
        for stage, trace_range in zip(stages, stage_trace_ranges):
            stage_traces = traces[
                trace_range["trace_start"] : trace_range["trace_end"] + 1
            ]
            if stage["target_name"] == "inserted_held":
                acceptance = render._insert_acceptance(stage_traces, stage)
            elif stage["target_name"] == "hang_complete":
                acceptance = render._hang_acceptance(stage_traces, stage)
            else:
                complete = [
                    bool(
                        render._subtask_complete(
                            stage_traces[-1], lane, stage["target_name"]
                        )
                    )
                    for lane in range(2)
                ]
                acceptance = {"complete": complete}
            stage_acceptance.append(
                {"name": stage["target_name"], **acceptance}
            )
        final_acceptance = stage_acceptance[-1]
        video = render._probe(args.output)
        checks = {
            "one_reset": True,
            "zero_inter_stage_resets": True,
            "history_chain_exact": True,
            "target_assets_loaded": (
                stages[-1]["target_mug"] == args.target_mug
                and stages[-1]["target_mug_tree"] == args.target_mug_tree
            ),
            "coded_task_success": bool(final_acceptance["complete"][1]),
            "all_mpc_stages_complete": all(
                bool(stage["complete"][1]) for stage in stage_acceptance
            ),
            "dynamic_frames": (
                max(row["mean"] for row in frame_stats)
                != min(row["mean"] for row in frame_stats)
            ),
            "h264_nonempty": (
                video["codec"] == "h264"
                and video["frame_count"] == len(frame_stats)
                and video["size_bytes"] > 0
            ),
            "fully_decodable": video["full_decode_returncode"] == 0,
        }
        result = {
            "status": "passed" if all(checks.values()) else "failed",
            "protocol": {
                "simulator_processes": 1,
                "scene_resets": 1,
                "inter_stage_resets": 0,
                "checkpoint_state": first["checkpoint_state"],
                "total_control_steps": global_step,
                "target_mug": args.target_mug,
                "target_mug_tree": args.target_mug_tree,
                "stages": boundaries,
                "stage_trace_ranges": stage_trace_ranges,
            },
            "terminal": {
                "left_grasp": traces[-1]["left_grasp"],
                "right_grasp": traces[-1]["right_grasp"],
                "stage2": traces[-1]["stage2"],
                "stage3": traces[-1]["stage3"],
            },
            "hang_acceptance": final_acceptance,
            "stage_acceptance": stage_acceptance,
            "checks": checks,
            "video": video,
        }
        with open(args.result_json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print(
            "HANGMUG_ADAPTATION_FINAL=" + json.dumps(result, sort_keys=True)
        )
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
