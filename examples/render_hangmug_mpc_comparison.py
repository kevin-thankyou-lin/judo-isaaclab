"""Render source-demo nominal and MPC controls side by side in IsaacLab."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from hangmug_grasp_keyframe_mpc import (
    HANG_SPEED_TOLERANCE_M_S,
    HANG_STABILITY_STEPS,
    TREE_INSERTION_POSITION_TOLERANCE_M,
    TREE_INSERTION_ROTATION_TOLERANCE_RAD,
    TREE_INSERTION_STABILITY_STEPS,
    _branch_frame,
    _mug_in_branch_frame,
    _quat_to_matrix,
    _rotation_matrix_error,
    _state_row,
    _tensor_tree,
    _torch_quat_multiply,
)


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
    parser.add_argument(
        "--target-mug-tree",
        help="Override the target MugTree instance recorded with the controls.",
    )
    parser.add_argument(
        "--target-mug",
        help="Override the target Mug instance recorded with the controls.",
    )
    parser.add_argument("--controls-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--friction-high", type=float, default=30.0)
    parser.add_argument("--friction-low", type=float, default=0.5)
    parser.add_argument(
        "--draw-coordinate-axes",
        action="store_true",
        help="Draw matched branch, desired EEF, and live EEF RGB axes.",
    )
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=("top_camera", "right_wrist_camera"),
        help="Camera rows to render for visual diagnosis.",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_inputs(args, device):
    import h5py
    import torch

    controls = np.load(args.controls_npz)
    nominal = np.asarray(controls["nominal"], dtype=np.float32)
    best = np.asarray(
        controls[
            "best_executed_actions"
            if "best_executed_actions" in controls.files
            else "best_sample"
        ],
        dtype=np.float32,
    )
    checkpoint_state = int(controls["checkpoint_state"])
    start_state = int(controls["start_state"])
    target_state = int(controls["target_state"])
    target_name = str(controls["target_name"])
    task_controller = str(
        controls["task_controller"]
        if "task_controller" in controls.files
        else "joint_residual"
    )
    source_mug_tree = str(
        controls["source_mug_tree"]
        if "source_mug_tree" in controls.files
        else os.path.join(args.objects_root, "MugTree", "mug_tree_000")
    )
    recorded_target_mug_tree = str(
        controls["target_mug_tree"]
        if "target_mug_tree" in controls.files
        else source_mug_tree
    )
    target_mug_tree = args.target_mug_tree or recorded_target_mug_tree
    if (
        args.target_mug_tree is not None
        and args.target_mug_tree != recorded_target_mug_tree
    ):
        raise ValueError(
            "target-mug-tree does not match the asset recorded with the controls"
        )
    source_mug = str(
        controls["source_mug"]
        if "source_mug" in controls.files
        else os.path.join(args.objects_root, "Mug", "mug_000")
    )
    recorded_target_mug = str(
        controls["target_mug"]
        if "target_mug" in controls.files
        else source_mug
    )
    target_mug = args.target_mug or recorded_target_mug
    if args.target_mug is not None and args.target_mug != recorded_target_mug:
        raise ValueError(
            "target-mug does not match the asset recorded with the controls"
        )
    tree_root_z_adjustment = float(
        controls["tree_root_z_adjustment"]
        if "tree_root_z_adjustment" in controls.files
        else 0.0
    )
    tree_offset_xyz = np.asarray(
        controls["tree_offset_xyz"]
        if "tree_offset_xyz" in controls.files
        else np.zeros(3),
        dtype=np.float32,
    )
    tree_yaw_deg = float(
        controls["tree_yaw_deg"]
        if "tree_yaw_deg" in controls.files
        else 0.0
    )
    source_branch_points = (
        np.asarray(controls["source_branch_points"], dtype=np.float32)
        if "source_branch_points" in controls.files
        and controls["source_branch_points"].size
        else None
    )
    target_branch_points = (
        np.asarray(controls["target_branch_points"], dtype=np.float32)
        if "target_branch_points" in controls.files
        and controls["target_branch_points"].size
        else None
    )
    target_eef_pose = (
        np.asarray(controls["reference_eef_poses"][-1], dtype=np.float32)
        if "reference_eef_poses" in controls.files
        and controls["reference_eef_poses"].size
        else None
    )
    target_mug_pose = (
        np.asarray(controls["target_mug_pose"][-1], dtype=np.float32)
        if "target_mug_pose" in controls.files
        and controls["target_mug_pose"].size
        else None
    )
    if nominal.shape != best.shape or nominal.ndim != 2:
        raise ValueError(
            f"Control shapes must match (horizon, action_dim): "
            f"{nominal.shape} versus {best.shape}"
        )
    with h5py.File(args.dataset, "r") as handle:
        group = handle[f"data/{args.episode}"]
        checkpoint = _tensor_tree(
            group["states"], checkpoint_state, device
        )
        tree_pose = checkpoint["rigid_object"]["mug_tree"]["root_pose"]
        tree_pose[:, 2] += tree_root_z_adjustment
        tree_pose[:, :3] += torch.as_tensor(
            tree_offset_xyz,
            dtype=tree_pose.dtype,
            device=tree_pose.device,
        )
        if tree_yaw_deg:
            half_yaw = np.deg2rad(tree_yaw_deg) / 2.0
            yaw = torch.tensor(
                [
                    np.cos(half_yaw),
                    0.0,
                    0.0,
                    np.sin(half_yaw),
                ],
                dtype=tree_pose.dtype,
                device=tree_pose.device,
            ).reshape(1, 4)
            tree_pose[:, 3:7] = _torch_quat_multiply(
                yaw, tree_pose[:, 3:7]
            )
        history = torch.as_tensor(
            group["actions"][checkpoint_state:start_state],
            dtype=torch.float32,
            device=device,
        )
        if "history_actions" in controls.files:
            history = torch.as_tensor(
                controls["history_actions"],
                dtype=torch.float32,
                device=device,
            )
            if history.shape[0] != start_state - checkpoint_state:
                raise ValueError(
                    "Recorded history_actions length does not match "
                    "checkpoint/start states"
                )
        target_reference = (
            np.asarray(
                controls["objective_reference_states"][-1],
                dtype=np.float32,
            )
            if "objective_reference_states" in controls.files
            else _state_row(group["states"], target_state)
        )
    candidate = torch.as_tensor(
        np.stack((nominal, best)), dtype=torch.float32, device=device
    )
    return {
        "checkpoint": checkpoint,
        "history": history,
        "history_control_overrides": (
            controls["history_control_overrides"].tolist()
            if "history_control_overrides" in controls.files
            else []
        ),
        "candidate": candidate,
        "checkpoint_state": checkpoint_state,
        "start_state": start_state,
        "target_state": target_state,
        "target_name": target_name,
        "task_controller": task_controller,
        "source_mug": source_mug,
        "target_mug": target_mug,
        "source_mug_tree": source_mug_tree,
        "target_mug_tree": target_mug_tree,
        "tree_root_z_adjustment": tree_root_z_adjustment,
        "tree_offset_xyz": tree_offset_xyz.tolist(),
        "tree_yaw_deg": tree_yaw_deg,
        "source_branch_points": (
            None
            if source_branch_points is None
            else source_branch_points.tolist()
        ),
        "target_branch_points": (
            None
            if target_branch_points is None
            else target_branch_points.tolist()
        ),
        "target_eef_pose": target_eef_pose,
        "target_mug_pose": target_mug_pose,
        "target_reference": target_reference,
    }


def _sample(env, step):
    import torch

    env_ids = torch.arange(env.num_envs, device=env.device)
    left = env.robot.arms["left_arm"].end_effector.is_grasping(
        "mug", env_ids=env_ids
    )
    right = env.robot.arms["right_arm"].end_effector.is_grasping(
        "mug", env_ids=env_ids
    )
    pose = env.scene["mug"].data.root_pose_w.detach().clone()
    pose[:, :3] -= env.scene.env_origins
    velocity = env.scene["mug"].data.root_vel_w.detach().clone()
    tree_pose = env.scene["mug_tree"].data.root_pose_w.detach().clone()
    tree_pose[:, :3] -= env.scene.env_origins
    return {
        "step": int(step),
        "left_grasp": left.detach().cpu().tolist(),
        "right_grasp": right.detach().cpu().tolist(),
        "stage2": env.stage2_success.detach().cpu().tolist(),
        "stage3": env.stage3_success.detach().cpu().tolist(),
        "mug_pose": pose.detach().cpu().tolist(),
        "mug_velocity": velocity.detach().cpu().tolist(),
        "tree_pose": tree_pose.detach().cpu().tolist(),
    }


def _rgb_frames(env, camera_name):
    env.sim.render()
    camera = env.scene[camera_name]
    camera.update(dt=0.0)
    rgb = camera.data.output["rgb"][:, :, :, :3].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        scale = 255.0 if float(rgb.max()) <= 1.0 else 1.0
        rgb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
    return rgb


def _draw_coordinate_axes(env, inputs, draw):
    if (
        draw is None
        or inputs["target_branch_points"] is None
        or inputs["target_eef_pose"] is None
    ):
        return
    draw.clear_lines()
    branch_origin, branch_rotation = _branch_frame(
        inputs["target_branch_points"]
    )
    tree_pose = env.scene["mug_tree"].data.root_pose_w.detach().cpu().numpy()
    eef_pose = (
        env.scene["right_arm"].data.body_pose_w[:, -1]
        .detach()
        .cpu()
        .numpy()
    )
    mug_pose = (
        env.scene["mug"].data.root_pose_w.detach().cpu().numpy()
    )
    env_origins = env.scene.env_origins.detach().cpu().numpy()
    target_pose = np.asarray(inputs["target_eef_pose"], dtype=np.float32)
    target_mug_pose = inputs["target_mug_pose"]
    starts = []
    ends = []
    colors = []
    sizes = []
    rgb = ((1.0, 0.1, 0.1, 1.0), (0.1, 1.0, 0.1, 1.0), (0.1, 0.4, 1.0, 1.0))
    for env_index in range(env.num_envs):
        tree_rotation = _quat_to_matrix(tree_pose[env_index, 3:7])
        branch_origin_w = (
            tree_pose[env_index, :3] + tree_rotation @ branch_origin
        )
        branch_rotation_w = tree_rotation @ branch_rotation
        target_origin_w = target_pose[:3] + env_origins[env_index]
        target_rotation_w = _quat_to_matrix(target_pose[3:7])
        live_origin_w = eef_pose[env_index, :3]
        live_rotation_w = _quat_to_matrix(eef_pose[env_index, 3:7])
        axes = [
            (branch_origin_w, branch_rotation_w, 0.050, 5.0),
            (target_origin_w, target_rotation_w, 0.035, 3.0),
            (live_origin_w, live_rotation_w, 0.020, 1.5),
        ]
        if target_mug_pose is not None:
            axes.extend(
                (
                    (
                        target_mug_pose[:3] + env_origins[env_index],
                        _quat_to_matrix(target_mug_pose[3:7]),
                        0.040,
                        4.0,
                    ),
                    (
                        mug_pose[env_index, :3],
                        _quat_to_matrix(mug_pose[env_index, 3:7]),
                        0.025,
                        2.0,
                    ),
                )
            )
        for origin, rotation, length, size in axes:
            for axis, color in enumerate(rgb):
                starts.append(tuple(float(value) for value in origin))
                end = origin + length * rotation[:, axis]
                ends.append(tuple(float(value) for value in end))
                colors.append(color)
                sizes.append(size)
    draw.draw_lines(starts, ends, colors, sizes)


def _write_frame(
    env,
    sample,
    frame_index,
    frames_dir,
    target_name,
    *,
    inputs,
    coordinate_draw=None,
):
    import cv2

    _draw_coordinate_axes(env, inputs, coordinate_draw)
    labels = ("SOURCE-DEMO NOMINAL", "MPC BEST SAMPLE")
    camera_rows = []
    for camera_name in inputs["camera_names"]:
        panels = []
        images = _rgb_frames(env, camera_name)
        for env_index, (image, label) in enumerate(zip(images, labels)):
            panel = image.copy()
            lines = [
                f"{label} / {camera_name}",
                f"candidate step: {sample['step']}",
                f"target: {target_name}",
                (
                    f"left grasp: {sample['left_grasp'][env_index]}  "
                    f"right grasp: {sample['right_grasp'][env_index]}"
                ),
                f"handover latched: {sample['stage2'][env_index]}",
                f"hang latched: {sample['stage3'][env_index]}",
            ]
            if coordinate_draw is not None:
                lines.append(
                    "RGB=XYZ: branch / desired+live mug / desired+live EEF"
                )
            for index, line in enumerate(lines):
                color = (
                    (70, 230, 255) if index == 0 else (245, 245, 245)
                )
                cv2.putText(
                    panel,
                    line,
                    (12, 30 + index * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            panels.append(panel)
        camera_rows.append(np.concatenate(panels, axis=1))
    combined = np.concatenate(camera_rows, axis=0)
    path = os.path.join(frames_dir, f"frame_{frame_index:06d}.png")
    if not cv2.imwrite(path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write {path}")
    return {
        "mean": float(combined.mean()),
        "std": float(combined.std()),
        "width": int(combined.shape[1]),
        "height": int(combined.shape[0]),
    }


def _encode(frames_dir, fps, output):
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(frames_dir, "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output,
        ],
        check=True,
    )


def _probe(path):
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration,size",
            "-show_entries",
            "stream=codec_name,width,height,nb_read_frames",
            "-of",
            "json",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            path,
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    stream = result["streams"][0]
    return {
        "path": os.path.abspath(path),
        "sha256": _sha256(path),
        "size_bytes": int(result["format"]["size"]),
        "duration_s": float(result["format"]["duration"]),
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_count": int(stream["nb_read_frames"]),
        "full_decode_returncode": decoded.returncode,
    }


def _quaternion_error(left, right):
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return float(
        2.0 * np.arccos(np.clip(abs(float(np.dot(left, right))), 0.0, 1.0))
    )


def _subtask_complete(sample, lane, target_name):
    if target_name == "left_grasp":
        return sample["left_grasp"][lane]
    if target_name == "right_grasp":
        return sample["right_grasp"][lane]
    if target_name == "handover_latched":
        return sample["stage2"][lane]
    if target_name == "tree_approach":
        return sample["stage2"][lane] and sample["right_grasp"][lane]
    return sample["stage3"][lane]


def _tree_geometry(traces, inputs):
    rows = np.zeros((len(traces), 2, 36), dtype=np.float32)
    rows[:, :, :7] = np.asarray([row["mug_pose"] for row in traces])
    rows[:, :, 7:13] = np.asarray(
        [row["mug_velocity"] for row in traces]
    )
    rows[:, :, 29:36] = np.asarray(
        [row["tree_pose"] for row in traces]
    )
    target_points = np.asarray(
        inputs["target_branch_points"], dtype=np.float32
    )
    source_points = np.asarray(
        inputs["source_branch_points"], dtype=np.float32
    )
    actual_position, actual_rotation = _mug_in_branch_frame(
        rows, target_points
    )
    reference_position, reference_rotation = _mug_in_branch_frame(
        np.asarray(inputs["target_reference"])[None, :], source_points
    )
    position_error = np.linalg.norm(
        actual_position - reference_position[None, :, :], axis=-1
    )
    rotation_error = _rotation_matrix_error(
        actual_rotation,
        reference_rotation[None, :, :, :],
    )
    speed = np.linalg.norm(rows[:, :, 7:10], axis=-1)
    return position_error, rotation_error, speed


def _insert_acceptance(traces, inputs):
    position_error, rotation_error, _ = _tree_geometry(traces, inputs)
    success = np.asarray(
        [
            [
                row["stage2"][lane] and row["right_grasp"][lane]
                for lane in range(2)
            ]
            for row in traces
        ],
        dtype=bool,
    )
    success &= position_error <= TREE_INSERTION_POSITION_TOLERANCE_M
    success &= rotation_error <= TREE_INSERTION_ROTATION_TOLERANCE_RAD
    window = success[-TREE_INSERTION_STABILITY_STEPS:]
    return {
        "complete": window.all(axis=0).tolist(),
        "acceptance_fraction": window.mean(axis=0).tolist(),
        "terminal_position_error_m": position_error[-1].tolist(),
        "terminal_rotation_error_rad": rotation_error[-1].tolist(),
    }


def _hang_acceptance(traces, inputs):
    position_error, rotation_error, speed = _tree_geometry(traces, inputs)
    success = np.asarray(
        [
            [
                row["stage3"][lane]
                and not row["left_grasp"][lane]
                and not row["right_grasp"][lane]
                for lane in range(2)
            ]
            for row in traces
        ],
        dtype=bool,
    )
    success &= speed <= HANG_SPEED_TOLERANCE_M_S
    return {
        "complete": success[-1].tolist(),
        "acceptance_fraction": success[-1].astype(float).tolist(),
        "task_stage3_terminal": [
            bool(value) for value in traces[-1]["stage3"]
        ],
        "task_stability_steps": HANG_STABILITY_STEPS,
        "terminal_position_error_m": position_error[-1].tolist(),
        "terminal_rotation_error_rad": rotation_error[-1].tolist(),
        "terminal_speed_m_s": speed[-1].tolist(),
    }


def main():
    args = _parser()
    sys.path.insert(0, os.path.abspath(args.gear_repo))

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(
        {"headless": True, "device": "cpu", "enable_cameras": True}
    ).app
    runner = None
    try:
        import torch
        from dc_study.planning import (
            RigidObjectMpcState,
            create_cpu_batched_planning_runner,
        )

        controls = np.load(args.controls_npz)
        recorded_tree = str(
            controls["target_mug_tree"]
            if "target_mug_tree" in controls.files
            else os.path.join(args.objects_root, "MugTree", "mug_tree_000")
        )
        recorded_mug = str(
            controls["target_mug"]
            if "target_mug" in controls.files
            else os.path.join(args.objects_root, "Mug", "mug_000")
        )
        target_tree = args.target_mug_tree or recorded_tree
        target_mug = args.target_mug or recorded_mug
        assets = {
            "mug": target_mug,
            "mug_tree": target_tree,
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
            num_envs=2,
            observation_modalities=["rgb", "proprioception"],
            enable_cameras=True,
            grasp_assist_config=assist_config,
            objects_randomization=None,
            init_joint_pos_randomization=0.0,
            enable_gripper_grasp_clamp=False,
            planning_substep_contact_sensors=True,
            camera_width=640,
            camera_height=480,
        )
        env = runner.env
        env.reset(warm_up=False, seed=args.seed)
        inputs = _load_inputs(args, env.device)
        inputs["camera_names"] = list(args.camera_names)
        coordinate_draw = None
        if args.draw_coordinate_axes:
            from isaacsim.core.utils.extensions import enable_extension

            if not enable_extension("isaacsim.util.debug_draw"):
                raise RuntimeError("Could not enable isaacsim.util.debug_draw")
            import isaacsim.util.debug_draw._debug_draw as _debug_draw

            coordinate_draw = _debug_draw.acquire_debug_draw_interface()
        runner.reset(
            inputs["checkpoint"],
            {
                "mug": RigidObjectMpcState(False),
                "mug_tree": RigidObjectMpcState(False),
            },
            is_relative=True,
        )
        for action in inputs["history"]:
            repeated = action.reshape(1, -1).expand(2, -1)
            env.step(repeated)

        traces = []
        frame_stats = []
        with tempfile.TemporaryDirectory(
            prefix="hangmug-mpc-comparison-"
        ) as frames_dir:
            sample = _sample(env, -1)
            traces.append(sample)
            frame_stats.append(
                _write_frame(
                    env,
                    sample,
                    0,
                    frames_dir,
                    inputs["target_name"],
                    inputs=inputs,
                    coordinate_draw=coordinate_draw,
                )
            )
            for step in range(inputs["candidate"].shape[1]):
                _, _, terminated, truncated, _ = env.step(
                    inputs["candidate"][:, step]
                )
                if bool(terminated.any()) or bool(truncated.any()):
                    raise RuntimeError(f"Unexpected done at candidate step {step}")
                sample = _sample(env, step)
                traces.append(sample)
                frame_stats.append(
                    _write_frame(
                        env,
                        sample,
                        step + 1,
                        frames_dir,
                        inputs["target_name"],
                        inputs=inputs,
                        coordinate_draw=coordinate_draw,
                    )
                )
            _encode(frames_dir, args.fps, args.output)
        video = _probe(args.output)
        positions = np.asarray(
            [[row["mug_pose"][lane][:3] for lane in range(2)] for row in traces]
        )
        rotations = np.asarray(
            [[row["mug_pose"][lane][3:7] for lane in range(2)] for row in traces]
        )
        translation = np.linalg.norm(positions[:, 0] - positions[:, 1], axis=1)
        rotation = np.asarray(
            [
                _quaternion_error(pair[0], pair[1])
                for pair in rotations
            ]
        )
        result = {
            "status": "passed",
            "configuration": {
                "physics": "parallel two-clone CPU PhysX scene",
                "left_lane": "source-demo nominal",
                "right_lane": "MPC best sample",
                "checkpoint_state": inputs["checkpoint_state"],
                "history_steps": int(inputs["history"].shape[0]),
                "history_control_overrides": inputs[
                    "history_control_overrides"
                ],
                "start_state": inputs["start_state"],
                "target_state": inputs["target_state"],
                "target_name": inputs["target_name"],
                "task_controller": inputs["task_controller"],
                "source_mug": inputs["source_mug"],
                "target_mug": inputs["target_mug"],
                "source_mug_tree": inputs["source_mug_tree"],
                "target_mug_tree": inputs["target_mug_tree"],
                "tree_root_z_adjustment_m": inputs[
                    "tree_root_z_adjustment"
                ],
                "tree_offset_xyz_m": inputs["tree_offset_xyz"],
                "tree_yaw_deg": inputs["tree_yaw_deg"],
                "source_branch_points_tree_local": inputs[
                    "source_branch_points"
                ],
                "target_branch_points_tree_local": inputs[
                    "target_branch_points"
                ],
                "candidate_horizon": int(inputs["candidate"].shape[1]),
                "coordinate_axes": args.draw_coordinate_axes,
                "camera_names": inputs["camera_names"],
            },
            "terminal": {
                "left_grasp": traces[-1]["left_grasp"],
                "right_grasp": traces[-1]["right_grasp"],
                "stage2": traces[-1]["stage2"],
                "stage3": traces[-1]["stage3"],
            },
            "nominal_vs_mpc_mug_divergence": {
                "translation_m_max": float(translation.max()),
                "translation_m_terminal": float(translation[-1]),
                "rotation_rad_max": float(rotation.max()),
                "rotation_rad_terminal": float(rotation[-1]),
            },
            "render": {
                "minimum_frame_std": min(row["std"] for row in frame_stats),
                "mean_pixel_range": [
                    min(row["mean"] for row in frame_stats),
                    max(row["mean"] for row in frame_stats),
                ],
            },
            "video": video,
        }
        insertion_acceptance = (
            _insert_acceptance(traces, inputs)
            if inputs["target_name"] == "inserted_held"
            and inputs["source_branch_points"] is not None
            else None
        )
        hang_acceptance = (
            _hang_acceptance(traces, inputs)
            if inputs["target_name"] == "hang_complete"
            and inputs["source_branch_points"] is not None
            else None
        )
        strict_acceptance = insertion_acceptance or hang_acceptance
        subtask_complete = {
            name: bool(
                strict_acceptance["complete"][lane]
                if strict_acceptance is not None
                else _subtask_complete(
                    traces[-1], lane, inputs["target_name"]
                )
            )
            for lane, name in enumerate(("nominal", "mpc"))
        }
        result["subtask_complete"] = subtask_complete
        if insertion_acceptance is not None:
            result["insertion_acceptance"] = insertion_acceptance
        if hang_acceptance is not None:
            result["hang_acceptance"] = hang_acceptance
        checks = {
            "parallel_lanes": env.num_envs == 2,
            "mpc_subtask_complete": subtask_complete["mpc"],
            "dynamic_frames": (
                result["render"]["mean_pixel_range"][1]
                != result["render"]["mean_pixel_range"][0]
            ),
            "h264_nonempty": (
                video["codec"] == "h264"
                and video["frame_count"] == len(frame_stats)
                and video["size_bytes"] > 0
            ),
            "fully_decodable": video["full_decode_returncode"] == 0,
        }
        result["checks"] = checks
        result["status"] = "passed" if all(checks.values()) else "failed"
        with open(args.result_json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print(
            "JUDO_ISAACLAB_HANGMUG_MPC_VIDEO="
            + json.dumps(result, sort_keys=True)
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
