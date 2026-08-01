"""Run one deterministic PutMarkerInDrawer replay or semantic skill program.

The skill mode uses sparse source-demo semantic keyframes, official USD geometry,
and closed-loop damped-least-squares IK.  It never samples candidate actions and
never resets or teleports between stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


SEMANTIC_INDICES = {
    "marker_pregrasp": 70,
    "marker_grasp": 98,
    "marker_lift": 119,
    "marker_clear_hold": 179,
    "handle_pregrasp": 297,
    "handle_grasp": 305,
    "drawer_open": 419,
    "marker_collision_clear": 447,
    "marker_cavity": 490,
    "marker_release": 521,
    "left_withdraw": 541,
    "drawer_closed": 567,
    "handle_release": 607,
}


def _parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gear-repo", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--objects-root", required=True)
    parser.add_argument("--mode", choices=("replay", "skill"), required=True)
    parser.add_argument("--replay-actions-from", choices=("source", "target"), default="source")
    parser.add_argument(
        "--expect-failure",
        action="store_true",
        help="Accept a technically valid replay only when coded task success remains false.",
    )
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--damping", type=float, default=0.045)
    parser.add_argument("--max-joint-delta", type=float, default=0.16)
    parser.add_argument("--max-position-step", type=float, default=0.025)
    parser.add_argument("--max-rotation-step", type=float, default=0.16)
    parser.add_argument("--handle-pull-dls-gain", type=float, default=0.0)
    parser.add_argument("--handle-pull-joint-extension", type=float, default=-0.05)
    parser.add_argument("--drawer-placement-q-m", type=float, default=0.055)
    parser.add_argument("--drawer-pull-extra-m", type=float, default=0.010)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--draw-coordinate-axes", action="store_true")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video")
    parser.add_argument("--trace-npz", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument(
        "--direct-replay-result",
        help="Fail-closed direct-source-action replay result for this target pair.",
    )
    return parser.parse_args()


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_attr(value: object) -> dict[str, str]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _dataset_assets(path: str, objects_root: str) -> dict[str, str]:
    import h5py

    with h5py.File(path, "r") as handle:
        relative = _json_attr(handle["data"].attrs["ASSETS_INSTANCE_PATHS"])
    result = {name: os.path.join(objects_root, value) for name, value in relative.items()}
    missing = [path for path in result.values() if not os.path.isdir(path)]
    if missing:
        raise FileNotFoundError(f"official asset directories missing: {missing}")
    return result


def _tensor_tree(group, index: int, device):
    import h5py
    import torch

    result = {}
    for name, value in group.items():
        if isinstance(value, h5py.Group):
            result[name] = _tensor_tree(value, index, device)
        else:
            result[name] = torch.as_tensor(
                np.asarray(value[index : index + 1]), dtype=torch.float32, device=device
            )
    return result


def _load_dataset(path: str, episode: str, device) -> dict[str, object]:
    import h5py
    import torch

    with h5py.File(path, "r") as handle:
        group = handle[f"data/{episode}"]
        return {
            "initial_state": _tensor_tree(group["states"], 0, device),
            "actions": torch.as_tensor(np.asarray(group["actions"]), device=device),
            "eef_left": np.asarray(group["obs/datagen_info/eef_pose/left_arm"]),
            "eef_right": np.asarray(group["obs/datagen_info/eef_pose/right_arm"]),
            "marker_pose": np.asarray(group["states/rigid_object/obj_0/root_pose"]),
            "cabinet_pose": np.asarray(group["states/articulation/obj_1/root_pose"]),
            "drawer_joint": np.asarray(group["states/articulation/obj_1/joint_position"]),
            "num_samples": int(group.attrs["num_samples"]),
            "recorded_success": bool(group.attrs["success"]),
        }


def _reset_scene_to_state(scene, state, env_ids) -> None:
    """Restore the one initial state only, with zero velocities."""
    import torch

    for name, asset_state in state.get("articulation", {}).items():
        asset = scene[name]
        pose = asset_state["root_pose"].clone()
        joints = asset_state["joint_position"].clone()
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)
        asset.set_joint_position_target(joints, env_ids=env_ids)
        asset.set_joint_velocity_target(torch.zeros_like(joints), env_ids=env_ids)
    for name, asset_state in state.get("rigid_object", {}).items():
        asset = scene[name]
        pose = asset_state["root_pose"].clone()
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=pose.device), env_ids=env_ids)
    scene.write_data_to_sim()


def _asset_root_usd(path: str) -> str:
    root = Path(path)
    candidate = root / f"{root.name}.usd"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return str(candidate)


def _asset_size(path: str) -> np.ndarray:
    with open(Path(path) / "asset_size.json", encoding="utf-8") as stream:
        size = json.load(stream)["size"]
    return np.asarray([size[axis] for axis in "xyz"], dtype=np.float64)


def _drawer_geometry(asset_path: str, root_pose: np.ndarray):
    from pxr import Usd, UsdGeom, UsdPhysics

    from judo_isaaclab.put_marker import DrawerGeometry

    stage = Usd.Stage.Open(_asset_root_usd(asset_path))
    joint = UsdPhysics.PrismaticJoint.Get(stage, "/object/joints/drawer_slider_1")
    if not joint:
        raise ValueError("lower drawer joint /object/joints/drawer_slider_1 is missing")
    axis_name = str(joint.GetAxisAttr().Get()).upper()
    axis = np.zeros(3, dtype=np.float64)
    axis[{"X": 0, "Y": 1, "Z": 2}[axis_name]] = 1.0
    origin = np.asarray(joint.GetLocalPos0Attr().Get(), dtype=np.float64)

    handle_prims = [
        prim
        for prim in stage.TraverseAll()
        if "/object/link_2/collisions/" in str(prim.GetPath())
        and "handle_col" in prim.GetName()
    ]
    if not handle_prims:
        raise ValueError("lower drawer handle collision prims are missing")
    handle_points = []
    for prim in handle_prims:
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        handle_points.append(np.asarray(transform.ExtractTranslation(), dtype=np.float64))
    # The central grip bar is the handle component furthest along the pull axis;
    # the two symmetric lower-X components attach it to the drawer face.
    handle = max(handle_points, key=lambda point: float(np.dot(point, axis)))
    size = _asset_size(asset_path)
    cavity_size = np.asarray([0.72 * size[0], 0.82 * size[1], 0.22 * size[2]])
    return DrawerGeometry(
        root_pose=np.asarray(root_pose, dtype=np.float64),
        slide_axis_local=axis,
        joint_origin_local=origin,
        handle_point_local=handle,
        lower_limit_m=float(joint.GetLowerLimitAttr().Get()),
        upper_limit_m=float(joint.GetUpperLimitAttr().Get()),
        cavity_size=cavity_size,
    )


def _pose_at(matrices: np.ndarray, index: int) -> np.ndarray:
    from judo_isaaclab.put_marker import pose_from_matrix

    return pose_from_matrix(matrices[index])


def _build_skill(
    source,
    target,
    source_geometry,
    target_geometry,
    target_left_start,
    target_right_start,
    drawer_placement_q_m,
    drawer_pull_extra_m,
):
    from judo_isaaclab.put_marker import (
        PutMarkerSkillProgram,
        compose_pose,
        inverse_pose,
        quaternion_rotate,
        transfer_pose,
    )

    source_marker = np.asarray(source["marker_pose"])[0]
    target_marker = np.asarray(target["marker_pose"])[0]
    source_marker_size = _asset_size(source["assets"]["obj_0"])
    target_marker_size = _asset_size(target["assets"]["obj_0"])
    marker_scale = target_marker_size / source_marker_size
    # MimicGen's datagen EEF frame is a calibrated tool-center frame, while
    # IsaacLab exposes the arm attachment link for Jacobians. Their rigid
    # transform is recovered once at reset and applied to every sparse source
    # keyframe; this is not a trajectory binding.
    left_tool_to_attach = compose_pose(
        inverse_pose(_pose_at(target["eef_left"], 0)), target_left_start
    )
    right_tool_to_attach = compose_pose(
        inverse_pose(_pose_at(target["eef_right"], 0)), target_right_start
    )

    def source_attach(arm: str, index: int) -> np.ndarray:
        matrices = source[f"eef_{arm}"]
        offset = left_tool_to_attach if arm == "left" else right_tool_to_attach
        return compose_pose(_pose_at(matrices, index), offset)

    def left_marker(name: str) -> np.ndarray:
        index = SEMANTIC_INDICES[name]
        return transfer_pose(
            source_attach("left", index),
            source_marker,
            target_marker,
            local_position_scale=marker_scale,
        )

    def left_drawer(name: str) -> np.ndarray:
        index = SEMANTIC_INDICES[name]
        source_q = float(np.asarray(source["drawer_joint"])[index, 1])
        target_q = float(
            np.clip(
                drawer_placement_q_m,
                target_geometry.lower_limit_m,
                target_geometry.upper_limit_m,
            )
        )
        return transfer_pose(
            source_attach("left", index),
            source_geometry.drawer_frame(source_q),
            target_geometry.drawer_frame(target_q),
            local_position_scale=target_geometry.cavity_size / source_geometry.cavity_size,
        )

    def right_handle(name: str) -> np.ndarray:
        index = SEMANTIC_INDICES[name]
        source_q = float(np.asarray(source["drawer_joint"])[index, 1])
        return target_geometry.transfer_handle_pose(
            source_geometry, source_attach("right", index), source_q
        )

    handle_open = right_handle("drawer_open")
    handle_open[:3] += quaternion_rotate(
        target_geometry.root_pose[3:], target_geometry.slide_axis_local
    ) * float(drawer_pull_extra_m)

    program = PutMarkerSkillProgram(
        target_left_start, target_right_start
    )
    program.grasp_marker(
        left_marker("marker_pregrasp"),
        left_marker("marker_grasp"),
        left_marker("marker_lift"),
        approach_steps=70,
        close_steps=28,
        lift_steps=21,
    )
    program.open_drawer(
        left_marker("marker_clear_hold"),
        right_handle("handle_pregrasp"),
        right_handle("handle_grasp"),
        handle_open,
        hold_steps=60,
        approach_steps=118,
        close_steps=8,
        pull_steps=114,
    )
    program.place_marker_in_drawer(
        left_drawer("marker_collision_clear"),
        left_drawer("marker_cavity"),
        transit_steps=28,
        lower_steps=43,
    )
    program.release_marker(
        left_drawer("marker_cavity"),
        left_drawer("left_withdraw"),
        release_steps=12,
        settle_steps=27,
        withdraw_steps=12,
    )
    program.close_drawer(
        right_handle("drawer_closed"),
        right_handle("handle_release"),
        push_steps=26,
        release_steps=40,
    )
    trajectory = program.build()
    if trajectory.steps != 607:
        raise AssertionError(f"expected 607 programmed steps, got {trajectory.steps}")
    return trajectory


def _sparse_joint_nominal(source, handle_pull_joint_extension: float) -> np.ndarray:
    """Interpolate only the semantic joint keyframes, never the full demo path."""
    actions = np.asarray(source["actions"].detach().cpu(), dtype=np.float64)
    endpoints = list(SEMANTIC_INDICES.values())
    parts = []
    previous_index = 0
    previous = actions[0]
    for index in endpoints:
        target = actions[index].copy()
        if index == SEMANTIC_INDICES["drawer_open"]:
            grasp = actions[SEMANTIC_INDICES["handle_grasp"]]
            target[7:13] += float(handle_pull_joint_extension) * (
                target[7:13] - grasp[7:13]
            )
        steps = index - previous_index
        fraction = np.linspace(1.0 / steps, 1.0, steps)
        smooth = fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)
        parts.append(previous[None] + smooth[:, None] * (target - previous)[None])
        previous = target
        previous_index = index
    nominal = np.concatenate(parts)
    if nominal.shape != (607, 14):
        raise AssertionError(f"unexpected sparse joint nominal shape {nominal.shape}")
    return nominal


def _clamp_norm(value, maximum: float):
    norm = value.norm(dim=-1, keepdim=True)
    return value * (maximum / norm.clamp_min(1.0e-8)).clamp(max=1.0)


def _ik_action(
    env,
    desired_left,
    desired_right,
    grippers,
    joint_nominal,
    args,
    *,
    right_dls_gain: float = 1.0,
    integrate_right_ik: bool = False,
):
    import torch
    from isaaclab.utils.math import compute_pose_error, subtract_frame_transforms

    from judo_isaaclab.task_space import damped_least_squares, resolve_end_effector_body_index

    action = torch.as_tensor(
        joint_nominal, dtype=torch.float32, device=env.device
    ).reshape(1, 14).clone()
    for arm_name, desired, action_start, gain in (
        ("left_arm", desired_left, 0, 1.0),
        ("right_arm", desired_right, 7, right_dls_gain),
    ):
        arm = env.scene[arm_name]
        body_index = resolve_end_effector_body_index(env, arm_name)
        jacobian_index = body_index - 1 if arm.is_fixed_base else body_index
        jacobian = arm.root_physx_view.get_jacobians()[:, jacobian_index, :, :6]
        current = arm.data.body_pose_w[:, body_index]
        base = arm.data.root_pose_w
        desired = torch.as_tensor(desired, dtype=torch.float32, device=env.device).reshape(1, 7)
        desired = desired.clone()
        desired[:, :3] += env.scene.env_origins
        current_pos_b, current_quat_b = subtract_frame_transforms(
            base[:, :3], base[:, 3:7], current[:, :3], current[:, 3:7]
        )
        desired_pos_b, desired_quat_b = subtract_frame_transforms(
            base[:, :3], base[:, 3:7], desired[:, :3], desired[:, 3:7]
        )
        pos_error, rot_error = compute_pose_error(
            current_pos_b,
            current_quat_b,
            desired_pos_b,
            desired_quat_b,
            rot_error_type="axis_angle",
        )
        twist = torch.cat(
            (
                _clamp_norm(pos_error, args.max_position_step),
                _clamp_norm(rot_error, args.max_rotation_step),
            ),
            dim=-1,
        )
        delta = damped_least_squares(jacobian, twist, args.damping).clamp(
            -args.max_joint_delta, args.max_joint_delta
        )
        anchor = (
            arm.data.joint_pos[:, :6]
            if arm_name == "right_arm" and integrate_right_ik
            else action[:, action_start : action_start + 6]
        )
        targets = anchor + float(gain) * delta
        limits = arm.data.joint_pos_limits[:, :6]
        action[:, action_start : action_start + 6] = torch.maximum(
            torch.minimum(targets, limits[:, :, 1]), limits[:, :, 0]
        )
    action[:, 6] = float(grippers[0])
    action[:, 13] = float(grippers[1])
    return action


def _eef_pose(env, arm_name: str) -> np.ndarray:
    from judo_isaaclab.task_space import resolve_end_effector_body_index

    arm = env.scene[arm_name]
    index = resolve_end_effector_body_index(env, arm_name)
    pose = arm.data.body_pose_w[0, index].detach().cpu().numpy().copy()
    pose[:3] -= env.scene.env_origins[0].detach().cpu().numpy()
    return pose


def _sample(env, step: int, stage: str, info=None) -> dict[str, object]:
    import torch

    left_grasp, _ = env.robot.is_grasping()
    _, right_grasp = env.robot.is_grasping(target_object="obj_1")
    marker = env.scene["obj_0"]
    cabinet = env.scene["obj_1"]
    marker_pose = marker.data.root_pose_w[0].detach().cpu().numpy().copy()
    cabinet_pose = cabinet.data.root_pose_w[0].detach().cpu().numpy().copy()
    origin = env.scene.env_origins[0].detach().cpu().numpy()
    marker_pose[:3] -= origin
    cabinet_pose[:3] -= origin
    placed_now = bool(env._marker_placed_in_drawer()[0].item())
    task_success = bool(env.get_task_success()[0].item())
    if info is not None and bool(info.get("success", torch.tensor([False]))[0].item()):
        task_success = True
    return {
        "step": int(step),
        "program_stage": stage,
        "left_grasp": bool(left_grasp[0].item()),
        "right_handle_grasp": bool(right_grasp[0].item()),
        "stage1": bool(env.stage1_success[0].item()),
        "stage2": bool(env.stage2_success[0].item()),
        "stage3": bool(env.stage3_success[0].item()),
        "task_success": task_success,
        "placed_predicate_now": placed_now,
        "marker_pose": marker_pose.tolist(),
        "marker_velocity": marker.data.root_vel_w[0].detach().cpu().tolist(),
        "cabinet_pose": cabinet_pose.tolist(),
        "drawer_joint_position": cabinet.data.joint_pos[0].detach().cpu().tolist(),
        "drawer_joint_velocity": cabinet.data.joint_vel[0].detach().cpu().tolist(),
        "left_eef_pose": _eef_pose(env, "left_arm").tolist(),
        "right_eef_pose": _eef_pose(env, "right_arm").tolist(),
    }


class _Encoder:
    def __init__(self, fps: int, output: str):
        self.fps = fps
        self.output = output
        self.process = None

    def write(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        height, width = frame.shape[:2]
        if self.process is None:
            self.process = subprocess.Popen(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
                    "-r", str(self.fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.output,
                ],
                stdin=subprocess.PIPE,
            )
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.process is None:
            raise RuntimeError("no frames were rendered")
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("ffmpeg encoder failed")


def _quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _draw_axes(env, draw, geometry, sample, desired_left, desired_right) -> None:
    if draw is None:
        return
    draw.clear_lines()
    marker = np.asarray(sample["marker_pose"], dtype=np.float64)
    drawer_q = float(sample["drawer_joint_position"][1])
    handle = geometry.handle_frame(drawer_q)
    cavity = geometry.drawer_frame(drawer_q)
    poses = (
        (marker, 0.035, 4.0),
        (handle, 0.045, 5.0),
        (cavity, 0.060, 5.0),
        (desired_left, 0.025, 2.0),
        (desired_right, 0.025, 2.0),
    )
    origin = env.scene.env_origins[0].detach().cpu().numpy()
    starts, ends, colors, sizes = [], [], [], []
    rgb = ((1.0, 0.1, 0.1, 1.0), (0.1, 1.0, 0.1, 1.0), (0.1, 0.4, 1.0, 1.0))
    for pose, length, size in poses:
        if pose is None:
            continue
        pose = np.asarray(pose, dtype=np.float64)
        position = pose[:3] + origin
        rotation = _quat_to_matrix(pose[3:])
        for axis, color in enumerate(rgb):
            starts.append(tuple(position))
            ends.append(tuple(position + length * rotation[:, axis]))
            colors.append(color)
            sizes.append(size)
    draw.draw_lines(starts, ends, colors, sizes)


def _frame(
    env,
    sample,
    desired_left=None,
    desired_right=None,
    *,
    coordinate_draw=None,
    target_geometry=None,
) -> np.ndarray:
    import cv2

    if coordinate_draw is not None:
        _draw_axes(
            env, coordinate_draw, target_geometry, sample, desired_left, desired_right
        )
    panels = []
    env.sim.render()
    for camera_name in ("top_camera", "left_wrist_camera", "right_wrist_camera"):
        camera = env.scene[camera_name]
        camera.update(dt=0.0)
        image = camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy()
        if image.dtype != np.uint8:
            scale = 255.0 if float(image.max()) <= 1.0 else 1.0
            image = np.clip(image * scale, 0, 255).astype(np.uint8)
        image = image.copy()
        lines = [
            f"{camera_name} / deterministic nominal",
            f"step {sample['step']} / {sample['program_stage']}",
            f"pick={sample['stage1']} open={sample['stage2']} place={sample['stage3']}",
            f"left-grasp={sample['left_grasp']} right-handle={sample['right_handle_grasp']}",
            f"drawer q={max(sample['drawer_joint_position']):.4f} m",
        ]
        if desired_left is not None and desired_right is not None:
            lines.append("RGB axes: marker/cavity and desired wrists")
        for row, line in enumerate(lines):
            cv2.putText(
                image, line, (12, 28 + 25 * row), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (245, 245, 245), 1, cv2.LINE_AA,
            )
        panels.append(cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA))
    return np.concatenate(panels, axis=1)


def _probe(path: str) -> dict[str, object]:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "format=duration,size",
            "-show_entries", "stream=codec_name,width,height,nb_read_frames", "-of", "json", path,
        ],
        check=True, capture_output=True, text=True,
    )
    decoded = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-f", "null", "-"],
        capture_output=True, text=True,
    )
    data = json.loads(probe.stdout)
    stream = data["streams"][0]
    return {
        "path": os.path.abspath(path),
        "sha256": _sha256(path),
        "size_bytes": int(data["format"]["size"]),
        "duration_s": float(data["format"]["duration"]),
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_count": int(stream["nb_read_frames"]),
        "full_decode_returncode": decoded.returncode,
    }


def _configure_offline_ground() -> dict[str, object]:
    """Replace Isaac's network-backed decorative grid with a static local box.

    The box top remains at z=0, matching GroundPlaneCfg's collision surface.  It
    changes neither official task assets nor their poses, geometry, or dynamics.
    """
    import isaaclab.sim as sim_utils
    from dc_study.envs.tasks.put_marker_in_drawer_manager_cfg import (
        PutMarkerInDrawerManagerEnvCfg,
    )

    original_init = PutMarkerInDrawerManagerEnvCfg.__init__

    def offline_init(instance, *init_args, **init_kwargs):
        original_init(instance, *init_args, **init_kwargs)
        ground = instance.scene.ground
        ground.init_state.pos = (0.0, 0.0, -0.05)
        ground.spawn = sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18), roughness=0.8
            ),
            semantic_tags=[("class", "ground")],
        )

    PutMarkerInDrawerManagerEnvCfg.__init__ = offline_init
    return {
        "reason": "network-backed Isaac default_environment.usd unavailable",
        "implementation": "procedural static CuboidCfg",
        "size_m": [100.0, 100.0, 0.1],
        "position_m": [0.0, 0.0, -0.05],
        "collision_surface_z_m": 0.0,
    }


def _asset_provenance(path: str) -> dict[str, object]:
    usd = _asset_root_usd(path)
    size = Path(path) / "asset_size.json"
    asset_hash = Path(path) / ".asset_hash"
    return {
        "path": os.path.abspath(path),
        "usd": {"path": usd, "sha256": _sha256(usd)},
        "asset_size": {"path": str(size), "sha256": _sha256(size)},
        "asset_hash_file": (
            {"path": str(asset_hash), "sha256": _sha256(asset_hash), "value": asset_hash.read_text().strip()}
            if asset_hash.is_file() else None
        ),
    }


def _transition_trace(samples: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    previous = None
    for sample in samples:
        state = tuple(sample[name] for name in ("stage1", "stage2", "stage3", "task_success"))
        if state != previous or sample is samples[-1]:
            result.append(
                {
                    key: sample[key]
                    for key in (
                        "step", "program_stage", "stage1", "stage2", "stage3",
                        "task_success", "marker_pose", "drawer_joint_position",
                    )
                }
            )
            previous = state
    return result


def main() -> None:
    args = _parser()
    if args.render and not args.video:
        raise ValueError("--render requires --video")
    if not 0.0 <= args.handle_pull_dls_gain <= 1.0:
        raise ValueError("--handle-pull-dls-gain must be in [0, 1]")
    if not -0.5 <= args.handle_pull_joint_extension <= 0.5:
        raise ValueError("--handle-pull-joint-extension must be in [-0.5, 0.5]")
    # Exact task-owned outputs are removed up front so a crashed process cannot
    # leave a stale artifact that a wrapper mistakes for this run's evidence.
    for path in (args.result_json, args.trace_npz, args.video):
        if path and os.path.isfile(path):
            os.unlink(path)
    sys.path.insert(0, os.path.abspath(args.gear_repo))

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(
        # The YAM scene always instantiates calibrated tiled camera prims, even
        # when RGB is not sampled. Isaac 5.1 therefore requires this app flag.
        {"headless": True, "device": args.device, "enable_cameras": True}
    ).app
    env = None
    encoder = None
    try:
        import torch
        from dc_study.utils.task_creation import create_task_environment

        offline_ground = _configure_offline_ground()
        source_assets = _dataset_assets(args.source_dataset, args.objects_root)
        target_assets = _dataset_assets(args.target_dataset, args.objects_root)
        modalities = ["proprioception"] + (["rgb"] if args.render else [])
        env = create_task_environment(
            task_name="PutMarkerInDrawer-v0",
            assets_instance_paths=target_assets,
            objects_randomization=None,
            init_joint_pos_randomization=0.0,
            mode="replay",
            device=args.device,
            observation_modalities=modalities,
            enable_self_collisions=False,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            image_downsample_factor=1,
            enable_gripper_grasp_clamp=False,
            enable_grasp_ray_viz=False,
        )
        env.reset(warm_up=False, seed=args.seed)
        source = _load_dataset(args.source_dataset, args.episode, env.device)
        target = _load_dataset(args.target_dataset, args.episode, env.device)
        source["assets"] = source_assets
        target["assets"] = target_assets

        env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
        _reset_scene_to_state(env.scene, target["initial_state"], env_ids)
        env.sim.forward()
        env.reset_success_check(env_ids)
        source_geometry = _drawer_geometry(
            source_assets["obj_1"], np.asarray(source["cabinet_pose"])[0]
        )
        target_geometry = _drawer_geometry(
            target_assets["obj_1"], np.asarray(target["cabinet_pose"])[0]
        )
        handle_displacement_m = float(
            np.linalg.norm(
                target_geometry.handle_frame(0.0)[:3]
                - source_geometry.handle_frame(0.0)[:3]
            )
        )
        integrated_target_handle_ik = handle_displacement_m > 0.04
        trajectory = (
            _build_skill(
                source,
                target,
                source_geometry,
                target_geometry,
                _eef_pose(env, "left_arm"),
                _eef_pose(env, "right_arm"),
                args.drawer_placement_q_m,
                args.drawer_pull_extra_m,
            )
            if args.mode == "skill" else None
        )
        joint_nominal = (
            _sparse_joint_nominal(source, args.handle_pull_joint_extension)
            if trajectory is not None else None
        )
        replay_data = source if args.replay_actions_from == "source" else target
        total_steps = trajectory.steps if trajectory is not None else len(replay_data["actions"])

        actions = []
        marker_poses = []
        cabinet_joints = []
        left_eef = []
        right_eef = []
        desired_left_trace = []
        desired_right_trace = []
        samples = []
        frame_stats = []
        if args.render:
            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            encoder = _Encoder(args.fps, args.video)
        coordinate_draw = None
        if args.draw_coordinate_axes:
            if not args.render:
                raise ValueError("--draw-coordinate-axes requires --render")
            from isaacsim.core.utils.extensions import enable_extension

            if not enable_extension("isaacsim.util.debug_draw"):
                raise RuntimeError("could not enable isaacsim.util.debug_draw")
            import isaacsim.util.debug_draw._debug_draw as debug_draw

            coordinate_draw = debug_draw.acquire_debug_draw_interface()

        samples.append(_sample(env, -1, "reset"))
        for step in range(total_steps):
            if trajectory is None:
                action = replay_data["actions"][step : step + 1]
                stage = "direct_action_replay"
                desired_left = desired_right = None
            else:
                desired_left = trajectory.left_poses[step]
                desired_right = trajectory.right_poses[step]
                stage = trajectory.stage_names[step]
                right_dls_gain = (
                    args.handle_pull_dls_gain
                    if stage == "open_drawer" and step >= SEMANTIC_INDICES["handle_grasp"]
                    else 1.0
                )
                integrate_right_ik = (
                    integrated_target_handle_ik
                    and step >= SEMANTIC_INDICES["marker_lift"]
                )
                action = _ik_action(
                    env,
                    desired_left,
                    desired_right,
                    trajectory.grippers[step],
                    joint_nominal[step],
                    args,
                    right_dls_gain=right_dls_gain,
                    integrate_right_ik=integrate_right_ik,
                )
                desired_left_trace.append(desired_left)
                desired_right_trace.append(desired_right)
            _, _, terminated, truncated, info = env.step(action)
            sample = _sample(env, step, stage, info)
            samples.append(sample)
            actions.append(action[0].detach().cpu().numpy())
            marker_poses.append(sample["marker_pose"])
            cabinet_joints.append(sample["drawer_joint_position"])
            left_eef.append(sample["left_eef_pose"])
            right_eef.append(sample["right_eef_pose"])
            if encoder is not None:
                frame = _frame(
                    env,
                    sample,
                    desired_left,
                    desired_right,
                    coordinate_draw=coordinate_draw,
                    target_geometry=target_geometry,
                )
                encoder.write(frame)
                frame_stats.append((float(frame.mean()), float(frame.std())))
            if (step + 1) % 50 == 0 or sample["task_success"]:
                print(
                    "PUTMARKER_PROGRESS="
                    + json.dumps(
                        {
                            key: sample[key]
                            for key in (
                                "step", "program_stage", "stage1", "stage2", "stage3",
                                "task_success", "left_grasp", "right_handle_grasp",
                                "drawer_joint_position", "marker_pose",
                            )
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if bool(truncated[0].item()):
                raise RuntimeError(f"unexpected timeout/reset at step {step}")
            if bool(terminated[0].item()) and not sample["task_success"]:
                raise RuntimeError(f"unexpected failure termination at step {step}")

        if encoder is not None:
            encoder.close()
            encoder = None
        Path(args.trace_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.trace_npz,
            actions=np.asarray(actions, dtype=np.float32),
            marker_poses=np.asarray(marker_poses, dtype=np.float32),
            cabinet_joint_positions=np.asarray(cabinet_joints, dtype=np.float32),
            left_eef_poses=np.asarray(left_eef, dtype=np.float32),
            right_eef_poses=np.asarray(right_eef, dtype=np.float32),
            desired_left_eef_poses=np.asarray(desired_left_trace, dtype=np.float32),
            desired_right_eef_poses=np.asarray(desired_right_trace, dtype=np.float32),
            sparse_joint_nominal=(
                np.asarray(joint_nominal, dtype=np.float32)
                if joint_nominal is not None else np.empty((0, 14), dtype=np.float32)
            ),
        )
        marker_z = np.asarray(marker_poses)[:, 2]
        final = samples[-1]
        video = _probe(args.video) if args.render else None
        direct_replay_result = None
        if args.direct_replay_result:
            with open(args.direct_replay_result, encoding="utf-8") as stream:
                direct_replay_result = json.load(stream)
        checks = {
            "one_reset": True,
            "zero_inter_stage_resets": True,
            "real_target_assets": target_assets == _dataset_assets(args.target_dataset, args.objects_root),
            "coded_task_success": bool(final["task_success"]),
            "all_stages_latched": bool(final["stage1"] and final["stage2"] and final["stage3"]),
            "marker_released": not bool(final["left_grasp"]),
            "drawer_closed_exact_contract": max(abs(v) for v in final["drawer_joint_position"]) < float(env.drawer_closed_threshold_m),
            "stable_support_window": bool(final["placed_predicate_now"]),
            "h264_nonempty": (
                video is None or (
                    video["codec"] == "h264" and video["size_bytes"] > 0
                    and video["frame_count"] == len(frame_stats)
                )
            ),
            "fully_decodable": video is None or video["full_decode_returncode"] == 0,
        }
        if args.expect_failure:
            acceptance_checks = {
                name: checks[name]
                for name in (
                    "one_reset",
                    "zero_inter_stage_resets",
                    "real_target_assets",
                    "h264_nonempty",
                    "fully_decodable",
                )
            }
            acceptance_checks["expected_coded_task_failure"] = not bool(
                final["task_success"]
            )
        else:
            acceptance_checks = checks
            if direct_replay_result is not None:
                acceptance_checks = dict(acceptance_checks)
                acceptance_checks["direct_source_action_replay_failed"] = bool(
                    direct_replay_result.get("status") == "passed"
                    and not direct_replay_result.get("terminal", {}).get(
                        "task_success", True
                    )
                )
        result = {
            "status": "passed" if all(acceptance_checks.values()) else "failed",
            "mode": args.mode,
            "protocol": {
                "controller": (
                    "direct_source_action_replay"
                    if trajectory is None
                    else "semantic_keyframe_joint_spline_with_cartesian_dls"
                ),
                "candidate_sampling": False,
                "scene_resets": 1,
                "inter_stage_resets": 0,
                "teleports_after_reset": 0,
                "control_rate_hz": 30,
                "steps": total_steps,
                "seed": args.seed,
                "grasp_assistance": "none",
                "semantic_coordinate_axes_rendered": bool(args.draw_coordinate_axes),
                "offline_ground_override": offline_ground,
                "source_semantic_indices": SEMANTIC_INDICES,
                "parameters": {
                    "damping": args.damping,
                    "max_joint_delta": args.max_joint_delta,
                    "max_position_step": args.max_position_step,
                    "max_rotation_step": args.max_rotation_step,
                    "handle_pull_dls_gain": args.handle_pull_dls_gain,
                    "handle_pull_joint_extension": args.handle_pull_joint_extension,
                    "handle_displacement_m": handle_displacement_m,
                    "integrated_target_handle_ik": integrated_target_handle_ik,
                    "drawer_pull_extra_m": args.drawer_pull_extra_m,
                    "drawer_placement_q_m": args.drawer_placement_q_m,
                },
            },
            "provenance": {
                "source_dataset": {"path": os.path.abspath(args.source_dataset), "sha256": _sha256(args.source_dataset)},
                "target_dataset": {"path": os.path.abspath(args.target_dataset), "sha256": _sha256(args.target_dataset)},
                "source_assets": {name: _asset_provenance(path) for name, path in source_assets.items()},
                "target_assets": {name: _asset_provenance(path) for name, path in target_assets.items()},
                "task_manager": {
                    "path": os.path.join(args.gear_repo, "dc_study/envs/tasks/put_marker_in_drawer_manager.py"),
                    "sha256": _sha256(os.path.join(args.gear_repo, "dc_study/envs/tasks/put_marker_in_drawer_manager.py")),
                },
                "task_config": {
                    "path": os.path.join(args.gear_repo, "dc_study/envs/tasks/put_marker_in_drawer_manager_cfg.py"),
                    "sha256": _sha256(os.path.join(args.gear_repo, "dc_study/envs/tasks/put_marker_in_drawer_manager_cfg.py")),
                },
                "trace": {"path": os.path.abspath(args.trace_npz), "sha256": _sha256(args.trace_npz)},
            },
            "geometry": {
                "source": {
                    "slide_axis_local": source_geometry.slide_axis_local.tolist(),
                    "joint_origin_local": source_geometry.joint_origin_local.tolist(),
                    "handle_point_local": source_geometry.handle_point_local.tolist(),
                    "limits_m": [source_geometry.lower_limit_m, source_geometry.upper_limit_m],
                    "cavity_size_m": source_geometry.cavity_size.tolist(),
                },
                "target": {
                    "slide_axis_local": target_geometry.slide_axis_local.tolist(),
                    "joint_origin_local": target_geometry.joint_origin_local.tolist(),
                    "handle_point_local": target_geometry.handle_point_local.tolist(),
                    "limits_m": [target_geometry.lower_limit_m, target_geometry.upper_limit_m],
                    "cavity_size_m": target_geometry.cavity_size.tolist(),
                },
            },
            "stage_success_trace": _transition_trace(samples),
            "metrics": {
                "left_grasp_frames": sum(bool(row["left_grasp"]) for row in samples),
                "right_handle_grasp_frames": sum(bool(row["right_handle_grasp"]) for row in samples),
                "maximum_drawer_open_m": float(np.max(cabinet_joints)),
                "terminal_drawer_abs_max_m": float(np.max(np.abs(cabinet_joints[-1]))),
                "marker_z_min_m": float(marker_z.min()),
                "marker_z_max_m": float(marker_z.max()),
                "terminal_marker_speed_mps": float(np.linalg.norm(final["marker_velocity"][:3])),
                "terminal_marker_angular_speed_rps": float(np.linalg.norm(final["marker_velocity"][3:])),
                "support_predicate_terminal": bool(final["placed_predicate_now"]),
                "collision_proxy": {
                    "marker_below_table_termination": False,
                    "unexpected_termination": False,
                    "contact_backed_grasps_only": True,
                },
            },
            "terminal": final,
            "checks": checks,
            "acceptance_checks": acceptance_checks,
            "video": video,
            "direct_replay_baseline": direct_replay_result,
        }
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print("PUTMARKER_FINAL=" + json.dumps(result, sort_keys=True), flush=True)
        if result["status"] != "passed":
            raise RuntimeError(f"acceptance checks failed: {acceptance_checks}")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if encoder is not None:
            encoder.close()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
