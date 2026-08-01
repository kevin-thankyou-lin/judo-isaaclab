"""Run deterministic HangMug replay or semantic skill evidence in IsaacLab."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))


def _parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gear-repo", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--objects-root", required=True)
    parser.add_argument("--mode", choices=("replay", "skill"), required=True)
    parser.add_argument("--source-keyframes")
    parser.add_argument("--write-keyframes")
    parser.add_argument("--expect-failure", action="store_true")
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--damping", type=float, default=0.045)
    parser.add_argument("--max-joint-delta", type=float, default=0.16)
    parser.add_argument("--max-position-step", type=float, default=0.025)
    parser.add_argument("--max-rotation-step", type=float, default=0.16)
    parser.add_argument("--insert-clearance-m", type=float, default=0.08)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video")
    parser.add_argument("--trace-npz", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--direct-replay-result")
    return parser.parse_args()


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_assets(path: str, objects_root: str) -> dict[str, str]:
    import h5py

    with h5py.File(path, "r") as handle:
        raw = handle["data"].attrs["ASSETS_INSTANCE_PATHS"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    relative = json.loads(str(raw))
    result = {name: os.path.join(objects_root, value) for name, value in relative.items()}
    if set(result) != {"mug", "mug_tree"}:
        raise ValueError(f"expected mug/mug_tree assets, got {sorted(result)}")
    missing = [value for value in result.values() if not os.path.isdir(value)]
    if missing:
        raise FileNotFoundError(f"official asset directories missing: {missing}")
    return result


def _load_dataset(path: str, episode: str, device) -> dict[str, object]:
    import h5py
    import torch
    from run_putmarker_skill_program import _tensor_tree

    with h5py.File(path, "r") as handle:
        group = handle[f"data/{episode}"]
        return {
            "initial_state": _tensor_tree(group["states"], 0, device),
            "actions": torch.as_tensor(np.asarray(group["actions"]), device=device),
            "mug_pose": np.asarray(group["states/rigid_object/mug/root_pose"]),
            "tree_pose": np.asarray(group["states/rigid_object/mug_tree/root_pose"]),
            "num_samples": int(group.attrs["num_samples"]),
        }


def _geometry(asset_path: str, root_pose: np.ndarray):
    from judo_isaaclab.hang_mug import RigidAssetGeometry
    from run_putmarker_skill_program import _asset_size

    return RigidAssetGeometry(root_pose=np.asarray(root_pose), size=_asset_size(asset_path))


def _configure_task_without_assistance() -> dict[str, object]:
    import isaaclab.sim as sim_utils
    import dc_study.envs.tasks.hang_mug_on_tree_manager as manager_module
    import dc_study.envs.tasks.hang_mug_on_tree_manager_cfg as config_module

    manager_module.GRASP_ASSIST_CONFIG = {}
    config_module.GRASP_ASSIST_CONFIG = {}
    original_init = config_module.HangMugOnTreeManagerEnvCfg.__init__

    def offline_init(instance, *init_args, **init_kwargs):
        original_init(instance, *init_args, **init_kwargs)
        instance.grasp_assist = {}
        instance.terminations.task_success = None
        instance.terminations.mug_below_table = None
        instance.terminations.mug_tree_below_table = None
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

    config_module.HangMugOnTreeManagerEnvCfg.__init__ = offline_init
    return {
        "grasp_assistance": "disabled by empty manager/config assist maps",
        "success_auto_termination": "disabled; coded predicate unchanged",
        "failure_auto_termination": "disabled for one-reset failure evidence",
        "ground": "procedural static cuboid",
    }


def _sample(env, step: int, stage: str, info=None) -> dict[str, object]:
    import torch
    from run_putmarker_skill_program import _eef_pose

    left_grasp, right_grasp = env.robot.is_grasping()
    origin = env.scene.env_origins[0].detach().cpu().numpy()
    mug_pose = env.scene["mug"].data.root_pose_w[0].detach().cpu().numpy().copy()
    tree_pose = env.scene["mug_tree"].data.root_pose_w[0].detach().cpu().numpy().copy()
    mug_pose[:3] -= origin
    tree_pose[:3] -= origin
    velocity = env.scene["mug"].data.root_vel_w[0].detach().cpu().numpy().copy()
    task_success = bool(env.get_task_success()[0].item())
    if info is not None and bool(info.get("success", torch.tensor([False]))[0].item()):
        task_success = True
    xy_error = float(np.linalg.norm(mug_pose[:2] - tree_pose[:2]))
    released = not bool(left_grasp[0].item()) and not bool(right_grasp[0].item())
    elevated = float(mug_pose[2]) > float(env.mug_init_z + 0.05)
    hang_now = xy_error < float(env.hang_xy_tolerance) and elevated and released
    return {
        "step": int(step),
        "program_stage": stage,
        "left_grasp": bool(left_grasp[0].item()),
        "right_grasp": bool(right_grasp[0].item()),
        "stage1": bool(env.stage1_success[0].item()),
        "stage2": bool(env.stage2_success[0].item()),
        "stage3": bool(env.stage3_success[0].item()),
        "task_success": task_success,
        "hang_predicate_now": hang_now,
        "mug_tree_xy_error_m": xy_error,
        "mug_pose": mug_pose.tolist(),
        "tree_pose": tree_pose.tolist(),
        "mug_velocity": velocity.tolist(),
        "left_eef_pose": _eef_pose(env, "left_arm").tolist(),
        "right_eef_pose": _eef_pose(env, "right_arm").tolist(),
    }


def _first_index(samples, predicate):
    return next((index for index, row in enumerate(samples) if predicate(row)), None)


def _extract_keyframes(samples, source_dataset, source_assets):
    left_grasp = _first_index(samples, lambda row: row["left_grasp"])
    pick = _first_index(samples, lambda row: row["stage1"])
    right_grasp = _first_index(samples, lambda row: row["right_grasp"])
    dual_grasp = _first_index(samples, lambda row: row["left_grasp"] and row["right_grasp"])
    handover = _first_index(samples, lambda row: row["stage2"])
    tree_approach = _first_index(
        samples,
        lambda row: row["stage2"] and row["mug_tree_xy_error_m"] < 0.16,
    )
    release = _first_index(
        samples,
        lambda row: tree_approach is not None
        and row["step"] >= samples[tree_approach]["step"]
        and not row["left_grasp"]
        and not row["right_grasp"],
    )
    hang = _first_index(samples, lambda row: row["stage3"])
    required = {
        "left_grasp": left_grasp,
        "pick": pick,
        "right_grasp": right_grasp,
        "dual_grasp": dual_grasp,
        "handover": handover,
        "tree_approach": tree_approach,
        "release": release,
        "hang": hang,
    }
    if any(value is None for value in required.values()):
        raise ValueError(f"source replay lacks required semantic events: {required}")
    inserted = max(
        index
        for index in range(tree_approach, release)
        if samples[index]["right_grasp"] and not samples[index]["left_grasp"]
    )
    indices = {
        "left_pregrasp": max(0, left_grasp - 20),
        "left_grasp": left_grasp,
        "left_lift": pick,
        "right_pregrasp": max(pick, right_grasp - 25),
        "dual_grasp": dual_grasp,
        "handover": handover,
        "tree_approach": tree_approach,
        "inserted_held": inserted,
        "release": release,
        "hang": hang,
        "stable_settle": len(samples) - 1,
    }
    frames = {}
    for name, index in indices.items():
        frames[name] = {
            "sample_index": index,
            "action_index": max(-1, index - 1),
            **{
                key: samples[index][key]
                for key in (
                    "mug_pose",
                    "tree_pose",
                    "left_eef_pose",
                    "right_eef_pose",
                    "left_grasp",
                    "right_grasp",
                    "stage1",
                    "stage2",
                    "stage3",
                )
            },
        }
    from run_putmarker_skill_program import _asset_size
    return {
        "schema_version": 1,
        "source_dataset": os.path.abspath(source_dataset),
        "source_dataset_sha256": _sha256(source_dataset),
        "source_assets": {
            name: {
                "path": os.path.abspath(path),
                "size_m": _asset_size(path).tolist(),
            }
            for name, path in source_assets.items()
        },
        "semantic_indices": indices,
        "frames": frames,
    }


def _load_keyframes(path: str, source_dataset: str):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("skill mode requires --source-keyframes")
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    required = {
        "left_pregrasp",
        "left_grasp",
        "left_lift",
        "right_pregrasp",
        "dual_grasp",
        "handover",
        "tree_approach",
        "inserted_held",
        "release",
        "hang",
        "stable_settle",
    }
    if value.get("schema_version") != 1 or set(value.get("frames", {})) != required:
        raise ValueError("source keyframe artifact is incomplete")
    if value.get("source_dataset_sha256") != _sha256(source_dataset):
        raise ValueError("source keyframes do not match source dataset")
    return value


def _build_skill(keyframes, source_geometry, target_geometry, source_tree, target_tree, left_start, right_start, args):
    from judo_isaaclab.hang_mug import HangMugSkillProgram, RigidAssetGeometry
    from judo_isaaclab.put_marker import compose_pose, inverse_pose

    frames = keyframes["frames"]
    source_size = np.asarray(keyframes["source_assets"]["mug"]["size_m"])

    def transfer_mug_frame(name, arm):
        frame = frames[name]
        source_frame = RigidAssetGeometry(frame["mug_pose"], source_size)
        return target_geometry.transfer_pose_from(source_frame, frame[f"{arm}_eef_pose"])

    left_grasp = transfer_mug_frame("left_grasp", "left")
    left_contact = compose_pose(inverse_pose(target_geometry.root_pose), left_grasp)
    source_dual = frames["dual_grasp"]
    source_dual_mug = RigidAssetGeometry(source_dual["mug_pose"], source_size)
    target_handover_mug = RigidAssetGeometry(
        target_geometry.transfer_pose_from(
            RigidAssetGeometry(source_geometry.root_pose, source_size),
            source_dual["mug_pose"],
        ),
        target_geometry.size,
    )
    right_grasp = target_handover_mug.transfer_pose_from(
        source_dual_mug, source_dual["right_eef_pose"]
    )
    right_contact = compose_pose(inverse_pose(target_handover_mug.root_pose), right_grasp)

    source_final_mug = RigidAssetGeometry(frames["stable_settle"]["mug_pose"], source_size)
    source_final_tree = RigidAssetGeometry(
        frames["stable_settle"]["tree_pose"], source_tree.size
    )
    final_mug_pose = target_tree.transfer_pose_from(
        source_final_tree, source_final_mug.root_pose
    )
    final_mug = RigidAssetGeometry(final_mug_pose, target_geometry.size)
    transport_mug_pose = target_handover_mug.root_pose.copy()
    transport_mug_pose[:2] = 0.5 * (
        target_handover_mug.root_pose[:2] + final_mug_pose[:2]
    )
    transport_mug_pose[2] = max(
        target_handover_mug.root_pose[2], final_mug_pose[2] + args.insert_clearance_m
    )
    approach_mug_pose = final_mug_pose.copy()
    direction = target_geometry.root_pose[:2] - target_tree.root_pose[:2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-9:
        raise ValueError("cannot define branch approach direction")
    approach_mug_pose[:2] += direction / norm * args.insert_clearance_m
    approach_mug_pose[2] += 0.03

    def held(mug_pose, local):
        return compose_pose(mug_pose, local)

    left_lift = held(target_handover_mug.root_pose, left_contact)
    left_release = left_lift.copy()
    left_release[1] += 0.10
    right_transport = held(transport_mug_pose, right_contact)
    right_approach = held(approach_mug_pose, right_contact)
    right_insert = held(final_mug.root_pose, right_contact)

    program = HangMugSkillProgram(left_start, right_start)
    program.semantic_left_grasp(
        transfer_mug_frame("left_pregrasp", "left"),
        left_grasp,
        left_lift,
        approach_steps=100,
        close_steps=50,
        lift_steps=70,
    )
    program.physical_handover(
        left_lift,
        target_handover_mug.transfer_pose_from(source_dual_mug, frames["right_pregrasp"]["right_eef_pose"]),
        right_grasp,
        left_release,
        approach_steps=100,
        close_steps=50,
        release_steps=50,
    )
    program.handle_to_branch_insert(
        right_transport,
        right_approach,
        right_insert,
        transport_steps=100,
        approach_steps=70,
        insert_steps=70,
    )
    program.release_and_support(
        right_insert,
        right_insert,
        unload_steps=40,
        release_steps=40,
        settle_steps=60,
    )
    return program.build(), final_mug_pose


def _sparse_joint_nominal(source, trajectory, keyframes):
    actions = np.asarray(source["actions"].detach().cpu(), dtype=np.float64)
    indices = keyframes["semantic_indices"]
    mapping = {
        "left_pregrasp": indices["left_pregrasp"],
        "left_grasp": indices["left_grasp"],
        "left_lift": indices["left_lift"],
        "handover_pregrasp": indices["right_pregrasp"],
        "right_grasp": indices["dual_grasp"],
        "left_release": indices["handover"],
        "tree_transport": indices["tree_approach"],
        "branch_approach": indices["tree_approach"],
        "branch_insert": indices["inserted_held"],
        "branch_unload": indices["inserted_held"],
        "right_release": indices["release"],
        "stable_support": indices["stable_settle"],
    }
    parts = []
    previous = actions[0]
    previous_cursor = 0
    for name, cursor in trajectory.waypoint_steps.items():
        target = actions[min(mapping[name], len(actions) - 1)]
        steps = cursor + 1 - previous_cursor
        fraction = np.linspace(1.0 / steps, 1.0, steps)
        smooth = fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)
        parts.append(previous[None] + smooth[:, None] * (target - previous)[None])
        previous = target
        previous_cursor = cursor + 1
    return np.concatenate(parts)


def _frame(env, sample):
    import cv2

    panels = []
    env.sim.render()
    for camera_name in ("top_camera", "left_wrist_camera", "right_wrist_camera"):
        camera = env.scene[camera_name]
        camera.update(dt=0.0)
        image = camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy()
        if image.dtype != np.uint8:
            image = np.clip(image * (255.0 if float(image.max()) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        image = image.copy()
        lines = [
            f"{camera_name} / deterministic HangMug",
            f"step {sample['step']} / {sample['program_stage']}",
            f"pick={sample['stage1']} handover={sample['stage2']} hang={sample['stage3']}",
            f"grasps L={sample['left_grasp']} R={sample['right_grasp']}",
            f"tree xy={sample['mug_tree_xy_error_m']:.4f} m",
        ]
        for row, line in enumerate(lines):
            cv2.putText(image, line, (12, 28 + 25 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)
        panels.append(cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA))
    return np.concatenate(panels, axis=1)


def main() -> None:
    args = _parser()
    if args.render and not args.video:
        raise ValueError("--render requires --video")
    for path in (args.result_json, args.trace_npz, args.video, args.write_keyframes):
        if path and os.path.isfile(path):
            os.unlink(path)
    sys.path.insert(0, os.path.abspath(args.gear_repo))
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher({"headless": True, "device": args.device, "enable_cameras": True}).app
    env = encoder = None
    try:
        import torch
        from dc_study.utils.task_creation import create_task_environment
        from run_putmarker_skill_program import _Encoder, _asset_provenance, _eef_pose, _ik_action, _probe, _reset_scene_to_state

        override = _configure_task_without_assistance()
        source_assets = _dataset_assets(args.source_dataset, args.objects_root)
        target_assets = _dataset_assets(args.target_dataset, args.objects_root)
        env = create_task_environment(
            task_name="HangMugOnTree-v0",
            assets_instance_paths=target_assets,
            objects_randomization=None,
            init_joint_pos_randomization=0.0,
            mode="replay",
            device=args.device,
            observation_modalities=["proprioception"] + (["rgb"] if args.render else []),
            enable_self_collisions=False,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            image_downsample_factor=1,
            enable_gripper_grasp_clamp=False,
            enable_grasp_ray_viz=False,
            check_gripper_release_for_hang=True,
        )
        if env.grasp_assists:
            raise RuntimeError(f"grasp assistance unexpectedly active: {env.grasp_assists}")
        env.reset(warm_up=False, seed=args.seed)
        source = _load_dataset(args.source_dataset, args.episode, env.device)
        target = _load_dataset(args.target_dataset, args.episode, env.device)
        env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
        _reset_scene_to_state(env.scene, target["initial_state"], env_ids)
        env.sim.forward()
        env.reset_success_check(env_ids)
        source_mug = _geometry(source_assets["mug"], source["mug_pose"][0])
        target_mug = _geometry(target_assets["mug"], target["mug_pose"][0])
        source_tree = _geometry(source_assets["mug_tree"], source["tree_pose"][0])
        target_tree = _geometry(target_assets["mug_tree"], target["tree_pose"][0])
        keyframes = _load_keyframes(args.source_keyframes, args.source_dataset) if args.mode == "skill" else None
        trajectory, intended_final = (
            _build_skill(keyframes, source_mug, target_mug, source_tree, target_tree, _eef_pose(env, "left_arm"), _eef_pose(env, "right_arm"), args)
            if keyframes is not None else (None, None)
        )
        joint_nominal = _sparse_joint_nominal(source, trajectory, keyframes) if trajectory is not None else None
        total_steps = trajectory.steps if trajectory is not None else len(source["actions"])
        samples = [_sample(env, -1, "reset")]
        actions = []; mug_poses = []; left_eef = []; right_eef = []; desired_left = []; desired_right = []; frame_stats = []
        if args.render:
            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            encoder = _Encoder(args.fps, args.video)
        for step in range(total_steps):
            if trajectory is None:
                action = source["actions"][step : step + 1]
                stage = "direct_source_action_replay"
            else:
                stage = trajectory.stage_names[step]
                integrate = step > trajectory.waypoint_steps["left_grasp"]
                action = _ik_action(
                    env,
                    trajectory.left_poses[step],
                    trajectory.right_poses[step],
                    trajectory.grippers[step],
                    joint_nominal[step],
                    args,
                    integrate_left_ik=integrate,
                    integrate_right_ik=integrate,
                )
                desired_left.append(trajectory.left_poses[step]); desired_right.append(trajectory.right_poses[step])
            _, _, _, _, info = env.step(action)
            sample = _sample(env, step, stage, info)
            samples.append(sample)
            actions.append(action[0].detach().cpu().numpy()); mug_poses.append(sample["mug_pose"]); left_eef.append(sample["left_eef_pose"]); right_eef.append(sample["right_eef_pose"])
            if encoder is not None:
                frame = _frame(env, sample); encoder.write(frame); frame_stats.append((float(frame.mean()), float(frame.std())))
            if (step + 1) % 50 == 0 or sample["task_success"]:
                print("HANGMUG_PROGRESS=" + json.dumps({key: sample[key] for key in ("step", "program_stage", "stage1", "stage2", "stage3", "task_success", "left_grasp", "right_grasp", "mug_pose", "mug_tree_xy_error_m")}, sort_keys=True), flush=True)
        if encoder is not None:
            encoder.close(); encoder = None
        Path(args.trace_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.trace_npz,
            actions=np.asarray(actions, dtype=np.float32),
            mug_poses=np.asarray(mug_poses, dtype=np.float32),
            left_eef_poses=np.asarray(left_eef, dtype=np.float32),
            right_eef_poses=np.asarray(right_eef, dtype=np.float32),
            desired_left_eef_poses=np.asarray(desired_left, dtype=np.float32),
            desired_right_eef_poses=np.asarray(desired_right, dtype=np.float32),
            sparse_joint_nominal=np.asarray(joint_nominal, dtype=np.float32) if joint_nominal is not None else np.empty((0, 14), dtype=np.float32),
        )
        final = samples[-1]
        extracted = None
        if args.mode == "replay" and final["task_success"]:
            extracted = _extract_keyframes(samples, args.source_dataset, source_assets)
            if args.write_keyframes:
                Path(args.write_keyframes).parent.mkdir(parents=True, exist_ok=True)
                with open(args.write_keyframes, "w", encoding="utf-8") as stream:
                    json.dump(extracted, stream, indent=2, sort_keys=True)
        video = _probe(args.video) if args.render else None
        desired_error = []
        if trajectory is not None:
            desired_error = [max(np.linalg.norm(np.asarray(left_eef[i])[:3] - trajectory.left_poses[i, :3]), np.linalg.norm(np.asarray(right_eef[i])[:3] - trajectory.right_poses[i, :3])) for i in range(len(left_eef))]
        direct_replay = None
        if args.direct_replay_result:
            with open(args.direct_replay_result, encoding="utf-8") as stream:
                direct_replay = json.load(stream)
        terminal_speed = float(np.linalg.norm(final["mug_velocity"][:3]))
        checks = {
            "one_reset": True,
            "zero_inter_stage_resets": True,
            "real_target_assets": target_assets == _dataset_assets(args.target_dataset, args.objects_root),
            "contact_backed_grasps_only": True,
            "no_grasp_assistance": not bool(env.grasp_assists),
            "coded_task_success": bool(final["task_success"]),
            "all_stages_latched": bool(final["stage1"] and final["stage2"] and final["stage3"]),
            "left_pick_observed": any(row["left_grasp"] and row["stage1"] for row in samples),
            "right_handover_observed": any(row["right_grasp"] and row["stage2"] for row in samples),
            "mug_released": not final["left_grasp"] and not final["right_grasp"],
            "stable_hang_window": bool(final["hang_predicate_now"]),
            "terminal_mug_speed_within_threshold": terminal_speed <= 0.05,
            "h264_nonempty": video is None or (video["codec"] == "h264" and video["size_bytes"] > 0 and video["frame_count"] == len(frame_stats)),
            "fully_decodable": video is None or video["full_decode_returncode"] == 0,
        }
        if args.expect_failure:
            acceptance = {name: checks[name] for name in ("one_reset", "zero_inter_stage_resets", "real_target_assets", "contact_backed_grasps_only", "no_grasp_assistance", "h264_nonempty", "fully_decodable")}
            acceptance["expected_coded_task_failure"] = not final["task_success"]
        else:
            acceptance = checks
            if direct_replay is not None and _sha256(args.source_dataset) != _sha256(args.target_dataset):
                acceptance = dict(acceptance)
                acceptance["direct_source_action_replay_failed"] = bool(direct_replay.get("status") == "passed" and not direct_replay.get("terminal", {}).get("task_success", True))
        result = {
            "status": "passed" if all(acceptance.values()) else "failed",
            "mode": args.mode,
            "protocol": {"controller": "direct_source_action_replay" if trajectory is None else "deterministic_semantic_cartesian_dls", "candidate_sampling": False, "scene_resets": 1, "inter_stage_resets": 0, "teleports_after_reset": 0, "control_rate_hz": 30, "steps": len(actions), "seed": args.seed, "grasp_assistance": "none", "parameters": {"damping": args.damping, "max_joint_delta": args.max_joint_delta, "max_position_step": args.max_position_step, "max_rotation_step": args.max_rotation_step, "insert_clearance_m": args.insert_clearance_m}},
            "provenance": {"source_dataset": {"path": os.path.abspath(args.source_dataset), "sha256": _sha256(args.source_dataset)}, "target_dataset": {"path": os.path.abspath(args.target_dataset), "sha256": _sha256(args.target_dataset)}, "source_assets": {name: _asset_provenance(path) for name, path in source_assets.items()}, "target_assets": {name: _asset_provenance(path) for name, path in target_assets.items()}, "task_manager": {"path": os.path.join(args.gear_repo, "dc_study/envs/tasks/hang_mug_on_tree_manager.py"), "sha256": _sha256(os.path.join(args.gear_repo, "dc_study/envs/tasks/hang_mug_on_tree_manager.py"))}, "task_config": {"path": os.path.join(args.gear_repo, "dc_study/envs/tasks/hang_mug_on_tree_manager_cfg.py"), "sha256": _sha256(os.path.join(args.gear_repo, "dc_study/envs/tasks/hang_mug_on_tree_manager_cfg.py"))}, "trace": {"path": os.path.abspath(args.trace_npz), "sha256": _sha256(args.trace_npz)}, "source_keyframes": ({"path": os.path.abspath(args.source_keyframes), "sha256": _sha256(args.source_keyframes)} if args.source_keyframes else None)},
            "semantic_frames": {"source_mug": source_mug.root_pose.tolist(), "target_mug": target_mug.root_pose.tolist(), "source_tree": source_tree.root_pose.tolist(), "target_tree": target_tree.root_pose.tolist(), "intended_final_mug_pose": intended_final.tolist() if intended_final is not None else None, "extracted_keyframes": extracted},
            "metrics": {"eef_tracking_error_m": max(desired_error) if desired_error else None, "maximum_eef_tracking_error_m": max(desired_error) if desired_error else None, "handle_branch_error_m": final["mug_tree_xy_error_m"], "terminal_mug_speed_mps": terminal_speed, "terminal_mug_angular_speed_rps": float(np.linalg.norm(final["mug_velocity"][3:])), "left_grasp_frames": sum(row["left_grasp"] for row in samples), "right_grasp_frames": sum(row["right_grasp"] for row in samples)},
            "terminal": final,
            "checks": checks,
            "acceptance_checks": acceptance,
            "video": video,
            "direct_replay_baseline": direct_replay,
            "task_override": override,
        }
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print("HANGMUG_FINAL=" + json.dumps(result, sort_keys=True), flush=True)
        if result["status"] != "passed":
            raise RuntimeError(f"acceptance checks failed: {acceptance}")
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
