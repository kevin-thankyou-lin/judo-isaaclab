"""Run direct replay or a deterministic semantic PutPot skill in IsaacLab.

Replay mode performs a one-reset free-running action replay and, on a successful
source run, extracts simulator-backed semantic keyframes.  Skill mode consumes
that fail-closed keyframe artifact and transfers the bimanual handle/support
strategy to the selected target assets without sampling, assistance, or an
inter-stage reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
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
    parser.add_argument("--mode", choices=("replay", "replay_center", "skill"), required=True)
    parser.add_argument("--source-keyframes")
    parser.add_argument("--write-keyframes")
    parser.add_argument("--expect-failure", action="store_true")
    parser.add_argument(
        "--classification-run",
        action="store_true",
        help="Accept a technically valid replay whether task success passes or fails.",
    )
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--damping", type=float, default=0.045)
    parser.add_argument("--max-joint-delta", type=float, default=0.16)
    parser.add_argument("--max-position-step", type=float, default=0.025)
    parser.add_argument("--max-rotation-step", type=float, default=0.16)
    parser.add_argument("--support-clearance-m", type=float, default=0.006)
    parser.add_argument("--transport-clearance-m", type=float, default=0.025)
    parser.add_argument("--collision-clearance-m", type=float, default=0.025)
    parser.add_argument("--transport-steps", type=int, default=180)
    parser.add_argument("--lower-steps", type=int, default=40)
    parser.add_argument("--release-steps", type=int, default=20)
    parser.add_argument("--withdraw-steps", type=int, default=30)
    parser.add_argument("--settle-steps", type=int, default=35)
    parser.add_argument("--center-repair-steps", type=int, default=60)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video")
    parser.add_argument("--trace-npz", required=True)
    parser.add_argument("--demo-hdf5")
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--direct-replay-result")
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
    if set(result) != {"pot", "cooktop"}:
        raise ValueError(f"expected pot/cooktop dataset assets, got {sorted(result)}")
    missing = [path for path in result.values() if not os.path.isdir(path)]
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
            "pot_pose": np.asarray(group["states/rigid_object/pot/root_pose"]),
            "cooktop_pose": np.asarray(group["states/rigid_object/cooktop/root_pose"]),
            "num_samples": int(group.attrs["num_samples"]),
        }


def _geometry(asset_path: str, root_pose: np.ndarray):
    from judo_isaaclab.put_pot import RigidSupportGeometry
    from run_putmarker_skill_program import _asset_size

    return RigidSupportGeometry(root_pose=np.asarray(root_pose), size=_asset_size(asset_path))


def _configure_offline_ground() -> dict[str, object]:
    import isaaclab.sim as sim_utils
    from dc_study.envs.tasks.put_pot_on_cooktop_manager_cfg import PutPotOnCooktopManagerEnvCfg

    original_init = PutPotOnCooktopManagerEnvCfg.__init__

    def offline_init(instance, *init_args, **init_kwargs):
        original_init(instance, *init_args, **init_kwargs)
        # Keep the task's exact stage latches and get_task_success predicate, but
        # do not let ManagerBasedRLEnv auto-reset on the first successful frame.
        # The evidence contract additionally requires a stable terminal window.
        instance.terminations.task_success = None
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

    PutPotOnCooktopManagerEnvCfg.__init__ = offline_init
    return {
        "reason": "network-backed Isaac default_environment.usd unavailable",
        "implementation": "procedural static CuboidCfg",
        "collision_surface_z_m": 0.0,
        "success_auto_termination": "disabled; coded task predicate unchanged",
    }


def _sample(env, step: int, stage: str, info=None) -> dict[str, object]:
    import torch
    from isaaclab.utils.math import quat_apply
    from judo_isaaclab.put_pot import cooktop_center_error_m
    from run_putmarker_skill_program import _eef_pose

    left_grasp, right_grasp = env.robot.is_grasping()
    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)

    def finger_evidence(
        arm_name: str,
    ) -> tuple[
        list[float], list[float], list[list[float]], list[list[float]]
    ]:
        gripper = env.robot.arms[arm_name].end_effector
        arm = env.scene[arm_name]
        forces, pad_fractions, pad_axes_world, pad_centers_world = [], [], [], []
        for finger in gripper.fingers:
            forces.append(
                float(finger.contact_force(gripper.default_target, env_ids)[0].item())
            )
            fraction, valid = finger.contact_pad_fraction(
                gripper.default_target, env_ids
            )
            pad_fractions.append(
                float(fraction[0].item()) if bool(valid[0].item()) else float("nan")
            )
            body_idx = arm.data.body_names.index(finger.link)
            finger_pose = arm.data.body_link_pose_w[env_ids, body_idx, :]
            tip, axis_unit, axis_length = finger._tip_base_axis(env.device)
            axis_world = quat_apply(
                finger_pose[:, 3:], axis_unit.expand(len(env_ids), -1)
            )[0]
            pad_axes_world.append(axis_world.detach().cpu().numpy().tolist())
            center_local = tip + 0.5 * axis_length * axis_unit
            center_world = finger_pose[0, :3] + quat_apply(
                finger_pose[:, 3:], center_local.expand(len(env_ids), -1)
            )[0]
            pad_centers_world.append(
                center_world.detach().cpu().numpy().tolist()
            )
        return forces, pad_fractions, pad_axes_world, pad_centers_world

    (
        left_finger_forces,
        left_pad_fractions,
        left_pad_axes,
        left_pad_centers,
    ) = finger_evidence("left_arm")
    (
        right_finger_forces,
        right_pad_fractions,
        right_pad_axes,
        right_pad_centers,
    ) = finger_evidence("right_arm")
    origin = env.scene.env_origins[0].detach().cpu().numpy()
    pot = env.scene["pot"]
    cooktop = env.scene["cooktop"]
    pot_pose = pot.data.root_pose_w[0].detach().cpu().numpy().copy()
    cooktop_pose = cooktop.data.root_pose_w[0].detach().cpu().numpy().copy()
    pot_pose[:3] -= origin
    cooktop_pose[:3] -= origin
    task_success = bool(env.get_task_success()[0].item())
    if info is not None and bool(info.get("success", torch.tensor([False]))[0].item()):
        task_success = True
    expected_z = cooktop_pose[2] + 0.5 * (env.cooktop_height + env.pot_height)
    center_error = cooktop_center_error_m(pot_pose, cooktop_pose)
    xy_error = center_error
    support_error = float(abs(pot_pose[2] - expected_z))
    qx, qy = pot_pose[4], pot_pose[5]
    orientation_error = float(np.arccos(np.clip(abs(1.0 - 2.0 * (qx * qx + qy * qy)), 0.0, 1.0)))
    on_top_now = (
        xy_error < float(env.ontop_xy_threshold)
        and support_error < 0.02
        and orientation_error < 0.2
        and not bool(left_grasp[0].item())
        and not bool(right_grasp[0].item())
    )
    return {
        "step": int(step),
        "program_stage": stage,
        "left_grasp": bool(left_grasp[0].item()),
        "right_grasp": bool(right_grasp[0].item()),
        "left_finger_forces_n": left_finger_forces,
        "left_pad_fractions": left_pad_fractions,
        "left_pad_axes_world": left_pad_axes,
        "left_pad_centers_world": left_pad_centers,
        "right_finger_forces_n": right_finger_forces,
        "right_pad_fractions": right_pad_fractions,
        "right_pad_axes_world": right_pad_axes,
        "right_pad_centers_world": right_pad_centers,
        "stage1": bool(env.stage1_success[0].item()),
        "stage2": bool(env.stage2_success[0].item()),
        "task_success": task_success,
        "on_top_predicate_now": on_top_now,
        "support_error_m": support_error,
        "center_error_m": center_error,
        "xy_error_m": xy_error,
        "orientation_error_rad": orientation_error,
        "pot_pose": pot_pose.tolist(),
        "pot_velocity": pot.data.root_vel_w[0].detach().cpu().tolist(),
        "cooktop_pose": cooktop_pose.tolist(),
        "left_eef_pose": _eef_pose(env, "left_arm").tolist(),
        "right_eef_pose": _eef_pose(env, "right_arm").tolist(),
    }


def _first(samples, predicate, name: str) -> int:
    for index, sample in enumerate(samples):
        if predicate(sample):
            return index
    raise RuntimeError(f"could not extract semantic keyframe: {name}")


def _extract_keyframes(samples, actions, source_dataset, source_assets) -> dict[str, object]:
    # samples[0] is reset and samples[action + 1] is the post-action state.
    left_close = int(np.flatnonzero(np.asarray(actions)[:, 6] > -0.04749)[0])
    right_close = int(np.flatnonzero(np.asarray(actions)[:, 13] > -0.04749)[0])
    left_grasp = _first(samples, lambda row: row["left_grasp"], "left_handle_grasp")
    both_grasp = _first(samples, lambda row: row["left_grasp"] and row["right_grasp"], "right_handle_grasp")
    pick = _first(samples, lambda row: row["stage1"], "pot_lift")
    released = _first(samples[pick:], lambda row: not row["left_grasp"] and not row["right_grasp"], "pot_release") + pick
    transported = max(range(pick, released), key=lambda i: samples[i]["pot_pose"][2])
    aligned = min(range(transported, released), key=lambda i: samples[i]["support_error_m"] + samples[i]["xy_error_m"])
    lower = max(both_grasp, released - 1)
    indices = {
        "left_pregrasp": max(0, left_close),
        "right_pregrasp": max(0, right_close),
        "left_handle_grasp": left_grasp,
        "right_handle_grasp": both_grasp,
        "pot_lift": pick,
        "pot_transport": transported,
        "support_align": aligned,
        "support_lower": lower,
        "pot_release": released,
        "stable_settle": len(samples) - 1,
    }
    frames = {}
    for name, index in indices.items():
        row = samples[index]
        frames[name] = {
            "sample_index": index,
            "action_index": max(-1, index - 1),
            "left_eef_pose": row["left_eef_pose"],
            "right_eef_pose": row["right_eef_pose"],
            "pot_pose": row["pot_pose"],
            "cooktop_pose": row["cooktop_pose"],
            "left_grasp": row["left_grasp"],
            "right_grasp": row["right_grasp"],
            "stage1": row["stage1"],
            "stage2": row["stage2"],
        }
    from run_putmarker_skill_program import _asset_size
    return {
        "schema_version": 1,
        "source_dataset": os.path.abspath(source_dataset),
        "source_dataset_sha256": _sha256(source_dataset),
        "source_assets": {
            name: {"path": os.path.abspath(path), "size_m": _asset_size(path).tolist()}
            for name, path in source_assets.items()
        },
        "semantic_indices": indices,
        "frames": frames,
    }


def _load_keyframes(path: str, source_dataset: str) -> dict[str, object]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("skill mode requires an existing --source-keyframes artifact")
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    required = {
        "left_pregrasp", "right_pregrasp", "left_handle_grasp",
        "right_handle_grasp", "pot_lift", "pot_transport",
        "support_align", "support_lower", "pot_release", "stable_settle",
    }
    if value.get("schema_version") != 1 or set(value.get("frames", {})) != required:
        raise ValueError("source keyframe artifact is incomplete or has the wrong schema")
    if value.get("source_dataset_sha256") != _sha256(source_dataset):
        raise ValueError("source keyframes do not match the selected source dataset")
    return value


def _build_skill(
    keyframes,
    source,
    target,
    source_geometry,
    target_geometry,
    source_parts,
    target_parts,
    source_components,
    target_components,
    left_start,
    right_start,
    args,
):
    from judo_isaaclab.put_marker import (
        compose_pose,
        inverse_pose,
        quaternion_rotate,
        transfer_pose,
    )
    from judo_isaaclab.put_pot import (
        HANDLE_PAD_GEOMETRIC_MARGIN_M,
        TRANSPORT_PLANNING_MARGIN_M,
        PutPotSkillProgram,
        RigidSupportGeometry,
        support_aligned_pot_pose,
    )
    from judo_isaaclab.semantic_parts import bimanual_handle_sides

    frames = keyframes["frames"]
    target_initial = target_geometry
    grasp_frame = frames["right_handle_grasp"]
    left_side, right_side = bimanual_handle_sides(
        grasp_frame["pot_pose"],
        source_parts,
        grasp_frame["left_eef_pose"],
        grasp_frame["right_eef_pose"],
    )

    def handle(parts, side):
        if side < 0:
            return parts.negative_handle_frame
        return parts.positive_handle_frame

    def handle_size(parts, side):
        if side < 0:
            return parts.negative_handle_size
        return parts.positive_handle_size

    def transfer_surface(frame_name: str, arm: str) -> np.ndarray:
        frame = frames[frame_name]
        side = left_side if arm == "left" else right_side
        source_handle = handle(source_parts, side)
        target_handle = handle(target_parts, side)
        source_frame = compose_pose(frame["pot_pose"], source_handle)
        target_frame = compose_pose(target_initial.root_pose, target_handle)
        # The authored handle center and extent change with the asset.  Scale
        # outward reach along its axis and preserve fingertip clearance from
        # the measured transverse surfaces.
        from judo_isaaclab.put_pot import (
            transfer_handle_pose_preserving_surface_clearance,
        )

        return transfer_handle_pose_preserving_surface_clearance(
            frame[f"{arm}_eef_pose"],
            source_frame,
            target_frame,
            handle_size(source_parts, side),
            handle_size(target_parts, side),
            target_parts.handle_axis,
        )

    contact_frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    grasp_poses: dict[str, np.ndarray] = {}
    grasp_geometry: dict[str, dict[str, float]] = {}

    def grasp_contact_frames(arm: str, side: int) -> tuple[np.ndarray, np.ndarray]:
        if arm in contact_frames:
            return contact_frames[arm]
        frame_name = f"{arm}_handle_grasp"
        frame = frames[frame_name]
        grasp_surface = transfer_surface(frame_name, arm)
        from judo_isaaclab.semantic_parts import infer_pot_handle_contact_frame

        source_reference = compose_pose(
            inverse_pose(frame["pot_pose"]), frame[f"{arm}_eef_pose"]
        )
        target_reference = compose_pose(
            inverse_pose(target_initial.root_pose), grasp_surface
        )
        contact_frames[arm] = (
            infer_pot_handle_contact_frame(
                source_components, source_parts, side, source_reference[:3]
            ),
            infer_pot_handle_contact_frame(
                target_components, target_parts, side, target_reference[:3]
            ),
        )
        return contact_frames[arm]

    def transfer_initial(frame_name: str, arm: str) -> np.ndarray:
        frame = frames[frame_name]
        side = left_side if arm == "left" else right_side
        surface_pose = transfer_surface(frame_name, arm)
        source_contact, target_contact = grasp_contact_frames(arm, side)
        from judo_isaaclab.put_pot import (
            transfer_handle_pose_through_contact_frames,
        )

        # The whole-handle surface transfer above deterministically bootstraps
        # the target collision segment.  Execute from the complete local
        # contact-frame correspondence so thin/curved targets do not combine a
        # segment-local rotation with an unrelated bounding-box position.
        surface_pose = transfer_handle_pose_through_contact_frames(
            frame[f"{arm}_eef_pose"],
            frames[f"{arm}_handle_grasp"]["pot_pose"],
            target_initial.root_pose,
            source_contact,
            target_contact,
        )
        if frame_name == f"{arm}_handle_grasp":
            from judo_isaaclab.put_pot import (
                balance_handle_contact_across_finger_pads,
                bounded_handle_pad_balance,
                center_handle_between_finger_pads,
                geometry_conditioned_handle_pad_depth,
                geometry_conditioned_handle_balance_limit,
                handle_finger_pad_depth_imbalance,
                handle_jaw_center_offset_m,
                seat_handle_inside_finger_pads,
            )
            from judo_isaaclab.put_marker import quaternion_rotate

            boundary = (
                target_parts.body_xy_min[target_parts.handle_axis]
                if side < 0
                else target_parts.body_xy_max[target_parts.handle_axis]
            )
            handle_points = np.concatenate(
                [
                    points
                    for points in target_components
                    if (
                        np.min(points[:, target_parts.handle_axis])
                        < boundary - 1.0e-4
                        if side < 0
                        else np.max(points[:, target_parts.handle_axis])
                        > boundary + 1.0e-4
                    )
                ]
            )
            jaw_center_offset = handle_jaw_center_offset_m(
                surface_pose, target_initial.root_pose, handle_points
            )
            surface_pose = center_handle_between_finger_pads(
                surface_pose, jaw_center_offset
            )
            target_handle_world = compose_pose(
                target_initial.root_pose, handle(target_parts, side)
            )
            pad_axis_world = quaternion_rotate(
                surface_pose[3:], np.asarray([0.0, 0.0, 1.0])
            )
            pad_axis_in_handle_frame = quaternion_rotate(
                inverse_pose(target_handle_world)[3:], pad_axis_world
            )

            pad_depth = geometry_conditioned_handle_pad_depth(
                handle_size(source_parts, side),
                handle_size(target_parts, side),
                target_parts.handle_axis,
                pad_axis_in_handle_frame,
            )
            surface_pose = seat_handle_inside_finger_pads(
                surface_pose, pad_depth
            )
            predicted_imbalance = handle_finger_pad_depth_imbalance(
                surface_pose, target_initial.root_pose, handle_points
            )
            balance_limit = geometry_conditioned_handle_balance_limit(
                handle_size(source_parts, side),
                handle_size(target_parts, side),
                target_parts.handle_axis,
                predicted_imbalance,
            )
            relative_balance = bounded_handle_pad_balance(
                predicted_imbalance,
                balance_limit,
            )
            surface_pose = balance_handle_contact_across_finger_pads(
                surface_pose, relative_balance
            )
            grasp_geometry[arm] = {
                "handle_side": int(side),
                "pad_depth_m": pad_depth,
                "jaw_center_offset_m": jaw_center_offset,
                "balance_limit_m": balance_limit,
                "predicted_pad_imbalance_m": predicted_imbalance,
                "relative_balance_m": relative_balance,
            }
            grasp_poses[arm] = surface_pose.copy()
        else:
            from judo_isaaclab.put_pot import expand_handle_pregrasp_clearance

            surface_pose = expand_handle_pregrasp_clearance(
                surface_pose,
                grasp_poses[arm],
                handle_size(source_parts, side),
                handle_size(target_parts, side),
                target_parts.handle_axis,
            )
        return surface_pose

    left_grasp = transfer_initial("left_handle_grasp", "left")
    right_grasp = transfer_initial("right_handle_grasp", "right")
    left_pregrasp = transfer_initial("left_pregrasp", "left")
    right_pregrasp = transfer_initial("right_pregrasp", "right")
    from judo_isaaclab.put_pot import (
        geometry_conditioned_target_handle_symmetry,
    )

    if geometry_conditioned_target_handle_symmetry(
        source_parts.negative_handle_size,
        source_parts.positive_handle_size,
        target_parts.negative_handle_size,
        target_parts.positive_handle_size,
        target_parts.handle_axis,
    ):
        right_handle_world = compose_pose(
            target_initial.root_pose, handle(target_parts, right_side)
        )
        left_handle_world = compose_pose(
            target_initial.root_pose, handle(target_parts, left_side)
        )
        from judo_isaaclab.put_pot import (
            mirror_handle_position_in_receiving_jaw_frame,
        )

        boundary = target_parts.body_xy_min[target_parts.handle_axis]
        left_handle_points = np.concatenate(
            [
                points
                for points in target_components
                if np.min(points[:, target_parts.handle_axis])
                < boundary - 1.0e-4
            ]
        )
        mirrored_left_grasp, post_mirror_jaw_offset = (
            mirror_handle_position_in_receiving_jaw_frame(
                right_grasp,
                right_handle_world,
                left_grasp,
                left_handle_world,
                target_initial.root_pose,
                left_handle_points,
            )
        )
        mirrored_left_pregrasp = transfer_pose(
            right_pregrasp, right_handle_world, left_handle_world
        )
        receiving_jaw_correction = mirrored_left_grasp[:3] - transfer_pose(
            right_grasp, right_handle_world, left_handle_world
        )[:3]
        left_grasp = mirrored_left_grasp
        left_pregrasp[:3] = (
            mirrored_left_pregrasp[:3] + receiving_jaw_correction
        )
        grasp_poses["left"] = left_grasp.copy()
        grasp_geometry["left"]["target_symmetric_position_from"] = "right"
        grasp_geometry["left"]["post_mirror_jaw_center_offset_m"] = (
            post_mirror_jaw_offset
        )
    # Anchor both contact transforms to the target pot at the transferred
    # grasp.  This makes the first smooth-transport sample continuous for every
    # asset scale instead of switching to a source-root convention.
    contact_root = target_initial.root_pose
    left_contact_world = left_grasp
    right_contact_world = right_grasp
    left_contact_local = compose_pose(inverse_pose(contact_root), left_contact_world)
    right_contact_local = compose_pose(inverse_pose(contact_root), right_contact_world)

    target_cooktop = RigidSupportGeometry(
        target["cooktop_pose"][0], target["cooktop_size"]
    )
    final_pot_pose = support_aligned_pot_pose(
        target_initial,
        target_cooktop,
        xy_offset_local=(0.0, 0.0),
        clearance_m=args.support_clearance_m,
    )

    def held(pot_pose, local):
        return compose_pose(pot_pose, local)

    from judo_isaaclab.put_pot import (
        CONTACT_FEEDBACK_HORIZON_STEPS,
        MISSING_FINGER_CONTACT_SETTLE_STEPS,
        geometry_conditioned_grasp_hold_steps,
        geometry_conditioned_peer_contact_transfer,
        geometry_conditioned_right_first_close,
        geometry_conditioned_transport_steps,
        geometry_conditioned_vertical_rise_fraction,
    )

    right_first_close = geometry_conditioned_right_first_close(
        handle_size(source_parts, left_side),
        handle_size(target_parts, left_side),
        target_parts.handle_axis,
        grasp_geometry["left"]["predicted_pad_imbalance_m"],
    )
    grasp_geometry["left"]["right_first_close"] = right_first_close
    peer_contact_hold_steps = (
        MISSING_FINGER_CONTACT_SETTLE_STEPS
        if geometry_conditioned_peer_contact_transfer(
            handle_size(target_parts, left_side),
            handle_size(target_parts, right_side),
            grasp_geometry["left"]["predicted_pad_imbalance_m"],
        )
        else 0
    )
    grasp_geometry["left"]["peer_contact_hold_steps"] = (
        peer_contact_hold_steps
    )
    if right_first_close:
        approach = left_pregrasp[:3] - left_grasp[:3]
        approach_norm = float(np.linalg.norm(approach))
        if approach_norm <= 1.0e-9:
            raise ValueError("left pregrasp and grasp positions must be distinct")
        transverse = [
            axis for axis in range(3) if axis != target_parts.handle_axis
        ]
        regrasp_clearance = (
            0.5
            * max(
                float(handle_size(target_parts, left_side)[axis])
                for axis in transverse
            )
            + args.collision_clearance_m
        )
        left_pregrasp[:3] += regrasp_clearance * approach / approach_norm
        approach_world = left_grasp[:3] - left_pregrasp[:3]
        approach_local = quaternion_rotate(
            inverse_pose(target_initial.root_pose)[3:], approach_world
        )
        grasp_geometry["left"]["regrasp_clearance_m"] = regrasp_clearance
        grasp_geometry["left"]["regrasp_approach_local_m"] = (
            approach_local.tolist()
        )

    transport_final_pot = final_pot_pose.copy()
    supported_center_slide = bool(right_first_close)
    if supported_center_slide:
        from judo_isaaclab.put_pot import support_boundary_staging_pose

        transport_final_pot = support_boundary_staging_pose(
            target_initial.root_pose,
            final_pot_pose,
            target_cooktop,
            support_inset_m=args.support_clearance_m,
        )
    grasp_geometry["left"]["supported_center_slide"] = supported_center_slide
    grasp_geometry["left"]["support_unload_m"] = (
        args.support_clearance_m + HANDLE_PAD_GEOMETRIC_MARGIN_M
        if supported_center_slide
        else 0.0
    )
    grasp_geometry["left"]["support_staging_offset_m"] = float(
        np.linalg.norm(transport_final_pot[:2] - final_pot_pose[:2])
    )
    transport_target = transport_final_pot.copy()
    transport_target[2] += (
        max(args.transport_clearance_m, args.collision_clearance_m)
        + TRANSPORT_PLANNING_MARGIN_M
    )
    left_lower = held(transport_final_pot, left_contact_local)
    right_lower = held(transport_final_pot, right_contact_local)
    left_withdraw = left_lower.copy()
    left_withdraw[:3] += np.asarray([0.0, 0.08, 0.12])
    right_center = held(final_pot_pose, right_contact_local)
    right_withdraw = right_center.copy()
    right_withdraw[:3] += np.asarray([0.0, -0.08, 0.12])

    grasp_hold_steps = max(
        geometry_conditioned_grasp_hold_steps(
            30,
            handle_size(source_parts, side),
            handle_size(target_parts, side),
            target_parts.handle_axis,
        )
        for side in (left_side, right_side)
    )
    for geometry in grasp_geometry.values():
        geometry["grasp_hold_steps"] = grasp_hold_steps

    transport_steps = max(
        geometry_conditioned_transport_steps(
            args.transport_steps,
            handle_size(source_parts, side),
            handle_size(target_parts, side),
            target_parts.handle_axis,
        )
        for side in (left_side, right_side)
    )
    center_slide_steps = max(
        geometry_conditioned_transport_steps(
            args.center_repair_steps,
            handle_size(source_parts, side),
            handle_size(target_parts, side),
            target_parts.handle_axis,
        )
        for side in (left_side, right_side)
    )
    grasp_geometry["left"]["center_slide_steps"] = center_slide_steps
    transport_vertical_rise_fraction = (
        geometry_conditioned_vertical_rise_fraction(
            transport_steps, peer_contact_hold_steps
        )
    )
    grasp_geometry["left"]["transport_vertical_rise_fraction"] = (
        transport_vertical_rise_fraction
    )
    transport_frontload_horizontal_axis = 0 if peer_contact_hold_steps else None
    grasp_geometry["left"]["transport_frontload_horizontal_axis"] = (
        transport_frontload_horizontal_axis
    )

    program = PutPotSkillProgram(left_start, right_start)
    program.bimanual_handle_grasp(
        left_pregrasp,
        right_pregrasp,
        left_grasp,
        right_grasp,
        approach_steps=110,
        left_close_steps=grasp_hold_steps if right_first_close else 60,
        right_close_steps=60 if right_first_close else grasp_hold_steps,
        simultaneous=not right_first_close,
        right_first=right_first_close,
        contact_hold_steps=peer_contact_hold_steps,
    )
    transport = program.smooth_bimanual_transport_to_center(
        target_initial.root_pose,
        transport_target,
        left_contact_local,
        right_contact_local,
        target_geometry.size,
        target_cooktop,
        steps=transport_steps,
        collision_clearance_m=args.collision_clearance_m,
        vertical_rise_fraction=transport_vertical_rise_fraction,
        frontload_horizontal_axis=transport_frontload_horizontal_axis,
    )
    if supported_center_slide:
        program.supported_center_slide_and_settle(
            left_lower,
            right_lower,
            left_withdraw,
            right_center,
            right_withdraw,
            lower_steps=args.lower_steps,
            left_release_steps=args.release_steps,
            center_steps=center_slide_steps,
            right_release_steps=args.release_steps,
            withdraw_steps=args.withdraw_steps,
            settle_steps=args.settle_steps,
        )
    else:
        program.short_lower_release_and_settle(
            left_lower,
            right_lower,
            left_withdraw,
            right_withdraw,
            lower_steps=args.lower_steps,
            release_steps=args.release_steps,
            withdraw_steps=args.withdraw_steps,
            settle_steps=args.settle_steps,
        )
    trajectory = program.build()
    from judo_isaaclab.put_pot import cartesian_smoothness_metrics

    transport_start = max(
        trajectory.waypoint_steps["left_handle_grasp"],
        trajectory.waypoint_steps["right_handle_grasp"],
    ) + 1
    transport_end = trajectory.waypoint_steps["smooth_transport"]
    plan_metrics = cartesian_smoothness_metrics(
        trajectory.left_poses[transport_start : transport_end + 1],
        trajectory.right_poses[transport_start : transport_end + 1],
    )
    plan_metrics.update(
        {
            "start_step": transport_start,
            "end_step": transport_end,
            "minimum_cooktop_clearance_m": transport.minimum_cooktop_clearance_m,
            "cooktop_overlap_samples": transport.cooktop_overlap_samples,
            "vertical_rise_steps": transport.vertical_rise_steps,
        }
    )
    return (
        trajectory,
        final_pot_pose,
        transport_final_pot,
        plan_metrics,
        grasp_geometry,
    )


def _build_center_repair(sample, args):
    """Slide an already-supported, right-held pot to center before release."""
    from judo_isaaclab.put_marker import SkillTrajectory, compose_pose, interpolate_poses, inverse_pose

    pot_pose = np.asarray(sample["pot_pose"], dtype=np.float64)
    cooktop_pose = np.asarray(sample["cooktop_pose"], dtype=np.float64)
    left_pose = np.asarray(sample["left_eef_pose"], dtype=np.float64)
    right_pose = np.asarray(sample["right_eef_pose"], dtype=np.float64)
    right_contact = compose_pose(inverse_pose(pot_pose), right_pose)
    centered_pot = pot_pose.copy()
    centered_pot[:2] = cooktop_pose[:2]
    right_center = compose_pose(centered_pot, right_contact)
    right_withdraw = right_center.copy()
    right_withdraw[:3] += np.asarray([0.0, -0.08, 0.12])

    center = int(args.center_repair_steps)
    release = int(args.release_steps)
    withdraw = int(args.withdraw_steps)
    settle = int(args.settle_steps)
    left = np.repeat(left_pose[None], center + release + withdraw + settle, axis=0)
    right = np.concatenate(
        (
            interpolate_poses(right_pose, right_center, center),
            np.repeat(right_center[None], release, axis=0),
            interpolate_poses(right_center, right_withdraw, withdraw),
            np.repeat(right_withdraw[None], settle, axis=0),
        )
    )
    grippers = np.empty((len(left), 2), dtype=np.float64)
    grippers[:center] = (-0.0475, 0.0)
    grippers[center:] = (-0.0475, -0.0475)
    stages = (
        ["supported_center_repair"] * center
        + ["unload_release"] * release
        + ["stable_settle"] * (withdraw + settle)
    )
    return SkillTrajectory(
        left_poses=left,
        right_poses=right,
        grippers=grippers,
        stage_names=stages,
        waypoint_steps={
            "center_slide": center - 1,
            "pot_release": center + release - 1,
            "bimanual_withdraw": center + release + withdraw - 1,
            "stable_settle": len(left) - 1,
        },
    )


def _sparse_joint_nominal(source, trajectory, keyframes) -> np.ndarray:
    actions = np.asarray(source["actions"].detach().cpu(), dtype=np.float64)
    source_indices = keyframes["semantic_indices"]
    mapping = {
        "bimanual_pregrasp": (source_indices["left_pregrasp"], source_indices["right_pregrasp"]),
        "left_handle_grasp": (source_indices["left_handle_grasp"], source_indices["right_pregrasp"]),
        "right_handle_grasp": (source_indices["right_handle_grasp"], source_indices["right_handle_grasp"]),
        "bimanual_contact_hold": (source_indices["left_handle_grasp"], source_indices["right_handle_grasp"]),
        "smooth_transport": (source_indices["support_align"], source_indices["support_align"]),
        "pot_lift": (source_indices["pot_lift"], source_indices["pot_lift"]),
        "pot_transport": (source_indices["pot_transport"], source_indices["pot_transport"]),
        "support_align": (source_indices["support_align"], source_indices["support_align"]),
        "support_lower": (source_indices["support_lower"], source_indices["support_lower"]),
        "pot_unload": (source_indices["support_lower"], source_indices["support_lower"]),
        "left_unload_release": (source_indices["stable_settle"], source_indices["support_lower"]),
        "center_slide": (source_indices["stable_settle"], source_indices["support_align"]),
        "pot_release": (source_indices["pot_release"], source_indices["pot_release"]),
        "bimanual_withdraw": (source_indices["stable_settle"], source_indices["stable_settle"]),
        "stable_settle": (source_indices["stable_settle"], source_indices["stable_settle"]),
    }
    if (
        trajectory.waypoint_steps["right_handle_grasp"]
        < trajectory.waypoint_steps["left_handle_grasp"]
    ):
        mapping["right_handle_grasp"] = (
            source_indices["left_pregrasp"],
            source_indices["right_handle_grasp"],
        )
        mapping["left_handle_grasp"] = (
            source_indices["left_handle_grasp"],
            source_indices["right_handle_grasp"],
        )
    parts = []
    previous = actions[0]
    previous_cursor = 0
    for name, cursor in trajectory.waypoint_steps.items():
        left_index, right_index = mapping[name]
        target = np.concatenate(
            (
                actions[min(left_index, len(actions) - 1), :7],
                actions[min(right_index, len(actions) - 1), 7:],
            )
        )
        steps = cursor + 1 - previous_cursor
        fraction = np.linspace(1.0 / steps, 1.0, steps)
        smooth = fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)
        parts.append(previous[None] + smooth[:, None] * (target - previous)[None])
        previous = target
        previous_cursor = cursor + 1
    result = np.concatenate(parts)
    if result.shape != (trajectory.steps, 14):
        raise AssertionError(f"unexpected nominal shape: {result.shape}")
    return result


def _frame(env, sample) -> np.ndarray:
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
            f"{camera_name} / deterministic PutPot",
            f"step {sample['step']} / {sample['program_stage']}",
            f"pick={sample['stage1']} place={sample['stage2']}",
            f"grasps L={sample['left_grasp']} R={sample['right_grasp']}",
            f"support dz={sample['support_error_m']:.4f} center={sample['center_error_m']:.4f} m",
        ]
        for row, line in enumerate(lines):
            cv2.putText(image, line, (12, 28 + 25 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
        panels.append(cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA))
    return np.concatenate(panels, axis=1)


def _transition_trace(samples):
    result, previous = [], None
    for sample in samples:
        state = (sample["stage1"], sample["stage2"], sample["task_success"])
        if state != previous or sample is samples[-1]:
            result.append({key: sample[key] for key in ("step", "program_stage", "stage1", "stage2", "task_success", "left_grasp", "right_grasp", "pot_pose", "support_error_m", "center_error_m", "xy_error_m")})
            previous = state
    return result


def main() -> None:
    args = _parser()
    if args.render and not args.video:
        raise ValueError("--render requires --video")
    if args.mode in {"skill", "replay_center"} and not args.source_keyframes:
        raise ValueError(f"{args.mode} mode requires --source-keyframes")
    if (
        args.support_clearance_m < 0.0
        or args.transport_clearance_m <= 0.0
        or args.collision_clearance_m < 0.0
        or min(
            args.transport_steps,
            args.lower_steps,
            args.release_steps,
            args.withdraw_steps,
            args.settle_steps,
        ) < 1
    ):
        raise ValueError("support/transport clearances are invalid")
    for path in (args.result_json, args.trace_npz, args.video, args.write_keyframes):
        if path and os.path.isfile(path):
            os.unlink(path)
    # Validate cheap dataset/asset provenance before the expensive app launch.
    source_assets = _dataset_assets(args.source_dataset, args.objects_root)
    target_assets = _dataset_assets(args.target_dataset, args.objects_root)
    sys.path.insert(0, os.path.abspath(args.gear_repo))
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher({"headless": True, "device": args.device, "enable_cameras": True}).app
    env = encoder = None
    try:
        import torch
        from dc_study.utils.task_creation import create_task_environment
        from run_putmarker_skill_program import (
            _Encoder, _asset_provenance, _eef_pose, _ik_action, _probe, _reset_scene_to_state,
        )

        offline_ground = _configure_offline_ground()
        env = create_task_environment(
            task_name="PutPotOnCooktop-v0",
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
        )
        env.reset(warm_up=False, seed=args.seed)
        source = _load_dataset(args.source_dataset, args.episode, env.device)
        target = _load_dataset(args.target_dataset, args.episode, env.device)
        target["cooktop_size"] = _geometry(target_assets["cooktop"], target["cooktop_pose"][0]).size
        env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
        _reset_scene_to_state(env.scene, target["initial_state"], env_ids)
        env.sim.forward()
        env.reset_success_check(env_ids)
        source_geometry = _geometry(source_assets["pot"], source["pot_pose"][0])
        target_geometry = _geometry(target_assets["pot"], target["pot_pose"][0])
        from semantic_asset_geometry import collision_components, jsonable, pot_parts

        source_components = collision_components(source_assets["pot"])
        target_components = collision_components(target_assets["pot"])
        source_parts = pot_parts(source_assets["pot"])
        target_parts = pot_parts(target_assets["pot"])
        target_cooktop_geometry = _geometry(
            target_assets["cooktop"], target["cooktop_pose"][0]
        )
        keyframes = _load_keyframes(args.source_keyframes, args.source_dataset) if args.mode in {"skill", "replay_center"} else None
        (
            trajectory,
            intended_final_pot,
            transport_final_pot,
            transport_plan,
            handle_grasp_geometry,
        ) = (
            _build_skill(
                keyframes,
                source,
                target,
                source_geometry,
                target_geometry,
                source_parts,
                target_parts,
                source_components,
                target_components,
                _eef_pose(env, "left_arm"),
                _eef_pose(env, "right_arm"),
                args,
            )
            if args.mode == "skill" else (None, None, None, None, None)
        )
        joint_nominal = _sparse_joint_nominal(source, trajectory, keyframes) if trajectory is not None else None
        # Centering deliberately departs from the edge-biased source support
        # pose, even when source and target geometry are identical.  Track the
        # post-grasp semantic path with integrated Cartesian IK in every skill
        # rollout instead of falling back to the demonstration joint nominal.
        integrate_target_ik = trajectory is not None
        contact_close_complete_step = (
            max(
                trajectory.waypoint_steps["left_handle_grasp"],
                trajectory.waypoint_steps["right_handle_grasp"],
            )
            if trajectory is not None else None
        )
        grasp_complete_step = (
            max(
                contact_close_complete_step,
                trajectory.waypoint_steps.get(
                    "bimanual_contact_hold", contact_close_complete_step
                ),
            )
            if trajectory is not None else None
        )
        pregrasp_complete_step = (
            trajectory.waypoint_steps["bimanual_pregrasp"]
            if trajectory is not None else None
        )
        left_grasp_step = (
            trajectory.waypoint_steps["left_handle_grasp"]
            if trajectory is not None else None
        )
        right_grasp_step = (
            trajectory.waypoint_steps["right_handle_grasp"]
            if trajectory is not None else None
        )
        if trajectory is not None:
            from judo_isaaclab.put_marker import (
                compose_pose,
                inverse_pose,
            )

            left_handle_contact = compose_pose(
                inverse_pose(target_geometry.root_pose),
                trajectory.left_poses[left_grasp_step],
            )
            right_handle_contact = compose_pose(
                inverse_pose(target_geometry.root_pose),
                trajectory.right_poses[right_grasp_step],
            )
        else:
            left_handle_contact = right_handle_contact = None
        missing_finger_corrections = {"left": 0.0, "right": 0.0}
        missing_finger_depth_corrections = {"left": 0.0, "right": 0.0}
        missing_finger_streaks = {"left": 0, "right": 0}
        right_first_close = bool(
            trajectory is not None
            and handle_grasp_geometry["left"].get("right_first_close", False)
        )
        milestone_jaw_center_residual_m = None
        milestone_translation_m = None
        milestone_applied_translation_m = None
        milestone_translation_limit_m = None
        milestone_reanchor_accepted = None
        milestone_reanchor_source = None
        milestone_feedback_horizon_steps = None
        milestone_reanchor_step = None
        milestone_gripper_hold_steps = None
        milestone_gripper_close_start_step = None
        milestone_open_pad_reseat_m = 0.0
        milestone_open_pad_reseat_residuals_m = []
        peer_contact_transfer = False
        peer_supported_contact_streak = 0
        peer_single_contact_latch_step = None
        peer_single_contact_latch_support_frames = None
        peer_single_contact_latch_local_m = None
        peer_single_contact_tracking_residual_world_m = None
        peer_contact_latch_jaw_residual_m = None
        peer_contact_jaw_twist_rad = None
        peer_contact_jaw_twist_fraction = None
        peer_contact_pre_twist_jaw_residual_m = None
        peer_contact_authored_jaw_center_locked = False
        peer_contact_latch_centering_applied_m = 0.0
        peer_contact_gripper_retime = None
        peer_contact_position_locked = False
        peer_contact_pad_center_tracking = []
        peer_contact_recovery_residuals_m = []
        peer_contact_pad_reseat_m = 0.0
        peer_contact_pad_reseat_residuals_m = []
        contact_hold_latch_step = None
        contact_hold_loaded_residual_world_m = None
        contact_hold_retention_local_m = None
        contact_hold_tracking_corrections_local_m = []
        reference_hold_left_contact_local = None
        reference_hold_right_contact_local = None
        transport_reanchor_steps = []
        transport_reanchor_evaluation_steps = []
        transport_reanchor_signed_residuals_world_m = []
        transport_reanchor_rejections = []
        transport_reference_left_contact_local = None
        transport_reference_right_contact_local = None
        transport_expected_left_tracking_residual_local = None
        transport_expected_right_tracking_residual_local = None
        transport_motion_preload_local_m = None
        loaded_transport_contact_tracking = []
        center_slide_reanchor_steps = []
        center_slide_reanchor_signed_residuals_local_m = []
        center_slide_reference_right_contact_local = None
        center_slide_contact_recovery_end_step = None
        transverse_handle_axes = [
            axis for axis in range(3) if axis != target_parts.handle_axis
        ]
        transport_contact_tracking_tolerance_m = 0.5 * min(
            float(size[axis])
            for size in (
                target_parts.negative_handle_size,
                target_parts.positive_handle_size,
            )
            for axis in transverse_handle_axes
        )
        from judo_isaaclab.put_pot import (
            HANDLE_PAD_GEOMETRIC_MARGIN_M,
            TRANSPORT_CONTACT_REANCHOR_MIN_STEPS,
            transport_reanchor_position_step_limit_m,
        )

        transport_reanchor_position_limit_m = (
            transport_reanchor_position_step_limit_m(
                args.max_position_step,
                transport_contact_tracking_tolerance_m,
            )
        )
        center_lowering_signed_residual_world_m = None
        release_signed_residual_world_m = None
        target_left_handle_points = None
        if right_first_close:
            side = int(handle_grasp_geometry["left"]["handle_side"])
            boundary = (
                target_parts.body_xy_min[target_parts.handle_axis]
                if side < 0
                else target_parts.body_xy_max[target_parts.handle_axis]
            )
            target_left_handle_points = np.concatenate(
                [
                    points
                    for points in target_components
                    if (
                        np.min(points[:, target_parts.handle_axis])
                        < boundary - 1.0e-4
                        if side < 0
                        else np.max(points[:, target_parts.handle_axis])
                        > boundary + 1.0e-4
                    )
                ]
            )
        repair_prefix_steps = (
            int(keyframes["frames"]["support_align"]["action_index"]) + 1
            if args.mode == "replay_center" else None
        )
        repair_trajectory = None
        repair_joint_nominal = None
        total_steps = (
            repair_prefix_steps + args.center_repair_steps + args.release_steps + args.withdraw_steps + args.settle_steps
            if repair_prefix_steps is not None
            else trajectory.steps if trajectory is not None else len(source["actions"])
        )
        from judo_isaaclab.demo_artifact import DemonstrationRecorder

        demo_recorder = DemonstrationRecorder()
        demo_recorder.start(env.scene.get_state(is_relative=False))
        samples = [_sample(env, -1, "reset")]
        actions = []; pot_poses = []; left_eef = []; right_eef = []; desired_left = []; desired_right = []
        frame_stats = []
        if args.render:
            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            encoder = _Encoder(args.fps, args.video)
        for step in range(total_steps):
            if repair_prefix_steps is not None and step < repair_prefix_steps:
                action = source["actions"][step : step + 1]
                stage = "source_action_prefix"
            elif repair_prefix_steps is not None:
                if repair_trajectory is None:
                    repair_trajectory = _build_center_repair(samples[-1], args)
                    repair_joint_nominal = np.asarray(
                        source["actions"][repair_prefix_steps - 1].detach().cpu(),
                        dtype=np.float64,
                    )
                suffix_step = step - repair_prefix_steps
                stage = repair_trajectory.stage_names[suffix_step]
                action = _ik_action(
                    env,
                    repair_trajectory.left_poses[suffix_step],
                    repair_trajectory.right_poses[suffix_step],
                    repair_trajectory.grippers[suffix_step],
                    repair_joint_nominal,
                    args,
                    integrate_left_ik=True,
                    integrate_right_ik=True,
                )
                desired_left.append(repair_trajectory.left_poses[suffix_step])
                desired_right.append(repair_trajectory.right_poses[suffix_step])
            elif trajectory is None:
                action = source["actions"][step : step + 1]
                stage = "direct_source_action_replay"
            else:
                stage = trajectory.stage_names[step]
                integrate_ik = bool(integrate_target_ik)
                action = _ik_action(
                    env,
                    trajectory.left_poses[step],
                    trajectory.right_poses[step],
                    trajectory.grippers[step],
                    joint_nominal[step],
                    args,
                    integrate_left_ik=integrate_ik,
                    integrate_right_ik=integrate_ik,
                )
                desired_left.append(trajectory.left_poses[step]); desired_right.append(trajectory.right_poses[step])
            observation, _, terminated, truncated, info = env.step(action)
            sample = _sample(env, step, stage, info)
            demo_recorder.append(
                action,
                env.scene.get_state(is_relative=False),
                observation=observation,
                semantic_observation=sample,
            )
            samples.append(sample)
            actions.append(action[0].detach().cpu().numpy())
            pot_poses.append(sample["pot_pose"]); left_eef.append(sample["left_eef_pose"]); right_eef.append(sample["right_eef_pose"])
            if (
                right_first_close
                and step == right_grasp_step
                and sample["right_grasp"]
            ):
                from judo_isaaclab.put_pot import (
                    HANDLE_PAD_DEPTH_MARGIN_M,
                    geometry_conditioned_peer_contact_transfer,
                    mirror_handle_position_in_receiving_jaw_frame,
                    reanchor_authored_handle_in_observed_jaw,
                    retime_loaded_gripper_close_for_pad_reseat,
                    select_geometry_conditioned_milestone_reanchor,
                )

                original_left_contact = left_handle_contact.copy()
                peer_contact_transfer = (
                    geometry_conditioned_peer_contact_transfer(
                        target_parts.negative_handle_size,
                        target_parts.positive_handle_size,
                        handle_grasp_geometry["left"][
                            "predicted_pad_imbalance_m"
                        ],
                    )
                )
                if peer_contact_transfer:
                    left_side = int(
                        handle_grasp_geometry["left"]["handle_side"]
                    )
                    right_side = int(
                        handle_grasp_geometry["right"]["handle_side"]
                    )
                    left_part_frame = (
                        target_parts.negative_handle_frame
                        if left_side < 0
                        else target_parts.positive_handle_frame
                    )
                    right_part_frame = (
                        target_parts.negative_handle_frame
                        if right_side < 0
                        else target_parts.positive_handle_frame
                    )
                    candidate_left_world, milestone_jaw_center_residual_m = (
                        mirror_handle_position_in_receiving_jaw_frame(
                            sample["right_eef_pose"],
                            compose_pose(sample["pot_pose"], right_part_frame),
                            sample["left_eef_pose"],
                            compose_pose(sample["pot_pose"], left_part_frame),
                            sample["pot_pose"],
                            target_left_handle_points,
                        )
                    )
                    candidate_left_contact = compose_pose(
                        inverse_pose(sample["pot_pose"]),
                        candidate_left_world,
                    )
                    milestone_translation_m = float(
                        np.linalg.norm(
                            candidate_left_world[:3]
                            - np.asarray(sample["left_eef_pose"])[:3]
                        )
                    )
                else:
                    (
                        candidate_left_contact,
                        milestone_jaw_center_residual_m,
                        milestone_translation_m,
                    ) = reanchor_authored_handle_in_observed_jaw(
                        sample["pot_pose"],
                        sample["left_eef_pose"],
                        sample["left_pad_centers_world"],
                        target_left_handle_points,
                        handle_grasp_geometry["left"][
                            "regrasp_approach_local_m"
                        ],
                    )
                milestone_translation_limit_m = (
                    handle_grasp_geometry["left"]["regrasp_clearance_m"]
                    + HANDLE_PAD_DEPTH_MARGIN_M
                )
                left_handle_contact, milestone_reanchor_accepted = (
                    select_geometry_conditioned_milestone_reanchor(
                        original_left_contact,
                        candidate_left_contact,
                        milestone_translation_m,
                        handle_grasp_geometry["left"]["regrasp_clearance_m"],
                        peer_contact_transfer=peer_contact_transfer,
                    )
                )
                milestone_reanchor_source = (
                    "observed_peer_contact"
                    if peer_contact_transfer
                    else (
                        "authored_open_jaw"
                        if milestone_reanchor_accepted
                        else "original_object_local_contact"
                    )
                )
                applied_left_world = compose_pose(
                    sample["pot_pose"], left_handle_contact
                )
                milestone_applied_translation_m = float(
                    np.linalg.norm(
                        applied_left_world[:3]
                        - np.asarray(sample["left_eef_pose"])[:3]
                    )
                )
                milestone_feedback_horizon_steps = max(
                    1,
                    int(
                        np.ceil(
                            milestone_applied_translation_m
                            / args.max_position_step
                        )
                    ),
                )
                milestone_reanchor_step = step
                if peer_contact_transfer:
                    trajectory, milestone_gripper_hold_steps = (
                        retime_loaded_gripper_close_for_pad_reseat(
                            trajectory,
                            step,
                            milestone_applied_translation_m,
                            reseat_step_m=0.002,
                        )
                    )
                    milestone_gripper_close_start_step = (
                        step + milestone_gripper_hold_steps
                    )
                missing_finger_corrections["left"] = 0.0
                missing_finger_depth_corrections["left"] = 0.0
                missing_finger_streaks["left"] = 0
            if (
                trajectory is not None
                and pregrasp_complete_step <= step < grasp_complete_step
            ):
                from judo_isaaclab.put_pot import (
                    CONTACT_FEEDBACK_HORIZON_STEPS,
                    MISSING_FINGER_CONTACT_DELAY_STEPS,
                    SINGLE_FINGER_CONTACT_LATCH_STEPS,
                    reanchor_missing_finger_contact,
                    reanchor_missing_finger_pad_depth,
                    reanchor_single_contact_pad_fraction,
                    geometry_conditioned_loaded_jaw_rotation_fraction,
                    retime_loaded_gripper_close_for_pad_reseat,
                    single_contact_pad_base_residual_m,
                    twist_loaded_jaw_about_observed_contact,
                    preserve_loaded_contact_target,
                    single_finger_contact_observed,
                    track_bimanual_handle_targets,
                    track_loaded_pad_center_from_observation,
                    LOADED_JAW_REACH_AVOIDANCE_FRACTION,
                    single_contact_pad_reseat_saturated,
                )

                if (
                    peer_contact_transfer
                    and peer_single_contact_latch_step is None
                    and milestone_gripper_close_start_step is not None
                    and step < milestone_gripper_close_start_step
                ):
                    previous_open_pad_reseat_m = milestone_open_pad_reseat_m
                    (
                        left_handle_contact,
                        milestone_open_pad_reseat_m,
                        open_pad_reseat_residual_m,
                    ) = reanchor_single_contact_pad_fraction(
                        left_handle_contact,
                        sample["pot_pose"],
                        sample["left_finger_forces_n"],
                        sample["left_pad_fractions"],
                        sample["left_pad_axes_world"],
                        milestone_open_pad_reseat_m,
                    )
                    milestone_open_pad_reseat_residuals_m.append(
                        {
                            "step": step,
                            "signed_residual_m": open_pad_reseat_residual_m,
                            "cumulative_applied_m": (
                                milestone_open_pad_reseat_m
                            ),
                        }
                    )
                    if (
                        single_contact_pad_reseat_saturated(
                            previous_open_pad_reseat_m,
                            milestone_open_pad_reseat_m,
                            open_pad_reseat_residual_m,
                        )
                        and single_finger_contact_observed(
                            sample["left_finger_forces_n"],
                            sample["left_pad_fractions"],
                        )
                    ):
                        trajectory, _ = (
                            retime_loaded_gripper_close_for_pad_reseat(
                                trajectory,
                                step,
                                0.0,
                            )
                        )
                        milestone_gripper_close_start_step = step
                        milestone_gripper_hold_steps = (
                            step - milestone_reanchor_step
                        )

                supported_peer_contact = bool(
                    peer_contact_transfer
                    and (
                        milestone_gripper_close_start_step is None
                        or step >= milestone_gripper_close_start_step
                    )
                    and single_finger_contact_observed(
                        sample["left_finger_forces_n"],
                        sample["left_pad_fractions"],
                    )
                )
                peer_single_contact_latch_support_frames = (
                    SINGLE_FINGER_CONTACT_LATCH_STEPS
                )
                peer_supported_contact_streak = (
                    peer_supported_contact_streak + 1
                    if supported_peer_contact
                    else 0
                )
                if (
                    peer_single_contact_latch_step is None
                    and peer_supported_contact_streak
                    >= SINGLE_FINGER_CONTACT_LATCH_STEPS
                ):
                    loaded_left_world, retained_residual = (
                        preserve_loaded_contact_target(
                            sample["left_eef_pose"],
                            trajectory.left_poses[step],
                            maximum_position_residual_m=args.max_position_step,
                        )
                    )
                    from judo_isaaclab.put_pot import (
                        HANDLE_PAD_GEOMETRIC_MARGIN_M,
                        center_handle_between_finger_pads,
                        handle_jaw_center_offset_m,
                    )

                    peer_contact_pre_twist_jaw_residual_m = (
                        handle_jaw_center_offset_m(
                            loaded_left_world,
                            sample["pot_pose"],
                            target_left_handle_points,
                        )
                    )
                    peer_contact_authored_jaw_center_locked = bool(
                        abs(peer_contact_pre_twist_jaw_residual_m)
                        <= HANDLE_PAD_GEOMETRIC_MARGIN_M
                    )
                    jaw_residual_for_twist_m = (
                        peer_contact_pre_twist_jaw_residual_m
                    )
                    if not peer_contact_authored_jaw_center_locked:
                        loaded_left_world = center_handle_between_finger_pads(
                            loaded_left_world,
                            peer_contact_pre_twist_jaw_residual_m,
                            maximum_correction_m=abs(
                                peer_contact_pre_twist_jaw_residual_m
                            ),
                        )
                        peer_contact_latch_centering_applied_m = (
                            peer_contact_pre_twist_jaw_residual_m
                        )
                        jaw_residual_for_twist_m = (
                            handle_jaw_center_offset_m(
                                loaded_left_world,
                                sample["pot_pose"],
                                target_left_handle_points,
                            )
                        )
                        milestone_feedback_horizon_steps = max(
                            1,
                            int(
                                np.ceil(
                                    abs(peer_contact_pre_twist_jaw_residual_m)
                                    / 0.002
                                )
                            ),
                        )
                        peer_contact_pad_reseat_m = (
                            milestone_open_pad_reseat_m
                        )
                    else:
                        milestone_feedback_horizon_steps = 1
                    peer_contact_jaw_twist_fraction = (
                        geometry_conditioned_loaded_jaw_rotation_fraction(
                            sample["left_finger_forces_n"],
                            sample["left_pad_fractions"],
                            jaw_center_residual_m=(
                                jaw_residual_for_twist_m
                            ),
                        )
                    )
                    (
                        loaded_left_world,
                        peer_contact_jaw_twist_rad,
                        peer_contact_position_locked,
                    ) = twist_loaded_jaw_about_observed_contact(
                        sample["left_eef_pose"],
                        loaded_left_world,
                        sample["left_finger_forces_n"],
                        sample["left_pad_centers_world"],
                        sample["left_pad_axes_world"],
                        peer_contact_jaw_twist_fraction,
                    )
                    peer_contact_position_locked = bool(
                        peer_contact_position_locked
                        or peer_contact_authored_jaw_center_locked
                    )
                    retained_residual = (
                        loaded_left_world[:3]
                        - np.asarray(sample["left_eef_pose"])[:3]
                    )
                    initial_pad_reseat_residual_m = (
                        single_contact_pad_base_residual_m(
                            sample["left_finger_forces_n"],
                            sample["left_pad_fractions"],
                        )
                    )
                    effective_pad_reseat_residual_m = (
                        0.0
                        if peer_contact_position_locked
                        else initial_pad_reseat_residual_m
                    )
                    trajectory, gripper_hold_steps = (
                        retime_loaded_gripper_close_for_pad_reseat(
                            trajectory,
                            step,
                            effective_pad_reseat_residual_m,
                        )
                    )
                    peer_contact_gripper_retime = {
                        "step": step,
                        "retained_command": float(
                            trajectory.grippers[step, 0]
                        ),
                        "pad_reseat_residual_m": (
                            initial_pad_reseat_residual_m
                        ),
                        "effective_pad_reseat_residual_m": (
                            effective_pad_reseat_residual_m
                        ),
                        "hold_steps": gripper_hold_steps,
                        "close_end_step": grasp_complete_step,
                    }
                    left_handle_contact = compose_pose(
                        inverse_pose(sample["pot_pose"]),
                        loaded_left_world,
                    )
                    peer_single_contact_latch_step = step
                    peer_single_contact_latch_local_m = (
                        left_handle_contact[:3].tolist()
                    )
                    peer_single_contact_tracking_residual_world_m = (
                        retained_residual.tolist()
                    )
                    from judo_isaaclab.put_pot import handle_jaw_center_offset_m

                    peer_contact_latch_jaw_residual_m = (
                        handle_jaw_center_offset_m(
                            loaded_left_world,
                            sample["pot_pose"],
                            target_left_handle_points,
                        )
                    )

                for arm in ("left", "right"):
                    if (
                        right_first_close
                        and arm == "left"
                        and step < right_grasp_step
                    ):
                        continue
                    contacting = (
                        np.asarray(sample[f"{arm}_finger_forces_n"]) >= 0.1
                    )
                    missing_finger_streaks[arm] = (
                        missing_finger_streaks[arm] + 1
                        if int(np.sum(contacting)) == 1
                        else 0
                    )
                    if (
                        arm == "left"
                        and peer_single_contact_latch_step is not None
                    ):
                        continue
                    if (
                        missing_finger_streaks[arm]
                        <= MISSING_FINGER_CONTACT_DELAY_STEPS
                    ):
                        continue
                    contact = (
                        left_handle_contact if arm == "left"
                        else right_handle_contact
                    )
                    contact, missing_finger_corrections[arm] = (
                        reanchor_missing_finger_contact(
                            contact,
                            sample["pot_pose"],
                            sample[f"{arm}_finger_forces_n"],
                            sample[f"{arm}_pad_centers_world"],
                            missing_finger_corrections[arm],
                        )
                    )
                    contact, missing_finger_depth_corrections[arm] = (
                        reanchor_missing_finger_pad_depth(
                            contact,
                            sample["pot_pose"],
                            sample[f"{arm}_finger_forces_n"],
                            sample[f"{arm}_pad_fractions"],
                            sample[f"{arm}_pad_axes_world"],
                            missing_finger_depth_corrections[arm],
                        )
                    )
                    if arm == "left":
                        left_handle_contact = contact
                    else:
                        right_handle_contact = contact

                if (
                    peer_single_contact_latch_step is not None
                    and not bool(sample["left_grasp"])
                ):
                    from judo_isaaclab.put_pot import (
                        reanchor_handle_jaw_center_step,
                    )

                    if peer_contact_position_locked:
                        pad_reseat_residual_m = (
                            single_contact_pad_base_residual_m(
                                sample["left_finger_forces_n"],
                                sample["left_pad_fractions"],
                            )
                        )
                    else:
                        (
                            left_handle_contact,
                            peer_contact_pad_reseat_m,
                            pad_reseat_residual_m,
                        ) = reanchor_single_contact_pad_fraction(
                            left_handle_contact,
                            sample["pot_pose"],
                            sample["left_finger_forces_n"],
                            sample["left_pad_fractions"],
                            sample["left_pad_axes_world"],
                            peer_contact_pad_reseat_m,
                        )
                    peer_contact_pad_reseat_residuals_m.append(
                        {
                            "step": step,
                            "signed_residual_m": pad_reseat_residual_m,
                            "cumulative_applied_m": peer_contact_pad_reseat_m,
                        }
                    )

                    if peer_contact_position_locked:
                        from judo_isaaclab.put_pot import handle_jaw_center_offset_m

                        jaw_residual_m = handle_jaw_center_offset_m(
                            compose_pose(
                                sample["pot_pose"], left_handle_contact
                            ),
                            sample["pot_pose"],
                            target_left_handle_points,
                        )
                        jaw_applied_m = 0.0
                    else:
                        (
                            left_handle_contact,
                            jaw_residual_m,
                            jaw_applied_m,
                        ) = reanchor_handle_jaw_center_step(
                            left_handle_contact,
                            sample["pot_pose"],
                            target_left_handle_points,
                        )
                    missing_finger_corrections["left"] -= jaw_applied_m
                    peer_contact_recovery_residuals_m.append(
                        {
                            "step": step,
                            "signed_residual_m": jaw_residual_m,
                            "applied_m": jaw_applied_m,
                        }
                    )

                trajectory = track_bimanual_handle_targets(
                    trajectory,
                    step,
                    sample["pot_pose"],
                    sample["left_eef_pose"],
                    sample["right_eef_pose"],
                    left_handle_contact,
                    right_handle_contact,
                    left_contact_latched=bool(sample["left_grasp"]),
                    right_contact_latched=bool(sample["right_grasp"]),
                    right_first_close=right_first_close,
                    feedback_horizon_steps=(
                        milestone_feedback_horizon_steps
                        if milestone_feedback_horizon_steps is not None
                        else CONTACT_FEEDBACK_HORIZON_STEPS
                    ),
                )
                left_contacting = (
                    np.asarray(sample["left_finger_forces_n"]) >= 0.1
                )
                if (
                    peer_contact_position_locked
                    and not bool(sample["left_grasp"])
                    and int(np.sum(left_contacting)) == 1
                ):
                    retained_left_world = compose_pose(
                        sample["pot_pose"], left_handle_contact
                    )
                    trajectory, pad_center_correction = (
                        track_loaded_pad_center_from_observation(
                            trajectory,
                            step,
                            sample["left_eef_pose"],
                            sample["left_finger_forces_n"],
                            sample["left_pad_centers_world"],
                            retained_left_world,
                        )
                    )
                    peer_contact_pad_center_tracking.append(
                        {
                            "step": step,
                            "signed_correction_world_m": (
                                pad_center_correction.tolist()
                            ),
                        }
                    )
            if (
                trajectory is not None
                and contact_close_complete_step <= step < grasp_complete_step
            ):
                from judo_isaaclab.put_pot import (
                    cartesian_smoothness_metrics,
                    compensate_retained_contact_tracking,
                    maximum_bimanual_position_step_m,
                    reanchor_bimanual_contact_hold,
                    reanchor_bimanual_transport_from_observation,
                    reinforce_loaded_contact_for_motion,
                )

                if (
                    reference_hold_left_contact_local is None
                    and sample["left_grasp"]
                    and sample["right_grasp"]
                ):
                    loaded_left_world, contact_hold_loaded_residual = (
                        preserve_loaded_contact_target(
                            sample["left_eef_pose"],
                            trajectory.left_poses[step],
                            maximum_position_residual_m=args.max_position_step,
                        )
                    )
                    reference_hold_left_contact_local = compose_pose(
                        inverse_pose(sample["pot_pose"]),
                        loaded_left_world,
                    )
                    reference_hold_right_contact_local = compose_pose(
                        inverse_pose(sample["pot_pose"]),
                        sample["right_eef_pose"],
                    )
                    contact_hold_latch_step = step
                    contact_hold_loaded_residual_world_m = (
                        contact_hold_loaded_residual.tolist()
                    )
                if reference_hold_left_contact_local is not None:
                    (
                        retained_hold_left_contact_local,
                        contact_hold_retention_local_m,
                    ) = compensate_retained_contact_tracking(
                        reference_hold_left_contact_local,
                        sample["pot_pose"],
                        sample["left_eef_pose"],
                    )
                    contact_hold_tracking_corrections_local_m.append(
                        {
                            "step": step,
                            "left": contact_hold_retention_local_m.tolist(),
                        }
                    )
                    trajectory = reanchor_bimanual_contact_hold(
                        trajectory,
                        step,
                        sample["pot_pose"],
                        retained_hold_left_contact_local,
                        reference_hold_right_contact_local,
                    )
                    if step + 1 == grasp_complete_step:
                        observed_inverse = inverse_pose(sample["pot_pose"])
                        observed_left_local = compose_pose(
                            observed_inverse, sample["left_eef_pose"]
                        )
                        observed_right_local = compose_pose(
                            observed_inverse, sample["right_eef_pose"]
                        )
                        (
                            retained_hold_left_contact_local,
                            transport_motion_preload,
                        ) = reinforce_loaded_contact_for_motion(
                            retained_hold_left_contact_local,
                            observed_left_local,
                            (
                                target_parts.negative_handle_size
                                if int(
                                    handle_grasp_geometry["left"]["handle_side"]
                                )
                                < 0
                                else target_parts.positive_handle_size
                            ),
                            target_parts.handle_axis,
                        )
                        transport_motion_preload_local_m = (
                            transport_motion_preload.tolist()
                        )
                        trajectory, retained_transport = (
                            reanchor_bimanual_transport_from_observation(
                                trajectory,
                                sample["pot_pose"],
                                sample["left_eef_pose"],
                                sample["right_eef_pose"],
                                transport_final_pot,
                                target_geometry.size,
                                target_cooktop_geometry,
                                transport_clearance_m=args.transport_clearance_m,
                                collision_clearance_m=args.collision_clearance_m,
                                current_step=step,
                                left_contact_local=(
                                    retained_hold_left_contact_local
                                ),
                                right_contact_local=(
                                    reference_hold_right_contact_local
                                ),
                                vertical_rise_fraction=handle_grasp_geometry[
                                    "left"
                                ]["transport_vertical_rise_fraction"],
                                frontload_horizontal_axis=handle_grasp_geometry[
                                    "left"
                                ]["transport_frontload_horizontal_axis"],
                            )
                        )
                        transport_reference_left_contact_local = (
                            retained_hold_left_contact_local
                        )
                        transport_reference_right_contact_local = (
                            reference_hold_right_contact_local
                        )
                        transport_expected_left_tracking_residual_local = (
                            retained_hold_left_contact_local[:3]
                            - observed_left_local[:3]
                        )
                        transport_expected_right_tracking_residual_local = (
                            reference_hold_right_contact_local[:3]
                            - observed_right_local[:3]
                        )
                        transport_reanchor_evaluation_steps.append(step)
                        transport_reanchor_steps.append(step)
                        start = step + 1
                        end = trajectory.waypoint_steps["smooth_transport"]
                        maximum_step_m = maximum_bimanual_position_step_m(
                            trajectory.left_poses[start : end + 1],
                            trajectory.right_poses[start : end + 1],
                        )
                        transport_plan = cartesian_smoothness_metrics(
                            trajectory.left_poses[start : end + 1],
                            trajectory.right_poses[start : end + 1],
                        )
                        transport_plan.update(
                            {
                                "start_step": start,
                                "end_step": end,
                                "maximum_per_arm_step_m": maximum_step_m,
                                "minimum_cooktop_clearance_m": (
                                    retained_transport.minimum_cooktop_clearance_m
                                ),
                                "cooktop_overlap_samples": (
                                    retained_transport.cooktop_overlap_samples
                                ),
                                "initial_clearance_recovery_m": (
                                    retained_transport.initial_clearance_recovery_m
                                ),
                                "vertical_rise_steps": (
                                    retained_transport.vertical_rise_steps
                                ),
                            }
                        )
            if (
                trajectory is not None
                and grasp_complete_step <= step
                < trajectory.waypoint_steps["smooth_transport"]
                and sample["left_grasp"]
                and sample["right_grasp"]
            ):
                from judo_isaaclab.put_pot import (
                    cartesian_smoothness_metrics,
                    compensate_retained_contact_tracking,
                    maximum_bimanual_position_step_m,
                    reanchor_bimanual_transport_from_observation,
                    transport_contact_reanchor_required,
                )

                if transport_contact_reanchor_required(
                    trajectory,
                    step,
                    sample["pot_pose"],
                    sample["left_eef_pose"],
                    sample["right_eef_pose"],
                    transport_reference_left_contact_local,
                    transport_reference_right_contact_local,
                    last_reanchor_step=(
                        transport_reanchor_evaluation_steps[-1]
                        if transport_reanchor_evaluation_steps
                        else None
                    ),
                    tracking_tolerance_m=transport_contact_tracking_tolerance_m,
                    expected_left_tracking_residual_local=(
                        transport_expected_left_tracking_residual_local
                    ),
                    expected_right_tracking_residual_local=(
                        transport_expected_right_tracking_residual_local
                    ),
                    minimum_interval_steps=max(
                        1, int(transport_plan.get("vertical_rise_steps", 1))
                    ),
                ):
                    transport_reanchor_evaluation_steps.append(step)
                    observed_pot_inverse = inverse_pose(sample["pot_pose"])
                    observed_left_contact_local = compose_pose(
                        observed_pot_inverse, sample["left_eef_pose"]
                    )
                    observed_right_contact_local = compose_pose(
                        observed_pot_inverse, sample["right_eef_pose"]
                    )
                    retained_left_contact_local = (
                        observed_left_contact_local
                        if transport_reference_left_contact_local is None
                        else compensate_retained_contact_tracking(
                            transport_reference_left_contact_local,
                            sample["pot_pose"],
                            sample["left_eef_pose"],
                        )[0]
                    )
                    retained_right_contact_local = (
                        observed_right_contact_local
                        if transport_reference_right_contact_local is None
                        else compensate_retained_contact_tracking(
                            transport_reference_right_contact_local,
                            sample["pot_pose"],
                            sample["right_eef_pose"],
                        )[0]
                    )
                    signed_residual = {
                        "step": step,
                        "left": (
                            trajectory.left_poses[step, :3]
                            - np.asarray(sample["left_eef_pose"])[:3]
                        ).tolist(),
                        "right": (
                            trajectory.right_poses[step, :3]
                            - np.asarray(sample["right_eef_pose"])[:3]
                        ).tolist(),
                        "contact_frame_local": {
                            "left": (
                                np.zeros(3, dtype=np.float64)
                                if transport_reference_left_contact_local is None
                                else observed_left_contact_local[:3]
                                - transport_reference_left_contact_local[:3]
                            ).tolist(),
                            "right": (
                                np.zeros(3, dtype=np.float64)
                                if transport_reference_right_contact_local is None
                                else observed_right_contact_local[:3]
                                - transport_reference_right_contact_local[:3]
                            ).tolist(),
                        },
                        "retained_contact_correction_local": {
                            "left": (
                                retained_left_contact_local[:3]
                                - (
                                    observed_left_contact_local[:3]
                                    if transport_reference_left_contact_local is None
                                    else transport_reference_left_contact_local[:3]
                                )
                            ).tolist(),
                            "right": (
                                retained_right_contact_local[:3]
                                - (
                                    observed_right_contact_local[:3]
                                    if transport_reference_right_contact_local is None
                                    else transport_reference_right_contact_local[:3]
                                )
                            ).tolist(),
                        },
                    }
                    candidate_trajectory, observed_transport = (
                        reanchor_bimanual_transport_from_observation(
                            trajectory,
                            sample["pot_pose"],
                            sample["left_eef_pose"],
                            sample["right_eef_pose"],
                            transport_final_pot,
                            target_geometry.size,
                            target_cooktop_geometry,
                            transport_clearance_m=args.transport_clearance_m,
                            collision_clearance_m=args.collision_clearance_m,
                            current_step=step,
                            left_contact_local=retained_left_contact_local,
                            right_contact_local=retained_right_contact_local,
                            vertical_rise_fraction=handle_grasp_geometry[
                                "left"
                            ]["transport_vertical_rise_fraction"],
                            frontload_horizontal_axis=handle_grasp_geometry[
                                "left"
                            ]["transport_frontload_horizontal_axis"],
                        )
                    )
                    start = step + 1
                    end = candidate_trajectory.waypoint_steps["smooth_transport"]
                    candidate_maximum_step_m = maximum_bimanual_position_step_m(
                        candidate_trajectory.left_poses[start : end + 1],
                        candidate_trajectory.right_poses[start : end + 1],
                    )
                    if candidate_maximum_step_m <= transport_reanchor_position_limit_m:
                        trajectory = candidate_trajectory
                        if transport_reference_left_contact_local is None:
                            transport_reference_left_contact_local = (
                                observed_left_contact_local
                            )
                            transport_expected_left_tracking_residual_local = (
                                np.zeros(3, dtype=np.float64)
                            )
                        if transport_reference_right_contact_local is None:
                            transport_reference_right_contact_local = (
                                observed_right_contact_local
                            )
                            transport_expected_right_tracking_residual_local = (
                                np.zeros(3, dtype=np.float64)
                            )
                        transport_reanchor_steps.append(step)
                        transport_reanchor_signed_residuals_world_m.append(
                            signed_residual
                        )
                        transport_plan = cartesian_smoothness_metrics(
                            trajectory.left_poses[start : end + 1],
                            trajectory.right_poses[start : end + 1],
                        )
                        transport_plan.update(
                            {
                                "start_step": start,
                                "end_step": end,
                                "maximum_per_arm_step_m": candidate_maximum_step_m,
                                "minimum_cooktop_clearance_m": (
                                    observed_transport.minimum_cooktop_clearance_m
                                ),
                                "cooktop_overlap_samples": (
                                    observed_transport.cooktop_overlap_samples
                                ),
                                "initial_clearance_recovery_m": (
                                    observed_transport.initial_clearance_recovery_m
                                ),
                                "vertical_rise_steps": (
                                    observed_transport.vertical_rise_steps
                                ),
                            }
                        )
                    else:
                        transport_reanchor_rejections.append(
                            {
                                **signed_residual,
                                "maximum_per_arm_step_m": candidate_maximum_step_m,
                                "position_step_limit_m": transport_reanchor_position_limit_m,
                            }
                        )
            if (
                trajectory is not None
                and transport_reference_left_contact_local is not None
                and grasp_complete_step <= step
                < trajectory.waypoint_steps["smooth_transport"]
                and sample["right_grasp"]
            ):
                from judo_isaaclab.put_pot import (
                    track_retained_contact_from_observed_object,
                )

                observed_left_contact_local = compose_pose(
                    inverse_pose(sample["pot_pose"]),
                    sample["left_eef_pose"],
                )
                next_left_target = compose_pose(
                    sample["pot_pose"],
                    transport_reference_left_contact_local,
                )
                loaded_transport_contact_tracking.append(
                    {
                        "step": step,
                        "target_residual_world_m": (
                            next_left_target[:3]
                            - np.asarray(sample["left_eef_pose"])[:3]
                        ).tolist(),
                        "contact_residual_local_m": (
                            observed_left_contact_local[:3]
                            - transport_reference_left_contact_local[:3]
                        ).tolist(),
                    }
                )
                trajectory = track_retained_contact_from_observed_object(
                    trajectory,
                    step,
                    sample["pot_pose"],
                    transport_reference_left_contact_local,
                )
            if trajectory is not None and "smooth_transport" in trajectory.waypoint_steps and step == trajectory.waypoint_steps["smooth_transport"]:
                from judo_isaaclab.put_pot import reanchor_centered_lowering

                measured_center_correction = (
                    np.asarray(sample["cooktop_pose"], dtype=np.float64)[:2]
                    - np.asarray(sample["pot_pose"], dtype=np.float64)[:2]
                )
                center_lowering_signed_residual_world_m = [
                    float(measured_center_correction[0]),
                    float(measured_center_correction[1]),
                    float(intended_final_pot[2] - sample["pot_pose"][2]),
                ]
                trajectory = reanchor_centered_lowering(
                    trajectory,
                    (
                        np.zeros(2, dtype=np.float64)
                        if "center_slide" in trajectory.waypoint_steps
                        else measured_center_correction
                    ),
                    sample["left_eef_pose"],
                    sample["right_eef_pose"],
                    vertical_correction_m=(
                        intended_final_pot[2] - sample["pot_pose"][2]
                    ),
                )
            if (
                trajectory is not None
                and "support_align" in trajectory.waypoint_steps
                and "center_slide" not in trajectory.waypoint_steps
                and step == trajectory.waypoint_steps["support_align"]
            ):
                from judo_isaaclab.put_pot import reanchor_centered_support

                center_correction = (
                    np.asarray(sample["cooktop_pose"], dtype=np.float64)[:2]
                    - np.asarray(sample["pot_pose"], dtype=np.float64)[:2]
                )
                trajectory = reanchor_centered_support(
                    trajectory,
                    center_correction,
                    sample["left_eef_pose"],
                    sample["right_eef_pose"],
                )
            if trajectory is not None and "pot_unload" in trajectory.waypoint_steps and step == trajectory.waypoint_steps["support_lower"]:
                from judo_isaaclab.put_pot import reanchor_centered_unload

                center_correction = (
                    np.asarray(sample["cooktop_pose"], dtype=np.float64)[:2]
                    - np.asarray(sample["pot_pose"], dtype=np.float64)[:2]
                )
                trajectory = reanchor_centered_unload(
                    trajectory,
                    center_correction,
                    sample["left_eef_pose"],
                    sample["right_eef_pose"],
                )
            if trajectory is not None and "center_slide" in trajectory.waypoint_steps and step == trajectory.waypoint_steps["left_unload_release"]:
                from judo_isaaclab.put_pot import reanchor_supported_center_slide

                trajectory = reanchor_supported_center_slide(
                    trajectory,
                    sample["pot_pose"],
                    sample["cooktop_pose"],
                    sample["right_eef_pose"],
                    support_unload_m=(
                        args.support_clearance_m
                        + HANDLE_PAD_GEOMETRIC_MARGIN_M
                    ),
                )
                center_slide_reference_right_contact_local = compose_pose(
                    inverse_pose(sample["pot_pose"]), sample["right_eef_pose"]
                )
                center_slide_reanchor_steps.append(step)
                center_slide_reanchor_signed_residuals_local_m.append(
                    {
                        "step": step,
                        "phase": "supported_contact_anchor",
                        "right": [0.0, 0.0, 0.0],
                    }
                )
            elif (
                trajectory is not None
                and "center_slide" in trajectory.waypoint_steps
                and trajectory.waypoint_steps["left_unload_release"] < step
                < trajectory.waypoint_steps["center_slide"]
                and sample["right_grasp"]
                and center_slide_reference_right_contact_local is not None
            ):
                observed_right_contact_local = compose_pose(
                    inverse_pose(sample["pot_pose"]), sample["right_eef_pose"]
                )
                center_slide_contact_residual = (
                    observed_right_contact_local[:3]
                    - center_slide_reference_right_contact_local[:3]
                )
                if step == center_slide_contact_recovery_end_step:
                    from judo_isaaclab.put_pot import reanchor_supported_center_slide

                    trajectory = reanchor_supported_center_slide(
                        trajectory,
                        sample["pot_pose"],
                        sample["cooktop_pose"],
                        sample["right_eef_pose"],
                        current_step=step,
                    )
                    center_slide_reference_right_contact_local = (
                        observed_right_contact_local
                    )
                    center_slide_contact_recovery_end_step = None
                    center_slide_reanchor_steps.append(step)
                    center_slide_reanchor_signed_residuals_local_m.append(
                        {
                            "step": step,
                            "phase": "recovered_contact_latched",
                            "right": center_slide_contact_residual.tolist(),
                        }
                    )
                elif (
                    center_slide_contact_recovery_end_step is None
                    and step - center_slide_reanchor_steps[-1]
                    >= TRANSPORT_CONTACT_REANCHOR_MIN_STEPS
                    and trajectory.waypoint_steps["center_slide"] - step >= 9
                    and np.linalg.norm(center_slide_contact_residual)
                    > transport_contact_tracking_tolerance_m
                ):
                    from judo_isaaclab.put_pot import reanchor_supported_center_slide

                    trajectory = reanchor_supported_center_slide(
                        trajectory,
                        sample["pot_pose"],
                        sample["cooktop_pose"],
                        sample["right_eef_pose"],
                        current_step=step,
                        reference_right_contact_local=(
                            center_slide_reference_right_contact_local
                        ),
                    )
                    center_slide_contact_recovery_end_step = step + min(
                        CONTACT_FEEDBACK_HORIZON_STEPS,
                        trajectory.waypoint_steps["center_slide"] - step - 8,
                    )
                    center_slide_reanchor_steps.append(step)
                    center_slide_reanchor_signed_residuals_local_m.append(
                        {
                            "step": step,
                            "phase": "contact_recovery_started",
                            "right": center_slide_contact_residual.tolist(),
                        }
                    )
            release_anchor = (
                "center_slide"
                if trajectory is not None and "center_slide" in trajectory.waypoint_steps
                else "pot_unload"
                if trajectory is not None and "pot_unload" in trajectory.waypoint_steps
                else "support_lower"
            )
            if trajectory is not None and step == trajectory.waypoint_steps[release_anchor]:
                from judo_isaaclab.put_pot import reanchor_centered_release

                center_correction = (
                    np.asarray(sample["cooktop_pose"], dtype=np.float64)[:2]
                    - np.asarray(sample["pot_pose"], dtype=np.float64)[:2]
                )
                release_signed_residual_world_m = [
                    float(center_correction[0]),
                    float(center_correction[1]),
                ]
                trajectory = reanchor_centered_release(
                    trajectory,
                    center_correction,
                    sample["left_eef_pose"],
                    sample["right_eef_pose"],
                )
            if encoder is not None:
                frame = _frame(env, sample); encoder.write(frame); frame_stats.append((float(frame.mean()), float(frame.std())))
            if (step + 1) % 50 == 0 or sample["task_success"]:
                print("PUTPOT_PROGRESS=" + json.dumps({key: sample[key] for key in ("step", "program_stage", "stage1", "stage2", "task_success", "left_grasp", "right_grasp", "pot_pose", "support_error_m", "center_error_m", "xy_error_m")}, sort_keys=True), flush=True)
            if bool(truncated[0].item()):
                raise RuntimeError(f"unexpected timeout/reset at step {step}")
            if bool(terminated[0].item()) and not sample["task_success"]:
                print(
                    "PUTPOT_FAILURE="
                    + json.dumps(
                        {
                            "reason": "unexpected_failure_termination",
                            "sample": sample,
                            "info_keys": sorted(info),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                raise RuntimeError(f"unexpected failure termination at step {step}")
        if encoder is not None:
            encoder.close(); encoder = None
        Path(args.trace_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.trace_npz,
            actions=np.asarray(actions, dtype=np.float32),
            pot_poses=np.asarray(pot_poses, dtype=np.float32),
            left_eef_poses=np.asarray(left_eef, dtype=np.float32),
            right_eef_poses=np.asarray(right_eef, dtype=np.float32),
            desired_left_eef_poses=np.asarray(desired_left, dtype=np.float32),
            desired_right_eef_poses=np.asarray(desired_right, dtype=np.float32),
            left_finger_forces_n=np.asarray(
                [sample["left_finger_forces_n"] for sample in samples[1:]],
                dtype=np.float32,
            ),
            left_pad_fractions=np.asarray(
                [sample["left_pad_fractions"] for sample in samples[1:]],
                dtype=np.float32,
            ),
            right_finger_forces_n=np.asarray(
                [sample["right_finger_forces_n"] for sample in samples[1:]],
                dtype=np.float32,
            ),
            right_pad_fractions=np.asarray(
                [sample["right_pad_fractions"] for sample in samples[1:]],
                dtype=np.float32,
            ),
            left_pad_axes_world=np.asarray(
                [sample["left_pad_axes_world"] for sample in samples[1:]],
                dtype=np.float32,
            ),
            right_pad_axes_world=np.asarray(
                [sample["right_pad_axes_world"] for sample in samples[1:]],
                dtype=np.float32,
            ),
            left_pad_centers_world=np.asarray(
                [sample["left_pad_centers_world"] for sample in samples[1:]],
                dtype=np.float32,
            ),
            right_pad_centers_world=np.asarray(
                [sample["right_pad_centers_world"] for sample in samples[1:]],
                dtype=np.float32,
            ),
            sparse_joint_nominal=np.asarray(joint_nominal, dtype=np.float32) if joint_nominal is not None else np.empty((0, 14), dtype=np.float32),
            program_stages=np.asarray(
                trajectory.stage_names
                if trajectory is not None
                else (["source_action_prefix"] * repair_prefix_steps + repair_trajectory.stage_names)
                if repair_trajectory is not None
                else ["direct_source_action_replay"] * len(actions)
            ),
        )
        final = samples[-1]
        extracted = None
        if args.mode == "replay" and final["task_success"]:
            extracted = _extract_keyframes(samples, np.asarray(actions), args.source_dataset, source_assets)
            if args.write_keyframes:
                Path(args.write_keyframes).parent.mkdir(parents=True, exist_ok=True)
                with open(args.write_keyframes, "w", encoding="utf-8") as stream:
                    json.dump(extracted, stream, indent=2, sort_keys=True)
        video = _probe(args.video) if args.render else None
        desired_error = []
        if trajectory is not None:
            desired_error = [max(np.linalg.norm(np.asarray(left_eef[i])[:3] - trajectory.left_poses[i, :3]), np.linalg.norm(np.asarray(right_eef[i])[:3] - trajectory.right_poses[i, :3])) for i in range(len(left_eef))]
        waypoint_errors = [desired_error[index] for index in trajectory.waypoint_steps.values() if trajectory is not None and index < len(desired_error)] if trajectory is not None else []
        executed_transport_metrics = None
        if trajectory is not None and transport_plan is not None:
            from judo_isaaclab.put_pot import (
                cartesian_smoothness_metrics,
                minimum_cooktop_clearance_m,
            )

            start = int(transport_plan["start_step"])
            end = int(transport_plan["end_step"]) + 1
            executed_transport_metrics = cartesian_smoothness_metrics(
                np.asarray(left_eef)[start:end], np.asarray(right_eef)[start:end]
            )
            executed_transport_metrics["minimum_cooktop_clearance_m"] = (
                minimum_cooktop_clearance_m(
                    np.asarray(pot_poses)[start:end],
                    target_geometry.size,
                    _geometry(target_assets["cooktop"], target["cooktop_pose"][0]),
                )
            )
        direct_replay = None
        if args.direct_replay_result:
            with open(args.direct_replay_result, encoding="utf-8") as stream:
                direct_replay = json.load(stream)
        from judo_isaaclab.put_pot import (
            CENTERED_ON_COOKTOP_TOLERANCE_M,
            HANDLE_PAD_DEPTH_MARGIN_M,
            LOADED_JAW_REACH_AVOIDANCE_FRACTION,
            MISSING_FINGER_CONTACT_LIMIT_M,
            MISSING_FINGER_CONTACT_DELAY_STEPS,
            MISSING_FINGER_JAW_AXIS_MIN_M,
            MISSING_FINGER_CONTACT_STEP_M,
            MISSING_FINGER_PAD_DEPTH_LIMIT_M,
            MISSING_FINGER_PAD_DEPTH_STEP_M,
            MISSING_FINGER_PAD_TARGET_FRACTION,
        )

        centered_on_cooktop = bool(
            final["center_error_m"] <= CENTERED_ON_COOKTOP_TOLERANCE_M
        )
        checks = {
            "one_reset": True,
            "zero_inter_stage_resets": True,
            "real_target_assets": target_assets == _dataset_assets(args.target_dataset, args.objects_root),
            "contact_backed_grasps_only": True,
            "smooth_collision_aware_transport": bool(
                trajectory is None
                or (
                    "smooth_transport" in trajectory.waypoint_steps
                    and "pot_lift" not in trajectory.waypoint_steps
                    and "pot_transport" not in trajectory.waypoint_steps
                    and "support_align" not in trajectory.waypoint_steps
                    and transport_plan["minimum_cooktop_clearance_m"]
                    + 1.0e-9
                    >= args.collision_clearance_m
                    and executed_transport_metrics["minimum_cooktop_clearance_m"]
                    + 1.0e-9
                    >= 0.0
                )
            ),
            "transport_no_internal_stops": bool(
                trajectory is None or transport_plan["internal_stop_count"] == 0
            ),
            "bimanual_transport_completed": bool(
                trajectory is None
                or (
                    samples[int(transport_plan["end_step"]) + 1]["left_grasp"]
                    and samples[int(transport_plan["end_step"]) + 1]["right_grasp"]
                )
            ),
            "coded_task_success": bool(final["task_success"]),
            "centered_on_cooktop": centered_on_cooktop,
            "accepted_task_success": bool(final["task_success"] and centered_on_cooktop),
            "all_stages_latched": bool(final["stage1"] and final["stage2"]),
            "bimanual_pick_observed": any(row["left_grasp"] and row["right_grasp"] for row in samples),
            "pot_released": not final["left_grasp"] and not final["right_grasp"],
            "stable_support_window": bool(final["on_top_predicate_now"]),
            "terminal_pot_speed_within_threshold": bool(
                float(np.linalg.norm(final["pot_velocity"][:3])) <= 0.05
            ),
            "h264_nonempty": video is None or (video["codec"] == "h264" and video["size_bytes"] > 0 and video["frame_count"] == len(frame_stats)),
            "fully_decodable": video is None or video["full_decode_returncode"] == 0,
        }
        if args.classification_run:
            if args.mode != "replay":
                raise ValueError("--classification-run is only valid in replay mode")
            acceptance_checks = {
                name: checks[name]
                for name in (
                    "one_reset", "zero_inter_stage_resets", "real_target_assets",
                    "contact_backed_grasps_only", "h264_nonempty", "fully_decodable",
                )
            }
        elif args.expect_failure:
            acceptance_checks = {name: checks[name] for name in ("one_reset", "zero_inter_stage_resets", "real_target_assets", "contact_backed_grasps_only", "h264_nonempty", "fully_decodable")}
            acceptance_checks["expected_coded_task_failure"] = not bool(final["task_success"])
        else:
            acceptance_checks = checks
            if trajectory is not None and direct_replay is None:
                # Source proof validates the smooth semantic program and coded
                # task success.  Full bimanual retention is the adaptation gate
                # for the selected replay-failing target below.
                acceptance_checks = dict(acceptance_checks)
                acceptance_checks.pop("bimanual_transport_completed")
            if direct_replay is not None:
                acceptance_checks = dict(acceptance_checks)
                acceptance_checks["direct_source_action_replay_failed"] = bool(
                    direct_replay.get("status") == "passed"
                    and not direct_replay.get("checks", {}).get(
                        "accepted_task_success",
                        direct_replay.get("terminal", {}).get("task_success", True),
                    )
                )
        demo_artifact = None
        if args.demo_hdf5 and checks["accepted_task_success"] and all(acceptance_checks.values()):
            from judo_isaaclab.demo_artifact import relative_asset_paths

            demo_recorder.write(
                args.demo_hdf5,
                assets_instance_paths=relative_asset_paths(target_assets, args.objects_root),
                success=True,
                metadata={
                    "task": "PutPotOnCooktop-v0",
                    "controller": (
                        "source_action_prefix_with_supported_center_repair"
                        if repair_trajectory is not None
                        else "direct_source_action_replay" if trajectory is None
                        else "deterministic_semantic_skill"
                    ),
                    "candidate_sampling": False,
                    "source_dataset_sha256": _sha256(args.source_dataset),
                    "target_dataset_sha256": _sha256(args.target_dataset),
                },
            )
            demo_artifact = {"path": os.path.abspath(args.demo_hdf5), "sha256": _sha256(args.demo_hdf5)}
        from run_putmarker_skill_program import _asset_provenance
        result = {
            "status": "passed" if all(acceptance_checks.values()) else "failed",
            "mode": args.mode,
            "protocol": {
                "controller": (
                    "source_action_prefix_with_supported_center_repair"
                    if repair_trajectory is not None
                    else "direct_source_action_replay" if trajectory is None
                    else "semantic_support_frames_with_cartesian_dls"
                ),
                "candidate_sampling": False,
                "scene_resets": 1,
                "inter_stage_resets": 0,
                "teleports_after_reset": 0,
                "control_rate_hz": 30,
                "steps": len(actions),
                "seed": args.seed,
                "grasp_assistance": "none",
                "milestone_feedback_horizon_steps": (
                    milestone_feedback_horizon_steps
                ),
                "milestone_translation_m": milestone_translation_m,
                "milestone_applied_translation_m": (
                    milestone_applied_translation_m
                ),
                "milestone_translation_limit_m": milestone_translation_limit_m,
                "milestone_reanchor_accepted": milestone_reanchor_accepted,
                "milestone_reanchor_source": milestone_reanchor_source,
                "milestone_reanchor_step": milestone_reanchor_step,
                "milestone_gripper_hold_steps": milestone_gripper_hold_steps,
                "milestone_gripper_close_start_step": (
                    milestone_gripper_close_start_step
                ),
                "milestone_open_pad_reseat_m": milestone_open_pad_reseat_m,
                "milestone_open_pad_reseat_residuals_m": (
                    milestone_open_pad_reseat_residuals_m
                ),
                "peer_single_contact_latch_step": peer_single_contact_latch_step,
                "peer_single_contact_latch_support_frames": (
                    peer_single_contact_latch_support_frames
                ),
                "peer_single_contact_latch_local_m": (
                    peer_single_contact_latch_local_m
                ),
                "peer_single_contact_tracking_residual_world_m": (
                    peer_single_contact_tracking_residual_world_m
                ),
                "peer_contact_latch_jaw_residual_m": (
                    peer_contact_latch_jaw_residual_m
                ),
                "peer_contact_jaw_twist_rad": peer_contact_jaw_twist_rad,
                "peer_contact_jaw_twist_fraction": (
                    peer_contact_jaw_twist_fraction
                ),
                "peer_contact_pre_twist_jaw_residual_m": (
                    peer_contact_pre_twist_jaw_residual_m
                ),
                "peer_contact_authored_jaw_center_locked": (
                    peer_contact_authored_jaw_center_locked
                ),
                "peer_contact_latch_centering_applied_m": (
                    peer_contact_latch_centering_applied_m
                ),
                "peer_contact_gripper_retime": peer_contact_gripper_retime,
                "peer_contact_position_locked": peer_contact_position_locked,
                "peer_contact_pad_center_tracking": (
                    peer_contact_pad_center_tracking
                ),
                "peer_contact_recovery_residuals_m": (
                    peer_contact_recovery_residuals_m
                ),
                "peer_contact_pad_reseat_m": peer_contact_pad_reseat_m,
                "peer_contact_pad_reseat_residuals_m": (
                    peer_contact_pad_reseat_residuals_m
                ),
                "contact_hold_latch_step": contact_hold_latch_step,
                "contact_hold_loaded_residual_world_m": (
                    contact_hold_loaded_residual_world_m
                ),
                "contact_hold_retention_local_m": (
                    None
                    if contact_hold_retention_local_m is None
                    else contact_hold_retention_local_m.tolist()
                ),
                "contact_hold_tracking_corrections_local_m": (
                    contact_hold_tracking_corrections_local_m
                ),
                "transport_reanchor_step": (
                    transport_reanchor_steps[0] if transport_reanchor_steps else None
                ),
                "transport_reanchor_steps": transport_reanchor_steps,
                "transport_reanchor_evaluation_steps": transport_reanchor_evaluation_steps,
                "transport_contact_tracking_tolerance_m": transport_contact_tracking_tolerance_m,
                "transport_reanchor_position_limit_m": transport_reanchor_position_limit_m,
                "transport_reanchor_signed_residuals_world_m": transport_reanchor_signed_residuals_world_m,
                "transport_reanchor_rejections": transport_reanchor_rejections,
                "transport_reanchor_minimum_interval_steps": (
                    None
                    if transport_plan is None
                    else max(
                        1, int(transport_plan.get("vertical_rise_steps", 1))
                    )
                ),
                "transport_expected_tracking_residual_local_m": {
                    "left": (
                        None
                        if transport_expected_left_tracking_residual_local is None
                        else transport_expected_left_tracking_residual_local.tolist()
                    ),
                    "right": (
                        None
                        if transport_expected_right_tracking_residual_local is None
                        else transport_expected_right_tracking_residual_local.tolist()
                    ),
                },
                "transport_motion_preload_local_m": (
                    transport_motion_preload_local_m
                ),
                "loaded_transport_contact_tracking": (
                    loaded_transport_contact_tracking
                ),
                "center_slide_reanchor_steps": center_slide_reanchor_steps,
                "center_slide_reanchor_signed_residuals_local_m": center_slide_reanchor_signed_residuals_local_m,
                "center_lowering_signed_residual_world_m": center_lowering_signed_residual_world_m,
                "release_signed_residual_world_m": release_signed_residual_world_m,
                "parameters": {"damping": args.damping, "max_joint_delta": args.max_joint_delta, "max_position_step": args.max_position_step, "max_rotation_step": args.max_rotation_step, "support_clearance_m": args.support_clearance_m, "transport_clearance_m": args.transport_clearance_m, "collision_clearance_m": args.collision_clearance_m, "executed_collision_minimum_m": 0.0, "handle_pad_depth_margin_m": HANDLE_PAD_DEPTH_MARGIN_M, "loaded_jaw_reach_avoidance_fraction": LOADED_JAW_REACH_AVOIDANCE_FRACTION, "geometry_conditioned_handle_grasp": handle_grasp_geometry, "missing_finger_contact_feedback": {"step_m": MISSING_FINGER_CONTACT_STEP_M, "limit_m": MISSING_FINGER_CONTACT_LIMIT_M, "minimum_observed_jaw_axis_m": MISSING_FINGER_JAW_AXIS_MIN_M, "milestone_jaw_center_residual_m": milestone_jaw_center_residual_m, "delay_steps": MISSING_FINGER_CONTACT_DELAY_STEPS, "final_signed_corrections_m": missing_finger_corrections, "pad_depth_step_m": MISSING_FINGER_PAD_DEPTH_STEP_M, "pad_depth_limit_m": MISSING_FINGER_PAD_DEPTH_LIMIT_M, "pad_target_fraction": MISSING_FINGER_PAD_TARGET_FRACTION, "final_pad_depth_corrections_m": missing_finger_depth_corrections}, "target_direct_generation": trajectory is not None, "source_semantic_success_required": False, "object_to_gripper_contact_frame_transfer": trajectory is not None, "requested_transport_steps": args.transport_steps, "transport_steps": (int(transport_plan["end_step"] - transport_plan["start_step"] + 1) if transport_plan is not None else args.transport_steps), "lower_steps": args.lower_steps, "release_steps": args.release_steps, "withdraw_steps": args.withdraw_steps, "settle_steps": args.settle_steps, "center_repair_steps": args.center_repair_steps, "integrated_target_ik": integrate_target_ik or repair_trajectory is not None, "smooth_collision_aware_transport": trajectory is not None, "bimanual_target_transport_required": bool(trajectory is not None and direct_replay is not None), "supported_center_slide": bool(trajectory is not None and "center_slide" in trajectory.waypoint_steps) or repair_trajectory is not None, "source_action_prefix_steps": repair_prefix_steps, "center_feedback_reanchor": trajectory is not None, "center_feedback_release_correction": trajectory is not None, "center_tolerance_m": CENTERED_ON_COOKTOP_TOLERANCE_M},
            },
            "provenance": {
                "source_dataset": {"path": os.path.abspath(args.source_dataset), "sha256": _sha256(args.source_dataset)},
                "target_dataset": {"path": os.path.abspath(args.target_dataset), "sha256": _sha256(args.target_dataset)},
                "source_assets": {name: _asset_provenance(path) for name, path in source_assets.items()},
                "target_assets": {name: _asset_provenance(path) for name, path in target_assets.items()},
                "task_manager": {"path": os.path.join(args.gear_repo, "dc_study/envs/tasks/put_pot_on_cooktop_manager.py"), "sha256": _sha256(os.path.join(args.gear_repo, "dc_study/envs/tasks/put_pot_on_cooktop_manager.py"))},
                "task_config": {"path": os.path.join(args.gear_repo, "dc_study/envs/tasks/put_pot_on_cooktop_manager_cfg.py"), "sha256": _sha256(os.path.join(args.gear_repo, "dc_study/envs/tasks/put_pot_on_cooktop_manager_cfg.py"))},
                "trace": {"path": os.path.abspath(args.trace_npz), "sha256": _sha256(args.trace_npz)},
                "demonstration": demo_artifact,
                "source_keyframes": ({"path": os.path.abspath(args.source_keyframes), "sha256": _sha256(args.source_keyframes)} if args.source_keyframes else None),
            },
            "semantic_frames": {
                "source_pot_bottom": source_geometry.bottom_frame.tolist(),
                "target_pot_bottom": target_geometry.bottom_frame.tolist(),
                "target_cooktop_top": _geometry(target_assets["cooktop"], target["cooktop_pose"][0]).top_frame.tolist(),
                "intended_final_pot_pose": intended_final_pot.tolist() if intended_final_pot is not None else None,
                "transport_final_pot_pose": transport_final_pot.tolist() if transport_final_pot is not None else None,
                "extracted_keyframes": extracted,
                "source_pot_parts": jsonable(source_parts),
                "target_pot_parts": jsonable(target_parts),
            },
            "stage_success_trace": _transition_trace(samples),
            "metrics": {
                "eef_tracking_error_m": max(waypoint_errors) if waypoint_errors else None,
                "maximum_eef_tracking_error_m": max(desired_error) if desired_error else None,
                "support_error_m": final["support_error_m"],
                "center_error_m": final["center_error_m"],
                "xy_error_m": final["xy_error_m"],
                "terminal_pot_speed_mps": float(np.linalg.norm(final["pot_velocity"][:3])),
                "terminal_pot_angular_speed_rps": float(np.linalg.norm(final["pot_velocity"][3:])),
                "left_grasp_frames": sum(row["left_grasp"] for row in samples),
                "right_grasp_frames": sum(row["right_grasp"] for row in samples),
                "transport_plan": transport_plan,
                "transport_executed": executed_transport_metrics,
            },
            "terminal": final,
            "checks": checks,
            "acceptance_checks": acceptance_checks,
            "video": video,
            "direct_replay_baseline": direct_replay,
            "offline_ground_override": offline_ground,
        }
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print("PUTPOT_FINAL=" + json.dumps(result, sort_keys=True), flush=True)
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
