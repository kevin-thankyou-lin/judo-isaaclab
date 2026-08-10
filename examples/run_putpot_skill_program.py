"""Run direct replay or a deterministic semantic PutPot skill in IsaacLab.

Replay mode performs a one-reset free-running action replay and, on a successful
source run, extracts simulator-backed semantic keyframes.  Skill mode consumes
that fail-closed keyframe artifact and transfers the bimanual handle/support
strategy to the selected target assets without sampling, assistance, or an
inter-stage reset.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))


def _milestone_reanchor_enabled(
    *, right_first_close: bool, forced_right_first_stabilization: bool
) -> bool:
    """Keep forced source chronology isolated from older feedback mechanisms."""
    return bool(right_first_close and not forced_right_first_stabilization)


def _parser(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gear-repo", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--objects-root", required=True)
    parser.add_argument("--mode", choices=("replay", "replay_center", "skill"), required=True)
    parser.add_argument("--source-keyframes")
    parser.add_argument("--source-demo-card")
    parser.add_argument("--write-keyframes")
    parser.add_argument("--expect-failure", action="store_true")
    parser.add_argument(
        "--acquisition-only",
        action="store_true",
        help=(
            "Stop after the complete grasp/hold window. This mode never emits "
            "a transport command and must be paired with --expect-failure."
        ),
    )
    parser.add_argument(
        "--classification-run",
        action="store_true",
        help="Accept a technically valid replay whether task success passes or fails.",
    )
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--program-spec-json")
    parser.add_argument("--controller-plugin-py")
    parser.add_argument("--controller-plugin-sha256")
    parser.add_argument("--controller-plugin-log")
    parser.add_argument("--controller-timeout-s", type=float, default=5.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video")
    parser.add_argument(
        "--render-diagnostic-only",
        action="store_true",
        help=(
            "Render left-handle contact axes without changing commands. This "
            "mode is non-training, consumes no repair attempt, and must replay "
            "an immutable acquisition trace exactly."
        ),
    )
    parser.add_argument(
        "--diagnostic-reference-trace",
        help="Immutable trace whose actions must exactly match this diagnostic replay.",
    )
    parser.add_argument(
        "--diagnostic-physical-request-id",
        help="Physical-attempt request identity owned by this non-consuming replay.",
    )
    parser.add_argument("--trace-npz", required=True)
    parser.add_argument("--demo-hdf5")
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--direct-replay-result")
    parser.add_argument("--persistent-session", action="store_true")
    parser.add_argument("--lifetime-attempt-number", type=int)
    parser.add_argument("--repair-epoch")
    parser.add_argument("--repair-epoch-attempt", type=int)
    parser.add_argument("--repair-epoch-attempt-limit", type=int, default=4)
    parser.add_argument("--runtime-receipt-json")
    parser.add_argument(
        "--target-left-grasp-orientation-local-wxyz",
        nargs=4,
        type=float,
        metavar=("W", "X", "Y", "Z"),
        help=(
            "Calibration-only left grasp orientation in the target pot frame. "
            "Transport still requires the measured bilateral latch gate."
        ),
    )
    parser.add_argument(
        "--target-left-precontact-calibration-trace",
        help=(
            "Immutable prior trace whose open-jaw sample measures the static "
            "left precontact jaw-axis translation. Acquisition-only mode only."
        ),
    )
    parser.add_argument(
        "--target-left-precontact-calibration-step",
        type=int,
        help="Zero-based open-jaw sample in the immutable calibration trace.",
    )
    parser.add_argument(
        "--target-left-source-contact-calibration-trace",
        help=(
            "Immutable failed acquisition trace used only to recover the rigid "
            "open-jaw pad frame for a source-contact-frame correction."
        ),
    )
    parser.add_argument(
        "--target-left-source-contact-calibration-step",
        type=int,
        help="Zero-based open-jaw sample for the source-contact-frame correction.",
    )
    parser.add_argument(
        "--target-left-source-contact-critic-json",
        help="Immutable critic receipt that owns the calibration trace.",
    )
    parser.add_argument(
        "--target-left-source-approach-corridor",
        action="store_true",
        help=(
            "Map both source left pregrasp and grasp frames through the target "
            "handle contact frame. Acquisition-only mode only."
        ),
    )
    parser.add_argument(
        "--target-left-contact-frame-preorientation-complete-step",
        type=int,
        help=(
            "Complete the source-mapped left wrist orientation at this force-free "
            "acquisition step, without changing any Cartesian translation."
        ),
    )
    parser.add_argument(
        "--target-left-contact-frame-prior-first-force-step",
        type=int,
        help=(
            "Measured first left force step from the immutable diagnostic that "
            "bounds contact-frame preorientation."
        ),
    )
    parser.add_argument(
        "--target-left-contact-frame-radial-clearance-m",
        type=float,
        help=(
            "Measured outward target-contact-normal clearance for a tangent-safe "
            "left pregrasp waypoint."
        ),
    )
    parser.add_argument(
        "--target-left-contact-frame-radial-waypoint-step",
        type=int,
        help="Acquisition step at which the tangent-safe radial waypoint is reached.",
    )
    parser.add_argument(
        "--target-source-left-first-acquisition",
        action="store_true",
        help=(
            "Preserve the source card's left-then-right acquisition order by "
            "holding the right gripper open at pregrasp through left closure."
        ),
    )
    parser.add_argument(
        "--target-right-first-stabilized-acquisition",
        action="store_true",
        help=(
            "Acquire the proven right endpoint first, hold the left arm at reset, "
            "then run the source-mapped left acquisition. Acquisition-only mode."
        ),
    )
    parser.add_argument(
        "--target-handle-local-mpc-acquisition",
        action="store_true",
        help=(
            "Replace only the post-right-latch left contact corridor with a "
            "deterministic bounded handle-local receding-horizon controller."
        ),
    )
    parser.add_argument(
        "--target-right-handle-local-mpc-bootstrap",
        action="store_true",
        help=(
            "Use the same bounded handle-local controller to acquire a robust "
            "right dual-pad latch before enabling the left contact window."
        ),
    )
    return parser.parse_args(argv)


_PERSISTENT_RUNTIME: dict[str, object] | None = None
_LAST_ATTEMPT_RUNTIME_RECEIPT: dict[str, object] | None = None


def close_persistent_runtime() -> dict[str, object] | None:
    """Close the one loaded PutPot runtime at an explicit worker boundary."""

    global _PERSISTENT_RUNTIME
    runtime = _PERSISTENT_RUNTIME
    if runtime is None:
        return None
    started = time.monotonic()
    runtime["env"].close()
    runtime["simulation_app"].close()
    receipt = {
        "pid": os.getpid(),
        "runtime_key": runtime["key"],
        "attempts": runtime["attempts"],
        "shutdown_s": time.monotonic() - started,
    }
    _PERSISTENT_RUNTIME = None
    return receipt


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


def _install_procedural_ground(scene, sim_utils, asset_base_cfg) -> str:
    """Install the offline collider whether the Gear scene has a ground slot."""

    spawn = sim_utils.CuboidCfg(
        size=(100.0, 100.0, 0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.18, 0.18, 0.18), roughness=0.8
        ),
        semantic_tags=[("class", "ground")],
    )
    if scene.ground is None:
        scene.ground = asset_base_cfg(
            prim_path="/World/Ground",
            spawn=spawn,
            init_state=asset_base_cfg.InitialStateCfg(
                pos=(0.0, 0.0, -0.05)
            ),
        )
        return "created_missing_scene_ground"
    scene.ground.init_state.pos = (0.0, 0.0, -0.05)
    scene.ground.spawn = spawn
    return "replaced_existing_scene_ground"


def _configure_offline_ground() -> dict[str, object]:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from dc_study.envs.tasks.put_pot_on_cooktop_manager_cfg import PutPotOnCooktopManagerEnvCfg

    original_init = PutPotOnCooktopManagerEnvCfg.__init__

    def offline_init(instance, *init_args, **init_kwargs):
        original_init(instance, *init_args, **init_kwargs)
        # Keep the task's exact stage latches and get_task_success predicate, but
        # do not let ManagerBasedRLEnv auto-reset on the first successful frame.
        # The evidence contract additionally requires a stable terminal window.
        instance.terminations.task_success = None
        instance._putpot_offline_ground_mode = _install_procedural_ground(
            instance.scene,
            sim_utils,
            AssetBaseCfg,
        )

    PutPotOnCooktopManagerEnvCfg.__init__ = offline_init
    return {
        "reason": "network-backed Isaac default_environment.usd unavailable",
        "implementation": (
            "procedural static CuboidCfg; create AssetBaseCfg when the "
            "Gear scene deliberately omits its ground slot"
        ),
        "collision_surface_z_m": 0.0,
        "success_auto_termination": "disabled; coded task predicate unchanged",
    }


def _sample(env, step: int, stage: str, info=None) -> dict[str, object]:
    import torch
    from judo_isaaclab.task_space import pose_runtime_to_wxyz
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
    pot_pose = pose_runtime_to_wxyz(pot_pose)
    cooktop_pose = pose_runtime_to_wxyz(cooktop_pose)
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
    for arm in ("left", "right"):
        source_contact, target_contact = contact_frames[arm]
        grasp_geometry[arm]["source_contact_frame_local"] = (
            source_contact.tolist()
        )
        grasp_geometry[arm]["target_contact_frame_local"] = (
            target_contact.tolist()
        )
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
        CONTACT_BACKED_PICK_SETTLE_STEPS,
        complete_peer_contact_transport_steps,
        geometry_conditioned_grasp_hold_steps,
        geometry_conditioned_peer_contact_hold_steps,
        geometry_conditioned_right_first_close,
        geometry_conditioned_transport_steps,
        geometry_conditioned_vertical_rise_fraction,
    )

    forced_right_first_stabilization = bool(
        getattr(args, "target_right_first_stabilized_acquisition", False)
    )
    right_first_close = (
        forced_right_first_stabilization
        or geometry_conditioned_right_first_close(
            handle_size(source_parts, left_side),
            handle_size(target_parts, left_side),
            target_parts.handle_axis,
            grasp_geometry["left"]["predicted_pad_imbalance_m"],
        )
    )
    grasp_geometry["left"]["right_first_close"] = right_first_close
    grasp_geometry["left"]["forced_right_first_stabilization"] = (
        forced_right_first_stabilization
    )
    grasp_geometry["left"]["defer_left_pregrasp"] = (
        forced_right_first_stabilization
    )
    peer_contact_hold_steps = geometry_conditioned_peer_contact_hold_steps(
        handle_size(target_parts, left_side),
        handle_size(target_parts, right_side),
        grasp_geometry["left"]["predicted_pad_imbalance_m"],
    )
    grasp_geometry["left"]["peer_contact_hold_steps"] = (
        peer_contact_hold_steps
    )
    if right_first_close and not forced_right_first_stabilization:
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
    if peer_contact_hold_steps:
        transport_steps = complete_peer_contact_transport_steps(
            transport_steps,
            peer_contact_hold_steps - CONTACT_BACKED_PICK_SETTLE_STEPS,
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
    source_left_first = bool(
        getattr(args, "target_source_left_first_acquisition", False)
    )
    if source_left_first and right_first_close:
        raise ValueError(
            "source left-first acquisition conflicts with right-first close"
        )
    grasp_geometry["left"]["source_left_first_acquisition"] = (
        source_left_first
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
        simultaneous=not right_first_close and not source_left_first,
        right_first=right_first_close,
        defer_left_pregrasp=bool(
            grasp_geometry["left"].get("defer_left_pregrasp", False)
        ),
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
        "left_pregrasp": (source_indices["left_pregrasp"], source_indices["right_handle_grasp"]),
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


def _debug_axis_primitives(
    target_contact_frame,
    pad_centers_world,
    pad_axes_world,
    actual_wrist,
    desired_wrist,
    *,
    env_origin_world=(0.0, 0.0, 0.0),
) -> dict[str, list[object]]:
    """Build render-only left-contact geometry in simulator world coordinates."""

    from run_putmarker_skill_program import _quat_to_matrix

    origin = np.asarray(env_origin_world, dtype=np.float64)
    target_contact = np.asarray(target_contact_frame, dtype=np.float64).copy()
    actual = np.asarray(actual_wrist, dtype=np.float64).copy()
    desired = np.asarray(desired_wrist, dtype=np.float64).copy()
    for pose in (target_contact, actual, desired):
        pose[:3] += origin
    pads = np.asarray(pad_centers_world, dtype=np.float64)
    pad_axes = np.asarray(pad_axes_world, dtype=np.float64)
    if pads.shape != (2, 3) or pad_axes.shape != (2, 3):
        raise ValueError("debug axes require exactly two left pad centers and axes")

    starts: list[tuple[float, float, float]] = []
    ends: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float, float]] = []
    sizes: list[float] = []
    labels: list[str] = []

    def line(start, end, color, size, label):
        starts.append(tuple(np.asarray(start, dtype=np.float64)))
        ends.append(tuple(np.asarray(end, dtype=np.float64)))
        colors.append(color)
        sizes.append(float(size))
        labels.append(label)

    target_rgb = (
        (1.0, 0.08, 0.08, 1.0),
        (0.08, 1.0, 0.08, 1.0),
        (0.08, 0.35, 1.0, 1.0),
    )
    target_rotation = _quat_to_matrix(target_contact[3:])
    for axis, (length, color) in enumerate(
        zip((0.090, 0.055, 0.055), target_rgb, strict=True)
    ):
        line(
            target_contact[:3],
            target_contact[:3] + length * target_rotation[:, axis],
            color,
            6.0 if axis == 0 else 4.0,
            "target_tangent" if axis == 0 else f"target_axis_{axis}",
        )

    pad_color = (1.0, 0.05, 0.85, 1.0)
    cross_half_width = 0.005
    for pad_index, center in enumerate(pads):
        for axis in range(3):
            delta = np.zeros(3, dtype=np.float64)
            delta[axis] = cross_half_width
            line(
                center - delta,
                center + delta,
                pad_color,
                5.0,
                f"pad_{pad_index}_center",
            )
    line(pads[0], pads[1], pad_color, 5.0, "jaw_closing_line")

    depth_color = (0.55, 1.0, 0.05, 1.0)
    for pad_index, (center, axis) in enumerate(zip(pads, pad_axes, strict=True)):
        line(
            center,
            center + 0.055 * axis / np.linalg.norm(axis),
            depth_color,
            4.0,
            f"pad_{pad_index}_depth_axis",
        )
    mean_depth_axis = np.mean(pad_axes, axis=0)
    mean_depth_norm = float(np.linalg.norm(mean_depth_axis))
    if mean_depth_norm <= 1.0e-9:
        raise ValueError("left pad depth axes have no finite mean direction")
    mean_depth_axis /= mean_depth_norm
    pad_mean = np.mean(pads, axis=0)
    line(
        pad_mean,
        pad_mean + 0.075 * mean_depth_axis,
        depth_color,
        6.0,
        "mean_pad_depth_axis",
    )

    for pose, color, size, label in (
        (actual, (0.05, 0.95, 1.0, 1.0), 4.0, "actual_wrist"),
        (desired, (1.0, 1.0, 1.0, 1.0), 5.0, "target_wrist"),
    ):
        rotation = _quat_to_matrix(pose[3:])
        for axis in range(3):
            line(
                pose[:3],
                pose[:3] + 0.060 * rotation[:, axis],
                color,
                size,
                f"{label}_axis_{axis}",
            )
    line(
        actual[:3],
        desired[:3],
        (1.0, 0.55, 0.02, 1.0),
        7.0,
        "actual_to_desired_correction",
    )
    line(
        pad_mean,
        target_contact[:3],
        (1.0, 0.55, 0.02, 1.0),
        5.0,
        "jaw_midpoint_to_target_contact_correction",
    )
    return {
        "starts": starts,
        "ends": ends,
        "colors": colors,
        "sizes": sizes,
        "labels": labels,
    }


def _debug_scene_primitives(
    pot_pose,
    cooktop_pose,
    target_contact_frames,
    pad_centers_world,
    pad_axes_world,
    actual_wrist_frames,
    desired_wrist_frames,
    *,
    control_vectors_world=None,
    env_origin_world=(0.0, 0.0, 0.0),
) -> dict[str, list[object]]:
    """Build the complete bimanual render-only frame/control diagnostic."""

    from run_putmarker_skill_program import _quat_to_matrix

    origin = np.asarray(env_origin_world, dtype=np.float64)
    starts: list[tuple[float, float, float]] = []
    ends: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float, float]] = []
    sizes: list[float] = []
    labels: list[str] = []

    def line(start, end, color, size, label):
        starts.append(tuple(np.asarray(start, dtype=np.float64)))
        ends.append(tuple(np.asarray(end, dtype=np.float64)))
        colors.append(color)
        sizes.append(float(size))
        labels.append(label)

    def frame_axes(pose, prefix, colors_rgb, length=0.055, size=4.0):
        value = np.asarray(pose, dtype=np.float64).copy()
        value[:3] += origin
        rotation = _quat_to_matrix(value[3:])
        for axis in range(3):
            line(
                value[:3],
                value[:3] + length * rotation[:, axis],
                colors_rgb[axis],
                size,
                f"{prefix}_axis_{axis}",
            )

    xyz = (
        (1.0, 0.08, 0.08, 1.0),
        (0.08, 1.0, 0.08, 1.0),
        (0.08, 0.35, 1.0, 1.0),
    )
    frame_axes(pot_pose, "pot_body", xyz, length=0.070, size=5.0)
    frame_axes(
        cooktop_pose,
        "cooktop_target",
        (
            (0.95, 0.75, 0.10, 1.0),
            (0.80, 0.60, 0.08, 1.0),
            (1.0, 0.92, 0.30, 1.0),
        ),
        length=0.075,
        size=5.0,
    )
    for arm, tint in (
        ("left", (1.0, 0.20, 0.85, 1.0)),
        ("right", (1.0, 0.50, 0.08, 1.0)),
    ):
        frame_axes(target_contact_frames[arm], f"{arm}_handle_contact", xyz, length=0.065, size=5.0)
        pads = np.asarray(pad_centers_world[arm], dtype=np.float64)
        axes = np.asarray(pad_axes_world[arm], dtype=np.float64)
        if pads.shape != (2, 3) or axes.shape != (2, 3):
            raise ValueError(f"debug axes require two {arm} pad centers and axes")
        for pad_index, (center, axis) in enumerate(zip(pads, axes, strict=True)):
            for coordinate in range(3):
                delta = np.zeros(3, dtype=np.float64)
                delta[coordinate] = 0.0045
                line(
                    center - delta,
                    center + delta,
                    tint,
                    5.0,
                    f"{arm}_pad_{pad_index}_center",
                )
            line(
                center,
                center + 0.050 * axis / np.linalg.norm(axis),
                tint,
                4.0,
                f"{arm}_pad_{pad_index}_axis",
            )
        line(pads[0], pads[1], tint, 6.0, f"{arm}_jaw_closing_line")
        frame_axes(
            actual_wrist_frames[arm],
            f"{arm}_actual_wrist",
            ((0.05, 0.95, 1.0, 1.0),) * 3,
            length=0.052,
            size=4.0,
        )
        frame_axes(
            desired_wrist_frames[arm],
            f"{arm}_desired_wrist",
            ((1.0, 1.0, 1.0, 1.0),) * 3,
            length=0.058,
            size=5.0,
        )
        target = np.asarray(target_contact_frames[arm], dtype=np.float64).copy()
        target[:3] += origin
        line(
            pads.mean(axis=0),
            target[:3],
            (1.0, 0.15, 0.15, 1.0),
            7.0,
            f"{arm}_signed_residual",
        )
        actual = np.asarray(actual_wrist_frames[arm], dtype=np.float64).copy()
        actual[:3] += origin
        control = (
            np.asarray(desired_wrist_frames[arm], dtype=np.float64)[:3]
            - np.asarray(actual_wrist_frames[arm], dtype=np.float64)[:3]
            if control_vectors_world is None
            else np.asarray(control_vectors_world[arm], dtype=np.float64)
        )
        line(
            actual[:3],
            actual[:3] + control,
            (0.15, 1.0, 0.15, 1.0),
            7.0,
            f"{arm}_signed_control",
        )
    return {
        "starts": starts,
        "ends": ends,
        "colors": colors,
        "sizes": sizes,
        "labels": labels,
    }


def _draw_left_contact_debug(
    env,
    draw,
    sample,
    target_contact_frame,
    desired_left_wrist,
) -> None:
    if draw is None:
        return
    draw.clear_lines()
    primitives = _debug_axis_primitives(
        target_contact_frame,
        sample["left_pad_centers_world"],
        sample["left_pad_axes_world"],
        sample["left_eef_pose"],
        desired_left_wrist,
        env_origin_world=env.scene.env_origins[0].detach().cpu().numpy(),
    )
    draw.draw_lines(
        primitives["starts"],
        primitives["ends"],
        primitives["colors"],
        primitives["sizes"],
    )


def _draw_contact_debug(
    env,
    draw,
    sample,
    target_contact_frames,
    desired_wrist_frames,
    control_vectors_world=None,
) -> None:
    if draw is None:
        return
    draw.clear_lines()
    primitives = _debug_scene_primitives(
        sample["pot_pose"],
        sample["cooktop_pose"],
        target_contact_frames,
        {
            "left": sample["left_pad_centers_world"],
            "right": sample["right_pad_centers_world"],
        },
        {
            "left": sample["left_pad_axes_world"],
            "right": sample["right_pad_axes_world"],
        },
        {
            "left": sample["left_eef_pose"],
            "right": sample["right_eef_pose"],
        },
        desired_wrist_frames,
        control_vectors_world=control_vectors_world,
        env_origin_world=env.scene.env_origins[0].detach().cpu().numpy(),
    )
    draw.draw_lines(
        primitives["starts"],
        primitives["ends"],
        primitives["colors"],
        primitives["sizes"],
    )


def _debug_axis_legend(frame: np.ndarray) -> np.ndarray:
    import cv2

    frame = frame.copy()
    band_top = frame.shape[0] - 22
    frame[band_top:, :] = (0.18 * frame[band_top:, :]).astype(np.uint8)
    items = (
        ("FRAMES xyz", (255, 60, 60)),
        ("POT/COOK", (245, 205, 40)),
        ("L/R PADS", (255, 20, 220)),
        ("WRIST ACT/TGT", (20, 240, 255)),
        ("RESIDUAL", (255, 40, 40)),
        ("CONTROL", (40, 255, 40)),
    )
    x = 8
    for label, rgb in items:
        frame[band_top + 6 : band_top + 16, x : x + 10] = rgb
        cv2.putText(
            frame,
            label,
            (x + 14, band_top + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        x += 24 + int(7.2 * len(label))
    return frame


def _frame(
    env,
    sample,
    *,
    debug_axis_draw=None,
    target_contact_frames=None,
    desired_wrist_frames=None,
    control_vectors_world=None,
) -> np.ndarray:
    import cv2

    if debug_axis_draw is not None:
        _draw_contact_debug(
            env,
            debug_axis_draw,
            sample,
            target_contact_frames,
            desired_wrist_frames,
            control_vectors_world,
        )
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
    frame = np.concatenate(panels, axis=1)
    return _debug_axis_legend(frame) if debug_axis_draw is not None else frame


def _transition_trace(samples):
    result, previous = [], None
    for sample in samples:
        state = (sample["stage1"], sample["stage2"], sample["task_success"])
        if state != previous or sample is samples[-1]:
            result.append({key: sample[key] for key in ("step", "program_stage", "stage1", "stage2", "task_success", "left_grasp", "right_grasp", "pot_pose", "support_error_m", "center_error_m", "xy_error_m")})
            previous = state
    return result


def _controller_observation(sample):
    """Bounded simulator observation exposed to reloadable controller code."""

    keys = (
        "step",
        "program_stage",
        "stage1",
        "stage2",
        "task_success",
        "left_grasp",
        "right_grasp",
        "pot_pose",
        "pot_velocity",
        "left_eef_pose",
        "right_eef_pose",
        "left_finger_forces_n",
        "right_finger_forces_n",
        "left_pad_fractions",
        "right_pad_fractions",
        "left_pad_axes_world",
        "right_pad_axes_world",
        "left_pad_centers_world",
        "right_pad_centers_world",
        "support_error_m",
        "center_error_m",
        "xy_error_m",
    )
    from judo_isaaclab.putpot_controller_protocol import jsonable

    return jsonable(
        {key: sample[key] for key in keys if key in sample},
        nonfinite="null",
    )


def _write_rollout_trace(
    path,
    *,
    actions,
    pot_poses,
    left_eef,
    right_eef,
    desired_left,
    desired_right,
    samples,
    joint_nominal,
    local_mpc_frame_receipts,
    partial: bool,
) -> None:
    """Write a fresh complete or partial physical trace before process teardown."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite PutPot trace: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    count = len(actions)
    local_active = np.zeros(count, dtype=bool)
    local_translation = np.full((count, 3), np.nan, dtype=np.float32)
    local_rotation = np.full((count, 3), np.nan, dtype=np.float32)
    local_jaw = np.full(count, np.nan, dtype=np.float32)
    local_source_weight = np.full(count, np.nan, dtype=np.float32)
    local_fail_closed = np.zeros(count, dtype=bool)
    local_active_arm = np.full(count, "", dtype="<U5")
    for item in local_mpc_frame_receipts:
        step = int(item["program_step"])
        if not 0 <= step < count:
            continue
        receipt = item["receipt"]
        control = receipt["executed_control"]
        local_active[step] = True
        active_arm = str(item.get("active_arm", "left"))
        if active_arm not in {"left", "right"}:
            raise ValueError(f"invalid handle-local MPC active arm: {active_arm}")
        local_active_arm[step] = active_arm
        local_translation[step] = control["translation_world_m"]
        local_rotation[step] = control["rotation_axis_angle_world_rad"]
        local_jaw[step] = control["jaw_increment"]
        local_source_weight[step] = receipt["source_prior_weight"]
        local_fail_closed[step] = receipt["fail_closed"]
    rows = samples[1 : count + 1]
    np.savez_compressed(
        target,
        actions=np.asarray(actions[:count], dtype=np.float32),
        pot_poses=np.asarray(pot_poses[:count], dtype=np.float32),
        cooktop_poses=np.asarray(
            [sample["cooktop_pose"] for sample in rows], dtype=np.float32
        ),
        left_eef_poses=np.asarray(left_eef[:count], dtype=np.float32),
        right_eef_poses=np.asarray(right_eef[:count], dtype=np.float32),
        desired_left_eef_poses=np.asarray(desired_left[:count], dtype=np.float32),
        desired_right_eef_poses=np.asarray(desired_right[:count], dtype=np.float32),
        left_finger_forces_n=np.asarray(
            [sample["left_finger_forces_n"] for sample in rows], dtype=np.float32
        ),
        left_pad_fractions=np.asarray(
            [sample["left_pad_fractions"] for sample in rows], dtype=np.float32
        ),
        right_finger_forces_n=np.asarray(
            [sample["right_finger_forces_n"] for sample in rows], dtype=np.float32
        ),
        right_pad_fractions=np.asarray(
            [sample["right_pad_fractions"] for sample in rows], dtype=np.float32
        ),
        left_pad_axes_world=np.asarray(
            [sample["left_pad_axes_world"] for sample in rows], dtype=np.float32
        ),
        right_pad_axes_world=np.asarray(
            [sample["right_pad_axes_world"] for sample in rows], dtype=np.float32
        ),
        left_pad_centers_world=np.asarray(
            [sample["left_pad_centers_world"] for sample in rows], dtype=np.float32
        ),
        right_pad_centers_world=np.asarray(
            [sample["right_pad_centers_world"] for sample in rows], dtype=np.float32
        ),
        sparse_joint_nominal=(
            np.asarray(joint_nominal, dtype=np.float32)
            if joint_nominal is not None
            else np.empty((0, 14), dtype=np.float32)
        ),
        program_stages=np.asarray([sample["program_stage"] for sample in rows]),
        local_mpc_active=local_active,
        local_mpc_active_arm=local_active_arm,
        local_mpc_translation_control_world_m=local_translation,
        local_mpc_rotation_control_axis_angle_world_rad=local_rotation,
        local_mpc_jaw_increment=local_jaw,
        local_mpc_source_prior_weight=local_source_weight,
        local_mpc_fail_closed=local_fail_closed,
        partial_trace=np.asarray(bool(partial)),
    )


def main(argv: list[str] | None = None) -> None:
    global _LAST_ATTEMPT_RUNTIME_RECEIPT, _PERSISTENT_RUNTIME
    attempt_wall_started_monotonic = time.monotonic()
    _LAST_ATTEMPT_RUNTIME_RECEIPT = None
    from judo_isaaclab.putpot_runtime import (
        AttemptIdentity,
        PhaseTimers,
        ensure_fresh_output_paths,
        instantiated_scene_sensor_inventory,
        timing_accounting,
        without_scene_camera_sensors,
    )
    from judo_isaaclab.putpot_program_spec import (
        apply_program_spec,
        load_program_spec,
    )

    args = _parser(argv)
    if args.mode in {"skill", "replay_center"} and not args.program_spec_json:
        raise ValueError(f"{args.mode} mode requires --program-spec-json")
    controller_flags = (
        args.controller_plugin_py,
        args.controller_plugin_sha256,
        args.controller_plugin_log,
    )
    if any(value is not None for value in controller_flags) and not all(
        value is not None for value in controller_flags
    ):
        raise ValueError(
            "controller plugin path, sha256, and log must be supplied together"
        )
    if args.controller_plugin_py and args.mode != "skill":
        raise ValueError("controller plugins are only valid in skill mode")
    selected_program_spec = (
        Path(args.program_spec_json)
        if args.program_spec_json
        else REPO_ROOT / "configs/putpot_semantic_program_v4.json"
    )
    # Deliberately reload on every main() call.  Persistent Isaac state is
    # reusable; semantic trajectory/controller parameters are not cached.
    program_spec = load_program_spec(selected_program_spec)
    apply_program_spec(args, program_spec)
    timers = PhaseTimers()
    attempt_identity = None
    identity_values = (
        args.lifetime_attempt_number,
        args.repair_epoch,
        args.repair_epoch_attempt,
    )
    if any(value is not None for value in identity_values):
        if not all(value is not None for value in identity_values):
            raise ValueError(
                "lifetime attempt, repair epoch, and repair-epoch attempt "
                "must be supplied together"
            )
        attempt_identity = AttemptIdentity(
            args.lifetime_attempt_number,
            args.repair_epoch,
            args.repair_epoch_attempt,
            args.repair_epoch_attempt_limit,
        )
    if args.render and not args.video:
        raise ValueError("--render requires --video")
    diagnostic_requested = bool(
        args.render_diagnostic_only or args.diagnostic_reference_trace
    )
    if diagnostic_requested and not (
        args.render_diagnostic_only and args.diagnostic_reference_trace
    ):
        raise ValueError(
            "render diagnostic requires --render-diagnostic-only and "
            "--diagnostic-reference-trace"
        )
    if args.render_diagnostic_only:
        if not (args.render and args.acquisition_only and args.expect_failure):
            raise ValueError(
                "render diagnostic requires render, acquisition-only, and "
                "expect-failure modes"
            )
        if args.demo_hdf5:
            raise ValueError("render diagnostic is non-training and forbids HDF5 output")
        if any(value is not None for value in identity_values):
            raise ValueError("render diagnostic consumes no repair-attempt identity")
        if args.controller_plugin_py:
            raise ValueError("render diagnostic forbids controller plugins")
        reference_trace = Path(args.diagnostic_reference_trace).resolve()
        if not reference_trace.is_file():
            raise FileNotFoundError(
                f"diagnostic reference trace is missing: {reference_trace}"
            )
        if reference_trace == Path(args.trace_npz).resolve():
            raise ValueError("diagnostic output trace must be fresh")
    if args.mode in {"skill", "replay_center"} and not args.source_keyframes:
        raise ValueError(f"{args.mode} mode requires --source-keyframes")
    calibration_requested = bool(
        args.target_left_precontact_calibration_trace is not None
        or args.target_left_precontact_calibration_step is not None
    )
    if calibration_requested and not (
        args.target_left_precontact_calibration_trace is not None
        and args.target_left_precontact_calibration_step is not None
    ):
        raise ValueError(
            "precontact calibration requires both a trace and sample step"
        )
    if args.acquisition_only and (
        args.mode != "skill" or not args.expect_failure
    ):
        raise ValueError(
            "--acquisition-only requires skill mode and --expect-failure"
        )
    if args.acquisition_only and not args.source_demo_card:
        raise ValueError("--acquisition-only requires an immutable source-demo card")
    if calibration_requested and not args.acquisition_only:
        raise ValueError("static precontact calibration is acquisition-only")
    source_contact_requested = any(
        value is not None
        for value in (
            args.target_left_source_contact_calibration_trace,
            args.target_left_source_contact_calibration_step,
            args.target_left_source_contact_critic_json,
        )
    )
    if source_contact_requested and not all(
        value is not None
        for value in (
            args.target_left_source_contact_calibration_trace,
            args.target_left_source_contact_calibration_step,
            args.target_left_source_contact_critic_json,
        )
    ):
        raise ValueError(
            "source-contact correction requires trace, sample step, and critic"
        )
    if source_contact_requested and not args.acquisition_only:
        raise ValueError("source-contact correction is acquisition-only")
    if source_contact_requested and calibration_requested:
        raise ValueError(
            "source-contact correction cannot reuse static translation calibration"
        )
    if args.target_left_source_approach_corridor and not source_contact_requested:
        raise ValueError(
            "source approach corridor requires source-contact calibration inputs"
        )
    preorientation_requested = any(
        value is not None
        for value in (
            args.target_left_contact_frame_preorientation_complete_step,
            args.target_left_contact_frame_prior_first_force_step,
        )
    )
    if preorientation_requested and not all(
        value is not None
        for value in (
            args.target_left_contact_frame_preorientation_complete_step,
            args.target_left_contact_frame_prior_first_force_step,
        )
    ):
        raise ValueError(
            "contact-frame preorientation requires complete and first-force steps"
        )
    if preorientation_requested and not args.target_left_source_approach_corridor:
        raise ValueError(
            "contact-frame preorientation requires the source approach corridor"
        )
    radial_waypoint_requested = any(
        value is not None
        for value in (
            args.target_left_contact_frame_radial_clearance_m,
            args.target_left_contact_frame_radial_waypoint_step,
        )
    )
    if radial_waypoint_requested and not all(
        value is not None
        for value in (
            args.target_left_contact_frame_radial_clearance_m,
            args.target_left_contact_frame_radial_waypoint_step,
        )
    ):
        raise ValueError("radial clearance waypoint requires distance and step")
    if radial_waypoint_requested and not preorientation_requested:
        raise ValueError(
            "radial clearance waypoint requires measured contact-frame preorientation"
        )
    if args.target_right_first_stabilized_acquisition:
        if not (args.acquisition_only and args.target_left_source_approach_corridor):
            raise ValueError(
                "right-first stabilization requires acquisition-only source corridor"
            )
        if args.target_source_left_first_acquisition:
            raise ValueError("right-first and source-left-first acquisition conflict")
        if preorientation_requested or radial_waypoint_requested:
            raise ValueError(
                "right-first stabilization cannot reuse exhausted entry corrections"
            )
    if args.target_handle_local_mpc_acquisition:
        if not (
            args.target_right_first_stabilized_acquisition
            and args.acquisition_only
            and args.target_left_source_approach_corridor
            and source_contact_requested
        ):
            raise ValueError(
                "handle-local MPC requires right-first acquisition-only source "
                "contact warm-start inputs"
            )
        if args.controller_plugin_py:
            raise ValueError("handle-local MPC cannot be combined with a controller plugin")
    if (
        args.target_right_handle_local_mpc_bootstrap
        and not args.target_handle_local_mpc_acquisition
    ):
        raise ValueError(
            "right handle-local bootstrap requires handle-local MPC acquisition"
        )
    if (
        args.target_source_left_first_acquisition
        and not args.target_left_source_approach_corridor
    ):
        raise ValueError(
            "source left-first acquisition requires the measured source corridor"
        )
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
    output_paths = (
        args.result_json,
        args.trace_npz,
        args.demo_hdf5,
        args.video,
        args.write_keyframes,
        args.runtime_receipt_json,
        args.controller_plugin_log,
    )
    if args.persistent_session:
        ensure_fresh_output_paths(list(output_paths))
    else:
        for path in output_paths:
            if path and os.path.isfile(path):
                os.unlink(path)
    # Validate cheap dataset/asset provenance before the expensive app launch.
    asset_load_started = time.monotonic()
    source_assets = _dataset_assets(args.source_dataset, args.objects_root)
    target_assets = _dataset_assets(args.target_dataset, args.objects_root)
    source_demo_card_receipt = None
    if args.source_demo_card:
        from judo_isaaclab.putpot_repair_policy import load_source_demo_card

        source_demo_card = load_source_demo_card(args.source_demo_card)
        if source_demo_card["source_dataset_sha256"] != _sha256(
            args.source_dataset
        ):
            raise ValueError("source-demo card does not match the source dataset")
        source_demo_card_receipt = {
            "path": os.path.abspath(args.source_demo_card),
            "sha256": _sha256(args.source_demo_card),
            "schema_version": source_demo_card["schema_version"],
            "contact_order": source_demo_card["contact_order"],
            "latch_contract": source_demo_card["latch_contract"],
        }
    timers.add("asset_env_load", time.monotonic() - asset_load_started)
    sys.path.insert(0, os.path.abspath(args.gear_repo))
    from isaaclab.app import AppLauncher

    runtime_key = {
        "target_assets": target_assets,
        "device": args.device,
        "render": bool(args.render),
        "camera_width": args.camera_width if args.render else None,
        "camera_height": args.camera_height if args.render else None,
    }
    runtime_reused = False
    reset_index = 1
    scene_sensor_inventory = None
    if args.persistent_session and _PERSISTENT_RUNTIME is not None:
        if _PERSISTENT_RUNTIME["key"] != runtime_key:
            raise RuntimeError(
                "persistent PutPot worker boundary changed; restart worker for "
                "assets, device, or camera capability"
            )
        simulation_app = _PERSISTENT_RUNTIME["simulation_app"]
        env = _PERSISTENT_RUNTIME["env"]
        offline_ground = _PERSISTENT_RUNTIME["offline_ground"]
        scene_sensor_inventory = _PERSISTENT_RUNTIME["scene_sensor_inventory"]
        runtime_reused = True
        reset_index = int(_PERSISTENT_RUNTIME["attempts"]) + 1
    else:
        app_started = time.monotonic()
        simulation_app = AppLauncher(
            {
                "headless": True,
                "device": args.device,
                "enable_cameras": bool(args.render),
            }
        ).app
        timers.add("app_startup", time.monotonic() - app_started)
        env = None
    encoder = None
    controller_client = None
    controller_receipt = None
    controller_command_count = 0
    try:
        import torch
        from dc_study.utils.task_creation import create_task_environment
        from run_putmarker_skill_program import (
            _Encoder, _asset_provenance, _eef_pose, _ik_action, _probe, _reset_scene_to_state,
        )

        if not runtime_reused:
            env_load_started = time.monotonic()
            offline_ground = _configure_offline_ground()
            observation_modalities = ["proprioception"] + (
                ["rgb"] if args.render else []
            )
            from dc_study.envs.yam_bimanual_scene import YamBimanualSceneCfg

            camera_policy = (
                nullcontext()
                if args.render
                else without_scene_camera_sensors(YamBimanualSceneCfg)
            )
            with camera_policy:
                env = create_task_environment(
                    task_name="PutPotOnCooktop-v0",
                    assets_instance_paths=target_assets,
                    objects_randomization=None,
                    init_joint_pos_randomization=0.0,
                    mode="replay",
                    device=args.device,
                    observation_modalities=observation_modalities,
                    enable_self_collisions=False,
                    camera_width=args.camera_width,
                    camera_height=args.camera_height,
                    image_downsample_factor=1,
                    enable_gripper_grasp_clamp=False,
                    enable_grasp_ray_viz=False,
                )
            scene_sensor_inventory = instantiated_scene_sensor_inventory(env.scene)
            timers.add("asset_env_load", time.monotonic() - env_load_started)
            if args.persistent_session:
                _PERSISTENT_RUNTIME = {
                    "key": runtime_key,
                    "simulation_app": simulation_app,
                    "env": env,
                    "offline_ground": offline_ground,
                    "scene_sensor_inventory": scene_sensor_inventory,
                    "attempts": 0,
                }
        # Re-read the instantiated scene on every retry.  The persistent copy is
        # provenance, not a substitute for live camera-absence evidence.
        scene_sensor_inventory = instantiated_scene_sensor_inventory(env.scene)
        if (
            not args.render
            and scene_sensor_inventory["instantiated_scene_camera_sensor_count"]
            != 0
        ):
            raise RuntimeError(
                "diagnostic mode instantiated scene camera sensors: "
                + repr(
                    scene_sensor_inventory[
                        "instantiated_scene_camera_sensor_names"
                    ]
                )
            )
        if args.persistent_session and _PERSISTENT_RUNTIME is not None:
            _PERSISTENT_RUNTIME["scene_sensor_inventory"] = scene_sensor_inventory
        reset_started = time.monotonic()
        env.reset(warm_up=False, seed=args.seed)
        source = _load_dataset(args.source_dataset, args.episode, env.device)
        target = _load_dataset(args.target_dataset, args.episode, env.device)
        target["cooktop_size"] = _geometry(target_assets["cooktop"], target["cooktop_pose"][0]).size
        env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
        _reset_scene_to_state(env.scene, target["initial_state"], env_ids)
        env.sim.forward()
        env.reset_success_check(env_ids)
        timers.add("reset", time.monotonic() - reset_started)
        trajectory_started = time.monotonic()
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
        from judo_isaaclab.put_pot import (
            loaded_pick_height_for_support_clearance,
            minimum_cooktop_clearance_m,
        )

        loaded_pick_height_m = loaded_pick_height_for_support_clearance(
            target_geometry.root_pose,
            target_geometry.size,
            target_cooktop_geometry,
            collision_clearance_m=args.collision_clearance_m,
        )
        keyframes = _load_keyframes(args.source_keyframes, args.source_dataset) if args.mode in {"skill", "replay_center"} else None
        left_reset_pose = _eef_pose(env, "left_arm")
        right_reset_pose = _eef_pose(env, "right_arm")
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
                left_reset_pose,
                right_reset_pose,
                args,
            )
            if args.mode == "skill" else (None, None, None, None, None)
        )
        target_left_handle_points = None
        if trajectory is not None and (
            calibration_requested
            or bool(
                handle_grasp_geometry["left"].get("right_first_close", False)
            )
        ):
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
        target_left_grasp_orientation_override_local_wxyz = None
        static_precontact_jaw_translation = None
        source_contact_frame_correction = None
        diagnostic_target_left_contact_frame = None
        diagnostic_target_contact_frames_local = None
        local_mpc_left_jaw_axis_prior_local = None
        local_mpc_left_pad_axis_prior_local = None
        local_mpc_right_contact_prior_local = None
        local_mpc_right_jaw_axis_prior_local = None
        local_mpc_right_pad_axis_prior_local = None
        local_mpc_right_frame_receipt = None
        if trajectory is not None:
            from judo_isaaclab.put_pot import (
                MEASURED_TARGET_LEFT_GRASP_ORIENTATION_LOCAL_WXYZ,
                apply_object_local_receiving_grasp_orientation,
            )

            target_pot_name = Path(target_assets["pot"]).name.lower()
            measured_orientation = (
                np.asarray(
                    args.target_left_grasp_orientation_local_wxyz,
                    dtype=np.float64,
                )
                if args.target_left_grasp_orientation_local_wxyz is not None
                else MEASURED_TARGET_LEFT_GRASP_ORIENTATION_LOCAL_WXYZ.get(
                    target_pot_name
                )
            )
            if measured_orientation is not None:
                trajectory = apply_object_local_receiving_grasp_orientation(
                    trajectory,
                    target_geometry.root_pose,
                    measured_orientation,
                )
                target_left_grasp_orientation_override_local_wxyz = (
                    measured_orientation.tolist()
                )
            diagnostic_target_contact_frames_local = {
                arm: np.asarray(
                    handle_grasp_geometry[arm]["target_contact_frame_local"],
                    dtype=np.float64,
                )
                for arm in ("left", "right")
            }
        if calibration_requested:
            from judo_isaaclab.put_pot import (
                HANDLE_PAD_DEPTH_MARGIN_M,
                apply_static_precontact_jaw_axis_translation,
                measure_handle_center_in_open_jaw,
            )

            calibration_path = Path(
                args.target_left_precontact_calibration_trace
            ).resolve()
            if not calibration_path.is_file():
                raise FileNotFoundError(
                    f"precontact calibration trace is missing: {calibration_path}"
                )
            with np.load(calibration_path, allow_pickle=False) as calibration:
                required_arrays = {"pot_poses", "left_pad_centers_world"}
                if not required_arrays.issubset(calibration.files):
                    raise ValueError(
                        "precontact calibration trace lacks pot/jaw geometry"
                    )
                calibration_step = int(
                    args.target_left_precontact_calibration_step
                )
                if not 0 <= calibration_step < len(calibration["pot_poses"]):
                    raise ValueError("precontact calibration step is out of range")
                calibration_pot_pose = np.asarray(
                    calibration["pot_poses"][calibration_step],
                    dtype=np.float64,
                )
                calibration_pad_centers = np.asarray(
                    calibration["left_pad_centers_world"][calibration_step],
                    dtype=np.float64,
                )
            jaw_axis_world, signed_translation_m = (
                measure_handle_center_in_open_jaw(
                    calibration_pot_pose,
                    calibration_pad_centers,
                    target_left_handle_points,
                )
            )
            transverse_axes = [
                axis for axis in range(3) if axis != target_parts.handle_axis
            ]
            handle_size = (
                target_parts.negative_handle_size
                if side < 0
                else target_parts.positive_handle_size
            )
            collision_free_pregrasp_m = (
                0.5
                * max(float(handle_size[axis]) for axis in transverse_axes)
                + args.collision_clearance_m
            )
            maximum_translation_m = (
                collision_free_pregrasp_m + HANDLE_PAD_DEPTH_MARGIN_M
            )
            trajectory, static_precontact_jaw_translation = (
                apply_static_precontact_jaw_axis_translation(
                    trajectory,
                    jaw_axis_world,
                    signed_translation_m,
                    maximum_translation_m,
                )
            )
            calibration_root = calibration_path.parent
            calibration_result = calibration_root / "skill_result.json"
            calibration_video = calibration_root / "skill.mp4"
            static_precontact_jaw_translation.update(
                {
                    "mechanism": "static_precontact_grasp_center_translation",
                    "calibration_trace": {
                        "path": str(calibration_path),
                        "sha256": _sha256(calibration_path),
                        "sample_step": calibration_step,
                    },
                    "calibration_result": (
                        {
                            "path": str(calibration_result.resolve()),
                            "sha256": _sha256(calibration_result),
                        }
                        if calibration_result.is_file()
                        else None
                    ),
                    "calibration_video": (
                        {
                            "path": str(calibration_video.resolve()),
                            "sha256": _sha256(calibration_video),
                        }
                        if calibration_video.is_file()
                        else None
                    ),
                    "measured_pot_pose": calibration_pot_pose.tolist(),
                    "measured_pad_centers_world": (
                        calibration_pad_centers.tolist()
                    ),
                    "collision_free_pregrasp_m": collision_free_pregrasp_m,
                }
            )
            handle_grasp_geometry["left"][
                "static_precontact_jaw_translation"
            ] = static_precontact_jaw_translation
        if source_contact_requested:
            from judo_isaaclab.put_marker import (
                compose_pose as compose_marker_pose,
                inverse_pose as inverse_marker_pose,
                quaternion_rotate as rotate_marker_vector,
                transfer_pose as transfer_marker_pose,
            )
            from judo_isaaclab.put_pot import (
                apply_contact_frame_preorientation,
                apply_contact_frame_radial_clearance_waypoint,
                apply_precontact_source_frame_correction,
                apply_source_demo_approach_corridor,
                source_contact_frame_grasp_pose,
            )

            calibration_path = Path(
                args.target_left_source_contact_calibration_trace
            ).resolve()
            critic_path = Path(
                args.target_left_source_contact_critic_json
            ).resolve()
            if not calibration_path.is_file():
                raise FileNotFoundError(
                    f"source-contact calibration trace is missing: {calibration_path}"
                )
            if not critic_path.is_file():
                raise FileNotFoundError(
                    f"source-contact critic is missing: {critic_path}"
                )
            with open(critic_path, encoding="utf-8") as stream:
                critic = json.load(stream)
            critic_gate = critic.get(
                "gate_decision", critic.get("strict_gate_decision", {})
            )
            critic_acquisition_failed = (
                critic_gate.get("robust_bilateral_latch") is False
                or critic_gate.get("robust_acquisition_passed") is False
            )
            if (
                critic.get("classification") != "failure_or_critic"
                or not critic_acquisition_failed
                or critic.get("artifacts", {}).get("trace_sha256")
                != _sha256(calibration_path)
            ):
                raise ValueError(
                    "source-contact critic does not own a failed calibration trace"
                )
            with np.load(calibration_path, allow_pickle=False) as calibration:
                required_arrays = {
                    "pot_poses",
                    "left_eef_poses",
                    "left_pad_centers_world",
                    "left_pad_axes_world",
                }
                if args.target_right_handle_local_mpc_bootstrap:
                    required_arrays.update(
                        {
                            "right_eef_poses",
                            "right_pad_centers_world",
                            "right_pad_axes_world",
                        }
                    )
                if not required_arrays.issubset(calibration.files):
                    raise ValueError(
                        "source-contact calibration trace lacks wrist/pad geometry"
                    )
                calibration_step = int(
                    args.target_left_source_contact_calibration_step
                )
                if not 0 <= calibration_step < len(
                    calibration["left_eef_poses"]
                ):
                    raise ValueError(
                        "source-contact calibration step is out of range"
                    )
                calibration_wrist = np.asarray(
                    calibration["left_eef_poses"][calibration_step],
                    dtype=np.float64,
                )
                calibration_pot_pose = np.asarray(
                    calibration["pot_poses"][calibration_step],
                    dtype=np.float64,
                )
                calibration_pad_centers = np.asarray(
                    calibration["left_pad_centers_world"][calibration_step],
                    dtype=np.float64,
                )
                calibration_pad_axes = np.asarray(
                    calibration["left_pad_axes_world"][calibration_step],
                    dtype=np.float64,
                )
                if args.target_right_handle_local_mpc_bootstrap:
                    calibration_right_wrist = np.asarray(
                        calibration["right_eef_poses"][calibration_step],
                        dtype=np.float64,
                    )
                    calibration_right_pad_centers = np.asarray(
                        calibration["right_pad_centers_world"][calibration_step],
                        dtype=np.float64,
                    )
                    calibration_right_pad_axes = np.asarray(
                        calibration["right_pad_axes_world"][calibration_step],
                        dtype=np.float64,
                    )
            source_left_grasp = keyframes["frames"]["left_handle_grasp"]
            desired_grasp, frame_receipt = source_contact_frame_grasp_pose(
                source_left_grasp["left_eef_pose"],
                source_left_grasp["pot_pose"],
                calibration_pot_pose,
                handle_grasp_geometry["left"][
                    "source_contact_frame_local"
                ],
                handle_grasp_geometry["left"][
                    "target_contact_frame_local"
                ],
                calibration_wrist,
                calibration_pad_centers,
                calibration_pad_axes,
            )
            calibration_pot_inverse = inverse_marker_pose(calibration_pot_pose)
            local_mpc_left_jaw_axis_prior_local = rotate_marker_vector(
                calibration_pot_inverse[3:],
                np.asarray(frame_receipt["jaw_axis_world"], dtype=np.float64),
            )
            local_mpc_left_pad_axis_prior_local = rotate_marker_vector(
                calibration_pot_inverse[3:],
                np.asarray(frame_receipt["mean_pad_axis_world"], dtype=np.float64),
            )
            if args.target_right_handle_local_mpc_bootstrap:
                source_right_grasp = keyframes["frames"]["right_handle_grasp"]
                (
                    desired_right_source_contact_wrist,
                    local_mpc_right_frame_receipt,
                ) = source_contact_frame_grasp_pose(
                    source_right_grasp["right_eef_pose"],
                    source_right_grasp["pot_pose"],
                    calibration_pot_pose,
                    handle_grasp_geometry["right"][
                        "source_contact_frame_local"
                    ],
                    handle_grasp_geometry["right"][
                        "target_contact_frame_local"
                    ],
                    calibration_right_wrist,
                    calibration_right_pad_centers,
                    calibration_right_pad_axes,
                )
                right_predicted_fractions = np.asarray(
                    local_mpc_right_frame_receipt[
                        "predicted_contact_pad_fractions"
                    ],
                    dtype=np.float64,
                )
                if not np.all(
                    np.isfinite(right_predicted_fractions)
                    & (right_predicted_fractions >= 0.10)
                    & (right_predicted_fractions <= 0.90)
                ):
                    raise ValueError(
                        "right source-contact prediction does not center both pads"
                    )
                local_mpc_right_contact_prior_local = compose_marker_pose(
                    calibration_pot_inverse,
                    desired_right_source_contact_wrist,
                )
                local_mpc_right_jaw_axis_prior_local = rotate_marker_vector(
                    calibration_pot_inverse[3:],
                    np.asarray(
                        local_mpc_right_frame_receipt["jaw_axis_world"],
                        dtype=np.float64,
                    ),
                )
                local_mpc_right_pad_axis_prior_local = rotate_marker_vector(
                    calibration_pot_inverse[3:],
                    np.asarray(
                        local_mpc_right_frame_receipt["mean_pad_axis_world"],
                        dtype=np.float64,
                    ),
                )
            predicted_fractions = np.asarray(
                frame_receipt["predicted_contact_pad_fractions"],
                dtype=np.float64,
            )
            if not np.all(
                np.isfinite(predicted_fractions)
                & (predicted_fractions >= 0.10)
                & (predicted_fractions <= 0.90)
            ):
                raise ValueError(
                    "source-contact prediction does not center both pad contacts"
                )
            target_handle_size = (
                target_parts.negative_handle_size
                if int(handle_grasp_geometry["left"]["handle_side"]) < 0
                else target_parts.positive_handle_size
            )
            position_bound = float(
                np.linalg.norm(target_handle_size) + args.collision_clearance_m
            )
            source_left_pregrasp = keyframes["frames"]["left_pregrasp"]
            target_contact_world = compose_marker_pose(
                calibration_pot_pose,
                handle_grasp_geometry["left"]["target_contact_frame_local"],
            )
            diagnostic_target_left_contact_frame = target_contact_world.copy()
            if args.target_left_source_approach_corridor:
                source_pregrasp_contact_world = compose_marker_pose(
                    source_left_pregrasp["pot_pose"],
                    handle_grasp_geometry["left"][
                        "source_contact_frame_local"
                    ],
                )
                desired_pregrasp = transfer_marker_pose(
                    source_left_pregrasp["left_eef_pose"],
                    source_pregrasp_contact_world,
                    target_contact_world,
                )
                desired_pregrasp[:3] += np.asarray(
                    frame_receipt["jaw_centering_translation_world_m"],
                    dtype=np.float64,
                )
                trajectory, trajectory_receipt = (
                    apply_source_demo_approach_corridor(
                        trajectory,
                        left_reset_pose,
                        desired_pregrasp,
                        desired_grasp,
                        maximum_position_correction_m=position_bound,
                        maximum_position_step_m=args.max_position_step,
                        maximum_orientation_step_rad=args.max_rotation_step,
                        preserve_left_hold_until_step=(
                            trajectory.waypoint_steps["right_handle_grasp"]
                            if args.target_right_first_stabilized_acquisition
                            else None
                        ),
                    )
                )
                if preorientation_requested:
                    trajectory, preorientation_receipt = (
                        apply_contact_frame_preorientation(
                            trajectory,
                            left_reset_pose,
                            desired_pregrasp,
                            alignment_complete_step=(
                                args.target_left_contact_frame_preorientation_complete_step
                            ),
                            prior_first_force_step=(
                                args.target_left_contact_frame_prior_first_force_step
                            ),
                            maximum_orientation_step_rad=args.max_rotation_step,
                        )
                    )
                    trajectory_receipt["contact_frame_preorientation"] = (
                        preorientation_receipt
                    )
                    if radial_waypoint_requested:
                        target_contact_normal_world = rotate_marker_vector(
                            target_contact_world[3:],
                            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
                        )
                        trajectory, radial_waypoint_receipt = (
                            apply_contact_frame_radial_clearance_waypoint(
                                trajectory,
                                left_reset_pose,
                                desired_pregrasp,
                                target_contact_normal_world,
                                clearance_m=(
                                    args.target_left_contact_frame_radial_clearance_m
                                ),
                                waypoint_step=(
                                    args.target_left_contact_frame_radial_waypoint_step
                                ),
                                maximum_position_step_m=args.max_position_step,
                            )
                        )
                        trajectory_receipt["radial_clearance_waypoint"] = (
                            radial_waypoint_receipt
                        )
                        mechanism = (
                            "target_contact_frame_radial_clearance_waypoint"
                        )
                    else:
                        mechanism = (
                            "target_contact_frame_preorientation_before_contact"
                        )
                else:
                    mechanism = (
                        "right_first_handle_local_receding_horizon_bootstrap"
                        if args.target_right_handle_local_mpc_bootstrap
                        else (
                            "deterministic_handle_local_receding_horizon_acquisition"
                            if args.target_handle_local_mpc_acquisition
                            else (
                                "right_first_stabilized_source_contact_acquisition"
                                if args.target_right_first_stabilized_acquisition
                                else (
                                    "source_demo_left_first_acquisition_chronology"
                                    if args.target_source_left_first_acquisition
                                    else "source_demo_pregrasp_contact_corridor"
                                )
                            )
                        )
                    )
            else:
                trajectory, trajectory_receipt = (
                    apply_precontact_source_frame_correction(
                        trajectory,
                        desired_grasp,
                        maximum_position_correction_m=position_bound,
                    )
                )
                mechanism = "source_local_contact_frame_wrist_correction"
            source_contact_frame_correction = {
                "mechanism": mechanism,
                "calibration_trace": {
                    "path": str(calibration_path),
                    "sha256": _sha256(calibration_path),
                    "sample_step": calibration_step,
                },
                "deterministic_target_pot_pose": (
                    calibration_pot_pose.tolist()
                ),
                "critic": {
                    "path": str(critic_path),
                    "sha256": _sha256(critic_path),
                },
                "source_keyframe": {
                    "name": "left_handle_grasp",
                    "sample_index": int(source_left_grasp["sample_index"]),
                },
                "source_pregrasp_keyframe": (
                    {
                        "name": "left_pregrasp",
                        "sample_index": int(
                            source_left_pregrasp["sample_index"]
                        ),
                    }
                    if args.target_left_source_approach_corridor
                    else None
                ),
                "right_trajectory_unchanged": not bool(
                    args.target_source_left_first_acquisition
                    or args.target_right_handle_local_mpc_bootstrap
                ),
                "right_acquisition_endpoint_preserved": not bool(
                    args.target_right_handle_local_mpc_bootstrap
                ),
                "right_handle_local_bootstrap": bool(
                    args.target_right_handle_local_mpc_bootstrap
                ),
                "right_acquisition_delayed_until_left_close": bool(
                    args.target_source_left_first_acquisition
                ),
                "right_first_stabilization": bool(
                    args.target_right_first_stabilized_acquisition
                ),
                "frame_measurement": frame_receipt,
                "right_bootstrap_frame_measurement": (
                    local_mpc_right_frame_receipt
                ),
                "trajectory_correction": trajectory_receipt,
            }
            handle_grasp_geometry["left"][
                "source_contact_frame_correction"
            ] = source_contact_frame_correction
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
                quaternion_rotate,
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
        local_mpc_config = None
        local_mpc_frame_receipts = []
        local_mpc_contact_window_step = 0
        local_mpc_robust_streak = 0
        local_mpc_latch_ready = False
        local_mpc_right_contact_window_step = 0
        local_mpc_right_robust_streak = 0
        local_mpc_right_latch_ready = False
        local_mpc_right_bootstrap_start_step = None
        local_mpc_fail_closed = False
        local_mpc_fail_reason = None
        local_mpc_last_control_vectors_world = {
            "left": np.zeros(3, dtype=np.float64),
            "right": np.zeros(3, dtype=np.float64),
        }
        local_mpc_left_contact_prior = (
            None if left_handle_contact is None else left_handle_contact.copy()
        )
        if args.target_handle_local_mpc_acquisition:
            from judo_isaaclab.putpot_local_mpc import HandleLocalMpcConfig

            local_mpc_config = HandleLocalMpcConfig(
                maximum_translation_step_m=min(0.004, args.max_position_step),
                maximum_rotation_step_rad=min(0.08, args.max_rotation_step),
            )
            if args.target_right_handle_local_mpc_bootstrap:
                right_jaw = np.asarray(trajectory.grippers[:, 1], dtype=np.float64)
                right_closure = np.flatnonzero(
                    right_jaw > right_jaw[0] + 1.0e-9
                )
                if not len(right_closure):
                    raise ValueError(
                        "right handle-local bootstrap has no source-timed jaw closure"
                    )
                local_mpc_right_bootstrap_start_step = int(right_closure[0])
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
        peer_contact_jaw_center_translation_m = None
        peer_contact_authored_jaw_center_locked = False
        peer_contact_latch_centering_applied_m = 0.0
        peer_contact_handle_center_rotation_rad = None
        peer_contact_handle_center_post_pivot_residual_m = None
        peer_contact_latch_centering_translation_m = 0.0
        peer_contact_centered_tracking_latch_step = None
        peer_contact_observed_jaw_residuals_m = []
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
        contact_hold_pick_lift_steps = []
        contact_hold_pick_lift_command_m = 0.0
        contact_hold_pick_recovery_correction_m = 0.0
        contact_hold_pick_recovery_steps = []
        contact_hold_pad_support_correction_m = 0.0
        contact_hold_pad_support_steps = []
        coded_pick_hold_retime_step = None
        coded_pick_hold_removed_steps = 0
        reference_hold_left_contact_local = None
        reference_hold_right_contact_local = None
        transport_reanchor_steps = []
        transport_reanchor_evaluation_steps = []
        transport_reanchor_signed_residuals_world_m = []
        transport_reanchor_rejections = []
        transport_pad_support_correction_m = 0.0
        transport_pad_support_steps = []
        transport_reference_observation_init_steps = []
        transport_reference_left_contact_local = None
        transport_reference_right_contact_local = None
        transport_expected_left_tracking_residual_local = None
        transport_expected_right_tracking_residual_local = None
        transport_motion_preload_local_m = None
        transport_loaded_vertical_rise_fraction = None
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
        if args.acquisition_only:
            total_steps = int(grasp_complete_step) + 1
        timers.add("trajectory_build", time.monotonic() - trajectory_started)
        from judo_isaaclab.demo_artifact import DemonstrationRecorder

        demo_recorder = DemonstrationRecorder()
        demo_recorder.start(env.scene.get_state(is_relative=False))
        samples = [_sample(env, -1, "reset")]
        actions = []; pot_poses = []; left_eef = []; right_eef = []; desired_left = []; desired_right = []
        if args.controller_plugin_py:
            from judo_isaaclab.putpot_controller_protocol import (
                ControllerPluginClient,
            )

            controller_client = ControllerPluginClient(
                args.controller_plugin_py,
                args.controller_plugin_sha256,
                runner_path=REPO_ROOT / "examples/run_putpot_controller_plugin.py",
                log_path=args.controller_plugin_log,
                timeout_s=args.controller_timeout_s,
            )
            initialized = controller_client.initialize(
                {
                    "attempt_identity": (
                        None
                        if attempt_identity is None
                        else attempt_identity.receipt()
                    ),
                    "program_parameters": dict(program_spec.parameters),
                    "initial_observation": _controller_observation(samples[-1]),
                    "base_trajectory": {
                        "steps": int(trajectory.steps),
                        "left_poses": trajectory.left_poses,
                        "right_poses": trajectory.right_poses,
                        "grippers": trajectory.grippers,
                        "stage_names": trajectory.stage_names,
                        "waypoint_steps": trajectory.waypoint_steps,
                    },
                    "geometry": {
                        "target_pot_root_pose": target_geometry.root_pose,
                        "target_pot_size": target_geometry.size,
                        "target_cooktop_root_pose": target_cooktop_geometry.root_pose,
                        "target_cooktop_size": target_cooktop_geometry.size,
                        "target_handle_axis": int(target_parts.handle_axis),
                        "target_negative_handle_size": target_parts.negative_handle_size,
                        "target_positive_handle_size": target_parts.positive_handle_size,
                    },
                },
                int(total_steps),
            )
            total_steps = initialized["total_steps"]
            controller_receipt = controller_client.receipt()
        frame_stats = []
        acquisition_fail_closed = False
        acquisition_fail_closed_step = None
        from judo_isaaclab.put_pot import (
            ROBUST_BIMANUAL_LATCH_STEPS,
            robust_bimanual_latch_ready,
        )

        debug_axis_draw = None
        if args.render:
            render_started = time.monotonic()
            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            encoder = _Encoder(args.fps, args.video)
            timers.add("render_encode", time.monotonic() - render_started)
        if args.render_diagnostic_only:
            if diagnostic_target_contact_frames_local is None:
                raise RuntimeError("target bimanual contact frames were not measured")
            from isaacsim.core.utils.extensions import enable_extension

            if not enable_extension("isaacsim.util.debug_draw"):
                raise RuntimeError("could not enable isaacsim.util.debug_draw")
            import isaacsim.util.debug_draw._debug_draw as debug_draw

            debug_axis_draw = debug_draw.acquire_debug_draw_interface()
        rollout_started = time.monotonic()
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
                if step == grasp_complete_step + 1:
                    acquisition_fail_closed = not robust_bimanual_latch_ready(
                        [row["left_grasp"] for row in samples],
                        [row["right_grasp"] for row in samples],
                        left_finger_forces_n=[
                            row["left_finger_forces_n"] for row in samples
                        ],
                        right_finger_forces_n=[
                            row["right_finger_forces_n"] for row in samples
                        ],
                        left_pad_fractions=[
                            row["left_pad_fractions"] for row in samples
                        ],
                        right_pad_fractions=[
                            row["right_pad_fractions"] for row in samples
                        ],
                    )
                    if acquisition_fail_closed:
                        acquisition_fail_closed_step = step
                        print(
                            "PUTPOT_FAIL_CLOSED="
                            + json.dumps(
                                {
                                    "reason": "robust_bimanual_latch_missing",
                                    "required_consecutive_frames": (
                                        ROBUST_BIMANUAL_LATCH_STEPS
                                    ),
                                    "step": step,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                if acquisition_fail_closed:
                    held_grippers = (
                        np.asarray([actions[-1][6], actions[-1][13]], dtype=np.float64)
                        if args.target_handle_local_mpc_acquisition and actions
                        else np.zeros(2, dtype=np.float64)
                    )
                    base_command = {
                        "stage": "bimanual_handle_grasp_fail_closed",
                        "left_pose": np.asarray(samples[-1]["left_eef_pose"]),
                        "right_pose": np.asarray(samples[-1]["right_eef_pose"]),
                        "grippers": held_grippers,
                        "joint_nominal": joint_nominal[step],
                    }
                else:
                    base_command = {
                        "stage": trajectory.stage_names[step],
                        "left_pose": trajectory.left_poses[step],
                        "right_pose": trajectory.right_poses[step],
                        "grippers": trajectory.grippers[step],
                        "joint_nominal": joint_nominal[step],
                    }
                command = (
                    None
                    if controller_client is None
                    else controller_client.command(
                        step=step,
                        base_command=base_command,
                        observation=_controller_observation(samples[-1]),
                    )
                )
                if command is not None and command["terminate"]:
                    break
                stage = (
                    trajectory.stage_names[step]
                    if command is None
                    else command["stage"]
                )
                integrate_ik = bool(integrate_target_ik)
                if command is not None and command["kind"] == "joint_action":
                    action = torch.as_tensor(
                        [command["action"]],
                        device=env.device,
                        dtype=source["actions"].dtype,
                    )
                    desired_left.append(np.asarray(samples[-1]["left_eef_pose"]))
                    desired_right.append(np.asarray(samples[-1]["right_eef_pose"]))
                else:
                    left_target = (
                        trajectory.left_poses[step]
                        if command is None
                        else np.asarray(command["left_pose"], dtype=np.float64)
                    )
                    right_target = (
                        trajectory.right_poses[step]
                        if command is None
                        else np.asarray(command["right_pose"], dtype=np.float64)
                    )
                    grippers = (
                        trajectory.grippers[step]
                        if command is None
                        else np.asarray(command["grippers"], dtype=np.float64)
                    )
                    grippers = np.asarray(grippers, dtype=np.float64).copy()
                    joint_nominal_weight = None
                    if args.target_handle_local_mpc_acquisition:
                        from judo_isaaclab.put_marker import compose_pose
                        from judo_isaaclab.putpot_local_mpc import (
                            contact_window_joint_nominal_weight,
                            handle_local_bootstrap_active,
                            handle_local_mpc_active,
                            handle_local_mpc_step,
                        )

                        right_bootstrap_active = bool(
                            args.target_right_handle_local_mpc_bootstrap
                            and handle_local_bootstrap_active(
                                enabled=not local_mpc_fail_closed,
                                latch_ready=local_mpc_right_latch_ready,
                                step=step,
                                bootstrap_start_step=(
                                    local_mpc_right_bootstrap_start_step
                                ),
                                grasp_complete_step=grasp_complete_step,
                            )
                        )
                        left_mpc_active = handle_local_mpc_active(
                            enabled=not local_mpc_fail_closed,
                            peer_latched=(
                                local_mpc_right_latch_ready
                                if args.target_right_handle_local_mpc_bootstrap
                                else bool(samples[-1]["right_grasp"])
                            ),
                            step=step,
                            peer_latch_step=right_grasp_step,
                            grasp_complete_step=grasp_complete_step,
                        )
                        mpc_active = right_bootstrap_active or left_mpc_active
                        right_contact_hold = bool(
                            args.target_right_handle_local_mpc_bootstrap
                            and local_mpc_right_latch_ready
                        )
                        joint_nominal_weight = contact_window_joint_nominal_weight(
                            active=True
                        ) if (mpc_active or right_contact_hold) else None
                        if mpc_active:
                            active_arm = (
                                "right" if right_bootstrap_active else "left"
                            )
                            peer_arm = "left" if active_arm == "right" else "right"
                            contact_origin = None
                            for prior_sample in samples:
                                all_forces = np.concatenate(
                                    (
                                        np.asarray(
                                            prior_sample["left_finger_forces_n"],
                                            dtype=np.float64,
                                        ),
                                        np.asarray(
                                            prior_sample["right_finger_forces_n"],
                                            dtype=np.float64,
                                        ),
                                    )
                                )
                                all_fractions = np.concatenate(
                                    (
                                        np.asarray(
                                            prior_sample["left_pad_fractions"],
                                            dtype=np.float64,
                                        ),
                                        np.asarray(
                                            prior_sample["right_pad_fractions"],
                                            dtype=np.float64,
                                        ),
                                    )
                                )
                                physical = (
                                    (all_forces >= 0.1)
                                    & np.isfinite(all_fractions)
                                    & (all_fractions >= 0.0)
                                    & (all_fractions <= 1.0)
                                )
                                if np.any(physical):
                                    contact_origin = np.asarray(
                                        prior_sample["pot_pose"], dtype=np.float64
                                    )[:3]
                                    break
                            if contact_origin is None:
                                contact_origin = np.asarray(
                                    samples[-1]["pot_pose"], dtype=np.float64
                                )[:3]
                            pre_peer_displacement_m = float(
                                np.linalg.norm(
                                    np.asarray(
                                        samples[-1]["pot_pose"], dtype=np.float64
                                    )[:3]
                                    - contact_origin
                                )
                            )
                            jaw_index = 13 if active_arm == "right" else 6
                            gripper_index = 1 if active_arm == "right" else 0
                            current_jaw = float(
                                actions[-1][jaw_index]
                                if actions
                                else grippers[gripper_index]
                            )
                            observed_handle = compose_pose(
                                samples[-1]["pot_pose"],
                                diagnostic_target_contact_frames_local[active_arm],
                            )
                            contact_prior_local = (
                                local_mpc_right_contact_prior_local
                                if active_arm == "right"
                                else local_mpc_left_contact_prior
                            )
                            object_relative_prior = compose_pose(
                                samples[-1]["pot_pose"],
                                contact_prior_local,
                            )
                            jaw_axis_prior_local = (
                                local_mpc_right_jaw_axis_prior_local
                                if active_arm == "right"
                                else local_mpc_left_jaw_axis_prior_local
                            )
                            pad_axis_prior_local = (
                                local_mpc_right_pad_axis_prior_local
                                if active_arm == "right"
                                else local_mpc_left_pad_axis_prior_local
                            )
                            warm_start_target = (
                                right_target
                                if active_arm == "right"
                                else left_target
                            )
                            active_streak = (
                                local_mpc_right_robust_streak
                                if active_arm == "right"
                                else local_mpc_robust_streak
                            )
                            local_command = handle_local_mpc_step(
                                contact_window_step=(
                                    local_mpc_right_contact_window_step
                                    if active_arm == "right"
                                    else local_mpc_contact_window_step
                                ),
                                observed_pot_pose=samples[-1]["pot_pose"],
                                observed_handle_contact_frame=observed_handle,
                                active_wrist_pose=samples[-1][
                                    f"{active_arm}_eef_pose"
                                ],
                                object_relative_wrist_prior=object_relative_prior,
                                object_relative_jaw_axis_prior=jaw_axis_prior_local,
                                object_relative_pad_depth_axis_prior=(
                                    pad_axis_prior_local
                                ),
                                source_warm_start_wrist_pose=warm_start_target,
                                active_pad_centers_world=samples[-1][
                                    f"{active_arm}_pad_centers_world"
                                ],
                                active_pad_axes_world=samples[-1][
                                    f"{active_arm}_pad_axes_world"
                                ],
                                active_pad_fractions=samples[-1][
                                    f"{active_arm}_pad_fractions"
                                ],
                                active_finger_forces_n=samples[-1][
                                    f"{active_arm}_finger_forces_n"
                                ],
                                peer_pad_fractions=samples[-1][
                                    f"{peer_arm}_pad_fractions"
                                ],
                                peer_finger_forces_n=samples[-1][
                                    f"{peer_arm}_finger_forces_n"
                                ],
                                active_grasp=bool(
                                    samples[-1][f"{active_arm}_grasp"]
                                ),
                                peer_grasp=bool(
                                    samples[-1][f"{peer_arm}_grasp"]
                                ),
                                pre_peer_pot_displacement_m=pre_peer_displacement_m,
                                current_jaw_command=current_jaw,
                                robust_streak=active_streak,
                                require_peer_latch=active_arm == "left",
                                config=local_mpc_config,
                            )
                            local_mpc_frame_receipts.append(
                                {
                                    "program_step": step,
                                    "active_arm": active_arm,
                                    "receipt": local_command.frame_receipt,
                                }
                            )
                            if active_arm == "right":
                                local_mpc_right_contact_window_step += 1
                                local_mpc_right_robust_streak = (
                                    local_command.robust_streak
                                )
                                local_mpc_right_latch_ready = (
                                    local_command.robust_latch_ready
                                )
                                right_target = local_command.wrist_target_pose
                                left_target = np.asarray(
                                    samples[-1]["left_eef_pose"],
                                    dtype=np.float64,
                                )
                                grippers[1] = local_command.jaw_command
                                if actions:
                                    grippers[0] = float(actions[-1][6])
                                stage = "right_handle_local_mpc_bootstrap"
                            else:
                                local_mpc_contact_window_step += 1
                                local_mpc_robust_streak = (
                                    local_command.robust_streak
                                )
                                local_mpc_latch_ready = (
                                    local_command.robust_latch_ready
                                )
                                left_target = local_command.wrist_target_pose
                                right_target = np.asarray(
                                    samples[-1]["right_eef_pose"],
                                    dtype=np.float64,
                                )
                                grippers[0] = local_command.jaw_command
                                if actions:
                                    grippers[1] = float(actions[-1][13])
                                stage = "handle_local_mpc_contact_window"
                            local_mpc_last_control_vectors_world = {
                                "left": np.zeros(3, dtype=np.float64),
                                "right": np.zeros(3, dtype=np.float64),
                            }
                            local_mpc_last_control_vectors_world[active_arm] = (
                                np.asarray(
                                    local_command.frame_receipt[
                                        "executed_control"
                                    ]["translation_world_m"],
                                    dtype=np.float64,
                                )
                            )
                            if local_command.fail_closed:
                                local_mpc_fail_closed = True
                                local_mpc_fail_reason = local_command.fail_reason
                                acquisition_fail_closed = True
                                acquisition_fail_closed_step = step
                                print(
                                    "PUTPOT_FAIL_CLOSED="
                                    + json.dumps(
                                        {
                                            "reason": local_mpc_fail_reason,
                                            "step": step,
                                            "controller": (
                                                "deterministic_handle_local_mpc"
                                            ),
                                        },
                                        sort_keys=True,
                                    ),
                                    flush=True,
                                )
                        elif right_contact_hold:
                            right_target = np.asarray(
                                samples[-1]["right_eef_pose"], dtype=np.float64
                            )
                            if actions:
                                grippers[1] = float(actions[-1][13])
                    action = _ik_action(
                        env,
                        left_target,
                        right_target,
                        grippers,
                        joint_nominal[step],
                        args,
                        integrate_left_ik=integrate_ik,
                        integrate_right_ik=integrate_ik,
                        joint_nominal_weight=joint_nominal_weight,
                    )
                    desired_left.append(left_target)
                    desired_right.append(right_target)
                if command is not None:
                    controller_command_count += 1
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
                _milestone_reanchor_enabled(
                    right_first_close=right_first_close,
                    forced_right_first_stabilization=bool(
                        args.target_right_first_stabilized_acquisition
                    ),
                )
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
                from judo_isaaclab.put_pot import (
                    peer_contact_transfer_horizon_steps,
                )

                milestone_feedback_horizon_steps = (
                    peer_contact_transfer_horizon_steps(
                        milestone_applied_translation_m
                    )
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
                and not args.target_handle_local_mpc_acquisition
                and pregrasp_complete_step <= step < grasp_complete_step
            ):
                from judo_isaaclab.put_pot import (
                    CONTACT_FEEDBACK_HORIZON_STEPS,
                    MISSING_FINGER_CONTACT_DELAY_STEPS,
                    SINGLE_FINGER_CONTACT_LATCH_STEPS,
                    reanchor_missing_finger_contact,
                    reanchor_missing_finger_pad_depth,
                    reanchor_single_contact_pad_fraction,
                    peer_contact_gripper_reseat_distance_m,
                    retime_loaded_gripper_close_for_pad_reseat,
                    single_contact_pad_base_residual_m,
                    preserve_loaded_contact_target,
                    peer_contact_latch_supported,
                    single_finger_contact_observed,
                    track_bimanual_handle_targets,
                    track_loaded_pad_center_from_observation,
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

                supported_peer_contact = peer_contact_latch_supported(
                    peer_contact_transfer,
                    sample["left_finger_forces_n"],
                    sample["left_pad_fractions"],
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
                        bounded_receiving_jaw_reorientation,
                        center_handle_between_finger_pads,
                        handle_jaw_center_offset_m,
                        receiving_jaw_center_translation_m,
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
                    peer_contact_jaw_center_translation_m = (
                        receiving_jaw_center_translation_m(
                            peer_contact_pre_twist_jaw_residual_m,
                            args.receiving_jaw_center_translation_fraction,
                            args.max_position_step,
                        )
                    )
                    loaded_left_world = center_handle_between_finger_pads(
                        loaded_left_world,
                        peer_contact_jaw_center_translation_m,
                        maximum_correction_m=args.max_position_step,
                    )
                    jaw_residual_for_twist_m = (
                        handle_jaw_center_offset_m(
                            loaded_left_world,
                            sample["pot_pose"],
                            target_left_handle_points,
                        )
                    )
                    milestone_feedback_horizon_steps = 1
                    peer_contact_pad_reseat_m = milestone_open_pad_reseat_m
                    (
                        loaded_left_world,
                        peer_contact_jaw_twist_rad,
                        peer_contact_position_locked,
                        peer_contact_jaw_twist_fraction,
                    ) = bounded_receiving_jaw_reorientation(
                        sample["left_eef_pose"],
                        loaded_left_world,
                        sample["left_finger_forces_n"],
                        sample["left_pad_centers_world"],
                        sample["left_pad_axes_world"],
                        jaw_residual_for_twist_m,
                        rotation_fraction=(
                            args.receiving_jaw_reorientation_fraction
                        ),
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
                        peer_contact_gripper_reseat_distance_m(
                            initial_pad_reseat_residual_m,
                            peer_contact_latch_centering_translation_m,
                            position_locked=peer_contact_position_locked,
                            additional_jaw_centering_translation_m=abs(
                                peer_contact_jaw_center_translation_m
                            ),
                        )
                    )
                    # The receiving wrist is loaded against one pad here.
                    # Retain the conservative 1 mm/control-step horizon for
                    # measured jaw translations as well as pad reseating;
                    # the 2 mm horizon closed before the CPU IK caught up on
                    # long, object-local corrections.
                    gripper_retime_step_m = 0.001
                    trajectory, gripper_hold_steps = (
                        retime_loaded_gripper_close_for_pad_reseat(
                            trajectory,
                            step,
                            effective_pad_reseat_residual_m,
                            reseat_step_m=gripper_retime_step_m,
                            close_steps=(
                                None
                                if args.receiving_jaw_close_horizon_steps == 0
                                else args.receiving_jaw_close_horizon_steps
                            ),
                        )
                    )
                    available_close_steps = (
                        grasp_complete_step - step - gripper_hold_steps
                    )
                    applied_close_steps = min(
                        available_close_steps,
                        (
                            available_close_steps
                            if args.receiving_jaw_close_horizon_steps == 0
                            else args.receiving_jaw_close_horizon_steps
                        ),
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
                        "jaw_center_translation_m": (
                            peer_contact_jaw_center_translation_m
                        ),
                        "hold_steps": gripper_hold_steps,
                        "hold_step_m": gripper_retime_step_m,
                        "requested_close_horizon_steps": (
                            args.receiving_jaw_close_horizon_steps
                        ),
                        "applied_close_steps": applied_close_steps,
                        "close_start_step": step + gripper_hold_steps,
                        "close_end_step": (
                            step + gripper_hold_steps + applied_close_steps
                        ),
                        "grasp_end_step": grasp_complete_step,
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

                if (
                    peer_single_contact_latch_step is not None
                    and abs(peer_contact_latch_centering_applied_m) > 0.0
                    and not peer_contact_position_locked
                ):
                    from judo_isaaclab.put_pot import (
                        HANDLE_PAD_GEOMETRIC_MARGIN_M,
                        handle_jaw_center_offset_m,
                    )

                    observed_jaw_residual_m = handle_jaw_center_offset_m(
                        sample["left_eef_pose"],
                        sample["pot_pose"],
                        target_left_handle_points,
                    )
                    peer_contact_observed_jaw_residuals_m.append(
                        {
                            "step": step,
                            "signed_residual_m": observed_jaw_residual_m,
                        }
                    )
                    if (
                        abs(observed_jaw_residual_m)
                        <= HANDLE_PAD_GEOMETRIC_MARGIN_M
                        and single_finger_contact_observed(
                            sample["left_finger_forces_n"],
                            sample["left_pad_fractions"],
                        )
                    ):
                        left_handle_contact = compose_pose(
                            inverse_pose(sample["pot_pose"]),
                            sample["left_eef_pose"],
                        )
                        peer_contact_position_locked = True
                        peer_contact_centered_tracking_latch_step = step

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
                            gripper_pose_world=sample[f"{arm}_eef_pose"],
                            limit_m=args.missing_finger_contact_limit_m,
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
                    peer_single_contact_latch_step is not None
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
                and not args.target_handle_local_mpc_acquisition
                and contact_close_complete_step <= step < grasp_complete_step
            ):
                from judo_isaaclab.put_pot import (
                    TRANSPORT_CONTACT_REANCHOR_MIN_STEPS,
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
                    left_contacting = (
                        np.asarray(sample["left_finger_forces_n"]) >= 0.1
                    )
                    if sample["left_grasp"] and sample["right_grasp"]:
                        from judo_isaaclab.put_pot import (
                            reanchor_two_contact_pad_support,
                        )

                        (
                            reference_hold_left_contact_local,
                            contact_hold_pad_support_correction_m,
                            pad_support_residual_m,
                        ) = reanchor_two_contact_pad_support(
                            reference_hold_left_contact_local,
                            sample["pot_pose"],
                            sample["left_finger_forces_n"],
                            sample["left_pad_fractions"],
                            sample["left_pad_axes_world"],
                            contact_hold_pad_support_correction_m,
                        )
                        if abs(pad_support_residual_m) > 0.0:
                            contact_hold_pad_support_steps.append(
                                {
                                    "step": step,
                                    "signed_residual_m": pad_support_residual_m,
                                    "cumulative_applied_m": (
                                        contact_hold_pad_support_correction_m
                                    ),
                                }
                            )
                    if (
                        not sample["left_grasp"]
                        and sample["right_grasp"]
                        and int(np.sum(left_contacting)) == 1
                    ):
                        from judo_isaaclab.put_pot import (
                            reanchor_missing_finger_contact,
                        )

                        (
                            reference_hold_left_contact_local,
                            contact_hold_pick_recovery_correction_m,
                        ) = reanchor_missing_finger_contact(
                            reference_hold_left_contact_local,
                            sample["pot_pose"],
                            sample["left_finger_forces_n"],
                            sample["left_pad_centers_world"],
                            contact_hold_pick_recovery_correction_m,
                            gripper_pose_world=sample["left_eef_pose"],
                            limit_m=args.missing_finger_contact_limit_m,
                        )
                        contact_hold_pick_recovery_steps.append(
                            {
                                "step": step,
                                "signed_correction_m": (
                                    contact_hold_pick_recovery_correction_m
                                ),
                            }
                        )
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
                    from judo_isaaclab.put_pot import (
                        CONTACT_HOLD_SUPPORT_ALIGNMENT_FRACTION,
                        advance_loaded_contact_hold_lift,
                    )

                    if sample["right_grasp"] and np.any(left_contacting):
                        (
                            contact_hold_pick_lift_command_m,
                            hold_lift_residual_m,
                        ) = advance_loaded_contact_hold_lift(
                            target_geometry.root_pose,
                            sample["pot_pose"],
                            contact_hold_pick_lift_command_m,
                            pick_height_m=loaded_pick_height_m,
                            # Saturate the existing Cartesian action limiter;
                            # CPU loaded-arm tracking otherwise reaches the
                            # pick height only after the thin left pad unloads.
                            maximum_residual_m=2.0 * args.max_position_step,
                        )
                    else:
                        hold_lift_residual_m = 0.0
                    if hold_lift_residual_m > 0.0:
                        contact_hold_pick_lift_steps.append(
                            {
                                "step": step,
                                "commanded_lift_m": (
                                    contact_hold_pick_lift_command_m
                                ),
                                "object_local_lift_residual_m": (
                                    hold_lift_residual_m
                                ),
                                "observed_lift_m": float(
                                    np.asarray(sample["pot_pose"])[2]
                                    - target_geometry.root_pose[2]
                                ),
                            }
                        )
                    trajectory = reanchor_bimanual_contact_hold(
                        trajectory,
                        step,
                        sample["pot_pose"],
                        retained_hold_left_contact_local,
                        reference_hold_right_contact_local,
                        object_local_lift_residual_m=hold_lift_residual_m,
                        support_normal_world=quaternion_rotate(
                            target_cooktop_geometry.root_pose[3:],
                            np.asarray([0.0, 0.0, 1.0]),
                        ),
                        support_aligned_object_orientation_wxyz=(
                            target_geometry.root_pose[3:]
                        ),
                        support_alignment_fraction=(
                            CONTACT_HOLD_SUPPORT_ALIGNMENT_FRACTION
                        ),
                    )
                    if (
                        sample["stage1"]
                        and sample["left_grasp"]
                        and sample["right_grasp"]
                        and minimum_cooktop_clearance_m(
                            np.asarray([sample["pot_pose"]]),
                            target_geometry.size,
                            target_cooktop_geometry,
                        )
                        >= args.collision_clearance_m
                        and step + 1 < grasp_complete_step
                    ):
                        from judo_isaaclab.put_pot import (
                            finish_contact_hold_after_coded_pick,
                        )

                        (
                            trajectory,
                            coded_pick_hold_removed_steps,
                        ) = finish_contact_hold_after_coded_pick(
                            trajectory, step
                        )
                        grasp_complete_step = step + 1
                        coded_pick_hold_retime_step = step
                    if step + 1 == grasp_complete_step:
                        from judo_isaaclab.put_pot import (
                            remaining_contact_vertical_rise_fraction,
                        )

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
                        transport_loaded_vertical_rise_fraction = (
                            remaining_contact_vertical_rise_fraction(
                                trajectory.waypoint_steps["smooth_transport"]
                                - step,
                                int(
                                    handle_grasp_geometry["left"][
                                        "peer_contact_hold_steps"
                                    ]
                                ),
                                (
                                    0
                                    if contact_hold_latch_step is None
                                    else step - contact_hold_latch_step
                                ),
                            )
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
                                vertical_rise_fraction=(
                                    transport_loaded_vertical_rise_fraction
                                ),
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
                    reanchor_two_contact_pad_support,
                    resolve_bimanual_contact_frames_local,
                    transport_reanchor_has_smooth_horizon,
                    transport_contact_reanchor_required,
                )

                if (
                    transport_reference_left_contact_local is None
                    or transport_reference_right_contact_local is None
                ):
                    (
                        transport_reference_left_contact_local,
                        transport_reference_right_contact_local,
                    ) = resolve_bimanual_contact_frames_local(
                        sample["pot_pose"],
                        sample["left_eef_pose"],
                        sample["right_eef_pose"],
                        transport_reference_left_contact_local,
                        transport_reference_right_contact_local,
                    )
                    transport_expected_left_tracking_residual_local = (
                        np.zeros(3, dtype=np.float64)
                    )
                    transport_expected_right_tracking_residual_local = (
                        np.zeros(3, dtype=np.float64)
                    )
                    transport_reference_observation_init_steps.append(step)
                (
                    pad_supported_left_contact_local,
                    candidate_pad_support_correction_m,
                    transport_pad_support_residual_m,
                ) = reanchor_two_contact_pad_support(
                    transport_reference_left_contact_local,
                    sample["pot_pose"],
                    sample["left_finger_forces_n"],
                    sample["left_pad_fractions"],
                    sample["left_pad_axes_world"],
                    transport_pad_support_correction_m,
                )
                transport_pad_support_requested = bool(
                    abs(
                        candidate_pad_support_correction_m
                        - transport_pad_support_correction_m
                    )
                    > 1.0e-12
                )
                if (
                    transport_pad_support_requested
                    or transport_contact_reanchor_required(
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
                        tracking_tolerance_m=(
                            transport_contact_tracking_tolerance_m
                        ),
                        expected_left_tracking_residual_local=(
                            transport_expected_left_tracking_residual_local
                        ),
                        expected_right_tracking_residual_local=(
                            transport_expected_right_tracking_residual_local
                        ),
                        # Contact feedback must run on its measured horizon. A
                        # geometry-dependent vertical rise can be much longer
                        # and allowed a newly acquired second handle to detach
                        # before the first drift correction.
                        minimum_interval_steps=(
                            TRANSPORT_CONTACT_REANCHOR_MIN_STEPS
                        ),
                    )
                ) and transport_reanchor_has_smooth_horizon(trajectory, step):
                    transport_reanchor_evaluation_steps.append(step)
                    observed_pot_inverse = inverse_pose(sample["pot_pose"])
                    observed_left_contact_local = compose_pose(
                        observed_pot_inverse, sample["left_eef_pose"]
                    )
                    observed_right_contact_local = compose_pose(
                        observed_pot_inverse, sample["right_eef_pose"]
                    )
                    retained_left_contact_local = (
                        pad_supported_left_contact_local
                        if transport_pad_support_requested
                        else (
                            observed_left_contact_local
                            if transport_reference_left_contact_local is None
                            else compensate_retained_contact_tracking(
                                transport_reference_left_contact_local,
                                sample["pot_pose"],
                                sample["left_eef_pose"],
                            )[0]
                        )
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
                            vertical_rise_fraction=(
                                transport_loaded_vertical_rise_fraction
                                if transport_loaded_vertical_rise_fraction
                                is not None
                                else handle_grasp_geometry["left"][
                                    "transport_vertical_rise_fraction"
                                ]
                            ),
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
                        if transport_pad_support_requested:
                            transport_reference_left_contact_local = (
                                pad_supported_left_contact_local
                            )
                            transport_pad_support_correction_m = (
                                candidate_pad_support_correction_m
                            )
                            transport_expected_left_tracking_residual_local = (
                                transport_reference_left_contact_local[:3]
                                - observed_left_contact_local[:3]
                            )
                            transport_pad_support_steps.append(
                                {
                                    "step": step,
                                    "signed_residual_m": (
                                        transport_pad_support_residual_m
                                    ),
                                    "cumulative_applied_m": (
                                        transport_pad_support_correction_m
                                    ),
                                }
                            )
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
                render_started = time.monotonic()
                diagnostic_target_frames_world = None
                diagnostic_desired_wrist_frames = None
                diagnostic_control_vectors_world = None
                if debug_axis_draw is not None:
                    from judo_isaaclab.put_marker import compose_pose

                    diagnostic_target_frames_world = {
                        arm: compose_pose(
                            sample["pot_pose"],
                            diagnostic_target_contact_frames_local[arm],
                        )
                        for arm in ("left", "right")
                    }
                    diagnostic_desired_wrist_frames = {
                        "left": desired_left[-1],
                        "right": desired_right[-1],
                    }
                    diagnostic_control_vectors_world = (
                        local_mpc_last_control_vectors_world
                        if args.target_handle_local_mpc_acquisition
                        and local_mpc_frame_receipts
                        else {
                            "left": np.asarray(desired_left[-1])[:3]
                            - np.asarray(sample["left_eef_pose"])[:3],
                            "right": np.asarray(desired_right[-1])[:3]
                            - np.asarray(sample["right_eef_pose"])[:3],
                        }
                    )
                frame = _frame(
                    env,
                    sample,
                    debug_axis_draw=debug_axis_draw,
                    target_contact_frames=diagnostic_target_frames_world,
                    desired_wrist_frames=diagnostic_desired_wrist_frames,
                    control_vectors_world=diagnostic_control_vectors_world,
                )
                encoder.write(frame)
                frame_stats.append((float(frame.mean()), float(frame.std())))
                timers.add("render_encode", time.monotonic() - render_started)
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
            render_started = time.monotonic()
            encoder.close(); encoder = None
            timers.add("render_encode", time.monotonic() - render_started)
        rollout_total_s = time.monotonic() - rollout_started
        timers.add(
            "rollout",
            max(0.0, rollout_total_s - timers.seconds["render_encode"]),
        )
        trace_started = time.monotonic()
        _write_rollout_trace(
            args.trace_npz,
            actions=actions,
            pot_poses=pot_poses,
            left_eef=left_eef,
            right_eef=right_eef,
            desired_left=desired_left,
            desired_right=desired_right,
            samples=samples,
            joint_nominal=joint_nominal,
            local_mpc_frame_receipts=local_mpc_frame_receipts,
            partial=False,
        )
        timers.add("trace_demo", time.monotonic() - trace_started)
        acquisition_latch = None
        if trajectory is not None:
            from judo_isaaclab.putpot_repair_policy import trace_latch_evidence

            acquisition_latch = trace_latch_evidence(args.trace_npz).receipt()
        if (
            args.target_handle_local_mpc_acquisition
            and not bool(
                acquisition_latch is not None
                and acquisition_latch["passes_robust_latch"]
            )
        ):
            local_mpc_fail_closed = True
            local_mpc_fail_reason = (
                local_mpc_fail_reason or "robust_four_pad_latch_not_sustained"
            )
        final = samples[-1]
        extracted = None
        if args.mode == "replay" and final["task_success"]:
            extracted = _extract_keyframes(samples, np.asarray(actions), args.source_dataset, source_assets)
            if args.write_keyframes:
                Path(args.write_keyframes).parent.mkdir(parents=True, exist_ok=True)
                with open(args.write_keyframes, "w", encoding="utf-8") as stream:
                    json.dump(extracted, stream, indent=2, sort_keys=True)
        validation_started = time.monotonic()
        trace_demo_at_validation_start_s = timers.seconds["trace_demo"]
        video = _probe(args.video) if args.render else None
        diagnostic_replay = None
        if args.render_diagnostic_only:
            reference_path = Path(args.diagnostic_reference_trace).resolve()
            with np.load(reference_path, allow_pickle=False) as reference:
                if "actions" not in reference.files:
                    raise ValueError("diagnostic reference trace has no actions")
                reference_actions = np.asarray(reference["actions"], dtype=np.float32)
            replay_actions = np.asarray(actions, dtype=np.float32)
            exact_actions = bool(
                reference_actions.shape == replay_actions.shape
                and np.array_equal(reference_actions, replay_actions)
            )
            max_abs_action_difference = (
                0.0
                if exact_actions
                else (
                    float(np.max(np.abs(reference_actions - replay_actions)))
                    if reference_actions.shape == replay_actions.shape
                    else None
                )
            )
            diagnostic_replay = {
                "classification": "deterministic_render_diagnostic_only",
                "physical_request_id": args.diagnostic_physical_request_id,
                "training_eligible": False,
                "causal_mechanism_attempt_consumed": False,
                "physics_or_controller_changes": False,
                "reference_trace": {
                    "path": str(reference_path),
                    "sha256": _sha256(reference_path),
                    "actions_shape": list(reference_actions.shape),
                    "actions_bytes_sha256": hashlib.sha256(
                        reference_actions.tobytes()
                    ).hexdigest(),
                },
                "replay_actions": {
                    "shape": list(replay_actions.shape),
                    "bytes_sha256": hashlib.sha256(
                        replay_actions.tobytes()
                    ).hexdigest(),
                    "exactly_equal": exact_actions,
                    "maximum_absolute_difference": max_abs_action_difference,
                },
                "overlay": {
                    "pot_body_frame": True,
                    "cooktop_target_frame": True,
                    "left_handle_contact_frame": True,
                    "right_handle_contact_frame": True,
                    "left_gripper_wrist_frames": True,
                    "right_gripper_wrist_frames": True,
                    "left_pad_centers_axes": True,
                    "right_pad_centers_axes": True,
                    "left_jaw_closing_line": True,
                    "right_jaw_closing_line": True,
                    "signed_residual_vectors": True,
                    "signed_control_vectors": True,
                    "screen_space_color_legend": True,
                },
            }
        desired_error = []
        if trajectory is not None:
            desired_error = [max(np.linalg.norm(np.asarray(left_eef[i])[:3] - np.asarray(desired_left[i])[:3]), np.linalg.norm(np.asarray(right_eef[i])[:3] - np.asarray(desired_right[i])[:3])) for i in range(len(left_eef))]
        waypoint_errors = [] if controller_client is not None else [desired_error[index] for index in trajectory.waypoint_steps.values() if trajectory is not None and index < len(desired_error)] if trajectory is not None else []
        executed_transport_metrics = None
        if (
            trajectory is not None
            and transport_plan is not None
            and not args.acquisition_only
        ):
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
            CONTACT_HOLD_SUPPORT_ALIGNMENT_FRACTION,
            HANDLE_PAD_DEPTH_MARGIN_M,
            LOADED_JAW_REACH_AVOIDANCE_FRACTION,
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
                not args.acquisition_only
                and (
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
                )
            ),
            "transport_no_internal_stops": bool(
                not args.acquisition_only
                and (
                    trajectory is None
                    or transport_plan["internal_stop_count"] == 0
                )
            ),
            "bimanual_transport_completed": bool(
                not args.acquisition_only
                and (
                    trajectory is None
                or (
                    samples[int(transport_plan["end_step"]) + 1]["left_grasp"]
                    and samples[int(transport_plan["end_step"]) + 1]["right_grasp"]
                )
                )
            ),
            "coded_task_success": bool(final["task_success"]),
            "centered_on_cooktop": centered_on_cooktop,
            "accepted_task_success": bool(final["task_success"] and centered_on_cooktop),
            "all_stages_latched": bool(final["stage1"] and final["stage2"]),
            "bimanual_pick_observed": any(row["left_grasp"] and row["right_grasp"] for row in samples),
            "robust_bilateral_latch": bool(
                acquisition_latch is not None
                and acquisition_latch["passes_robust_latch"]
            ),
            "pot_released": not final["left_grasp"] and not final["right_grasp"],
            "stable_support_window": bool(final["on_top_predicate_now"]),
            "terminal_pot_speed_within_threshold": bool(
                float(np.linalg.norm(final["pot_velocity"][:3])) <= 0.05
            ),
            "h264_nonempty": bool(
                video is not None
                and video["codec"] == "h264"
                and video["size_bytes"] > 0
                and video["frame_count"] == len(frame_stats)
            ),
            "fully_decodable": bool(
                video is not None and video["full_decode_returncode"] == 0
            ),
            "diagnostic_actions_identical": bool(
                diagnostic_replay is None
                or diagnostic_replay["replay_actions"]["exactly_equal"]
            ),
        }
        if args.target_handle_local_mpc_acquisition:
            checks["handle_local_mpc_bounded"] = bool(
                local_mpc_frame_receipts
                and all(
                    all(
                        receipt["receipt"]["hard_constraints"][name]
                        for name in (
                            "translation_step_within_bound",
                            "rotation_step_within_bound",
                            "jaw_step_within_bound",
                        )
                    )
                    for receipt in local_mpc_frame_receipts
                )
            )
            checks["handle_local_mpc_robust_latch"] = bool(
                local_mpc_latch_ready
                and acquisition_latch is not None
                and acquisition_latch["passes_robust_latch"]
            )
            checks["handle_local_mpc_fail_close_contract"] = bool(
                all(
                    (
                        all(
                            receipt["receipt"]["hard_constraints"][name]
                            for name in (
                                "pre_peer_pot_motion_within_limit",
                                "active_contact_pad_margin_valid",
                                "peer_contact_pad_margin_valid",
                            )
                        )
                        or receipt["receipt"]["fail_closed"]
                    )
                    for receipt in local_mpc_frame_receipts
                )
                and (
                    checks["handle_local_mpc_robust_latch"]
                    or local_mpc_fail_closed
                )
            )
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
            if args.render_diagnostic_only:
                acceptance_checks["diagnostic_actions_identical"] = checks[
                    "diagnostic_actions_identical"
                ]
        else:
            acceptance_checks = checks
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

            demo_started = time.monotonic()
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
                    "controller_plugin_sha256": (
                        None
                        if controller_receipt is None
                        else controller_receipt["sha256"]
                    ),
                    "source_dataset_sha256": _sha256(args.source_dataset),
                    "target_dataset_sha256": _sha256(args.target_dataset),
                },
            )
            demo_artifact = {"path": os.path.abspath(args.demo_hdf5), "sha256": _sha256(args.demo_hdf5)}
            timers.add("trace_demo", time.monotonic() - demo_started)
        from run_putmarker_skill_program import _asset_provenance
        local_mpc_protocol = None
        if args.target_handle_local_mpc_acquisition:
            from judo_isaaclab.putpot_local_mpc import (
                handle_local_mpc_config_receipt,
            )

            local_mpc_protocol = {
                "classification": (
                    "deterministic_handle_local_receding_horizon_mpc_lite"
                ),
                "global_trajectory_optimizer": False,
                "candidate_sampling": False,
                "random_rollout_search": False,
                "active_wrist_sequence": (
                    ["right", "left"]
                    if args.target_right_handle_local_mpc_bootstrap
                    else ["left"]
                ),
                "peer_strategy": (
                    "right_robust_dual_pad_bootstrap_then_observed_wrist_hold"
                    if args.target_right_handle_local_mpc_bootstrap
                    else "right_first_observed_wrist_hold"
                ),
                "source_demo_authority": [
                    "coarse_stage_order",
                    "object_relative_pregrasp_grasp_prior",
                    "gripper_timing_warm_start",
                ],
                "source_demo_near_contact_action_authority": False,
                "source_joint_nominal_weight_in_contact_window": 0.0,
                "config": handle_local_mpc_config_receipt(local_mpc_config),
                "frame_receipts": local_mpc_frame_receipts,
                "contact_window_frames": len(local_mpc_frame_receipts),
                "right_bootstrap": {
                    "enabled": bool(
                        args.target_right_handle_local_mpc_bootstrap
                    ),
                    "start_step": local_mpc_right_bootstrap_start_step,
                    "contact_window_frames": local_mpc_right_contact_window_step,
                    "robust_streak": local_mpc_right_robust_streak,
                    "robust_latch_ready": local_mpc_right_latch_ready,
                    "source_contact_frame_measurement": (
                        local_mpc_right_frame_receipt
                    ),
                },
                "robust_streak": local_mpc_robust_streak,
                "robust_latch_ready": local_mpc_latch_ready,
                "fail_closed": local_mpc_fail_closed,
                "fail_reason": local_mpc_fail_reason,
            }
        result = {
            "status": "passed" if all(acceptance_checks.values()) else "failed",
            "mode": args.mode,
            "protocol": {
                "controller": (
                    "source_action_prefix_with_supported_center_repair"
                    if repair_trajectory is not None
                    else "direct_source_action_replay" if trajectory is None
                    else (
                        "reloadable_python_controller_subprocess"
                        if controller_receipt is not None
                        else (
                            "deterministic_handle_local_mpc_lite_with_cartesian_dls"
                            if args.target_handle_local_mpc_acquisition
                            else "semantic_support_frames_with_cartesian_dls"
                        )
                    )
                ),
                "candidate_sampling": False,
                "scene_resets": 1,
                "inter_stage_resets": 0,
                "teleports_after_reset": 0,
                "control_rate_hz": 30,
                "steps": len(actions),
                "seed": args.seed,
                "attempt_identity": (
                    None if attempt_identity is None else attempt_identity.receipt()
                ),
                "program_spec": (
                    None if program_spec is None else program_spec.receipt()
                ),
                "controller_plugin": controller_receipt,
                "controller_plugin_command_count": controller_command_count,
                "persistent_runtime": {
                    "pid": os.getpid(),
                    "reused": runtime_reused,
                    "reset_index": reset_index,
                    "app_cameras_enabled": bool(args.render),
                    "observation_modalities": ["proprioception"]
                    + (["rgb"] if args.render else []),
                    **scene_sensor_inventory,
                    "worker_boundary": (
                        "same assets, device, camera capability, and code head"
                    ),
                },
                "grasp_assistance": "none",
                "acquisition_only": bool(args.acquisition_only),
                "render_diagnostic": diagnostic_replay,
                "acquisition_latch": acquisition_latch,
                "static_precontact_jaw_translation": (
                    static_precontact_jaw_translation
                ),
                "source_contact_frame_correction": (
                    source_contact_frame_correction
                ),
                "handle_local_mpc": local_mpc_protocol,
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
                "peer_contact_jaw_center_translation_m": (
                    peer_contact_jaw_center_translation_m
                ),
                "peer_contact_authored_jaw_center_locked": (
                    peer_contact_authored_jaw_center_locked
                ),
                "peer_contact_latch_centering_applied_m": (
                    peer_contact_latch_centering_applied_m
                ),
                "peer_contact_handle_center_rotation_rad": (
                    peer_contact_handle_center_rotation_rad
                ),
                "peer_contact_handle_center_post_pivot_residual_m": (
                    peer_contact_handle_center_post_pivot_residual_m
                ),
                "peer_contact_latch_centering_translation_m": (
                    peer_contact_latch_centering_translation_m
                ),
                "peer_contact_centered_tracking_latch_step": (
                    peer_contact_centered_tracking_latch_step
                ),
                "peer_contact_observed_jaw_residuals_m": (
                    peer_contact_observed_jaw_residuals_m
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
                "target_left_grasp_orientation_override_local_wxyz": (
                    target_left_grasp_orientation_override_local_wxyz
                ),
                "contact_hold_latch_step": contact_hold_latch_step,
                "acquisition_fail_closed_step": acquisition_fail_closed_step,
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
                "contact_hold_pick_lift_steps": contact_hold_pick_lift_steps,
                "contact_hold_pick_recovery_steps": (
                    contact_hold_pick_recovery_steps
                ),
                "contact_hold_pad_support_correction_m": (
                    contact_hold_pad_support_correction_m
                ),
                "contact_hold_pad_support_steps": contact_hold_pad_support_steps,
                "coded_pick_hold_retime_step": coded_pick_hold_retime_step,
                "coded_pick_hold_removed_steps": (
                    coded_pick_hold_removed_steps
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
                "transport_pad_support_correction_m": (
                    transport_pad_support_correction_m
                ),
                "transport_pad_support_steps": transport_pad_support_steps,
                "transport_reference_observation_init_steps": (
                    transport_reference_observation_init_steps
                ),
                "transport_reanchor_minimum_interval_steps": (
                    None
                    if transport_plan is None
                    else TRANSPORT_CONTACT_REANCHOR_MIN_STEPS
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
                "transport_loaded_vertical_rise_fraction": (
                    transport_loaded_vertical_rise_fraction
                ),
                "loaded_transport_contact_tracking": (
                    loaded_transport_contact_tracking
                ),
                "center_slide_reanchor_steps": center_slide_reanchor_steps,
                "center_slide_reanchor_signed_residuals_local_m": center_slide_reanchor_signed_residuals_local_m,
                "center_lowering_signed_residual_world_m": center_lowering_signed_residual_world_m,
                "release_signed_residual_world_m": release_signed_residual_world_m,
                "parameters": {"damping": args.damping, "max_joint_delta": args.max_joint_delta, "max_position_step": args.max_position_step, "max_rotation_step": args.max_rotation_step, "support_clearance_m": args.support_clearance_m, "transport_clearance_m": args.transport_clearance_m, "collision_clearance_m": args.collision_clearance_m, "executed_collision_minimum_m": 0.0, "handle_pad_depth_margin_m": HANDLE_PAD_DEPTH_MARGIN_M, "loaded_jaw_reach_avoidance_fraction": LOADED_JAW_REACH_AVOIDANCE_FRACTION, "missing_finger_contact_limit_m": args.missing_finger_contact_limit_m, "receiving_jaw_center_translation_fraction": args.receiving_jaw_center_translation_fraction, "receiving_jaw_reorientation_fraction": args.receiving_jaw_reorientation_fraction, "receiving_jaw_close_horizon_steps": args.receiving_jaw_close_horizon_steps, "geometry_conditioned_handle_grasp": handle_grasp_geometry, "missing_finger_contact_feedback": {"step_m": MISSING_FINGER_CONTACT_STEP_M, "limit_m": args.missing_finger_contact_limit_m, "minimum_observed_jaw_axis_m": MISSING_FINGER_JAW_AXIS_MIN_M, "milestone_jaw_center_residual_m": milestone_jaw_center_residual_m, "delay_steps": MISSING_FINGER_CONTACT_DELAY_STEPS, "final_signed_corrections_m": missing_finger_corrections, "pad_depth_step_m": MISSING_FINGER_PAD_DEPTH_STEP_M, "pad_depth_limit_m": MISSING_FINGER_PAD_DEPTH_LIMIT_M, "pad_target_fraction": MISSING_FINGER_PAD_TARGET_FRACTION, "final_pad_depth_corrections_m": missing_finger_depth_corrections}, "target_direct_generation": trajectory is not None, "source_semantic_success_required": False, "object_to_gripper_contact_frame_transfer": trajectory is not None, "requested_transport_steps": args.transport_steps, "transport_steps": (int(transport_plan["end_step"] - transport_plan["start_step"] + 1) if transport_plan is not None else args.transport_steps), "lower_steps": args.lower_steps, "release_steps": args.release_steps, "withdraw_steps": args.withdraw_steps, "settle_steps": args.settle_steps, "center_repair_steps": args.center_repair_steps, "integrated_target_ik": integrate_target_ik or repair_trajectory is not None, "smooth_collision_aware_transport": trajectory is not None, "bimanual_target_transport_required": bool(trajectory is not None), "supported_center_slide": bool(trajectory is not None and "center_slide" in trajectory.waypoint_steps) or repair_trajectory is not None, "source_action_prefix_steps": repair_prefix_steps, "center_feedback_reanchor": trajectory is not None, "center_feedback_release_correction": trajectory is not None, "center_tolerance_m": CENTERED_ON_COOKTOP_TOLERANCE_M},
            },
            "provenance": {
                "program_spec": (
                    None if program_spec is None else program_spec.receipt()
                ),
                "controller_plugin": controller_receipt,
                "source_dataset": {"path": os.path.abspath(args.source_dataset), "sha256": _sha256(args.source_dataset)},
                "target_dataset": {"path": os.path.abspath(args.target_dataset), "sha256": _sha256(args.target_dataset)},
                "source_assets": {name: _asset_provenance(path) for name, path in source_assets.items()},
                "target_assets": {name: _asset_provenance(path) for name, path in target_assets.items()},
                "task_manager": {"path": os.path.join(args.gear_repo, "dc_study/envs/tasks/put_pot_on_cooktop_manager.py"), "sha256": _sha256(os.path.join(args.gear_repo, "dc_study/envs/tasks/put_pot_on_cooktop_manager.py"))},
                "task_config": {"path": os.path.join(args.gear_repo, "dc_study/envs/tasks/put_pot_on_cooktop_manager_cfg.py"), "sha256": _sha256(os.path.join(args.gear_repo, "dc_study/envs/tasks/put_pot_on_cooktop_manager_cfg.py"))},
                "trace": {"path": os.path.abspath(args.trace_npz), "sha256": _sha256(args.trace_npz)},
                "demonstration": demo_artifact,
                "source_keyframes": ({"path": os.path.abspath(args.source_keyframes), "sha256": _sha256(args.source_keyframes)} if args.source_keyframes else None),
                "source_demo_card": source_demo_card_receipt,
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
        timers.add(
            "validation_decode_hash",
            max(
                0.0,
                time.monotonic()
                - validation_started
                - (
                    timers.seconds["trace_demo"]
                    - trace_demo_at_validation_start_s
                ),
            ),
        )
        result["protocol"]["phase_timings_s"] = timers.receipt()
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print("PUTPOT_FINAL=" + json.dumps(result, sort_keys=True), flush=True)
        if result["status"] != "passed":
            raise RuntimeError(f"acceptance checks failed: {acceptance_checks}")
    except BaseException:
        if (
            "actions" in locals()
            and actions
            and not Path(args.trace_npz).exists()
            and "samples" in locals()
            and len(samples) > 1
        ):
            try:
                _write_rollout_trace(
                    args.trace_npz,
                    actions=actions,
                    pot_poses=pot_poses,
                    left_eef=left_eef,
                    right_eef=right_eef,
                    desired_left=desired_left,
                    desired_right=desired_right,
                    samples=samples,
                    joint_nominal=joint_nominal,
                    local_mpc_frame_receipts=local_mpc_frame_receipts,
                    partial=True,
                )
                print(
                    "PUTPOT_PARTIAL_TRACE="
                    + json.dumps(
                        {
                            "path": str(Path(args.trace_npz).resolve()),
                            "actions": len(actions),
                            "sha256": _sha256(args.trace_npz),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except BaseException:
                traceback.print_exc()
        traceback.print_exc()
        raise
    finally:
        runtime_receipt_via_monitor = False
        if controller_client is not None:
            if controller_receipt is None:
                controller_receipt = controller_client.receipt()
            controller_client.close(force=sys.exc_info()[0] is not None)
        if encoder is not None:
            render_started = time.monotonic()
            encoder.close()
            timers.add("render_encode", time.monotonic() - render_started)
        if args.persistent_session and _PERSISTENT_RUNTIME is not None:
            _PERSISTENT_RUNTIME["attempts"] = reset_index
        else:
            shutdown_started = time.monotonic()
            if env is not None:
                env.close()
            if args.runtime_receipt_json:
                provisional_receipt = {
                    "pid": os.getpid(),
                    "persistent": False,
                    "runtime_reused": runtime_reused,
                    "reset_index": reset_index,
                    "attempt_identity": (
                        None
                        if attempt_identity is None
                        else attempt_identity.receipt()
                    ),
                    "phase_timings_s": timers.receipt(),
                    "attempt_wall_started_monotonic": (
                        attempt_wall_started_monotonic
                    ),
                    "scene_sensor_inventory": scene_sensor_inventory,
                    "program_spec": (
                        None if program_spec is None else program_spec.receipt()
                    ),
                    "controller_plugin": controller_receipt,
                }
                subprocess.Popen(
                    [
                        sys.executable,
                        str(REPO_ROOT / "src/judo_isaaclab/shutdown_monitor.py"),
                        "--pid",
                        str(os.getpid()),
                        "--started-monotonic",
                        repr(shutdown_started),
                        "--receipt-json",
                        args.runtime_receipt_json,
                        "--payload-json",
                        json.dumps(provisional_receipt, sort_keys=True),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                runtime_receipt_via_monitor = True
            simulation_app.close()
            timers.add("shutdown", time.monotonic() - shutdown_started)
        phase_timings = timers.receipt()
        _LAST_ATTEMPT_RUNTIME_RECEIPT = {
            "pid": os.getpid(),
            "persistent": bool(args.persistent_session),
            "runtime_reused": runtime_reused,
            "reset_index": reset_index,
            "attempt_identity": (
                None if attempt_identity is None else attempt_identity.receipt()
            ),
            "phase_timings_s": phase_timings,
            "timing_accounting": timing_accounting(
                time.monotonic() - attempt_wall_started_monotonic,
                phase_timings,
            ),
            "scene_sensor_inventory": scene_sensor_inventory,
            "program_spec": (
                None if program_spec is None else program_spec.receipt()
            ),
            "controller_plugin": controller_receipt,
            "controller_plugin_command_count": controller_command_count,
        }
        if args.runtime_receipt_json and not runtime_receipt_via_monitor:
            receipt_path = Path(args.runtime_receipt_json)
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_tmp = receipt_path.with_name(receipt_path.name + ".tmp")
            with open(receipt_tmp, "x", encoding="utf-8") as stream:
                json.dump(
                    _LAST_ATTEMPT_RUNTIME_RECEIPT,
                    stream,
                    indent=2,
                    sort_keys=True,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(receipt_tmp, receipt_path)


if __name__ == "__main__":
    main()
