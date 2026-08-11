"""Run deterministic HangMug replay or semantic skill evidence in IsaacLab."""

from __future__ import annotations

import argparse
import copy
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

from judo_isaaclab.dataset_aliases import (
    canonicalize_named_mapping,
    canonicalize_rigid_object_state,
    parse_object_aliases,
)
from judo_isaaclab.hang_mug_cli import (
    add_replay_repair_arguments,
    validate_replay_repair_arguments,
)
from judo_isaaclab.semantic_execution import (
    SemanticExecutionHooks,
    SemanticProtocolRecorder,
)


def _parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gear-repo", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--objects-root", required=True)
    parser.add_argument(
        "--mode", choices=("replay", "replay_hang", "skill"), required=True
    )
    parser.add_argument("--source-keyframes")
    parser.add_argument("--write-keyframes")
    parser.add_argument("--expect-failure", action="store_true")
    parser.add_argument(
        "--classification-run",
        action="store_true",
        help="Accept a technically valid replay whether task success passes or fails.",
    )
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument(
        "--dataset-object-alias",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="Rename legacy HDF5 object labels at load time without modifying the file.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--require-cpu-physics",
        action="store_true",
        help="Fail before rollout unless both requested and actual physics devices are CPU.",
    )
    parser.add_argument(
        "--grasp-assist-mechanism",
        choices=("task_config", "friction", "fixed_joint"),
        default="task_config",
        help="Use the configured datagen assist or select another mechanism exposed by it.",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--damping", type=float, default=0.045)
    parser.add_argument("--max-joint-delta", type=float, default=0.16)
    parser.add_argument("--max-position-step", type=float, default=0.025)
    parser.add_argument("--max-rotation-step", type=float, default=0.16)
    parser.add_argument("--insert-clearance-m", type=float, default=0.08)
    parser.add_argument(
        "--require-clean-insertion",
        action="store_true",
        help="Require zero exact cup-body/tree collisions before release.",
    )
    parser.add_argument("--clean-insertion-tracking-margin-m", type=float,
                        default=0.0, help="Extra radial tracking clearance.")
    parser.add_argument(
        "--handover-contact-settle-steps",
        type=int,
        default=0,
        help="Approach the observed-state receiving contact while open, then close in place.",
    )
    parser.add_argument(
        "--replay-handover-latch-grace-steps",
        type=int,
        default=0,
        help=(
            "Replay this many additional source actions after the source handover "
            "keyframe before starting a replay_hang suffix."
        ),
    )
    add_replay_repair_arguments(parser)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video")
    parser.add_argument("--trace-npz", required=True)
    parser.add_argument("--demo-hdf5")
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--direct-replay-result")
    args = parser.parse_args()
    validate_replay_repair_arguments(parser, args)
    return args


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _physics_device_receipt(
    requested: object,
    actual: object | None = None,
    *,
    require_cpu: bool,
) -> dict[str, object]:
    """Return a fail-closed physics-device receipt for CPU-only campaigns."""

    requested_name = str(requested)
    actual_name = None if actual is None else str(actual)
    if require_cpu and requested_name != "cpu":
        raise RuntimeError(
            f"CPU physics is required, but the requested device is {requested_name!r}"
        )
    if require_cpu and actual_name is not None and actual_name != "cpu":
        raise RuntimeError(
            f"CPU physics is required, but the actual device is {actual_name!r}"
        )
    return {
        "required": "cpu" if require_cpu else None,
        "requested": requested_name,
        "actual": actual_name,
        "passed": not require_cpu
        or (requested_name == "cpu" and actual_name in (None, "cpu")),
    }


def _dataset_assets(
    path: str, objects_root: str, aliases: dict[str, str] | None = None
) -> dict[str, str]:
    import h5py

    with h5py.File(path, "r") as handle:
        raw = handle["data"].attrs["ASSETS_INSTANCE_PATHS"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    relative = canonicalize_named_mapping(
        json.loads(str(raw)), aliases or {}, expected_names={"mug", "mug_tree"}
    )
    result = {name: os.path.join(objects_root, value) for name, value in relative.items()}
    missing = [value for value in result.values() if not os.path.isdir(value)]
    if missing:
        raise FileNotFoundError(f"official asset directories missing: {missing}")
    return result


def _load_dataset(
    path: str,
    episode: str,
    device,
    aliases: dict[str, str] | None = None,
) -> dict[str, object]:
    import h5py
    import torch
    from run_putmarker_skill_program import _tensor_tree

    with h5py.File(path, "r") as handle:
        group = handle[f"data/{episode}"]
        object_aliases = aliases or {}
        rigid_objects = group["states/rigid_object"]
        canonical_rigid = canonicalize_named_mapping(
            {name: name for name in rigid_objects},
            object_aliases,
            expected_names={"mug", "mug_tree"},
        )
        source_by_canonical = canonical_rigid
        initial_state = canonicalize_rigid_object_state(
            _tensor_tree(group["states"], 0, device), object_aliases
        )
        return {
            "initial_state": initial_state,
            "actions": torch.as_tensor(np.asarray(group["actions"]), device=device),
            "mug_pose": np.asarray(
                group[f"states/rigid_object/{source_by_canonical['mug']}/root_pose"]
            ),
            "tree_pose": np.asarray(
                group[f"states/rigid_object/{source_by_canonical['mug_tree']}/root_pose"]
            ),
            "num_samples": int(group.attrs["num_samples"]),
        }


def _load_assets(args, aliases):
    return (
        _dataset_assets(args.source_dataset, args.objects_root, aliases),
        _dataset_assets(args.target_dataset, args.objects_root, aliases),
    )


def _load_datasets(args, device, aliases):
    return (
        _load_dataset(args.source_dataset, args.episode, device, aliases),
        _load_dataset(args.target_dataset, args.episode, device, aliases),
    )


def _geometry(asset_path: str, root_pose: np.ndarray):
    from judo_isaaclab.hang_mug import RigidAssetGeometry
    from run_putmarker_skill_program import _asset_size

    return RigidAssetGeometry(root_pose=np.asarray(root_pose), size=_asset_size(asset_path))


def _select_grasp_assist_config(config, mechanism: str):
    selected = copy.deepcopy(config)
    if mechanism != "task_config":
        for spec in selected.values():
            spec["mechanism"] = mechanism
    return selected


def _add_right_handover_assist(config):
    """Mirror the datagen-supported mug assist onto the receiving hand."""

    selected = copy.deepcopy(config)
    if "left" not in selected:
        raise RuntimeError("HangMug right assist requires the canonical left assist")
    right = copy.deepcopy(selected["left"])
    right["arm"] = "right_arm"
    right["mechanism"] = "fixed_joint"
    right["grasp_delay_s"] = 0.0
    selected["right"] = right
    return selected


def _install_grasp_assist_config(manager_module, config_module, config) -> None:
    manager_module.GRASP_ASSIST_CONFIG = copy.deepcopy(config)
    config_module.GRASP_ASSIST_CONFIG = copy.deepcopy(config)


def _configure_task_for_evidence(mechanism: str = "task_config") -> dict[str, object]:
    import isaaclab.sim as sim_utils
    import dc_study.envs.tasks.hang_mug_on_tree_manager as manager_module
    import dc_study.envs.tasks.hang_mug_on_tree_manager_cfg as config_module

    assist_config = _select_grasp_assist_config(
        config_module.GRASP_ASSIST_CONFIG, mechanism
    )
    assist_config = _add_right_handover_assist(assist_config)
    if not assist_config:
        raise RuntimeError("HangMug datagen grasp-assist config is empty")
    if manager_module.GRASP_ASSIST_CONFIG != config_module.GRASP_ASSIST_CONFIG:
        raise RuntimeError("HangMug manager/config grasp-assist maps disagree")
    _install_grasp_assist_config(manager_module, config_module, assist_config)
    original_init = config_module.HangMugOnTreeManagerEnvCfg.__init__

    def offline_init(instance, *init_args, **init_kwargs):
        original_init(instance, *init_args, **init_kwargs)
        instance.grasp_assist = copy.deepcopy(assist_config)
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
        "grasp_assistance": "datagen-supported grasp assist selected",
        "grasp_assistance_selection": mechanism,
        "grasp_assistance_config": assist_config,
        "success_auto_termination": "disabled; coded predicate unchanged",
        "failure_auto_termination": "disabled for one-reset failure evidence",
        "ground": "procedural static cuboid",
    }


def _validate_datagen_grasp_assists(env, expected_config) -> str:
    expected_config = dict(expected_config or {})
    expected_names = {str(name) for name, spec in expected_config.items() if spec}
    actual_names = set(env.grasp_assists)
    if actual_names != expected_names:
        raise RuntimeError(
            f"grasp-assist names differ: expected {sorted(expected_names)}, "
            f"got {sorted(actual_names)}"
        )
    expected_classes = {
        "friction": "FrictionGraspAssist",
        "fixed_joint": "FixedJointGraspAssist",
        "none": "NullGraspAssist",
    }
    entries = []
    for name in sorted(expected_names):
        spec = expected_config[name]
        mechanism = str(spec.get("mechanism", "friction"))
        expected_class = expected_classes.get(mechanism)
        actual_class = type(env.grasp_assists[name]).__name__
        if expected_class is None or actual_class != expected_class:
            raise RuntimeError(
                f"grasp assist {name!r}: expected {mechanism!r}/{expected_class}, "
                f"got {actual_class}"
            )
        entries.append(f"{name}={mechanism}")
    return "task_config:" + ",".join(entries)


def _update_authored_assist_releases(env, trajectory, step: int) -> None:
    """Release grasp assists at the coded handover and unload boundaries.

    The task manager normally drops the left friction assist while both hands
    overlap during handover.  Some valid geometries transition directly from
    left to right contact without a simultaneous-grasp controller sample, so
    that event alone is not a reliable release signal.  The semantic program's
    left-release boundary is deterministic and already commands the left hand
    open; use it as a fail-closed release signal without advancing the left
    assist state machine twice during the grasp phase.
    """
    import torch

    left_grasping, right_grasping = env.robot.is_grasping()
    left_assist = env.grasp_assists.get("left")
    left_release_step = trajectory.waypoint_steps.get("left_release")
    releasing_left = left_release_step is not None and step >= left_release_step
    if left_assist is not None and releasing_left:
        left_assist.update(
            engage=left_grasping,
            disable=torch.ones_like(left_grasping, dtype=torch.bool),
        )

    right_assist = env.grasp_assists.get("right")
    if right_assist is not None:
        releasing_right = step > trajectory.waypoint_steps["branch_unload"]
        right_assist.update(
            engage=right_grasping,
            disable=torch.full_like(right_grasping, releasing_right),
        )


def _schema_aware_success_acceptance(
    checks: dict[str, bool], *, coded_skill: bool
) -> dict[str, bool]:
    """Select only mechanism-relevant checks without weakening task success.

    Direct action replay does not drive the skill runner's receiving-hand
    fixed-joint state machine.  Its physical right grasp and handover remain
    mandatory through ``right_handover_observed``, while the skill-only assist
    engagement bit is inapplicable.
    """
    acceptance = dict(checks)
    if not coded_skill:
        acceptance.pop("right_grasp_assist_engaged", None)
    return acceptance


def _requires_observed_handover_reanchor(mug_parts) -> bool:
    """Use live handover feedback for mugs taller than both lateral spans."""
    size = np.asarray(mug_parts.body_size, dtype=np.float64)
    if size.shape != (3,) or np.any(size <= 0.0):
        raise ValueError("mug body size must contain three positive values")
    return bool(size[2] > max(size[0], size[1]))


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
    assist_engaged = {
        name: bool(assist.engaged[0].item())
        for name, assist in env.grasp_assists.items()
    }
    return {
        "step": int(step),
        "program_stage": stage,
        "left_grasp": bool(left_grasp[0].item()),
        "right_grasp": bool(right_grasp[0].item()),
        "grasp_assist_engaged": assist_engaged,
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
        "clean_insertion_path": {
            "start_sample_index": tree_approach,
            "end_sample_index": inserted,
            "mug_poses": [
                samples[index]["mug_pose"]
                for index in range(tree_approach, inserted + 1)
            ],
        },
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
    clean_path = value.get("clean_insertion_path", {}).get("mug_poses", [])
    if len(clean_path) < 2:
        raise ValueError("source keyframes lack a clean insertion path")
    if value.get("source_dataset_sha256") != _sha256(source_dataset):
        raise ValueError("source keyframes do not match source dataset")
    return value


def _build_skill(
    keyframes,
    source_geometry,
    target_geometry,
    source_tree,
    target_tree,
    source_parts,
    target_parts,
    source_branches,
    target_branches,
    left_start,
    right_start,
    args,
):
    from judo_isaaclab.hang_mug import (
        HangMugSkillProgram,
        RigidAssetGeometry,
        ensure_pick_latch_clearance,
        geometry_conditioned_hang_pose,
    )
    from judo_isaaclab.put_marker import (
        compose_pose,
        inverse_pose,
        quaternion_rotate,
        transfer_pose,
    )

    frames = keyframes["frames"]
    source_initial_body = compose_pose(
        source_geometry.root_pose, source_parts.body_frame
    )
    target_initial_body = compose_pose(
        target_geometry.root_pose, target_parts.body_frame
    )

    def transfer_mug_frame(name, arm):
        frame = frames[name]
        source_frame = compose_pose(frame["mug_pose"], source_parts.body_frame)
        return transfer_pose(
            frame[f"{arm}_eef_pose"],
            source_frame,
            target_initial_body,
            local_position_scale=target_parts.body_size / source_parts.body_size,
        )

    left_grasp = transfer_mug_frame("left_grasp", "left")
    left_contact = compose_pose(inverse_pose(target_geometry.root_pose), left_grasp)
    source_dual = frames["dual_grasp"]
    source_dual_body = compose_pose(
        source_dual["mug_pose"], source_parts.body_frame
    )
    target_handover_body = transfer_pose(
        source_dual_body,
        source_initial_body,
        target_initial_body,
        local_position_scale=target_parts.body_size / source_parts.body_size,
    )
    pick_latch_body = ensure_pick_latch_clearance(
        target_handover_body,
        target_initial_body,
        target_parts.body_size[2],
    )
    target_handover_mug = RigidAssetGeometry(
        compose_pose(target_handover_body, inverse_pose(target_parts.body_frame)),
        target_geometry.size,
    )
    pick_latch_mug_pose = compose_pose(
        pick_latch_body, inverse_pose(target_parts.body_frame)
    )
    right_grasp = transfer_pose(
        source_dual["right_eef_pose"],
        source_dual_body,
        target_handover_body,
        local_position_scale=target_parts.body_size / source_parts.body_size,
    )
    right_contact = compose_pose(
        inverse_pose(target_handover_mug.root_pose), right_grasp
    )

    final_mug_pose, source_branch, target_branch = geometry_conditioned_hang_pose(
        frames["stable_settle"]["mug_pose"],
        frames["stable_settle"]["tree_pose"],
        source_parts,
        target_parts,
        source_branches,
        target_tree.root_pose,
        target_branches,
    )
    target_branch_world = compose_pose(target_tree.root_pose, target_branch.frame)
    final_mug = RigidAssetGeometry(final_mug_pose, target_geometry.size)
    transport_mug_pose = target_handover_mug.root_pose.copy()
    transport_mug_pose[:2] = 0.5 * (
        target_handover_mug.root_pose[:2] + final_mug_pose[:2]
    )
    transport_mug_pose[2] = max(
        target_handover_mug.root_pose[2], final_mug_pose[2] + args.insert_clearance_m
    )
    approach_mug_pose = final_mug_pose.copy()
    branch_tangent_world = quaternion_rotate(
        target_branch_world[3:], [1.0, 0.0, 0.0]
    )
    approach_mug_pose[:3] += branch_tangent_world * args.insert_clearance_m
    approach_mug_pose[2] += 0.03

    def held(mug_pose, local):
        return compose_pose(mug_pose, local)

    left_lift = held(pick_latch_mug_pose, left_contact)
    left_handover = held(target_handover_mug.root_pose, left_contact)
    left_release = left_handover.copy()
    left_release[1] += 0.10
    right_transport = held(transport_mug_pose, right_contact)
    right_approach = held(approach_mug_pose, right_contact)
    right_insert = held(final_mug.root_pose, right_contact)
    source_insert = frames["inserted_held"]
    left_branch_observer = target_tree.transfer_pose_from(
        RigidAssetGeometry(source_insert["tree_pose"], source_tree.size),
        source_insert["left_eef_pose"],
    )

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
        left_handover,
        transfer_pose(
            frames["right_pregrasp"]["right_eef_pose"],
            source_dual_body,
            target_handover_body,
            local_position_scale=target_parts.body_size / source_parts.body_size,
        ),
        right_grasp,
        left_release,
        approach_steps=100,
        contact_settle_steps=args.handover_contact_settle_steps,
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
        left_observer=left_branch_observer,
    )
    program.release_and_support(
        right_insert,
        right_insert,
        unload_steps=40,
        release_steps=40,
        settle_steps=60,
    )
    return (
        program.build(),
        final_mug_pose,
        target_handover_mug.root_pose,
        right_contact,
        source_branch,
        target_branch,
    )


def _sparse_joint_nominal(source, trajectory, keyframes):
    actions = np.asarray(source["actions"].detach().cpu(), dtype=np.float64)
    indices = keyframes["semantic_indices"]
    mapping = {
        "left_pregrasp": indices["left_pregrasp"],
        "left_grasp": indices["left_grasp"],
        "left_lift": indices["left_lift"],
        "handover_pregrasp": indices["right_pregrasp"],
        "right_grasp_settle": indices["dual_grasp"],
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
            f"assist={sample['grasp_assist_engaged']}",
            f"tree xy={sample['mug_tree_xy_error_m']:.4f} m",
        ]
        for row, line in enumerate(lines):
            cv2.putText(image, line, (12, 28 + 25 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)
        panels.append(cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA))
    return np.concatenate(panels, axis=1)


def main() -> None:
    args = _parser()
    object_aliases = parse_object_aliases(args.dataset_object_alias)
    protocol_recorder = SemanticProtocolRecorder()
    execution_hooks = SemanticExecutionHooks()
    _physics_device_receipt(
        args.device,
        require_cpu=args.require_cpu_physics,
    )
    if args.render and not args.video:
        raise ValueError("--render requires --video")
    for path in (args.result_json, args.trace_npz, args.video, args.write_keyframes):
        if path and os.path.isfile(path):
            os.unlink(path)
    # Validate cheap dataset/asset provenance before the expensive app launch.
    source_assets, target_assets = _load_assets(args, object_aliases)
    sys.path.insert(0, os.path.abspath(args.gear_repo))
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher({"headless": True, "device": args.device, "enable_cameras": True}).app
    env = encoder = None
    try:
        import torch
        from dc_study.utils.task_creation import create_task_environment
        from run_putmarker_skill_program import _Encoder, _asset_provenance, _eef_pose, _ik_action, _probe, _reset_scene_to_state

        override = _configure_task_for_evidence(args.grasp_assist_mechanism)
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
        physics_device = _physics_device_receipt(
            args.device,
            env.device,
            require_cpu=args.require_cpu_physics,
        )
        print(
            "HANGMUG_PHYSICS_DEVICE="
            + json.dumps(physics_device, sort_keys=True),
            flush=True,
        )
        grasp_assistance = _validate_datagen_grasp_assists(
            env, override["grasp_assistance_config"]
        )
        env.reset(warm_up=False, seed=args.seed)
        protocol_recorder.record_environment_reset(
            reason="task_environment_reset"
        )
        source, target = _load_datasets(args, env.device, object_aliases)
        env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
        _reset_scene_to_state(env.scene, target["initial_state"], env_ids)
        protocol_recorder.record_state_restore(
            initial=True,
            reason="target_dataset_initial_state",
        )
        env.sim.forward()
        env.reset_success_check(env_ids)
        source_mug = _geometry(source_assets["mug"], source["mug_pose"][0])
        target_mug = _geometry(target_assets["mug"], target["mug_pose"][0])
        source_tree = _geometry(source_assets["mug_tree"], source["tree_pose"][0])
        target_tree = _geometry(target_assets["mug_tree"], target["tree_pose"][0])
        from semantic_asset_geometry import jsonable, mug_parts, tree_branches

        source_parts = mug_parts(source_assets["mug"])
        target_parts = mug_parts(target_assets["mug"])
        source_branches = tree_branches(source_assets["mug_tree"])
        target_branches = tree_branches(target_assets["mug_tree"])
        keyframes = (
            _load_keyframes(args.source_keyframes, args.source_dataset)
            if args.source_keyframes
            else None
        )
        trajectory, intended_final, nominal_handover_mug, nominal_right_contact, source_branch, target_branch = (
            _build_skill(
                keyframes,
                source_mug,
                target_mug,
                source_tree,
                target_tree,
                source_parts,
                target_parts,
                source_branches,
                target_branches,
                _eef_pose(env, "left_arm"),
                _eef_pose(env, "right_arm"),
                args,
            )
            if args.mode == "skill" else (None, None, None, None, None, None)
        )
        joint_nominal = _sparse_joint_nominal(source, trajectory, keyframes) if trajectory is not None else None
        observed_handover_reanchor = bool(
            trajectory is not None
            and _requires_observed_handover_reanchor(target_parts)
        )
        repair_prefix_steps = None
        if args.mode == "replay_hang":
            from judo_isaaclab.hang_mug_replay_tail import (
                replay_prefix_steps,
                replay_tail_steps,
            )

            repair_prefix_steps = replay_prefix_steps(
                keyframes,
                latch_grace_steps=args.replay_handover_latch_grace_steps,
            )
            total_steps = repair_prefix_steps + replay_tail_steps(
                keyframes,
                unload_steps=args.replay_hang_unload_steps,
                release_steps=args.replay_hang_release_steps,
            )
        else:
            total_steps = trajectory.steps if trajectory is not None else len(source["actions"])
        from judo_isaaclab.demo_artifact import DemonstrationRecorder

        demo_recorder = DemonstrationRecorder()
        demo_recorder.start(env.scene.get_state(is_relative=False))
        samples = [_sample(env, -1, "reset")]
        actions = []; mug_poses = []; left_eef = []; right_eef = []; desired_left = []; desired_right = []; desired_steps = []; frame_stats = []
        if args.render:
            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            encoder = _Encoder(args.fps, args.video)
        from hangmug_rollout import execute_hangmug_rollout

        rollout = execute_hangmug_rollout(
            env=env, source=source, keyframes=keyframes,
            source_parts=source_parts, target_parts=target_parts,
            source_branches=source_branches, target_tree=target_tree,
            target_branches=target_branches, target_assets=target_assets,
            trajectory=trajectory,
            joint_nominal=joint_nominal, intended_final=intended_final,
            nominal_handover_mug=nominal_handover_mug,
            nominal_right_contact=nominal_right_contact,
            source_branch=source_branch, target_branch=target_branch,
            observed_handover_reanchor=observed_handover_reanchor,
            repair_prefix_steps=repair_prefix_steps, total_steps=total_steps,
            args=args, demo_recorder=demo_recorder, encoder=encoder,
            samples=samples, protocol_recorder=protocol_recorder,
            execution_hooks=execution_hooks, ik_action=_ik_action,
            sample_environment=_sample,
            update_assist_releases=_update_authored_assist_releases,
            render_frame=_frame,
        )
        samples = rollout.samples; actions = rollout.actions
        mug_poses = rollout.mug_poses; left_eef = rollout.left_eef
        right_eef = rollout.right_eef; desired_left = rollout.desired_left
        desired_right = rollout.desired_right; desired_steps = rollout.desired_steps
        trajectory = rollout.trajectory; joint_nominal = rollout.joint_nominal
        intended_final = rollout.intended_final
        nominal_right_contact = rollout.nominal_right_contact
        source_branch = rollout.source_branch; target_branch = rollout.target_branch
        frame_stats = rollout.frame_stats
        planned_clean_insertion = rollout.planned_clean_insertion
        if encoder is not None:
            encoder.close(); encoder = None
        Path(args.trace_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.trace_npz,
            actions=np.asarray(actions, dtype=np.float32),
            mug_poses=np.asarray(mug_poses, dtype=np.float32),
            left_eef_poses=np.asarray(left_eef, dtype=np.float32),
            right_eef_poses=np.asarray(right_eef, dtype=np.float32),
            program_stages=np.asarray(
                [row["program_stage"] for row in samples], dtype="U32"
            ),
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
        clean_insertion = None
        if args.require_clean_insertion:
            audit_keyframes = keyframes or extracted
            if audit_keyframes is None:
                raise RuntimeError(
                    "clean insertion audit requires simulator-derived source keyframes"
                )
            if args.mode == "replay_hang":
                clean_start = int(repair_prefix_steps)
                clean_release = int(
                    repair_prefix_steps
                    + trajectory.waypoint_steps["branch_unload"]
                    + 1
                )
            else:
                clean_start = int(
                    audit_keyframes["frames"]["tree_approach"]["action_index"]
                )
                clean_release = int(
                    audit_keyframes["frames"]["release"]["action_index"]
                )
            from judo_isaaclab.hang_mug_clean_insertion import (
                exact_body_collision_receipt,
            )

            clean_insertion = exact_body_collision_receipt(
                mug_poses,
                tree_pose=final["tree_pose"],
                target_assets=target_assets,
                start_step=clean_start,
                release_step=clean_release,
            )
            print(
                "HANGMUG_CLEAN_INSERTION="
                + json.dumps(clean_insertion, sort_keys=True),
                flush=True,
            )
        video = _probe(args.video) if args.render else None
        desired_error = [
            max(
                np.linalg.norm(
                    np.asarray(left_eef[actual_step])[:3]
                    - np.asarray(desired_left[index])[:3]
                ),
                np.linalg.norm(
                    np.asarray(right_eef[actual_step])[:3]
                    - np.asarray(desired_right[index])[:3]
                ),
            )
            for index, actual_step in enumerate(desired_steps)
        ]
        direct_replay = None
        if args.direct_replay_result:
            with open(args.direct_replay_result, encoding="utf-8") as stream:
                direct_replay = json.load(stream)
        terminal_speed = float(np.linalg.norm(final["mug_velocity"][:3]))
        protocol_receipt = protocol_recorder.receipt()
        protocol_checks = protocol_receipt["checks"]
        contact_backed_grasps_only = bool(
            protocol_receipt["contact_channels"]
        ) and all(
            channel["source"] == "env.robot.is_grasping"
            for channel in protocol_receipt["contact_channels"].values()
        )
        checks = {
            "one_reset": protocol_checks["one_reset"],
            "zero_inter_stage_resets": protocol_checks[
                "zero_inter_stage_resets"
            ],
            "zero_post_start_state_writes": protocol_checks[
                "zero_post_start_state_writes"
            ],
            "no_truncation_observed": protocol_checks["no_truncation_observed"],
            "no_unexpected_termination": bool(
                not protocol_receipt["termination_events"] or final["task_success"]
            ),
            "real_target_assets": target_assets
            == _dataset_assets(args.target_dataset, args.objects_root, object_aliases),
            "contact_backed_grasps_only": contact_backed_grasps_only,
            "datagen_grasp_assist_configured": bool(env.grasp_assists),
            "left_grasp_assist_engaged": any(
                row["grasp_assist_engaged"].get("left", False) for row in samples
            ),
            "left_grasp_assist_released": not final[
                "grasp_assist_engaged"
            ].get("left", False),
            "right_grasp_assist_engaged": any(
                row["grasp_assist_engaged"].get("right", False) for row in samples
            ),
            "right_grasp_assist_released": not final[
                "grasp_assist_engaged"
            ].get("right", False),
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
        if args.require_clean_insertion:
            checks["clean_insertion_exact_mesh_audited"] = bool(
                clean_insertion and clean_insertion["method"]
                == "python-fcl exact mesh intersection"
            )
            checks["clean_insertion_body_collision_free"] = bool(
                clean_insertion and clean_insertion["passed"]
            )
        if args.require_cpu_physics:
            checks["physics_device_cpu"] = bool(
                physics_device["passed"] and physics_device["actual"] == "cpu"
            )
        if args.classification_run:
            if args.mode != "replay":
                raise ValueError("--classification-run is only valid in replay mode")
            acceptance = {
                name: checks[name]
                for name in (
                    "one_reset", "zero_inter_stage_resets", "real_target_assets",
                    "contact_backed_grasps_only", "datagen_grasp_assist_configured",
                    "h264_nonempty", "fully_decodable",
                )
            }
        elif args.expect_failure:
            acceptance = {name: checks[name] for name in ("one_reset", "zero_inter_stage_resets", "real_target_assets", "contact_backed_grasps_only", "datagen_grasp_assist_configured", "left_grasp_assist_engaged", "h264_nonempty", "fully_decodable")}
            acceptance["expected_coded_task_failure"] = not final["task_success"]
        else:
            acceptance = _schema_aware_success_acceptance(
                checks, coded_skill=args.mode in {"replay_hang", "skill"}
            )
            if args.require_clean_insertion:
                acceptance = dict(acceptance)
                acceptance["clean_insertion_exact_mesh_audited"] = checks[
                    "clean_insertion_exact_mesh_audited"
                ]
                acceptance["clean_insertion_body_collision_free"] = checks[
                    "clean_insertion_body_collision_free"
                ]
            if direct_replay is not None and _sha256(args.source_dataset) != _sha256(args.target_dataset):
                acceptance = dict(acceptance)
                direct_checks = direct_replay.get("checks", {})
                acceptance["direct_source_action_replay_failed"] = bool(
                    direct_replay.get("status") == "passed"
                    and (
                        not direct_replay.get("terminal", {}).get(
                            "task_success", True
                        )
                        or direct_checks.get(
                            "clean_insertion_body_collision_free"
                        ) is False
                    )
                )
                acceptance["direct_replay_grasp_assistance_matched"] = (
                    direct_replay.get("protocol", {}).get("grasp_assistance")
                    == grasp_assistance
                )
                acceptance["direct_replay_physics_device_matched"] = (
                    direct_replay.get("protocol", {}).get("physics_device_actual")
                    == str(env.device)
                )
        controller = (
            "source_handover_prefix_with_clean_relationship_insertion"
            if args.mode == "replay_hang"
            else "direct_source_action_replay"
            if args.mode == "replay"
            else "deterministic_semantic_skill"
        )
        demo_artifact = None
        if args.demo_hdf5 and final["task_success"] and all(acceptance.values()):
            from judo_isaaclab.demo_artifact import relative_asset_paths

            demo_recorder.write(
                args.demo_hdf5,
                assets_instance_paths=relative_asset_paths(target_assets, args.objects_root),
                success=True,
                metadata={
                    "task": "HangMugOnTree-v0",
                    "controller": controller,
                    "candidate_sampling": False,
                    "grasp_assistance": grasp_assistance,
                    "source_dataset_sha256": _sha256(args.source_dataset),
                    "target_dataset_sha256": _sha256(args.target_dataset),
                },
            )
            demo_artifact = {"path": os.path.abspath(args.demo_hdf5), "sha256": _sha256(args.demo_hdf5)}
        result = {
            "status": "passed" if all(acceptance.values()) else "failed",
            "mode": args.mode,
            "protocol": {"controller": controller, "candidate_sampling": False, "scene_resets": protocol_receipt["environment_resets"], "initial_state_restores": protocol_receipt["initial_state_restores"], "inter_stage_resets": protocol_receipt["inter_stage_resets"], "teleports_after_reset": protocol_receipt["teleports_after_rollout_start"], "execution_instrumentation": protocol_receipt, "semantic_execution_hooks": {"enabled": execution_hooks.enabled, "policy": "optional_observation_only"}, "dataset_object_aliases": object_aliases, "dataset_alias_policy": "load_time_keys_only_source_bytes_unchanged", "control_rate_hz": 30, "steps": len(actions), "seed": args.seed, "physics_device_requested": args.device, "physics_device_actual": str(env.device), "physics_device_requirement": "cpu" if args.require_cpu_physics else None, "physics_device_receipt": physics_device, "grasp_assistance": grasp_assistance, "planned_clean_insertion": planned_clean_insertion, "clean_insertion": clean_insertion, "parameters": {"damping": args.damping, "max_joint_delta": args.max_joint_delta, "max_position_step": args.max_position_step, "max_rotation_step": args.max_rotation_step, "insert_clearance_m": args.insert_clearance_m, "clean_insertion_tracking_margin_m": args.clean_insertion_tracking_margin_m, "handover_contact_settle_steps": args.handover_contact_settle_steps, "source_action_prefix_steps": repair_prefix_steps, "replay_target_branch_policy": args.replay_target_branch_policy, "observed_left_anchor_held_during_handover": observed_handover_reanchor, "observed_handover_reanchor": observed_handover_reanchor, "right_contact_feedback_reanchor": trajectory is not None, "pick_clearance_uses_measured_body_height": True, "mug_body_frame_scaling": True, "handle_hole_branch_frame_transfer": True, "branch_support_midpoint": True}},
            "provenance": {"source_dataset": {"path": os.path.abspath(args.source_dataset), "sha256": _sha256(args.source_dataset)}, "target_dataset": {"path": os.path.abspath(args.target_dataset), "sha256": _sha256(args.target_dataset)}, "dataset_object_aliases": object_aliases, "source_assets": {name: _asset_provenance(path) for name, path in source_assets.items()}, "target_assets": {name: _asset_provenance(path) for name, path in target_assets.items()}, "task_manager": {"path": os.path.join(args.gear_repo, "dc_study/envs/tasks/hang_mug_on_tree_manager.py"), "sha256": _sha256(os.path.join(args.gear_repo, "dc_study/envs/tasks/hang_mug_on_tree_manager.py"))}, "task_config": {"path": os.path.join(args.gear_repo, "dc_study/envs/tasks/hang_mug_on_tree_manager_cfg.py"), "sha256": _sha256(os.path.join(args.gear_repo, "dc_study/envs/tasks/hang_mug_on_tree_manager_cfg.py"))}, "trace": {"path": os.path.abspath(args.trace_npz), "sha256": _sha256(args.trace_npz)}, "demonstration": demo_artifact, "source_keyframes": ({"path": os.path.abspath(args.source_keyframes), "sha256": _sha256(args.source_keyframes)} if args.source_keyframes else None)},
            "semantic_frames": {
                "source_mug": source_mug.root_pose.tolist(),
                "target_mug": target_mug.root_pose.tolist(),
                "source_tree": source_tree.root_pose.tolist(),
                "target_tree": target_tree.root_pose.tolist(),
                "source_mug_parts": jsonable(source_parts),
                "target_mug_parts": jsonable(target_parts),
                "source_branch": jsonable(source_branch),
                "target_branch": jsonable(target_branch),
                "intended_final_mug_pose": (
                    intended_final.tolist() if intended_final is not None else None
                ),
                "extracted_keyframes": extracted,
            },
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
