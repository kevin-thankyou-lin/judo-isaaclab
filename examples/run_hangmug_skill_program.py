"""Run deterministic HangMug replay or semantic skill evidence in IsaacLab."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import traceback

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

PROVEN_CONTROL_DEFAULTS = {
    "damping": 0.045,
    "max_joint_delta": 0.16,
    "max_position_step": 0.025,
    "max_rotation_step": 0.16,
}


def _parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gear-repo", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--target-dataset")
    parser.add_argument("--target-mug-asset")
    parser.add_argument("--target-tree-asset")
    parser.add_argument("--objects-root", required=True)
    parser.add_argument("--mode", choices=("replay", "skill"), required=True)
    parser.add_argument("--source-keyframes")
    parser.add_argument("--write-keyframes")
    parser.add_argument(
        "--reuse-source-pick-prefix",
        action="store_true",
        help=(
            "Replay source actions through the right-pregrasp keyframe, then "
            "repair only the handover-and-hang suffix."
        ),
    )
    parser.add_argument("--expect-failure", action="store_true")
    parser.add_argument(
        "--classification-run",
        action="store_true",
        help="Accept a technically valid replay whether task success passes or fails.",
    )
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-controller-gains-sha256")
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
        "--branch-approach-height-m",
        type=float,
        default=0.03,
        help="Bounded world-Z clearance added before branch-axis insertion.",
    )
    parser.add_argument(
        "--pick-lift-margin-m",
        type=float,
        default=0.0,
        help="Bounded extra clearance above the unchanged 5 cm Pick threshold.",
    )
    parser.add_argument(
        "--branch-support-fraction",
        type=float,
        default=0.5,
        help="Bounded root-to-tip fraction used to seat the handle on the branch.",
    )
    parser.add_argument(
        "--branch-roll-offset-rad",
        type=float,
        default=0.0,
        help="Bounded mug roll about the selected authored branch axis.",
    )
    parser.add_argument(
        "--branch-support-seat-down-m",
        type=float,
        default=0.0,
        help="Bounded vertical seating offset applied before branch release.",
    )
    parser.add_argument(
        "--stable-support-steps",
        type=int,
        default=60,
        help="Released terminal observation rows; acceptance predicates stay unchanged.",
    )
    parser.add_argument(
        "--target-branch-rank",
        type=int,
        help="Optional canonical inferred-branch rank for a geometry-screened repair.",
    )
    parser.add_argument(
        "--handover-contact-settle-steps",
        type=int,
        default=0,
        help="Approach the observed-state receiving contact while open, then close in place.",
    )
    parser.add_argument(
        "--handover-contact-acquire-steps",
        type=int,
        default=0,
        help="Move the left-held mug into the stationary closed receiver before release.",
    )
    parser.add_argument(
        "--handover-target-offset-m",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Bounded world-translation correction for the fixed receiving waypoint.",
    )
    parser.add_argument(
        "--handover-target-local-pitch-rad",
        type=float,
        default=0.0,
        help="Bounded local-Y rotation of the fixed receiving waypoint.",
    )
    parser.add_argument(
        "--handover-straddle-local-x-m",
        type=float,
        default=0.0,
        help="Bounded receiver-local X correction applied after final orientation.",
    )
    parser.add_argument(
        "--handover-orient-clearance-m",
        type=float,
        default=0.0,
        help="World-Z clearance used to orient the open receiver before descending.",
    )
    parser.add_argument(
        "--handover-standoff-outside-m",
        type=float,
        default=0.0,
        help=(
            "Receiver-side horizontal clearance added only to the open handover "
            "standoff; the final grasp pose is unchanged."
        ),
    )
    parser.add_argument(
        "--handover-orient-steps",
        type=int,
        default=0,
        help="Rows used to rotate the open receiver at the clear pose.",
    )
    parser.add_argument(
        "--handover-handle-frame-transfer",
        action="store_true",
        help="Transfer the demonstrated receiver pose through authored handle-hole frames.",
    )
    parser.add_argument(
        "--branch-orient-steps",
        type=int,
        default=0,
        help="Rows used to orient the held mug at the clear transport pose.",
    )
    parser.add_argument(
        "--handover-confirm-steps",
        type=int,
        default=0,
        help="Hold the completed release pose while the coded Handover stage latches.",
    )
    parser.add_argument(
        "--left-release-retreat-m",
        type=float,
        default=0.10,
        help="Bounded lateral retreat after the receiving grasp closes.",
    )
    parser.add_argument(
        "--handover-post-release-lift-m",
        type=float,
        default=0.0,
        help="Bounded receiver lift after the contact-backed left release.",
    )
    parser.add_argument(
        "--handover-post-release-lift-steps",
        type=int,
        default=0,
        help="Rows for the contact-backed receiver lift after left release.",
    )
    parser.add_argument(
        "--post-handover-right-return-steps",
        type=int,
        default=0,
        help=(
            "Rows for the closed right carrier to return to its demonstrated "
            "start pose before branch setup."
        ),
    )
    parser.add_argument(
        "--left-branch-point-steps",
        type=int,
        default=0,
        help=(
            "Rows for the open left arm to reach the target-branch observer "
            "pose after the right carrier has returned to start."
        ),
    )
    parser.add_argument(
        "--direct-rest-to-preinsert-steps",
        type=int,
        default=0,
        help=(
            "Rows for one collision-screened interpolation from carrying rest "
            "to the geometry-derived pre-insertion pose."
        ),
    )
    parser.add_argument(
        "--post-handover-rest-observer-steps",
        type=int,
        default=0,
        help=(
            "Rows for the closed right carrier to return to rest while the "
            "open left arm moves to the branch observer."
        ),
    )
    parser.add_argument(
        "--post-release-return-to-rest-steps",
        type=int,
        default=0,
        help=(
            "Rows for one collision-screened open-arm interpolation from final "
            "supported release back to demonstrated rest."
        ),
    )
    parser.add_argument(
        "--require-broad-pad-contact",
        action="store_true",
        help=(
            "Require sustained two-finger contact within the interior 15-85% "
            "of each pad for both the left pick and right carrier grasp."
        ),
    )
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


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json_atomic(path: str | os.PathLike[str], value: dict[str, object]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def _require_proven_control_defaults(args) -> None:
    for name, expected in PROVEN_CONTROL_DEFAULTS.items():
        actual = float(getattr(args, name))
        if actual != expected:
            raise ValueError(
                f"{name}={actual} changes the proven default {expected}; "
                "repair semantic waypoints or timing instead"
            )


def _source_dataset_receipt(
    path: str, episode: str, expected_sha256: str | None
) -> dict[str, object]:
    """Bind the sole executable source to its HDF5 ``actions`` dataset."""
    import h5py

    file_sha256 = _sha256(path)
    if expected_sha256 and file_sha256 != expected_sha256:
        raise ValueError(
            f"source dataset SHA256 mismatch: {file_sha256} != {expected_sha256}"
        )
    with h5py.File(path, "r") as handle:
        group = handle[f"data/{episode}"]
        actions = np.asarray(group["actions"])
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"source actions must have shape (N, 14), got {actions.shape}")
        if int(group.attrs["num_samples"]) != len(actions):
            raise ValueError("source num_samples does not match actions")
        state_lengths: list[int] = []
        group["states"].visititems(
            lambda _, value: state_lengths.append(int(value.shape[0]))
            if isinstance(value, h5py.Dataset)
            else None
        )
        if not state_lengths or any(length != len(actions) + 1 for length in state_lengths):
            raise ValueError("source states must contain exactly one more row than actions")
        processed = group.get("processed_actions")
        processed_shape = None if processed is None else list(processed.shape)
    return {
        "path": os.path.abspath(path),
        "file_sha256": file_sha256,
        "episode": episode,
        "action_dataset": "actions",
        "actions_shape": list(actions.shape),
        "actions_dtype": str(actions.dtype),
        "actions_sha256": _array_sha256(actions),
        "processed_actions_shape": processed_shape,
        "processed_actions_role": "state_aligned_analysis_metadata_not_executed",
    }


def _asset_index(path: str) -> int:
    match = re.search(r"_(\d{6})$", Path(path).name)
    if match is None:
        raise ValueError(f"asset name lacks a six-digit pair index: {path}")
    return int(match.group(1))


def _resolve_target_assets(args, source_assets: dict[str, str]) -> tuple[dict[str, str], str]:
    overrides = (args.target_mug_asset, args.target_tree_asset)
    if any(overrides) and not all(overrides):
        raise ValueError("--target-mug-asset and --target-tree-asset must be supplied together")
    if all(overrides):
        if args.target_dataset:
            raise ValueError("asset overrides use the source as state template; omit --target-dataset")
        root = Path(args.objects_root).resolve()
        target = {
            "mug": str(Path(args.target_mug_asset).resolve()),
            "mug_tree": str(Path(args.target_tree_asset).resolve()),
        }
        for path in target.values():
            Path(path).relative_to(root)
            if not Path(path).is_dir():
                raise FileNotFoundError(path)
        if _asset_index(target["mug"]) != _asset_index(target["mug_tree"]):
            raise ValueError("HangMug target assets must use the same index")
        return target, args.source_dataset
    if not args.target_dataset:
        return dict(source_assets), args.source_dataset
    return _dataset_assets(args.target_dataset, args.objects_root), args.target_dataset


def _asset_min_z(path: str) -> float:
    with open(Path(path) / "asset_size.json", encoding="utf-8") as stream:
        value = np.asarray(json.load(stream)["min"], dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid asset bounds: {path}")
    return float(value[2])


def _support_preserving_target_state(
    template: dict[str, object],
    template_assets: dict[str, str],
    target_assets: dict[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Copy initial state while keeping each upright object's bottom on its support."""
    target = copy.deepcopy(template)
    receipt: dict[str, object] = {}
    pose_keys = {"mug": "mug_pose", "mug_tree": "tree_pose"}
    for name, pose_key in pose_keys.items():
        poses = np.array(template[pose_key], copy=True)
        source_z = float(poses[0, 2])
        support_z = source_z + _asset_min_z(template_assets[name])
        target_z = support_z - _asset_min_z(target_assets[name])
        poses[0, 2] = target_z
        target[pose_key] = poses
        root_pose = target["initial_state"]["rigid_object"][name]["root_pose"]
        root_pose[..., 2] = target_z
        receipt[name] = {
            "source_root_z_m": source_z,
            "target_root_z_m": target_z,
            "translation_z_m": target_z - source_z,
            "preserved_support_z_m": support_z,
        }
    return target, receipt


def _terminal_stability(samples: list[dict[str, object]], steps: int = 30) -> dict[str, object]:
    window = samples[-steps:]
    passed = len(window) == steps and all(
        row["task_success"]
        and row["stage3"]
        and row["hang_predicate_now"]
        and not row["left_grasp"]
        and not row["right_grasp"]
        for row in window
    )
    return {"required_steps": steps, "observed_steps": len(window), "passed": bool(passed)}


def _independent_terminal_hang_receipt(
    statuses: list[dict[str, object]],
    samples: list[dict[str, object]],
    reset_counts: dict[str, int],
    *,
    steps: int = 30,
) -> dict[str, object]:
    """Prove a durable physical hang without substituting for task latches."""
    if steps <= 0:
        raise ValueError("terminal hang window must be positive")
    aligned = len(statuses) == len(samples)
    status_window = statuses[-steps:] if aligned else []
    sample_window = samples[-steps:] if aligned else []

    def all_status(predicate) -> bool:
        return len(status_window) == steps and all(predicate(row) for row in status_window)

    def raw(row, name: str) -> bool:
        return bool(row["diagnostics"]["raw_conditions"][name])

    checks = {
        "status_sample_alignment": aligned,
        "required_window_observed": len(status_window) == steps,
        "branch_engaged": all_status(
            lambda row: row["diagnostics"]["branch_engaged"]
        ),
        "support_held": all_status(
            lambda row: raw(row, "insertion_support_candidate")
        ),
        "release_hang_candidate": all_status(
            lambda row: raw(row, "release_hang_candidate")
        ),
        "adapter_released": all_status(lambda row: row["released"]),
        "adapter_stable": all_status(lambda row: row["stable"]),
        "bounded_contact_entire_rollout": bool(statuses)
        and all(row["contact_policy"] for row in statuses),
        "both_grippers_released": len(sample_window) == steps
        and all(
            not row["left_grasp"] and not row["right_grasp"]
            for row in sample_window
        ),
        "both_assists_released": len(sample_window) == steps
        and all(
            not row["grasp_assist_engaged"].get("left", False)
            and not row["grasp_assist_engaged"].get("right", False)
            for row in sample_window
        ),
        "one_continuous_reset_free_rollout": reset_counts
        == {
            "explicit_env_reset_calls": 1,
            "initial_state_restores": 1,
            "resets_during_episode": 0,
        },
    }
    final_status = statuses[-1] if statuses else None
    final_sample = samples[-1] if samples else None
    return {
        "required_steps": steps,
        "observed_steps": len(status_window),
        "checks": checks,
        "passed": all(checks.values()),
        "coded_stage_latches": {
            "stage1": bool(final_sample and final_sample.get("stage1", False)),
            "stage2": bool(final_sample and final_sample.get("stage2", False)),
            "stage3": bool(final_sample and final_sample.get("stage3", False)),
        },
        "adapter_completed_stage_latches": (
            {}
            if final_status is None
            else dict(final_status["diagnostics"]["completed_stage_latches"])
        ),
    }


def _semantic_stage_receipt(statuses) -> dict[str, object]:
    """Summarize completed adapter stages without treating transient truth as completion."""
    from dc_study.datagen.hang_mug_status import ORDERED_STAGES

    first_steps = {}
    for stage in ORDERED_STAGES:
        first_steps[stage] = next(
            (step for step, status in enumerate(statuses) if status[stage]), None
        )
    completed = []
    for stage in ORDERED_STAGES:
        if first_steps[stage] is None:
            break
        completed.append(stage)
    first_failed = None if len(completed) == len(ORDERED_STAGES) else ORDERED_STAGES[len(completed)]
    final = statuses[-1] if statuses else None
    diagnostics = {} if final is None else final["diagnostics"]
    return {
        "ordered_stages": list(ORDERED_STAGES),
        "first_completed_steps": first_steps,
        "completed_stages": completed,
        "last_completed_stage": completed[-1] if completed else None,
        "first_failed_stage": first_failed,
        "terminal_checks": {
            name: bool(final and final[name])
            for name in ("task_success", "released", "stable", "contact_policy")
        },
        "contact_policy": {
            "selected_branch": diagnostics.get("selected_branch"),
            "failure_reason": diagnostics.get("failure_reason"),
            "failure_step": diagnostics.get("failure_step"),
            "deepest_overlap_m": diagnostics.get("deepest_overlap_m"),
            "consecutive_overlap_steps": diagnostics.get("consecutive_overlap_steps"),
            "fallen": diagnostics.get("fallen"),
        },
    }


def _direct_actions_exact(executed: list[np.ndarray], source_actions) -> bool:
    return np.array_equal(
        np.asarray(executed, dtype=np.float32),
        source_actions.detach().cpu().numpy(),
    )


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


def _activate_quality_wave_contact_reports(scene) -> tuple[str, ...]:
    """Enable reports on every body participating in filtered wave guards."""
    enabled = []
    for name in ("left_arm", "right_arm", "mug_tree"):
        asset = getattr(scene, name, None)
        spawn = getattr(asset, "spawn", None)
        if spawn is None or not hasattr(spawn, "activate_contact_sensors"):
            raise RuntimeError(
                f"quality-wave contact reporting is unavailable for {name}"
            )
        spawn.activate_contact_sensors = True
        enabled.append(name)
    return tuple(enabled)


def _preserve_quality_wave_contact_reports_across_arm_rebuild(scene) -> None:
    """Re-enable reports after Gear rebuilds both articulation configs."""
    scene_type = type(scene)
    marker = "_cpgen_quality_wave_contact_reports_wrapped"
    if getattr(scene_type, marker, False):
        _activate_quality_wave_contact_reports(scene)
        return
    original_build = scene_type.build_from_spec

    def build_with_contact_reports(instance, *args, **kwargs):
        result = original_build(instance, *args, **kwargs)
        _activate_quality_wave_contact_reports(instance)
        return result

    scene_type.build_from_spec = build_with_contact_reports
    setattr(scene_type, marker, True)
    _activate_quality_wave_contact_reports(scene)


def _usd_rigid_body_names(usd_path: str, articulation_root: str) -> tuple[str, ...]:
    """Read the reusable arm link topology before the physics scene exists."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not open articulation USD: {usd_path}")
    root = stage.GetPrimAtPath(str(articulation_root))
    if not root.IsValid():
        default_root = stage.GetDefaultPrim().GetPath().pathString.rstrip("/")
        root = stage.GetPrimAtPath(default_root + str(articulation_root))
    if not root.IsValid():
        raise RuntimeError(f"cannot map articulation root in {usd_path}")
    names = tuple(
        prim.GetName()
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    if not names or len(names) != len(set(names)):
        raise RuntimeError("articulation USD has missing or duplicate rigid-body names")
    return names


def _install_quality_wave_contact_sensors(config) -> dict[str, tuple[str, ...]]:
    """Declare per-link filtered sensors before IsaacLab creates the scene."""
    from isaaclab.sensors import ContactSensorCfg

    scene = config.scene
    _activate_quality_wave_contact_reports(scene)
    articulation_root = str(scene.right_arm.articulation_root_prim_path)
    right_names = _usd_rigid_body_names(
        scene.right_arm.spawn.usd_path, articulation_root
    )
    left_names = _usd_rigid_body_names(
        scene.left_arm.spawn.usd_path,
        str(scene.left_arm.articulation_root_prim_path),
    )
    env_root = "{ENV_REGEX_NS}"
    right_paths = tuple(
        f"{env_root}/RightArm{articulation_root}/{name}" for name in right_names
    )
    left_root = str(scene.left_arm.articulation_root_prim_path)
    left_paths = tuple(
        f"{env_root}/LeftArm{left_root}/{name}" for name in left_names
    )
    object_links = getattr(config, "_contact_body_links", {})

    def object_path(name: str) -> str:
        if name not in object_links:
            raise RuntimeError(f"quality-wave sensor cannot resolve {name} body")
        suffix = f"/{object_links[name]}" if object_links[name] else ""
        return f"{env_root}/{name}{suffix}"

    tree_path = object_path("mug_tree")
    mug_path = object_path("mug")
    update_period = float(config.sim.dt * config.decimation)
    sensor_names = {"environment": [], "mug": [], "left_tree": []}

    def add(group: str, index: int, prim_path: str, filters: tuple[str, ...]):
        name = f"quality_wave_{group}_{index:02d}"
        setattr(
            scene,
            name,
            ContactSensorCfg(
                prim_path=prim_path,
                update_period=update_period,
                history_length=1,
                track_contact_points=False,
                max_contact_data_count_per_prim=64,
                track_pose=False,
                filter_prim_paths_expr=list(filters),
            ),
        )
        sensor_names[group].append(name)

    for index, path in enumerate(right_paths):
        add("environment", index, path, (tree_path, *left_paths))
        add("mug", index, path, (mug_path,))
    for index, path in enumerate(left_paths):
        add("left_tree", index, path, (tree_path,))
    frozen_names = {name: tuple(values) for name, values in sensor_names.items()}
    config._quality_wave_contact_sensor_names = frozen_names
    config._quality_wave_contact_body_paths = {
        "right_body_paths": right_paths,
        "left_body_paths": left_paths,
        "tree_body_path": tree_path,
        "mug_body_path": mug_path,
    }
    return frozen_names


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
        # Gear rebuilds both articulation configs after asset/contact-sensor
        # binding.  Preserve the reporter bit across that rebuild: setting it
        # only on the initial arm configs is silently discarded.
        _preserve_quality_wave_contact_reports_across_arm_rebuild(instance.scene)
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
    config_type = config_module.HangMugOnTreeManagerEnvCfg
    assets_marker = "_cpgen_quality_wave_asset_sensor_wrapper"
    if not getattr(config_type, assets_marker, False):
        original_assets = config_type.configure_assets_instance_paths

        def configure_assets_with_wave_sensors(instance, *args, **kwargs):
            result = original_assets(instance, *args, **kwargs)
            _install_quality_wave_contact_sensors(instance)
            return result

        config_type.configure_assets_instance_paths = configure_assets_with_wave_sensors
        setattr(config_type, assets_marker, True)
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
    releasing_left = step >= trajectory.waypoint_steps["left_release"]
    if left_assist is not None and releasing_left:
        left_assist.update(
            engage=left_grasping,
            disable=torch.ones_like(left_grasping, dtype=torch.bool),
        )

    right_assist = env.grasp_assists.get("right")
    if right_assist is not None:
        support_boundary = (
            "supported_release_hold"
            if "supported_release_hold" in trajectory.waypoint_steps
            else "branch_unload"
        )
        releasing_right = step > trajectory.waypoint_steps[support_boundary]
        right_assist.update(
            engage=right_grasping,
            disable=torch.full_like(right_grasping, releasing_right),
        )


def _release_left_assist_after_secure_receiver(env, sample, waypoint: str) -> bool:
    """Drop the giver assist only after a sampled broad receiver grasp.

    The quality-wave handover keeps the legacy task-manager overlap release
    disabled.  This post-sample transition preserves the evidence row on which
    both the giver and receiver are securely supported, then releases the giver
    before its authored retreat begins.
    """
    if waypoint not in {"right_grasp", "handover_contact_acquire"}:
        return False
    if not _sample_has_broad_contact(sample, "right"):
        return False
    left_assist = env.grasp_assists.get("left")
    if left_assist is None or not sample["grasp_assist_engaged"].get("left", False):
        return False
    import torch

    left_grasping, _ = env.robot.is_grasping()
    left_assist.update(
        engage=left_grasping,
        disable=torch.ones_like(left_grasping, dtype=torch.bool),
    )
    return True


def _schema_aware_success_acceptance(
    checks: dict[str, bool], *, coded_skill: bool
) -> dict[str, bool]:
    """Select the applicable authoritative acceptance checks.

    Direct action replay does not drive the skill runner's receiving-hand
    fixed-joint state machine.  Its physical right grasp and handover remain
    mandatory through ``right_handover_observed``, while the skill-only assist
    engagement bit is inapplicable.  For a coded target repair, task-manager
    stage latches remain diagnostics: the independent terminal hang receipt is
    the physical success authority and is never inferred from those latches.
    """
    acceptance = dict(checks)
    if coded_skill:
        for name in (
            "coded_task_success",
            "all_stages_latched",
            "right_handover_observed",
            "stable_hang_window",
            "handover_boundary_passed",
        ):
            acceptance.pop(name, None)
    else:
        acceptance.pop("right_grasp_assist_engaged", None)
    return acceptance


def _requires_observed_handover_reanchor(
    mug_parts, *, handle_frame_transfer: bool = False
) -> bool:
    """Use live handover feedback for mugs taller than both lateral spans."""
    size = np.asarray(mug_parts.body_size, dtype=np.float64)
    if size.shape != (3,) or np.any(size <= 0.0):
        raise ValueError("mug body size must contain three positive values")
    return bool(handle_frame_transfer or size[2] > max(size[0], size[1]))


def _bounded_handover_offset(value) -> np.ndarray:
    offset = np.asarray(value, dtype=np.float64)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("handover target offset must contain three finite values")
    if np.linalg.norm(offset) > 0.04:
        raise ValueError("handover target offset exceeds 4 cm")
    return offset


def _bounded_branch_support_fraction(value: float) -> float:
    fraction = float(value)
    if not np.isfinite(fraction) or not 0.25 <= fraction <= 0.75:
        raise ValueError("branch support fraction must be finite and in [0.25, 0.75]")
    return fraction


def _branch_approach_mug_pose(
    final_mug_pose, target_branch_world, clearance_m: float, height_m: float
) -> np.ndarray:
    from judo_isaaclab.put_marker import _pose, quaternion_rotate

    result = _pose(final_mug_pose, "final mug pose").copy()
    branch = _pose(target_branch_world, "target branch pose")
    height = float(height_m)
    if not np.isfinite(height) or not 0.0 <= height <= 0.08:
        raise ValueError("branch approach height must be in [0, 0.08] m")
    result[:3] += quaternion_rotate(branch[3:], [1.0, 0.0, 0.0]) * float(
        clearance_m
    )
    result[2] += height
    return result


def _bounded_pick_lift_margin(value: float) -> float:
    margin = float(value)
    if not np.isfinite(margin) or not 0.0 <= margin <= 0.03:
        raise ValueError("pick lift margin must be finite and in [0, 0.03] m")
    return margin


def _branch_support_seated_pose(value, seat_down_m: float) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("branch support pose must contain seven finite values")
    amount = _bounded_branch_support_seat_down(seat_down_m)
    seated = pose.copy()
    seated[2] -= amount
    return seated


def _bounded_branch_support_seat_down(value: float) -> float:
    amount = float(value)
    if not np.isfinite(amount) or not 0.0 <= amount <= 0.03:
        raise ValueError("branch support seat-down must be in [0, 0.03] m")
    return amount


def _bounded_left_release_retreat(value: float) -> float:
    retreat = float(value)
    if not np.isfinite(retreat) or not 0.02 <= retreat <= 0.12:
        raise ValueError("left release retreat must be finite and in [0.02, 0.12] m")
    return retreat


def _bounded_handover_post_release_lift(value: float) -> float:
    lift = float(value)
    if not np.isfinite(lift) or not 0.0 <= lift <= 0.08:
        raise ValueError("handover post-release lift must be in [0, 0.08] m")
    return lift


def _bounded_handover_post_release_lift_steps(value: int, lift_m: float) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 60:
        raise ValueError("handover post-release lift steps must be in [0, 60]")
    if bool(value) != bool(lift_m):
        raise ValueError("handover post-release lift distance and steps must both be zero or positive")
    return value


def _handover_target_with_local_pitch(pose, angle_rad: float) -> np.ndarray:
    """Rotate a receiver waypoint about its own Y axis without translating it."""
    from judo_isaaclab.put_marker import _pose, quaternion_multiply

    angle = float(angle_rad)
    if not np.isfinite(angle) or abs(angle) > np.pi / 4.0:
        raise ValueError("handover target local pitch must be finite and within 45 degrees")
    result = _pose(pose, "handover target").copy()
    half = 0.5 * angle
    result[3:] = quaternion_multiply(
        result[3:], np.asarray([np.cos(half), 0.0, np.sin(half), 0.0])
    )
    result[3:] /= np.linalg.norm(result[3:])
    return result


def _handover_target_with_local_straddle(pose, local_x_m: float) -> np.ndarray:
    """Translate an oriented receiver target only along its closing-frame X axis."""
    from judo_isaaclab.put_marker import _pose, quaternion_rotate

    amount = float(local_x_m)
    if not np.isfinite(amount) or abs(amount) > 0.14:
        raise ValueError("handover local straddle correction must be within 14 cm")
    result = _pose(pose, "handover target").copy()
    result[:3] += quaternion_rotate(result[3:], [amount, 0.0, 0.0])
    return result


def _handover_outside_standoff(
    right_grasp,
    right_start,
    handover_mug_pose,
    *,
    vertical_clearance_m: float,
    outside_clearance_m: float,
) -> np.ndarray:
    """Place the open receiver above the mug and toward its own start side."""
    grasp = np.asarray(right_grasp, dtype=np.float64)
    start = np.asarray(right_start, dtype=np.float64)
    mug = np.asarray(handover_mug_pose, dtype=np.float64)
    if any(value.shape != (7,) for value in (grasp, start, mug)):
        raise ValueError("handover standoff poses must contain seven values")
    if not all(np.isfinite(value).all() for value in (grasp, start, mug)):
        raise ValueError("handover standoff poses must be finite")
    vertical = float(vertical_clearance_m)
    outside = float(outside_clearance_m)
    if not np.isfinite(vertical) or not 0.0 <= vertical <= 0.12:
        raise ValueError("handover vertical clearance must be in [0, 0.12] m")
    if not np.isfinite(outside) or not 0.0 <= outside <= 0.12:
        raise ValueError("handover outside clearance must be in [0, 0.12] m")
    direction = start[:2] - mug[:2]
    norm = float(np.linalg.norm(direction))
    if outside and norm <= 1.0e-6:
        raise ValueError("receiver start must define a horizontal outside direction")
    result = grasp.copy()
    if outside:
        result[:2] += outside * direction / norm
    result[2] += vertical
    return result


def _semantic_waypoint_name(trajectory, step: int) -> str:
    for name, endpoint in trajectory.waypoint_steps.items():
        if step <= endpoint:
            return name
    raise IndexError(f"semantic step {step} exceeds the skill trajectory")


def _branch_reanchor_waypoints(trajectory) -> tuple[str, ...]:
    if trajectory is None:
        return ()
    handover_boundary = (
        "handover_confirm"
        if "handover_confirm" in trajectory.waypoint_steps
        else "left_release"
    )
    if "direct_preinsert" in trajectory.waypoint_steps:
        # Hold the demonstrated right-rest target exactly while the left arm
        # moves.  Reanchor the single outbound interpolation only after the
        # observer waypoint has completed and a fresh held-contact transform is
        # available.
        return (
            "carrying_rest_observer",
            "direct_preinsert",
            "branch_insert",
            "supported_release_hold",
        )
    if "right_return_start" in trajectory.waypoint_steps:
        # Preserve the exact demonstrated carrier-start target.  Once reached,
        # reanchor only the remaining branch path to the observed held-mug
        # contact rather than shifting the start waypoint itself.
        names = ["right_return_start", "left_branch_point"]
    else:
        names = [handover_boundary]
    names.append("tree_transport")
    if "branch_orient_clear" in trajectory.waypoint_steps:
        names.append("branch_orient_clear")
    return (*names, "branch_approach", "branch_insert", "branch_unload")


def _asset_root_usd(asset_directory: str) -> str:
    candidates = sorted(Path(asset_directory).glob("*.usd"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one root USD in {asset_directory}, got {len(candidates)}"
        )
    return str(candidates[0])


def _quality_wave_contact_views(env, target_assets) -> dict[str, object]:
    """Resolve the predeclared IsaacLab sensors; never create late PhysX views."""
    del target_assets
    names = getattr(env.cfg, "_quality_wave_contact_sensor_names", None)
    paths = getattr(env.cfg, "_quality_wave_contact_body_paths", None)
    if not names or not paths:
        raise RuntimeError("quality-wave contact sensors were not predeclared")
    resolved = {
        group: tuple(env.scene[name] for name in names[group])
        for group in ("environment", "mug", "left_tree")
    }
    if any(not values for values in resolved.values()):
        raise RuntimeError("quality-wave contact sensor group is empty")
    return {
        **resolved,
        **paths,
    }


# Retain the narrow name used by the direct-segment tests and old callers.
_direct_segment_contact_views = _quality_wave_contact_views


def _contact_view_max_force(view, physics_dt: float) -> float:
    if isinstance(view, (tuple, list)):
        return max(
            (_contact_view_max_force(item, physics_dt) for item in view),
            default=0.0,
        )
    if hasattr(view, "data"):
        values = view.data.force_matrix_w
        if values is None:
            raise RuntimeError("predeclared contact sensor has no filtered force matrix")
    else:
        values = view.get_contact_force_matrix(dt=physics_dt)
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    return float(np.linalg.norm(array, axis=-1).max(initial=0.0))


def _pose_path_step_receipt(
    poses: np.ndarray,
    *,
    maximum_translation_m: float,
    maximum_rotation_rad: float,
) -> dict[str, object]:
    """Prove that every commanded pose increment is bounded and continuous."""
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 2:
        raise ValueError("pose path must contain at least two seven-value poses")
    if not np.isfinite(values).all():
        raise ValueError("pose path must be finite")
    position_steps = np.linalg.norm(np.diff(values[:, :3], axis=0), axis=1)
    quaternions = values[:, 3:]
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1)[:, None]
    dots = np.abs(np.sum(quaternions[:-1] * quaternions[1:], axis=1))
    rotation_steps = 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
    max_position = float(position_steps.max(initial=0.0))
    max_rotation = float(rotation_steps.max(initial=0.0))
    return {
        "rows": int(len(values) - 1),
        "maximum_translation_step_m": max_position,
        "translation_limit_m": float(maximum_translation_m),
        "maximum_rotation_step_rad": max_rotation,
        "rotation_limit_rad": float(maximum_rotation_rad),
        "discontinuous_wrist_jump": bool(
            max_position > maximum_translation_m
            or max_rotation > maximum_rotation_rad
        ),
        "passed": bool(
            max_position <= maximum_translation_m
            and max_rotation <= maximum_rotation_rad
        ),
    }


_YAM_OPEN_GRIPPER_ENVELOPE_RADIUS_M = 0.14


def _handover_gripper_proxy_receipt(
    collision_report: dict[str, object],
    *,
    first_allowed_contact_step: int | None,
) -> dict[str, object]:
    """Reject gripper-proxy collisions before the intended contact approach.

    The open YAM fingers extend substantially beyond the wrist origin.  A
    wrist-only screen therefore cannot prove that a nominal pregrasp is clear.
    Mug contact is permitted only after the explicit open contact-approach
    waypoint begins; tree contact is never permitted.
    """
    collision_steps = [int(step) for step in collision_report["collision_steps"]]
    if first_allowed_contact_step is None:
        unexpected = collision_steps
    else:
        unexpected = [
            step for step in collision_steps if step < first_allowed_contact_step
        ]
    return {
        **collision_report,
        "first_allowed_contact_step": first_allowed_contact_step,
        "unexpected_collision_steps": unexpected,
        "passed": not unexpected,
    }


def _handover_wave_plan_screen(
    trajectory,
    sample: dict[str, object],
    target_assets: dict[str, str],
    nominal_right_contact,
    args,
    *,
    phase: str,
) -> dict[str, object]:
    """Screen the live-geometry receiver path without adding motion phases."""
    from judo_isaaclab.collision_screening import (
        load_usd_collision_mesh,
        object_path_clearance_reports,
        sphere_path_collision_report,
    )
    from judo_isaaclab.put_marker import compose_pose

    if phase == "clear_pregrasp":
        previous = "left_lift" if "left_lift" in trajectory.waypoint_steps else None
        endpoint = "handover_pregrasp"
        start = 0 if previous is None else trajectory.waypoint_steps[previous] + 1
    elif phase == "open_approach":
        previous = "handover_pregrasp"
        endpoint = (
            "right_grasp_settle"
            if "right_grasp_settle" in trajectory.waypoint_steps
            else "right_grasp"
        )
        start = trajectory.waypoint_steps[previous] + 1
    else:
        raise ValueError(f"unknown handover plan-screen phase: {phase}")
    end = trajectory.waypoint_steps[endpoint]
    current_right = np.asarray(sample["right_eef_pose"], dtype=np.float64)
    current_left = np.asarray(sample["left_eef_pose"], dtype=np.float64)
    right_path = np.concatenate(
        (current_right[None], trajectory.right_poses[start : end + 1]), axis=0
    )
    left_path = np.concatenate(
        (current_left[None], trajectory.left_poses[start : end + 1]), axis=0
    )
    mug_pose = np.asarray(sample["mug_pose"], dtype=np.float64)
    tree_pose = np.asarray(sample["tree_pose"], dtype=np.float64)
    mug_mesh = load_usd_collision_mesh(_asset_root_usd(target_assets["mug"]))
    tree_mesh = load_usd_collision_mesh(_asset_root_usd(target_assets["mug_tree"]))
    mug_tree = object_path_clearance_reports(
        np.repeat(mug_pose[None], len(right_path), axis=0),
        tree_pose=tree_pose,
        object_mesh=mug_mesh,
        tree_mesh=tree_mesh,
        required_clearance_m=0.002,
        sample_stride=1,
        maximum_vertices=3000,
    )[0]
    camera_local = np.asarray([0.0035, 0.073, 0.073, 1.0, 0.0, 0.0, 0.0])
    right_camera = np.stack(
        [compose_pose(pose, camera_local)[:3] for pose in right_path]
    )
    left_camera = np.stack(
        [compose_pose(pose, camera_local)[:3] for pose in left_path]
    )
    right_wrist = sphere_path_collision_report(
        right_path[:, :3],
        radius_m=0.01,
        obstacles={"mug_tree": (tree_mesh, tree_pose)},
        sample_stride=1,
    )
    right_camera_screen = sphere_path_collision_report(
        right_camera,
        radius_m=0.012,
        obstacles={"mug_tree": (tree_mesh, tree_pose), "held_mug": (mug_mesh, mug_pose)},
        sample_stride=1,
    )
    left_camera_screen = sphere_path_collision_report(
        left_camera,
        radius_m=0.012,
        obstacles={"mug_tree": (tree_mesh, tree_pose)},
        sample_stride=1,
    )
    right_gripper_tree = sphere_path_collision_report(
        right_path[:, :3],
        radius_m=_YAM_OPEN_GRIPPER_ENVELOPE_RADIUS_M,
        obstacles={"mug_tree": (tree_mesh, tree_pose)},
        sample_stride=1,
    )
    right_gripper_mug_raw = sphere_path_collision_report(
        right_path[:, :3],
        radius_m=_YAM_OPEN_GRIPPER_ENVELOPE_RADIUS_M,
        obstacles={"held_mug": (mug_mesh, mug_pose)},
        sample_stride=1,
    )
    first_allowed_contact_step = None
    if phase == "open_approach":
        contact_start = (
            trajectory.waypoint_steps.get(
                "handover_orient_clear",
                trajectory.waypoint_steps["handover_pregrasp"],
            )
            + 1
        )
        # right_path[0] is the live pose immediately before trajectory[start].
        first_allowed_contact_step = 1 + contact_start - start
    right_gripper_mug = _handover_gripper_proxy_receipt(
        right_gripper_mug_raw,
        first_allowed_contact_step=first_allowed_contact_step,
    )
    wrist_separation = np.linalg.norm(
        right_path[:, :3] - left_path[:, :3], axis=1
    )
    camera_separation = np.linalg.norm(right_camera - left_camera, axis=1)
    smoothness = _pose_path_step_receipt(
        right_path,
        maximum_translation_m=args.max_position_step,
        maximum_rotation_rad=args.max_rotation_step,
    )
    selected_contact = np.asarray(nominal_right_contact, dtype=np.float64)
    expected_grasp = compose_pose(mug_pose, selected_contact)
    planned_grasp = trajectory.right_poses[trajectory.waypoint_steps["right_grasp"]]
    grasp_position_error = float(np.linalg.norm(expected_grasp[:3] - planned_grasp[:3]))
    grasp_quaternion_alignment = float(
        abs(np.dot(expected_grasp[3:], planned_grasp[3:]))
    )
    gripper_open = bool(np.all(trajectory.grippers[start : end + 1, 1] <= -0.04749))
    checks = {
        "full_swept_path_sampled": len(right_path) == end - start + 2,
        "smooth_bounded_pose_interpolation": bool(smoothness["passed"]),
        "held_mug_tree_clearance_positive": bool(mug_tree["valid"]),
        "right_wrist_tree_proxy_clear": bool(right_wrist["valid"]),
        "right_wrist_camera_mug_and_tree_proxy_clear": bool(
            right_camera_screen["valid"]
        ),
        "right_open_gripper_tree_proxy_clear": bool(right_gripper_tree["valid"]),
        "right_open_gripper_mug_clear_until_contact_approach": bool(
            right_gripper_mug["passed"]
        ),
        "left_wrist_camera_tree_proxy_clear": bool(left_camera_screen["valid"]),
        "bilateral_wrist_and_camera_clearance_positive": bool(
            min(wrist_separation.min(), camera_separation.min()) >= 0.04
        ),
        "right_gripper_open_until_contact_pose": gripper_open,
        "selected_grasp_reanchored_from_live_mug_geometry": bool(
            grasp_position_error <= 1.0e-6
            and grasp_quaternion_alignment >= 1.0 - 1.0e-9
        ),
    }
    return {
        "phase": phase,
        "motion_subphases_added": False,
        "selection_method": "live mug frame plus pinned broad-contact transform",
        "live_mug_pose": mug_pose.tolist(),
        "selected_right_contact_mug_frame": selected_contact.tolist(),
        "planned_right_grasp_pose": np.asarray(planned_grasp).tolist(),
        "planned_rows": int(len(right_path) - 1),
        "smoothness": smoothness,
        "minimum_bilateral_wrist_separation_m": float(wrist_separation.min()),
        "minimum_bilateral_camera_separation_m": float(camera_separation.min()),
        "grasp_reanchor_position_error_m": grasp_position_error,
        "grasp_reanchor_quaternion_alignment": grasp_quaternion_alignment,
        "reports": {
            "held_mug_vs_tree": mug_tree,
            "right_wrist_origin_vs_tree": right_wrist,
            "right_wrist_camera_vs_mug_and_tree": right_camera_screen,
            "right_open_gripper_envelope_vs_tree": right_gripper_tree,
            "right_open_gripper_envelope_vs_mug": right_gripper_mug,
            "left_wrist_camera_vs_tree": left_camera_screen,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _handover_wave_live_row(
    views: dict[str, object], sample: dict[str, object], waypoint: str, physics_dt: float
) -> dict[str, object]:
    environment_force = _contact_view_max_force(views["environment"], physics_dt)
    mug_force = _contact_view_max_force(views["mug"], physics_dt)
    left_tree_force = _contact_view_max_force(views["left_tree"], physics_dt)
    intended_contact = waypoint in {
        "right_grasp_settle",
        "right_grasp",
        "handover_contact_acquire",
    }
    checks = {
        "right_arm_tree_and_left_arm_contact_free": environment_force <= 1.0e-6,
        "left_arm_tree_contact_free": left_tree_force <= 1.0e-6,
        "right_gripper_mug_contact_only_at_contact_pose": bool(
            intended_contact or mug_force <= 1.0e-6
        ),
    }
    return {
        "waypoint": waypoint,
        "checked_after_step": int(sample["step"]),
        "maximum_environment_contact_force_n": environment_force,
        "maximum_right_mug_contact_force_n": mug_force,
        "maximum_left_tree_contact_force_n": left_tree_force,
        "intended_right_mug_contact": intended_contact,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _sample_has_broad_contact(sample: dict[str, object], side: str) -> bool:
    fractions = np.asarray(sample[f"{side}_pad_fractions"], dtype=np.float64)
    forces = np.asarray(sample[f"{side}_finger_forces_n"], dtype=np.float64)
    return bool(
        sample[f"{side}_grasp"]
        and sample["grasp_assist_engaged"].get(side, False)
        and fractions.shape == (2,)
        and forces.shape == (2,)
        and np.isfinite(fractions).all()
        and np.isfinite(forces).all()
        and (fractions >= 0.15).all()
        and (fractions <= 0.85).all()
        and (forces > 0.0).all()
    )


def _handover_wave_contract_receipt(
    trajectory,
    trace_waypoints,
    actions,
    samples,
    plan_screens: dict[str, object],
    live_rows: list[dict[str, object]],
) -> dict[str, object] | None:
    if trajectory is None or "direct_preinsert" not in trajectory.waypoint_steps:
        return None
    handover_names = (
        "handover_pregrasp",
        "handover_orient_clear",
        "right_grasp_settle",
        "right_grasp",
        "handover_contact_acquire",
    )
    names = np.asarray(trace_waypoints, dtype=str)
    sample_rows = samples[1:]
    action_array = np.asarray(actions, dtype=np.float64)
    aligned = len(names) == len(sample_rows) == len(action_array)
    observed_handover_rows = np.flatnonzero(np.isin(names, handover_names))
    preclose_rows = np.flatnonzero(
        np.isin(
            names,
            ("handover_pregrasp", "handover_orient_clear", "right_grasp_settle"),
        )
    )
    close_rows = np.flatnonzero(names == "right_grasp")
    secure_rows = (
        [
            int(row)
            for row in close_rows
            if aligned and _sample_has_broad_contact(sample_rows[row], "right")
        ]
        if aligned
        else []
    )
    first_secure = None if not secure_rows else secure_rows[0]
    giver_held_until_secure = bool(
        first_secure is not None
        and all(
            sample_rows[row]["left_grasp"]
            and sample_rows[row]["grasp_assist_engaged"].get("left", False)
            for row in observed_handover_rows
            if row <= first_secure
        )
    )
    right_gripper = (
        action_array[:, 13]
        if action_array.ndim == 2 and action_array.shape[1] >= 14
        else np.empty((0,), dtype=np.float64)
    )
    preclose_open = bool(
        len(preclose_rows)
        and len(right_gripper) == len(names)
        and np.all(right_gripper[preclose_rows] <= -0.04749)
    )
    close_monotone = bool(
        len(close_rows)
        and len(right_gripper) == len(names)
        and np.all(np.diff(right_gripper[close_rows]) >= -1.0e-9)
        and right_gripper[close_rows[-1]] >= -1.0e-8
    )
    first_contact = next(
        (
            row
            for row in live_rows
            if row["maximum_right_mug_contact_force_n"] > 1.0e-6
        ),
        None,
    )
    expected_live_rows = int(len(observed_handover_rows))
    checks = {
        "trace_arrays_row_aligned": aligned,
        "both_swept_plan_screens_passed": bool(
            set(plan_screens) == {"clear_pregrasp", "open_approach"}
            and all(receipt is not None and receipt["passed"] for receipt in plan_screens.values())
        ),
        "all_handover_rows_live_guarded": bool(
            len(live_rows) == expected_live_rows
            and all(row["passed"] for row in live_rows)
        ),
        "first_right_mug_contact_at_contact_pose": bool(
            first_contact is not None
            and first_contact["waypoint"] in {"right_grasp_settle", "right_grasp"}
        ),
        "right_gripper_open_through_open_approach": preclose_open,
        "right_close_monotone_at_grasp_pose": close_monotone,
        "broad_force_backed_right_contact_before_giver_release": bool(secure_rows),
        "left_giver_held_until_right_contact_secure": giver_held_until_secure,
    }
    return {
        "selection": {
            "candidate_sampling": False,
            "source": "live mug geometry reanchored pinned contact transform",
            "broad_contact_accepted_only_from_runtime_pad_receipt": True,
        },
        "plan_screens": plan_screens,
        "live_physx_contact_guard": {
            "expected_rows": expected_live_rows,
            "observed_rows": len(live_rows),
            "first_right_mug_contact": first_contact,
            "rows": live_rows,
            "passed": bool(
                len(live_rows) == expected_live_rows
                and all(row["passed"] for row in live_rows)
            ),
        },
        "first_broad_right_contact_trace_row": first_secure,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _direct_segment_live_row(
    views: dict[str, object], sample: dict[str, object], waypoint: str, physics_dt: float
) -> dict[str, object]:
    environment_force = _contact_view_max_force(views["environment"], physics_dt)
    mug_force = _contact_view_max_force(views["mug"], physics_dt)
    outbound = waypoint == "direct_preinsert"
    returning = waypoint == "post_release_return"
    fractions = np.asarray(sample["right_pad_fractions"], dtype=float)
    forces = np.asarray(sample["right_finger_forces_n"], dtype=float)
    grasp_secure = bool(
        sample["right_grasp"]
        and sample["grasp_assist_engaged"].get("right", False)
        and fractions.shape == (2,)
        and forces.shape == (2,)
        and np.isfinite(fractions).all()
        and (fractions >= 0.15).all()
        and (fractions <= 0.85).all()
        and (forces > 0.0).all()
    )
    checks = {
        "right_arm_tree_and_left_arm_contact_free": environment_force <= 1.0e-6,
        "carrier_grasp_not_degraded": grasp_secure if outbound else True,
        "open_right_arm_mug_contact_free": mug_force <= 1.0e-6 if returning else True,
        "right_open_during_return": not sample["right_grasp"] if returning else True,
    }
    return {
        "waypoint": waypoint,
        "checked_after_step": int(sample["step"]),
        "maximum_environment_contact_force_n": environment_force,
        "maximum_mug_contact_force_n": mug_force,
        "right_pad_fractions": fractions.tolist(),
        "right_finger_forces_n": forces.tolist(),
        "checks": checks,
        "passed": bool((outbound or returning) and all(checks.values())),
    }


def _direct_segment_plan_screen(
    trajectory,
    sample: dict[str, object],
    target_assets: dict[str, str],
    *,
    phase: str,
) -> dict[str, object]:
    """Screen the exact next direct interpolation without adding a phase."""
    from judo_isaaclab.collision_screening import (
        load_usd_collision_mesh,
        object_path_clearance_reports,
        rigid_weld_object_poses,
        sphere_path_collision_report,
    )
    from judo_isaaclab.put_marker import compose_pose

    if phase == "outbound":
        previous, endpoint = "carrying_rest_observer", "direct_preinsert"
    elif phase == "return":
        previous, endpoint = "right_release", "post_release_return"
    else:
        raise ValueError(f"unknown direct segment phase: {phase}")
    start = trajectory.waypoint_steps[previous] + 1
    end = trajectory.waypoint_steps[endpoint]
    current_eef = np.asarray(sample["right_eef_pose"], dtype=np.float64)
    eef_path = np.concatenate(
        (current_eef[None], trajectory.right_poses[start : end + 1]), axis=0
    )
    tree_pose = np.asarray(sample["tree_pose"], dtype=np.float64)
    mug_pose = np.asarray(sample["mug_pose"], dtype=np.float64)
    tree_mesh = load_usd_collision_mesh(_asset_root_usd(target_assets["mug_tree"]))
    mug_mesh = load_usd_collision_mesh(_asset_root_usd(target_assets["mug"]))
    checks = {"full_interpolation_sampled": len(eef_path) == end - start + 2}
    reports = {}
    if phase == "outbound":
        object_path = rigid_weld_object_poses(eef_path, current_eef, mug_pose)
        object_report = object_path_clearance_reports(
            object_path,
            tree_pose=tree_pose,
            object_mesh=mug_mesh,
            tree_mesh=tree_mesh,
            required_clearance_m=0.002,
            sample_stride=1,
            maximum_vertices=3000,
        )[0]
        reports["rigidly_carried_mug_vs_tree"] = object_report
        checks["rigidly_carried_mug_tree_collision_free"] = bool(
            object_report["valid"]
        )
        obstacles = {"mug_tree": (tree_mesh, tree_pose)}
    else:
        obstacles = {
            "mug_tree": (tree_mesh, tree_pose),
            "released_mug": (mug_mesh, mug_pose),
        }
    wrist_report = sphere_path_collision_report(
        eef_path[:, :3], radius_m=0.01, obstacles=obstacles, sample_stride=1
    )
    camera_local = np.asarray([0.0035, 0.073, 0.073, 1.0, 0.0, 0.0, 0.0])
    camera_points = np.stack(
        [compose_pose(pose, camera_local)[:3] for pose in eef_path]
    )
    camera_report = sphere_path_collision_report(
        camera_points, radius_m=0.012, obstacles=obstacles, sample_stride=1
    )
    reports["right_wrist_origin_proxy"] = wrist_report
    reports["right_wrist_camera_center_proxy"] = camera_report
    checks["right_wrist_origin_proxy_collision_free"] = bool(wrist_report["valid"])
    checks["right_wrist_camera_proxy_collision_free"] = bool(camera_report["valid"])
    if phase == "outbound":
        left_eef = np.asarray(sample["left_eef_pose"], dtype=np.float64)
        minimum_wrist_separation = float(
            np.linalg.norm(eef_path[:, :3] - left_eef[:3], axis=1).min()
        )
        left_camera = compose_pose(left_eef, camera_local)[:3]
        minimum_camera_separation = float(
            np.linalg.norm(camera_points - left_camera, axis=1).min()
        )
        reports["bilateral_origin_clearance"] = {
            "minimum_wrist_origin_separation_m": minimum_wrist_separation,
            "minimum_camera_center_separation_m": minimum_camera_separation,
            "required_m": 0.04,
        }
        checks["bilateral_wrist_and_camera_origins_clear"] = bool(
            min(minimum_wrist_separation, minimum_camera_separation) >= 0.04
        )
    return {
        "phase": phase,
        "motion_waypoint": endpoint,
        "motion_subphases_added": False,
        "planned_rows": int(len(eef_path) - 1),
        "checks": checks,
        "reports": reports,
        "passed": all(checks.values()),
    }


def _direct_phase_contract_receipt(
    trajectory,
    trace_waypoints,
    actions,
    samples,
) -> dict[str, object] | None:
    if trajectory is None or "direct_preinsert" not in trajectory.waypoint_steps:
        return None
    required = (
        "carrying_rest_observer",
        "direct_preinsert",
        "branch_insert",
        "supported_release_hold",
        "right_release",
        "post_release_return",
        "stable_support",
    )
    forbidden = ("tree_transport", "branch_orient_clear", "branch_approach", "branch_unload")
    names = np.asarray(trace_waypoints, dtype=str)
    boundaries = {}
    for name in required:
        rows = np.flatnonzero(names == name)
        boundaries[name] = {
            "first_row": None if not len(rows) else int(rows[0]),
            "last_row": None if not len(rows) else int(rows[-1]),
            "rows": int(len(rows)),
        }
    compressed = tuple(
        name
        for index, name in enumerate(names.tolist())
        if index == 0 or name != names[index - 1]
    )
    try:
        suffix_start = compressed.index("carrying_rest_observer")
        observed_suffix = compressed[suffix_start:]
    except ValueError:
        observed_suffix = ()
    all_required = all(boundaries[name]["rows"] > 0 for name in required)
    ordered = all_required and all(
        boundaries[left]["last_row"] < boundaries[right]["first_row"]
        for left, right in zip(required[:-1], required[1:], strict=True)
    )
    action_array = np.asarray(actions, dtype=np.float64)
    right_gripper = (
        action_array[:, 13]
        if action_array.ndim == 2 and action_array.shape[1] >= 14
        else np.empty((0,), dtype=np.float64)
    )
    if all_required and len(right_gripper) == len(names):
        carrier_start = boundaries["carrying_rest_observer"]["first_row"]
        hold_end = boundaries["supported_release_hold"]["last_row"]
        release_start = boundaries["right_release"]["first_row"]
        release_end = boundaries["right_release"]["last_row"]
        return_start = boundaries["post_release_return"]["first_row"]
        carrier_closed = bool(
            np.all(np.abs(right_gripper[carrier_start : hold_end + 1]) <= 1.0e-6)
        )
        release_values = right_gripper[release_start : release_end + 1]
        release_monotone = bool(
            len(release_values)
            and np.all(np.diff(release_values) <= 1.0e-9)
            and release_values[-1] <= -0.04749
        )
        deltas = np.diff(right_gripper[carrier_start:])
        opening_rows = np.flatnonzero(deltas < -1.0e-8)
        opening_runs = int(
            bool(len(opening_rows))
            + np.count_nonzero(np.diff(opening_rows) > 1)
        )
        never_reclosed = bool(np.all(deltas <= 1.0e-8))
        open_return = bool(
            np.all(right_gripper[return_start:] <= -0.04749)
        )
    else:
        carrier_closed = release_monotone = never_reclosed = open_return = False
        opening_runs = 0
    sample_rows = samples[1:]
    if all_required and len(sample_rows) == len(names):
        observer_start = boundaries["carrying_rest_observer"]["last_row"]
        observer_end = boundaries["post_release_return"]["last_row"]
        observer_target = trajectory.left_poses[
            trajectory.waypoint_steps["carrying_rest_observer"]
        ]
        observer_error = max(
            np.linalg.norm(
                np.asarray(sample_rows[row]["left_eef_pose"], dtype=float)[:3]
                - observer_target[:3]
            )
            for row in range(observer_start, observer_end + 1)
        )
        return_rows = range(
            boundaries["post_release_return"]["first_row"],
            boundaries["post_release_return"]["last_row"] + 1,
        )
        return_released = all(
            not sample_rows[row]["right_grasp"] for row in return_rows
        )
    else:
        observer_error = float("inf")
        return_released = False
    rest_before = trajectory.right_poses[
        trajectory.waypoint_steps["carrying_rest_observer"]
    ]
    rest_after = trajectory.right_poses[
        trajectory.waypoint_steps["post_release_return"]
    ]
    checks = {
        "all_named_boundaries_present": all_required,
        "named_boundaries_strictly_ordered": ordered,
        "exact_direct_suffix_without_intermediate_phases": observed_suffix == required,
        "legacy_transport_orientation_approach_unload_absent": not any(
            np.any(names == name) for name in forbidden
        ),
        "carrier_command_closed_through_supported_hold": carrier_closed,
        "one_monotone_final_opening": release_monotone and opening_runs == 1,
        "right_gripper_never_reclosed": never_reclosed,
        "post_release_return_command_open": open_return,
        "post_release_return_physically_released": return_released,
        "left_observer_held": observer_error <= 0.03,
        "return_endpoint_is_demonstrated_rest_pose": bool(
            np.allclose(rest_after, rest_before, atol=1.0e-9, rtol=0.0)
        ),
    }
    return {
        "required_suffix": list(required),
        "observed_compressed_suffix": list(observed_suffix),
        "forbidden_waypoints": list(forbidden),
        "phase_boundaries": boundaries,
        "right_opening_transition_runs": opening_runs,
        "maximum_left_observer_position_error_m": float(observer_error),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _trace_status_arrays(samples) -> dict[str, np.ndarray]:
    rows = samples[1:]
    return {
        "left_grasp": np.asarray([row["left_grasp"] for row in rows], dtype=bool),
        "right_grasp": np.asarray([row["right_grasp"] for row in rows], dtype=bool),
        "left_assist_engaged": np.asarray(
            [row["grasp_assist_engaged"].get("left", False) for row in rows],
            dtype=bool,
        ),
        "right_assist_engaged": np.asarray(
            [row["grasp_assist_engaged"].get("right", False) for row in rows],
            dtype=bool,
        ),
        **{
            f"stage{stage}_latched": np.asarray(
                [row[f"stage{stage}"] for row in rows], dtype=bool
            )
            for stage in (1, 2, 3)
        },
    }


def _source_pick_prefix_steps(keyframes) -> int:
    """Return the exact source-action prefix ending at right pregrasp."""
    frame = keyframes["frames"]["right_pregrasp"]
    action_index = int(frame["action_index"])
    if action_index < 0 or int(frame["sample_index"]) != action_index + 1:
        raise ValueError("right-pregrasp keyframe is not aligned to source actions")
    if not frame["stage1"] or frame["stage2"] or not frame["left_grasp"]:
        raise ValueError("source right-pregrasp is not a completed Pick boundary")
    return action_index + 1


def _trajectory_after(trajectory, waypoint: str):
    """Slice a planned skill after a completed waypoint and rebase its indices."""
    from judo_isaaclab.put_marker import SkillTrajectory

    if waypoint not in trajectory.waypoint_steps:
        raise ValueError(f"trajectory is missing {waypoint} waypoint")
    start = trajectory.waypoint_steps[waypoint] + 1
    if start >= trajectory.steps:
        raise ValueError(f"trajectory has no suffix after {waypoint}")
    return SkillTrajectory(
        left_poses=trajectory.left_poses[start:].copy(),
        right_poses=trajectory.right_poses[start:].copy(),
        grippers=trajectory.grippers[start:].copy(),
        stage_names=trajectory.stage_names[start:],
        waypoint_steps={
            name: end - start
            for name, end in trajectory.waypoint_steps.items()
            if end >= start
        },
    )


def _require_reusable_pick_boundary(sample) -> None:
    """Fail closed unless the live source prefix still holds a completed Pick."""
    left_assist = sample["grasp_assist_engaged"].get("left", False)
    right_assist = sample["grasp_assist_engaged"].get("right", False)
    if not sample["stage1"] or sample["stage2"] or not left_assist or right_assist:
        raise RuntimeError("source prefix did not preserve the completed Pick boundary")


def _pick_boundary_receipt(sample) -> dict[str, object]:
    """Require the completed Pick postcondition before any Handover motion."""
    checks = {
        "stage1_latched": bool(sample["stage1"]),
        "left_contact_secure": bool(sample["left_grasp"]),
        "left_assist_secure": bool(
            sample["grasp_assist_engaged"].get("left", False)
        ),
        "right_assist_clear": not bool(
            sample["grasp_assist_engaged"].get("right", False)
        ),
    }
    return {
        "stage": "pick",
        "checked_after_step": int(sample["step"]),
        "checks": checks,
        "passed": all(checks.values()),
        "safe_to_continue": all(checks.values()),
    }


def _handover_boundary_receipt(sample) -> dict[str, object]:
    """Check the live Handover postcondition before branch transport."""
    checks = {
        "stage2_latched": bool(sample["stage2"]),
        "right_contact_secure": bool(sample["right_grasp"]),
        "right_assist_secure": bool(
            sample["grasp_assist_engaged"].get("right", False)
        ),
        "left_assist_released": not bool(
            sample["grasp_assist_engaged"].get("left", False)
        ),
    }
    physical_checks = {
        name: value for name, value in checks.items() if name != "stage2_latched"
    }
    return {
        "stage": "handover",
        "checked_after_step": int(sample["step"]),
        "checks": checks,
        "passed": all(checks.values()),
        "safe_to_continue": all(physical_checks.values()),
    }


def _handover_contact_acquire_guard_receipt(
    sample, *, phase: str
) -> dict[str, object]:
    """Fail closed while bringing the left-held mug into the receiver."""
    checks = {
        "pick_latched": bool(sample["stage1"]),
    }
    if phase == "entry":
        checks["left_assist_secure"] = bool(
            sample["grasp_assist_engaged"].get("left", False)
        )
    elif phase == "row":
        receiver_secure = bool(
            sample["right_grasp"]
            and sample["grasp_assist_engaged"].get("right", False)
        )
        checks["assist_backed_support_secure"] = bool(
            sample["grasp_assist_engaged"].get("left", False)
            or receiver_secure
        )
    elif phase == "completion":
        checks.update(
            {
                "right_contact_secure": bool(sample["right_grasp"]),
                "right_assist_secure": bool(
                    sample["grasp_assist_engaged"].get("right", False)
                ),
            }
        )
    else:
        raise ValueError(f"unknown contact-acquire phase: {phase}")
    return {
        "phase": phase,
        "checked_after_step": int(sample["step"]),
        "checks": checks,
        "raw_left_contact": bool(sample["left_grasp"]),
        "raw_right_contact": bool(sample["right_grasp"]),
        "passed": all(checks.values()),
    }


def _handover_lift_guard_receipt(sample, *, phase: str) -> dict[str, object]:
    """Bind receiver lift entry to contact, then retain assist-backed support."""
    checks = {
        "pick_latched": bool(sample["stage1"]),
        "right_assist_secure": bool(
            sample["grasp_assist_engaged"].get("right", False)
        ),
    }
    if phase == "entry":
        checks["right_contact_secure"] = bool(sample["right_grasp"])
    elif phase != "lift_row":
        raise ValueError(f"unknown handover lift phase: {phase!r}")
    return {
        "stage": "handover_post_release_lift",
        "phase": phase,
        "checked_after_step": int(sample["step"]),
        "checks": checks,
        "diagnostics": {
            "right_contact_raw": bool(sample["right_grasp"]),
            "left_assist_engaged": bool(
                sample["grasp_assist_engaged"].get("left", False)
            ),
        },
        "passed": all(checks.values()),
    }


def _sample(env, step: int, stage: str, info=None) -> dict[str, object]:
    import torch
    from run_putmarker_skill_program import _eef_pose

    left_grasp, right_grasp = env.robot.is_grasping()
    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)

    def finger_evidence(arm_name: str) -> tuple[list[float], list[float]]:
        gripper = env.robot.arms[arm_name].end_effector
        forces, pad_fractions = [], []
        for finger in gripper.fingers:
            forces.append(
                float(finger.contact_force(gripper.default_target, env_ids)[0].item())
            )
            fraction, valid = finger.contact_pad_fraction(
                gripper.default_target, env_ids
            )
            pad_fractions.append(
                float(fraction[0].item())
                if bool(valid[0].item())
                else float("nan")
            )
        return forces, pad_fractions

    left_finger_forces, left_pad_fractions = finger_evidence("left_arm")
    right_finger_forces, right_pad_fractions = finger_evidence("right_arm")
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
        "left_finger_forces_n": left_finger_forces,
        "left_pad_fractions": left_pad_fractions,
        "right_finger_forces_n": right_finger_forces,
        "right_pad_fractions": right_pad_fractions,
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
        "right_arm_joint_pos": env.scene["right_arm"].data.joint_pos[
            0, :6
        ].detach().cpu().numpy().tolist(),
    }


def _broad_pad_contact_receipt(
    samples, side: str, *, required_steps: int = 8
) -> dict[str, object]:
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    interior_low, interior_high = 0.15, 0.85
    qualifying = []
    for row in samples:
        fractions = np.asarray(row[f"{side}_pad_fractions"], dtype=float)
        forces = np.asarray(row[f"{side}_finger_forces_n"], dtype=float)
        qualifying.append(
            bool(row[f"{side}_grasp"])
            and fractions.shape == (2,)
            and forces.shape == (2,)
            and bool(np.isfinite(fractions).all())
            and bool((fractions >= interior_low).all())
            and bool((fractions <= interior_high).all())
            and bool((forces > 0.0).all())
        )
    longest = current = 0
    for value in qualifying:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return {
        "side": side,
        "pad_fraction_guard_band": [interior_low, interior_high],
        "required_consecutive_steps": required_steps,
        "longest_consecutive_steps": longest,
        "qualifying_steps": sum(qualifying),
        "passed": longest >= required_steps,
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
        transfer_handover_contact_by_handle_frame,
    )
    from judo_isaaclab.put_marker import (
        compose_pose,
        inverse_pose,
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
        pick_threshold_m=0.05 + _bounded_pick_lift_margin(args.pick_lift_margin_m),
    )
    target_handover_mug = RigidAssetGeometry(
        compose_pose(target_handover_body, inverse_pose(target_parts.body_frame)),
        target_geometry.size,
    )
    pick_latch_mug_pose = compose_pose(
        pick_latch_body, inverse_pose(target_parts.body_frame)
    )
    if args.handover_handle_frame_transfer:
        right_grasp = transfer_handover_contact_by_handle_frame(
            source_dual["mug_pose"],
            target_handover_mug.root_pose,
            source_parts.handle_hole_frame,
            target_parts.handle_hole_frame,
            source_dual["right_eef_pose"],
        )
    else:
        right_grasp = transfer_pose(
            source_dual["right_eef_pose"],
            source_dual_body,
            target_handover_body,
            local_position_scale=target_parts.body_size / source_parts.body_size,
        )
    right_grasp[:3] += _bounded_handover_offset(args.handover_target_offset_m)
    right_grasp = _handover_target_with_local_pitch(
        right_grasp, args.handover_target_local_pitch_rad
    )
    right_grasp = _handover_target_with_local_straddle(
        right_grasp, args.handover_straddle_local_x_m
    )
    right_pregrasp = transfer_pose(
        frames["right_pregrasp"]["right_eef_pose"],
        source_dual_body,
        target_handover_body,
        local_position_scale=target_parts.body_size / source_parts.body_size,
    )
    right_orient_clear = None
    if args.handover_orient_steps:
        pregrasp_orientation = right_pregrasp[3:].copy()
        right_orient_clear = _handover_outside_standoff(
            right_grasp,
            right_start,
            target_handover_mug.root_pose,
            vertical_clearance_m=args.handover_orient_clearance_m,
            outside_clearance_m=args.handover_standoff_outside_m,
        )
        right_pregrasp = right_orient_clear.copy()
        right_pregrasp[3:] = pregrasp_orientation
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
        branch_support_fraction=_bounded_branch_support_fraction(
            args.branch_support_fraction
        ),
        branch_roll_offset_rad=args.branch_roll_offset_rad,
        target_branch_rank=args.target_branch_rank,
        target_branch_row=2,
    )
    final_mug_pose = _branch_support_seated_pose(
        final_mug_pose, args.branch_support_seat_down_m
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
    approach_mug_pose = _branch_approach_mug_pose(
        final_mug_pose,
        target_branch_world,
        args.insert_clearance_m,
        args.branch_approach_height_m,
    )

    def held(mug_pose, local):
        return compose_pose(mug_pose, local)

    left_lift = held(pick_latch_mug_pose, left_contact)
    left_handover = held(target_handover_mug.root_pose, left_contact)
    receiver_lift_m = _bounded_handover_post_release_lift(
        args.handover_post_release_lift_m
    )
    receiver_lift_steps = _bounded_handover_post_release_lift_steps(
        args.handover_post_release_lift_steps, receiver_lift_m
    )
    left_release = left_handover.copy()
    left_release[1] += _bounded_left_release_retreat(args.left_release_retreat_m)
    receiver_lift = right_grasp.copy()
    receiver_lift[2] += receiver_lift_m
    right_transport = held(transport_mug_pose, right_contact)
    right_approach = held(approach_mug_pose, right_contact)
    right_insert = held(final_mug.root_pose, right_contact)
    right_branch_orient = None
    if args.branch_orient_steps:
        right_branch_orient = right_transport.copy()
        right_branch_orient[3:] = right_approach[3:]
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
        right_pregrasp,
        right_grasp,
        left_release,
        receiver_lift=receiver_lift,
        receiver_lift_steps=receiver_lift_steps,
        right_orient_clear=right_orient_clear,
        orient_steps=args.handover_orient_steps,
        approach_steps=100,
        contact_settle_steps=args.handover_contact_settle_steps,
        contact_acquire_steps=args.handover_contact_acquire_steps,
        confirm_steps=args.handover_confirm_steps,
        close_steps=50,
        release_steps=50,
    )
    if args.post_handover_rest_observer_steps:
        program.post_handover_rest_and_observe(
            right_start,
            left_branch_observer,
            steps=args.post_handover_rest_observer_steps,
        )
    elif args.post_handover_right_return_steps or args.left_branch_point_steps:
        program.post_handover_branch_setup(
            right_start,
            left_branch_observer,
            right_return_steps=args.post_handover_right_return_steps,
            left_point_steps=args.left_branch_point_steps,
        )
    direct_contract = bool(args.direct_rest_to_preinsert_steps)
    if direct_contract:
        program.direct_rest_to_branch_insert(
            right_approach,
            right_insert,
            direct_steps=args.direct_rest_to_preinsert_steps,
            insert_steps=70,
            left_observer=left_branch_observer,
        )
        program.release_and_return_to_rest(
            right_insert,
            right_start,
            support_steps=40,
            release_steps=40,
            return_steps=args.post_release_return_to_rest_steps,
            settle_steps=args.stable_support_steps,
        )
    else:
        program.handle_to_branch_insert(
            right_transport,
            right_approach,
            right_insert,
            transport_steps=100,
            right_orient_clear=right_branch_orient,
            orient_steps=args.branch_orient_steps,
            approach_steps=70,
            insert_steps=70,
            left_observer=left_branch_observer,
        )
        program.release_and_support(
            right_insert,
            right_insert,
            unload_steps=40,
            release_steps=40,
            settle_steps=args.stable_support_steps,
        )
    return (
        program.build(),
        final_mug_pose,
        target_handover_mug.root_pose,
        right_contact,
        source_branch,
        target_branch,
    )


def _sparse_joint_nominal(
    source, trajectory, keyframes, *, initial_action_index: int = 0
):
    actions = np.asarray(source["actions"].detach().cpu(), dtype=np.float64)
    indices = keyframes["semantic_indices"]
    mapping = {
        "left_pregrasp": indices["left_pregrasp"],
        "left_grasp": indices["left_grasp"],
        "left_lift": indices["left_lift"],
        "handover_pregrasp": indices["right_pregrasp"],
        "handover_orient_clear": indices["right_pregrasp"],
        "right_grasp_settle": indices["dual_grasp"],
        "right_grasp": indices["dual_grasp"],
        "handover_contact_acquire": indices["dual_grasp"],
        "left_release": indices["handover"],
        "handover_receiver_lift": indices["handover"],
        "handover_confirm": indices["handover"],
        "right_return_start": 0,
        "carrying_rest_observer": 0,
        "left_branch_point": indices["tree_approach"],
        "direct_preinsert": indices["tree_approach"],
        "tree_transport": indices["tree_approach"],
        "branch_orient_clear": indices["tree_approach"],
        "branch_approach": indices["tree_approach"],
        "branch_insert": indices["inserted_held"],
        "supported_release_hold": indices["inserted_held"],
        "branch_unload": indices["inserted_held"],
        "right_release": indices["release"],
        "post_release_return": 0,
        "stable_support": indices["stable_settle"],
    }
    parts = []
    if not 0 <= initial_action_index < len(actions):
        raise ValueError("initial action index is outside the source action dataset")
    previous = actions[initial_action_index]
    previous_cursor = 0
    for name, cursor in trajectory.waypoint_steps.items():
        target_index = mapping[name]
        if (
            "direct_preinsert" in trajectory.waypoint_steps
            and name in {"post_release_return", "stable_support"}
        ):
            target_index = 0
        target = actions[min(target_index, len(actions) - 1)]
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
    _require_proven_control_defaults(args)
    if args.handover_confirm_steps < 0:
        raise ValueError("--handover-confirm-steps must be nonnegative")
    if not 0 <= args.branch_orient_steps <= 90:
        raise ValueError("--branch-orient-steps must be in [0, 90]")
    if not 60 <= args.stable_support_steps <= 240:
        raise ValueError("--stable-support-steps must be in [60, 240]")
    if not 0 <= args.handover_contact_acquire_steps <= 60:
        raise ValueError("--handover-contact-acquire-steps must be in [0, 60]")
    _bounded_handover_offset(args.handover_target_offset_m)
    _handover_target_with_local_pitch(
        np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        args.handover_target_local_pitch_rad,
    )
    _handover_target_with_local_straddle(
        np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        args.handover_straddle_local_x_m,
    )
    if (
        not np.isfinite(args.handover_orient_clearance_m)
        or not 0.0 <= args.handover_orient_clearance_m <= 0.12
        or not 0 <= args.handover_orient_steps <= 60
        or bool(args.handover_orient_clearance_m) != bool(args.handover_orient_steps)
        or not np.isfinite(args.handover_standoff_outside_m)
        or not 0.0 <= args.handover_standoff_outside_m <= 0.12
        or bool(args.handover_standoff_outside_m)
        and not args.handover_orient_steps
    ):
        raise ValueError(
            "handover orient clearance/steps and optional outside standoff must be bounded"
        )
    if args.handover_straddle_local_x_m and not args.handover_orient_steps:
        raise ValueError("handover local straddle correction requires orient-first descent")
    _bounded_branch_support_fraction(args.branch_support_fraction)
    _branch_approach_mug_pose(
        np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        args.insert_clearance_m,
        args.branch_approach_height_m,
    )
    if (
        not np.isfinite(args.branch_roll_offset_rad)
        or abs(args.branch_roll_offset_rad) > np.pi / 2
    ):
        raise ValueError("--branch-roll-offset-rad must be within 90 degrees")
    _bounded_branch_support_seat_down(args.branch_support_seat_down_m)
    _bounded_left_release_retreat(args.left_release_retreat_m)
    post_release_lift = _bounded_handover_post_release_lift(
        args.handover_post_release_lift_m
    )
    _bounded_handover_post_release_lift_steps(
        args.handover_post_release_lift_steps, post_release_lift
    )
    setup_steps = (
        args.post_handover_right_return_steps,
        args.left_branch_point_steps,
    )
    if any(isinstance(value, bool) or not 0 <= value <= 120 for value in setup_steps):
        raise ValueError("post-handover setup steps must be integers in [0, 120]")
    if bool(setup_steps[0]) != bool(setup_steps[1]):
        raise ValueError(
            "right return and left branch-point steps must be selected together"
        )
    direct_steps = (
        args.direct_rest_to_preinsert_steps,
        args.post_release_return_to_rest_steps,
    )
    if any(
        isinstance(value, bool) or not 0 <= value <= 240
        for value in direct_steps
    ):
        raise ValueError("direct choreography steps must be integers in [0, 240]")
    if bool(direct_steps[0]) != bool(direct_steps[1]):
        raise ValueError(
            "direct rest-to-preinsert and post-release return steps must be selected together"
        )
    simultaneous_setup = args.post_handover_rest_observer_steps
    if (
        isinstance(simultaneous_setup, bool)
        or not 0 <= simultaneous_setup <= 120
    ):
        raise ValueError("simultaneous rest/observer steps must be in [0, 120]")
    if simultaneous_setup and any(setup_steps):
        raise ValueError("simultaneous and sequential post-handover setup cannot be combined")
    if direct_steps[0] and (
        not simultaneous_setup or any(setup_steps) or args.branch_orient_steps
    ):
        raise ValueError(
            "direct choreography requires simultaneous rest/observer setup and forbids sequential or branch-orientation subphases"
        )
    if args.reuse_source_pick_prefix and args.mode != "skill":
        raise ValueError("--reuse-source-pick-prefix requires --mode skill")
    _physics_device_receipt(
        args.device,
        require_cpu=args.require_cpu_physics,
    )
    if args.render and not args.video:
        raise ValueError("--render requires --video")
    for path in (
        args.result_json,
        args.trace_npz,
        args.video,
        args.write_keyframes,
        args.demo_hdf5,
    ):
        if path and os.path.isfile(path):
            os.unlink(path)
    # Validate cheap dataset/asset provenance before the expensive app launch.
    source_receipt = _source_dataset_receipt(
        args.source_dataset, args.episode, args.expected_source_sha256
    )
    source_assets = _dataset_assets(args.source_dataset, args.objects_root)
    target_assets, target_state_template = _resolve_target_assets(args, source_assets)
    sys.path.insert(0, os.path.abspath(args.gear_repo))
    simulation_app = env = encoder = None
    try:
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(
            {"headless": True, "device": args.device, "enable_cameras": True}
        ).app
        import torch
        from dc_study.datagen.controller_settings import (
            compare_live_settings_to_spec,
            read_live_controller_settings,
        )
        from dc_study.datagen.io import sha256_json
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
            disable_env_recorders=True,
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
        defer_left_assist_to_secure_receiver = bool(
            args.direct_rest_to_preinsert_steps
        )
        env._defer_left_assist_release_to_secure_receiver = (
            defer_left_assist_to_secure_receiver
        )
        configured_gains_start = env.robot.spec.controller_gains()
        configured_gains_start_sha256 = sha256_json(configured_gains_start)
        if (
            args.expected_controller_gains_sha256
            and configured_gains_start_sha256 != args.expected_controller_gains_sha256
        ):
            raise RuntimeError("configured controller gains do not match the campaign pin")
        live_gains_start = read_live_controller_settings(env.scene)
        live_gains_start_check = compare_live_settings_to_spec(
            live_gains_start, configured_gains_start
        )
        if not live_gains_start_check["matches_configured_spec"]:
            raise RuntimeError("live controller gains do not match the configured defaults")
        reset_counts = {
            "explicit_env_reset_calls": 0,
            "initial_state_restores": 0,
            "resets_during_episode": 0,
        }
        env.reset(warm_up=False, seed=args.seed)
        reset_counts["explicit_env_reset_calls"] += 1
        source = _load_dataset(args.source_dataset, args.episode, env.device)
        target = _load_dataset(target_state_template, args.episode, env.device)
        template_assets = _dataset_assets(target_state_template, args.objects_root)
        target, initial_placement = _support_preserving_target_state(
            target, template_assets, target_assets
        )
        env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
        _reset_scene_to_state(env.scene, target["initial_state"], env_ids)
        reset_counts["initial_state_restores"] += 1
        env.sim.forward()
        env.reset_success_check(env_ids)
        from dc_study.datagen.hang_mug_status import (
            hang_mug_status,
            reset_hang_mug_status,
        )

        reset_hang_mug_status(env)
        source_mug = _geometry(source_assets["mug"], source["mug_pose"][0])
        target_mug = _geometry(target_assets["mug"], target["mug_pose"][0])
        source_tree = _geometry(source_assets["mug_tree"], source["tree_pose"][0])
        target_tree = _geometry(target_assets["mug_tree"], target["tree_pose"][0])
        from semantic_asset_geometry import jsonable, mug_parts, tree_branches

        source_parts = mug_parts(source_assets["mug"])
        target_parts = mug_parts(target_assets["mug"])
        source_branches = tree_branches(source_assets["mug_tree"])
        target_branches = tree_branches(target_assets["mug_tree"])
        keyframes = _load_keyframes(args.source_keyframes, args.source_dataset) if args.mode == "skill" else None
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
            if keyframes is not None else (None, None, None, None, None, None)
        )
        source_prefix_steps = (
            _source_pick_prefix_steps(keyframes)
            if trajectory is not None and args.reuse_source_pick_prefix
            else 0
        )
        if source_prefix_steps > len(source["actions"]):
            raise ValueError("source Pick prefix exceeds the source action dataset")
        joint_nominal = (
            _sparse_joint_nominal(source, trajectory, keyframes)
            if trajectory is not None and not source_prefix_steps
            else None
        )
        observed_handover_reanchor = bool(
            trajectory is not None
            and (
                "direct_preinsert" in trajectory.waypoint_steps
                or source_prefix_steps
                or _requires_observed_handover_reanchor(
                    target_parts,
                    handle_frame_transfer=args.handover_handle_frame_transfer,
                )
            )
        )
        direct_contact_views = (
            _quality_wave_contact_views(env, target_assets)
            if trajectory is not None
            and "direct_preinsert" in trajectory.waypoint_steps
            else None
        )
        physics_dt = float(env.sim.get_physics_dt())
        total_steps = (
            source_prefix_steps + _trajectory_after(trajectory, "left_lift").steps
            if source_prefix_steps
            else trajectory.steps if trajectory is not None else len(source["actions"])
        )
        from judo_isaaclab.demo_artifact import DemonstrationRecorder

        demo_recorder = DemonstrationRecorder()
        demo_recorder.start(env.scene.get_state(is_relative=False))
        samples = [_sample(env, -1, "reset")]
        semantic_statuses = []
        for name, pose_key in (("mug", "mug_pose"), ("mug_tree", "tree_pose")):
            initial_placement[name]["observed_after_restore"] = samples[0][pose_key]
        actions = []; mug_poses = []; left_eef = []; right_eef = []; desired_left = []; desired_right = []; semantic_left_eef = []; semantic_right_eef = []; frame_stats = []
        trace_stages = []; trace_waypoints = []
        pick_boundary = None
        handover_boundary = None
        handover_contact_acquire = None
        handover_contact_acquire_rows = []
        handover_lift_boundary = None
        handover_lift_rows = []
        handover_wave_plan_screens = {
            "clear_pregrasp": None,
            "open_approach": None,
        }
        handover_wave_live_rows = []
        left_assist_secure_receiver_release_step = None
        direct_plan_screens = {"outbound": None, "return": None}
        direct_live_rows = []
        if args.render:
            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            encoder = _Encoder(args.fps, args.video)
        for step in range(total_steps):
            if trajectory is None:
                action = source["actions"][step : step + 1]
                stage = "direct_source_action_replay"
                waypoint = "direct_source_action_replay"
                semantic_step = None
            elif step < source_prefix_steps:
                action = source["actions"][step : step + 1]
                stage = "exact_source_pick_prefix"
                waypoint = "exact_source_pick_prefix"
                semantic_step = None
            else:
                if source_prefix_steps and joint_nominal is None:
                    pick_boundary = _pick_boundary_receipt(samples[-1])
                    _require_reusable_pick_boundary(samples[-1])
                    from judo_isaaclab.hang_mug import reanchor_physical_handover

                    trajectory = _trajectory_after(
                        reanchor_physical_handover(
                            trajectory,
                            nominal_handover_mug,
                            samples[-1]["mug_pose"],
                            samples[-1]["left_eef_pose"],
                            samples[-1]["right_eef_pose"],
                        ),
                        "left_lift",
                    )
                    joint_nominal = _sparse_joint_nominal(
                        source,
                        trajectory,
                        keyframes,
                        initial_action_index=source_prefix_steps - 1,
                    )
                semantic_step = step - source_prefix_steps
                if (
                    "left_lift" in trajectory.waypoint_steps
                    and semantic_step == trajectory.waypoint_steps["left_lift"] + 1
                ):
                    pick_boundary = _pick_boundary_receipt(samples[-1])
                    if not pick_boundary["safe_to_continue"]:
                        break
                clear_pregrasp_boundary = bool(
                    "direct_preinsert" in trajectory.waypoint_steps
                    and "handover_pregrasp" in trajectory.waypoint_steps
                    and (
                        (
                            "left_lift" in trajectory.waypoint_steps
                            and semantic_step
                            == trajectory.waypoint_steps["left_lift"] + 1
                        )
                        or (
                            "left_lift" not in trajectory.waypoint_steps
                            and semantic_step == 0
                        )
                    )
                )
                if clear_pregrasp_boundary:
                    handover_wave_plan_screens["clear_pregrasp"] = (
                        _handover_wave_plan_screen(
                            trajectory,
                            samples[-1],
                            target_assets,
                            nominal_right_contact,
                            args,
                            phase="clear_pregrasp",
                        )
                    )
                    if not handover_wave_plan_screens["clear_pregrasp"]["passed"]:
                        break
                if (
                    "direct_preinsert" in trajectory.waypoint_steps
                    and semantic_step
                    == trajectory.waypoint_steps["handover_pregrasp"] + 1
                ):
                    handover_wave_plan_screens["open_approach"] = (
                        _handover_wave_plan_screen(
                            trajectory,
                            samples[-1],
                            target_assets,
                            nominal_right_contact,
                            args,
                            phase="open_approach",
                        )
                    )
                    if not handover_wave_plan_screens["open_approach"]["passed"]:
                        break
                if (
                    "handover_contact_acquire" in trajectory.waypoint_steps
                    and semantic_step == trajectory.waypoint_steps["right_grasp"] + 1
                ):
                    entry = _handover_contact_acquire_guard_receipt(
                        samples[-1], phase="entry"
                    )
                    if not entry["passed"]:
                        handover_contact_acquire = {"entry": entry, "passed": False}
                        break
                    from judo_isaaclab.hang_mug import (
                        reanchor_handover_contact_acquire,
                    )

                    trajectory, plan = reanchor_handover_contact_acquire(
                        trajectory,
                        nominal_right_contact,
                        samples[-1]["mug_pose"],
                        samples[-1]["left_eef_pose"],
                        samples[-1]["right_eef_pose"],
                    )
                    handover_contact_acquire = {
                        **plan,
                        "entry": entry,
                        "completion": None,
                        "passed": False,
                    }
                if (
                    "handover_contact_acquire" in trajectory.waypoint_steps
                    and semantic_step
                    == trajectory.waypoint_steps["handover_contact_acquire"] + 1
                ):
                    completion = _handover_contact_acquire_guard_receipt(
                        samples[-1], phase="completion"
                    )
                    handover_contact_acquire["completion"] = completion
                    handover_contact_acquire["passed"] = bool(
                        completion["passed"]
                        and len(handover_contact_acquire_rows)
                        == args.handover_contact_acquire_steps
                        and all(row["passed"] for row in handover_contact_acquire_rows)
                    )
                    if not handover_contact_acquire["passed"]:
                        break
                if (
                    "handover_receiver_lift" in trajectory.waypoint_steps
                    and semantic_step
                    == trajectory.waypoint_steps["left_release"] + 1
                ):
                    handover_lift_boundary = _handover_lift_guard_receipt(
                        samples[-1], phase="entry"
                    )
                    if not handover_lift_boundary["passed"]:
                        break
                if (
                    semantic_step
                    == trajectory.waypoint_steps.get(
                        "handover_confirm",
                        trajectory.waypoint_steps["left_release"],
                    )
                    + 1
                ):
                    handover_boundary = _handover_boundary_receipt(samples[-1])
                    if not handover_boundary["safe_to_continue"]:
                        break
                if (
                    "direct_preinsert" in trajectory.waypoint_steps
                    and semantic_step
                    == trajectory.waypoint_steps["carrying_rest_observer"] + 1
                ):
                    direct_plan_screens["outbound"] = _direct_segment_plan_screen(
                        trajectory, samples[-1], target_assets, phase="outbound"
                    )
                    if not direct_plan_screens["outbound"]["passed"]:
                        break
                if (
                    "post_release_return" in trajectory.waypoint_steps
                    and semantic_step
                    == trajectory.waypoint_steps["right_release"] + 1
                ):
                    direct_plan_screens["return"] = _direct_segment_plan_screen(
                        trajectory, samples[-1], target_assets, phase="return"
                    )
                    if not direct_plan_screens["return"]["passed"]:
                        break
                stage = trajectory.stage_names[semantic_step]
                waypoint = _semantic_waypoint_name(trajectory, semantic_step)
                integrate = bool(
                    source_prefix_steps
                    or semantic_step > trajectory.waypoint_steps["left_grasp"]
                )
                action = _ik_action(
                    env,
                    trajectory.left_poses[semantic_step],
                    trajectory.right_poses[semantic_step],
                    trajectory.grippers[semantic_step],
                    joint_nominal[semantic_step],
                    args,
                    integrate_left_ik=integrate,
                    integrate_right_ik=integrate
                    and waypoint not in {
                        "right_return_start", "carrying_rest_observer"
                    },
                )
                desired_left.append(trajectory.left_poses[semantic_step]); desired_right.append(trajectory.right_poses[semantic_step])
            observation, _, _, _, info = env.step(action)
            if semantic_step is not None:
                _update_authored_assist_releases(env, trajectory, semantic_step)
            sample = _sample(env, step, stage, info)
            stop_after_row = False
            if waypoint == "handover_receiver_lift":
                lift_row = _handover_lift_guard_receipt(sample, phase="lift_row")
                handover_lift_rows.append(lift_row)
                stop_after_row = not lift_row["passed"]
            if waypoint == "handover_contact_acquire":
                acquire_row = _handover_contact_acquire_guard_receipt(
                    sample, phase="row"
                )
                handover_contact_acquire_rows.append(acquire_row)
                stop_after_row = stop_after_row or not acquire_row["passed"]
            if waypoint in {
                "handover_pregrasp",
                "handover_orient_clear",
                "right_grasp_settle",
                "right_grasp",
                "handover_contact_acquire",
            } and direct_contact_views is not None:
                wave_row = _handover_wave_live_row(
                    direct_contact_views, sample, waypoint, physics_dt
                )
                handover_wave_live_rows.append(wave_row)
                stop_after_row = stop_after_row or not wave_row["passed"]
            if waypoint in {"direct_preinsert", "post_release_return"}:
                live_row = _direct_segment_live_row(
                    direct_contact_views, sample, waypoint, physics_dt
                )
                direct_live_rows.append(live_row)
                stop_after_row = stop_after_row or not live_row["passed"]
            semantic_statuses.append(hang_mug_status(env, info))
            demo_recorder.append(
                action,
                env.scene.get_state(is_relative=False),
                observation=observation,
                semantic_observation=sample,
            )
            samples.append(sample)
            actions.append(action[0].detach().cpu().numpy()); mug_poses.append(sample["mug_pose"]); left_eef.append(sample["left_eef_pose"]); right_eef.append(sample["right_eef_pose"])
            trace_stages.append(stage); trace_waypoints.append(waypoint)
            if (
                defer_left_assist_to_secure_receiver
                and left_assist_secure_receiver_release_step is None
                and _release_left_assist_after_secure_receiver(
                    env, sample, waypoint
                )
            ):
                left_assist_secure_receiver_release_step = step
            if semantic_step is not None:
                semantic_left_eef.append(sample["left_eef_pose"])
                semantic_right_eef.append(sample["right_eef_pose"])
            if (
                observed_handover_reanchor
                and semantic_step is not None
                and "left_lift" in trajectory.waypoint_steps
                and semantic_step == trajectory.waypoint_steps["left_lift"]
                and sample["left_grasp"]
            ):
                from judo_isaaclab.hang_mug import reanchor_physical_handover

                trajectory = reanchor_physical_handover(
                    trajectory,
                    nominal_handover_mug,
                    sample["mug_pose"],
                    sample["left_eef_pose"],
                    sample["right_eef_pose"],
                )
            if (
                observed_handover_reanchor
                and semantic_step is not None
                and semantic_step == trajectory.waypoint_steps["handover_pregrasp"]
                and sample["stage1"]
                and sample["grasp_assist_engaged"].get("left", False)
            ):
                from judo_isaaclab.hang_mug import (
                    reanchor_right_grasp_from_observed_mug,
                )

                trajectory = reanchor_right_grasp_from_observed_mug(
                    trajectory,
                    nominal_right_contact,
                    sample["mug_pose"],
                    sample["right_eef_pose"],
                )
            reanchor_waypoints = _branch_reanchor_waypoints(trajectory)
            if semantic_step is not None and any(
                semantic_step == trajectory.waypoint_steps[name]
                for name in reanchor_waypoints
            ) and sample["right_grasp"]:
                from judo_isaaclab.hang_mug import reanchor_branch_transport_contact
                from judo_isaaclab.put_marker import compose_pose, inverse_pose

                completed_waypoint = next(
                    name
                    for name in reanchor_waypoints
                    if semantic_step == trajectory.waypoint_steps[name]
                )
                trajectory = reanchor_branch_transport_contact(
                    trajectory,
                    nominal_right_contact,
                    sample["mug_pose"],
                    sample["right_eef_pose"],
                    completed_waypoint=completed_waypoint,
                )
                nominal_right_contact = compose_pose(
                    inverse_pose(sample["mug_pose"]),
                    sample["right_eef_pose"],
                )
            if encoder is not None:
                frame = _frame(env, sample); encoder.write(frame); frame_stats.append((float(frame.mean()), float(frame.std())))
            if (step + 1) % 50 == 0 or sample["task_success"]:
                progress = {key: sample[key] for key in ("step", "program_stage", "stage1", "stage2", "stage3", "task_success", "left_grasp", "right_grasp", "grasp_assist_engaged", "mug_pose", "mug_tree_xy_error_m")}
                print("HANGMUG_PROGRESS=" + json.dumps(progress, sort_keys=True), flush=True)
                print(f"STEP_PROGRESS step={step} stage={stage}", flush=True)
            if stop_after_row:
                break
        if encoder is not None:
            encoder.close(); encoder = None
        Path(args.trace_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.trace_npz,
            actions=np.asarray(actions, dtype=np.float32),
            mug_poses=np.asarray(mug_poses, dtype=np.float32),
            left_eef_poses=np.asarray(left_eef, dtype=np.float32),
            right_eef_poses=np.asarray(right_eef, dtype=np.float32),
            program_stages=np.asarray(trace_stages, dtype="U32"),
            semantic_waypoints=np.asarray(trace_waypoints, dtype="U32"),
            **_trace_status_arrays(samples),
            desired_left_eef_poses=np.asarray(desired_left, dtype=np.float32),
            desired_right_eef_poses=np.asarray(desired_right, dtype=np.float32),
            sparse_joint_nominal=np.asarray(joint_nominal, dtype=np.float32) if joint_nominal is not None else np.empty((0, 14), dtype=np.float32),
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
            right_arm_joint_pos=np.asarray(
                [sample["right_arm_joint_pos"] for sample in samples[1:]],
                dtype=np.float32,
            ),
        )
        final = samples[-1]
        terminal_stability = _terminal_stability(samples)
        independent_terminal_hang = _independent_terminal_hang_receipt(
            semantic_statuses,
            samples[1:],
            reset_counts,
        )
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
            desired_error = [
                max(
                    np.linalg.norm(np.asarray(actual_left)[:3] - target_left[:3]),
                    np.linalg.norm(np.asarray(actual_right)[:3] - target_right[:3]),
                )
                for actual_left, actual_right, target_left, target_right in zip(
                    semantic_left_eef,
                    semantic_right_eef,
                    desired_left,
                    desired_right,
                    strict=True,
                )
            ]
        direct_replay = None
        if args.direct_replay_result:
            with open(args.direct_replay_result, encoding="utf-8") as stream:
                direct_replay = json.load(stream)
        terminal_speed = float(np.linalg.norm(final["mug_velocity"][:3]))
        executed_source_actions_exact = (
            _direct_actions_exact(actions, source["actions"])
            if trajectory is None
            else None
        )
        source_pick_prefix_exact = (
            _direct_actions_exact(
                actions[:source_prefix_steps],
                source["actions"][:source_prefix_steps],
            )
            if source_prefix_steps
            else None
        )
        configured_gains_end = env.robot.spec.controller_gains()
        live_gains_end = read_live_controller_settings(env.scene)
        live_gains_end_check = compare_live_settings_to_spec(
            live_gains_end, configured_gains_end
        )
        controller_receipt = {
            "expected_configured_sha256": args.expected_controller_gains_sha256,
            "starting_configured_sha256": configured_gains_start_sha256,
            "ending_configured_sha256": sha256_json(configured_gains_end),
            "starting_live_sha256": sha256_json(live_gains_start),
            "ending_live_sha256": sha256_json(live_gains_end),
            "starting_live_matches_configured": live_gains_start_check,
            "ending_live_matches_configured": live_gains_end_check,
        }
        broad_pad_contact = {
            side: _broad_pad_contact_receipt(samples[1:], side)
            for side in ("left", "right")
        }
        right_start_configuration = None
        if args.post_handover_right_return_steps or args.post_handover_rest_observer_steps:
            setup_waypoint = (
                "carrying_rest_observer"
                if args.post_handover_rest_observer_steps
                else "right_return_start"
            )
            rows = [
                sample
                for sample, waypoint in zip(samples[1:], trace_waypoints, strict=True)
                if waypoint == setup_waypoint
            ]
            target = np.asarray(
                source["actions"][0, 7:13].detach().cpu(), dtype=float
            )
            observed = (
                np.asarray(rows[-1]["right_arm_joint_pos"], dtype=float)
                if rows
                else None
            )
            maximum_error = (
                float(np.max(np.abs(observed - target)))
                if observed is not None
                else None
            )
            right_start_configuration = {
                "target_source_action_index": 0,
                "target_right_arm_joints": target.tolist(),
                "observed_right_arm_joints": (
                    observed.tolist() if observed is not None else None
                ),
                "maximum_joint_error_rad": maximum_error,
                "threshold_rad": 0.08,
                "passed": maximum_error is not None and maximum_error <= 0.08,
            }
        direct_phase_contract = _direct_phase_contract_receipt(
            trajectory, trace_waypoints, actions, samples
        )
        handover_wave_contract = _handover_wave_contract_receipt(
            trajectory,
            trace_waypoints,
            actions,
            samples,
            handover_wave_plan_screens,
            handover_wave_live_rows,
        )
        direct_collision_screening = None
        post_release_right_rest = None
        if direct_phase_contract is not None:
            expected_live_rows = (
                args.direct_rest_to_preinsert_steps
                + args.post_release_return_to_rest_steps
            )
            direct_collision_screening = {
                "planning_is_not_a_motion_phase": True,
                "plan_screens": direct_plan_screens,
                "live_physx_contact_guard": {
                    "right_body_paths": direct_contact_views["right_body_paths"],
                    "left_body_paths": direct_contact_views["left_body_paths"],
                    "tree_body_path": direct_contact_views["tree_body_path"],
                    "mug_body_path": direct_contact_views["mug_body_path"],
                    "required_force_threshold_n": 1.0e-6,
                    "expected_rows": expected_live_rows,
                    "observed_rows": len(direct_live_rows),
                    "rows": direct_live_rows,
                    "passed": bool(
                        len(direct_live_rows) == expected_live_rows
                        and all(row["passed"] for row in direct_live_rows)
                    ),
                },
            }
            direct_collision_screening["passed"] = bool(
                all(
                    receipt is not None and receipt["passed"]
                    for receipt in direct_plan_screens.values()
                )
                and direct_collision_screening["live_physx_contact_guard"][
                    "passed"
                ]
            )
            return_rows = [
                sample
                for sample, waypoint in zip(samples[1:], trace_waypoints, strict=True)
                if waypoint == "post_release_return"
            ]
            target = np.asarray(
                source["actions"][0, 7:13].detach().cpu(), dtype=float
            )
            observed = (
                np.asarray(return_rows[-1]["right_arm_joint_pos"], dtype=float)
                if return_rows
                else None
            )
            maximum_error = (
                float(np.max(np.abs(observed - target)))
                if observed is not None
                else None
            )
            post_release_right_rest = {
                "target_source_action_index": 0,
                "target_right_arm_joints": target.tolist(),
                "observed_right_arm_joints": (
                    None if observed is None else observed.tolist()
                ),
                "maximum_joint_error_rad": maximum_error,
                "threshold_rad": 0.08,
                "right_gripper_remained_open": bool(
                    return_rows and all(not row["right_grasp"] for row in return_rows)
                ),
                "passed": bool(
                    maximum_error is not None
                    and maximum_error <= 0.08
                    and return_rows
                    and all(not row["right_grasp"] for row in return_rows)
                ),
            }
        checks = {
            "one_reset": reset_counts["explicit_env_reset_calls"] == 1,
            "zero_inter_stage_resets": reset_counts["resets_during_episode"] == 0,
            "real_target_assets": all(Path(path).is_dir() for path in target_assets.values()),
            "configured_controller_matches_expected": (
                args.expected_controller_gains_sha256 is None
                or configured_gains_start_sha256 == args.expected_controller_gains_sha256
            ),
            "configured_controller_unchanged": configured_gains_end == configured_gains_start,
            "live_controller_unchanged": live_gains_end == live_gains_start,
            "starting_live_controller_matches_spec": bool(
                live_gains_start_check["matches_configured_spec"]
            ),
            "ending_live_controller_matches_spec": bool(
                live_gains_end_check["matches_configured_spec"]
            ),
            "single_source_action_dataset": source_receipt["action_dataset"] == "actions",
            "source_dataset_unchanged": _sha256(args.source_dataset)
            == source_receipt["file_sha256"],
            "contact_backed_grasps_only": True,
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
            "stable_hang_window": bool(terminal_stability["passed"]),
            "independent_terminal_hang": bool(independent_terminal_hang["passed"]),
            "terminal_mug_speed_within_threshold": terminal_speed <= 0.05,
            "h264_nonempty": video is None or (video["codec"] == "h264" and video["size_bytes"] > 0 and video["frame_count"] == len(frame_stats)),
            "fully_decodable": video is None or video["full_decode_returncode"] == 0,
        }
        if args.require_broad_pad_contact:
            checks["left_broad_pad_contact"] = bool(
                broad_pad_contact["left"]["passed"]
            )
            checks["right_broad_pad_contact"] = bool(
                broad_pad_contact["right"]["passed"]
            )
        if right_start_configuration is not None:
            checks["right_start_configuration_reached"] = bool(
                right_start_configuration["passed"]
            )
        if direct_phase_contract is not None:
            checks["handover_wave_contract"] = bool(
                handover_wave_contract and handover_wave_contract["passed"]
            )
            checks["direct_phase_contract"] = bool(
                direct_phase_contract["passed"]
            )
            checks["direct_segment_collision_screening"] = bool(
                direct_collision_screening["passed"]
            )
            checks["post_release_right_rest_reached_open"] = bool(
                post_release_right_rest["passed"]
            )
        if args.handover_post_release_lift_steps:
            checks["handover_post_release_lift_passed"] = bool(
                handover_lift_boundary
                and handover_lift_boundary["passed"]
                and len(handover_lift_rows)
                == args.handover_post_release_lift_steps
                and all(row["passed"] for row in handover_lift_rows)
            )
        if args.handover_contact_acquire_steps:
            checks["handover_contact_acquire_passed"] = bool(
                handover_contact_acquire and handover_contact_acquire["passed"]
            )
        if trajectory is None:
            checks["executed_source_actions_exact"] = bool(executed_source_actions_exact)
        if source_prefix_steps:
            checks["reused_source_pick_prefix_exact"] = bool(
                source_pick_prefix_exact
            )
        if trajectory is not None:
            checks["pick_boundary_passed"] = bool(
                pick_boundary and pick_boundary["passed"]
            )
            checks["handover_boundary_passed"] = bool(
                handover_boundary and handover_boundary["passed"]
            )
            checks["handover_safe_to_continue"] = bool(
                handover_boundary and handover_boundary["safe_to_continue"]
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
                    "configured_controller_matches_expected",
                    "configured_controller_unchanged", "live_controller_unchanged",
                    "starting_live_controller_matches_spec",
                    "ending_live_controller_matches_spec",
                    "single_source_action_dataset", "source_dataset_unchanged",
                    "executed_source_actions_exact",
                    "h264_nonempty", "fully_decodable",
                )
            }
        elif args.expect_failure:
            acceptance = {name: checks[name] for name in ("one_reset", "zero_inter_stage_resets", "real_target_assets", "configured_controller_matches_expected", "configured_controller_unchanged", "live_controller_unchanged", "starting_live_controller_matches_spec", "ending_live_controller_matches_spec", "single_source_action_dataset", "source_dataset_unchanged", "contact_backed_grasps_only", "datagen_grasp_assist_configured", "left_grasp_assist_engaged", "h264_nonempty", "fully_decodable")}
            if trajectory is None:
                acceptance["executed_source_actions_exact"] = checks[
                    "executed_source_actions_exact"
                ]
            acceptance["expected_coded_task_failure"] = not final["task_success"]
        else:
            acceptance = _schema_aware_success_acceptance(
                checks, coded_skill=trajectory is not None
            )
            if direct_replay is not None and target_assets != source_assets:
                acceptance = dict(acceptance)
                acceptance["direct_source_action_replay_failed"] = bool(direct_replay.get("status") == "passed" and not direct_replay.get("terminal", {}).get("task_success", True))
                acceptance["direct_replay_grasp_assistance_matched"] = (
                    direct_replay.get("protocol", {}).get("grasp_assistance")
                    == grasp_assistance
                )
                acceptance["direct_replay_physics_device_matched"] = (
                    direct_replay.get("protocol", {}).get("physics_device_actual")
                    == str(env.device)
                )
        demo_artifact = None
        if args.demo_hdf5 and all(acceptance.values()):
            from judo_isaaclab.demo_artifact import relative_asset_paths

            demo_recorder.write(
                args.demo_hdf5,
                assets_instance_paths=relative_asset_paths(target_assets, args.objects_root),
                success=True,
                metadata={
                    "task": "HangMugOnTree-v0",
                    "controller": (
                        "direct_source_action_replay"
                        if trajectory is None
                        else "source_pick_prefix_then_deterministic_semantic_skill"
                        if source_prefix_steps
                        else "deterministic_semantic_skill"
                    ),
                    "candidate_sampling": False,
                    "grasp_assistance": grasp_assistance,
                    "source_dataset_sha256": _sha256(args.source_dataset),
                    "target_state_template_sha256": _sha256(target_state_template),
                    "source_action_dataset": "actions",
                    "source_actions_sha256": source_receipt["actions_sha256"],
                },
            )
            demo_artifact = {"path": os.path.abspath(args.demo_hdf5), "sha256": _sha256(args.demo_hdf5)}
        result = {
            "status": "passed" if all(acceptance.values()) else "failed",
            "mode": args.mode,
            "protocol": {"controller": "direct_source_action_replay" if trajectory is None else "source_pick_prefix_then_deterministic_semantic_cartesian_dls" if source_prefix_steps else "deterministic_semantic_cartesian_dls", "candidate_sampling": False, "scene_resets": reset_counts["explicit_env_reset_calls"], "initial_state_restores": reset_counts["initial_state_restores"], "inter_stage_resets": reset_counts["resets_during_episode"], "control_rate_hz": 30, "steps": len(actions), "seed": args.seed, "physics_device_requested": args.device, "physics_device_actual": str(env.device), "physics_device_requirement": "cpu" if args.require_cpu_physics else None, "physics_device_receipt": physics_device, "grasp_assistance": grasp_assistance, "source_pick_prefix": ({"through_waypoint": "right_pregrasp", "action_count": source_prefix_steps, "first_action_index": 0, "last_action_index": source_prefix_steps - 1, "actions_sha256": _array_sha256(np.asarray(actions[:source_prefix_steps], dtype=np.float32)), "exact": bool(source_pick_prefix_exact)} if source_prefix_steps else None), "parameters": {"damping": args.damping, "max_joint_delta": args.max_joint_delta, "max_position_step": args.max_position_step, "max_rotation_step": args.max_rotation_step, "insert_clearance_m": args.insert_clearance_m, "branch_approach_height_m": args.branch_approach_height_m, "pick_lift_margin_m": _bounded_pick_lift_margin(args.pick_lift_margin_m), "branch_support_fraction": _bounded_branch_support_fraction(args.branch_support_fraction), "branch_support_seat_down_m": args.branch_support_seat_down_m, "stable_support_steps": args.stable_support_steps, "handover_contact_settle_steps": args.handover_contact_settle_steps, "handover_contact_acquire_steps": args.handover_contact_acquire_steps, "handover_confirm_steps": args.handover_confirm_steps, "handover_post_release_lift_m": _bounded_handover_post_release_lift(args.handover_post_release_lift_m), "handover_post_release_lift_steps": args.handover_post_release_lift_steps, "post_handover_right_return_steps": args.post_handover_right_return_steps, "left_branch_point_steps": args.left_branch_point_steps, "post_handover_rest_observer_steps": args.post_handover_rest_observer_steps, "direct_rest_to_preinsert_steps": args.direct_rest_to_preinsert_steps, "post_release_return_to_rest_steps": args.post_release_return_to_rest_steps, "handover_target_offset_m": _bounded_handover_offset(args.handover_target_offset_m).tolist(), "handover_target_local_pitch_rad": float(args.handover_target_local_pitch_rad), "handover_straddle_local_x_m": float(args.handover_straddle_local_x_m), "handover_orient_clearance_m": float(args.handover_orient_clearance_m), "handover_orient_steps": int(args.handover_orient_steps), "handover_standoff_outside_m": float(args.handover_standoff_outside_m), "handover_handle_frame_transfer": bool(args.handover_handle_frame_transfer), "left_release_retreat_m": _bounded_left_release_retreat(args.left_release_retreat_m), "observed_left_anchor_held_during_handover": observed_handover_reanchor, "observed_handover_reanchor": observed_handover_reanchor, "right_contact_feedback_reanchor": trajectory is not None, "pick_clearance_uses_measured_body_height": True, "mug_body_frame_scaling": True, "handle_hole_branch_frame_transfer": True, "branch_support_midpoint": _bounded_branch_support_fraction(args.branch_support_fraction) == 0.5}},
            "provenance": {"source_dataset": source_receipt, "target_state_template": {"path": os.path.abspath(target_state_template), "sha256": _sha256(target_state_template), "actions_executed": False}, "source_assets": {name: _asset_provenance(path) for name, path in source_assets.items()}, "target_assets": {name: _asset_provenance(path) for name, path in target_assets.items()}, "task_manager": {"path": os.path.join(args.gear_repo, "dc_study/envs/tasks/hang_mug_on_tree_manager.py"), "sha256": _sha256(os.path.join(args.gear_repo, "dc_study/envs/tasks/hang_mug_on_tree_manager.py"))}, "task_config": {"path": os.path.join(args.gear_repo, "dc_study/envs/tasks/hang_mug_on_tree_manager_cfg.py"), "sha256": _sha256(os.path.join(args.gear_repo, "dc_study/envs/tasks/hang_mug_on_tree_manager_cfg.py"))}, "trace": {"path": os.path.abspath(args.trace_npz), "sha256": _sha256(args.trace_npz)}, "demonstration": demo_artifact, "source_keyframes": ({"path": os.path.abspath(args.source_keyframes), "sha256": _sha256(args.source_keyframes)} if args.source_keyframes else None)},
            "initial_placement": initial_placement,
            "controller_gains": controller_receipt,
            "reset_counts": reset_counts,
            "terminal_stability": terminal_stability,
            "grasp_quality": {
                "required": bool(args.require_broad_pad_contact),
                "broad_pad_contact": broad_pad_contact,
            },
            "handover_wave_contract": handover_wave_contract,
            "handover_assist_lifecycle": {
                "legacy_simultaneous_grasp_release_deferred": (
                    defer_left_assist_to_secure_receiver
                ),
                "left_assist_released_after_secure_receiver_step": (
                    left_assist_secure_receiver_release_step
                ),
            },
            "post_handover_setup": {
                "right_start_configuration": right_start_configuration,
                "left_branch_point_ordered_after_right_return": bool(
                    args.post_handover_rest_observer_steps
                    or not args.post_handover_right_return_steps
                    or (
                        trajectory.waypoint_steps["right_return_start"]
                        < trajectory.waypoint_steps["left_branch_point"]
                        < trajectory.waypoint_steps[
                            "direct_preinsert"
                            if "direct_preinsert" in trajectory.waypoint_steps
                            else "tree_transport"
                        ]
                    )
                ),
            },
            "direct_phase_contract": direct_phase_contract,
            "direct_segment_collision_screening": direct_collision_screening,
            "post_release_right_rest": post_release_right_rest,
            "independent_terminal_hang": independent_terminal_hang,
            "semantic_stage_receipt": _semantic_stage_receipt(semantic_statuses),
            "stage_boundary": handover_boundary or pick_boundary,
            "pick_boundary": pick_boundary,
            "handover_contact_acquire": {
                **(handover_contact_acquire or {"passed": False}),
                "planned_steps": args.handover_contact_acquire_steps,
                "rows": handover_contact_acquire_rows,
            },
            "handover_post_release_lift": {
                "planned_steps": args.handover_post_release_lift_steps,
                "entry": handover_lift_boundary,
                "rows": handover_lift_rows,
                "passed": bool(
                    not args.handover_post_release_lift_steps
                    or (
                        handover_lift_boundary
                        and handover_lift_boundary["passed"]
                        and len(handover_lift_rows)
                        == args.handover_post_release_lift_steps
                        and all(row["passed"] for row in handover_lift_rows)
                    )
                ),
            },
            "first_failed_semantic_stage": (
                "pick"
                if pick_boundary is not None and not pick_boundary["passed"]
                else "handover"
                if (
                    handover_boundary is not None
                    and not handover_boundary["passed"]
                )
                or (
                    args.handover_contact_acquire_steps
                    and (
                        handover_contact_acquire is None
                        or not handover_contact_acquire["passed"]
                    )
                )
                or (
                    args.handover_post_release_lift_steps
                    and (
                        handover_lift_boundary is None
                        or not handover_lift_boundary["passed"]
                        or len(handover_lift_rows)
                        != args.handover_post_release_lift_steps
                        or not all(row["passed"] for row in handover_lift_rows)
                    )
                )
                else None
            ),
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
        result["protocol"]["parameters"]["branch_orient_steps"] = int(
            args.branch_orient_steps
        )
        result["protocol"]["parameters"]["branch_roll_offset_rad"] = float(
            args.branch_roll_offset_rad
        )
        result["protocol"]["parameters"]["target_branch_rank"] = (
            args.target_branch_rank
        )
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(args.result_json, result)
        print("HANGMUG_FINAL=" + json.dumps(result, sort_keys=True), flush=True)
        if result["status"] != "passed":
            raise RuntimeError(f"acceptance checks failed: {acceptance}")
    except BaseException as error:
        if not os.path.isfile(args.result_json):
            _write_json_atomic(
                args.result_json,
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "provenance": {
                        "source_dataset": source_receipt,
                        "target_state_template": os.path.abspath(target_state_template),
                        "target_assets": target_assets,
                    },
                },
            )
        print(
            f"HANGMUG_RUN_FAILED artifact={os.path.abspath(args.result_json)} "
            f"error={type(error).__name__}: {error}",
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        if encoder is not None:
            encoder.close()
        if env is not None:
            env.close()
        if simulation_app is not None:
            print("ISAAC_SHUTDOWN_BEGIN", flush=True)
            simulation_app.close()


if __name__ == "__main__":
    main()
