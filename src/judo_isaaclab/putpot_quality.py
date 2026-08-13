"""Opt-in quality audits and lane guards for parallel PutPot generation.

This module is intentionally simulator-independent and is not imported by the
legacy PutPot runner.  A quality-wave launcher must explicitly load the quality
configuration and supply measured traces to these fail-closed audits.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
MODE = "putpot_quality_wave"
_LANE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _pair(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((str(first), str(second))))


@dataclass(frozen=True)
class PutPotQualityConfig:
    """Validated opt-in PutPot quality contract."""

    path: Path
    sha256: str
    grasp: Mapping[str, Any]
    motion: Mapping[str, Any]
    perturbation: Mapping[str, Any]
    collision: Mapping[str, Any]
    lane: Mapping[str, Any]

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": MODE,
            "path": str(self.path.resolve()),
            "sha256": self.sha256,
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_quality_config(path: str | Path) -> PutPotQualityConfig:
    """Load the explicit quality mode without changing legacy behavior."""

    source = Path(path)
    raw = source.read_bytes()
    value = json.loads(raw)
    expected = {
        "schema_version",
        "mode",
        "grasp",
        "motion",
        "perturbation",
        "collision",
        "lane",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("quality config has unexpected top-level keys")
    if value["schema_version"] != SCHEMA_VERSION or value["mode"] != MODE:
        raise ValueError("unsupported PutPot quality config")

    grasp = value["grasp"]
    if grasp.get("order") != ["left", "right"]:
        raise ValueError("quality grasp order must be left then right")
    if int(grasp.get("stable_steps", 0)) != 15:
        raise ValueError("quality grasp requires a 15-frame four-pad latch")
    for name in (
        "minimum_force_n",
        "minimum_pad_fraction_margin",
        "minimum_contact_area_fraction",
        "maximum_flush_angle_deg",
        "maximum_pre_latch_object_motion_m",
    ):
        if not np.isfinite(float(grasp.get(name, np.nan))):
            raise ValueError(f"grasp.{name} must be finite")

    motion = value["motion"]
    if motion.get("planner_mode") != "object_first_rigid_weld":
        raise ValueError("quality motion must be object-first rigid-weld motion")
    perturbation = value["perturbation"]
    if int(perturbation.get("case_count", 0)) < 1:
        raise ValueError("perturbation.case_count must be positive")
    required_fraction = float(perturbation.get("required_pass_fraction", np.nan))
    if not 0.0 <= required_fraction <= 1.0:
        raise ValueError("perturbation.required_pass_fraction must be in [0, 1]")

    collision = value["collision"]
    required = tuple(collision.get("required_components", ()))
    expected_components = {
        "left_arm",
        "right_arm",
        "left_gripper",
        "right_gripper",
        "left_wrist_camera",
        "right_wrist_camera",
    }
    if not expected_components.issubset(required):
        raise ValueError("collision audit must include both arms, grippers, and cameras")
    exclusions = {_pair(*item) for item in collision.get("intended_contact_exclusions", ())}
    if exclusions != {
        _pair("left_gripper", "pot_left_handle"),
        _pair("right_gripper", "pot_right_handle"),
    }:
        raise ValueError("only assigned gripper-to-handle contacts may be intended")

    lane = value["lane"]
    if lane.get("one_worker_per_gpu") is not True or lane.get("immutable_receipts") is not True:
        raise ValueError("quality lanes require exclusive GPUs and immutable receipts")
    return PutPotQualityConfig(
        path=source,
        sha256=_sha256_bytes(raw),
        grasp=grasp,
        motion=motion,
        perturbation=perturbation,
        collision=collision,
        lane=lane,
    )


def _first_run_end(mask: np.ndarray, required: int) -> int | None:
    count = 0
    for step, active in enumerate(mask):
        count = count + 1 if bool(active) else 0
        if count >= required:
            return step
    return None


def audit_broad_contact(
    *,
    left_forces_n: Any,
    right_forces_n: Any,
    left_pad_fractions: Any,
    right_pad_fractions: Any,
    left_contact_area_fractions: Any,
    right_contact_area_fractions: Any,
    left_flush_angles_deg: Any,
    right_flush_angles_deg: Any,
    pot_positions_m: Any,
    config: PutPotQualityConfig,
) -> dict[str, Any]:
    """Audit left-first broad two-pad acquisition and the four-pad latch."""

    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (
            left_forces_n,
            right_forces_n,
            left_pad_fractions,
            right_pad_fractions,
            left_contact_area_fractions,
            right_contact_area_fractions,
            left_flush_angles_deg,
            right_flush_angles_deg,
        )
    ]
    steps = arrays[0].shape[0]
    if steps < 1 or any(value.shape != (steps, 2) for value in arrays):
        raise ValueError("all per-pad measurements must have shape (steps, 2)")
    pot = np.asarray(pot_positions_m, dtype=np.float64)
    if pot.shape != (steps, 3) or not all(np.all(np.isfinite(value)) for value in arrays + [pot]):
        raise ValueError("contact audit inputs must be finite and aligned")

    rule = config.grasp
    force = float(rule["minimum_force_n"])
    margin = float(rule["minimum_pad_fraction_margin"])
    area = float(rule["minimum_contact_area_fraction"])
    angle = float(rule["maximum_flush_angle_deg"])
    required = int(rule["stable_steps"])

    def quality(values: Sequence[np.ndarray]) -> np.ndarray:
        forces, fractions, areas, angles = values
        return (
            np.all(forces >= force, axis=1)
            & np.all((fractions >= margin) & (fractions <= 1.0 - margin), axis=1)
            & np.all(areas >= area, axis=1)
            & np.all(np.abs(angles) <= angle, axis=1)
        )

    left = quality((arrays[0], arrays[2], arrays[4], arrays[6]))
    right = quality((arrays[1], arrays[3], arrays[5], arrays[7]))
    left_end = _first_run_end(left, required)
    first_right = next((int(i) for i in np.flatnonzero(right)), None)
    latch_end = _first_run_end(left & right, required)
    order_ok = left_end is not None and (first_right is None or first_right > left_end)
    pre_latch_end = steps - 1 if latch_end is None else latch_end
    displacement = np.linalg.norm(pot[: pre_latch_end + 1] - pot[0], axis=1)
    maximum_motion = float(np.max(displacement))
    motion_ok = maximum_motion <= float(rule["maximum_pre_latch_object_motion_m"])
    passed = bool(order_ok and latch_end is not None and motion_ok)
    return {
        "passed": passed,
        "left_first": bool(order_ok),
        "first_left_stable_step": left_end,
        "first_right_quality_step": first_right,
        "first_four_pad_latch_step": latch_end,
        "required_four_pad_frames": required,
        "minimum_force_n": float(min(np.min(arrays[0]), np.min(arrays[1]))),
        "minimum_pad_edge_margin": float(
            min(
                np.min(np.minimum(arrays[2], 1.0 - arrays[2])),
                np.min(np.minimum(arrays[3], 1.0 - arrays[3])),
            )
        ),
        "minimum_contact_area_fraction": float(min(np.min(arrays[4]), np.min(arrays[5]))),
        "maximum_abs_flush_angle_deg": float(
            max(np.max(np.abs(arrays[6])), np.max(np.abs(arrays[7])))
        ),
        "maximum_pre_latch_object_motion_m": maximum_motion,
    }


def _quat_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(left * right, axis=1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def audit_object_first_motion(
    *,
    pot_poses: Any,
    left_eef_poses: Any,
    right_eef_poses: Any,
    planner_mode: str,
    config: PutPotQualityConfig,
) -> dict[str, Any]:
    """Measure smooth coordinated motion and live grasp-transform retention."""

    pot = np.asarray(pot_poses, dtype=np.float64)
    left = np.asarray(left_eef_poses, dtype=np.float64)
    right = np.asarray(right_eef_poses, dtype=np.float64)
    if pot.ndim != 2 or pot.shape[1] != 7 or left.shape != pot.shape or right.shape != pot.shape:
        raise ValueError("pot and EEF pose paths must share shape (steps, 7)")
    if len(pot) < 2 or not np.all(np.isfinite(np.concatenate((pot, left, right)))):
        raise ValueError("pose paths must contain at least two finite steps")
    rule = config.motion
    position_step = np.linalg.norm(np.diff(pot[:, :3], axis=0), axis=1)
    rotation_step = _quat_distance(pot[1:, 3:], pot[:-1, 3:])
    left_relative = left[:, :3] - pot[:, :3]
    right_relative = right[:, :3] - pot[:, :3]
    translation_drift = max(
        float(np.max(np.linalg.norm(left_relative - left_relative[0], axis=1))),
        float(np.max(np.linalg.norm(right_relative - right_relative[0], axis=1))),
    )
    left_relative_rotation = _quat_distance(left[:, 3:], pot[:, 3:])
    right_relative_rotation = _quat_distance(right[:, 3:], pot[:, 3:])
    rotation_drift = max(
        float(np.max(np.abs(left_relative_rotation - left_relative_rotation[0]))),
        float(np.max(np.abs(right_relative_rotation - right_relative_rotation[0]))),
    )
    passed = bool(
        planner_mode == rule["planner_mode"]
        and np.max(position_step) <= float(rule["maximum_position_step_m"])
        and np.max(rotation_step) <= float(rule["maximum_rotation_step_rad"])
        and translation_drift <= float(rule["maximum_grasp_translation_drift_m"])
        and rotation_drift <= float(rule["maximum_grasp_rotation_drift_rad"])
    )
    acceleration = np.diff(np.diff(pot[:, :3], axis=0), axis=0)
    jerk = np.diff(acceleration, axis=0)
    return {
        "passed": passed,
        "planner_mode": planner_mode,
        "maximum_position_step_m": float(np.max(position_step)),
        "maximum_rotation_step_rad": float(np.max(rotation_step)),
        "maximum_grasp_translation_drift_m": translation_drift,
        "maximum_grasp_rotation_drift_rad": rotation_drift,
        "maximum_discrete_acceleration_m": (
            float(np.max(np.linalg.norm(acceleration, axis=1)))
            if len(acceleration)
            else 0.0
        ),
        "maximum_discrete_jerk_m": (
            float(np.max(np.linalg.norm(jerk, axis=1))) if len(jerk) else 0.0
        ),
    }


def audit_release_and_return(
    *,
    stage_events: Sequence[str],
    supported: Any,
    left_open: Any,
    right_open: Any,
    left_positions_m: Any,
    right_positions_m: Any,
    left_start_m: Any,
    right_start_m: Any,
    config: PutPotQualityConfig,
) -> dict[str, Any]:
    """Require one supported bilateral opening and open-arm return to starts."""

    expected = (
        "left_handle_stable",
        "right_handle_stable",
        "four_pad_latch",
        "bimanual_lift",
        "coordinated_transfer",
        "supported_lower",
        "open_both",
        "return_both_open_to_start",
    )
    positions = []
    cursor = -1
    for name in expected:
        try:
            cursor = list(stage_events).index(name, cursor + 1)
        except ValueError:
            cursor = -1
            break
        positions.append(cursor)
    supported_mask = np.asarray(supported, dtype=bool)
    left_mask = np.asarray(left_open, dtype=bool)
    right_mask = np.asarray(right_open, dtype=bool)
    left_path = np.asarray(left_positions_m, dtype=np.float64)
    right_path = np.asarray(right_positions_m, dtype=np.float64)
    if (
        supported_mask.ndim != 1
        or left_mask.shape != supported_mask.shape
        or right_mask.shape != supported_mask.shape
    ):
        raise ValueError("support and gripper histories must be aligned")
    if left_path.shape != (len(supported_mask), 3) or right_path.shape != left_path.shape:
        raise ValueError("arm position histories must have shape (steps, 3)")
    bilateral_open = left_mask & right_mask
    transitions = np.flatnonzero(bilateral_open & ~np.r_[False, bilateral_open[:-1]])
    opening_step = int(transitions[0]) if len(transitions) == 1 else None
    opening_supported = opening_step is not None and bool(supported_mask[opening_step])
    tolerance = float(config.motion["return_to_start_tolerance_m"])
    left_error = float(np.linalg.norm(left_path[-1] - np.asarray(left_start_m, dtype=np.float64)))
    right_error = float(
        np.linalg.norm(right_path[-1] - np.asarray(right_start_m, dtype=np.float64))
    )
    return {
        "passed": bool(
            len(positions) == len(expected)
            and len(transitions) == 1
            and opening_supported
            and bilateral_open[-1]
            and np.all(bilateral_open[opening_step:])
            and left_error <= tolerance
            and right_error <= tolerance
        ),
        "stage_order": list(stage_events),
        "bilateral_open_event_count": int(len(transitions)),
        "opening_step": opening_step,
        "opening_supported": bool(opening_supported),
        "left_return_error_m": left_error,
        "right_return_error_m": right_error,
        "final_both_open": bool(bilateral_open[-1]),
    }


def deterministic_perturbation_cases(
    config: PutPotQualityConfig, *, joint_dof: int
) -> list[dict[str, Any]]:
    """Create a reproducible perturbation bank for physical reruns."""

    if joint_dof < 1:
        raise ValueError("joint_dof must be positive")
    rule = config.perturbation
    rng = np.random.default_rng(int(rule["seed"]))
    cases = []
    for index in range(int(rule["case_count"])):
        value = {
            "case_index": index,
            "seed": int(rule["seed"]),
            "grasp_translation_m": rng.normal(0.0, float(rule["translation_std_m"]), 3).tolist(),
            "eef_rotation_vector_rad": rng.normal(0.0, float(rule["rotation_std_rad"]), 3).tolist(),
            "joint_action_rad": rng.normal(0.0, float(rule["joint_std_rad"]), joint_dof).tolist(),
        }
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        value["case_sha256"] = _sha256_bytes(canonical)
        cases.append(value)
    return cases


def audit_perturbation_outcomes(
    cases: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    config: PutPotQualityConfig,
) -> dict[str, Any]:
    """Match immutable case hashes to success receipts and enforce the pass rate."""

    expected = {str(case["case_sha256"]) for case in cases}
    observed = {str(item.get("case_sha256")): bool(item.get("passed")) for item in outcomes}
    if set(observed) != expected:
        raise ValueError("perturbation outcomes do not match the generated case bank")
    passed_count = sum(observed.values())
    fraction = passed_count / len(expected) if expected else 0.0
    required = float(config.perturbation["required_pass_fraction"])
    return {
        "passed": bool(fraction >= required),
        "case_count": len(expected),
        "passed_count": passed_count,
        "pass_fraction": fraction,
        "required_pass_fraction": required,
        "case_sha256": sorted(expected),
    }


def audit_swept_self_collision(
    *,
    component_centers_m: Mapping[str, Any],
    component_radii_m: Mapping[str, float],
    config: PutPotQualityConfig,
    structural_adjacencies: Sequence[Sequence[str]] = (),
) -> dict[str, Any]:
    """Screen swept sphere proxies for robot, gripper, and wrist-camera collisions.

    Callers may include pot-handle proxies; only the two configured assigned
    gripper/handle pairs are treated as intended contacts. Structural adjacency
    must be declared separately and is reported for provenance.
    """

    paths = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in component_centers_m.items()
    }
    required = set(config.collision["required_components"])
    if not required.issubset(paths):
        raise ValueError("swept collision audit is missing required robot components")
    steps = next(iter(paths.values())).shape[0]
    if steps < 1 or any(
        value.shape != (steps, 3) or not np.all(np.isfinite(value))
        for value in paths.values()
    ):
        raise ValueError("component center paths must share finite shape (steps, 3)")
    if set(paths) != set(component_radii_m):
        raise ValueError("each swept component requires exactly one radius")
    if any(
        not np.isfinite(float(radius)) or float(radius) <= 0.0
        for radius in component_radii_m.values()
    ):
        raise ValueError("component radii must be finite and positive")
    adjacent = {_pair(*item) for item in structural_adjacencies}
    intended = {_pair(*item) for item in config.collision["intended_contact_exclusions"]}
    excluded = adjacent | intended
    minimum_required = float(config.collision["minimum_clearance_m"])
    reports = []
    names = sorted(paths)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            pair = _pair(first, second)
            distance = np.linalg.norm(paths[first] - paths[second], axis=1)
            clearance = (
                distance
                - float(component_radii_m[first])
                - float(component_radii_m[second])
            )
            minimum_step = int(np.argmin(clearance))
            reports.append(
                {
                    "components": list(pair),
                    "minimum_clearance_m": float(clearance[minimum_step]),
                    "minimum_clearance_step": minimum_step,
                    "excluded": pair in excluded,
                    "exclusion_kind": (
                        "intended_contact"
                        if pair in intended
                        else ("structural_adjacency" if pair in adjacent else None)
                    ),
                    "collision": bool(
                        clearance[minimum_step] < minimum_required
                        and pair not in excluded
                    ),
                }
            )
    collisions = [item for item in reports if item["collision"]]
    return {
        "passed": not collisions,
        "minimum_required_clearance_m": minimum_required,
        "collisions": collisions,
        "pairs": reports,
    }


def validate_lane_contract(
    *,
    pair_index: int,
    lane_id: str,
    cuda_visible_devices: str,
    output_root: str | Path,
    config: PutPotQualityConfig,
) -> dict[str, Any]:
    """Validate explicit pair ownership, one visible GPU, and pair-local output."""

    if pair_index < 0 or not _LANE_ID.fullmatch(lane_id):
        raise ValueError("pair index and lane ID must be explicit and unambiguous")
    devices = [part.strip() for part in cuda_visible_devices.split(",") if part.strip()]
    if len(devices) != 1 or not devices[0].isdigit():
        raise ValueError("quality worker must see exactly one numeric GPU")
    root = Path(output_root).resolve()
    expected_suffix = str(config.lane["pair_directory_format"]).format(pair_index=pair_index)
    if tuple(root.parts[-2:]) != tuple(Path(expected_suffix).parts[-2:]):
        raise ValueError("quality output root must be pair-local")
    return {
        "pair_index": pair_index,
        "lane_id": lane_id,
        "gpu_id": devices[0],
        "output_root": str(root),
        "receipt_path": str(root / str(config.lane["receipt_name"])),
        "config_sha256": config.sha256,
    }


class GpuLease:
    """Exclusive process-scoped GPU lease for one parallel lane."""

    def __init__(self, lease_root: str | Path, gpu_id: str, lane_id: str):
        if not gpu_id.isdigit() or not _LANE_ID.fullmatch(lane_id):
            raise ValueError("GPU and lane identifiers must be unambiguous")
        self.path = Path(lease_root) / f"gpu-{gpu_id}.lease"
        self.lane_id = lane_id
        self._payload = json.dumps({"lane_id": lane_id, "pid": os.getpid()}, sort_keys=True)
        self._owned = False

    def __enter__(self) -> "GpuLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(self._payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._owned = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._owned and self.path.read_text(encoding="utf-8").strip() == self._payload:
            self.path.unlink()
        self._owned = False


def write_immutable_receipt(path: str | Path, value: Mapping[str, Any]) -> str:
    """Publish canonical JSON exactly once without exposing a partial final file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_bytes(payload)
