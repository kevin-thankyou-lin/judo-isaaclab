"""Source-first admission policy for bounded PutPot repair attempts.

This module keeps the coding agent from turning a durable failure boundary into
unbounded pair-specific hill climbing.  It makes the source demonstration, the
earliest physical failure, robust contact margins, and the repair mechanism
part of the immutable queue contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


STAGES = (
    "bimanual_handle_grasp",
    "smooth_bimanual_transport",
    "support_alignment",
    "release_and_settle",
    "accepted",
)

REPAIR_FAMILY_STAGE = {
    "staged_bilateral_acquisition": "bimanual_handle_grasp",
    "grasp_depth_correction": "bimanual_handle_grasp",
    "contact_frame_entry_alignment": "bimanual_handle_grasp",
    "zero_jump_transport": "smooth_bimanual_transport",
    "object_local_corotation": "smooth_bimanual_transport",
    "support_frame_alignment": "support_alignment",
    "release_settle": "release_and_settle",
    "bounded_geometric_search": None,
}

DEFAULT_POLICY = {
    "schema_version": 1,
    "maximum_attempts_per_visit": 4,
    "maximum_nonimproving_attempts_per_family": 2,
    "minimum_bilateral_latch_frames": 15,
    "minimum_finger_force_n": 1.0,
    "minimum_pad_fraction_margin": 0.10,
    "maximum_pre_peer_object_motion_m": 0.003,
    "minimum_material_latch_frame_delta": 15,
    "minimum_material_transport_path_delta_m": 0.05,
    "minimum_material_center_error_delta_m": 0.01,
    "require_action_identical_diagnostic_before_next_attempt": True,
}

DIAGNOSTIC_OVERLAY_REQUIREMENTS = (
    "target_handle_contact_frame",
    "target_handle_tangent_axis",
    "actual_pad_centers",
    "actual_pad_axes",
    "jaw_closing_line",
    "target_wrist_frame",
    "actual_wrist_frame",
    "signed_correction_vectors",
    "screen_space_color_legend",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pose(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (7,):
        raise ValueError(f"{name} must have shape (7,)")
    norm = float(np.linalg.norm(result[3:]))
    if norm < 1.0e-8:
        raise ValueError(f"{name} has a zero quaternion")
    result = result.copy()
    result[3:] /= norm
    return result


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    twice_cross = 2.0 * np.cross(quaternion[1:], vector)
    return vector + quaternion[0] * twice_cross + np.cross(
        quaternion[1:], twice_cross
    )


def _inverse_pose(value: Any) -> np.ndarray:
    value = _pose(value, "pose")
    result = np.empty(7, dtype=np.float64)
    result[3:] = value[3:] * np.asarray([1.0, -1.0, -1.0, -1.0])
    result[:3] = _quat_rotate(result[3:], -value[:3])
    return result


def _compose_pose(left: Any, right: Any) -> np.ndarray:
    left = _pose(left, "left")
    right = _pose(right, "right")
    result = np.empty(7, dtype=np.float64)
    result[:3] = left[:3] + _quat_rotate(left[3:], right[:3])
    result[3:] = _quat_multiply(left[3:], right[3:])
    result[3:] /= np.linalg.norm(result[3:])
    return result


def _relative_pose(frame: Any, value: Any) -> list[float]:
    return _compose_pose(_inverse_pose(frame), value).tolist()


def build_source_demo_card(source_keyframes: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact semantic strategy card from proven source keyframes."""

    if source_keyframes.get("schema_version") != 1:
        raise ValueError("source keyframes must use schema_version 1")
    frames = source_keyframes.get("frames")
    if not isinstance(frames, Mapping):
        raise ValueError("source keyframes must contain a frames mapping")
    required = (
        "left_pregrasp",
        "left_handle_grasp",
        "right_pregrasp",
        "right_handle_grasp",
        "pot_lift",
        "pot_transport",
        "support_align",
        "pot_release",
        "stable_settle",
    )
    missing = [name for name in required if name not in frames]
    if missing:
        raise ValueError(f"source keyframes are missing {missing!r}")

    semantic_frames: dict[str, Any] = {}
    for name in required:
        frame = frames[name]
        semantic_frames[name] = {
            "sample_index": int(frame["sample_index"]),
            "pot_pose": list(frame["pot_pose"]),
            "left_eef_in_pot": _relative_pose(
                frame["pot_pose"], frame["left_eef_pose"]
            ),
            "right_eef_in_pot": _relative_pose(
                frame["pot_pose"], frame["right_eef_pose"]
            ),
            "left_grasp": bool(frame["left_grasp"]),
            "right_grasp": bool(frame["right_grasp"]),
        }

    left_index = semantic_frames["left_handle_grasp"]["sample_index"]
    right_index = semantic_frames["right_handle_grasp"]["sample_index"]
    contact_order = ["left", "right"] if left_index < right_index else ["right", "left"]
    lift_start = np.asarray(frames["right_handle_grasp"]["pot_pose"][:3])
    lift_end = np.asarray(frames["pot_lift"]["pot_pose"][:3])
    lift_delta = lift_end - lift_start
    lift_norm = float(np.linalg.norm(lift_delta))
    if lift_norm < 1.0e-8:
        raise ValueError("source lift direction is degenerate")

    return {
        "schema_version": 1,
        "source_dataset": source_keyframes["source_dataset"],
        "source_dataset_sha256": source_keyframes["source_dataset_sha256"],
        "source_assets": source_keyframes["source_assets"],
        "stage_sequence": [
            "staged_bilateral_acquisition",
            "robust_bimanual_latch",
            "lift",
            "object_pose_closed_loop_transport",
            "support_alignment",
            "release_and_settle",
        ],
        "contact_order": contact_order,
        "semantic_frames": semantic_frames,
        "latch_contract": {
            "minimum_bilateral_frames": DEFAULT_POLICY[
                "minimum_bilateral_latch_frames"
            ],
            "minimum_finger_force_n": DEFAULT_POLICY["minimum_finger_force_n"],
            "minimum_pad_fraction_margin": DEFAULT_POLICY[
                "minimum_pad_fraction_margin"
            ],
            "maximum_pre_peer_object_motion_m": DEFAULT_POLICY[
                "maximum_pre_peer_object_motion_m"
            ],
        },
        "lift_direction_in_source_world": (lift_delta / lift_norm).tolist(),
        "transport_contract": {
            "frame": "observed_pot_pose",
            "preserve_loaded_object_local_grasp_transforms": True,
            "zero_jump_at_handoff": True,
        },
        "release_contract": {
            "requires_centered_support": True,
            "requires_stable_support_window": True,
            "requires_terminal_speed_gate": True,
        },
    }


def write_source_demo_card(
    source_keyframes_json: str | Path, output_json: str | Path
) -> dict[str, Any]:
    source = json.loads(Path(source_keyframes_json).read_text(encoding="utf-8"))
    card = build_source_demo_card(source)
    target = Path(output_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "x", encoding="utf-8") as stream:
        json.dump(card, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return source_demo_card_receipt(target)


def load_source_demo_card(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "source_dataset",
        "source_dataset_sha256",
        "source_assets",
        "stage_sequence",
        "contact_order",
        "semantic_frames",
        "latch_contract",
        "lift_direction_in_source_world",
        "transport_contract",
        "release_contract",
    }
    if set(value) != required or value["schema_version"] != 1:
        raise ValueError("invalid PutPot source-demo card schema")
    if value["contact_order"] not in (["left", "right"], ["right", "left"]):
        raise ValueError("source-demo contact_order must contain left and right")
    if value["transport_contract"] != {
        "frame": "observed_pot_pose",
        "preserve_loaded_object_local_grasp_transforms": True,
        "zero_jump_at_handoff": True,
    }:
        raise ValueError("source-demo card must require object-local zero-jump transport")
    return value


def source_demo_card_receipt(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    value = load_source_demo_card(target)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "schema_version": value["schema_version"],
        "contact_order": value["contact_order"],
    }


def action_tensor_receipt(trace_npz: str | Path) -> dict[str, Any]:
    """Hash the exact executable action tensor carried by one immutable trace."""

    target = Path(trace_npz).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"PutPot action trace is missing: {target}")
    with np.load(target, allow_pickle=False) as trace:
        if "actions" not in trace.files:
            raise ValueError("PutPot trace has no actions tensor")
        actions = np.ascontiguousarray(trace["actions"])
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"PutPot actions must have shape (steps, 14), got {actions.shape}")
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "shape": list(actions.shape),
        "dtype": actions.dtype.str,
        "bytes_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
    }


def _require_vector(value: Any, name: str) -> list[float]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"diagnostic {name} must be a finite length-3 vector")
    return vector.tolist()


def load_failed_attempt_diagnostic(path: str | Path) -> dict[str, Any]:
    """Load the machine-checkable receipt that unlocks the next physical attempt."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "classification",
        "physical_attempt",
        "diagnostic",
        "action_parity",
        "protocol",
        "overlay",
        "measured_residuals",
        "next_mechanism_prediction",
    }
    if not required.issubset(value) or value.get("schema_version") != 1:
        raise ValueError("invalid failed-attempt diagnostic receipt schema")
    if value.get("classification") != "action_identical_render_diagnostic":
        raise ValueError("diagnostic receipt classification is not action-identical")
    parity = value.get("action_parity")
    if not isinstance(parity, Mapping) or not (
        parity.get("exactly_equal") is True
        and parity.get("reference_shape") == parity.get("replay_shape")
        and parity.get("reference_actions_bytes_sha256")
        == parity.get("replay_actions_bytes_sha256")
        and float(parity.get("maximum_absolute_difference", float("inf"))) == 0.0
    ):
        raise ValueError("diagnostic action parity gate failed")
    protocol = value.get("protocol")
    if not isinstance(protocol, Mapping) or not (
        protocol.get("physics_or_controller_changes") is False
        and protocol.get("causal_mechanism_attempt_consumed") is False
        and protocol.get("training_eligible") is False
        and protocol.get("attempt_identity") is None
    ):
        raise ValueError("diagnostic consuming/non-training protocol gate failed")
    overlay = value.get("overlay")
    if not isinstance(overlay, Mapping) or any(
        overlay.get(name) is not True for name in DIAGNOSTIC_OVERLAY_REQUIREMENTS
    ):
        raise ValueError("diagnostic coordinate-axis overlay gate failed")
    residuals = value.get("measured_residuals")
    if not isinstance(residuals, Mapping):
        raise ValueError("diagnostic receipt has no measured residuals")
    _require_vector(
        residuals.get("signed_translation_residual_world_m"),
        "signed translation residual",
    )
    _require_vector(
        residuals.get("signed_rotation_residual_axis_angle_deg"),
        "signed rotation residual",
    )
    for name in ("translation_norm_m", "rotation_norm_deg"):
        number = residuals.get(name)
        if not isinstance(number, (int, float)) or not np.isfinite(number):
            raise ValueError(f"diagnostic {name} must be finite")
    prediction = value.get("next_mechanism_prediction")
    if not isinstance(prediction, Mapping) or not isinstance(
        prediction.get("mechanism_id"), str
    ) or not prediction["mechanism_id"].strip():
        raise ValueError("diagnostic receipt has no next-mechanism prediction")
    _require_vector(
        prediction.get("signed_translation_mm"),
        "predicted signed translation",
    )
    _require_vector(
        prediction.get("signed_rotation_axis_angle_deg"),
        "predicted signed rotation",
    )
    if not isinstance(prediction.get("sign_basis"), str) or not prediction[
        "sign_basis"
    ].strip():
        raise ValueError("diagnostic next-mechanism sign basis is missing")
    return value


def validate_failed_attempt_diagnostic(
    path: str | Path,
    *,
    previous_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove a failed physical attempt has an exact, decoded, non-consuming replay."""

    value = load_failed_attempt_diagnostic(path)
    physical = value.get("physical_attempt")
    diagnostic = value.get("diagnostic")
    if not isinstance(physical, Mapping) or not isinstance(diagnostic, Mapping):
        raise ValueError("diagnostic receipt artifact mappings are missing")
    if physical.get("request_id") != previous_receipt.get("request_id"):
        raise ValueError("diagnostic does not belong to the latest physical attempt")
    result_path = previous_receipt.get("result_json")
    if not isinstance(result_path, str) or not Path(result_path).is_file():
        raise ValueError("latest physical attempt has no immutable result")
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    if result.get("checks", {}).get("accepted_task_success") is True:
        raise ValueError("successful physical attempts must stop instead of being repaired")
    physical_trace = result_trace_path(result)
    physical_video_value = result.get("video")
    physical_video = (
        Path(physical_video_value["path"])
        if isinstance(physical_video_value, Mapping)
        and isinstance(physical_video_value.get("path"), str)
        else None
    )
    if physical_trace is None or not physical_trace.is_file():
        raise ValueError("failed physical attempt has no immutable trace")
    if physical_video is None or not physical_video.is_file():
        raise ValueError("failed physical attempt has no immutable MP4")
    if not (
        physical_video_value.get("codec") == "h264"
        and int(physical_video_value.get("full_decode_returncode", -1)) == 0
        and int(physical_video_value.get("frame_count", 0)) > 0
        and result.get("checks", {}).get("h264_nonempty") is True
        and result.get("checks", {}).get("fully_decodable") is True
    ):
        raise ValueError("failed physical attempt MP4 gate failed")

    def verify_file(receipt: Any, actual: Path, name: str) -> None:
        if not isinstance(receipt, Mapping):
            raise ValueError(f"diagnostic {name} receipt is missing")
        if Path(str(receipt.get("path", ""))).resolve() != actual.resolve():
            raise ValueError(f"diagnostic {name} path mismatch")
        if receipt.get("sha256") != sha256_file(actual):
            raise ValueError(f"diagnostic {name} hash mismatch")

    verify_file(physical.get("trace"), physical_trace, "physical trace")
    verify_file(physical.get("video"), physical_video, "physical video")
    diagnostic_result = Path(str(diagnostic.get("result", {}).get("path", "")))
    diagnostic_trace = Path(str(diagnostic.get("trace", {}).get("path", "")))
    diagnostic_video = Path(str(diagnostic.get("video", {}).get("path", "")))
    for artifact, actual, name in (
        (diagnostic.get("result"), diagnostic_result, "result"),
        (diagnostic.get("trace"), diagnostic_trace, "trace"),
        (diagnostic.get("video"), diagnostic_video, "video"),
    ):
        if not actual.is_file():
            raise ValueError(f"diagnostic {name} artifact is missing")
        verify_file(artifact, actual, name)

    reference = action_tensor_receipt(physical_trace)
    replay = action_tensor_receipt(diagnostic_trace)
    parity = value["action_parity"]
    if not (
        reference["shape"] == replay["shape"] == parity["reference_shape"]
        and replay["shape"] == parity["replay_shape"]
        and reference["bytes_sha256"]
        == parity["reference_actions_bytes_sha256"]
        and replay["bytes_sha256"] == parity["replay_actions_bytes_sha256"]
    ):
        raise ValueError("diagnostic live action tensor hash gate failed")
    with np.load(physical_trace, allow_pickle=False) as left, np.load(
        diagnostic_trace, allow_pickle=False
    ) as right:
        if not np.array_equal(left["actions"], right["actions"]):
            raise ValueError("diagnostic live action tensor equality gate failed")

    diagnostic_result_value = json.loads(
        diagnostic_result.read_text(encoding="utf-8")
    )
    render_receipt = diagnostic_result_value.get("protocol", {}).get(
        "render_diagnostic"
    )
    video_receipt = diagnostic_result_value.get("video")
    if not isinstance(render_receipt, Mapping) or not (
        render_receipt.get("physics_or_controller_changes") is False
        and render_receipt.get("causal_mechanism_attempt_consumed") is False
        and render_receipt.get("training_eligible") is False
        and render_receipt.get("replay_actions", {}).get("exactly_equal") is True
    ):
        raise ValueError("diagnostic result protocol gate failed")
    result_overlay = render_receipt.get("overlay", {})
    result_overlay_passes = {
        "target_handle_contact_frame": result_overlay.get(
            "target_left_contact_frame"
        )
        is True,
        "target_handle_tangent_axis": result_overlay.get("target_tangent_axis")
        == "local_x",
        "actual_pad_centers": result_overlay.get("actual_left_pad_centers") == 2,
        "actual_pad_axes": result_overlay.get("actual_left_pad_axes") == 2,
        "jaw_closing_line": result_overlay.get("jaw_closing_line") is True,
        "target_wrist_frame": result_overlay.get("target_left_wrist_frame") is True,
        "actual_wrist_frame": result_overlay.get("actual_left_wrist_frame") is True,
        "signed_correction_vectors": (
            result_overlay.get("actual_to_desired_correction_vector") is True
            and result_overlay.get(
                "jaw_midpoint_to_target_contact_correction_vector"
            )
            is True
        ),
        "screen_space_color_legend": result_overlay.get(
            "screen_space_color_legend"
        )
        is True,
    }
    if result_overlay_passes != {
        name: True for name in DIAGNOSTIC_OVERLAY_REQUIREMENTS
    }:
        raise ValueError("diagnostic result coordinate-axis overlay gate failed")
    if not isinstance(video_receipt, Mapping) or not (
        video_receipt.get("codec") == "h264"
        and int(video_receipt.get("full_decode_returncode", -1)) == 0
        and int(video_receipt.get("frame_count", 0)) > 0
    ):
        raise ValueError("diagnostic H.264 full-decode gate failed")
    if diagnostic_result_value.get("checks", {}).get(
        "diagnostic_actions_identical"
    ) is not True:
        raise ValueError("diagnostic result rejected action identity")
    return value


@dataclass(frozen=True)
class LatchEvidence:
    first_contact_step: int | None
    bilateral_latch_step: int | None
    pre_peer_object_motion_m: float | None
    longest_robust_bilateral_frames: int
    robust_window_minimum_force_n: float | None
    robust_window_minimum_pad_fraction_margin: float | None
    passes_robust_latch: bool

    def receipt(self) -> dict[str, Any]:
        return asdict(self)


def _longest_true_run(values: np.ndarray) -> tuple[int, slice | None]:
    best_start = None
    best_length = 0
    start = None
    for index, value in enumerate(values.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            length = index - start
            if length > best_length:
                best_start = start
                best_length = length
            start = None
    return best_length, (
        None if best_start is None else slice(best_start, best_start + best_length)
    )


def trace_latch_evidence(
    trace_npz: str | Path,
    *,
    policy: Mapping[str, Any] = DEFAULT_POLICY,
) -> LatchEvidence:
    """Measure robust bilateral latch margins from an immutable rollout trace."""

    with np.load(trace_npz, allow_pickle=False) as trace:
        pot = np.asarray(trace["pot_poses"], dtype=np.float64)
        left_forces = np.asarray(trace["left_finger_forces_n"], dtype=np.float64)
        right_forces = np.asarray(trace["right_finger_forces_n"], dtype=np.float64)
        left_fractions = np.asarray(trace["left_pad_fractions"], dtype=np.float64)
        right_fractions = np.asarray(trace["right_pad_fractions"], dtype=np.float64)
    if not (
        pot.ndim == 2
        and pot.shape[1] == 7
        and left_forces.shape == right_forces.shape == left_fractions.shape == right_fractions.shape
        and left_forces.shape == (len(pot), 2)
    ):
        raise ValueError("PutPot trace has incompatible contact-array shapes")

    force_threshold = float(policy["minimum_finger_force_n"])
    fraction_margin = float(policy["minimum_pad_fraction_margin"])
    physical_margin = np.minimum(
        np.concatenate((left_fractions, right_fractions), axis=1),
        1.0 - np.concatenate((left_fractions, right_fractions), axis=1),
    )
    all_forces = np.concatenate((left_forces, right_forces), axis=1)
    robust = np.all(all_forces >= force_threshold, axis=1) & np.all(
        np.isfinite(physical_margin) & (physical_margin >= fraction_margin), axis=1
    )
    longest, window = _longest_true_run(robust)

    physical_contact = np.concatenate(
        (
            (left_forces >= 0.1) & np.isfinite(left_fractions) & (left_fractions >= 0.0) & (left_fractions <= 1.0),
            (right_forces >= 0.1) & np.isfinite(right_fractions) & (right_fractions >= 0.0) & (right_fractions <= 1.0),
        ),
        axis=1,
    )
    any_contact = np.any(physical_contact, axis=1)
    bilateral_physical = np.all(physical_contact, axis=1)
    first_contact_indices = np.flatnonzero(any_contact)
    bilateral_indices = np.flatnonzero(bilateral_physical)
    first_contact = int(first_contact_indices[0]) if first_contact_indices.size else None
    bilateral_step = int(bilateral_indices[0]) if bilateral_indices.size else None
    motion = None
    if (
        first_contact is not None
        and bilateral_step is not None
        and bilateral_step >= first_contact
    ):
        relative = pot[first_contact : bilateral_step + 1, :3] - pot[first_contact, :3]
        motion = float(np.max(np.linalg.norm(relative, axis=1)))

    min_force = None
    min_fraction = None
    if window is not None:
        min_force = float(np.min(all_forces[window]))
        min_fraction = float(np.min(physical_margin[window]))
    passes = bool(
        longest >= int(policy["minimum_bilateral_latch_frames"])
        and motion is not None
        and motion <= float(policy["maximum_pre_peer_object_motion_m"])
    )
    return LatchEvidence(
        first_contact_step=first_contact,
        bilateral_latch_step=bilateral_step,
        pre_peer_object_motion_m=motion,
        longest_robust_bilateral_frames=longest,
        robust_window_minimum_force_n=min_force,
        robust_window_minimum_pad_fraction_margin=min_fraction,
        passes_robust_latch=passes,
    )


def result_failed_stage(result: Mapping[str, Any] | None) -> str:
    if not isinstance(result, Mapping):
        return "bimanual_handle_grasp"
    checks = result.get("checks", {})
    if not isinstance(checks, Mapping):
        return "bimanual_handle_grasp"
    if checks.get("bimanual_pick_observed") is not True:
        return "bimanual_handle_grasp"
    if checks.get("bimanual_transport_completed") is not True:
        return "smooth_bimanual_transport"
    if checks.get("centered_on_cooktop") is not True or checks.get("coded_task_success") is not True:
        return "support_alignment"
    if checks.get("pot_released") is not True or checks.get("stable_support_window") is not True:
        return "release_and_settle"
    if checks.get("accepted_task_success") is True:
        return "accepted"
    return "release_and_settle"


def result_trace_path(result: Mapping[str, Any] | None) -> Path | None:
    if not isinstance(result, Mapping):
        return None
    path = result.get("provenance", {}).get("trace", {}).get("path")
    return Path(path) if isinstance(path, str) and path else None


def repair_evidence(result: Mapping[str, Any] | None) -> dict[str, Any]:
    reported_stage = result_failed_stage(result)
    trace_path = result_trace_path(result)
    latch = None
    if trace_path is not None and trace_path.is_file():
        latch = trace_latch_evidence(trace_path)
    effective_stage = reported_stage
    if reported_stage == "smooth_bimanual_transport" and (
        latch is None or not latch.passes_robust_latch
    ):
        effective_stage = "bimanual_handle_grasp"
    return {
        "reported_failed_stage": reported_stage,
        "earliest_failed_stage": effective_stage,
        "latch": None if latch is None else latch.receipt(),
    }


def _nested_number(value: Mapping[str, Any], path: Sequence[str]) -> float | None:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return float(current) if isinstance(current, (int, float)) else None


def material_progress(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> tuple[bool, str]:
    """Use stage progress and robust task margins, never raw proxy counts."""

    before_evidence = repair_evidence(before)
    after_evidence = repair_evidence(after)
    before_stage = before_evidence["earliest_failed_stage"]
    after_stage = after_evidence["earliest_failed_stage"]
    if STAGES.index(after_stage) > STAGES.index(before_stage):
        return True, f"earliest_stage_advanced:{before_stage}->{after_stage}"
    if after_stage != before_stage:
        return False, f"earliest_stage_regressed:{before_stage}->{after_stage}"

    if before_stage == "bimanual_handle_grasp":
        before_frames = (before_evidence["latch"] or {}).get(
            "longest_robust_bilateral_frames", 0
        )
        after_latch = after_evidence["latch"] or {}
        after_frames = after_latch.get("longest_robust_bilateral_frames", 0)
        improved = bool(
            after_frames - before_frames
            >= DEFAULT_POLICY["minimum_material_latch_frame_delta"]
            and after_latch.get("pre_peer_object_motion_m") is not None
            and after_latch["pre_peer_object_motion_m"]
            <= DEFAULT_POLICY["maximum_pre_peer_object_motion_m"]
        )
        return improved, f"robust_latch_frames:{before_frames}->{after_frames}"

    if before_stage == "smooth_bimanual_transport":
        before_path = _nested_number(before, ("metrics", "transport_executed", "path_length_m")) or 0.0
        after_path = _nested_number(after, ("metrics", "transport_executed", "path_length_m")) or 0.0
        before_center = _nested_number(before, ("metrics", "center_error_m"))
        after_center = _nested_number(after, ("metrics", "center_error_m"))
        path_progress = after_path - before_path >= DEFAULT_POLICY[
            "minimum_material_transport_path_delta_m"
        ]
        center_progress = bool(
            before_center is not None
            and after_center is not None
            and before_center - after_center
            >= DEFAULT_POLICY["minimum_material_center_error_delta_m"]
            and after_path >= before_path
        )
        return bool(path_progress or center_progress), (
            f"transport_path_m:{before_path:.6f}->{after_path:.6f};"
            f"center_error_m:{before_center}->{after_center}"
        )

    if before_stage == "support_alignment":
        before_center = _nested_number(before, ("metrics", "center_error_m"))
        after_center = _nested_number(after, ("metrics", "center_error_m"))
        improved = bool(
            before_center is not None
            and after_center is not None
            and before_center - after_center
            >= DEFAULT_POLICY["minimum_material_center_error_delta_m"]
        )
        return improved, f"center_error_m:{before_center}->{after_center}"
    return False, "release_requires_end_to_end_acceptance"


def load_repair_proposal(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "source_demo_card_sha256",
        "baseline_request_id",
        "target_stage",
        "mechanism_id",
        "repair_family",
        "hypothesis",
        "expected_task_delta",
        "changed_primitives",
    }
    if set(value) != required or value["schema_version"] != 1:
        raise ValueError("invalid PutPot repair-proposal schema")
    for key in ("source_demo_card_sha256", "baseline_request_id", "mechanism_id", "hypothesis", "expected_task_delta"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"repair proposal {key} must be nonempty")
    if value["target_stage"] not in STAGES[:-1]:
        raise ValueError("repair proposal target_stage is unsupported")
    if value["repair_family"] not in REPAIR_FAMILY_STAGE:
        raise ValueError("repair proposal must use the shared repair library")
    family_stage = REPAIR_FAMILY_STAGE[value["repair_family"]]
    if family_stage is not None and family_stage != value["target_stage"]:
        raise ValueError("repair family does not match target_stage")
    if not isinstance(value["changed_primitives"], list) or not value["changed_primitives"]:
        raise ValueError("repair proposal changed_primitives must be nonempty")
    if not all(isinstance(item, str) and item.strip() for item in value["changed_primitives"]):
        raise ValueError("repair proposal changed_primitives must contain strings")
    return value


def validate_repair_admission(
    proposal: Mapping[str, Any],
    *,
    session: Mapping[str, Any],
    prior_requests: Sequence[Mapping[str, Any]],
    prior_receipts: Sequence[Mapping[str, Any]],
) -> None:
    """Reject wrong-stage, repeated, or exhausted repair mechanisms."""

    source_card = session.get("source_demo_card")
    if not isinstance(source_card, Mapping):
        raise ValueError("source-first session has no source-demo card")
    if proposal["source_demo_card_sha256"] != source_card.get("sha256"):
        raise ValueError("repair proposal source-demo card hash mismatch")
    if not prior_receipts:
        raise ValueError("repair proposal requires a completed baseline receipt")
    previous = prior_receipts[-1]
    if proposal["baseline_request_id"] != previous.get("request_id"):
        raise ValueError("repair proposal must cite the latest acknowledged request")
    previous_result_path = previous.get("result_json")
    if not isinstance(previous_result_path, str) or not Path(previous_result_path).is_file():
        raise ValueError("latest receipt has no readable immutable result")
    previous_result = json.loads(Path(previous_result_path).read_text(encoding="utf-8"))
    earliest = repair_evidence(previous_result)["earliest_failed_stage"]
    if proposal["target_stage"] != earliest:
        raise ValueError(
            f"earliest-failure rule requires {earliest}, not {proposal['target_stage']}"
        )

    prior_by_id = {
        request.get("request_id"): request
        for request in prior_requests
        if request.get("type") == "attempt"
    }
    used_mechanisms = {
        request.get("repair_proposal", {}).get("mechanism_id")
        for request in prior_by_id.values()
        if isinstance(request.get("repair_proposal"), Mapping)
    }
    if proposal["mechanism_id"] in used_mechanisms:
        raise ValueError("one rollout is allowed per distinct causal mechanism")

    nonimproving = 0
    for receipt in prior_receipts:
        request = prior_by_id.get(receipt.get("request_id"), {})
        prior_proposal = request.get("repair_proposal")
        if not isinstance(prior_proposal, Mapping) or prior_proposal.get("repair_family") != proposal["repair_family"]:
            continue
        baseline_id = prior_proposal.get("baseline_request_id")
        baseline_receipt = next(
            (row for row in prior_receipts if row.get("request_id") == baseline_id),
            None,
        )
        if baseline_receipt is None:
            raise ValueError("repair history has a missing baseline receipt")
        paths = (baseline_receipt.get("result_json"), receipt.get("result_json"))
        if not all(isinstance(path, str) and Path(path).is_file() for path in paths):
            raise ValueError("repair history has a missing result artifact")
        before = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
        after = json.loads(Path(paths[1]).read_text(encoding="utf-8"))
        improved, _ = material_progress(before, after)
        if not improved:
            nonimproving += 1
    if nonimproving >= int(
        session["repair_policy"]["maximum_nonimproving_attempts_per_family"]
    ):
        raise ValueError("repair family is exhausted after two non-improving attempts")


def training_data_route(
    result: Mapping[str, Any] | None, error: str | None = None
) -> str:
    """Separate strict demonstrations from useful failures and bad artifacts."""

    if error is not None or not isinstance(result, Mapping):
        return "reject_artifact"
    checks = result.get("checks", {})
    if not isinstance(checks, Mapping):
        return "reject_artifact"
    if checks.get("h264_nonempty") is not True or checks.get("fully_decodable") is not True:
        return "reject_artifact"
    evidence = repair_evidence(result)
    if checks.get("accepted_task_success") is True and (
        evidence["latch"] or {}
    ).get("passes_robust_latch") is True:
        return "accepted_demo"
    return "failure_or_critic"
