"""Build the action-identical visual receipt required before another PutPot attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from judo_isaaclab.putpot_repair_policy import (
    _compose_pose,
    _quat_multiply,
    action_tensor_receipt,
    result_trace_path,
    sha256_file,
)


def _artifact(path: str | Path) -> dict[str, object]:
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def _video(result: dict[str, object], name: str) -> tuple[Path, dict[str, object]]:
    receipt = result.get("video")
    if not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str):
        raise ValueError(f"{name} result has no video receipt")
    path = Path(receipt["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if (
        receipt.get("codec") != "h264"
        or int(receipt.get("full_decode_returncode", -1)) != 0
        or decode.returncode != 0
    ):
        raise ValueError(f"{name} video is not fully decoded H.264: {decode.stderr}")
    return path, receipt


def _axis_angle_deg(actual: np.ndarray, target: np.ndarray) -> list[float]:
    inverse_actual = actual * np.asarray([1.0, -1.0, -1.0, -1.0])
    delta = _quat_multiply(target, inverse_actual)
    if delta[0] < 0.0:
        delta = -delta
    vector_norm = float(np.linalg.norm(delta[1:]))
    if vector_norm <= 1.0e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * np.arctan2(vector_norm, float(delta[0]))
    return (np.degrees(angle) * delta[1:] / vector_norm).tolist()


def _local_mpc_active_arm(trace: object, step: int) -> str:
    """Resolve the active wrist from a new trace, preserving legacy left traces."""

    files = getattr(trace, "files", trace)
    if "local_mpc_active_arm" not in files:
        return "left"
    arm = str(trace["local_mpc_active_arm"][step])
    if arm not in {"left", "right"}:
        raise ValueError(f"invalid local-MPC active arm at step {step}: {arm!r}")
    return arm


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-request-id", required=True)
    parser.add_argument("--physical-result-json", required=True)
    parser.add_argument("--diagnostic-result-json", required=True)
    parser.add_argument("--next-mechanism-id", required=True)
    parser.add_argument("--predicted-translation-mm", nargs=3, type=float, required=True)
    parser.add_argument(
        "--predicted-rotation-axis-angle-deg", nargs=3, type=float, required=True
    )
    parser.add_argument("--sign-basis", required=True)
    parser.add_argument(
        "--measurement-step",
        type=int,
        help=(
            "Trace step for the current earliest-failure frame; defaults to "
            "the first physical contact for legacy diagnostics."
        ),
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    output = Path(args.output_json).resolve()
    if output.exists():
        raise FileExistsError(output)
    physical_result_path = Path(args.physical_result_json).resolve()
    diagnostic_result_path = Path(args.diagnostic_result_json).resolve()
    physical_result = json.loads(physical_result_path.read_text(encoding="utf-8"))
    diagnostic_result = json.loads(diagnostic_result_path.read_text(encoding="utf-8"))
    physical_trace = result_trace_path(physical_result)
    diagnostic_trace = result_trace_path(diagnostic_result)
    if physical_trace is None or diagnostic_trace is None:
        raise ValueError("physical and diagnostic results require trace receipts")
    physical_trace = physical_trace.resolve()
    diagnostic_trace = diagnostic_trace.resolve()
    physical_video, _ = _video(physical_result, "physical")
    diagnostic_video, diagnostic_video_receipt = _video(
        diagnostic_result, "diagnostic"
    )
    reference = action_tensor_receipt(physical_trace)
    replay = action_tensor_receipt(diagnostic_trace)
    with np.load(physical_trace, allow_pickle=False) as left, np.load(
        diagnostic_trace, allow_pickle=False
    ) as right:
        left_actions = np.asarray(left["actions"])
        right_actions = np.asarray(right["actions"])
        exact = bool(
            left_actions.shape == right_actions.shape
            and np.array_equal(left_actions, right_actions)
        )
        maximum_difference = (
            float(
                np.max(
                    np.abs(left_actions.astype(float) - right_actions.astype(float))
                )
            )
            if left_actions.shape == right_actions.shape and left_actions.size
            else None
        )
        protocol = diagnostic_result["protocol"]
        correction = protocol["source_contact_frame_correction"]
        geometry = protocol["parameters"]["geometry_conditioned_handle_grasp"]
        first_contact = int(protocol["acquisition_latch"]["first_contact_step"])
        measurement_step = (
            first_contact
            if args.measurement_step is None
            else int(args.measurement_step)
        )
        if not 0 <= measurement_step < len(right_actions):
            raise ValueError("diagnostic measurement step is out of range")
        pot_pose = np.asarray(right["pot_poses"][measurement_step], dtype=float)
        cooktop_pose = np.asarray(
            right["cooktop_poses"][measurement_step]
            if "cooktop_poses" in right.files
            else diagnostic_result["terminal"]["cooktop_pose"],
            dtype=float,
        )
        target_contacts = {
            arm: _compose_pose(
                pot_pose,
                geometry[arm]["target_contact_frame_local"],
            )
            for arm in ("left", "right")
        }
        actual_wrists = {
            arm: np.asarray(right[f"{arm}_eef_poses"][measurement_step], dtype=float)
            for arm in ("left", "right")
        }
        target_wrists = {
            arm: np.asarray(
                right[f"desired_{arm}_eef_poses"][measurement_step], dtype=float
            )
            for arm in ("left", "right")
        }
        pads = {
            arm: np.asarray(
                right[f"{arm}_pad_centers_world"][measurement_step], dtype=float
            )
            for arm in ("left", "right")
        }
        pad_axes = {
            arm: np.asarray(
                right[f"{arm}_pad_axes_world"][measurement_step], dtype=float
            )
            for arm in ("left", "right")
        }
        control_vectors = {
            arm: target_wrists[arm][:3] - actual_wrists[arm][:3]
            for arm in ("left", "right")
        }
        if (
            "local_mpc_active" in right.files
            and bool(right["local_mpc_active"][measurement_step])
        ):
            active_arm = _local_mpc_active_arm(right, measurement_step)
            control_vectors[active_arm] = np.asarray(
                right["local_mpc_translation_control_world_m"][measurement_step],
                dtype=float,
            )
    if not exact or maximum_difference != 0.0:
        raise ValueError("diagnostic replay actions are not exactly identical")
    render = diagnostic_result.get("protocol", {}).get("render_diagnostic", {})
    overlay_source = render.get("overlay", {})
    overlay_names = (
        "pot_body_frame",
        "cooktop_target_frame",
        "left_handle_contact_frame",
        "right_handle_contact_frame",
        "left_gripper_wrist_frames",
        "right_gripper_wrist_frames",
        "left_pad_centers_axes",
        "right_pad_centers_axes",
        "left_jaw_closing_line",
        "right_jaw_closing_line",
        "signed_residual_vectors",
        "signed_control_vectors",
        "screen_space_color_legend",
    )
    overlay = {name: overlay_source.get(name) is True for name in overlay_names}
    translation = target_wrists["left"][:3] - actual_wrists["left"][:3]
    rotation = _axis_angle_deg(
        actual_wrists["left"][3:], target_wrists["left"][3:]
    )
    frame_receipt = {
        "pot_body_frame": pot_pose.tolist(),
        "cooktop_target_frame": cooktop_pose.tolist(),
        "handle_contact_frames": {
            arm: target_contacts[arm].tolist() for arm in ("left", "right")
        },
        "gripper_wrist_frames": {
            arm: {
                "actual": actual_wrists[arm].tolist(),
                "desired": target_wrists[arm].tolist(),
            }
            for arm in ("left", "right")
        },
        "pad_centers": {
            arm: pads[arm].tolist() for arm in ("left", "right")
        },
        "pad_axes": {
            arm: pad_axes[arm].tolist() for arm in ("left", "right")
        },
        "jaw_closing_lines": {
            arm: (pads[arm][1] - pads[arm][0]).tolist()
            for arm in ("left", "right")
        },
        "signed_residual_vectors_world_m": {
            arm: (target_contacts[arm][:3] - pads[arm].mean(axis=0)).tolist()
            for arm in ("left", "right")
        },
        "signed_control_vectors_world_m": {
            arm: control_vectors[arm].tolist() for arm in ("left", "right")
        },
    }
    receipt = {
        "schema_version": 1,
        "classification": "action_identical_render_diagnostic",
        "physical_attempt": {
            "request_id": args.physical_request_id,
            "result": _artifact(physical_result_path),
            "trace": _artifact(physical_trace),
            "video": _artifact(physical_video),
        },
        "diagnostic": {
            "result": _artifact(diagnostic_result_path),
            "trace": _artifact(diagnostic_trace),
            "video": {
                **_artifact(diagnostic_video),
                "codec": diagnostic_video_receipt["codec"],
                "frame_count": diagnostic_video_receipt["frame_count"],
                "full_decode_returncode": 0,
            },
        },
        "action_parity": {
            "reference_shape": reference["shape"],
            "replay_shape": replay["shape"],
            "reference_actions_bytes_sha256": reference["bytes_sha256"],
            "replay_actions_bytes_sha256": replay["bytes_sha256"],
            "exactly_equal": exact,
            "maximum_absolute_difference": maximum_difference,
        },
        "protocol": {
            "physics_or_controller_changes": False,
            "causal_mechanism_attempt_consumed": False,
            "training_eligible": False,
            "attempt_identity": diagnostic_result.get("protocol", {}).get(
                "attempt_identity"
            ),
        },
        "overlay": overlay,
        "frame_receipt": frame_receipt,
        "measured_residuals": {
            "sample_step": measurement_step,
            "first_physical_contact_step": first_contact,
            "signed_translation_residual_world_m": translation.tolist(),
            "translation_norm_m": float(np.linalg.norm(translation)),
            "signed_rotation_residual_axis_angle_deg": rotation,
            "rotation_norm_deg": float(np.linalg.norm(rotation)),
            "jaw_midpoint_to_target_contact_world_m": (
                target_contacts["left"][:3] - pads["left"].mean(axis=0)
            ).tolist(),
            "target_contact_frame_world": target_contacts["left"].tolist(),
            "actual_pad_centers_world": pads["left"].tolist(),
            "actual_pad_axes_world": pad_axes["left"].tolist(),
            "actual_wrist_pose": actual_wrists["left"].tolist(),
            "target_wrist_pose": target_wrists["left"].tolist(),
        },
        "next_mechanism_prediction": {
            "mechanism_id": args.next_mechanism_id,
            "signed_translation_mm": args.predicted_translation_mm,
            "signed_rotation_axis_angle_deg": args.predicted_rotation_axis_angle_deg,
            "sign_basis": args.sign_basis,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        "PUTPOT_DIAGNOSTIC_RECEIPT="
        + json.dumps(
            {"path": str(output), "sha256": sha256_file(output)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
