"""Fail-closed independent audit for one PutPot quality-wave artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from judo_isaaclab.putpot_quality import (  # noqa: E402
    audit_broad_contact,
    audit_object_first_motion,
    audit_perturbation_outcomes,
    audit_release_and_return,
    audit_swept_self_collision,
    deterministic_perturbation_cases,
    load_quality_config,
    write_immutable_receipt,
)


TRACE_FIELDS = {
    "actions",
    "pot_poses",
    "left_eef_poses",
    "right_eef_poses",
    "left_finger_forces_n",
    "right_finger_forces_n",
    "left_pad_fractions",
    "right_pad_fractions",
    "partial_trace",
}
CONTACT_FIELDS = {
    "left_contact_area_fractions",
    "right_contact_area_fractions",
    "left_flush_angles_deg",
    "right_flush_angles_deg",
    "supported",
    "left_open",
    "right_open",
    "stage_events",
    "object_first_start_step",
    "object_first_end_step",
    "left_start_m",
    "right_start_m",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _media_probe(path: Path) -> dict[str, Any]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
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
            str(path),
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        return {"passed": False, "probe_error": probe.stderr.strip()}
    value = json.loads(probe.stdout)
    streams = value.get("streams", [])
    stream = streams[0] if streams else {}
    return {
        "passed": bool(
            stream.get("codec_name") == "h264"
            and int(stream.get("nb_read_frames", 0)) > 0
            and decoded.returncode == 0
        ),
        "codec": stream.get("codec_name"),
        "frame_count": int(stream.get("nb_read_frames", 0)),
        "full_decode_returncode": decoded.returncode,
    }


def _hdf5_receipt(
    path: Path, trace_actions: np.ndarray, result: Mapping[str, Any]
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if "data/demo_0/actions" not in handle or "data/demo_0/states" not in handle:
            raise ValueError("demonstration is missing actions or states")
        demo = handle["data/demo_0"]
        actions = np.asarray(demo["actions"], dtype=np.float32)
        state_lengths: list[int] = []

        def collect(_name, item):
            if isinstance(item, h5py.Dataset):
                state_lengths.append(int(item.shape[0]))

        demo["states"].visititems(collect)
        if not state_lengths or len(set(state_lengths)) != 1:
            raise ValueError("demonstration state arrays are missing or misaligned")
        state_count = state_lengths[0]
        asset_paths = json.loads(handle["data"].attrs["ASSETS_INSTANCE_PATHS"])
        success = bool(demo.attrs.get("success", False))
    expected_assets = result.get("provenance", {}).get("target_assets", {})
    asset_match = bool(
        set(asset_paths) == set(expected_assets)
        and all(
            Path(expected_assets[name]["path"]).name == Path(asset_paths[name]).name
            for name in asset_paths
        )
    )
    return {
        "passed": bool(
            success
            and actions.shape == trace_actions.shape
            and np.array_equal(actions, trace_actions)
            and state_count == len(actions) + 1
            and asset_match
        ),
        "success": success,
        "actions": len(actions),
        "states": state_count,
        "actions_match_trace": bool(
            actions.shape == trace_actions.shape
            and np.array_equal(actions, trace_actions)
        ),
        "assets_match_result": asset_match,
        "sha256": sha256_file(path),
    }


def _collision_inputs(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, float], list, dict[str, list[str]]]:
    with np.load(path, allow_pickle=False) as telemetry:
        center_names = sorted(
            name.removeprefix("center__")
            for name in telemetry.files
            if name.startswith("center__")
        )
        radius_names = sorted(
            name.removeprefix("radius__")
            for name in telemetry.files
            if name.startswith("radius__")
        )
        if center_names != radius_names:
            raise ValueError("collision telemetry centers and radii do not match")
        centers = {
            name: np.asarray(telemetry[f"center__{name}"], dtype=np.float64)
            for name in center_names
        }
        radii = {
            name: float(np.asarray(telemetry[f"radius__{name}"]).reshape(()))
            for name in radius_names
        }
        structural = (
            json.loads(str(np.asarray(telemetry["structural_adjacencies_json"]).item()))
            if "structural_adjacencies_json" in telemetry.files
            else []
        )
        groups = (
            json.loads(str(np.asarray(telemetry["component_groups_json"]).item()))
            if "component_groups_json" in telemetry.files
            else {}
        )
    return centers, radii, structural, groups


def _check_result(result: Mapping[str, Any], config_sha256: str) -> dict[str, Any]:
    checks = result.get("checks", {})
    acceptance = result.get("acceptance_checks", {})
    required = (
        "coded_task_success",
        "accepted_task_success",
        "bimanual_pick_observed",
        "bimanual_transport_completed",
        "pot_released",
        "stable_support_window",
        "one_reset",
        "zero_inter_stage_resets",
    )
    quality = result.get("protocol", {}).get("quality_config")
    missing = [name for name in required if checks.get(name) is not True]
    failed_acceptance = [name for name, passed in acceptance.items() if passed is not True]
    return {
        "passed": bool(
            result.get("status") == "passed"
            and not missing
            and not failed_acceptance
            and isinstance(quality, Mapping)
            and quality.get("sha256") == config_sha256
        ),
        "result_status": result.get("status"),
        "missing_or_failed_checks": missing,
        "failed_acceptance_checks": failed_acceptance,
        "quality_config_sha256": (
            quality.get("sha256") if isinstance(quality, Mapping) else None
        ),
    }


def audit_bundle(
    *,
    result_json: str | Path,
    trace_npz: str | Path,
    contact_telemetry_npz: str | Path,
    collision_telemetry_npz: str | Path,
    demo_hdf5: str | Path,
    video: str | Path,
    runtime_receipt_json: str | Path,
    quality_config_json: str | Path,
    perturbation_outcomes_json: str | Path,
    media_probe: Callable[[Path], Mapping[str, Any]] = _media_probe,
) -> dict[str, Any]:
    """Audit a complete nominal bundle; return an explicit terminal receipt."""

    inputs = {
        "result_json": Path(result_json),
        "trace_npz": Path(trace_npz),
        "contact_telemetry_npz": Path(contact_telemetry_npz),
        "collision_telemetry_npz": Path(collision_telemetry_npz),
        "demo_hdf5": Path(demo_hdf5),
        "video": Path(video),
        "runtime_receipt_json": Path(runtime_receipt_json),
        "perturbation_outcomes_json": Path(perturbation_outcomes_json),
    }
    missing_artifacts = sorted(name for name, path in inputs.items() if not path.is_file())
    if missing_artifacts:
        return {
            "status": "failed",
            "terminal": True,
            "reason": "missing_quality_artifacts",
            "missing_artifacts": missing_artifacts,
            "pair_owner_action": "produce the named artifacts without reusing another lane",
        }

    config = load_quality_config(quality_config_json)
    result = _read_json(inputs["result_json"])
    runtime = _read_json(inputs["runtime_receipt_json"])
    outcomes_value = _read_json(inputs["perturbation_outcomes_json"])
    with np.load(inputs["trace_npz"], allow_pickle=False) as trace_file:
        missing_trace = sorted(TRACE_FIELDS - set(trace_file.files))
        trace = {name: np.asarray(trace_file[name]) for name in trace_file.files}
    with np.load(inputs["contact_telemetry_npz"], allow_pickle=False) as contact_file:
        missing_contact = sorted(CONTACT_FIELDS - set(contact_file.files))
        contact = {name: np.asarray(contact_file[name]) for name in contact_file.files}
    missing_telemetry = {
        "trace": missing_trace,
        "contact": missing_contact,
    }
    if missing_trace or missing_contact:
        return {
            "status": "failed",
            "terminal": True,
            "reason": "missing_required_quality_telemetry",
            "missing_telemetry_fields": missing_telemetry,
            "pair_owner_action": (
                "instrument the physical runner and retry this pair; do not infer "
                "contact area, flush angle, support, gripper state, or phase boundaries"
            ),
        }
    if bool(np.asarray(trace["partial_trace"]).reshape(())):
        return {
            "status": "failed",
            "terminal": True,
            "reason": "partial_physical_trace",
            "pair_owner_action": "retry after diagnosing the interrupted worker",
        }

    count = len(trace["actions"])
    aligned = all(
        len(trace[name]) == count
        for name in TRACE_FIELDS - {"partial_trace"}
    ) and all(
        len(contact[name]) == count
        for name in (
            "left_contact_area_fractions",
            "right_contact_area_fractions",
            "left_flush_angles_deg",
            "right_flush_angles_deg",
            "supported",
            "left_open",
            "right_open",
        )
    )
    if not aligned:
        return {
            "status": "failed",
            "terminal": True,
            "reason": "quality_telemetry_time_axes_are_not_aligned",
            "pair_owner_action": "fix recorder alignment before another acceptance audit",
        }

    broad = audit_broad_contact(
        left_forces_n=trace["left_finger_forces_n"],
        right_forces_n=trace["right_finger_forces_n"],
        left_pad_fractions=trace["left_pad_fractions"],
        right_pad_fractions=trace["right_pad_fractions"],
        left_contact_area_fractions=contact["left_contact_area_fractions"],
        right_contact_area_fractions=contact["right_contact_area_fractions"],
        left_flush_angles_deg=contact["left_flush_angles_deg"],
        right_flush_angles_deg=contact["right_flush_angles_deg"],
        pot_positions_m=trace["pot_poses"][:, :3],
        config=config,
    )
    start = int(np.asarray(contact["object_first_start_step"]).reshape(()))
    end = int(np.asarray(contact["object_first_end_step"]).reshape(()))
    if not 0 <= start < end < count:
        motion = {"passed": False, "reason": "invalid_object_first_window"}
    else:
        motion = audit_object_first_motion(
            pot_poses=trace["pot_poses"][start : end + 1],
            left_eef_poses=trace["left_eef_poses"][start : end + 1],
            right_eef_poses=trace["right_eef_poses"][start : end + 1],
            planner_mode="object_first_rigid_weld",
            config=config,
        )
    release = audit_release_and_return(
        stage_events=[str(value) for value in contact["stage_events"].tolist()],
        supported=contact["supported"],
        left_open=contact["left_open"],
        right_open=contact["right_open"],
        left_positions_m=trace["left_eef_poses"][:, :3],
        right_positions_m=trace["right_eef_poses"][:, :3],
        left_start_m=contact["left_start_m"],
        right_start_m=contact["right_start_m"],
        config=config,
    )
    centers, radii, structural, groups = _collision_inputs(
        inputs["collision_telemetry_npz"]
    )
    collision = audit_swept_self_collision(
        component_centers_m=centers,
        component_radii_m=radii,
        structural_adjacencies=structural,
        component_groups=groups,
        config=config,
    )
    collision["step_count"] = (
        len(next(iter(centers.values()))) if centers else 0
    )
    if collision["step_count"] != count:
        collision["passed"] = False
        collision["reason"] = "collision_telemetry_does_not_cover_full_trace"
    result_check = _check_result(result, config.sha256)
    hdf5 = _hdf5_receipt(inputs["demo_hdf5"], trace["actions"], result)
    video_receipt = dict(media_probe(inputs["video"]))
    video_receipt["sha256"] = sha256_file(inputs["video"])
    provenance = result.get("provenance", {})
    quality_sidecars = result.get("protocol", {}).get("quality_sidecars", {})
    hashes = {
        "passed": bool(
            provenance.get("trace", {}).get("sha256")
            == sha256_file(inputs["trace_npz"])
            and provenance.get("demonstration", {}).get("sha256")
            == hdf5["sha256"]
            and result.get("video", {}).get("sha256") == video_receipt["sha256"]
            and quality_sidecars.get("contact", {}).get("sha256")
            == sha256_file(inputs["contact_telemetry_npz"])
            and quality_sidecars.get("collision", {}).get("sha256")
            == sha256_file(inputs["collision_telemetry_npz"])
        ),
        "trace_sha256": sha256_file(inputs["trace_npz"]),
        "demo_sha256": hdf5["sha256"],
        "video_sha256": video_receipt["sha256"],
        "contact_telemetry_sha256": sha256_file(
            inputs["contact_telemetry_npz"]
        ),
        "collision_telemetry_sha256": sha256_file(
            inputs["collision_telemetry_npz"]
        ),
    }
    worker = {
        "passed": bool(
            runtime.get("persistent") is False
            and runtime.get("shutdown", {}).get("completion")
            == "process_exit_observed"
        ),
        "persistent": runtime.get("persistent"),
        "shutdown_completion": runtime.get("shutdown", {}).get("completion"),
        "pid": runtime.get("pid"),
    }
    cases = deterministic_perturbation_cases(
        config, joint_dof=int(outcomes_value.get("joint_dof", 0))
    )
    perturbation = audit_perturbation_outcomes(
        cases, outcomes_value.get("outcomes", []), config
    )
    checks = {
        "nominal_task": result_check,
        "broad_contact": broad,
        "object_first_motion": motion,
        "release_and_return": release,
        "swept_collision": collision,
        "hdf5": hdf5,
        "video": video_receipt,
        "hashes": hashes,
        "clean_worker": worker,
        "perturbation": perturbation,
    }
    failed = [name for name, value in checks.items() if value.get("passed") is not True]
    return {
        "schema_version": 1,
        "status": "accepted" if not failed else "failed",
        "terminal": True,
        "failed_checks": failed,
        "quality_config": config.receipt(),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "checks": checks,
        "pair_owner_action": (
            None
            if not failed
            else "diagnose the first failed quality check and retry this pair"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--trace-npz", required=True)
    parser.add_argument("--contact-telemetry-npz", required=True)
    parser.add_argument("--collision-telemetry-npz", required=True)
    parser.add_argument("--demo-hdf5", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--runtime-receipt-json", required=True)
    parser.add_argument("--quality-config-json", required=True)
    parser.add_argument("--perturbation-outcomes-json", required=True)
    parser.add_argument("--receipt-json", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = audit_bundle(
            result_json=args.result_json,
            trace_npz=args.trace_npz,
            contact_telemetry_npz=args.contact_telemetry_npz,
            collision_telemetry_npz=args.collision_telemetry_npz,
            demo_hdf5=args.demo_hdf5,
            video=args.video,
            runtime_receipt_json=args.runtime_receipt_json,
            quality_config_json=args.quality_config_json,
            perturbation_outcomes_json=args.perturbation_outcomes_json,
        )
    except Exception as error:
        receipt = {
            "schema_version": 1,
            "status": "failed",
            "terminal": True,
            "reason": "quality_auditor_exception",
            "error": f"{type(error).__name__}: {error}",
            "pair_owner_action": "repair missing or malformed evidence and retry the audit",
        }
    write_immutable_receipt(args.receipt_json, receipt)
    print("PUTPOT_QUALITY_AUDIT=" + json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
