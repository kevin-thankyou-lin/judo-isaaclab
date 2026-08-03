"""Fail-closed audit of the 40 selected PutMarker campaign records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from run_three_task_asset_campaign import _atomic_json, _task_success, validate_demo


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_provenance(value: Any, errors: list[str], prefix: str = "provenance") -> int:
    verified = 0
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            path = Path(value["path"])
            if not path.is_file():
                errors.append(f"{prefix}: missing {path}")
            elif _sha256(path) != value["sha256"]:
                errors.append(f"{prefix}: hash mismatch {path}")
            else:
                verified += 1
        for key, child in value.items():
            verified += _check_provenance(child, errors, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            verified += _check_provenance(child, errors, f"{prefix}[{index}]")
    return verified


def _video_receipt(path: Path) -> dict[str, Any]:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=codec_name,nb_read_frames,width,height",
            "-of", "json", str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout).get("streams", []) if probe.returncode == 0 else []
    stream = streams[0] if len(streams) == 1 else {}
    return {
        "path": str(path),
        "sha256": _sha256(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "probe_returncode": probe.returncode,
        "codec": stream.get("codec_name"),
        "decoded_frames": int(stream.get("nb_read_frames", 0) or 0),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "full_decode_returncode": decode.returncode,
        "probe_stderr": probe.stderr.strip(),
        "decode_stderr": decode.stderr.strip(),
    }


def audit_pair(key: str, record: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    result_path = Path(record.get("result", ""))
    video_path = Path(record.get("video", ""))
    demo_record = record.get("demonstration", {})
    demo_path = Path(demo_record.get("path", ""))
    if not result_path.is_file():
        return {"pair": key, "errors": [f"missing result {result_path}"]}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checks = result.get("acceptance_checks", {})
    protocol = result.get("protocol", {})
    if result.get("status") != "passed" or not _task_success(result):
        errors.append("coded task success/result status failed")
    if not checks or not all(value is True for value in checks.values()):
        errors.append(f"acceptance checks failed: {checks}")
    if not (
        protocol.get("candidate_sampling") is False
        and protocol.get("scene_resets") == 1
        and protocol.get("inter_stage_resets") == 0
        and protocol.get("teleports_after_reset") == 0
        and protocol.get("grasp_assistance")
        in {
            "none",
            "task_config:right=friction",
            "task_config:right=fixed_joint",
        }
    ):
        errors.append("deterministic continuous-rollout protocol failed")
    device = protocol.get("actual_device_receipt", {})
    actual = device.get("actual", {})
    devices = [actual.get("manager_environment"), actual.get("simulation_context")]
    devices.extend(actual.get("action_tensors", []))
    if not (
        device.get("matched") is True
        and device.get("expected") == "cpu"
        and device.get("requested") == "cpu"
        and devices
        and all(name == "cpu" for name in devices)
    ):
        errors.append(f"actual CPU receipt failed: {device}")
    if result.get("provenance", {}).get("target_dataset", {}).get("path") != record.get("dataset"):
        errors.append("selected target dataset does not match result provenance")
    provenance_files = _check_provenance(result.get("provenance", {}), errors)
    if not demo_path.is_file():
        errors.append(f"missing demonstration {demo_path}")
        demonstration = {}
    else:
        demonstration = validate_demo(demo_path, record["assets"])
        if demonstration["actions"] <= 0:
            errors.append("demonstration contains no actions")
        if demonstration["sha256"] != demo_record.get("sha256"):
            errors.append("demonstration hash differs from ledger")
        if demonstration["actions"] != demo_record.get("actions"):
            errors.append("demonstration action count differs from ledger")
    video = _video_receipt(video_path)
    expected_video = result.get("video", {})
    if not (
        video["size_bytes"] > 0
        and video["codec"] == "h264"
        and video["decoded_frames"] > 0
        and video["full_decode_returncode"] == 0
        and video["sha256"] == expected_video.get("sha256")
        and video["decoded_frames"] == int(expected_video.get("frame_count", -1))
    ):
        errors.append(f"video decode/hash receipt failed: {video}")
    return {
        "pair": key,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "accepted_source": record.get("accepted_source"),
        "result": str(result_path),
        "demonstration": demonstration,
        "video": video,
        "provenance_files_verified": provenance_files,
        "protocol": protocol,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    ledger_path = Path(args.ledger)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    selected = {
        key: value for key, value in ledger["pairs"].items() if value["task"] == "putmarker"
    }
    records = [audit_pair(key, value) for key, value in sorted(selected.items())]
    errors = []
    if len(records) != 40:
        errors.append(f"expected 40 PutMarker records, found {len(records)}")
    pending = [key for key, value in selected.items() if value.get("status") != "accepted"]
    if pending:
        errors.append(f"non-accepted records: {pending}")
    errors.extend(f"{row['pair']}: {error}" for row in records for error in row["errors"])
    report = {
        "status": "passed" if not errors else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ledger": str(ledger_path.resolve()),
        "ledger_sha256": _sha256(ledger_path),
        "summary": {
            "total": len(records),
            "passed": sum(not row["errors"] for row in records),
            "failed": sum(bool(row["errors"]) for row in records),
            "decoded_frames": sum(row.get("video", {}).get("decoded_frames", 0) for row in records),
            "provenance_files_verified": sum(row.get("provenance_files_verified", 0) for row in records),
        },
        "errors": errors,
        "pairs": records,
    }
    _atomic_json(Path(args.output_json), report)
    print("PUTMARKER_CAMPAIGN_AUDIT=" + json.dumps(report["summary"], sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
