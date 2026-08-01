"""Fail-closed validation for a PutPot simulator harness invocation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


CENTER_TOLERANCE_M = 0.03
COLLISION_CLEARANCE_M = 0.025


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--trace-npz", required=True)
    parser.add_argument("--video")
    args = parser.parse_args()
    errors = []
    if not os.path.isfile(args.log):
        errors.append(f"missing log: {args.log}")
        log = ""
    else:
        with open(args.log, encoding="utf-8", errors="replace") as stream:
            log = stream.read()
        if "Traceback (most recent call last)" in log:
            errors.append("log contains a Python traceback")
        if "PUTPOT_FINAL=" not in log:
            errors.append("log has no PUTPOT_FINAL sentinel")
    if not os.path.isfile(args.result_json):
        errors.append(f"missing result JSON: {args.result_json}")
        result = {}
    else:
        try:
            with open(args.result_json, encoding="utf-8") as stream:
                result = json.load(stream)
        except (OSError, ValueError) as error:
            result = {}
            errors.append(f"invalid result JSON: {error}")
        if result.get("status") != "passed":
            errors.append(f"result status is {result.get('status')!r}, not 'passed'")
        checks = result.get("acceptance_checks", result.get("checks", {}))
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            errors.append(f"failed result checks: {failed}")
        expected_failure = "expected_coded_task_failure" in checks
        if not expected_failure:
            center_error = result.get("metrics", {}).get("center_error_m")
            if not isinstance(center_error, (int, float)):
                errors.append("missing numeric metrics.center_error_m")
            elif center_error > CENTER_TOLERANCE_M:
                errors.append(
                    f"center error {center_error:.6f} m exceeds {CENTER_TOLERANCE_M:.2f} m"
                )
            result_checks = result.get("checks", {})
            if result_checks.get("centered_on_cooktop") is not True:
                errors.append("checks.centered_on_cooktop is not true")
            if result_checks.get("accepted_task_success") is not True:
                errors.append("checks.accepted_task_success is not true")
            if result_checks.get("smooth_collision_aware_transport") is not True:
                errors.append("checks.smooth_collision_aware_transport is not true")
            if result_checks.get("transport_no_internal_stops") is not True:
                errors.append("checks.transport_no_internal_stops is not true")
            adapted_target = result.get("direct_replay_baseline") is not None
            if (
                adapted_target
                and result_checks.get("bimanual_transport_completed") is not True
            ):
                errors.append("checks.bimanual_transport_completed is not true")
            transport = result.get("metrics", {}).get("transport_plan", {})
            internal_stops = transport.get("internal_stop_count")
            if internal_stops != 0:
                errors.append(
                    f"transport internal stop count is {internal_stops!r}, not 0"
                )
            clearance = transport.get("minimum_cooktop_clearance_m")
            if not isinstance(clearance, (int, float)):
                errors.append("missing numeric transport cooktop clearance")
            elif clearance + 1.0e-9 < COLLISION_CLEARANCE_M:
                errors.append(
                    f"transport clearance {clearance:.6f} m is below "
                    f"{COLLISION_CLEARANCE_M:.3f} m"
                )
            executed_clearance = result.get("metrics", {}).get(
                "transport_executed", {}
            ).get("minimum_cooktop_clearance_m")
            if not isinstance(executed_clearance, (int, float)):
                errors.append("missing numeric executed transport cooktop clearance")
            elif executed_clearance + 1.0e-9 < 0.0:
                errors.append(
                    f"executed transport clearance {executed_clearance:.6f} m is negative"
                )
    if not os.path.isfile(args.trace_npz) or os.path.getsize(args.trace_npz) == 0:
        errors.append(f"missing or empty trace: {args.trace_npz}")
    if args.video:
        if not os.path.isfile(args.video) or os.path.getsize(args.video) == 0:
            errors.append(f"missing or empty video: {args.video}")
        else:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", args.video],
                capture_output=True,
                text=True,
            )
            if probe.returncode != 0 or probe.stdout.strip() != "h264":
                errors.append("video is not a readable H.264 stream")
            decoded = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", args.video, "-f", "null", "-"],
                capture_output=True,
                text=True,
            )
            if decoded.returncode != 0:
                errors.append("video failed full decode")
    if errors:
        print("PUTPOT_CHECK_FAILED=" + json.dumps(errors), file=sys.stderr)
        raise SystemExit(2)
    print("PUTPOT_CHECK_PASSED")


if __name__ == "__main__":
    main()
