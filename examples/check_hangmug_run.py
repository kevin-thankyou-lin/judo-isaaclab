"""Fail-closed validation for a HangMug evidence invocation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


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
        if "HANGMUG_FINAL=" not in log:
            errors.append("log has no HANGMUG_FINAL sentinel")
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
        grasp_assistance = result.get("protocol", {}).get("grasp_assistance", "")
        if not str(grasp_assistance).startswith("task_config:"):
            errors.append("canonical task-configured grasp assistance was not used")
        for name in (
            "datagen_grasp_assist_configured",
            "left_grasp_assist_engaged",
        ):
            if checks.get(name) is not True:
                errors.append(f"missing grasp-assist evidence: {name}")
        if result.get("terminal", {}).get("task_success") is True:
            if checks.get("left_grasp_assist_released") is not True:
                errors.append("left grasp assist remained engaged after handover")
        if result.get("protocol", {}).get("candidate_sampling") is not False:
            errors.append("candidate sampling was not disabled")
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
        print("HANGMUG_CHECK_FAILED=" + json.dumps(errors), file=sys.stderr)
        raise SystemExit(2)
    print("HANGMUG_CHECK_PASSED")


if __name__ == "__main__":
    main()
