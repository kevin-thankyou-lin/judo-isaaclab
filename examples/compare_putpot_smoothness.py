"""Compare a smooth PutPot trace with the locked segmented baseline."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from judo_isaaclab.put_pot import cartesian_smoothness_metrics


def _digest(path: str | Path) -> str:
    digest = sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(trace, start: int, end: int) -> dict[str, float | int]:
    if not 0 <= start < end <= len(trace["desired_left_eef_poses"]):
        raise ValueError(f"invalid trace window [{start}, {end})")
    return cartesian_smoothness_metrics(
        trace["desired_left_eef_poses"][start:end],
        trace["desired_right_eef_poses"][start:end],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trace", required=True)
    parser.add_argument("--candidate-trace", required=True)
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-transport-start", type=int, default=200)
    parser.add_argument("--baseline-segmented-end", type=int, default=430)
    parser.add_argument("--baseline-centered-end", type=int, default=610)
    args = parser.parse_args()

    baseline = np.load(args.baseline_trace)
    candidate = np.load(args.candidate_trace)
    stages = np.asarray(candidate["program_stages"]).astype(str)
    smooth_indices = np.flatnonzero(stages == "smooth_bimanual_transport")
    if not len(smooth_indices) or not np.all(np.diff(smooth_indices) == 1):
        raise ValueError("candidate trace has no single contiguous smooth transport")
    start = int(smooth_indices[0])
    end = int(smooth_indices[-1]) + 1
    baseline_segmented = _metrics(
        baseline, args.baseline_transport_start, args.baseline_segmented_end
    )
    baseline_centered = _metrics(
        baseline, args.baseline_transport_start, args.baseline_centered_end
    )
    candidate_metrics = _metrics(candidate, start, end)
    with open(args.candidate_result, encoding="utf-8") as stream:
        result = json.load(stream)
    result_metrics = result.get("metrics", {}).get("transport_plan", {})
    checks = {
        "baseline_is_segmented_a8b7c79": args.baseline_revision.startswith("a8b7c79"),
        "candidate_single_contiguous_transport": len(smooth_indices) == end - start,
        "candidate_has_no_internal_stops": candidate_metrics["internal_stop_count"] == 0,
        "candidate_jerk_below_segmented_baseline": (
            candidate_metrics["peak_jerk_mps3"]
            < baseline_segmented["peak_jerk_mps3"]
        ),
        "candidate_path_shorter_than_old_centered_sequence": (
            candidate_metrics["path_length_m"]
            < baseline_centered["path_length_m"]
        ),
        "candidate_result_matches_trace": (
            result_metrics.get("internal_stop_count")
            == candidate_metrics["internal_stop_count"]
            and abs(
                float(result_metrics.get("peak_jerk_mps3", float("inf")))
                - candidate_metrics["peak_jerk_mps3"]
            )
            <= max(1.0e-3, 0.01 * candidate_metrics["peak_jerk_mps3"])
        ),
    }
    value = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "revisions": {
            "baseline": args.baseline_revision,
            "candidate": args.candidate_revision,
        },
        "artifacts": {
            "baseline_trace": {
                "path": str(Path(args.baseline_trace).resolve()),
                "sha256": _digest(args.baseline_trace),
            },
            "candidate_trace": {
                "path": str(Path(args.candidate_trace).resolve()),
                "sha256": _digest(args.candidate_trace),
            },
            "candidate_result": {
                "path": str(Path(args.candidate_result).resolve()),
                "sha256": _digest(args.candidate_result),
            },
        },
        "windows": {
            "baseline_segmented": [args.baseline_transport_start, args.baseline_segmented_end],
            "baseline_centered": [args.baseline_transport_start, args.baseline_centered_end],
            "candidate_smooth": [start, end],
        },
        "metrics": {
            "baseline_segmented": baseline_segmented,
            "baseline_centered_sequence": baseline_centered,
            "candidate_smooth": candidate_metrics,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PUTPOT_SMOOTHNESS=" + json.dumps(value, sort_keys=True))
    if value["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
