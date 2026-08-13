"""Resume the one-source, same-index Task2 HangMug campaign serially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_hangmug_skill_program import (  # noqa: E402
    PROVEN_CONTROL_DEFAULTS,
    _independent_terminal_hang_receipt,
    _source_dataset_receipt,
)
from run_putmarker_skill_program import _probe  # noqa: E402
from run_three_task_asset_campaign import validate_demo  # noqa: E402

SOURCE = Path("/home/linke/datasets/gear-dc-study/task2/teleop/mug_teacup_000000.hdf5")
SOURCE_SHA256 = "dbb2882af9d99f9042043f1e4a74944fc9907f057a30e885fcf68d02d31bcae5"
SOURCE_ACTIONS_SHA256 = "c36c7cba4c3221969771a69c75faaaebe6f4ca7fae7133ee8488a1721b016203"
SOURCE_PREFIX_STEPS = 393
CONTROLLER_SHA256 = "f383179e50239db34e449c75d1a4f2cc2a670631d041a7370957a136bab12fce"
LIVE_GAINS_SHA256 = "aa5ab8f58e558ca4acb182c9c3128effac161a6e6e83ace2c73f402cdd31b878"
OBJECTS = Path("/home/linke/datasets/gear-dc-study/task2/objects")
GEAR_REPO = Path("/home/linke/Projects/gear-dc-study-privileged-mug-datagen-20260812")
PYTHON = Path("/home/linke/miniforge3/envs/yam_lab/bin/python")
GUARD = GEAR_REPO / "scripts/run_local_isaac_guarded.sh"
KEYFRAMES = Path("results/task2/source/attempt_001_compact_replay/source_keyframes.json")
TIMING = Path("results/task2/pairs/000001/attempt_004_handover_confirm_hold/accepted_runtime_timing.json")
RESULTS = Path("results/task2")
MIDDLE_ROW_BRANCHES = frozenset({"branch_layer_2_a", "branch_layer_2_b"})
BRANCH_SUFFIX_STRATEGY_FIELDS = frozenset({
    "require_broad_pad_contact",
    "post_handover_right_return_steps",
    "left_branch_point_steps",
    "post_handover_rest_observer_steps",
    "direct_rest_to_preinsert_steps",
    "post_release_return_to_rest_steps",
    "branch_orient_steps",
    "insert_clearance_m",
    "branch_approach_height_m",
    "branch_roll_offset_rad",
    "branch_support_fraction",
    "branch_support_seat_down_m",
    "stable_support_steps",
})
LD_LIBRARY_PATH = ":".join(
    (
        "/home/linke/miniforge3/envs/yam_lab/lib",
        "/usr/local/cuda-12.8/lib64",
        "/usr/local/cuda-12.8/lib64",
    )
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_middle_row_branch(selected_branch: str | None) -> None:
    if selected_branch not in MIDDLE_ROW_BRANCHES:
        raise RuntimeError(
            f"accepted hang must use a middle-row branch, got {selected_branch!r}"
        )


def _load(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _atomic_json(path: Path, value: dict, *, immutable: bool = False) -> None:
    if immutable and path.exists():
        raise FileExistsError(f"immutable receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _asset_pair(index: int) -> dict[str, Path]:
    pair = {
        "mug": OBJECTS / "MugHangable" / f"mug_teacup_{index:06d}",
        "mug_tree": OBJECTS / "ThreeLayerMugTree" / f"mug_tree_{index:06d}",
    }
    missing = [str(path) for path in pair.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Pair {index:06d} assets missing: {missing}")
    return pair


def _attempt_directory(index: int, label: str) -> Path:
    pair_root = RESULTS / "pairs" / f"{index:06d}"
    numbers = []
    for path in pair_root.glob("attempt_[0-9][0-9][0-9]_*"):
        try:
            numbers.append(int(path.name.split("_", 2)[1]))
        except ValueError:
            pass
    return pair_root / f"attempt_{max(numbers, default=0) + 1:03d}_{label}"


def _steady_state_seconds() -> int:
    timing = _load(TIMING)
    runtime = float(timing["accepted_runtime_seconds"])
    calculated = int(np.ceil(max(1.5 * runtime, runtime + 120.0)))
    if calculated != int(timing["steady_state_guard_seconds"]):
        raise RuntimeError("accepted runtime timing formula no longer matches its receipt")
    return calculated


def _common_workload(index: int, attempt: Path) -> list[str]:
    assets = _asset_pair(index)
    return [
        "env",
        f"PYTHONPATH={REPO_ROOT}:{GEAR_REPO}",
        f"LD_LIBRARY_PATH={LD_LIBRARY_PATH}",
        str(PYTHON),
        "examples/run_hangmug_skill_program.py",
        "--gear-repo", str(GEAR_REPO),
        "--source-dataset", str(SOURCE),
        "--objects-root", str(OBJECTS),
        "--target-mug-asset", str(assets["mug"]),
        "--target-tree-asset", str(assets["mug_tree"]),
        "--episode", "demo_0",
        "--expected-source-sha256", SOURCE_SHA256,
        "--expected-controller-gains-sha256", CONTROLLER_SHA256,
        "--device", "cpu",
        "--require-cpu-physics",
        "--grasp-assist-mechanism", "task_config",
        "--render",
        "--video", str(attempt / "video.mp4"),
        "--trace-npz", str(attempt / "trace.npz"),
        "--demo-hdf5", str(attempt / "demo.hdf5"),
        "--result-json", str(attempt / "result.json"),
    ]


def _guarded(attempt: Path, workload: list[str]) -> list[str]:
    return [str(GUARD), str(_steady_state_seconds()), str(attempt / "replay.log"), *workload]


def _classification_command(index: int, attempt: Path) -> list[str]:
    workload = _common_workload(index, attempt)
    workload[workload.index("--device"):workload.index("--device")] = [
        "--mode", "replay", "--classification-run",
    ]
    return _guarded(attempt, workload)


def _repair_command(
    index: int, attempt: Path, classification_result: Path, selection: dict,
    strategy: dict | None = None,
) -> list[str]:
    strategy = strategy or {}
    workload = _common_workload(index, attempt)
    arguments = [
        "--mode", "skill",
        "--source-keyframes", str(KEYFRAMES),
        "--direct-replay-result", str(classification_result),
        "--handover-confirm-steps",
        str(strategy.get("handover_confirm_steps", 12)),
    ]
    if strategy.get("require_broad_pad_contact"):
        arguments.append("--require-broad-pad-contact")
    if "pick_lift_margin_m" in strategy:
        arguments.extend([
            "--pick-lift-margin-m", str(strategy["pick_lift_margin_m"])
        ])
    if strategy.get("handover_handle_frame_transfer"):
        arguments.append("--handover-handle-frame-transfer")
    for field, option in (
        ("post_handover_right_return_steps", "--post-handover-right-return-steps"),
        ("left_branch_point_steps", "--left-branch-point-steps"),
        ("post_handover_rest_observer_steps", "--post-handover-rest-observer-steps"),
        ("direct_rest_to_preinsert_steps", "--direct-rest-to-preinsert-steps"),
        ("post_release_return_to_rest_steps", "--post-release-return-to-rest-steps"),
        ("branch_orient_steps", "--branch-orient-steps"),
        ("insert_clearance_m", "--insert-clearance-m"),
        ("branch_approach_height_m", "--branch-approach-height-m"),
        ("branch_roll_offset_rad", "--branch-roll-offset-rad"),
        ("branch_support_fraction", "--branch-support-fraction"),
        ("branch_support_seat_down_m", "--branch-support-seat-down-m"),
        ("stable_support_steps", "--stable-support-steps"),
        ("target_branch_rank", "--target-branch-rank"),
    ):
        if field in strategy:
            arguments.extend([option, str(strategy[field])])
    if selection["actual_repair_boundary"] == "reset":
        arguments.extend([
            "--handover-contact-settle-steps",
            str(strategy.get("handover_contact_settle_steps", 30)),
        ])
        if strategy.get("handover_contact_acquire_steps"):
            arguments.extend([
                "--handover-contact-acquire-steps",
                str(strategy["handover_contact_acquire_steps"]),
            ])
        if strategy.get("handover_contact_acquire_reference_steps"):
            arguments.extend([
                "--handover-contact-acquire-reference-steps",
                str(strategy["handover_contact_acquire_reference_steps"]),
            ])
        if "handover_contact_acquire_target_mug_position_m" in strategy:
            arguments.extend([
                "--handover-contact-acquire-target-mug-position-m",
                *map(
                    str,
                    strategy["handover_contact_acquire_target_mug_position_m"],
                ),
            ])
        if "handover_contact_acquire_target_mug_quaternion_wxyz" in strategy:
            arguments.extend([
                "--handover-contact-acquire-target-mug-quaternion-wxyz",
                *map(
                    str,
                    strategy["handover_contact_acquire_target_mug_quaternion_wxyz"],
                ),
            ])
        if strategy.get("handover_contact_acquire_reanchor_right_assist"):
            arguments.append("--handover-contact-acquire-reanchor-right-assist")
        if "handover_target_offset_m" in strategy:
            arguments.extend([
                "--handover-target-offset-m",
                *map(str, strategy["handover_target_offset_m"]),
            ])
        if "handover_target_local_pitch_rad" in strategy:
            arguments.extend([
                "--handover-target-local-pitch-rad",
                str(strategy["handover_target_local_pitch_rad"]),
            ])
        if "handover_target_local_roll_rad" in strategy:
            arguments.extend([
                "--handover-target-local-roll-rad",
                str(strategy["handover_target_local_roll_rad"]),
            ])
        if "handover_straddle_local_x_m" in strategy:
            arguments.extend([
                "--handover-straddle-local-x-m",
                str(strategy["handover_straddle_local_x_m"]),
            ])
        for field, option in (
            ("handover_orient_clearance_m", "--handover-orient-clearance-m"),
            ("handover_orient_steps", "--handover-orient-steps"),
            ("handover_standoff_outside_m", "--handover-standoff-outside-m"),
        ):
            if field in strategy:
                arguments.extend([option, str(strategy[field])])
        if "left_release_retreat_m" in strategy:
            arguments.extend([
                "--left-release-retreat-m",
                str(strategy["left_release_retreat_m"]),
            ])
        if "handover_post_release_lift_m" in strategy:
            arguments.extend([
                "--handover-post-release-lift-m",
                str(strategy["handover_post_release_lift_m"]),
                "--handover-post-release-lift-steps",
                str(strategy["handover_post_release_lift_steps"]),
            ])
    elif selection["actual_repair_boundary"] == "pick":
        if set(strategy) - BRANCH_SUFFIX_STRATEGY_FIELDS:
            raise ValueError("pair repair strategy is valid only from reset")
        arguments.append("--reuse-source-pick-prefix")
    else:
        raise RuntimeError(f"unsupported actual repair boundary: {selection}")
    workload[workload.index("--device"):workload.index("--device")] = arguments
    return _guarded(attempt, workload)


def _repair_selection(first_failed_stage: str, last_completed_stage: str | None) -> dict:
    ordered = (
        "pick", "handover", "alignment", "insertion_and_support", "release_and_hang"
    )
    if first_failed_stage not in ordered:
        raise ValueError(f"unknown failed semantic stage: {first_failed_stage!r}")
    if first_failed_stage == "pick":
        boundary, coarse = "reset", False
    elif first_failed_stage == "handover":
        boundary, coarse = "pick", False
    else:
        # Pick is the latest currently proven executable resume boundary. Run
        # one clean reset-to-finish suffix without claiming a later-stage resume.
        boundary, coarse = "pick", True
    return {
        "requested_failed_stage": first_failed_stage,
        "requested_last_completed_stage": last_completed_stage,
        "actual_repair_boundary": boundary,
        "coarse_fallback": coarse,
    }


def _repair_strategy(index: int) -> dict:
    path = RESULTS / "pairs" / f"{index:06d}" / "repair_candidate.json"
    if not path.is_file():
        return {}
    value = _load(path)
    allowed = {
        "force_semantic_regeneration",
        "handover_contact_settle_steps",
        "handover_contact_acquire_steps",
        "handover_contact_acquire_reference_steps",
        "handover_contact_acquire_target_mug_position_m",
        "handover_contact_acquire_target_mug_quaternion_wxyz",
        "handover_contact_acquire_reanchor_right_assist",
        "handover_confirm_steps",
        "handover_post_release_lift_m",
        "handover_post_release_lift_steps",
        "handover_target_offset_m",
        "handover_target_local_pitch_rad",
        "handover_target_local_roll_rad",
        "handover_straddle_local_x_m",
        "handover_orient_clearance_m",
        "handover_orient_steps",
        "handover_standoff_outside_m",
        "handover_handle_frame_transfer",
        "left_release_retreat_m",
        "pick_lift_margin_m",
        "require_broad_pad_contact",
        "post_handover_right_return_steps",
        "left_branch_point_steps",
        "post_handover_rest_observer_steps",
        "direct_rest_to_preinsert_steps",
        "post_release_return_to_rest_steps",
        "branch_orient_steps",
        "insert_clearance_m",
        "branch_approach_height_m",
        "branch_roll_offset_rad",
        "branch_support_fraction",
        "branch_support_seat_down_m",
        "stable_support_steps",
    }
    if set(value) - allowed:
        raise ValueError(f"unsupported repair candidate fields: {sorted(value)}")
    late_support_fields = set(BRANCH_SUFFIX_STRATEGY_FIELDS)
    handover_fields = allowed - late_support_fields - {
        "force_semantic_regeneration", "pick_lift_margin_m"
    }
    strategy = {}
    if "force_semantic_regeneration" in value:
        if value["force_semantic_regeneration"] is not True:
            raise ValueError(
                "semantic regeneration must be true when selected"
            )
    if "require_broad_pad_contact" in value:
        if value["require_broad_pad_contact"] is not True:
            raise ValueError(
                "broad pad contact must be true when selected"
            )
        strategy["require_broad_pad_contact"] = True
    if "handover_handle_frame_transfer" in value:
        enabled = value["handover_handle_frame_transfer"]
        if enabled is not True:
            raise ValueError(
                "handover handle-frame transfer must be true when selected"
            )
        strategy["handover_handle_frame_transfer"] = True
    if "pick_lift_margin_m" in value:
        margin = value["pick_lift_margin_m"]
        if (
            isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not np.isfinite(margin)
            or not 0.0 <= margin <= 0.03
        ):
            raise ValueError("pick lift margin must be in [0, 0.03] m")
        strategy["pick_lift_margin_m"] = float(margin)
    setup_steps = (
        value.get("post_handover_right_return_steps", 0),
        value.get("left_branch_point_steps", 0),
    )
    if any(
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or not 0 <= steps <= 120
        for steps in setup_steps
    ):
        raise ValueError("post-handover setup steps must be integers in [0, 120]")
    if bool(setup_steps[0]) != bool(setup_steps[1]):
        raise ValueError(
            "right return and left branch-point steps must be selected together"
        )
    if setup_steps[0]:
        strategy["post_handover_right_return_steps"] = setup_steps[0]
        strategy["left_branch_point_steps"] = setup_steps[1]
    simultaneous_setup = value.get("post_handover_rest_observer_steps", 0)
    if (
        isinstance(simultaneous_setup, bool)
        or not isinstance(simultaneous_setup, int)
        or not 0 <= simultaneous_setup <= 120
    ):
        raise ValueError("simultaneous rest/observer steps must be in [0, 120]")
    if simultaneous_setup and any(setup_steps):
        raise ValueError(
            "simultaneous and sequential post-handover setup cannot be combined"
        )
    if simultaneous_setup:
        strategy["post_handover_rest_observer_steps"] = simultaneous_setup
    direct_steps = (
        value.get("direct_rest_to_preinsert_steps", 0),
        value.get("post_release_return_to_rest_steps", 0),
    )
    if any(
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or not 0 <= steps <= 240
        for steps in direct_steps
    ):
        raise ValueError("direct choreography steps must be integers in [0, 240]")
    if bool(direct_steps[0]) != bool(direct_steps[1]):
        raise ValueError(
            "direct rest-to-preinsert and post-release return steps must be selected together"
        )
    if direct_steps[0]:
        if not simultaneous_setup or any(setup_steps):
            raise ValueError(
                "direct choreography requires simultaneous right-rest and left-observer setup"
            )
        if value.get("branch_orient_steps", 0):
            raise ValueError(
                "direct choreography forbids branch orientation subphases"
            )
        strategy["direct_rest_to_preinsert_steps"] = direct_steps[0]
        strategy["post_release_return_to_rest_steps"] = direct_steps[1]
    if "branch_orient_steps" in value:
        branch_orient_steps = value["branch_orient_steps"]
        if (
            isinstance(branch_orient_steps, bool)
            or not isinstance(branch_orient_steps, int)
            or not 0 <= branch_orient_steps <= 90
        ):
            raise ValueError("branch orient steps must be in [0, 90]")
        strategy["branch_orient_steps"] = branch_orient_steps
    if "insert_clearance_m" in value:
        clearance = value["insert_clearance_m"]
        if (
            isinstance(clearance, bool)
            or not isinstance(clearance, (int, float))
            or not np.isfinite(clearance)
            or not 0.03 <= clearance <= 0.10
        ):
            raise ValueError("insert clearance must be in [0.03, 0.10] m")
        strategy["insert_clearance_m"] = float(clearance)
    if "branch_approach_height_m" in value:
        height = value["branch_approach_height_m"]
        if (
            isinstance(height, bool)
            or not isinstance(height, (int, float))
            or not np.isfinite(height)
            or not 0.0 <= height <= 0.08
        ):
            raise ValueError("branch approach height must be in [0, 0.08] m")
        strategy["branch_approach_height_m"] = float(height)
    if "branch_roll_offset_rad" in value:
        roll = value["branch_roll_offset_rad"]
        if (
            isinstance(roll, bool)
            or not isinstance(roll, (int, float))
            or not np.isfinite(roll)
            or abs(roll) > np.pi / 2
        ):
            raise ValueError("branch roll offset must be within 90 degrees")
        strategy["branch_roll_offset_rad"] = float(roll)
    if "branch_support_fraction" in value:
        fraction = value["branch_support_fraction"]
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not np.isfinite(fraction)
            or not 0.25 <= fraction <= 0.75
        ):
            raise ValueError("branch support fraction must be in [0.25, 0.75]")
        strategy["branch_support_fraction"] = float(fraction)
    if "branch_support_seat_down_m" in value:
        seat_down = value["branch_support_seat_down_m"]
        if (
            isinstance(seat_down, bool)
            or not isinstance(seat_down, (int, float))
            or not np.isfinite(seat_down)
            or not 0.0 <= seat_down <= 0.03
        ):
            raise ValueError("branch support seat-down must be in [0, 0.03] m")
        strategy["branch_support_seat_down_m"] = float(seat_down)
    if "stable_support_steps" in value:
        steps = value["stable_support_steps"]
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or not 60 <= steps <= 240
        ):
            raise ValueError("stable support steps must be in [60, 240]")
        strategy["stable_support_steps"] = steps
    if not (set(value) & handover_fields):
        return strategy
    settle = value.get("handover_contact_settle_steps", 30)
    acquire = value.get("handover_contact_acquire_steps", 0)
    acquire_reference = value.get("handover_contact_acquire_reference_steps")
    acquire_target = value.get(
        "handover_contact_acquire_target_mug_position_m"
    )
    acquire_target_quaternion = value.get(
        "handover_contact_acquire_target_mug_quaternion_wxyz"
    )
    acquire_reanchor_right = value.get(
        "handover_contact_acquire_reanchor_right_assist", False
    )
    confirm = value.get("handover_confirm_steps", 12)
    offset = np.asarray(value.get("handover_target_offset_m", (0, 0, 0)), dtype=float)
    pitch = value.get("handover_target_local_pitch_rad", 0.0)
    roll = value.get("handover_target_local_roll_rad", 0.0)
    straddle = value.get("handover_straddle_local_x_m", 0.0)
    orient_clearance = value.get("handover_orient_clearance_m", 0.0)
    orient_steps = value.get("handover_orient_steps", 0)
    outside_standoff = value.get("handover_standoff_outside_m", 0.0)
    if not isinstance(settle, int) or not 0 <= settle <= 60:
        raise ValueError("handover contact settle must be an integer in [0, 60]")
    if not isinstance(acquire, int) or not 0 <= acquire <= 60:
        raise ValueError("handover contact acquire must be an integer in [0, 60]")
    if acquire_reference is not None and (
        isinstance(acquire_reference, bool)
        or not isinstance(acquire_reference, int)
        or not acquire <= acquire_reference <= 60
        or not acquire
    ):
        raise ValueError(
            "handover contact-acquire reference must be between the positive "
            "executed steps and 60"
        )
    if acquire_target is not None:
        acquire_target = np.asarray(acquire_target, dtype=float)
        if acquire_target.shape != (3,) or not np.isfinite(acquire_target).all():
            raise ValueError(
                "handover contact-acquire target must contain three finite values"
            )
        if not acquire:
            raise ValueError(
                "handover contact-acquire target requires positive acquisition steps"
            )
    if acquire_target_quaternion is not None:
        acquire_target_quaternion = np.asarray(
            acquire_target_quaternion, dtype=float
        )
        if (
            acquire_target_quaternion.shape != (4,)
            or not np.isfinite(acquire_target_quaternion).all()
            or abs(np.linalg.norm(acquire_target_quaternion) - 1.0) > 1.0e-6
        ):
            raise ValueError(
                "handover contact-acquire quaternion must be a unit quaternion"
            )
        if acquire_target is None:
            raise ValueError(
                "handover contact-acquire quaternion requires a target position"
            )
    if (
        "handover_contact_acquire_reanchor_right_assist" in value
        and acquire_reanchor_right is not True
    ):
        raise ValueError("right-assist contact reanchor must be true when selected")
    if acquire_reanchor_right:
        if acquire_target is None or acquire_target_quaternion is None:
            raise ValueError(
                "right-assist contact reanchor requires an explicit target pose"
            )
    if acquire_reference is not None and not acquire_reanchor_right:
        raise ValueError(
            "handover contact-acquire reference requires right-assist reanchor"
        )
    if not isinstance(confirm, int) or not 0 <= confirm <= 60:
        raise ValueError("handover confirmation must be an integer in [0, 60]")
    post_release_lift = value.get("handover_post_release_lift_m", 0.0)
    if (
        isinstance(post_release_lift, bool)
        or not isinstance(post_release_lift, (int, float))
        or not np.isfinite(post_release_lift)
        or not 0.0 <= post_release_lift <= 0.08
    ):
        raise ValueError("handover post-release lift must be in [0, 0.08] m")
    post_release_steps = value.get("handover_post_release_lift_steps", 0)
    if (
        isinstance(post_release_steps, bool)
        or not isinstance(post_release_steps, int)
        or not 0 <= post_release_steps <= 60
        or bool(post_release_steps) != bool(post_release_lift)
    ):
        raise ValueError(
            "handover post-release lift distance and steps must both be zero or positive"
        )
    if offset.shape != (3,) or not np.all(np.isfinite(offset)) or np.linalg.norm(offset) > 0.04:
        raise ValueError("handover target offset must be three finite values within 4 cm")
    if (
        isinstance(pitch, bool)
        or not isinstance(pitch, (int, float))
        or not np.isfinite(pitch)
        or abs(pitch) > np.pi / 4.0
    ):
        raise ValueError("handover target local pitch must be within 45 degrees")
    if (
        isinstance(roll, bool)
        or not isinstance(roll, (int, float))
        or not np.isfinite(roll)
        or abs(roll) > np.pi / 4.0
    ):
        raise ValueError("handover target local roll must be within 45 degrees")
    if (
        isinstance(straddle, bool)
        or not isinstance(straddle, (int, float))
        or not np.isfinite(straddle)
        or abs(straddle) > 0.14
    ):
        raise ValueError("handover local straddle correction must be within 14 cm")
    if (
        isinstance(orient_clearance, bool)
        or not isinstance(orient_clearance, (int, float))
        or not np.isfinite(orient_clearance)
        or not 0.0 <= orient_clearance <= 0.12
        or isinstance(orient_steps, bool)
        or not isinstance(orient_steps, int)
        or not 0 <= orient_steps <= 60
        or bool(orient_clearance) != bool(orient_steps)
        or isinstance(outside_standoff, bool)
        or not isinstance(outside_standoff, (int, float))
        or not np.isfinite(outside_standoff)
        or not 0.0 <= outside_standoff <= 0.12
        or bool(outside_standoff) and not orient_steps
    ):
        raise ValueError(
            "handover orient clearance/steps and outside standoff must be bounded"
        )
    strategy.update({
        "handover_contact_settle_steps": settle,
        "handover_contact_acquire_steps": acquire,
        "handover_confirm_steps": confirm,
        "handover_post_release_lift_m": float(post_release_lift),
        "handover_post_release_lift_steps": post_release_steps,
        "handover_target_offset_m": offset.tolist(),
    })
    if acquire_reference is not None:
        strategy["handover_contact_acquire_reference_steps"] = acquire_reference
    if acquire_target is not None:
        strategy["handover_contact_acquire_target_mug_position_m"] = (
            acquire_target.tolist()
        )
    if acquire_target_quaternion is not None:
        strategy["handover_contact_acquire_target_mug_quaternion_wxyz"] = (
            acquire_target_quaternion.tolist()
        )
    if acquire_reanchor_right:
        strategy["handover_contact_acquire_reanchor_right_assist"] = True
    if "handover_target_local_pitch_rad" in value:
        strategy["handover_target_local_pitch_rad"] = float(pitch)
    if "handover_target_local_roll_rad" in value:
        strategy["handover_target_local_roll_rad"] = float(roll)
    if orient_steps:
        strategy["handover_orient_clearance_m"] = float(orient_clearance)
        strategy["handover_orient_steps"] = orient_steps
    if outside_standoff:
        strategy["handover_standoff_outside_m"] = float(outside_standoff)
    if straddle:
        if not orient_steps:
            raise ValueError(
                "handover local straddle correction requires orient-first descent"
            )
        strategy["handover_straddle_local_x_m"] = float(straddle)
    if "left_release_retreat_m" in value:
        retreat = value["left_release_retreat_m"]
        if (
            isinstance(retreat, bool)
            or not isinstance(retreat, (int, float))
            or not np.isfinite(retreat)
            or not 0.02 <= retreat <= 0.12
        ):
            raise ValueError("left release retreat must be in [0.02, 0.12] m")
        strategy["left_release_retreat_m"] = float(retreat)
    return strategy


def _force_semantic_regeneration(index: int) -> bool:
    """Require a fresh reset-to-finish skill after direct replay diagnosis."""
    path = RESULTS / "pairs" / f"{index:06d}" / "repair_candidate.json"
    if not path.is_file():
        return False
    value = _load(path).get("force_semantic_regeneration", False)
    if value not in (False, True):
        raise ValueError("force_semantic_regeneration must be boolean")
    return value


def _worker_pids() -> list[int]:
    workers = []
    for process in Path("/proc").glob("[0-9]*"):
        try:
            tokens = process.joinpath("cmdline").read_bytes().split(b"\0")
            names = [Path(token.decode(errors="replace")).name for token in tokens if token]
            comm = process.joinpath("comm").read_text().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm in {"isaac-sim", "kit"} or "run_hangmug_skill_program.py" in names:
            workers.append(int(process.name))
    return sorted(workers)


def _require_zero_workers() -> None:
    workers = _worker_pids()
    if workers:
        raise RuntimeError(f"local Isaac worker gate failed: {workers}")


def _guard_lifecycle(attempt: Path) -> dict:
    required = {
        "exit": (attempt / "replay.log.exit", "GUARDED_RUN_EXIT=0"),
        "stall": (attempt / "replay.log.stall", "NO_STEP_PROGRESS_STALL_TRIGGERED=0"),
        "post_run": (attempt / "replay.log", "POST_RUN_ZERO_WORKER=PASS"),
    }
    missing = []
    for name, (path, marker) in required.items():
        text = path.read_text(errors="replace") if path.is_file() else ""
        if marker not in text.splitlines():
            missing.append(name)
    workers = _worker_pids()
    if missing or workers:
        raise RuntimeError(
            f"guard lifecycle incomplete: missing={missing}, workers={workers}"
        )
    return {
        "guarded_run_exit": 0,
        "no_step_progress_stall_triggered": False,
        "post_run_zero_worker": True,
        "workers_at_audit": [],
    }


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(values, dtype=bool))
    return None if not len(indices) else int(indices[0])


def _semantic_audit(
    demo: Path,
    assets: dict[str, Path],
    mug_init_z: float,
    reset_counts: dict[str, int],
) -> dict:
    """Recompute diagnostic stages and the independent terminal hang."""
    import h5py

    sys.path.insert(0, str(GEAR_REPO))
    from dc_study.datagen.hang_mug_status import (
        ORDERED_STAGES,
        hang_mug_status,
        reset_hang_mug_status,
    )

    with h5py.File(demo, "r") as handle:
        semantic = handle["data/demo_0/obs/semantic"]
        names = (
            "step", "mug_pose", "tree_pose", "left_grasp", "right_grasp",
            "task_success", "stage1", "stage2", "stage3",
        )
        values = {name: np.asarray(semantic[name]) for name in names}
        values["grasp_assist_engaged/left"] = np.asarray(
            semantic["grasp_assist_engaged/left"]
        )
        values["grasp_assist_engaged/right"] = np.asarray(
            semantic["grasp_assist_engaged/right"]
        )

    class Body:
        def __init__(self):
            self.data = SimpleNamespace(root_pos_w=None, root_quat_w=None)

    mug, tree = Body(), Body()
    env = SimpleNamespace(
        scene={"mug": mug, "mug_tree": tree},
        mug_init_z=float(mug_init_z),
        hang_mug_asset_directories={name: str(path) for name, path in assets.items()},
    )
    env.robot = SimpleNamespace(is_grasping=lambda: (
        np.asarray([env._left_grasp]), np.asarray([env._right_grasp])
    ))
    env.grasp_assists = {
        "left": SimpleNamespace(engaged=np.asarray([False])),
        "right": SimpleNamespace(engaged=np.asarray([False])),
    }
    env.get_task_success = lambda: np.asarray([env._task_success])
    reset_hang_mug_status(env)
    statuses = []
    samples = []
    maximum_opening = maximum_body = maximum_deep = 0.0
    maximum_deep_streak = 0
    for row in range(len(values["step"])):
        mug.data.root_pos_w = values["mug_pose"][row : row + 1, :3]
        mug.data.root_quat_w = values["mug_pose"][row : row + 1, 3:]
        tree.data.root_pos_w = values["tree_pose"][row : row + 1, :3]
        tree.data.root_quat_w = values["tree_pose"][row : row + 1, 3:]
        env._left_grasp = bool(values["left_grasp"][row])
        env._right_grasp = bool(values["right_grasp"][row])
        env._task_success = bool(values["task_success"][row])
        env.stage1_success = np.asarray([values["stage1"][row]])
        env.stage2_success = np.asarray([values["stage2"][row]])
        env.stage3_success = np.asarray([values["stage3"][row]])
        env.grasp_assists["left"].engaged = np.asarray(
            [values["grasp_assist_engaged/left"][row]]
        )
        env.grasp_assists["right"].engaged = np.asarray(
            [values["grasp_assist_engaged/right"][row]]
        )
        status = hang_mug_status(env)
        diagnostics = status["diagnostics"]
        maximum_opening = max(maximum_opening, diagnostics["opening_overlap_m"])
        maximum_body = max(maximum_body, diagnostics["body_post_overlap_proxy_m"])
        maximum_deep = max(maximum_deep, diagnostics["deepest_overlap_m"])
        maximum_deep_streak = max(
            maximum_deep_streak, diagnostics["consecutive_overlap_steps"]
        )
        statuses.append(status)
        samples.append(
            {
                "stage1": bool(values["stage1"][row]),
                "stage2": bool(values["stage2"][row]),
                "stage3": bool(values["stage3"][row]),
                "left_grasp": bool(values["left_grasp"][row]),
                "right_grasp": bool(values["right_grasp"][row]),
                "grasp_assist_engaged": {
                    "left": bool(values["grasp_assist_engaged/left"][row]),
                    "right": bool(values["grasp_assist_engaged/right"][row]),
                },
            }
        )
    firsts = {stage: _first_true([row[stage] for row in statuses]) for stage in ORDERED_STAGES}
    seen_missing = False
    for stage in ORDERED_STAGES:
        if firsts[stage] is None:
            seen_missing = True
        elif seen_missing:
            raise RuntimeError(f"diagnostic semantic stage order failed: {firsts}")
    completed_steps = [value for value in firsts.values() if value is not None]
    if completed_steps != sorted(completed_steps):
        raise RuntimeError(f"diagnostic semantic stage order failed: {firsts}")
    terminal = _independent_terminal_hang_receipt(
        statuses,
        samples,
        reset_counts,
    )
    if not terminal["passed"]:
        raise RuntimeError(f"independent terminal hang audit failed: {terminal['checks']}")
    final_diagnostics = statuses[-1]["diagnostics"]
    return {
        "ordered_semantic_stages": firsts,
        "terminal_30": {
            "passed": True,
            **terminal["checks"],
            "coded_stage_latches": terminal["coded_stage_latches"],
            "adapter_completed_stage_latches": terminal[
                "adapter_completed_stage_latches"
            ],
        },
        "contact_policy": {
            "contact_policy_held": all(row["contact_policy"] for row in statuses),
            "selected_branch": final_diagnostics["selected_branch"],
            "failure_reason": final_diagnostics["failure_reason"],
            "failure_step": final_diagnostics["failure_step"],
            "fallen": final_diagnostics["fallen"],
            "maximum_opening_overlap_m": maximum_opening,
            "maximum_body_post_overlap_proxy_m": maximum_body,
            "deepest_overlap_m": maximum_deep,
            "maximum_consecutive_deep_overlap_steps": maximum_deep_streak,
        },
    }


def _direct_choreography_audit(result: dict, trace) -> dict:
    """Recompute the direct phase/gripper contract from immutable trace data."""
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
    names = trace["semantic_waypoints"].astype(str)
    actions = np.asarray(trace["actions"], dtype=np.float64)
    desired = np.asarray(trace["desired_right_eef_poses"], dtype=np.float64)
    if len(names) != len(actions) or len(desired) != len(names):
        raise RuntimeError("direct choreography trace arrays are not row aligned")
    compressed = tuple(
        name
        for row, name in enumerate(names.tolist())
        if row == 0 or name != names[row - 1]
    )
    if "carrying_rest_observer" not in compressed:
        raise RuntimeError("direct choreography lacks the carrying-rest boundary")
    suffix = compressed[compressed.index("carrying_rest_observer") :]
    boundaries = {}
    for name in required:
        rows = np.flatnonzero(names == name)
        if not len(rows):
            raise RuntimeError(f"direct choreography lacks {name}")
        boundaries[name] = (int(rows[0]), int(rows[-1]))

    def direct_line(previous: str, segment: str) -> dict:
        start = desired[boundaries[previous][1], :3]
        rows = np.flatnonzero(names == segment)
        points = desired[rows, :3]
        direction = points[-1] - start
        squared = float(direction @ direction)
        if squared <= 0.0:
            raise RuntimeError(f"{segment} has no Cartesian displacement")
        fractions = (points - start) @ direction / squared
        residuals = np.linalg.norm(
            points - (start + fractions[:, None] * direction), axis=1
        )
        return {
            "rows": int(len(rows)),
            "maximum_line_residual_m": float(residuals.max(initial=0.0)),
            "fractions_monotone": bool(np.all(np.diff(fractions) >= -1.0e-9)),
            "endpoint_fraction": float(fractions[-1]),
            "passed": bool(
                residuals.max(initial=0.0) <= 1.0e-6
                and np.all(np.diff(fractions) >= -1.0e-9)
                and abs(float(fractions[-1]) - 1.0) <= 1.0e-6
            ),
        }

    outbound = direct_line("carrying_rest_observer", "direct_preinsert")
    returning = direct_line("right_release", "post_release_return")
    right_gripper = actions[:, 13]
    carrier_start = boundaries["carrying_rest_observer"][0]
    hold_end = boundaries["supported_release_hold"][1]
    release_start, release_end = boundaries["right_release"]
    return_start = boundaries["post_release_return"][0]
    deltas = np.diff(right_gripper[carrier_start:])
    opening_rows = np.flatnonzero(deltas < -1.0e-8)
    opening_runs = int(
        bool(len(opening_rows)) + np.count_nonzero(np.diff(opening_rows) > 1)
    )
    runner_contract = result.get("direct_phase_contract") or {}
    collision = result.get("direct_segment_collision_screening") or {}
    return_rest = result.get("post_release_right_rest") or {}
    checks = {
        "exact_named_suffix": suffix == required,
        "forbidden_intermediate_phases_absent": not any(
            np.any(names == name) for name in forbidden
        ),
        "outbound_one_direct_interpolation": outbound["passed"],
        "return_one_direct_interpolation": returning["passed"],
        "closed_carrier_through_supported_hold": bool(
            np.all(np.abs(right_gripper[carrier_start : hold_end + 1]) <= 1.0e-6)
        ),
        "one_final_monotone_opening": bool(
            opening_runs == 1
            and np.all(np.diff(right_gripper[release_start : release_end + 1]) <= 1.0e-9)
            and right_gripper[release_end] <= -0.04749
        ),
        "no_reclose_and_open_return": bool(
            np.all(deltas <= 1.0e-8)
            and np.all(right_gripper[return_start:] <= -0.04749)
        ),
        "runner_phase_receipt_passed": bool(runner_contract.get("passed")),
        "both_full_segment_screens_passed": bool(collision.get("passed")),
        "open_return_reached_demonstrated_rest": bool(return_rest.get("passed")),
    }
    if not all(checks.values()):
        raise RuntimeError(f"direct choreography audit failed: {checks}")
    return {
        "passed": True,
        "required_suffix": list(required),
        "observed_suffix": list(suffix),
        "phase_boundaries": {
            name: {"first_row": first, "last_row": last}
            for name, (first, last) in boundaries.items()
        },
        "outbound_interpolation": outbound,
        "post_release_return_interpolation": returning,
        "right_opening_transition_runs": opening_runs,
        "checks": checks,
        "collision_screening": collision,
        "post_release_right_rest": return_rest,
    }


def _handover_wave_audit(result: dict, trace) -> dict:
    """Recompute the wave handover ordering/contact contract from the trace."""
    receipt = result.get("handover_wave_contract") or {}
    names = trace["semantic_waypoints"].astype(str)
    actions = np.asarray(trace["actions"], dtype=np.float64)
    if len(names) != len(actions):
        raise RuntimeError("handover wave trace arrays are not row aligned")
    handover_names = (
        "handover_pregrasp",
        "handover_orient_clear",
        "right_grasp_settle",
        "right_grasp",
        "handover_contact_acquire",
    )
    handover_rows = np.flatnonzero(np.isin(names, handover_names))
    preclose_rows = np.flatnonzero(
        np.isin(
            names,
            ("handover_pregrasp", "handover_orient_clear", "right_grasp_settle"),
        )
    )
    close_rows = np.flatnonzero(names == "right_grasp")
    required_arrays = (
        "left_grasp",
        "left_assist_engaged",
        "right_grasp",
        "right_assist_engaged",
        "right_finger_forces_n",
        "right_pad_fractions",
    )
    if any(len(trace[name]) != len(names) for name in required_arrays):
        raise RuntimeError("handover contact arrays are not row aligned")
    forces = np.asarray(trace["right_finger_forces_n"], dtype=np.float64)
    fractions = np.asarray(trace["right_pad_fractions"], dtype=np.float64)
    secure = (
        np.asarray(trace["right_grasp"], dtype=bool)
        & np.asarray(trace["right_assist_engaged"], dtype=bool)
        & np.isfinite(forces).all(axis=1)
        & np.isfinite(fractions).all(axis=1)
        & (forces > 0.0).all(axis=1)
        & (fractions >= 0.15).all(axis=1)
        & (fractions <= 0.85).all(axis=1)
        & (names == "right_grasp")
    )
    secure_rows = np.flatnonzero(secure)
    first_secure = None if not len(secure_rows) else int(secure_rows[0])
    giver_held = bool(
        first_secure is not None
        and np.asarray(trace["left_grasp"], dtype=bool)[
            handover_rows[handover_rows <= first_secure]
        ].all()
        and np.asarray(trace["left_assist_engaged"], dtype=bool)[
            handover_rows[handover_rows <= first_secure]
        ].all()
    )
    right_gripper = actions[:, 13]
    plan_screens = receipt.get("plan_screens") or {}
    live = receipt.get("live_physx_contact_guard") or {}
    checks = {
        "runner_wave_receipt_passed": bool(receipt.get("passed")),
        "both_live_geometry_swept_screens_passed": bool(
            set(plan_screens) == {"clear_pregrasp", "open_approach"}
            and all(plan_screens[name].get("passed") for name in plan_screens)
        ),
        "all_handover_rows_live_guarded": bool(
            live.get("passed")
            and live.get("expected_rows") == len(handover_rows)
            and live.get("observed_rows") == len(handover_rows)
        ),
        "right_open_through_pregrasp_and_approach": bool(
            len(preclose_rows)
            and np.all(right_gripper[preclose_rows] <= -0.04749)
        ),
        "right_close_monotone_only_at_grasp_pose": bool(
            len(close_rows)
            and np.all(np.diff(right_gripper[close_rows]) >= -1.0e-9)
            and right_gripper[close_rows[-1]] >= -1.0e-8
        ),
        "broad_force_backed_contact_reached_while_giver_held": bool(
            len(secure_rows) and giver_held
        ),
        "first_contact_at_contact_pose": bool(
            (live.get("first_right_mug_contact") or {}).get("waypoint")
            in {"right_grasp_settle", "right_grasp"}
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"handover wave audit failed: {checks}")
    return {
        "passed": True,
        "first_broad_right_contact_trace_row": first_secure,
        "handover_trace_rows": int(len(handover_rows)),
        "checks": checks,
        "plan_screens": plan_screens,
        "live_physx_contact_guard": live,
    }


def independent_audit(index: int, attempt: Path) -> dict:
    guard = _guard_lifecycle(attempt)
    result_path, trace_path = attempt / "result.json", attempt / "trace.npz"
    video_path, demo_path = attempt / "video.mp4", attempt / "demo.hdf5"
    manifest_path, log_path = attempt / "manifest.json", attempt / "replay.log"
    result, manifest = _load(result_path), _load(manifest_path)
    if result.get("status") != "passed" or not all(result.get("acceptance_checks", {}).values()):
        raise RuntimeError("runner did not independently qualify this attempt")
    subprocess.run(
        [str(PYTHON), "examples/check_hangmug_run.py", "--log", str(log_path),
         "--result-json", str(result_path), "--trace-npz", str(trace_path),
         "--video", str(video_path)],
        cwd=REPO_ROOT, check=True,
    )
    assets = _asset_pair(index)
    relative_assets = {
        "mug": f"MugHangable/mug_teacup_{index:06d}",
        "mug_tree": f"ThreeLayerMugTree/mug_tree_{index:06d}",
    }
    demo = validate_demo(demo_path, relative_assets)
    import h5py

    with h5py.File(SOURCE, "r") as source_handle, h5py.File(demo_path, "r") as handle:
        source_actions = np.asarray(source_handle["data/demo_0/actions"])
        group = handle["data/demo_0"]
        actions = np.asarray(group["actions"])
        if "processed_actions" in group:
            raise RuntimeError("processed_actions must not appear in an accepted demo")
        count = len(actions)
        semantic = group["obs/semantic"]
        terminal_velocity = np.asarray(semantic["mug_velocity"])[-30:]
        terminal_pose = np.asarray(semantic["mug_pose"])[-30:]
    trace = np.load(trace_path)
    if not np.array_equal(actions, trace["actions"]):
        raise RuntimeError("HDF5 actions differ from the executed trace")
    video = _probe(str(video_path))
    if video["codec"] != "h264" or video["full_decode_returncode"] != 0 or video["frame_count"] != count:
        raise RuntimeError("video decode/frame alignment failed")
    provenance = result["provenance"]
    source = provenance["source_dataset"]
    gains = result["controller_gains"]
    resets = result["reset_counts"]
    protocol = result["protocol"]
    method = manifest["method"]
    if method == "direct_source_action_replay":
        action_binding = len(actions) == len(source_actions) and np.array_equal(actions, source_actions)
        command = _classification_command(index, attempt)
    else:
        selection = {
            name: manifest["classification"][name]
            for name in (
                "requested_failed_stage", "requested_last_completed_stage",
                "actual_repair_boundary", "coarse_fallback",
            )
        }
        action_binding = (
            selection["actual_repair_boundary"] != "pick"
            or np.array_equal(actions[:SOURCE_PREFIX_STEPS], source_actions[:SOURCE_PREFIX_STEPS])
        )
        command = _repair_command(
            index, attempt, Path(manifest["classification"]["result_path"]), selection,
            manifest.get("repair_strategy"),
        )
    if not (
        source["file_sha256"] == SOURCE_SHA256
        and source["actions_sha256"] == SOURCE_ACTIONS_SHA256
        and source["action_dataset"] == "actions"
        and provenance["target_state_template"]["actions_executed"] is False
        and _asset_provenance_matches(provenance["target_assets"], assets)
        and action_binding
        and protocol["candidate_sampling"] is False
        and protocol["physics_device_actual"] == "cpu"
        and protocol["parameters"] | PROVEN_CONTROL_DEFAULTS == protocol["parameters"]
        and all(protocol["parameters"][name] == value for name, value in PROVEN_CONTROL_DEFAULTS.items())
        and gains["starting_configured_sha256"] == gains["ending_configured_sha256"] == CONTROLLER_SHA256
        and gains["starting_live_sha256"] == gains["ending_live_sha256"] == LIVE_GAINS_SHA256
        and gains["starting_live_matches_configured"]["matches_configured_spec"]
        and gains["ending_live_matches_configured"]["matches_configured_spec"]
        and resets == {"explicit_env_reset_calls": 1, "initial_state_restores": 1, "resets_during_episode": 0}
        and manifest["launch_command"] == command
    ):
        raise RuntimeError("source/pair/controller/reset/manifest campaign pins failed")
    semantic_audit = _semantic_audit(
        demo_path,
        assets,
        result["initial_placement"]["mug"]["target_root_z_m"],
        resets,
    )
    selected_branch = semantic_audit["contact_policy"]["selected_branch"]
    _require_middle_row_branch(selected_branch)
    runner_terminal = result.get("independent_terminal_hang")
    if (
        not runner_terminal
        or not runner_terminal.get("passed")
        or runner_terminal.get("checks")
        != {
            name: semantic_audit["terminal_30"][name]
            for name in runner_terminal.get("checks", {})
        }
    ):
        raise RuntimeError("runner/independent terminal hang receipts disagree")
    if not semantic_audit["contact_policy"]["contact_policy_held"]:
        raise RuntimeError("bounded-contact policy failed")
    direct_choreography = None
    handover_wave = None
    if manifest.get("repair_strategy", {}).get("direct_rest_to_preinsert_steps"):
        handover_wave = _handover_wave_audit(result, trace)
        direct_choreography = _direct_choreography_audit(result, trace)
    return {
        "accepted": True,
        "pair_index": index,
        "artifact_hashes": {
            "manifest_sha256": _sha256(manifest_path),
            "result_sha256": _sha256(result_path),
            "trace_sha256": _sha256(trace_path),
            "video_sha256": _sha256(video_path),
            "demo_hdf5_sha256": demo["sha256"],
        },
        "provenance": {
            "source_dataset_sha256": SOURCE_SHA256,
            "source_action_dataset": "actions",
            "source_actions_sha256": SOURCE_ACTIONS_SHA256,
            "source_prefix_action_count": (
                SOURCE_PREFIX_STEPS
                if method != "direct_source_action_replay"
                and manifest["classification"]["actual_repair_boundary"] == "pick"
                else None
            ),
            "source_prefix_bit_exact": (
                bool(action_binding)
                if method == "direct_source_action_replay"
                or manifest["classification"]["actual_repair_boundary"] == "pick"
                else None
            ),
            "target_state_template_actions_executed": False,
            "candidate_sampling": False,
        },
        "controller_gains": {
            "configured_sha256": CONTROLLER_SHA256,
            "configured_unchanged": True,
            "starting_live_sha256": LIVE_GAINS_SHA256,
            "ending_live_sha256": LIVE_GAINS_SHA256,
            "live_unchanged": True,
            "starting_live_matches_configured": True,
            "ending_live_matches_configured": True,
        },
        "reset_counts": resets,
        "hdf5": {
            "actions": count,
            "states": count + 1,
            "observations": count,
            "actions_equal_trace": True,
            "processed_actions_present": False,
            "asset_paths": relative_assets,
            "success": True,
        },
        "media": video,
        "terminal_30": {
            **semantic_audit["terminal_30"],
            "mug_z_span_m": float(np.ptp(terminal_pose[:, 2])),
            "maximum_linear_speed_mps": float(np.linalg.norm(terminal_velocity[:, :3], axis=1).max()),
            "maximum_angular_speed_rps": float(np.linalg.norm(terminal_velocity[:, 3:], axis=1).max()),
        },
        "ordered_semantic_stages": semantic_audit["ordered_semantic_stages"],
        "contact_policy": semantic_audit["contact_policy"],
        "direct_choreography": direct_choreography,
        "handover_wave": handover_wave,
        "guard": guard,
    }


def _asset_provenance_matches(recorded: dict, expected: dict[str, Path]) -> bool:
    """Compare asset identities after resolving lane-local data symlinks."""
    if set(recorded) != set(expected):
        return False
    try:
        return all(
            Path(recorded[name]["path"]).resolve(strict=True)
            == Path(expected[name]).resolve(strict=True)
            for name in expected
        )
    except (KeyError, OSError, TypeError):
        return False


def classification_audit(index: int, attempt: Path) -> dict:
    guard = _guard_lifecycle(attempt)
    result = _load(attempt / "result.json")
    manifest = _load(attempt / "manifest.json")
    if (
        result.get("status") != "passed"
        or result.get("mode") != "replay"
        or not all(result.get("acceptance_checks", {}).values())
        or manifest.get("method") != "direct_source_action_replay"
        or manifest.get("launch_command") != _classification_command(index, attempt)
    ):
        raise RuntimeError("direct classification technical checks failed")
    trace = np.load(attempt / "trace.npz")
    import h5py

    with h5py.File(SOURCE, "r") as handle:
        source_actions = np.asarray(handle["data/demo_0/actions"])
    if not np.array_equal(trace["actions"], source_actions):
        raise RuntimeError("classification did not execute exactly the pinned source actions")
    video = _probe(str(attempt / "video.mp4"))
    if video["codec"] != "h264" or video["full_decode_returncode"] != 0 or video["frame_count"] != len(source_actions):
        raise RuntimeError("classification video failed decode/frame alignment")
    receipt = result.get("semantic_stage_receipt")
    if not receipt or receipt.get("ordered_stages") != [
        "pick", "handover", "alignment", "insertion_and_support", "release_and_hang"
    ]:
        raise RuntimeError("classification lacks the ordered completed-stage receipt")
    completed = receipt.get("completed_stages", [])
    failed = receipt.get("first_failed_stage")
    if completed != receipt["ordered_stages"][: len(completed)]:
        raise RuntimeError("classification completed stages are not a contiguous prefix")
    if failed != (None if len(completed) == 5 else receipt["ordered_stages"][len(completed)]):
        raise RuntimeError("classification first failed stage disagrees with completed prefix")
    gains = result["controller_gains"]
    resets = result["reset_counts"]
    source = result["provenance"]["source_dataset"]
    if not (
        source["file_sha256"] == SOURCE_SHA256
        and source["actions_sha256"] == SOURCE_ACTIONS_SHA256
        and source["action_dataset"] == "actions"
        and result["checks"]["executed_source_actions_exact"]
        and gains["starting_configured_sha256"] == gains["ending_configured_sha256"] == CONTROLLER_SHA256
        and gains["starting_live_sha256"] == gains["ending_live_sha256"] == LIVE_GAINS_SHA256
        and resets == {"explicit_env_reset_calls": 1, "initial_state_restores": 1, "resets_during_episode": 0}
    ):
        raise RuntimeError("classification source/gain/reset pins failed")
    return {
        "status": "direct_success" if failed is None and result["terminal"]["task_success"] else "repair_required",
        "pair_index": index,
        "completed_stages": completed,
        "last_completed_stage": receipt.get("last_completed_stage"),
        "first_failed_stage": failed,
        "first_completed_steps": receipt["first_completed_steps"],
        "terminal_checks": receipt["terminal_checks"],
        "contact_policy": receipt["contact_policy"],
        "artifacts": {
            "manifest_sha256": _sha256(attempt / "manifest.json"),
            "result_sha256": _sha256(attempt / "result.json"),
            "trace_sha256": _sha256(attempt / "trace.npz"),
            "video_sha256": _sha256(attempt / "video.mp4"),
        },
        "source_actions_exact": True,
        "video": video,
        "guard": guard,
        "result_path": str(attempt / "result.json"),
    }


def _validate_accepted(index: int, entry: dict) -> None:
    if index == 0:
        accepted = _load(RESULTS / "source/accepted_source.json")
        paths = accepted["artifacts"]
        for artifact in ("result", "video", "demonstration"):
            path = Path(paths[artifact]["path"])
            if not path.is_absolute():
                path = REPO_ROOT / path
            if not path.is_file() or _sha256(path) != paths[artifact]["sha256"]:
                raise RuntimeError(f"accepted source {artifact} artifact hash disagrees")
        expected = {
            "result_sha256": paths["result"]["sha256"],
            "video_sha256": paths["video"]["sha256"],
            "demonstration_sha256": paths["demonstration"]["sha256"],
        }
    else:
        attempt = RESULTS / "pairs" / f"{index:06d}" / entry["attempt"]
        video = attempt / ("video.mp4" if (attempt / "video.mp4").is_file() else "skill.mp4")
        expected = {
            "result_sha256": _sha256(attempt / "result.json"),
            "video_sha256": _sha256(video),
            "demonstration_sha256": _sha256(attempt / "demo.hdf5"),
            "independent_audit_sha256": _sha256(attempt / "independent_audit.json"),
        }
    if entry.get("status") != "accepted" or any(entry.get(name) != value for name, value in expected.items()):
        raise RuntimeError(f"accepted Pair {index:06d} ledger/artifact hashes disagree")


def _first_missing(ledger: dict) -> int:
    missing = None
    for index in range(40):
        entry = ledger.get("pairs", {}).get(f"{index:06d}")
        if entry is None:
            missing = index
            break
        _validate_accepted(index, entry)
    if missing is None:
        return 40
    later = [key for key, value in ledger.get("pairs", {}).items() if int(key) > missing and value.get("status") == "accepted"]
    if later:
        raise RuntimeError(f"accepted ledger entries exist past missing Pair {missing:06d}: {later}")
    return missing


def _accept(
    index: int,
    attempt: Path,
    audit: dict,
    predecessor_sha256: str,
    *,
    replace_existing: bool = False,
) -> str:
    ledger_path = RESULTS / "ledger.json"
    if _sha256(ledger_path) != predecessor_sha256:
        raise RuntimeError("ledger changed during attempt; refusing atomic transition")
    ledger = _load(ledger_path)
    key = f"{index:06d}"
    existing = ledger["pairs"].get(key)
    if existing is not None and not replace_existing:
        raise RuntimeError(f"Pair {key} already has a ledger entry")
    if existing is None and replace_existing:
        raise RuntimeError(f"Pair {key} has no acceptance to supersede")
    hashes = audit["artifact_hashes"]
    replacement = {
        "status": "accepted",
        "mug": f"MugHangable/mug_teacup_{key}",
        "mug_tree": f"ThreeLayerMugTree/mug_tree_{key}",
        "attempt": attempt.name,
        "result_sha256": hashes["result_sha256"],
        "video_sha256": hashes["video_sha256"],
        "demonstration_sha256": hashes["demo_hdf5_sha256"],
        "independent_audit_sha256": _sha256(attempt / "independent_audit.json"),
    }
    if replace_existing:
        history = list(existing.get("superseded_acceptances", ()))
        history.append(
            {name: value for name, value in existing.items()
             if name != "superseded_acceptances"}
        )
        replacement["superseded_acceptances"] = history
    ledger["pairs"][key] = replacement
    _atomic_json(ledger_path, ledger)
    return _sha256(ledger_path)


def _recover_audited_attempt(index: int) -> bool:
    """Finish a crash-interrupted audit-to-ledger transition without Isaac."""
    pair_root = RESULTS / "pairs" / f"{index:06d}"
    attempts = sorted(pair_root.glob("attempt_[0-9][0-9][0-9]_*") , reverse=True)
    for attempt in attempts:
        audit_path = attempt / "independent_audit.json"
        if not audit_path.is_file():
            continue
        recorded = _load(audit_path)
        recomputed = independent_audit(index, attempt)
        if recorded != recomputed or recorded.get("accepted") is not True:
            raise RuntimeError(f"preserved independent audit changed: {audit_path}")
        _accept(index, attempt, recorded, _sha256(RESULTS / "ledger.json"))
        print(f"TASK2_HANGMUG_RECOVERED_ACCEPTANCE={index:06d} attempt={attempt}", flush=True)
        return True
    return False


def _classification_binding(
    attempt: Path, classification: dict, *, force_from_reset: bool = False
) -> dict:
    selection = (
        _repair_selection("pick", None)
        if force_from_reset
        else _repair_selection(
            classification["first_failed_stage"],
            classification["last_completed_stage"],
        )
    )
    return {
        "result_path": classification["result_path"],
        "result_sha256": classification["artifacts"]["result_sha256"],
        "audit_path": str(attempt / "classification_audit.json"),
        "audit_sha256": _sha256(attempt / "classification_audit.json"),
        "completed_stages": classification["completed_stages"],
        "quality_regeneration_from_direct_success": force_from_reset,
        **selection,
    }


def _manifest(
    index: int,
    attempt: Path,
    command: list[str],
    ledger_sha256: str,
    *,
    method: str,
    classification: dict | None = None,
    repair_strategy: dict | None = None,
    ledger_transition: dict | None = None,
) -> dict:
    value = {
        "schema_version": 1,
        "immutable": True,
        "purpose": f"same-index Task2 HangMug {method}",
        "judo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "pair_index": index,
        "source_dataset": str(SOURCE),
        "source_sha256": SOURCE_SHA256,
        "source_action_dataset": "actions",
        "source_actions_sha256": SOURCE_ACTIONS_SHA256,
        "source_prefix_action_count": (
            SOURCE_PREFIX_STEPS
            if classification is not None
            and classification["actual_repair_boundary"] == "pick"
            else None
        ),
        "target_assets": {name: str(path) for name, path in _asset_pair(index).items()},
        "controller_gains_sha256": CONTROLLER_SHA256,
        "control_defaults": PROVEN_CONTROL_DEFAULTS,
        "source_keyframes_sha256": _sha256(KEYFRAMES),
        "accepted_runtime_timing_sha256": _sha256(TIMING),
        "runner_sha256": _sha256(REPO_ROOT / "examples/run_hangmug_skill_program.py"),
        "compact_skill_sha256": _sha256(REPO_ROOT / "src/judo_isaaclab/hang_mug.py"),
        "campaign_driver_sha256": _sha256(__file__),
        "guard_sha256": _sha256(GUARD),
        "ledger_predecessor_sha256": ledger_sha256,
        "method": method,
        "launch_command": command,
    }
    if classification is not None:
        value["classification"] = classification
    if repair_strategy:
        value["repair_strategy"] = repair_strategy
    if ledger_transition is not None:
        value["ledger_transition"] = ledger_transition
    return value


def _preserve_failure(index: int, attempt: Path, **details) -> None:
    failure = {
        "status": "not_accepted",
        "pair_index": index,
        "manifest_sha256": _sha256(attempt / "manifest.json"),
        **details,
        "preserved_artifacts": {
            path.name: _sha256(path)
            for path in attempt.iterdir()
            if path.is_file() and path.name != "driver_failure.json"
        },
    }
    _atomic_json(attempt / "driver_failure.json", failure, immutable=True)


def _execute(index: int, attempt: Path, command: list[str]) -> None:
    _require_zero_workers()
    print(f"TASK2_HANGMUG_ATTEMPT_START={index:06d} attempt={attempt}", flush=True)
    print("TASK2_HANGMUG_COMMAND=" + json.dumps(command), flush=True)
    returncode = subprocess.run(command, cwd=REPO_ROOT).returncode
    _require_zero_workers()
    if returncode != 0:
        _preserve_failure(index, attempt, guard_returncode=returncode)
        raise RuntimeError(f"Pair {index:06d} failed; preserved at {attempt}")


def _reusable_classification(index: int) -> tuple[Path, dict] | None:
    pair_root = RESULTS / "pairs" / f"{index:06d}"
    for attempt in sorted(pair_root.glob("attempt_[0-9][0-9][0-9]_direct_source_classification"), reverse=True):
        path = attempt / "classification_audit.json"
        if not path.is_file():
            continue
        recorded = _load(path)
        if recorded != classification_audit(index, attempt):
            raise RuntimeError(f"classification receipt changed: {path}")
        print(f"TASK2_HANGMUG_REUSE_CLASSIFICATION={index:06d} attempt={attempt}", flush=True)
        return attempt, recorded
    return None


def _accept_attempt(
    index: int,
    attempt: Path,
    ledger_sha256: str,
    *,
    replace_existing: bool = False,
) -> None:
    try:
        audit = independent_audit(index, attempt)
        _atomic_json(attempt / "independent_audit.json", audit, immutable=True)
        final_ledger_sha256 = _accept(
            index, attempt, audit, ledger_sha256,
            replace_existing=replace_existing,
        )
    except BaseException as error:
        if not (attempt / "driver_failure.json").exists():
            _preserve_failure(index, attempt, error=f"{type(error).__name__}: {error}")
        raise
    print(
        f"TASK2_HANGMUG_PAIR_ACCEPTED={index:06d} "
        f"audit={_sha256(attempt / 'independent_audit.json')} ledger={final_ledger_sha256}",
        flush=True,
    )


def run_one(index: int, *, replace_existing: bool = False) -> None:
    _require_zero_workers()
    ledger_path = RESULTS / "ledger.json"
    ledger_sha256 = _sha256(ledger_path)
    ledger = _load(ledger_path)
    existing = ledger.get("pairs", {}).get(f"{index:06d}")
    if replace_existing:
        if existing is None:
            raise RuntimeError(f"Pair {index:06d} has no acceptance to requalify")
        _validate_accepted(index, existing)
    elif existing is not None:
        raise RuntimeError(f"Pair {index:06d} is already accepted")
    ledger_transition = (
        {"mode": "replace_superseded_acceptance", "previous_entry": existing}
        if replace_existing else None
    )
    reusable = _reusable_classification(index)
    if reusable is None:
        classification_attempt = _attempt_directory(index, "direct_source_classification")
        classification_attempt.mkdir(parents=True, exist_ok=False)
        command = _classification_command(index, classification_attempt)
        _atomic_json(
            classification_attempt / "manifest.json",
            _manifest(
                index, classification_attempt, command, ledger_sha256,
                method="direct_source_action_replay",
                ledger_transition=ledger_transition,
            ),
            immutable=True,
        )
        _execute(index, classification_attempt, command)
        try:
            classification = classification_audit(index, classification_attempt)
            _atomic_json(
                classification_attempt / "classification_audit.json",
                classification,
                immutable=True,
            )
        except BaseException as error:
            _preserve_failure(
                index, classification_attempt,
                error=f"{type(error).__name__}: {error}",
            )
            raise
    else:
        classification_attempt, classification = reusable
    force_regeneration = _force_semantic_regeneration(index)
    if classification["status"] == "direct_success" and not force_regeneration:
        _accept_attempt(
            index, classification_attempt, ledger_sha256,
            replace_existing=replace_existing,
        )
        return
    failed_stage = (
        "quality_regeneration"
        if force_regeneration
        else classification["first_failed_stage"]
    )
    repair_attempt = _attempt_directory(index, f"repair_{failed_stage}")
    repair_attempt.mkdir(parents=True, exist_ok=False)
    classification_binding = _classification_binding(
        classification_attempt,
        classification,
        force_from_reset=force_regeneration,
    )
    repair_strategy = _repair_strategy(index)
    command = _repair_command(
        index, repair_attempt, Path(classification["result_path"]),
        classification_binding, repair_strategy,
    )
    _atomic_json(
        repair_attempt / "manifest.json",
        _manifest(
            index, repair_attempt, command, ledger_sha256,
            method=(
                "semantic_coarse_boundary_repair"
                if classification_binding["coarse_fallback"]
                else "semantic_stage_boundary_repair"
            ),
            classification=classification_binding,
            repair_strategy=repair_strategy,
            ledger_transition=ledger_transition,
        ),
        immutable=True,
    )
    _execute(index, repair_attempt, command)
    _accept_attempt(
        index, repair_attempt, ledger_sha256,
        replace_existing=replace_existing,
    )


def _run_serial(indices) -> None:
    for index in indices:
        run_one(index)


def _claim_explicit_lane(index: int, lane_id: str) -> Path:
    """Bind one isolated results tree to one explicit pair/lane assignment."""
    lane = lane_id.strip()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not lane or any(character not in allowed for character in lane):
        raise ValueError("lane ID must contain only letters, digits, '-' or '_'")
    receipt = {
        "schema_version": 1,
        "pair_index": index,
        "human_pair": index + 1,
        "lane_id": lane,
        "judo_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "results_root": str(RESULTS.resolve()),
    }
    path = RESULTS / "pairs" / f"{index:06d}" / "lane_assignment.json"
    if path.exists():
        recorded = _load(path)
        immutable_fields = (
            "schema_version",
            "pair_index",
            "human_pair",
            "lane_id",
            "results_root",
        )
        fields_changed = any(
            recorded.get(field) != receipt[field] for field in immutable_fields
        )
        recorded_head = recorded.get("judo_head")
        current_head = receipt["judo_head"]
        head_is_authorized = recorded_head == current_head
        if not head_is_authorized and isinstance(recorded_head, str):
            head_is_authorized = subprocess.run(
                ["git", "merge-base", "--is-ancestor", recorded_head, current_head],
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode == 0
        if fields_changed or not head_is_authorized:
            raise RuntimeError(f"pair/lane assignment changed: {path}")
        return path
    _atomic_json(path, receipt, immutable=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    ownership = parser.add_mutually_exclusive_group()
    ownership.add_argument("--max-new-pairs", type=int)
    ownership.add_argument("--pair-index", type=int)
    ownership.add_argument("--requalify-pair", type=int)
    parser.add_argument("--lane-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_new_pairs is not None and args.max_new_pairs < 1:
        raise ValueError("--max-new-pairs must be positive")
    if args.pair_index is not None and not 0 <= args.pair_index < 40:
        raise ValueError("--pair-index must be in [0, 39]")
    if args.requalify_pair is not None and not 1 <= args.requalify_pair < 40:
        raise ValueError("--requalify-pair must be in [1, 39]")
    if (args.pair_index is None) != (args.lane_id is None):
        raise ValueError("--pair-index and --lane-id must be supplied together")
    os.chdir(REPO_ROOT)
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=REPO_ROOT).returncode
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT).returncode
    if dirty != 0 or staged != 0:
        raise RuntimeError("tracked worktree changes must be checkpointed before campaign launch")
    source = _source_dataset_receipt(str(SOURCE), "demo_0", SOURCE_SHA256)
    if source["actions_sha256"] != SOURCE_ACTIONS_SHA256:
        raise RuntimeError("pinned source actions hash mismatch")
    for index in range(40):
        _asset_pair(index)
    ledger = _load(RESULTS / "ledger.json")
    if args.requalify_pair is not None:
        run_one(args.requalify_pair, replace_existing=True)
        return
    if args.pair_index is not None:
        _claim_explicit_lane(args.pair_index, args.lane_id)
        if args.dry_run:
            attempt = _attempt_directory(
                args.pair_index, "direct_source_classification"
            )
            print("TASK2_HANGMUG_DRY_RUN=" + json.dumps({
                "pair_index": args.pair_index,
                "human_pair": args.pair_index + 1,
                "lane_id": args.lane_id,
                "attempt": str(attempt),
                "classification_command": _classification_command(
                    args.pair_index, attempt
                ),
            }, sort_keys=True))
            return
        run_one(args.pair_index)
        return
    start = _first_missing(ledger)
    while start < 40 and _recover_audited_attempt(start):
        ledger = _load(RESULTS / "ledger.json")
        start = _first_missing(ledger)
    if start == 40:
        print("TASK2_HANGMUG_CAMPAIGN_COMPLETE=40/40", flush=True)
        return
    selected = list(range(start, min(40, start + (args.max_new_pairs or 40))))
    if args.dry_run:
        for index in selected:
            attempt = _attempt_directory(index, "direct_source_classification")
            print("TASK2_HANGMUG_DRY_RUN=" + json.dumps({
                "pair_index": index,
                "attempt": str(attempt),
                "classification_command": _classification_command(index, attempt),
            }, sort_keys=True))
        return
    _run_serial(selected)


if __name__ == "__main__":
    main()
