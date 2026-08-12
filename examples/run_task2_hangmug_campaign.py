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
    index: int, attempt: Path, classification_result: Path, selection: dict
) -> list[str]:
    workload = _common_workload(index, attempt)
    arguments = [
        "--mode", "skill",
        "--source-keyframes", str(KEYFRAMES),
        "--direct-replay-result", str(classification_result),
        "--handover-confirm-steps", "12",
    ]
    if selection["actual_repair_boundary"] == "reset":
        arguments.extend(["--handover-contact-settle-steps", "30"])
    elif selection["actual_repair_boundary"] == "pick":
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


def _semantic_audit(demo: Path, assets: dict[str, Path], mug_init_z: float) -> dict:
    """Recompute five ordered stages/contact with the campaign status adapter."""
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
    firsts = {stage: _first_true([row[stage] for row in statuses]) for stage in ORDERED_STAGES}
    if any(value is None for value in firsts.values()) or list(firsts.values()) != sorted(firsts.values()):
        raise RuntimeError(f"ordered semantic stages failed: {firsts}")
    terminal = statuses[-30:]
    if len(terminal) != 30 or not all(
        row["task_success"] and row["released"] and row["stable"] and row["contact_policy"]
        for row in terminal
    ):
        raise RuntimeError("terminal 30-step physical success audit failed")
    final_diagnostics = statuses[-1]["diagnostics"]
    return {
        "ordered_semantic_stages": firsts,
        "terminal_30": {
            "all_task_success": True,
            "all_released": True,
            "all_stable": True,
            "all_contact_policy": True,
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
            index, attempt, Path(manifest["classification"]["result_path"]), selection
        )
    if not (
        source["file_sha256"] == SOURCE_SHA256
        and source["actions_sha256"] == SOURCE_ACTIONS_SHA256
        and source["action_dataset"] == "actions"
        and provenance["target_state_template"]["actions_executed"] is False
        and {name: value["path"] for name, value in provenance["target_assets"].items()}
        == {name: str(path) for name, path in assets.items()}
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
        demo_path, assets, result["initial_placement"]["mug"]["target_root_z_m"]
    )
    if not semantic_audit["contact_policy"]["contact_policy_held"]:
        raise RuntimeError("bounded-contact policy failed")
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
        "guard": guard,
    }


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


def _accept(index: int, attempt: Path, audit: dict, predecessor_sha256: str) -> str:
    ledger_path = RESULTS / "ledger.json"
    if _sha256(ledger_path) != predecessor_sha256:
        raise RuntimeError("ledger changed during attempt; refusing atomic transition")
    ledger = _load(ledger_path)
    key = f"{index:06d}"
    if key in ledger["pairs"]:
        raise RuntimeError(f"Pair {key} already has a ledger entry")
    hashes = audit["artifact_hashes"]
    ledger["pairs"][key] = {
        "status": "accepted",
        "mug": f"MugHangable/mug_teacup_{key}",
        "mug_tree": f"ThreeLayerMugTree/mug_tree_{key}",
        "attempt": attempt.name,
        "result_sha256": hashes["result_sha256"],
        "video_sha256": hashes["video_sha256"],
        "demonstration_sha256": hashes["demo_hdf5_sha256"],
        "independent_audit_sha256": _sha256(attempt / "independent_audit.json"),
    }
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


def _classification_binding(attempt: Path, classification: dict) -> dict:
    selection = _repair_selection(
        classification["first_failed_stage"], classification["last_completed_stage"]
    )
    return {
        "result_path": classification["result_path"],
        "result_sha256": classification["artifacts"]["result_sha256"],
        "audit_path": str(attempt / "classification_audit.json"),
        "audit_sha256": _sha256(attempt / "classification_audit.json"),
        "completed_stages": classification["completed_stages"],
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


def _accept_attempt(index: int, attempt: Path, ledger_sha256: str) -> None:
    try:
        audit = independent_audit(index, attempt)
        _atomic_json(attempt / "independent_audit.json", audit, immutable=True)
        final_ledger_sha256 = _accept(index, attempt, audit, ledger_sha256)
    except BaseException as error:
        if not (attempt / "driver_failure.json").exists():
            _preserve_failure(index, attempt, error=f"{type(error).__name__}: {error}")
        raise
    print(
        f"TASK2_HANGMUG_PAIR_ACCEPTED={index:06d} "
        f"audit={_sha256(attempt / 'independent_audit.json')} ledger={final_ledger_sha256}",
        flush=True,
    )


def run_one(index: int) -> None:
    _require_zero_workers()
    ledger_path = RESULTS / "ledger.json"
    ledger_sha256 = _sha256(ledger_path)
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
    if classification["status"] == "direct_success":
        _accept_attempt(index, classification_attempt, ledger_sha256)
        return
    failed_stage = classification["first_failed_stage"]
    repair_attempt = _attempt_directory(index, f"repair_{failed_stage}")
    repair_attempt.mkdir(parents=True, exist_ok=False)
    classification_binding = _classification_binding(
        classification_attempt, classification
    )
    command = _repair_command(
        index, repair_attempt, Path(classification["result_path"]),
        classification_binding,
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
        ),
        immutable=True,
    )
    _execute(index, repair_attempt, command)
    _accept_attempt(index, repair_attempt, ledger_sha256)


def _run_serial(indices) -> None:
    for index in indices:
        run_one(index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-pairs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_new_pairs is not None and args.max_new_pairs < 1:
        raise ValueError("--max-new-pairs must be positive")
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
