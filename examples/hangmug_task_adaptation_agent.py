"""Run a resumable evidence-driven HangMug asset-adaptation loop."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from judo_isaaclab.adaptation import (
    TaskAdaptationBundle,
    TrialEvidence,
    corrected_insert_offset,
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--gear-repo", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--render-failures", action="store_true")
    return parser.parse_args()


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _extend(command: list[str], name: str, value: Any) -> None:
    if value is None or value is False:
        return
    command.append(_flag(name))
    if value is True:
        return
    if isinstance(value, (list, tuple)):
        command.extend(_argument(item) for item in value)
    else:
        command.append(_argument(value))


def _argument(value: Any) -> str:
    """Avoid scientific-notation negatives that argparse treats as options."""
    if isinstance(value, float):
        return f"{value:.12f}"
    return str(value)


def _run(command: list[str], log_path: Path, dry_run: bool) -> int:
    log_path.write_text("COMMAND=" + json.dumps(command) + "\n")
    if dry_run:
        return 0
    with open(log_path, "a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return completed.returncode


def _stage_command(
    args,
    bundle,
    stage,
    parameters,
    promoted_controls,
    result_path,
    controls_path,
):
    command = [
        args.python,
        str(REPO_ROOT / "examples" / "hangmug_grasp_keyframe_mpc.py"),
        "--gear-repo",
        args.gear_repo,
        "--dataset",
        bundle.dataset,
        "--objects-root",
        bundle.objects_root,
        "--episode",
        bundle.episode,
        "--target-mug",
        bundle.target_assets["mug"],
        "--target-mug-tree",
        bundle.target_assets["mug_tree"],
        "--checkpoint-state",
        str(bundle.checkpoint_state),
        "--start-state",
        str(stage.start_state),
        "--target-state",
        str(stage.target_state),
        "--target-name",
        stage.target_name,
        "--horizon",
        str(stage.horizon),
        "--result-json",
        str(result_path),
        "--controls-npz",
        str(controls_path),
    ]
    if promoted_controls:
        command.append("--history-controls-npz")
        command.extend(str(path) for path in promoted_controls)
    branch = bundle.correspondences.get("mug_tree.branch", {})
    if branch:
        _extend(command, "source_branch_points", branch["source_points"])
        _extend(command, "target_branch_points", branch["target_points"])
    handle = bundle.correspondences.get("mug.handle", {})
    if handle:
        _extend(command, "source_handle_point_mug", handle["source_point"])
        _extend(command, "target_handle_point_mug", handle["target_point"])
    grasp = bundle.correspondences.get("mug.left_grasp", {})
    if stage.target_name == "left_grasp" and grasp:
        _extend(command, "source_grasp_eef_pose", grasp["source_eef_pose"])
        _extend(
            command,
            "source_grasp_contact_pose",
            grasp.get("source_contact_pose"),
        )
    for name, value in parameters.items():
        if name != "attempts":
            _extend(command, name, value)
    return command


def _failure_render_command(
    args, bundle, controls_path, output_path, result_path
):
    return [
        args.python,
        str(REPO_ROOT / "examples" / "render_hangmug_mpc_comparison.py"),
        "--gear-repo",
        args.gear_repo,
        "--dataset",
        bundle.dataset,
        "--objects-root",
        bundle.objects_root,
        "--episode",
        bundle.episode,
        "--target-mug",
        bundle.target_assets["mug"],
        "--target-mug-tree",
        bundle.target_assets["mug_tree"],
        "--controls-npz",
        str(controls_path),
        "--output",
        str(output_path),
        "--result-json",
        str(result_path),
        "--camera-names",
        "top_camera",
        "left_wrist_camera",
        "right_wrist_camera",
        "--draw-coordinate-axes",
    ]


def _resumable_stage_record(ledger, stage_name):
    """Return the latest incomplete record instead of repeating completed trials."""
    for record in reversed(ledger["stages"]):
        if record.get("name") == stage_name and record.get("status") != "promoted":
            record["status"] = "running"
            return record
    record = {"name": stage_name, "status": "running", "trials": []}
    ledger["stages"].append(record)
    return record


def _discard_incomplete_trailing_trials(stage_record):
    """Retry launches that exited before producing both evidence artifacts."""
    while stage_record["trials"]:
        trailing = stage_record["trials"][-1]
        if Path(trailing["result"]).exists() and Path(
            trailing["controls"]
        ).exists():
            break
        stage_record["trials"].pop()


def _fallback_trial(stage_record, fallback):
    """Select a robust near-keyframe trial for downstream coded validation."""
    eligible = []
    for trial in stage_record["trials"]:
        metrics = trial.get("metrics", {})
        count = metrics.get("count", 0)
        generic_minima = fallback.get("metric_min", {})
        generic_maxima = fallback.get("metric_max", {})
        generic_match = bool(generic_minima or generic_maxima) and all(
            metrics.get(name, float("-inf")) >= threshold
            for name, threshold in generic_minima.items()
        ) and all(
            metrics.get(name, float("inf")) <= threshold
            for name, threshold in generic_maxima.items()
        )
        legacy_match = (
            count > 0
            and metrics.get("keyframe_right_grasp_count") == count
            and metrics.get("keyframe_stage2_count") == count
            and metrics.get("keyframe_position_error_m_max", float("inf"))
            <= fallback.get("position_tolerance_m", float("-inf"))
            and metrics.get("keyframe_rotation_error_rad_max", float("inf"))
            <= fallback.get("rotation_tolerance_rad", float("-inf"))
        )
        if generic_match or legacy_match:
            eligible.append(trial)
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda trial: (
            -trial["metrics"].get("acceptance_success_count", 0),
            -trial["metrics"].get("post_keyframe_target_fraction_mean", 0.0),
            trial["metrics"].get("keyframe_position_error_m_max", float("inf")),
            trial["metrics"].get("keyframe_rotation_error_rad_max", float("inf")),
        ),
    )


def _rendered_mpc_subtask_passed(result_path):
    """Require a fresh rendered execution, not only batched search metrics."""
    result_path = Path(result_path)
    if not result_path.exists():
        return False
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        result.get("status") == "passed"
        and result.get("subtask_complete", {}).get("mpc")
        and result.get("checks", {}).get("fully_decodable")
    )


def main():
    args = _parser()
    bundle = TaskAdaptationBundle.load(args.bundle)
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    ledger_path = workspace / "ledger.json"
    ledger = (
        json.loads(ledger_path.read_text())
        if args.resume and ledger_path.exists()
        else {
            "status": "running",
            "bundle": str(Path(args.bundle).resolve()),
            "task_name": bundle.task_name,
            "success_check": bundle.success_check,
            "source_assets": bundle.source_assets,
            "target_assets": bundle.target_assets,
            "stages": [],
        }
    )
    if args.resume:
        ledger["status"] = "running"
    promoted_controls = [
        Path(stage["promoted_controls"])
        for stage in ledger["stages"]
        if stage.get("status") == "promoted"
    ]
    completed_names = {
        stage["name"]
        for stage in ledger["stages"]
        if stage.get("status") == "promoted"
    }

    for stage in bundle.stages:
        if stage.name in completed_names:
            continue
        parameters = copy.deepcopy(stage.parameters)
        attempts = parameters.pop("attempts", [{}])
        fallback = parameters.pop("provisional_fallback", None)
        stage_record = (
            _resumable_stage_record(ledger, stage.name)
            if args.resume
            else {"name": stage.name, "status": "running", "trials": []}
        )
        if not args.resume:
            ledger["stages"].append(stage_record)
        ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True))
        evidence = None
        previous_parameters = None
        _discard_incomplete_trailing_trials(stage_record)
        if stage_record["trials"]:
            previous_trial = stage_record["trials"][-1]
            previous_result = Path(previous_trial["result"])
            previous_controls = Path(previous_trial["controls"])
            previous_parameters = previous_trial.get("parameters")
            if previous_result.exists() and previous_controls.exists():
                evidence = TrialEvidence.from_result(
                    previous_result, previous_controls
                )
        for index, override in enumerate(attempts):
            if index < len(stage_record["trials"]):
                continue
            trial_parameters = copy.deepcopy(parameters)
            trial_parameters.update(override)
            if (
                evidence is not None
                and stage.target_name == "inserted_held"
                and "insert_eef_position_offset_branch" not in override
            ):
                prior_offset = (
                    previous_parameters or trial_parameters
                ).get(
                    "insert_eef_position_offset_branch",
                    [0.0, 0.0, 0.0],
                )
                trial_parameters["insert_eef_position_offset_branch"] = (
                    corrected_insert_offset(
                        prior_offset,
                        evidence,
                    )
                )
            prefix = workspace / f"{stage.name}_trial{index:02d}"
            result_path = prefix.with_suffix(".json")
            controls_path = prefix.with_name(prefix.name + "_controls.npz")
            log_path = prefix.with_suffix(".log")
            command = _stage_command(
                args,
                bundle,
                stage,
                trial_parameters,
                promoted_controls,
                result_path,
                controls_path,
            )
            returncode = _run(command, log_path, args.dry_run)
            trial = {
                "index": index,
                "command": command,
                "returncode": returncode,
                "result": str(result_path),
                "controls": str(controls_path),
                "log": str(log_path),
                "parameters": trial_parameters,
            }
            if result_path.exists() and controls_path.exists():
                evidence = TrialEvidence.from_result(
                    result_path, controls_path
                )
                trial["reached"] = evidence.reached
                trial["metrics"] = evidence.metrics
            stage_record["trials"].append(trial)
            previous_parameters = trial_parameters
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True))
            if args.dry_run:
                promoted_controls.append(controls_path)
                stage_record["status"] = "promoted"
                stage_record["promoted_controls"] = str(controls_path)
                break
            if evidence is not None and evidence.reached:
                promoted_controls.append(controls_path)
                stage_record["status"] = "promoted"
                stage_record["promoted_controls"] = str(controls_path)
                break
            if (
                args.render_failures
                and controls_path.exists()
                and not args.dry_run
            ):
                _run(
                    _failure_render_command(
                        args,
                        bundle,
                        controls_path,
                        prefix.with_name(prefix.name + "_all_views.mp4"),
                        prefix.with_name(prefix.name + "_render.json"),
                    ),
                    prefix.with_name(prefix.name + "_render.log"),
                    False,
                )
        if stage_record["status"] != "promoted" and fallback is not None:
            provisional = _fallback_trial(stage_record, fallback)
            if provisional is not None:
                controls_path = Path(provisional["controls"])
                if fallback.get("require_rendered_mpc_success"):
                    prefix_name = controls_path.name.removesuffix(
                        "_controls.npz"
                    )
                    render_output = controls_path.with_name(
                        prefix_name + "_all_views.mp4"
                    )
                    render_result = controls_path.with_name(
                        prefix_name + "_render.json"
                    )
                    render_log = controls_path.with_name(
                        prefix_name + "_render.log"
                    )
                    if not _rendered_mpc_subtask_passed(render_result):
                        _run(
                            _failure_render_command(
                                args,
                                bundle,
                                controls_path,
                                render_output,
                                render_result,
                            ),
                            render_log,
                            False,
                        )
                    render_passed = _rendered_mpc_subtask_passed(
                        render_result
                    )
                    stage_record["fallback_render"] = {
                        "passed": render_passed,
                        "result": str(render_result),
                        "video": str(render_output),
                        "log": str(render_log),
                    }
                    if not render_passed:
                        provisional = None
            if provisional is not None:
                controls_path = Path(provisional["controls"])
                promoted_controls.append(controls_path)
                stage_record["status"] = "promoted"
                stage_record["promoted_controls"] = str(controls_path)
                stage_record["promotion"] = {
                    "kind": "provisional_final_gate",
                    "trial": provisional["index"],
                    "thresholds": fallback,
                    "metrics": provisional["metrics"],
                }
        if stage_record["status"] != "promoted":
            stage_record["status"] = "failed"
            ledger["status"] = "failed"
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True))
            raise RuntimeError(f"stage {stage.name} exhausted its trials")
        ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True))

    output = workspace / "hangmug_target_asset_demo_all_views.mp4"
    final_result = workspace / "final_validation.json"
    final_log = workspace / "final_validation.log"
    command = [
        args.python,
        str(REPO_ROOT / "examples" / "render_hangmug_continuous_sequence.py"),
        "--gear-repo",
        args.gear_repo,
        "--dataset",
        bundle.dataset,
        "--objects-root",
        bundle.objects_root,
        "--episode",
        bundle.episode,
        "--target-mug",
        bundle.target_assets["mug"],
        "--target-mug-tree",
        bundle.target_assets["mug_tree"],
        "--controls-npz",
        *(str(path) for path in promoted_controls),
        "--output",
        str(output),
        "--result-json",
        str(final_result),
        "--camera-names",
        "top_camera",
        "left_wrist_camera",
        "right_wrist_camera",
        "--draw-coordinate-axes",
    ]
    returncode = _run(command, final_log, args.dry_run)
    ledger["final"] = {
        "command": command,
        "returncode": returncode,
        "result": str(final_result),
        "video": str(output),
        "log": str(final_log),
    }
    final = json.loads(final_result.read_text()) if final_result.exists() else {}
    ledger["status"] = (
        "passed"
        if returncode == 0 and final.get("status") == "passed"
        else ("dry_run" if args.dry_run else "failed")
    )
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True))
    print("HANGMUG_ADAPTATION_AGENT=" + json.dumps(ledger, sort_keys=True))
    if ledger["status"] not in ("passed", "dry_run"):
        raise RuntimeError("continuous target-asset validation failed")


if __name__ == "__main__":
    main()
