"""Plan or execute one explicit pair-local PutPot quality lane serially."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from judo_isaaclab.putpot_quality import (  # noqa: E402
    GpuLease,
    deterministic_perturbation_cases,
    load_quality_config,
    validate_lane_contract,
    write_immutable_receipt,
)


_OUTPUT_FLAGS = {
    "--result-json",
    "--trace-npz",
    "--video",
    "--demo-hdf5",
    "--runtime-receipt-json",
}


def _load_args(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("runner args JSON must be a list of strings")
    if "--quality-config-json" in value:
        raise ValueError("driver owns --quality-config-json")
    return value


def _expand_args(
    arguments: list[str], *, attempt_root: Path, pair_index: int, lane_id: str
) -> list[str]:
    values = {
        "attempt_root": str(attempt_root.resolve()),
        "pair_index": f"{pair_index:06d}",
        "lane_id": lane_id,
    }
    expanded = [argument.format(**values) for argument in arguments]
    positions = {flag: expanded.index(flag) for flag in _OUTPUT_FLAGS if flag in expanded}
    missing = sorted(_OUTPUT_FLAGS - set(positions))
    if missing:
        raise ValueError(f"runner args are missing pair-local outputs: {missing}")
    for flag, index in positions.items():
        if index + 1 >= len(expanded):
            raise ValueError(f"runner output flag has no path: {flag}")
        output = Path(expanded[index + 1]).resolve()
        if not output.is_relative_to(attempt_root.resolve()):
            raise ValueError(f"runner output escapes attempt root: {flag}={output}")
    return expanded


def build_plan(
    *,
    pair_index: int,
    lane_id: str,
    gpu_id: str,
    output_root: Path,
    quality_config_json: Path,
    runner: Path,
    runner_args_json: Path,
    perturbation_adapter: Path | None,
    joint_dof: int,
) -> dict[str, Any]:
    if not runner.is_file():
        raise FileNotFoundError(f"PutPot runner does not exist: {runner}")
    if perturbation_adapter is not None and not perturbation_adapter.is_file():
        raise FileNotFoundError(
            f"perturbation adapter does not exist: {perturbation_adapter}"
        )
    config = load_quality_config(quality_config_json)
    lane = validate_lane_contract(
        pair_index=pair_index,
        lane_id=lane_id,
        cuda_visible_devices=gpu_id,
        output_root=output_root,
        config=config,
    )
    base_args = _load_args(runner_args_json)
    cases: list[dict[str, Any] | None] = [
        None,
        *deterministic_perturbation_cases(config, joint_dof=joint_dof),
    ]
    attempts = []
    for sequence, case in enumerate(cases):
        name = "nominal" if case is None else f"perturbation-{case['case_index']:03d}"
        attempt_root = output_root / "attempts" / name
        runner_command = [
            sys.executable,
            str(runner.resolve()),
            *_expand_args(
                base_args,
                attempt_root=attempt_root,
                pair_index=pair_index,
                lane_id=lane_id,
            ),
            "--quality-config-json",
            str(quality_config_json.resolve()),
        ]
        command = runner_command
        if case is not None:
            case_path = attempt_root / "perturbation_case.json"
            if perturbation_adapter is None:
                command = None
            else:
                command = [
                    sys.executable,
                    str(perturbation_adapter.resolve()),
                    "--case-json",
                    str(case_path.resolve()),
                    "--",
                    *runner_command,
                ]
        attempts.append(
            {
                "sequence": sequence,
                "name": name,
                "attempt_root": str(attempt_root.resolve()),
                "case": case,
                "command": command,
            }
        )
    return {
        "schema_version": 1,
        "status": "planned",
        "lane": lane,
        "runner": str(runner.resolve()),
        "runner_args_json": str(runner_args_json.resolve()),
        "perturbation_adapter": (
            None if perturbation_adapter is None else str(perturbation_adapter.resolve())
        ),
        "joint_dof": joint_dof,
        "attempts": attempts,
    }


def execute_plan(plan: dict[str, Any], *, lease_root: Path) -> dict[str, Any]:
    """Execute nominal then perturbations serially under one external GPU lease."""

    if plan["perturbation_adapter"] is None:
        return {
            **plan,
            "status": "failed",
            "terminal": True,
            "reason": "missing_physical_perturbation_adapter",
            "pair_owner_action": (
                "provide an adapter that applies each case to the physical runner; "
                "do not mark repeated nominal runs as perturbations"
            ),
        }
    lane = plan["lane"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = lane["gpu_id"]
    receipts = []
    with GpuLease(lease_root, lane["gpu_id"], lane["lane_id"]):
        for attempt in plan["attempts"]:
            root = Path(attempt["attempt_root"])
            root.mkdir(parents=True, exist_ok=False)
            if attempt["case"] is not None:
                write_immutable_receipt(
                    root / "perturbation_case.json", attempt["case"]
                )
            log = root / "driver.log"
            with log.open("xb") as stream:
                completed = subprocess.run(
                    attempt["command"],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                stream.flush()
                os.fsync(stream.fileno())
            result_path = root / "result.json"
            result_status = None
            if result_path.is_file():
                try:
                    result_status = json.loads(
                        result_path.read_text(encoding="utf-8")
                    ).get("status")
                except (OSError, ValueError):
                    result_status = "invalid"
            receipt = {
                "sequence": attempt["sequence"],
                "name": attempt["name"],
                "case_sha256": (
                    None
                    if attempt["case"] is None
                    else attempt["case"]["case_sha256"]
                ),
                "returncode": completed.returncode,
                "result_status": result_status,
                "passed": completed.returncode == 0 and result_status == "passed",
                "log": str(log.resolve()),
            }
            write_immutable_receipt(root / "driver_receipt.json", receipt)
            receipts.append(receipt)
            if not receipt["passed"]:
                break
    completed_all = len(receipts) == len(plan["attempts"])
    all_passed = completed_all and all(receipt["passed"] for receipt in receipts)
    perturbation_outcomes = {
        "joint_dof": plan["joint_dof"],
        "outcomes": [
            {
                "case_sha256": receipt["case_sha256"],
                "passed": receipt["passed"],
            }
            for receipt in receipts
            if receipt["case_sha256"] is not None
        ],
    }
    outcomes_path = Path(lane["output_root"]) / "perturbation_outcomes.json"
    outcomes_sha256 = write_immutable_receipt(
        outcomes_path, perturbation_outcomes
    )
    return {
        **plan,
        "status": "completed" if all_passed else "failed",
        "terminal": True,
        "attempt_receipts": receipts,
        "perturbation_outcomes": {
            "path": str(outcomes_path.resolve()),
            "sha256": outcomes_sha256,
            **perturbation_outcomes,
        },
        "pair_owner_action": (
            None
            if all_passed
            else "diagnose the first terminal attempt before continuing this pair"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-index", type=int, required=True)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--quality-config-json", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--runner-args-json", required=True)
    parser.add_argument("--perturbation-adapter")
    parser.add_argument("--joint-dof", type=int, default=14)
    parser.add_argument("--lease-root", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--receipt-json", required=True)
    args = parser.parse_args(argv)
    plan = build_plan(
        pair_index=args.pair_index,
        lane_id=args.lane_id,
        gpu_id=args.gpu_id,
        output_root=Path(args.output_root),
        quality_config_json=Path(args.quality_config_json),
        runner=Path(args.runner),
        runner_args_json=Path(args.runner_args_json),
        perturbation_adapter=(
            None if args.perturbation_adapter is None else Path(args.perturbation_adapter)
        ),
        joint_dof=args.joint_dof,
    )
    receipt = execute_plan(plan, lease_root=Path(args.lease_root)) if args.execute else plan
    write_immutable_receipt(args.receipt_json, receipt)
    print("PUTPOT_QUALITY_CAMPAIGN=" + json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] in {"planned", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
