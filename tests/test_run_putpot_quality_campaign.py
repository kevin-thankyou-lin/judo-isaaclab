import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from run_putpot_quality_campaign import build_plan, execute_plan


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/putpot_quality_wave_v1.json"


def _runner_args(tmp_path):
    path = tmp_path / "runner_args.json"
    path.write_text(
        json.dumps(
            [
                "--result-json",
                "{attempt_root}/result.json",
                "--trace-npz",
                "{attempt_root}/trace.npz",
                "--video",
                "{attempt_root}/video.mp4",
                "--demo-hdf5",
                "{attempt_root}/demo.hdf5",
                "--runtime-receipt-json",
                "{attempt_root}/runtime.json",
            ]
        )
    )
    return path


def _fake_runner(tmp_path):
    path = tmp_path / "fake_runner.py"
    path.write_text(
        """import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--result-json', required=True)
parser.add_argument('--trace-npz')
parser.add_argument('--video')
parser.add_argument('--demo-hdf5')
parser.add_argument('--runtime-receipt-json')
parser.add_argument('--quality-config-json')
args, _ = parser.parse_known_args()
result = Path(args.result_json)
result.parent.mkdir(parents=True, exist_ok=True)
result.write_text(json.dumps({'status': 'passed'}))
"""
    )
    return path


def _fake_adapter(tmp_path):
    path = tmp_path / "fake_adapter.py"
    path.write_text(
        """import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--case-json', required=True)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
raise SystemExit(subprocess.run(command).returncode)
"""
    )
    return path


def _plan(tmp_path, adapter):
    output = tmp_path / "pairs/000012"
    return build_plan(
        pair_index=12,
        lane_id="node1-gpu3-pair12",
        gpu_id="3",
        output_root=output,
        quality_config_json=CONFIG,
        runner=_fake_runner(tmp_path),
        runner_args_json=_runner_args(tmp_path),
        perturbation_adapter=adapter,
        joint_dof=14,
    )


def test_plan_is_nominal_then_fixed_cases_and_binds_quality_config(tmp_path):
    adapter = _fake_adapter(tmp_path)
    plan = _plan(tmp_path, adapter)
    assert len(plan["attempts"]) == 9
    assert plan["attempts"][0]["name"] == "nominal"
    assert [attempt["sequence"] for attempt in plan["attempts"]] == list(range(9))
    assert all(
        "--quality-config-json" in attempt["command"]
        for attempt in plan["attempts"]
    )
    assert plan["attempts"][1]["case"]["case_sha256"]


def test_execute_fails_closed_without_physical_perturbation_adapter(tmp_path):
    plan = _plan(tmp_path, None)
    receipt = execute_plan(plan, lease_root=tmp_path / "leases")
    assert receipt["status"] == "failed"
    assert receipt["reason"] == "missing_physical_perturbation_adapter"
    assert not (tmp_path / "pairs/000012/attempts").exists()


def test_execute_runs_all_attempts_serially_under_one_released_gpu_lease(tmp_path):
    plan = _plan(tmp_path, _fake_adapter(tmp_path))
    receipt = execute_plan(plan, lease_root=tmp_path / "leases")
    assert receipt["status"] == "completed"
    assert len(receipt["attempt_receipts"]) == 9
    assert all(item["passed"] for item in receipt["attempt_receipts"])
    assert len(receipt["perturbation_outcomes"]["outcomes"]) == 8
    assert Path(receipt["perturbation_outcomes"]["path"]).is_file()
    assert not (tmp_path / "leases/gpu-3.lease").exists()
    for index, item in enumerate(receipt["attempt_receipts"]):
        assert item["sequence"] == index
