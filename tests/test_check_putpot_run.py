import json
from pathlib import Path
import subprocess
import sys


def _run_checker(
    tmp_path: Path,
    center_error_m: float,
    *,
    bimanual_transport_completed: bool = True,
    adapted_target: bool = False,
    executed_clearance_m: float = 0.025,
):
    log = tmp_path / "run.log"
    result = tmp_path / "result.json"
    trace = tmp_path / "trace.npz"
    log.write_text("PUTPOT_FINAL={}\n", encoding="utf-8")
    trace.write_bytes(b"trace")
    centered = center_error_m <= 0.03
    result.write_text(
        json.dumps(
            {
                "status": "passed",
                "metrics": {
                    "center_error_m": center_error_m,
                    "transport_plan": {
                        "internal_stop_count": 0,
                        "minimum_cooktop_clearance_m": 0.025,
                    },
                    "transport_executed": {
                        "minimum_cooktop_clearance_m": executed_clearance_m,
                    },
                },
                "checks": {
                    "centered_on_cooktop": centered,
                    "accepted_task_success": centered,
                    "smooth_collision_aware_transport": True,
                    "transport_no_internal_stops": True,
                    "bimanual_transport_completed": bimanual_transport_completed,
                },
                "acceptance_checks": {"centered_on_cooktop": centered},
                "direct_replay_baseline": ({"status": "passed"} if adapted_target else None),
            }
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        [
            sys.executable,
            "examples/check_putpot_run.py",
            "--log",
            str(log),
            "--result-json",
            str(result),
            "--trace-npz",
            str(trace),
        ],
        capture_output=True,
        text=True,
    )


def test_checker_accepts_center_error_at_threshold(tmp_path):
    checked = _run_checker(tmp_path, 0.03)
    assert checked.returncode == 0, checked.stderr


def test_checker_rejects_old_edge_biased_putpot_result(tmp_path):
    checked = _run_checker(tmp_path, 0.1748468)
    assert checked.returncode == 2
    assert "center error 0.174847 m exceeds 0.03 m" in checked.stderr


def test_checker_requires_bimanual_completion_for_adapted_target(tmp_path):
    checked = _run_checker(
        tmp_path,
        0.02,
        bimanual_transport_completed=False,
        adapted_target=True,
    )
    assert checked.returncode == 2
    assert "checks.bimanual_transport_completed is not true" in checked.stderr


def test_checker_allows_positive_executed_margin_below_planned_clearance(tmp_path):
    checked = _run_checker(tmp_path, 0.02, executed_clearance_m=0.01)
    assert checked.returncode == 0, checked.stderr
