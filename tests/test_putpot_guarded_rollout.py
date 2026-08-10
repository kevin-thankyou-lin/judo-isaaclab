import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).parents[1]
WRAPPER = REPO_ROOT / "examples/run_putpot_guarded_rollout.py"


def _run(tmp_path, returncode):
    log = tmp_path / "rollout.log"
    status = tmp_path / "wrapper_status.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(WRAPPER),
            "--log",
            str(log),
            "--status-json",
            str(status),
            "--",
            sys.executable,
            "-c",
            f"print('physical output'); raise SystemExit({returncode})",
        ],
        capture_output=True,
        text=True,
    )
    return completed, log, status


def test_guarded_rollout_records_success_without_pipe_status_bookkeeping(tmp_path):
    completed, log, status = _run(tmp_path, 0)

    assert completed.returncode == 0
    assert log.read_text() == "physical output\n"
    receipt = json.loads(status.read_text())
    assert receipt["returncode"] == 0
    assert receipt["bookkeeping_complete"] is True


def test_guarded_rollout_preserves_child_failure_and_still_writes_receipt(tmp_path):
    completed, log, status = _run(tmp_path, 7)

    assert completed.returncode == 7
    assert log.read_text() == "physical output\n"
    receipt = json.loads(status.read_text())
    assert receipt["returncode"] == 7
    assert receipt["bookkeeping_complete"] is True
