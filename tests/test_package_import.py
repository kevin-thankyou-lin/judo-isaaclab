import json
import os
from pathlib import Path
import subprocess
import sys


def test_evidence_cli_does_not_require_judo_runtime(tmp_path):
    root = Path(__file__).parents[1]
    workspace = tmp_path / "workspace"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "evidence_adaptation_harness.py"),
            "--bundle",
            str(root / "configs" / "putpot_evidence_agent.json"),
            "--workspace",
            str(workspace),
            "init",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["task_name"] == "PutPotOnCooktop-v0"
