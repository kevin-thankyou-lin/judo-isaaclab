import importlib.util
from pathlib import Path

import pytest


def _runner_module():
    path = Path(__file__).parents[1] / "examples/run_putmarker_skill_program.py"
    spec = importlib.util.spec_from_file_location("putmarker_runner_device", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_actual_device_receipt_accepts_cpu_physics_and_action_tensors():
    receipt = _runner_module()._actual_device_receipt(
        "cpu", "cpu", "cpu", ["cpu", "cpu"]
    )

    assert receipt["matched"] is True
    assert receipt["actual"]["action_tensors"] == ["cpu"]


def test_actual_device_receipt_fails_closed_on_device_drift():
    with pytest.raises(RuntimeError, match="actual device receipt mismatch"):
        _runner_module()._actual_device_receipt(
            "cpu", "cpu", "cuda:0", ["cpu"]
        )
