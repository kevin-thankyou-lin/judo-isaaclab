import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "examples/run_hangmug_skill_program.py"
    spec = importlib.util.spec_from_file_location("run_hangmug_skill_program", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cpu_physics_receipt_requires_cpu_request():
    module = _module()

    with pytest.raises(RuntimeError, match="requested device"):
        module._physics_device_receipt("cuda:0", require_cpu=True)


def test_cpu_physics_receipt_requires_cpu_actual_device():
    module = _module()

    with pytest.raises(RuntimeError, match="actual device"):
        module._physics_device_receipt("cpu", "cuda:0", require_cpu=True)


def test_cpu_physics_receipt_proves_requested_and_actual_cpu():
    module = _module()

    assert module._physics_device_receipt("cpu", "cpu", require_cpu=True) == {
        "required": "cpu",
        "requested": "cpu",
        "actual": "cpu",
        "passed": True,
    }
