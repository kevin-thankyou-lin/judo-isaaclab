import json
from pathlib import Path
import sys

import pytest

from judo_isaaclab.putpot_controller_protocol import (
    ControllerPluginClient,
    jsonable,
    sha256_file,
    validate_command_response,
)


REPO_ROOT = Path(__file__).parents[1]
RUNNER = REPO_ROOT / "examples/run_putpot_controller_plugin.py"


def _plugin(tmp_path, name, offset):
    path = tmp_path / f"{name}.py"
    path.write_text(
        f'''\
class Controller:
    def initialize(self, context):
        self.offset = {offset!r}
        return {{"protocol_version": 1, "program_name": {name!r}, "total_steps": context["base_trajectory"]["steps"], "metadata": {{"offset": self.offset}}}}
    def command(self, request):
        base = request["base_command"]
        left = list(base["left_pose"])
        left[0] += self.offset
        return {{"kind": "cartesian", "stage": {name!r}, "left_pose": left, "right_pose": base["right_pose"], "grippers": base["grippers"], "terminate": False}}
def create_controller():
    return Controller()
''',
        encoding="utf-8",
    )
    return path


def _context():
    return {
        "base_trajectory": {
            "steps": 3,
            "left_poses": [[0, 0, 0, 1, 0, 0, 0]] * 3,
            "right_poses": [[0, 0, 0, 1, 0, 0, 0]] * 3,
            "grippers": [[0, 0]] * 3,
            "stage_names": ["base"] * 3,
            "waypoint_steps": {},
        }
    }


def _base():
    return {
        "stage": "base",
        "left_pose": [0, 0, 0, 1, 0, 0, 0],
        "right_pose": [0, 0, 0, 1, 0, 0, 0],
        "grippers": [0, 0],
        "joint_nominal": [0] * 14,
    }


def test_two_python_revisions_reload_as_distinct_child_processes(tmp_path):
    receipts = []
    outputs = []
    for index, offset in enumerate((0.001, 0.004), 1):
        plugin = _plugin(tmp_path, f"revision_{index}", offset)
        client = ControllerPluginClient(
            plugin,
            sha256_file(plugin),
            runner_path=RUNNER,
            log_path=tmp_path / f"controller_{index}.log",
            python=sys.executable,
        )
        initialized = client.initialize(_context(), 3)
        outputs.append(
            client.command(step=0, base_command=_base(), observation={})
        )
        receipts.append(client.receipt())
        client.close()
        assert initialized["program_name"] == f"revision_{index}"
    assert receipts[0]["pid"] != receipts[1]["pid"]
    assert receipts[0]["sha256"] != receipts[1]["sha256"]
    assert outputs[0]["left_pose"][0] == pytest.approx(0.001)
    assert outputs[1]["left_pose"][0] == pytest.approx(0.004)


def test_command_schema_rejects_nonfinite_or_partial_targets():
    with pytest.raises(ValueError, match="keys must be exactly"):
        validate_command_response(
            {"kind": "cartesian", "stage": "bad", "terminate": False}
        )


def test_observation_codec_nulls_nonfinite_sensor_sentinels_only_when_requested():
    assert jsonable({"pad_fraction": float("nan")}, nonfinite="null") == {
        "pad_fraction": None
    }
    with pytest.raises(ValueError, match="must be finite"):
        jsonable({"command": float("nan")})
    with pytest.raises(ValueError, match="finite"):
        validate_command_response(
            {
                "kind": "joint_action",
                "stage": "bad",
                "terminate": False,
                "action": [0.0] * 13 + [float("nan")],
            }
        )
