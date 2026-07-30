import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from hangmug_grasp_keyframe_mpc import _objective_components
from render_hangmug_mpc_comparison import _quaternion_error


@pytest.fixture
def rollout_inputs():
    states = np.zeros((2, 2, 36), dtype=np.float32)
    reference = np.zeros((2, 36), dtype=np.float32)
    states[:, :, 3] = 1.0
    reference[:, 3] = 1.0
    states[:, :, 32] = 1.0
    reference[:, 32] = 1.0
    sensors = np.zeros((2, 2, 10), dtype=np.float32)
    controls = np.zeros((2, 2, 14), dtype=np.float32)
    nominal = np.zeros((2, 14), dtype=np.float32)
    return states, sensors, controls, reference, nominal


@pytest.mark.parametrize(
    ("target_name", "sensor_index"),
    (
        ("left_grasp", 2),
        ("right_grasp", 6),
        ("handover_latched", 8),
    ),
)
def test_target_success_selects_expected_sensor(
    rollout_inputs, target_name, sensor_index
):
    states, sensors, controls, reference, nominal = rollout_inputs
    sensors[0, :, sensor_index] = 1.0
    if target_name == "left_grasp":
        sensors[0, :, 3] = 1.0
    elif target_name == "right_grasp":
        sensors[0, :, 2] = 1.0
        sensors[0, :, 7] = 1.0
    else:
        sensors[0, :, 6] = 1.0

    result = _objective_components(
        states,
        sensors,
        controls,
        reference=reference,
        nominal=nominal,
        keyframe_offset=0,
        target_name=target_name,
    )

    assert result["keyframe_target_success"].tolist() == [True, False]
    assert result["rewards"][0] > result["rewards"][1]


def test_video_quaternion_error_is_sign_invariant():
    quaternion = np.array([0.5, 0.5, 0.5, 0.5])

    assert _quaternion_error(quaternion, -quaternion) == pytest.approx(0.0)


@pytest.mark.parametrize("target_name", ("tree_approach", "inserted_held"))
def test_tree_targets_require_latched_handover_and_right_grasp(
    rollout_inputs, target_name
):
    states, sensors, controls, reference, nominal = rollout_inputs
    sensors[0, :, 6] = 1.0
    sensors[0, :, 8] = 1.0
    sensors[1, :, 6] = 1.0

    result = _objective_components(
        states,
        sensors,
        controls,
        reference=reference,
        nominal=nominal,
        keyframe_offset=0,
        target_name=target_name,
    )

    assert result["keyframe_target_success"].tolist() == [True, False]
    assert result["rewards"][0] > result["rewards"][1]


def test_tree_relative_target_is_invariant_to_tree_translation_and_yaw(
    rollout_inputs,
):
    states, sensors, controls, reference, nominal = rollout_inputs
    reference[:, :3] = [1.0, 0.0, 0.0]
    yaw = np.sqrt(0.5)
    states[0, :, :3] = [2.0, 4.0, 0.0]
    states[0, :, 3:7] = [yaw, 0.0, 0.0, yaw]
    states[0, :, 29:32] = [2.0, 3.0, 0.0]
    states[0, :, 32:36] = [yaw, 0.0, 0.0, yaw]
    states[1, :, :3] = [2.1, 4.0, 0.0]
    states[1, :, 3:7] = [yaw, 0.0, 0.0, yaw]
    states[1, :, 29:32] = [2.0, 3.0, 0.0]
    states[1, :, 32:36] = [yaw, 0.0, 0.0, yaw]
    sensors[:, :, 6] = 1.0
    sensors[:, :, 8] = 1.0

    result = _objective_components(
        states,
        sensors,
        controls,
        reference=reference,
        nominal=nominal,
        keyframe_offset=0,
        target_name="inserted_held",
    )

    assert result["target_frame"] == "mug_relative_to_tree"
    assert result["keyframe_position_error_m"][0] == pytest.approx(
        0.0, abs=1.0e-6
    )
    assert result["keyframe_rotation_error_rad"][0] == pytest.approx(0.0)
    assert result["keyframe_position_error_m"][1] == pytest.approx(0.1)
