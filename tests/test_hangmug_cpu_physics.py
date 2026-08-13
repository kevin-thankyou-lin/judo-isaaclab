import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def test_left_release_retreat_is_bounded():
    module = _module()

    assert module._bounded_left_release_retreat(0.03) == pytest.approx(0.03)
    with pytest.raises(ValueError, match=r"\[0.02, 0.12\]"):
        module._bounded_left_release_retreat(0.15)

    lift = module._bounded_handover_post_release_lift(0.055)
    assert lift == pytest.approx(0.055)
    assert module._bounded_handover_post_release_lift_steps(30, lift) == 30
    with pytest.raises(ValueError, match="post-release lift"):
        module._bounded_handover_post_release_lift(0.081)
    with pytest.raises(ValueError, match="distance and steps"):
        module._bounded_handover_post_release_lift_steps(0, lift)


def test_semantic_waypoint_identity_uses_executed_row_endpoints():
    module = _module()
    trajectory = SimpleNamespace(
        waypoint_steps={"approach": 1, "close": 4, "release": 6}
    )

    assert [module._semantic_waypoint_name(trajectory, row) for row in range(7)] == [
        "approach", "approach", "close", "close", "close", "release", "release"
    ]
    with pytest.raises(IndexError, match="exceeds"):
        module._semantic_waypoint_name(trajectory, 7)


def test_branch_reanchor_waits_for_confirmation_when_present():
    module = _module()

    assert module._branch_reanchor_waypoints(None) == ()
    assert module._branch_reanchor_waypoints(
        SimpleNamespace(waypoint_steps={"left_release": 10, "handover_confirm": 20})
    )[:2] == ("handover_confirm", "tree_transport")
    assert module._branch_reanchor_waypoints(
        SimpleNamespace(waypoint_steps={"left_release": 10})
    )[0] == "left_release"


def test_trace_status_arrays_are_one_to_one_with_executed_rows():
    module = _module()
    reset = {"left_grasp": False, "right_grasp": False,
             "grasp_assist_engaged": {}, "stage1": False, "stage2": False,
             "stage3": False,
             "gripper_contact_diagnostics": {
                 arm: [
                     {"force_n": 0.0, "touching": False,
                      "pad_fraction": float("nan"), "pad_valid": False,
                      "pad_in_band": False,
                      "contact_position": [float("nan")] * 3,
                      "pad_tip_position": [0.0, 0.0, 0.0],
                      "pad_base_position": [0.0, 0.0, 0.068]}
                     for _ in range(2)
                 ]
                 for arm in ("left", "right")
             }}
    rows = [
        {**reset, "left_grasp": True, "grasp_assist_engaged": {"left": True},
         "stage1": True,
         "gripper_contact_diagnostics": {
             **reset["gripper_contact_diagnostics"],
             "left": [
                 {"force_n": 1.5, "touching": True, "pad_fraction": 0.4,
                  "pad_valid": True, "pad_in_band": True,
                  "contact_position": [1.0, 2.0, 3.0],
                  "pad_tip_position": [0.0, 0.0, 0.0],
                  "pad_base_position": [0.0, 0.0, 0.068]},
                 {"force_n": 2.0, "touching": True, "pad_fraction": 0.6,
                  "pad_valid": True, "pad_in_band": True,
                  "contact_position": [4.0, 5.0, 6.0],
                  "pad_tip_position": [0.1, 0.2, 0.3],
                  "pad_base_position": [0.1, 0.2, 0.368]},
             ],
         }},
        {**reset, "right_grasp": True, "grasp_assist_engaged": {"right": True},
         "stage1": True, "stage2": True},
    ]

    trace = module._trace_status_arrays([reset, *rows])

    assert {
        "left_grasp", "right_grasp", "left_assist_engaged",
        "right_assist_engaged", "stage1_latched", "stage2_latched",
        "stage3_latched",
    } < set(trace)
    assert all(trace[name].shape == (2,) and trace[name].dtype == bool for name in (
        "left_grasp", "right_grasp", "left_assist_engaged",
        "right_assist_engaged", "stage1_latched", "stage2_latched",
        "stage3_latched",
    ))
    assert trace["left_finger_contact_force_n"].shape == (2, 2)
    assert trace["right_finger_pad_fraction"].shape == (2, 2)
    assert trace["left_finger_contact_position"].shape == (2, 2, 3)
    assert trace["right_finger_pad_tip_position"].shape == (2, 2, 3)
    assert trace["left_finger_pad_base_position"].shape == (2, 2, 3)
    assert trace["left_finger_touching"].dtype == bool
    assert trace["left_finger_pad_in_band"].dtype == bool
    assert trace["left_finger_contact_force_n"][0].tolist() == pytest.approx([1.5, 2.0])
    assert trace["left_finger_pad_fraction"][0].tolist() == pytest.approx([0.4, 0.6])
    assert trace["left_finger_pad_in_band"][0].tolist() == [True, True]
    np.testing.assert_allclose(
        trace["left_finger_contact_position"][0],
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )
    assert trace["left_grasp"].tolist() == [True, False]
    assert trace["right_grasp"].tolist() == [False, True]
    assert trace["stage2_latched"].tolist() == [False, True]
