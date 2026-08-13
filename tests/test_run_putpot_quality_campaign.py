import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from run_putpot_quality_campaign import build_plan, execute_plan
from run_putpot_skill_program import (
    _collision_clear_peer_pregrasp,
    _critic_owned_precontact_pad_balance,
    _quality_left_first_local_mpc_enabled,
    _quality_source_contact_requires_sequential_corridor,
    _quality_static_centering_contract_missing,
    _robot_arm_registry_key,
    _source_contact_requires_acquisition_only,
    _source_left_first_requires_measured_corridor,
    _static_precontact_requires_acquisition_only,
    _translate_source_corridor_endpoints,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/putpot_quality_wave_v1.json"


def test_quality_mode_allows_explicit_left_first_without_legacy_calibration():
    assert not _source_left_first_requires_measured_corridor(
        requested=True,
        has_measured_corridor=False,
        quality_mode=True,
    )
    assert _source_left_first_requires_measured_corridor(
        requested=True,
        has_measured_corridor=False,
        quality_mode=False,
    )
    assert not _source_left_first_requires_measured_corridor(
        requested=True,
        has_measured_corridor=True,
        quality_mode=False,
    )


def test_quality_source_contact_repair_requires_left_first_measured_corridor():
    assert _source_contact_requires_acquisition_only(
        requested=True,
        acquisition_only=False,
        quality_mode=False,
    )
    assert not _source_contact_requires_acquisition_only(
        requested=True,
        acquisition_only=False,
        quality_mode=True,
    )
    assert _quality_source_contact_requires_sequential_corridor(
        requested=True,
        acquisition_only=False,
        quality_mode=True,
        has_measured_corridor=False,
        left_first=True,
    )
    assert _quality_source_contact_requires_sequential_corridor(
        requested=True,
        acquisition_only=False,
        quality_mode=True,
        has_measured_corridor=True,
        left_first=False,
    )
    assert not _quality_source_contact_requires_sequential_corridor(
        requested=True,
        acquisition_only=False,
        quality_mode=True,
        has_measured_corridor=True,
        left_first=True,
    )


def test_quality_static_centering_requires_same_sequential_corridor_sample():
    assert _static_precontact_requires_acquisition_only(
        requested=True,
        acquisition_only=False,
        quality_mode=False,
    )
    assert _quality_static_centering_contract_missing(
        requested=True,
        acquisition_only=False,
        quality_mode=True,
        source_contact_requested=True,
        has_measured_corridor=True,
        left_first=True,
        same_calibration_sample=False,
    )
    assert not _quality_static_centering_contract_missing(
        requested=True,
        acquisition_only=False,
        quality_mode=True,
        source_contact_requested=True,
        has_measured_corridor=True,
        left_first=True,
        same_calibration_sample=True,
    )


def test_quality_local_mpc_requires_full_left_first_failed_trace_contract():
    required = {
        "requested": True,
        "acquisition_only": False,
        "quality_mode": True,
        "left_first": True,
        "has_measured_corridor": True,
        "source_contact_requested": True,
    }
    assert _quality_left_first_local_mpc_enabled(**required)
    for name in (
        "requested",
        "quality_mode",
        "left_first",
        "has_measured_corridor",
        "source_contact_requested",
    ):
        rejected = dict(required)
        rejected[name] = False
        assert not _quality_left_first_local_mpc_enabled(**rejected)
    acquisition_only = dict(required)
    acquisition_only["acquisition_only"] = True
    assert not _quality_left_first_local_mpc_enabled(**acquisition_only)


def test_semantic_arm_labels_map_to_live_yam_registry_keys():
    assert _robot_arm_registry_key("left") == "left_arm"
    assert _robot_arm_registry_key("right") == "right_arm"
    try:
        _robot_arm_registry_key("peer")
    except ValueError as error:
        assert "left or right" in str(error)
    else:
        raise AssertionError("invalid semantic arm label was accepted")


def test_collision_clear_peer_pregrasp_moves_only_outward_position():
    pregrasp = [0.72, -0.16, 0.91, 0.1, 0.2, 0.3, 0.9]
    pot = [0.71, 0.09, 0.81, 1.0, 0.0, 0.0, 0.0]
    staged, receipt = _collision_clear_peer_pregrasp(
        pregrasp, pot, clearance_m=0.025
    )
    before = np.asarray(pregrasp[:3]) - np.asarray(pot[:3])
    after = staged[:3] - np.asarray(pot[:3])

    assert np.linalg.norm(after) == pytest.approx(
        np.linalg.norm(before) + 0.025
    )
    assert staged[3:] == pytest.approx(pregrasp[3:])
    assert receipt["translation_norm_m"] == pytest.approx(0.025)
    assert receipt["orientation_unchanged"]
    assert receipt["grasp_endpoint_unchanged"]


def test_critic_owned_precontact_pad_balance_maximizes_edge_margin(tmp_path):
    trace = tmp_path / "trace.npz"
    pot = np.asarray(
        [
            [0.7, 0.1, 0.8, 1.0, 0.0, 0.0, 0.0],
            [0.7, 0.1, 0.8, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    np.savez_compressed(
        trace,
        pot_poses=pot,
        left_finger_forces_n=[[0.0, 0.0], [3.0, 4.0]],
        left_pad_fractions=[[np.nan, np.nan], [0.45, -0.05]],
        left_pad_axes_world=[
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        partial_trace=np.asarray(False),
    )
    expected_translation = [0.0, -0.02041984274983406, 0.0]
    critic = tmp_path / "critic.json"
    critic.write_text(
        json.dumps(
            {
                "lane_id": "pair15",
                "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                "pad_balance_calibration": {
                    "step": 1,
                    "classification": "two_pad_force_backed_edge_only",
                    "left_finger_forces_n": [3.0, 4.0],
                    "left_pad_fractions": [0.45, -0.05],
                    "finger_pad_axis_extent_m": 0.06806614430096655,
                    "precontact_translation_world_m": expected_translation,
                },
            }
        )
    )

    receipt = _critic_owned_precontact_pad_balance(
        trace,
        critic,
        1,
        lane_id="pair15",
        minimum_force_n=1.0,
        maximum_pre_latch_motion_m=0.003,
        maximum_translation_m=0.025,
    )

    assert receipt["translation_world_m"] == pytest.approx(
        expected_translation
    )
    assert receipt["predicted_pad_fractions"] == pytest.approx([0.75, 0.25])
    assert receipt["predicted_minimum_edge_margin"] == pytest.approx(0.25)
    assert receipt["orientation_unchanged"]


def test_critic_owned_precontact_pad_balance_caps_before_measured_collision(tmp_path):
    trace = tmp_path / "trace.npz"
    pot = np.asarray(
        [
            [0.7, 0.1, 0.8, 1.0, 0.0, 0.0, 0.0],
            [0.7, 0.1, 0.8, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    np.savez_compressed(
        trace,
        pot_poses=pot,
        left_finger_forces_n=[[0.0, 0.0], [3.0, 4.0]],
        left_pad_fractions=[[np.nan, np.nan], [0.45, -0.05]],
        left_pad_axes_world=[
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        partial_trace=np.asarray(False),
    )
    expected_uncapped = [0.0, -0.02041984274983406, 0.0]
    critic = tmp_path / "critic.json"
    critic.write_text(
        json.dumps(
            {
                "lane_id": "pair15",
                "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                "pad_balance_calibration": {
                    "step": 1,
                    "classification": "two_pad_force_backed_edge_only",
                    "left_finger_forces_n": [3.0, 4.0],
                    "left_pad_fractions": [0.45, -0.05],
                    "finger_pad_axis_extent_m": 0.06806614430096655,
                    "precontact_translation_world_m": expected_uncapped,
                },
            }
        )
    )

    receipt = _critic_owned_precontact_pad_balance(
        trace,
        critic,
        1,
        lane_id="pair15",
        minimum_force_n=1.0,
        minimum_pad_fraction_margin=0.1,
        maximum_pre_latch_motion_m=0.003,
        maximum_translation_m=0.025,
        applied_translation_cap_m=0.015,
    )

    assert receipt["uncapped_translation_world_m"] == pytest.approx(
        expected_uncapped
    )
    assert receipt["translation_world_m"] == pytest.approx([0.0, -0.015, 0.0])
    assert receipt["translation_norm_m"] == pytest.approx(0.015)
    assert receipt["translation_was_capped"]
    applied_delta = 0.30 * 0.015 / np.linalg.norm(expected_uncapped)
    assert receipt["predicted_pad_fractions"] == pytest.approx(
        np.asarray([0.45, -0.05]) + applied_delta
    )
    assert receipt["predicted_minimum_edge_margin"] > 0.1


def test_measured_static_translation_moves_both_source_corridor_endpoints():
    pregrasp = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    grasp = [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0]
    translated_pregrasp, translated_grasp = _translate_source_corridor_endpoints(
        pregrasp,
        grasp,
        {"translation_world_m": [-0.1, 0.2, -0.3]},
    )
    assert translated_pregrasp.tolist() == [
        0.9,
        2.2,
        2.7,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert translated_grasp.tolist() == [
        3.9,
        5.2,
        5.7,
        1.0,
        0.0,
        0.0,
        0.0,
    ]


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
                "--quality-contact-telemetry-npz",
                "{attempt_root}/contact.npz",
                "--quality-collision-telemetry-npz",
                "{attempt_root}/collision.npz",
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
parser.add_argument('--quality-contact-telemetry-npz')
parser.add_argument('--quality-collision-telemetry-npz')
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


def test_default_plan_uses_real_physical_perturbation_adapter(tmp_path):
    plan = _plan(tmp_path, None)
    assert Path(plan["perturbation_adapter"]).name == (
        "apply_putpot_quality_perturbation.py"
    )
    assert "--quality-perturbation-case-json" not in plan["attempts"][0]["command"]


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
