import json
from pathlib import Path

import numpy as np
import pytest

from judo_isaaclab.putpot_quality import (
    GpuLease,
    audit_broad_contact,
    audit_object_first_motion,
    audit_perturbation_outcomes,
    audit_release_and_return,
    audit_swept_self_collision,
    deterministic_perturbation_cases,
    load_quality_config,
    validate_lane_contract,
    write_immutable_receipt,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/putpot_quality_wave_v1.json"


def _config():
    return load_quality_config(CONFIG)


def _contact_trace(*, steps=50, right_start=20, latch_start=30):
    left_forces = np.zeros((steps, 2))
    right_forces = np.zeros((steps, 2))
    left_forces[2:] = 2.0
    right_forces[right_start:] = 2.0
    values = {
        "left_forces_n": left_forces,
        "right_forces_n": right_forces,
        "left_pad_fractions": np.full((steps, 2), 0.5),
        "right_pad_fractions": np.full((steps, 2), 0.5),
        "left_contact_area_fractions": np.where(left_forces > 0, 0.5, 0.0),
        "right_contact_area_fractions": np.where(right_forces > 0, 0.5, 0.0),
        "left_flush_angles_deg": np.zeros((steps, 2)),
        "right_flush_angles_deg": np.zeros((steps, 2)),
        "pot_positions_m": np.zeros((steps, 3)),
    }
    if latch_start > right_start:
        values["right_forces_n"][right_start:latch_start] = 0.0
        values["right_contact_area_fractions"][right_start:latch_start] = 0.0
    return values


def test_quality_config_is_explicit_and_hash_pinned():
    config = _config()
    assert config.receipt()["mode"] == "putpot_quality_wave"
    assert len(config.sha256) == 64
    assert config.grasp["order"] == ["left", "right"]


def test_broad_contact_requires_left_first_and_fifteen_four_pad_frames():
    values = _contact_trace()
    receipt = audit_broad_contact(**values, config=_config())
    assert receipt["passed"] is True
    assert receipt["first_left_stable_step"] == 16
    assert receipt["first_four_pad_latch_step"] == 44


def test_broad_contact_rejects_right_before_left_is_stable():
    values = _contact_trace(right_start=5, latch_start=5)
    receipt = audit_broad_contact(**values, config=_config())
    assert receipt["passed"] is False
    assert receipt["left_first"] is False


def test_broad_contact_rejects_fingertip_and_pre_latch_object_motion():
    values = _contact_trace()
    values["right_pad_fractions"][-15:, 0] = 0.02
    values["pot_positions_m"][20:, 0] = 0.004
    receipt = audit_broad_contact(**values, config=_config())
    assert receipt["passed"] is False
    assert receipt["maximum_pre_latch_object_motion_m"] == pytest.approx(0.004)


def test_object_first_motion_reports_smooth_rigid_coordination():
    steps = 20
    positions = np.linspace([0, 0, 0], [0.2, 0, 0], steps)
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (steps, 1))
    pot = np.c_[positions, quaternions]
    left = pot.copy()
    right = pot.copy()
    left[:, 1] -= 0.1
    right[:, 1] += 0.1
    receipt = audit_object_first_motion(
        pot_poses=pot,
        left_eef_poses=left,
        right_eef_poses=right,
        planner_mode="object_first_rigid_weld",
        config=_config(),
    )
    assert receipt["passed"] is True
    assert receipt["maximum_grasp_translation_drift_m"] < 1e-12


def test_release_is_once_supported_and_returns_both_open_to_start():
    steps = 12
    supported = np.zeros(steps, dtype=bool)
    supported[7:] = True
    left_open = np.zeros(steps, dtype=bool)
    right_open = np.zeros(steps, dtype=bool)
    left_open[8:] = True
    right_open[8:] = True
    left = np.zeros((steps, 3))
    right = np.zeros((steps, 3))
    receipt = audit_release_and_return(
        stage_events=[
            "left_handle_stable",
            "right_handle_stable",
            "four_pad_latch",
            "bimanual_lift",
            "coordinated_transfer",
            "supported_lower",
            "open_both",
            "return_both_open_to_start",
        ],
        supported=supported,
        left_open=left_open,
        right_open=right_open,
        left_positions_m=left,
        right_positions_m=right,
        left_start_m=[0, 0, 0],
        right_start_m=[0, 0, 0],
        config=_config(),
    )
    assert receipt["passed"] is True
    assert receipt["bilateral_open_event_count"] == 1


def test_perturbations_are_fixed_seed_and_receipts_are_complete():
    first = deterministic_perturbation_cases(_config(), joint_dof=14)
    second = deterministic_perturbation_cases(_config(), joint_dof=14)
    assert first == second
    outcomes = [{"case_sha256": item["case_sha256"], "passed": True} for item in first]
    assert audit_perturbation_outcomes(first, outcomes, _config())["passed"] is True
    outcomes[-1]["passed"] = False
    assert audit_perturbation_outcomes(first, outcomes, _config())["passed"] is False


def _collision_components(steps=4):
    return {
        "left_arm": np.tile([0.0, -0.4, 0.0], (steps, 1)),
        "right_arm": np.tile([0.0, 0.4, 0.0], (steps, 1)),
        "left_gripper": np.tile([0.5, -0.2, 0.0], (steps, 1)),
        "right_gripper": np.tile([0.5, 0.2, 0.0], (steps, 1)),
        "left_wrist_camera": np.tile([0.3, -0.3, 0.2], (steps, 1)),
        "right_wrist_camera": np.tile([0.3, 0.3, 0.2], (steps, 1)),
        "pot_left_handle": np.tile([0.5, -0.2, 0.0], (steps, 1)),
        "pot_right_handle": np.tile([0.5, 0.2, 0.0], (steps, 1)),
    }


def test_swept_collision_allows_only_declared_handle_contacts():
    components = _collision_components()
    radii = {name: 0.02 for name in components}
    receipt = audit_swept_self_collision(
        component_centers_m=components,
        component_radii_m=radii,
        config=_config(),
    )
    assert receipt["passed"] is True
    exclusions = {tuple(item["components"]): item["exclusion_kind"] for item in receipt["pairs"]}
    assert exclusions[("left_gripper", "pot_left_handle")] == "intended_contact"


def test_swept_collision_rejects_wrist_camera_collision():
    components = _collision_components()
    components["right_wrist_camera"][:] = components["left_wrist_camera"]
    radii = {name: 0.02 for name in components}
    receipt = audit_swept_self_collision(
        component_centers_m=components,
        component_radii_m=radii,
        config=_config(),
    )
    assert receipt["passed"] is False
    assert any("wrist_camera" in " ".join(item["components"]) for item in receipt["collisions"])


def test_lane_contract_requires_one_gpu_and_pair_local_output(tmp_path):
    root = tmp_path / "pairs/000012"
    receipt = validate_lane_contract(
        pair_index=12,
        lane_id="node1-gpu3-pair12",
        cuda_visible_devices="3",
        output_root=root,
        config=_config(),
    )
    assert receipt["gpu_id"] == "3"
    with pytest.raises(ValueError, match="exactly one"):
        validate_lane_contract(
            pair_index=12,
            lane_id="node1-gpu3-pair12",
            cuda_visible_devices="3,4",
            output_root=root,
            config=_config(),
        )


def test_gpu_lease_is_exclusive_and_released(tmp_path):
    with GpuLease(tmp_path, "3", "lane-one"):
        assert (tmp_path / "gpu-3.lease").exists()
        with pytest.raises(FileExistsError):
            with GpuLease(tmp_path, "3", "lane-two"):
                pass
    assert not (tmp_path / "gpu-3.lease").exists()


def test_immutable_receipt_cannot_be_overwritten(tmp_path):
    path = tmp_path / "quality_receipt.json"
    digest = write_immutable_receipt(path, {"accepted": True})
    assert len(digest) == 64
    assert json.loads(path.read_text()) == {"accepted": True}
    with pytest.raises(FileExistsError):
        write_immutable_receipt(path, {"accepted": False})
