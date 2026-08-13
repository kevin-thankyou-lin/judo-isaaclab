import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "src"))

from apply_putpot_quality_perturbation import adapted_command
from judo_isaaclab.putpot_quality import (
    deterministic_perturbation_cases,
    load_quality_config,
)
from judo_isaaclab.putpot_quality_runtime import (
    collision_sidecar_fields,
    load_and_validate_perturbation_case,
    perturb_grasp_pose,
    perturb_joint_action,
    quality_stage_telemetry,
)


CONFIG = ROOT / "configs/putpot_quality_wave_v1.json"


def test_real_adapter_binds_hashed_case_and_runner_applies_both_noise_spaces(tmp_path):
    config = load_quality_config(CONFIG)
    case = deterministic_perturbation_cases(config, joint_dof=14)[0]
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    loaded = load_and_validate_perturbation_case(
        path, deterministic_perturbation_cases(config, joint_dof=14)
    )
    command = adapted_command(
        path,
        ["python", "runner.py", "--quality-config-json", str(CONFIG)],
    )
    assert command[-2:] == ["--quality-perturbation-case-json", str(path.resolve())]
    pose = perturb_grasp_pose([0, 0, 0, 1, 0, 0, 0], loaded)
    assert np.allclose(pose[:3], loaded["grasp_translation_m"])
    action = perturb_joint_action(np.zeros(14), loaded)
    assert action[6] == action[13] == 0.0
    assert np.any(action[np.r_[0:6, 7:13]] != 0.0)
    tampered = dict(case)
    tampered["joint_action_rad"] = [0.0] * 14
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="hash"):
        load_and_validate_perturbation_case(path, [case])


def _row(step, *, left_quality, right_quality, support=False, stage1=False):
    return {
        "left_contact_area_fractions": [0.5, 0.5] if left_quality else [0.0, 0.0],
        "right_contact_area_fractions": [0.5, 0.5] if right_quality else [0.0, 0.0],
        "left_flush_angles_deg": [2.0, 2.0],
        "right_flush_angles_deg": [2.0, 2.0],
        "left_finger_forces_n": [2.0, 2.0] if left_quality else [0.0, 0.0],
        "right_finger_forces_n": [2.0, 2.0] if right_quality else [0.0, 0.0],
        "support_geometry_now": support,
        "stage1": stage1,
        "left_grasp": left_quality,
        "right_grasp": right_quality,
        "left_eef_pose": [-0.2, -0.3, 0.4, 1, 0, 0, 0],
        "right_eef_pose": [-0.2, 0.3, 0.4, 1, 0, 0, 0],
        "quality_collision_centers_m": {
            "left_arm__link": [-0.3, -0.3, 0.2],
            "right_arm__link": [-0.3, 0.3, 0.2],
        },
    }


def test_measured_sidecars_encode_left_first_latch_object_window_and_one_open():
    rows = [_row(-1, left_quality=False, right_quality=False)]
    actions = np.zeros((55, 14), dtype=np.float64)
    actions[:, [6, 13]] = 0.0
    for step in range(55):
        left = 2 <= step < 45
        right = 20 <= step < 45
        support = step >= 40
        rows.append(
            _row(
                step,
                left_quality=left,
                right_quality=right,
                support=support,
                stage1=step >= 35,
            )
        )
    actions[45:, [6, 13]] = -0.0475
    contact = quality_stage_telemetry(
        samples=rows,
        actions=actions,
        left_start_m=[-0.2, -0.3, 0.4],
        right_start_m=[-0.2, 0.3, 0.4],
        stable_steps=15,
        minimum_force_n=1.0,
        minimum_area_fraction=0.25,
        maximum_flush_angle_deg=12.0,
        return_tolerance_m=0.015,
    )
    assert contact["object_first_start_step"].item() == 34
    assert contact["object_first_end_step"].item() == 40
    assert contact["stage_events"].tolist() == [
        "left_handle_stable",
        "right_handle_stable",
        "four_pad_latch",
        "bimanual_lift",
        "coordinated_transfer",
        "supported_lower",
        "open_both",
        "return_both_open_to_start",
    ]


def test_collision_sidecar_covers_every_measured_body_frame():
    rows = [_row(-1, left_quality=False, right_quality=False)]
    rows.extend(_row(i, left_quality=False, right_quality=False) for i in range(3))
    model = {
        "radii": {"left_arm__link": 0.1, "right_arm__link": 0.1},
        "groups": {
            "left_arm": ["left_arm__link"],
            "right_arm": ["right_arm__link"],
        },
        "structural": [],
    }
    fields = collision_sidecar_fields(rows, model)
    assert fields["center__left_arm__link"].shape == (3, 3)
    assert json.loads(fields["component_groups_json"].item())["left_arm"] == [
        "left_arm__link"
    ]
