import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from audit_putpot_quality_run import audit_bundle
from judo_isaaclab.putpot_quality import (
    deterministic_perturbation_cases,
    load_quality_config,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/putpot_quality_wave_v1.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bundle(tmp_path: Path):
    count = 60
    actions = np.zeros((count, 14), dtype=np.float32)
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[35:50, 0] = np.linspace(0.0, 0.1, 15)
    positions[50:, 0] = 0.1
    quaternion = np.tile([1.0, 0.0, 0.0, 0.0], (count, 1))
    pot = np.c_[positions, quaternion].astype(np.float32)
    left = pot.copy()
    right = pot.copy()
    left[:, 1] -= 0.1
    right[:, 1] += 0.1
    left[50:, :3] = [-0.2, -0.3, 0.4]
    right[50:, :3] = [-0.2, 0.3, 0.4]
    left_force = np.zeros((count, 2), dtype=np.float32)
    right_force = np.zeros((count, 2), dtype=np.float32)
    left_force[2:50] = 2.0
    right_force[20:50] = 2.0
    fractions = np.full((count, 2), 0.5, dtype=np.float32)
    trace = tmp_path / "trace.npz"
    np.savez_compressed(
        trace,
        actions=actions,
        pot_poses=pot,
        left_eef_poses=left,
        right_eef_poses=right,
        left_finger_forces_n=left_force,
        right_finger_forces_n=right_force,
        left_pad_fractions=fractions,
        right_pad_fractions=fractions,
        partial_trace=np.asarray(False),
    )
    contact = tmp_path / "contact.npz"
    supported = np.zeros(count, dtype=bool)
    supported[49:] = True
    opened = np.zeros(count, dtype=bool)
    opened[50:] = True
    np.savez_compressed(
        contact,
        left_contact_area_fractions=np.where(left_force > 0, 0.5, 0.0),
        right_contact_area_fractions=np.where(right_force > 0, 0.5, 0.0),
        left_flush_angles_deg=np.zeros((count, 2)),
        right_flush_angles_deg=np.zeros((count, 2)),
        supported=supported,
        left_open=opened,
        right_open=opened,
        stage_events=np.asarray(
            [
                "left_handle_stable",
                "right_handle_stable",
                "four_pad_latch",
                "bimanual_lift",
                "coordinated_transfer",
                "supported_lower",
                "open_both",
                "return_both_open_to_start",
            ]
        ),
        object_first_start_step=np.asarray(35),
        object_first_end_step=np.asarray(49),
        left_start_m=np.asarray([-0.2, -0.3, 0.4]),
        right_start_m=np.asarray([-0.2, 0.3, 0.4]),
    )
    components = {
        "left_arm": [-0.4, -0.4, 0.0],
        "right_arm": [-0.4, 0.4, 0.0],
        "left_gripper": [0.0, -0.2, 0.0],
        "right_gripper": [0.0, 0.2, 0.0],
        "left_wrist_camera": [-0.2, -0.3, 0.2],
        "right_wrist_camera": [-0.2, 0.3, 0.2],
        "pot_left_handle": [0.0, -0.2, 0.0],
        "pot_right_handle": [0.0, 0.2, 0.0],
    }
    collision_values = {}
    for name, position in components.items():
        collision_values[f"center__{name}"] = np.tile(position, (count, 1))
        collision_values[f"radius__{name}"] = np.asarray(0.02)
    collision = tmp_path / "collision.npz"
    np.savez_compressed(collision, **collision_values)

    demo = tmp_path / "demo.hdf5"
    with h5py.File(demo, "w") as handle:
        data = handle.create_group("data")
        data.attrs["ASSETS_INSTANCE_PATHS"] = json.dumps(
            {
                "pot": "PotSimple/cooking_pot_000001",
                "cooktop": "CooktopSimple/induction_cooktop_000001",
            }
        )
        group = data.create_group("demo_0")
        group.attrs["success"] = True
        group.create_dataset("actions", data=actions)
        group.create_group("states").create_dataset(
            "robot", data=np.zeros((count + 1, 3))
        )
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic-video")
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "pid": 100,
                "persistent": False,
                "shutdown": {"completion": "process_exit_observed"},
            }
        )
    )
    config = load_quality_config(CONFIG)
    cases = deterministic_perturbation_cases(config, joint_dof=14)
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text(
        json.dumps(
            {
                "joint_dof": 14,
                "outcomes": [
                    {"case_sha256": case["case_sha256"], "passed": True}
                    for case in cases
                ],
            }
        )
    )
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "status": "passed",
                "protocol": {
                    "quality_config": config.receipt(),
                    "quality_sidecars": {
                        "contact": {"sha256": _sha256(contact)},
                        "collision": {"sha256": _sha256(collision)},
                    },
                },
                "checks": {
                    "coded_task_success": True,
                    "accepted_task_success": True,
                    "bimanual_pick_observed": True,
                    "bimanual_transport_completed": True,
                    "pot_released": True,
                    "stable_support_window": True,
                    "one_reset": True,
                    "zero_inter_stage_resets": True,
                },
                "acceptance_checks": {"all": True},
                "provenance": {
                    "trace": {"sha256": _sha256(trace)},
                    "demonstration": {"sha256": _sha256(demo)},
                    "target_assets": {
                        "pot": {"path": "/objects/cooking_pot_000001"},
                        "cooktop": {"path": "/objects/induction_cooktop_000001"},
                    },
                },
                "video": {"sha256": _sha256(video)},
            }
        )
    )
    return {
        "result_json": result,
        "trace_npz": trace,
        "contact_telemetry_npz": contact,
        "collision_telemetry_npz": collision,
        "demo_hdf5": demo,
        "video": video,
        "runtime_receipt_json": runtime,
        "quality_config_json": CONFIG,
        "perturbation_outcomes_json": outcomes,
    }


def test_complete_quality_bundle_is_accepted(tmp_path):
    receipt = audit_bundle(
        **_bundle(tmp_path),
        media_probe=lambda _path: {
            "passed": True,
            "codec": "h264",
            "frame_count": 60,
        },
    )
    assert receipt["status"] == "accepted"
    assert receipt["failed_checks"] == []


def test_missing_contact_telemetry_is_terminal_and_not_inferred(tmp_path):
    bundle = _bundle(tmp_path)
    contact = bundle["contact_telemetry_npz"]
    with np.load(contact, allow_pickle=False) as handle:
        values = {
            name: handle[name]
            for name in handle.files
            if name != "left_flush_angles_deg"
        }
    contact.unlink()
    np.savez_compressed(contact, **values)
    receipt = audit_bundle(
        **bundle,
        media_probe=lambda _path: {"passed": True},
    )
    assert receipt["status"] == "failed"
    assert receipt["reason"] == "missing_required_quality_telemetry"
    assert receipt["missing_telemetry_fields"]["contact"] == [
        "left_flush_angles_deg"
    ]


def test_short_collision_sidecar_fails_full_sweep_gate(tmp_path):
    bundle = _bundle(tmp_path)
    collision = bundle["collision_telemetry_npz"]
    with np.load(collision, allow_pickle=False) as handle:
        values = {
            name: (handle[name][:-1] if name.startswith("center__") else handle[name])
            for name in handle.files
        }
    collision.unlink()
    np.savez_compressed(collision, **values)
    receipt = audit_bundle(
        **bundle,
        media_probe=lambda _path: {"passed": True, "codec": "h264"},
    )
    assert receipt["status"] == "failed"
    assert "swept_collision" in receipt["failed_checks"]
