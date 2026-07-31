import json

import numpy as np
import pytest

from judo_isaaclab.adaptation import (
    TaskAdaptationBundle,
    TrialEvidence,
    asset_relative_grasp_pose,
    corrected_insert_offset,
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from hangmug_task_adaptation_agent import (
    _discard_incomplete_trailing_trials,
    _extend,
    _fallback_trial,
    _rendered_mpc_subtask_passed,
    _resumable_stage_record,
)


def _bundle():
    return {
        "task_name": "HangMugOnTree-v0",
        "dataset": "/tmp/source.hdf5",
        "episode": "demo_0",
        "objects_root": "/tmp/objects",
        "source_assets": {"mug": "mug_000", "mug_tree": "tree_000"},
        "target_assets": {"mug": "mug_029", "mug_tree": "tree_037"},
        "success_check": "check_task_success",
        "checkpoint_state": 0,
        "correspondences": {},
        "stages": [
            {
                "name": "bootstrap",
                "target_name": "handover_latched",
                "start_state": 0,
                "target_state": 4,
                "horizon": 5,
            },
            {
                "name": "release",
                "target_name": "hang_complete",
                "start_state": 5,
                "target_state": 8,
                "horizon": 4,
            },
        ],
    }


def test_bundle_requires_contiguous_stages_and_real_asset_swap(tmp_path):
    path = tmp_path / "task.json"
    path.write_text(json.dumps(_bundle()))
    loaded = TaskAdaptationBundle.load(path)
    assert loaded.stages[1].start_state == 5

    broken = _bundle()
    broken["stages"][1]["start_state"] = 6
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="contiguous"):
        TaskAdaptationBundle.load(path)


def test_trial_evidence_and_signed_insert_correction(tmp_path):
    result = tmp_path / "trial.json"
    result.write_text(
        json.dumps(
            {
                "status": "failed",
                "best_sample_reached_keyframe": False,
                "repeat_evaluation": {
                    "best_sample": {
                        "keyframe_position_error_vector_m_mean": [
                            0.02,
                            -0.004,
                            0.001,
                        ]
                    }
                },
            }
        )
    )
    evidence = TrialEvidence.from_result(result, tmp_path / "controls.npz")
    assert corrected_insert_offset([0.0, 0.0, 0.0], evidence) == pytest.approx(
        [-0.01, 0.002, -0.0005]
    )


def test_cli_vector_avoids_negative_scientific_notation():
    command = []
    _extend(command, "source_branch_points", [0.00001, -0.000084349, 0.2])
    assert command == [
        "--source-branch-points",
        "0.000010000000",
        "-0.000084349000",
        "0.200000000000",
    ]


def test_resume_reuses_latest_incomplete_stage():
    ledger = {
        "stages": [
            {"name": "insert", "status": "failed", "trials": [{"index": 0}]}
        ]
    }
    record = _resumable_stage_record(ledger, "insert")
    assert record is ledger["stages"][0]
    assert record["status"] == "running"
    assert len(record["trials"]) == 1


def test_resume_discards_only_incomplete_trailing_trials(tmp_path):
    result = tmp_path / "complete.json"
    controls = tmp_path / "complete.npz"
    result.write_text("{}")
    controls.write_bytes(b"npz")
    record = {
        "trials": [
            {"result": str(result), "controls": str(controls)},
            {
                "result": str(tmp_path / "missing.json"),
                "controls": str(tmp_path / "missing.npz"),
            },
        ]
    }
    _discard_incomplete_trailing_trials(record)
    assert record["trials"] == [
        {"result": str(result), "controls": str(controls)}
    ]


def test_fallback_requires_robust_grasp_and_selects_best_pose():
    trials = []
    for index, position in enumerate((0.014, 0.011)):
        trials.append(
            {
                "index": index,
                "controls": f"trial{index}.npz",
                "metrics": {
                    "count": 8,
                    "keyframe_right_grasp_count": 8,
                    "keyframe_stage2_count": 8,
                    "keyframe_position_error_m_max": position,
                    "keyframe_rotation_error_rad_max": 0.2,
                },
            }
        )
    selected = _fallback_trial(
        {"trials": trials},
        {"position_tolerance_m": 0.015, "rotation_tolerance_rad": 0.22},
    )
    assert selected["index"] == 1


def test_fallback_supports_stage_specific_metric_thresholds():
    selected = _fallback_trial(
        {
            "trials": [
                {
                    "index": 0,
                    "metrics": {
                        "acceptance_success_count": 4,
                        "post_keyframe_target_fraction_mean": 0.9,
                    },
                },
                {
                    "index": 1,
                    "metrics": {
                        "acceptance_success_count": 5,
                        "post_keyframe_target_fraction_mean": 0.83,
                    },
                },
            ]
        },
        {
            "metric_min": {
                "acceptance_success_count": 5,
                "post_keyframe_target_fraction_mean": 0.8,
            }
        },
    )
    assert selected["index"] == 1


def test_rendered_fallback_requires_mpc_success_and_decodable_video(tmp_path):
    result = tmp_path / "render.json"
    result.write_text(
        json.dumps(
            {
                "status": "passed",
                "subtask_complete": {"mpc": True},
                "checks": {"fully_decodable": True},
            }
        )
    )
    assert _rendered_mpc_subtask_passed(result)
    payload = json.loads(result.read_text())
    payload["subtask_complete"]["mpc"] = False
    result.write_text(json.dumps(payload))
    assert not _rendered_mpc_subtask_passed(result)


def test_asset_relative_grasp_tracks_target_pose_and_dimensions():
    identity = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    source_eef = identity.copy()
    source_eef[0] = 0.1
    half = np.sqrt(0.5)
    target_object = np.asarray(
        [1.0, 2.0, 3.0, half, 0.0, 0.0, half]
    )

    target = asset_relative_grasp_pose(
        source_eef,
        identity,
        target_object,
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
    )

    assert target[:3] == pytest.approx([1.0, 2.2, 3.0], abs=1.0e-6)
    assert target[3:] == pytest.approx(target_object[3:], abs=1.0e-6)


def test_asset_relative_grasp_preserves_vertical_tool_standoff():
    source_object = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    source_eef = source_object.copy()
    source_eef[:3] = [0.1, 0.2, 0.3]

    target = asset_relative_grasp_pose(
        source_eef,
        source_object,
        source_object,
        [1.0, 1.0, 1.0],
        [2.0, 0.5, 1.5],
    )

    assert target[:3] == pytest.approx([0.2, 0.1, 0.3], abs=1.0e-6)


def test_asset_relative_grasp_scales_contact_but_not_tool_transform():
    identity = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    source_contact = identity.copy()
    source_contact[:3] = [0.1, 0.2, 0.3]
    source_eef = identity.copy()
    source_eef[:3] = [0.1, 0.2, 0.5]

    target = asset_relative_grasp_pose(
        source_eef,
        identity,
        identity,
        [1.0, 1.0, 1.0],
        [2.0, 0.5, 1.5],
        source_contact_pose=source_contact,
    )

    assert target[:3] == pytest.approx([0.2, 0.1, 0.65], abs=1.0e-6)


def test_asset_relative_grasp_rejects_invalid_sizes():
    pose = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="dimensions must be positive"):
        asset_relative_grasp_pose(
            pose, pose, pose, [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]
        )
