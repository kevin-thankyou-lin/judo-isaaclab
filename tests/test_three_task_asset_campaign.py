import json

import h5py
import numpy as np
import pytest

from run_three_task_asset_campaign import (
    _reusable_classification,
    enumerate_pairs,
    validate_asset_inventory,
    validate_demo,
)
from run_replay_success_semantic_audit import (
    reusable_semantic_result,
    select_replay_success_pairs,
    semantic_acceptance_satisfied,
)


def _dataset(path, assets):
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["ASSETS_INSTANCE_PATHS"] = json.dumps(assets)


def test_enumeration_requires_exact_unique_pair_set(tmp_path):
    for index in range(2):
        _dataset(tmp_path / f"task_{index}.hdf5", {"object": f"Object/object_{index:03d}"})
    task = {
        "name": "task", "expected_pairs": 2,
        "source_dataset": str(tmp_path / "task_0.hdf5"),
        "dataset_globs": [str(tmp_path / "task_*.hdf5")],
    }
    pairs = enumerate_pairs(task)
    assert [pair["pair_id"] for pair in pairs] == ["object_000", "object_001"]


def test_enumeration_fails_closed_on_wrong_count(tmp_path):
    _dataset(tmp_path / "task_0.hdf5", {"object": "Object/object_000"})
    with pytest.raises(RuntimeError, match="expected 2"):
        enumerate_pairs({
            "name": "task", "expected_pairs": 2,
            "source_dataset": str(tmp_path / "task_0.hdf5"),
            "dataset_globs": [str(tmp_path / "*.hdf5")],
        })


def test_asset_inventory_fails_before_simulator_startup(tmp_path):
    dataset = tmp_path / "task_0.hdf5"
    _dataset(dataset, {"object": "Object/object_000"})
    task = {
        "name": "task",
        "objects_root": str(tmp_path / "objects"),
    }
    pair = {
        "dataset": str(dataset),
        "pair_id": "object_000",
        "assets": {"object": "Object/object_000"},
    }
    with pytest.raises(RuntimeError, match="missing 1 official asset directories"):
        validate_asset_inventory(task, [pair])

    (tmp_path / "objects/Object/object_000").mkdir(parents=True)
    assert validate_asset_inventory(task, [pair]) == {
        "objects_root": str((tmp_path / "objects").resolve()),
        "pairs": 1,
        "asset_directories": 1,
    }


def test_demo_validation_checks_alignment_and_assets(tmp_path):
    path = tmp_path / "demo.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["ASSETS_INSTANCE_PATHS"] = json.dumps({"object": "Object/object_000"})
        demo = data.create_group("demo_0")
        demo.attrs["num_samples"] = 2
        demo.attrs["success"] = True
        demo.create_dataset("actions", data=np.zeros((2, 14)))
        demo.create_dataset("states/object", data=np.zeros((3, 7)))
        demo.create_dataset("obs/proprio", data=np.zeros((2, 14)))
    value = validate_demo(path, {"object": "Object/object_000"})
    assert value["actions"] == 2
    assert len(value["sha256"]) == 64


def test_reuses_only_hash_verified_classification(tmp_path):
    target = tmp_path / "target.hdf5"
    video = tmp_path / "video.mp4"
    trace = tmp_path / "trace.npz"
    target.write_bytes(b"target")
    video.write_bytes(b"video")
    trace.write_bytes(b"trace")
    import hashlib

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "status": "passed",
        "mode": "replay",
        "acceptance_checks": {"technical": True},
        "provenance": {
            "target_dataset": {"sha256": digest(target)},
            "trace": {"path": str(trace), "sha256": digest(trace)},
        },
        "video": {"path": str(video), "sha256": digest(video)},
    }
    assert _reusable_classification(result, str(target))
    video.write_bytes(b"corrupt")
    assert not _reusable_classification(result, str(target))


def test_semantic_audit_selects_only_plain_replay_successes():
    pairs = [
        {"pair_id": "replay"},
        {"pair_id": "adapted"},
        {"pair_id": "failed"},
    ]
    ledger = {
        "pairs": {
            "replay": {"status": "accepted", "method": "source_action_replay"},
            "adapted": {
                "status": "accepted",
                "method": "deterministic_semantic_skill",
            },
            "failed": {
                "status": "adaptation_failed",
                "method": "deterministic_semantic_skill",
            },
        }
    }
    assert select_replay_success_pairs(pairs, ledger) == [{"pair_id": "replay"}]


def test_semantic_audit_reuses_hash_verified_task_failure(tmp_path):
    target = tmp_path / "target.hdf5"
    video = tmp_path / "skill.mp4"
    trace = tmp_path / "skill_trace.npz"
    target.write_bytes(b"target")
    video.write_bytes(b"video")
    trace.write_bytes(b"trace")
    import hashlib

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "status": "failed",
        "mode": "skill",
        "acceptance_checks": {
            "coded_task_success": False,
            "fully_decodable": True,
            "h264_nonempty": True,
            "one_reset": True,
            "real_target_assets": True,
            "zero_inter_stage_resets": True,
        },
        "provenance": {
            "target_dataset": {"sha256": digest(target)},
            "trace": {"path": str(trace), "sha256": digest(trace)},
        },
        "video": {"path": str(video), "sha256": digest(video)},
    }
    assert reusable_semantic_result(result, str(target))
    trace.write_bytes(b"corrupt")
    assert not reusable_semantic_result(result, str(target))


def test_semantic_audit_excludes_only_direct_replay_failure_control():
    result = {
        "mode": "skill",
        "status": "failed",
        "checks": {"coded_task_success": True},
        "acceptance_checks": {
            "coded_task_success": True,
            "all_stages_latched": True,
            "stable_support_window": True,
            "fully_decodable": True,
            "direct_source_action_replay_failed": False,
        },
    }
    assert semantic_acceptance_satisfied(result)
    result["acceptance_checks"]["stable_support_window"] = False
    assert not semantic_acceptance_satisfied(result)
