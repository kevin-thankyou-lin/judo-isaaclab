import json

import h5py
import numpy as np
import pytest

from run_three_task_asset_campaign import enumerate_pairs, validate_demo


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
