from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from build_putpot_quality_targets import (
    build_target_hdf5,
    validate_asset_catalog,
    validate_source_hdf5,
)


def _make_assets(root: Path, *, skip_pot: int | None = None) -> None:
    for index in range(40):
        if index != skip_pot:
            pot = root / "objects" / "PotSimple" / f"cooking_pot_{index:06d}"
            pot.mkdir(parents=True)
            (pot / f"cooking_pot_{index:06d}.usd").write_text("pot", encoding="utf-8")
        cooktop = root / "objects" / "CooktopSimple" / f"induction_cooktop_{index:06d}"
        cooktop.mkdir(parents=True)
        (cooktop / f"induction_cooktop_{index:06d}.usd").write_text(
            "cooktop", encoding="utf-8"
        )


def _make_source(path: Path, *, equal_length_states: bool = False) -> None:
    actions = np.arange(12, dtype=np.float32).reshape(3, 4)
    state_count = len(actions) if equal_length_states else len(actions) + 1
    with h5py.File(path, "w") as handle:
        handle.attrs["root_marker"] = "unchanged"
        data = handle.create_group("data")
        data.attrs["ASSETS_INSTANCE_PATHS"] = json.dumps(
            {
                "pot": "PotSimple/cooking_pot_000000",
                "cooktop": "CooktopSimple/induction_cooktop_000000",
            }
        )
        data.attrs["total"] = 999999  # Intentionally stale; demo_* is authoritative.
        demo = data.create_group("demo_7")
        demo.attrs["num_samples"] = len(actions)
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("processed_actions", data=np.vstack([actions, actions[-1]]))
        states = demo.create_group("states")
        states.create_dataset("arm/joint_position", data=np.arange(state_count * 2).reshape(-1, 2))
        obs = demo.create_group("obs")
        obs.create_dataset("joint_position", data=np.arange(len(actions) * 2).reshape(-1, 2))


def _dataset_values(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        def record(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset):
                values[name] = obj[()]

        handle.visititems(record)
    return values


def test_build_target_preserves_datasets_and_writes_verified_receipt(tmp_path: Path) -> None:
    task = tmp_path / "task1"
    _make_assets(task)
    source = tmp_path / "source.hdf5"
    output = tmp_path / "targets" / "pot_000013.hdf5"
    _make_source(source)

    receipt = build_target_hdf5(source, task, 13, output)

    assert output.is_file()
    assert _dataset_values(output).keys() == _dataset_values(source).keys()
    for name, source_value in _dataset_values(source).items():
        np.testing.assert_array_equal(_dataset_values(output)[name], source_value)
    with h5py.File(source, "r") as source_handle, h5py.File(output, "r") as output_handle:
        assert output_handle.attrs["root_marker"] == source_handle.attrs["root_marker"]
        assert output_handle["data"].attrs["total"] == 999999
        assert json.loads(output_handle["data"].attrs["ASSETS_INSTANCE_PATHS"]) == {
            "pot": "PotSimple/cooking_pot_000013",
            "cooktop": "CooktopSimple/induction_cooktop_000013",
        }
    assert receipt["source_validation"]["computed_total_actions"] == 3
    assert receipt["source_validation"]["demos"][0]["name"] == "demo_7"
    receipt_path = output.with_suffix(".hdf5.provenance.json")
    receipt_bytes = receipt_path.read_bytes()
    assert json.loads(receipt_bytes) == receipt
    expected_receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    assert output.with_suffix(".hdf5.provenance.json.sha256").read_text().split()[0] == (
        expected_receipt_sha
    )


def test_builder_refuses_overwrite_by_default(tmp_path: Path) -> None:
    task = tmp_path / "task1"
    _make_assets(task)
    source = tmp_path / "source.hdf5"
    output = tmp_path / "pot_000000.hdf5"
    _make_source(source)
    build_target_hdf5(source, task, 0, output)
    original_sha = hashlib.sha256(output.read_bytes()).hexdigest()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_target_hdf5(source, task, 0, output)

    assert hashlib.sha256(output.read_bytes()).hexdigest() == original_sha


def test_nonstandard_equal_length_source_requires_explicit_acknowledgement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.hdf5"
    _make_source(source, equal_length_states=True)

    with pytest.raises(ValueError, match="expected actions \\+ 1"):
        validate_source_hdf5(source)

    result = validate_source_hdf5(source, acknowledge_nonstandard_alignment=True)
    assert result["demos"][0]["alignment"] == "equal_length"
    assert result["acknowledged_nonstandard_alignment"] is True


def test_asset_catalog_requires_all_40_contiguous_pairs(tmp_path: Path) -> None:
    task = tmp_path / "task1"
    _make_assets(task, skip_pot=17)

    with pytest.raises(ValueError, match="missing=\\[17\\]"):
        validate_asset_catalog(task)
