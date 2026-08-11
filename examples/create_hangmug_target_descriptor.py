"""Recover one missing HangMug target-state descriptor without copying actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_first_state(source: h5py.Group, target: h5py.Group) -> None:
    for name, value in source.items():
        if isinstance(value, h5py.Group):
            child = target.create_group(name)
            for key, attribute in value.attrs.items():
                child.attrs[key] = attribute
            _copy_first_state(value, child)
            continue
        if value.ndim < 1 or value.shape[0] < 1:
            raise ValueError(f"state dataset lacks an initial sample: {value.name}")
        child = target.create_dataset(name, data=np.asarray(value[0:1]))
        for key, attribute in value.attrs.items():
            child.attrs[key] = attribute


def create_target_descriptor(
    *,
    template_dataset: str | Path,
    corrupt_original: str | Path,
    canonical_action_source: str | Path,
    output: str | Path,
    template_assets: dict[str, str],
    target_assets: dict[str, str],
    expected_template_sha256: str,
    expected_corrupt_sha256: str,
    expected_action_source_sha256: str,
) -> dict[str, object]:
    """Copy only target initial state; never copy a second demo's actions."""

    template_dataset = Path(template_dataset).resolve()
    corrupt_original = Path(corrupt_original).resolve()
    canonical_action_source = Path(canonical_action_source).resolve()
    output = Path(output).resolve()
    if _sha256(template_dataset) != expected_template_sha256:
        raise ValueError("template dataset hash mismatch")
    if _sha256(corrupt_original) != expected_corrupt_sha256:
        raise ValueError("corrupt original hash mismatch")
    if _sha256(canonical_action_source) != expected_action_source_sha256:
        raise ValueError("canonical action source hash mismatch")

    provenance = {
        "schema_version": 1,
        "artifact_role": "target_initial_state_descriptor_not_source_demo",
        "state_template": {
            "path": str(template_dataset),
            "sha256": expected_template_sha256,
        },
        "replaced_corrupt_input": {
            "path": str(corrupt_original),
            "sha256": expected_corrupt_sha256,
        },
        "target_assets": target_assets,
        "canonical_action_source": {
            "path": str(canonical_action_source),
            "sha256": expected_action_source_sha256,
        },
        "action_policy": "one synthetic zero action; never used as source actions",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(template_dataset, "r") as source:
        raw_assets = source["data"].attrs["ASSETS_INSTANCE_PATHS"]
        if isinstance(raw_assets, bytes):
            raw_assets = raw_assets.decode("utf-8")
        if json.loads(str(raw_assets)) != template_assets:
            raise ValueError("template asset mapping mismatch")
        source_demo = source["data/demo_0"]
        action_shape = source_demo["actions"].shape
        if len(action_shape) != 2 or action_shape[1] <= 0:
            raise ValueError("template actions must have shape (steps, dimensions)")
        with h5py.File(temporary, "w") as target:
            data = target.create_group("data")
            data.attrs["ASSETS_INSTANCE_PATHS"] = json.dumps(
                target_assets, sort_keys=True
            )
            data.attrs["TARGET_DESCRIPTOR_PROVENANCE"] = json.dumps(
                provenance, sort_keys=True
            )
            data.attrs["TARGET_SCENE_DESCRIPTOR_KIND"] = (
                "official_target_initial_state_with_zero_actions_not_source_demo"
            )
            data.attrs["TARGET_SCENE_DESCRIPTOR_ACTION_SOURCE_SHA256"] = (
                expected_action_source_sha256
            )
            data.attrs["TARGET_SCENE_DESCRIPTOR_STATE_TEMPLATE_SHA256"] = (
                expected_template_sha256
            )
            demo = data.create_group("demo_0")
            demo.attrs["num_samples"] = 1
            demo.attrs["success"] = False
            demo.attrs["descriptor_only"] = True
            demo.create_dataset(
                "actions", data=np.zeros((1, action_shape[1]), dtype=np.float32)
            )
            states = demo.create_group("states")
            _copy_first_state(source_demo["states"], states)
    os.replace(temporary, output)
    return {
        "path": str(output),
        "sha256": _sha256(output),
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-dataset", required=True)
    parser.add_argument("--corrupt-original", required=True)
    parser.add_argument("--canonical-action-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-template-sha256", required=True)
    parser.add_argument("--expected-corrupt-sha256", required=True)
    parser.add_argument("--expected-action-source-sha256", required=True)
    parser.add_argument("--template-mug-asset", required=True)
    parser.add_argument("--template-tree-asset", required=True)
    parser.add_argument("--target-mug-asset", required=True)
    parser.add_argument("--target-tree-asset", required=True)
    args = parser.parse_args()
    receipt = create_target_descriptor(
        template_dataset=args.template_dataset,
        corrupt_original=args.corrupt_original,
        canonical_action_source=args.canonical_action_source,
        output=args.output,
        template_assets={
            "obj_0": args.template_mug_asset,
            "obj_1": args.template_tree_asset,
        },
        target_assets={
            "obj_0": args.target_mug_asset,
            "obj_1": args.target_tree_asset,
        },
        expected_template_sha256=args.expected_template_sha256,
        expected_corrupt_sha256=args.expected_corrupt_sha256,
        expected_action_source_sha256=args.expected_action_source_sha256,
    )
    print("HANGMUG_TARGET_DESCRIPTOR=" + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
