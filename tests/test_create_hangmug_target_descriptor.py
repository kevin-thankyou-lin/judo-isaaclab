from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np

from create_hangmug_target_descriptor import create_target_descriptor


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_descriptor_copies_only_initial_state_and_uses_zero_action(tmp_path):
    template = tmp_path / "rescaled.hdf5"
    corrupt = tmp_path / "corrupt.hdf5"
    output = tmp_path / "descriptor.hdf5"
    action_source = tmp_path / "canonical_source.hdf5"
    corrupt.write_bytes(b"truncated")
    action_source.write_bytes(b"canonical source")
    template_assets = {"obj_0": "mug/rescaled", "obj_1": "tree/rescaled"}
    target_assets = {"obj_0": "mug/base", "obj_1": "tree/base"}
    with h5py.File(template, "w") as handle:
        data = handle.create_group("data")
        data.attrs["ASSETS_INSTANCE_PATHS"] = json.dumps(template_assets)
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.full((3, 14), 7.0))
        states = demo.create_group("states")
        rigid = states.create_group("rigid_object")
        mug = rigid.create_group("obj_0")
        mug.create_dataset("root_pose", data=np.arange(21).reshape(3, 7))
        tree = rigid.create_group("obj_1")
        tree.create_dataset("root_pose", data=np.arange(21, 42).reshape(3, 7))

    receipt = create_target_descriptor(
        template_dataset=template,
        corrupt_original=corrupt,
        canonical_action_source=action_source,
        output=output,
        template_assets=template_assets,
        target_assets=target_assets,
        expected_template_sha256=_sha256(template),
        expected_corrupt_sha256=_sha256(corrupt),
        expected_action_source_sha256=_sha256(action_source),
    )

    assert receipt["sha256"] == _sha256(output)
    with h5py.File(output, "r") as handle:
        demo = handle["data/demo_0"]
        assert json.loads(handle["data"].attrs["ASSETS_INSTANCE_PATHS"]) == target_assets
        assert demo.attrs["descriptor_only"]
        assert not demo.attrs["success"]
        assert handle["data"].attrs[
            "TARGET_SCENE_DESCRIPTOR_ACTION_SOURCE_SHA256"
        ] == _sha256(action_source)
        np.testing.assert_array_equal(demo["actions"][:], np.zeros((1, 14)))
        np.testing.assert_array_equal(
            demo["states/rigid_object/obj_0/root_pose"][:],
            np.arange(7).reshape(1, 7),
        )
