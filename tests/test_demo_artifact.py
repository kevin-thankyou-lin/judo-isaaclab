import json

import h5py
import numpy as np
import pytest

from judo_isaaclab.demo_artifact import DemonstrationRecorder, relative_asset_paths


def _state(offset=0.0):
    return {
        "articulation": {
            "arm": {
                "joint_position": np.asarray([[offset, offset + 1]], dtype=np.float32),
                "joint_velocity": np.asarray([[0.0, 0.0]], dtype=np.float32),
            }
        },
        "rigid_object": {
            "object": {"root_pose": np.asarray([[offset] * 7], dtype=np.float32)}
        },
    }


def test_writes_replayable_success_demo(tmp_path):
    recorder = DemonstrationRecorder()
    recorder.start(_state())
    recorder.append(
        np.asarray([[0.1, 0.2]], dtype=np.float32),
        _state(1.0),
        observation={"policy": np.asarray([[3.0, 4.0]], dtype=np.float32)},
        semantic_observation={"stage1": True, "rgb": np.zeros((1, 32, 32, 3))},
    )
    path = tmp_path / "demo.hdf5"
    recorder.write(
        path,
        assets_instance_paths={"object": "Objects/object_007"},
        success=True,
        metadata={"controller": "deterministic_skill", "candidate_sampling": False},
    )

    with h5py.File(path, "r") as handle:
        demo = handle["data/demo_0"]
        assert demo["actions"].shape == (1, 2)
        assert demo["states/articulation/arm/joint_position"].shape == (2, 2)
        assert demo["initial_state/articulation/arm/joint_position"].shape == (1, 2)
        assert demo["obs/policy"].shape == (1, 2)
        assert demo["obs/semantic/stage1"].shape == (1,)
        assert "obs/semantic/rgb" not in demo
        assert demo.attrs["success"]
        assert json.loads(handle["data"].attrs["ASSETS_INSTANCE_PATHS"])["object"] == "Objects/object_007"


def test_refuses_unsuccessful_demo(tmp_path):
    recorder = DemonstrationRecorder()
    recorder.start(_state())
    recorder.append(np.zeros((1, 2)), _state())
    with pytest.raises(RuntimeError, match="unsuccessful"):
        recorder.write(
            tmp_path / "bad.hdf5",
            assets_instance_paths={},
            success=False,
            metadata={},
        )


def test_relative_asset_paths(tmp_path):
    object_dir = tmp_path / "Mug" / "mug_003"
    object_dir.mkdir(parents=True)
    assert relative_asset_paths({"mug": str(object_dir)}, tmp_path) == {
        "mug": "Mug/mug_003"
    }
