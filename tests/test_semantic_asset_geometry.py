import importlib

import numpy as np
import pytest

from semantic_asset_geometry import asset_root_usd, jsonable


def test_asset_root_usd_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        asset_root_usd(str(tmp_path / "missing"))


def test_jsonable_preserves_numeric_part_evidence():
    semantic_parts = importlib.import_module("judo_isaaclab.semantic_parts")
    value = semantic_parts.PotParts(
        negative_handle_frame=np.asarray([-1, 0, 0, 1, 0, 0, 0], dtype=float),
        positive_handle_frame=np.asarray([1, 0, 0, 1, 0, 0, 0], dtype=float),
        negative_handle_size=np.asarray([0.1, 0.02, 0.03]),
        positive_handle_size=np.asarray([0.1, 0.02, 0.03]),
        handle_axis=0,
        bottom_z=-0.05,
        body_xy_min=np.asarray([-0.1, -0.1]),
        body_xy_max=np.asarray([0.1, 0.1]),
    )

    result = jsonable(value)

    assert result["handle_axis"] == 0
    assert result["positive_handle_frame"][:3] == [1.0, 0.0, 0.0]
    assert result["body_xy_min"] == [-0.1, -0.1]
