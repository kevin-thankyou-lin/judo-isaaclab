import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "examples/audit_semantic_asset_geometry.py"
    spec = importlib.util.spec_from_file_location("audit_semantic_asset_geometry", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bounds_records_object_local_top_support_plane():
    value = _module()._bounds(
        (
            np.asarray(
                [
                    [-0.2, -0.1, -0.05],
                    [0.2, 0.1, 0.05],
                    [-0.2, 0.1, 0.05],
                    [0.2, -0.1, -0.05],
                ]
            ),
        )
    )

    assert value["center_local_m"] == [0.0, 0.0, 0.0]
    assert value["size_m"] == [0.4, 0.2, 0.1]
    assert value["top_support_frame_local"] == [
        0.0,
        0.0,
        0.05,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
