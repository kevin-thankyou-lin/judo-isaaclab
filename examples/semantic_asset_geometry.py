"""Read authored USD collision meshes and infer deterministic semantic parts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


def asset_root_usd(asset_path: str) -> str:
    root = Path(asset_path)
    candidate = root / f"{root.name}.usd"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return str(candidate)


def collision_components(asset_path: str) -> tuple[np.ndarray, ...]:
    """Return collision-mesh vertices in the asset root frame.

    Importing Pixar USD stays inside this runner-side module so the core
    geometry inference and its unit tests do not require an Isaac installation.
    """

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(asset_root_usd(asset_path))
    if not stage:
        raise ValueError(f"could not open USD stage: {asset_path}")
    result = []
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Mesh) or "/collisions/" not in str(prim.GetPath()):
            continue
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
        transform = np.asarray(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            ),
            dtype=np.float64,
        )
        homogeneous = np.concatenate(
            (points, np.ones((len(points), 1), dtype=np.float64)), axis=1
        )
        result.append((homogeneous @ transform)[:, :3])
    if not result:
        raise ValueError(f"asset contains no authored collision meshes: {asset_path}")
    return tuple(result)


def pot_parts(asset_path: str):
    from judo_isaaclab.semantic_parts import infer_pot_parts

    return infer_pot_parts(collision_components(asset_path))


def mug_parts(asset_path: str):
    from judo_isaaclab.semantic_parts import infer_mug_parts

    return infer_mug_parts(collision_components(asset_path))


def tree_branches(asset_path: str):
    from judo_isaaclab.semantic_parts import infer_tree_branches

    return infer_tree_branches(collision_components(asset_path))


def jsonable(value: Any) -> Any:
    """Convert inferred dataclasses and arrays into result-JSON values."""

    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value
