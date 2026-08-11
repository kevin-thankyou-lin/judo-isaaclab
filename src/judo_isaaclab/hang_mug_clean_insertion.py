"""Exact collision receipts for clean HangMug insertion paths."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np


def _asset_root_usd(asset_path: str) -> str:
    root = Path(asset_path)
    usd = root / f"{root.name}.usd"
    if not usd.is_file():
        raise FileNotFoundError(usd)
    return str(usd)


def _indexed_collision_components(asset_path: str) -> dict[int, np.ndarray]:
    """Read authored collision points keyed by their USD component suffix."""

    from pxr import Gf, Usd, UsdGeom

    usd = _asset_root_usd(asset_path)
    stage = Usd.Stage.Open(usd)
    if not stage:
        raise ValueError(f"could not open USD stage: {usd}")
    transforms = UsdGeom.XformCache()
    grouped: dict[int, list[np.ndarray]] = {}
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or "/collisions/" not in prim_path:
            continue
        match = re.search(r"obj_link_collision_(\d+)(?:/|$)", prim_path)
        if match is None:
            raise ValueError(f"collision mesh lacks numeric component id: {prim_path}")
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points:
            continue
        transform = transforms.GetLocalToWorldTransform(prim)
        vertices = np.asarray(
            [transform.Transform(Gf.Vec3d(point)) for point in points],
            dtype=np.float64,
        )
        grouped.setdefault(int(match.group(1)), []).append(vertices)
    if not grouped:
        raise ValueError(f"no indexed collision meshes found in {usd}")
    return {
        index: np.concatenate(parts, axis=0)
        for index, parts in sorted(grouped.items())
    }


def _mug_body_collision_indices(asset_path: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Infer body and handle component IDs from authored mug geometry."""

    from .semantic_parts import infer_mug_handle_component_indices

    indexed = _indexed_collision_components(asset_path)
    component_ids = tuple(indexed)
    handle_positions = infer_mug_handle_component_indices(indexed.values())
    handle_ids = tuple(component_ids[position] for position in handle_positions)
    body_ids = tuple(index for index in component_ids if index not in handle_ids)
    if not body_ids:
        raise ValueError("mug body collision components could not be isolated")
    return body_ids, handle_ids


def exact_body_collision_receipt(
    mug_poses: Any,
    *,
    tree_pose: Any,
    target_assets: dict[str, str],
    start_step: int = 0,
    release_step: int | None = None,
) -> dict[str, Any]:
    """Require zero cup-body/tree intersections while allowing handle contact."""

    from .collision_screening import (
        load_usd_collision_mesh,
        object_path_collision_reports,
    )

    poses = np.asarray(mug_poses, dtype=np.float64)
    release = len(poses) if release_step is None else int(release_step)
    if not 0 <= start_step < release <= len(poses):
        raise ValueError("clean insertion window must lie within the mug path")
    body_indices, handle_indices = _mug_body_collision_indices(
        target_assets["mug"]
    )
    body = load_usd_collision_mesh(
        _asset_root_usd(target_assets["mug"]), body_indices
    )
    tree = load_usd_collision_mesh(_asset_root_usd(target_assets["mug_tree"]))
    report = object_path_collision_reports(
        poses[start_step:release],
        tree_pose=np.asarray(tree_pose, dtype=np.float64),
        object_mesh=body,
        tree_mesh=tree,
        sample_stride=1,
    )[0]
    collisions = [start_step + step for step in report["collision_steps"]]
    return {
        "method": report["method"],
        "semantic_contract": (
            "exact cup-body/tree intersection forbidden before release; "
            "handle/tree contact allowed"
        ),
        "mug_body_collision_indices": list(body_indices),
        "allowed_mug_handle_collision_indices": list(handle_indices),
        "start_step": int(start_step),
        "release_step": release,
        "sampled_steps": release - start_step,
        "collision_steps": collisions,
        "collision_count": len(collisions),
        "passed": not collisions,
    }
