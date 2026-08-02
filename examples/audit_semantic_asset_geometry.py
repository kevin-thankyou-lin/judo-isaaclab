"""Validate measured semantic part frames for every configured campaign pair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from run_three_task_asset_campaign import (
    _atomic_json,
    _expand,
    _sha256,
    enumerate_pairs,
    validate_asset_inventory,
)
from semantic_asset_geometry import (
    asset_root_usd,
    collision_components,
    jsonable,
    mug_parts,
    pot_parts,
    tree_branches,
)


def _bounds(components) -> dict[str, object]:
    points = np.concatenate(tuple(components), axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    return {
        "minimum_local_m": minimum.tolist(),
        "maximum_local_m": maximum.tolist(),
        "center_local_m": center.tolist(),
        "size_m": (maximum - minimum).tolist(),
        "top_support_frame_local": [
            float(center[0]),
            float(center[1]),
            float(maximum[2]),
            1.0,
            0.0,
            0.0,
            0.0,
        ],
    }


def _asset_receipt(path: Path) -> dict[str, object]:
    usd = asset_root_usd(str(path))
    return {"path": str(path.resolve()), "usd": usd, "usd_sha256": _sha256(usd)}


def _pair_geometry(task: dict, pair: dict) -> dict[str, object]:
    objects_root = Path(_expand(task["objects_root"]))
    assets = {name: objects_root / value for name, value in pair["assets"].items()}
    if task["name"] == "putpot":
        return {
            "pot": {
                **_asset_receipt(assets["pot"]),
                "parts": jsonable(pot_parts(str(assets["pot"]))),
            },
            "cooktop": {
                **_asset_receipt(assets["cooktop"]),
                "support_region": _bounds(collision_components(str(assets["cooktop"]))),
            },
        }
    if task["name"] == "putmarker":
        from run_putmarker_skill_program import _drawer_geometry

        drawer = _drawer_geometry(
            str(assets["obj_1"]),
            np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        )
        return {
            "marker": {
                **_asset_receipt(assets["obj_0"]),
                "part_region": _bounds(collision_components(str(assets["obj_0"]))),
            },
            "drawer": {
                **_asset_receipt(assets["obj_1"]),
                "slide_axis_local": drawer.slide_axis_local.tolist(),
                "joint_origin_local_m": drawer.joint_origin_local.tolist(),
                "handle_point_local_m": drawer.handle_point_local.tolist(),
                "cavity_center_local_m": drawer.cavity_center_local.tolist(),
                "cavity_size_m": drawer.cavity_size.tolist(),
                "joint_limits_m": [drawer.lower_limit_m, drawer.upper_limit_m],
            },
        }
    if task["name"] == "hangmug":
        branches = tree_branches(str(assets["mug_tree"]))
        return {
            "mug": {
                **_asset_receipt(assets["mug"]),
                "parts": jsonable(mug_parts(str(assets["mug"]))),
            },
            "tree": {
                **_asset_receipt(assets["mug_tree"]),
                "branches": jsonable(branches),
            },
        }
    raise ValueError(f"unknown task: {task['name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    records = {}
    task_counts = {}
    for task in config["tasks"]:
        pairs = enumerate_pairs(task)
        validate_asset_inventory(task, pairs)
        task_counts[task["name"]] = len(pairs)
        for pair in pairs:
            key = f"{task['name']}:{pair['pair_id']}"
            records[key] = {
                "task": task["name"],
                "pair_id": pair["pair_id"],
                "dataset": pair["dataset"],
                "assets": pair["assets"],
                "geometry": _pair_geometry(task, pair),
            }
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    receipt = {
        "schema_version": 1,
        "code_head": head,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "authored USD collision geometry and prismatic joint metadata",
        "records": records,
        "summary": {
            "pairs": len(records),
            "tasks": task_counts,
            "all_geometry_frames_inferred": len(records) == sum(task_counts.values()),
        },
    }
    _atomic_json(Path(args.output), receipt)
    print("SEMANTIC_GEOMETRY_AUDIT=" + json.dumps(receipt["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
