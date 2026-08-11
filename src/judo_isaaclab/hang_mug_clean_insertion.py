"""Exact collision receipts for clean HangMug insertion paths."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np

from .put_marker import SkillTrajectory, compose_pose, inverse_pose, quaternion_rotate


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


def apply_branch_radial_clearance(
    mug_poses: Any,
    *,
    tree_pose: Any,
    mug_body_frame: Any,
    mug_body_size: Any,
    target_branch: Any,
    collision_steps: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Move one colliding path segment outward without routing it downward.

    This is a deterministic geometry correction, not candidate search.  The
    direction is the cup-body radial vector away from the selected branch axis
    at the collision-window midpoint.  A downward radial vector is projected
    into the world-horizontal plane and given clearance equal to the measured
    mug-body horizontal half-width.  Smooth tapers preserve the incoming path
    and the original final support relationship.
    """

    poses = np.asarray(mug_poses, dtype=np.float64)
    steps = sorted({int(step) for step in collision_steps})
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError("mug_poses must have shape (steps, 7)")
    if not steps or steps[0] < 0 or steps[-1] >= len(poses):
        raise ValueError("collision_steps must select at least one mug pose")
    tree = np.asarray(tree_pose, dtype=np.float64)
    if tree.shape != (7,):
        raise ValueError("tree_pose must have shape (7,)")
    body_size = np.asarray(mug_body_size, dtype=np.float64)
    if body_size.shape != (3,) or np.any(body_size <= 0.0):
        raise ValueError("mug_body_size must have three positive values")

    inner = tree[:3] + quaternion_rotate(tree[3:], target_branch.inner_point)
    tip = tree[:3] + quaternion_rotate(tree[3:], target_branch.tip_point)
    axis = tip - inner
    length = float(np.linalg.norm(axis))
    radius = float(target_branch.radius_m)
    if length <= 0.0 or radius <= 0.0:
        raise ValueError("target branch must have positive length and radius")
    tangent = axis / length
    midpoint = (steps[0] + steps[-1]) // 2
    body_center = compose_pose(poses[midpoint], mug_body_frame)[:3]
    along = float(np.clip(np.dot(body_center - inner, tangent), 0.0, length))
    radial = body_center - (inner + along * tangent)
    radial -= float(np.dot(radial, tangent)) * tangent
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm <= 1.0e-8:
        raise ValueError("cup body lies on branch axis; radial correction is undefined")
    raw_direction = radial / radial_norm
    direction = raw_direction.copy()
    gravity_safe = bool(direction[2] < 0.0)
    maximum_displacement = radius
    if gravity_safe:
        direction[2] = 0.0
        horizontal_norm = float(np.linalg.norm(direction))
        if horizontal_norm <= 1.0e-8:
            raise ValueError(
                "downward radial correction has no horizontal component"
            )
        direction /= horizontal_norm
        maximum_displacement = max(radius, 0.5 * float(np.max(body_size[:2])))

    span = steps[-1] - steps[0] + 1
    taper_steps = max(12, 3 * span)
    start = max(0, steps[0] - taper_steps)
    end = min(len(poses) - 1, steps[-1] + taper_steps)
    weights = np.zeros(len(poses), dtype=np.float64)

    def smooth(value: float) -> float:
        return value * value * (3.0 - 2.0 * value)

    for index in range(start, steps[0] + 1):
        fraction = (index - start) / max(1, steps[0] - start)
        weights[index] = smooth(float(fraction))
    weights[steps[0] : steps[-1] + 1] = 1.0
    for index in range(steps[-1], end + 1):
        fraction = (end - index) / max(1, end - steps[-1])
        weights[index] = smooth(float(fraction))

    corrected = poses.copy()
    corrected[:, :3] += (
        maximum_displacement * weights[:, None] * direction[None, :]
    )
    return corrected, {
        "method": "single_pass_branch_radial_clearance",
        "source_collision_steps": steps,
        "correction_window": [start, end],
        "raw_direction_world": raw_direction.tolist(),
        "direction_world": direction.tolist(),
        "gravity_safe_horizontal_projection": gravity_safe,
        "maximum_displacement_m": maximum_displacement,
        "preserves_final_pose": bool(
            np.allclose(corrected[-1], poses[-1], atol=1.0e-12)
        ),
    }


def _executed_prefix_receipt(
    poses: Any | None,
    *,
    completed_step: int,
    tree_pose: Any,
    target_assets: dict[str, str],
) -> dict[str, Any] | None:
    """Fail closed on observed collisions before editing future commands."""

    if poses is None:
        return None
    executed = np.asarray(poses, dtype=np.float64)
    if executed.ndim != 2 or executed.shape[1] != 7:
        raise ValueError("executed_mug_poses must have shape (steps, 7)")
    if len(executed) != completed_step + 1:
        raise ValueError("executed_mug_poses must end at completed_step")
    receipt = exact_body_collision_receipt(
        executed,
        tree_pose=tree_pose,
        target_assets=target_assets,
    )
    if not receipt["passed"]:
        raise RuntimeError(
            "observation-compensated insertion already collided in execution"
        )
    return receipt


def repair_compensated_insertion_path(
    trajectory: SkillTrajectory,
    *,
    completed_step: int,
    right_contact_in_mug: Any,
    tree_pose: Any,
    mug_body_frame: Any,
    mug_body_size: Any,
    target_branch: Any,
    target_assets: dict[str, str],
    executed_mug_poses: Any | None = None,
) -> tuple[SkillTrajectory, dict[str, Any]]:
    """Exact-screen and repair an observation-compensated insertion suffix."""

    if "branch_insert" not in trajectory.waypoint_steps:
        raise ValueError("branch trajectory is missing branch_insert")
    insert = int(trajectory.waypoint_steps["branch_insert"])
    unload = int(trajectory.waypoint_steps.get("branch_unload", insert))
    completed = int(completed_step)
    if not -1 <= completed < unload:
        raise ValueError("completed_step must precede branch_unload")
    audit_end = insert if completed < insert else unload
    future_start = completed + 1
    contact = np.asarray(right_contact_in_mug, dtype=np.float64)
    if contact.shape != (7,):
        raise ValueError("right_contact_in_mug must have shape (7,)")
    contact_inverse = inverse_pose(contact)
    mug_path = np.asarray([
        compose_pose(pose, contact_inverse)
        for pose in trajectory.right_poses[: audit_end + 1]
    ], dtype=np.float64)
    executed_receipt = _executed_prefix_receipt(
        executed_mug_poses,
        completed_step=completed,
        tree_pose=tree_pose,
        target_assets=target_assets,
    )
    initial = exact_body_collision_receipt(
        mug_path,
        tree_pose=tree_pose,
        target_assets=target_assets,
        start_step=future_start,
        release_step=audit_end + 1,
    )
    receipt = dict(initial)
    receipt["observation_compensated_path"] = True
    receipt["audit_end_step"] = audit_end
    receipt["executed_prefix"] = executed_receipt
    if initial["passed"]:
        receipt["geometry_correction"] = None
        return trajectory, receipt

    collision_steps = [int(step) for step in initial["collision_steps"]]
    corrected_future, correction = apply_branch_radial_clearance(
        mug_path[future_start:],
        tree_pose=tree_pose,
        mug_body_frame=mug_body_frame,
        mug_body_size=mug_body_size,
        target_branch=target_branch,
        collision_steps=[step - future_start for step in collision_steps],
    )
    corrected = mug_path.copy()
    corrected[future_start:] = corrected_future
    correction = dict(correction)
    correction["future_start_step"] = future_start
    correction["source_collision_steps"] = collision_steps
    correction["correction_window"] = [
        future_start + int(step) for step in correction["correction_window"]
    ]
    receipt = exact_body_collision_receipt(
        corrected,
        tree_pose=tree_pose,
        target_assets=target_assets,
        start_step=future_start,
        release_step=audit_end + 1,
    )
    receipt["observation_compensated_path"] = True
    receipt["audit_end_step"] = audit_end
    receipt["executed_prefix"] = executed_receipt
    receipt["pre_correction_collision_count"] = initial["collision_count"]
    receipt["pre_correction_collision_steps"] = initial["collision_steps"]
    receipt["geometry_correction"] = correction
    if not receipt["passed"]:
        raise RuntimeError("observation-compensated insertion remains collision-unsafe")

    right = np.asarray(trajectory.right_poses, dtype=np.float64).copy()
    right[future_start : audit_end + 1] = np.asarray(
        [compose_pose(pose, contact) for pose in corrected[future_start:]],
        dtype=np.float64,
    )
    return SkillTrajectory(
        left_poses=trajectory.left_poses.copy(),
        right_poses=right,
        grippers=trajectory.grippers.copy(),
        stage_names=trajectory.stage_names,
        waypoint_steps=dict(trajectory.waypoint_steps),
    ), receipt
