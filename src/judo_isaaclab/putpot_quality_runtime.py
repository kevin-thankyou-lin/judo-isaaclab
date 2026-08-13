"""Runtime-only evidence helpers for opt-in PutPot quality runs.

The legacy runner never imports this module unless ``--quality-config-json`` is
present.  The helpers deliberately operate on measured simulator observations;
missing raw contact or robot-body geometry raises instead of producing a proxy
that could pass the independent audit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_and_validate_perturbation_case(
    path: str | Path, expected_cases: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Load exactly one generated case and reject relabelled nominal runs."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "case_sha256" not in value:
        raise ValueError("quality perturbation case must be a hashed JSON object")
    unhashed = {key: item for key, item in value.items() if key != "case_sha256"}
    if _canonical_sha256(unhashed) != value["case_sha256"]:
        raise ValueError("quality perturbation case hash does not match its payload")
    by_hash = {str(case["case_sha256"]): dict(case) for case in expected_cases}
    expected = by_hash.get(str(value["case_sha256"]))
    if expected is None or expected != value:
        raise ValueError("quality perturbation case is not in the configured fixed bank")
    translation = np.asarray(value["grasp_translation_m"], dtype=np.float64)
    rotation = np.asarray(value["eef_rotation_vector_rad"], dtype=np.float64)
    joints = np.asarray(value["joint_action_rad"], dtype=np.float64)
    if translation.shape != (3,) or rotation.shape != (3,) or joints.ndim != 1:
        raise ValueError("quality perturbation vectors have invalid shapes")
    if not np.all(np.isfinite(np.concatenate((translation, rotation, joints)) )):
        raise ValueError("quality perturbation vectors must be finite")
    return expected


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def perturb_grasp_pose(pose: Any, case: Mapping[str, Any]) -> np.ndarray:
    """Apply the fixed Cartesian grasp perturbation to one wxyz pose."""

    result = np.asarray(pose, dtype=np.float64).copy()
    if result.shape != (7,):
        raise ValueError("grasp pose must have shape (7,)")
    result[:3] += np.asarray(case["grasp_translation_m"], dtype=np.float64)
    vector = np.asarray(case["eef_rotation_vector_rad"], dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    delta = (
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if angle == 0.0
        else np.r_[np.cos(0.5 * angle), np.sin(0.5 * angle) * vector / angle]
    )
    result[3:] = _quat_multiply(delta, result[3:])
    result[3:] /= np.linalg.norm(result[3:])
    return result


def perturb_joint_action(action: Any, case: Mapping[str, Any]) -> np.ndarray:
    """Apply the fixed arm-joint perturbation; gripper commands stay binary."""

    result = np.asarray(action, dtype=np.float64).copy()
    delta = np.asarray(case["joint_action_rad"], dtype=np.float64)
    if result.shape != delta.shape:
        raise ValueError("joint perturbation does not match runner action width")
    if len(delta) == 14 and (delta[6] != 0.0 or delta[13] != 0.0):
        raise ValueError("quality perturbations must not alter gripper commands")
    return result + delta


def _convex_hull_area(points: np.ndarray) -> float:
    """Area of a small 2-D contact-point footprint without SciPy."""

    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) < 3:
        return 0.0

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])) - (
            (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return 0.5 * abs(
        sum(
            hull[index][0] * hull[(index + 1) % len(hull)][1]
            - hull[(index + 1) % len(hull)][0] * hull[index][1]
            for index in range(len(hull))
        )
    )


def measured_pad_contact_quality(finger, target: str, env_id: int = 0) -> dict[str, Any]:
    """Measure raw PhysX contact footprint and pad-normal alignment.

    The area is the convex hull of *all* raw contact points projected onto the
    authored rectangular pad, divided by that pad's authored area.  A single
    aggregate contact point therefore cannot pass as broad contact.
    """

    import torch
    from isaaclab.utils.math import quat_apply_inverse

    sensor = getattr(finger, "_sensor", None)
    filter_index = int(finger._filter_index(target))
    view = getattr(sensor, "contact_physx_view", None)
    if sensor is None or view is None or filter_index < 0:
        raise RuntimeError("raw filtered PhysX contact data is unavailable")
    _, points, normals, _, counts, starts = view.get_contact_data(
        dt=float(sensor._sim_physics_dt)
    )
    body_count = int(sensor.num_bodies)
    filter_count = int(view.filter_count)
    count_grid = counts.view(-1, body_count, filter_count)
    start_grid = starts.view(-1, body_count, filter_count)
    count = int(count_grid[env_id, 0, filter_index].item())
    start = int(start_grid[env_id, 0, filter_index].item())
    if count < 1:
        return {
            "contact_point_count": 0,
            "contact_area_fraction": 0.0,
            "flush_angle_deg": 180.0,
        }
    points_world = points[start : start + count]
    normals_world = normals[start : start + count]
    arm = finger.scene[finger.arm_name]
    body_index = arm.data.body_names.index(finger.link)
    pose = arm.data.body_link_pose_w[env_id, body_index]
    pose_batch = pose[None].expand(count, -1)
    local = quat_apply_inverse(
        pose_batch[:, 3:], points_world - pose_batch[:, :3]
    )
    local_normals = quat_apply_inverse(pose_batch[:, 3:], normals_world)
    keypoints = finger._keypoints.to(device=local.device, dtype=local.dtype)
    tip = keypoints[0]
    base_midpoint = 0.5 * (keypoints[1] + keypoints[2])
    length_vector = base_midpoint - tip
    width_vector = keypoints[2] - keypoints[1]
    length = torch.linalg.norm(length_vector)
    width = torch.linalg.norm(width_vector)
    if float(length.item()) <= 0.0 or float(width.item()) <= 0.0:
        raise RuntimeError("authored finger pad keypoints are degenerate")
    length_axis = length_vector / length
    width_axis = width_vector / width
    normal_axis = torch.linalg.cross(length_axis, width_axis)
    normal_axis = normal_axis / torch.linalg.norm(normal_axis)
    projected = torch.stack(
        (
            torch.sum((local - tip) * length_axis, dim=1),
            torch.sum((local - tip) * width_axis, dim=1),
        ),
        dim=1,
    ).detach().cpu().numpy()
    footprint = _convex_hull_area(projected)
    area_fraction = float(np.clip(footprint / float((length * width).item()), 0.0, 1.0))
    normal_alignment = torch.abs(local_normals @ normal_axis)
    flush = torch.rad2deg(torch.acos(torch.clamp(normal_alignment, 0.0, 1.0)))
    return {
        "contact_point_count": count,
        "contact_area_fraction": area_fraction,
        "flush_angle_deg": float(torch.max(flush).item()),
    }


def quality_stage_telemetry(
    *,
    samples: Sequence[Mapping[str, Any]],
    actions: Any,
    left_start_m: Any,
    right_start_m: Any,
    stable_steps: int,
    minimum_force_n: float,
    minimum_area_fraction: float,
    maximum_flush_angle_deg: float,
    return_tolerance_m: float,
) -> dict[str, Any]:
    """Build measured contact/motion/release fields for the sidecar."""

    rows = list(samples)[1:]
    action_array = np.asarray(actions, dtype=np.float64)
    count = len(action_array)
    if len(rows) != count or action_array.shape != (count, 14):
        raise ValueError("quality stage telemetry must align to 14-DoF actions")
    left_area = np.asarray([row["left_contact_area_fractions"] for row in rows])
    right_area = np.asarray([row["right_contact_area_fractions"] for row in rows])
    left_flush = np.asarray([row["left_flush_angles_deg"] for row in rows])
    right_flush = np.asarray([row["right_flush_angles_deg"] for row in rows])
    left_forces = np.asarray([row["left_finger_forces_n"] for row in rows])
    right_forces = np.asarray([row["right_finger_forces_n"] for row in rows])
    finite = all(
        value.shape == (count, 2) and np.all(np.isfinite(value))
        for value in (left_area, right_area, left_flush, right_flush)
    )
    if not finite:
        raise ValueError("measured per-finger quality evidence is missing or non-finite")

    def good(forces, areas, angles):
        return (
            np.all(forces >= minimum_force_n, axis=1)
            & np.all(areas >= minimum_area_fraction, axis=1)
            & np.all(np.abs(angles) <= maximum_flush_angle_deg, axis=1)
        )

    left_good = good(left_forces, left_area, left_flush)
    right_good = good(right_forces, right_area, right_flush)

    def first_streak(mask):
        streak = 0
        for index, active in enumerate(mask):
            streak = streak + 1 if bool(active) else 0
            if streak >= stable_steps:
                return index
        return None

    left_stable = first_streak(left_good)
    right_stable = first_streak(right_good & (np.arange(count) > (-1 if left_stable is None else left_stable)))
    latch = first_streak(left_good & right_good)
    support = np.asarray([row["support_geometry_now"] for row in rows], dtype=bool)
    left_open = action_array[:, 6] <= -0.04749
    right_open = action_array[:, 13] <= -0.04749
    stages: list[str] = []
    if left_stable is not None:
        stages.append("left_handle_stable")
    if right_stable is not None:
        stages.append("right_handle_stable")
    if latch is not None:
        stages.append("four_pad_latch")
    lift = next((i for i, row in enumerate(rows) if latch is not None and i >= latch and row["stage1"]), None)
    if lift is not None:
        stages.append("bimanual_lift")
    lower = next((i for i in range((lift or 0), count) if support[i] and rows[i]["left_grasp"] and rows[i]["right_grasp"]), None)
    if lower is not None:
        stages.extend(("coordinated_transfer", "supported_lower"))
    bilateral = left_open & right_open
    open_events = np.flatnonzero(bilateral & ~np.r_[False, bilateral[:-1]])
    opening = int(open_events[0]) if len(open_events) == 1 else None
    if opening is not None and support[opening]:
        stages.append("open_both")
    left_final = np.asarray(rows[-1]["left_eef_pose"], dtype=np.float64)[:3]
    right_final = np.asarray(rows[-1]["right_eef_pose"], dtype=np.float64)[:3]
    if (
        opening is not None
        and np.all(bilateral[opening:])
        and np.linalg.norm(left_final - np.asarray(left_start_m)) <= return_tolerance_m
        and np.linalg.norm(right_final - np.asarray(right_start_m)) <= return_tolerance_m
    ):
        stages.append("return_both_open_to_start")
    return {
        "left_contact_area_fractions": left_area,
        "right_contact_area_fractions": right_area,
        "left_flush_angles_deg": left_flush,
        "right_flush_angles_deg": right_flush,
        "supported": support,
        "left_open": left_open,
        "right_open": right_open,
        "stage_events": np.asarray(stages),
        "object_first_start_step": np.asarray(-1 if latch is None else latch),
        "object_first_end_step": np.asarray(-1 if lower is None else lower),
        "left_start_m": np.asarray(left_start_m, dtype=np.float64),
        "right_start_m": np.asarray(right_start_m, dtype=np.float64),
    }


def write_quality_sidecar(path: str | Path, fields: Mapping[str, Any]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite quality sidecar: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **fields)


def authored_robot_collision_model(env) -> dict[str, Any]:
    """Build conservative spheres from every authored robot collision body.

    Radii come from USD collision bounds in each rigid body's local frame.  No
    guessed per-link dimensions are accepted.  The model includes every body
    on both arms, including both fingers and both wrist-camera rigid bodies.
    """

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    model: dict[str, Any] = {"radii": {}, "groups": {}, "structural": []}
    for arm_name, side in (("left_arm", "left"), ("right_arm", "right")):
        arm = env.scene[arm_name]
        usd_path = Path(arm.cfg.spawn.usd_path).resolve()
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise RuntimeError(f"could not open robot USD for quality audit: {usd_path}")
        meters = float(UsdGeom.GetStageMetersPerUnit(stage))
        rigid = {
            prim.GetName(): prim
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        }
        labels: dict[str, str] = {}
        for body_name in arm.data.body_names:
            body = rigid.get(body_name)
            if body is None:
                raise RuntimeError(f"robot USD lacks runtime body {body_name!r}")
            collision_prims = [
                prim for prim in Usd.PrimRange(body)
                if prim.HasAPI(UsdPhysics.CollisionAPI)
            ]
            if not collision_prims:
                # Articulation bookkeeping bodies with no collision geometry are
                # intentionally absent from the swept collision model.
                continue
            body_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            world_body = body_world.GetInverse()
            radius = 0.0
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [
                    UsdGeom.Tokens.default_,
                    UsdGeom.Tokens.render,
                    UsdGeom.Tokens.proxy,
                ],
                True,
                False,
            )
            for collision in collision_prims:
                bounds = cache.ComputeWorldBound(collision).ComputeAlignedBox()
                minimum, maximum = bounds.GetMin(), bounds.GetMax()
                for x in (minimum[0], maximum[0]):
                    for y in (minimum[1], maximum[1]):
                        for z in (minimum[2], maximum[2]):
                            point = world_body.Transform(Gf.Vec3d(x, y, z))
                            radius = max(
                                radius,
                                float(np.linalg.norm(np.asarray(point))) * meters,
                            )
            if not np.isfinite(radius) or radius <= 0.0:
                raise RuntimeError(f"invalid authored collision radius for {body_name}")
            label = f"{arm_name}__{body_name}"
            labels[str(body.GetPath())] = label
            model["radii"][label] = radius
            lowered = body_name.lower()
            if "camera" in lowered:
                group = f"{side}_wrist_camera"
            elif "finger" in lowered or body_name == "link_6":
                group = f"{side}_gripper"
            else:
                group = f"{side}_arm"
            model["groups"].setdefault(group, []).append(label)
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            targets0 = joint.GetBody0Rel().GetTargets()
            targets1 = joint.GetBody1Rel().GetTargets()
            if len(targets0) == 1 and len(targets1) == 1:
                first = labels.get(str(targets0[0]))
                second = labels.get(str(targets1[0]))
                if first is not None and second is not None:
                    model["structural"].append(sorted((first, second)))
    expected = {
        "left_arm", "right_arm", "left_gripper", "right_gripper",
        "left_wrist_camera", "right_wrist_camera",
    }
    if set(model["groups"]) != expected or any(
        not model["groups"][name] for name in expected
    ):
        raise RuntimeError("robot collision model lacks an arm/gripper/camera group")
    model["structural"] = sorted({tuple(item) for item in model["structural"]})
    return model


def capture_robot_collision_centers(env, model: Mapping[str, Any]) -> dict[str, list[float]]:
    """Capture actual world centers for every collision-bearing rigid body."""

    centers: dict[str, list[float]] = {}
    for arm_name in ("left_arm", "right_arm"):
        arm = env.scene[arm_name]
        for body_index, body_name in enumerate(arm.data.body_names):
            label = f"{arm_name}__{body_name}"
            if label in model["radii"]:
                centers[label] = (
                    arm.data.body_link_pose_w[0, body_index, :3]
                    .detach().cpu().numpy().astype(np.float64).tolist()
                )
    if set(centers) != set(model["radii"]):
        raise RuntimeError("live collision body centers do not match authored model")
    return centers


def collision_sidecar_fields(
    samples: Sequence[Mapping[str, Any]], model: Mapping[str, Any]
) -> dict[str, Any]:
    """Serialize full-trace collision centers and exact authored sphere pins."""

    rows = list(samples)[1:]
    fields: dict[str, Any] = {}
    for label, radius in model["radii"].items():
        path = np.asarray(
            [row["quality_collision_centers_m"][label] for row in rows],
            dtype=np.float64,
        )
        if path.shape != (len(rows), 3) or not np.all(np.isfinite(path)):
            raise ValueError(f"collision path is missing for {label}")
        fields[f"center__{label}"] = path
        fields[f"radius__{label}"] = np.asarray(float(radius))
    fields["component_groups_json"] = np.asarray(
        json.dumps(model["groups"], sort_keys=True)
    )
    fields["structural_adjacencies_json"] = np.asarray(
        json.dumps(model["structural"], sort_keys=True)
    )
    fields["geometry_contract"] = np.asarray(
        "measured_body_origins_with_authored_collision_bound_spheres"
    )
    return fields
