"""Deterministic part frames inferred from authored collision geometry.

The campaign assets do not contain semantic prim names for pot handles, mug
handles, or mug-tree branches.  They do, however, contain convex collision
components.  This module turns those measured components into stable local
part frames without asset identifiers, sampling, or simulator rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from judo_isaaclab.put_marker import pose_from_matrix


LOCAL_HANDLE_TANGENT_NEIGHBORS = 32


def _points(value: object) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 4:
        raise ValueError("collision component points must have shape (N, 3), N >= 4")
    if not np.all(np.isfinite(result)):
        raise ValueError("collision component points must be finite")
    return result


def _components(values: Iterable[object]) -> tuple[np.ndarray, ...]:
    result = tuple(_points(value) for value in values)
    if not result:
        raise ValueError("at least one collision component is required")
    return result


def _bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def _union_bounds(values: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate(tuple(values), axis=0)
    return _bounds(points)


def _frame(origin: object, x_axis: object, z_hint: object = (0.0, 0.0, 1.0)) -> np.ndarray:
    origin = np.asarray(origin, dtype=np.float64)
    x_axis = np.asarray(x_axis, dtype=np.float64)
    z_hint = np.asarray(z_hint, dtype=np.float64)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_hint, x_axis)
    if np.linalg.norm(y_axis) < 1.0e-8:
        raise ValueError("part-frame x axis is parallel to its z hint")
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    matrix[:3, 3] = origin
    return pose_from_matrix(matrix)


def _projected_size(points: np.ndarray, frame: np.ndarray) -> np.ndarray:
    from judo_isaaclab.put_marker import inverse_pose, quaternion_rotate

    inverse = inverse_pose(frame)
    local = np.stack(
        [quaternion_rotate(inverse[3:], point - frame[:3]) for point in points]
    )
    return local.max(axis=0) - local.min(axis=0)


@dataclass(frozen=True)
class PotParts:
    """Pot handle frames plus the measured bottom support plane."""

    negative_handle_frame: np.ndarray
    positive_handle_frame: np.ndarray
    negative_handle_size: np.ndarray
    positive_handle_size: np.ndarray
    handle_axis: int
    bottom_z: float
    body_xy_min: np.ndarray
    body_xy_max: np.ndarray

    def __post_init__(self) -> None:
        for name in ("negative_handle_frame", "positive_handle_frame"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (7,):
                raise ValueError(f"{name} must have shape (7,)")
            object.__setattr__(self, name, value)
        for name in (
            "negative_handle_size",
            "positive_handle_size",
            "body_xy_min",
            "body_xy_max",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            expected = (3,) if "handle_size" in name else (2,)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class MugParts:
    """Mug body and handle-hole frames in the asset root frame."""

    body_frame: np.ndarray
    body_size: np.ndarray
    handle_hole_frame: np.ndarray
    handle_outer_size: np.ndarray
    handle_thickness_m: float
    handle_axis: int
    handle_sign: int

    def __post_init__(self) -> None:
        for name in ("body_frame", "handle_hole_frame"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (7,):
                raise ValueError(f"{name} must have shape (7,)")
            object.__setattr__(self, name, value)
        for name in ("body_size", "handle_outer_size"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or np.any(value <= 0.0):
                raise ValueError(f"{name} must have three positive values")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class BranchPart:
    """One mug-tree branch, directed from the trunk toward its tip."""

    frame: np.ndarray
    inner_point: np.ndarray
    tip_point: np.ndarray
    tangent: np.ndarray
    length_m: float
    radius_m: float
    normalized_height: float
    azimuth_rad: float

    def __post_init__(self) -> None:
        for name, shape in (
            ("frame", (7,)),
            ("inner_point", (3,)),
            ("tip_point", (3,)),
            ("tangent", (3,)),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            object.__setattr__(self, name, value)


def _footprint(components: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    all_points = np.concatenate(components)
    global_min, global_max = _bounds(all_points)
    height = global_max[2] - global_min[2]
    floor = []
    for points in components:
        minimum, maximum = _bounds(points)
        if (
            minimum[2] <= global_min[2] + 0.08 * height
            and maximum[2] - minimum[2] <= 0.28 * height
        ):
            floor.append(points)
    if not floor:
        raise ValueError("could not infer the object's bottom footprint")
    return _union_bounds(floor)


def infer_pot_parts(values: Iterable[object]) -> PotParts:
    """Infer the two pot handles from overhang beyond the bottom footprint."""

    components = _components(values)
    all_points = np.concatenate(components)
    global_min, global_max = _bounds(all_points)
    body_min, body_max = _footprint(components)
    overhang = np.asarray(
        [
            body_min[0] - global_min[0],
            global_max[0] - body_max[0],
            body_min[1] - global_min[1],
            global_max[1] - body_max[1],
        ]
    )
    best = int(np.argmax(overhang))
    axis = best // 2
    if overhang[best] < 0.008:
        raise ValueError("pot collision geometry has no measurable handle overhang")
    side_sets: dict[int, list[np.ndarray]] = {-1: [], 1: []}
    for points in components:
        minimum, maximum = _bounds(points)
        for sign in (-1, 1):
            reach = (
                body_min[axis] - minimum[axis]
                if sign < 0
                else maximum[axis] - body_max[axis]
            )
            side_overhang = (
                body_min[axis] - global_min[axis]
                if sign < 0
                else global_max[axis] - body_max[axis]
            )
            if side_overhang >= 0.008 and reach > 0.08 * side_overhang:
                side_sets[sign].append(points)
    if not all(side_sets.values()):
        raise ValueError("pot collision geometry does not contain two handle sides")

    frames: dict[int, np.ndarray] = {}
    sizes: dict[int, np.ndarray] = {}
    for sign, selected in side_sets.items():
        points = np.concatenate(selected)
        minimum, maximum = _bounds(points)
        origin = 0.5 * (minimum + maximum)
        outward = np.zeros(3, dtype=np.float64)
        outward[axis] = sign
        frames[sign] = _frame(origin, outward)
        sizes[sign] = _projected_size(points, frames[sign])
    return PotParts(
        negative_handle_frame=frames[-1],
        positive_handle_frame=frames[1],
        negative_handle_size=sizes[-1],
        positive_handle_size=sizes[1],
        handle_axis=axis,
        bottom_z=float(global_min[2]),
        body_xy_min=body_min[:2],
        body_xy_max=body_max[:2],
    )


def infer_pot_handle_contact_frame(
    values: Iterable[object],
    parts: PotParts,
    side: int,
    reference_point: object,
) -> np.ndarray:
    """Infer the nearest authored handle-segment frame and principal tangent."""

    components = _components(values)
    if side not in (-1, 1):
        raise ValueError("handle side must be -1 or 1")
    reference = np.asarray(reference_point, dtype=np.float64)
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_point must contain three finite values")
    axis = parts.handle_axis
    boundary = parts.body_xy_min[axis] if side < 0 else parts.body_xy_max[axis]
    candidates = []
    for points in components:
        reaches_handle = (
            np.min(points[:, axis]) < boundary - 1.0e-4
            if side < 0
            else np.max(points[:, axis]) > boundary + 1.0e-4
        )
        if reaches_handle:
            candidates.append(points)
    if not candidates:
        raise ValueError("no collision component reaches the selected pot handle")
    component = min(
        candidates,
        key=lambda points: float(np.min(np.linalg.norm(points - reference, axis=1))),
    )
    nearest = component[np.argmin(np.linalg.norm(component - reference, axis=1))]
    # Curved handles are commonly authored as one convex component.  A PCA of
    # every vertex describes the whole arc rather than the tangent under the
    # transferred wrist.  Use the nearest authored surface neighborhood so the
    # frame remains local to the actual contact region.
    distances = np.linalg.norm(component - reference, axis=1)
    count = min(LOCAL_HANDLE_TANGENT_NEIGHBORS, len(component))
    neighborhood = component[np.argsort(distances)[:count]]
    covariance = np.cov(
        neighborhood - neighborhood.mean(axis=0), rowvar=False
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
    # PCA eigenvectors have arbitrary sign.  Canonicalize from the dominant
    # in-plane handle direction instead of allowing a numerically tiny
    # vertical component to choose a 180-degree wrist rotation.  Pot019's
    # negative handle measured (transverse, vertical)=(-0.5783, +0.0045): the
    # old vertical-first rule inverted its local correspondence even though
    # the authored segment was overwhelmingly transverse.
    transverse_axis = (axis + 1) % 2
    canonical_axis = (
        transverse_axis
        if abs(tangent[transverse_axis]) >= abs(tangent[2])
        else 2
    )
    if tangent[canonical_axis] < 0.0:
        tangent = -tangent
    outward = np.zeros(3, dtype=np.float64)
    outward[axis] = side
    return _frame(nearest, tangent, outward)


def bimanual_handle_sides(
    pot_root_pose: object,
    parts: PotParts,
    left_eef_position: object,
    right_eef_position: object,
) -> tuple[int, int]:
    """Assign distinct authored handle sides to the two observed wrists."""

    from judo_isaaclab.put_marker import compose_pose

    handles = {
        -1: compose_pose(pot_root_pose, parts.negative_handle_frame)[:3],
        1: compose_pose(pot_root_pose, parts.positive_handle_frame)[:3],
    }
    left = np.asarray(left_eef_position, dtype=np.float64)[:3]
    right = np.asarray(right_eef_position, dtype=np.float64)[:3]
    direct = np.linalg.norm(left - handles[-1]) + np.linalg.norm(right - handles[1])
    crossed = np.linalg.norm(left - handles[1]) + np.linalg.norm(right - handles[-1])
    return (-1, 1) if direct <= crossed else (1, -1)


def infer_mug_parts(values: Iterable[object]) -> MugParts:
    """Infer a mug body frame and the center of its handle opening."""

    components = _components(values)
    all_points = np.concatenate(components)
    global_min, global_max = _bounds(all_points)
    body_min, body_max = _footprint(components)
    candidates = []
    for axis in (0, 1):
        candidates.extend(
            [
                (body_min[axis] - global_min[axis], axis, -1),
                (global_max[axis] - body_max[axis], axis, 1),
            ]
        )
    overhang, axis, sign = max(candidates)
    if overhang < 0.006:
        raise ValueError("mug collision geometry has no measurable handle overhang")
    transverse = 1 - axis
    selected = []
    for points in components:
        minimum, maximum = _bounds(points)
        reach = (
            body_min[axis] - minimum[axis]
            if sign < 0
            else maximum[axis] - body_max[axis]
        )
        transverse_span = maximum[transverse] - minimum[transverse]
        body_span = body_max[transverse] - body_min[transverse]
        if reach > 0.05 * overhang and transverse_span < 0.55 * body_span:
            selected.append(points)
    if not selected:
        raise ValueError("mug handle collision components could not be isolated")
    handle_points = np.concatenate(selected)
    handle_min, handle_max = _bounds(handle_points)
    outward = np.zeros(3, dtype=np.float64)
    outward[axis] = sign
    hole_center = 0.5 * (handle_min + handle_max)
    # The center of the handle material bounds is not generally the center of
    # its opening.  Convex-decomposed handles contain an outer rail plus upper
    # and lower connectors; averaging all their vertices can put the inferred
    # point inside the outer rail (notably on shallow HangMug handles).  Infer
    # the authored empty cavity from the body boundary, the inner face of the
    # outer rail, and the facing surfaces of the two connectors.
    selected_bounds = tuple(_bounds(points) for points in selected)
    selected_centers = tuple(
        0.5 * (minimum + maximum) for minimum, maximum in selected_bounds
    )
    body_boundary = body_max[axis] if sign > 0 else body_min[axis]
    outer_extent = handle_max[axis] if sign > 0 else handle_min[axis]
    outer_cut = body_boundary + 0.55 * (outer_extent - body_boundary)
    rail_indices = tuple(
        index
        for index, center in enumerate(selected_centers)
        if sign * center[axis] >= sign * outer_cut
    )
    connector_indices = tuple(
        index for index in range(len(selected_bounds)) if index not in rail_indices
    )
    if rail_indices:
        rail_inner_faces = [
            selected_bounds[index][0 if sign > 0 else 1][axis]
            for index in rail_indices
        ]
        rail_inner = float(np.median(rail_inner_faces))
        if sign * (rail_inner - body_boundary) > 1.0e-6:
            hole_center[axis] = 0.5 * (body_boundary + rail_inner)
    lower_faces = [
        selected_bounds[index][1][2]
        for index in connector_indices
        if selected_centers[index][2] < hole_center[2]
    ]
    upper_faces = [
        selected_bounds[index][0][2]
        for index in connector_indices
        if selected_centers[index][2] > hole_center[2]
    ]
    if lower_faces and upper_faces:
        cavity_lower = max(lower_faces)
        cavity_upper = min(upper_faces)
        if cavity_upper > cavity_lower:
            hole_center[2] = 0.5 * (cavity_lower + cavity_upper)
    hole_frame = _frame(hole_center, outward)
    handle_size = _projected_size(handle_points, hole_frame)
    component_thicknesses = []
    for points in selected:
        size = _projected_size(points, hole_frame)
        component_thicknesses.append(min(size[0], size[2]))
    thickness = float(np.median(component_thicknesses))
    body_center = 0.5 * (body_min + body_max)
    body_center[2] = 0.5 * (global_min[2] + global_max[2])
    body_frame = _frame(body_center, outward)
    body_points = all_points[
        (all_points[:, axis] >= body_min[axis] - 1.0e-6)
        & (all_points[:, axis] <= body_max[axis] + 1.0e-6)
    ]
    return MugParts(
        body_frame=body_frame,
        body_size=_projected_size(body_points, body_frame),
        handle_hole_frame=hole_frame,
        handle_outer_size=handle_size,
        handle_thickness_m=thickness,
        handle_axis=axis,
        handle_sign=sign,
    )


def infer_tree_branches(values: Iterable[object]) -> tuple[BranchPart, ...]:
    """Infer every non-base radial branch and its deterministic support frame."""

    components = _components(values)
    all_points = np.concatenate(components)
    global_min, global_max = _bounds(all_points)
    height = global_max[2] - global_min[2]
    radial_global = np.linalg.norm(all_points[:, :2], axis=1).max()
    result = []
    for points in components:
        minimum, maximum = _bounds(points)
        radial = np.linalg.norm(points[:, :2], axis=1)
        radial_span = float(radial.max() - radial.min())
        z_span = float(maximum[2] - minimum[2])
        if 0.5 * (minimum[2] + maximum[2]) < global_min[2] + 0.20 * height:
            continue
        if radial.max() < 0.45 * radial_global or radial_span < 0.025:
            continue
        outer_seed = points[int(np.argmax(radial))]
        horizontal = outer_seed[:2] / np.linalg.norm(outer_seed[:2])
        projection = points[:, :2] @ horizontal
        low = np.quantile(projection, 0.12)
        high = np.quantile(projection, 0.88)
        inner = points[projection <= low].mean(axis=0)
        tip = points[projection >= high].mean(axis=0)
        tangent = tip - inner
        length = float(np.linalg.norm(tangent))
        if length < 0.035 or radial_span < 0.75 * z_span:
            continue
        tangent /= length
        along = (points - inner) @ tangent
        perpendicular = points - inner - along[:, None] * tangent
        radius = float(np.quantile(np.linalg.norm(perpendicular, axis=1), 0.55))
        support = tip - tangent * min(0.35 * length, max(2.5 * radius, 0.012))
        frame = _frame(support, tangent)
        result.append(
            BranchPart(
                frame=frame,
                inner_point=inner,
                tip_point=tip,
                tangent=tangent,
                length_m=length,
                radius_m=radius,
                normalized_height=float((support[2] - global_min[2]) / height),
                azimuth_rad=float(np.arctan2(tangent[1], tangent[0])),
            )
        )
    if len(result) < 2:
        raise ValueError("fewer than two mug-tree branches were inferred")
    return tuple(sorted(result, key=lambda item: (item.normalized_height, item.azimuth_rad)))


def infer_open_drawer_cavity(
    values: Iterable[object], slide_axis_local: object
) -> tuple[np.ndarray, np.ndarray]:
    """Infer the open-top interior region bounded by authored drawer walls."""

    components = _components(values)
    axis = np.asarray(slide_axis_local, dtype=np.float64)
    slide_dimension = int(np.argmax(np.abs(axis)))
    if slide_dimension not in (0, 1):
        raise ValueError("drawer slide axis must be horizontal")
    transverse_dimension = 1 - slide_dimension
    all_points = np.concatenate(components)
    global_min, global_max = _bounds(all_points)
    global_size = global_max - global_min
    bounds = [_bounds(points) for points in components]

    interior_min = global_min.copy()
    interior_max = global_max.copy()
    for dimension in (slide_dimension, transverse_dimension):
        low_walls = []
        high_walls = []
        tolerance = 0.08 * global_size[dimension]
        for minimum, maximum in bounds:
            span = maximum - minimum
            if span[2] < 0.45 * global_size[2]:
                continue
            if span[dimension] > 0.30 * global_size[dimension]:
                continue
            if minimum[dimension] <= global_min[dimension] + tolerance:
                low_walls.append(maximum[dimension])
            if maximum[dimension] >= global_max[dimension] - tolerance:
                high_walls.append(minimum[dimension])
        if not low_walls or not high_walls:
            raise ValueError("could not infer both authored drawer side walls")
        interior_min[dimension] = max(low_walls)
        interior_max[dimension] = min(high_walls)

    floor_tops = []
    for minimum, maximum in bounds:
        if (
            minimum[2] <= global_min[2] + 0.08 * global_size[2]
            and maximum[2] - minimum[2] <= 0.30 * global_size[2]
        ):
            floor_tops.append(maximum[2])
    if not floor_tops:
        raise ValueError("could not infer the authored drawer floor")
    interior_min[2] = max(floor_tops)
    size = interior_max - interior_min
    if np.any(size <= 0.0):
        raise ValueError("inferred drawer cavity is empty")
    return 0.5 * (interior_min + interior_max), size


def closest_branch(branches: Iterable[BranchPart], point: object) -> BranchPart:
    """Return the branch segment closest to a local-frame point."""

    point = np.asarray(point, dtype=np.float64)

    def distance(branch: BranchPart) -> float:
        segment = branch.tip_point - branch.inner_point
        fraction = np.clip(
            np.dot(point - branch.inner_point, segment) / np.dot(segment, segment),
            0.0,
            1.0,
        )
        return float(np.linalg.norm(point - (branch.inner_point + fraction * segment)))

    values = tuple(branches)
    if not values:
        raise ValueError("at least one branch is required")
    return min(values, key=distance)


def corresponding_branch(
    source: BranchPart, targets: Iterable[BranchPart]
) -> BranchPart:
    """Match a branch by object-local height rank and outward azimuth."""

    def angle_error(value: float) -> float:
        return abs(float(np.arctan2(np.sin(value), np.cos(value))))

    values = tuple(targets)
    if not values:
        raise ValueError("at least one target branch is required")
    return min(
        values,
        key=lambda target: (
            2.0 * abs(target.normalized_height - source.normalized_height)
            + angle_error(target.azimuth_rad - source.azimuth_rad)
        ),
    )
