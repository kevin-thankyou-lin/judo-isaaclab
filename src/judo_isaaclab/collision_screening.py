"""Collision screening for rigidly grasped object Cartesian paths."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import numpy as np


def _quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a scalar-first quaternion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        other = (axis + 1) % 3
        last = (axis + 2) % 3
        scale = np.sqrt(
            1.0 + matrix[axis, axis] - matrix[other, other] - matrix[last, last]
        ) * 2.0
        quaternion = np.zeros(4, dtype=np.float64)
        quaternion[0] = (matrix[last, other] - matrix[other, last]) / scale
        quaternion[axis + 1] = 0.25 * scale
        quaternion[other + 1] = (matrix[other, axis] + matrix[axis, other]) / scale
        quaternion[last + 1] = (matrix[last, axis] + matrix[axis, last]) / scale
    return quaternion / np.linalg.norm(quaternion)


def _pose_matrix(pose: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _quat_to_matrix(pose[3:7])
    result[:3, 3] = pose[:3]
    return result


def _matrix_pose(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix[:3, 3], _matrix_to_quat(matrix[:3, :3])))


def rigid_weld_object_poses(
    eef_path: np.ndarray,
    current_eef_pose: np.ndarray,
    current_object_pose: np.ndarray,
) -> np.ndarray:
    """Propagate an object along an EEF path with a fixed relative transform."""
    eef_path = np.asarray(eef_path, dtype=np.float64)
    eef_to_object = np.linalg.inv(_pose_matrix(current_eef_pose)) @ _pose_matrix(
        current_object_pose
    )
    return np.stack(
        [_matrix_pose(_pose_matrix(pose) @ eef_to_object) for pose in eef_path]
    )


def rigid_weld_eef_poses(
    object_path: np.ndarray,
    current_eef_pose: np.ndarray,
    current_object_pose: np.ndarray,
) -> np.ndarray:
    """Map an object-first path to EEF poses using the live grasp transform."""
    object_path = np.asarray(object_path, dtype=np.float64)
    object_to_eef = np.linalg.inv(_pose_matrix(current_object_pose)) @ _pose_matrix(
        current_eef_pose
    )
    return np.stack(
        [_matrix_pose(_pose_matrix(pose) @ object_to_eef) for pose in object_path]
    )


def _triangulate(counts, indices, offset):
    triangles = []
    cursor = 0
    for count in counts:
        face = indices[cursor : cursor + count]
        cursor += count
        for index in range(1, count - 1):
            triangles.append(
                [offset + face[0], offset + face[index], offset + face[index + 1]]
            )
    return triangles


def load_usd_collision_mesh(
    asset_path: str,
    collision_indices: tuple[int, ...] | list[int] | None = None,
):
    """Load selected composed USD collision meshes into one root-local mesh.

    ``collision_indices`` refers to the numeric suffix in prim names such as
    ``obj_link_collision_25``. Omitting it preserves the original all-mesh
    behavior.
    """
    indices = (
        None
        if collision_indices is None
        else tuple(sorted({int(index) for index in collision_indices}))
    )
    return _load_usd_collision_mesh(str(asset_path), indices)


@lru_cache(maxsize=32)
def _load_usd_collision_mesh(
    asset_path: str,
    collision_indices: tuple[int, ...] | None,
):
    import trimesh
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(Path(asset_path)))
    transforms = UsdGeom.XformCache()
    vertices = []
    faces = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or "/collisions/" not in prim_path:
            continue
        if collision_indices is not None:
            match = re.search(r"obj_link_collision_(\d+)(?:/|$)", prim_path)
            if match is None or int(match.group(1)) not in collision_indices:
                continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        transform = transforms.GetLocalToWorldTransform(prim)
        local_vertices = np.asarray(
            [transform.Transform(Gf.Vec3d(point)) for point in points],
            dtype=np.float64,
        )
        offset = sum(len(item) for item in vertices)
        vertices.append(local_vertices)
        faces.extend(
            _triangulate(
                list(mesh.GetFaceVertexCountsAttr().Get()),
                list(mesh.GetFaceVertexIndicesAttr().Get()),
                offset,
            )
        )
    if not vertices or not faces:
        raise ValueError(f"no composed collision meshes found in {asset_path}")
    return trimesh.Trimesh(
        vertices=np.concatenate(vertices, axis=0),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _quaternion_distance(left: np.ndarray, right: np.ndarray) -> float:
    dot = float(abs(np.dot(left, right)))
    return float(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def _sample_vertices(vertices: np.ndarray, maximum: int) -> np.ndarray:
    if len(vertices) <= maximum:
        return vertices
    return vertices[np.linspace(0, len(vertices) - 1, maximum).round().astype(int)]


def _sample_surface_points(mesh, maximum: int) -> np.ndarray:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    points = np.concatenate(
        (np.asarray(mesh.vertices, dtype=np.float64), triangles.mean(axis=1)),
        axis=0,
    )
    return _sample_vertices(points, maximum)


def _transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    matrix = _pose_matrix(pose)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _mesh_clearance(object_points, object_pose, tree_points, tree_query):
    """Approximate symmetric surface separation with cached KD trees."""
    from scipy.spatial import cKDTree

    object_world = _transform_points(object_points, object_pose)
    object_to_tree = tree_query.query(object_world, k=1, workers=1)[0]
    tree_to_object = cKDTree(object_world).query(
        tree_points, k=1, workers=1
    )[0]
    return float(min(object_to_tree.min(), tree_to_object.min()))


def screen_object_paths(
    object_paths: list[np.ndarray],
    *,
    tree_pose: np.ndarray,
    object_mesh,
    tree_mesh,
    preinsert_end_step: int,
    required_clearance_m: float = 0.002,
    sample_stride: int = 5,
    maximum_vertices: int = 1500,
):
    """Select the shortest pre-insertion path meeting mesh-clearance bounds."""
    from scipy.spatial import cKDTree

    tree_world = tree_mesh.copy()
    tree_world.apply_transform(_pose_matrix(tree_pose))
    tree_points = _sample_surface_points(tree_world, maximum_vertices)
    tree_query = cKDTree(tree_points)
    object_points = _sample_surface_points(object_mesh, maximum_vertices)
    reports = []
    for candidate_index, object_poses in enumerate(object_paths):
        steps = list(range(0, preinsert_end_step + 1, sample_stride))
        if steps[-1] != preinsert_end_step:
            steps.append(preinsert_end_step)
        clearances = [
            _mesh_clearance(
                object_points,
                object_poses[step],
                tree_points,
                tree_query,
            )
            for step in steps
        ]
        displacement = np.diff(object_poses[:, :3], axis=0)
        length = float(np.linalg.norm(displacement, axis=-1).sum())
        minimum = float(min(clearances))
        reports.append(
            {
                "candidate_index": candidate_index,
                "minimum_clearance_m": minimum,
                "path_length_m": length,
                "sampled_steps": steps,
                "sampled_clearance_m": clearances,
                "valid": minimum >= required_clearance_m,
            }
        )
    valid = [report for report in reports if report["valid"]]
    selected = (
        min(valid, key=lambda report: report["path_length_m"])
        if valid
        else max(
            reports,
            key=lambda report: (
                report["minimum_clearance_m"], -report["path_length_m"]
            ),
        )
    )
    return selected["candidate_index"], reports


def object_path_clearance_reports(
    object_paths: list[np.ndarray] | np.ndarray,
    *,
    tree_pose: np.ndarray,
    object_mesh,
    tree_mesh,
    required_clearance_m: float = 0.002,
    sample_stride: int = 1,
    maximum_vertices: int = 1500,
):
    """Measure whole-path body-to-tree clearance for semantic collision gates."""
    paths = np.asarray(object_paths, dtype=np.float64)
    if paths.ndim == 2:
        paths = paths[None, ...]
    _, reports = screen_object_paths(
        list(paths),
        tree_pose=np.asarray(tree_pose, dtype=np.float64),
        object_mesh=object_mesh,
        tree_mesh=tree_mesh,
        preinsert_end_step=paths.shape[1] - 1,
        required_clearance_m=required_clearance_m,
        sample_stride=sample_stride,
        maximum_vertices=maximum_vertices,
    )
    for report in reports:
        minimum_index = int(np.argmin(report["sampled_clearance_m"]))
        report["minimum_clearance_step"] = int(
            report["sampled_steps"][minimum_index]
        )
        report["semantic_surface"] = "object_body"
        report["allowed_tree_contact"] = False
    return reports


def object_path_collision_reports(
    object_paths: list[np.ndarray] | np.ndarray,
    *,
    tree_pose: np.ndarray,
    object_mesh,
    tree_mesh,
    sample_stride: int = 1,
):
    """Report exact FCL mesh intersections along one or more object paths."""
    import trimesh

    paths = np.asarray(object_paths, dtype=np.float64)
    if paths.ndim == 2:
        paths = paths[None, ...]
    manager = trimesh.collision.CollisionManager()
    manager.add_object("tree", tree_mesh, transform=_pose_matrix(tree_pose))
    manager.add_object("object_body", object_mesh)
    reports = []
    for candidate_index, object_poses in enumerate(paths):
        sampled_steps = list(range(0, len(object_poses), sample_stride))
        if sampled_steps[-1] != len(object_poses) - 1:
            sampled_steps.append(len(object_poses) - 1)
        collision_steps = []
        for step in sampled_steps:
            manager.set_transform(
                "object_body", _pose_matrix(object_poses[step])
            )
            if manager.in_collision_internal():
                collision_steps.append(step)
        reports.append(
            {
                "candidate_index": candidate_index,
                "sampled_steps": sampled_steps,
                "collision_steps": collision_steps,
                "collision_count": len(collision_steps),
                "first_collision_step": (
                    None if not collision_steps else int(collision_steps[0])
                ),
                "valid": not collision_steps,
                "semantic_surface": "object_body",
                "allowed_tree_contact": False,
                "method": "python-fcl exact mesh intersection",
            }
        )
    return reports


def select_robot_feasible_object_path(
    object_paths: np.ndarray,
    *,
    current_eef_pose: np.ndarray,
    current_object_pose: np.ndarray,
):
    """Select a supported object path with the easiest rigid-weld EEF motion.

    Translation and orientation travel are scored independently. The terminal
    orientation change is weighted most heavily because the right-arm tracker
    previously rotated the mug body into the branch while chasing a supported
    but awkward floating-object pose.
    """
    paths = np.asarray(object_paths, dtype=np.float64)
    if paths.ndim == 2:
        paths = paths[None, ...]
    reports = []
    for candidate_index, object_path in enumerate(paths):
        eef_path = rigid_weld_eef_poses(
            object_path, current_eef_pose, current_object_pose
        )
        translation_steps = np.linalg.norm(
            np.diff(eef_path[:, :3], axis=0), axis=-1
        )
        rotation_steps = np.asarray(
            [
                _quaternion_distance(left, right)
                for left, right in zip(eef_path[:-1, 3:7], eef_path[1:, 3:7])
            ]
        )
        terminal_rotation = _quaternion_distance(
            eef_path[0, 3:7], eef_path[-1, 3:7]
        )
        translation_length = float(translation_steps.sum())
        rotation_length = float(rotation_steps.sum())
        score = (
            translation_length
            + 0.05 * rotation_length
            + 0.20 * terminal_rotation
            + 0.10 * float(rotation_steps.max(initial=0.0))
        )
        reports.append(
            {
                "candidate_index": candidate_index,
                "robot_feasibility_score": score,
                "eef_translation_length_m": translation_length,
                "eef_rotation_length_rad": rotation_length,
                "eef_terminal_rotation_rad": terminal_rotation,
                "eef_max_rotation_step_rad": float(
                    rotation_steps.max(initial=0.0)
                ),
            }
        )
    selected = min(reports, key=lambda report: report["robot_feasibility_score"])
    return selected["candidate_index"], reports


def screen_rigid_weld_paths(
    paths: list[np.ndarray],
    *,
    current_eef_pose: np.ndarray,
    current_object_pose: np.ndarray,
    tree_pose: np.ndarray,
    object_mesh,
    tree_mesh,
    preinsert_end_step: int,
    required_clearance_m: float = 0.002,
    sample_stride: int = 5,
    maximum_vertices: int = 1500,
):
    """Screen EEF paths after propagating their rigidly grasped object."""
    object_paths = [
        rigid_weld_object_poses(path, current_eef_pose, current_object_pose)
        for path in paths
    ]
    return screen_object_paths(
        object_paths,
        tree_pose=tree_pose,
        object_mesh=object_mesh,
        tree_mesh=tree_mesh,
        preinsert_end_step=preinsert_end_step,
        required_clearance_m=required_clearance_m,
        sample_stride=sample_stride,
        maximum_vertices=maximum_vertices,
    )
