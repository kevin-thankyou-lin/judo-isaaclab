#!/usr/bin/env python3
"""Import one Articraft pot URDF as a dynamic rigid USD with provenance.

The attached Articraft pots are single-link URDFs.  This importer deliberately
uses IsaacLab's native convex-decomposition path, removes articulation metadata
that is inappropriate for a ``RigidObjectCfg``, measures the URDF collision
bounds, and writes the ``asset_size.json`` contract used by the PutPot task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _values(text: str | None, *, default: tuple[float, ...]) -> np.ndarray:
    return np.asarray(
        default if text is None else tuple(float(value) for value in text.split()),
        dtype=np.float64,
    )


def _rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cr, -sr), (0, sr, cr)))
    ry = np.asarray(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)))
    rz = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)))
    return rz @ ry @ rx


def _obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append(tuple(float(value) for value in line.split()[1:4]))
    if not vertices:
        raise ValueError(f"OBJ contains no vertices: {path}")
    return np.asarray(vertices, dtype=np.float64)


def urdf_collision_bounds(urdf_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Return local collision-space bounds and referenced mesh files."""

    urdf = Path(urdf_path).resolve()
    root = ET.parse(urdf).getroot()
    points: list[np.ndarray] = []
    meshes: list[Path] = []
    for collision in root.findall(".//collision"):
        mesh = collision.find("geometry/mesh")
        if mesh is None or not mesh.get("filename"):
            raise ValueError("Articraft importer requires mesh collision geometry")
        mesh_path = (urdf.parent / mesh.get("filename")).resolve()
        scale = _values(mesh.get("scale"), default=(1.0, 1.0, 1.0))
        origin = collision.find("origin")
        xyz = _values(origin.get("xyz") if origin is not None else None, default=(0, 0, 0))
        rpy = _values(origin.get("rpy") if origin is not None else None, default=(0, 0, 0))
        vertices = _obj_vertices(mesh_path) * scale
        points.append(vertices @ _rotation(rpy).T + xyz)
        meshes.append(mesh_path)
    if not points:
        raise ValueError(f"URDF contains no collision meshes: {urdf}")
    combined = np.concatenate(points, axis=0)
    return combined.min(axis=0), combined.max(axis=0), sorted(set(meshes))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Single-link Articraft URDF")
    parser.add_argument("--output", required=True, help="Output USD path")
    parser.add_argument("--mass-kg", type=float, default=1.0)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    from isaaclab.app import AppLauncher

    args = _parser().parse_args()
    launcher = AppLauncher(args)
    simulation_app = launcher.app
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from pxr import Usd, UsdPhysics

        source = Path(args.input).resolve()
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        lower, upper, meshes = urdf_collision_bounds(source)
        cfg = UrdfConverterCfg(
            asset_path=str(source),
            usd_dir=str(output.parent),
            usd_file_name=output.name,
            fix_base=False,
            merge_fixed_joints=True,
            force_usd_conversion=True,
            make_instanceable=False,
            collider_type="convex_decomposition",
            self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=0.0
                ),
                target_type="none",
            ),
        )
        converted = Path(UrdfConverter(cfg).usd_path).resolve()
        if converted != output:
            raise RuntimeError(f"converter wrote unexpected path: {converted}")

        stage = Usd.Stage.Open(str(output))
        if stage is None:
            raise RuntimeError(f"could not reopen converted USD: {output}")
        rigid_bodies = []
        for prim in list(stage.Traverse()):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_bodies.append(prim)
        if len(rigid_bodies) != 1:
            raise RuntimeError(f"expected one rigid body, found {len(rigid_bodies)}")
        mass_api = UsdPhysics.MassAPI.Apply(rigid_bodies[0])
        mass_api.CreateMassAttr(float(args.mass_kg))
        stage.GetRootLayer().Save()

        size = upper - lower
        asset_size = {"size": dict(zip(("x", "y", "z"), map(float, size)))}
        size_path = output.parent / "asset_size.json"
        size_path.write_text(json.dumps(asset_size, indent=2) + "\n", encoding="utf-8")
        receipt = {
            "schema_version": 1,
            "source_urdf": {"path": str(source), "sha256": _sha256(source)},
            "source_meshes": [
                {"path": str(path), "sha256": _sha256(path)} for path in meshes
            ],
            "bounds_m": {"min": lower.tolist(), "max": upper.tolist()},
            "mass_kg": float(args.mass_kg),
            "usd": {"path": str(output), "sha256": _sha256(output)},
            "asset_size": {"path": str(size_path), "sha256": _sha256(size_path)},
            "rigid_body_prim": str(rigid_bodies[0].GetPath()),
            "articulation_roots": 0,
        }
        receipt_path = output.parent / "import_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print("ARTICRAFT_PUTPOT_IMPORT=" + json.dumps(receipt, sort_keys=True))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
