"""Inspect mesh and collision bounds for one or more MugTree USD assets."""

import argparse
import json

import numpy as np

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="+")
    args = parser.parse_args()
    simulation_app = AppLauncher(
        {"headless": True, "device": "cpu", "enable_cameras": False}
    ).app
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        result = {}
        prims = {}
        for asset in args.assets:
            stage = Usd.Stage.Open(asset)
            prims[asset] = {
                "default_prim": str(stage.GetDefaultPrim().GetPath()),
                "paths": [str(prim.GetPath()) for prim in stage.Traverse()],
            }
            transforms = UsdGeom.XformCache()
            meshes = []
            for prim in stage.Traverse():
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                transform = transforms.GetLocalToWorldTransform(prim)
                world_points = (
                    np.asarray(
                        [
                            transform.Transform(Gf.Vec3d(point))
                            for point in points
                        ],
                        dtype=np.float64,
                    )
                    if points
                    else np.empty((0, 3))
                )
                center = world_points.mean(axis=0) if points else None
                covariance = (
                    np.cov(world_points - center, rowvar=False)
                    if len(world_points) > 2
                    else None
                )
                eigenvalues, eigenvectors = (
                    np.linalg.eigh(covariance)
                    if covariance is not None
                    else (None, None)
                )
                meshes.append(
                    {
                        "path": str(prim.GetPath()),
                        "point_count": len(points) if points else 0,
                        "world_bounds": (
                            [
                                world_points.min(axis=0).tolist(),
                                world_points.max(axis=0).tolist(),
                            ]
                            if points
                            else None
                        ),
                        "world_center": center.tolist() if points else None,
                        "principal_axis": (
                            eigenvectors[:, np.argmax(eigenvalues)].tolist()
                            if eigenvalues is not None
                            else None
                        ),
                        "collision": prim.HasAPI(UsdPhysics.CollisionAPI),
                    }
                )
            result[asset] = meshes
        print(
            "MUGTREE_GEOMETRY=" + json.dumps(result, sort_keys=True),
            flush=True,
        )
        print("MUGTREE_PRIMS=" + json.dumps(prims, sort_keys=True), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
