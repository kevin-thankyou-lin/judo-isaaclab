from pathlib import Path

import numpy as np

from import_articraft_putpot_asset import urdf_collision_bounds


def test_urdf_collision_bounds_applies_scale_origin_and_rotation(tmp_path: Path) -> None:
    mesh = tmp_path / "box.obj"
    mesh.write_text("v 0 0 0\nv 1 2 3\n", encoding="utf-8")
    urdf = tmp_path / "model.urdf"
    urdf.write_text(
        """<robot name="pot"><link name="pot">
        <collision><origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>
        <geometry><mesh filename="box.obj" scale="2 1 1"/></geometry></collision>
        </link></robot>""",
        encoding="utf-8",
    )

    lower, upper, meshes = urdf_collision_bounds(urdf)

    np.testing.assert_allclose(lower, [-1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(upper, [1.0, 2.0, 3.0], atol=1e-12)
    assert meshes == [mesh.resolve()]
