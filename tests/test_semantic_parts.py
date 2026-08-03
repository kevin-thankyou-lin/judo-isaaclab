import numpy as np

from judo_isaaclab.semantic_parts import (
    bimanual_handle_sides,
    closest_branch,
    corresponding_branch,
    infer_mug_parts,
    infer_open_drawer_cavity,
    infer_pot_handle_contact_frame,
    infer_pot_parts,
    infer_tree_branches,
)


def _box(minimum, maximum):
    minimum = np.asarray(minimum, dtype=float)
    maximum = np.asarray(maximum, dtype=float)
    return np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ]
    )


def _rod(inner, tip, radius=0.006):
    inner = np.asarray(inner, dtype=float)
    tip = np.asarray(tip, dtype=float)
    tangent = tip - inner
    tangent /= np.linalg.norm(tangent)
    side = np.cross([0.0, 0.0, 1.0], tangent)
    side /= np.linalg.norm(side)
    up = np.cross(tangent, side)
    return np.asarray(
        [
            endpoint + a * radius * side + b * radius * up
            for endpoint in (inner, tip)
            for a in (-1.0, 1.0)
            for b in (-1.0, 1.0)
        ]
    )


def test_infer_pot_parts_uses_handle_overhang_not_asset_id():
    components = [
        _box((-0.12, -0.11, -0.08), (0.12, 0.11, -0.07)),
        _box((-0.12, -0.11, -0.07), (0.12, 0.11, 0.08)),
        _box((0.12, -0.025, 0.01), (0.19, 0.025, 0.06)),
        _box((-0.19, -0.025, 0.01), (-0.12, 0.025, 0.06)),
    ]
    parts = infer_pot_parts(components)

    assert parts.handle_axis == 0
    np.testing.assert_allclose(parts.positive_handle_frame[:3], [0.155, 0.0, 0.035])
    np.testing.assert_allclose(parts.negative_handle_frame[:3], [-0.155, 0.0, 0.035])
    assert parts.bottom_z == -0.08
    assert bimanual_handle_sides(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        parts,
        [-0.16, 0.0, 0.04],
        [0.16, 0.0, 0.04],
    ) == (-1, 1)


def test_pot_handle_contact_frame_uses_nearest_authored_segment_tangent():
    negative = _rod((-0.12, -0.02, 0.01), (-0.19, 0.03, 0.06), radius=0.004)
    positive = _rod((0.12, 0.02, 0.01), (0.19, -0.03, 0.06), radius=0.004)
    components = [
        _box((-0.12, -0.11, -0.08), (0.12, 0.11, -0.07)),
        _box((-0.12, -0.11, -0.07), (0.12, 0.11, 0.08)),
        negative,
        positive,
    ]
    parts = infer_pot_parts(components)
    frame = infer_pot_handle_contact_frame(
        components, parts, -1, [-0.22, 0.04, 0.08]
    )

    from judo_isaaclab.put_marker import quaternion_rotate

    inferred_tangent = quaternion_rotate(frame[3:], [1.0, 0.0, 0.0])
    authored_tangent = negative[-1] - negative[0]
    authored_tangent /= np.linalg.norm(authored_tangent)
    assert abs(float(np.dot(inferred_tangent, authored_tangent))) > 0.9
    assert frame[0] < parts.body_xy_min[0]


def test_infer_mug_parts_separates_body_footprint_and_handle_hole():
    components = [
        _box((-0.05, -0.04, -0.04), (0.025, 0.04, -0.034)),
        _box((-0.05, -0.04, -0.034), (0.025, 0.04, 0.04)),
        _box((0.025, -0.006, -0.025), (0.055, 0.006, -0.012)),
        _box((0.045, -0.006, -0.012), (0.058, 0.006, 0.018)),
        _box((0.025, -0.006, 0.018), (0.055, 0.006, 0.03)),
    ]
    parts = infer_mug_parts(components)

    assert parts.handle_axis == 0
    assert parts.handle_sign == 1
    np.testing.assert_allclose(parts.body_frame[:2], [-0.0125, 0.0], atol=1.0e-8)
    np.testing.assert_allclose(parts.handle_hole_frame[:3], [0.0415, 0.0, 0.0025])
    assert 0.009 < parts.handle_thickness_m < 0.014


def test_infer_open_drawer_cavity_uses_authored_floor_and_walls():
    components = [
        _box((-0.12, -0.18, -0.24), (0.14, 0.18, -0.22)),
        _box((-0.12, -0.18, -0.24), (-0.10, 0.18, 0.0)),
        _box((0.12, -0.18, -0.24), (0.14, 0.18, 0.0)),
        _box((-0.12, -0.18, -0.24), (0.14, -0.16, 0.0)),
        _box((-0.12, 0.16, -0.24), (0.14, 0.18, 0.0)),
    ]

    center, size = infer_open_drawer_cavity(components, [1.0, 0.0, 0.0])

    np.testing.assert_allclose(center, [0.01, 0.0, -0.11])
    np.testing.assert_allclose(size, [0.22, 0.32, 0.22])


def test_tree_branch_selection_and_correspondence_use_local_geometry():
    source_components = [
        _box((-0.08, -0.08, -0.18), (0.08, 0.08, -0.16)),
        _box((-0.01, -0.01, -0.16), (0.01, 0.01, 0.18)),
        _rod((0.01, 0.0, -0.03), (0.08, 0.0, 0.01)),
        _rod((0.0, 0.01, 0.07), (0.0, 0.08, 0.10)),
        _rod((-0.01, 0.0, 0.14), (-0.08, 0.0, 0.18)),
    ]
    target_components = [
        _box((-0.09, -0.09, -0.20), (0.09, 0.09, -0.18)),
        _box((-0.012, -0.012, -0.18), (0.012, 0.012, 0.20)),
        _rod((0.012, 0.0, -0.02), (0.09, 0.0, 0.02)),
        _rod((0.0, 0.012, 0.08), (0.0, 0.09, 0.12)),
        _rod((-0.012, 0.0, 0.15), (-0.09, 0.0, 0.20)),
    ]
    source = infer_tree_branches(source_components)
    target = infer_tree_branches(target_components)

    selected = closest_branch(source, [0.06, 0.0, -0.005])
    matched = corresponding_branch(selected, target)

    assert len(source) == len(target) == 3
    assert selected.tangent[0] > 0.8
    assert matched.tangent[0] > 0.8
    assert matched.normalized_height < 0.6
