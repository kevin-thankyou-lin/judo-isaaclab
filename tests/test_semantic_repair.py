import numpy as np
import pytest

from judo_isaaclab.semantic_repair import diagnose_semantic_failure


def _pose(x=0.0, y=0.0, z=0.0):
    return [x, y, z, 1.0, 0.0, 0.0, 0.0]


def test_putpot_diagnosis_keeps_signed_cooktop_residuals():
    result = {
        "checks": {"bimanual_transport_completed": True, "centered_on_cooktop": False},
        "terminal": {
            "stage1": True,
            "cooktop_pose": _pose(0.7, -0.3, 0.8),
            "pot_pose": _pose(0.74, -0.32, 0.9),
            "center_error_m": 0.0447,
        },
        "metrics": {"left_grasp_frames": 100, "right_grasp_frames": 90},
        "protocol": {"steps": 505, "parameters": {"center_tolerance_m": 0.03}},
        "semantic_frames": {
            "intended_final_pot_pose": _pose(0.7, -0.3, 0.9),
            "target_cooktop_top": _pose(0.7, -0.3, 0.85),
            "target_pot_parts": {"bottom_z": -0.05},
        },
    }

    diagnosis = diagnose_semantic_failure("putpot", result)

    assert diagnosis.first_failed_stage == "support_alignment"
    np.testing.assert_allclose(
        diagnosis.signed_residuals["pot_center_in_cooktop_frame_m"],
        [0.04, -0.02, 0.1],
    )
    assert diagnosis.signed_residuals["center_margin_m"] < 0.0
    np.testing.assert_allclose(
        diagnosis.signed_residuals["pot_bottom_in_cooktop_support_frame_m"],
        [0.04, -0.02, 0.0],
    )


def test_putpot_task_success_with_clearance_failure_is_transport_failure():
    result = {
        "checks": {
            "bimanual_transport_completed": True,
            "smooth_collision_aware_transport": False,
            "centered_on_cooktop": True,
        },
        "terminal": {"stage1": True, "task_success": True},
        "metrics": {"transport_plan": {"end_step": 379}},
        "protocol": {"steps": 505},
    }

    assert diagnose_semantic_failure("putpot", result).first_failed_stage == (
        "bimanual_transport"
    )


def test_putpot_grasp_diagnosis_keeps_signed_first_contact_pad_miss():
    result = {
        "checks": {},
        "terminal": {"stage1": False, "center_error_m": 0.4},
        "metrics": {"left_grasp_frames": 0, "right_grasp_frames": 0},
        "protocol": {"steps": 505, "parameters": {"center_tolerance_m": 0.03}},
    }
    trace = {
        "left_finger_forces_n": np.asarray([[0.0, 0.0], [0.0, 11.0]]),
        "left_pad_fractions": np.asarray([[np.nan, np.nan], [np.nan, -0.041]]),
        "right_finger_forces_n": np.asarray([[0.0, 0.0], [4.0, 0.0]]),
        "right_pad_fractions": np.asarray([[np.nan, np.nan], [0.467, np.nan]]),
    }

    residuals = diagnose_semantic_failure(
        "putpot", result, trace
    ).signed_residuals

    assert residuals["left_first_contact_step"] == 1
    assert residuals["left_peak_contacting_fingers"] == 1
    assert residuals["left_first_contact_signed_pad_margin"] == pytest.approx(-0.041)
    assert residuals["right_first_contact_signed_pad_margin"] == pytest.approx(0.467)


def test_putmarker_diagnosis_reports_slide_axis_deficit():
    result = {
        "checks": {},
        "terminal": {
            "stage1": True,
            "stage2": False,
            "stage3": False,
            "cabinet_pose": _pose(0.8, 0.0, 0.9),
            "marker_pose": _pose(0.7, 0.02, 0.8),
            "drawer_joint_position": [0.0, 0.0],
        },
        "metrics": {"maximum_drawer_open_m": 0.031, "right_handle_grasp_frames": 0},
        "geometry": {
            "target": {
                "slide_axis_local": [1.0, 0.0, 0.0],
                "joint_origin_local": [0.1, 0.0, -0.1],
                "handle_point_local": [0.2, 0.0, -0.1],
                "cavity_size_m": [0.4, 0.3, 0.2],
            }
        },
        "protocol": {"steps": 607},
    }

    diagnosis = diagnose_semantic_failure("putmarker", result)

    assert diagnosis.first_failed_stage == "open_drawer"
    assert np.isclose(
        diagnosis.signed_residuals["drawer_open_residual_along_slide_axis_m"],
        -0.019,
    )
    assert "marker_center_in_drawer_cavity_frame_m" in diagnosis.signed_residuals


def test_hangmug_diagnosis_distinguishes_handover_from_support():
    base = {
        "checks": {},
        "metrics": {"left_grasp_frames": 100, "right_grasp_frames": 0},
        "protocol": {"steps": 800},
        "semantic_frames": {
            "intended_final_mug_pose": _pose(0.7, -0.2, 1.0),
            "target_mug_parts": {"handle_hole_frame": _pose(0.1, 0.0, 0.0)},
            "target_branch": {
                "frame": _pose(0.0, 0.0, 0.1),
                "tangent": [1.0, 0.0, 0.0],
                "tip_point": [0.2, 0.0, 0.1],
                "radius_m": 0.01,
            },
        },
    }
    handover = {
        **base,
        "terminal": {
            "stage1": True,
            "stage2": False,
            "tree_pose": _pose(0.75, -0.3, 0.9),
            "mug_pose": _pose(0.2, 0.0, 0.8),
            "mug_tree_xy_error_m": 0.5,
        },
    }
    support = {
        **base,
        "terminal": {
            "stage1": True,
            "stage2": True,
            "tree_pose": _pose(0.75, -0.3, 0.9),
            "mug_pose": _pose(0.72, -0.25, 0.82),
            "mug_tree_xy_error_m": 0.058,
        },
    }

    assert diagnose_semantic_failure("hangmug", handover).first_failed_stage == (
        "physical_handover"
    )
    assert diagnose_semantic_failure("hangmug", support).first_failed_stage == (
        "handle_to_branch_support"
    )
    support_diagnosis = diagnose_semantic_failure("hangmug", support)
    assert "handle_hole_center_in_branch_support_frame_m" in (
        support_diagnosis.signed_residuals
    )
