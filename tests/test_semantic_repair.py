import numpy as np

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
        "semantic_frames": {"intended_final_pot_pose": _pose(0.7, -0.3, 0.9)},
    }

    diagnosis = diagnose_semantic_failure("putpot", result)

    assert diagnosis.first_failed_stage == "support_alignment"
    np.testing.assert_allclose(
        diagnosis.signed_residuals["pot_center_in_cooktop_frame_m"],
        [0.04, -0.02, 0.1],
    )
    assert diagnosis.signed_residuals["center_margin_m"] < 0.0


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
        "geometry": {"target": {"slide_axis_local": [1.0, 0.0, 0.0]}},
        "protocol": {"steps": 607},
    }

    diagnosis = diagnose_semantic_failure("putmarker", result)

    assert diagnosis.first_failed_stage == "open_drawer"
    assert np.isclose(
        diagnosis.signed_residuals["drawer_open_residual_along_slide_axis_m"],
        -0.019,
    )


def test_hangmug_diagnosis_distinguishes_handover_from_support():
    base = {
        "checks": {},
        "metrics": {"left_grasp_frames": 100, "right_grasp_frames": 0},
        "protocol": {"steps": 800},
        "semantic_frames": {"intended_final_mug_pose": _pose(0.7, -0.2, 1.0)},
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
