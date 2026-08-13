from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from check_hangmug_run import _grasp_assist_errors, _handover_contact_target_errors


def test_classification_subset_uses_full_grasp_assist_lifecycle():
    result = {
        "acceptance_checks": {"one_reset": True},
        "checks": {
            "datagen_grasp_assist_configured": True,
            "left_only_fixed_joint_assist_configured": True,
            "left_grasp_assist_engaged": True,
            "left_grasp_assist_released": True,
            "right_grasp_assist_absent": True,
        },
        "terminal": {"task_success": True},
    }
    assert _grasp_assist_errors(result) == []
    result["checks"]["left_grasp_assist_engaged"] = False
    assert _grasp_assist_errors(result) == [
        "missing grasp-assist evidence: left_grasp_assist_engaged"
    ]
    result["checks"]["left_grasp_assist_engaged"] = True
    result["checks"]["right_grasp_assist_absent"] = False
    assert _grasp_assist_errors(result) == [
        "missing grasp-assist evidence: right_grasp_assist_absent"
    ]


def test_skill_result_requires_scaled_source_body_contact_receipt():
    result = {
        "mode": "skill",
        "protocol": {"parameters": {"require_source_dual_body_contact": True}},
        "handover_contact_target": {
            "method": "source_dual_grasp_right_eef_in_mug_body_scaled",
            "checks": {
                "target_position_is_scaled_source_body_contact": True,
                "target_orientation_is_source_body_contact": True,
            },
            "passed": True,
        },
    }
    assert _handover_contact_target_errors(result) == []
    result["handover_contact_target"]["checks"][
        "target_position_is_scaled_source_body_contact"
    ] = False
    assert _handover_contact_target_errors(result) == [
        "scaled source BODY-frame receiver contact receipt failed"
    ]
