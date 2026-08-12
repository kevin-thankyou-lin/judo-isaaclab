from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from check_hangmug_run import _grasp_assist_errors


def test_classification_subset_uses_full_grasp_assist_lifecycle():
    result = {
        "acceptance_checks": {"one_reset": True},
        "checks": {
            "datagen_grasp_assist_configured": True,
            "left_grasp_assist_engaged": True,
            "left_grasp_assist_released": True,
            "right_grasp_assist_released": True,
        },
        "terminal": {"task_success": True},
    }
    assert _grasp_assist_errors(result) == []
    result["checks"]["left_grasp_assist_engaged"] = False
    assert _grasp_assist_errors(result) == [
        "missing grasp-assist evidence: left_grasp_assist_engaged"
    ]
    result["checks"]["left_grasp_assist_engaged"] = True
    result["checks"]["right_grasp_assist_released"] = False
    assert _grasp_assist_errors(result) == [
        "right grasp assist remained engaged at terminal"
    ]
