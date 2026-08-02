import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "examples/run_semantic_repair_campaign.py"
    spec = importlib.util.spec_from_file_location("run_semantic_repair_campaign", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_strict_semantic_success_requires_demo_and_all_acceptance_checks():
    module = _module()
    source = {
        "result": {
            "mode": "skill",
            "status": "passed",
            "checks": {"coded_task_success": True},
            "acceptance_checks": {"coded_task_success": True, "fully_decodable": True},
            "provenance": {"demonstration": {"path": "/tmp/demo.hdf5"}},
        }
    }
    assert module._strict_semantic_success(source)
    source["result"]["acceptance_checks"]["fully_decodable"] = False
    assert not module._strict_semantic_success(source)


def test_audit_coded_success_is_classified_as_packaging_not_motion_failure():
    module = _module()
    source = {
        "source": "replay_success_semantic_audit",
        "result_path": "/tmp/result.json",
        "trace_path": "/tmp/trace.npz",
        "result": {
            "status": "failed",
            "checks": {"coded_task_success": True},
            "protocol": {"steps": 607, "control_rate_hz": 30},
            "video": {"path": "/tmp/skill.mp4", "sha256": "abc"},
        },
    }
    diagnosis = module._diagnosis("putmarker", source)

    assert diagnosis["first_failed_stage"] == "evidence_packaging"
    assert diagnosis["visual_evidence"]["frame"] == 606
