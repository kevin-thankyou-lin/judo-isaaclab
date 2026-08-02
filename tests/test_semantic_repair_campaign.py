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


def test_repair_ingestion_accepts_semantic_motion_without_replay_failure_control():
    module = _module()
    source = {
        "result": {
            "mode": "skill",
            "status": "failed",
            "checks": {"coded_task_success": True},
            "acceptance_checks": {
                "coded_task_success": True,
                "all_stages_latched": True,
                "stable_support_window": True,
                "direct_source_action_replay_failed": False,
            },
        }
    }

    assert module._semantic_motion_success(source)
    assert not module._strict_semantic_success(source)
    source["result"]["acceptance_checks"]["stable_support_window"] = False
    assert not module._semantic_motion_success(source)


def test_preserved_deterministic_primary_success_is_not_replayed():
    module = _module()
    entry = {
        "status": "accepted",
        "method": "source_action_prefix_with_supported_center_repair",
        "demonstration": {"path": "/tmp/preserved.hdf5"},
    }
    result = {"checks": {"coded_task_success": True}}

    assert module._preserved_primary_success(entry, result)
    entry["method"] = "source_action_replay"
    assert not module._preserved_primary_success(entry, result)
