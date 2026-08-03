import importlib.util
from pathlib import Path

import pytest


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


def test_failed_acceptance_check_is_diagnosed_even_if_task_success_was_seen():
    module = _module()
    source = {
        "source": "replay_success_semantic_audit",
        "result_path": "/tmp/result.json",
        "trace_path": "/tmp/trace.npz",
        "result": {
            "status": "failed",
            "checks": {"coded_task_success": True},
            "terminal": {"stage1": True, "stage2": False, "stage3": False},
            "metrics": {"maximum_drawer_open_m": 0.04},
            "protocol": {"steps": 607, "control_rate_hz": 30},
            "video": {"path": "/tmp/skill.mp4", "sha256": "abc"},
        },
    }
    diagnosis = module._diagnosis("putmarker", source)

    assert diagnosis["first_failed_stage"] == "open_drawer"
    assert diagnosis["visual_evidence"]["frame"] == 418


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


def test_taxonomy_groups_every_failure_record_by_first_failed_stage():
    module = _module()
    taxonomy = module._taxonomy(
        {
            "putmarker:a": {
                "task": "putmarker",
                "diagnoses": [
                    {
                        "first_failed_stage": "open_drawer",
                        "source": "primary_campaign",
                    },
                    {
                        "first_failed_stage": "open_drawer",
                        "source": "replay_success_semantic_audit",
                    },
                ],
            },
            "hangmug:b": {
                "task": "hangmug",
                "diagnoses": [
                    {
                        "first_failed_stage": "physical_handover",
                        "source": "primary_campaign",
                    }
                ],
            },
        }
    )

    assert taxonomy["diagnosed_failure_records"] == 3
    drawer = taxonomy["by_task_and_first_failed_stage"]["putmarker"]["open_drawer"]
    assert drawer["failure_records"] == 2
    assert drawer["pairs"] == ["putmarker:a"]


def test_preserved_demo_receipt_must_match_recorded_hash(monkeypatch):
    module = _module()
    monkeypatch.setattr(
        module,
        "validate_demo",
        lambda path, assets: {"path": path, "sha256": "actual", "actions": 607},
    )

    assert module._validate_demo_receipt(
        {"path": "/tmp/demo.hdf5", "sha256": "actual", "actions": 607}, {}
    )["sha256"] == "actual"
    with pytest.raises(RuntimeError, match="hash does not match"):
        module._validate_demo_receipt(
            {"path": "/tmp/demo.hdf5", "sha256": "stale", "actions": 607}, {}
        )


def test_fail_fast_lane_stops_only_after_a_failed_attempt():
    module = _module()

    assert not module._stop_after_attempt(True, {"status": "accepted"})
    assert module._stop_after_attempt(True, {"status": "pending"})
    assert not module._stop_after_attempt(False, {"status": "pending"})
