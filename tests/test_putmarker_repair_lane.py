import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _module():
    path = Path(__file__).parents[1] / "examples/run_putmarker_repair_lane.py"
    spec = importlib.util.spec_from_file_location("putmarker_repair_lane", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lane_bootstraps_repository_src_for_failure_diagnosis():
    module = _module()
    assert str(module.REPO_ROOT / "src") in sys.path


def test_actual_cpu_receipt_is_fail_closed():
    module = _module()
    result = {
        "protocol": {
            "actual_device_receipt": {
                "requested": "cpu",
                "expected": "cpu",
                "actual": {
                    "manager_environment": "cpu",
                    "simulation_context": "cpu",
                    "action_tensors": ["cpu"],
                },
                "matched": True,
            }
        }
    }

    assert module._actual_cpu_receipt(result)["matched"] is True
    result["protocol"]["actual_device_receipt"]["actual"][
        "simulation_context"
    ] = "cuda:0"
    with pytest.raises(RuntimeError, match="CPU physics/action receipt failed"):
        module._actual_cpu_receipt(result)


def test_summary_preserves_all_task_statuses():
    module = _module()
    summary = module._summary(
        {
            "putmarker:a": {"status": "accepted"},
            "putmarker:b": {"status": "pending"},
            "hangmug:c": {"status": "semantic_success_artifact_pending"},
        }
    )

    assert summary["total"] == 3
    assert summary["accepted"] == 1
    assert summary["pending"] == 1
    assert summary["semantic_success_artifact_pending"] == 1


def test_pair_merge_changes_only_target_record(tmp_path, monkeypatch):
    module = _module()
    ledger_path = tmp_path / "semantic_repair_ledger.json"
    original_other = {
        "task": "hangmug",
        "pair_id": "mug_000__tree_000",
        "status": "pending",
        "attempts": [{"foreign": "must survive"}],
    }
    ledger = {
        "pairs": {
            "hangmug:mug_000__tree_000": original_other,
            "putmarker:marker_001__drawer_001": {
                "task": "putmarker",
                "pair_id": "marker_001__drawer_001",
                "dataset": "/dataset/one.hdf5",
                "assets": {"obj_0": "marker", "obj_1": "drawer"},
                "status": "pending",
                "attempts": [],
                "diagnoses": [{"reason": "open"}],
            },
        },
        "summary": {},
    }
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    attempt_root = tmp_path / "stage" / "attempt_001"
    attempt_root.mkdir(parents=True)
    (attempt_root / "skill_result.json").write_text("{}", encoding="utf-8")
    (attempt_root / "skill_demo.hdf5").write_bytes(b"demo")
    monkeypatch.setattr(
        module,
        "validate_demo",
        lambda path, assets: {"path": str(path), "sha256": "demo", "actions": 607},
    )
    monkeypatch.setattr(module, "_artifact_receipt", lambda *args: {"ok": True})
    attempt = {
        "status": "accepted",
        "workflow_id": "workflow",
        "code_head": "deadbeef",
        "actual_device_receipt": {"matched": True},
    }

    receipt = module._merge_accepted_pair(
        ledger_path=ledger_path,
        output_root=tmp_path / "results",
        key="putmarker:marker_001__drawer_001",
        attempt=attempt,
        attempt_root=attempt_root,
        lane_id="isolated_lane",
    )

    merged = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "merged"
    assert merged["pairs"]["hangmug:mug_000__tree_000"] == original_other
    target = merged["pairs"]["putmarker:marker_001__drawer_001"]
    assert target["status"] == "accepted"
    assert target["accepted_source"] == "isolated_lane"
    assert target["diagnoses"] == [{"reason": "open"}]
    assert target["attempts"][0]["workflow_id"] == "workflow"


def test_merge_skips_pair_already_accepted_elsewhere(tmp_path):
    module = _module()
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "pairs": {
                    "putmarker:a": {
                        "task": "putmarker",
                        "pair_id": "a",
                        "status": "accepted",
                        "accepted_source": "other_lane",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    receipt = module._merge_accepted_pair(
        ledger_path=ledger_path,
        output_root=tmp_path,
        key="putmarker:a",
        attempt={},
        attempt_root=tmp_path / "unused",
        lane_id="this_lane",
    )

    assert receipt["status"] == "already_accepted"
    assert receipt["accepted_source"] == "other_lane"
