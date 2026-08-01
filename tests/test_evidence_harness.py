import json

import pytest

from judo_isaaclab.evidence_harness import (
    EvidenceContract,
    EvidenceLedger,
    evaluate_result,
)


def _contract(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(
        json.dumps(
            {
                "task_name": "PutPotOnCooktop-v0",
                "success_check": "check_task_success",
                "source": {"dataset": "source.hdf5", "assets": {"pot": "p0"}},
                "target_search": {"require_replay_failure": True},
                "stages": ["pick", "place"],
                "result_paths": {
                    "task_success": ["terminal.task_success"],
                    "stages": {
                        "pick": ["terminal.stage1"],
                        "place": ["terminal.stage2"],
                    },
                    "metrics": {
                        "eef_tracking_error_m": ["metrics.eef_error"]
                    },
                },
                "required_protocol_checks": ["one_reset", "real_target_assets"],
                "thresholds": {"eef_tracking_error_m": 0.02},
            }
        )
    )
    return path


def _result(success=False, pick=True, place=False, error=0.0):
    return {
        "terminal": {"task_success": success, "stage1": pick, "stage2": place},
        "metrics": {"eef_error": error},
        "checks": {"one_reset": True, "real_target_assets": True},
    }


def test_target_replay_requires_real_failure(tmp_path):
    contract = EvidenceContract.load(_contract(tmp_path))
    failed = evaluate_result(contract, "target_replay", _result())
    assert failed.accepted
    assert failed.diagnosis == "placement_release_support"

    too_easy = evaluate_result(
        contract, "target_replay", _result(success=True, place=True)
    )
    assert not too_easy.accepted
    assert too_easy.diagnosis == "target_too_easy"


def test_diagnosis_separates_reachability_from_contact_path(tmp_path):
    contract = EvidenceContract.load(_contract(tmp_path))
    reach = evaluate_result(
        contract, "target_skill", _result(error=0.08)
    )
    assert reach.diagnosis == "reachability"
    contact = evaluate_result(
        contract, "target_skill", _result(error=0.001)
    )
    assert contact.diagnosis == "placement_release_support"


def test_protocol_and_video_fail_closed(tmp_path):
    contract = EvidenceContract.load(_contract(tmp_path))
    result = _result(success=True, place=True)
    assert not evaluate_result(contract, "final_render", result).accepted
    result["checks"]["one_reset"] = False
    checked = evaluate_result(
        contract, "target_skill", result, video_exists=True
    )
    assert not checked.accepted
    assert checked.diagnosis == "infrastructure_or_protocol"


def test_ledger_requires_same_target_and_revision(tmp_path):
    bundle = _contract(tmp_path)
    ledger = EvidenceLedger(tmp_path / "ledger.json", bundle)
    log = tmp_path / "run.log"
    log.write_text("ok")
    trace = tmp_path / "trace.npz"
    trace.write_bytes(b"trace")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")

    def add(phase, success, target="target", revision="rev"):
        result = tmp_path / f"{phase}.json"
        result.write_text(json.dumps(_result(success, place=success)))
        return ledger.add_attempt(
            phase=phase,
            result_path=result,
            log_path=log,
            returncode=0,
            revision=revision,
            source_id="source",
            target_id=target,
            trace_path=trace,
            video_path=video if phase == "final_render" else None,
        )

    assert add("source_skill", True)["accepted"]
    assert add("target_replay", False)["accepted"]
    assert add("target_skill", True)["accepted"]
    assert add("final_render", True, revision="other")["accepted"]
    assert not ledger.proof_status()["complete"]
    assert add("final_render", True)["accepted"]
    assert ledger.proof_status()["complete"]
    assert json.loads((tmp_path / "ledger.json").read_text())["status"] == "complete"


def test_contract_rejects_missing_replay_failure_gate(tmp_path):
    path = _contract(tmp_path)
    value = json.loads(path.read_text())
    value["target_search"]["require_replay_failure"] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="replay failure"):
        EvidenceContract.load(path)


def test_contract_expands_environment_paths(tmp_path, monkeypatch):
    path = _contract(tmp_path)
    value = json.loads(path.read_text())
    value["source"]["dataset"] = "${TASK_DATA_ROOT}/source.hdf5"
    path.write_text(json.dumps(value))
    monkeypatch.setenv("TASK_DATA_ROOT", "/verified/data")
    assert EvidenceContract.load(path).source["dataset"] == "/verified/data/source.hdf5"
