import json
from pathlib import Path

import pytest

from judo_isaaclab.putpot_queue import (
    static_worker_argv,
    submit_program_request,
)
from judo_isaaclab.putpot_runtime import append_jsonl, read_jsonl


REPO_ROOT = Path(__file__).parents[1]
DEFAULT_SPEC = REPO_ROOT / "configs/putpot_semantic_program_v2.json"


def _spec(tmp_path, index):
    value = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
    value["parameters"]["receiving_jaw_reorientation_fraction"] = 0.40 + index / 100
    path = tmp_path / f"source_spec_{index}.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _session(tmp_path):
    request_path = tmp_path / "requests.jsonl"
    receipt_path = tmp_path / "receipts.jsonl"
    request_path.touch()
    session = {
        "schema_version": 1,
        "pair": "cooktop_001__pot_001",
        "code_head": "head",
        "repair_root": str(tmp_path / "semantic_repair"),
        "epoch_root": str(tmp_path / "epoch"),
        "repair_epoch": "epoch-a",
        "first_lifetime_attempt": 10,
        "attempt_limit": 4,
        "request_jsonl": str(request_path),
        "receipt_jsonl": str(receipt_path),
        "static_argv": ["--mode", "skill", "--device", "cpu"],
    }
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    return session_path, request_path, receipt_path


def test_static_worker_argv_removes_every_request_scoped_value():
    assert static_worker_argv(
        [
            "--mode",
            "skill",
            "--result-json",
            "old.json",
            "--program-spec-json",
            "old-spec.json",
            "--repair-epoch-attempt",
            "3",
        ]
    ) == ["--mode", "skill"]


def test_interactive_queue_enforces_ack_and_four_cycle_limit(tmp_path):
    session_path, request_path, receipt_path = _session(tmp_path)

    for cycle in range(1, 5):
        request = submit_program_request(session_path, _spec(tmp_path, cycle))
        assert request["repair_epoch_attempt"] == cycle
        assert request["lifetime_attempt"] == 9 + cycle
        assert Path(request["program_spec_json"]).is_file()
        if cycle == 1:
            with pytest.raises(RuntimeError, match="has not been acknowledged"):
                submit_program_request(session_path, _spec(tmp_path, 9))
        append_jsonl(
            receipt_path,
            {
                "type": "attempt",
                "request_id": request["request_id"],
                "program_spec": {"sha256": request["program_spec_sha256"]},
                "diagnostic_classification": "diagnosed_physics_failure",
            },
        )

    with pytest.raises(ValueError, match="cycle limit exceeded"):
        submit_program_request(session_path, _spec(tmp_path, 5))
    assert len([row for row in read_jsonl(request_path) if row["type"] == "attempt"]) == 4
