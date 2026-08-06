import importlib.util
import json
import os
from pathlib import Path
import sys

from judo_isaaclab.putpot_program_spec import load_program_spec
from judo_isaaclab.putpot_controller_protocol import sha256_file
from judo_isaaclab.putpot_runtime import append_jsonl, read_jsonl


REPO_ROOT = Path(__file__).parents[1]
DEFAULT_CONTROLLER = REPO_ROOT / "controllers/putpot_passthrough.py"
DEFAULT_SPEC = REPO_ROOT / "configs/putpot_semantic_program_v4.json"


def _module():
    path = REPO_ROOT / "examples/run_putpot_persistent_worker.py"
    spec = importlib.util.spec_from_file_location("run_putpot_persistent_worker", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_spec(tmp_path, index):
    value = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
    value["parameters"]["receiving_jaw_reorientation_fraction"] += index / 100
    path = tmp_path / f"spec_{index}.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def test_eight_different_specs_share_one_mocked_isaac_pid_and_ninth_is_rejected(
    monkeypatch, tmp_path
):
    module = _module()
    request_path = tmp_path / "requests.jsonl"
    receipt_path = tmp_path / "receipts.jsonl"
    request_path.touch()
    specs = [_write_spec(tmp_path, index) for index in range(1, 10)]

    for index, spec_path in enumerate(specs, 1):
        attempt = tmp_path / f"attempt_{index:03d}"
        spec = load_program_spec(spec_path)
        append_jsonl(
            request_path,
            {
                "type": "attempt",
                "request_id": f"epoch:{index}",
                "pair": "cooktop_001__pot_001",
                "code_head": "test-head",
                "argv": [
                    "--result-json",
                    str(attempt / "skill_result.json"),
                    "--program-spec-json",
                    str(spec_path),
                ],
                "log": str(attempt / "skill.log"),
                "result_json": str(attempt / "skill_result.json"),
                "program_spec_json": str(spec_path),
                "program_spec_sha256": spec.sha256,
                "controller_plugin_py": str(DEFAULT_CONTROLLER),
                "controller_plugin_sha256": sha256_file(DEFAULT_CONTROLLER),
            },
        )
    append_jsonl(
        request_path,
        {"type": "shutdown", "request_id": "epoch:shutdown", "reason": "test"},
    )

    monkeypatch.setattr(module, "_git_head", lambda: "test-head")
    monkeypatch.setattr(module, "_git_status", lambda: "")
    monkeypatch.setattr(module, "_process_started_monotonic", lambda: 0.0)
    module.putpot._PERSISTENT_RUNTIME = None
    calls = []

    def fake_main(argv):
        calls.append(list(argv))
        result_path = Path(argv[argv.index("--result-json") + 1])
        spec_path = Path(argv[argv.index("--program-spec-json") + 1])
        program_spec = load_program_spec(spec_path)
        reset_index = len(calls)
        controller_receipt = {
            "path": str(DEFAULT_CONTROLLER.resolve()),
            "sha256": sha256_file(DEFAULT_CONTROLLER),
            "pid": 777 + reset_index,
            "protocol_version": 1,
            "program": {"program_name": f"revision-{reset_index}"},
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "checks": {
                        "bimanual_pick_observed": False,
                        "coded_task_success": False,
                        "bimanual_transport_completed": False,
                        "h264_nonempty": False,
                        "fully_decodable": False,
                    },
                    "stage_success_trace": [{"step": -1}],
                    "protocol": {
                        "parameters": dict(program_spec.parameters),
                        "peer_contact_gripper_retime": {
                            "requested_close_horizon_steps": program_spec.parameters[
                                "receiving_jaw_close_horizon_steps"
                            ],
                            "applied_close_steps": 10,
                            "close_start_step": 100,
                            "close_end_step": 110,
                            "grasp_end_step": 200,
                        },
                    },
                    "provenance": {
                        "trace": {"path": str(result_path.with_name("trace.npz"))},
                        "program_spec": program_spec.receipt(),
                        "controller_plugin": controller_receipt,
                    },
                }
            ),
            encoding="utf-8",
        )
        module.putpot._LAST_ATTEMPT_RUNTIME_RECEIPT = {
            "pid": os.getpid(),
            "persistent": True,
            "runtime_reused": reset_index > 1,
            "reset_index": reset_index,
            "phase_timings_s": {},
            "program_spec": program_spec.receipt(),
            "controller_plugin": controller_receipt,
        }

    monkeypatch.setattr(module.putpot, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(REPO_ROOT / "examples/run_putpot_persistent_worker.py"),
            "--request-jsonl",
            str(request_path),
            "--receipt-jsonl",
            str(receipt_path),
            "--poll-interval-s",
            "0.001",
        ],
    )
    module.main()

    receipts = [row for row in read_jsonl(receipt_path) if row["type"] == "attempt"]
    assert len(receipts) == 8
    assert {row["pid"] for row in receipts} == {os.getpid()}
    assert len({row["program_spec"]["sha256"] for row in receipts}) == 8
    assert [row["runtime"]["reset_index"] for row in receipts] == list(range(1, 9))
    assert [row["runtime"]["runtime_reused"] for row in receipts] == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert all(row["acknowledged"] for row in receipts)
    assert all(row["failed_stage"] == "bimanual_handle_grasp" for row in receipts)
    assert all(
        "receiving_jaw_reorientation_fraction"
        in row["failed_stage_program_parameter_observations"]
        for row in receipts
    )
    assert all("--persistent-session" in argv for argv in calls)
    assert len(calls) == 8
    rejected = [row for row in read_jsonl(receipt_path) if row["type"] == "request_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["request_id"] == "epoch:9"
    assert rejected[0]["reason"] == "adaptive_diagnose_repair_cycle_limit"
