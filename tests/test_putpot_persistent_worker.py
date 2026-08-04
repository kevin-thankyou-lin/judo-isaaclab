import importlib.util
import json
import os
from pathlib import Path
import sys

from judo_isaaclab.putpot_program_spec import load_program_spec
from judo_isaaclab.putpot_runtime import append_jsonl, read_jsonl


REPO_ROOT = Path(__file__).parents[1]
DEFAULT_SPEC = REPO_ROOT / "configs/putpot_semantic_program_v3.json"


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


def test_two_different_specs_share_one_mocked_isaac_pid_and_reset(monkeypatch, tmp_path):
    module = _module()
    request_path = tmp_path / "requests.jsonl"
    receipt_path = tmp_path / "receipts.jsonl"
    request_path.touch()
    specs = [_write_spec(tmp_path, index) for index in (1, 2)]

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
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "checks": {
                        "coded_task_success": False,
                        "bimanual_transport_completed": False,
                        "h264_nonempty": False,
                        "fully_decodable": False,
                    },
                    "stage_success_trace": [{"step": -1}],
                    "provenance": {
                        "trace": {"path": str(result_path.with_name("trace.npz"))},
                        "program_spec": program_spec.receipt(),
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
    assert len(receipts) == 2
    assert {row["pid"] for row in receipts} == {os.getpid()}
    assert len({row["program_spec"]["sha256"] for row in receipts}) == 2
    assert [row["runtime"]["reset_index"] for row in receipts] == [1, 2]
    assert [row["runtime"]["runtime_reused"] for row in receipts] == [False, True]
    assert all(row["acknowledged"] for row in receipts)
    assert all("--persistent-session" in argv for argv in calls)
