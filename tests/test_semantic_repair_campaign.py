import importlib.util
import hashlib
import json
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


def test_putpot_semantic_repair_requires_bimanual_transport_without_replay():
    module = _module()
    source = {
        "source": "semantic_repair",
        "result": {
            "mode": "skill",
            "status": "passed",
            "checks": {
                "coded_task_success": True,
                "bimanual_transport_completed": False,
                "h264_nonempty": True,
                "fully_decodable": True,
            },
            "acceptance_checks": {"coded_task_success": True},
            "direct_replay_baseline": None,
            "provenance": {"demonstration": {"path": "/tmp/demo.hdf5"}},
            "video": {"path": "/tmp/skill.mp4"},
        },
    }

    assert not module._strict_semantic_success(source, "putpot")
    assert not module._semantic_motion_success(source, "putpot")
    source["result"]["checks"]["bimanual_transport_completed"] = True
    assert module._strict_semantic_success(source, "putpot")


def test_putpot_diagnostic_command_disables_render_and_video():
    module = _module()
    command = [
        "python",
        "runner.py",
        "--render",
        "--video",
        "/tmp/skill.mp4",
        "--trace-npz",
        "/tmp/trace.npz",
    ]
    assert module._without_render(command) == [
        "python",
        "runner.py",
        "--trace-npz",
        "/tmp/trace.npz",
    ]


def test_putpot_rendered_worker_accepts_without_second_isaac_launch(
    tmp_path, monkeypatch
):
    module = _module()
    monkeypatch.setattr(module, "_git_head", lambda: "candidate-head")
    revised_spec = tmp_path / "revised_spec.json"
    revised = json.loads(
        (module.REPO_ROOT / "configs/putpot_semantic_program_v4.json").read_text()
    )
    revised["parameters"]["receiving_jaw_reorientation_fraction"] = 0.9
    revised_spec.write_text(json.dumps(revised))

    def fake_command(*args, output, **kwargs):
        return [
            "python",
            "runner.py",
            "--render",
            "--video",
            str(output / "skill.mp4"),
            "--result-json",
            str(output / "skill_result.json"),
        ]

    monkeypatch.setattr(module, "_command", fake_command)

    class FakeWorker:
        def __init__(self, command, **_kwargs):
            self.request_path = Path(
                command[command.index("--request-jsonl") + 1]
            )
            self.receipt_path = Path(
                command[command.index("--receipt-jsonl") + 1]
            )
            self.returncode = None
            self.wrote_attempt = False
            self.wrote_boundary = False

        def poll(self):
            queue = [
                json.loads(line)
                for line in self.request_path.read_text(encoding="utf-8").splitlines()
            ]
            attempts = [row for row in queue if row["type"] == "attempt"]
            if attempts and not self.wrote_attempt:
                assert len(attempts) == 1
                request = attempts[0]
                assert "--render" in request["argv"]
                assert "--video" in request["argv"]
                assert request["video"].endswith("attempt_001/skill.mp4")
                submitted_spec = json.loads(
                    Path(request["program_spec_json"]).read_text()
                )
                assert (
                    submitted_spec["parameters"][
                        "receiving_jaw_reorientation_fraction"
                    ]
                    == 0.9
                )
                Path(request["result_json"]).parent.mkdir(
                    parents=True, exist_ok=True
                )
                Path(request["result_json"]).write_text(
                    json.dumps(
                        {
                            "provenance": {
                                "demonstration": {"path": "/tmp/demo.hdf5"}
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                with open(self.receipt_path, "a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "type": "attempt",
                                "request_index": 1,
                                "request_id": request["request_id"],
                                "pair": request["pair"],
                                "pid": 123,
                                "started_at": "start",
                                "finished_at": "finish",
                                "code_head": "candidate-head",
                                "result_json": request["result_json"],
                                "result_present": True,
                                "error": None,
                                "diagnostic_classification": "final_acceptance_candidate",
                                "render_recommendation": "final_acceptance_candidate",
                                "runtime": {"reset_index": 1},
                                "program_spec": {
                                    "path": request["program_spec_json"],
                                    "sha256": request["program_spec_sha256"],
                                },
                                "controller_plugin": {
                                    "path": request["controller_plugin_py"],
                                    "sha256": request[
                                        "controller_plugin_sha256"
                                    ],
                                },
                            }
                        )
                        + "\n"
                    )
                self.wrote_attempt = True
            if any(row["type"] == "shutdown" for row in queue) and not self.wrote_boundary:
                with open(self.receipt_path, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"type": "worker_boundary"}) + "\n")
                self.wrote_boundary = True
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", FakeWorker)
    monkeypatch.setattr(module, "_strict_semantic_success", lambda *_args: True)
    monkeypatch.setattr(
        module,
        "validate_demo",
        lambda path, assets: {"path": path, "assets": assets},
    )
    attempts = module._putpot_worker_visit(
        {"name": "putpot"},
        {
            "pair_id": "cooktop_999__pot_999",
            "dataset": "/target.hdf5",
            "assets": {"pot": "pot.usd", "cooktop": "cooktop.usd"},
        },
        output_root=tmp_path,
        python="python",
        gear_repo="/gear",
        attempt_limit=4,
        repair_epoch="epoch-a",
        initial_program_spec_json=revised_spec,
    )

    assert [attempt["status"] for attempt in attempts] == ["accepted"]
    assert attempts[0]["video"].endswith("attempt_001/skill.mp4")
    assert attempts[0]["demonstration"]["path"] == "/tmp/demo.hdf5"


def test_completed_visit_receipts_survive_early_acceptance_discovery(
    tmp_path, monkeypatch
):
    module = _module()
    ledger = {
        "pairs": {
            "putpot:pair": {
                "status": "pending",
                "attempts": [],
            }
        }
    }
    written = []
    monkeypatch.setattr(
        module,
        "_atomic_json",
        lambda _path, value: written.append(
            json.loads(json.dumps(value))
        ),
    )

    def discover_acceptance(_config, _root, _path):
        refreshed = json.loads(json.dumps(ledger))
        refreshed["pairs"]["putpot:pair"]["attempts"] = json.loads(
            json.dumps(written[-1]["pairs"]["putpot:pair"]["attempts"])
        )
        refreshed["pairs"]["putpot:pair"]["status"] = "accepted"
        ledger.clear()
        ledger.update(refreshed)
        return ledger

    monkeypatch.setattr(module, "refresh_ledger", discover_acceptance)
    result = module._record_completed_visit_attempts(
        {},
        tmp_path,
        tmp_path / "ledger.json",
        ledger,
        "putpot:pair",
        [
            {"attempt": "attempt_001", "status": "render_candidate"},
            {"attempt": "attempt_002", "status": "accepted"},
        ],
        4,
    )

    assert result["pairs"]["putpot:pair"]["status"] == "accepted"
    assert [
        attempt["attempt"]
        for attempt in result["pairs"]["putpot:pair"]["attempts"]
    ] == ["attempt_001", "attempt_002"]
    assert len(written) == 2


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


def test_inaccessible_provenance_reuses_matching_hash_verified_demo(
    tmp_path, monkeypatch
):
    module = _module()
    preserved = tmp_path / "skill_demo.hdf5"
    preserved.write_bytes(b"verified")
    monkeypatch.setattr(
        module,
        "validate_demo",
        lambda path, assets: {"path": path, "sha256": "same", "actions": 800},
    )

    actual = module._validate_demo_receipt(
        {"path": "/home/gear/missing.hdf5", "sha256": "same", "actions": 800},
        {},
        fallback={"path": str(preserved), "sha256": "same", "actions": 800},
    )
    assert actual["path"] == str(preserved)
    with pytest.raises(RuntimeError, match="preserved demonstration hash differs"):
        module._validate_demo_receipt(
            {"path": "/home/gear/missing.hdf5", "sha256": "other"},
            {},
            fallback={"path": str(preserved), "sha256": "same"},
        )


def test_result_record_rebases_missing_artifacts_to_hash_identical_siblings(tmp_path):
    module = _module()
    payloads = {
        "skill.mp4": b"video",
        "skill_trace.npz": b"trace",
        "skill_demo.hdf5": b"demo",
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)

    def receipt(name):
        return {
            "path": f"/home/gear/results/missing/{name}",
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
        }

    result_path = tmp_path / "skill_result.json"
    result_path.write_text(
        json.dumps(
            {
                "video": receipt("skill.mp4"),
                "provenance": {
                    "trace": receipt("skill_trace.npz"),
                    "demonstration": receipt("skill_demo.hdf5"),
                },
            }
        )
    )
    record = module._result_record(result_path, "semantic_repair")
    assert record is not None
    assert record["published_artifacts_rebased"] is True
    assert record["video_path"] == str((tmp_path / "skill.mp4").resolve())
    assert record["trace_path"] == str((tmp_path / "skill_trace.npz").resolve())
    assert record["demonstration"]["path"] == str(
        (tmp_path / "skill_demo.hdf5").resolve()
    )


def test_fail_fast_lane_stops_only_after_a_failed_attempt():
    module = _module()

    assert not module._stop_after_attempt(True, {"status": "accepted"})
    assert module._stop_after_attempt(True, {"status": "pending"})
    assert not module._stop_after_attempt(False, {"status": "pending"})


def test_hard_case_pending_survives_nonaccepting_refresh_classification():
    module = _module()

    assert module._pending_status({"status": "hard_case_pending"}) == (
        "hard_case_pending"
    )
    assert module._pending_status({"status": "pending"}) == "pending"


def test_putpot_round_robin_visit_is_capped_at_eight_repair_cycles():
    module = _module()

    assert module._validate_fresh_attempts_per_asset_visit(8, {"putpot"}) == 8
    with pytest.raises(ValueError, match="capped at eight"):
        module._validate_fresh_attempts_per_asset_visit(9, {"putpot"})
    with pytest.raises(ValueError, match="must be positive"):
        module._validate_fresh_attempts_per_asset_visit(0, {"putpot"})


def test_repair_policy_is_target_direct_and_preserves_successes(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "_git_head", lambda: "rev")
    ledger = module.refresh_ledger(
        {"tasks": []}, tmp_path / "results", tmp_path / "ledger.json"
    )

    assert "source semantic success is not required" in ledger["policy"]
    assert "preserve hash-verified successes" in ledger["policy"]
    assert "stop on first failed pair" in ledger["policy"]
