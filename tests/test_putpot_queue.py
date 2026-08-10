import json
from pathlib import Path

import pytest

from judo_isaaclab.putpot_queue import (
    static_worker_argv,
    submit_program_request,
)
from judo_isaaclab.putpot_program_spec import load_program_spec
from judo_isaaclab.putpot_runtime import append_jsonl, read_jsonl
from judo_isaaclab.putpot_repair_policy import (
    DEFAULT_POLICY,
    source_demo_card_receipt,
)


REPO_ROOT = Path(__file__).parents[1]
DEFAULT_SPEC = REPO_ROOT / "configs/putpot_semantic_program_v4.json"
DEFAULT_CONTROLLER = REPO_ROOT / "controllers/putpot_passthrough.py"


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
        "schema_version": 2,
        "pair": "cooktop_001__pot_001",
        "code_head": "head",
        "repair_root": str(tmp_path / "semantic_repair"),
        "epoch_root": str(tmp_path / "epoch"),
        "repair_epoch": "epoch-a",
        "first_lifetime_attempt": 10,
        "attempt_limit": 4,
        "initial_controller_plugin_py": str(DEFAULT_CONTROLLER),
        "request_jsonl": str(request_path),
        "receipt_jsonl": str(receipt_path),
        "static_argv": ["--mode", "skill", "--device", "cpu"],
    }
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    return session_path, request_path, receipt_path


def _enable_source_first_policy(session_path, tmp_path):
    card = {
        "schema_version": 1,
        "source_dataset": "/data/source.hdf5",
        "source_dataset_sha256": "a" * 64,
        "source_assets": {"pot": {}, "cooktop": {}},
        "stage_sequence": ["staged_bilateral_acquisition"],
        "contact_order": ["left", "right"],
        "semantic_frames": {},
        "latch_contract": {},
        "lift_direction_in_source_world": [0.0, 0.0, 1.0],
        "transport_contract": {
            "frame": "observed_pot_pose",
            "preserve_loaded_object_local_grasp_transforms": True,
            "zero_jump_at_handoff": True,
        },
        "release_contract": {},
    }
    card_path = tmp_path / "source_demo_card.json"
    card_path.write_text(json.dumps(card, sort_keys=True), encoding="utf-8")
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["schema_version"] = 4
    session["source_demo_card"] = source_demo_card_receipt(card_path)
    session["repair_policy"] = dict(DEFAULT_POLICY)
    session_path.write_text(json.dumps(session), encoding="utf-8")
    return session


def _failed_grasp_result(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "failed",
                "checks": {
                    "bimanual_pick_observed": False,
                    "bimanual_transport_completed": False,
                    "h264_nonempty": True,
                    "fully_decodable": True,
                },
                "metrics": {},
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )


def _proposal(path, session, baseline, mechanism, family="staged_bilateral_acquisition", target="bimanual_handle_grasp"):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_demo_card_sha256": session["source_demo_card"]["sha256"],
                "baseline_request_id": baseline,
                "target_stage": target,
                "mechanism_id": mechanism,
                "repair_family": family,
                "hypothesis": "source contact order avoids premature object motion",
                "expected_task_delta": "pass the robust bilateral latch gate",
                "changed_primitives": ["contact_order", "pregrasp"],
            }
        ),
        encoding="utf-8",
    )
    return path


def _ack_failed_grasp(receipt_path, request):
    result_path = Path(request["result_json"])
    _failed_grasp_result(result_path)
    append_jsonl(
        receipt_path,
        {
            "type": "attempt",
            "request_id": request["request_id"],
            "result_json": str(result_path),
            "program_spec": load_program_spec(request["program_spec_json"]).receipt(),
            "controller_plugin": {
                "path": request["controller_plugin_py"],
                "sha256": request["controller_plugin_sha256"],
            },
            "diagnostic_classification": "diagnosed_physics_failure",
            "failed_stage": "bimanual_handle_grasp",
            "failed_stage_program_parameter_observations": {},
        },
    )


def test_static_worker_argv_removes_every_request_scoped_value():
    assert static_worker_argv(
        [
            "--mode",
            "skill",
            "--result-json",
            "old.json",
            "--video",
            "old.mp4",
            "--program-spec-json",
            "old-spec.json",
            "--repair-epoch-attempt",
            "3",
            "--controller-plugin-py",
            "controller.py",
            "--controller-plugin-sha256",
            "hash",
            "--controller-plugin-log",
            "controller.log",
        ]
    ) == ["--mode", "skill"]


def test_rendered_session_assigns_fresh_video_to_every_request(tmp_path):
    session_path, _, _ = _session(tmp_path)
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["static_argv"].append("--render")
    session_path.write_text(json.dumps(session), encoding="utf-8")

    request = submit_program_request(session_path, _spec(tmp_path, 1))

    assert "--render" in request["argv"]
    video_argument = request["argv"][request["argv"].index("--video") + 1]
    assert video_argument == request["video"]
    assert request["video"].endswith("attempt_010/skill.mp4")


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
                "program_spec": load_program_spec(
                    request["program_spec_json"]
                ).receipt(),
                "controller_plugin": {
                    "path": request["controller_plugin_py"],
                    "sha256": request["controller_plugin_sha256"],
                },
                "diagnostic_classification": "diagnosed_physics_failure",
                "failed_stage": "bimanual_handle_grasp",
                "failed_stage_program_parameter_observations": {
                    "receiving_jaw_reorientation_fraction": {
                        "requested": load_program_spec(
                            request["program_spec_json"]
                        ).parameters["receiving_jaw_reorientation_fraction"],
                        "observed": load_program_spec(
                            request["program_spec_json"]
                        ).parameters["receiving_jaw_reorientation_fraction"],
                    }
                },
            },
        )

    with pytest.raises(ValueError, match="cycle limit exceeded"):
        submit_program_request(session_path, _spec(tmp_path, 5))
    assert len([row for row in read_jsonl(request_path) if row["type"] == "attempt"]) == 4


def test_queue_rejects_hash_change_for_unobserved_failed_stage_parameter(tmp_path):
    session_path, request_path, receipt_path = _session(tmp_path)
    first = submit_program_request(session_path, _spec(tmp_path, 1))
    first_spec = load_program_spec(first["program_spec_json"])
    append_jsonl(
        receipt_path,
        {
            "type": "attempt",
            "request_id": first["request_id"],
            "program_spec": first_spec.receipt(),
            "controller_plugin": {
                "path": first["controller_plugin_py"],
                "sha256": first["controller_plugin_sha256"],
            },
            "diagnostic_classification": "diagnosed_physics_failure",
            "failed_stage": "bimanual_handle_grasp",
            "failed_stage_program_parameter_observations": {
                "receiving_jaw_reorientation_fraction": {
                    "requested": first_spec.parameters[
                        "receiving_jaw_reorientation_fraction"
                    ],
                    "observed": first_spec.parameters[
                        "receiving_jaw_reorientation_fraction"
                    ],
                }
            },
        },
    )
    revised_path = _spec(tmp_path, 2)
    revised = json.loads(revised_path.read_text(encoding="utf-8"))
    revised["parameters"]["settle_steps"] = 40
    revised_path.write_text(json.dumps(revised), encoding="utf-8")
    with pytest.raises(ValueError, match="not observed at the failed stage"):
        submit_program_request(session_path, revised_path)
    assert len(read_jsonl(request_path)) == 1


def test_queue_accepts_new_python_controller_with_unchanged_spec(tmp_path):
    session_path, _, receipt_path = _session(tmp_path)
    spec = _spec(tmp_path, 1)
    first = submit_program_request(session_path, spec)
    first_spec = load_program_spec(first["program_spec_json"])
    append_jsonl(
        receipt_path,
        {
            "type": "attempt",
            "request_id": first["request_id"],
            "program_spec": first_spec.receipt(),
            "controller_plugin": {
                "path": first["controller_plugin_py"],
                "sha256": first["controller_plugin_sha256"],
            },
            "diagnostic_classification": "diagnosed_physics_failure",
            "failed_stage": "bimanual_handle_grasp",
            "failed_stage_program_parameter_observations": {},
        },
    )
    revised_controller = tmp_path / "revised_controller.py"
    revised_controller.write_text(
        DEFAULT_CONTROLLER.read_text(encoding="utf-8") + "\nREVISION = 2\n",
        encoding="utf-8",
    )
    second = submit_program_request(
        session_path,
        first["program_spec_json"],
        controller_plugin_py=revised_controller,
    )
    assert second["program_spec_sha256"] == first["program_spec_sha256"]
    assert second["controller_plugin_sha256"] != first["controller_plugin_sha256"]


def test_source_first_queue_requires_latest_earliest_stage_proposal(tmp_path):
    session_path, _, receipt_path = _session(tmp_path)
    session = _enable_source_first_policy(session_path, tmp_path)
    spec = _spec(tmp_path, 1)
    first = submit_program_request(session_path, spec)
    _ack_failed_grasp(receipt_path, first)
    controller = tmp_path / "controller.py"
    controller.write_text(DEFAULT_CONTROLLER.read_text() + "\nREVISION=1\n")

    with pytest.raises(ValueError, match="requires a repair proposal"):
        submit_program_request(
            session_path, first["program_spec_json"], controller_plugin_py=controller
        )
    wrong = _proposal(
        tmp_path / "wrong.json",
        session,
        first["request_id"],
        "transport-v1",
        family="zero_jump_transport",
        target="smooth_bimanual_transport",
    )
    with pytest.raises(ValueError, match="earliest-failure rule"):
        submit_program_request(
            session_path,
            first["program_spec_json"],
            controller_plugin_py=controller,
            repair_proposal_json=wrong,
        )


def test_source_first_queue_allows_one_rollout_per_mechanism_and_exhausts_family(tmp_path):
    session_path, _, receipt_path = _session(tmp_path)
    session = _enable_source_first_policy(session_path, tmp_path)
    spec = _spec(tmp_path, 1)
    first = submit_program_request(session_path, spec)
    _ack_failed_grasp(receipt_path, first)

    controllers = []
    for index in (1, 2):
        controller = tmp_path / f"controller_{index}.py"
        controller.write_text(
            DEFAULT_CONTROLLER.read_text() + f"\nREVISION={index}\n"
        )
        controllers.append(controller)
    proposal_one = _proposal(
        tmp_path / "proposal_one.json",
        session,
        first["request_id"],
        "staged-clearance-v1",
    )
    second = submit_program_request(
        session_path,
        first["program_spec_json"],
        controller_plugin_py=controllers[0],
        repair_proposal_json=proposal_one,
    )
    _ack_failed_grasp(receipt_path, second)

    reused = _proposal(
        tmp_path / "reused.json",
        session,
        second["request_id"],
        "staged-clearance-v1",
    )
    with pytest.raises(ValueError, match="one rollout"):
        submit_program_request(
            session_path,
            second["program_spec_json"],
            controller_plugin_py=controllers[1],
            repair_proposal_json=reused,
        )

    proposal_two = _proposal(
        tmp_path / "proposal_two.json",
        session,
        second["request_id"],
        "staged-backoff-v2",
    )
    third = submit_program_request(
        session_path,
        second["program_spec_json"],
        controller_plugin_py=controllers[1],
        repair_proposal_json=proposal_two,
    )
    _ack_failed_grasp(receipt_path, third)
    proposal_three = _proposal(
        tmp_path / "proposal_three.json",
        session,
        third["request_id"],
        "staged-sweep-v3",
    )
    controller_three = tmp_path / "controller_3.py"
    controller_three.write_text(DEFAULT_CONTROLLER.read_text() + "\nREVISION=3\n")
    with pytest.raises(ValueError, match="family is exhausted"):
        submit_program_request(
            session_path,
            third["program_spec_json"],
            controller_plugin_py=controller_three,
            repair_proposal_json=proposal_three,
        )
