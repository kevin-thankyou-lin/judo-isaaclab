import json
from pathlib import Path

import numpy as np
import pytest

from judo_isaaclab.putpot_queue import (
    static_worker_argv,
    submit_program_request,
)
from judo_isaaclab.putpot_program_spec import load_program_spec
from judo_isaaclab.putpot_runtime import append_jsonl, read_jsonl
from judo_isaaclab.putpot_repair_policy import (
    DEFAULT_POLICY,
    sha256_file,
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
    session["static_argv"].append("--render")
    session_path.write_text(json.dumps(session), encoding="utf-8")
    return session


def _failed_grasp_result(path, video_path=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = path.with_name("skill_trace.npz")
    actions = np.zeros((4, 14), dtype=np.float32)
    pot = np.zeros((4, 7), dtype=np.float32)
    pot[:, 3] = 1.0
    forces = np.zeros((4, 2), dtype=np.float32)
    fractions = np.full((4, 2), np.nan, dtype=np.float32)
    np.savez(
        trace_path,
        actions=actions,
        pot_poses=pot,
        left_finger_forces_n=forces,
        right_finger_forces_n=forces,
        left_pad_fractions=fractions,
        right_pad_fractions=fractions,
    )
    video_path = Path(video_path or path.with_name("skill.mp4"))
    video_path.write_bytes(b"physical-h264")
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
                "provenance": {
                    "trace": {"path": str(trace_path), "sha256": sha256_file(trace_path)}
                },
                "video": {
                    "path": str(video_path),
                    "sha256": sha256_file(video_path),
                    "codec": "h264",
                    "frame_count": 4,
                    "full_decode_returncode": 0,
                },
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
    _failed_grasp_result(result_path, request.get("video"))
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


def _diagnostic_receipt(
    tmp_path,
    request,
    *,
    action_delta=0.0,
    codec="h264",
    fully_decoded=True,
    consuming=False,
):
    physical_result_path = Path(request["result_json"])
    physical_result = json.loads(physical_result_path.read_text())
    physical_trace = Path(physical_result["provenance"]["trace"]["path"])
    physical_video = Path(physical_result["video"]["path"])
    diagnostic_root = tmp_path / f"diagnostic_{request['repair_epoch_attempt']}"
    diagnostic_root.mkdir(exist_ok=True)
    diagnostic_trace = diagnostic_root / "trace.npz"
    with np.load(physical_trace, allow_pickle=False) as trace:
        actions = np.asarray(trace["actions"]).copy()
    actions[0, 0] += action_delta
    np.savez(diagnostic_trace, actions=actions)
    diagnostic_video = diagnostic_root / "axes.mp4"
    diagnostic_video.write_bytes(b"diagnostic-h264")
    diagnostic_result_path = diagnostic_root / "result.json"
    action_bytes = __import__("hashlib").sha256(actions.tobytes()).hexdigest()
    with np.load(physical_trace, allow_pickle=False) as trace:
        reference = np.asarray(trace["actions"])
    reference_bytes = __import__("hashlib").sha256(reference.tobytes()).hexdigest()
    exact = bool(np.array_equal(reference, actions))
    diagnostic_result = {
        "status": "passed",
        "checks": {"diagnostic_actions_identical": exact},
        "provenance": {"trace": {"path": str(diagnostic_trace)}},
        "protocol": {
            "attempt_identity": None,
            "render_diagnostic": {
                "physics_or_controller_changes": False,
                "causal_mechanism_attempt_consumed": consuming,
                "training_eligible": False,
                "replay_actions": {"exactly_equal": exact},
                "overlay": {
                    "target_left_contact_frame": True,
                    "target_tangent_axis": "local_x",
                    "actual_left_pad_centers": 2,
                    "actual_left_pad_axes": 2,
                    "jaw_closing_line": True,
                    "target_left_wrist_frame": True,
                    "actual_left_wrist_frame": True,
                    "actual_to_desired_correction_vector": True,
                    "jaw_midpoint_to_target_contact_correction_vector": True,
                    "screen_space_color_legend": True,
                },
            },
        },
        "video": {
            "path": str(diagnostic_video),
            "codec": codec,
            "frame_count": 4,
            "full_decode_returncode": 0 if fully_decoded else 1,
        },
    }
    diagnostic_result_path.write_text(json.dumps(diagnostic_result))

    def artifact(path):
        return {"path": str(path), "sha256": sha256_file(path)}

    receipt = {
        "schema_version": 1,
        "classification": "action_identical_render_diagnostic",
        "physical_attempt": {
            "request_id": request["request_id"],
            "trace": artifact(physical_trace),
            "video": artifact(physical_video),
        },
        "diagnostic": {
            "result": artifact(diagnostic_result_path),
            "trace": artifact(diagnostic_trace),
            "video": artifact(diagnostic_video),
        },
        "action_parity": {
            "reference_shape": list(reference.shape),
            "replay_shape": list(actions.shape),
            "reference_actions_bytes_sha256": reference_bytes,
            "replay_actions_bytes_sha256": action_bytes,
            "exactly_equal": exact,
            "maximum_absolute_difference": float(np.max(np.abs(reference - actions))),
        },
        "protocol": {
            "physics_or_controller_changes": False,
            "causal_mechanism_attempt_consumed": consuming,
            "training_eligible": False,
            "attempt_identity": None,
        },
        "overlay": {
            "target_handle_contact_frame": True,
            "target_handle_tangent_axis": True,
            "actual_pad_centers": True,
            "actual_pad_axes": True,
            "jaw_closing_line": True,
            "target_wrist_frame": True,
            "actual_wrist_frame": True,
            "signed_correction_vectors": True,
            "screen_space_color_legend": True,
        },
        "measured_residuals": {
            "signed_translation_residual_world_m": [0.001, -0.002, 0.003],
            "translation_norm_m": 0.0037,
            "signed_rotation_residual_axis_angle_deg": [1.0, -2.0, 3.0],
            "rotation_norm_deg": 3.74,
        },
        "next_mechanism_prediction": {
            "mechanism_id": "structural-contact-frame-v2",
            "signed_translation_mm": [1.0, -2.0, 3.0],
            "signed_rotation_axis_angle_deg": [1.0, -2.0, 3.0],
            "sign_basis": "measured actual-to-target wrist frame",
        },
    }
    receipt_path = diagnostic_root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    return receipt_path


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
    diagnostic = _diagnostic_receipt(tmp_path, first)
    controller = tmp_path / "controller.py"
    controller.write_text(DEFAULT_CONTROLLER.read_text() + "\nREVISION=1\n")

    with pytest.raises(ValueError, match="requires an action-identical diagnostic"):
        submit_program_request(
            session_path, first["program_spec_json"], controller_plugin_py=controller
        )
    with pytest.raises(ValueError, match="requires a repair proposal"):
        submit_program_request(
            session_path,
            first["program_spec_json"],
            controller_plugin_py=controller,
            diagnostic_receipt_json=diagnostic,
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
            diagnostic_receipt_json=diagnostic,
        )


def test_source_first_queue_allows_one_rollout_per_mechanism_and_exhausts_family(tmp_path):
    session_path, _, receipt_path = _session(tmp_path)
    session = _enable_source_first_policy(session_path, tmp_path)
    spec = _spec(tmp_path, 1)
    first = submit_program_request(session_path, spec)
    _ack_failed_grasp(receipt_path, first)
    diagnostic_one = _diagnostic_receipt(tmp_path, first)

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
        diagnostic_receipt_json=diagnostic_one,
    )
    _ack_failed_grasp(receipt_path, second)
    diagnostic_two = _diagnostic_receipt(tmp_path, second)

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
            diagnostic_receipt_json=diagnostic_two,
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
        diagnostic_receipt_json=diagnostic_two,
    )
    _ack_failed_grasp(receipt_path, third)
    diagnostic_three = _diagnostic_receipt(tmp_path, third)
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
            diagnostic_receipt_json=diagnostic_three,
        )


@pytest.mark.parametrize(
    ("action_delta", "codec", "fully_decoded", "consuming", "error"),
    [
        (0.0, "h264", True, False, None),
        (0.01, "h264", True, False, "action parity gate failed"),
        (0.0, "vp9", True, False, "H.264 full-decode gate failed"),
        (0.0, "h264", False, False, "H.264 full-decode gate failed"),
        (0.0, "h264", True, True, "consuming/non-training protocol gate failed"),
    ],
)
def test_next_attempt_accepts_only_exact_decoded_nonconsuming_diagnostic(
    tmp_path, action_delta, codec, fully_decoded, consuming, error
):
    session_path, _, receipt_path = _session(tmp_path)
    session = _enable_source_first_policy(session_path, tmp_path)
    first = submit_program_request(session_path, _spec(tmp_path, 1))
    _ack_failed_grasp(receipt_path, first)
    diagnostic = _diagnostic_receipt(
        tmp_path,
        first,
        action_delta=action_delta,
        codec=codec,
        fully_decoded=fully_decoded,
        consuming=consuming,
    )
    proposal = _proposal(
        tmp_path / "proposal.json",
        session,
        first["request_id"],
        "structural-contact-frame-v2",
    )
    controller = tmp_path / "controller.py"
    controller.write_text(DEFAULT_CONTROLLER.read_text() + "\nREVISION=9\n")
    if error is not None:
        with pytest.raises(ValueError, match=error):
            submit_program_request(
                session_path,
                first["program_spec_json"],
                controller_plugin_py=controller,
                repair_proposal_json=proposal,
                diagnostic_receipt_json=diagnostic,
            )
        return
    second = submit_program_request(
        session_path,
        first["program_spec_json"],
        controller_plugin_py=controller,
        repair_proposal_json=proposal,
        diagnostic_receipt_json=diagnostic,
    )
    assert second["repair_epoch_attempt"] == 2
    assert second["prior_diagnostic_receipt_sha256"] == sha256_file(
        second["prior_diagnostic_receipt_json"]
    )
