import pytest

import judo_isaaclab.shutdown_monitor as shutdown_monitor

from judo_isaaclab.putpot_runtime import (
    AttemptIdentity,
    PHASE_NAMES,
    PhaseTimers,
    diagnostic_classification,
    append_jsonl,
    ensure_fresh_output_paths,
    full_render_required_for_merge,
    instantiated_scene_sensor_inventory,
    render_recommendation,
    read_jsonl,
    timing_accounting,
    failed_stage_program_parameter_observations,
    validate_material_spec_revision,
    validate_same_spec_retry,
    without_scene_camera_sensors,
)


def test_jsonl_reader_ignores_partial_trailing_append(tmp_path):
    path = tmp_path / "queue.jsonl"
    append_jsonl(path, {"request_id": "complete"})
    with open(path, "a", encoding="utf-8") as stream:
        stream.write('{"request_id":"partial"')

    assert read_jsonl(path) == [{"request_id": "complete"}]


def test_same_spec_retry_requires_ambiguous_receipt_and_explicit_reason():
    diagnosed = {
        "program_spec": {"sha256": "same"},
        "diagnostic_classification": "diagnosed_physics_failure",
    }
    with pytest.raises(ValueError, match="prior ambiguous_failure"):
        validate_same_spec_retry(diagnosed, "same", "looks stochastic")

    ambiguous = {
        "program_spec": {"sha256": "same"},
        "diagnostic_classification": "ambiguous_failure",
    }
    with pytest.raises(ValueError, match="explicit ambiguity reason"):
        validate_same_spec_retry(ambiguous, "same", None)
    validate_same_spec_retry(ambiguous, "same", "trace ended before first contact")
    validate_same_spec_retry(diagnosed, "different", None)


def test_changed_hash_requires_every_changed_parameter_at_failed_stage():
    previous = {
        "program_spec": {
            "sha256": "old",
            "parameters": {
                "receiving_jaw_close_horizon_steps": 0,
                "settle_steps": 35,
            },
        },
        "failed_stage": "bimanual_handle_grasp",
        "failed_stage_program_parameter_observations": {
            "receiving_jaw_close_horizon_steps": {
                "requested": 0,
                "applied_close_steps": 269,
            }
        },
    }
    validate_material_spec_revision(
        previous,
        "new",
        {"receiving_jaw_close_horizon_steps": 1, "settle_steps": 35},
    )
    with pytest.raises(ValueError, match="not observed at the failed stage"):
        validate_material_spec_revision(
            previous,
            "newer",
            {"receiving_jaw_close_horizon_steps": 1, "settle_steps": 40},
        )
    with pytest.raises(ValueError, match="no effective parameter change"):
        validate_material_spec_revision(
            previous,
            "format-only",
            {"receiving_jaw_close_horizon_steps": 0, "settle_steps": 35},
        )


def test_failed_grasp_receipt_observes_executed_close_horizon():
    parameters = {
        "damping": 0.045,
        "max_joint_delta": 0.16,
        "max_position_step": 0.025,
        "max_rotation_step": 0.16,
        "missing_finger_contact_limit_m": 0.012,
        "receiving_jaw_center_translation_fraction": 0.0,
        "receiving_jaw_reorientation_fraction": 0.45,
        "receiving_jaw_close_horizon_steps": 1,
    }
    result = {
        "checks": {"bimanual_pick_observed": False},
        "protocol": {
            "parameters": dict(parameters),
            "peer_contact_gripper_retime": {
                "requested_close_horizon_steps": 1,
                "applied_close_steps": 1,
                "close_start_step": 198,
                "close_end_step": 199,
                "grasp_end_step": 467,
            },
        },
    }
    stage, observed = failed_stage_program_parameter_observations(
        result, parameters
    )
    assert stage == "bimanual_handle_grasp"
    assert observed["receiving_jaw_close_horizon_steps"] == {
        "requested": 1,
        "applied_close_steps": 1,
        "close_start_step": 198,
        "close_end_step": 199,
        "grasp_end_step": 467,
    }


def test_attempt_identity_separates_lifetime_and_four_attempt_epoch():
    identity = AttemptIdentity(37, "epoch-20260804-a", 4)
    assert identity.receipt() == {
        "lifetime_attempt": 37,
        "repair_epoch": "epoch-20260804-a",
        "repair_epoch_attempt": 4,
        "repair_epoch_attempt_limit": 4,
    }
    with pytest.raises(ValueError, match="exceeds"):
        AttemptIdentity(38, "epoch-20260804-a", 5)
    assert AttemptIdentity(38, "epoch-20260804-a", 1, 3).receipt()[
        "repair_epoch_attempt_limit"
    ] == 3
    with pytest.raises(ValueError, match="capped at four"):
        AttemptIdentity(38, "epoch-20260804-a", 1, 5)


def test_phase_timers_expose_every_required_phase():
    timers = PhaseTimers()
    timers.add("reset", 0.25)
    receipt = timers.receipt()
    assert tuple(receipt) == PHASE_NAMES
    assert receipt["reset"] == pytest.approx(0.25)
    assert receipt["shutdown"] == pytest.approx(0.0)


def test_timing_accounting_exposes_unattributed_wall_time():
    phases = {name: 0.0 for name in PHASE_NAMES}
    phases["app_startup"] = 10.0
    phases["rollout"] = 30.0
    assert timing_accounting(62.0, phases) == {
        "attempt_wall_time_s": 62.0,
        "named_phase_sum_s": 40.0,
        "unattributed_time_s": 22.0,
    }


def test_instantiated_scene_sensor_inventory_uses_runtime_sensor_instances():
    class ContactSensor:
        pass

    class Camera:
        pass

    class Scene:
        sensors = {
            "pot_contact": ContactSensor(),
            "top_camera": Camera(),
        }

    assert instantiated_scene_sensor_inventory(Scene()) == {
        "instantiated_scene_sensor_names": ["pot_contact", "top_camera"],
        "instantiated_scene_sensor_count": 2,
        "instantiated_scene_camera_sensor_names": ["top_camera"],
        "instantiated_scene_camera_sensor_count": 1,
    }

    Scene.sensors = {"pot_contact": ContactSensor()}
    inventory = instantiated_scene_sensor_inventory(Scene())
    assert inventory["instantiated_scene_camera_sensor_names"] == []
    assert inventory["instantiated_scene_camera_sensor_count"] == 0


def test_fresh_output_guard_refuses_overwrite(tmp_path):
    existing = tmp_path / "skill.log"
    existing.write_text("preserve")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ensure_fresh_output_paths([existing])
    ensure_fresh_output_paths([tmp_path / "new.json", None])


def test_render_recommendation_is_narrow_and_merge_requires_full_video():
    candidate = {
        "checks": {
            "coded_task_success": True,
            "bimanual_transport_completed": True,
            "h264_nonempty": False,
            "fully_decodable": False,
        },
        "stage_success_trace": [{"step": 1}],
        "provenance": {"trace": {"path": "trace.npz"}},
    }
    assert render_recommendation(candidate) == "final_acceptance_candidate"
    assert not full_render_required_for_merge(candidate)

    diagnosed = {
        **candidate,
        "checks": {**candidate["checks"], "bimanual_transport_completed": False},
    }
    assert render_recommendation(diagnosed) is None
    assert render_recommendation(None) == "ambiguous_failure"

    rendered = {
        **candidate,
        "video": {"path": "skill.mp4"},
        "checks": {
            **candidate["checks"],
            "h264_nonempty": True,
            "fully_decodable": True,
        },
    }
    assert full_render_required_for_merge(rendered)


def test_deterministic_controller_exception_never_recommends_render():
    error = "ValueError: smooth transport requires at least eight steps"
    assert (
        diagnostic_classification(None, error)
        == "deterministic_controller_or_config_exception"
    )
    assert render_recommendation(None, error) is None


def test_camera_free_scene_policy_removes_sensors_and_restores_builder():
    class Scene:
        def build_from_spec(self, spec):
            for name in spec.camera_names:
                setattr(self, f"{name}_camera", object())
            return "built"

    class Spec:
        camera_names = ("top", "left_wrist")

    original = Scene.build_from_spec
    scene = Scene()
    with without_scene_camera_sensors(Scene):
        assert scene.build_from_spec(Spec()) == "built"
        assert scene.top_camera is None
        assert scene.left_wrist_camera is None
    assert Scene.build_from_spec is original
    fresh = Scene()
    fresh.build_from_spec(Spec())
    assert fresh.top_camera is not None


def test_shutdown_monitor_writes_completion_for_gone_process(tmp_path):
    receipt = tmp_path / "receipt.jsonl"
    shutdown_monitor.main(
        [
            "--pid",
            "999999999",
            "--started-monotonic",
            "0",
            "--receipt-jsonl",
            str(receipt),
            "--payload-json",
            '{"type":"worker_summary","shutdown":{"attempts":2}}',
        ]
    )
    value = __import__("json").loads(receipt.read_text(encoding="utf-8"))
    assert value["shutdown"]["completion"] == "process_exit_observed"
    assert value["shutdown"]["attempts"] == 2
    assert value["shutdown"]["shutdown_s"] >= 0.0


def test_shutdown_monitor_atomically_writes_phase_timing_json(tmp_path):
    receipt = tmp_path / "runtime.json"
    started = __import__("time").monotonic() - 2.0
    shutdown_monitor.main(
        [
            "--pid",
            "999999999",
            "--started-monotonic",
            str(__import__("time").monotonic()),
            "--receipt-json",
            str(receipt),
            "--payload-json",
            __import__("json").dumps(
                {
                    "phase_timings_s": {"shutdown": 0.0},
                    "attempt_wall_started_monotonic": started,
                }
            ),
        ]
    )
    value = __import__("json").loads(receipt.read_text(encoding="utf-8"))
    assert value["shutdown"]["completion"] == "process_exit_observed"
    assert value["phase_timings_s"]["shutdown"] >= 0.0
    assert value["timing_accounting"]["attempt_wall_time_s"] >= 2.0
    assert value["timing_accounting"]["named_phase_sum_s"] >= 0.0
    assert value["timing_accounting"]["unattributed_time_s"] >= 1.9
