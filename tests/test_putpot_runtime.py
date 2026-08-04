import pytest

import judo_isaaclab.shutdown_monitor as shutdown_monitor

from judo_isaaclab.putpot_runtime import (
    AttemptIdentity,
    PHASE_NAMES,
    PhaseTimers,
    diagnostic_classification,
    ensure_fresh_output_paths,
    full_render_required_for_merge,
    render_recommendation,
    without_scene_camera_sensors,
)


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
    with pytest.raises(ValueError, match="exactly four"):
        AttemptIdentity(38, "epoch-20260804-a", 1, 3)


def test_phase_timers_expose_every_required_phase():
    timers = PhaseTimers()
    timers.add("reset", 0.25)
    receipt = timers.receipt()
    assert tuple(receipt) == PHASE_NAMES
    assert receipt["reset"] == pytest.approx(0.25)
    assert receipt["shutdown"] == pytest.approx(0.0)


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
    shutdown_monitor.main(
        [
            "--pid",
            "999999999",
            "--started-monotonic",
            str(__import__("time").monotonic()),
            "--receipt-json",
            str(receipt),
            "--payload-json",
            '{"phase_timings_s":{"shutdown":0.0}}',
        ]
    )
    value = __import__("json").loads(receipt.read_text(encoding="utf-8"))
    assert value["shutdown"]["completion"] == "process_exit_observed"
    assert value["phase_timings_s"]["shutdown"] >= 0.0
