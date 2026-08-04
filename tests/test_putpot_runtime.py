import pytest

from judo_isaaclab.putpot_runtime import (
    AttemptIdentity,
    PHASE_NAMES,
    PhaseTimers,
    ensure_fresh_output_paths,
    full_render_required_for_merge,
    render_recommendation,
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
