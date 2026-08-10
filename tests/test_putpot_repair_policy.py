import json
from pathlib import Path

import numpy as np
import pytest

from judo_isaaclab.putpot_repair_policy import (
    DEFAULT_POLICY,
    build_source_demo_card,
    load_repair_proposal,
    material_progress,
    repair_evidence,
    trace_latch_evidence,
    training_data_route,
)


def _frame(index, *, left=False, right=False, pot_x=0.0):
    return {
        "sample_index": index,
        "action_index": max(0, index - 1),
        "pot_pose": [pot_x, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
        "cooktop_pose": [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
        "left_eef_pose": [0.0, 0.2, 0.9, 1.0, 0.0, 0.0, 0.0],
        "right_eef_pose": [0.0, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0],
        "left_grasp": left,
        "right_grasp": right,
        "stage1": False,
        "stage2": False,
    }


def _keyframes():
    values = {
        "left_pregrasp": _frame(10),
        "left_handle_grasp": _frame(20, left=True),
        "right_pregrasp": _frame(30, left=True),
        "right_handle_grasp": _frame(40, left=True, right=True),
        "pot_lift": _frame(50, left=True, right=True, pot_x=0.05),
        "pot_transport": _frame(60, left=True, right=True, pot_x=0.20),
        "support_align": _frame(70, right=True, pot_x=0.30),
        "pot_release": _frame(80, pot_x=0.30),
        "stable_settle": _frame(90, pot_x=0.30),
    }
    return {
        "schema_version": 1,
        "source_dataset": "/data/putpot_000.hdf5",
        "source_dataset_sha256": "a" * 64,
        "source_assets": {"pot": {}, "cooktop": {}},
        "frames": values,
    }


def _trace(path: Path, *, robust_frames=20, motion=0.001, fraction=0.5):
    count = 40
    pot = np.zeros((count, 7), dtype=np.float32)
    pot[:, 3] = 1.0
    pot[5:11, 0] = np.linspace(0.0, motion, 6)
    pot[11:, 0] = motion
    left_forces = np.zeros((count, 2), dtype=np.float32)
    right_forces = np.zeros((count, 2), dtype=np.float32)
    left_fractions = np.full((count, 2), np.nan, dtype=np.float32)
    right_fractions = np.full((count, 2), np.nan, dtype=np.float32)
    left_forces[5:, :] = 5.0
    left_fractions[5:, :] = fraction
    start = count - robust_frames
    right_forces[start:, :] = 5.0
    right_fractions[start:, :] = fraction
    np.savez(
        path,
        pot_poses=pot,
        left_finger_forces_n=left_forces,
        right_finger_forces_n=right_forces,
        left_pad_fractions=left_fractions,
        right_pad_fractions=right_fractions,
    )


def _result(trace: Path, *, pick=True, transport=False, path=0.0, center=0.4):
    return {
        "status": "failed",
        "checks": {
            "bimanual_pick_observed": pick,
            "bimanual_transport_completed": transport,
            "centered_on_cooktop": False,
            "coded_task_success": False,
            "pot_released": False,
            "stable_support_window": False,
            "accepted_task_success": False,
            "h264_nonempty": True,
            "fully_decodable": True,
        },
        "metrics": {
            "center_error_m": center,
            "transport_executed": {"path_length_m": path},
        },
        "provenance": {"trace": {"path": str(trace)}},
    }


def test_source_demo_card_preserves_strategy_and_object_relative_frames():
    card = build_source_demo_card(_keyframes())

    assert card["contact_order"] == ["left", "right"]
    assert card["stage_sequence"][0] == "staged_bilateral_acquisition"
    assert card["transport_contract"] == {
        "frame": "observed_pot_pose",
        "preserve_loaded_object_local_grasp_transforms": True,
        "zero_jump_at_handoff": True,
    }
    assert len(card["semantic_frames"]["left_handle_grasp"]["left_eef_in_pot"]) == 7


def test_trace_latch_requires_force_fraction_stability_and_low_pre_peer_motion(tmp_path):
    good = tmp_path / "good.npz"
    _trace(good, robust_frames=20, motion=0.001, fraction=0.5)
    evidence = trace_latch_evidence(good)

    assert evidence.passes_robust_latch
    assert evidence.longest_robust_bilateral_frames == 20
    assert evidence.pre_peer_object_motion_m == pytest.approx(0.001)
    assert evidence.robust_window_minimum_force_n == pytest.approx(5.0)
    assert evidence.robust_window_minimum_pad_fraction_margin == pytest.approx(0.5)

    edge = tmp_path / "edge.npz"
    _trace(edge, robust_frames=20, motion=0.001, fraction=0.02)
    assert not trace_latch_evidence(edge).passes_robust_latch


def test_earliest_failure_returns_to_grasp_when_transport_latch_is_marginal(tmp_path):
    trace = tmp_path / "marginal.npz"
    _trace(trace, robust_frames=10)
    evidence = repair_evidence(_result(trace, pick=True, transport=False))

    assert evidence["reported_failed_stage"] == "smooth_bimanual_transport"
    assert evidence["earliest_failed_stage"] == "bimanual_handle_grasp"


def test_material_progress_uses_robust_margins_and_task_scale_delta(tmp_path):
    before_trace = tmp_path / "before.npz"
    after_trace = tmp_path / "after.npz"
    _trace(before_trace, robust_frames=20)
    _trace(after_trace, robust_frames=20)
    before = _result(before_trace, path=0.10)
    after = _result(after_trace, path=0.16)

    improved, reason = material_progress(before, after)

    assert improved
    assert "transport_path_m" in reason


def test_training_route_separates_strict_demos_failures_and_bad_media(tmp_path):
    trace = tmp_path / "trace.npz"
    _trace(trace, robust_frames=20)
    failure = _result(trace)
    assert training_data_route(failure) == "failure_or_critic"
    failure["checks"]["fully_decodable"] = False
    assert training_data_route(failure) == "reject_artifact"


def test_repair_proposal_rejects_pair_specific_unregistered_family(tmp_path):
    proposal = {
        "schema_version": 1,
        "source_demo_card_sha256": "a" * 64,
        "baseline_request_id": "epoch:1",
        "target_stage": "bimanual_handle_grasp",
        "mechanism_id": "measured-depth-v1",
        "repair_family": "pair_037_custom_patch",
        "hypothesis": "the pads are shallow",
        "expected_task_delta": "robust latch",
        "changed_primitives": ["grasp_depth"],
    }
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(proposal), encoding="utf-8")

    with pytest.raises(ValueError, match="shared repair library"):
        load_repair_proposal(path)
