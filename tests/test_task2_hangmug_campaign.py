import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

import run_task2_hangmug_campaign as campaign


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_same_index_assets_and_proven_command_are_pinned(tmp_path, monkeypatch):
    objects = tmp_path / "objects"
    for kind, name in (
        ("MugHangable", "mug_teacup_000002"),
        ("ThreeLayerMugTree", "mug_tree_000002"),
    ):
        (objects / kind / name).mkdir(parents=True)
    monkeypatch.setattr(campaign, "OBJECTS", objects)
    monkeypatch.setattr(campaign, "_steady_state_seconds", lambda: 211)
    attempt = Path("results/task2/pairs/000002/attempt_001_direct_source_classification")
    command = campaign._classification_command(2, attempt)
    assert str(objects / "MugHangable/mug_teacup_000002") in command
    assert str(objects / "ThreeLayerMugTree/mug_tree_000002") in command
    assert command.count(str(campaign.SOURCE)) == 1
    assert "processed_actions" not in command
    assert "--classification-run" in command
    assert command[command.index("--mode") + 1] == "replay"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--expected-controller-gains-sha256") + 1] == campaign.CONTROLLER_SHA256
    for name in campaign.PROVEN_CONTROL_DEFAULTS:
        assert f"--{name.replace('_', '-')}" not in command


def test_evidence_runner_disables_unused_environment_hdf5_recorder():
    runner = Path(campaign.__file__).with_name("run_hangmug_skill_program.py")
    source = runner.read_text()

    assert "disable_env_recorders=True" in source


def test_missing_or_cross_index_asset_fails_before_command(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "OBJECTS", tmp_path)
    (tmp_path / "MugHangable/mug_teacup_000003").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="000003"):
        campaign._asset_pair(3)


def test_atomic_immutable_receipt_never_overwrites(tmp_path):
    path = tmp_path / "manifest.json"
    campaign._atomic_json(path, {"value": 1}, immutable=True)
    with pytest.raises(FileExistsError):
        campaign._atomic_json(path, {"value": 2}, immutable=True)
    assert json.loads(path.read_text()) == {"value": 1}


def test_asset_provenance_matches_lane_symlink_to_canonical_assets(tmp_path):
    canonical = tmp_path / "canonical"
    mug = canonical / "MugHangable/mug_teacup_000010"
    tree = canonical / "ThreeLayerMugTree/mug_tree_000010"
    mug.mkdir(parents=True)
    tree.mkdir(parents=True)
    alias = tmp_path / "lane-data"
    alias.symlink_to(canonical, target_is_directory=True)
    expected = {
        "mug": alias / "MugHangable/mug_teacup_000010",
        "mug_tree": alias / "ThreeLayerMugTree/mug_tree_000010",
    }
    recorded = {
        "mug": {"path": str(mug)},
        "mug_tree": {"path": str(tree)},
    }
    assert campaign._asset_provenance_matches(recorded, expected)


def test_asset_provenance_rejects_wrong_or_incomplete_assets(tmp_path):
    mug = tmp_path / "MugHangable/mug_teacup_000010"
    tree = tmp_path / "ThreeLayerMugTree/mug_tree_000010"
    wrong = tmp_path / "ThreeLayerMugTree/mug_tree_000011"
    mug.mkdir(parents=True)
    tree.mkdir(parents=True)
    wrong.mkdir(parents=True)
    expected = {"mug": mug, "mug_tree": tree}
    assert not campaign._asset_provenance_matches(
        {"mug": {"path": str(mug)}}, expected
    )
    assert not campaign._asset_provenance_matches(
        {
            "mug": {"path": str(mug)},
            "mug_tree": {"path": str(wrong)},
        },
        expected,
    )


def test_explicit_lane_claim_is_pair_specific_and_immutable(tmp_path, monkeypatch):
    results = tmp_path / "task2"
    results.mkdir()
    monkeypatch.setattr(campaign, "RESULTS", results)
    monkeypatch.setattr(
        campaign.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "f623e360\n",
    )

    path = campaign._claim_explicit_lane(12, "node1-gpu3")
    receipt = json.loads(path.read_text())
    assert receipt == {
        "schema_version": 1,
        "pair_index": 12,
        "human_pair": 13,
        "lane_id": "node1-gpu3",
        "judo_head": "f623e360",
        "results_root": str(results.resolve()),
    }
    assert campaign._claim_explicit_lane(12, "node1-gpu3") == path
    with pytest.raises(RuntimeError, match="assignment changed"):
        campaign._claim_explicit_lane(12, "node2-gpu0")


def test_explicit_lane_claim_allows_descendant_commit_without_rewriting_receipt(
    tmp_path, monkeypatch,
):
    results = tmp_path / "task2"
    results.mkdir()
    monkeypatch.setattr(campaign, "RESULTS", results)
    heads = iter(("base-head\n", "repair-head\n"))
    monkeypatch.setattr(
        campaign.subprocess,
        "check_output",
        lambda *_args, **_kwargs: next(heads),
    )
    ancestry_calls = []

    def run(command, **kwargs):
        ancestry_calls.append((command, kwargs))
        return campaign.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(campaign.subprocess, "run", run)
    path = campaign._claim_explicit_lane(12, "node1-gpu3")
    original = path.read_bytes()
    assert campaign._claim_explicit_lane(12, "node1-gpu3") == path
    assert path.read_bytes() == original
    assert ancestry_calls[0][0] == [
        "git", "merge-base", "--is-ancestor", "base-head", "repair-head",
    ]


def test_explicit_lane_claim_rejects_non_descendant_commit(tmp_path, monkeypatch):
    results = tmp_path / "task2"
    results.mkdir()
    monkeypatch.setattr(campaign, "RESULTS", results)
    heads = iter(("base-head\n", "unrelated-head\n"))
    monkeypatch.setattr(
        campaign.subprocess,
        "check_output",
        lambda *_args, **_kwargs: next(heads),
    )
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda command, **_kwargs: campaign.subprocess.CompletedProcess(command, 1),
    )
    campaign._claim_explicit_lane(12, "node1-gpu3")
    with pytest.raises(RuntimeError, match="assignment changed"):
        campaign._claim_explicit_lane(12, "node1-gpu3")


@pytest.mark.parametrize("lane", ("", "node/gpu", "node gpu"))
def test_explicit_lane_claim_rejects_ambiguous_ids(tmp_path, monkeypatch, lane):
    monkeypatch.setattr(campaign, "RESULTS", tmp_path)
    with pytest.raises(ValueError, match="lane ID"):
        campaign._claim_explicit_lane(12, lane)


def test_resume_skips_only_hash_valid_contiguous_acceptances(tmp_path, monkeypatch):
    results = tmp_path / "task2"
    source = results / "source"
    source.mkdir(parents=True)
    artifacts = {}
    for name in ("result", "video", "demonstration"):
        path = source / name
        path.write_text(name)
        artifacts[name] = {"path": str(path), "sha256": _digest(path)}
    (source / "accepted_source.json").write_text(json.dumps({"artifacts": artifacts}))
    pair = results / "pairs/000001/attempt_004"
    pair.mkdir(parents=True)
    files = {
        "result_sha256": pair / "result.json",
        "video_sha256": pair / "skill.mp4",
        "demonstration_sha256": pair / "demo.hdf5",
        "independent_audit_sha256": pair / "independent_audit.json",
    }
    for path in files.values():
        path.write_text(path.name)
    ledger = {"pairs": {
        "000000": {"status": "accepted", **{name: artifacts[name.removesuffix("_sha256")]["sha256"] for name in ("result_sha256", "video_sha256", "demonstration_sha256")}},
        "000001": {"status": "accepted", "attempt": "attempt_004", **{name: _digest(path) for name, path in files.items()}},
    }}
    monkeypatch.setattr(campaign, "RESULTS", results)
    assert campaign._first_missing(ledger) == 2
    ledger["pairs"]["000001"]["video_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="ledger/artifact"):
        campaign._first_missing(ledger)


def test_resume_rejects_accepted_gap(monkeypatch):
    monkeypatch.setattr(campaign, "_validate_accepted", lambda *_: None)
    ledger = {"pairs": {
        "000000": {"status": "accepted"},
        "000002": {"status": "accepted"},
    }}
    with pytest.raises(RuntimeError, match="past missing Pair 000001"):
        campaign._first_missing(ledger)


def test_atomic_accept_requires_unchanged_ledger_and_writes_only_acceptance(tmp_path, monkeypatch):
    results = tmp_path / "task2"
    results.mkdir()
    ledger_path = results / "ledger.json"
    ledger_path.write_text(json.dumps({"pairs": {"000000": {"status": "accepted"}}}))
    attempt = results / "pairs/000001/attempt_001"
    attempt.mkdir(parents=True)
    (attempt / "independent_audit.json").write_text("audit")
    monkeypatch.setattr(campaign, "RESULTS", results)
    before = _digest(ledger_path)
    audit = {"artifact_hashes": {
        "result_sha256": "r", "video_sha256": "v", "demo_hdf5_sha256": "d"
    }}
    after = campaign._accept(1, attempt, audit, before)
    assert after == _digest(ledger_path)
    assert json.loads(ledger_path.read_text())["pairs"]["000001"] == {
        "status": "accepted",
        "mug": "MugHangable/mug_teacup_000001",
        "mug_tree": "ThreeLayerMugTree/mug_tree_000001",
        "attempt": "attempt_001",
        "result_sha256": "r",
        "video_sha256": "v",
        "demonstration_sha256": "d",
        "independent_audit_sha256": _digest(attempt / "independent_audit.json"),
    }
    with pytest.raises(RuntimeError, match="ledger changed"):
        campaign._accept(2, attempt, audit, before)


def test_atomic_requalification_preserves_superseded_acceptance(tmp_path, monkeypatch):
    results = tmp_path / "task2"
    results.mkdir()
    old = {
        "status": "accepted", "attempt": "attempt_001",
        "result_sha256": "old-result", "video_sha256": "old-video",
        "demonstration_sha256": "old-demo",
        "independent_audit_sha256": "old-audit",
    }
    ledger_path = results / "ledger.json"
    ledger_path.write_text(json.dumps({"pairs": {"000006": old}}))
    attempt = results / "pairs/000006/attempt_024"
    attempt.mkdir(parents=True)
    (attempt / "independent_audit.json").write_text("new-audit")
    monkeypatch.setattr(campaign, "RESULTS", results)
    audit = {"artifact_hashes": {
        "result_sha256": "new-result", "video_sha256": "new-video",
        "demo_hdf5_sha256": "new-demo",
    }}
    campaign._accept(
        6, attempt, audit, _digest(ledger_path), replace_existing=True,
    )
    replacement = json.loads(ledger_path.read_text())["pairs"]["000006"]
    assert replacement["attempt"] == "attempt_024"
    assert replacement["superseded_acceptances"] == [old]
    with pytest.raises(RuntimeError, match="no acceptance to supersede"):
        campaign._accept(
            7, attempt, audit, _digest(ledger_path), replace_existing=True,
        )


def test_first_true_is_fail_closed():
    assert campaign._first_true([False, True, True]) == 1
    assert campaign._first_true([False, False]) is None


def test_worker_gate_matches_runner_token_not_prompt_text(tmp_path, monkeypatch):
    proc = tmp_path / "123"
    proc.mkdir()
    (proc / "comm").write_text("python\n")
    (proc / "cmdline").write_bytes(b"python\0examples/run_hangmug_skill_program.py\0")
    monkeypatch.setattr(campaign.Path, "glob", lambda _self, _pattern: [proc])
    assert campaign._worker_pids() == [123]
    (proc / "cmdline").write_bytes(b"codex\0prompt mentions run_hangmug_skill_program.py\0")
    assert campaign._worker_pids() == []


def test_pick_failure_repair_cannot_use_exact_pick_prefix(tmp_path, monkeypatch):
    objects = tmp_path / "objects"
    for kind, name in (
        ("MugHangable", "mug_teacup_000002"),
        ("ThreeLayerMugTree", "mug_tree_000002"),
    ):
        (objects / kind / name).mkdir(parents=True)
    monkeypatch.setattr(campaign, "OBJECTS", objects)
    monkeypatch.setattr(campaign, "_steady_state_seconds", lambda: 211)
    pick = campaign._repair_selection("pick", None)
    command = campaign._repair_command(
        2, tmp_path / "repair", tmp_path / "classification/result.json", pick
    )
    assert "--direct-replay-result" in command
    assert "--reuse-source-pick-prefix" not in command
    assert command[command.index("--handover-confirm-steps") + 1] == "12"
    assert command[command.index("--handover-contact-settle-steps") + 1] == "30"
    handover_selection = campaign._repair_selection("handover", "pick")
    handover = campaign._repair_command(
        2, tmp_path / "repair2", tmp_path / "classification/result.json",
        handover_selection,
    )
    assert "--reuse-source-pick-prefix" in handover
    assert handover[handover.index("--handover-confirm-steps") + 1] == "12"
    assert "--handover-contact-settle-steps" not in handover


def test_pair_repair_candidate_is_bounded_and_pinned_in_command(tmp_path, monkeypatch):
    results = tmp_path / "task2"
    candidate = results / "pairs/000002/repair_candidate.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(json.dumps({
        "handover_contact_settle_steps": 30,
        "handover_contact_acquire_steps": 24,
        "handover_confirm_steps": 20,
        "handover_post_release_lift_m": 0.055,
        "handover_post_release_lift_steps": 30,
        "left_release_retreat_m": 0.03,
        "post_handover_right_return_steps": 45,
        "left_branch_point_steps": 35,
        "require_broad_pad_contact": True,
        "pick_lift_margin_m": 0.01,
        "handover_handle_frame_transfer": True,
        "handover_target_local_pitch_rad": 0.7853981633974483,
        "handover_orient_clearance_m": 0.08,
        "handover_orient_steps": 30,
        "handover_standoff_outside_m": 0.08,
        "handover_straddle_local_x_m": -0.124,
        "branch_orient_steps": 60,
        "insert_clearance_m": 0.04,
        "branch_approach_height_m": 0.0,
        "branch_roll_offset_rad": 0.5235987755982988,
        "branch_support_fraction": 0.75,
        "stable_support_steps": 180,
    }))
    monkeypatch.setattr(campaign, "RESULTS", results)
    strategy = campaign._repair_strategy(2)
    selection = campaign._repair_selection("pick", None)
    monkeypatch.setattr(campaign, "_common_workload", lambda *_: ["--device", "cpu"])
    monkeypatch.setattr(campaign, "_guarded", lambda _attempt, workload: workload)
    command = campaign._repair_command(
        2, tmp_path / "attempt", tmp_path / "result.json", selection, strategy
    )
    cursor = command.index("--left-release-retreat-m")
    assert float(command[cursor + 1]) == pytest.approx(0.03)
    assert command[command.index("--post-handover-right-return-steps") + 1] == "45"
    assert command[command.index("--left-branch-point-steps") + 1] == "35"
    assert "--require-broad-pad-contact" in command
    assert float(command[command.index("--pick-lift-margin-m") + 1]) == 0.01
    assert command[command.index("--handover-confirm-steps") + 1] == "20"
    assert command[command.index("--handover-contact-acquire-steps") + 1] == "24"
    assert float(command[command.index("--insert-clearance-m") + 1]) == 0.04
    assert float(command[command.index("--branch-approach-height-m") + 1]) == 0.0
    assert float(command[command.index("--branch-roll-offset-rad") + 1]) == pytest.approx(
        0.5235987755982988
    )
    assert float(command[command.index("--branch-support-fraction") + 1]) == 0.75
    assert command[command.index("--stable-support-steps") + 1] == "180"
    assert float(command[command.index("--handover-post-release-lift-m") + 1]) == 0.055
    assert command[command.index("--handover-post-release-lift-steps") + 1] == "30"
    assert "--handover-handle-frame-transfer" in command
    assert float(command[command.index("--handover-target-local-pitch-rad") + 1]) == pytest.approx(
        0.7853981633974483
    )
    assert float(command[command.index("--handover-orient-clearance-m") + 1]) == 0.08
    assert command[command.index("--handover-orient-steps") + 1] == "30"
    assert float(command[command.index("--handover-standoff-outside-m") + 1]) == 0.08
    assert float(command[command.index("--handover-straddle-local-x-m") + 1]) == -0.124
    assert command[command.index("--branch-orient-steps") + 1] == "60"
    assert "--target-branch-rank" not in command
    candidate.write_text(json.dumps({"handover_target_offset_m": [0.05, 0, 0]}))
    with pytest.raises(ValueError, match="within 4 cm"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"left_release_retreat_m": 0.01}))
    with pytest.raises(ValueError, match=r"\[0.02, 0.12\]"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"handover_confirm_steps": 61}))
    with pytest.raises(ValueError, match="confirmation"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({
        "handover_post_release_lift_m": 0.081,
        "handover_post_release_lift_steps": 30,
    }))
    with pytest.raises(ValueError, match="post-release lift"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"handover_post_release_lift_m": 0.055}))
    with pytest.raises(ValueError, match="distance and steps"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"pick_lift_margin_m": 0.031}))
    with pytest.raises(ValueError, match="pick lift margin"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"stable_support_steps": 241}))
    with pytest.raises(ValueError, match="stable support steps"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"post_handover_right_return_steps": 45}))
    with pytest.raises(ValueError, match="selected together"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({
        "post_handover_right_return_steps": 45,
        "left_branch_point_steps": 35,
        "direct_rest_to_preinsert_steps": 170,
    }))
    with pytest.raises(ValueError, match="must be selected together"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({
        "direct_rest_to_preinsert_steps": 170,
        "post_release_return_to_rest_steps": 120,
    }))
    with pytest.raises(ValueError, match="requires simultaneous"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({
        "post_handover_rest_observer_steps": 60,
        "direct_rest_to_preinsert_steps": 170,
        "post_release_return_to_rest_steps": 120,
        "branch_orient_steps": 20,
    }))
    with pytest.raises(ValueError, match="forbids branch orientation"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"require_broad_pad_contact": False}))
    with pytest.raises(ValueError, match="must be true"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"handover_handle_frame_transfer": False}))
    with pytest.raises(ValueError, match="must be true"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"handover_contact_acquire_steps": 61}))
    with pytest.raises(ValueError, match="contact acquire"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"handover_target_local_pitch_rad": 0.8}))
    with pytest.raises(ValueError, match="within 45 degrees"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"handover_orient_clearance_m": 0.08}))
    with pytest.raises(ValueError, match="orient clearance"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"handover_standoff_outside_m": 0.08}))
    with pytest.raises(ValueError, match="outside standoff"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({
        "handover_orient_clearance_m": 0.08,
        "handover_orient_steps": 30,
        "handover_standoff_outside_m": 0.121,
    }))
    with pytest.raises(ValueError, match="outside standoff"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"handover_straddle_local_x_m": -0.124}))
    with pytest.raises(ValueError, match="requires orient-first"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({
        "handover_orient_clearance_m": 0.08,
        "handover_orient_steps": 30,
        "handover_straddle_local_x_m": -0.141,
    }))
    with pytest.raises(ValueError, match="within 14 cm"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"branch_orient_steps": 91}))
    with pytest.raises(ValueError, match="branch orient steps"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"insert_clearance_m": 0.02}))
    with pytest.raises(ValueError, match="insert clearance"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"branch_approach_height_m": 0.081}))
    with pytest.raises(ValueError, match="branch approach height"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"branch_roll_offset_rad": 3.141592653589793}))
    with pytest.raises(ValueError, match="branch roll offset"):
        campaign._repair_strategy(2)
    candidate.write_text(json.dumps({"target_branch_rank": 2}))
    with pytest.raises(ValueError, match="unsupported repair candidate"):
        campaign._repair_strategy(2)


def test_campaign_accepts_only_middle_row_branches():
    for branch in ("branch_layer_2_a", "branch_layer_2_b"):
        campaign._require_middle_row_branch(branch)
    for branch in (None, "branch_layer_1_a", "branch_layer_3_b"):
        with pytest.raises(RuntimeError, match="middle-row branch"):
            campaign._require_middle_row_branch(branch)


def test_direct_choreography_candidate_pins_both_single_segment_counts(
    tmp_path, monkeypatch
):
    results = tmp_path / "task2"
    candidate = results / "pairs/000002/repair_candidate.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(json.dumps({
        "post_handover_rest_observer_steps": 60,
        "direct_rest_to_preinsert_steps": 170,
        "post_release_return_to_rest_steps": 120,
    }))
    monkeypatch.setattr(campaign, "RESULTS", results)
    strategy = campaign._repair_strategy(2)
    monkeypatch.setattr(campaign, "_common_workload", lambda *_: ["--device", "cpu"])
    monkeypatch.setattr(campaign, "_guarded", lambda _attempt, workload: workload)
    command = campaign._repair_command(
        2,
        tmp_path / "attempt",
        tmp_path / "result.json",
        campaign._repair_selection("pick", None),
        strategy,
    )

    assert command[command.index("--post-handover-rest-observer-steps") + 1] == "60"
    assert command[command.index("--direct-rest-to-preinsert-steps") + 1] == "170"
    assert command[command.index("--post-release-return-to-rest-steps") + 1] == "120"
    assert "--branch-orient-steps" not in command


def test_pick_lift_margin_is_reset_boundary_only(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "_common_workload", lambda *_: ["--device", "cpu"])
    monkeypatch.setattr(campaign, "_guarded", lambda _attempt, workload: workload)
    strategy = {"pick_lift_margin_m": 0.01}
    reset = campaign._repair_command(
        6, tmp_path / "reset", tmp_path / "classification.json",
        campaign._repair_selection("pick", None), strategy,
    )
    assert reset[reset.index("--pick-lift-margin-m") + 1] == "0.01"
    with pytest.raises(ValueError, match="valid only from reset"):
        campaign._repair_command(
            6, tmp_path / "prefix", tmp_path / "classification.json",
            campaign._repair_selection("handover", "pick"), strategy,
        )


def test_branch_support_candidate_is_allowed_from_exact_pick_prefix(tmp_path, monkeypatch):
    results = tmp_path / "task2"
    candidate = results / "pairs/000004/repair_candidate.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(json.dumps({"branch_support_fraction": 0.35}))
    monkeypatch.setattr(campaign, "RESULTS", results)
    strategy = campaign._repair_strategy(4)
    assert strategy == {"branch_support_fraction": 0.35}
    monkeypatch.setattr(campaign, "_common_workload", lambda *_: ["--device", "cpu"])
    monkeypatch.setattr(campaign, "_guarded", lambda _attempt, workload: workload)
    command = campaign._repair_command(
        4,
        tmp_path / "attempt",
        tmp_path / "classification/result.json",
        campaign._repair_selection("handover", "pick"),
        strategy,
    )
    assert "--reuse-source-pick-prefix" in command
    assert command[command.index("--branch-support-fraction") + 1] == "0.35"
    candidate.write_text(json.dumps({"branch_support_fraction": 0.2}))
    with pytest.raises(ValueError, match="support fraction"):
        campaign._repair_strategy(4)

    candidate.write_text(json.dumps({"branch_support_seat_down_m": 0.01}))
    strategy = campaign._repair_strategy(4)
    assert strategy == {"branch_support_seat_down_m": 0.01}
    command = campaign._repair_command(
        4,
        tmp_path / "attempt2",
        tmp_path / "classification/result.json",
        campaign._repair_selection("release_and_hang", "insertion_and_support"),
        strategy,
    )
    assert command[command.index("--branch-support-seat-down-m") + 1] == "0.01"
    candidate.write_text(json.dumps({"branch_support_seat_down_m": 0.031}))
    with pytest.raises(ValueError, match="seat-down"):
        campaign._repair_strategy(4)


def test_branch_approach_height_is_allowed_from_exact_pick_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "_common_workload", lambda *_: ["--device", "cpu"])
    monkeypatch.setattr(campaign, "_guarded", lambda _attempt, workload: workload)
    selection = campaign._repair_selection("alignment", "handover")

    command = campaign._repair_command(
        8,
        tmp_path / "attempt",
        tmp_path / "classification/result.json",
        selection,
        {"branch_approach_height_m": 0.0},
    )

    assert "--reuse-source-pick-prefix" in command
    assert command[command.index("--branch-approach-height-m") + 1] == "0.0"


@pytest.mark.parametrize(
    "failed,last_completed",
    (
        ("alignment", "handover"),
        ("insertion_and_support", "alignment"),
        ("release_and_hang", "insertion_and_support"),
    ),
)
def test_later_stage_failure_uses_truthful_coarse_pick_boundary(
    failed, last_completed, tmp_path, monkeypatch
):
    objects = tmp_path / "objects"
    for kind, name in (
        ("MugHangable", "mug_teacup_000002"),
        ("ThreeLayerMugTree", "mug_tree_000002"),
    ):
        (objects / kind / name).mkdir(parents=True)
    monkeypatch.setattr(campaign, "OBJECTS", objects)
    monkeypatch.setattr(campaign, "_steady_state_seconds", lambda: 211)
    selection = campaign._repair_selection(failed, last_completed)
    assert selection == {
        "requested_failed_stage": failed,
        "requested_last_completed_stage": last_completed,
        "actual_repair_boundary": "pick",
        "coarse_fallback": True,
    }
    command = campaign._repair_command(
        2, tmp_path / "repair", tmp_path / "classification/result.json", selection
    )
    assert "--reuse-source-pick-prefix" in command
    assert "--direct-replay-result" in command


def test_classification_binding_and_manifest_preserve_actual_boundary(
    tmp_path, monkeypatch
):
    attempt = tmp_path / "classification"
    attempt.mkdir()
    (attempt / "classification_audit.json").write_text("audit")
    classification = {
        "first_failed_stage": "alignment",
        "last_completed_stage": "handover",
        "completed_stages": ["pick", "handover"],
        "result_path": str(attempt / "result.json"),
        "artifacts": {"result_sha256": "result"},
    }
    binding = campaign._classification_binding(attempt, classification)
    assert binding["requested_failed_stage"] == "alignment"
    assert binding["requested_last_completed_stage"] == "handover"
    assert binding["actual_repair_boundary"] == "pick"
    assert binding["coarse_fallback"] is True
    assert "first_failed_stage" not in binding
    objects = tmp_path / "objects"
    for kind, name in (
        ("MugHangable", "mug_teacup_000002"),
        ("ThreeLayerMugTree", "mug_tree_000002"),
    ):
        (objects / kind / name).mkdir(parents=True)
    monkeypatch.setattr(campaign, "OBJECTS", objects)
    monkeypatch.setattr(campaign, "_sha256", lambda _path: "hash")
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *_a, **_k: "head\n")
    manifest = campaign._manifest(
        2, tmp_path / "repair", ["command"], "ledger",
        method="semantic_coarse_boundary_repair", classification=binding,
        repair_strategy={"handover_contact_settle_steps": 30},
    )
    assert manifest["classification"] == binding
    assert manifest["source_prefix_action_count"] == campaign.SOURCE_PREFIX_STEPS
    assert manifest["method"] == "semantic_coarse_boundary_repair"
    assert manifest["repair_strategy"] == {"handover_contact_settle_steps": 30}


def test_guard_lifecycle_requires_all_markers_and_live_zero_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "_worker_pids", lambda: [])
    (tmp_path / "replay.log").write_text("POST_RUN_ZERO_WORKER=PASS\n")
    (tmp_path / "replay.log.exit").write_text("GUARDED_RUN_EXIT=0\n")
    (tmp_path / "replay.log.stall").write_text(
        "NO_STEP_PROGRESS_STALL_TRIGGERED=0\n"
    )
    assert campaign._guard_lifecycle(tmp_path)["post_run_zero_worker"] is True
    (tmp_path / "replay.log.exit").unlink()
    with pytest.raises(RuntimeError, match=r"missing=\['exit'\]"):
        campaign._guard_lifecycle(tmp_path)
    (tmp_path / "replay.log.exit").write_text("GUARDED_RUN_EXIT=1\n")
    with pytest.raises(RuntimeError, match=r"missing=\['exit'\]"):
        campaign._guard_lifecycle(tmp_path)
    (tmp_path / "replay.log.exit").write_text("GUARDED_RUN_EXIT=0\n")
    monkeypatch.setattr(campaign, "_worker_pids", lambda: [99])
    with pytest.raises(RuntimeError, match=r"workers=\[99\]"):
        campaign._guard_lifecycle(tmp_path)


def _run_one_fixture(tmp_path, monkeypatch, classification):
    results = tmp_path / "task2"
    results.mkdir()
    (results / "ledger.json").write_text(json.dumps({"pairs": {}}))
    monkeypatch.setattr(campaign, "RESULTS", results)
    monkeypatch.setattr(campaign, "_require_zero_workers", lambda: None)
    monkeypatch.setattr(campaign, "_reusable_classification", lambda _index: None)
    monkeypatch.setattr(
        campaign, "_manifest",
        lambda *_args, method, classification=None, repair_strategy=None,
        ledger_transition=None: {
            "method": method, "classification": classification,
            "repair_strategy": repair_strategy, "ledger_transition": ledger_transition,
        },
    )
    events = []
    monkeypatch.setattr(
        campaign, "_classification_command", lambda *_: ["classification"]
    )
    monkeypatch.setattr(
        campaign, "_repair_command",
        lambda *_args: events.append("repair_command") or ["repair"],
    )
    def execute(_index, attempt, command):
        events.append(command[0])
        (attempt / "result.json").write_text("{}")
    monkeypatch.setattr(campaign, "_execute", execute)
    monkeypatch.setattr(
        campaign, "classification_audit",
        lambda *_: events.append("classification_audit") or classification,
    )
    monkeypatch.setattr(
        campaign, "_accept_attempt",
        lambda *_, **__: events.append("accepted"),
    )
    campaign.run_one(2)
    return events


def test_classification_precedes_repair(tmp_path, monkeypatch):
    result = {
        "status": "repair_required", "first_failed_stage": "handover",
        "last_completed_stage": "pick", "completed_stages": ["pick"],
        "result_path": str(tmp_path / "result.json"),
        "artifacts": {"result_sha256": "r"},
    }
    events = _run_one_fixture(tmp_path, monkeypatch, result)
    assert events == ["classification", "classification_audit", "repair_command", "repair", "accepted"]


def test_direct_success_skips_repair(tmp_path, monkeypatch):
    result = {
        "status": "direct_success", "first_failed_stage": None,
        "last_completed_stage": "release_and_hang",
        "completed_stages": ["pick", "handover", "alignment", "insertion_and_support", "release_and_hang"],
        "result_path": str(tmp_path / "result.json"),
        "artifacts": {"result_sha256": "r"},
    }
    events = _run_one_fixture(tmp_path, monkeypatch, result)
    assert events == ["classification", "classification_audit", "accepted"]


def test_quality_wave_direct_success_forces_fresh_skill(tmp_path, monkeypatch):
    result = {
        "status": "direct_success", "first_failed_stage": None,
        "last_completed_stage": "release_and_hang",
        "completed_stages": ["pick", "handover", "alignment", "insertion_and_support", "release_and_hang"],
        "result_path": str(tmp_path / "result.json"),
        "artifacts": {"result_sha256": "r"},
    }
    monkeypatch.setattr(campaign, "_force_semantic_regeneration", lambda _index: True)
    events = _run_one_fixture(tmp_path, monkeypatch, result)
    assert events == ["classification", "classification_audit", "repair_command", "repair", "accepted"]


def test_quality_wave_direct_failure_still_forces_fresh_skill_from_reset(
    tmp_path, monkeypatch
):
    result = {
        "status": "repair_required", "first_failed_stage": "handover",
        "last_completed_stage": "pick", "completed_stages": ["pick"],
        "result_path": str(tmp_path / "result.json"),
        "artifacts": {"result_sha256": "r"},
    }
    monkeypatch.setattr(campaign, "_force_semantic_regeneration", lambda _index: True)

    events = _run_one_fixture(tmp_path, monkeypatch, result)

    assert events == ["classification", "classification_audit", "repair_command", "repair", "accepted"]
    attempt = (
        tmp_path / "task2/pairs/000002/attempt_002_repair_quality_regeneration"
    )
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["classification"]["actual_repair_boundary"] == "reset"
    assert manifest["classification"]["requested_failed_stage"] == "pick"
    assert manifest["classification"][
        "quality_regeneration_from_direct_success"
    ] is True


def test_forced_quality_binding_restarts_from_reset(tmp_path, monkeypatch):
    audit = tmp_path / "classification_audit.json"
    audit.write_text("{}")
    classification = {
        "first_failed_stage": None,
        "last_completed_stage": "release_and_hang",
        "completed_stages": ["pick", "handover", "alignment", "insertion_and_support", "release_and_hang"],
        "result_path": str(tmp_path / "result.json"),
        "artifacts": {"result_sha256": "r"},
    }
    monkeypatch.setattr(campaign, "_sha256", lambda _path: "hash")
    binding = campaign._classification_binding(
        tmp_path, classification, force_from_reset=True
    )
    assert binding["actual_repair_boundary"] == "reset"
    assert binding["quality_regeneration_from_direct_success"] is True


def test_serial_campaign_never_advances_after_first_failure(monkeypatch):
    seen = []
    def fail(index):
        seen.append(index)
        raise RuntimeError("not accepted")
    monkeypatch.setattr(campaign, "run_one", fail)
    with pytest.raises(RuntimeError, match="not accepted"):
        campaign._run_serial([2, 3, 4])
    assert seen == [2]
