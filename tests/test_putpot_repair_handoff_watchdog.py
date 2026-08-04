import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).parents[1]
        / "examples/watch_putpot_repair_handoffs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "watch_putpot_repair_handoffs", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _boundary(**overrides):
    value = {
        "action": "diagnose_and_repair",
        "fingerprint": "epoch:2",
        "pair": "cooktop_012__pot_012",
        "session_json": "/results/session.json",
        "attempts_completed": 2,
        "attempt_limit": 4,
        "latest_result_json": "/results/attempt_002/skill_result.json",
        "ready_since_epoch_s": 100.0,
    }
    value.update(overrides)
    return value


def test_waits_through_diagnosis_grace_period():
    module = _module()

    assert not module._should_wake(
        _boundary(),
        {"boundaries": {}},
        now=699.0,
        grace_seconds=600.0,
        repeat_seconds=1200.0,
        max_wakes=2,
    )
    assert module._should_wake(
        _boundary(),
        {"boundaries": {}},
        now=700.0,
        grace_seconds=600.0,
        repeat_seconds=1200.0,
        max_wakes=2,
    )


def test_deduplicates_wakes_for_same_receipt_and_allows_bounded_retry():
    module = _module()
    state = {
        "boundaries": {
            "epoch:2": {
                "wake_count": 1,
                "last_wake_epoch_s": 1000.0,
            }
        }
    }

    assert not module._should_wake(
        _boundary(),
        state,
        now=2199.0,
        grace_seconds=600.0,
        repeat_seconds=1200.0,
        max_wakes=2,
    )
    assert module._should_wake(
        _boundary(),
        state,
        now=2200.0,
        grace_seconds=600.0,
        repeat_seconds=1200.0,
        max_wakes=2,
    )
    state["boundaries"]["epoch:2"]["wake_count"] = 2
    assert not module._should_wake(
        _boundary(),
        state,
        now=4000.0,
        grace_seconds=600.0,
        repeat_seconds=1200.0,
        max_wakes=2,
    )


def test_completed_visit_uses_short_rotation_grace():
    module = _module()
    boundary = _boundary(
        action="rotate_after_visit",
        fingerprint="epoch:worker_summary",
    )

    assert not module._should_wake(
        boundary,
        {"boundaries": {}},
        now=129.0,
        grace_seconds=600.0,
        rotation_grace_seconds=30.0,
        repeat_seconds=1200.0,
        max_wakes=2,
    )
    assert module._should_wake(
        boundary,
        {"boundaries": {}},
        now=130.0,
        grace_seconds=600.0,
        rotation_grace_seconds=30.0,
        repeat_seconds=1200.0,
        max_wakes=2,
    )
def test_prompt_requires_diagnosis_and_duplicate_preflight():
    module = _module()

    prompt = module._prompt(_boundary())

    assert "inspect /results/attempt_002/skill_result.json" in prompt
    assert "exactly one materially revised program spec" in prompt
    assert "Do not blind-repeat" in prompt
    assert "do nothing duplicate" in prompt


def test_rotation_prompt_requires_live_process_preflight():
    module = _module()
    prompt = module._prompt(
        _boundary(action="rotate_after_visit", session_json="/results/finished.json")
    )

    assert "rotate to the next pending asset" in prompt
    assert "do not launch a duplicate simulator" in prompt


def test_tmux_wake_waits_for_paste_before_carriage_return(monkeypatch):
    module = _module()
    events = []

    def fake_run(argv, **_kwargs):
        events.append(tuple(argv))

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.time, "sleep", lambda value: events.append(("sleep", value)))

    module._wake_tmux("agent", "continue")

    assert events[-3:] == [
        ("tmux", "send-keys", "-t", "agent", "-l", "--", "continue"),
        ("sleep", 0.25),
        ("tmux", "send-keys", "-t", "agent", "C-m"),
    ]


def test_daemon_retries_transient_remote_snapshot_failure(
    tmp_path, monkeypatch, capsys
):
    module = _module()
    calls = []

    def fake_snapshot(_ssh, _root, *, timeout_seconds):
        calls.append(timeout_seconds)
        if len(calls) == 1:
            raise module.subprocess.CalledProcessError(255, ["ssh"])
        return {
            "ledger_summary": {"accepted": 120, "total": 120},
            "boundary": None,
        }

    monkeypatch.setattr(module, "_snapshot", fake_snapshot)
    monkeypatch.setattr(module.time, "sleep", lambda _value: None)

    module.main(
        [
            "--ssh-command",
            "ssh host",
            "--results-root",
            "/results",
            "--agent-tmux",
            "agent",
            "--state-json",
            str(tmp_path / "state.json"),
            "--poll-seconds",
            "0",
            "--ssh-timeout-seconds",
            "7",
        ]
    )

    assert calls == [7.0, 7.0]
    output = capsys.readouterr().out.splitlines()
    error = json.loads(output[0].split("=", 1)[1])
    assert error["consecutive_failures"] == 1
    assert output[-1].startswith("PUTPOT_HANDOFF_WATCHDOG_COMPLETE=")


def test_once_mode_surfaces_remote_snapshot_failure(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(
        module,
        "_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module.subprocess.TimeoutExpired(["ssh"], 3)
        ),
    )

    with pytest.raises(module.subprocess.TimeoutExpired):
        module.main(
            [
                "--ssh-command",
                "ssh host",
                "--results-root",
                "/results",
                "--agent-tmux",
                "agent",
                "--state-json",
                str(tmp_path / "state.json"),
                "--once",
            ]
        )
