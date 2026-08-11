from judo_isaaclab.semantic_execution import (
    SemanticExecutionEvent,
    SemanticExecutionHooks,
    SemanticProtocolRecorder,
)


def _started_recorder() -> SemanticProtocolRecorder:
    recorder = SemanticProtocolRecorder()
    recorder.record_environment_reset(reason="task_environment_reset")
    recorder.record_state_restore(initial=True, reason="target_initial_state")
    recorder.start_rollout()
    return recorder


def test_protocol_receipt_measures_one_continuous_rollout():
    recorder = _started_recorder()
    recorder.record_step(
        step=0,
        stage="grasp",
        terminated=False,
        truncated=False,
    )
    recorder.record_contact_observation(
        "left_object_grasp",
        step=0,
        active=True,
        source="env.robot.is_grasping",
    )
    recorder.finish_rollout()

    receipt = recorder.receipt()
    assert receipt["checks"] == {
        "one_reset": True,
        "zero_inter_stage_resets": True,
        "zero_post_start_state_writes": True,
        "no_truncation_observed": True,
    }
    assert receipt["steps"] == 1
    assert receipt["reset_events"] == [
        {
            "index": 0,
            "reason": "task_environment_reset",
            "after_rollout_start": False,
        }
    ]
    assert receipt["state_restore_events"] == [
        {
            "index": 0,
            "reason": "target_initial_state",
            "initial": True,
            "after_rollout_start": False,
        }
    ]
    assert receipt["contact_channels"]["left_object_grasp"] == {
        "source": "env.robot.is_grasping",
        "observations": 1,
        "active_steps": 1,
        "first_active_step": 0,
        "last_active_step": 0,
    }


def test_protocol_receipt_exposes_post_start_mutations_and_step_flags():
    recorder = _started_recorder()
    recorder.record_step(
        step=0,
        stage="grasp",
        terminated=True,
        truncated=True,
    )
    recorder.record_state_restore(initial=False, reason="unexpected_reanchor_write")
    recorder.record_environment_reset(reason="unexpected_stage_reset")
    recorder.finish_rollout()

    receipt = recorder.receipt()
    assert receipt["checks"] == {
        "one_reset": False,
        "zero_inter_stage_resets": False,
        "zero_post_start_state_writes": False,
        "no_truncation_observed": False,
    }
    assert receipt["teleports_after_rollout_start"] == 1
    assert receipt["termination_events"] == [{"step": 0, "stage": "grasp"}]
    assert receipt["truncation_events"] == [{"step": 0, "stage": "grasp"}]


def test_empty_hooks_are_noop_and_callbacks_receive_read_only_events():
    empty = SemanticExecutionHooks()
    assert not empty.enabled
    empty.emit(SemanticExecutionEvent(kind="rollout_start"))

    events = []
    hooks = SemanticExecutionHooks((events.append,))
    event = SemanticExecutionEvent(
        kind="milestone",
        step=4,
        stage="handover",
        milestone="right_grasp",
        observation={"right_grasp": True},
    )
    hooks.emit(event)

    assert hooks.enabled
    assert events == [event]
