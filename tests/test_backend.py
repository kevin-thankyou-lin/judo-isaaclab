import numpy as np
import pytest

from judo_isaaclab import BranchContext, HistoryConditionedIsaacLabBackend


class FakeEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.device = "cpu"
        self.state = np.zeros((num_envs, 1), dtype=np.float32)

    def step(self, actions):
        self.state += np.asarray(actions)[:, :1]
        done = np.zeros(self.num_envs, dtype=bool)
        return None, None, done, done, {}


class FakeRunner:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.env = FakeEnv(num_envs)
        self.closed = False

    def reset(self, checkpoint_state, rigid_object_states, **kwargs):
        del rigid_object_states, kwargs
        self.env.state[:] = checkpoint_state["value"]

    def close(self):
        self.closed = True


def make_backend(num_envs=3):
    runner = FakeRunner(num_envs)
    return HistoryConditionedIsaacLabBackend(
        runner,
        state_encoder=lambda env: env.state.copy(),
        action_adapter=lambda actions, env: actions,
    )


def test_history_is_shared_before_candidates_branch():
    backend = make_backend()
    backend.set_branch_context(
        BranchContext(
            checkpoint_state={"value": 10.0},
            action_history=np.array([[1.0], [2.0]], dtype=np.float32),
            rigid_object_states={},
        )
    )
    controls = np.array(
        [
            [[0.0], [1.0]],
            [[2.0], [1.0]],
            [[4.0], [1.0]],
        ],
        dtype=np.float32,
    )

    states, sensors, policy_output = backend.rollout(np.empty(0), controls)

    np.testing.assert_allclose(states[:, 0, 0], [13.0, 15.0, 17.0])
    np.testing.assert_allclose(states[:, 1, 0], [14.0, 16.0, 18.0])
    assert sensors.shape == (3, 2, 0)
    assert policy_output is None
    assert backend.last_diagnostics.history_steps == 2


def test_context_is_required():
    backend = make_backend()
    with pytest.raises(RuntimeError, match="set_branch_context"):
        backend.rollout(np.empty(0), np.zeros((3, 2, 1)))


def test_fixed_clone_count_cannot_be_resized():
    backend = make_backend()
    backend.update(3)
    with pytest.raises(ValueError, match="fixed"):
        backend.update(4)


def test_close_delegates_to_runner():
    backend = make_backend()
    backend.close()
    assert backend.runner.closed


def test_encoder_shape_is_checked():
    backend = HistoryConditionedIsaacLabBackend(
        FakeRunner(3),
        state_encoder=lambda env: np.zeros((1, 4)),
        action_adapter=lambda actions, env: actions,
    )
    backend.set_branch_context(
        BranchContext(
            checkpoint_state={"value": 0.0},
            action_history=np.empty((0, 1)),
            rigid_object_states={},
        )
    )
    with pytest.raises(ValueError, match="state_encoder"):
        backend.rollout(np.empty(0), np.zeros((3, 1, 1)))


def test_step_observer_sees_reset_history_and_candidates():
    events = []
    runner = FakeRunner(2)
    backend = HistoryConditionedIsaacLabBackend(
        runner,
        state_encoder=lambda env: env.state.copy(),
        action_adapter=lambda actions, env: actions,
        step_observer=lambda env, phase, step: events.append(
            (phase, step, env.state.copy())
        ),
    )
    backend.set_branch_context(
        BranchContext(
            checkpoint_state={"value": 4.0},
            action_history=np.array([[1.0], [2.0]], dtype=np.float32),
            rigid_object_states={},
        )
    )

    backend.rollout(np.empty(0), np.zeros((2, 2, 1), dtype=np.float32))

    assert [(phase, step) for phase, step, _ in events] == [
        ("reset", -1),
        ("history", 0),
        ("history", 1),
        ("candidate", 0),
        ("candidate", 1),
    ]
    np.testing.assert_allclose(events[0][2], [[4.0], [4.0]])
    np.testing.assert_allclose(events[2][2], [[7.0], [7.0]])


def test_candidate_adapter_can_change_action_dimension():
    runner = FakeRunner(2)
    backend = HistoryConditionedIsaacLabBackend(
        runner,
        state_encoder=lambda env: env.state.copy(),
        action_adapter=lambda actions, env: actions,
        candidate_action_adapter=lambda controls, env: controls[:, :1] * 2.0,
    )
    backend.set_branch_context(
        BranchContext(
            checkpoint_state={"value": 0.0},
            action_history=np.array([[1.0]], dtype=np.float32),
            rigid_object_states={},
        )
    )

    states, _, _ = backend.rollout(
        np.empty(0),
        np.array([[[2.0, 9.0]], [[3.0, 9.0]]], dtype=np.float32),
    )

    np.testing.assert_allclose(states[:, 0, 0], [5.0, 7.0])
    np.testing.assert_allclose(
        backend.last_executed_candidate_actions[:, 0, 0], [4.0, 6.0]
    )


def test_candidate_adapter_is_reset_before_each_rollout():
    class StatefulAdapter:
        def __init__(self):
            self.begin_calls = 0
            self.step = 0

        def begin_candidate_rollout(self, env):
            del env
            self.begin_calls += 1
            self.step = 0

        def __call__(self, controls, env):
            del env
            self.step += 1
            return controls[:, :1] + self.step

    adapter = StatefulAdapter()
    runner = FakeRunner(2)
    backend = HistoryConditionedIsaacLabBackend(
        runner,
        state_encoder=lambda env: env.state.copy(),
        action_adapter=lambda actions, env: actions,
        candidate_action_adapter=adapter,
    )
    backend.set_branch_context(
        BranchContext(
            checkpoint_state={"value": 0.0},
            action_history=np.empty((0, 1), dtype=np.float32),
            rigid_object_states={},
        )
    )
    controls = np.zeros((2, 2, 2), dtype=np.float32)

    first, _, _ = backend.rollout(np.empty(0), controls)
    second, _, _ = backend.rollout(np.empty(0), controls)

    assert adapter.begin_calls == 2
    np.testing.assert_allclose(first, second)
