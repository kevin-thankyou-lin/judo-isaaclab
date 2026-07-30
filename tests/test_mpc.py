import numpy as np

from judo_isaaclab import BranchContext, JudoIsaacLabMPC


class FakeOptimizer:
    num_rollouts = 4
    num_nodes = 2
    nu = 1

    def pre_optimization(self, old_times, new_times):
        pass

    def sample_control_knots(self, nominal):
        return np.array(
            [
                nominal,
                nominal,
                nominal + 1.0,
                nominal - 1.0,
            ]
        )

    def update_nominal_knots(self, sampled, rewards):
        return sampled[np.argmax(rewards)]


class FakeBackend:
    def set_branch_context(self, context):
        self.context = context

    def rollout(self, x0, controls):
        del x0
        states = np.cumsum(controls, axis=1)
        sensors = np.empty((*states.shape[:2], 0))
        return states, sensors, None


CONTEXT = BranchContext(
    checkpoint_state={},
    action_history=np.empty((0, 1)),
    rigid_object_states={},
)


def test_accepts_candidate_above_noise_gate():
    objective = lambda states, sensors, controls: states[:, -1, 0]
    mpc = JudoIsaacLabMPC(
        FakeOptimizer(),
        FakeBackend(),
        objective,
        duplicate_nominal=2,
        min_improvement=0.5,
    )

    plan = mpc.plan(CONTEXT, np.zeros((2, 1)))

    assert plan.accepted_update
    assert plan.best_rollout == 2
    np.testing.assert_allclose(plan.action, [1.0])


def test_rejects_update_inside_improvement_gate():
    objective = lambda states, sensors, controls: np.array([0.0, 0.0, 0.1, -0.1])
    mpc = JudoIsaacLabMPC(
        FakeOptimizer(),
        FakeBackend(),
        objective,
        duplicate_nominal=2,
        min_improvement=0.5,
    )

    plan = mpc.plan(CONTEXT, np.zeros((2, 1)))

    assert not plan.accepted_update
    np.testing.assert_allclose(plan.action, [0.0])
