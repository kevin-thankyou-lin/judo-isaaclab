"""Judo rollout backend backed by a vectorized IsaacLab planning scene."""

from collections.abc import Callable
from typing import Any

import numpy as np
from judo.utils.rollout_backend import RolloutBackend

from judo_isaaclab.types import BranchContext, RolloutDiagnostics

Encoder = Callable[[Any], np.ndarray]
ActionAdapter = Callable[[np.ndarray, Any], Any]
StepObserver = Callable[[Any, str, int], None]


def _default_action_adapter(actions: np.ndarray, env: Any) -> Any:
    import torch

    return torch.as_tensor(actions, dtype=torch.float32, device=env.device)


def _as_batch(value: Any, num_envs: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] != num_envs:
        raise ValueError(
            f"{label} must encode ({num_envs}, features), got {array.shape}"
        )
    return array


class HistoryConditionedIsaacLabBackend(RolloutBackend):
    """Reconstruct contact history, then execute Judo candidates in parallel.

    The runner must own a planning-only IsaacLab scene whose number of cloned
    environments equals Judo's rollout count. A fresh reset is followed by an
    identical action prefix in every clone. Candidate controls only diverge after
    that prefix, avoiding unsupported cold resets into a hidden contact manifold.
    """

    def __init__(
        self,
        runner: Any,
        state_encoder: Encoder,
        *,
        sensor_encoder: Encoder | None = None,
        action_adapter: ActionAdapter | None = None,
        candidate_action_adapter: ActionAdapter | None = None,
        step_observer: StepObserver | None = None,
    ) -> None:
        self.runner = runner
        self.env = runner.env
        self.num_threads = int(runner.num_envs)
        self.state_encoder = state_encoder
        self.sensor_encoder = sensor_encoder
        self.action_adapter = action_adapter or _default_action_adapter
        self.candidate_action_adapter = candidate_action_adapter
        self.step_observer = step_observer
        self.context: BranchContext | None = None
        self.last_diagnostics: RolloutDiagnostics | None = None
        self.last_executed_candidate_actions: np.ndarray | None = None

    def _observe(self, phase: str, step_index: int) -> None:
        if self.step_observer is not None:
            self.step_observer(self.env, phase, step_index)

    def set_branch_context(self, context: BranchContext) -> None:
        """Set the checkpoint and shared history for subsequent rollouts."""
        history = np.asarray(context.action_history)
        if history.ndim != 2:
            raise ValueError(
                f"action_history must have shape (steps, action_dim), got {history.shape}"
            )
        self.context = context

    def clear_branch_context(self) -> None:
        """Drop the current branch context."""
        self.context = None

    def _step(self, actions: np.ndarray, phase: str) -> Any:
        adapter = (
            self.candidate_action_adapter
            if phase == "candidate"
            and self.candidate_action_adapter is not None
            else self.action_adapter
        )
        adapted = adapter(actions, self.env)
        _, _, terminated, truncated, _ = self.env.step(adapted)
        if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
            raise RuntimeError(
                "IsaacLab planning environment produced a done signal; "
                "auto-reset must be disabled for MPC rollouts"
            )
        return adapted

    def _encode(self, encoder: Encoder | None, label: str) -> np.ndarray:
        if encoder is None:
            return np.empty((self.num_threads, 0), dtype=np.float32)
        return _as_batch(encoder(self.env), self.num_threads, label)

    def rollout(
        self,
        x0: np.ndarray,
        controls: np.ndarray,
        last_policy_output: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Run one history-conditioned batch in Judo's backend format."""
        del x0, last_policy_output
        if self.context is None:
            raise RuntimeError("set_branch_context() must be called before rollout()")

        candidates = np.asarray(controls, dtype=np.float32)
        if candidates.ndim != 3 or candidates.shape[0] != self.num_threads:
            raise ValueError(
                "controls must have shape "
                f"({self.num_threads}, horizon, action_dim), got {candidates.shape}"
            )
        history = np.asarray(self.context.action_history, dtype=np.float32)
        if (
            self.candidate_action_adapter is None
            and history.shape[1] != candidates.shape[2]
        ):
            raise ValueError(
                "History and candidate action dimensions differ: "
                f"{history.shape[1]} != {candidates.shape[2]}"
            )

        self.runner.reset(
            self.context.checkpoint_state,
            self.context.rigid_object_states,
            assist_states=self.context.assist_states,
            is_relative=self.context.is_relative,
            deformable_policy=self.context.deformable_policy,
            free_body_velocity_fallback=self.context.free_body_velocity_fallback,
        )
        self._observe("reset", -1)
        for step, action in enumerate(history):
            self._step(
                np.broadcast_to(action, (self.num_threads, action.size)).copy(),
                "history",
            )
            self._observe("history", step)

        states = []
        sensors = []
        executed_actions = []
        for step in range(candidates.shape[1]):
            adapted = self._step(candidates[:, step, :], "candidate")
            if hasattr(adapted, "detach"):
                adapted = adapted.detach().cpu().numpy()
            executed_actions.append(np.asarray(adapted, dtype=np.float32))
            self._observe("candidate", step)
            states.append(self._encode(self.state_encoder, "state_encoder"))
            sensors.append(self._encode(self.sensor_encoder, "sensor_encoder"))

        self.last_diagnostics = RolloutDiagnostics(
            num_rollouts=self.num_threads,
            history_steps=history.shape[0],
            horizon_steps=candidates.shape[1],
            action_dim=candidates.shape[2],
            reset_completed=True,
        )
        self.last_executed_candidate_actions = np.stack(
            executed_actions, axis=1
        )
        return (
            np.stack(states, axis=1),
            np.stack(sensors, axis=1),
            None,
        )

    def update(self, num_threads: int) -> None:
        """Validate Judo's requested rollout count against the fixed scene."""
        if num_threads != self.num_threads:
            raise ValueError(
                "IsaacLab clone count is fixed at scene creation: "
                f"requested {num_threads}, available {self.num_threads}"
            )

    def close(self) -> None:
        """Close the owned runner."""
        self.runner.close()
