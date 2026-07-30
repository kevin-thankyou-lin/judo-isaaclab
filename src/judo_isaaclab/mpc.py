"""Thin Judo optimizer loop for history-conditioned IsaacLab rollouts."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from judo_isaaclab.backend import HistoryConditionedIsaacLabBackend
from judo_isaaclab.types import BranchContext

Objective = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
ControlExpander = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class MPCPlan:
    """Result of one MPC optimization call."""

    action: np.ndarray
    optimized_knots: np.ndarray
    sampled_knots: np.ndarray
    best_sampled_knots: np.ndarray
    rewards: np.ndarray
    best_rollout: int
    best_iteration: int
    accepted_update: bool
    improvement: float
    nominal_reward_mean: float
    nominal_reward_std: float


class JudoIsaacLabMPC:
    """Use a Judo optimizer with a history-conditioned IsaacLab backend."""

    def __init__(
        self,
        optimizer: Any,
        backend: HistoryConditionedIsaacLabBackend,
        objective: Objective,
        *,
        control_expander: ControlExpander | None = None,
        num_iterations: int = 1,
        duplicate_nominal: int = 2,
        min_improvement: float = 0.0,
        noise_std_multiplier: float = 2.0,
    ) -> None:
        if num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if duplicate_nominal < 1 or duplicate_nominal > optimizer.num_rollouts:
            raise ValueError("duplicate_nominal must fit within num_rollouts")
        self.optimizer = optimizer
        self.backend = backend
        self.objective = objective
        self.control_expander = control_expander or (lambda knots: knots)
        self.num_iterations = num_iterations
        self.duplicate_nominal = duplicate_nominal
        self.min_improvement = min_improvement
        self.noise_std_multiplier = noise_std_multiplier

    def plan(
        self,
        context: BranchContext,
        nominal_knots: np.ndarray,
    ) -> MPCPlan:
        """Optimize one action sequence and apply a measured-noise acceptance gate."""
        nominal = np.asarray(nominal_knots, dtype=np.float64).copy()
        if nominal.shape != (self.optimizer.num_nodes, self.optimizer.nu):
            raise ValueError(
                "nominal_knots must have shape "
                f"{(self.optimizer.num_nodes, self.optimizer.nu)}, got {nominal.shape}"
            )

        self.backend.set_branch_context(context)
        times = np.arange(self.optimizer.num_nodes, dtype=np.float64)
        self.optimizer.pre_optimization(times, times)

        starting_nominal = nominal.copy()
        sampled = np.empty((0, 0, 0))
        rewards = np.empty(0)
        starting_nominal_rewards = np.empty(0)
        best_sampled_knots = starting_nominal.copy()
        best_reward = -np.inf
        best_rollout = 0
        best_iteration = 0
        for iteration in range(self.num_iterations):
            sampled = np.asarray(
                self.optimizer.sample_control_knots(nominal), dtype=np.float64
            )
            sampled[: self.duplicate_nominal] = nominal
            controls = np.asarray(self.control_expander(sampled), dtype=np.float32)
            states, sensors, _ = self.backend.rollout(
                np.empty(0, dtype=np.float64), controls
            )
            rewards = np.asarray(
                self.objective(states, sensors, controls), dtype=np.float64
            )
            if rewards.shape != (self.optimizer.num_rollouts,):
                raise ValueError(
                    "objective must return one reward per rollout, got "
                    f"{rewards.shape}"
                )
            if iteration == 0:
                starting_nominal_rewards = rewards[
                    : self.duplicate_nominal
                ].copy()
            iteration_best = int(np.argmax(rewards))
            if float(rewards[iteration_best]) > best_reward:
                best_reward = float(rewards[iteration_best])
                best_sampled_knots = sampled[iteration_best].copy()
                best_rollout = iteration_best
                best_iteration = iteration
            nominal = self.optimizer.update_nominal_knots(sampled, rewards)

        nominal_mean = float(starting_nominal_rewards.mean())
        nominal_std = float(starting_nominal_rewards.std())
        improvement = float(best_reward - nominal_mean)
        threshold = max(
            self.min_improvement,
            self.noise_std_multiplier * nominal_std,
        )
        accepted = improvement > threshold
        action = (
            best_sampled_knots[0] if accepted else starting_nominal[0]
        )
        return MPCPlan(
            action=np.asarray(action).copy(),
            optimized_knots=np.asarray(nominal).copy(),
            sampled_knots=sampled.copy(),
            best_sampled_knots=best_sampled_knots,
            rewards=rewards.copy(),
            best_rollout=best_rollout,
            best_iteration=best_iteration,
            accepted_update=accepted,
            improvement=improvement,
            nominal_reward_mean=nominal_mean,
            nominal_reward_std=nominal_std,
        )
