"""Task-agnostic instrumentation for continuous semantic skill execution.

The task runners own semantic geometry and control.  This module only records
runner-visible protocol operations and exposes optional observation hooks; an
empty hook set is deliberately a no-op so existing fixed-duration programs keep
their exact behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


EventKind = Literal[
    "rollout_start",
    "before_step",
    "after_step",
    "milestone",
    "rollout_end",
]


def _flag(value: Any) -> bool:
    """Reduce scalar, array, or tensor-like termination flags to one bool."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "any"):
        value = value.any()
    if hasattr(value, "item"):
        value = value.item()
    return bool(value)


@dataclass(frozen=True)
class SemanticExecutionEvent:
    """Read-only event delivered to optional task-owned observers."""

    kind: EventKind
    step: int | None = None
    stage: str | None = None
    milestone: str | None = None
    observation: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SemanticExecutionHooks:
    """Fan out execution events without prescribing task phases or policy."""

    def __init__(
        self,
        callbacks: tuple[Callable[[SemanticExecutionEvent], None], ...] = (),
    ) -> None:
        self._callbacks = tuple(callbacks)

    @property
    def enabled(self) -> bool:
        return bool(self._callbacks)

    def emit(self, event: SemanticExecutionEvent) -> None:
        for callback in self._callbacks:
            callback(event)


@dataclass
class _ContactChannel:
    source: str
    observations: int = 0
    active_steps: int = 0
    first_active_step: int | None = None
    last_active_step: int | None = None

    def record(self, step: int, active: bool, source: str) -> None:
        if source != self.source:
            raise ValueError(
                f"contact channel source changed from {self.source!r} to {source!r}"
            )
        self.observations += 1
        if active:
            self.active_steps += 1
            if self.first_active_step is None:
                self.first_active_step = int(step)
            self.last_active_step = int(step)

    def receipt(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "observations": self.observations,
            "active_steps": self.active_steps,
            "first_active_step": self.first_active_step,
            "last_active_step": self.last_active_step,
        }


@dataclass
class SemanticProtocolRecorder:
    """Measure runner-owned reset/state-write operations and step outcomes.

    The receipt intentionally states its measurement boundary.  It can observe
    calls made through this recorder and the flags returned by ``env.step``; it
    does not claim to introspect hidden simulator operations.
    """

    environment_resets: int = 0
    initial_state_restores: int = 0
    post_start_state_writes: int = 0
    inter_stage_resets: int = 0
    steps: int = 0
    rollout_started: bool = False
    rollout_finished: bool = False
    termination_events: list[dict[str, Any]] = field(default_factory=list)
    truncation_events: list[dict[str, Any]] = field(default_factory=list)
    _contacts: dict[str, _ContactChannel] = field(default_factory=dict)

    def record_environment_reset(self, *, reason: str) -> None:
        self.environment_resets += 1
        if self.rollout_started:
            self.inter_stage_resets += 1

    def record_state_restore(self, *, initial: bool, reason: str) -> None:
        if initial:
            if self.rollout_started:
                raise RuntimeError("initial state cannot be restored after rollout start")
            self.initial_state_restores += 1
        else:
            if self.rollout_started:
                self.post_start_state_writes += 1

    def start_rollout(self) -> None:
        if self.rollout_started:
            raise RuntimeError("rollout already started")
        if self.rollout_finished:
            raise RuntimeError("finished recorder cannot be restarted")
        self.rollout_started = True

    def record_step(
        self,
        *,
        step: int,
        stage: str,
        terminated: Any,
        truncated: Any,
    ) -> None:
        if not self.rollout_started or self.rollout_finished:
            raise RuntimeError("step recorded outside an active rollout")
        if int(step) != self.steps:
            raise ValueError(f"expected step {self.steps}, got {step}")
        self.steps += 1
        if _flag(terminated):
            self.termination_events.append({"step": int(step), "stage": stage})
        if _flag(truncated):
            self.truncation_events.append({"step": int(step), "stage": stage})

    def record_contact_observation(
        self,
        name: str,
        *,
        step: int,
        active: Any,
        source: str,
    ) -> None:
        if not self.rollout_started or self.rollout_finished:
            raise RuntimeError("contact recorded outside an active rollout")
        channel = self._contacts.setdefault(name, _ContactChannel(source=source))
        channel.record(int(step), _flag(active), source)

    def finish_rollout(self) -> None:
        if not self.rollout_started:
            raise RuntimeError("rollout was never started")
        if self.rollout_finished:
            raise RuntimeError("rollout already finished")
        self.rollout_finished = True

    def checks(self) -> dict[str, bool]:
        return {
            "one_reset": (
                self.environment_resets == 1
                and self.initial_state_restores == 1
                and self.inter_stage_resets == 0
            ),
            "zero_inter_stage_resets": self.inter_stage_resets == 0,
            "zero_post_start_state_writes": self.post_start_state_writes == 0,
            "no_truncation_observed": not self.truncation_events,
        }

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "measurement_scope": "runner_operations_and_env_step_flags",
            "environment_resets": self.environment_resets,
            "initial_state_restores": self.initial_state_restores,
            "inter_stage_resets": self.inter_stage_resets,
            "post_start_state_writes": self.post_start_state_writes,
            "teleports_after_rollout_start": self.post_start_state_writes,
            "steps": self.steps,
            "rollout_started": self.rollout_started,
            "rollout_finished": self.rollout_finished,
            "termination_events": list(self.termination_events),
            "truncation_events": list(self.truncation_events),
            "contact_channels": {
                name: channel.receipt()
                for name, channel in sorted(self._contacts.items())
            },
            "checks": self.checks(),
        }
