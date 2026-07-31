"""Versioned task-adaptation bundles and simulator trial evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageSpec:
    """One executable subtask in a continuous task strategy."""

    name: str
    target_name: str
    start_state: int
    target_state: int
    horizon: int
    parameters: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("stage name must be nonempty")
        if self.start_state < 0 or self.horizon < 1:
            raise ValueError(f"invalid stage interval for {self.name}")
        if not (
            self.start_state
            < self.target_state
            <= self.start_state + self.horizon
        ):
            raise ValueError(
                f"target_state for {self.name} must lie inside its horizon"
            )


@dataclass(frozen=True)
class TaskAdaptationBundle:
    """Immutable inputs plus an editable semantic strategy."""

    task_name: str
    dataset: str
    episode: str
    objects_root: str
    source_assets: dict[str, str]
    target_assets: dict[str, str]
    success_check: str
    checkpoint_state: int
    correspondences: dict[str, Any]
    stages: tuple[StageSpec, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "TaskAdaptationBundle":
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream)
        stages = tuple(StageSpec(**stage) for stage in value.pop("stages"))
        bundle = cls(stages=stages, **value)
        bundle.validate()
        return bundle

    def validate(self) -> None:
        if not self.task_name or not self.success_check:
            raise ValueError("task_name and success_check are required")
        if self.checkpoint_state < 0:
            raise ValueError("checkpoint_state must be nonnegative")
        if set(self.source_assets) != set(self.target_assets):
            raise ValueError("source and target asset categories must match")
        if self.source_assets == self.target_assets:
            raise ValueError("target assets must differ from source assets")
        previous_end = None
        for stage in self.stages:
            stage.validate()
            if previous_end is not None and stage.start_state != previous_end:
                raise ValueError("stages must be physically contiguous")
            previous_end = stage.start_state + stage.horizon


@dataclass(frozen=True)
class TrialEvidence:
    """The small, stable contract consumed by the orchestration loop."""

    status: str
    reached: bool
    metrics: dict[str, Any]
    result_path: str
    controls_path: str

    @classmethod
    def from_result(
        cls,
        result_path: str | Path,
        controls_path: str | Path,
    ) -> "TrialEvidence":
        with open(result_path, encoding="utf-8") as stream:
            result = json.load(stream)
        best = result.get("repeat_evaluation", {}).get("best_sample", {})
        return cls(
            status=str(result.get("status", "unknown")),
            reached=bool(result.get("best_sample_reached_keyframe", False)),
            metrics=best,
            result_path=str(result_path),
            controls_path=str(controls_path),
        )


def corrected_insert_offset(
    current: list[float] | tuple[float, float, float],
    evidence: TrialEvidence,
    *,
    gain: float = 0.5,
    maximum_step: float = 0.01,
) -> list[float]:
    """Use signed branch-frame error to propose one bounded correction."""
    vector = evidence.metrics.get("keyframe_position_error_vector_m_mean")
    if vector is None:
        return list(current)
    return [
        float(value - max(-maximum_step, min(maximum_step, gain * error)))
        for value, error in zip(current, vector)
    ]
