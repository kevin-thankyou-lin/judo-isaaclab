"""Shared data structures."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BranchContext:
    """State needed to reconstruct one trustworthy MPC branch point.

    ``checkpoint_state`` should come from a contact-free state whenever possible.
    ``action_history`` is replayed identically in every planning environment before
    candidate controls are allowed to diverge.
    """

    checkpoint_state: Mapping[str, Any]
    action_history: np.ndarray
    rigid_object_states: Mapping[str, Any]
    assist_states: Mapping[str, Any] = field(default_factory=dict)
    is_relative: bool = True
    deformable_policy: str = "error"
    free_body_velocity_fallback: str | None = None


@dataclass(frozen=True)
class RolloutDiagnostics:
    """Metadata from the most recent backend rollout."""

    num_rollouts: int
    history_steps: int
    horizon_steps: int
    action_dim: int
    reset_completed: bool
