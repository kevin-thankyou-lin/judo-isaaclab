"""History-conditioned IsaacLab rollouts and asset-adaptation utilities."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "BranchContext": "judo_isaaclab.types",
    "RolloutDiagnostics": "judo_isaaclab.types",
    "HistoryConditionedIsaacLabBackend": "judo_isaaclab.backend",
    "JudoIsaacLabMPC": "judo_isaaclab.mpc",
    "MPCPlan": "judo_isaaclab.mpc",
    "DampedLeastSquaresPoseTrackingAdapter": "judo_isaaclab.task_space",
    "DampedLeastSquaresTaskSpaceAdapter": "judo_isaaclab.task_space",
    "resolve_end_effector_body_index": "judo_isaaclab.task_space",
    "StageSpec": "judo_isaaclab.adaptation",
    "TaskAdaptationBundle": "judo_isaaclab.adaptation",
    "TrialEvidence": "judo_isaaclab.adaptation",
    "asset_relative_grasp_pose": "judo_isaaclab.adaptation",
    "corrected_insert_offset": "judo_isaaclab.adaptation",
    "DrawerGeometry": "judo_isaaclab.put_marker",
    "PutMarkerSkillProgram": "judo_isaaclab.put_marker",
    "SkillTrajectory": "judo_isaaclab.put_marker",
    "SkillWaypoint": "judo_isaaclab.put_marker",
    "compose_pose": "judo_isaaclab.put_marker",
    "inverse_pose": "judo_isaaclab.put_marker",
    "pose_from_matrix": "judo_isaaclab.put_marker",
    "transfer_pose": "judo_isaaclab.put_marker",
    "AttemptEvaluation": "judo_isaaclab.evidence_harness",
    "EvidenceContract": "judo_isaaclab.evidence_harness",
    "EvidenceLedger": "judo_isaaclab.evidence_harness",
    "evaluate_result": "judo_isaaclab.evidence_harness",
    "execute_attempt": "judo_isaaclab.evidence_harness",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Load simulator-heavy modules only when their public symbol is used."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
