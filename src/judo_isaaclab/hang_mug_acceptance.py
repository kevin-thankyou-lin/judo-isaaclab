"""Task-specific HangMug quality receipts and acceptance checks."""

from __future__ import annotations

from typing import Any


def audit_clean_insertion(
    args: Any,
    *,
    keyframes: dict[str, Any] | None,
    extracted_keyframes: dict[str, Any] | None,
    repair_prefix_steps: int | None,
    trajectory: Any,
    mug_poses: Any,
    tree_poses: Any,
    target_assets: dict[str, str],
) -> dict[str, Any] | None:
    """Run the exact-mesh audit with the configured explicit contact budget."""

    if not args.require_clean_insertion:
        return None
    audit_keyframes = keyframes or extracted_keyframes
    if audit_keyframes is None:
        raise RuntimeError(
            "clean insertion audit requires simulator-derived source keyframes"
        )
    if args.mode == "replay_hang":
        clean_start = int(repair_prefix_steps)
        clean_release = int(
            repair_prefix_steps + trajectory.waypoint_steps["branch_unload"] + 1
        )
    else:
        clean_start = int(
            audit_keyframes["frames"]["tree_approach"]["action_index"]
        )
        clean_release = int(
            audit_keyframes["frames"]["release"]["action_index"]
        )

    from .hang_mug_clean_insertion import exact_body_collision_receipt

    return exact_body_collision_receipt(
        mug_poses,
        tree_pose=tree_poses,
        target_assets=target_assets,
        start_step=clean_start,
        release_step=clean_release,
        maximum_collision_frames=(
            args.max_clean_insertion_body_collision_frames
        ),
        maximum_consecutive_collision_frames=(
            args.max_clean_insertion_consecutive_body_collision_frames
        ),
        maximum_penetration_depth_m=(
            args.max_clean_insertion_body_penetration_depth_m
        ),
    )


def controller_gain_receipt(args: Any) -> dict[str, Any]:
    """Prove waypoint repair did not scale the original controller gains."""

    passed = bool(
        args.replay_hang_insert_dls_gain == 1.0
        and args.replay_hang_support_dls_gain == 1.0
        and args.replay_hang_support_gain_lead_steps == 0
    )
    return {
        "required_default": bool(args.require_default_controller_gains),
        "insert_dls_gain": float(args.replay_hang_insert_dls_gain),
        "support_dls_gain": float(args.replay_hang_support_dls_gain),
        "support_gain_lead_steps": int(
            args.replay_hang_support_gain_lead_steps
        ),
        "passed": passed,
    }


def add_quality_checks(
    checks: dict[str, bool],
    *,
    args: Any,
    clean_insertion: dict[str, Any] | None,
    gain_receipt: dict[str, Any],
) -> None:
    """Add measured quality checks without changing generic harness policy."""

    checks["default_controller_gains_unchanged"] = bool(
        gain_receipt["passed"]
    )
    if args.require_clean_insertion:
        checks["clean_insertion_exact_mesh_audited"] = bool(
            clean_insertion
            and clean_insertion["method"]
            == "python-fcl exact mesh intersection"
        )
        checks["clean_insertion_body_collision_free"] = bool(
            clean_insertion and clean_insertion["strict_collision_free"]
        )
        checks["clean_insertion_body_contact_within_budget"] = bool(
            clean_insertion and clean_insertion["within_contact_budget"]
        )
    if args.require_default_controller_gains:
        checks["default_controller_gains_required"] = bool(
            gain_receipt["passed"]
        )


def add_quality_acceptance(
    acceptance: dict[str, bool],
    *,
    args: Any,
    checks: dict[str, bool],
) -> dict[str, bool]:
    """Require only quality gates explicitly selected by this task adapter."""

    quality = dict(acceptance)
    if args.require_clean_insertion:
        quality["clean_insertion_exact_mesh_audited"] = checks[
            "clean_insertion_exact_mesh_audited"
        ]
        quality["clean_insertion_body_contact_within_budget"] = checks[
            "clean_insertion_body_contact_within_budget"
        ]
    if args.require_default_controller_gains:
        quality["default_controller_gains_required"] = checks[
            "default_controller_gains_required"
        ]
    return quality
