"""Reusable command-line options for HangMug repair execution."""

from __future__ import annotations

from typing import Any


def add_replay_repair_arguments(parser: Any) -> None:
    """Expose optional branch selection and source-like release timing."""

    parser.add_argument(
        "--replay-target-branch-policy",
        choices=("source_corresponding", "nearest_eef"),
        default="source_corresponding",
        help=(
            "Retain the source-corresponding branch or select the one requiring "
            "the least EEF translation from the observed handover."
        ),
    )
    parser.add_argument(
        "--replay-hang-unload-steps",
        type=int,
        default=80,
        help="Hold the inserted mug for this many steps before opening the right hand.",
    )
    parser.add_argument(
        "--replay-hang-release-steps",
        type=int,
        default=40,
        help="Open the right hand over this many steps after insertion.",
    )
    parser.add_argument(
        "--replay-hang-insert-dls-gain",
        type=float,
        default=1.0,
        help=(
            "Scale only the receiving arm's incremental IK correction from "
            "branch approach through insertion; defaults to unchanged control."
        ),
    )
    parser.add_argument(
        "--replay-hang-support-dls-gain",
        type=float,
        default=1.0,
        help=(
            "Scale only the receiving arm's incremental IK correction after "
            "insertion while the mug is held on the branch; defaults to "
            "unchanged control."
        ),
    )
    parser.add_argument(
        "--replay-hang-support-gain-lead-steps",
        type=int,
        default=0,
        help=(
            "Begin the support gain this many controller steps before the "
            "insert waypoint; defaults to support-only behavior."
        ),
    )
    parser.add_argument(
        "--replay-hang-insert-time-scale",
        type=int,
        default=1,
        help=(
            "Densify each source-relative insertion interval by this integer "
            "factor; defaults to the unchanged source timing."
        ),
    )
    parser.add_argument(
        "--require-default-controller-gains",
        action="store_true",
        help=(
            "Fail closed unless insertion and support DLS gains remain at "
            "their original 1.0 values and no early support-gain lead is used."
        ),
    )
    parser.add_argument(
        "--max-clean-insertion-body-collision-frames",
        type=int,
        default=0,
        help="Allow at most this many exact body-contact frames before release.",
    )
    parser.add_argument(
        "--max-clean-insertion-consecutive-body-collision-frames",
        type=int,
        default=0,
        help="Allow at most this many consecutive exact body-contact frames.",
    )
    parser.add_argument(
        "--max-clean-insertion-body-penetration-depth-m",
        type=float,
        default=0.0,
        help="Allow exact body contact only up to this penetration depth in meters.",
    )


def validate_replay_repair_arguments(parser: Any, args: Any) -> None:
    """Fail closed on invalid clean-repair geometry or timing."""

    if args.clean_insertion_tracking_margin_m < 0.0:
        parser.error("--clean-insertion-tracking-margin-m must be nonnegative")
    if args.replay_hang_unload_steps < 0:
        parser.error("--replay-hang-unload-steps must be nonnegative")
    if args.replay_hang_release_steps <= 0:
        parser.error("--replay-hang-release-steps must be positive")
    if not 0.0 < args.replay_hang_insert_dls_gain <= 4.0:
        parser.error("--replay-hang-insert-dls-gain must be in (0, 4]")
    if not 0.0 < args.replay_hang_support_dls_gain <= 8.0:
        parser.error("--replay-hang-support-dls-gain must be in (0, 8]")
    if args.replay_hang_support_gain_lead_steps < 0:
        parser.error("--replay-hang-support-gain-lead-steps must be nonnegative")
    if args.replay_hang_insert_time_scale < 1:
        parser.error("--replay-hang-insert-time-scale must be positive")
    if args.max_clean_insertion_body_collision_frames < 0:
        parser.error(
            "--max-clean-insertion-body-collision-frames must be nonnegative"
        )
    if args.max_clean_insertion_consecutive_body_collision_frames < 0:
        parser.error(
            "--max-clean-insertion-consecutive-body-collision-frames must be "
            "nonnegative"
        )
    if args.max_clean_insertion_body_penetration_depth_m < 0.0:
        parser.error(
            "--max-clean-insertion-body-penetration-depth-m must be nonnegative"
        )
    if args.require_default_controller_gains and (
        args.replay_hang_insert_dls_gain != 1.0
        or args.replay_hang_support_dls_gain != 1.0
        or args.replay_hang_support_gain_lead_steps != 0
    ):
        parser.error(
            "--require-default-controller-gains forbids DLS gain scaling and "
            "support-gain lead overrides"
        )
