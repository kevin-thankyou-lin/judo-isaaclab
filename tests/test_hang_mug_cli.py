import argparse

import pytest

from judo_isaaclab.hang_mug_cli import (
    add_replay_repair_arguments,
    validate_replay_repair_arguments,
)


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean-insertion-tracking-margin-m", type=float, default=0.0
    )
    add_replay_repair_arguments(parser)
    return parser


def test_support_gain_accepts_bounded_gain_eight():
    parser = _parser()
    args = parser.parse_args(["--replay-hang-support-dls-gain", "8.0"])

    validate_replay_repair_arguments(parser, args)


def test_support_gain_rejects_values_above_eight():
    parser = _parser()
    args = parser.parse_args(["--replay-hang-support-dls-gain", "8.01"])

    with pytest.raises(SystemExit):
        validate_replay_repair_arguments(parser, args)


def test_default_controller_gain_contract_accepts_original_values():
    parser = _parser()
    args = parser.parse_args(["--require-default-controller-gains"])

    validate_replay_repair_arguments(parser, args)


def test_default_controller_gain_contract_rejects_scaled_gain():
    parser = _parser()
    args = parser.parse_args(
        [
            "--require-default-controller-gains",
            "--replay-hang-insert-dls-gain",
            "2.0",
        ]
    )

    with pytest.raises(SystemExit):
        validate_replay_repair_arguments(parser, args)


def test_minor_contact_budget_accepts_nonnegative_bounds():
    parser = _parser()
    args = parser.parse_args(
        [
            "--max-clean-insertion-body-collision-frames",
            "3",
            "--max-clean-insertion-consecutive-body-collision-frames",
            "2",
            "--max-clean-insertion-body-penetration-depth-m",
            "0.0005",
        ]
    )

    validate_replay_repair_arguments(parser, args)
