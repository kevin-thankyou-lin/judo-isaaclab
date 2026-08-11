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
