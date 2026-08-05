"""Reference PutPot controller plugin preserving the host's base trajectory."""

from __future__ import annotations


class PassthroughController:
    def initialize(self, context):
        return {
            "protocol_version": 1,
            "program_name": "putpot_passthrough",
            "total_steps": context["base_trajectory"]["steps"],
            "metadata": {"authoritative_targets": "host_base_trajectory"},
        }

    def command(self, request):
        base = request["base_command"]
        return {
            "kind": "cartesian",
            "stage": base["stage"],
            "left_pose": base["left_pose"],
            "right_pose": base["right_pose"],
            "grippers": base["grippers"],
            "terminate": False,
        }


def create_controller():
    return PassthroughController()
