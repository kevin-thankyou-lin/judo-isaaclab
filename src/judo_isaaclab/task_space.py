"""Fast batched task-space residual adaptation for joint-target environments."""

from dataclasses import dataclass
from typing import Any


def damped_least_squares(jacobian: Any, twist: Any, damping: float) -> Any:
    """Map batched Cartesian residuals to joint residuals without an SVD."""
    import torch

    identity = torch.eye(
        jacobian.shape[-2],
        dtype=jacobian.dtype,
        device=jacobian.device,
    ).expand(jacobian.shape[0], -1, -1)
    system = jacobian @ jacobian.transpose(-1, -2)
    solved = torch.linalg.solve(
        system + damping**2 * identity,
        twist.unsqueeze(-1),
    )
    return (jacobian.transpose(-1, -2) @ solved).squeeze(-1)


@dataclass
class DampedLeastSquaresTaskSpaceAdapter:
    """Turn ``[base action, Cartesian residual]`` controls into joint targets."""

    arm_name: str = "right_arm"
    base_action_dim: int = 14
    arm_action_start: int = 7
    arm_joint_count: int = 6
    damping: float = 0.05
    max_joint_delta: float = 0.35

    @property
    def control_dim(self) -> int:
        return self.base_action_dim + 6

    def __call__(self, controls: Any, env: Any) -> Any:
        import torch

        controls = torch.as_tensor(
            controls, dtype=torch.float32, device=env.device
        )
        if controls.ndim != 2 or controls.shape[1] != self.control_dim:
            raise ValueError(
                f"Task-space controls must have shape "
                f"(batch, {self.control_dim}), got {tuple(controls.shape)}"
            )
        arm = env.scene[self.arm_name]
        body_index = arm.num_bodies - 1
        jacobian_index = body_index - 1 if arm.is_fixed_base else body_index
        jacobian = arm.root_physx_view.get_jacobians()[
            :, jacobian_index, :, : self.arm_joint_count
        ]
        joint_delta = damped_least_squares(
            jacobian,
            controls[:, self.base_action_dim :],
            self.damping,
        ).clamp(-self.max_joint_delta, self.max_joint_delta)

        actions = controls[:, : self.base_action_dim].clone()
        arm_slice = slice(
            self.arm_action_start,
            self.arm_action_start + self.arm_joint_count,
        )
        actions[:, arm_slice] += joint_delta
        limits = arm.data.joint_pos_limits[:, : self.arm_joint_count]
        actions[:, arm_slice] = torch.maximum(
            torch.minimum(actions[:, arm_slice], limits[:, :, 1]),
            limits[:, :, 0],
        )
        return actions


@dataclass
class DampedLeastSquaresPoseTrackingAdapter:
    """Track a reference EEF trajectory plus Cartesian residuals in closed loop."""

    reference_poses: Any
    arm_name: str = "right_arm"
    base_action_dim: int = 14
    arm_action_start: int = 7
    arm_joint_count: int = 6
    damping: float = 0.05
    max_joint_delta: float = 0.35
    max_position_step: float = 0.03
    max_rotation_step: float = 0.2

    def __post_init__(self) -> None:
        if (
            len(self.reference_poses.shape) != 2
            or self.reference_poses.shape[1] != 7
        ):
            raise ValueError(
                "reference_poses must have shape (horizon, 7), got "
                f"{self.reference_poses.shape}"
            )
        self._step = 0

    @property
    def control_dim(self) -> int:
        return self.base_action_dim + 6

    def begin_candidate_rollout(self, env: Any) -> None:
        import torch

        self._reference = torch.as_tensor(
            self.reference_poses,
            dtype=torch.float32,
            device=env.device,
        )
        self._step = 0

    @staticmethod
    def _clamp_norm(value: Any, maximum: float) -> Any:
        norm = value.norm(dim=-1, keepdim=True)
        scale = (maximum / norm.clamp_min(1.0e-8)).clamp(max=1.0)
        return value * scale

    def __call__(self, controls: Any, env: Any) -> Any:
        import torch
        from isaaclab.utils.math import (
            compute_pose_error,
            quat_from_angle_axis,
            quat_mul,
            subtract_frame_transforms,
        )

        controls = torch.as_tensor(
            controls, dtype=torch.float32, device=env.device
        )
        if controls.ndim != 2 or controls.shape[1] != self.control_dim:
            raise ValueError(
                f"Task-space controls must have shape "
                f"(batch, {self.control_dim}), got {tuple(controls.shape)}"
            )
        if self._step >= len(self._reference):
            raise RuntimeError("Candidate horizon exceeds reference trajectory")

        arm = env.scene[self.arm_name]
        body_index = arm.num_bodies - 1
        jacobian_index = body_index - 1 if arm.is_fixed_base else body_index
        jacobian = arm.root_physx_view.get_jacobians()[
            :, jacobian_index, :, : self.arm_joint_count
        ]
        current_pose = arm.data.body_pose_w[:, body_index]
        base_pose = arm.data.root_pose_w
        reference = self._reference[self._step].expand(controls.shape[0], -1)
        desired_position = (
            reference[:, :3]
            + env.scene.env_origins
            + controls[:, self.base_action_dim : self.base_action_dim + 3]
        )
        rotation_offset = controls[:, self.base_action_dim + 3 :]
        angle = rotation_offset.norm(dim=-1)
        axis = rotation_offset / angle.unsqueeze(-1).clamp_min(1.0e-8)
        desired_quaternion = quat_mul(
            quat_from_angle_axis(angle, axis),
            reference[:, 3:7],
        )
        current_position_b, current_quaternion_b = subtract_frame_transforms(
            base_pose[:, :3],
            base_pose[:, 3:7],
            current_pose[:, :3],
            current_pose[:, 3:7],
        )
        desired_position_b, desired_quaternion_b = subtract_frame_transforms(
            base_pose[:, :3],
            base_pose[:, 3:7],
            desired_position,
            desired_quaternion,
        )
        position_error, rotation_error = compute_pose_error(
            current_position_b,
            current_quaternion_b,
            desired_position_b,
            desired_quaternion_b,
            rot_error_type="axis_angle",
        )
        twist = torch.cat(
            (
                self._clamp_norm(position_error, self.max_position_step),
                self._clamp_norm(rotation_error, self.max_rotation_step),
            ),
            dim=-1,
        )
        joint_delta = damped_least_squares(
            jacobian, twist, self.damping
        ).clamp(-self.max_joint_delta, self.max_joint_delta)

        actions = controls[:, : self.base_action_dim].clone()
        arm_slice = slice(
            self.arm_action_start,
            self.arm_action_start + self.arm_joint_count,
        )
        actions[:, arm_slice] = (
            arm.data.joint_pos[:, : self.arm_joint_count] + joint_delta
        )
        limits = arm.data.joint_pos_limits[:, : self.arm_joint_count]
        actions[:, arm_slice] = torch.maximum(
            torch.minimum(actions[:, arm_slice], limits[:, :, 1]),
            limits[:, :, 0],
        )
        self._step += 1
        return actions
