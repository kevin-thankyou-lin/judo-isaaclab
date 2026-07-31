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
