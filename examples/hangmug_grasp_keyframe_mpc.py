"""Run history-conditioned Judo CEM toward a HangMug keyframe."""

import argparse
import json
import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

HANG_SPEED_TOLERANCE_M_S = 0.05
HANG_STABILITY_STEPS = 30
TREE_APPROACH_POSITION_TOLERANCE_M = 0.06
TREE_INSERTION_POSITION_TOLERANCE_M = 0.01
TREE_INSERTION_ROTATION_TOLERANCE_RAD = 0.15
TREE_INSERTION_STABILITY_STEPS = 3
HANGMUG_INSERT_ANCHOR_STATE = 774
HANGMUG_INSERT_CLEARANCE_OFFSET_BRANCH_M = (-0.025, 0.025, -0.055)
HANGMUG_INSERT_APPROACH_OFFSET_BRANCH_M = (0.05, 0.0, 0.0)
HANGMUG_INSERT_SEAT_OFFSET_BRANCH_M = (0.0, 0.0, 0.0)
HANGMUG_INSERT_EEF_POSITION_OFFSET_BRANCH_M = (0.0, 0.0, 0.0)
HANGMUG_INSERT_EEF_ROTATION_OFFSET_BRANCH_RAD = (0.0, 0.0, 0.0)
HANGMUG_INSERT_CLEARANCE_ROTATION_OFFSET_BRANCH_RAD = (0.0, 0.0, 0.0)
HANGMUG_INSERT_CLEARANCE_FRACTION = 0.20
HANGMUG_INSERT_APPROACH_FRACTION = 0.60
HANGMUG_INSERT_SEAT_FRACTION = 0.85
HANGMUG_RELEASE_OPEN_VALUE = -0.0475
HANGMUG_RELEASE_START_FRACTION = 0.05
HANGMUG_RELEASE_END_FRACTION = 0.20


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gear-repo",
        default=(
            "/home/linke/Projects/"
            "gear-dc-study-judo-hangmug-friction-20260730"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=(
            "/tmp/hangmug_source_demo/tasks_data/HangMugOnTree/"
            "teleop/hangmug_000.hdf5"
        ),
    )
    parser.add_argument(
        "--objects-root",
        default=(
            "/tmp/hangmug_source_demo/tasks_data/"
            "HangMugOnTree/objects"
        ),
    )
    parser.add_argument(
        "--target-mug-tree",
        help=(
            "Target MugTree instance directory. Defaults to the source "
            "mug_tree_000 asset."
        ),
    )
    parser.add_argument(
        "--target-mug",
        help=(
            "Target Mug instance directory. Defaults to the source "
            "mug_000 asset."
        ),
    )
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument(
        "--history-controls-npz",
        nargs="*",
        default=(),
        help=(
            "Earlier-stage controls whose best samples replace matching "
            "segments of the source action history."
        ),
    )
    parser.add_argument(
        "--checkpoint-state",
        type=int,
        default=None,
        help="Contact-free reset state; defaults to --start-state.",
    )
    parser.add_argument("--start-state", type=int, default=110)
    parser.add_argument("--target-state", type=int, default=116)
    parser.add_argument(
        "--target-name",
        choices=(
            "left_grasp",
            "right_grasp",
            "handover_latched",
            "tree_approach",
            "inserted_held",
            "hang_complete",
        ),
        default="left_grasp",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--search-space",
        choices=("joint", "task"),
        default="joint",
        help="Optimize smooth joint corrections or Cartesian right-EEF residuals.",
    )
    parser.add_argument(
        "--task-controller",
        choices=("joint_residual", "pose_tracking", "semantic_pose"),
        default="joint_residual",
        help=(
            "Map task residuals around source joint targets, track a recorded "
            "source EEF trajectory, or generate a fresh semantic-keyframe path."
        ),
    )
    parser.add_argument(
        "--control-knots",
        type=int,
        help=(
            "Number of smooth correction knots. Defaults to one knot per "
            "control step."
        ),
    )
    parser.add_argument("--num-rollouts", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--num-elites", type=int, default=4)
    parser.add_argument("--duplicate-nominal", type=int, default=4)
    parser.add_argument("--candidate-repeats", type=int, default=1)
    parser.add_argument(
        "--candidate-repeat-reducer",
        choices=("mean", "min"),
        default="mean",
        help="Aggregate duplicate candidate scores by mean or worst repeat.",
    )
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=0.03)
    parser.add_argument("--max-action-delta", type=float, default=0.08)
    parser.add_argument(
        "--task-translation-goal",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="World-frame Cartesian warm-start offset at the target keyframe.",
    )
    parser.add_argument(
        "--task-translation-start",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="World-frame Cartesian warm-start offset at the stage start.",
    )
    parser.add_argument("--max-task-translation-delta", type=float, default=0.08)
    parser.add_argument("--max-task-rotation-delta", type=float, default=0.3)
    parser.add_argument("--dls-damping", type=float, default=0.05)
    parser.add_argument("--dls-max-joint-delta", type=float, default=0.35)
    parser.add_argument("--dls-max-position-step", type=float, default=0.03)
    parser.add_argument("--dls-max-rotation-step", type=float, default=0.2)
    parser.add_argument(
        "--insert-eef-position-offset-branch",
        type=float,
        nargs=3,
        default=HANGMUG_INSERT_EEF_POSITION_OFFSET_BRANCH_M,
        metavar=("X", "Y", "Z"),
        help="Calibrated final EEF position correction in the branch frame.",
    )
    parser.add_argument(
        "--insert-eef-rotation-offset-branch",
        type=float,
        nargs=3,
        default=HANGMUG_INSERT_EEF_ROTATION_OFFSET_BRANCH_RAD,
        metavar=("RX", "RY", "RZ"),
        help="Calibrated final EEF axis-angle correction in the branch frame.",
    )
    parser.add_argument(
        "--insert-clearance-rotation-offset-branch",
        type=float,
        nargs=3,
        default=HANGMUG_INSERT_CLEARANCE_ROTATION_OFFSET_BRANCH_RAD,
        metavar=("RX", "RY", "RZ"),
        help=(
            "Pre-approach EEF orientation relative to the final EEF "
            "orientation, as branch-frame axis-angle."
        ),
    )
    parser.add_argument(
        "--insert-clearance-offset-branch",
        type=float,
        nargs=3,
        default=HANGMUG_INSERT_CLEARANCE_OFFSET_BRANCH_M,
        metavar=("X", "Y", "Z"),
        help="Collision-free pre-approach EEF offset in the branch frame.",
    )
    parser.add_argument(
        "--insert-approach-offset-branch",
        type=float,
        nargs=3,
        default=HANGMUG_INSERT_APPROACH_OFFSET_BRANCH_M,
        metavar=("X", "Y", "Z"),
        help="Pre-insert EEF offset in the matched target-branch frame.",
    )
    parser.add_argument(
        "--insert-seat-offset-branch",
        type=float,
        nargs=3,
        default=HANGMUG_INSERT_SEAT_OFFSET_BRANCH_M,
        metavar=("X", "Y", "Z"),
        help="Final seating EEF offset in the matched target-branch frame.",
    )
    parser.add_argument(
        "--insert-clearance-fraction",
        type=float,
        default=HANGMUG_INSERT_CLEARANCE_FRACTION,
    )
    parser.add_argument(
        "--insert-approach-fraction",
        type=float,
        default=HANGMUG_INSERT_APPROACH_FRACTION,
    )
    parser.add_argument(
        "--insert-seat-fraction",
        type=float,
        default=HANGMUG_INSERT_SEAT_FRACTION,
    )
    parser.add_argument(
        "--insert-anchor-state",
        type=int,
        default=HANGMUG_INSERT_ANCHOR_STATE,
        help=(
            "Stable released-hang state whose mug-to-branch transform defines "
            "actual insertion geometry."
        ),
    )
    parser.add_argument(
        "--semantic-anchor-state",
        type=int,
        help=(
            "Optional source-demo state supplying the semantic EEF target "
            "independently of rollout timing."
        ),
    )
    parser.add_argument(
        "--release-open-value",
        type=float,
        help="Program the right gripper to this open target for hang_complete.",
    )
    parser.add_argument(
        "--release-start-fraction",
        type=float,
        default=HANGMUG_RELEASE_START_FRACTION,
    )
    parser.add_argument(
        "--release-end-fraction",
        type=float,
        default=HANGMUG_RELEASE_END_FRACTION,
    )
    parser.add_argument("--right-joint-path-weight", type=float, default=0.0)
    parser.add_argument("--right-joint-accel-weight", type=float, default=0.0)
    parser.add_argument("--right-joint-jerk-weight", type=float, default=0.0)
    parser.add_argument("--friction-high", type=float, default=30.0)
    parser.add_argument("--friction-low", type=float, default=0.5)
    parser.add_argument(
        "--tree-offset-xyz",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Translate the planning tree while preserving its relative target.",
    )
    parser.add_argument(
        "--tree-yaw-deg",
        type=float,
        default=0.0,
        help="Rotate the planning tree about world Z.",
    )
    parser.add_argument(
        "--source-branch-points",
        type=float,
        nargs=9,
        metavar=(
            "X0",
            "Y0",
            "Z0",
            "X1",
            "Y1",
            "Z1",
            "X2",
            "Y2",
            "Z2",
        ),
        help="Three source-tree-local points defining the matched branch frame.",
    )
    parser.add_argument(
        "--target-branch-points",
        type=float,
        nargs=9,
        metavar=(
            "X0",
            "Y0",
            "Z0",
            "X1",
            "Y1",
            "Z1",
            "X2",
            "Y2",
            "Z2",
        ),
        help="The three corresponding target-tree-local branch points.",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--result-json",
        default="/tmp/judo_isaaclab_hangmug_grasp_keyframe_mpc.json",
    )
    parser.add_argument(
        "--controls-npz",
        help="Optional nominal/optimized/best controls for replay rendering.",
    )
    return parser.parse_args()


def _tensor_tree(group, index, device):
    import h5py
    import torch

    result = {}
    for name, value in group.items():
        if isinstance(value, h5py.Group):
            result[name] = _tensor_tree(value, index, device)
        else:
            result[name] = torch.as_tensor(
                value[index : index + 1],
                dtype=torch.float32,
                device=device,
            )
    return result


def _source_tree_path(args):
    return os.path.join(args.objects_root, "MugTree", "mug_tree_000")


def _target_tree_path(args):
    return args.target_mug_tree or _source_tree_path(args)


def _source_mug_path(args):
    return os.path.join(args.objects_root, "Mug", "mug_000")


def _target_mug_path(args):
    return args.target_mug or _source_mug_path(args)


def _asset_height(path):
    with open(os.path.join(path, "asset_size.json"), encoding="utf-8") as stream:
        return float(json.load(stream)["size"]["z"])


def _tree_root_z_adjustment(args):
    return (
        _asset_height(_target_tree_path(args))
        - _asset_height(_source_tree_path(args))
    ) / 2.0


def _state_row(group, index):
    return np.concatenate(
        (
            np.asarray(group["rigid_object/mug/root_pose"][index]),
            np.asarray(group["rigid_object/mug/root_velocity"][index]),
            np.asarray(group["articulation/left_arm/joint_position"][index]),
            np.asarray(group["articulation/right_arm/joint_position"][index]),
            np.asarray(group["rigid_object/mug_tree/root_pose"][index]),
        )
    ).astype(np.float32)


def _apply_history_control_overrides(
    history,
    *,
    checkpoint_state,
    start_state,
    controls_paths,
):
    history = np.asarray(history, dtype=np.float32).copy()
    for path in controls_paths:
        with np.load(path) as controls:
            override = np.asarray(
                controls[
                    "best_executed_actions"
                    if "best_executed_actions" in controls.files
                    else "best_sample"
                ],
                dtype=np.float32,
            )
            override_start = int(controls["start_state"])
        override_end = override_start + len(override)
        if not checkpoint_state <= override_start < override_end <= start_state:
            raise ValueError(
                f"History override {path} spans [{override_start}, "
                f"{override_end}), outside [{checkpoint_state}, {start_state})"
            )
        if override.shape[1:] != history.shape[1:]:
            raise ValueError(
                f"History override {path} has action shape "
                f"{override.shape[1:]}, expected {history.shape[1:]}"
            )
        begin = override_start - checkpoint_state
        history[begin : begin + len(override)] = override
    return history


def _load_demo(args, device):
    import h5py
    import torch

    checkpoint_state = (
        args.start_state
        if args.checkpoint_state is None
        else args.checkpoint_state
    )
    if checkpoint_state > args.start_state:
        raise ValueError("checkpoint-state must not exceed start-state")
    with h5py.File(args.dataset, "r") as handle:
        group = handle[f"data/{args.episode}"]
        states = group["states"]
        checkpoint = _tensor_tree(states, checkpoint_state, device)
        tree_pose = checkpoint["rigid_object"]["mug_tree"]["root_pose"]
        tree_pose[:, 2] += _tree_root_z_adjustment(args)
        tree_pose[:, :3] += torch.as_tensor(
            args.tree_offset_xyz,
            dtype=tree_pose.dtype,
            device=tree_pose.device,
        )
        if args.tree_yaw_deg:
            half_yaw = np.deg2rad(args.tree_yaw_deg) / 2.0
            yaw = torch.tensor(
                [
                    np.cos(half_yaw),
                    0.0,
                    0.0,
                    np.sin(half_yaw),
                ],
                dtype=tree_pose.dtype,
                device=tree_pose.device,
            ).reshape(1, 4)
            tree_pose[:, 3:7] = _torch_quat_multiply(
                yaw, tree_pose[:, 3:7]
            )
        source_history = np.asarray(
            group["actions"][checkpoint_state : args.start_state],
            dtype=np.float32,
        )
        history = _apply_history_control_overrides(
            source_history,
            checkpoint_state=checkpoint_state,
            start_state=args.start_state,
            controls_paths=args.history_controls_npz,
        )
        nominal = np.asarray(
            group["actions"][
                args.start_state : args.start_state + args.horizon
            ],
            dtype=np.float32,
        )
        reference = np.stack(
            [
                _state_row(states, index)
                for index in range(
                    args.start_state + 1,
                    args.start_state + args.horizon + 1,
                )
            ]
        )
    return checkpoint, history, nominal, reference, source_history


def _load_demo_state_row(args, index):
    import h5py

    with h5py.File(args.dataset, "r") as handle:
        group = handle[f"data/{args.episode}"]
        states = group["states"]
        state_count = len(group["actions"]) + 1
        if not 0 <= index < state_count:
            raise ValueError(
                f"state index {index} is outside [0, {state_count})"
            )
        return _state_row(states, index)


def _load_demo_actions(args, start, end):
    import h5py

    with h5py.File(args.dataset, "r") as handle:
        actions = handle[f"data/{args.episode}/actions"]
        if not 0 <= start <= end <= len(actions):
            raise ValueError(
                f"action interval [{start}, {end}) is outside "
                f"[0, {len(actions)})"
            )
        return np.asarray(actions[start:end], dtype=np.float32)


def _encode_state(env):
    import torch

    state = env.scene.get_state(is_relative=True)
    return (
        torch.cat(
            (
                state["rigid_object"]["mug"]["root_pose"],
                state["rigid_object"]["mug"]["root_velocity"],
                state["articulation"]["left_arm"]["joint_position"],
                state["articulation"]["right_arm"]["joint_position"],
                state["rigid_object"]["mug_tree"]["root_pose"],
            ),
            dim=-1,
        )
        .detach()
        .cpu()
        .numpy()
    )


def _encode_sensors(env):
    import torch

    env_ids = torch.arange(env.num_envs, device=env.device)
    left = env.robot.arms["left_arm"].end_effector
    right = env.robot.arms["right_arm"].end_effector
    left_forces = [
        finger.contact_force("mug", env_ids).reshape(-1, 1)
        for finger in left.fingers
    ]
    right_forces = [
        finger.contact_force("mug", env_ids).reshape(-1, 1)
        for finger in right.fingers
    ]
    left_grasp = left.is_grasping(
        "mug", env_ids=env_ids
    ).float().reshape(-1, 1)
    right_grasp = right.is_grasping(
        "mug", env_ids=env_ids
    ).float().reshape(-1, 1)
    left_assist = env.grasp_assists["left"].engaged.float().reshape(-1, 1)
    stages = [
        stage.float().reshape(-1, 1)
        for stage in (
            env.stage1_success,
            env.stage2_success,
            env.stage3_success,
        )
    ]
    return (
        torch.cat(
            (
                *left_forces,
                left_grasp,
                left_assist,
                *right_forces,
                right_grasp,
                *stages,
            ),
            dim=1,
        )
        .detach()
        .cpu()
        .numpy()
    )


def _replay_context(runner, context):
    import torch

    env = runner.env
    runner.reset(
        context.checkpoint_state,
        context.rigid_object_states,
        assist_states=context.assist_states,
        is_relative=context.is_relative,
        deformable_policy=context.deformable_policy,
        free_body_velocity_fallback=context.free_body_velocity_fallback,
    )
    for action in context.action_history:
        repeated = torch.as_tensor(
            action, dtype=torch.float32, device=env.device
        ).reshape(1, -1).expand(env.num_envs, -1)
        env.step(repeated)


def _right_eef_pose_relative(env):
    pose = env.scene["right_arm"].data.body_pose_w[0, -1].detach().clone()
    pose[:3] -= env.scene.env_origins[0]
    return pose.cpu().numpy()


def _tree_pose_relative(env):
    pose = env.scene["mug_tree"].data.root_pose_w[0].detach().clone()
    pose[:3] -= env.scene.env_origins[0]
    return pose.cpu().numpy()


def _mug_pose_relative(env):
    pose = env.scene["mug"].data.root_pose_w[0].detach().clone()
    pose[:3] -= env.scene.env_origins[0]
    return pose.cpu().numpy()


def _record_right_eef_reference(runner, context, nominal):
    """Replay source controls once and record the environment-relative EEF pose."""
    import torch

    env = runner.env
    _replay_context(runner, context)
    poses = []
    for action in nominal:
        repeated = torch.as_tensor(
            action, dtype=torch.float32, device=env.device
        ).reshape(1, -1).expand(env.num_envs, -1)
        _, _, terminated, truncated, _ = env.step(repeated)
        if bool(terminated.any()) or bool(truncated.any()):
            raise RuntimeError("Reference EEF replay produced a done signal")
        poses.append(_right_eef_pose_relative(env))
    return np.stack(poses)


def _record_right_eef_keyframe(runner, context, nominal):
    """Replay the demo only to extract its final semantic EEF keyframe."""
    import torch

    env = runner.env
    _replay_context(runner, context)
    for action in nominal:
        repeated = torch.as_tensor(
            action, dtype=torch.float32, device=env.device
        ).reshape(1, -1).expand(env.num_envs, -1)
        _, _, terminated, truncated, _ = env.step(repeated)
        if bool(terminated.any()) or bool(truncated.any()):
            raise RuntimeError("Semantic keyframe replay produced a done signal")
    return _right_eef_pose_relative(env)


def _semantic_reference_trajectory(start_pose, target_pose, horizon):
    """Build a fresh smooth Cartesian path between two semantic keyframes."""
    start_pose = np.asarray(start_pose, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32)
    phase = np.linspace(
        1.0 / horizon, 1.0, horizon, dtype=np.float32
    )
    smooth = phase * phase * (3.0 - 2.0 * phase)
    position = (
        start_pose[None, :3]
        + smooth[:, None] * (target_pose[:3] - start_pose[:3])
    )
    left = start_pose[3:7] / np.linalg.norm(start_pose[3:7])
    right = target_pose[3:7] / np.linalg.norm(target_pose[3:7])
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    if dot > 0.9995:
        quaternion = (
            left[None, :] + smooth[:, None] * (right - left)[None, :]
        )
        quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    else:
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        quaternion = (
            np.sin((1.0 - smooth) * angle)[:, None] * left[None, :]
            + np.sin(smooth * angle)[:, None] * right[None, :]
        ) / np.sin(angle)
    return np.concatenate((position, quaternion), axis=-1).astype(np.float32)


def insert(
    start_pose,
    target_pose,
    branch_rotation,
    horizon,
    *,
    target_position_offset_branch=HANGMUG_INSERT_EEF_POSITION_OFFSET_BRANCH_M,
    target_rotation_offset_branch=(
        HANGMUG_INSERT_EEF_ROTATION_OFFSET_BRANCH_RAD
    ),
    clearance_rotation_offset_branch=(
        HANGMUG_INSERT_CLEARANCE_ROTATION_OFFSET_BRANCH_RAD
    ),
    clearance_offset_branch=None,
    approach_offset_branch=HANGMUG_INSERT_APPROACH_OFFSET_BRANCH_M,
    seat_offset_branch=HANGMUG_INSERT_SEAT_OFFSET_BRANCH_M,
    clearance_fraction=HANGMUG_INSERT_CLEARANCE_FRACTION,
    approach_fraction=HANGMUG_INSERT_APPROACH_FRACTION,
    seat_fraction=HANGMUG_INSERT_SEAT_FRACTION,
):
    """Clear the tree, approach beyond the branch tip, seat inward, and hold.

    The branch frame's positive X axis runs from the branch root toward its
    tip. A positive approach X offset therefore backs the mug handle beyond
    the tip; interpolation to the seated target threads it inward along the
    branch tangent.
    """
    if clearance_offset_branch is None:
        phase_order_valid = 0.0 < approach_fraction < seat_fraction < 1.0
    else:
        phase_order_valid = (
            0.0
            < clearance_fraction
            < approach_fraction
            < seat_fraction
            < 1.0
        )
    if not phase_order_valid:
        raise ValueError(
            "insert fractions must satisfy "
            "0 < clearance < approach < seat < 1"
        )
    start_pose = np.asarray(start_pose, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32).copy()
    branch_rotation = np.asarray(branch_rotation, dtype=np.float32)
    if branch_rotation.shape != (3, 3):
        raise ValueError("branch_rotation must have shape (3, 3)")
    target_pose[:3] += branch_rotation @ np.asarray(
        target_position_offset_branch, dtype=np.float32
    )
    rotation_offset = branch_rotation @ np.asarray(
        target_rotation_offset_branch, dtype=np.float32
    )
    rotation_angle = np.linalg.norm(rotation_offset)
    if rotation_angle > 0.0:
        rotation_axis = rotation_offset / rotation_angle
        rotation_quaternion = np.concatenate(
            (
                np.asarray([np.cos(rotation_angle / 2.0)]),
                rotation_axis * np.sin(rotation_angle / 2.0),
            )
        )
        target_pose[3:7] = _quat_multiply(
            rotation_quaternion, target_pose[3:7]
        )
        target_pose[3:7] /= np.linalg.norm(target_pose[3:7])

    approach = target_pose[:3] + branch_rotation @ np.asarray(
        approach_offset_branch, dtype=np.float32
    )
    seated = target_pose[:3] + branch_rotation @ np.asarray(
        seat_offset_branch, dtype=np.float32
    )
    waypoint_phase = [0.0]
    waypoint_position = [start_pose[:3]]
    if clearance_offset_branch is not None:
        clearance = target_pose[:3] + branch_rotation @ np.asarray(
            clearance_offset_branch, dtype=np.float32
        )
        waypoint_phase.append(clearance_fraction)
        waypoint_position.append(clearance)
    waypoint_phase.extend((approach_fraction, seat_fraction, 1.0))
    waypoint_position.extend((approach, seated, seated))
    waypoint_phase = np.asarray(waypoint_phase, dtype=np.float32)
    waypoint_position = np.stack(waypoint_position)
    phase = np.linspace(
        1.0 / horizon, 1.0, horizon, dtype=np.float32
    )
    position = np.empty((horizon, 3), dtype=np.float32)
    for index, value in enumerate(phase):
        segment = min(
            np.searchsorted(waypoint_phase, value, side="right") - 1,
            len(waypoint_phase) - 2,
        )
        local = (
            (value - waypoint_phase[segment])
            / (waypoint_phase[segment + 1] - waypoint_phase[segment])
        )
        smooth = local * local * (3.0 - 2.0 * local)
        position[index] = (
            waypoint_position[segment]
            + smooth
            * (
                waypoint_position[segment + 1]
                - waypoint_position[segment]
            )
        )

    orientation_phase = [0.0]
    orientation_waypoint = [start_pose[3:7]]
    clearance_rotation = branch_rotation @ np.asarray(
        clearance_rotation_offset_branch, dtype=np.float32
    )
    clearance_angle = np.linalg.norm(clearance_rotation)
    if clearance_offset_branch is not None and clearance_angle > 0.0:
        clearance_axis = clearance_rotation / clearance_angle
        clearance_quaternion = np.concatenate(
            (
                np.asarray([np.cos(clearance_angle / 2.0)]),
                clearance_axis * np.sin(clearance_angle / 2.0),
            )
        )
        orientation_phase.append(clearance_fraction)
        orientation_waypoint.append(
            _quat_multiply(clearance_quaternion, target_pose[3:7])
        )
    orientation_phase.extend((seat_fraction, 1.0))
    orientation_waypoint.extend((target_pose[3:7], target_pose[3:7]))
    orientation_phase = np.asarray(orientation_phase, dtype=np.float32)
    orientation_waypoint = np.asarray(
        orientation_waypoint, dtype=np.float32
    )
    orientation_waypoint /= np.linalg.norm(
        orientation_waypoint, axis=-1, keepdims=True
    )
    orientation = np.empty((horizon, 4), dtype=np.float32)
    for index, value in enumerate(phase):
        segment = min(
            np.searchsorted(orientation_phase, value, side="right") - 1,
            len(orientation_phase) - 2,
        )
        local = (
            (value - orientation_phase[segment])
            / (
                orientation_phase[segment + 1]
                - orientation_phase[segment]
            )
        )
        smooth = local * local * (3.0 - 2.0 * local)
        left = orientation_waypoint[segment]
        right = orientation_waypoint[segment + 1]
        dot = float(np.dot(left, right))
        if dot < 0.0:
            right = -right
            dot = -dot
        if dot > 0.9995:
            quaternion = left + smooth * (right - left)
        else:
            angle = np.arccos(np.clip(dot, -1.0, 1.0))
            quaternion = (
                np.sin((1.0 - smooth) * angle) * left
                + np.sin(smooth * angle) * right
            ) / np.sin(angle)
        orientation[index] = quaternion / np.linalg.norm(quaternion)
    return np.concatenate((position, orientation), axis=-1).astype(np.float32)


def release(
    controls,
    *,
    open_value=HANGMUG_RELEASE_OPEN_VALUE,
    start_fraction=HANGMUG_RELEASE_START_FRACTION,
    end_fraction=HANGMUG_RELEASE_END_FRACTION,
):
    """Smoothly open the right gripper, then hold it open."""
    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError(
            "release fractions must satisfy 0 <= start < end <= 1"
        )
    controls = np.asarray(controls, dtype=np.float32).copy()
    horizon = len(controls)
    phase = (np.arange(horizon, dtype=np.float32) + 1.0) / horizon
    blend = np.clip(
        (phase - start_fraction) / (end_fraction - start_fraction),
        0.0,
        1.0,
    )
    blend = blend * blend * (3.0 - 2.0 * blend)
    closed_value = controls[0, 13]
    controls[:, 13] = (
        closed_value + blend * (float(open_value) - closed_value)
    )
    return controls


def _semantic_base_controls(nominal, target_name):
    """Keep only stage-level hold/release intent from the demonstration."""
    controls = np.asarray(nominal, dtype=np.float32).copy()
    controls[:, :6] = controls[0, :6]
    controls[:, 6] = controls[0, 6]
    if target_name != "hang_complete":
        controls[:, 13] = controls[0, 13]
    return controls


def _quaternion_error(actual, target):
    actual = actual / np.linalg.norm(actual, axis=-1, keepdims=True)
    target = target / np.linalg.norm(target, axis=-1, keepdims=True)
    dots = np.abs(np.sum(actual * target, axis=-1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def _expand_control_corrections(
    knots,
    nominal,
    *,
    max_action_delta,
    right_arm_only=False,
):
    """Interpolate smooth joint-target corrections onto the rollout horizon."""
    knots = np.asarray(knots, dtype=np.float32)
    nominal = np.asarray(nominal, dtype=np.float32)
    corrections = _interpolate_knots(knots, nominal.shape[0])
    corrections = np.clip(
        corrections, -max_action_delta, max_action_delta
    )
    corrections[:, :, (6, 13)] = 0.0
    if right_arm_only:
        corrections[:, :, :7] = 0.0
    return nominal[None, :, :] + corrections


def _interpolate_knots(knots, horizon):
    knots = np.asarray(knots, dtype=np.float32)
    source = np.linspace(0.0, 1.0, knots.shape[1])
    target = np.linspace(0.0, 1.0, horizon)
    return np.stack(
        [
            np.stack(
                [
                    np.interp(target, source, rollout[:, action])
                    for action in range(knots.shape[2])
                ],
                axis=-1,
            )
            for rollout in knots
        ]
    )


def _task_program_knots(num_knots, translation_start, translation_goal):
    phase = (
        np.ones(1, dtype=np.float32)
        if num_knots == 1
        else np.linspace(0.0, 1.0, num_knots, dtype=np.float32)
    )
    smooth = phase * phase * (3.0 - 2.0 * phase)
    knots = np.zeros((num_knots, 6), dtype=np.float32)
    start = np.asarray(translation_start, dtype=np.float32)
    goal = np.asarray(translation_goal, dtype=np.float32)
    knots[:, :3] = start + smooth[:, None] * (goal - start)
    return knots


def _expand_task_space_program(
    knots,
    nominal,
    program_nominal,
    *,
    max_translation_delta,
    max_rotation_delta,
):
    knots = np.asarray(knots, dtype=np.float32)
    delta = knots - program_nominal[None, :, :]
    delta[:, :, :3] = np.clip(
        delta[:, :, :3],
        -max_translation_delta,
        max_translation_delta,
    )
    delta[:, :, 3:] = np.clip(
        delta[:, :, 3:],
        -max_rotation_delta,
        max_rotation_delta,
    )
    residuals = _interpolate_knots(
        program_nominal[None, :, :] + delta, nominal.shape[0]
    )
    base = np.broadcast_to(
        nominal[None, :, :],
        (knots.shape[0], *nominal.shape),
    )
    return np.concatenate((base, residuals), axis=-1)


def _torch_quat_multiply(left, right):
    import torch

    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_multiply(left, right):
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_rotate(quaternion, vector):
    quaternion = quaternion / np.linalg.norm(
        quaternion, axis=-1, keepdims=True
    )
    q_vector = quaternion[..., 1:4]
    twice_cross = 2.0 * np.cross(q_vector, vector)
    return (
        vector
        + quaternion[..., :1] * twice_cross
        + np.cross(q_vector, twice_cross)
    )


def _pose_compose(left, right):
    """Compose two ``[position, wxyz]`` poses."""
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    result = np.empty(7, dtype=np.float32)
    result[:3] = left[:3] + _quat_rotate(left[3:7], right[:3])
    result[3:7] = _quat_multiply(left[3:7], right[3:7])
    result[3:7] /= np.linalg.norm(result[3:7])
    return result


def _pose_inverse(pose):
    """Invert a ``[position, wxyz]`` pose."""
    pose = np.asarray(pose, dtype=np.float32)
    result = np.empty(7, dtype=np.float32)
    result[3:7] = pose[3:7] / np.linalg.norm(pose[3:7])
    result[4:7] *= -1.0
    result[:3] = _quat_rotate(result[3:7], -pose[:3])
    return result


def _eef_target_for_mug_target(current_eef_pose, current_mug_pose, target_mug_pose):
    """Preserve the live grasp transform while moving the mug to a target."""
    eef_to_mug = _pose_compose(
        _pose_inverse(current_eef_pose), current_mug_pose
    )
    return _pose_compose(target_mug_pose, _pose_inverse(eef_to_mug))


def _quat_to_matrix(quaternion):
    quaternion = quaternion / np.linalg.norm(
        quaternion, axis=-1, keepdims=True
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _matrix_to_quat(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        first = axis
        second = (axis + 1) % 3
        third = (axis + 2) % 3
        scale = np.sqrt(
            1.0
            + matrix[first, first]
            - matrix[second, second]
            - matrix[third, third]
        ) * 2.0
        vector = np.zeros(3, dtype=np.float32)
        vector[first] = 0.25 * scale
        vector[second] = (
            matrix[second, first] + matrix[first, second]
        ) / scale
        vector[third] = (
            matrix[third, first] + matrix[first, third]
        ) / scale
        quaternion = np.concatenate(
            (
                np.asarray(
                    [
                        (
                            matrix[third, second]
                            - matrix[second, third]
                        )
                        / scale
                    ]
                ),
                vector,
            )
        )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion.astype(np.float32)


def _rotation_matrix_error(actual, target):
    relative = np.einsum("...ji,...jk->...ik", actual, target)
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _branch_frame(points):
    points = np.asarray(points, dtype=np.float32).reshape(3, 3)
    x_axis = points[1] - points[0]
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1.0e-6:
        raise ValueError("First two branch correspondence points must differ")
    x_axis /= x_norm
    z_axis = np.cross(x_axis, points[2] - points[0])
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1.0e-6:
        raise ValueError("Branch correspondence points must be non-collinear")
    z_axis /= z_norm
    y_axis = np.cross(z_axis, x_axis)
    return points[0], np.stack((x_axis, y_axis, z_axis), axis=-1)


def _correspond_pose_between_branches(
    pose,
    source_tree_pose,
    target_tree_pose,
    source_branch_points,
    target_branch_points,
):
    """Transport a world-relative pose through aligned branch frames."""
    pose = np.asarray(pose, dtype=np.float32)
    source_tree_pose = np.asarray(source_tree_pose, dtype=np.float32)
    target_tree_pose = np.asarray(target_tree_pose, dtype=np.float32)
    source_origin, source_branch_rotation = _branch_frame(
        source_branch_points
    )
    target_origin, target_branch_rotation = _branch_frame(
        target_branch_points
    )
    source_tree_rotation = _quat_to_matrix(source_tree_pose[3:7])
    target_tree_rotation = _quat_to_matrix(target_tree_pose[3:7])
    source_branch_rotation_w = (
        source_tree_rotation @ source_branch_rotation
    )
    target_branch_rotation_w = (
        target_tree_rotation @ target_branch_rotation
    )
    source_branch_origin_w = (
        source_tree_pose[:3] + source_tree_rotation @ source_origin
    )
    target_branch_origin_w = (
        target_tree_pose[:3] + target_tree_rotation @ target_origin
    )
    position_branch = source_branch_rotation_w.T @ (
        pose[:3] - source_branch_origin_w
    )
    rotation_branch = (
        source_branch_rotation_w.T @ _quat_to_matrix(pose[3:7])
    )
    target_pose = np.empty(7, dtype=np.float32)
    target_pose[:3] = (
        target_branch_origin_w
        + target_branch_rotation_w @ position_branch
    )
    target_pose[3:7] = _matrix_to_quat(
        target_branch_rotation_w @ rotation_branch
    )
    return target_pose


def _mug_in_tree_frame(rows):
    mug_position = rows[..., :3]
    mug_quaternion = rows[..., 3:7]
    tree_position = rows[..., 29:32]
    tree_quaternion = rows[..., 32:36]
    mug_quaternion = mug_quaternion / np.linalg.norm(
        mug_quaternion, axis=-1, keepdims=True
    )
    tree_quaternion = tree_quaternion / np.linalg.norm(
        tree_quaternion, axis=-1, keepdims=True
    )
    tree_inverse = tree_quaternion.copy()
    tree_inverse[..., 1:4] *= -1.0
    return (
        _quat_rotate(tree_inverse, mug_position - tree_position),
        _quat_multiply(tree_inverse, mug_quaternion),
    )


def _mug_in_branch_frame(rows, points):
    mug_position, mug_quaternion = _mug_in_tree_frame(rows)
    origin, branch_rotation = _branch_frame(points)
    position = np.einsum(
        "ij,...j->...i",
        branch_rotation.T,
        mug_position - origin,
    )
    mug_rotation = _quat_to_matrix(mug_quaternion)
    rotation = np.einsum(
        "ij,...jk->...ik", branch_rotation.T, mug_rotation
    )
    return position, rotation


def _objective_components(
    states,
    sensors,
    controls,
    *,
    reference,
    nominal,
    control_reference=None,
    keyframe_offset,
    target_name,
    source_branch_points=None,
    target_branch_points=None,
    right_joint_path_weight=0.0,
    right_joint_accel_weight=0.0,
    right_joint_jerk_weight=0.0,
):
    world_position_error = np.linalg.norm(
        states[:, :, :3] - reference[None, :, :3], axis=-1
    )
    world_rotation_error = _quaternion_error(
        states[:, :, 3:7], reference[None, :, 3:7]
    )
    tree_target = target_name in (
        "tree_approach",
        "inserted_held",
        "hang_complete",
    )
    if tree_target and source_branch_points is not None:
        relative_position, relative_rotation = _mug_in_branch_frame(
            states, target_branch_points
        )
        reference_position, reference_rotation = _mug_in_branch_frame(
            reference, source_branch_points
        )
        position_error = np.linalg.norm(
            relative_position - reference_position[None, :, :],
            axis=-1,
        )
        rotation_error = _rotation_matrix_error(
            relative_rotation,
            reference_rotation[None, :, :, :],
        )
        position_error_vector = (
            relative_position - reference_position[None, :, :]
        )
        target_frame = "mug_relative_to_corresponded_branch"
    elif tree_target:
        relative_position, relative_quaternion = _mug_in_tree_frame(states)
        reference_position, reference_quaternion = _mug_in_tree_frame(reference)
        position_error = np.linalg.norm(
            relative_position - reference_position[None, :, :],
            axis=-1,
        )
        rotation_error = _quaternion_error(
            relative_quaternion, reference_quaternion[None, :, :]
        )
        position_error_vector = (
            relative_position - reference_position[None, :, :]
        )
        target_frame = "mug_relative_to_tree_root"
    else:
        position_error = world_position_error
        rotation_error = world_rotation_error
        position_error_vector = (
            states[:, :, :3] - reference[None, :, :3]
        )
        target_frame = "environment_origin"
    left_joint_error = np.sqrt(
        np.mean(
            np.square(
                states[:, :, 13:21] - reference[None, :, 13:21]
            ),
            axis=-1,
        )
    )
    right_joint_error = np.sqrt(
        np.mean(
            np.square(
                states[:, :, 21:29] - reference[None, :, 21:29]
            ),
            axis=-1,
        )
    )
    if control_reference is None:
        control_reference = nominal
    action_delta = np.sqrt(
        np.mean(
            np.square(
                controls - np.asarray(control_reference)[None, :, :]
            ),
            axis=(1, 2),
        )
    )
    right_joint_positions = states[:, :, 21:27]
    right_joint_velocity = np.diff(right_joint_positions, axis=1)
    right_joint_acceleration = np.diff(right_joint_velocity, axis=1)
    right_joint_jerk = np.diff(right_joint_acceleration, axis=1)

    def trajectory_cost(value):
        if value.shape[1] == 0:
            return np.zeros(value.shape[0], dtype=np.float32)
        return np.linalg.norm(value, axis=-1).sum(axis=1)

    right_joint_path = trajectory_cost(right_joint_velocity)
    right_joint_accel = trajectory_cost(right_joint_acceleration)
    right_joint_jerk = trajectory_cost(right_joint_jerk)
    left_grasp = sensors[:, :, 2]
    left_assist = sensors[:, :, 3]
    right_grasp = sensors[:, :, 6]
    stage1 = sensors[:, :, 7]
    stage2 = sensors[:, :, 8]
    stage3 = sensors[:, :, 9]
    mug_linear_speed = np.linalg.norm(states[:, :, 7:10], axis=-1)
    post_keyframe = slice(keyframe_offset, None)
    rewards = (
        -80.0 * position_error[:, keyframe_offset]
        -4.0 * rotation_error[:, keyframe_offset]
        -1.0 * left_joint_error[:, keyframe_offset]
        -1.0 * right_joint_error[:, keyframe_offset]
        -5.0 * position_error.mean(axis=1)
        -0.5 * rotation_error.mean(axis=1)
        -2.0 * action_delta
        -right_joint_path_weight * right_joint_path
        -right_joint_accel_weight * right_joint_accel
        -right_joint_jerk_weight * right_joint_jerk
    )
    if target_name == "left_grasp":
        target_success = left_grasp
        rewards += (
            2.0 * left_grasp[:, keyframe_offset]
            +1.0 * left_assist[:, keyframe_offset]
            +1.0 * left_grasp[:, post_keyframe].mean(axis=1)
        )
    elif target_name == "right_grasp":
        target_success = right_grasp
        rewards += (
            2.0 * right_grasp[:, keyframe_offset]
            +1.0 * left_grasp[:, keyframe_offset]
            +1.0 * right_grasp[:, post_keyframe].mean(axis=1)
            +0.5 * stage1[:, keyframe_offset]
        )
    elif target_name == "handover_latched":
        target_success = stage2
        rewards += (
            3.0 * stage2[:, keyframe_offset]
            +2.0 * right_grasp[:, keyframe_offset]
            +1.0 * (1.0 - left_grasp[:, keyframe_offset])
            +1.0 * right_grasp[:, post_keyframe].mean(axis=1)
        )
    elif target_name in ("tree_approach", "inserted_held"):
        position_tolerance = (
            TREE_APPROACH_POSITION_TOLERANCE_M
            if target_name == "tree_approach"
            else TREE_INSERTION_POSITION_TOLERANCE_M
        )
        target_success = (
            stage2
            * right_grasp
            * (position_error <= position_tolerance)
        )
        if target_name == "inserted_held":
            target_success *= (
                rotation_error <= TREE_INSERTION_ROTATION_TOLERANCE_RAD
            )
        rewards += (
            3.0 * stage2[:, keyframe_offset]
            +3.0 * right_grasp[:, keyframe_offset]
            +1.0 * (1.0 - left_grasp[:, keyframe_offset])
            +2.0 * right_grasp[:, post_keyframe].mean(axis=1)
        )
        if target_name == "inserted_held":
            rewards += (
                -40.0 * position_error[:, keyframe_offset]
                -4.0 * rotation_error[:, keyframe_offset]
                +12.0 * target_success[:, keyframe_offset]
                +4.0 * target_success[:, post_keyframe].mean(axis=1)
            )
    elif target_name == "hang_complete":
        target_success = (
            stage3
            * (mug_linear_speed <= HANG_SPEED_TOLERANCE_M_S)
            * (1.0 - left_grasp)
            * (1.0 - right_grasp)
        )
        rewards += (
            8.0 * stage3[:, keyframe_offset]
            +4.0 * stage3[:, post_keyframe].mean(axis=1)
            +2.0 * stage2[:, keyframe_offset]
        )
    else:
        raise ValueError(f"Unsupported target-name: {target_name}")
    if target_name == "hang_complete":
        # The task's stage-3 latch already requires 30 stable steps. Requiring
        # another 30 latched frames here would double-count stabilization.
        acceptance_fraction = target_success[:, keyframe_offset]
        rewards += 12.0 * acceptance_fraction
    elif target_name == "inserted_held":
        acceptance_window = slice(
            max(
                0,
                keyframe_offset - TREE_INSERTION_STABILITY_STEPS + 1,
            ),
            keyframe_offset + 1,
        )
        acceptance_fraction = target_success[:, acceptance_window].mean(axis=1)
        rewards += 12.0 * acceptance_fraction
    else:
        acceptance_fraction = target_success[:, keyframe_offset]
    return {
        "rewards": rewards,
        "target_frame": target_frame,
        "keyframe_position_error_m": position_error[:, keyframe_offset],
        "keyframe_position_error_vector_m": position_error_vector[
            :, keyframe_offset
        ],
        "keyframe_rotation_error_rad": rotation_error[:, keyframe_offset],
        "keyframe_left_joint_rms_rad": left_joint_error[:, keyframe_offset],
        "keyframe_right_joint_rms_rad": right_joint_error[:, keyframe_offset],
        "action_delta_rms": action_delta,
        "right_joint_path_l2": right_joint_path,
        "right_joint_accel_l2": right_joint_accel,
        "right_joint_jerk_l2": right_joint_jerk,
        "keyframe_target_success": target_success[:, keyframe_offset] > 0.5,
        "keyframe_left_grasp": left_grasp[:, keyframe_offset] > 0.5,
        "keyframe_right_grasp": right_grasp[:, keyframe_offset] > 0.5,
        "keyframe_left_assist": left_assist[:, keyframe_offset] > 0.5,
        "keyframe_stage2": stage2[:, keyframe_offset] > 0.5,
        "post_keyframe_target_fraction": target_success[
            :, post_keyframe
        ].mean(axis=1),
        "acceptance_window_fraction": acceptance_fraction,
        "keyframe_mug_linear_speed_m_s": mug_linear_speed[:, keyframe_offset],
    }


def _group_summary(name, rows, components):
    rewards = components["rewards"][rows]
    position = components["keyframe_position_error_m"][rows]
    position_vector = components["keyframe_position_error_vector_m"][rows]
    rotation = components["keyframe_rotation_error_rad"][rows]
    action_delta = components["action_delta_rms"][rows]
    right_joint_path = components["right_joint_path_l2"][rows]
    right_joint_accel = components["right_joint_accel_l2"][rows]
    right_joint_jerk = components["right_joint_jerk_l2"][rows]
    target_success = components["keyframe_target_success"][rows]
    left_grasp = components["keyframe_left_grasp"][rows]
    right_grasp = components["keyframe_right_grasp"][rows]
    left_assist = components["keyframe_left_assist"][rows]
    stage2 = components["keyframe_stage2"][rows]
    retention = components["post_keyframe_target_fraction"][rows]
    acceptance = components["acceptance_window_fraction"][rows]
    speed = components["keyframe_mug_linear_speed_m_s"][rows]
    return {
        "name": name,
        "count": int(len(rewards)),
        "reward_mean": float(rewards.mean()),
        "reward_max": float(rewards.max()),
        "keyframe_position_error_m_mean": float(position.mean()),
        "keyframe_position_error_m_max": float(position.max()),
        "keyframe_position_error_vector_m_mean": (
            position_vector.mean(axis=0).tolist()
        ),
        "keyframe_rotation_error_rad_mean": float(rotation.mean()),
        "keyframe_rotation_error_rad_max": float(rotation.max()),
        "action_delta_rms_mean": float(action_delta.mean()),
        "right_joint_path_l2_mean": float(right_joint_path.mean()),
        "right_joint_path_l2_max": float(right_joint_path.max()),
        "right_joint_accel_l2_mean": float(right_joint_accel.mean()),
        "right_joint_accel_l2_max": float(right_joint_accel.max()),
        "right_joint_jerk_l2_mean": float(right_joint_jerk.mean()),
        "right_joint_jerk_l2_max": float(right_joint_jerk.max()),
        "keyframe_target_success_count": int(target_success.sum()),
        "keyframe_left_grasp_count": int(left_grasp.sum()),
        "keyframe_right_grasp_count": int(right_grasp.sum()),
        "keyframe_left_assist_count": int(left_assist.sum()),
        "keyframe_stage2_count": int(stage2.sum()),
        "post_keyframe_target_fraction_mean": float(retention.mean()),
        "acceptance_window_fraction_mean": float(acceptance.mean()),
        "acceptance_success_count": int((acceptance >= 1.0).sum()),
        "keyframe_mug_linear_speed_m_s_mean": float(speed.mean()),
        "keyframe_mug_linear_speed_m_s_max": float(speed.max()),
    }


def _subtask_reached(group):
    return group["acceptance_success_count"] == group["count"]


def main():
    args = _parser()
    if (args.source_branch_points is None) != (
        args.target_branch_points is None
    ):
        raise ValueError(
            "source-branch-points and target-branch-points must be supplied together"
        )
    source_branch_points = (
        None
        if args.source_branch_points is None
        else np.asarray(args.source_branch_points, dtype=np.float32).reshape(3, 3)
    )
    target_branch_points = (
        None
        if args.target_branch_points is None
        else np.asarray(args.target_branch_points, dtype=np.float32).reshape(3, 3)
    )
    keyframe_offset = args.target_state - args.start_state - 1
    if not 0 <= keyframe_offset < args.horizon:
        raise ValueError("target-state must fall inside the rollout horizon")
    if args.num_rollouts < 3:
        raise ValueError("num-rollouts must be at least three")
    num_control_knots = args.control_knots or args.horizon
    if not 1 <= num_control_knots <= args.horizon:
        raise ValueError("control-knots must be within [1, horizon]")
    if not (
        0.0
        < args.insert_clearance_fraction
        < args.insert_approach_fraction
        < args.insert_seat_fraction
        < 1.0
    ):
        raise ValueError(
            "insert fractions must satisfy "
            "0 < clearance < approach < seat < 1"
        )
    if not (
        0.0
        <= args.release_start_fraction
        < args.release_end_fraction
        <= 1.0
    ):
        raise ValueError(
            "release fractions must satisfy 0 <= start < end <= 1"
        )
    motion_weights = (
        args.right_joint_path_weight,
        args.right_joint_accel_weight,
        args.right_joint_jerk_weight,
    )
    if any(weight < 0.0 for weight in motion_weights):
        raise ValueError("right-joint motion weights must be nonnegative")
    sys.path.insert(0, os.path.abspath(args.gear_repo))

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(
        {"headless": True, "device": "cpu", "enable_cameras": False}
    ).app
    runner = None
    try:
        from dc_study.planning import (
            RigidObjectMpcState,
            create_cpu_batched_planning_runner,
        )
        from judo.optimizers.cem import (
            CrossEntropyMethod,
            CrossEntropyMethodConfig,
        )
        from judo_isaaclab import (
            BranchContext,
            DampedLeastSquaresPoseTrackingAdapter,
            DampedLeastSquaresTaskSpaceAdapter,
            HistoryConditionedIsaacLabBackend,
            JudoIsaacLabMPC,
        )

        np.random.seed(args.seed)
        assets = {
            "mug": _target_mug_path(args),
            "mug_tree": _target_tree_path(args),
        }
        assist_config = {
            "left": {
                "arm": "left_arm",
                "target": {"object": "mug"},
                "grasp_delay_s": 0.0,
                "mechanism": "friction",
                "friction": {
                    "high": args.friction_high,
                    "low": args.friction_low,
                },
            }
        }
        runner = create_cpu_batched_planning_runner(
            task_name="HangMugOnTree-v0",
            assets_instance_paths=assets,
            num_envs=args.num_rollouts,
            observation_modalities=["proprioception"],
            enable_cameras=False,
            grasp_assist_config=assist_config,
            objects_randomization=None,
            init_joint_pos_randomization=0.0,
            enable_gripper_grasp_clamp=False,
            planning_substep_contact_sensors=True,
        )
        runner.env.reset(warm_up=False, seed=args.seed)
        checkpoint, history, nominal, reference, source_history = _load_demo(
            args, runner.env.device
        )
        objective_reference = reference
        insert_anchor_reference = None
        if args.target_name == "inserted_held":
            insert_anchor_reference = _load_demo_state_row(
                args, args.insert_anchor_state
            )
            objective_reference = reference.copy()
            objective_reference[:, :13] = insert_anchor_reference[:13]
            objective_reference[:, 29:36] = insert_anchor_reference[29:36]
        context = BranchContext(
            checkpoint_state=checkpoint,
            action_history=history,
            rigid_object_states={
                "mug": RigidObjectMpcState(False),
                "mug_tree": RigidObjectMpcState(False),
            },
            is_relative=True,
        )
        task_translation_goal = args.task_translation_goal
        if args.search_space == "task" and task_translation_goal is None:
            task_translation_goal = np.asarray(
                args.tree_offset_xyz, dtype=np.float32
            )
            task_translation_goal[2] += _tree_root_z_adjustment(args)
            if source_branch_points is not None:
                task_translation_goal += (
                    target_branch_points[0] - source_branch_points[0]
                )
        reference_eef_poses = None
        target_mug_pose = None
        base_controls = nominal
        if args.search_space == "task" and args.task_controller in (
            "pose_tracking",
            "semantic_pose",
        ):
            reference_context = BranchContext(
                checkpoint_state=checkpoint,
                action_history=source_history,
                rigid_object_states=context.rigid_object_states,
                is_relative=True,
            )
            if args.task_controller == "semantic_pose":
                if args.target_name == "inserted_held":
                    _replay_context(runner, context)
                    current_eef_pose = _right_eef_pose_relative(runner.env)
                    current_mug_pose = _mug_pose_relative(runner.env)
                    target_tree_pose = _tree_pose_relative(runner.env)
                    if target_branch_points is not None:
                        target_mug_pose = _correspond_pose_between_branches(
                            insert_anchor_reference[:7],
                            insert_anchor_reference[29:36],
                            target_tree_pose,
                            source_branch_points,
                            target_branch_points,
                        )
                    else:
                        mug_in_source_tree = _pose_compose(
                            _pose_inverse(
                                insert_anchor_reference[29:36]
                            ),
                            insert_anchor_reference[:7],
                        )
                        target_mug_pose = _pose_compose(
                            target_tree_pose, mug_in_source_tree
                        )
                    target_eef_pose = _eef_target_for_mug_target(
                        current_eef_pose,
                        current_mug_pose,
                        target_mug_pose,
                    )
                else:
                    semantic_nominal = nominal
                    if args.semantic_anchor_state is not None:
                        semantic_nominal = _load_demo_actions(
                            args,
                            args.start_state,
                            args.semantic_anchor_state,
                        )
                    target_eef_pose = _record_right_eef_keyframe(
                        runner, reference_context, semantic_nominal
                    )
                    _replay_context(runner, context)
                    current_eef_pose = _right_eef_pose_relative(runner.env)
                    target_tree_pose = _tree_pose_relative(runner.env)
                    if target_branch_points is not None:
                        target_eef_pose = _correspond_pose_between_branches(
                            target_eef_pose,
                            reference[-1, 29:36],
                            target_tree_pose,
                            source_branch_points,
                            target_branch_points,
                        )
                    else:
                        target_eef_pose[:3] += np.asarray(
                            task_translation_goal, dtype=np.float32
                        )
                if (
                    args.target_name == "inserted_held"
                    and target_branch_points is not None
                ):
                    target_branch_rotation = _branch_frame(
                        target_branch_points
                    )[1]
                    branch_rotation_world = (
                        _quat_to_matrix(target_tree_pose[3:7])
                        @ target_branch_rotation
                    )
                    reference_eef_poses = insert(
                        current_eef_pose,
                        target_eef_pose,
                        branch_rotation_world,
                        args.horizon,
                        target_position_offset_branch=(
                            args.insert_eef_position_offset_branch
                        ),
                        target_rotation_offset_branch=(
                            args.insert_eef_rotation_offset_branch
                        ),
                        clearance_rotation_offset_branch=(
                            args.insert_clearance_rotation_offset_branch
                        ),
                        clearance_offset_branch=(
                            args.insert_clearance_offset_branch
                        ),
                        approach_offset_branch=(
                            args.insert_approach_offset_branch
                        ),
                        seat_offset_branch=args.insert_seat_offset_branch,
                        clearance_fraction=args.insert_clearance_fraction,
                        approach_fraction=args.insert_approach_fraction,
                        seat_fraction=args.insert_seat_fraction,
                    )
                else:
                    reference_eef_poses = _semantic_reference_trajectory(
                        current_eef_pose,
                        target_eef_pose,
                        args.horizon,
                    )
                base_controls = _semantic_base_controls(
                    nominal, args.target_name
                )
                if (
                    args.target_name == "hang_complete"
                    and args.release_open_value is not None
                ):
                    base_controls = release(
                        base_controls,
                        open_value=args.release_open_value,
                        start_fraction=args.release_start_fraction,
                        end_fraction=args.release_end_fraction,
                    )
            else:
                reference_eef_poses = _record_right_eef_reference(
                    runner, reference_context, nominal
                )
            task_adapter = DampedLeastSquaresPoseTrackingAdapter(
                reference_poses=reference_eef_poses,
                damping=args.dls_damping,
                max_joint_delta=args.dls_max_joint_delta,
                max_position_step=args.dls_max_position_step,
                max_rotation_step=args.dls_max_rotation_step,
            )
        elif args.search_space == "task":
            task_adapter = DampedLeastSquaresTaskSpaceAdapter(
                damping=args.dls_damping,
                max_joint_delta=args.dls_max_joint_delta,
            )
        else:
            task_adapter = None
        backend = HistoryConditionedIsaacLabBackend(
            runner,
            state_encoder=_encode_state,
            sensor_encoder=_encode_sensors,
            candidate_action_adapter=task_adapter,
        )
        optimizer_dim = 6 if task_adapter is not None else nominal.shape[1]
        optimizer = CrossEntropyMethod(
            CrossEntropyMethodConfig(
                num_rollouts=args.num_rollouts,
                num_nodes=num_control_knots,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                num_elites=args.num_elites,
            ),
            nu=optimizer_dim,
        )

        if task_adapter is None:
            correction_nominal = np.zeros(
                (num_control_knots, nominal.shape[1]), dtype=np.float32
            )
        elif args.task_controller == "semantic_pose":
            correction_nominal = np.zeros(
                (num_control_knots, optimizer_dim), dtype=np.float32
            )
        else:
            correction_nominal = _task_program_knots(
                num_control_knots,
                args.task_translation_start,
                task_translation_goal,
            )

        def expand(knots):
            if task_adapter is None:
                controls = _expand_control_corrections(
                    knots,
                    base_controls,
                    max_action_delta=args.max_action_delta,
                    right_arm_only=args.target_name
                    in ("tree_approach", "inserted_held", "hang_complete"),
                )
            else:
                controls = _expand_task_space_program(
                    knots,
                    base_controls,
                    correction_nominal,
                    max_translation_delta=args.max_task_translation_delta,
                    max_rotation_delta=args.max_task_rotation_delta,
                )
            if args.target_name == "inserted_held":
                controls[:, : keyframe_offset + 1, 13] = base_controls[0, 13]
            return controls

        objective_control_reference = expand(
            correction_nominal[None, ...]
        )[0]
        iteration_summaries = []

        def objective(states, sensors, controls):
            components = _objective_components(
                states,
                sensors,
                controls,
                reference=objective_reference,
                nominal=base_controls,
                control_reference=objective_control_reference,
                keyframe_offset=keyframe_offset,
                target_name=args.target_name,
                source_branch_points=source_branch_points,
                target_branch_points=target_branch_points,
                right_joint_path_weight=args.right_joint_path_weight,
                right_joint_accel_weight=args.right_joint_accel_weight,
                right_joint_jerk_weight=args.right_joint_jerk_weight,
            )
            iteration_summaries.append(
                {
                    "iteration": len(iteration_summaries),
                    "reward_max": float(components["rewards"].max()),
                    "reward_mean": float(components["rewards"].mean()),
                    "keyframe_target_success_count": int(
                        components["keyframe_target_success"].sum()
                    ),
                    "keyframe_left_grasp_count": int(
                        components["keyframe_left_grasp"].sum()
                    ),
                    "keyframe_right_grasp_count": int(
                        components["keyframe_right_grasp"].sum()
                    ),
                    "keyframe_stage2_count": int(
                        components["keyframe_stage2"].sum()
                    ),
                    "best_keyframe_position_error_m": float(
                        components["keyframe_position_error_m"][
                            np.argmax(components["rewards"])
                        ]
                    ),
                }
            )
            return components["rewards"]

        mpc = JudoIsaacLabMPC(
            optimizer,
            backend,
            objective,
            num_iterations=args.num_iterations,
            duplicate_nominal=args.duplicate_nominal,
            candidate_repeats=args.candidate_repeats,
            candidate_repeat_reducer=args.candidate_repeat_reducer,
            min_improvement=0.01,
            noise_std_multiplier=2.0,
            control_expander=expand,
        )
        plan = mpc.plan(context, correction_nominal)

        best_sample = plan.best_sampled_knots
        group_size = args.num_rollouts // 3
        nominal_rows = slice(0, group_size)
        optimized_rows = slice(group_size, 2 * group_size)
        best_rows = slice(2 * group_size, args.num_rollouts)
        evaluation_knots = np.empty(
            (args.num_rollouts, num_control_knots, optimizer_dim),
            dtype=np.float64,
        )
        evaluation_knots[nominal_rows] = correction_nominal
        evaluation_knots[optimized_rows] = plan.optimized_knots
        evaluation_knots[best_rows] = best_sample
        evaluation_controls = expand(evaluation_knots)
        states, sensors, _ = backend.rollout(
            np.empty(0), evaluation_controls
        )
        evaluation = _objective_components(
            states,
            sensors,
            evaluation_controls,
            reference=objective_reference,
            nominal=base_controls,
            control_reference=objective_control_reference,
            keyframe_offset=keyframe_offset,
            target_name=args.target_name,
            source_branch_points=source_branch_points,
            target_branch_points=target_branch_points,
            right_joint_path_weight=args.right_joint_path_weight,
            right_joint_accel_weight=args.right_joint_accel_weight,
            right_joint_jerk_weight=args.right_joint_jerk_weight,
        )
        groups = {
            "nominal": _group_summary(
                "nominal", nominal_rows, evaluation
            ),
            "optimized_mean": _group_summary(
                "optimized_mean", optimized_rows, evaluation
            ),
            "best_sample": _group_summary(
                "best_sample", best_rows, evaluation
            ),
        }
        executed_evaluation_controls = (
            backend.last_executed_candidate_actions
        )
        optimized_executed_actions = executed_evaluation_controls[
            optimized_rows.start
        ]
        best_executed_actions = executed_evaluation_controls[best_rows.start]
        best_group = groups["best_sample"]
        reached = _subtask_reached(best_group)
        result = {
            "status": "passed" if reached else "failed",
            "source": {
                "episode": args.episode,
                "checkpoint_state": (
                    args.start_state
                    if args.checkpoint_state is None
                    else args.checkpoint_state
                ),
                "history_steps": int(history.shape[0]),
                "history_control_overrides": list(
                    args.history_controls_npz
                ),
                "start_state": args.start_state,
                "target_state": args.target_state,
                "target_name": args.target_name,
                "insert_anchor_state": (
                    args.insert_anchor_state
                    if args.target_name == "inserted_held"
                    else None
                ),
                "semantic_anchor_state": args.semantic_anchor_state,
                "target_frame": evaluation["target_frame"],
                "acceptance": "strict_geometric_subtask_success",
                "hang_acceptance": {
                    "existing_task_stage3": True,
                    "both_grippers_released": True,
                    "speed_tolerance_m_s": HANG_SPEED_TOLERANCE_M_S,
                    "task_stability_steps": HANG_STABILITY_STEPS,
                    "branch_relative_pose": "diagnostic_only",
                },
                "tree_acceptance": {
                    "approach_position_tolerance_m": (
                        TREE_APPROACH_POSITION_TOLERANCE_M
                    ),
                    "insertion_position_tolerance_m": (
                        TREE_INSERTION_POSITION_TOLERANCE_M
                    ),
                    "insertion_rotation_tolerance_rad": (
                        TREE_INSERTION_ROTATION_TOLERANCE_RAD
                    ),
                    "insertion_consecutive_steps": (
                        TREE_INSERTION_STABILITY_STEPS
                    ),
                },
                "right_gripper_held_through_target": (
                    args.target_name == "inserted_held"
                ),
                "tree_offset_xyz_m": list(args.tree_offset_xyz),
                "tree_yaw_deg": args.tree_yaw_deg,
                "source_mug": _source_mug_path(args),
                "target_mug": _target_mug_path(args),
                "source_mug_tree": _source_tree_path(args),
                "target_mug_tree": _target_tree_path(args),
                "tree_root_z_adjustment_m": _tree_root_z_adjustment(args),
                "source_branch_points_tree_local": (
                    None
                    if source_branch_points is None
                    else source_branch_points.tolist()
                ),
                "target_branch_points_tree_local": (
                    None
                    if target_branch_points is None
                    else target_branch_points.tolist()
                ),
                "horizon": args.horizon,
                "control_knots": num_control_knots,
                "control_hz": 30,
            },
            "optimizer": {
                "type": "judo.optimizers.cem.CrossEntropyMethod",
                "search_space": args.search_space,
                "task_controller": args.task_controller,
                "task_reference": (
                    "fresh_live_pose_to_semantic_keyframe"
                    if args.task_controller == "semantic_pose"
                    else (
                        "recorded_source_eef_trajectory"
                        if args.task_controller == "pose_tracking"
                        else "source_joint_targets"
                    )
                ),
                "num_rollouts": args.num_rollouts,
                "num_iterations": args.num_iterations,
                "num_elites": args.num_elites,
                "duplicate_nominal": args.duplicate_nominal,
                "candidate_repeats": args.candidate_repeats,
                "candidate_repeat_reducer": (
                    args.candidate_repeat_reducer
                ),
                "sigma_min": args.sigma_min,
                "sigma_max": args.sigma_max,
                "max_action_delta": args.max_action_delta,
                "task_translation_start_m": list(
                    args.task_translation_start
                ),
                "task_translation_goal_m": (
                    None
                    if task_translation_goal is None
                    else np.asarray(task_translation_goal).tolist()
                ),
                "max_task_translation_delta_m": (
                    args.max_task_translation_delta
                ),
                "max_task_rotation_delta_rad": (
                    args.max_task_rotation_delta
                ),
                "right_joint_motion_weights": {
                    "path": args.right_joint_path_weight,
                    "acceleration": args.right_joint_accel_weight,
                    "jerk": args.right_joint_jerk_weight,
                },
                "dls_damping": args.dls_damping,
                "dls_max_joint_delta_rad": args.dls_max_joint_delta,
                "dls_max_position_step_m": args.dls_max_position_step,
                "dls_max_rotation_step_rad": args.dls_max_rotation_step,
                "insert": {
                    "enabled": (
                        args.task_controller == "semantic_pose"
                        and args.target_name == "inserted_held"
                        and target_branch_points is not None
                    ),
                    "clearance_offset_branch_m": list(
                        args.insert_clearance_offset_branch
                    ),
                    "clearance_rotation_offset_branch_rad": list(
                        args.insert_clearance_rotation_offset_branch
                    ),
                    "eef_position_offset_branch_m": list(
                        args.insert_eef_position_offset_branch
                    ),
                    "eef_rotation_offset_branch_rad": list(
                        args.insert_eef_rotation_offset_branch
                    ),
                    "approach_offset_branch_m": list(
                        args.insert_approach_offset_branch
                    ),
                    "seat_offset_branch_m": list(
                        args.insert_seat_offset_branch
                    ),
                    "clearance_fraction": args.insert_clearance_fraction,
                    "approach_fraction": args.insert_approach_fraction,
                    "seat_fraction": args.insert_seat_fraction,
                },
                "release": {
                    "enabled": (
                        args.target_name == "hang_complete"
                        and args.release_open_value is not None
                    ),
                    "open_value": args.release_open_value,
                    "start_fraction": args.release_start_fraction,
                    "end_fraction": args.release_end_fraction,
                },
            },
            "iteration_summaries": iteration_summaries,
            "plan": {
                "accepted_update": bool(plan.accepted_update),
                "best_rollout": int(plan.best_rollout),
                "best_iteration": int(plan.best_iteration),
                "improvement": float(plan.improvement),
                "nominal_reward_mean": float(plan.nominal_reward_mean),
                "nominal_reward_std": float(plan.nominal_reward_std),
                "first_action_delta_l2": float(
                    np.linalg.norm(
                        best_executed_actions[0] - nominal[0]
                    )
                ),
            },
            "repeat_evaluation": groups,
            "best_sample_reached_keyframe": reached,
        }
        with open(args.result_json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        if args.controls_npz:
            np.savez_compressed(
                args.controls_npz,
                nominal=nominal,
                base_controls=base_controls,
                optimized_mean=expand(plan.optimized_knots[None, ...])[0],
                best_sample=expand(plan.best_sampled_knots[None, ...])[0],
                optimized_executed_actions=optimized_executed_actions,
                best_executed_actions=best_executed_actions,
                search_space=np.asarray(args.search_space),
                task_translation_start=np.asarray(
                    args.task_translation_start, dtype=np.float32
                ),
                task_translation_goal=(
                    np.empty((0,), dtype=np.float32)
                    if task_translation_goal is None
                    else np.asarray(
                        task_translation_goal, dtype=np.float32
                    )
                ),
                task_controller=np.asarray(args.task_controller),
                reference_eef_poses=(
                    np.empty((0, 7), dtype=np.float32)
                    if reference_eef_poses is None
                    else reference_eef_poses
                ),
                target_mug_pose=(
                    np.empty((0, 7), dtype=np.float32)
                    if target_mug_pose is None
                    else target_mug_pose[None, :]
                ),
                insert_approach_offset_branch=np.asarray(
                    args.insert_approach_offset_branch, dtype=np.float32
                ),
                insert_clearance_offset_branch=np.asarray(
                    args.insert_clearance_offset_branch, dtype=np.float32
                ),
                insert_clearance_rotation_offset_branch=np.asarray(
                    args.insert_clearance_rotation_offset_branch,
                    dtype=np.float32,
                ),
                insert_eef_position_offset_branch=np.asarray(
                    args.insert_eef_position_offset_branch, dtype=np.float32
                ),
                insert_eef_rotation_offset_branch=np.asarray(
                    args.insert_eef_rotation_offset_branch, dtype=np.float32
                ),
                insert_seat_offset_branch=np.asarray(
                    args.insert_seat_offset_branch, dtype=np.float32
                ),
                insert_clearance_fraction=np.float32(
                    args.insert_clearance_fraction
                ),
                insert_approach_fraction=np.float32(
                    args.insert_approach_fraction
                ),
                insert_seat_fraction=np.float32(args.insert_seat_fraction),
                insert_anchor_state=np.int64(args.insert_anchor_state),
                semantic_anchor_state=(
                    np.int64(-1)
                    if args.semantic_anchor_state is None
                    else np.int64(args.semantic_anchor_state)
                ),
                release_open_value=(
                    np.float32(np.nan)
                    if args.release_open_value is None
                    else np.float32(args.release_open_value)
                ),
                release_start_fraction=np.float32(
                    args.release_start_fraction
                ),
                release_end_fraction=np.float32(args.release_end_fraction),
                insert_anchor_reference=(
                    np.empty((0, 36), dtype=np.float32)
                    if insert_anchor_reference is None
                    else insert_anchor_reference[None, :]
                ),
                objective_reference_states=objective_reference,
                history_actions=history,
                history_control_overrides=np.asarray(
                    args.history_controls_npz
                ),
                checkpoint_state=np.int64(
                    args.start_state
                    if args.checkpoint_state is None
                    else args.checkpoint_state
                ),
                start_state=np.int64(args.start_state),
                target_state=np.int64(args.target_state),
                target_name=np.asarray(args.target_name),
                tree_offset_xyz=np.asarray(
                    args.tree_offset_xyz, dtype=np.float32
                ),
                tree_yaw_deg=np.float32(args.tree_yaw_deg),
                source_mug_tree=np.asarray(_source_tree_path(args)),
                target_mug_tree=np.asarray(_target_tree_path(args)),
                source_mug=np.asarray(_source_mug_path(args)),
                target_mug=np.asarray(_target_mug_path(args)),
                tree_root_z_adjustment=np.float32(
                    _tree_root_z_adjustment(args)
                ),
                source_branch_points=(
                    np.empty((0, 3), dtype=np.float32)
                    if source_branch_points is None
                    else source_branch_points
                ),
                target_branch_points=(
                    np.empty((0, 3), dtype=np.float32)
                    if target_branch_points is None
                    else target_branch_points
                ),
            )
        print(
            "JUDO_ISAACLAB_HANGMUG_KEYFRAME_MPC="
            + json.dumps(result, sort_keys=True)
        )
        if not reached:
            raise RuntimeError(result)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if runner is not None:
            runner.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
