"""Run history-conditioned Judo CEM toward a HangMug keyframe."""

import argparse
import json
import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

HANG_POSITION_TOLERANCE_M = 0.03
HANG_SPEED_TOLERANCE_M_S = 0.05
HANG_STABILITY_STEPS = 30
TREE_APPROACH_POSITION_TOLERANCE_M = 0.06
TREE_INSERTION_POSITION_TOLERANCE_M = 0.03


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
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=0.03)
    parser.add_argument("--max-action-delta", type=float, default=0.08)
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
                controls["best_sample"], dtype=np.float32
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
        history = np.asarray(
            group["actions"][checkpoint_state : args.start_state],
            dtype=np.float32,
        )
        history = _apply_history_control_overrides(
            history,
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
    return checkpoint, history, nominal, reference


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
    source = np.linspace(0.0, 1.0, knots.shape[1])
    target = np.linspace(0.0, 1.0, nominal.shape[0])
    corrections = np.stack(
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
    corrections = np.clip(
        corrections, -max_action_delta, max_action_delta
    )
    corrections[:, :, (6, 13)] = 0.0
    if right_arm_only:
        corrections[:, :, :7] = 0.0
    return nominal[None, :, :] + corrections


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
    keyframe_offset,
    target_name,
    source_branch_points=None,
    target_branch_points=None,
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
        target_frame = "mug_relative_to_tree_root"
    else:
        position_error = world_position_error
        rotation_error = world_rotation_error
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
    action_delta = np.sqrt(
        np.mean(
            np.square(controls - nominal[None, :, :]),
            axis=(1, 2),
        )
    )
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
            )
    elif target_name == "hang_complete":
        target_success = (
            stage3
            * (position_error <= HANG_POSITION_TOLERANCE_M)
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
        acceptance_window = slice(
            max(0, keyframe_offset - HANG_STABILITY_STEPS + 1),
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
        "keyframe_rotation_error_rad": rotation_error[:, keyframe_offset],
        "keyframe_left_joint_rms_rad": left_joint_error[:, keyframe_offset],
        "keyframe_right_joint_rms_rad": right_joint_error[:, keyframe_offset],
        "action_delta_rms": action_delta,
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
    rotation = components["keyframe_rotation_error_rad"][rows]
    action_delta = components["action_delta_rms"][rows]
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
        "keyframe_rotation_error_rad_mean": float(rotation.mean()),
        "keyframe_rotation_error_rad_max": float(rotation.max()),
        "action_delta_rms_mean": float(action_delta.mean()),
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
            HistoryConditionedIsaacLabBackend,
            JudoIsaacLabMPC,
        )

        np.random.seed(args.seed)
        assets = {
            "mug": os.path.join(args.objects_root, "Mug", "mug_000"),
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
        checkpoint, history, nominal, reference = _load_demo(
            args, runner.env.device
        )
        context = BranchContext(
            checkpoint_state=checkpoint,
            action_history=history,
            rigid_object_states={
                "mug": RigidObjectMpcState(False),
                "mug_tree": RigidObjectMpcState(False),
            },
            is_relative=True,
        )
        backend = HistoryConditionedIsaacLabBackend(
            runner,
            state_encoder=_encode_state,
            sensor_encoder=_encode_sensors,
        )
        optimizer = CrossEntropyMethod(
            CrossEntropyMethodConfig(
                num_rollouts=args.num_rollouts,
                num_nodes=num_control_knots,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                num_elites=args.num_elites,
            ),
            nu=nominal.shape[1],
        )

        correction_nominal = np.zeros(
            (num_control_knots, nominal.shape[1]), dtype=np.float32
        )

        def expand(knots):
            controls = _expand_control_corrections(
                knots,
                nominal,
                max_action_delta=args.max_action_delta,
                right_arm_only=args.target_name
                in ("tree_approach", "inserted_held", "hang_complete"),
            )
            if args.target_name == "inserted_held":
                controls[:, : keyframe_offset + 1, 13] = nominal[0, 13]
            return controls

        iteration_summaries = []

        def objective(states, sensors, controls):
            components = _objective_components(
                states,
                sensors,
                controls,
                reference=reference,
                nominal=nominal,
                keyframe_offset=keyframe_offset,
                target_name=args.target_name,
                source_branch_points=source_branch_points,
                target_branch_points=target_branch_points,
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
            (args.num_rollouts, num_control_knots, nominal.shape[1]),
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
            reference=reference,
            nominal=nominal,
            keyframe_offset=keyframe_offset,
            target_name=args.target_name,
            source_branch_points=source_branch_points,
            target_branch_points=target_branch_points,
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
                "target_frame": evaluation["target_frame"],
                "acceptance": "task_subtask_success",
                "hang_acceptance": {
                    "position_tolerance_m": HANG_POSITION_TOLERANCE_M,
                    "speed_tolerance_m_s": HANG_SPEED_TOLERANCE_M_S,
                    "consecutive_steps": HANG_STABILITY_STEPS,
                },
                "tree_acceptance": {
                    "approach_position_tolerance_m": (
                        TREE_APPROACH_POSITION_TOLERANCE_M
                    ),
                    "insertion_position_tolerance_m": (
                        TREE_INSERTION_POSITION_TOLERANCE_M
                    ),
                },
                "right_gripper_held_through_target": (
                    args.target_name == "inserted_held"
                ),
                "tree_offset_xyz_m": list(args.tree_offset_xyz),
                "tree_yaw_deg": args.tree_yaw_deg,
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
                "num_rollouts": args.num_rollouts,
                "num_iterations": args.num_iterations,
                "num_elites": args.num_elites,
                "duplicate_nominal": args.duplicate_nominal,
                "candidate_repeats": args.candidate_repeats,
                "sigma_min": args.sigma_min,
                "sigma_max": args.sigma_max,
                "max_action_delta": args.max_action_delta,
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
                        expand(plan.best_sampled_knots[None, ...])[0, 0]
                        - nominal[0]
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
                optimized_mean=expand(plan.optimized_knots[None, ...])[0],
                best_sample=expand(plan.best_sampled_knots[None, ...])[0],
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
