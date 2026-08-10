import pytest

torch = pytest.importorskip("torch")

from judo_isaaclab.task_space import (
    DampedLeastSquaresPoseTrackingAdapter,
    damped_least_squares,
    pose_runtime_to_wxyz,
    pose_wxyz_to_runtime,
    resolve_end_effector_body_index,
    resolve_link_jacobian,
)


class _Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_damped_least_squares_is_batched_and_finite():
    jacobian = torch.eye(6).repeat(2, 1, 1)
    twist = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]]
    )

    result = damped_least_squares(jacobian, twist, damping=0.1)

    assert result.shape == (2, 6)
    torch.testing.assert_close(result, twist / 1.01)
    assert torch.isfinite(result).all()


def test_pose_order_conversion_round_trips_xyzw_runtime():
    legacy = torch.tensor([[1.0, 2.0, 3.0, 0.7, 0.1, 0.2, 0.3]])

    runtime = pose_wxyz_to_runtime(legacy, runtime_xyzw=True)

    torch.testing.assert_close(
        runtime, torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.7]])
    )
    torch.testing.assert_close(
        pose_runtime_to_wxyz(runtime, runtime_xyzw=True), legacy
    )


def test_pose_order_conversion_is_identity_for_legacy_runtime():
    legacy = torch.tensor([1.0, 2.0, 3.0, 0.7, 0.1, 0.2, 0.3])

    runtime = pose_wxyz_to_runtime(legacy, runtime_xyzw=False)

    torch.testing.assert_close(runtime, legacy)
    assert runtime.data_ptr() != legacy.data_ptr()


def test_resolve_link_jacobian_uses_new_proxy_torch_view():
    jacobians = torch.arange(1 * 3 * 6 * 7).reshape(1, 3, 6, 7)
    arm = _Namespace(
        is_fixed_base=True,
        data=_Namespace(
            body_link_jacobian_w=_Namespace(torch=jacobians)
        ),
    )

    result = resolve_link_jacobian(arm, body_index=2, joint_count=6)

    torch.testing.assert_close(result, jacobians[:, 1, :, :6])


def test_resolve_link_jacobian_preserves_legacy_physx_view():
    jacobians = torch.arange(1 * 3 * 6 * 7).reshape(1, 3, 6, 7)
    arm = _Namespace(
        is_fixed_base=False,
        data=_Namespace(),
        root_physx_view=_Namespace(get_jacobians=lambda: jacobians),
    )

    result = resolve_link_jacobian(arm, body_index=2, joint_count=5)

    torch.testing.assert_close(result, jacobians[:, 2, :, :5])


def test_pose_tracking_adapter_rejects_invalid_reference_shape():
    with pytest.raises(ValueError, match="horizon, 7"):
        DampedLeastSquaresPoseTrackingAdapter(
            reference_poses=torch.zeros((3, 6))
        )


def test_resolve_end_effector_uses_robot_attachment_link():
    arm = _Namespace(
        num_bodies=4,
        data=_Namespace(
            body_names=["base", "link_6", "left_finger", "right_finger"]
        ),
    )
    env = _Namespace(
        scene={"right_arm": arm},
        robot=_Namespace(
            arms={
                "right_arm": _Namespace(
                    end_effector=_Namespace(attach_link_name="link_6")
                )
            }
        ),
    )

    assert resolve_end_effector_body_index(env, "right_arm") == 1


def test_resolve_end_effector_allows_explicit_body_and_legacy_fallback():
    arm = _Namespace(
        num_bodies=3,
        data=_Namespace(body_names=["base", "tool", "finger"]),
    )
    env = _Namespace(scene={"arm": arm})

    assert resolve_end_effector_body_index(env, "arm", "tool") == 1
    assert resolve_end_effector_body_index(env, "arm") == 2


def test_resolve_end_effector_rejects_unknown_attachment_link():
    arm = _Namespace(
        num_bodies=2,
        data=_Namespace(body_names=["base", "tool"]),
    )
    env = _Namespace(scene={"arm": arm})

    with pytest.raises(ValueError, match="missing"):
        resolve_end_effector_body_index(env, "arm", "missing")
