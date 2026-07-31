import pytest

torch = pytest.importorskip("torch")

from judo_isaaclab.task_space import (
    DampedLeastSquaresPoseTrackingAdapter,
    damped_least_squares,
    resolve_end_effector_body_index,
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
