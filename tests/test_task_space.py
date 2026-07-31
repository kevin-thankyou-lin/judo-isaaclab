import pytest

torch = pytest.importorskip("torch")

from judo_isaaclab.task_space import (
    DampedLeastSquaresPoseTrackingAdapter,
    damped_least_squares,
)


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
