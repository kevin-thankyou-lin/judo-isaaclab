import numpy as np
import pytest

from build_putpot_failed_attempt_diagnostic import _local_mpc_active_arm


def test_diagnostic_builder_uses_trace_active_arm_and_legacy_left_fallback():
    trace = {
        "local_mpc_active_arm": np.asarray(["right", "left"], dtype="<U5")
    }
    assert _local_mpc_active_arm(trace, 0) == "right"
    assert _local_mpc_active_arm(trace, 1) == "left"
    assert _local_mpc_active_arm({}, 0) == "left"


def test_diagnostic_builder_rejects_invalid_active_arm():
    trace = {"local_mpc_active_arm": np.asarray(["both"], dtype="<U5")}
    with pytest.raises(ValueError, match="invalid local-MPC active arm"):
        _local_mpc_active_arm(trace, 0)
