import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "examples/run_putmarker_skill_program.py"
    spec = importlib.util.spec_from_file_location("putmarker_skill_program", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_right_handle_assist_covers_official_cabinet_with_feet_workspace():
    module = _module()

    assert module._right_handle_friction_assist_reason(
        "/dataset/annotated_cabinet_with_feet/annotated_drawer_with_feet_003.hdf5",
        1.0044,
    ) == "official_cabinet_with_feet_workspace"


def test_right_handle_assist_preserves_ordinary_cabinet_physics():
    module = _module()

    assert module._right_handle_friction_assist_reason(
        "/dataset/annotated/annotated_drawer_003.hdf5", 1.0044
    ) is None
    assert module._right_handle_friction_assist_reason(
        "/dataset/annotated/annotated_drawer_000.hdf5", 1.0618
    ) == "elevated_handle_workspace"
