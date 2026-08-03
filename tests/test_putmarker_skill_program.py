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


def test_low_feet_cabinet_handle_uses_horizontal_collision_clear_pull():
    module = _module()

    assert module._effective_handle_pull_vertical_offset_m(
        "/dataset/annotated_cabinet_with_feet/annotated_drawer_with_feet_009.hdf5",
        1.0048,
        -0.1311,
        -0.04,
    ) == 0.0
    assert module._effective_handle_pull_vertical_offset_m(
        "/dataset/annotated_cabinet_with_feet/annotated_drawer_with_feet_008.hdf5",
        1.0601,
        -0.1258,
        -0.04,
    ) == 0.0
    assert module._effective_handle_pull_vertical_offset_m(
        "/dataset/annotated/annotated_drawer_009.hdf5", 1.0048, -0.1311, -0.04
    ) == -0.04


def test_low_feet_handle_uses_target_semantic_frame_and_labeled_friction():
    module = _module()

    assert module._right_handle_assist_spec(
        "/dataset/annotated_cabinet_with_feet/annotated_drawer_with_feet_009.hdf5",
        1.0048,
        -0.1311,
    ) == ("friction", "target_semantic_low_feet_handle")
    assert module._uses_target_handle_keyframe(
        "/dataset/annotated_cabinet_with_feet/annotated_drawer_with_feet_009.hdf5",
        1.0048,
        -0.1311,
    )
    assert module._right_handle_assist_spec(
        "/dataset/annotated_cabinet_with_feet/annotated_drawer_with_feet_008.hdf5",
        1.0601,
        -0.1258,
    ) == ("friction", "official_cabinet_with_feet_workspace")


def test_target_handle_grasp_index_comes_from_lower_drawer_motion():
    module = _module()
    drawer = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0002], [0.0, 0.01]]

    assert module._target_drawer_grasp_index({"drawer_joint": drawer}) == 1
