import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from render_hangmug_continuous_sequence import _classify_release_collisions


def test_release_collision_contract_rejects_insertion_body_contact():
    report = _classify_release_collisions(
        [74, 145, 146], release_start_step=125, terminal_step=240
    )

    assert report["pre_release_collision_steps"] == [74]
    assert report["release_settling_collision_steps"] == [145, 146]
    assert report["pre_release_valid"] is False
    assert report["terminal_clear"] is True


def test_release_collision_contract_allows_transient_release_settling():
    report = _classify_release_collisions(
        [145, 146], release_start_step=125, terminal_step=240
    )

    assert report["pre_release_valid"] is True
    assert report["terminal_clear"] is True
