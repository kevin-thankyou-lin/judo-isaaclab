import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from search_floating_object_hang import _candidate_paths, _score


def test_candidate_path_ends_at_branch_frame_offset():
    start = np.asarray([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
    seed = np.asarray([0.4, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0])
    parameters = np.zeros((2, 8), dtype=np.float32)
    parameters[1, :3] = [0.01, -0.02, 0.03]

    paths = _candidate_paths(start, seed, np.eye(3), parameters, horizon=20)

    assert paths.shape == (2, 20, 7)
    np.testing.assert_allclose(paths[0, -1, :3], seed[:3], atol=1e-6)
    np.testing.assert_allclose(
        paths[1, -1, :3], seed[:3] + parameters[1, :3], atol=1e-6
    )


def test_coded_support_dominates_release_fallback_score():
    scores = _score(
        success=[False, True],
        terminal_drift=[0.0, 1.0],
        peak_speed=[0.0, 10.0],
        terminal_height=[1.0, 0.0],
    )

    assert scores[1] > scores[0]


def test_score_prefers_deeper_supported_seating_when_stability_matches():
    scores = _score(
        success=[True, True],
        terminal_drift=[0.002, 0.002],
        peak_speed=[0.02, 0.02],
        terminal_height=[1.0, 1.0],
        seating_depth=[0.0, 0.01],
    )

    assert scores[1] > scores[0]
