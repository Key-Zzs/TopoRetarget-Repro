from __future__ import annotations

import numpy as np
import pytest

from toporetarget.retarget.first_frame_seed import load_first_frame_seed_authority_v2


def test_v2_candidates_are_deterministic_and_bound_respecting() -> None:
    authority = load_first_frame_seed_authority_v2()
    lower = np.array([-1.0, -2.0, -3.0])
    upper = np.array([1.0, 2.0, 3.0])
    first = authority.candidates(np.zeros(3), lower, upper)
    second = authority.candidates(np.zeros(3), lower, upper)

    assert [name for name, _ in first] == list(authority.candidate_ids)
    for (_, left), (_, right) in zip(first, second, strict=True):
        assert np.array_equal(left, right)
        assert np.all(left >= lower)
        assert np.all(left <= upper)


@pytest.mark.parametrize(
    ("neutral", "lower", "upper", "message"),
    [
        (np.array([np.nan]), np.array([-1.0]), np.array([1.0]), "NONFINITE"),
        (np.zeros(2), np.array([-1.0]), np.array([1.0]), "SHAPE"),
        (np.array([2.0]), np.array([-1.0]), np.array([1.0]), "OUT_OF_BOUNDS"),
    ],
)
def test_v2_candidates_fail_closed(
    neutral: np.ndarray, lower: np.ndarray, upper: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_first_frame_seed_authority_v2().candidates(neutral, lower, upper)


def test_v2_selection_prefers_success_then_objective_then_fixed_order() -> None:
    authority = load_first_frame_seed_authority_v2()
    records = [
        {
            "seed_id": "joint_range_upper_quartile",
            "finite": True,
            "within_bounds": True,
            "probe_progressed": True,
            "probe_success": False,
            "probe_final_objective": 0.01,
        },
        {
            "seed_id": "joint_range_midpoint",
            "finite": True,
            "within_bounds": True,
            "probe_progressed": True,
            "probe_success": True,
            "probe_final_objective": 0.02,
        },
    ]
    assert authority.select(records)["seed_id"] == "joint_range_midpoint"
    with pytest.raises(ValueError, match="NO_VALID"):
        authority.select([])
