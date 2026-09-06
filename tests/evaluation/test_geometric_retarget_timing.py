from __future__ import annotations

import numpy as np
import pytest

from toporetarget.evaluation.geometric_retarget_timing import (
    aggregate_frame_statistics,
    cold_warm_totals,
    frame_statistics,
    runtime_estimate,
    stage_total,
    timed_call,
    validate_frame_rows,
)


def test_instrumentation_does_not_change_deterministic_output() -> None:
    inputs = np.array([3.0, 1.0, 4.0])

    def reference(values: np.ndarray) -> np.ndarray:
        return values**2 + 2.0 * values

    enabled = timed_call(reference, inputs.copy(), enabled=True)
    disabled = timed_call(reference, inputs.copy(), enabled=False)

    assert np.array_equal(enabled.value, disabled.value)
    assert enabled.start is not None and enabled.end is not None
    assert enabled.seconds is not None and enabled.seconds >= 0
    assert disabled.start is None and disabled.end is None


def test_timing_math_and_cold_warm_decomposition() -> None:
    first = [1.0, 3.0]
    second = [2.0, 2.0, 2.0]
    stats = frame_statistics(first)
    aggregate = aggregate_frame_statistics([first, second])

    assert stats["mean_sec_per_frame"] == 2.0
    assert stats["p50_sec_per_frame"] == 2.0
    assert aggregate["weighted_solver_sec_per_frame"] == pytest.approx(2.0)
    assert aggregate["unweighted_episode_mean_sec_per_frame"] == pytest.approx(2.0)
    assert cold_warm_totals(5.0, 10.0, 11.0) == {
        "cold_dev_01_sec": 15.0,
        "warm_dev_02_sec": 11.0,
    }
    assert (
        stage_total(
            {
                "episode_load": 1.0,
                "prepare": 2.0,
                "solver_episode": 3.0,
                "semantic_validation": 4.0,
                "html": 5.0,
            }
        )
        == 15.0
    )


def test_frame_timer_rows_are_complete_unique_and_monotonic() -> None:
    rows = [
        {
            "frame_ordinal": 0,
            "source_frame_id": 100,
            "solver_start": 10.0,
            "solver_end": 10.5,
            "solver_sec": 0.5,
        },
        {
            "frame_ordinal": 1,
            "source_frame_id": 101,
            "solver_start": 11.0,
            "solver_end": 11.25,
            "solver_sec": 0.25,
        },
    ]
    validate_frame_rows(rows, 2)
    with pytest.raises(ValueError, match="DUPLICATE_FRAME_ID"):
        validate_frame_rows([{**rows[0]}, {**rows[1], "source_frame_id": 100}], 2)


def test_frame_timer_accepts_cumulative_retry_solver_duration() -> None:
    row = {
        "frame_ordinal": 0,
        "source_frame_id": 100,
        "solver_start": 20.0,
        "solver_end": 20.5,
        "solver_sec": 1.25,
        "solver_sec_scope": "all_production_refine_attempt_active_set_and_final_audit_timers",
        "accepted_solver_sec": 0.5,
        "unrecorded_window_joint_optimization": False,
    }
    validate_frame_rows([row], 1)
    with pytest.raises(ValueError, match="UNRECORDED_WINDOW_OPTIMIZATION"):
        validate_frame_rows([{**row, "unrecorded_window_joint_optimization": True}], 1)


def test_batch_estimate_amortizes_shared_initialization_once() -> None:
    estimate = runtime_estimate(2.0, 100.0, shared_init=7.0)
    assert estimate["seconds"] == 207.0
    assert estimate["hours"] == pytest.approx(207.0 / 3600.0)
