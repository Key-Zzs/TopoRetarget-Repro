"""Non-invasive timing math for exact geometric retarget pilots."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class TimedResult:
    """A function result plus optional monotonic clock boundaries."""

    value: Any
    start: float | None
    end: float | None

    @property
    def seconds(self) -> float | None:
        if self.start is None or self.end is None:
            return None
        return self.end - self.start


def timed_call(
    function: Callable[..., T], *args: Any, enabled: bool = True, **kwargs: Any
) -> TimedResult:
    """Call ``function`` exactly once; instrumentation cannot change its inputs."""

    if not enabled:
        return TimedResult(function(*args, **kwargs), None, None)
    started = time.perf_counter()
    value = function(*args, **kwargs)
    ended = time.perf_counter()
    return TimedResult(value, started, ended)


def percentile(values: np.ndarray, value: float) -> float:
    """Use NumPy's linear percentile definition for all O5 reports."""

    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def frame_statistics(seconds: Iterable[float]) -> dict[str, float | int]:
    """Summarize one complete episode's positive per-frame solver times."""

    values = np.asarray(list(seconds), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("GEOMETRIC_RETARGET_TIMING_VALUES_INVALID")
    if np.any(values < 0):
        raise ValueError("GEOMETRIC_RETARGET_TIMING_NEGATIVE_DURATION")
    return {
        "n_frames": int(values.size),
        "solver_total_sec": float(values.sum()),
        "mean_sec_per_frame": float(values.mean()),
        "std_sec_per_frame": float(values.std()),
        "min_sec_per_frame": float(values.min()),
        "p50_sec_per_frame": percentile(values, 50),
        "p90_sec_per_frame": percentile(values, 90),
        "p95_sec_per_frame": percentile(values, 95),
        "p99_sec_per_frame": percentile(values, 99),
        "max_sec_per_frame": float(values.max()),
    }


def validate_frame_rows(rows: list[dict[str, Any]], expected_frames: int) -> None:
    """Fail closed on missing, duplicate, non-monotonic, or negative frame timers."""

    if len(rows) != expected_frames:
        raise ValueError(f"GEOMETRIC_RETARGET_TIMING_FRAME_COUNT:{len(rows)}:{expected_frames}")
    ordinals = [int(row["frame_ordinal"]) for row in rows]
    frame_ids = [int(row["source_frame_id"]) for row in rows]
    if ordinals != list(range(expected_frames)):
        raise ValueError("GEOMETRIC_RETARGET_TIMING_ORDINALS_INVALID")
    if len(set(frame_ids)) != expected_frames:
        raise ValueError("GEOMETRIC_RETARGET_TIMING_DUPLICATE_FRAME_ID")
    for row in rows:
        start = float(row["solver_start"])
        end = float(row["solver_end"])
        seconds = float(row["solver_sec"])
        if not np.isfinite([start, end, seconds]).all() or end < start or seconds < 0:
            raise ValueError("GEOMETRIC_RETARGET_TIMING_CLOCK_INVALID")
        if (
            row.get("solver_sec_scope")
            == "all_production_refine_attempt_active_set_and_final_audit_timers"
        ):
            accepted = float(row["accepted_solver_sec"])
            if not np.isfinite(accepted) or accepted < 0:
                raise ValueError("GEOMETRIC_RETARGET_TIMING_ACCEPTED_DURATION_INVALID")
            if not np.isclose(end - start, accepted, atol=1e-9, rtol=1e-6):
                raise ValueError("GEOMETRIC_RETARGET_TIMING_ACCEPTED_DURATION_MISMATCH")
            if bool(row.get("unrecorded_window_joint_optimization", False)):
                raise ValueError("GEOMETRIC_RETARGET_TIMING_UNRECORDED_WINDOW_OPTIMIZATION")
        elif not np.isclose(end - start, seconds, atol=1e-9, rtol=1e-6):
            raise ValueError("GEOMETRIC_RETARGET_TIMING_DURATION_MISMATCH")


def stage_total(stages: dict[str, float]) -> float:
    """Compute episode machine total without shared initialization."""

    required = ("episode_load", "prepare", "solver_episode", "semantic_validation", "html")
    return float(sum(float(stages[name]) for name in required))


def cold_warm_totals(
    shared_init: float, dev_01_machine: float, dev_02_machine: float
) -> dict[str, float]:
    """Apply the O5 same-process cold/warm decomposition."""

    return {
        "cold_dev_01_sec": float(shared_init + dev_01_machine),
        "warm_dev_02_sec": float(dev_02_machine),
    }


def aggregate_frame_statistics(episodes: Iterable[Iterable[float]]) -> dict[str, float | int]:
    """Return weighted and unweighted pilot aggregates."""

    arrays = [np.asarray(list(values), dtype=np.float64) for values in episodes]
    if len(arrays) != 2 or any(value.size == 0 for value in arrays):
        raise ValueError("GEOMETRIC_RETARGET_TIMING_REQUIRES_SAME_TWO_EPISODES")
    combined = np.concatenate(arrays)
    return {
        "n_timed_episodes": 2,
        "n_timed_frames": int(combined.size),
        "weighted_solver_sec_per_frame": float(combined.mean()),
        "unweighted_episode_mean_sec_per_frame": float(np.mean([value.mean() for value in arrays])),
        "all_frame_p50_sec": percentile(combined, 50),
        "all_frame_p90_sec": percentile(combined, 90),
        "all_frame_p95_sec": percentile(combined, 95),
        "all_frame_max_sec": float(combined.max()),
    }


def runtime_estimate(
    seconds_per_frame: float, frame_count: float, shared_init: float = 0.0
) -> dict[str, float]:
    """Estimate solver wall time with shared initialization added exactly once."""

    seconds = float(shared_init + seconds_per_frame * frame_count)
    return {"seconds": seconds, "hours": seconds / 3600.0, "days": seconds / 86400.0}


__all__ = [
    "TimedResult",
    "aggregate_frame_statistics",
    "cold_warm_totals",
    "frame_statistics",
    "percentile",
    "runtime_estimate",
    "stage_total",
    "timed_call",
    "validate_frame_rows",
]
