"""Runtime-health classification for trajectory and static retarget samples.

The policy is intentionally pure: it only classifies immutable frame-runtime
evidence.  It cannot modify solver, geometry, artifact, or scheduling inputs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

DYNAMIC_SINGLE_FRAME_HARD_S = 90.0
DYNAMIC_ROLLING_P95_HARD_S = 30.0
DYNAMIC_CONSECUTIVE_SLOW_FRAME_S = 45.0
DYNAMIC_CONSECUTIVE_SLOW_FRAME_COUNT = 3
STATIC_SINGLE_FRAME_SOFT_WARNING_S = 90.0
STATIC_SINGLE_FRAME_HARD_STOP_S = 300.0

STATIC_FRAME_ACCEPTED_WITH_RUNTIME_WARNING = "STATIC_FRAME_ACCEPTED_WITH_RUNTIME_WARNING"
STATIC_FRAME_ACCEPTED = "STATIC_FRAME_ACCEPTED"
STATIC_FRAME_HARD_RUNTIME_FAILURE = "STATIC_FRAME_HARD_RUNTIME_FAILURE"
STATIC_FRAME_SOLVER_FAILURE = "STATIC_FRAME_SOLVER_FAILURE"
STATIC_FRAME_GEOMETRY_FAILURE = "STATIC_FRAME_GEOMETRY_FAILURE"


@dataclass(frozen=True)
class RuntimeHealthDecision:
    """A serializable classification that has no side effects on retargeting."""

    sample_kind: str
    status: str
    elapsed_s: float
    rolling_p95_gate: str
    consecutive_slow_frame_gate: str
    terminal_reason: str | None
    warning: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_value(
    selection: Mapping[str, Any], canonical_metadata: Mapping[str, Any], *keys: str
) -> Any:
    for values in (selection, canonical_metadata):
        for key in keys:
            if key in values:
                return values[key]
    return None


def is_static_single_frame_contract(
    selection: Mapping[str, Any],
    canonical_metadata: Mapping[str, Any],
    *,
    frame_count: int,
) -> bool:
    """Identify a static singleton through the canonical selection contract.

    Dataset identity is deliberately not read.  A static classification needs
    all three contract facts: static sample mode, inapplicable temporal metrics,
    and one articulated/canonical frame.
    """

    sample_mode = _first_value(selection, canonical_metadata, "sample_mode", "sample_type")
    temporal_metrics = _first_value(selection, canonical_metadata, "temporal_metrics_applicable")
    articulated_frames = _first_value(
        selection, canonical_metadata, "articulated_frame_count", "frame_count"
    )
    if not isinstance(sample_mode, str) or not sample_mode.lower().startswith("static"):
        return False
    try:
        one_articulated_frame = int(articulated_frames) == 1
    except (TypeError, ValueError):
        one_articulated_frame = False
    return temporal_metrics is False and one_articulated_frame and int(frame_count) == 1


def classify_runtime_health(
    *,
    elapsed_s: float,
    frame_times_s: Sequence[float],
    static_single_frame: bool,
) -> RuntimeHealthDecision:
    """Classify health timing without changing acceptance or artifacts."""

    elapsed = float(elapsed_s)
    if not math.isfinite(elapsed):
        return RuntimeHealthDecision(
            sample_kind="static_single_frame" if static_single_frame else "dynamic_trajectory",
            status=(
                STATIC_FRAME_HARD_RUNTIME_FAILURE
                if static_single_frame
                else "DYNAMIC_FRAME_HARD_RUNTIME_FAILURE"
            ),
            elapsed_s=elapsed,
            rolling_p95_gate="NOT_APPLICABLE" if static_single_frame else "enabled",
            consecutive_slow_frame_gate="NOT_APPLICABLE" if static_single_frame else "enabled",
            terminal_reason="nonfinite_frame_runtime",
            warning=None,
        )

    if static_single_frame:
        if elapsed > STATIC_SINGLE_FRAME_HARD_STOP_S:
            return RuntimeHealthDecision(
                sample_kind="static_single_frame",
                status=STATIC_FRAME_HARD_RUNTIME_FAILURE,
                elapsed_s=elapsed,
                rolling_p95_gate="NOT_APPLICABLE",
                consecutive_slow_frame_gate="NOT_APPLICABLE",
                terminal_reason=f"static_frame_over_300s:{elapsed:.3f}",
                warning=None,
            )
        if elapsed > STATIC_SINGLE_FRAME_SOFT_WARNING_S:
            return RuntimeHealthDecision(
                sample_kind="static_single_frame",
                status=STATIC_FRAME_ACCEPTED_WITH_RUNTIME_WARNING,
                elapsed_s=elapsed,
                rolling_p95_gate="NOT_APPLICABLE",
                consecutive_slow_frame_gate="NOT_APPLICABLE",
                terminal_reason=None,
                warning=(
                    "static single-frame runtime exceeded 90 s; retained because the "
                    "300 s static hard stop was not exceeded"
                ),
            )
        return RuntimeHealthDecision(
            sample_kind="static_single_frame",
            status=STATIC_FRAME_ACCEPTED,
            elapsed_s=elapsed,
            rolling_p95_gate="NOT_APPLICABLE",
            consecutive_slow_frame_gate="NOT_APPLICABLE",
            terminal_reason=None,
            warning=None,
        )

    if elapsed > DYNAMIC_SINGLE_FRAME_HARD_S:
        return RuntimeHealthDecision(
            sample_kind="dynamic_trajectory",
            status="DYNAMIC_FRAME_HARD_RUNTIME_FAILURE",
            elapsed_s=elapsed,
            rolling_p95_gate="enabled",
            consecutive_slow_frame_gate="enabled",
            terminal_reason=f"single_frame_over_90s:{elapsed:.3f}",
            warning=None,
        )
    times = [float(value) for value in frame_times_s]
    if len(times) >= 10 and all(math.isfinite(value) for value in times[-10:]):
        ordered = sorted(times[-10:])
        position = 0.95 * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        rolling_p95 = ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
        if rolling_p95 > DYNAMIC_ROLLING_P95_HARD_S:
            return RuntimeHealthDecision(
                sample_kind="dynamic_trajectory",
                status="DYNAMIC_TRAJECTORY_HARD_RUNTIME_FAILURE",
                elapsed_s=elapsed,
                rolling_p95_gate="enabled",
                consecutive_slow_frame_gate="enabled",
                terminal_reason="rolling_10_frame_p95_over_30s",
                warning=None,
            )
    if len(times) >= DYNAMIC_CONSECUTIVE_SLOW_FRAME_COUNT and all(
        value > DYNAMIC_CONSECUTIVE_SLOW_FRAME_S
        for value in times[-DYNAMIC_CONSECUTIVE_SLOW_FRAME_COUNT:]
    ):
        return RuntimeHealthDecision(
            sample_kind="dynamic_trajectory",
            status="DYNAMIC_TRAJECTORY_HARD_RUNTIME_FAILURE",
            elapsed_s=elapsed,
            rolling_p95_gate="enabled",
            consecutive_slow_frame_gate="enabled",
            terminal_reason="three_consecutive_frames_over_45s",
            warning=None,
        )
    return RuntimeHealthDecision(
        sample_kind="dynamic_trajectory",
        status="DYNAMIC_FRAME_ACCEPTED",
        elapsed_s=elapsed,
        rolling_p95_gate="enabled",
        consecutive_slow_frame_gate="enabled",
        terminal_reason=None,
        warning=None,
    )
