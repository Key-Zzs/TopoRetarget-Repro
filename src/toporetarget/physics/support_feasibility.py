"""Turn support evidence and real gravity diagnostics into timeline semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .support_contract import SupportClassification, SupportMode


def _runs(values: np.ndarray) -> list[tuple[int, int, str]]:
    if values.ndim != 1:
        raise ValueError("SUPPORT_TIMELINE_VALUES_MUST_BE_1D")
    result: list[tuple[int, int, str]] = []
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[stop] == values[start]:
            stop += 1
        result.append((start, stop, str(values[start])))
        start = stop
    return result


def build_support_timeline(
    *,
    runtime_index: np.ndarray,
    source_expected_contact: np.ndarray,
    gravity_label_by_state: Mapping[int, str],
    source_support_available: bool,
) -> list[dict[str, object]]:
    """Build time intervals, never silently replacing unknown support with a plane."""

    indices = np.asarray(runtime_index, dtype=np.int64)
    expected = np.asarray(source_expected_contact, dtype=bool)
    if indices.shape != expected.shape or not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError("SUPPORT_TIMELINE_INDEX_INVALID")
    labels: list[str] = []
    for index, source_contact in zip(indices, expected, strict=True):
        gravity = gravity_label_by_state.get(int(index))
        if source_support_available:
            labels.append(SupportClassification.SUPPORT_RECOVERED_VALIDATED.value)
        elif source_contact and gravity == "GRAVITY_SAFE":
            labels.append(SupportClassification.HAND_SUPPORTED_VALIDATED.value)
        elif not source_contact and gravity == "GRAVITY_RISK":
            labels.append(SupportClassification.UNSUPPORTED_REFERENCE.value)
        else:
            labels.append(SupportClassification.SUPPORT_UNKNOWN.value)
    encoded = np.asarray(labels, dtype="U32")
    return [
        {
            "start_runtime_index": start,
            "end_runtime_index_exclusive": stop,
            "classification": classification,
            "evidence": (
                "source_support_asset"
                if source_support_available
                else "source_contact_plus_gravity_diagnostic"
            ),
        }
        for start, stop, classification in _runs(encoded)
    ]


def decide_support_mode(
    *,
    support_timeline: Sequence[Mapping[str, object]],
    safe_bank_names: Sequence[str],
    hidden_support: bool,
) -> dict[str, object]:
    """Resolve the P3 support gate without pretending frame zero is known."""

    if hidden_support:
        raise ValueError("SUPPORT_DECISION_HIDDEN_SUPPORT_FORBIDDEN")
    classes = {str(item["classification"]) for item in support_timeline}
    initial = {"CONTACT_READY_SAFE", "PERSISTENT_SAFE", "MANIPULATION_SAFE"}
    present = set(str(value) for value in safe_bank_names)
    if SupportClassification.SUPPORT_EXPLICIT_VALIDATED.value in classes:
        mode = SupportMode.SOURCE_SUPPORT
        classification = SupportClassification.SUPPORT_EXPLICIT_VALIDATED.value
    elif SupportClassification.SUPPORT_RECOVERED_VALIDATED.value in classes:
        mode = SupportMode.SOURCE_SUPPORT
        classification = SupportClassification.SUPPORT_RECOVERED_VALIDATED.value
    elif initial & present:
        mode = SupportMode.CONTACT_READY_ONLY_VALIDATED
        classification = SupportMode.CONTACT_READY_ONLY_VALIDATED.value
    elif SupportClassification.HAND_SUPPORTED_VALIDATED.value in classes:
        mode = SupportMode.HAND_SUPPORTED
        classification = SupportClassification.HAND_SUPPORTED_VALIDATED.value
    else:
        mode = SupportMode.BLOCKED
        classification = SupportClassification.SUPPORT_UNKNOWN.value
    return {
        "schema_version": "Stage16SupportFeasibilityDecisionV1",
        "support_mode": mode.value,
        "p3_gate_classification": classification,
        "frame_zero_full_gravity_authorized": mode is SupportMode.SOURCE_SUPPORT,
        "hidden_support": False,
        "p3_allowed_reset_banks": sorted(initial & present),
        "prohibited_reset_states": ["PRE_CONTACT", "AMBIGUOUS", "GRAVITY_RISK", "INVALID_RESET"],
    }


def support_diagnostic_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Keep the no-support counterfactual explicit when no source asset exists."""

    if not rows:
        raise ValueError("SUPPORT_DIAGNOSTIC_ROWS_EMPTY")
    finite = all(not bool(row.get("nonfinite", False)) for row in rows)
    return {
        "schema_version": "Stage16FullGravitySupportDiagnosticV1",
        "source_support_active": "NOT_RUN_NO_RECOVERABLE_SOURCE_SUPPORT_ASSET",
        "support_removed_counterfactual": "NOT_APPLICABLE_NO_SOURCE_SUPPORT_ASSET",
        "hand_supported_diagnostic": "REUSED_FROM_TRUE_PHYSX_P1_ROWS",
        "all_rows_finite": finite,
        "hidden_support": False,
    }


__all__ = [
    "build_support_timeline",
    "decide_support_mode",
    "support_diagnostic_summary",
]
