"""Reference-contact interpretation used by the Stage 16-D R2 audit.

This module is intentionally independent of the historical Reward V3 mask.
It describes the evidence that is available for an audit; callers must never
feed its result back into an already-trained V3 policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import numpy as np

FINGER_ORDER: Final = ("thumb", "index", "middle", "ring", "pinky")
EVIDENCE_CLASSES: Final = (
    "NO_CONTACT_EXPECTED",
    "PROXIMITY_ONLY_AMBIGUOUS",
    "GEOMETRIC_STRONG_CONTACT_CANDIDATE",
    "SOURCE_SUPPORTED_CONTACT",
    "REFERENCE_CONTACT_EVIDENCE_CONFLICT",
)
EVIDENCE_SOURCES: Final = (
    "SOURCE_EXPLICIT",
    "SOURCE_DERIVED",
    "TOPOLOGY_DERIVED",
    "GEOMETRIC_PROXIMITY",
    "UNAVAILABLE",
)


@dataclass(frozen=True)
class ReferenceContactContractV2:
    """Frozen, diagnostic-only contact interpretation thresholds."""

    identifier: str = "ReferenceContactContractV2"
    schema_version: str = "Stage16DReferenceContactContractV2"
    finger_order: tuple[str, ...] = FINGER_ORDER
    strong_distance_m: float = 0.02
    historical_v3_distance_m: float = 0.03
    severe_source_geometry_conflict_distance_m: float = 0.05
    persistent_window_control_steps: int = 3
    training_use: str = "forbidden_diagnostic_only"

    def __post_init__(self) -> None:
        if self.finger_order != FINGER_ORDER:
            raise ValueError("REFERENCE_CONTACT_V2_FINGER_ORDER_DRIFT")
        if (self.strong_distance_m, self.historical_v3_distance_m) != (0.02, 0.03):
            raise ValueError("REFERENCE_CONTACT_V2_THRESHOLDS_ARE_FROZEN")
        if self.severe_source_geometry_conflict_distance_m != 0.05:
            raise ValueError("REFERENCE_CONTACT_V2_CONFLICT_THRESHOLD_DRIFT")
        if self.persistent_window_control_steps != 3:
            raise ValueError("REFERENCE_CONTACT_V2_PERSISTENT_WINDOW_DRIFT")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_support(values: np.ndarray | None, shape: tuple[int, int], label: str) -> np.ndarray:
    if values is None:
        return np.zeros(shape, dtype=bool)
    result = np.asarray(values, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"REFERENCE_CONTACT_V2_{label}_SHAPE_INVALID:{result.shape}")
    return result


def evaluate_reference_contact(
    distances_m: np.ndarray,
    *,
    source_explicit_contact: np.ndarray | None = None,
    source_derived_contact: np.ndarray | None = None,
    topology_contact: np.ndarray | None = None,
    contract: ReferenceContactContractV2 | None = None,
) -> dict[str, np.ndarray]:
    """Classify a ``[T, 5]`` reference without changing the V3 mask.

    All support arrays are optional because HOCap has no verified per-finger
    annotation in the frozen R2 inputs.  Their provenance is reported by the
    caller rather than promoted to ground truth here.
    """

    frozen = contract or ReferenceContactContractV2()
    distance = np.asarray(distances_m, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[1] != len(FINGER_ORDER):
        raise ValueError("REFERENCE_CONTACT_V2_DISTANCE_MUST_BE_[T,5]")
    if not np.isfinite(distance).all():
        raise ValueError("REFERENCE_CONTACT_V2_DISTANCE_NONFINITE")
    shape = distance.shape
    explicit = _as_support(source_explicit_contact, shape, "SOURCE_EXPLICIT")
    derived = _as_support(source_derived_contact, shape, "SOURCE_DERIVED")
    topology = _as_support(topology_contact, shape, "TOPOLOGY")
    source_supported = explicit | derived
    topology_supported = topology
    supported = source_supported | topology_supported
    strong_geometric = distance <= frozen.strong_distance_m
    v3_primary = distance < frozen.historical_v3_distance_m
    proximity_only = (distance > frozen.strong_distance_m) & v3_primary & ~supported
    no_contact = (distance >= frozen.historical_v3_distance_m) & ~supported
    conflict = supported & (distance > frozen.severe_source_geometry_conflict_distance_m)
    strong = (strong_geometric | supported) & ~conflict
    evidence_class = np.full(shape, "NO_CONTACT_EXPECTED", dtype="<U40")
    evidence_class[proximity_only] = "PROXIMITY_ONLY_AMBIGUOUS"
    evidence_class[strong_geometric & ~supported] = "GEOMETRIC_STRONG_CONTACT_CANDIDATE"
    evidence_class[supported & ~conflict] = "SOURCE_SUPPORTED_CONTACT"
    evidence_class[conflict] = "REFERENCE_CONTACT_EVIDENCE_CONFLICT"
    source_class = np.full(shape, "GEOMETRIC_PROXIMITY", dtype="<U24")
    source_class[distance >= frozen.historical_v3_distance_m] = "UNAVAILABLE"
    source_class[topology] = "TOPOLOGY_DERIVED"
    source_class[derived] = "SOURCE_DERIVED"
    source_class[explicit] = "SOURCE_EXPLICIT"
    if not np.array_equal(v3_primary, (distance < 0.03)):
        raise AssertionError("REFERENCE_CONTACT_V2_HISTORICAL_V3_MASK_DRIFT")
    if np.any(strong & proximity_only) or np.any(strong & no_contact):
        raise AssertionError("REFERENCE_CONTACT_V2_PARTITION_INVALID")
    return {
        "strong_contact_expected": strong,
        "proximity_only": proximity_only,
        "no_contact_expected": no_contact,
        "source_contact_supported": source_supported,
        "topology_contact_supported": topology_supported,
        "reference_distance_m": distance.astype(np.float32),
        "reference_evidence_class": evidence_class,
        "reference_evidence_source": source_class,
        "historical_v3_primary_mask": v3_primary,
        "reference_contact_evidence_conflict": conflict,
    }


def persistent_windows(
    strong_contact_expected: np.ndarray,
    *,
    evidence_source: np.ndarray,
    distances_m: np.ndarray,
    contract: ReferenceContactContractV2 | None = None,
) -> list[dict[str, object]]:
    """Return every persistent V2 expectation window with auditable composition."""

    frozen = contract or ReferenceContactContractV2()
    strong = np.asarray(strong_contact_expected, dtype=bool)
    source = np.asarray(evidence_source)
    distances = np.asarray(distances_m, dtype=np.float64)
    if strong.ndim != 2 or strong.shape[1] != len(FINGER_ORDER):
        raise ValueError("REFERENCE_CONTACT_V2_STRONG_MASK_MUST_BE_[T,5]")
    if source.shape != strong.shape or distances.shape != strong.shape:
        raise ValueError("REFERENCE_CONTACT_V2_WINDOW_INPUT_SHAPE_INVALID")
    windows: list[dict[str, object]] = []
    for finger_index, finger in enumerate(FINGER_ORDER):
        start = 0
        while start < strong.shape[0]:
            if not strong[start, finger_index]:
                start += 1
                continue
            end = start + 1
            while end < strong.shape[0] and strong[end, finger_index]:
                end += 1
            if end - start >= frozen.persistent_window_control_steps:
                segment_source = source[start:end, finger_index]
                unique, counts = np.unique(segment_source, return_counts=True)
                segment_distance = distances[start:end, finger_index]
                windows.append(
                    {
                        "finger": finger,
                        "start": start,
                        "end": end,
                        "length_control_steps": end - start,
                        "evidence_source_composition": {
                            str(key): int(value) for key, value in zip(unique, counts, strict=True)
                        },
                        "source_or_topology_supported_fraction": float(
                            np.isin(
                                segment_source,
                                ("SOURCE_EXPLICIT", "SOURCE_DERIVED", "TOPOLOGY_DERIVED"),
                            ).mean()
                        ),
                        "distance_m": {
                            "min": float(segment_distance.min()),
                            "mean": float(segment_distance.mean()),
                            "max": float(segment_distance.max()),
                        },
                    }
                )
            start = end
    return windows
