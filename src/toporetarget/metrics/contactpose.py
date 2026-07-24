"""ContactPose Appendix A.3 metrics, with strict unit and input contracts."""

# ruff: noqa: E501

from __future__ import annotations

from typing import Any

import numpy as np


def _vectors(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape [C,3]")
    if array.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    return array


def contact_precision_eq10(
    source_hand_points: Any,
    robot_hand_points: Any,
    source_object_points: Any,
    robot_object_points: Any,
) -> float:
    """Equation (10), returned in millimetres."""

    hs, hr, os, or_ = (
        _vectors(value, name)
        for value, name in (
            (source_hand_points, "source_hand_points"),
            (robot_hand_points, "robot_hand_points"),
            (source_object_points, "source_object_points"),
            (robot_object_points, "robot_object_points"),
        )
    )
    if not (hs.shape == hr.shape == os.shape == or_.shape):
        raise ValueError("Eq. 10 inputs must have identical [C,3] shapes")
    return float(np.mean(np.linalg.norm((hr - or_) - (hs - os), axis=1)) * 1000.0)


def contact_alignment_eq11(
    source_points: Any,
    robot_points: Any,
    source_parents: Any,
    robot_parents: Any,
) -> float:
    """Equation (11), returned in degrees; zero-length segments are invalid."""

    sr, rr, sp, rp = (
        _vectors(value, name)
        for value, name in (
            (source_points, "source_points"),
            (robot_points, "robot_points"),
            (source_parents, "source_parents"),
            (robot_parents, "robot_parents"),
        )
    )
    if not (sr.shape == rr.shape == sp.shape == rp.shape):
        raise ValueError("Eq. 11 inputs must have identical [C,3] shapes")
    source = sr - sp
    robot = rr - rp
    source_norm = np.linalg.norm(source, axis=1)
    robot_norm = np.linalg.norm(robot, axis=1)
    if np.any(source_norm <= 1e-12) or np.any(robot_norm <= 1e-12):
        raise ValueError("Eq. 11 rejects zero-length source or robot segments")
    cosine = np.sum(source * robot, axis=1) / (source_norm * robot_norm)
    return float(np.mean(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))


def penetration_eq12(signed_distance: Any, *, threshold_m: float = 0.002) -> dict[str, float]:
    """Equation (12) for one unit; positive signed distance means outside."""

    distances = np.asarray(signed_distance, dtype=np.float64)
    if distances.size == 0 or not np.all(np.isfinite(distances)):
        raise ValueError("signed_distance must be finite and non-empty")
    if threshold_m <= 0:
        raise ValueError("threshold_m must be positive")
    penetration = np.maximum(-distances, 0.0)
    per_frame = penetration.max(axis=-1) if penetration.ndim > 1 else penetration
    return {
        "max_penetration_mm": float(np.max(per_frame) * 1000.0),
        "penetration_rate_2mm": float(np.mean(per_frame > threshold_m)),
        "min_signed_distance_mm": float(np.min(distances) * 1000.0),
        "threshold_mm": float(threshold_m * 1000.0),
    }


def compute_contactpose_metrics(inputs: dict[str, Any]) -> dict[str, Any]:
    """Compute only when all official attribution inputs are available."""

    required = {
        "source_hand_points",
        "robot_hand_points",
        "source_object_points",
        "robot_object_points",
        "source_parents",
        "robot_parents",
        "signed_distance",
    }
    missing = sorted(required - set(inputs))
    if missing:
        return {"status": "N/A", "missing_inputs": missing, "semantics": "PAPER_EXACT"}
    result = {
        "status": "pass",
        "semantics": "PAPER_EXACT",
        "contact_precision_eq10": contact_precision_eq10(
            inputs["source_hand_points"],
            inputs["robot_hand_points"],
            inputs["source_object_points"],
            inputs["robot_object_points"],
        ),
        "contact_alignment_eq11": contact_alignment_eq11(
            inputs["source_hand_points"],
            inputs["robot_hand_points"],
            inputs["source_parents"],
            inputs["robot_parents"],
        ),
    }
    result.update(penetration_eq12(inputs["signed_distance"]))
    return result


__all__ = [
    "compute_contactpose_metrics",
    "contact_alignment_eq11",
    "contact_precision_eq10",
    "penetration_eq12",
]
