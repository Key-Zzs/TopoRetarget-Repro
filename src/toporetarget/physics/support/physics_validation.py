"""Backend-neutral summaries for causal support experiments.

Isaac/PhysX runners feed their recorded telemetry here.  Keeping the reduction
outside the environment makes the causal gates testable without importing
Isaac, and prevents a visual replay from being mistaken for a physics rollout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from .types import PhysicsValidation


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())
    except (TypeError, ValueError):
        return False


def summarize_static_support_test(
    records: Sequence[Mapping[str, object]],
    *,
    support_active: bool,
    mass_kg: float,
    gravity_world_mps2: Sequence[float],
    support_normal: Sequence[float],
    drift_tolerance_m: float = 0.01,
    rotation_tolerance_rad: float = 0.1,
) -> dict[str, object]:
    """Summarize object-only settling telemetry with explicit causality flags."""

    if not records:
        raise ValueError("SUPPORT_STATIC_TEST_RECORDS_EMPTY")
    required = {"position_world_m", "linear_velocity_world_mps", "angular_velocity_world_radps"}
    if any(not required.issubset(row) for row in records):
        raise ValueError("SUPPORT_STATIC_TEST_TELEMETRY_INCOMPLETE")
    positions = np.asarray([row["position_world_m"] for row in records], dtype=np.float64)
    linear = np.asarray([row["linear_velocity_world_mps"] for row in records], dtype=np.float64)
    angular = np.asarray([row["angular_velocity_world_radps"] for row in records], dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or not all(_finite(value) for value in (positions, linear, angular))
    ):
        raise ValueError("SUPPORT_STATIC_TEST_TELEMETRY_NONFINITE")
    gravity = np.asarray(gravity_world_mps2, dtype=np.float64)
    normal = np.asarray(support_normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    displacement = positions - positions[0]
    vertical_drift = np.abs(displacement @ normal)
    support_force = np.asarray(
        [row.get("support_force_world_n", [0.0, 0.0, 0.0]) for row in records], dtype=np.float64
    )
    orientation_values = [row.get("orientation_world_wxyz") for row in records]
    if all(value is not None for value in orientation_values):
        orientations = np.asarray(orientation_values, dtype=np.float64)
        if orientations.shape != (len(records), 4) or not _finite(orientations):
            raise ValueError("SUPPORT_STATIC_TEST_ORIENTATION_NONFINITE")
        orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)
        orientation_dot = np.abs(orientations @ orientations[0])
        rotation_drift = 2.0 * np.arccos(np.clip(orientation_dot, -1.0, 1.0))
        orientation_source = "quaternion_pose_drift_from_first_record"
    else:
        # Older backend-neutral callers may only provide angular velocity.  Do
        # not pretend it is pose drift, but retain a conservative compatibility
        # fallback for those records.
        rotation_drift = np.linalg.norm(angular, axis=1)
        orientation_source = "angular_velocity_fallback"
    normal_force = support_force @ normal
    contact = np.asarray([bool(row.get("support_contact", False)) for row in records], dtype=bool)
    stable = bool(
        np.max(np.linalg.norm(displacement, axis=1)) <= drift_tolerance_m
        and np.max(rotation_drift) <= rotation_tolerance_rad
        and np.max(np.linalg.norm(linear, axis=1)) <= 0.25
        and (not support_active or bool(contact.any()))
    )
    expected_force = float(mass_kg * np.linalg.norm(gravity))
    force_ratio = float(np.mean(normal_force[contact]) / expected_force) if contact.any() else 0.0
    return {
        "status": "PASS" if stable else "FAIL",
        "support_active": support_active,
        "record_count": int(len(records)),
        "position_drift_max_m": float(np.max(np.linalg.norm(displacement, axis=1))),
        "vertical_drift_max_m": float(np.max(vertical_drift)),
        "xy_drift_max_m": float(
            np.max(np.linalg.norm(displacement - np.outer(displacement @ normal, normal), axis=1))
        ),
        "rotation_drift_max_rad": float(np.max(rotation_drift)),
        "rotation_drift_source": orientation_source,
        "angular_speed_max_radps": float(np.max(np.linalg.norm(angular, axis=1))),
        "linear_speed_max_mps": float(np.max(np.linalg.norm(linear, axis=1))),
        "support_contact_frames": int(np.count_nonzero(contact)),
        "support_contact_fraction": float(np.mean(contact)),
        "support_normal_force_mean_n": (
            float(np.mean(normal_force[contact])) if contact.any() else 0.0
        ),
        "support_normal_force_p05_n": (
            float(np.percentile(normal_force[contact], 5.0)) if contact.any() else 0.0
        ),
        "support_normal_force_p95_n": (
            float(np.percentile(normal_force[contact], 95.0)) if contact.any() else 0.0
        ),
        "expected_mg_n": expected_force,
        "support_force_to_mg_ratio": force_ratio,
        "gravity_world_mps2": gravity.tolist(),
        "support_normal": normal.tolist(),
        "external_guidance": bool(any(row.get("external_guidance", False) for row in records)),
        "object_rollout_state_writes": int(
            sum(int(cast(Any, row.get("object_state_writes", 0))) for row in records)
        ),
        "hidden_attachment": bool(any(row.get("hidden_attachment", False) for row in records)),
        "kinematic_support_force": bool(
            any(row.get("kinematic_support_force", False) for row in records)
        ),
    }


def compare_support_counterfactuals(
    with_support: Mapping[str, object],
    without_support: Mapping[str, object],
    *,
    fall_drift_threshold_m: float = 0.05,
) -> dict[str, object]:
    """Establish the support effect from matched object-only A/B runs."""

    with_drift = float(cast(Any, with_support.get("position_drift_max_m", np.nan)))
    without_drift = float(cast(Any, without_support.get("position_drift_max_m", np.nan)))
    without_falls = without_drift >= fall_drift_threshold_m
    with_stable = with_support.get("status") == "PASS"
    causal = bool(with_stable and without_falls)
    return {
        "status": "PASS" if causal else "FAIL",
        "with_support_position_drift_max_m": with_drift,
        "without_support_position_drift_max_m": without_drift,
        "without_support_fall_threshold_m": fall_drift_threshold_m,
        "without_support_falls": without_falls,
        "with_support_stable": with_stable,
        "causal_support_effect": causal,
        "interpretation": (
            "with-support physically prevents matched gravity fall"
            if causal
            else "causal support effect not established"
        ),
    }


def build_physics_validation(
    *,
    with_support: Mapping[str, object],
    without_support: Mapping[str, object],
    causal_comparison: Mapping[str, object],
) -> PhysicsValidation:
    flags = (with_support, without_support)
    safe_causality = all(
        not bool(row.get(key, False))
        for row in flags
        for key in ("external_guidance", "hidden_attachment", "kinematic_support_force")
    ) and all(int(cast(Any, row.get("object_rollout_state_writes", 0))) == 0 for row in flags)
    status = "PASS" if causal_comparison.get("status") == "PASS" and safe_causality else "FAIL"
    return PhysicsValidation(
        object_only_with_support=dict(with_support),
        object_only_without_support=dict(without_support),
        causal_comparison=dict(causal_comparison),
        status=status,
        external_guidance=not safe_causality,
        object_rollout_state_writes=sum(
            int(cast(Any, row.get("object_rollout_state_writes", 0))) for row in flags
        ),
        hidden_attachment=any(bool(row.get("hidden_attachment", False)) for row in flags),
        kinematic_support_force=any(
            bool(row.get("kinematic_support_force", False)) for row in flags
        ),
    )


__all__ = [
    "build_physics_validation",
    "compare_support_counterfactuals",
    "summarize_static_support_test",
]
