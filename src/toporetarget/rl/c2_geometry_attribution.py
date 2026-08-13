"""Pure, fail-closed contracts for Stage16 P3-B.5 geometry attribution.

The module consumes already-captured geometry/contact/controller telemetry.  It
does not configure a simulator, mutate a policy, or reinterpret formal gates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

DECISION_SCHEMA = "C2GeometryAttributionDecisionV1"
TEMPORAL_CLASSES = (
    "INITIAL_GEOMETRY_INVALID",
    "CONTACT_TRANSIENT_GEOMETRY_FAILURE",
    "SUSTAINED_LOAD_GEOMETRY_FAILURE",
    "LATE_POLICY_GEOMETRY_FAILURE",
    "UNKNOWN_TEMPORAL_FAILURE",
)
PHYSICS_VARIANTS = {
    "A": (0.50, 1.50),
    "B": (0.50, 1.00),
    "C": (0.25, 1.50),
    "D": (0.25, 1.00),
}


@dataclass(frozen=True)
class GeometryGateV1:
    """The existing absolute gate, represented without any relaxation."""

    max_penetration_exclusive_m: float
    p95_penetration_inclusive_m: float
    inter_finger_inclusive_m: float

    def passes(self, *, maximum_m: float, p95_m: float, inter_finger_m: float) -> bool:
        return bool(
            maximum_m < self.max_penetration_exclusive_m
            and p95_m <= self.p95_penetration_inclusive_m
            and inter_finger_m <= self.inter_finger_inclusive_m
        )


def decision_contract(gate: GeometryGateV1) -> dict[str, object]:
    """Return the fixed pre-result decision contract for P3-B.5."""

    return {
        "schema_version": DECISION_SCHEMA,
        "geometry_gate": asdict(gate),
        "physics_variants": {
            name: {"gravity_scale": gravity, "friction_scale": friction}
            for name, (gravity, friction) in PHYSICS_VARIANTS.items()
        },
        "frozen_invariants": [
            "mass",
            "inertia",
            "restitution",
            "damping",
            "solver_iterations",
            "dt",
            "substeps",
            "contact_offset",
            "rest_offset",
            "collision_geometry",
            "controller",
            "action_scale",
            "rsi_reset",
            "reward",
            "policy_weights",
        ],
        "prohibited": [
            "ppo_training",
            "optimizer_step",
            "support",
            "guidance",
            "rollout_object_state_write",
            "rollout_wrist_root_write",
            "geometry_gate_mutation",
        ],
        "criteria": {
            "initial_geometry_invalid": "reset frame violates the frozen absolute gate",
            "friction_moderate": "A_to_B_p95_or_max_reduction_at_least_30_percent and B "
            "improves more than C",
            "gravity_moderate": "A_to_C_p95_or_max_reduction_at_least_30_percent and C "
            "improves more than B",
            "mixed": "A fails; B and C have limited improvement; D passes",
            "policy_reaction": "open-loop is safe while frozen-policy closed-loop fails under "
            "the same alternate physics",
        },
    }


def _one_dimensional(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name}_MUST_BE_FINITE_NONEMPTY_VECTOR")
    return result


def first_violation(
    frame_worst_penetration_m: np.ndarray,
    *,
    gate: GeometryGateV1,
) -> int | None:
    """Return the first frame violating either formal hand-object limit."""

    values = _one_dimensional(frame_worst_penetration_m, name="FRAME_WORST_PENETRATION")
    # A frame is an actionable absolute-geometry violation once it exceeds the
    # frozen 3 mm p95 limit; the strict 10 mm limit remains the catastrophic
    # condition.  Episode p95 aggregation is intentionally not used to hide a
    # one-frame onset.
    failed = values > gate.p95_penetration_inclusive_m
    return None if not failed.any() else int(np.flatnonzero(failed)[0])


def temporal_classification(
    *,
    first_violation_frame: int | None,
    maximum_frame: int,
    frame_count: int,
    first_contact_frame: int | None,
    reset_violates: bool,
) -> str:
    """Classify timing without treating a contact observation as a cause."""

    if frame_count < 1 or maximum_frame < 0 or maximum_frame >= frame_count:
        raise ValueError("TEMPORAL_CLASSIFICATION_FRAME_RANGE_INVALID")
    if reset_violates:
        return "INITIAL_GEOMETRY_INVALID"
    if first_violation_frame is None:
        return "UNKNOWN_TEMPORAL_FAILURE"
    if first_contact_frame is None:
        return "UNKNOWN_TEMPORAL_FAILURE"
    if first_violation_frame <= first_contact_frame + 5:
        return "CONTACT_TRANSIENT_GEOMETRY_FAILURE"
    if first_violation_frame >= int(frame_count * 0.75):
        return "LATE_POLICY_GEOMETRY_FAILURE"
    return "SUSTAINED_LOAD_GEOMETRY_FAILURE"


def controller_indicators(
    *,
    finger_target: np.ndarray,
    finger_actual: np.ndarray,
    wrist_target: np.ndarray,
    wrist_actual: np.ndarray,
    actuator_effort: np.ndarray,
    effort_limit: float | None,
    contact_force_world: np.ndarray,
    center_frame: int,
    half_window: int = 10,
) -> dict[str, float | None]:
    """Extract bounded, recorded controller indicators around a violation."""

    finger_t = np.asarray(finger_target, dtype=np.float64)
    finger_a = np.asarray(finger_actual, dtype=np.float64)
    wrist_t = np.asarray(wrist_target, dtype=np.float64)
    wrist_a = np.asarray(wrist_actual, dtype=np.float64)
    effort = np.asarray(actuator_effort, dtype=np.float64)
    force = np.asarray(contact_force_world, dtype=np.float64)
    if (
        finger_t.shape != finger_a.shape
        or wrist_t.shape != wrist_a.shape
        or finger_t.ndim != 2
        or wrist_t.shape != (finger_t.shape[0], 7)
        or effort.shape != (finger_t.shape[0], 26)
        or force.shape != (finger_t.shape[0], 3)
    ):
        raise ValueError("CONTROLLER_INDICATOR_SHAPE_INVALID")
    if not all(
        np.isfinite(value).all() for value in (finger_t, finger_a, wrist_t, wrist_a, effort, force)
    ):
        raise ValueError("CONTROLLER_INDICATOR_NONFINITE")
    start = max(0, center_frame - half_window)
    stop = min(finger_t.shape[0], center_frame + half_window + 1)
    finger_error = np.abs(finger_t[start:stop] - finger_a[start:stop])
    wrist_error = np.linalg.norm(wrist_t[start:stop, :3] - wrist_a[start:stop, :3], axis=-1)
    # The first six entries are virtual-wrist force/torque components with
    # mixed N/Nm units.  The remaining twenty entries are identically bounded
    # finger drives, so only those may be compared to ``effort_limit``.
    finger_effort_window = np.abs(effort[start:stop, 6:])
    return {
        "window_start": float(start),
        "window_end_exclusive": float(stop),
        "joint_target_error_peak_rad": float(finger_error.max(initial=0.0)),
        "wrist_target_error_peak_m": float(wrist_error.max(initial=0.0)),
        "contact_force_spike_n": float(np.linalg.norm(force[start:stop], axis=-1).max(initial=0.0)),
        "effort_saturation_fraction": (
            None
            if effort_limit is None or effort_limit <= 0.0
            else float(np.mean(finger_effort_window >= effort_limit))
        ),
    }


def relative_reduction(baseline: float, alternate: float) -> float:
    if baseline < 0.0 or alternate < 0.0:
        raise ValueError("PENETRATION_MUST_BE_NONNEGATIVE")
    return 0.0 if baseline == 0.0 else float((baseline - alternate) / baseline)


def physics_attribution(rows: Mapping[str, Mapping[str, float | bool]], *, mode: str) -> str:
    """Classify A/B/C/D evidence under the frozen threshold-free rule."""

    if set(rows) != set(PHYSICS_VARIANTS):
        raise ValueError("COUNTERFACTUAL_VARIANT_SET_INVALID")
    for name, row in rows.items():
        if not {"p95_penetration_m", "max_penetration_m", "gate_pass"}.issubset(row):
            raise ValueError(f"COUNTERFACTUAL_ROW_INCOMPLETE:{name}")
    a = rows["A"]
    b = rows["B"]
    c = rows["C"]
    d = rows["D"]
    a_metric = max(float(a["p95_penetration_m"]), float(a["max_penetration_m"]))
    b_metric = max(float(b["p95_penetration_m"]), float(b["max_penetration_m"]))
    c_metric = max(float(c["p95_penetration_m"]), float(c["max_penetration_m"]))
    b_reduction = relative_reduction(a_metric, b_metric)
    c_reduction = relative_reduction(a_metric, c_metric)
    if (
        not bool(a["gate_pass"])
        and bool(d["gate_pass"])
        and not bool(b["gate_pass"])
        and not bool(c["gate_pass"])
    ):
        return "GRAVITY_FRICTION_COUPLING"
    if bool(b["gate_pass"]) or (b_reduction >= 0.30 and b_reduction > c_reduction):
        return (
            "HIGH_FRICTION_STICKING_PRIMARY"
            if mode == "closed_loop"
            else "HIGH_FRICTION_CONTRIBUTOR"
        )
    if bool(c["gate_pass"]) or (c_reduction >= 0.30 and c_reduction > b_reduction):
        return "GRAVITY_LOAD_PRIMARY" if mode == "closed_loop" else "GRAVITY_LOAD_CONTRIBUTOR"
    return "PHYSICS_PARAMETER_EFFECT_NOT_SUPPORTED"


def root_cause_matrix(
    *,
    reset_fraction: float,
    friction_label: str,
    gravity_label: str,
    controller_overdrive: bool,
    policy_reaction: bool,
    proxy_discrepancy: bool | None,
) -> dict[str, str]:
    """Map pre-defined evidence into the handoff's four-level matrix."""

    if not 0.0 <= reset_fraction <= 1.0:
        raise ValueError("RESET_FAILURE_FRACTION_INVALID")
    return {
        "Reset": (
            "STRONG" if reset_fraction >= 0.75 else "MODERATE" if reset_fraction >= 0.25 else "WEAK"
        ),
        "Gravity": "MODERATE" if "GRAVITY" in gravity_label else "NOT_SUPPORTED",
        "Friction": "MODERATE" if "FRICTION" in friction_label else "NOT_SUPPORTED",
        "Controller": "MODERATE" if controller_overdrive else "NOT_SUPPORTED",
        "Policy reaction": "MODERATE" if policy_reaction else "NOT_SUPPORTED",
        "Proxy": (
            "MODERATE"
            if proxy_discrepancy is True
            else "NOT_SUPPORTED"
            if proxy_discrepancy is False
            else "WEAK"
        ),
    }


__all__ = [
    "DECISION_SCHEMA",
    "PHYSICS_VARIANTS",
    "TEMPORAL_CLASSES",
    "GeometryGateV1",
    "controller_indicators",
    "decision_contract",
    "first_violation",
    "physics_attribution",
    "relative_reduction",
    "root_cause_matrix",
    "temporal_classification",
]
