"""Versioned, offline-only Stage16 dynamic physical qualification helpers.

This module intentionally consumes recorded trajectory telemetry.  It neither
loads a policy nor touches an Isaac scene, which keeps the new qualification
receipt separate from the immutable Evaluation Suite V2 result.
"""

# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .grasp_lift_skill_collapse import grasp_lift_episode_metrics

DYNAMIC_PHYSICAL_QUALIFICATION_SCHEMA = "Stage16DynamicPhysicalQualificationV1"
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
PHASE_NAMES = np.asarray(
    ("PRE_CONTACT", "APPROACH", "CONTACT", "GRASP", "LIFT", "MANIPULATION", "TERMINAL")
)


@dataclass(frozen=True)
class DynamicTerminalGate:
    """Legacy terminal constants applied to reference-relative twist error."""

    terminal_window_control_steps: int
    terminal_linear_speed_mps: float
    terminal_angular_speed_radps: float
    terminal_free_object_linear_speed_mps: float
    terminal_free_object_angular_speed_radps: float

    @classmethod
    def from_frozen_gate(cls, value: Mapping[str, object]) -> DynamicTerminalGate:
        required = (
            "terminal_window_control_steps",
            "terminal_linear_speed_mps",
            "terminal_angular_speed_radps",
            "terminal_free_object_linear_speed_mps",
            "terminal_free_object_angular_speed_radps",
        )
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f"DYNAMIC_QUALIFICATION_LEGACY_GATE_MISSING:{','.join(missing)}")
        result = cls(**{name: value[name] for name in required})  # type: ignore[arg-type]
        if result.terminal_window_control_steps <= 0 or any(
            scalar <= 0.0
            for scalar in (
                result.terminal_linear_speed_mps,
                result.terminal_angular_speed_radps,
                result.terminal_free_object_linear_speed_mps,
                result.terminal_free_object_angular_speed_radps,
            )
        ):
            raise ValueError("DYNAMIC_QUALIFICATION_LEGACY_GATE_INVALID")
        return result

    def as_dict(self) -> dict[str, object]:
        return {
            "terminal_window_control_steps": self.terminal_window_control_steps,
            "terminal_linear_speed_mps": self.terminal_linear_speed_mps,
            "terminal_angular_speed_radps": self.terminal_angular_speed_radps,
            "terminal_free_object_linear_speed_mps": self.terminal_free_object_linear_speed_mps,
            "terminal_free_object_angular_speed_radps": self.terminal_free_object_angular_speed_radps,
        }


def phase_labels_from_reference_index(
    reference_index: np.ndarray, *, frame_count: int = 321
) -> np.ndarray:
    """Return the existing Stage16 reference-index-only phase labels.

    This is the same mapping emitted by the V4 runtime's ``phase_code``:
    ``floor(reference_index * 7 / frame_count)`` clipped to the seven ordered
    labels.  It is metadata only and cannot affect execution.
    """

    indices = np.asarray(reference_index, dtype=np.int64)
    if (
        indices.ndim != 1
        or not len(indices)
        or np.any(indices < 0)
        or np.any(indices >= frame_count)
    ):
        raise ValueError("DYNAMIC_QUALIFICATION_REFERENCE_INDEX_INVALID")
    code = np.clip((indices * len(PHASE_NAMES)) // frame_count, 0, len(PHASE_NAMES) - 1)
    return PHASE_NAMES[code]


def _valid_rows(value: np.ndarray, *, length: int) -> np.ndarray:
    result = np.asarray(value, dtype=bool)
    if result.shape != (length,) or not result[1:].all() or result[0]:
        raise ValueError("DYNAMIC_QUALIFICATION_FORCE_VALIDITY_INVALID")
    return result


def _twist_array(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 6 or not np.isfinite(result).all():
        raise ValueError(f"DYNAMIC_QUALIFICATION_{name}_INVALID")
    return result


def dynamic_twist_metrics(
    *,
    actual_twist_world: np.ndarray,
    reference_twist_world: np.ndarray,
    hand_object_contact: np.ndarray,
    valid: np.ndarray,
    gate: DynamicTerminalGate,
) -> dict[str, object]:
    """Measure world-frame reference-relative object twist and its inherited gate.

    The values remain world-frame ``actual - reference`` quantities.  Unlike
    the V2 terminal-stability evaluator, no absolute object-world velocity is
    consulted when calculating ``reference_twist_dynamic_pass``.
    """

    actual = _twist_array(actual_twist_world, name="ACTUAL_TWIST")
    reference = _twist_array(reference_twist_world, name="REFERENCE_TWIST")
    if actual.shape != reference.shape:
        raise ValueError("DYNAMIC_QUALIFICATION_TWIST_SHAPE_MISMATCH")
    contacts = np.asarray(hand_object_contact, dtype=bool)
    if contacts.shape != (len(actual),):
        raise ValueError("DYNAMIC_QUALIFICATION_HAND_CONTACT_INVALID")
    rows = _valid_rows(valid, length=len(actual))
    delta = actual - reference
    linear = np.linalg.norm(delta[:, :3], axis=-1)
    angular = np.linalg.norm(delta[:, 3:], axis=-1)
    selected = np.flatnonzero(rows)
    terminal = selected[-min(gate.terminal_window_control_steps, len(selected)) :]
    linear_limit = np.where(
        contacts[terminal],
        gate.terminal_linear_speed_mps,
        gate.terminal_free_object_linear_speed_mps,
    )
    angular_limit = np.where(
        contacts[terminal],
        gate.terminal_angular_speed_radps,
        gate.terminal_free_object_angular_speed_radps,
    )
    return {
        "frame_convention": "world_frame__delta_equals_actual_minus_reference",
        "full_motion": {
            "Delta_v_mean_mps": float(linear[rows].mean()),
            "Delta_v_p95_mps": float(np.quantile(linear[rows], 0.95)),
            "Delta_omega_mean_radps": float(angular[rows].mean()),
            "Delta_omega_p95_radps": float(np.quantile(angular[rows], 0.95)),
        },
        "legacy_terminal_window_equivalent": {
            "control_steps": int(len(terminal)),
            "Delta_v_terminal_mean_mps": float(linear[terminal].mean()),
            "Delta_v_terminal_p95_mps": float(np.quantile(linear[terminal], 0.95)),
            "Delta_v_terminal_max_mps": float(linear[terminal].max()),
            "Delta_omega_terminal_mean_radps": float(angular[terminal].mean()),
            "Delta_omega_terminal_p95_radps": float(np.quantile(angular[terminal], 0.95)),
            "Delta_omega_terminal_max_radps": float(angular[terminal].max()),
            "linear_limit_mps": linear_limit.tolist(),
            "angular_limit_radps": angular_limit.tolist(),
        },
        "reference_twist_dynamic_pass": bool(
            np.all(linear[terminal] <= linear_limit) and np.all(angular[terminal] <= angular_limit)
        ),
        "absolute_world_terminal_velocity_used": False,
    }


def dynamic_interaction_metrics(trace: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Reuse the established persistent-grasp/lift semantic for V4 telemetry."""

    required = (
        "hand_object_pair_force_valid",
        "hand_object_pair_presence",
        "tip_pair_presence",
        "fingertip_object_pair_force_world",
        "source_contact_mask",
        "object_pose",
        "phase",
    )
    missing = [name for name in required if name not in trace]
    if missing:
        raise ValueError(f"DYNAMIC_QUALIFICATION_INTERACTION_FIELD_MISSING:{','.join(missing)}")
    frame_count = len(np.asarray(trace["object_pose"]))
    reward = np.asarray(trace.get("r_contact_v4", np.zeros(frame_count)), dtype=np.float64)
    result = grasp_lift_episode_metrics(
        {
            "hand_object_pair_force_valid": np.asarray(trace["hand_object_pair_force_valid"]),
            "hand_object_pair_presence": np.asarray(trace["hand_object_pair_presence"]),
            "actual_contact_mask": np.asarray(trace["tip_pair_presence"]),
            "fingertip_object_pair_force_world": np.asarray(
                trace["fingertip_object_pair_force_world"]
            ),
            "reference_contact_mask": np.asarray(trace["source_contact_mask"]),
            "contact_reward": reward,
            "phase": np.asarray(trace["phase"]),
            "object_pose": np.asarray(trace["object_pose"]),
        }
    )
    result["interaction_dynamic_pass"] = bool(
        result["persistent_grasp"]
        and result["grasp_and_lift"]
        and result["persistent_grasp_at_semantic_lift"]
    )
    result["interaction_contract"] = (
        "persistent_multi_finger_grasp_and_object_lift_with_grasp_at_reference_LIFT"
    )
    return result


def dynamic_qualification(
    *,
    legacy_kinematic_success: bool,
    interaction: Mapping[str, object],
    twist: Mapping[str, object],
    geometry_safe: bool,
    action_bounds_safe: bool,
    causal_execution_safe: bool,
) -> dict[str, object]:
    """Compose the additive dynamic result without modifying V2 fields."""

    interaction_pass = bool(interaction["interaction_dynamic_pass"])
    twist_pass = bool(twist["reference_twist_dynamic_pass"])
    result = {
        "SRkin": bool(legacy_kinematic_success),
        "interaction_dynamic": interaction_pass,
        "reference_twist_dynamic": twist_pass,
        "geometry_safe": bool(geometry_safe),
        "action_bounds_safe": bool(action_bounds_safe),
        "causal_execution_safe": bool(causal_execution_safe),
        "ABSOLUTE_WORLD_TERMINAL_ZERO_SPEED_REQUIRED": "NO",
        "SR_HOLD_IMPLEMENTED": "NO",
    }
    result["SR_dynamic"] = bool(
        all(
            value
            for key, value in result.items()
            if key.endswith("safe")
            or key in {"SRkin", "interaction_dynamic", "reference_twist_dynamic"}
        )
    )
    if result["SR_dynamic"]:
        primary = "DYNAMIC_SUCCESS"
    elif not result["SRkin"]:
        primary = "KINEMATIC_FAILURE"
    elif not result["interaction_dynamic"]:
        primary = "DYNAMIC_INTERACTION_FAILURE"
    elif not result["reference_twist_dynamic"]:
        primary = "DYNAMIC_TWIST_FAILURE"
    elif not result["geometry_safe"]:
        primary = "GEOMETRY_FAILURE"
    else:
        primary = "CAUSALITY_FAILURE"
    result["primary_classification"] = primary
    result["secondary_failures"] = [
        name
        for name, passed in (
            ("KINEMATIC_FAILURE", result["SRkin"]),
            ("DYNAMIC_INTERACTION_FAILURE", result["interaction_dynamic"]),
            ("DYNAMIC_TWIST_FAILURE", result["reference_twist_dynamic"]),
            ("GEOMETRY_FAILURE", result["geometry_safe"]),
            ("ACTION_BOUNDS_FAILURE", result["action_bounds_safe"]),
            ("CAUSALITY_FAILURE", result["causal_execution_safe"]),
        )
        if not passed and name != primary
    ]
    return result


def object_local_points(points_world: np.ndarray, object_pose_wxyz: np.ndarray) -> np.ndarray:
    """Express points in each frame's object coordinate system."""

    points = np.asarray(points_world, dtype=np.float64)
    pose = np.asarray(object_pose_wxyz, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3 or pose.shape != (points.shape[0], 7):
        raise ValueError("DYNAMIC_QUALIFICATION_OBJECT_LOCAL_SHAPE_INVALID")
    quaternion = pose[:, 3:]
    norm = np.linalg.norm(quaternion, axis=-1)
    if not np.isfinite(points).all() or not np.isfinite(pose).all() or np.any(norm < 1.0e-8):
        raise ValueError("DYNAMIC_QUALIFICATION_OBJECT_LOCAL_NONFINITE")
    w, x, y, z = (quaternion[:, index] / norm for index in range(4))
    rotation = np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), axis=-1),
            np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), axis=-1),
            np.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), axis=-1),
        ),
        axis=-2,
    )
    return np.einsum("tji,tfi->tfj", rotation, points - pose[:, None, :3])


__all__ = [
    "DYNAMIC_PHYSICAL_QUALIFICATION_SCHEMA",
    "FINGER_ORDER",
    "PHASE_NAMES",
    "DynamicTerminalGate",
    "dynamic_interaction_metrics",
    "dynamic_qualification",
    "dynamic_twist_metrics",
    "object_local_points",
    "phase_labels_from_reference_index",
]
