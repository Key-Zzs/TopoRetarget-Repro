"""Pure diagnostics for Stage16 contact-skill collapse localization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PHASE_NAMES = (
    "PRE_CONTACT",
    "APPROACH",
    "CONTACT",
    "GRASP",
    "LIFT",
    "MANIPULATION",
    "TERMINAL",
)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def first_persistent_true(values: np.ndarray, *, consecutive: int) -> int | None:
    """Return the first index of the first qualifying consecutive true run."""

    mask = np.asarray(values, dtype=bool)
    if mask.ndim != 1 or consecutive <= 0:
        raise ValueError("CONTACT_COLLAPSE_PERSISTENCE_INPUT_INVALID")
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if bool(value) else 0
        if run >= consecutive:
            return index - consecutive + 1
    return None


def detect_contact_milestones(
    rows: Sequence[Mapping[str, Any]], *, baseline_contact_episodes: int
) -> dict[str, dict[str, int] | None]:
    """Detect pre-registered contact milestones from ordered update evaluations."""

    ordered = sorted(rows, key=lambda row: int(row["update"]))
    if not ordered or baseline_contact_episodes <= 0:
        raise ValueError("CONTACT_COLLAPSE_MILESTONE_INPUT_INVALID")
    milestones: dict[str, dict[str, int] | None] = {
        "U_FIRST_DEGRADATION": None,
        "U_MAJOR_COLLAPSE": None,
        "U_ZERO_CONTACT": None,
        "U_PERSISTENT_ZERO": None,
    }
    zero_run = 0
    zero_run_start: Mapping[str, Any] | None = None
    for row in ordered:
        episodes = int(row["episodes"])
        contacts = int(row["contact_episodes"])
        if episodes <= 0 or not 0 <= contacts <= episodes:
            raise ValueError("CONTACT_COLLAPSE_MILESTONE_ROW_INVALID")
        receipt = {"update": int(row["update"]), "samples": int(row["samples"])}
        if milestones["U_FIRST_DEGRADATION"] is None and contacts < baseline_contact_episodes:
            milestones["U_FIRST_DEGRADATION"] = receipt
        if milestones["U_MAJOR_COLLAPSE"] is None and 2 * contacts <= episodes:
            milestones["U_MAJOR_COLLAPSE"] = receipt
        if contacts == 0:
            if zero_run == 0:
                zero_run_start = row
            zero_run += 1
            if milestones["U_ZERO_CONTACT"] is None:
                milestones["U_ZERO_CONTACT"] = receipt
            if zero_run == 3 and zero_run_start is not None:
                milestones["U_PERSISTENT_ZERO"] = {
                    **receipt,
                    "run_start_update": int(zero_run_start["update"]),
                    "run_start_samples": int(zero_run_start["samples"]),
                }
        else:
            zero_run = 0
            zero_run_start = None
    return milestones


def quaternion_angle_rad(first_wxyz: np.ndarray, second_wxyz: np.ndarray) -> np.ndarray:
    """Shortest unsigned quaternion distance with arbitrary leading dimensions."""

    first = np.asarray(first_wxyz, dtype=np.float64)
    second = np.asarray(second_wxyz, dtype=np.float64)
    if first.shape != second.shape or first.shape[-1] != 4:
        raise ValueError("CONTACT_COLLAPSE_QUATERNION_SHAPE_INVALID")
    first = first / np.linalg.norm(first, axis=-1, keepdims=True).clip(min=1.0e-12)
    second = second / np.linalg.norm(second, axis=-1, keepdims=True).clip(min=1.0e-12)
    dot = np.abs(np.sum(first * second, axis=-1)).clip(0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _mean_p95(values: np.ndarray) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(vector)), "p95": float(np.quantile(vector, 0.95))}


def command_tracking_metrics(trace: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Decompose reference-to-command and command-to-actual errors."""

    wrist_ref = np.asarray(trace["wrist_reference"], dtype=np.float64)
    wrist_cmd = np.asarray(trace["wrist_target"], dtype=np.float64)
    wrist_actual = np.asarray(trace["wrist_pose"], dtype=np.float64)
    finger_ref = np.asarray(trace["finger_reference"], dtype=np.float64)
    finger_cmd = np.asarray(trace["finger_target"], dtype=np.float64)
    finger_actual = np.asarray(trace["finger_q"], dtype=np.float64)
    if not (
        wrist_ref.shape == wrist_cmd.shape == wrist_actual.shape
        and finger_ref.shape == finger_cmd.shape == finger_actual.shape
        and wrist_ref.shape[-1] == 7
        and finger_ref.shape[-1] == 20
    ):
        raise ValueError("CONTACT_COLLAPSE_COMMAND_TRACE_SHAPE_INVALID")
    phase = np.asarray(trace["phase"])
    focus = np.isin(phase, ("CONTACT", "GRASP"))
    if not np.any(focus):
        raise ValueError("CONTACT_COLLAPSE_CONTACT_GRASP_WINDOW_EMPTY")
    finger_ref_cmd = finger_cmd - finger_ref
    finger_cmd_actual = finger_actual - finger_cmd
    result: dict[str, Any] = {
        "wrist_position_ref_to_command_m": _mean_p95(
            np.linalg.norm(wrist_cmd[:, :3] - wrist_ref[:, :3], axis=-1)
        ),
        "wrist_position_command_to_actual_m": _mean_p95(
            np.linalg.norm(wrist_actual[:, :3] - wrist_cmd[:, :3], axis=-1)
        ),
        "wrist_rotation_ref_to_command_rad": _mean_p95(
            quaternion_angle_rad(wrist_ref[:, 3:], wrist_cmd[:, 3:])
        ),
        "wrist_rotation_command_to_actual_rad": _mean_p95(
            quaternion_angle_rad(wrist_cmd[:, 3:], wrist_actual[:, 3:])
        ),
        "finger_ref_to_command_rad": _mean_p95(np.abs(finger_ref_cmd[focus])),
        "finger_command_to_actual_rad": _mean_p95(np.abs(finger_cmd_actual[focus])),
        "per_joint": [],
        "per_finger": [],
    }
    for joint in range(20):
        values = finger_ref_cmd[focus, joint]
        result["per_joint"].append(
            {
                "joint": joint,
                "mean_abs_rad": float(np.mean(np.abs(values))),
                "p95_abs_rad": float(np.quantile(np.abs(values), 0.95)),
                "signed_mean_rad": float(np.mean(values)),
            }
        )
    for finger, name in enumerate(FINGER_NAMES):
        values = finger_ref_cmd[focus, 4 * finger : 4 * (finger + 1)]
        result["per_finger"].append(
            {
                "finger": name,
                "mean_abs_rad": float(np.mean(np.abs(values))),
                "p95_abs_rad": float(np.quantile(np.abs(values), 0.95)),
                "signed_mean_rad": float(np.mean(values)),
            }
        )
    return result


def lift_timing(
    trace: Mapping[str, np.ndarray],
    *,
    wrist_up_velocity_mps: float = 0.02,
    consecutive_frames: int = 3,
    object_lift_displacement_m: float = 0.005,
) -> dict[str, int | bool | None]:
    """Apply the pre-registered deterministic lift/contact timing rules."""

    phase = np.asarray(trace["phase"])
    contact = np.asarray(
        np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=-1),
        dtype=np.bool_,
    )
    wrist_up = np.asarray(trace["wrist_twist_world"], dtype=np.float64)[:, 2]
    object_z = np.asarray(trace["object_pose"], dtype=np.float64)[:, 2]
    first_contact_values = np.flatnonzero(contact)
    persistent_contact = first_persistent_true(contact, consecutive=consecutive_frames)
    actual_up = first_persistent_true(
        wrist_up > wrist_up_velocity_mps, consecutive=consecutive_frames
    )
    object_lift = first_persistent_true(
        object_z > object_z[0] + object_lift_displacement_m,
        consecutive=consecutive_frames,
    )

    def phase_onset(name: str) -> int | None:
        values = np.flatnonzero(phase == name)
        return None if not values.size else int(values[0])

    return {
        "first_contact": None if not first_contact_values.size else int(first_contact_values[0]),
        "persistent_contact": persistent_contact,
        "reference_grasp_onset": phase_onset("GRASP"),
        "reference_lift_onset": phase_onset("LIFT"),
        "actual_wrist_up_onset": actual_up,
        "object_lift_onset": object_lift,
        "premature_lift": bool(
            actual_up is not None and (persistent_contact is None or actual_up < persistent_contact)
        ),
    }


__all__ = [
    "FINGER_NAMES",
    "PHASE_NAMES",
    "command_tracking_metrics",
    "detect_contact_milestones",
    "first_persistent_true",
    "lift_timing",
    "quaternion_angle_rad",
]
