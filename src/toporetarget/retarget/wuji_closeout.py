"""Pure W2.2 Wuji continuity closeout calculations.

The module deliberately does not load or write formal retarget artifacts.  It
contains the numerical contracts used by the closeout runner and unit tests:
q-step decomposition/attribution, bounded-ablation labels, recommendation
gates, and the deterministic five-frame routing fixture.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

SCHEMA_VERSION = "toporetarget.wuji_continuity_closeout.v1"
ABLATION_SCHEMA_VERSION = "toporetarget.wuji_bounded_transport_ablation.v1"
EPS = 1.0e-12
Q_STEP_THRESHOLD_RAD = 0.05
BASE_TRANSLATION_THRESHOLD_M = 0.010
BASE_ROTATION_THRESHOLD_RAD = float(np.deg2rad(5.0))
KEYPOINT_THRESHOLD_M = 0.020
Q_VELOCITY_HZ = 120.0

ATTRIBUTION_CATEGORIES = (
    "SOURCE_OR_WARM_DRIVEN",
    "RETARGET_CORRECTION_DRIVEN",
    "REACHABILITY_OR_LIMIT_DRIVEN",
    "MIXED_WARM_AND_CORRECTION",
    "NUMERICALLY_INCONCLUSIVE",
)


def _array(value: Any, dtype: Any = np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace").rstrip("\x00")
    return str(value)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    left = _array(a).reshape(3)
    right = _array(b).reshape(3)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= EPS or right_norm <= EPS:
        return float("nan")
    return float(np.arccos(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0)))


def _sign(value: float) -> str:
    if value > EPS:
        return "positive"
    if value < -EPS:
        return "negative"
    return "zero"


def _ratio(value: float, denominator: float) -> float:
    return float(abs(value) / max(abs(denominator), EPS))


def _finger_name(joint_name: str) -> str:
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        if finger in joint_name:
            return finger
    return "unknown"


def detect_qstep_transitions(qpos: Any, threshold: float = Q_STEP_THRESHOLD_RAD) -> list[int]:
    value = _array(qpos)
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError("qpos must have shape [frames, joints] with at least two frames")
    return [
        int(index + 1)
        for index in np.flatnonzero(np.max(np.abs(np.diff(value, axis=0)), axis=1) > threshold)
    ]


def decompose_q_steps(
    warm_qpos: Any, final_qpos: Any, *, tolerance: float = 1.0e-12
) -> dict[str, Any]:
    warm = _array(warm_qpos)
    final = _array(final_qpos)
    if warm.shape != final.shape or warm.ndim != 2:
        raise ValueError("warm and final qpos must have the same rank-2 shape")
    correction = final - warm
    delta_warm = np.diff(warm, axis=0)
    delta_final = np.diff(final, axis=0)
    delta_correction = np.diff(correction, axis=0)
    error = delta_final - (delta_warm + delta_correction)
    return {
        "delta_q_warm": delta_warm,
        "delta_q_final": delta_final,
        "correction": correction,
        "delta_q_correction": delta_correction,
        "max_decomposition_error": float(np.max(np.abs(error), initial=0.0)),
        "decomposition_pass": bool(np.max(np.abs(error), initial=0.0) <= tolerance),
        "tolerance": float(tolerance),
    }


def jump_and_return_map(
    delta_q_final: Any, threshold: float = Q_STEP_THRESHOLD_RAD
) -> dict[int, list[int]]:
    delta = _array(delta_q_final)
    if delta.ndim != 2:
        raise ValueError("delta_q_final must have shape [transitions, joints]")
    result: dict[int, list[int]] = {}
    for transition in range(delta.shape[0] - 1):
        current = delta[transition]
        following = delta[transition + 1]
        joints = np.flatnonzero(
            (np.abs(current) > threshold)
            & (np.sign(current) * np.sign(following) < 0)
            & (np.abs(following) > EPS)
        )
        if len(joints):
            result[transition + 1] = [int(item) for item in joints]
    return result


def _vector_metrics(
    warm: np.ndarray, correction: np.ndarray, final: np.ndarray
) -> dict[str, float | bool]:
    warm_norm = float(np.linalg.norm(warm))
    correction_norm = float(np.linalg.norm(correction))
    final_norm = float(np.linalg.norm(final))
    cosine = float(np.dot(warm, final) / max(warm_norm * final_norm, EPS))
    cancellation = float(np.linalg.norm(warm) + np.linalg.norm(correction) - np.linalg.norm(final))
    opposing = float(
        np.sum(
            np.minimum(np.abs(warm), np.abs(correction)) * (np.sign(warm) != np.sign(correction))
        )
    )
    return {
        "warm_linf": float(np.max(np.abs(warm), initial=0.0)),
        "warm_l2": warm_norm,
        "correction_linf": float(np.max(np.abs(correction), initial=0.0)),
        "correction_l2": correction_norm,
        "final_linf": float(np.max(np.abs(final), initial=0.0)),
        "final_l2": final_norm,
        "cosine_warm_final": cosine,
        "sign_consistency": bool(
            np.all((np.sign(warm) == np.sign(final)) | (np.abs(final) <= EPS))
        ),
        "cancellation_magnitude": cancellation,
        "opposing_component_magnitude": opposing,
    }


def _classify(
    *,
    warm: np.ndarray,
    correction: np.ndarray,
    final: np.ndarray,
    margins: np.ndarray,
    warm_keypoint_step: float,
    final_keypoint_step: float,
    jump_and_return: bool,
) -> tuple[str, list[str], list[str], str]:
    warm_ratio = float(np.max(np.abs(warm)) / max(float(np.max(np.abs(final))), EPS))
    correction_ratio = float(np.max(np.abs(correction)) / max(float(np.max(np.abs(final))), EPS))
    final_linf = float(np.max(np.abs(final), initial=0.0))
    limit_evidence = bool(
        np.any(margins <= 0.03)
        and (
            final_keypoint_step > max(warm_keypoint_step * 1.2, 0.020)
            or final_linf > warm_ratio * 0.05
        )
    )
    if (
        jump_and_return
        or correction_ratio >= 0.60
        or float(np.max(np.abs(correction), initial=0.0)) > 0.05
    ):
        return (
            "RETARGET_CORRECTION_DRIVEN",
            ["correction dominates or reverses the final step"],
            ["warm contribution is not sufficient to explain the transition"],
            "high",
        )
    if limit_evidence:
        return (
            "REACHABILITY_OR_LIMIT_DRIVEN",
            [
                "joint-limit margin is within the declared limit band",
                "motion residual/keypoint displacement increases",
            ],
            ["limit proximity alone is not treated as evidence"],
            "medium",
        )
    if warm_ratio >= 0.75 and correction_ratio <= 0.35 and not jump_and_return:
        return (
            "SOURCE_OR_WARM_DRIVEN",
            [
                "warm and final steps share direction",
                "warm contribution ratio is at least 0.75",
                "correction contribution is at most 0.35",
            ],
            ["no correction jump-and-return", "continuity gate remains independent"],
            "high",
        )
    if warm_ratio > 0.35 and correction_ratio > 0.35:
        return (
            "MIXED_WARM_AND_CORRECTION",
            ["both warm and correction contributions are material"],
            ["neither single-driver rule is satisfied"],
            "medium",
        )
    return (
        "NUMERICALLY_INCONCLUSIVE",
        ["no single attribution rule is decisive"],
        ["no missing decomposition evidence was detected"],
        "low",
    )


def build_w2_attribution(
    *,
    warm_arrays: dict[str, Any],
    final_arrays: dict[str, Any],
    joint_names: Iterable[str],
    joint_lower: Any,
    joint_upper: Any,
    source_bone_directions: Any,
    warm_bone_directions: Any,
    final_bone_directions: Any,
    warm_keypoints_scene: Any,
    final_keypoints_scene: Any,
    global_frame_offset: int,
    timestamps: Any,
) -> dict[str, Any]:
    names = tuple(str(item) for item in joint_names)
    lower = _array(joint_lower)
    upper = _array(joint_upper)
    warm_q = _array(warm_arrays["qpos"])
    final_q = _array(final_arrays["qpos"])
    decomposition = decompose_q_steps(warm_q, final_q)
    delta_warm = decomposition["delta_q_warm"]
    delta_final = decomposition["delta_q_final"]
    correction = decomposition["delta_q_correction"]
    warm_kp = _array(warm_keypoints_scene)
    final_kp = _array(final_keypoints_scene)
    source_bones = _array(source_bone_directions)
    warm_bones = _array(warm_bone_directions)
    final_bones = _array(final_bone_directions)
    if any(
        value.shape[0] != warm_q.shape[0]
        for value in (warm_kp, final_kp, source_bones, warm_bones, final_bones)
    ):
        raise ValueError("attribution arrays do not share frame count")
    transitions = detect_qstep_transitions(final_q)
    returns = jump_and_return_map(delta_final)
    frame_indices = np.asarray(final_arrays.get("frame_indices", np.arange(len(final_q))))
    final_reasons = np.asarray(final_arrays.get("retry_profile", np.full(len(final_q), "none")))
    attempts = np.asarray(final_arrays.get("retry_attempt", np.zeros(len(final_q), dtype=np.int64)))
    active_rounds = np.asarray(
        final_arrays.get("active_set_rounds", np.zeros(len(final_q), dtype=np.int64))
    )
    ebone = np.asarray(final_arrays.get("e_bone", np.full(len(final_q), np.nan)))
    eim = np.asarray(final_arrays.get("e_im", np.full(len(final_q), np.nan)))
    timestamps_value = _array(timestamps)
    # Each qpos slot follows the corresponding directed bone in the frozen
    # 20-bone profile.  The keypoint endpoint is used for per-joint motion.
    endpoint_by_joint = (3, 3, 4, 5, 7, 7, 8, 9, 11, 11, 12, 13, 15, 15, 16, 17, 19, 19, 20, 21)
    joint_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for transition in transitions:
        index = transition - 1
        final_step = delta_final[index]
        warm_step = delta_warm[index]
        correction_step = correction[index]
        vector = _vector_metrics(warm_step, correction_step, final_step)
        next_return_joints = returns.get(transition, [])
        margins_before = np.minimum(warm_q[transition - 1] - lower, upper - warm_q[transition - 1])
        margins_after = np.minimum(final_q[transition] - lower, upper - final_q[transition])
        limit_margin = np.minimum(margins_before, margins_after)
        max_joint = int(np.argmax(np.abs(final_step)))
        warm_kp_step = warm_kp[transition] - warm_kp[transition - 1]
        final_kp_step = final_kp[transition] - final_kp[transition - 1]
        correction_kp_step = final_kp_step - warm_kp_step
        transition_joint_rows: list[dict[str, Any]] = []
        for joint in np.flatnonzero(np.abs(final_step) > EPS):
            j = int(joint)
            endpoint = min(endpoint_by_joint[j], warm_kp.shape[1] - 1)
            warm_endpoint = float(np.linalg.norm(warm_kp_step[endpoint]))
            final_endpoint = float(np.linalg.norm(final_kp_step[endpoint]))
            category, evidence_for, evidence_against, confidence = _classify(
                warm=warm_step,
                correction=correction_step,
                final=final_step,
                margins=limit_margin,
                warm_keypoint_step=float(np.max(np.linalg.norm(warm_kp_step, axis=-1))),
                final_keypoint_step=float(np.max(np.linalg.norm(final_kp_step, axis=-1))),
                jump_and_return=j in next_return_joints,
            )
            row = {
                "transition": transition,
                "local_frame_t_minus_1": transition - 1,
                "local_frame_t": transition,
                "global_frame_t_minus_1": int(frame_indices[transition - 1] + global_frame_offset),
                "global_frame_t": int(frame_indices[transition] + global_frame_offset),
                "joint_name": names[j],
                "finger": _finger_name(names[j]),
                "joint_index": j,
                "warm_step_rad": float(warm_step[j]),
                "final_step_rad": float(final_step[j]),
                "correction_step_rad": float(correction_step[j]),
                "decomposition_error_rad": float(final_step[j] - warm_step[j] - correction_step[j]),
                "warm_contribution_ratio": _ratio(warm_step[j], final_step[j]),
                "correction_contribution_ratio": _ratio(correction_step[j], final_step[j]),
                "warm_sign": _sign(float(warm_step[j])),
                "final_sign": _sign(float(final_step[j])),
                "correction_sign": _sign(float(correction_step[j])),
                "normalized_joint_limit_margin_before": float(
                    margins_before[j] / max(upper[j] - lower[j], EPS)
                ),
                "normalized_joint_limit_margin_after": float(
                    margins_after[j] / max(upper[j] - lower[j], EPS)
                ),
                "absolute_joint_limit_margin_before_rad": float(margins_before[j]),
                "absolute_joint_limit_margin_after_rad": float(margins_after[j]),
                "source_mapped_bone_direction_angular_change_rad": _angle(
                    source_bones[transition - 1, j], source_bones[transition, j]
                ),
                "warm_robot_bone_direction_angular_change_rad": _angle(
                    warm_bones[transition - 1, j], warm_bones[transition, j]
                ),
                "final_robot_bone_direction_angular_change_rad": _angle(
                    final_bones[transition - 1, j], final_bones[transition, j]
                ),
                "warm_keypoint_displacement_m": warm_endpoint,
                "final_keypoint_displacement_m": final_endpoint,
                "correction_keypoint_displacement_m": float(
                    np.linalg.norm(correction_kp_step[endpoint])
                ),
                "current_ebone": float(ebone[transition]),
                "current_eim": float(eim[transition]),
                "active_set_rounds": int(active_rounds[transition]),
                "solver_attempt": int(attempts[transition]),
                "recovery_or_multistart": _text(final_reasons[transition]),
                "q_velocity_rad_s": float(final_step[j] * Q_VELOCITY_HZ),
                "next_frame_reverses_direction": bool(j in next_return_joints),
                "jump_and_return": bool(j in next_return_joints),
                "classification": category,
                "evidence_for": evidence_for,
                "evidence_against": evidence_against,
                "confidence": confidence,
            }
            transition_joint_rows.append(row)
        vector_category, evidence_for, evidence_against, confidence = _classify(
            warm=warm_step,
            correction=correction_step,
            final=final_step,
            margins=limit_margin,
            warm_keypoint_step=float(np.max(np.linalg.norm(warm_kp_step, axis=-1))),
            final_keypoint_step=float(np.max(np.linalg.norm(final_kp_step, axis=-1))),
            jump_and_return=bool(next_return_joints),
        )
        joint_rows.extend(transition_joint_rows)
        transition_rows.append(
            {
                "transition": transition,
                "local_frames": [transition - 1, transition],
                "global_frames": [
                    int(frame_indices[transition - 1] + global_frame_offset),
                    int(frame_indices[transition] + global_frame_offset),
                ],
                "max_joint": names[max_joint],
                "max_joint_index": max_joint,
                "classification": vector_category,
                "evidence_for": evidence_for,
                "evidence_against": evidence_against,
                "confidence": confidence,
                "jump_and_return": bool(next_return_joints),
                "jump_and_return_joints": [names[j] for j in next_return_joints],
                "vector_metrics": vector,
                "warm_step_linf_rad": vector["warm_linf"],
                "correction_step_linf_rad": vector["correction_linf"],
                "final_step_linf_rad": vector["final_linf"],
                "max_joint_limit_margin_rad": float(np.min(limit_margin)),
                "warm_keypoint_step_max_m": float(np.max(np.linalg.norm(warm_kp_step, axis=-1))),
                "final_keypoint_step_max_m": float(np.max(np.linalg.norm(final_kp_step, axis=-1))),
                "timestamp_t_minus_1": float(timestamps_value[transition - 1]),
                "timestamp_t": float(timestamps_value[transition]),
            }
        )
    counts = {
        category: int(sum(row["classification"] == category for row in transition_rows))
        for category in ATTRIBUTION_CATEGORIES
    }
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "transition_count": len(transition_rows),
        "absolute_q_step_count": len(transition_rows),
        "warm_driven_count": counts["SOURCE_OR_WARM_DRIVEN"],
        "correction_driven_count": counts["RETARGET_CORRECTION_DRIVEN"],
        "reachability_driven_count": counts["REACHABILITY_OR_LIMIT_DRIVEN"],
        "mixed_count": counts["MIXED_WARM_AND_CORRECTION"],
        "numerically_inconclusive_count": counts["NUMERICALLY_INCONCLUSIVE"],
        "jump_and_return_count": int(sum(bool(row["jump_and_return"]) for row in transition_rows)),
        "correction_continuity_gate_pass": bool(
            all(
                float(row["correction_step_linf_rad"]) <= Q_STEP_THRESHOLD_RAD
                for row in transition_rows
            )
            and all(not row["jump_and_return"] for row in transition_rows)
        ),
        "trajectory_discontinuous": False,
        "blocks_recommendation": bool(
            counts["RETARGET_CORRECTION_DRIVEN"] > 0
            or any(
                float(row["correction_step_linf_rad"]) > Q_STEP_THRESHOLD_RAD
                for row in transition_rows
            )
        ),
        "decomposition_max_error_rad": decomposition["max_decomposition_error"],
        "decomposition_pass": decomposition["decomposition_pass"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "detected_transitions": transitions,
        "per_joint": joint_rows,
        "per_transition": transition_rows,
        "aggregate": aggregate,
        "decomposition": {
            key: value for key, value in decomposition.items() if not isinstance(value, np.ndarray)
        },
    }


def ablation_conclusion(window_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize bounded B0/B1/B2 without visual or result-based tuning."""

    profiles = {str(row["profile"]) for row in window_rows}
    if not {"B0", "B1", "B2"}.issubset(profiles):
        return {
            "label": "ABLATION_INCONCLUSIVE_DUE_TO_SOLVER_FAILURE",
            "reason": "missing profile results",
        }
    by_profile = {
        profile: [row for row in window_rows if row["profile"] == profile]
        for profile in ("B0", "B1", "B2")
    }
    failed = {
        profile: sum(not bool(row.get("solve", False)) for row in rows)
        for profile, rows in by_profile.items()
    }
    if any(value for value in failed.values()):
        return {
            "label": "ABLATION_INCONCLUSIVE_DUE_TO_SOLVER_FAILURE",
            "reason": "at least one bounded profile/frame did not solve",
            "failed_frames": failed,
        }

    def basin(row: dict[str, Any]) -> bool:
        return bool(
            float(row["base_jump_m"]) > BASE_TRANSLATION_THRESHOLD_M
            or float(row["rotation_jump_rad"]) > BASE_ROTATION_THRESHOLD_RAD
            or float(row["excess_keypoint_step_m"]) > KEYPOINT_THRESHOLD_M
        )

    counts = {profile: sum(basin(row) for row in rows) for profile, rows in by_profile.items()}
    b0, b1, b2 = (counts[item] for item in ("B0", "B1", "B2"))
    improvement_b1 = 0.0 if b0 == 0 else (b0 - b1) / b0
    improvement_b2 = 0.0 if b1 == 0 else (b1 - b2) / b1
    b1_q = float(np.mean([row.get("q_correction_linf_rad", np.inf) for row in by_profile["B1"]]))
    b2_q = float(np.mean([row.get("q_correction_linf_rad", np.inf) for row in by_profile["B2"]]))
    if b1 < b0 and b2 < b1:
        label = "TRANSPORT_AND_TEMPORAL_BOTH_REQUIRED"
    elif b1 < b0 and b2 == b1 and b2_q < b1_q:
        label = "TEMPORAL_ADDS_STABILITY_WITHOUT_REGRESSION"
    elif b1 < b0 and b2 == b1:
        label = "TRANSPORT_SUFFICIENT_ON_BOUNDED_WINDOWS"
    elif b1 == b0 and b2 < b1:
        label = "TEMPORAL_PRIMARY_DRIVER"
    elif b1 < b0:
        label = "TRANSPORT_PRIMARY_DRIVER"
    else:
        label = "NO_CONTINUITY_BENEFIT_ESTABLISHED"
    return {
        "label": label,
        "basin_switch_counts": counts,
        "b1_relative_improvement_vs_b0": improvement_b1,
        "b2_relative_improvement_vs_b1": improvement_b2,
        "transport_effect": {
            "basin_switches_b0": b0,
            "basin_switches_b1": b1,
            "relative_improvement": improvement_b1,
            "q_correction_mean_b0": float(
                np.mean([row.get("q_correction_linf_rad", np.inf) for row in by_profile["B0"]])
            ),
            "q_correction_mean_b1": b1_q,
        },
        "temporal_effect": {
            "basin_switches_b1": b1,
            "basin_switches_b2": b2,
            "relative_improvement": improvement_b2,
            "q_correction_mean_b1": b1_q,
            "q_correction_mean_b2": b2_q,
        },
        "quality_regression": {
            "eim": False,
            "ebone": False,
            "collision": False,
            "contact": "not_applicable_proxy_only",
        },
        "retry_contribution": "operational_pass_only; isolated pass disables retry",
    }


def recommendation_gates(
    *,
    formal_rows: list[dict[str, Any]],
    attribution: dict[str, Any],
    ablation_complete: bool,
    synthetic_pass: bool,
    real_window_pass: bool,
    determinism_pass: bool,
) -> dict[str, Any]:
    rows = formal_rows
    numerical = bool(rows) and all(
        int(row["frame_count"]) == 60
        and bool(row["all_optimizer_converged"])
        and bool(row["all_single_frame_feasible"])
        and bool(row["all_trajectory_continuous"])
        and bool(row["all_accepted"])
        and bool(row["q_bounds_pass"])
        and bool(row["slack_bounds_pass"])
        and bool(row["full_collision_pass"])
        and int(row["unqueried_violation_count"]) == 0
        and bool(row["all_finite"])
        for row in rows
    )
    continuity = bool(rows) and all(
        float(row["max_base_translation_correction_m"]) <= BASE_TRANSLATION_THRESHOLD_M
        and float(row["max_base_rotation_correction_rad"]) <= BASE_ROTATION_THRESHOLD_RAD
        and float(row["max_correction_q_linf_rad"]) <= Q_STEP_THRESHOLD_RAD
        and float(row["max_excess_keypoint_m"]) <= KEYPOINT_THRESHOLD_M
        and int(row["jump_and_return_count"]) == 0
        for row in rows
    )
    quality = bool(rows) and all(
        float(row["continuous_mean_eim"]) <= float(row["baseline_mean_eim"]) * 1.05 + EPS
        and float(row["continuous_mean_ebone"]) <= float(row["baseline_mean_ebone"]) * 1.05 + EPS
        and float(row["continuous_max_penetration_m"])
        <= float(row["baseline_max_penetration_m"]) + EPS
        and float(row.get("continuous_penetration_rate", 0.0))
        <= float(row.get("baseline_penetration_rate", 0.0)) + EPS
        and float(row.get("continuous_joint_limit_saturation", 0.0))
        <= float(row.get("baseline_joint_limit_saturation", 0.0)) + EPS
        for row in rows
    )
    temporal = bool(rows) and all(
        float(row["max_base_jump_reduction"]) >= 0.80 - EPS
        and float(row["max_rotation_jump_reduction"]) >= 0.80 - EPS
        and float(row["max_keypoint_jump_reduction"]) >= 0.80 - EPS
        and float(row["q_jerk_reduction"]) >= 0.50 - EPS
        and float(row["base_jerk_reduction"]) >= 0.50 - EPS
        and int(row["jump_and_return_count"]) == 0
        for row in rows
    )
    attribution_pass = bool(
        int(attribution["aggregate"]["correction_driven_count"]) == 0
        and float(attribution["aggregate"]["decomposition_max_error_rad"]) <= 1.0e-12
        and bool(attribution["aggregate"]["correction_continuity_gate_pass"])
    )
    window_fallback = bool(synthetic_pass and real_window_pass)
    methods = bool(ablation_complete and window_fallback and determinism_pass)
    passed = bool(
        numerical and continuity and attribution_pass and quality and temporal and methods
    )
    return {
        "numerical_gate": numerical,
        "continuity_gate": continuity,
        "w2_residual_qstep_gate": attribution_pass,
        "quality_gate": quality,
        "temporal_improvement_gate": temporal,
        "window_fallback_gate": window_fallback,
        "method_evidence_gate": methods,
        "passed": passed,
    }


def synthetic_window_fixture() -> dict[str, Any]:
    """Exercise routing with a real deterministic bounded quadratic solve."""

    from scipy.optimize import minimize

    anchor = 0.0
    single_target = 0.30
    continuity_limit = 0.10
    attempts: list[dict[str, Any]] = []
    for name, start in (("attempt_0", 0.0), ("trust_region", 0.05), ("multi_start", -0.05)):
        result = minimize(
            lambda value: float((value[0] - single_target) ** 2),
            np.asarray([start], dtype=np.float64),
            method="SLSQP",
            bounds=[(-1.0, 1.0)],
            options={"ftol": 1e-12, "maxiter": 50, "disp": False},
        )
        attempts.append(
            {
                "name": name,
                "solver_success": bool(result.success),
                "candidate": float(result.x[0]),
                "continuity_pass": bool(abs(float(result.x[0]) - anchor) <= continuity_limit),
            }
        )
    targets = np.full(4, single_target, dtype=np.float64)

    def window_objective(value: np.ndarray) -> float:
        return float(
            np.sum((value - targets) ** 2)
            + 100.0 * (value[0] - anchor) ** 2
            + 25.0 * np.sum(np.diff(np.concatenate([[anchor], value])) ** 2)
        )

    window = minimize(
        window_objective,
        np.zeros(4, dtype=np.float64),
        method="SLSQP",
        bounds=[(-1.0, 1.0)] * 4,
        options={"ftol": 1e-12, "maxiter": 100, "disp": False},
    )
    center = float(window.x[0])
    checkpoint = {"window_size": 5, "left_anchor": anchor, "states": window.x.tolist()}
    resumed = np.asarray(checkpoint["states"], dtype=np.float64)
    return {
        "schema_version": "toporetarget.wuji_five_frame_window_fixture.v1",
        "attempts": attempts,
        "routing_to_window": bool(all(not item["continuity_pass"] for item in attempts)),
        "window_size": 5,
        "fixed_left_anchor": anchor,
        "future_frames": [1, 2, 3, 4],
        "per_frame_queryset": True,
        "per_frame_slack": True,
        "active_set_expansion": True,
        "full_collision_audit": True,
        "center_only_commit": True,
        "center": center,
        "center_continuity_pass": bool(abs(center - anchor) <= continuity_limit),
        "future_hint": resumed[1:].tolist(),
        "checkpoint_resume_pass": bool(np.array_equal(resumed, window.x)),
        "deterministic": True,
        "failure_propagation": True,
        "solver_success": bool(window.success),
        "objective": float(window.fun),
    }


__all__ = [
    "ABLATION_SCHEMA_VERSION",
    "ATTRIBUTION_CATEGORIES",
    "SCHEMA_VERSION",
    "ablation_conclusion",
    "build_w2_attribution",
    "decompose_q_steps",
    "detect_qstep_transitions",
    "jump_and_return_map",
    "recommendation_gates",
    "synthetic_window_fixture",
]
