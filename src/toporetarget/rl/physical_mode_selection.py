"""Fail-closed global C2 contact-mode selection for the P3 curriculum.

The two contact rewards are compared only after both clips finish the same
C2 physics condition.  This module intentionally has no Isaac dependency so
the selection evidence can be reviewed and tested before any C3 process is
allowed to start.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .physical_evaluation import CLIPS

CONTACT_MODES = ("aggregate_v3", "strict_per_finger_v4")
SELECTION_SCHEMA = "Stage16P3GlobalPhysicalContactModeSelectionV1"
_EXPECTED_PHYSICS = {"curriculum_stage": "C2", "gravity_scale": 0.5, "friction_scale": 1.5}
_TIE_TOLERANCE = 0.05


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}_MUST_BE_MAPPING")
    return value


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name}_MUST_BE_FINITE_NUMBER")
    return float(value)


def _rate(report: Mapping[str, Any], *, name: str) -> float:
    aggregate = _mapping(
        _mapping(report["evaluation_suite_v2"], name="SELECTION_SUITE")["aggregate"],
        name="SELECTION_SUITE_AGGREGATE",
    )
    metric = _mapping(aggregate[name], name=f"SELECTION_{name}")
    return _number(metric["rate"], name=f"SELECTION_{name}_RATE")


def _metric(report: Mapping[str, Any], *path: str) -> float:
    value: object = report
    for part in path:
        value = _mapping(value, name=f"SELECTION_{part}")[part]
    return _number(value, name="SELECTION_METRIC")


def _validate_report(report: Mapping[str, Any], *, mode: str, clip: str) -> None:
    if mode not in CONTACT_MODES or clip not in CLIPS:
        raise ValueError("SELECTION_MODE_OR_CLIP_INVALID")
    if report.get("status") != "P3_CONTACT_READY_DEVELOPMENT_EVALUATION_COMPLETE":
        raise ValueError("SELECTION_DEVELOPMENT_REPORT_INCOMPLETE")
    if report.get("kind") != "development" or report.get("curriculum_stage") != "C2":
        raise ValueError("SELECTION_REQUIRES_C2_DEVELOPMENT_ONLY")
    if report.get("clip") != clip or report.get("contact_mode") != mode:
        raise ValueError("SELECTION_REPORT_IDENTITY_MISMATCH")
    physics = _mapping(report.get("curriculum_physics"), name="SELECTION_PHYSICS")
    if {key: physics.get(key) for key in _EXPECTED_PHYSICS} != _EXPECTED_PHYSICS:
        raise ValueError("SELECTION_C2_PHYSICS_MISMATCH")
    if int(report.get("finite_episode_count", -1)) != 20:
        raise ValueError("SELECTION_REQUIRES_EXACTLY_20_FINITE_EPISODES")


def _safety(report: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    penetration = _mapping(report["penetration"], name="SELECTION_PENETRATION")
    failures = _mapping(report["physical_failure"], name="SELECTION_FAILURES")
    causal = _mapping(report["causal_contract"], name="SELECTION_CAUSAL")
    if not bool(penetration.get("absolute_geometry_pass")):
        reasons.append("ABSOLUTE_GEOMETRY_GATE_FAILED")
    if int(failures.get("catastrophic_contact", -1)) != 0:
        reasons.append("CATASTROPHIC_CONTACT_FAILURE")
    if int(failures.get("joint_limit", -1)) != 0:
        reasons.append("JOINT_LIMIT_FAILURE")
    causal_expected = {
        "external_guidance": False,
        "support": "none",
        "frame_zero_full_gravity": False,
        "rollout_object_state_writes": 0,
        "rollout_wrist_root_writes": 0,
    }
    if {key: causal.get(key) for key in causal_expected} != causal_expected:
        reasons.append("CAUSAL_CONTROL_CONTRACT_FAILED")
    return not reasons, reasons


def _per_clip_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    return {
        "SRqualified": _rate(report, name="qualified_success"),
        "SRphysics": _rate(report, name="physics_success"),
        "source_persistent_tip_recall": _metric(
            report, "interaction", "aggregate", "source_persistent_tip_recall"
        ),
        "cross_finger_compensation": _metric(
            report, "interaction", "aggregate", "cross_finger_compensation"
        ),
        "no_hand_object_contact_fraction": _metric(
            report, "flight", "no_hand_object_contact_fraction"
        ),
        "terminal_Delta_omega_radps": _metric(report, "twist", "Delta_omega_radps", "terminal"),
        "terminal_Delta_v_mps": _metric(report, "twist", "Delta_v_mps", "terminal"),
        "terminal_stability_rate": _metric(report, "twist", "terminal_stability_rate"),
        "hand_object_p95_penetration_m": _metric(
            report, "penetration", "hand_object_p95_penetration_m"
        ),
    }


def _macro(per_clip: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    values = list(per_clip.values())
    if len(values) != len(CLIPS):
        raise ValueError("SELECTION_CLIP_SET_INVALID")
    return {
        "minimum_per_clip_SRqualified": min(item["SRqualified"] for item in values),
        "macro_SRqualified": float(np.mean([item["SRqualified"] for item in values])),
        "minimum_per_clip_SRphysics": min(item["SRphysics"] for item in values),
        "macro_source_persistent_tip_recall": float(
            np.mean([item["source_persistent_tip_recall"] for item in values])
        ),
        "macro_cross_finger_compensation": float(
            np.mean([item["cross_finger_compensation"] for item in values])
        ),
        "macro_no_hand_object_contact_fraction": float(
            np.mean([item["no_hand_object_contact_fraction"] for item in values])
        ),
        "macro_terminal_Delta_omega_radps": float(
            np.mean([item["terminal_Delta_omega_radps"] for item in values])
        ),
        "macro_terminal_Delta_v_mps": float(
            np.mean([item["terminal_Delta_v_mps"] for item in values])
        ),
        "macro_terminal_stability_rate": float(
            np.mean([item["terminal_stability_rate"] for item in values])
        ),
        "macro_hand_object_p95_penetration_m": float(
            np.mean([item["hand_object_p95_penetration_m"] for item in values])
        ),
    }


def _selection_key(metrics: Mapping[str, float]) -> tuple[float, ...]:
    """Negate lower-is-better criteria for a descending lexicographic sort."""

    return (
        metrics["minimum_per_clip_SRqualified"],
        metrics["macro_SRqualified"],
        metrics["minimum_per_clip_SRphysics"],
        metrics["macro_source_persistent_tip_recall"],
        -metrics["macro_cross_finger_compensation"],
        -metrics["macro_no_hand_object_contact_fraction"],
        -metrics["macro_terminal_Delta_omega_radps"],
        -metrics["macro_terminal_Delta_v_mps"],
        metrics["macro_terminal_stability_rate"],
        -metrics["macro_hand_object_p95_penetration_m"],
    )


def _near_primary_tie(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    keys = (
        "minimum_per_clip_SRqualified",
        "macro_SRqualified",
        "minimum_per_clip_SRphysics",
        "macro_source_persistent_tip_recall",
        "macro_cross_finger_compensation",
        "macro_no_hand_object_contact_fraction",
    )
    return all(abs(left[key] - right[key]) < _TIE_TOLERANCE for key in keys)


def select_global_physical_contact_mode(
    reports: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, object]:
    """Select one global C2 mode or return a non-promotable safety block.

    ``reports`` must contain exactly ``mode -> clip -> C2 development report``.
    Formal artifacts are intentionally not accepted by this API.
    """

    if set(reports) != set(CONTACT_MODES):
        raise ValueError("SELECTION_MODE_SET_INVALID")
    candidates: dict[str, dict[str, object]] = {}
    for mode in CONTACT_MODES:
        by_clip = _mapping(reports[mode], name=f"SELECTION_{mode}")
        if set(by_clip) != set(CLIPS):
            raise ValueError("SELECTION_CLIP_SET_INVALID")
        per_clip: dict[str, dict[str, object]] = {}
        metrics_by_clip: dict[str, dict[str, float]] = {}
        for clip in CLIPS:
            report = _mapping(by_clip[clip], name="SELECTION_REPORT")
            _validate_report(report, mode=mode, clip=clip)
            safety_pass, safety_reasons = _safety(report)
            metrics = _per_clip_metrics(report)
            metrics_by_clip[clip] = metrics
            per_clip[clip] = {
                "safety_pass": safety_pass,
                "safety_reasons": safety_reasons,
                "metrics": metrics,
            }
        candidates[mode] = {
            "safety_pass": all(bool(per_clip[clip]["safety_pass"]) for clip in CLIPS),
            "per_clip_metrics": per_clip,
            "macro_metrics": _macro(metrics_by_clip),
        }

    eligible = [mode for mode in CONTACT_MODES if candidates[mode]["safety_pass"]]
    base: dict[str, object] = {
        "schema_version": SELECTION_SCHEMA,
        "selection_stage": "C2",
        "selection_kind": "development_only",
        "required_physics": dict(_EXPECTED_PHYSICS),
        "tie_tolerance": _TIE_TOLERANCE,
        "candidates": candidates,
        "clip_specific_selection_forbidden": True,
    }
    if not eligible:
        return {
            **base,
            "status": "GLOBAL_PHYSICAL_CONTACT_MODE_SELECTION_BLOCKED",
            "selected_mode": None,
            "rejected_mode": None,
            "rejected_modes": list(CONTACT_MODES),
            "selection_reason": (
                "Neither global mode passed C2 absolute-geometry and causal-controller safety "
                "requirements on both clips; C3/C4 promotion is forbidden."
            ),
        }
    if len(eligible) == 1:
        selected = eligible[0]
        reason = "Only one global mode passed the mandatory C2 safety requirements on both clips."
    else:
        v3 = _mapping(candidates["aggregate_v3"]["macro_metrics"], name="SELECTION_V3")
        v4 = _mapping(candidates["strict_per_finger_v4"]["macro_metrics"], name="SELECTION_V4")
        if _near_primary_tie(v3, v4):
            selected = "aggregate_v3"
            reason = (
                "The primary global metrics are within the pre-registered 0.05 tie tolerance; "
                "aggregate_v3 is the stable baseline preference."
            )
        else:
            selected = max(
                eligible, key=lambda mode: _selection_key(candidates[mode]["macro_metrics"])
            )
            reason = "Selected by the pre-registered global C2 lexicographic metric order."
    rejected = next(mode for mode in CONTACT_MODES if mode != selected)
    return {
        **base,
        "status": "GLOBAL_PHYSICAL_CONTACT_MODE_SELECTED",
        "selected_mode": selected,
        "rejected_mode": rejected,
        "rejected_modes": [rejected],
        "selection_reason": reason,
        "selection_key": list(_selection_key(candidates[selected]["macro_metrics"])),
    }


__all__ = ["CONTACT_MODES", "SELECTION_SCHEMA", "select_global_physical_contact_mode"]
