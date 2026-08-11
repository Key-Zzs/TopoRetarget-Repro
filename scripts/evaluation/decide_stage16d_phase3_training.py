#!/usr/bin/env python3
"""Make the frozen P1/4M Phase 3 continuation decision from dev evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rate(suite: dict[str, Any], field: str) -> float:
    value = suite.get("aggregate", {}).get(field, {}).get("rate")
    if not isinstance(value, (int, float)):
        raise ValueError(f"PHASE3_SUITE_METRIC_MISSING:{field}")
    return float(value)


def _metrics(
    qualification: dict[str, Any], suite: dict[str, Any], evaluation: dict[str, Any]
) -> dict[str, float | bool]:
    frame_zero = evaluation.get("frame_zero")
    if not isinstance(frame_zero, list) or len(frame_zero) != 20:
        raise ValueError("PHASE3_DECISION_REQUIRES_20_DEV_EPISODES")
    final_error: list[float] = []
    for row in frame_zero:
        if isinstance(row.get("final_object_position_error_m"), (int, float)):
            final_error.append(float(row["final_object_position_error_m"]))
            continue
        tracking = row.get("object_tracking_error_m")
        if not isinstance(tracking, dict) or not isinstance(tracking.get("final"), (int, float)):
            raise ValueError("PHASE3_FINAL_OBJECT_POSITION_ERROR_MISSING")
        final_error.append(float(tracking["final"]))
    residuals = qualification.get("twist_residuals", {})
    episodes = qualification.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 20:
        raise ValueError("PHASE3_DECISION_REQUIRES_20_QUALIFIED_EPISODES")

    def pass_rate(field: str) -> float:
        if any(field not in row for row in episodes):
            raise ValueError(f"PHASE3_SAFETY_METRIC_MISSING:{field}")
        return float(np.mean([bool(row[field]) for row in episodes]))

    return {
        "terminal_stability_rate": float(qualification["terminal_stability_rate"]),
        "terminal_contact_rate": float(qualification["terminal_contact_rate"]),
        "terminal_delta_v_median_mps": float(
            residuals["terminal_delta_v_mps"]["per_episode_median"]
        ),
        "terminal_delta_omega_median_radps": float(
            residuals["terminal_delta_omega_radps"]["per_episode_median"]
        ),
        "final_object_position_error_median_m": float(np.median(final_error)),
        "SR_kinematic": _rate(suite, "kinematic_success"),
        "SR_physics": _rate(suite, "physics_success"),
        "SR_qualified": _rate(suite, "qualified_success"),
        "absolute_geometry_pass": bool(qualification["geometry_absolute_pass"]),
        "action_bounds_pass_rate": pass_rate("action_bounds_pass"),
        "inter_finger_penetration_pass_rate": pass_rate("inter_finger_penetration_pass"),
        "contact_causality_pass_rate": pass_rate("contact_causality_pass"),
        "contact_topology_pass_rate": pass_rate("contact_topology_pass"),
    }


def _comparison(
    candidate: dict[str, float | bool], baseline: dict[str, float | bool]
) -> dict[str, Any]:
    return {
        "terminal_stability_gain": float(
            candidate["terminal_stability_rate"] - baseline["terminal_stability_rate"]
        ),
        "terminal_delta_v_reduction": (
            None
            if baseline["terminal_delta_v_median_mps"] == 0.0
            else 1.0
            - float(candidate["terminal_delta_v_median_mps"])
            / float(baseline["terminal_delta_v_median_mps"])
        ),
        "terminal_delta_omega_reduction": (
            None
            if baseline["terminal_delta_omega_median_radps"] == 0.0
            else 1.0
            - float(candidate["terminal_delta_omega_median_radps"])
            / float(baseline["terminal_delta_omega_median_radps"])
        ),
        "final_object_position_error_reduction": (
            None
            if baseline["final_object_position_error_median_m"] == 0.0
            else 1.0
            - float(candidate["final_object_position_error_median_m"])
            / float(baseline["final_object_position_error_median_m"])
        ),
        "SR_physics_gain": float(candidate["SR_physics"] - baseline["SR_physics"]),
        "SR_qualified_gain": float(candidate["SR_qualified"] - baseline["SR_qualified"]),
        "SR_kinematic_change": float(candidate["SR_kinematic"] - baseline["SR_kinematic"]),
        "terminal_contact_change": float(
            candidate["terminal_contact_rate"] - baseline["terminal_contact_rate"]
        ),
        "action_bounds_pass_rate_change": float(
            candidate["action_bounds_pass_rate"] - baseline["action_bounds_pass_rate"]
        ),
        "inter_finger_penetration_pass_rate_change": float(
            candidate["inter_finger_penetration_pass_rate"]
            - baseline["inter_finger_penetration_pass_rate"]
        ),
        "contact_causality_pass_rate_change": float(
            candidate["contact_causality_pass_rate"] - baseline["contact_causality_pass_rate"]
        ),
        "contact_topology_pass_rate_change": float(
            candidate["contact_topology_pass_rate"] - baseline["contact_topology_pass_rate"]
        ),
    }


def _load_triplet(
    qualification: Path, suite: Path, evaluation: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return _read(qualification.resolve()), _read(suite.resolve()), _read(evaluation.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("p1", "4m"), required=True)
    parser.add_argument("--candidate-qualification", type=Path, required=True)
    parser.add_argument("--candidate-suite", type=Path, required=True)
    parser.add_argument("--candidate-evaluation", type=Path, required=True)
    parser.add_argument("--v1-l0-qualification", type=Path, required=True)
    parser.add_argument("--v1-l0-suite", type=Path, required=True)
    parser.add_argument("--v1-l0-evaluation", type=Path, required=True)
    parser.add_argument("--v1-4m-qualification", type=Path, required=True)
    parser.add_argument("--v1-4m-suite", type=Path, required=True)
    parser.add_argument("--v1-4m-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_q, candidate_suite, candidate_eval = _load_triplet(
        args.candidate_qualification, args.candidate_suite, args.candidate_evaluation
    )
    l0_q, l0_suite, l0_eval = _load_triplet(
        args.v1_l0_qualification, args.v1_l0_suite, args.v1_l0_evaluation
    )
    v1_4m_q, v1_4m_suite, v1_4m_eval = _load_triplet(
        args.v1_4m_qualification, args.v1_4m_suite, args.v1_4m_evaluation
    )
    candidate = _metrics(candidate_q, candidate_suite, candidate_eval)
    l0 = _metrics(l0_q, l0_suite, l0_eval)
    baseline = _metrics(v1_4m_q, v1_4m_suite, v1_4m_eval)
    comparison = _comparison(candidate, baseline)
    l0_comparison = _comparison(candidate, l0)
    guards = {
        "SR_kinematic_not_down_more_than_0_05": comparison["SR_kinematic_change"] >= -0.05,
        "terminal_contact_not_down_more_than_0_10": comparison["terminal_contact_change"] >= -0.10,
        "absolute_geometry_pass": bool(candidate["absolute_geometry_pass"]),
        "inter_finger_penetration_not_regressed": (
            comparison["inter_finger_penetration_pass_rate_change"] >= 0.0
        ),
    }
    if args.stage == "p1":
        improvements = {
            "terminal_stability_gain_at_least_0_10": comparison["terminal_stability_gain"] >= 0.10,
            "terminal_delta_v_reduction_at_least_15_percent": (
                comparison["terminal_delta_v_reduction"] is not None
                and comparison["terminal_delta_v_reduction"] >= 0.15
            ),
            "terminal_delta_omega_reduction_at_least_15_percent": (
                comparison["terminal_delta_omega_reduction"] is not None
                and comparison["terminal_delta_omega_reduction"] >= 0.15
            ),
            "SR_physics_gain_at_least_0_10": comparison["SR_physics_gain"] >= 0.10,
        }
        if all(guards.values()) and any(improvements.values()):
            status = "PHASE3_P1_IMPROVING"
            authorized_target = 4_194_304
        elif all(guards.values()):
            status = "PHASE3_P1_NEUTRAL_CONTINUE_TO_4M"
            authorized_target = 4_194_304
        else:
            status = "PHASE3_P1_COLLAPSE_STOP"
            authorized_target = None
    else:
        improvements = {
            "terminal_stability_gain_at_least_0_15": comparison["terminal_stability_gain"] >= 0.15,
            "terminal_delta_v_reduction_at_least_20_percent": (
                comparison["terminal_delta_v_reduction"] is not None
                and comparison["terminal_delta_v_reduction"] >= 0.20
            ),
            "terminal_delta_omega_reduction_at_least_20_percent": (
                comparison["terminal_delta_omega_reduction"] is not None
                and comparison["terminal_delta_omega_reduction"] >= 0.20
            ),
            "SR_physics_gain_at_least_0_15": comparison["SR_physics_gain"] >= 0.15,
            "SR_qualified_gain_at_least_0_15": comparison["SR_qualified_gain"] >= 0.15,
            "final_object_error_reduction_at_least_15_percent": (
                comparison["final_object_position_error_reduction"] is not None
                and comparison["final_object_position_error_reduction"] >= 0.15
            ),
        }
        strict_guards = {
            **guards,
            "SR_kinematic_at_least_0_90": candidate["SR_kinematic"] >= 0.90,
            "terminal_contact_at_least_0_80": candidate["terminal_contact_rate"] >= 0.80,
            "action_bounds_not_regressed": comparison["action_bounds_pass_rate_change"] >= 0.0,
            "inter_finger_penetration_not_regressed": (
                comparison["inter_finger_penetration_pass_rate_change"] >= 0.0
            ),
            "contact_causality_not_regressed": (
                comparison["contact_causality_pass_rate_change"] >= 0.0
            ),
            "contact_topology_not_regressed": (
                comparison["contact_topology_pass_rate_change"] >= 0.0
            ),
        }
        if all(strict_guards.values()) and sum(improvements.values()) >= 2:
            status = "PHASE3_REWARD_V2_EFFECTIVE"
            authorized_target = 16_777_216
        else:
            status = "PHASE3_REWARD_V2_INSUFFICIENT_AT_4M"
            authorized_target = None
        guards = strict_guards
    result = {
        "schema_version": "Stage16DPhase3TrainingDecisionV1",
        "stage": args.stage,
        "status": status,
        "authorized_next_reward_v2_samples": authorized_target,
        "candidate": candidate,
        "v1_l0_baseline": l0,
        "v1_4m_baseline": baseline,
        "comparison_to_v1_4m": comparison,
        "comparison_to_v1_l0": l0_comparison,
        "improvements": improvements,
        "guards": guards,
        "selection_note": (
            "V1 4M is the primary comparator; V1 L0 remains reported as the common "
            "actor-initialization baseline."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
