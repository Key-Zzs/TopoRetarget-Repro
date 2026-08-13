#!/usr/bin/env python3
"""Apply the frozen Reward V3 4M-to-16M continuation decision exactly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V3_16M_DECISION_JSON_OBJECT_REQUIRED:{path}")
    return value


def _rate(suite: dict[str, Any], field: str) -> float:
    value = suite.get("aggregate", {}).get(field, {}).get("rate")
    if not isinstance(value, (int, float)):
        raise ValueError(f"V3_16M_DECISION_SUITE_RATE_MISSING:{field}")
    return float(value)


def _metric(contact: dict[str, Any], field: str) -> float:
    value = contact.get("aggregate", {}).get(field)
    if not isinstance(value, (int, float)):
        raise ValueError(f"V3_16M_DECISION_CONTACT_METRIC_MISSING:{field}")
    return float(value)


def _residual(qualification: dict[str, Any], field: str) -> float:
    value = qualification.get("twist_residuals", {}).get(field, {}).get("per_episode_median")
    if not isinstance(value, (int, float)):
        raise ValueError(f"V3_16M_DECISION_TWIST_METRIC_MISSING:{field}")
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-selection", type=Path, required=True)
    parser.add_argument("--v3-qualification", type=Path, required=True)
    parser.add_argument("--v3-suite", type=Path, required=True)
    parser.add_argument("--v3-contact", type=Path, required=True)
    parser.add_argument("--v3-geometry", type=Path, required=True)
    parser.add_argument("--v1-qualification", type=Path, required=True)
    parser.add_argument("--v1-suite", type=Path, required=True)
    parser.add_argument("--v1-contact", type=Path, required=True)
    parser.add_argument("--v1-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = _read(args.v3_selection.resolve())
    v3_qualification = _read(args.v3_qualification.resolve())
    v3_suite = _read(args.v3_suite.resolve())
    v3_contact = _read(args.v3_contact.resolve())
    v3_geometry = _read(args.v3_geometry.resolve())
    v1_qualification = _read(args.v1_qualification.resolve())
    v1_suite = _read(args.v1_suite.resolve())
    v1_contact = _read(args.v1_contact.resolve())
    v1_geometry = _read(args.v1_geometry.resolve())
    selected = selection.get("selected", {})
    if (
        selection.get("status") != "STAGE16D_REWARD_V3_DEVELOPMENT_CHECKPOINT_SELECTED"
        or v3_qualification.get("kind") != "development"
        or v3_qualification.get("checkpoint_sha256") != selected.get("checkpoint_sha256")
        or v1_qualification.get("kind") != "development"
        or v3_qualification.get("clip") != v1_qualification.get("clip")
    ):
        raise ValueError("V3_16M_DECISION_PROVENANCE_INVALID")
    v3 = {
        "expected_contact_recall": _metric(v3_contact, "expected_contact_recall"),
        "persistent_contact_recall": _metric(v3_contact, "persistent_contact_recall"),
        "terminal_stability": float(v3_qualification["terminal_stability_rate"]),
        "SRphysics": _rate(v3_suite, "physics_success"),
        "SRqualified": _rate(v3_suite, "qualified_success"),
        "longest_contact_loss_gap": _metric(v3_contact, "longest_contact_loss_gap"),
        "terminal_delta_omega": _residual(v3_qualification, "terminal_delta_omega_radps"),
        "terminal_delta_v": _residual(v3_qualification, "terminal_delta_v_mps"),
        "SRkin": _rate(v3_suite, "kinematic_success"),
        "force_p95_over_v1": _metric(v3_contact, "force_p95_over_v1"),
    }
    v1 = {
        "expected_contact_recall": _metric(v1_contact, "expected_contact_recall"),
        "persistent_contact_recall": _metric(v1_contact, "persistent_contact_recall"),
        "terminal_stability": float(v1_qualification["terminal_stability_rate"]),
        "SRphysics": _rate(v1_suite, "physics_success"),
        "SRqualified": _rate(v1_suite, "qualified_success"),
        "longest_contact_loss_gap": _metric(v1_contact, "longest_contact_loss_gap"),
        "terminal_delta_omega": _residual(v1_qualification, "terminal_delta_omega_radps"),
        "terminal_delta_v": _residual(v1_qualification, "terminal_delta_v_mps"),
    }
    criteria = {
        "expected_contact_recall_plus_015": v3["expected_contact_recall"]
        >= v1["expected_contact_recall"] + 0.15,
        "persistent_contact_recall_plus_015": v3["persistent_contact_recall"]
        >= v1["persistent_contact_recall"] + 0.15,
        "terminal_stability_plus_015": v3["terminal_stability"] >= v1["terminal_stability"] + 0.15,
        "SRphysics_plus_015": v3["SRphysics"] >= v1["SRphysics"] + 0.15,
        "SRqualified_plus_015": v3["SRqualified"] >= v1["SRqualified"] + 0.15,
        "longest_loss_gap_reduced_30pct": v3["longest_contact_loss_gap"]
        <= v1["longest_contact_loss_gap"] * 0.70,
        "terminal_delta_omega_reduced_20pct": v3["terminal_delta_omega"]
        <= v1["terminal_delta_omega"] * 0.80,
        "terminal_delta_v_reduced_20pct": v3["terminal_delta_v"] <= v1["terminal_delta_v"] * 0.80,
    }
    improved_count = sum(criteria.values())
    geometry_pass = bool(v3_qualification.get("geometry_absolute_pass"))
    v3_max_penetration_m = v3_geometry.get("corrected", {}).get("max_penetration_m")
    v1_max_penetration_m = v1_geometry.get("corrected", {}).get("max_penetration_m")
    if not isinstance(v3_max_penetration_m, (int, float)) or not isinstance(
        v1_max_penetration_m, (int, float)
    ):
        raise ValueError("V3_16M_DECISION_GEOMETRY_METRIC_MISSING")
    penetration_worsened = float(v3_max_penetration_m) > float(v1_max_penetration_m)
    severe_force_farming = bool(
        v3["force_p95_over_v1"] > 3.0
        and (penetration_worsened or v3["SRphysics"] < v1["SRphysics"])
    )
    effective = (
        improved_count >= 2 and v3["SRkin"] >= 0.90 and geometry_pass and not severe_force_farming
    )
    result = {
        "schema_version": "Stage16DRewardV3FourMillionDecisionV1",
        "status": "V3_EFFECTIVE_AT_4M" if effective else "STOP_AT_V3_4M_BEST",
        "clip": v3_qualification["clip"],
        "selected_checkpoint": selected,
        "v3_metrics": v3,
        "v1_baseline_metrics": v1,
        "improvement_criteria": criteria,
        "improvement_count": improved_count,
        "absolute_geometry_pass": geometry_pass,
        "v3_max_penetration_m": float(v3_max_penetration_m),
        "v1_max_penetration_m": float(v1_max_penetration_m),
        "penetration_worsened": penetration_worsened,
        "severe_force_farming": severe_force_farming,
        "continue_target_reward_v3_samples": 16_777_216 if effective else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "improvement_count": improved_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
