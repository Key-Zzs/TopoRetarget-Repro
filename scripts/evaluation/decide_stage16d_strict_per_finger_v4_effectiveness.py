#!/usr/bin/env python3
"""Apply the frozen Strict Per-Finger V4 4M-to-16M continuation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"STRICT_V4_EFFECTIVENESS_JSON_OBJECT_REQUIRED:{path}")
    return value


def _rate(suite: dict[str, Any], name: str) -> float:
    value = suite.get("aggregate", {}).get(name, {}).get("rate")
    if not isinstance(value, (int, float)):
        raise ValueError(f"STRICT_V4_EFFECTIVENESS_SUITE_RATE_MISSING:{name}")
    return float(value)


def _number(mapping: dict[str, Any], *keys: str) -> float:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            raise ValueError(f"STRICT_V4_EFFECTIVENESS_METRIC_MISSING:{'.'.join(keys)}")
        value = value.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"STRICT_V4_EFFECTIVENESS_METRIC_MISSING:{'.'.join(keys)}")
    return float(value)


def _cross_finger(audit: dict[str, Any]) -> float:
    values = [
        row.get("cross_finger_group_compensation_fraction")
        for row in audit.get("per_finger", [])
        if isinstance(row, dict)
        and isinstance(row.get("cross_finger_group_compensation_fraction"), (int, float))
    ]
    if not values:
        raise ValueError("STRICT_V4_EFFECTIVENESS_CROSS_FINGER_METRICS_MISSING")
    return float(sum(float(value) for value in values) / len(values))


def _metrics(
    *, suite: dict[str, Any], audit: dict[str, Any], qualification: dict[str, Any]
) -> dict[str, float]:
    aggregate = audit.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("STRICT_V4_EFFECTIVENESS_AUDIT_AGGREGATE_MISSING")
    return {
        "persistent_source_tip_recall": _number(aggregate, "persistent_source_tip_recall"),
        "source_tip_recall": _number(aggregate, "source_tip_recall"),
        "cross_finger_compensation": _cross_finger(audit),
        "no_hand_object_contact_flight": _number(
            aggregate, "no_hand_object_contact_flight_fraction"
        ),
        "SRphysics": _rate(suite, "physics_success"),
        "SRqualified": _rate(suite, "qualified_success"),
        "SRkin": _rate(suite, "kinematic_success"),
        "terminal_delta_omega_radps": _number(
            qualification,
            "twist_residuals",
            "terminal_delta_omega_radps",
            "per_episode_median",
        ),
        "terminal_delta_v_mps": _number(
            qualification,
            "twist_residuals",
            "terminal_delta_v_mps",
            "per_episode_median",
        ),
    }


def _absolute_geometry_pass(geometry: dict[str, Any]) -> bool:
    gates = geometry.get("absolute_gates")
    if not isinstance(gates, dict) or not gates:
        raise ValueError("STRICT_V4_EFFECTIVENESS_ABSOLUTE_GEOMETRY_GATES_MISSING")
    return all(bool(value) for value in gates.values())


def _force_p95(audit: dict[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for row in audit.get("per_finger", []):
        if not isinstance(row, dict) or not isinstance(row.get("finger"), str):
            continue
        statistic = row.get("tip_pair_force_n_when_source_required")
        value = statistic.get("p95") if isinstance(statistic, dict) else None
        result[str(row["finger"])] = float(value) if isinstance(value, (int, float)) else None
    if len(result) != 5:
        raise ValueError("STRICT_V4_EFFECTIVENESS_PER_FINGER_FORCE_MISSING")
    return result


def _relative_reduction(v3: float, v4: float) -> float:
    return 0.0 if v3 <= 0.0 else (v3 - v4) / v3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--v1-suite", type=Path, required=True)
    parser.add_argument("--v1-audit", type=Path, required=True)
    parser.add_argument("--v1-geometry", type=Path, required=True)
    parser.add_argument("--v3-suite", type=Path, required=True)
    parser.add_argument("--v3-audit", type=Path, required=True)
    parser.add_argument("--v3-qualification", type=Path, required=True)
    parser.add_argument("--v3-geometry", type=Path, required=True)
    parser.add_argument("--v4-suite", type=Path, required=True)
    parser.add_argument("--v4-audit", type=Path, required=True)
    parser.add_argument("--v4-qualification", type=Path, required=True)
    parser.add_argument("--v4-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = _read(args.selection.resolve())
    v1_suite, v1_audit, v1_geometry = (
        _read(args.v1_suite.resolve()),
        _read(args.v1_audit.resolve()),
        _read(args.v1_geometry.resolve()),
    )
    v3_suite, v3_audit, v3_qualification, v3_geometry = (
        _read(args.v3_suite.resolve()),
        _read(args.v3_audit.resolve()),
        _read(args.v3_qualification.resolve()),
        _read(args.v3_geometry.resolve()),
    )
    v4_suite, v4_audit, v4_qualification, v4_geometry = (
        _read(args.v4_suite.resolve()),
        _read(args.v4_audit.resolve()),
        _read(args.v4_qualification.resolve()),
        _read(args.v4_geometry.resolve()),
    )
    selected = selection.get("selected")
    if (
        selection.get("status") != "STAGE16D_STRICT_V4_DEVELOPMENT_CHECKPOINT_SELECTED"
        or not isinstance(selected, dict)
        or not isinstance(selected.get("reward_v4_samples"), int)
        or int(selected["reward_v4_samples"]) > 4_194_304
        or v3_audit.get("status") != "STRICT_V4_SOURCE_CONTACT_AUDIT_COMPLETE"
        or v4_audit.get("status") != "STRICT_V4_SOURCE_CONTACT_AUDIT_COMPLETE"
        or v4_qualification.get("status") != "STAGE16D_STRICT_V4_FORMAL_COMPLETE"
    ):
        raise ValueError("STRICT_V4_EFFECTIVENESS_REQUIRES_FROZEN_4M_FORMAL_EVIDENCE")
    clip = str(v4_qualification.get("clip"))
    if any(
        str(value.get("clip")) != clip
        for value in (
            v1_suite,
            v1_audit,
            v1_geometry,
            v3_suite,
            v3_audit,
            v3_qualification,
            v3_geometry,
            v4_suite,
            v4_audit,
            v4_geometry,
        )
    ):
        raise ValueError("STRICT_V4_EFFECTIVENESS_CLIP_PROVENANCE_MISMATCH")

    v1_physics = _rate(v1_suite, "physics_success")
    v3 = _metrics(suite=v3_suite, audit=v3_audit, qualification=v3_qualification)
    v4 = _metrics(suite=v4_suite, audit=v4_audit, qualification=v4_qualification)
    improvements = {
        "persistent_source_tip_recall": {
            "threshold": 0.15,
            "delta": v4["persistent_source_tip_recall"] - v3["persistent_source_tip_recall"],
        },
        "source_tip_recall": {
            "threshold": 0.15,
            "delta": v4["source_tip_recall"] - v3["source_tip_recall"],
        },
        "cross_finger_compensation": {
            "threshold_relative_reduction": 0.30,
            "relative_reduction": _relative_reduction(
                v3["cross_finger_compensation"], v4["cross_finger_compensation"]
            ),
        },
        "no_hand_object_contact_flight": {
            "threshold_relative_reduction": 0.30,
            "relative_reduction": _relative_reduction(
                v3["no_hand_object_contact_flight"], v4["no_hand_object_contact_flight"]
            ),
        },
        "SRphysics": {"threshold": 0.10, "delta": v4["SRphysics"] - v3["SRphysics"]},
        "SRqualified": {"threshold": 0.10, "delta": v4["SRqualified"] - v3["SRqualified"]},
        "terminal_delta_omega_radps": {
            "threshold_relative_reduction": 0.20,
            "relative_reduction": _relative_reduction(
                v3["terminal_delta_omega_radps"], v4["terminal_delta_omega_radps"]
            ),
        },
        "terminal_delta_v_mps": {
            "threshold_relative_reduction": 0.20,
            "relative_reduction": _relative_reduction(
                v3["terminal_delta_v_mps"], v4["terminal_delta_v_mps"]
            ),
        },
    }
    passes = {
        name: bool(
            value["delta"] >= value["threshold"]
            if "delta" in value
            else value["relative_reduction"] >= value["threshold_relative_reduction"]
        )
        for name, value in improvements.items()
    }
    v1_force, v4_force = _force_p95(v1_audit), _force_p95(v4_audit)
    v4_geometry_pass = _absolute_geometry_pass(v4_geometry)
    force_farming = [
        finger
        for finger, baseline in v1_force.items()
        if baseline is not None
        and baseline > 0.0
        and v4_force[finger] is not None
        and float(v4_force[finger]) > 3.0 * baseline
        and (not v4_geometry_pass or v4["SRphysics"] < v1_physics)
    ]
    guards = {
        "SRkin_at_least_0_90": v4["SRkin"] >= 0.90,
        "absolute_geometry_safety_pass": v4_geometry_pass,
        "no_severe_force_farming": not force_farming,
    }
    effective = sum(passes.values()) >= 2 and all(guards.values())
    result = {
        "schema_version": "Stage16DStrictPerFingerV4EffectivenessGateV1",
        "clip": clip,
        "selection": selected,
        "comparison_split": "Formal20 frozen holdout; selection remained development-only",
        "v3_metrics": v3,
        "v4_metrics": v4,
        "primary_improvements": improvements,
        "primary_improvement_passes": passes,
        "primary_improvement_pass_count": sum(passes.values()),
        "guards": guards,
        "v1_force_p95_n": v1_force,
        "v1_SRphysics": v1_physics,
        "v4_force_p95_n": v4_force,
        "force_farming_fingers": force_farming,
        "status": ("STRICT_V4_EFFECTIVE_AT_4M" if effective else "STOP_AT_STRICT_V4_4M_BEST"),
        "continue_training_to_v4_samples": 16_777_216 if effective else None,
        "stop_reason": (
            "at least two primary improvements and every guard passed"
            if effective
            else "frozen 4M effectiveness gate did not pass; no 16M continuation authorized"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
