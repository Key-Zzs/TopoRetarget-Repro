#!/usr/bin/env python3
"""Write the six-row V1/V3/V4 Strict Per-Finger Formal20 comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"STRICT_V4_REPORT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _number(value: Any) -> float | str:
    return float(value) if isinstance(value, (int, float)) else "N/A"


def _nested(value: dict[str, Any], *keys: str) -> float | str:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return "N/A"
        current = current.get(key)
    return _number(current)


def _mean_per_finger(audit: dict[str, Any], key: str) -> float | str:
    values = [
        row.get(key)
        for row in audit.get("per_finger", [])
        if isinstance(row, dict) and isinstance(row.get(key), (int, float))
    ]
    return float(sum(float(value) for value in values) / len(values)) if values else "N/A"


def _samples(qualification: dict[str, Any]) -> int | str:
    for key in ("reward_v4_samples", "reward_v3_samples", "cumulative_training_samples"):
        if isinstance(qualification.get(key), int):
            return int(qualification[key])
    return "N/A"


def _geometry_penetration_mm(geometry: dict[str, Any]) -> float | str:
    for keys in (("corrected", "max_penetration_m"), ("max_penetration_m",)):
        value = _nested(geometry, *keys)
        if isinstance(value, float):
            return value * 1000.0
    return "N/A"


def _row(values: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    clip, contract, qualification_arg, suite_arg, audit_arg, geometry_arg = values
    qualification = _read(Path(qualification_arg).resolve())
    suite = _read(Path(suite_arg).resolve())
    audit = _read(Path(audit_arg).resolve())
    geometry = _read(Path(geometry_arg).resolve())
    if qualification.get("clip") != clip or audit.get("clip") != clip or suite.get("clip") != clip:
        raise ValueError("STRICT_V4_REPORT_CLIP_PROVENANCE_MISMATCH")
    aggregate = audit.get("aggregate", {})
    if not isinstance(aggregate, dict):
        raise ValueError("STRICT_V4_REPORT_AUDIT_AGGREGATE_MISSING")
    suite_aggregate = suite.get("aggregate", {})
    if not isinstance(suite_aggregate, dict):
        raise ValueError("STRICT_V4_REPORT_SUITE_AGGREGATE_MISSING")
    row = {
        "Clip": clip.removeprefix("hocap_"),
        "Contract": contract,
        "Samples": _samples(qualification),
        "Er_deg": _nested(suite_aggregate, "E_r_mean_deg", "mean"),
        "Et_cm": _nested(suite_aggregate, "E_t_mean_cm", "mean"),
        "Ej_cm": _nested(suite_aggregate, "E_j_mean_cm", "mean"),
        "Eft_cm": _nested(suite_aggregate, "E_ft_mean_cm", "mean"),
        "SRkin": _nested(suite_aggregate, "kinematic_success", "rate"),
        "SRphysics": _nested(suite_aggregate, "physics_success", "rate"),
        "SRqualified": _nested(suite_aggregate, "qualified_success", "rate"),
        "Source_tip_recall": _nested(aggregate, "source_tip_recall"),
        "Persistent_tip_recall": _nested(aggregate, "persistent_source_tip_recall"),
        "Cross_finger_compensation": _mean_per_finger(
            audit, "cross_finger_group_compensation_fraction"
        ),
        "Persistent_cross_finger_compensation": _mean_per_finger(
            audit, "persistent_cross_finger_group_compensation_fraction"
        ),
        "Same_finger_non_tip": _mean_per_finger(audit, "same_finger_non_tip_substitution_fraction"),
        "Fully_missing": _mean_per_finger(audit, "fully_missing_fraction"),
        "No_tip_flight": _nested(aggregate, "no_tip_contact_flight_fraction"),
        "No_hand_flight": _nested(aggregate, "no_hand_object_contact_flight_fraction"),
        "Longest_no_hand_flight_gap": _nested(aggregate, "longest_no_hand_flight_gap"),
        "Stability": _number(qualification.get("terminal_stability_rate")),
        "Delta_v_mps": _nested(
            qualification, "twist_residuals", "terminal_delta_v_mps", "per_episode_median"
        ),
        "Delta_omega_radps": _nested(
            qualification, "twist_residuals", "terminal_delta_omega_radps", "per_episode_median"
        ),
        "Max_penetration_mm": _geometry_penetration_mm(geometry),
        "Force_p95_N": _nested(aggregate, "tip_pair_force_n", "p95"),
        "Force_global_max_N": _nested(aggregate, "tip_pair_force_n", "max"),
        "Force_farming_flag": "PENDING_BASELINE_COMPARISON",
    }
    return row, audit


def _farming(rows: list[dict[str, Any]]) -> None:
    for clip in ("170105", "170650"):
        baseline = next(row for row in rows if row["Clip"] == clip and row["Contract"] == "V1")
        baseline_force = baseline["Force_p95_N"]
        baseline_penetration = baseline["Max_penetration_mm"]
        baseline_physics = baseline["SRphysics"]
        for row in (item for item in rows if item["Clip"] == clip):
            if row["Contract"] == "V1":
                row["Force_farming_flag"] = "BASELINE"
                continue
            force = row["Force_p95_N"]
            penetration = row["Max_penetration_mm"]
            physics = row["SRphysics"]
            suspect = (
                isinstance(force, float)
                and isinstance(baseline_force, float)
                and baseline_force > 0.0
                and force > 3.0 * baseline_force
                and (
                    (
                        isinstance(penetration, float)
                        and isinstance(baseline_penetration, float)
                        and penetration > baseline_penetration
                    )
                    or (
                        isinstance(physics, float)
                        and isinstance(baseline_physics, float)
                        and physics < baseline_physics
                    )
                )
            )
            row["Force_farming_flag"] = "SUSPECTED" if suspect else "NOT_SUSPECTED"


def _markdown(title: str, rows: list[dict[str, Any]], fields: list[str]) -> str:
    output = [f"# {title}", "", "| " + " | ".join(fields) + " |"]
    output.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        output.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    output.append("")
    return "\n".join(output)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in fields} for row in rows])


def _delta(v4: dict[str, Any], v3: dict[str, Any], name: str) -> float | str:
    left, right = v4[name], v3[name]
    return left - right if isinstance(left, float) and isinstance(right, float) else "N/A"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row",
        nargs=6,
        action="append",
        required=True,
        metavar=("CLIP", "CONTRACT", "QUALIFICATION", "SUITE", "AUDIT", "GEOMETRY"),
    )
    parser.add_argument("--core-csv", type=Path, required=True)
    parser.add_argument("--core-markdown", type=Path, required=True)
    parser.add_argument("--per-finger-csv", type=Path, required=True)
    parser.add_argument("--v3-v4-170105", type=Path, required=True)
    parser.add_argument("--v3-v4-170650", type=Path, required=True)
    parser.add_argument("--flight-csv", type=Path, required=True)
    args = parser.parse_args()

    paired = [_row(row) for row in args.row]
    rows = [item[0] for item in paired]
    expected = {
        (clip, contract) for clip in ("170105", "170650") for contract in ("V1", "V3", "V4")
    }
    if len(rows) != 6 or {(row["Clip"], row["Contract"]) for row in rows} != expected:
        raise ValueError("STRICT_V4_REPORT_EXACT_SIX_V1_V3_V4_ROWS_REQUIRED")
    _farming(rows)
    rows.sort(key=lambda row: (row["Clip"], ("V1", "V3", "V4").index(str(row["Contract"]))))
    fields = list(rows[0])
    _write_csv(args.core_csv.resolve(), rows, fields)
    args.core_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.core_markdown.write_text(
        _markdown("Stage 16-D Strict Per-Finger V4 core experiment table", rows, fields),
        encoding="utf-8",
    )
    per_finger: list[dict[str, Any]] = []
    for row, audit in paired:
        for finger in audit.get("per_finger", []):
            if not isinstance(finger, dict):
                continue
            force = finger.get("tip_pair_force_n_when_source_required")
            per_finger.append(
                {
                    "Clip": row["Clip"],
                    "Contract": row["Contract"],
                    "Finger": finger.get("finger"),
                    "Source_expected_percent": float(finger["source_required_runtime_frames"])
                    / 321.0,
                    "Tip_recall": finger.get("source_tip_recall"),
                    "Persistent_recall": finger.get("persistent_source_tip_recall"),
                    "Cross_finger_compensation": finger.get(
                        "cross_finger_group_compensation_fraction"
                    ),
                    "Persistent_cross_finger_compensation": finger.get(
                        "persistent_cross_finger_group_compensation_fraction"
                    ),
                    "Same_finger_substitute": finger.get(
                        "same_finger_non_tip_substitution_fraction"
                    ),
                    "Fully_missing_percent": finger.get("fully_missing_fraction"),
                    "Force_p95_N": force.get("p95") if isinstance(force, dict) else "N/A",
                }
            )
    _write_csv(args.per_finger_csv.resolve(), per_finger, list(per_finger[0]))
    flight_rows = [
        {
            "Clip": row["Clip"],
            "Contract": row["Contract"],
            "No_tip_flight_fraction": row["No_tip_flight"],
            "No_hand_flight_fraction": row["No_hand_flight"],
            "Longest_no_hand_flight_gap": row["Longest_no_hand_flight_gap"],
            "Recontact_events": _nested(audit, "aggregate", "recontact_event_count"),
        }
        for row, audit in paired
    ]
    _write_csv(args.flight_csv.resolve(), flight_rows, list(flight_rows[0]))
    for clip, output in (("170105", args.v3_v4_170105), ("170650", args.v3_v4_170650)):
        v3 = next(row for row in rows if row["Clip"] == clip and row["Contract"] == "V3")
        v4 = next(row for row in rows if row["Clip"] == clip and row["Contract"] == "V4")
        metrics = [
            "Source_tip_recall",
            "Persistent_tip_recall",
            "Cross_finger_compensation",
            "Fully_missing",
            "No_hand_flight",
            "Longest_no_hand_flight_gap",
            "SRphysics",
            "SRqualified",
            "Et_cm",
            "Er_deg",
            "Delta_v_mps",
            "Delta_omega_radps",
            "Max_penetration_mm",
            "Force_p95_N",
        ]
        comparison = [
            {
                "Metric": metric,
                "V3 Aggregate": v3[metric],
                "V4 Strict": v4[metric],
                "Delta": _delta(v4, v3, metric),
            }
            for metric in metrics
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            _markdown(
                f"Stage 16-D hocap_{clip} V3 to V4 interaction fidelity",
                comparison,
                list(comparison[0]),
            ),
            encoding="utf-8",
        )
    print(json.dumps({"status": "STAGE16D_STRICT_V4_COMPARISON_WRITTEN", "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
