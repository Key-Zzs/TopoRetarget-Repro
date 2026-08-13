#!/usr/bin/env python3
"""Write the required Stage 16-D Reward V3 core and per-clip comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

NA = "N/A"


def _read(path_text: str) -> dict[str, Any] | None:
    if path_text == NA:
        return None
    path = Path(path_text).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V3_COMPARISON_JSON_OBJECT_REQUIRED:{path}")
    return value


def _value(value: Any) -> float | str:
    return float(value) if isinstance(value, (int, float)) else NA


def _rate(suite: dict[str, Any] | None, field: str) -> float | str:
    if suite is None:
        return NA
    return _value(suite.get("aggregate", {}).get(field, {}).get("rate"))


def _mean(suite: dict[str, Any] | None, field: str) -> float | str:
    if suite is None:
        return NA
    return _value(suite.get("aggregate", {}).get(field, {}).get("mean"))


def _nested(value: dict[str, Any] | None, *keys: str) -> float | str:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return NA
        current = current.get(key)
    return _value(current)


def _row(values: list[str]) -> dict[str, Any]:
    clip, contract, samples, qualification_arg, suite_arg, contact_arg, geometry_arg = values
    qualification = _read(qualification_arg)
    suite = _read(suite_arg)
    contact = _read(contact_arg)
    geometry = _read(geometry_arg)
    if qualification is None:
        raise ValueError("V3_COMPARISON_QUALIFICATION_REQUIRED")
    if qualification.get("clip") != clip:
        raise ValueError("V3_COMPARISON_CLIP_MISMATCH")
    max_penetration_mm = _nested(geometry, "corrected", "max_penetration_m")
    if isinstance(max_penetration_mm, float):
        max_penetration_mm *= 1000.0
    return {
        "Clip": clip.removeprefix("hocap_"),
        "Contract": contract,
        "Samples": samples,
        "Er_deg": _mean(suite, "E_r_mean_deg"),
        "Et_cm": _mean(suite, "E_t_mean_cm"),
        "Ej_cm": _mean(suite, "E_j_mean_cm"),
        "Eft_cm": _mean(suite, "E_ft_mean_cm"),
        "SRkin": _rate(suite, "kinematic_success"),
        "SRphysics": _rate(suite, "physics_success"),
        "SRqualified": _rate(suite, "qualified_success"),
        "Expected_contact_recall": _nested(contact, "aggregate", "expected_contact_recall"),
        "Persistent_contact_recall": _nested(contact, "aggregate", "persistent_contact_recall"),
        "Longest_loss_gap_steps": _nested(contact, "aggregate", "longest_contact_loss_gap"),
        "Terminal_contact": _value(qualification.get("terminal_contact_rate")),
        "Stability": _value(qualification.get("terminal_stability_rate")),
        "Delta_v_mps": _nested(
            qualification, "twist_residuals", "terminal_delta_v_mps", "per_episode_median"
        ),
        "Delta_omega_radps": _nested(
            qualification,
            "twist_residuals",
            "terminal_delta_omega_radps",
            "per_episode_median",
        ),
        "Contact_force_p95_N": _nested(contact, "aggregate", "contact_force_n", "p95"),
        "Max_penetration_mm": max_penetration_mm,
        "qualification": qualification_arg,
        "evaluation_suite": suite_arg,
        "contact_summary": contact_arg,
        "geometry": geometry_arg,
    }


def _markdown(title: str, rows: list[dict[str, Any]], fields: list[str]) -> str:
    result = [f"# {title}", "", "| " + " | ".join(fields) + " |"]
    result.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        result.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    result.append("")
    return "\n".join(result)


def _delta(v3: dict[str, Any], v1: dict[str, Any], field: str) -> float | str:
    left, right = v3[field], v1[field]
    if not isinstance(left, float) or not isinstance(right, float):
        return NA
    return left - right


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row",
        nargs=7,
        action="append",
        required=True,
        metavar=("CLIP", "CONTRACT", "SAMPLES", "QUALIFICATION", "SUITE", "CONTACT", "GEOMETRY"),
        help=f"Use literal {NA} for an unavailable historical Suite/Contact/Geometry artifact.",
    )
    parser.add_argument("--core-csv", type=Path, required=True)
    parser.add_argument("--core-markdown", type=Path, required=True)
    parser.add_argument("--ablation-170650-markdown", type=Path, required=True)
    parser.add_argument("--comparison-170105-markdown", type=Path, required=True)
    args = parser.parse_args()

    rows = [_row(value) for value in args.row]
    required = {
        ("170650", "V1 best baseline"),
        ("170650", "Reward V2 P1"),
        ("170650", "Reward V3 best"),
        ("170105", "V1 best baseline"),
        ("170105", "Reward V3 best"),
    }
    found = {(str(row["Clip"]), str(row["Contract"])) for row in rows}
    if len(rows) != 5 or found != required:
        raise ValueError(f"V3_COMPARISON_EXACT_FIVE_ROWS_REQUIRED:{sorted(found)}")
    fields = [
        "Clip",
        "Contract",
        "Samples",
        "Er_deg",
        "Et_cm",
        "Ej_cm",
        "Eft_cm",
        "SRkin",
        "SRphysics",
        "SRqualified",
        "Expected_contact_recall",
        "Persistent_contact_recall",
        "Longest_loss_gap_steps",
        "Terminal_contact",
        "Stability",
        "Delta_v_mps",
        "Delta_omega_radps",
        "Contact_force_p95_N",
        "Max_penetration_mm",
    ]
    for path in (
        args.core_csv,
        args.core_markdown,
        args.ablation_170650_markdown,
        args.comparison_170105_markdown,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.core_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in fields} for row in rows])
    args.core_markdown.write_text(
        _markdown("Stage 16-D Reward V3 core experiment table", rows, fields), encoding="utf-8"
    )
    ablation = [row for row in rows if row["Clip"] == "170650"]
    args.ablation_170650_markdown.write_text(
        _markdown("Stage 16-D hocap_170650 ablation", ablation, fields[1:]), encoding="utf-8"
    )
    comparison_rows = [row for row in rows if row["Clip"] == "170105"]
    v1 = next(row for row in comparison_rows if row["Contract"] == "V1 best baseline")
    v3 = next(row for row in comparison_rows if row["Contract"] == "Reward V3 best")
    delta_fields = [field for field in fields[3:] if field not in {"Max_penetration_mm"}]
    comparison = [
        {
            "Metric": field,
            "V1 baseline": v1[field],
            "V3 best": v3[field],
            "Delta": _delta(v3, v1, field),
        }
        for field in delta_fields
    ]
    args.comparison_170105_markdown.write_text(
        _markdown(
            "Stage 16-D hocap_170105 Reward V3 comparison",
            comparison,
            ["Metric", "V1 baseline", "V3 best", "Delta"],
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "STAGE16D_REWARD_V3_COMPARISON_WRITTEN", "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
