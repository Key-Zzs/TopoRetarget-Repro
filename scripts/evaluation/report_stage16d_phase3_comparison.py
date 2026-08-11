#!/usr/bin/env python3
"""Write the Phase 3 comparison CSV and Markdown from qualified evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"PHASE3_COMPARISON_EXPECTS_JSON_OBJECT:{path}")
    return value


def _rate(suite: dict[str, Any], field: str) -> float:
    value = suite.get("aggregate", {}).get(field, {}).get("rate")
    if not isinstance(value, (int, float)):
        raise ValueError(f"PHASE3_COMPARISON_RATE_MISSING:{field}")
    return float(value)


def _mean(suite: dict[str, Any], field: str) -> float:
    value = suite.get("aggregate", {}).get(field, {}).get("mean")
    if not isinstance(value, (int, float)):
        raise ValueError(f"PHASE3_COMPARISON_MEAN_MISSING:{field}")
    return float(value)


def _row(
    label: str, qualification_path: Path, suite_path: Path, evaluation_path: Path
) -> dict[str, Any]:
    qualification = _read(qualification_path.resolve())
    suite = _read(suite_path.resolve())
    evaluation = _read(evaluation_path.resolve())
    frame_zero = evaluation.get("frame_zero")
    if not isinstance(frame_zero, list) or len(frame_zero) != 20:
        raise ValueError("PHASE3_COMPARISON_REQUIRES_20_FRAME_ZERO_EPISODES")
    residuals = qualification.get("twist_residuals", {})
    rows = qualification.get("episodes")
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("PHASE3_COMPARISON_REQUIRES_20_QUALIFIED_EPISODES")

    def safety_rate(field: str) -> float:
        if any(field not in row for row in rows):
            raise ValueError(f"PHASE3_COMPARISON_SAFETY_FIELD_MISSING:{field}")
        return sum(bool(row[field]) for row in rows) / len(rows)

    return {
        "label": label,
        "checkpoint": evaluation["checkpoint"],
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "policy_reference_version": qualification["policy_reference_version"],
        "reward_v2_samples": evaluation.get("reward_v2_samples"),
        "qualification_status": qualification["status"],
        "SR_kinematic": _rate(suite, "kinematic_success"),
        "SR_physics": _rate(suite, "physics_success"),
        "SR_qualified": _rate(suite, "qualified_success"),
        "terminal_contact_rate": qualification["terminal_contact_rate"],
        "terminal_stability_rate": qualification["terminal_stability_rate"],
        "terminal_delta_v_median_mps": residuals["terminal_delta_v_mps"]["per_episode_median"],
        "terminal_delta_omega_median_radps": residuals["terminal_delta_omega_radps"][
            "per_episode_median"
        ],
        "E_t_mean_cm": _mean(suite, "E_t_mean_cm"),
        "E_r_mean_deg": _mean(suite, "E_r_mean_deg"),
        "E_ft_mean_cm": _mean(suite, "E_ft_mean_cm"),
        "E_j_mean_cm": _mean(suite, "E_j_mean_cm"),
        "absolute_geometry_pass": qualification["geometry_absolute_pass"],
        "action_bounds_pass_rate": safety_rate("action_bounds_pass"),
        "inter_finger_penetration_pass_rate": safety_rate("inter_finger_penetration_pass"),
        "contact_causality_pass_rate": safety_rate("contact_causality_pass"),
        "contact_topology_pass_rate": safety_rate("contact_topology_pass"),
        "qualification": str(qualification_path.resolve()),
        "evaluation_suite": str(suite_path.resolve()),
        "evaluation": str(evaluation_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row",
        type=Path,
        nargs=4,
        action="append",
        metavar=("LABEL", "QUALIFICATION", "SUITE", "EVALUATION"),
        required=True,
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    result = [
        _row(str(label), qualification, suite, evaluation)
        for label, qualification, suite, evaluation in args.row
    ]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    display = [
        "label",
        "policy_reference_version",
        "reward_v2_samples",
        "SR_kinematic",
        "SR_physics",
        "SR_qualified",
        "terminal_contact_rate",
        "terminal_stability_rate",
        "terminal_delta_v_median_mps",
        "terminal_delta_omega_median_radps",
        "E_t_mean_cm",
        "E_r_mean_deg",
        "E_ft_mean_cm",
        "E_j_mean_cm",
        "absolute_geometry_pass",
    ]
    markdown = [
        "# Stage 16-D Phase 3 comparison",
        "",
        (
            "All rows use the frozen 20-episode development set. "
            "Source-relative geometry remains a diagnostic."
        ),
        "",
        "| " + " | ".join(display) + " |",
        "| " + " | ".join("---" for _ in display) + " |",
    ]
    for row in result:
        markdown.append("| " + " | ".join(str(row[field]) for field in display) + " |")
    markdown.append("")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({"rows": len(result), "csv": str(args.csv.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
