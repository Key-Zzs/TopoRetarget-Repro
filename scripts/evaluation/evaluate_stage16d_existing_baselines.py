#!/usr/bin/env python3
"""Evaluate both frozen Stage 16-D R7 baselines with Evaluation Suite V2."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _comparison_row(clip: str, summary: dict[str, object]) -> dict[str, object]:
    aggregate = summary["aggregate"]
    if not isinstance(aggregate, dict):
        raise ValueError("invalid Evaluation Suite V2 aggregate")
    row: dict[str, object] = {"clip": clip}
    for metric in ("E_r_mean_deg", "E_t_mean_cm", "E_j_mean_cm", "E_ft_mean_cm"):
        values = aggregate[metric]
        if not isinstance(values, dict):
            raise ValueError("invalid metric aggregate")
        row[f"{metric}_mean"] = values["mean"]
        row[f"{metric}_std"] = values["std"]
        row[f"{metric}_median"] = values["median"]
        row[f"{metric}_p95_over_rollouts"] = values["p95_over_rollouts"]
    for metric in ("kinematic_success", "physics_success", "qualified_success"):
        values = aggregate[metric]
        if not isinstance(values, dict):
            raise ValueError("invalid success aggregate")
        row[metric] = values["rate"]
        row[f"{metric}_pass_count"] = values["pass_count"]
        row[f"{metric}_total"] = values["total"]
    legacy = summary["legacy"]
    if not isinstance(legacy, dict):
        raise ValueError("invalid legacy summary")
    old = legacy["old_ppo_task_success"]
    if not isinstance(old, dict):
        raise ValueError("invalid legacy task success")
    row["old_ppo_task_success_pass_count"] = old["pass_count"]
    row["old_ppo_task_success_total"] = old["total"]
    row["source_relative_geometry_fidelity"] = legacy["source_relative_geometry_fidelity"]
    row["absolute_hand_object_penetration"] = legacy["absolute_hand_object_penetration"]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    evaluator = REPO_ROOT / "scripts/evaluation/evaluate_stage16d_trajectory.py"
    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    for clip in ("hocap_170105", "hocap_170650"):
        source = input_root / clip
        per_episode = output_root / f"{clip}_per_episode.csv"
        summary = output_root / f"{clip}_summary.json"
        timeline = output_root / f"{clip}_metric_timeline.jsonl"
        subprocess.run(
            [
                sys.executable,
                str(evaluator),
                "--trace",
                str(source / "ppo_r7_formal_trace_replica0.npz"),
                "--reference",
                str(
                    REPO_ROOT
                    / ".local/reports/stage16d_ppo26d_clip_repair/reference"
                    / f"{clip}.reference.npz"
                ),
                "--qualification",
                str(source / "r7_formal_qualification.json"),
                "--per-episode",
                str(per_episode),
                "--summary",
                str(summary),
                "--timeline",
                str(timeline),
            ],
            check=True,
        )
        summaries[clip] = _read_json(summary)
        rows.append(_comparison_row(clip, summaries[clip]))
    comparison_csv = output_root / "baseline_comparison.csv"
    with comparison_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = [
        "| Metric | hocap_170105 | hocap_170650 |",
        "| --- | ---: | ---: |",
    ]
    labels = (
        ("E_r mean±std (deg)", "E_r_mean_deg_mean", "E_r_mean_deg_std"),
        ("E_t mean±std (cm)", "E_t_mean_cm_mean", "E_t_mean_cm_std"),
        ("E_j mean±std (cm)", "E_j_mean_cm_mean", "E_j_mean_cm_std"),
        ("E_ft mean±std (cm)", "E_ft_mean_cm_mean", "E_ft_mean_cm_std"),
    )
    for label, mean, std in labels:
        table.append(
            f"| {label} | {rows[0][mean]:.4f} ± {rows[0][std]:.4f} | "
            f"{rows[1][mean]:.4f} ± {rows[1][std]:.4f} |"
        )
    for label, name in (
        ("SR_kinematic", "kinematic_success"),
        ("SR_physics", "physics_success"),
        ("SR_qualified", "qualified_success"),
        ("Legacy PPO task success", "old_ppo_task_success"),
    ):
        count = f"{name}_pass_count"
        total = f"{name}_total"
        table.append(
            f"| {label} | {rows[0][count]}/{rows[0][total]} | {rows[1][count]}/{rows[1][total]} |"
        )
    (output_root / "baseline_comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    (output_root / "evaluation_suite_v2_summary.md").write_text(
        "# Evaluation Suite V2 baseline re-evaluation\n\n"
        "The metrics use the common world/env frame after environment-origin removal. "
        "`SR_physics` uses absolute safety gates; legacy source-relative geometry fidelity remains "
        "a separate diagnostic.\n\n" + "\n".join(table) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"comparison": str(comparison_csv.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
