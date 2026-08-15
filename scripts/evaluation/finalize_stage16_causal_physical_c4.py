#!/usr/bin/env python3
"""Materialize the required C4 comparison tables and replay manifest.

The input qualifications are immutable Formal20 postprocessing results.  This
script only reads them, keeps all unsuccessful episodes, and uses ``N/A`` for
legacy zero-g quantities which were never captured under the C4 contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIPS = ("hocap_170105", "hocap_170650")
MODES = (("v3", "aggregate_v3", "V3"), ("v4", "strict_per_finger_v4", "V4"))
LEGACY_ZERO_G_PATHS = {
    "V3": REPO_ROOT
    / ".local/reports/stage16d_reward_v3_pairforce_unblock/v1_pairforce/{clip}"
    / "ppo_r7_pairforce_evaluation.json",
    "V4": REPO_ROOT
    / ".local/reports/stage16d_strict_per_finger_v4/{clip}/formal"
    / "v4_formal_selected_1064960_source_contact_evaluation.json",
}
TABLE_COLUMNS = (
    "Contract",
    "Physics",
    "Reward",
    "Er°",
    "Et cm",
    "Ej cm",
    "Eft cm",
    "SRkin",
    "SRphysics",
    "SRqualified",
    "Tip recall",
    "Persistent recall",
    "Cross-comp",
    "Persistent cross-comp",
    "Fully missing",
    "No-hand fraction",
    "Longest gap",
    "Δv mean",
    "Δv p95",
    "Δv terminal",
    "Δω mean",
    "Δω p95",
    "Δω terminal",
    "Stability",
    "H-O max mm",
    "H-O p95 mm",
    "Active p95 mm",
    "H-T max mm",
    "Inter-finger mm",
    "Geometry pass",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"C4_FINALIZE_JSON_OBJECT_REQUIRED:{path}")
    return value


def _receipt(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"C4_FINALIZE_REQUIRED_ARTIFACT_MISSING:{path}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, object]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(columns) + " |\n"
    separator = "| " + " | ".join("---" for _ in columns) + " |\n"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "N/A")) for key in columns) + " |\n")
    path.write_text("".join(lines), encoding="utf-8")


def _value(value: object, digits: int = 6) -> object:
    return "N/A" if value is None else round(float(value), digits)


def _c4_row(qualification: dict[str, Any]) -> dict[str, object]:
    suite = qualification["evaluation_suite_v2"]["aggregate"]
    interaction = qualification["interaction"]["aggregate"]
    flight = qualification["flight"]
    twist = qualification["twist"]
    penetration = qualification["penetration"]
    geometry = penetration["aggregate"]
    table = penetration["hand_table"]
    return {
        "Contract": "C4 Formal20 table-supported",
        "Physics": "full gravity; nominal friction; inferred table",
        "Reward": "V3" if qualification["contact_mode"] == "aggregate_v3" else "V4",
        "Er°": _value(suite["E_r_mean_deg"]["mean"]),
        "Et cm": _value(suite["E_t_mean_cm"]["mean"]),
        "Ej cm": _value(suite["E_j_mean_cm"]["mean"]),
        "Eft cm": _value(suite["E_ft_mean_cm"]["mean"]),
        "SRkin": _value(suite["kinematic_success"]["rate"]),
        "SRphysics": _value(suite["physics_success"]["rate"]),
        "SRqualified": _value(suite["qualified_success"]["rate"]),
        "Tip recall": _value(interaction["source_tip_recall"]),
        "Persistent recall": _value(interaction["source_persistent_tip_recall"]),
        "Cross-comp": _value(interaction["cross_finger_compensation"]),
        "Persistent cross-comp": _value(interaction["persistent_cross_finger_compensation"]),
        "Fully missing": _value(interaction["fully_missing_source_contact"]),
        "No-hand fraction": _value(flight["no_hand_object_contact_fraction"]),
        "Longest gap": flight["longest_flight_gap"],
        "Δv mean": _value(twist["Delta_v_mean_mps"]),
        "Δv p95": _value(twist["Delta_v_p95_mps"]),
        "Δv terminal": _value(twist["Delta_v_terminal_mps"]),
        "Δω mean": _value(twist["Delta_omega_mean_radps"]),
        "Δω p95": _value(twist["Delta_omega_p95_radps"]),
        "Δω terminal": _value(twist["Delta_omega_terminal_radps"]),
        "Stability": _value(twist["terminal_stability_rate"]),
        "H-O max mm": _value(geometry["max_penetration_m"] * 1000.0),
        "H-O p95 mm": _value(geometry["p95_penetration_m"] * 1000.0),
        "Active p95 mm": _value(geometry["active_p95_penetration_m"] * 1000.0),
        "H-T max mm": _value(table["max_penetration_m"] * 1000.0),
        "Inter-finger mm": _value(penetration["interfinger_max_penetration_m"] * 1000.0),
        "Geometry pass": bool(
            penetration["absolute_geometry_pass"] and table["absolute_geometry_pass"]
        ),
    }


def _legacy_zero_g_evidence(*, clip: str, reward: str) -> dict[str, object]:
    """Read only quantities actually captured by an historical zero-g audit.

    Historical V3 and V4 used different evaluators, so this deliberately does
    not invent C4 metrics or reinterpret their terminal tracking values as
    Evaluation Suite V2 errors.
    """

    path = Path(str(LEGACY_ZERO_G_PATHS[reward]).format(clip=clip))
    value = _read(path)
    evidence: dict[str, object] = {
        "reward": reward,
        "clip": clip,
        "receipt": _receipt(path),
        "status": value.get("status"),
        "schema_version": value.get("schema_version"),
        "captured_metrics": {},
        "not_comparable_reason": (
            "Historical zero-g evaluator did not capture the C4 Formal20 "
            "table-support, exact-geometry, twist, or Evaluation Suite V2 contract."
        ),
    }
    metrics = evidence["captured_metrics"]
    assert isinstance(metrics, dict)
    if reward == "V3":
        summary = value.get("frame_zero_summary", {})
        metrics.update(
            {
                "episodes": summary.get("episodes"),
                "ever_contact_rate": summary.get("ever_contact_rate"),
                "terminal_contact_rate": summary.get("terminal_contact_rate"),
                "final_object_rotation_error_deg_mean": _legacy_rad_to_deg(
                    summary.get("final_object_rotation_error_rad", {}).get("mean")
                    if isinstance(summary.get("final_object_rotation_error_rad"), dict)
                    else None
                ),
                "final_object_position_error_cm_mean": _legacy_m_to_cm(
                    summary.get("final_object_position_error_m", {}).get("mean")
                    if isinstance(summary.get("final_object_position_error_m"), dict)
                    else None
                ),
            }
        )
    else:
        aggregate = value.get("aggregate", {})
        metrics.update(
            {
                "source_tip_recall": aggregate.get("source_tip_recall"),
                "persistent_source_tip_recall": aggregate.get("persistent_source_tip_recall"),
                "no_hand_object_contact_flight_fraction": aggregate.get(
                    "no_hand_object_contact_flight_fraction"
                ),
                "longest_no_hand_flight_gap": aggregate.get("longest_no_hand_flight_gap"),
            }
        )
    return evidence


def _legacy_rad_to_deg(value: object) -> object:
    return "N/A" if value is None else _value(float(value) * 180.0 / math.pi)


def _legacy_m_to_cm(value: object) -> object:
    return "N/A" if value is None else _value(float(value) * 100.0)


def _legacy_zero_row(*, evidence: dict[str, object]) -> dict[str, object]:
    reward = str(evidence["reward"])
    metrics = evidence["captured_metrics"]
    assert isinstance(metrics, dict)
    row = {
        **{column: "N/A" for column in TABLE_COLUMNS},
        "Contract": "historical frozen zero-g (not recomputed)",
        "Physics": "zero gravity",
        "Reward": reward,
        "Geometry pass": "N/A (not captured under C4 contract)",
        "H-T max mm": "N/A (not captured)",
    }
    # These exact source-contact fields have the same definitions in the V4
    # historical audit and C4, unlike its missing table/geometry/twist suite.
    if reward == "V4":
        row.update(
            {
                "Tip recall": _value(metrics.get("source_tip_recall")),
                "Persistent recall": _value(metrics.get("persistent_source_tip_recall")),
                "No-hand fraction": _value(metrics.get("no_hand_object_contact_flight_fraction")),
                "Longest gap": metrics.get("longest_no_hand_flight_gap", "N/A"),
            }
        )
    return row


def _selection(rows: list[dict[str, object]]) -> dict[str, int]:
    ranked = sorted(
        rows,
        key=lambda row: (
            not bool(row["qualified_success"]),
            float(row["E_r_mean_deg"]) + float(row["E_t_mean_cm"]),
        ),
    )
    by_error = sorted(rows, key=lambda row: float(row["E_r_mean_deg"]) + float(row["E_t_mean_cm"]))
    failure = next(
        (row for row in reversed(by_error) if not bool(row["qualified_success"])), by_error[-1]
    )
    return {
        "representative_best": int(ranked[0]["episode"]),
        "median_typical": int(by_error[len(by_error) // 2]["episode"]),
        "representative_failure_or_worst": int(failure["episode"]),
    }


def _nested(value: dict[str, Any], *keys: str) -> float | None:
    cursor: object = value
    for key in keys:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return float(cursor) if cursor is not None else None


def _conclusion(
    v3: dict[str, Any], v4: dict[str, Any], metrics: tuple[tuple[str, bool, tuple[str, ...]], ...]
) -> dict[str, object]:
    """Compare each frozen metric independently; never turn this into a reward scalar."""

    rows = []
    wins = {"V3": 0, "V4": 0}
    for label, higher_is_better, path in metrics:
        left = _nested(v3, *path)
        right = _nested(v4, *path)
        winner = "N/A"
        if left is not None and right is not None and left != right:
            winner = "V3" if (left > right) == higher_is_better else "V4"
            wins[winner] += 1
        rows.append({"metric": label, "V3": left, "V4": right, "directional_winner": winner})
    if wins["V3"] and not wins["V4"]:
        verdict = "V3 better"
    elif wins["V4"] and not wins["V3"]:
        verdict = "V4 better"
    else:
        verdict = "mixed"
    return {"verdict": verdict, "wins": wins, "metrics": rows}


def _mode_conclusions(v3: dict[str, Any], v4: dict[str, Any]) -> dict[str, dict[str, object]]:
    return {
        "interaction": _conclusion(
            v3,
            v4,
            (
                ("source_tip_recall", True, ("interaction", "aggregate", "source_tip_recall")),
                (
                    "source_persistent_tip_recall",
                    True,
                    ("interaction", "aggregate", "source_persistent_tip_recall"),
                ),
                (
                    "cross_finger_compensation",
                    False,
                    ("interaction", "aggregate", "cross_finger_compensation"),
                ),
                (
                    "persistent_cross_finger_compensation",
                    False,
                    ("interaction", "aggregate", "persistent_cross_finger_compensation"),
                ),
                (
                    "fully_missing_source_contact",
                    False,
                    ("interaction", "aggregate", "fully_missing_source_contact"),
                ),
                (
                    "no_hand_object_contact_fraction",
                    False,
                    ("flight", "no_hand_object_contact_fraction"),
                ),
            ),
        ),
        "twist": _conclusion(
            v3,
            v4,
            (
                ("Delta_v_mean_mps", False, ("twist", "Delta_v_mean_mps")),
                ("Delta_v_p95_mps", False, ("twist", "Delta_v_p95_mps")),
                ("Delta_v_terminal_mps", False, ("twist", "Delta_v_terminal_mps")),
                ("Delta_omega_mean_radps", False, ("twist", "Delta_omega_mean_radps")),
                ("Delta_omega_p95_radps", False, ("twist", "Delta_omega_p95_radps")),
                ("Delta_omega_terminal_radps", False, ("twist", "Delta_omega_terminal_radps")),
                ("terminal_stability_rate", True, ("twist", "terminal_stability_rate")),
            ),
        ),
        "penetration": _conclusion(
            v3,
            v4,
            (
                ("hand_object_max", False, ("penetration", "aggregate", "max_penetration_m")),
                ("hand_object_p95", False, ("penetration", "aggregate", "p95_penetration_m")),
                (
                    "hand_table_max",
                    False,
                    ("penetration", "hand_table", "max_penetration_m"),
                ),
                (
                    "interfinger_max",
                    False,
                    ("penetration", "interfinger_max_penetration_m"),
                ),
            ),
        ),
        "evaluation_suite_v2": _conclusion(
            v3,
            v4,
            (
                (
                    "E_r_mean_deg",
                    False,
                    ("evaluation_suite_v2", "aggregate", "E_r_mean_deg", "mean"),
                ),
                (
                    "E_t_mean_cm",
                    False,
                    ("evaluation_suite_v2", "aggregate", "E_t_mean_cm", "mean"),
                ),
                (
                    "E_j_mean_cm",
                    False,
                    ("evaluation_suite_v2", "aggregate", "E_j_mean_cm", "mean"),
                ),
                (
                    "E_ft_mean_cm",
                    False,
                    ("evaluation_suite_v2", "aggregate", "E_ft_mean_cm", "mean"),
                ),
                (
                    "SRkin",
                    True,
                    ("evaluation_suite_v2", "aggregate", "kinematic_success", "rate"),
                ),
                (
                    "SRphysics",
                    True,
                    ("evaluation_suite_v2", "aggregate", "physics_success", "rate"),
                ),
                (
                    "SRqualified",
                    True,
                    ("evaluation_suite_v2", "aggregate", "qualified_success", "rate"),
                ),
            ),
        ),
    }


def _write_conclusions_markdown(path: Path, conclusion: dict[str, dict[str, object]]) -> None:
    lines = ["# C4 Reward V3 vs V4 Conclusions\n\n"]
    for category, value in conclusion.items():
        lines.append(f"## {category}\n\nVerdict: **{value['verdict']}**. ")
        wins = value["wins"]
        assert isinstance(wins, dict)
        lines.append(f"Directional metric wins — V3: {wins['V3']}; V4: {wins['V4']}.\n\n")
        lines.append("| Metric | V3 | V4 | Directional winner |\n| --- | ---: | ---: | --- |\n")
        metrics = value["metrics"]
        assert isinstance(metrics, list)
        for row in metrics:
            assert isinstance(row, dict)
            lines.append(
                f"| {row['metric']} | {row['V3']} | {row['V4']} | {row['directional_winner']} |\n"
            )
        lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")


def _stage_summary(root: Path, directory: str, clip: str, stage: str) -> dict[str, object]:
    path = root / "training" / directory / clip / stage.lower() / "training_result.json"
    value = _read(path)
    warning_root = path.parent / "saturation" / "warnings"
    return {
        "stage": stage,
        "status": value.get("status"),
        "cumulative_samples": value.get("cumulative_samples"),
        "checkpoint": value.get("checkpoint"),
        "warning_receipt_count": len(list(warning_root.glob("receipt*.json")))
        if warning_root.is_dir()
        else 0,
    }


def _simulation_summary(
    root: Path, directory: str, clip: str, result: dict[str, Any]
) -> dict[str, object]:
    trace_root = REPO_ROOT / ".local/sim_data/stage16_causal_physical_c4" / directory / clip
    episodes = result["episodes"]
    assert isinstance(episodes, list)
    success_count = sum(bool(row["qualified_success"]) for row in episodes if isinstance(row, dict))
    return {
        "directory": str(trace_root),
        "episode_count": len(episodes),
        "success_count": success_count,
        "failure_count": len(episodes) - success_count,
        "trace_count": len(list(trace_root.glob("episode_*.npz"))) if trace_root.is_dir() else 0,
    }


def _write_handoff(
    path: Path,
    *,
    root: Path,
    results: dict[tuple[str, str], dict[str, Any]],
    latest: list[dict[str, object]],
    conclusions: dict[str, dict[str, dict[str, object]]],
) -> None:
    lines = [
        "# Stage16 Causal Physical PPO C0→C4 Handoff\n\n",
        "## Contract\n\n",
        "- `BRANCH=feature/ppo-physical`\n",
        "- `SATURATION_0_25_HARD_STOP=NO`; `SATURATION_0_25_WARNING=YES`\n",
        "- `PERFORMANCE_PROMOTION_GATE=NO`; promotion is planned-sample completion only.\n",
        "- `REFERENCE_MODIFIED=NO`; `INFERRED_TABLE_ACTIVE=YES`.\n",
        "- `GUIDANCE_ADDED=NO`; `OBJECT_ROLLOUT_WRITE_ADDED=NO`; "
        "`WRIST_ROOT_ROLLOUT_WRITE_ADDED=NO`.\n\n",
        "## Four PPO lineages\n\n",
        "| Reward / clip | C0 | C1 | C2 | C3 | C4 |\n| --- | --- | --- | --- | --- | --- |\n",
    ]
    for directory, _mode, label in MODES:
        for clip in CLIPS:
            stages = [
                _stage_summary(root, directory, clip, stage)
                for stage in ("C0", "C1", "C2", "C3", "C4")
            ]
            cells = [
                f"{entry['status']} ({entry['cumulative_samples']} samples; "
                f"warnings={entry['warning_receipt_count']})"
                for entry in stages
            ]
            lines.append(f"| {label} / {clip} | " + " | ".join(cells) + " |\n")
    lines.extend(
        [
            "\n## Latest four C4 results\n\n",
            "See [comparison/latest_four_results.md](comparison/latest_four_results.md) for the "
            "complete checkpoint, samples, interaction, twist, and penetration table.\n\n",
        ]
    )
    for clip in CLIPS:
        lines.append(f"## {clip}: V3 vs V4\n\n")
        for category, value in conclusions[clip].items():
            lines.append(f"- {category}: **{value['verdict']}**.\n")
        lines.append(
            f"- Detail: [comparison/{clip}_v3_v4_conclusions.md]"
            f"(comparison/{clip}_v3_v4_conclusions.md).\n"
        )
        lines.append(
            f"- Four-row zero-g/C4 comparison: [comparison/{clip}_zero_g_vs_c4.md]"
            f"(comparison/{clip}_zero_g_vs_c4.md).\n\n"
        )
    lines.append("## Simulation data\n\n")
    lines.append("| Reward / clip | Directory | Episodes | Success | Failure | Traces |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |\n")
    for directory, _mode, label in MODES:
        for clip in CLIPS:
            summary = _simulation_summary(root, directory, clip, results[(directory, clip)])
            lines.append(
                f"| {label} / {clip} | `{summary['directory']}` | {summary['episode_count']} | "
                f"{summary['success_count']} | {summary['failure_count']} | "
                f"{summary['trace_count']} |\n"
            )
    lines.extend(
        [
            "\n## Replay\n\n",
            "The four executable main replay commands and representative failure replays are in "
            "[replay/visualization_commands.md](replay/visualization_commands.md).\n\n",
            "## Receipts\n\n",
            "- Frozen inputs: [frozen_inputs.json](frozen_inputs.json)\n",
            "- Curriculum: [curriculum_contract.json](curriculum_contract.json)\n",
            "- Warnings: [warning_contract.json](warning_contract.json)\n",
            "- Technical recoveries: [technical_failures.json](technical_failures.json)\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT / ".local/reports/stage16_causal_physical_c0_c4"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    results: dict[tuple[str, str], dict[str, Any]] = {}
    selections: dict[str, dict[str, int]] = {}
    for directory, _mode, _label in MODES:
        for clip in CLIPS:
            path = root / "formal" / directory / clip / "qualification.json"
            value = _read(path)
            if value.get("status") != "C4_FORMAL20_COMPLETE":
                raise ValueError(f"C4_FINALIZE_FORMAL_RESULT_INVALID:{path}")
            results[(directory, clip)] = value
            selections[f"{directory}/{clip}"] = _selection(value["episodes"])
    comparison = root / "comparison"
    interaction = root / "interaction"
    twist = root / "twist"
    penetration = root / "penetration"
    suite_root = root / "evaluation_suite_v2"
    latest: list[dict[str, object]] = []
    legacy_zero_g: dict[str, dict[str, object]] = {}
    conclusions: dict[str, dict[str, dict[str, object]]] = {}
    for clip in CLIPS:
        legacy_v3 = _legacy_zero_g_evidence(clip=clip, reward="V3")
        legacy_v4 = _legacy_zero_g_evidence(clip=clip, reward="V4")
        legacy_zero_g[clip] = {"V3": legacy_v3, "V4": legacy_v4}
        rows = [_legacy_zero_row(evidence=legacy_v3), _legacy_zero_row(evidence=legacy_v4)]
        for directory, _mode, _label in MODES:
            value = results[(directory, clip)]
            rows.append(_c4_row(value))
            _write_json(interaction / f"{directory}_{clip}.json", value["interaction"])
            _write_json(twist / f"{directory}_{clip}.json", value["twist"])
            _write_json(penetration / f"{directory}_{clip}.json", value["penetration"])
            _write_json(suite_root / f"{directory}_{clip}.json", value["evaluation_suite_v2"])
            aggregate = value["evaluation_suite_v2"]["aggregate"]
            checkpoint = value["provenance"]["checkpoint"]
            training = _read(root / "training" / directory / clip / "c4" / "training_result.json")
            latest.append(
                {
                    "Clip": clip,
                    "Reward": _label,
                    "C4 checkpoint": checkpoint["path"],
                    "Total curriculum samples": training["cumulative_samples"],
                    "SRkin": _value(aggregate["kinematic_success"]["rate"]),
                    "SRphysics": _value(aggregate["physics_success"]["rate"]),
                    "SRqualified": _value(aggregate["qualified_success"]["rate"]),
                    "Interaction summary": value["interaction"]["aggregate"],
                    "Twist summary": value["twist"],
                    "Penetration summary": value["penetration"],
                }
            )
        _write_csv(comparison / f"{clip}_zero_g_vs_c4.csv", rows, TABLE_COLUMNS)
        _write_md(comparison / f"{clip}_zero_g_vs_c4.md", rows, TABLE_COLUMNS)
        c4_rows = rows[2:]
        _write_csv(comparison / f"{clip}_v3_v4.csv", c4_rows, TABLE_COLUMNS)
        _write_md(comparison / f"{clip}_v3_v4.md", c4_rows, TABLE_COLUMNS)
        conclusion = _mode_conclusions(results[("v3", clip)], results[("v4", clip)])
        conclusions[clip] = conclusion
        _write_json(comparison / f"{clip}_v3_v4_conclusions.json", conclusion)
        _write_conclusions_markdown(comparison / f"{clip}_v3_v4_conclusions.md", conclusion)
    _write_json(comparison / "legacy_zero_g_sources.json", legacy_zero_g)
    latest_columns = tuple(latest[0])
    _write_csv(comparison / "latest_four_results.csv", latest, latest_columns)
    _write_md(comparison / "latest_four_results.md", latest, latest_columns)
    _write_json(root / "replay" / "selection.json", selections)
    commands = []
    for directory, _mode, label in MODES:
        for clip in CLIPS:
            selection = selections[f"{directory}/{clip}"]
            sim = REPO_ROOT / ".local/sim_data/stage16_causal_physical_c4" / directory / clip
            main = (
                "conda run -n toporetarget-isaaclab python "
                "scripts/rl/isaaclab/replay_stage16d_simulation_trace.py "
                "--accept-eula --headless --max-loops 1 "
                f"--trace {sim}/episode_{selection['representative_best']:03d}.npz "
                f"--object {clip} --validation-output {root}/replay/{directory}_{clip}_main.json"
            )
            failure = (
                "conda run -n toporetarget-isaaclab python "
                "scripts/rl/isaaclab/replay_stage16d_simulation_trace.py "
                "--accept-eula --headless --max-loops 1 "
                f"--trace {sim}/episode_{selection['representative_failure_or_worst']:03d}.npz "
                f"--object {clip}"
            )
            commands.append(
                f"### {label} / {clip} / C4\n\n`{main}`\n\nFailure/worst replay: `{failure}`\n"
            )
    (root / "replay").mkdir(parents=True, exist_ok=True)
    (root / "replay" / "visualization_commands.md").write_text(
        "\n".join(commands), encoding="utf-8"
    )
    summary = {
        "schema_version": "Stage16CausalPhysicalC4SummaryV1",
        "latest_four": latest,
        "replay_selection": selections,
        "legacy_zero_g_sources": str((comparison / "legacy_zero_g_sources.json").resolve()),
        "reward_mode_conclusions": conclusions,
    }
    _write_json(root / "final_summary.json", summary)
    frozen_inputs = {
        "schema_version": "Stage16CausalPhysicalFrozenInputsV1",
        "reference_kinematics": {
            clip: _receipt(
                REPO_ROOT
                / ".local/reports/stage16d_reference_kinematics_v2/references"
                / f"{clip}.reference_kinematics_v2.npz"
            )
            for clip in CLIPS
        },
        "inferred_support": {
            clip: _receipt(
                REPO_ROOT
                / ".local/reports/stage16_support_reconstruction/inference"
                / clip
                / "table_proxy.json"
            )
            for clip in CLIPS
        },
        "contact_contracts": {
            "aggregate_v3": _receipt(
                REPO_ROOT
                / ".local/reports/stage16d_reward_v3_pairforce_unblock/contact_reward_contract.json"
            ),
            "strict_per_finger_v4": _receipt(
                REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4/strict_v4_contract.json"
            ),
        },
    }
    _write_json(root / "frozen_inputs.json", frozen_inputs)
    curriculum_path = REPO_ROOT / "configs/rl/stage16/stage16_gravity_friction_curriculum_v1.yaml"
    _write_json(
        root / "curriculum_contract.json",
        {
            "receipt": _receipt(curriculum_path),
            "stages": {
                "C0": {"gravity_scale": 0.0, "friction_scale": 2.0},
                "C1": {"gravity_scale": 0.25, "friction_scale": 1.75},
                "C2": {"gravity_scale": 0.5, "friction_scale": 1.5},
                "C3": {"gravity_scale": 0.75, "friction_scale": 1.25},
                "C4": {"gravity_scale": 1.0, "friction_scale": 1.0},
            },
            "promotion_semantics": "planned_sample_budget_completed_only",
        },
    )
    _write_json(
        root / "warning_contract.json",
        {
            "schema_version": "Stage16CausalPhysicalWarningContractV1",
            "saturation_abs_actor_mean_threshold": 0.98,
            "saturation_fraction_warning_threshold": 0.25,
            "saturation_hard_stop": False,
            "warning_checkpoint_policy": "checkpoint_and_receipt_before_continuing",
            "optimization_health_hard_stop": False,
        },
    )
    gpu_receipts = {}
    for directory, _mode, _label in MODES:
        for clip in CLIPS:
            path = root / "training" / directory / clip / "c0" / "gpu_preflight.json"
            if path.is_file():
                gpu_receipts[f"{directory}/{clip}"] = _read(path)
    _write_json(root / "resource_usage.json", {"gpu_preflight": gpu_receipts})
    failures = []
    technical_path = root / "technical_failures.jsonl"
    if technical_path.is_file():
        failures = [json.loads(line) for line in technical_path.read_text().splitlines() if line]
    _write_json(root / "technical_failures.json", {"failures": failures})
    _write_handoff(
        root / "handoff.md",
        root=root,
        results=results,
        latest=latest,
        conclusions=conclusions,
    )
    (root / "final_summary.md").write_text(
        "# Stage16 Causal Physical PPO C0→C4 Summary\n\n"
        "All four C4 Formal20 results, including failures, are listed in "
        "[comparison/latest_four_results.md](comparison/latest_four_results.md).\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "C4_COMPARISON_COMPLETE", "root": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
