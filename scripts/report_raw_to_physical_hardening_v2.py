#!/usr/bin/env python3
"""Aggregate the five frozen P5 hardening outcomes without computing a success rate."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path

EPISODES = (
    "hocap_subject_9_20231027_125019__right__G16_3__ep00",
    "hocap_subject_6_20231025_112332__right__G09_4__ep00",
    "hocap_subject_2_20231023_164741__right__G22_3__ep00",
    "hocap_subject_3_20231024_161209__right__G16_2__ep00",
    "hocap_subject_1_20231025_170231__right__G10_3__ep00",
)

TERMINAL_STATUSES = {
    "HARDENING_PASS_FULL_CYCLE",
    "HARDENING_PASS_PICK_ONLY",
    "RAW_QUALITY_REJECTED",
    "RETARGET_FAILED",
    "SOURCE_CONTROLLER_FAILED",
    "SUPPORT_UNRESOLVED",
    "PPO_BUDGET_EXHAUSTED",
    "PF_PASS_DF_FAIL",
    "TECHNICAL_FAILURE",
}

MAIN_COLUMNS = (
    "episode",
    "input_quality",
    "retarget",
    "source_controller",
    "l0_updates",
    "support",
    "frozen_pf_v2",
    "ppo_updates",
    "pf_pick",
    "pf_transport",
    "pf_place",
    "pf_release",
    "pf_retreat",
    "pf_full",
    "df_pose",
    "df_linear",
    "df_angular",
    "status",
)

TIMING_PHASES = (
    "input_quality_scan",
    "retarget_solver",
    "retarget_io",
    "reference",
    "source_controller_l0",
    "support",
    "isaac_bootstrap",
    "frozen_eval",
    "ppo_training",
    "ppo_evaluation",
    "qualification",
    "export",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    return parser


def _read(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"P5_EPISODE_RESULT_OBJECT_REQUIRED:{path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _validate(episode: str, payload: Mapping[str, object]) -> None:
    if (
        payload.get("schema_version") != "HardeningV2P5EpisodeResultV1"
        or payload.get("episode") != episode
        or payload.get("dataset_role") != "PIPELINE_HARDENING_SET_V1"
        or payload.get("held_out") is not False
        or payload.get("status") not in TERMINAL_STATUSES
    ):
        raise ValueError(f"P5_EPISODE_RESULT_INVALID:{episode}")
    missing = sorted(set(MAIN_COLUMNS) - payload.keys())
    if missing:
        raise ValueError(f"P5_EPISODE_MAIN_METRICS_MISSING:{episode}:{missing}")
    timing = payload.get("timing_seconds")
    if not isinstance(timing, dict) or set(timing) != set(TIMING_PHASES):
        raise ValueError(f"P5_EPISODE_TIMING_INVALID:{episode}")
    for phase, value in timing.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"P5_EPISODE_TIMING_VALUE_INVALID:{episode}:{phase}")


def aggregate(report_root: Path) -> dict[str, object]:
    root = report_root.resolve()
    p5 = root / "p5_hardening_regression"
    results: list[dict[str, object]] = []
    for episode in EPISODES:
        path = p5 / "per_episode" / episode / "final_status.json"
        if not path.is_file():
            raise FileNotFoundError(f"P5_TERMINAL_RECEIPT_MISSING:{path}")
        payload = _read(path)
        _validate(episode, payload)
        results.append(payload)

    main_rows = [{column: row[column] for column in MAIN_COLUMNS} for row in results]
    _write_csv(p5 / "main_metrics.csv", main_rows, MAIN_COLUMNS)

    timing_rows = [
        {"episode": row["episode"], "phase": phase, "seconds": row["timing_seconds"][phase]}
        for row in results
        for phase in TIMING_PHASES
    ]
    _write_csv(p5 / "timing.csv", timing_rows, ("episode", "phase", "seconds"))

    replay_lines = [
        "# P5 hardening failure/qualification replay commands",
        "",
        "These traces are PIPELINE_HARDENING_SET_V1 diagnostics, not held-out accepted data.",
        "",
    ]
    for row in results:
        commands = row.get("replay_commands", [])
        if commands:
            replay_lines.extend((f"## {row['episode']}", ""))
            for command in commands:
                replay_lines.extend(("```bash", str(command), "```", ""))
    replay_path = p5 / "replay" / "visualization_commands.md"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text("\n".join(replay_lines).rstrip() + "\n", encoding="utf-8")

    technical = [row["technical_failure"] for row in results if row.get("technical_failure")]
    technical_path = root / "technical_failures.jsonl"
    technical_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in technical),
        encoding="utf-8",
    )

    summary = {
        "schema_version": "RawToPhysicalHardeningV2P5SummaryV1",
        "status": "COMPLETE",
        "dataset_role": "PIPELINE_HARDENING_SET_V1",
        "held_out_benchmark": False,
        "held_out_success_rate_computed": False,
        "contract_changed_after_first_result": False,
        "episode_order": list(EPISODES),
        "terminal_statuses": {row["episode"]: row["status"] for row in results},
        "main_metrics": str((p5 / "main_metrics.csv").resolve()),
        "timing": str((p5 / "timing.csv").resolve()),
        "replay_commands": str(replay_path.resolve()),
    }
    _write_json(p5 / "final_decision.json", summary)
    return summary


def main() -> int:
    args = _parser().parse_args()
    print(json.dumps(aggregate(args.report_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
