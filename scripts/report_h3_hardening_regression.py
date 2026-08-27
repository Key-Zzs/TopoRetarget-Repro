#!/usr/bin/env python3
"""Aggregate terminal H3-C Hardening5 results and evaluate pipeline readiness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

EPISODES = (
    "hocap_subject_9_20231027_125019__right__G16_3__ep00",
    "hocap_subject_6_20231025_112332__right__G09_4__ep00",
    "hocap_subject_2_20231023_164741__right__G22_3__ep00",
    "hocap_subject_3_20231024_161209__right__G16_2__ep00",
    "hocap_subject_1_20231025_170231__right__G10_3__ep00",
)

TERMINAL_STATUSES = {
    "ACCEPTED_FROZEN_FULL_CYCLE",
    "ACCEPTED_AFTER_REFINEMENT_FULL_CYCLE",
    "ACCEPTED_PICK_ONLY",
    "PPO_BUDGET_EXHAUSTED",
    "SOURCE_CONTROLLER_TRUE_HARD_FAILURE",
    "SUPPORT_UNRESOLVED",
    "RETARGET_FAILED_AFTER_VALID_INPUT",
    "PF_PASS_DF_FAIL",
    "TECHNICAL_FAILURE",
}

MAIN_COLUMNS = (
    "episode",
    "retarget",
    "source_route",
    "source_executable",
    "source_fidelity",
    "support",
    "frozen_pf",
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"H3C_JSON_OBJECT_REQUIRED:{path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"H3C_REPORT_OUTPUT_EXISTS:{path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"H3C_REPORT_OUTPUT_EXISTS:{path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _validate(episode: str, row: dict[str, Any]) -> None:
    if not (
        row.get("schema_version") == "H3HardeningRegressionEpisodeResultV1"
        and row.get("episode") == episode
        and row.get("dataset_role") == "PIPELINE_HARDENING_SET_V1"
        and row.get("held_out") is False
        and row.get("status") in TERMINAL_STATUSES
    ):
        raise ValueError(f"H3C_EPISODE_RESULT_INVALID:{episode}")
    missing = sorted(set(MAIN_COLUMNS) - row.keys())
    if missing:
        raise ValueError(f"H3C_MAIN_METRICS_MISSING:{episode}:{missing}")
    readiness = row.get("pipeline_readiness")
    if not isinstance(readiness, dict):
        raise ValueError(f"H3C_READINESS_EVIDENCE_MISSING:{episode}")
    required = {
        "exact_retarget_terminal",
        "entered_frozen_full_gravity_evaluation",
        "blocked_by_task_fidelity_only_gate",
        "unresolved_generic_technical_blocker",
        "initial_physical_failure",
        "ppo_terminal",
        "explicit_physical_invalid_state",
        "per_episode_tuning",
    }
    if set(readiness) != required:
        raise ValueError(f"H3C_READINESS_EVIDENCE_FIELDS_INVALID:{episode}")
    if (
        not isinstance(row.get("method_contract_hash"), str)
        or len(row["method_contract_hash"]) != 64
    ):
        raise ValueError(f"H3C_METHOD_HASH_INVALID:{episode}")
    timing = row.get("timing_seconds")
    if not isinstance(timing, dict) or not timing:
        raise ValueError(f"H3C_TIMING_REQUIRED:{episode}")
    for phase, value in timing.items():
        if (
            not isinstance(phase, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(f"H3C_TIMING_INVALID:{episode}:{phase}")
    failures = row.get("failure_taxonomy")
    if not isinstance(failures, list):
        raise ValueError(f"H3C_FAILURE_TAXONOMY_REQUIRED:{episode}")


def aggregate(root: Path) -> dict[str, Any]:
    report_root = root.resolve()
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []
    for episode in EPISODES:
        path = report_root / "per_episode" / episode / "final_status.json"
        row = _json(path)
        _validate(episode, row)
        rows.append(row)
        inputs.append({"path": str(path.resolve()), "sha256": _sha256(path)})

    _write_csv(
        report_root / "main_metrics.csv",
        [{field: row[field] for field in MAIN_COLUMNS} for row in rows],
        MAIN_COLUMNS,
    )
    timing_rows = [
        {"episode": row["episode"], "phase": phase, "seconds": seconds}
        for row in rows
        for phase, seconds in sorted(row["timing_seconds"].items())
    ]
    _write_csv(report_root / "timing.csv", timing_rows, ("episode", "phase", "seconds"))
    failure_rows = [
        {
            "episode": row["episode"],
            "classification": str(failure.get("classification", "UNCLASSIFIED")),
            "stage": str(failure.get("stage", "UNKNOWN")),
            "resolved": bool(failure.get("resolved", False)),
            "scientific_verdict_affected": bool(failure.get("scientific_verdict_affected", False)),
            "details": str(failure.get("details", "")),
        }
        for row in rows
        for failure in row["failure_taxonomy"]
        if isinstance(failure, dict)
    ]
    _write_csv(
        report_root / "failure_taxonomy.csv",
        failure_rows,
        (
            "episode",
            "classification",
            "stage",
            "resolved",
            "scientific_verdict_affected",
            "details",
        ),
    )

    hashes = {str(row["method_contract_hash"]) for row in rows}
    gates = {
        "A_5_OF_5_EXACT_RETARGET_TERMINAL": all(
            row["pipeline_readiness"]["exact_retarget_terminal"] is True for row in rows
        ),
        "B_5_OF_5_ENTERED_FROZEN_FULL_GRAVITY_EVAL": all(
            row["pipeline_readiness"]["entered_frozen_full_gravity_evaluation"] is True
            for row in rows
        ),
        "C_ZERO_FIDELITY_ONLY_SOURCE_BLOCKS": all(
            row["pipeline_readiness"]["blocked_by_task_fidelity_only_gate"] is False for row in rows
        ),
        "D_ZERO_UNRESOLVED_GENERIC_BLOCKERS": all(
            row["pipeline_readiness"]["unresolved_generic_technical_blocker"] is False
            for row in rows
        ),
        "E_INITIAL_FAILURES_TERMINAL_OR_EXPLICITLY_INVALID": all(
            row["pipeline_readiness"]["initial_physical_failure"] is False
            or row["pipeline_readiness"]["ppo_terminal"] is True
            or row["pipeline_readiness"]["explicit_physical_invalid_state"] is True
            for row in rows
        ),
        "F_METHOD_CONTRACT_HASHES_IDENTICAL": len(hashes) == 1,
        "G_PER_EPISODE_TUNING_ZERO": all(
            row["pipeline_readiness"]["per_episode_tuning"] is False for row in rows
        ),
    }
    ready = all(gates.values())
    readiness = {
        "schema_version": "H3PipelineReadinessGateV1",
        "status": "PASS" if ready else "FAIL",
        "H3C_READY_FOR_UNSEEN_OBJECT_EXECUTION": "YES" if ready else "NO",
        "gates": gates,
        "method_contract_hashes": sorted(hashes),
        "episode_count": len(rows),
        "inputs": inputs,
    }
    _write_new(
        report_root / "pipeline_readiness.json",
        json.dumps(readiness, indent=2, sort_keys=True) + "\n",
    )
    decision = {
        "schema_version": "H3HardeningRegressionDecisionV1",
        "status": "COMPLETE",
        "dataset_role": "PIPELINE_HARDENING_SET_V1",
        "held_out_benchmark": False,
        "held_out_success_rate_computed": False,
        "episode_order": list(EPISODES),
        "terminal_statuses": {str(row["episode"]): str(row["status"]) for row in rows},
        "pipeline_readiness": readiness,
    }
    _write_new(
        report_root / "final_decision.json",
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
    )
    return decision


def main() -> int:
    args = _parser().parse_args()
    print(json.dumps(aggregate(args.report_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
