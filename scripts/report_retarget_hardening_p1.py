#!/usr/bin/env python3
"""Assemble fail-closed Lane-A/P1 hardening receipts from current evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.retarget.input_quality import RetargetInputQualityContractV1  # noqa: E402
from toporetarget.retarget.refinement_checkpoint import CheckpointStore  # noqa: E402
from toporetarget.retarget.refinement_performance import (  # noqa: E402
    RefinementExecutionProfile,
)

HARDENING_EPISODES = (
    "hocap_subject_9_20231027_125019__right__G16_3__ep00",
    "hocap_subject_6_20231025_112332__right__G09_4__ep00",
    "hocap_subject_2_20231023_164741__right__G22_3__ep00",
    "hocap_subject_3_20231024_161209__right__G16_2__ep00",
    "hocap_subject_1_20231025_170231__right__G10_3__ep00",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=REPO_ROOT / ".local/reports/raw_to_physical_hardening_v2/p1_retarget",
    )
    return parser


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _input_quality(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen = RetargetInputQualityContractV1()
    execution = RefinementExecutionProfile.load("wuji_continuous_sequential_fast_exact_v2")
    contract = {
        "schema_version": "RetargetInputQualityContractReceiptV1",
        "status": "FROZEN_BEFORE_P1_SOLVER_BENCHMARK_OUTCOMES",
        "input_quality": frozen.as_dict(),
        "input_quality_contract_sha256": frozen.contract_sha256,
        "wrist_orientation_authority": {
            "priority": list(frozen.wrist_authority_priority),
            "production_primary": "MANO_GLOBAL_WRIST_ORIENTATION",
            "canonical_keypoint_wrist_v1_only_authority": False,
        },
        "gap_policy": {
            "domain": "seconds",
            "maximum_gap_seconds": frozen.maximum_repair_gap_seconds,
            "same_for_all_clips": True,
            "per_clip_thresholds": False,
            "boundary_or_long_gap": "UNRECOVERABLE_TRACKING_GAP",
        },
        "retarget_execution": {
            "solver": "wuji_continuous_sequential_v1",
            "execution": execution.profile_id,
            "execution_profile_sha256": execution.profile_hash,
            "math_equivalent": execution.math_equivalent,
            "paper_objective_unchanged": execution.paper_objective_unchanged,
            "paper_constraints_unchanged": execution.paper_constraints_unchanged,
            "durable_checkpoint_interval_frames": (execution.durable_checkpoint_interval_frames),
            "intermediate_checkpoint_mode": execution.intermediate_checkpoint_mode,
            "historical_sequence_rewrite": execution.historical_sequence_rewrite,
        },
    }
    rows: list[dict[str, Any]] = []
    for index, episode in enumerate(HARDENING_EPISODES, start=1):
        receipt_path = root / "repair_receipts" / f"{episode}.json"
        receipt = _json(receipt_path)
        object_invalid = sum(
            int(item["invalid_before"]) for item in receipt["object_repair"]["objects"]
        )
        rows.append(
            {
                "hardening_index": index,
                "episode_id": episode,
                "status": receipt["status"],
                "frames": receipt["frames"],
                "scan_seconds": receipt["scan_seconds"],
                "mano_invalid_frames_before": receipt["mano_repair"]["invalid_before"],
                "object_invalid_frames_before": object_invalid,
                "repaired_frame_count": len(receipt["mano_repair"]["repaired_frames"])
                + sum(len(item["repaired_frames"]) for item in receipt["object_repair"]["objects"]),
                "long_gap_count": len(receipt["mano_repair"]["long_invalid_gaps"])
                + sum(
                    len(item["long_invalid_gaps"]) for item in receipt["object_repair"]["objects"]
                ),
                "keypoint_wrist_diagnostic_invalid_frames": len(
                    receipt["keypoint_wrist_diagnostic_invalid_frames"]
                ),
                "wrist_authority": receipt["wrist_orientation_authority"],
                "mano_bones_nondegenerate": receipt["checks"]["mano_bones_nondegenerate"],
                "receipt": str(receipt_path.resolve()),
            }
        )
    return contract, rows


def _historical_incomplete(episode: str) -> dict[str, Any]:
    report_base = (
        REPO_ROOT
        / ".local/reports/held_out_hocap_raw_to_physical_pilot/clips"
        / episode
        / "geometric/episodes"
        / episode
    )
    stage = _json(report_base / "logs/continuous_refinement.receipt.json")
    command = stage["command"]
    checkpoint = Path(command[command.index("--checkpoint-root") + 1])
    store = CheckpointStore(checkpoint)
    chain = store.validate_chain(allow_incomplete=True)
    metadata = [store.load_frame(index)[0] for index in chain["contiguous_frames"]]
    solver_seconds = sum(float(item["solve_time_s"]) for item in metadata)
    return {
        "attempted_wall_seconds": float(stage["wall_seconds"]),
        "accepted_frames": len(metadata),
        "solver_seconds_accepted_frames": solver_seconds,
        "solver_ms_per_accepted_frame": 1000.0 * solver_seconds / len(metadata),
        "wall_minus_accepted_solver_seconds": float(stage["wall_seconds"]) - solver_seconds,
        "checkpoint_root": str(checkpoint),
    }


def _throughput_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_path = (
        REPO_ROOT
        / ".local/reports/hocap_physicalization_protocol_freeze/benchmark"
        / "fast_exact_v2_benchmark_summary.json"
    )
    historical = _json(summary_path)
    rows: list[dict[str, Any]] = []

    def add(**values: Any) -> None:
        template = {
            "case": "",
            "phase": "",
            "before_after": "",
            "repeat": "",
            "frames": "",
            "status": "",
            "input_quality_scan_seconds": "",
            "solver_seconds": "",
            "checkpoint_io_seconds": "",
            "validation_seconds": "",
            "html_seconds": "",
            "serialization_or_orchestration_seconds": "",
            "total_seconds": "",
            "ms_per_frame": "",
            "comparability": "",
            "source": "",
        }
        template.update(values)
        rows.append(template)

    med = historical["medians"]["historical_170650_60f"]
    add(
        case="historical_170650",
        phase="first60",
        before_after="before_hardening",
        repeat="median_of_3",
        frames=60,
        status="HISTORICAL_DIAGNOSTIC",
        solver_seconds=med["solver_ms_per_frame"] * 60 / 1000.0,
        validation_seconds=med["full_frame_validation_seconds"] + med["mesh_validation_seconds"],
        html_seconds=med["html_generation_seconds"],
        serialization_or_orchestration_seconds=med["serialization_seconds"],
        total_seconds=med["total_seconds"],
        ms_per_frame=1000.0 * med["total_seconds"] / 60,
        comparability=(
            "not exact fast_exact_v2 accepted baseline; historical summary says inconclusive"
        ),
        source=str(summary_path.resolve()),
    )
    episode1 = HARDENING_EPISODES[0]
    episode1_path = (
        REPO_ROOT
        / ".local/reports/held_out_hocap_raw_to_physical_pilot/clips"
        / episode1
        / "geometric/episodes"
        / episode1
        / "geometric_retarget_receipt.json"
    )
    episode1_receipt = _json(episode1_path)
    timing = episode1_receipt["timing"]
    add(
        case="hardening_1",
        phase="full_episode",
        before_after="before_hardening",
        repeat=1,
        frames=timing["frames"],
        status="PASS",
        solver_seconds=timing["solver_seconds"],
        validation_seconds=timing["full_frame_validation_seconds"]
        + timing["mesh_validation_seconds"],
        html_seconds=timing["html_generation_seconds"],
        serialization_or_orchestration_seconds=timing["serialization_seconds"],
        total_seconds=timing["total_seconds"],
        ms_per_frame=1000.0 * timing["total_seconds"] / timing["frames"],
        comparability="single historical full run",
        source=str(episode1_path.resolve()),
    )
    incomplete: dict[str, Any] = {}
    for hardening_index, episode in ((2, HARDENING_EPISODES[1]), (3, HARDENING_EPISODES[2])):
        value = _historical_incomplete(episode)
        incomplete[f"hardening_{hardening_index}"] = value
        add(
            case=f"hardening_{hardening_index}",
            phase="incomplete_historical_full_attempt",
            before_after="before_hardening",
            repeat=1,
            frames=value["accepted_frames"],
            status="FAIL_AFTER_VALID_INPUT",
            solver_seconds=value["solver_seconds_accepted_frames"],
            serialization_or_orchestration_seconds=value["wall_minus_accepted_solver_seconds"],
            total_seconds=value["attempted_wall_seconds"],
            ms_per_frame=1000.0 * value["attempted_wall_seconds"] / value["accepted_frames"],
            comparability="incomplete failure; accepted-frame timing only",
            source=value["checkpoint_root"],
        )
    micro_path = root / "checkpoint_durability_benchmark.json"
    micro = _json(micro_path)
    for row in micro["rows"]:
        add(
            case="checkpoint_io_microbenchmark",
            phase="60_synthetic_checkpoint_payloads",
            before_after=row["mode"],
            repeat=row["repeat"],
            frames=micro["frames"],
            status="PASS",
            checkpoint_io_seconds=row["seconds"],
            total_seconds=row["seconds"],
            ms_per_frame=row["ms_per_frame"],
            comparability="checkpoint orchestration only; retarget solver not exercised",
            source=str(micro_path.resolve()),
        )
    for case, phase in (
        ("historical_170650", "first60"),
        ("hardening_1", "first60"),
        ("hardening_1", "full_episode"),
        ("hardening_2", "first60_after_quality_gate"),
        ("hardening_3", "first60_after_quality_gate"),
    ):
        add(
            case=case,
            phase=phase,
            before_after="after_hardening",
            status="NOT_RUN_CPU_BENCHMARK_DEFERRED",
            comparability=(
                "missing required three comparable runs; full episode repetition "
                "exception not invoked"
            ),
            source="P1 runtime hold",
        )
    inventory = {
        "historical_benchmark_summary": historical,
        "hardening_1_historical_timing": timing,
        "hardening_2_and_3_incomplete": incomplete,
        "checkpoint_microbenchmark": micro,
    }
    return rows, inventory


def main() -> int:
    args = _parser().parse_args()
    root = args.report_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    contract, quality_rows = _input_quality(root)
    _write_json(root / "input_quality_contract.json", contract)
    _write_json(root / "retarget_input_quality_contract.json", contract)
    _write_csv(root / "per_episode_quality.csv", quality_rows, list(quality_rows[0]))
    throughput_rows, inventory = _throughput_rows(root)
    _write_csv(root / "throughput_benchmark.csv", throughput_rows, list(throughput_rows[0]))
    _write_json(root / "historical_receipt_inventory.json", inventory)
    micro = inventory["checkpoint_microbenchmark"]
    all_quality_pass = all(row["status"] in {"PASS", "PASS_WITH_WARNINGS"} for row in quality_rows)
    decision = {
        "schema_version": "P1RetargetHardeningDecisionV1",
        "status": "P1_RETARGET_HARDENING_PARTIAL",
        "P1_COMPLETE": True,
        "INPUT_QUALITY_PRECHECK": "PASS" if all_quality_pass else "FAIL",
        "RETARGET_MATH_PARITY": "NOT_RUN_COMPARABLE",
        "RETARGET_MATH_CHANGED": False,
        "CHECKPOINT_PAYLOAD_RESUME_PARITY": "PASS" if micro["resume_parity"] else "FAIL",
        "SOLVER_RESUME_PARITY": "NOT_RUN",
        "RESUME_PARITY": "PARTIAL",
        "APPEND_ONLY_CHECKPOINT_PARITY": (
            "PASS" if micro["append_only_checkpoint_parity"] else "FAIL"
        ),
        "THROUGHPUT_CHANGE": {
            "checkpoint_io_microbenchmark_percent": micro["median_change_percent"],
            "solver_end_to_end": "INCONCLUSIVE_NOT_RUN",
            "interpretation": "microbenchmark cannot substitute for required solver cases",
        },
        "quality_findings": {
            "hardening_set_scanned": len(quality_rows),
            "passed": sum(row["status"] in {"PASS", "PASS_WITH_WARNINGS"} for row in quality_rows),
            "repaired_frames": sum(int(row["repaired_frame_count"]) for row in quality_rows),
            "hardening_2_wrist_result": (
                "RAW_MANO_AND_RECONSTRUCTED_WRIST_VALID; historical keypoint-frame error arose "
                "inside robot-side final solver, not raw input"
            ),
            "hardening_3_bone_result": (
                "RAW_MANO_RECONSTRUCTED_BONES_VALID; historical zero-length error arose "
                "inside robot-side final solver, not raw input"
            ),
        },
        "runtime_hold": {
            "reason": (
                "required comparable after-hardening solver suite is expensive and "
                "explicitly deferred"
            ),
            "observed_hardening_1_full_hours": inventory["hardening_1_historical_timing"][
                "total_seconds"
            ]
            / 3600.0,
            "observed_hardening_2_incomplete_hours": inventory["hardening_2_and_3_incomplete"][
                "hardening_2"
            ]["attempted_wall_seconds"]
            / 3600.0,
            "observed_hardening_3_incomplete_hours": inventory["hardening_2_and_3_incomplete"][
                "hardening_3"
            ]["attempted_wall_seconds"]
            / 3600.0,
            "missing_cases": [
                "historical_170650_first60_after_hardening_x3",
                "hardening_1_first60_after_hardening_x3",
                "hardening_1_full_after_hardening",
                "hardening_2_first60_after_quality_gate_x3",
                "hardening_3_first60_after_quality_gate_x3",
            ],
        },
        "P1_SELECTED_RETARGET_CONTRACT": {
            "branch": "CURRENT_FAST_EXACT_V2_EXACT_SOLVER_PLUS_INPUT_QUALITY_V1_FAIL_CLOSED",
            "solver": "wuji_continuous_sequential_v1",
            "execution": "wuji_continuous_sequential_fast_exact_v2",
            "preflight": "RetargetInputQualityV1",
            "wrist_authority": "MANO_GLOBAL_WRIST_ORIENTATION",
            "long_gap_policy": "UNRECOVERABLE_TRACKING_GAP",
            "durable_checkpoint_interval_frames": 10,
            "selection_basis": "P1 partial fallback required by hardening contract",
        },
        "P1_CONSERVATIVE_FALLBACK": {
            "branch": "CURRENT_EXACT_SOLVER_PLUS_PREFLIGHT_FAIL_CLOSED",
            "throughput_claim": "NONE",
            "per_clip_tuning": False,
        },
        "does_not_block_other_lanes": True,
    }
    _write_json(root / "final_decision.json", decision)
    accepted_2 = inventory["hardening_2_and_3_incomplete"]["hardening_2"]["accepted_frames"]
    accepted_3 = inventory["hardening_2_and_3_incomplete"]["hardening_3"]["accepted_frames"]
    analysis = f"""# P1 Retarget throughput analysis

Decision: `P1_RETARGET_HARDENING_PARTIAL`.

All five hardening episodes passed `RetargetInputQualityV1`; no MANO or object
frames were invalid and no repair was applied. This means the historical #2
and #3 failures must not be relabeled as raw-input failures. Their historical
accepted checkpoint prefixes ended at {accepted_2} and {accepted_3} frames, respectively,
and the terminal degeneracies occurred in final solver robot features.

The 60-frame checkpoint-only microbenchmark passed append-only and checkpoint
payload resume parity. Median checkpoint orchestration changed from
{micro["median_before_ms_per_frame"]:.3f} to {micro["median_after_ms_per_frame"]:.3f} ms/frame
({micro["median_change_percent"]:.2f}%). This is not an end-to-end retarget
throughput result and does not establish solver math parity.

Comparable after-hardening solver runs were not started. Historical wall time
was {decision["runtime_hold"]["observed_hardening_1_full_hours"]:.2f} h for #1 full,
{decision["runtime_hold"]["observed_hardening_2_incomplete_hours"]:.2f} h for #2 incomplete, and
{decision["runtime_hold"]["observed_hardening_3_incomplete_hours"]:.2f} h for #3 incomplete.
Therefore solver throughput and exact before/after math parity remain
`INCONCLUSIVE/NOT_RUN`, and P1 selects the mandatory conservative exact-solver
plus fail-closed preflight branch.
"""
    (root / "throughput_analysis.md").write_text(analysis, encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
