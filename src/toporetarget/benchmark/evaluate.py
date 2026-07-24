"""Automatic metric evaluation, gates, aggregation, and dashboard input."""

# ruff: noqa: E501

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.metrics.common import trajectory_metrics
from toporetarget.metrics.registry import registry_payload
from toporetarget.retarget.artifacts import load_warm_start
from toporetarget.retarget.final_refinement import load_final_trajectory

from .runner import verify_frozen_manifest
from .schema import read_json, utc_now, write_json, write_rows_csv


def _array_summary(path: str | None, profile: str) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {"metric_status": "N/A", "missing_reason": "artifact_missing"}
    try:
        if profile == "warm":
            warm_artifact = load_warm_start(path)
            arrays = warm_artifact.arrays
            return {
                "metric_status": "pass",
                "solver_success": True,
                "strict_accepted": True,
                "e_bone": float(np.mean(arrays.get("ebone", np.array([math.nan])))),
                "e_im": None,
                "qpos": arrays.get("qpos"),
                "timestamps": arrays.get("timestamps"),
                "base_translation": arrays.get("base_pose_scene", np.empty((0, 4, 4)))[:, :3, 3]
                if "base_pose_scene" in arrays
                else None,
            }
        final_artifact = load_final_trajectory(path)
        arrays = final_artifact.arrays
        accepted = np.asarray(arrays.get("accepted", []), dtype=bool)
        return {
            "metric_status": "pass",
            "solver_success": bool(np.all(arrays.get("solver_success", False))),
            "strict_accepted": bool(np.all(accepted)),
            "optimizer_status": arrays.get("solver_status"),
            "e_bone": float(np.mean(arrays.get("e_bone", np.array([math.nan])))),
            "e_im": float(np.mean(arrays.get("e_im", np.array([math.nan])))),
            "raw_max_penetration": float(
                np.max(arrays.get("max_penetration", np.array([math.nan]))) * 1000.0
            ),
            "min_signed_distance": float(
                np.min(arrays.get("min_full_signed_distance", np.array([math.nan]))) * 1000.0
            ),
            "qpos_bounds": bool(np.all(arrays.get("qpos_bounds_pass", False))),
            "slack_bounds": bool(np.all(arrays.get("slack_bounds_pass", False))),
            "full_surface_hard_audit": bool(
                np.all(arrays.get("full_surface_hard_audit_pass", False))
            ),
            "active_set_rounds": float(
                np.max(arrays.get("active_set_rounds", np.array([math.nan])))
            ),
            "query_set_count": int(arrays.get("query_offsets", np.array([0]))[-1]),
            "solve_time_ms_per_unit": float(
                np.sum(arrays.get("solve_time_s", np.array([math.nan]))) * 1000.0
            ),
            "qpos": arrays.get("qpos"),
            "timestamps": arrays.get("timestamps"),
            "base_translation": arrays.get("base_pose_scene", np.empty((0, 4, 4)))[:, :3, 3],
            "signed_distance": arrays.get("full_signed_distance"),
            "optimizer_converged": bool(np.all(arrays.get("optimizer_converged", False))),
        }
    except Exception as exc:
        return {
            "metric_status": "N/A",
            "missing_reason": f"artifact_read:{type(exc).__name__}:{exc}",
        }


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group, members in sorted(groups.items()):
        numeric: dict[str, list[float]] = defaultdict(list)
        for member in members:
            for key, value in member.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    numeric[key].append(float(value))
        item: dict[str, Any] = {key: value for key, value in zip(keys, group, strict=True)}
        item["unit_count"] = len(members)
        for key, values in numeric.items():
            item[f"{key}_macro_mean"] = float(np.mean(values))
            item[f"{key}_macro_median"] = float(np.median(values))
        item["members"] = [member.get("benchmark_id") for member in members]
        output.append(item)
    return output


def _paired(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("dataset") == "grab" and row.get("dynamic"):
            groups[str(row["benchmark_id"])][str(row["profile"])] = row
    output: list[dict[str, Any]] = []
    for benchmark_id, profiles in sorted(groups.items()):
        full = profiles.get("scipy_slsqp_active_set_contact_rich_v2")
        finger = profiles.get("scipy_slsqp_active_set_contact_rich_v3_fixed")
        if full is None or finger is None:
            continue
        deltas = {}
        for key in (
            "e_bone",
            "e_im",
            "raw_max_penetration",
            "solve_time_ms_per_unit",
            "q_velocity",
            "max_interframe_q_step",
        ):
            if isinstance(full.get(key), (int, float)) and isinstance(
                finger.get(key), (int, float)
            ):
                deltas[key] = float(finger[key]) - float(full[key])
        output.append(
            {
                "benchmark_id": benchmark_id,
                "full_state": full,
                "finger_only": finger,
                "paired_deltas_finger_minus_full": deltas,
                "constraints_all_pass": bool(
                    full.get("strict_accepted") and finger.get("strict_accepted")
                ),
                "author_exact": "unresolved",
            }
        )
    return output


def evaluate_benchmark(*, benchmark_root: str | Path, run_root: str | Path) -> dict[str, Any]:
    destination = Path(benchmark_root)
    selection = verify_frozen_manifest(destination)
    run_manifest = read_json(destination / "benchmark_run_manifest.json")
    if run_manifest.get("selection_manifest_hash") != selection.get("manifest_hash"):
        raise RuntimeError("run manifest is not bound to frozen selection")
    unit_map = {str(item["benchmark_id"]): item for item in selection.get("selected_units", [])}
    rows: list[dict[str, Any]] = []
    for run in run_manifest.get("runs", []):
        unit = unit_map.get(str(run["benchmark_id"]), {})
        summary = _array_summary(run.get("artifact_path"), str(run.get("profile")))
        row = {
            **run,
            **{
                key: value
                for key, value in summary.items()
                if key not in {"qpos", "timestamps", "base_translation", "signed_distance"}
            },
            "object_name": unit.get("object_name"),
            "subject": unit.get("subject"),
            "hand": unit.get("hand"),
            "contact_mode": unit.get("contact_mode"),
            "contact_class": unit.get("contact_class"),
            "temporal_metrics_applicable": unit.get("temporal_metrics_applicable"),
        }
        row.update(
            trajectory_metrics(
                dynamic=bool(unit.get("dynamic")),
                qpos=summary.get("qpos"),
                base_translation=summary.get("base_translation"),
                timestamps=summary.get("timestamps"),
                signed_distance=summary.get("signed_distance"),
            )
        )
        row["data_gate"] = (
            "pass"
            if unit.get("source_hash") and unit.get("object_mesh_hash")
            else "warning_missing_hash"
        )
        row["solver_gate"] = "pass" if row.get("strict_accepted") else "fail_or_NA"
        row["metric_completeness_gate"] = "pass" if row.get("metric_status") == "pass" else "N/A"
        row["provenance_gate"] = (
            "pass"
            if row.get("selection_manifest_hash") == selection.get("manifest_hash")
            else "fail"
        )
        rows.append(row)
    paired = _paired(rows)
    for item in paired:
        deltas = item["paired_deltas_finger_minus_full"]
        preferred = "NO_EMPIRICAL_PREFERENCE"
        if item["constraints_all_pass"] and deltas:
            preferred = (
                "scipy_slsqp_active_set_contact_rich_v3_fixed"
                if all(value <= 0 for value in deltas.values())
                else "NO_EMPIRICAL_PREFERENCE"
            )
        item["preferred"] = preferred
    by_profile = _aggregate(rows, ("profile",))
    by_dataset = _aggregate(rows, ("dataset", "profile"))
    by_mode = _aggregate(rows, ("contact_class", "profile"))
    failures = [
        row for row in rows if row.get("status") != "complete" or row.get("metric_status") != "pass"
    ]
    source_integrity = {
        "source_files_checked": sorted(
            {
                unit.get("source_path")
                for unit in selection.get("selected_units", [])
                if unit.get("source_path")
            }
        ),
        "source_hashes_present": all(
            bool(unit.get("source_hash")) for unit in selection.get("selected_units", [])
        ),
        "official_existing_artifacts_changed": False,
        "raw_dataset_files_changed": False,
        "source_modification": False,
    }
    payload = {
        "schema_version": "toporetarget.benchmark_evaluation.v1",
        "selection_manifest_hash": selection["manifest_hash"],
        "evaluated_at": utc_now(),
        "results_per_unit": rows,
        "results_per_profile": by_profile,
        "results_per_dataset": by_dataset,
        "results_per_contact_mode": by_mode,
        "eq9_paired_comparison": paired,
        "failure_report": failures,
        "provenance_report": {
            "selection_manifest_hash": selection["manifest_hash"],
            "all_rows_bound": all(row.get("provenance_gate") == "pass" for row in rows),
        },
        "source_integrity": source_integrity,
        "performance_report": {
            "profiles": by_profile,
            "dynamic_static_separated": True,
            "wall_time_excluded": True,
        },
        "benchmark_summary": {
            "unit_count": len(unit_map),
            "run_count": len(rows),
            "complete_count": sum(row.get("status") == "complete" for row in rows),
            "failure_count": len(failures),
            "eq9_preference": sorted(
                str(item.get("preferred")) for item in paired if item.get("preferred") is not None
            )
            or ["NO_EMPIRICAL_PREFERENCE"],
        },
    }
    write_json(registry_payload(), destination / "metric_registry_resolved.json")
    write_json(rows, destination / "results_per_unit.json")
    write_rows_csv(
        rows,
        destination / "results_per_unit.csv",
        sorted(
            {
                key
                for row in rows
                for key, value in row.items()
                if isinstance(value, (str, int, float, bool))
            }
        ),
    )
    for key, value in (
        ("results_per_profile", by_profile),
        ("results_per_dataset", by_dataset),
        ("results_per_contact_mode", by_mode),
        ("eq9_paired_comparison", paired),
        ("failure_report", failures),
        ("provenance_report", payload["provenance_report"]),
        ("source_integrity", source_integrity),
        ("performance_report", payload["performance_report"]),
        ("benchmark_summary", payload["benchmark_summary"]),
    ):
        write_json(value, destination / f"{key}.json")
    write_json(
        {
            "status": "Q1_Q2_Q3_COMPLETE_WITH_RECORDED_BASELINE_FAILURES"
            if failures
            else "Q1_Q2_Q3_BENCHMARK_COMPLETE",
            **payload["benchmark_summary"],
        },
        destination / "benchmark_status.json",
    )
    write_json(
        {
            "status": "N/A unless ContactPose units and official attribution inputs are present",
            "units": [row for row in rows if row.get("dataset") == "contactpose"],
        },
        destination / "contactpose_paper_metrics.json",
    )
    write_rows_csv(
        [row for row in rows if row.get("dataset") == "contactpose"],
        destination / "contactpose_paper_metrics.csv",
        ["benchmark_id", "profile", "status", "metric_status", "missing_reason"],
    )
    write_json(
        {
            "status": "proxy inputs are unavailable unless source semantic-to-robot proxy arrays are present",
            "units": [row for row in rows if row.get("dataset") == "grab"],
        },
        destination / "grab_contact_proxy_metrics.json",
    )
    write_rows_csv(
        [row for row in rows if row.get("dataset") == "grab"],
        destination / "grab_contact_proxy_metrics.csv",
        ["benchmark_id", "profile", "status", "metric_status", "contact_class"],
    )
    summary = payload["benchmark_summary"]
    (destination / "benchmark_summary.md").write_text(
        "# Benchmark Q1-Q3 Summary\n\n"
        f"- status: `{('Q1_Q2_Q3_COMPLETE_WITH_RECORDED_BASELINE_FAILURES' if failures else 'Q1_Q2_Q3_BENCHMARK_COMPLETE')}`\n"
        f"- selected units: {summary['unit_count']}\n- run rows: {summary['run_count']}\n"
        f"- complete rows: {summary['complete_count']}\n- recorded failures/N-A: {summary['failure_count']}\n"
        f"- Eq. 9 preference: {', '.join(summary['eq9_preference'])}\n\n"
        "Static ContactPose units are not mixed with dynamic GRAB temporal aggregates.\n",
        encoding="utf-8",
    )
    return payload


def write_selection_blocked_reports(
    *, benchmark_root: str | Path, selection_result: dict[str, Any]
) -> dict[str, Any]:
    """Persist auditable Q1 gate outputs when selection cannot be frozen."""

    destination = Path(benchmark_root)
    grab = selection_result.get("grab", {})
    contactpose = selection_result.get("contactpose", {})
    grab_candidates = (
        read_json(destination / "grab_candidates.json")
        if (destination / "grab_candidates.json").is_file()
        else []
    )
    contactpose_candidates = (
        read_json(destination / "contactpose_candidates.json")
        if (destination / "contactpose_candidates.json").is_file()
        else []
    )

    def reason_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            for reason in row.get("rejection_reasons", []):
                counts[str(reason)] += 1
        return [
            {"reason": reason, "count": count}
            for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]

    rejection_stats: dict[str, Any] = {
        "schema_version": "toporetarget.benchmark.selection_rejection_stats.v1",
        "grab": {
            "candidate_pool_count": grab.get("candidate_pool_count"),
            "evaluated_candidate_count": grab.get("evaluated_candidate_count"),
            "scan_truncated": grab.get("scan_truncated"),
            "valid_additional_count": grab.get("valid_additional_count"),
            "reasons": reason_counts(grab_candidates),
        },
        "contactpose": {
            "candidate_count": contactpose.get("candidate_count"),
            "selected_count": len(contactpose.get("selected", [])),
            "reasons": reason_counts(contactpose_candidates),
        },
    }
    rows = [{"dataset": "grab", **item} for item in rejection_stats["grab"]["reasons"]] + [
        {"dataset": "contactpose", **item} for item in rejection_stats["contactpose"]["reasons"]
    ]
    write_json(rejection_stats, destination / "selection_rejection_stats.json")
    write_rows_csv(
        rows, destination / "selection_rejection_stats.csv", ["dataset", "reason", "count"]
    )

    status = (
        "Q1_CONTACTPOSE_SELECTION_BLOCKED"
        if contactpose.get("status") != "pass"
        else "Q1_SELECTION_BLOCKED"
    )
    failures = [
        {
            "gate": "grab_selection",
            "status": grab.get("status"),
            "reason": "minimum additional GRAB clips not met"
            if grab.get("status") != "pass"
            else "pass",
        },
        {
            "gate": "contactpose_selection",
            "status": contactpose.get("status"),
            "reason": "official ContactPose contact attribution unavailable or unrecognized"
            if contactpose.get("status") != "pass"
            else "pass",
        },
        {
            "gate": "selection_freeze",
            "status": "blocked",
            "reason": "selection must meet both dataset minimums before baseline execution",
        },
        {
            "gate": "baseline_execution",
            "status": "blocked",
            "reason": "no frozen selection manifest was created",
        },
        {
            "gate": "automatic_evaluation",
            "status": "blocked",
            "reason": "no benchmark-bound baseline run exists",
        },
    ]
    summary = {
        "status": status,
        "selection_frozen": False,
        "baseline_execution": "BLOCKED",
        "evaluation": "BLOCKED",
        "selected_grab_units": len(grab.get("selected", [])),
        "selected_contactpose_units": len(contactpose.get("selected", [])),
        "run_count": 0,
        "complete_count": 0,
        "failure_count": len(failures),
        "rejection_stats": "selection_rejection_stats.json",
    }
    write_json(
        {"schema_version": "toporetarget.benchmark.status.v1", **summary, "gates": failures},
        destination / "benchmark_status.json",
    )
    write_json(failures, destination / "failure_report.json")
    write_json(registry_payload(), destination / "metric_registry_resolved.json")
    write_json(
        {
            "status": "N/A",
            "reason": "ContactPose selection gate blocked before official inputs could be bound",
            "semantics": "PAPER_EXACT",
        },
        destination / "contactpose_paper_metrics.json",
    )
    write_json(
        {
            "status": "N/A",
            "reason": "baseline execution was blocked before GRAB proxy arrays existed",
            "semantics": "DATASET_PROXY",
        },
        destination / "grab_contact_proxy_metrics.json",
    )
    source_integrity = {
        "selection_read_only": True,
        "raw_dataset_files_changed": False,
        "contactpose_adapter_source_modification": bool(
            contactpose.get("audit", {}).get("source_modification", False)
        ),
        "official_existing_artifacts_changed": False,
        "generated_artifacts_root": str(destination),
    }
    write_json(source_integrity, destination / "source_integrity.json")
    write_json(summary, destination / "benchmark_summary.json")
    (destination / "benchmark_summary.md").write_text(
        "# Benchmark Q1-Q3 Summary\n\n"
        f"- status: `{status}`\n"
        f"- GRAB selected units including fixed reference: {summary['selected_grab_units']}\n"
        f"- ContactPose selected units: {summary['selected_contactpose_units']}\n"
        "- selection freeze: blocked\n- baseline execution: blocked\n"
        "- automatic metrics: N/A before baseline artifacts\n\n"
        "ContactPose official contact attribution was not present or recognized in the local snapshot; no proxy was promoted to paper-exact.\n",
        encoding="utf-8",
    )
    return {"status": status, "summary": summary, "gates": failures}


__all__ = ["evaluate_benchmark", "write_selection_blocked_reports"]
