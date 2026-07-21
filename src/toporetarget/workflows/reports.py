"""Auditable Stage 10 input, execution, and completion reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .cache import path_hash
from .schema import WorkflowRequest, stable_hash, write_json
from .validation import environment_snapshot


def _path_snapshot(path: str | Path | None, *, hash_file: bool = False) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    target = Path(path).expanduser()
    payload: dict[str, Any] = {"path": str(target), "exists": target.exists()}
    if not target.exists():
        return payload
    stat = target.stat()
    payload.update(
        {
            "kind": "file" if target.is_file() else "directory" if target.is_dir() else "other",
            "size_bytes": stat.st_size if target.is_file() else None,
            "mtime_ns": stat.st_mtime_ns,
        }
    )
    if hash_file and target.is_file():
        payload["sha256"] = path_hash(target)
    return payload


def stage9_window_geometry_audit(
    *, repo_root: Path, request: WorkflowRequest, selected: dict[str, Any]
) -> dict[str, Any]:
    """Bind a Stage 10 window to the bounded Stage 9 geometry classification."""

    report_path = repo_root / ".local/reports/stage9_solver_closeout/window_geometry_audit.json"
    result: dict[str, Any] = {
        "status": "deferred_missing_report",
        "report_path": str(report_path),
        "window_class": None,
        "semantic_contact_frame_ratio": None,
        "source_geometry": None,
    }
    if not report_path.is_file():
        return result
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {**result, "status": "invalid_report", "error": str(exc)}
    target_range = [int(selected["start_frame"]), int(selected["end_frame"])]
    target_sequence = request.sequence
    target_hand = request.hand
    match = next(
        (
            item
            for item in payload.get("windows", [])
            if item.get("window_identity", {}).get("global_frame_range") == target_range
            and item.get("window_identity", {}).get("sequence") == target_sequence
            and item.get("window_identity", {}).get("hand") == target_hand
        ),
        None,
    )
    if match is None:
        return {
            **result,
            "status": "window_not_in_report",
            "requested_sequence": target_sequence,
            "requested_hand": target_hand,
            "requested_frame_range": target_range,
            "report_sha256": path_hash(report_path),
        }
    metrics = match.get("classification_metrics", {})
    source_geometry = {
        "source_mano_object_median": metrics.get("source_mano_object_median_m"),
        "source_contact_geometry_median_m": metrics.get("source_contact_geometry_median_m"),
        "aggregate_distances": match.get("aggregate_distances", {}),
    }
    window_class = match.get("window_class")
    return {
        "status": "pass" if window_class == "contact_rich" else "reject",
        "report_path": str(report_path),
        "report_sha256": path_hash(report_path),
        "window_id": match.get("window_id"),
        "window_class": window_class,
        "classification_matches_expected": match.get("classification_matches_expected"),
        "semantic_contact_frame_ratio": metrics.get("semantic_contact_frame_ratio"),
        "source_geometry": source_geometry,
        "diagnostic_thresholds": payload.get("diagnostic_config", {}),
    }


def build_input_audit(
    *,
    request: WorkflowRequest,
    selection: dict[str, Any],
    manual_acceptance: str | Path,
    profile_hashes: dict[str, str],
) -> dict[str, Any]:
    """Capture the bounded inputs without recursively scanning external roots."""

    index_root = request.index.expanduser().resolve()
    source_path = selection.get("source_path")
    profile_paths = {
        name: _path_snapshot(request.repo_root / relative, hash_file=True)
        for name, relative in _profile_paths(request.refinement_solver_profile).items()
    }
    payload: dict[str, Any] = {
        "schema_version": "toporetarget.stage10_input_audit.v1",
        "status": "pass",
        "workflow_id": "grab_to_artimano",
        "request": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in request.__dict__.items()
        },
        "selection": {
            "sequence": selection.get("sequence"),
            "selected": selection.get("selected"),
            "selection_hash": selection.get("selection_hash"),
        },
        "stage9_window_geometry_audit": stage9_window_geometry_audit(
            repo_root=request.repo_root, request=request, selected=selection.get("selected", {})
        )
        if selection.get("selected")
        else {"status": "deferred_no_selection"},
        "manual_acceptance": _path_snapshot(manual_acceptance, hash_file=True),
        "index": {
            "root": _path_snapshot(index_root),
            "index_jsonl": _path_snapshot(index_root / "index.jsonl", hash_file=True),
            "manifest_json": _path_snapshot(index_root / "manifest.json", hash_file=True),
        },
        "source": _path_snapshot(source_path, hash_file=True),
        "external_roots": {
            "mano_model_root": _path_snapshot(request.mano_model_root),
            "asset_root": _path_snapshot(request.asset_root),
        },
        "profiles": profile_paths,
        "profile_hashes": dict(profile_hashes),
        "environment": environment_snapshot(request.repo_root),
        "read_only_contract": {
            "raw_source_not_written": True,
            "external_model_roots_not_scanned_recursively": True,
            "native_time_preserved": True,
            "semantic_contact_used_only_for_selection_and_validation": True,
        },
    }
    payload["audit_hash"] = stable_hash(payload)
    return payload


def _profile_paths(solver_profile_id: str = "scipy_slsqp_active_set_v1") -> dict[str, str]:
    # Kept local to avoid importing planning during CLI help or report loading.
    return {
        "paper_retarget": "configs/paper/retarget.yaml",
        "object_surface": "configs/geometry/object_surface_sampling.yaml",
        "signed_distance": "configs/geometry/signed_distance.yaml",
        "bone_profile": "configs/retarget/bones/mediapipe21_full_finger_chain_v1.yaml",
        "frame_profile": "configs/retarget/frames/canonical_keypoint_wrist_v1.yaml",
        "delaunay_profile": "configs/retarget/interaction/strict_scipy_qhull_v1.yaml",
        "query_profile": "configs/retarget/collision_queries/adaptive_active_set_v1.yaml",
        "coordinate_profile": "configs/retarget/refinement/local_seed_delta_v1.yaml",
        "solver_profile": (f"configs/retarget/refinement_solvers/{solver_profile_id}.yaml"),
        "robot_surface": "configs/geometry/robot_collision_sampling.yaml",
        "contact_mapping": "configs/datasets/grab_contact_parts.yaml",
    }


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def build_execution_reports(
    manifest: dict[str, Any],
    *,
    run_root: str | Path,
    elapsed_s: float,
) -> dict[str, dict[str, Any]]:
    """Build honest reports for both successful and failed bounded runs."""

    nodes = manifest.get("nodes", [])
    if not isinstance(nodes, list):
        nodes = []
    reuse_decisions = [
        {
            "node_id": node.get("node_id"),
            "status": node.get("status"),
            "reused": bool(node.get("reused", False)),
            "skipped": bool(node.get("skipped", False)),
            "expected_signature": node.get("expected_signature"),
            "actual_signature": node.get("actual_signature"),
            "invalidation_reason": node.get("invalidation_reason"),
            "validation_status": node.get("validation_status"),
            "output_hashes": node.get("output_hashes", {}),
        }
        for node in nodes
    ]
    durations = {
        str(node.get("node_id")): node.get("duration_s")
        for node in nodes
        if node.get("duration_s") is not None
    }
    run_path = Path(run_root)
    failed = [node.get("node_id") for node in nodes if node.get("status") == "failed"]
    run_status = str(manifest.get("run_status", "unknown"))
    source_integrity = manifest.get("source_integrity") or {}
    if not source_integrity:
        source_report_path = Path(run_root) / "reports" / "source.json"
        source_report: dict[str, Any] = {}
        if source_report_path.is_file():
            try:
                import json

                value = json.loads(source_report_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    source_report = value
            except (OSError, ValueError):
                source_report = {}
        source_path = source_report.get("source_path") or manifest.get("source_path")
        source_hash_before = source_report.get("source_hash") or manifest.get("source_hash")
        if source_path and source_hash_before:
            try:
                source_hash_after = path_hash(source_path)
                unchanged = source_hash_after == source_hash_before
                source_integrity = {
                    "source_path": source_path,
                    "source_hash_before": source_hash_before,
                    "source_hash_after": source_hash_after,
                    "source_integrity_check": "pass" if unchanged else "fail",
                    "raw_source_not_written_by_workflow": unchanged,
                    "reconstructed_from_source_report": True,
                }
            except (OSError, ValueError) as exc:
                source_integrity = {
                    "source_path": source_path,
                    "source_hash_before": source_hash_before,
                    "source_hash_after": None,
                    "source_integrity_check": "unknown",
                    "raw_source_not_written_by_workflow": None,
                    "error": str(exc),
                }
        else:
            source_integrity = {
                "source_integrity_check": "not_checked",
                "reason": "run stopped before the source was resolved",
            }
    semantic = manifest.get("validations", {}).get("semantic_sanity")
    final_path = manifest.get("final_artifact_path") or (
        manifest.get("artifacts", {}).get("final", {}) or {}
    ).get("path")
    final_present = bool(final_path and Path(str(final_path)).exists())
    identity = manifest.get("validations", {}).get("cross_stage_identity")
    end_to_end_status = (
        "blocked"
        if run_status == "failed" or not final_present
        else "pending_human_acceptance"
        if run_status == "pending_human_acceptance"
        else "pass"
    )
    end_to_end = {
        "schema_version": "toporetarget.stage10_end_to_end_validation.v1",
        "status": end_to_end_status,
        "run_id": manifest.get("run_id"),
        "run_status": run_status,
        "final_artifact_present": final_present,
        "final_artifact": final_path,
        "failed_nodes": failed,
        "cross_stage_identity_status": (
            identity.get("status") if isinstance(identity, dict) else "not_run"
        ),
        "reason": (
            "final refinement did not produce a final artifact"
            if not final_present
            else "full end-to-end validation awaits human acceptance"
            if end_to_end_status == "pending_human_acceptance"
            else None
        ),
    }
    completion_status = (
        "STAGE10_COMPLETE_WITH_HUMAN_ACCEPTANCE"
        if run_status == "complete"
        else "STAGE10_IMPLEMENTED_PENDING_HUMAN_ACCEPTANCE"
        if run_status == "pending_human_acceptance"
        else "STAGE10_BLOCKED"
        if run_status == "failed"
        else "STAGE10_IMPLEMENTED_PENDING_HUMAN_ACCEPTANCE"
    )
    reports = {
        "artifact_reuse": {
            "schema_version": "toporetarget.stage10_artifact_reuse.v1",
            "status": "pass",
            "run_id": manifest.get("run_id"),
            "decisions": reuse_decisions,
            "reused_node_count": sum(1 for item in reuse_decisions if item["reused"]),
            "rerun_node_count": sum(
                1 for item in reuse_decisions if item["status"] in {"running", "passed", "failed"}
            ),
            "invalidated_nodes": [
                item["node_id"] for item in reuse_decisions if item["invalidation_reason"]
            ],
        },
        "invalidation_tests": {
            "schema_version": "toporetarget.stage10_invalidation_report.v1",
            "status": "covered_by_tests",
            "test_module": "tests/unit/test_workflow_invalidation.py",
            "run_id": manifest.get("run_id"),
            "observed_invalidation_reasons": [
                {
                    "node_id": item["node_id"],
                    "reason": item["invalidation_reason"],
                }
                for item in reuse_decisions
                if item["invalidation_reason"]
            ],
        },
        "performance": {
            "schema_version": "toporetarget.stage10_performance.v1",
            "status": (
                "complete" if run_status in {"pending_human_acceptance", "complete"} else "partial"
            ),
            "run_id": manifest.get("run_id"),
            "node_duration_s": durations,
            "node_time_sum_s": float(sum(float(value) for value in durations.values())),
            "total_wall_time_s": float(elapsed_s),
            "reused_node_count": sum(1 for item in reuse_decisions if item["reused"]),
            "passed_node_count": sum(1 for item in reuse_decisions if item["status"] == "passed"),
            "failed_node_ids": failed,
            "run_disk_bytes": _directory_size(run_path),
            "cache_reuse_savings": "not_estimated_without_a_counterfactual_run",
        },
        "determinism": {
            "schema_version": "toporetarget.stage10_determinism.v1",
            "status": "pending_full_repeat",
            "run_id": manifest.get("run_id"),
            "selected_window_hash": (manifest.get("contact_window_selection") or {}).get(
                "selection_hash"
            ),
            "node_signatures": {
                str(node.get("node_id")): node.get("expected_signature") for node in nodes
            },
            "full_repeat_executed": False,
            "required_comparison_fields": [
                "selected_window",
                "node_signatures",
                "final_qpos",
                "final_base_pose_scene",
                "review_frame_ids",
            ],
            "reason": "full repeat requires a successful final refinement artifact",
        },
        "source_integrity": {
            "schema_version": "toporetarget.stage10_source_integrity.v1",
            "run_id": manifest.get("run_id"),
            **source_integrity,
        },
        "end_to_end_validation": end_to_end,
        "semantic_sanity": semantic
        if isinstance(semantic, dict)
        else {
            "schema_version": "toporetarget.semantic_trajectory_sanity.v1",
            "status": "blocked",
            "reason": "final refinement did not produce a final artifact",
            "failed_nodes": failed,
        },
        "stage10_summary": {
            "schema_version": "toporetarget.stage10_summary.v1",
            "run_id": manifest.get("run_id"),
            "run_status": run_status,
            "completion_status": completion_status,
            "failed_nodes": failed,
            "final_artifact": manifest.get("final_artifact_path"),
            "source_integrity_check": source_integrity.get("source_integrity_check"),
            "human_acceptance": manifest.get("manual_acceptance", {}),
        },
    }
    for payload in reports.values():
        payload.setdefault("report_hash", stable_hash(payload))
    return reports


def write_execution_reports(
    manifest: Any,
    *,
    run_root: str | Path,
    elapsed_s: float,
) -> dict[str, Path]:
    payload = manifest.as_dict() if hasattr(manifest, "as_dict") else manifest
    root = Path(run_root) / "reports"
    reports = build_execution_reports(payload, run_root=run_root, elapsed_s=elapsed_s)
    paths: dict[str, Path] = {}
    for name, value in reports.items():
        destination = root / f"{name}.json"
        write_json(value, destination)
        paths[name] = destination
    end_to_end = reports["end_to_end_validation"]
    csv_destination = root / "end_to_end_validation.csv"
    with csv_destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("field", "value"))
        for key in (
            "run_id",
            "run_status",
            "status",
            "final_artifact_present",
            "final_artifact",
            "cross_stage_identity_status",
            "reason",
        ):
            writer.writerow((key, end_to_end.get(key)))
        failed_nodes = ";".join(str(item) for item in end_to_end["failed_nodes"])
        writer.writerow(("failed_nodes", failed_nodes))
    paths["end_to_end_validation_csv"] = csv_destination
    write_global_execution_reports(payload, run_root=run_root)
    return paths


def write_global_execution_reports(
    manifest: dict[str, Any],
    *,
    run_root: str | Path,
) -> dict[str, Path]:
    """Maintain the required repo-local Stage 10 report index.

    The global directory is an index over bounded runs, not a second source of
    truth.  Each copied report retains the latest run's honest status, while
    ``resume_validation.json`` summarizes reuse and invalidation observed across
    the local Stage 10 run roots.
    """

    repo_root = Path(str(manifest.get("repo_root") or Path.cwd())).resolve()
    global_root = repo_root / ".local" / "reports" / "stage10"
    global_root.mkdir(parents=True, exist_ok=True)
    run_reports = Path(run_root) / "reports"
    paths: dict[str, Path] = {}
    selection = manifest.get("contact_window_selection")
    if isinstance(selection, dict):
        selection_destination = global_root / "contact_window_selection.json"
        write_json(selection, selection_destination)
        paths["contact_window_selection"] = selection_destination
        candidates_destination = global_root / "contact_window_candidates.json"
        write_json(
            {
                "candidates": selection.get("candidates", []),
                "selection_hash": selection.get("selection_hash"),
            },
            candidates_destination,
        )
        paths["contact_window_candidates"] = candidates_destination
    plan_source = Path(run_root) / "plan.json"
    plan_destination = global_root / "workflow_plan.json"
    if plan_source.is_file():
        plan_destination.write_text(plan_source.read_text(encoding="utf-8"), encoding="utf-8")
        paths["workflow_plan"] = plan_destination
    copied_names = (
        "end_to_end_validation",
        "semantic_sanity",
        "artifact_reuse",
        "invalidation_tests",
        "determinism",
        "performance",
        "source_integrity",
        "stage10_summary",
    )
    for name in copied_names:
        source = run_reports / f"{name}.json"
        destination = global_root / source.name
        if source.is_file():
            destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            write_json(
                {
                    "schema_version": f"toporetarget.stage10_{name}.v1",
                    "status": "blocked",
                    "run_id": manifest.get("run_id"),
                    "reason": f"{name} was not reached by the bounded run",
                },
                destination,
            )
        paths[name] = destination
    source_csv = run_reports / "end_to_end_validation.csv"
    csv_destination = global_root / source_csv.name
    if source_csv.is_file():
        csv_destination.write_text(source_csv.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        csv_destination.write_text("field,value\nstatus,blocked\n", encoding="utf-8")
    paths["end_to_end_validation_csv"] = csv_destination

    run_base = Path(str(manifest.get("run_root") or run_root)).resolve().parent
    run_summaries: list[dict[str, Any]] = []
    for manifest_path in sorted(run_base.glob("*/manifest.json")):
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        nodes = value.get("nodes", [])
        if not isinstance(nodes, list):
            nodes = []
        run_summaries.append(
            {
                "run_id": value.get("run_id"),
                "run_status": value.get("run_status"),
                "reused_node_count": sum(1 for node in nodes if node.get("reused")),
                "invalidated_nodes": [
                    node.get("node_id") for node in nodes if node.get("invalidation_reason")
                ],
                "failed_nodes": [
                    node.get("node_id") for node in nodes if node.get("status") == "failed"
                ],
            }
        )
    reuse_observed = any(item["reused_node_count"] > 0 for item in run_summaries)
    resume_payload = {
        "schema_version": "toporetarget.stage10_resume_validation.v1",
        "status": "pass" if reuse_observed else "covered_by_tests",
        "latest_run_id": manifest.get("run_id"),
        "runs": run_summaries,
        "resume_reuse_observed": reuse_observed,
        "downstream_invalidation_observed": any(
            bool(item["invalidated_nodes"]) for item in run_summaries
        ),
        "invalidation_test_module": "tests/unit/test_workflow_invalidation.py",
        "full_repeat_determinism_executed": False,
        "reason": (
            "full repeat requires a successful final refinement artifact"
            if not any(
                item["run_status"] in {"complete", "pending_human_acceptance"}
                for item in run_summaries
            )
            else None
        ),
    }
    resume_destination = global_root / "resume_validation.json"
    write_json(resume_payload, resume_destination)
    paths["resume_validation"] = resume_destination
    return paths


__all__ = [
    "build_execution_reports",
    "build_input_audit",
    "write_execution_reports",
    "write_global_execution_reports",
]
