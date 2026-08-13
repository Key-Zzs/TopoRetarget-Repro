#!/usr/bin/env python3
# ruff: noqa: E402
"""Run the frozen Stage 12 dataset-adapter validation matrix.

The command is intentionally sequence-scoped.  It reads the existing NAS
payload, writes only under ``.local/experiments/stage12_dataset_validation``,
and keeps the shared Wuji target/profile fixed for every dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

# Set the final-job CPU budget before NumPy/SciPy/Torch are imported.  The
# running legacy workers predate this guard; all newly launched jobs inherit it.
from toporetarget.retarget.final_jobs import (
    HEALTH_GATE_PAUSE_STATE,
    PAUSE_STATE,
    FinalJobPaused,
    FinalRefinementCPUConfig,
    append_heartbeat,
    assert_final_jobs_allowed,
    claim_final_job,
    configure_cpu_runtime,
    paused,
    release_final_job,
)

FINAL_CPU_RUNTIME = configure_cpu_runtime(FinalRefinementCPUConfig.load())

import numpy as np
import yaml

from toporetarget.adapters.datasets import get_dataset_adapter_registry
from toporetarget.cli.retarget import _run_checkpoint_refinement
from toporetarget.contracts.canonical import CanonicalHOIv2, load_canonical_hoi, save_canonical_hoi
from toporetarget.data.adapters.base import FrameRange
from toporetarget.geometry.object_geometry import sample_object_track
from toporetarget.geometry.robot_surface import (
    load_robot_surface_profile,
    sample_robot_collision_surface,
)
from toporetarget.geometry.surface_sampling import load_surface_profile
from toporetarget.quality.html import render_clip_html, smoke_html
from toporetarget.quality.schema import ClipSpec
from toporetarget.retarget.artifacts import load_warm_start, save_warm_start
from toporetarget.retarget.bones import load_bone_profile
from toporetarget.retarget.final_refinement import (
    CollisionQueryProfile,
    RefinementCoordinateProfile,
    RefinementSolverProfile,
    load_final_trajectory,
)
from toporetarget.retarget.frames import load_frame_profile
from toporetarget.retarget.interaction_artifacts import (
    load_interaction_evaluation,
    load_interaction_graph,
    save_interaction_evaluation,
    save_interaction_graph,
)
from toporetarget.retarget.interaction_evaluation import evaluate_interaction_graph
from toporetarget.retarget.interaction_graph import build_source_interaction_graph
from toporetarget.retarget.pipeline import build_warm_start_trajectory, source_cache_hash
from toporetarget.retarget.refinement_performance import RefinementExecutionProfile
from toporetarget.retarget.solver import load_solver_profile
from toporetarget.retarget.static_runtime_policy import (
    STATIC_FRAME_ACCEPTED,
    STATIC_FRAME_ACCEPTED_WITH_RUNTIME_WARNING,
    STATIC_FRAME_GEOMETRY_FAILURE,
    STATIC_FRAME_HARD_RUNTIME_FAILURE,
    STATIC_FRAME_SOLVER_FAILURE,
    classify_runtime_health,
    is_static_single_frame_contract,
)
from toporetarget.robots.registry import get_robot_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "benchmarks" / "stage12_selection.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local" / "experiments" / "stage12_dataset_validation"


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sequence"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _hash_rows(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _selection_rows(config: Path) -> list[dict[str, Any]]:
    values = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    target = values.get("target", {})
    if (
        target.get("robot") != "wuji_hand2_beta1_rh"
        or target.get("profile") != "wuji_continuous_sequential_v1"
    ):
        raise ValueError(
            "Stage 12 selection must remain frozen to the Wuji Hand2 RH target/profile"
        )
    rows = list(values.get("selections", []))
    if len(rows) != 8:
        raise ValueError(f"Stage 12 requires exactly 8 selected trajectories, got {len(rows)}")
    for row in rows:
        frame_range = row.get("frame_range")
        if (
            not isinstance(frame_range, list)
            or len(frame_range) != 2
            or int(frame_range[1]) <= int(frame_range[0])
        ):
            raise ValueError(f"invalid Stage 12 frame_range: {row}")
        if row.get("hand") != "right":
            raise ValueError(f"Stage 12 selection is not right-hand: {row}")
    return rows


def _dataset_manifest(adapter: Any, output_root: Path) -> Path:
    rows = adapter.discover()
    payload = {
        "schema_version": "toporetarget.stage12.dataset_manifest.v1",
        "dataset": adapter.descriptor.name,
        "version": adapter.descriptor.version,
        "source_path": str(adapter.data_root.resolve()),
        "source_index_hash": _hash_rows(rows),
        "hash_scope": "adapter index rows and metadata paths; no frame geometry loaded",
        "license": "unresolved; consult upstream dataset license before redistribution",
        "sequence_count": len(rows),
        "capability": adapter.descriptor.capabilities.as_dict(),
        "provenance": adapter.descriptor.provenance,
        "raw_data_policy": "read_existing_payload_only_no_redownload_no_reextract_no_full_copy",
    }
    path = output_root / adapter.descriptor.name / "dataset_manifest.json"
    _write_json(path, payload)
    return path


def _primary_object(sequence: CanonicalHOIv2, configured_object_id: str | None) -> Any:
    """Resolve the configured object id instead of trusting stale canonical metadata."""

    if configured_object_id is None:
        return sequence.primary_rigid_object()
    return sequence.rigid_object(str(configured_object_id))


def _metrics(
    sequence: CanonicalHOIv2, warm: Any, final: Any | None, runtime_s: float
) -> dict[str, Any]:
    source = np.asarray(
        sequence.hands[0].keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64
    )
    result: dict[str, Any] = {
        "runtime_s_total": float(runtime_s),
        "warm": {
            "ebone_mean": float(np.mean(warm.arrays["ebone"])),
            "ebone_p95": float(np.percentile(warm.arrays["ebone"], 95)),
            "solver_success_count": int(np.count_nonzero(warm.arrays.get("solver_success", []))),
            "frame_count": warm.frame_count,
        },
    }
    if final is None:
        result["final"] = {"status": "not_available"}
        return result
    robot = np.asarray(final.arrays["robot_keypoints_scene"], dtype=np.float64)
    error = np.linalg.norm(robot - source, axis=-1)
    names = [
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
    ]
    groups = [(1, 5), (5, 9), (9, 13), (13, 17), (17, 21)]
    per_finger = {
        name: float(np.sqrt(np.mean(error[:, start:stop] ** 2)))
        for name, (start, stop) in zip(names, groups, strict=True)
    }
    continuity = {
        key: float(np.nanmax(final.arrays[key]))
        for key in (
            "continuity_base_translation_m",
            "continuity_base_rotation_rad",
            "continuity_finger_inf_rad",
            "continuity_excess_keypoint_m",
        )
        if key in final.arrays
    }
    result["final"] = {
        "status": "available",
        "eim_rmse": float(np.sqrt(np.mean(error**2))),
        "eim_mean": float(np.mean(error)),
        "per_finger_rmse": per_finger,
        "ebone_mean": float(np.mean(final.arrays.get("e_bone", np.asarray([np.nan])))),
        "penetration_max_m": float(
            np.max(final.arrays.get("max_penetration", np.asarray([np.nan])))
        ),
        "continuity": continuity,
        "solver_success_count": int(np.count_nonzero(final.arrays.get("solver_success", []))),
        "accepted_count": int(np.count_nonzero(final.arrays.get("accepted", []))),
        "runtime_s_solver": float(np.sum(final.arrays.get("solve_time_s", np.asarray([0.0])))),
    }
    return result


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path.with_suffix(".json"), payload)
    lines = [
        f"# Stage 12 retarget report: {payload['dataset']} / {payload['sequence']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Input: `{payload['input']['canonical']}`",
        "- Retarget robot/profile: "
        f"`{payload['retarget']['robot']}` / `{payload['retarget']['profile']}`",
        f"- q dimension: `{payload['retarget'].get('q_dimension', 'not_available')}`",
        "",
        "## Quality",
        "",
        "```json",
        json.dumps(payload.get("quality", {}), indent=2, sort_keys=True, default=str),
        "```",
        "",
        "## Failure diagnosis",
        "",
        f"- Category: `{payload.get('failure_diagnosis', {}).get('category', 'not_available')}`",
        f"- Detail: {payload.get('failure_diagnosis', {}).get('detail', 'not_available')}",
    ]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report_path(output_root: Path, row: dict[str, Any]) -> Path:
    return (
        output_root
        / str(row["dataset"])
        / _safe(str(row["sequence"]))
        / "metrics"
        / "retarget_report.json"
    )


def _resume_checkpoint(root: Path) -> Path:
    """Prefer the latest resumable Stage-12 checkpoint without replacing it."""

    checkpoints = root / "checkpoints"
    candidates = (
        [
            path
            for path in checkpoints.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        ]
        if checkpoints.is_dir()
        else []
    )
    if not candidates:
        return checkpoints / "final_refinement_fast_exact_v2_r1"

    def key(path: Path) -> tuple[int, float, str]:
        progress = path / "progress.json"
        next_frame = -1
        if progress.is_file():
            next_frame = int(json.loads(progress.read_text(encoding="utf-8")).get("next_frame", -1))
        return (next_frame, path.stat().st_mtime_ns, path.name)

    return max(candidates, key=key)


def _final_output_path(root: Path, checkpoint: Path) -> Path:
    """Reuse historical versioned final artifacts without creating a duplicate lineage."""

    candidates = [
        root / "final" / checkpoint.name / "final_retarget.zarr",
        root / "final" / checkpoint.name.removeprefix("final_refinement_") / "final_retarget.zarr",
        *sorted((root / "final").glob("*/final_retarget.zarr")),
    ]
    return next((path for path in candidates if path.is_dir()), candidates[0])


def _health_gate(
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    static_single_frame: bool,
) -> str | None:
    """Stage-12 stop conditions evaluated from immutable per-frame evidence."""

    elapsed = float(metadata.get("solve_time_s", float("inf")))
    # Torch creates an idle interop pool at import time even after both public
    # limits are set to one.  Raw /proc task count would reject that harmless
    # pool, so check the configured BLAS environment and Torch's effective
    # execution limits instead.  A non-unit limit is a real fail-closed gate.
    numeric_env = FINAL_CPU_RUNTIME["blas_environment"]
    torch_runtime = FINAL_CPU_RUNTIME.get("torch", {})
    if any(value != "1" for value in numeric_env.values()) or (
        torch_runtime.get("available", False)
        and (
            int(torch_runtime.get("threads", 0)) != 1
            or int(torch_runtime.get("interop_threads", 0)) != 1
        )
    ):
        return "worker_thread_oversubscription"
    runtime = classify_runtime_health(
        elapsed_s=elapsed,
        frame_times_s=[float(item.get("solve_time_s", np.nan)) for item in rows],
        static_single_frame=static_single_frame,
    )
    if runtime.terminal_reason is not None:
        return runtime.terminal_reason
    if int(metadata.get("unqueried_soft_violation_count", 0)) != 0:
        return "unqueried_violation"
    if not bool(metadata.get("strict_accepted", False)):
        return "strict_acceptance_failed"
    return None


def _formal_v4_health_gate(
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    first_ten_report: Path | None,
    static_single_frame: bool,
) -> str | None:
    """Apply the v4 formal evidence gates without restarting a qualified job."""

    failure = _health_gate(metadata, rows, static_single_frame=static_single_frame)
    if failure is not None:
        return failure
    diagnostics = dict(metadata.get("diagnostics", {}))
    sign_cache = dict(diagnostics.get("sign_cache", {}))
    if int(diagnostics.get("sign_mismatch_count", 0)) != 0:
        return "sign_mismatch"
    if int(sign_cache.get("false_certified_reuse_count", 0)) != 0:
        return "false_certified_sign_reuse"
    if int(diagnostics.get("full_audit_call_count", 0)) != 1:
        return "full_audit_count_not_exactly_one"
    if len(rows) != 10 or first_ten_report is None:
        return None
    times = np.asarray([float(item["solve_time_s"]) for item in rows], dtype=np.float64)
    passed = bool(
        np.all(np.isfinite(times))
        and float(np.median(times)) <= 15.0
        and float(np.percentile(times, 95)) <= 25.0
        and float(np.max(times)) <= 45.0
    )
    evidence = {
        "schema_version": "toporetarget.stage12.v4.oakink_first_ten.v1",
        "status": "pass" if passed else "fail",
        "accepted_count": int(sum(bool(item.get("strict_accepted", False)) for item in rows)),
        "strict_accepted_count": int(
            sum(bool(item.get("strict_accepted", False)) for item in rows)
        ),
        "full_audit_count_exactly_one": bool(
            all(
                int(dict(item.get("diagnostics", {})).get("full_audit_call_count", 0)) == 1
                for item in rows
            )
        ),
        "sign_mismatch_count": int(
            sum(
                int(dict(item.get("diagnostics", {})).get("sign_mismatch_count", 0))
                for item in rows
            )
        ),
        "false_certified_reuse_count": int(
            sum(
                int(
                    dict(dict(item.get("diagnostics", {})).get("sign_cache", {})).get(
                        "false_certified_reuse_count", 0
                    )
                )
                for item in rows
            )
        ),
        "runtime_s": {
            "median": float(np.median(times)),
            "p95": float(np.percentile(times, 95)),
            "max": float(np.max(times)),
        },
        "continuity_pass": bool(
            all(not item.get("continuity_failure_reasons", []) for item in rows)
        ),
        "checkpoint_prefix": [int(item["local_frame_index"]) for item in rows],
        "continued_in_same_process_after_pass": passed,
    }
    _write_json(first_ten_report, evidence)
    return None if passed else "oakink_first_ten_runtime_gate_failed"


def _link_upstream_artifacts(
    *,
    root: Path,
    paths: dict[str, Path],
    upstream_root: Path | None,
    dataset: str,
    unit: str,
    formal_v4: bool,
) -> None:
    """Reuse immutable upstream artifacts by symlink, never by copying or mutation."""

    if upstream_root is None:
        return
    source_root = upstream_root / dataset / unit
    if not source_root.is_dir():
        raise ValueError(f"upstream Stage-12 selection root is missing: {source_root}")
    relative = {
        "canonical": Path("canonical/canonical_hoi_v2.zarr"),
        "warm": Path("warm/warm_start.zarr"),
        "graph": Path("exports/interaction_graph.zarr"),
        "evaluation": Path("exports/interaction_evaluation.zarr"),
        "object_samples": Path("exports/object_samples.npz"),
        "collision_samples": Path("exports/wuji_collision_samples.npz"),
    }
    compatibility = "reused"
    source_warm = source_root / relative["warm"]
    source_graph = source_root / relative["graph"]
    if formal_v4 and source_warm.is_dir() and source_graph.is_dir():
        upstream_warm = load_warm_start(source_warm)
        upstream_graph = load_interaction_graph(source_graph)
        if upstream_warm.metadata.get("source_cache_hash") != upstream_graph.metadata.get(
            "source_cache_hash"
        ):
            # Legacy v2 artifacts with incompatible source identities are
            # immutable audit inputs, not legal formal-v4 inputs.  Keep the
            # canonical/object sampling payload read-only and rebuild only the
            # incompatible warm/graph/evaluation closure in this new lineage.
            for key in ("warm", "graph", "evaluation"):
                relative.pop(key)
            compatibility = "rebuild_warm_graph_due_to_legacy_source_cache_hash_mismatch"
    reused: dict[str, str] = {}
    for key, suffix in relative.items():
        source = source_root / suffix
        destination = paths[key]
        if not source.exists():
            continue
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError(f"formal lineage refuses to replace existing {key}: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source)
        reused[key] = str(source)
    _write_json(
        root / "manifests" / "upstream_reuse.json",
        {
            "schema_version": "toporetarget.stage12.v4.upstream_reuse.v1",
            "mode": "read_only_symlink",
            "source_root": str(source_root),
            "warm_graph_compatibility": compatibility,
            "reused": reused,
        },
    )


def _aggregate_existing_reports(output_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the final handoff summary from completed sequence reports only."""

    results: list[dict[str, Any]] = []
    for row in rows:
        report_path = _report_path(output_root, row)
        if not report_path.is_file():
            results.append(
                {
                    "dataset": row["dataset"],
                    "sequence": row["sequence"],
                    "status": "missing_report",
                    "report": str(report_path),
                }
            )
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        final = dict(report.get("quality", {}).get("final", {}))
        results.append(
            {
                "dataset": report.get("dataset", row["dataset"]),
                "sequence": report.get("sequence", row["sequence"]),
                "status": report.get("status", "unknown"),
                "report": str(report_path),
                "html": report.get("paths", {}).get("html"),
                "runtime_s_total": report.get("quality", {}).get("runtime_s_total"),
                "eim_rmse": final.get("eim_rmse"),
                "ebone_mean": final.get("ebone_mean"),
                "penetration_max_m": final.get("penetration_max_m"),
                "accepted_count": final.get("accepted_count"),
                "frame_count": report.get("input", {}).get("frame_count"),
                "contact_benchmark_status": report.get("input", {}).get(
                    "contact_benchmark_status", "NOT_AVAILABLE"
                ),
            }
        )
    completed_statuses = {
        "pass",
        "completed_with_solver_failures",
        STATIC_FRAME_ACCEPTED,
        STATIC_FRAME_ACCEPTED_WITH_RUNTIME_WARNING,
    }
    completed = [item for item in results if item["status"] in completed_statuses]
    return {
        "schema_version": "toporetarget.stage12.summary.v1",
        "target": {"robot": "wuji_hand2_beta1_rh", "profile": "wuji_continuous_sequential_v1"},
        "selection_count": len(rows),
        "report_count": len(results) - sum(item["status"] == "missing_report" for item in results),
        "pass_count": sum(item["status"] in completed_statuses for item in results),
        "completed_with_solver_failures": sum(
            item["status"] == "completed_with_solver_failures" for item in results
        ),
        "blocked_count": sum(item["status"] == "blocked" for item in results),
        "paused_count": sum(str(item["status"]).startswith("PAUSED_") for item in results),
        "missing_report_count": sum(item["status"] == "missing_report" for item in results),
        "wuji_completion_rate": float(len(completed) / len(rows)) if rows else 0.0,
        "results": results,
    }


def run_one(
    *,
    repo: Path,
    output_root: Path,
    adapter: Any,
    row: dict[str, Any],
    robot_name: str,
    profile_name: str,
    execution_profile_name: str = "wuji_continuous_sequential_fast_exact_v2",
    mano_model_root: Path | None = None,
    upstream_root: Path | None = None,
    formal_v4: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = str(row["dataset"])
    sequence_id = str(row["sequence"])
    frame_start, frame_stop = (int(row["frame_range"][0]), int(row["frame_range"][1]))
    unit = _safe(sequence_id)
    job_id = f"stage12_{dataset}_{unit}"
    root = output_root / dataset / unit
    checkpoint = _resume_checkpoint(root)
    paths = {
        "canonical": root / "canonical" / "canonical_hoi_v2.zarr",
        "warm": root / "warm" / "warm_start.zarr",
        "graph": root / "exports" / "interaction_graph.zarr",
        "evaluation": root / "exports" / "interaction_evaluation.zarr",
        "object_samples": root / "exports" / "object_samples.npz",
        "collision_samples": root / "exports" / "wuji_collision_samples.npz",
        "final": _final_output_path(root, checkpoint),
        "checkpoint": checkpoint,
        "html": root / "html" / "source_warm_final_wuji.html",
        "progress": root / "metrics" / "final_progress.json",
        "report": root / "metrics" / "retarget_report",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _link_upstream_artifacts(
        root=root,
        paths=paths,
        upstream_root=upstream_root,
        dataset=dataset,
        unit=unit,
        formal_v4=formal_v4,
    )
    payload: dict[str, Any] = {
        "schema_version": "toporetarget.stage12.retarget_report.v1",
        "status": "running",
        "dataset": dataset,
        "sequence": sequence_id,
        "selection": row,
        "input": {
            "data_root": str(adapter.data_root.resolve()),
            "canonical": str(paths["canonical"]),
        },
        "retarget": {"robot": robot_name, "profile": profile_name, "solver_profile": profile_name},
        "paths": {key: str(value) for key, value in paths.items()},
        "final_job_id": job_id,
        "cpu_runtime": FINAL_CPU_RUNTIME,
    }

    def write_progress(phase: str, **detail: Any) -> None:
        """Persist a bounded heartbeat without changing retargeting behavior."""

        _write_json(
            paths["progress"],
            {
                "schema_version": "toporetarget.stage12.final_progress.v1",
                "status": "running",
                "dataset": dataset,
                "sequence": sequence_id,
                "phase": phase,
                "elapsed_s": float(time.perf_counter() - started),
                **detail,
            },
        )

    claimed = False
    try:
        if mano_model_root is not None:
            adapter.mano_model_root = mano_model_root
        if paths["canonical"].is_dir():
            canonical = load_canonical_hoi(paths["canonical"])
            write_progress("reuse_canonical", frame_count=canonical.num_frames)
        else:
            write_progress("load_sequence")
            raw = adapter.load_sequence(
                sequence_id,
                frame_range=FrameRange(frame_start, frame_stop),
                hand=str(row.get("hand", "right")),
                primary_object_id=row.get("primary_object"),
            )
            write_progress("canonicalize")
            canonical = adapter.convert_to_canonical(raw)
            save_canonical_hoi(canonical, paths["canonical"])
        payload["input"].update(
            {
                "source_sequence": canonical.metadata.provenance.source_sequence,
                "source_hash": canonical.metadata.provenance.source_hash,
                "canonical_schema": canonical.metadata.schema_version,
                "frame_count": canonical.num_frames,
                "object_ids": [item.object_id for item in canonical.rigid_objects],
                "contact_benchmark_status": canonical.metadata.metadata.get(
                    "contact_benchmark_status", "NOT_AVAILABLE"
                ),
                "sample_mode": row.get(
                    "sample_mode",
                    row.get(
                        "sample_type",
                        canonical.metadata.metadata.get(
                            "sample_mode",
                            canonical.metadata.metadata.get("sample_type"),
                        ),
                    ),
                ),
                "temporal_metrics_applicable": row.get(
                    "temporal_metrics_applicable",
                    canonical.metadata.metadata.get("temporal_metrics_applicable"),
                ),
            }
        )
        static_single_frame = is_static_single_frame_contract(
            row,
            canonical.metadata.metadata,
            frame_count=canonical.num_frames,
        )
        payload["input"]["runtime_sample_kind"] = (
            "static_single_frame" if static_single_frame else "dynamic_trajectory"
        )
        validation = adapter.validate(canonical)
        if validation["errors"]:
            raise ValueError(f"canonical validation failed: {validation['errors']}")

        write_progress("warm_start", frame_count=canonical.num_frames)
        robot = get_robot_registry(repo_root=repo).load(robot_name)
        frame_profile = load_frame_profile(
            "canonical_keypoint_wrist_v1", config_root=repo / "configs" / "retarget" / "frames"
        )
        bone_profile = load_bone_profile(
            "mediapipe21_full_finger_chain_v1", config_root=repo / "configs" / "retarget" / "bones"
        )
        warm_solver = load_solver_profile(
            "paper_repro_scipy_trf", config_root=repo / "configs" / "retarget" / "warm_start"
        )
        expected_source_cache_hash = source_cache_hash(paths["canonical"])
        warm_preexisting = paths["warm"].is_dir()
        warm_is_compatible = False
        if warm_preexisting:
            warm = load_warm_start(paths["warm"])
            warm_is_compatible = (
                warm.metadata.get("source_cache_hash") == expected_source_cache_hash
            )
            # The frozen v2 inputs predate source-cache hashes for some valid
            # selections.  A formal v4 lineage may reuse only those immutable
            # upstream warm artifacts after checking their sequence identity
            # and frame cardinality; it must never regenerate through the
            # read-only symlink.
            if (
                formal_v4
                and upstream_root is not None
                and warm.metadata.get("source_cache_hash") is None
            ):
                warm_is_compatible = bool(
                    warm.frame_count == canonical.num_frames
                    and warm.metadata.get("source_sequence_id") == canonical.metadata.sequence_id
                )
        if warm_is_compatible:
            warm_diagnostics = {"status": "reused_existing_artifact"}
        else:
            final_lineage_exists = (
                any(
                    (root / "final" / candidate / "final_retarget.zarr" / "zarr.json").is_file()
                    for candidate in (root / "final").iterdir()
                    if candidate.is_dir()
                )
                if (root / "final").is_dir()
                else False
            )
            checkpoint_lineage_exists = any(
                path.is_file() for path in (root / "checkpoints").glob("*/manifest.json")
            )
            if warm_preexisting and (final_lineage_exists or checkpoint_lineage_exists):
                raise ValueError(
                    "existing warm-start source cache hash is incompatible with an established "
                    "final/checkpoint lineage; preserve artifacts and create a versioned run root"
                )
            # A warm artifact without the canonical source hash cannot be paired
            # with a graph or checkpoint safely.  This path is reached before
            # final refinement; no accepted final/checkpoint lineage is replaced.
            warm, warm_diagnostics = build_warm_start_trajectory(
                canonical,
                "right_hand",
                robot,
                frame_profile,
                bone_profile,
                warm_solver,
                source_cache=paths["canonical"],
            )
            warm.metadata.update(
                {
                    "stage12_target_profile_id": profile_name,
                    "stage12_warm_solver_profile_id": warm_solver.profile_id,
                    "dataset_specific_solver": False,
                }
            )
            save_warm_start(warm, paths["warm"], force=True)
            warm_diagnostics = {
                **warm_diagnostics,
                "status": "regenerated_incompatible_source_cache"
                if warm_preexisting
                else "created",
            }
        payload["retarget"].update(
            {"q_dimension": robot.num_dofs, "warm_diagnostics": warm_diagnostics}
        )

        write_progress("interaction_artifacts", frame_count=canonical.num_frames)
        object_track = _primary_object(canonical, row.get("primary_object"))
        if paths["graph"].is_dir() and paths["collision_samples"].is_file():
            graph = load_interaction_graph(paths["graph"])
            graph_object_id = graph.metadata.get("object_id")
            if graph_object_id != object_track.object_id:
                raise ValueError(
                    "existing interaction graph targets "
                    f"{graph_object_id!r}, but the frozen selection targets "
                    f"{object_track.object_id!r}; preserve the old artifacts and rebuild "
                    "viewer-derived artifacts in a versioned repair root"
                )
        else:
            object_profile = load_surface_profile("paper_strict_area_uniform", repo_root=repo)
            object_samples = sample_object_track(object_track, object_profile)
            object_samples.save(paths["object_samples"], overwrite=True)
            graph = build_source_interaction_graph(
                canonical,
                "right_hand",
                object_track.object_id,
                object_samples,
                source_cache=paths["canonical"],
                object_sample_path=paths["object_samples"],
                frame_indices=np.arange(canonical.num_frames, dtype=np.int64),
            )
            save_interaction_graph(graph, paths["graph"], force=True)
            collision_profile = load_robot_surface_profile(
                "engineering_collision_32_per_geometry", repo_root=repo
            )
            collision_samples = sample_robot_collision_surface(
                robot, warm.arrays["qpos"][0], collision_profile
            )
            collision_samples.save(paths["collision_samples"], overwrite=True)
        if paths["evaluation"].is_dir():
            load_interaction_evaluation(paths["evaluation"])
        else:
            evaluation = evaluate_interaction_graph(graph, warm, robot)
            save_interaction_evaluation(evaluation, paths["evaluation"], force=True)

        write_progress("final_refinement", frame_count=canonical.num_frames)
        solver = RefinementSolverProfile.load(profile_name, repo)
        coordinate = RefinementCoordinateProfile.load("local_seed_delta_v1", repo)
        query = CollisionQueryProfile.load("adaptive_active_set_v1", repo)
        execution = RefinementExecutionProfile.load(execution_profile_name, repo)

        existing_progress = paths["checkpoint"] / "progress.json"
        checkpoint_complete = (
            existing_progress.is_file()
            and json.loads(existing_progress.read_text(encoding="utf-8")).get("status")
            == "complete"
        )
        if checkpoint_complete and paths["final"].is_dir():
            checkpoint_status = {"status": "complete", "reused_complete_checkpoint": True}
        else:
            # Upstream adapter/canonical/warm/graph work is intentionally
            # permitted while the final queue is paused.  Claim a worker only
            # at the point final refinement would actually begin.
            assert_final_jobs_allowed(repo)
            claim_final_job(
                repo,
                job_id=job_id,
                payload={"dataset": dataset, "sequence": sequence_id},
            )
            claimed = True
            append_heartbeat(
                job_id, "job_started", {"dataset": dataset, "sequence": sequence_id}, root=repo
            )
            checkpoint_status = _run_checkpoint_refinement(
                canonical=paths["canonical"],
                warm_start=paths["warm"],
                graph_path=paths["graph"],
                robot=robot_name,
                collision_samples=paths["collision_samples"],
                query_profile_id=query.profile_id,
                coordinate_profile_id=coordinate.profile_id,
                solver_profile_id=solver.profile_id,
                execution_profile_id=execution.profile_id,
                start_frame=0,
                end_frame=canonical.num_frames,
                checkpoint_root=paths["checkpoint"],
                output=paths["final"],
                asset_root=None,
                resume=paths["checkpoint"].exists(),
                max_wall_time=None,
                stop_after_frame=None,
                progress_json=paths["progress"],
                progress_log=repo
                / ".local"
                / "runtime"
                / "final_jobs"
                / job_id
                / "heartbeat.jsonl",
                force=False,
                pause_check=lambda: paused(repo),
                frame_health_gate=(
                    lambda metadata, rows: (
                        _formal_v4_health_gate(
                            metadata,
                            rows,
                            first_ten_report=(
                                root / "manifests" / "oakink_first_ten_qualification.json"
                            )
                            if formal_v4 and dataset == "oakink"
                            else None,
                            static_single_frame=static_single_frame,
                        )
                        if formal_v4
                        else _health_gate(metadata, rows, static_single_frame=static_single_frame)
                    )
                ),
            )
        if checkpoint_status.get("status") != "complete":
            raise FinalJobPaused(str(checkpoint_status.get("pause_reason", "checkpoint paused")))
        final = load_final_trajectory(paths["final"])
        diagnostics = {"checkpoint": checkpoint_status}
        final.metadata.update(
            {
                "stage12_dataset": dataset,
                "stage12_target_profile_id": profile_name,
                "dataset_specific_solver": False,
                "dataset_specific_temporal": False,
                "contact_benchmark_status": canonical.metadata.metadata.get(
                    "contact_benchmark_status", "NOT_AVAILABLE"
                ),
            }
        )
        clip = ClipSpec(
            unit_id=f"stage12_{dataset}_{unit}",
            sequence=sequence_id,
            subject=dataset,
            object_name=object_track.object_id,
            start_frame=frame_start,
            end_frame=frame_stop,
            hand="right",
            robot=robot_name,
            native_fps=float(canonical.metadata.native_fps or 30.0),
        )
        render_clip_html(
            clip=clip,
            canonical_path=paths["canonical"],
            source_path=paths["canonical"],
            profile_paths={
                "paper_warm": (paths["warm"], True, "warm Wuji"),
                profile_name: (paths["final"], False, "final Wuji"),
            },
            output=paths["html"],
            asset_root=None,
            recommended_profile=profile_name,
            graph_path=paths["graph"],
            evaluation_path=paths["evaluation"],
        )
        html_smoke = smoke_html(
            paths["html"],
            expected_frames=canonical.num_frames,
            profiles=2,
            expected_object_id=object_track.object_id,
            expected_context_object_ids={
                item.object_id
                for item in canonical.rigid_objects
                if item.object_id != object_track.object_id
            },
        )
        if html_smoke.get("status") != "pass":
            raise ValueError(f"Stage 12 HTML smoke failed: {html_smoke}")
        frame_rows = list(checkpoint_status.get("frame_rows", []))
        runtime = classify_runtime_health(
            elapsed_s=float(final.arrays.get("solve_time_s", np.asarray([np.nan]))[-1]),
            frame_times_s=[float(item.get("solve_time_s", np.nan)) for item in frame_rows],
            static_single_frame=static_single_frame,
        )
        accepted = bool(np.all(final.arrays.get("accepted", False)))
        final_status = (
            runtime.status
            if static_single_frame and accepted
            else STATIC_FRAME_SOLVER_FAILURE
            if static_single_frame
            else "pass"
            if accepted
            else "completed_with_solver_failures"
        )
        payload.update(
            {
                "status": final_status,
                "quality": _metrics(canonical, warm, final, time.perf_counter() - started),
                "final_diagnostics": diagnostics,
                "runtime_policy": runtime.as_dict(),
                "html_smoke": html_smoke,
                "failure_diagnosis": {
                    "category": "none" if accepted else "solver_issue",
                    "detail": "No failure diagnosed"
                    if accepted
                    else "final artifact retained with per-frame optimizer/audit state",
                },
            }
        )
    except FinalJobPaused as exc:
        reason = str(exc)
        pause_state = PAUSE_STATE if reason == PAUSE_STATE else HEALTH_GATE_PAUSE_STATE
        diagnosis = "operator_control" if pause_state == PAUSE_STATE else "health_gate"
        static_status = None
        if "static_single_frame" in locals() and static_single_frame and reason != PAUSE_STATE:
            static_status = (
                STATIC_FRAME_HARD_RUNTIME_FAILURE
                if "static_frame_over_300s" in reason
                else STATIC_FRAME_SOLVER_FAILURE
                if ("solver" in reason or "optimizer" in reason or "strict_acceptance" in reason)
                else STATIC_FRAME_GEOMETRY_FAILURE
            )
        if "static_single_frame" in locals() and static_single_frame:
            elapsed = float("nan")
            frame_rows = []
            if "checkpoint_status" in locals():
                frame_rows = list(checkpoint_status.get("frame_rows", []))
                if frame_rows:
                    elapsed = float(frame_rows[-1].get("solve_time_s", np.nan))
            payload["runtime_policy"] = classify_runtime_health(
                elapsed_s=elapsed,
                frame_times_s=[float(item.get("solve_time_s", np.nan)) for item in frame_rows],
                static_single_frame=True,
            ).as_dict()
        append_heartbeat(job_id, "job_paused", {"reason": reason, "state": pause_state}, root=repo)
        payload.update(
            {
                "status": static_status or pause_state,
                "error": reason,
                "failure_diagnosis": {"category": diagnosis, "detail": reason},
            }
        )
    except Exception as exc:
        category = "source_quality"
        text = str(exc).lower()
        if "mapping" in text or "mano" in text or "mediapipe" in text:
            category = "mapping_issue"
        elif "morph" in text:
            category = "morphology_gap"
        elif "solver" in text or "optimizer" in text or "slsqp" in text:
            category = "solver_issue"
        elif "collision" in text or "sdf" in text or "penetr" in text:
            category = "collision_issue"
        static_status = None
        if "static_single_frame" in locals() and static_single_frame:
            static_status = (
                STATIC_FRAME_SOLVER_FAILURE
                if category == "solver_issue"
                else STATIC_FRAME_GEOMETRY_FAILURE
            )
        payload.update(
            {
                "status": static_status or "blocked",
                "error": str(exc),
                "failure_diagnosis": {"category": category, "detail": str(exc)},
                "quality": _metrics(canonical, warm, None, time.perf_counter() - started)
                if "canonical" in locals() and "warm" in locals()
                else {"status": "not_available"},
            }
        )
    try:
        _write_report(paths["report"], payload)
    finally:
        if claimed:
            release_final_job(repo, job_id=job_id)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--selection-index", type=int, default=None)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument(
        "--execution-profile",
        default="wuji_continuous_sequential_fast_exact_v2",
        help="fixed Stage-12 execution profile; changing it creates a distinct checkpoint lineage",
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=None,
        help="read-only existing Stage-12 root whose canonical/warm/graph artifacts are symlinked",
    )
    parser.add_argument(
        "--formal-v4",
        action="store_true",
        help="enforce the v4 formal per-frame and OakInk first-ten gates in one solver process",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="write the all-selection handoff summary from existing reports without rerunning",
    )
    args = parser.parse_args()
    repo = args.repo_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if (
        args.formal_v4
        and args.execution_profile != "wuji_continuous_sequential_fast_exact_v4_compiled_sign"
    ):
        raise ValueError(
            "--formal-v4 requires wuji_continuous_sequential_fast_exact_v4_compiled_sign"
        )
    rows = _selection_rows(args.config.expanduser().resolve())
    if args.dataset:
        rows = [row for row in rows if row["dataset"] in set(args.dataset)]
    if args.selection_index is not None:
        rows = [rows[int(args.selection_index)]]
    if args.max_trajectories is not None:
        rows = rows[: int(args.max_trajectories)]
    if args.aggregate_only:
        summary = _aggregate_existing_reports(output_root, rows)
        _write_json(output_root / "stage12_summary.json", summary)
        return (
            0
            if summary["missing_report_count"] == 0
            and summary["blocked_count"] == 0
            and summary["paused_count"] == 0
            else 1
        )
    data_root = args.data_root or Path(
        yaml.safe_load(args.config.read_text(encoding="utf-8"))["data_root"]
    )
    registry = get_dataset_adapter_registry()
    adapters: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    for row in rows:
        dataset = str(row["dataset"])
        adapter = adapters.setdefault(dataset, registry.create(dataset, data_root=data_root))
        _dataset_manifest(adapter, output_root)
        result = run_one(
            repo=repo,
            output_root=output_root,
            adapter=adapter,
            row=row,
            robot_name="wuji_hand2_beta1_rh",
            profile_name="wuji_continuous_sequential_v1",
            execution_profile_name=str(args.execution_profile),
            upstream_root=None
            if args.upstream_root is None
            else args.upstream_root.expanduser().resolve(),
            formal_v4=bool(args.formal_v4),
        )
        results.append(result)
        print(
            json.dumps(
                {"dataset": dataset, "sequence": row["sequence"], "status": result["status"]},
                sort_keys=True,
            )
        )
    summary = {
        "schema_version": "toporetarget.stage12.summary.v1",
        "target": {"robot": "wuji_hand2_beta1_rh", "profile": "wuji_continuous_sequential_v1"},
        "selection_count": len(rows),
        "pass_count": sum(
            result["status"]
            in {
                "pass",
                STATIC_FRAME_ACCEPTED,
                STATIC_FRAME_ACCEPTED_WITH_RUNTIME_WARNING,
            }
            for result in results
        ),
        "completed_with_solver_failures": sum(
            result["status"] == "completed_with_solver_failures" for result in results
        ),
        "blocked_count": sum(result["status"] == "blocked" for result in results),
        "results": [
            {
                "dataset": result["dataset"],
                "sequence": result["sequence"],
                "status": result["status"],
                "report": result["paths"]["report"],
            }
            for result in results
        ],
    }
    _write_json(output_root / "stage12_summary.json", summary)
    return 0 if summary["blocked_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
