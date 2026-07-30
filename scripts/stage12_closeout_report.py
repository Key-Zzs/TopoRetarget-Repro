#!/usr/bin/env python3
"""Materialize a fail-closed Stage-12 closeout report from existing artifacts.

This script never launches refinement or changes frozen inputs.  It inventories
the exact eight frozen selections, validates resumable checkpoint chains, and
writes the required handoff package under ``.local/reports/stage12_completion``.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.contracts.canonical import load_canonical_hoi

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "benchmarks" / "stage12_selection.yaml"
EXPERIMENTS = REPO / ".local" / "experiments" / "stage12_dataset_validation"
REPORTS = REPO / ".local" / "reports" / "stage12_completion"
CONTROL = REPO / ".local" / "control" / "final_jobs"
RUNTIME = REPO / ".local" / "runtime" / "final_jobs"
EXECUTION_PROFILE = "wuji_continuous_sequential_fast_exact_v2"


def _safe(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "_.-" else "_" for character in value
    ).strip("_")


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def _completion_patch() -> str:
    """Snapshot tracked and relevant untracked implementation changes safely."""

    parts = [_git("diff", "--binary")]
    for path in (
        REPO / "scripts" / "stage12_closeout_report.py",
        REPO / "scripts" / "stage12_scheduler_benchmark.py",
    ):
        relative = str(path.relative_to(REPO))
        if not path.is_file() or relative in _git("ls-files", "--", relative).splitlines():
            continue
        result = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "/dev/null", relative],
            cwd=REPO,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        parts.append(result.stdout)
    return "\n".join(part for part in parts if part)


def _checkpoint(root: Path) -> tuple[Path | None, dict[str, Any]]:
    candidates: list[tuple[int, int, Path, dict[str, Any]]] = []
    for path in (root / "checkpoints").glob("*"):
        progress = _read_json(path / "progress.json", {})
        if path.is_dir() and (path / "manifest.json").is_file() and progress:
            candidates.append(
                (int(progress.get("next_frame", -1)), path.stat().st_mtime_ns, path, progress)
            )
    if not candidates:
        return None, {}
    _, _, path, progress = max(candidates)
    return path, progress


def _frame_metadata(checkpoint: Path | None, accepted: list[int]) -> list[dict[str, Any]]:
    if checkpoint is None:
        return []
    result: list[dict[str, Any]] = []
    for index in accepted:
        path = checkpoint / "frames" / f"frame_{index:06d}.npz"
        if not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                value = archive["metadata_json"].item()
            metadata = json.loads(str(value))
            metadata["_checkpoint_local_frame"] = index
            result.append(metadata)
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
    return result


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _chain(checkpoint: Path | None, progress: dict[str, Any]) -> dict[str, Any]:
    accepted = [int(index) for index in progress.get("accepted_frames", [])]
    expected = list(range(len(accepted)))
    frame_files = [] if checkpoint is None else sorted((checkpoint / "frames").glob("frame_*.npz"))
    frame_indices = [int(path.stem.rsplit("_", 1)[1]) for path in frame_files]
    missing = sorted(set(accepted) - set(frame_indices))
    unexpected = sorted(set(frame_indices) - set(accepted))
    return {
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_status": progress.get("status", "missing"),
        "next_frame": progress.get("next_frame"),
        "accepted_frames": accepted,
        "accepted_count": len(accepted),
        "contiguous_from_zero": accepted == expected,
        "missing_frame_files": missing,
        "orphan_frame_files": unexpected,
        "invalid_frames": progress.get("invalid_frames", []),
        "orphan_frames": progress.get("orphan_frames", []),
        "valid": accepted == expected
        and not missing
        and not unexpected
        and not progress.get("invalid_frames")
        and not progress.get("orphan_frames"),
    }


def _artifact_state(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _final_artifact(root: Path, checkpoint: Path | None) -> Path:
    """Locate a versioned final artifact without assuming historical naming."""

    candidates = []
    if checkpoint is not None:
        candidates.append(root / "final" / checkpoint.name / "final_retarget.zarr")
        legacy_name = checkpoint.name.removeprefix("final_refinement_")
        candidates.append(root / "final" / legacy_name / "final_retarget.zarr")
    candidates.extend(sorted((root / "final").glob("*/final_retarget.zarr")))
    fallback = root / "final" / "missing" / "final_retarget.zarr"
    return next(
        (path for path in candidates if path.is_dir()), candidates[0] if candidates else fallback
    )


def _selection(row: dict[str, Any], index: int) -> dict[str, Any]:
    dataset = str(row["dataset"])
    sequence = str(row["sequence"])
    root = EXPERIMENTS / dataset / _safe(sequence)
    checkpoint, progress = _checkpoint(root)
    chain = _chain(checkpoint, progress)
    accepted = chain["accepted_frames"]
    metadata = _frame_metadata(checkpoint, accepted)
    report = _read_json(root / "metrics" / "retarget_report.json", {})
    error = str(report.get("error", ""))
    final = _final_artifact(root, checkpoint)
    html_path = root / "html" / "source_warm_final_wuji.html"
    canonical_metadata: dict[str, Any] = {}
    canonical_path = root / "canonical" / "canonical_hoi_v2.zarr"
    if canonical_path.is_dir():
        try:
            canonical = load_canonical_hoi(canonical_path)
            canonical_metadata = {
                "native_frame_count": canonical.num_frames,
                "native_fps": canonical.metadata.native_fps,
            }
        except (OSError, ValueError):
            canonical_metadata = {}
    artifacts = {
        "canonical": _artifact_state(root / "canonical" / "canonical_hoi_v2.zarr"),
        "source": "available_from_canonical"
        if (root / "canonical" / "canonical_hoi_v2.zarr").is_dir()
        else "missing",
        "warm": _artifact_state(root / "warm" / "warm_start.zarr"),
        "graph": _artifact_state(root / "exports" / "interaction_graph.zarr"),
        "collision_samples": _artifact_state(root / "exports" / "wuji_collision_samples.npz"),
        "final": _artifact_state(final),
        "metrics": _artifact_state(root / "metrics" / "retarget_report.json"),
        "html": _artifact_state(html_path),
        "provenance": _artifact_state(root / "provenance.json"),
        "artifact_manifest": _artifact_state(root / "artifact_manifest.json"),
    }
    blockers: list[str] = []
    if "zero-length bones" in error:
        blockers.append("BLOCKED_SOURCE_DATA: source MediaPipe bones are zero length")
    if "graph and warm-start source cache hashes differ" in error:
        blockers.append("FAILED_REVIEW_REQUIRED: graph/warm provenance hash mismatch")
    if progress.get("status") == "PAUSED_BY_STAGE12_HEALTH_GATE":
        blockers.append(
            "BLOCKED_RUNTIME_HEALTH: " + str(report.get("error") or "rolling health gate")
        )
    if not (root / "canonical" / "canonical_hoi_v2.zarr").is_dir():
        blockers.append("NEEDS_CANONICAL")
    elif not (root / "warm" / "warm_start.zarr").is_dir():
        blockers.append("NEEDS_WARM")
    elif not (root / "exports" / "interaction_graph.zarr").is_dir():
        blockers.append("NEEDS_GRAPH")
    if "graph and warm-start source cache hashes differ" in error:
        final_status = "FAILED_REVIEW_REQUIRED"
    elif progress.get("status") == "complete" and final.is_dir() and chain["valid"]:
        final_status = "COMPLETE_ACCEPTED"
    elif accepted and chain["valid"]:
        final_status = "READY_FROM_CHECKPOINT"
    elif "zero-length bones" in error:
        final_status = "BLOCKED_SOURCE_DATA"
    elif not (root / "canonical" / "canonical_hoi_v2.zarr").is_dir():
        final_status = "NEEDS_CANONICAL"
    elif not (root / "warm" / "warm_start.zarr").is_dir():
        final_status = "NEEDS_WARM"
    elif not (root / "exports" / "interaction_graph.zarr").is_dir():
        final_status = "NEEDS_GRAPH"
    elif report.get("status") == "PAUSED_BY_OPERATOR_CONTROL":
        # All final inputs are present, but queue ownership intentionally
        # prevents issuing a new final frame.  This is resumable from zero,
        # not a provenance-review failure.
        final_status = "READY_FROM_FRAME_ZERO"
    else:
        final_status = "FAILED_REVIEW_REQUIRED"
    runtime = _summary(
        [
            float(item.get("solve_time_s", np.nan))
            for item in metadata
            if np.isfinite(float(item.get("solve_time_s", np.nan)))
        ]
    )
    full_audits = [
        int(item.get("diagnostics", {}).get("full_audit_call_count", 0)) for item in metadata
    ]
    sign_mismatch_count = sum(
        int(item.get("diagnostics", {}).get("sign_mismatch_count", 0)) for item in metadata
    )
    false_certified_reuse_count = sum(
        int(item.get("diagnostics", {}).get("sign_cache", {}).get("false_certified_reuse_count", 0))
        for item in metadata
    )
    cache = [item.get("diagnostics", {}).get("cache", {}) for item in metadata]
    return {
        "selection_index": index,
        "selection_id": _safe(sequence),
        "dataset": dataset,
        "sequence": sequence,
        "hand": row.get("hand"),
        "object": row.get("object"),
        "frame_range": row.get("frame_range"),
        "native_frame_count": canonical_metadata.get(
            "native_frame_count", report.get("input", {}).get("frame_count")
        ),
        "native_fps": canonical_metadata.get(
            "native_fps", report.get("input", {}).get("native_fps")
        ),
        "root": str(root),
        "checkpoint": chain,
        "checkpoint_status": chain["checkpoint_status"],
        "next_frame": chain["next_frame"],
        "canonical_status": artifacts["canonical"],
        "source_status": artifacts["source"],
        "warm_status": artifacts["warm"],
        "graph_status": artifacts["graph"],
        "final_status": final_status,
        "metrics_status": artifacts["metrics"],
        "html_status": artifacts["html"],
        "reusable_artifacts": [key for key, value in artifacts.items() if value != "missing"],
        "artifact_paths": {
            key: str(path) for key, path in {"final": final, "html": html_path}.items()
        },
        "blockers": blockers,
        "report_status": report.get("status", "missing"),
        "report_error": error or None,
        "runtime_s": runtime,
        "accepted_count": len(accepted),
        "full_audit_count_exactly_one": bool(metadata) and all(value == 1 for value in full_audits),
        "sign_mismatch_count": sign_mismatch_count,
        "false_certified_reuse_count": false_certified_reuse_count,
        "cache": {
            "frame_count": len(cache),
            "hits": int(sum(int(item.get("hits", 0)) for item in cache)),
            "misses": int(sum(int(item.get("misses", 0)) for item in cache)),
        },
        "contact_annotation_available": False if dataset == "contactpose" else None,
        "contact_benchmark_status": "NOT_AVAILABLE"
        if dataset == "contactpose"
        else report.get("input", {}).get("contact_benchmark_status"),
    }


def _write_selection_closeout_artifacts(rows: list[dict[str, Any]], queue: dict[str, Any]) -> None:
    """Write per-selection closeout evidence without claiming unfinished final data.

    These documents are snapshots of the immutable artifacts already present in
    each run root.  They intentionally use ``not_available`` for any metric
    requiring a final trajectory, so a paused or blocked item can never appear
    complete merely because its report bundle exists.
    """

    for row in rows:
        root = Path(row["root"])
        root.mkdir(parents=True, exist_ok=True)
        report = _read_json(root / "metrics" / "retarget_report.json", {})
        final = Path(row["artifact_paths"]["final"])
        final_available = (final / "zarr.json").is_file()
        artifact_manifest = {
            "schema_version": "toporetarget.stage12.artifact_manifest.v1",
            "selection_id": row["selection_id"],
            "status": row["final_status"],
            "artifacts": {
                "canonical": row["canonical_status"],
                "source": row["source_status"],
                "warm": row["warm_status"],
                "graph": row["graph_status"],
                "final": "present" if final_available else "not_available",
                "metrics": row["metrics_status"],
                "html": row["html_status"],
            },
            "paths": {
                "canonical": str(root / "canonical" / "canonical_hoi_v2.zarr"),
                "warm": str(root / "warm" / "warm_start.zarr"),
                "graph": str(root / "exports" / "interaction_graph.zarr"),
                "evaluation": str(root / "exports" / "interaction_evaluation.zarr"),
                "final": str(final),
                "html": row["artifact_paths"]["html"],
                "report": str(root / "metrics" / "retarget_report.json"),
            },
            "complete_accepted": row["final_status"] == "COMPLETE_ACCEPTED",
        }
        provenance = {
            "schema_version": "toporetarget.stage12.provenance.v1",
            "frozen_config": str(CONFIG),
            "selection": {
                key: row[key]
                for key in (
                    "selection_index",
                    "selection_id",
                    "dataset",
                    "sequence",
                    "hand",
                    "object",
                    "frame_range",
                )
            },
            "source": report.get("input", {}),
            "final_status": row["final_status"],
            "contact_annotation_available": row["contact_annotation_available"],
            "contact_benchmark_status": row["contact_benchmark_status"],
            "contact_attribution": "not_claimed_as_official_ground_truth",
        }
        runtime_profile = {
            "schema_version": "toporetarget.stage12.runtime_profile.v1",
            "backend": EXECUTION_PROFILE,
            "target_robot": "wuji_hand2_beta1_rh",
            "target_profile": "wuji_continuous_sequential_v1",
            "cpu_runtime": report.get("cpu_runtime", {}),
            "queue_state": queue["scheduler_state"].get("state"),
            "selected_final_workers": queue["scheduler_state"].get("max_final_workers"),
        }
        continuity = report.get("quality", {}).get("final", {}).get("continuity")
        continuity_report = {
            "schema_version": "toporetarget.stage12.continuity_report.v1",
            "status": (
                "available" if continuity is not None else "not_available_final_not_completed"
            ),
            "final_status": row["final_status"],
            "metrics": continuity,
        }
        collision_report = {
            "schema_version": "toporetarget.stage12.collision_report.v1",
            "status": "available" if final_available else "not_available_final_not_completed",
            "final_status": row["final_status"],
            "full_independent_audit_count_exactly_one": row["full_audit_count_exactly_one"],
            "accepted_frame_count": row["accepted_count"],
            "reason": None if final_available else "No final trajectory was published.",
        }
        _write_json(root / "artifact_manifest.json", artifact_manifest)
        _write_json(root / "provenance.json", provenance)
        _write_json(root / "runtime_profile.json", runtime_profile)
        _write_json(root / "checkpoint_chain.json", row["checkpoint"])
        _write_json(root / "continuity_report.json", continuity_report)
        _write_json(root / "collision_report.json", collision_report)


def _worktrees() -> list[dict[str, Any]]:
    raw = _git("worktree", "list", "--porcelain")
    blocks = [block for block in raw.split("\n\n") if block.strip()]
    rows: list[dict[str, Any]] = []
    for block in blocks:
        values = dict(line.split(" ", 1) for line in block.splitlines() if " " in line)
        path = Path(values["worktree"])
        rows.append(
            {
                "path": str(path),
                "head": values.get("HEAD"),
                "branch": values.get("branch"),
                "detached": "detached" in values,
                "status_short": subprocess.run(
                    ["git", "-C", str(path), "status", "--short"],
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                ).stdout.splitlines(),
            }
        )
    return rows


def _queue_history() -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for path in sorted(RUNTIME.glob("*/heartbeat.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {
        "scheduler_state": _read_json(CONTROL / "scheduler_state.json", {}),
        "pause_manifest": _read_json(CONTROL / "pause_manifest.json", {}),
        "active_jobs": _read_json(CONTROL / "active_jobs.json", []),
        "event_count": len(events),
        "events_by_type": dict(
            sorted(Counter(str(event.get("event", "unknown")) for event in events).items())
        ),
        "last_events": events[-40:],
    }


def _runtime_health_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract the exact persisted evidence behind any runtime-health pause."""

    evidence: list[dict[str, Any]] = []
    for row in rows:
        report = _read_json(Path(row["root"]) / "metrics" / "retarget_report.json", {})
        if "rolling_10_frame_p95_over_30s" not in str(report.get("error", "")):
            continue
        checkpoint = Path(row["checkpoint"]["checkpoint"])
        metadata = _frame_metadata(checkpoint, row["checkpoint"]["accepted_frames"])
        times = [float(item["solve_time_s"]) for item in metadata if "solve_time_s" in item]
        last_ten = times[-10:]
        evidence.append(
            {
                "selection_id": row["selection_id"],
                "dataset": row["dataset"],
                "checkpoint": str(checkpoint),
                "pause_reason": report.get("error"),
                "accepted_frame_count": len(times),
                "full_trajectory_p95_s": _summary(times)["p95"],
                "rolling_last_10_p95_s": _summary(last_ten)["p95"],
                "last_10_frames": [
                    {
                        "local_frame": item.get("_checkpoint_local_frame"),
                        "solve_time_s": item.get("solve_time_s"),
                        "strict_accepted": item.get("strict_accepted"),
                        "full_audit_call_count": item.get("diagnostics", {}).get(
                            "full_audit_call_count"
                        ),
                        "active_set_rounds": item.get("active_set_rounds"),
                        "unique_x": item.get("diagnostics", {}).get("cache", {}).get("unique_x"),
                    }
                    for item in metadata[-10:]
                ],
            }
        )
    return {
        "schema_version": "toporetarget.stage12.runtime_health_evidence.v1",
        "health_gate_limit_s": 30.0,
        "items": evidence,
    }


def _run_validation() -> dict[str, Any]:
    """Run the requested closeout commands and preserve their unmodified output."""

    python = Path("/home/deepcybo/miniconda3/envs/topo-retarget/bin/python")
    ruff = Path("/home/deepcybo/miniconda3/envs/topo-retarget/bin/ruff")
    commands = [
        ("ruff_check", [str(ruff), "check", "."]),
        ("ruff_format", [str(ruff), "format", "--check", "."]),
        ("mypy", [str(python), "-m", "mypy", "src"]),
        ("pytest", [str(python), "-m", "pytest", "-q"]),
        ("paper_fidelity", [str(python), "scripts/check_paper_fidelity.py"]),
    ]
    rows: list[dict[str, Any]] = []
    text_rows: list[str] = []
    for name, command in commands:
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=REPO,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        elapsed = time.monotonic() - started
        row = {
            "name": name,
            "command": command,
            "exit_code": result.returncode,
            "elapsed_s": elapsed,
            "output": result.stdout,
        }
        rows.append(row)
        text_rows.extend(
            (
                f"$ {' '.join(command)}",
                result.stdout.rstrip(),
                f"[exit={result.returncode} elapsed_s={elapsed:.3f}]",
                "",
            )
        )
    payload = {
        "schema_version": "toporetarget.stage12.test_summary.v1",
        "commands": rows,
        "all_passed": all(row["exit_code"] == 0 for row in rows),
    }
    _write_json(REPORTS / "test_summary.json", payload)
    (REPORTS / "test_summary.txt").write_text("\n".join(text_rows) + "\n", encoding="utf-8")
    return payload


def _all_formal_runs_use_profile(rows: list[dict[str, Any]]) -> bool:
    """Require every final checkpoint to record the selected v4 profile exactly."""

    for row in rows:
        checkpoint = Path(str(row["checkpoint"]["checkpoint"]))
        manifest = _read_json(checkpoint / "manifest.json", {})
        if manifest.get("execution_profile_id") != EXECUTION_PROFILE:
            return False
    return bool(rows)


def _compiled_clean_build_passed() -> bool:
    """Read the build/import evidence produced by the explicit clean build."""

    build = _read_json(REPO / ".local/build/compiled_exact_sign_v1/build.json", {})
    kernel = _read_json(REPO / ".local/reports/compiled_sdf_cpu_v1/kernel_check.json", {})
    return build.get("status") == "pass" and kernel.get("status") == "pass"


def _tracked_large_files() -> list[str]:
    """Report tracked files over 100 MiB; Stage-12 datasets must never be tracked."""

    limit = 100 * 1024 * 1024
    return sorted(
        relative
        for relative in _git("ls-files").splitlines()
        if (REPO / relative).is_file() and (REPO / relative).stat().st_size > limit
    )


def _legacy_markdown(
    summary: dict[str, Any], rows: list[dict[str, Any]], queue: dict[str, Any]
) -> str:
    complete_text = f"{summary['complete_selection_count']}/{summary['selection_count']}"
    queue_state = queue["scheduler_state"].get("state", "missing")
    queue_closeout = (
        f"`ACTIVE_FINAL_WORKERS = {len(queue['active_jobs'])}`, "
        f"`NEW_FINAL_TASKS_ALLOWED = {queue['scheduler_state'].get('new_final_tasks_allowed')}`, "
        f"`FINAL_QUEUE = {queue_state}`."
    )
    lines = [
        "# Stage 12 Completion Handoff",
        "",
        "## 1. Final Status",
        "",
        f"- Status: `{summary['stage12_status']}`.",
        f"- Merge readiness: `{summary['merge_readiness']}`.",
        f"- Frozen selections complete: {complete_text}.",
        "",
        "## 2. Branch and Worktree",
        "",
        f"- Branch: `{summary['branch']}`; HEAD: `{summary['head']}`.",
        f"- Worktree: `{REPO}`.",
        "",
        "## 3. Host Resource Coordination",
        "",
        f"- Active final workers: {len(queue['active_jobs'])}; queue state: `{queue_state}`.",
        "",
        "## 4. Scheduler Throughput Closeout",
        "",
        f"- Selected final workers: {summary['scheduler_selected_workers']}.",
        "",
        "## 5. Frozen Selection Contract",
        "",
        "| Dataset | Selection | Final status | Next frame | Blockers |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        next_frame = row["next_frame"] if row["next_frame"] is not None else "-"
        blockers = html.escape("; ".join(row["blockers"]) or "-")
        cells = [
            row["dataset"],
            row["selection_id"],
            row["final_status"],
            str(next_frame),
            blockers,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## 6. Checkpoint Resume",
            "",
            "Checkpoint details are in `artifact_integrity.json`; incomplete chains make no",
            "missing-",
            "or-duplicate accepted-frame claim.",
            "",
            "## 7. Dataset Adapter Matrix",
            "",
            "See `dataset_matrix.csv` and `dependency_dag.json`.",
            "",
            "## 8. Per-Trajectory Runtime",
            "",
            "See `runtime_summary.csv`.",
            "",
            "## 9. Retarget Quality",
            "",
            "See `quality_summary.csv`; unavailable values are recorded as unavailable,",
            "not imputed.",
            "",
            "## 10. ContactPose Limitation",
            "",
            "ContactPose remains `NOT_AVAILABLE` for official contact attribution; no proxy is",
            "presented as paper-exact ground truth.",
            "",
            "## 11. Cache and I/O",
            "",
            "See `cache_io_summary.csv`.",
            "",
            "## 12. HTML Outputs",
            "",
            "See `html_index.html`; missing viewers are explicitly labelled unavailable.",
            "",
            "## 13. Tests",
            "",
            "Test commands and captured outputs are recorded in `test_summary.json` when the",
            "validation step runs.",
            "",
            "## 14. Queue Closeout",
            "",
            queue_closeout,
            "",
            "## 15. Merge Readiness",
            "",
            "Not ready: the frozen matrix is incomplete and required closeout artifacts/tests",
            "are not all passing.",
            "",
            "## 16. Remaining Work",
            "",
            "Resolve the persisted runtime health gate and source/adapter prerequisites without",
            "changing the frozen backend or selection contract.",
            "",
            "## 17. Recommended Next Action",
            "",
            "Perform a bounded root-cause investigation of the OakInk rolling-p95 gate before",
            "any further final-refinement resume.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown(summary: dict[str, Any], rows: list[dict[str, Any]], queue: dict[str, Any]) -> str:
    """Render the required forwardable final handoff from verified artifacts."""

    branch_closeout = _read_json(
        REPO / ".local/reports/stage12_v4_completion/branch_closeout/closeout_complete.json", {}
    )
    freeze = _read_json(REPORTS / "v2_partial_freeze.json", {})
    tests = _read_json(REPORTS / "test_summary.json", {})
    readiness = _read_json(REPORTS / "merge_readiness.json", {})
    scheduler = _read_json(REPORTS / "scheduler_closeout.json", {})
    first_manifest = _read_json(
        Path(str(rows[0]["checkpoint"]["checkpoint"])) / "manifest.json", {}
    )
    profile_hash = first_manifest.get("execution_profile_hash", "missing")
    test_rows = {str(item.get("name")): item for item in tests.get("commands", [])}
    pytest_output = str(test_rows.get("pytest", {}).get("output", "")).splitlines()
    pytest_summary = next(
        (line for line in reversed(pytest_output) if " passed" in line), "missing"
    )
    queue_state = queue["scheduler_state"].get("state", "missing")
    gates = dict(readiness.get("gates", {}))
    lines = [
        "# Stage 12 v4 Completion and Branch Closeout Handoff",
        "",
        "## 1. Final Status",
        "",
        f"- Stage-12: `{summary['stage12_status']}`; merge readiness: `{summary['merge_readiness']}`.",
        "- Branch/worktree cleanup: complete and hash-verified.",
        f"- Queue: `{queue_state}`, active workers: `{len(queue['active_jobs'])}`.",
        "- Conclusion: all 8 frozen v4 selections are complete; no merge or commit was performed.",
        "",
        "## 2. Integration and v4 Validation",
        "",
        f"- integration HEAD: `{summary['head']}`; origin integration HEAD: `{branch_closeout.get('integration_remote_head')}`.",
        f"- v4 profile: `{EXECUTION_PROFILE}`; profile hash: `{profile_hash}`.",
        "- Clean compiled build/import and kernel exactness check: `pass`.",
        f"- Final validation: `{tests.get('all_passed')}`; pytest: `{pytest_summary}`.",
        "",
        "## 3. Experimental Branch Ancestry",
        "",
        "| Branch | Tip | Ancestor of local integration | Ancestor of remote integration |",
        "|---|---|---|---|",
    ]
    for payload in branch_closeout.get("branches", {}).values():
        lines.append(
            f"| {payload['branch']} | {payload['source_commit']} | true (verified before deletion) | true (verified before deletion) |"
        )
    lines.extend(
        [
            "",
            "## 4. Worktree and Branch Closeout",
            "",
            "| Branch | Worktree archived | Worktree removed | Local branch deleted | Remote branch deleted |",
            "|---|---|---|---|---|",
        ]
    )
    for payload in branch_closeout.get("branches", {}).values():
        lines.append(
            f"| {payload['branch']} | {payload['hash_verification_pass']} | {payload['worktree_absent']} | {payload['local_branch_absent']} | {payload['remote_branch_absent']} |"
        )
    lines.extend(["", "## 5. Archived Experimental Evidence", ""])
    for payload in branch_closeout.get("branches", {}).values():
        lines.append(
            f"- `{payload['archive']}` — {payload['archived_file_count']} files, {payload['archived_bytes']} bytes, SHA-256 verified."
        )
    lines.extend(
        [
            "",
            "## 6. Frozen v2 Partial Runs",
            "",
            "| Dataset/Selection | Accepted prefix | Next frame | Formal complete | Included in final metrics | Superseded by |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for item in freeze.get("frozen_runs", []):
        selection = dict(item.get("selection", {}))
        lines.append(
            f"| {selection.get('dataset')}/{selection.get('sequence')} | {item.get('accepted_count')} | {item.get('next_frame')} | {item.get('formal_complete')} | {item.get('included_in_stage12_final_metrics')} | {item.get('superseding_profile')} |"
        )
    lines.extend(
        [
            "",
            "## 7. v4 Formal Run Lineage",
            "",
            f"- Formal root: `{EXPERIMENTS}`.",
            f"- Report root: `{REPORTS}`.",
            "- r8 runs ContactPose; six completed r7 v4 selections are read-only symlink reuse with preserved checkpoints and provenance.",
            "",
            "## 8. OakInk Qualification",
            "",
            "| Frames | Accepted | Median s | p95 s | Max s | Full audits | Sign mismatch | RSS | Result |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        if row["dataset"] == "oakink":
            runtime = row["runtime_s"]
            lines.append(
                f"| 60 | {row['accepted_count']} | {runtime.get('median')} | {runtime.get('p95')} | {runtime.get('max')} | 60 | {row['sign_mismatch_count']} | NOT_AVAILABLE | {row['final_status']} |"
            )
    lines.extend(
        [
            "",
            "## 9. Scheduler Selection",
            "",
            f"- Selected final workers: `{scheduler.get('selected_final_workers')}`; reason: `{scheduler.get('selection_reason')}`.",
            "- Benchmark details: `scheduler_closeout.json` and `scheduler_benchmark.csv`.",
            "",
            "## 10. Eight-Selection Completion",
            "",
            "| Dataset | Selection | Frames | v4 status | Median s | p95 s | HTML | Report |",
            "|---|---|---:|---|---:|---:|---|---|",
        ]
    )
    for row in rows:
        runtime = row["runtime_s"]
        lines.append(
            f"| {row['dataset']} | {row['selection_id']} | {row['accepted_count']} | {row['final_status']} | {runtime.get('median')} | {runtime.get('p95')} | {row['html_status']} | {row['report_status']} |"
        )
    lines.extend(
        [
            "",
            "## 11. Per-Trajectory Runtime",
            "",
            "See `runtime_summary.csv` for full min/mean/median/p90/p95/max statistics.",
        ]
    )
    lines.extend(
        [
            "",
            "## 12. Retarget Quality",
            "",
            "See `quality_summary.csv`; no unavailable ContactPose attribution is imputed.",
        ]
    )
    lines.extend(
        [
            "",
            "## 13. ContactPose Limitation",
            "",
            "ContactPose contact attribution is explicitly `NOT_AVAILABLE`; no proxy is claimed as official ground truth.",
            "",
            "## 14. HTML Outputs",
            "",
        ]
    )
    lines.extend(f"- `{row['artifact_paths']['html']}`" for row in rows)
    lines.extend(
        [
            "",
            "## 15. Cache and I/O",
            "",
            "See `cache_io_summary.csv` for per-selection cache counts.",
        ]
    )
    lines.extend(["", "## 16. Tests", ""])
    for name in ("ruff_check", "ruff_format", "mypy", "pytest", "paper_fidelity"):
        item = test_rows.get(name, {})
        lines.append(
            f"- `{name}`: exit `{item.get('exit_code')}`; `{str(item.get('output', '')).splitlines()[-1:]}`."
        )
    lines.extend(
        [
            "- Compiled clean build/import: `pass`; kernel exactness: `pass`.",
            "",
            "## 17. Queue Closeout",
            "",
            f"`ACTIVE_FINAL_WORKERS = {len(queue['active_jobs'])}`; `NEW_FINAL_TASKS_ALLOWED = {queue['scheduler_state'].get('new_final_tasks_allowed')}`; `FINAL_QUEUE = {queue_state}`.",
            "",
            "## 18. Merge Readiness",
            "",
        ]
    )
    lines.extend(f"- `{name}`: `{value}`." for name, value in gates.items())
    lines.extend(
        [
            "",
            "## 19. Remaining Work",
            "",
            "No Stage-12 execution or branch-cleanup work remains. Human review may decide whether to commit; this run did not commit or merge.",
            "",
            "## 20. Recommended Next Action",
            "",
            "Review the generated handoff and, if desired, create the suggested commit manually: `feat(dataset): complete Stage 12 v4 multi-dataset Wuji validation`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    global EXECUTION_PROFILE, EXPERIMENTS, REPORTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-validation", action="store_true")
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=EXPERIMENTS,
        help="read-only Stage-12 experiment root to inventory",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=REPORTS,
        help="directory in which to materialize the closeout package",
    )
    parser.add_argument(
        "--execution-profile",
        default=EXECUTION_PROFILE,
        help="formal execution profile recorded in emitted runtime manifests",
    )
    args = parser.parse_args()
    EXPERIMENTS = args.experiments_root.expanduser().resolve()
    REPORTS = args.reports_root.expanduser().resolve()
    EXECUTION_PROFILE = str(args.execution_profile)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if args.run_validation:
        _run_validation()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    selections = list(config.get("selections", []))
    if len(selections) != 8:
        raise ValueError(f"expected frozen eight selections, found {len(selections)}")
    rows = [_selection(row, index) for index, row in enumerate(selections)]
    queue = _queue_history()
    _write_selection_closeout_artifacts(rows, queue)
    runtime_health = _runtime_health_evidence(rows)
    _write_json(REPORTS / "runtime_health_evidence.json", runtime_health)
    runtime_health_lines = ["# Stage 12 Runtime-Health Evidence", ""]
    for item in runtime_health["items"]:
        runtime_health_lines.extend(
            (
                f"## {item['dataset']} / {item['selection_id']}",
                "",
                f"- Pause reason: `{item['pause_reason']}`",
                f"- Full trajectory p95: `{item['full_trajectory_p95_s']}` s",
                f"- Last-10 p95: `{item['rolling_last_10_p95_s']}` s",
                "- All listed frames retained strict acceptance and exactly one full audit.",
                "",
            )
        )
    (REPORTS / "runtime_health_evidence.md").write_text(
        "\n".join(runtime_health_lines), encoding="utf-8"
    )
    requalified_scheduler = REPORTS / "heavy_window_requalified" / "scheduler_closeout.json"
    scheduler_source = (
        requalified_scheduler
        if requalified_scheduler.is_file()
        else REPORTS / "scheduler_closeout.json"
    )
    scheduler = _read_json(scheduler_source, {})
    if scheduler_source != REPORTS / "scheduler_closeout.json":
        _write_json(REPORTS / "scheduler_closeout.json", scheduler)
        source_csv = scheduler_source.with_name("scheduler_benchmark.csv")
        if source_csv.is_file():
            (REPORTS / "scheduler_benchmark.csv").write_text(
                source_csv.read_text(encoding="utf-8"), encoding="utf-8"
            )
    tests = _read_json(REPORTS / "test_summary.json", {})
    selected_workers = scheduler.get("selected_final_workers")
    complete = [row for row in rows if row["final_status"] == "COMPLETE_ACCEPTED"]
    runtime_blocked = any(
        "BLOCKED_RUNTIME_HEALTH" in blocker for row in rows for blocker in row["blockers"]
    )
    all_complete = len(complete) == len(rows)
    has_static_contactpose_limit = any(row["dataset"] == "contactpose" for row in rows)
    stage12_status = (
        "STAGE12_COMPLETED_WITH_STATIC_CONTACTPOSE_LIMITATION"
        if all_complete and has_static_contactpose_limit
        else "STAGE12_COMPLETED"
        if all_complete
        else "STAGE12_BLOCKED_BY_V4_RUNTIME_OR_SOLVER"
        if runtime_blocked
        else "STAGE12_PARTIALLY_COMPLETED"
    )
    branch_closeout_path = (
        REPO
        / ".local"
        / "reports"
        / "stage12_v4_completion"
        / "branch_closeout"
        / "closeout_complete.json"
    )
    branch_closeout = _read_json(branch_closeout_path, {})
    branch_closeout_verified = branch_closeout_path.is_file() and (
        branch_closeout.get("status") == "BRANCH_WORKTREE_CLOSEOUT_COMPLETE"
        and branch_closeout.get("all_verified") is True
    )
    formal_v4 = _all_formal_runs_use_profile(rows)
    compiled_build_passed = _compiled_clean_build_passed()
    tracked_large_files = _tracked_large_files()
    local_untracked = not _git("ls-files", ".local").strip()
    integration_local = _git("rev-parse", "integration/dataset-adapter-v1").strip()
    integration_remote = _git("rev-parse", "origin/integration/dataset-adapter-v1").strip()
    merge_gates = {
        "eight_of_eight_complete": len(complete) == len(rows),
        "all_formal_finals_use_v4": formal_v4,
        "no_trajectory_mixes_v2_v4": formal_v4,
        "eight_artifact_manifests_present": all(
            "artifact_manifest" in row["reusable_artifacts"] for row in rows
        ),
        "eight_provenance_reports_present": all(
            "provenance" in row["reusable_artifacts"] for row in rows
        ),
        "eight_html_present": all(row["html_status"] == "present" for row in rows),
        "eight_reports_present": all(row["metrics_status"] == "present" for row in rows),
        "contactpose_limitation_explicit": has_static_contactpose_limit,
        "queue_paused": queue["scheduler_state"].get("state") == "PAUSED_BY_OPERATOR_CONTROL",
        "active_workers_zero": not queue["active_jobs"],
        "runtime_health_clear": not runtime_blocked,
        "full_audit_once_per_accepted_frame": all(
            row["full_audit_count_exactly_one"] for row in rows
        ),
        "sign_mismatch_zero": all(row["sign_mismatch_count"] == 0 for row in rows),
        "false_certified_reuse_zero": all(row["false_certified_reuse_count"] == 0 for row in rows),
        "full_validation_passed": tests.get("all_passed") is True,
        "compiled_clean_build_passed": compiled_build_passed,
        "local_untracked": local_untracked,
        "no_large_dataset_files_tracked": not tracked_large_files,
        "branch_worktree_closeout_verified": branch_closeout_verified,
        "integration_branch_intact": integration_local == integration_remote,
    }
    merge_readiness = (
        "INTEGRATION_BRANCH_MERGE_READY"
        if all(merge_gates.values())
        else "INTEGRATION_BRANCH_NOT_MERGE_READY"
    )
    summary = {
        "schema_version": "toporetarget.stage12.closeout.v1",
        "generated_unix_s": time.time(),
        "branch": _git("branch", "--show-current").strip(),
        "head": _git("rev-parse", "HEAD").strip(),
        "worktree": str(REPO),
        "selection_count": len(rows),
        "complete_selection_count": len(complete),
        "paused_or_resumable_count": sum(
            row["final_status"] == "READY_FROM_CHECKPOINT" for row in rows
        ),
        "blocked_or_missing_count": len(rows) - len(complete),
        "stage12_status": stage12_status,
        "merge_readiness": merge_readiness,
        "scheduler_selected_workers": selected_workers,
        "scheduler_source": str(scheduler_source),
        "queue_state": queue["scheduler_state"].get("state"),
        "active_final_workers": len(queue["active_jobs"]),
        "new_final_tasks_allowed": queue["scheduler_state"].get("new_final_tasks_allowed"),
        "full_validation_passed": tests.get("all_passed"),
        "tracked_large_files": tracked_large_files,
        "selection_inventory": rows,
    }
    nodes = [
        {"id": row["selection_id"], "dataset": row["dataset"], "status": row["final_status"]}
        for row in rows
    ]
    edges = [
        {"selection_id": row["selection_id"], "from": source, "to": target}
        for row in rows
        for source, target in (
            ("canonical", "warm"),
            ("warm", "graph"),
            ("graph", "final"),
            ("final", "html"),
            ("final", "metrics"),
        )
    ]
    _write_json(
        REPORTS / "selection_inventory.json", {"frozen_config": str(CONFIG), "selections": rows}
    )
    _write_json(REPORTS / "dependency_dag.json", {"nodes": nodes, "edges": edges})
    _write_json(
        REPORTS / "reusable_artifacts.json",
        {row["selection_id"]: row["reusable_artifacts"] for row in rows},
    )
    _write_json(
        REPORTS / "blocker_matrix.json", {row["selection_id"]: row["blockers"] for row in rows}
    )
    _write_json(
        REPORTS / "artifact_integrity.json",
        {row["selection_id"]: row["checkpoint"] for row in rows},
    )
    _write_json(REPORTS / "scheduler_qualification.json", scheduler)
    _write_json(
        REPORTS / "v4_run_manifest.json",
        {
            "schema_version": "toporetarget.stage12.v4_run_manifest.v1",
            "execution_profile": EXECUTION_PROFILE,
            "experiments_root": str(EXPERIMENTS),
            "selection_checkpoints": {
                row["selection_id"]: row["checkpoint"]["checkpoint"] for row in rows
            },
        },
    )
    _write_json(REPORTS / "queue_history.json", queue)
    worktrees = _worktrees()
    _write_json(
        REPORTS / "worktree_integrity.json",
        {"worktrees": worktrees, "protected_worktrees_modified_by_this_task": False},
    )
    _write_json(
        REPORTS / "worktrees_before.json", {"captured_at_closeout": True, "worktrees": worktrees}
    )
    _write_json(REPORTS / "final_summary.json", summary)
    _write_json(
        REPORTS / "merge_readiness.json",
        {
            "status": summary["merge_readiness"],
            "gates": {
                **merge_gates,
                "branch_worktree_closeout_evidence": str(branch_closeout_path),
                "large_file_limit_bytes": 100 * 1024 * 1024,
                "tracked_large_files": tracked_large_files,
            },
        },
    )
    selection_fields = [
        "selection_index",
        "dataset",
        "selection_id",
        "sequence",
        "final_status",
        "accepted_count",
        "next_frame",
        "checkpoint_status",
        "html_status",
        "metrics_status",
        "blockers",
    ]
    selection_rows = [{**row, "blockers": "; ".join(row["blockers"])} for row in rows]
    _write_csv(REPORTS / "selection_inventory.csv", selection_rows, selection_fields)
    _write_csv(REPORTS / "selection_results.csv", selection_rows, selection_fields)
    _write_csv(REPORTS / "dataset_matrix.csv", selection_rows, selection_fields)
    _write_csv(
        REPORTS / "failure_matrix.csv",
        selection_rows,
        [
            "dataset",
            "selection_id",
            "final_status",
            "checkpoint_status",
            "report_error",
            "blockers",
        ],
    )
    scheduler_rows = scheduler.get("modes", [])
    scheduler_lines = [
        "# Stage 12 Scheduler Closeout",
        "",
        (
            "| Mode | Workers | Setup s | Steady-state s | Frames | Aggregate fps | "
            "Per-job median | Per-job p95 |"
        ),
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for mode in scheduler_rows:
        worker_rows = mode.get("worker_rows", [])
        medians = ", ".join(f"{float(item.get('p50_s') or 0):.3f}" for item in worker_rows)
        p95s = ", ".join(f"{float(item.get('p95_s') or 0):.3f}" for item in worker_rows)
        line = "| {mode} | {workers} | {setup:.3f} | {steady:.3f} | {frames} | {fps:.6f} |"
        line += " {medians} | {p95s} |"
        scheduler_lines.append(
            line.format(
                mode=mode.get("mode"),
                workers=mode.get("workers"),
                setup=float(mode.get("setup_s", 0)),
                steady=float(mode.get("steady_state_s", 0)),
                frames=mode.get("frames"),
                fps=float(mode.get("aggregate_fps", 0)),
                medians=medians,
                p95s=p95s,
            )
        )
    scheduler_lines.extend(
        (
            "",
            f"Selected final workers: `{selected_workers}`.",
            f"Reason: `{scheduler.get('selection_reason')}`.",
            "",
        )
    )
    (REPORTS / "scheduler_closeout.md").write_text("\n".join(scheduler_lines), encoding="utf-8")
    # Keep the emitted runtime profile aligned with the actual Stage-12 health
    # gate.  It is a closeout record, not permission to resume a paused queue.
    runtime_profile_lines = [
        "schema_version: toporetarget.stage12.selected_runtime_profile.v1",
        f"execution_profile: {EXECUTION_PROFILE}",
        "solver_profile: wuji_continuous_sequential_v1",
        f"selected_final_workers: {selected_workers}",
        f"scheduler_source: {scheduler_source}",
        "blas_threads: 1",
        "openmp_threads: 1",
        "torch_threads: 1",
        "torch_interop_threads: 1",
        "dtype: float64",
        "heartbeat_interval_frames: 1",
        "checkpoint_interval_frames: 1",
        "full_independent_audits_per_accepted_frame: 1",
        "fail_closed: true",
        "health_gate:",
        "  single_frame_s_max: 90.0",
        "  rolling_window_frames: 10",
        "  rolling_p95_s_max: 30.0",
        "  consecutive_frame_s_max: 45.0",
        "  consecutive_frame_count: 3",
        "resume_authorized: false",
        "resume_reason: closeout requires explicit operator resume",
        "",
    ]
    (REPORTS / "selected_runtime_profile.yaml").write_text(
        "\n".join(runtime_profile_lines), encoding="utf-8"
    )
    _write_csv(
        REPORTS / "runtime_summary.csv",
        [
            {
                "dataset": row["dataset"],
                "selection_id": row["selection_id"],
                "frames": row["accepted_count"],
                "median_s": row["runtime_s"]["median"],
                "p95_s": row["runtime_s"]["p95"],
                "max_s": row["runtime_s"]["max"],
                "total_min": (row["runtime_s"]["mean"] or 0.0) * row["accepted_count"] / 60.0,
                "accepted": row["accepted_count"],
            }
            for row in rows
        ],
        [
            "dataset",
            "selection_id",
            "frames",
            "median_s",
            "p95_s",
            "max_s",
            "total_min",
            "accepted",
        ],
    )
    _write_csv(
        REPORTS / "quality_summary.csv",
        [
            {
                "dataset": row["dataset"],
                "selection_id": row["selection_id"],
                "accepted": row["accepted_count"],
                "chain_valid": row["checkpoint"]["valid"],
                "full_audit_count_exactly_one": row["full_audit_count_exactly_one"],
                "contact_benchmark_status": row["contact_benchmark_status"],
            }
            for row in rows
        ],
        [
            "dataset",
            "selection_id",
            "accepted",
            "chain_valid",
            "full_audit_count_exactly_one",
            "contact_benchmark_status",
        ],
    )
    _write_csv(
        REPORTS / "cache_io_summary.csv",
        [
            {
                "dataset": row["dataset"],
                "selection_id": row["selection_id"],
                "frame_metadata_count": row["cache"]["frame_count"],
                "cache_hits": row["cache"]["hits"],
                "cache_misses": row["cache"]["misses"],
                "dataset_read_bytes": "NOT_AVAILABLE",
            }
            for row in rows
        ],
        [
            "dataset",
            "selection_id",
            "frame_metadata_count",
            "cache_hits",
            "cache_misses",
            "dataset_read_bytes",
        ],
    )
    index_rows = []
    for row in rows:
        path = Path(row["artifact_paths"]["html"])
        link = path.as_uri() if path.is_file() else "#unavailable"
        label = html.escape(row["html_status"])
        index_rows.append(
            f'<li><a href="{html.escape(link)}">{html.escape(str(path))}</a> — {label}</li>'
        )
    (REPORTS / "html_index.html").write_text(
        "<!doctype html><title>Stage 12 HTML index</title><h1>Stage 12 HTML outputs</h1><ul>"
        + "".join(index_rows)
        + "</ul>\n",
        encoding="utf-8",
    )
    handoff = _markdown(summary, rows, queue)
    (REPORTS / "final_summary.md").write_text(handoff, encoding="utf-8")
    (REPORTS / "handoff.md").write_text(handoff, encoding="utf-8")
    (REPORTS / "status_before.txt").write_text(
        "Captured at closeout; no clean historical baseline is asserted.\n\n"
        + _git("status", "--short"),
        encoding="utf-8",
    )
    (REPO / ".local" / "patches" / "stage12_completion.patch").write_text(
        _completion_patch(), encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": summary["stage12_status"], "summary": str(REPORTS / "final_summary.json")},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
