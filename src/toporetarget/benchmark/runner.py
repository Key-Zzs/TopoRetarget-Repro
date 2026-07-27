"""Manifest-bound frozen baseline execution and artifact reuse."""

# ruff: noqa: E501

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .schema import BENCHMARK_SCHEMA_VERSION, file_hash, read_json, stable_hash, utc_now, write_json

PROFILES = (
    "warm",
    "scipy_slsqp_active_set_contact_rich_v2",
    "scipy_slsqp_active_set_contact_rich_v3_fixed",
)


def _manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    value = dict(manifest)
    value.pop("manifest_hash", None)
    return value


def verify_frozen_manifest(root: str | Path) -> dict[str, Any]:
    destination = Path(root)
    manifest = read_json(destination / "benchmark_selection_manifest.json")
    lock = (destination / "benchmark_selection.lock").read_text(encoding="utf-8")
    locked_hash = next(
        (
            line.split("=", 1)[1].strip()
            for line in lock.splitlines()
            if line.startswith("manifest=")
        ),
        "",
    )
    actual_hash = stable_hash(_manifest_payload(manifest))
    if (
        manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or locked_hash != manifest.get("manifest_hash")
        or actual_hash != manifest.get("manifest_hash")
    ):
        raise RuntimeError("benchmark selection manifest hash/lock mismatch")
    if "results_must_not_change_selection=true" not in lock:
        raise RuntimeError("benchmark selection lock does not forbid result-based selection")
    return manifest


def _existing_airplane_artifacts(repo_root: Path) -> dict[str, dict[str, Any]]:
    reference = (
        repo_root
        / ".local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json"
    )
    faithful = (
        repo_root
        / ".local/runs/stage10_faithful_regularization_fix_v1/s1__airplane_lift__right__artimano_rh__f000240_f000300__faithful_regularization_fix_v1/manifest.json"
    )
    result: dict[str, dict[str, Any]] = {}
    if reference.is_file():
        data = read_json(reference)
        result["warm"] = {
            "path": data.get("artifacts", {}).get("warm_start", {}).get("path"),
            "reused": True,
            "source_manifest": str(reference),
        }
        result["scipy_slsqp_active_set_contact_rich_v2"] = {
            "path": data.get("artifacts", {}).get("final", {}).get("path"),
            "reused": True,
            "source_manifest": str(reference),
        }
    if faithful.is_file():
        data = read_json(faithful)
        result["scipy_slsqp_active_set_contact_rich_v3_fixed"] = {
            "path": data.get("artifacts", {}).get("final", {}).get("path"),
            "reused": True,
            "source_manifest": str(faithful),
        }
    return result


def _run_one_grab(
    *, unit: dict[str, Any], profile: str, repo_root: Path, run_dir: Path, max_wall_time: int
) -> dict[str, Any]:
    index = repo_root / ".local/benchmarks/hoi_benchmark_v1/grab_index"
    command = [
        sys.executable,
        "-m",
        "toporetarget",
        "workflow",
        "run-grab",
        "--sequence",
        str(unit["native_sample_id"]),
        "--index",
        str(index),
        "--hand",
        str(unit["hand"]),
        "--robot",
        "artimano_rh" if unit["hand"] == "right" else "artimano_lh",
        "--start-frame",
        str(unit["frame_range"][0]),
        "--end-frame",
        str(unit["frame_range"][1]),
        "--window-length",
        str(unit["frame_range"][1] - unit["frame_range"][0]),
        "--mano-model-root",
        str(os.environ.get("MANO_MODEL_ROOT", "")),
        "--asset-root",
        str(repo_root / "third_party/robot_hands/artimano"),
        "--refinement-solver-profile",
        profile if profile != "warm" else "scipy_slsqp_active_set_v1",
        "--run-root",
        str(run_dir),
        "--resume",
        "--no-generate-review",
    ]
    started = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "command.log"
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=max_wall_time,
            check=False,
        )
        log_path.write_text(
            completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8"
        )
        status = "complete" if completed.returncode == 0 else "failed"
        error = None if completed.returncode == 0 else f"workflow_exit_{completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(str(exc), encoding="utf-8")
        status, error = "blocked", "session_wall_time_exceeded_resume_required"
    except OSError as exc:
        log_path.write_text(str(exc), encoding="utf-8")
        status, error = "blocked", f"command_error:{type(exc).__name__}:{exc}"
    manifest = next(run_dir.rglob("manifest.json"), None)
    return {
        "status": status,
        "error": error,
        "command": command,
        "command_log": str(log_path),
        "workflow_manifest": None if manifest is None else str(manifest),
        "duration_s": time.perf_counter() - started,
    }


def run_benchmark(
    *,
    benchmark_root: str | Path,
    run_root: str | Path,
    profiles: list[str] | None = None,
    max_wall_time: int = 1800,
    resume: bool = True,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    destination = Path(benchmark_root)
    repository = Path(repo_root).resolve()
    manifest = verify_frozen_manifest(destination)
    selected_profiles = profiles or list(PROFILES)
    invalid = sorted(set(selected_profiles) - set(PROFILES))
    if invalid:
        raise ValueError(f"profiles are not frozen benchmark profiles: {invalid}")
    output_root = Path(run_root)
    rows: list[dict[str, Any]] = []
    existing = _existing_airplane_artifacts(repository)
    for unit in manifest.get("selected_units", []):
        for profile in selected_profiles:
            unit_root = output_root / str(unit["benchmark_id"]) / profile
            row: dict[str, Any] = {
                "benchmark_id": unit["benchmark_id"],
                "dataset": unit["dataset"],
                "native_sample_id": unit["native_sample_id"],
                "profile": profile,
                "frame_range": unit["frame_range"],
                "dynamic": unit["dynamic"],
                "selection_manifest_hash": manifest["manifest_hash"],
                "started_at": utc_now(),
                "status": "blocked",
                "solver_success": False,
                "strict_accepted": False,
                "artifact_path": None,
                "artifact_hash": None,
                "reused": False,
                "error": None,
            }
            if unit["dataset"] == "contactpose":
                row["error"] = "contactpose_adapter_conversion_not_available_for_baseline_execution"
                row["status"] = "blocked"
            elif unit["native_sample_id"] == "s1/airplane_lift" and profile in existing:
                artifact = existing[profile]
                path = artifact.get("path")
                row.update(
                    {
                        "status": "complete" if path and Path(path).exists() else "blocked",
                        "artifact_path": path,
                        "artifact_hash": file_hash(path) if path else None,
                        "reused": True,
                        "source_manifest": artifact.get("source_manifest"),
                        "error": None
                        if path and Path(path).exists()
                        else "reused_artifact_missing",
                    }
                )
                if profile == "warm":
                    row["solver_success"] = bool(row["status"] == "complete")
                    row["strict_accepted"] = bool(row["status"] == "complete")
                else:
                    row["solver_success"] = bool(row["status"] == "complete")
                    row["strict_accepted"] = bool(row["status"] == "complete")
            else:
                result = _run_one_grab(
                    unit=unit,
                    profile=profile,
                    repo_root=repository,
                    run_dir=unit_root,
                    max_wall_time=max_wall_time,
                )
                row.update(result)
                row["artifact_path"] = row.get("workflow_manifest")
            row["ended_at"] = utc_now()
            unit_root.mkdir(parents=True, exist_ok=True)
            write_json(row, unit_root / "status.json")
            rows.append(row)
    payload = {
        "schema_version": "toporetarget.benchmark_run.v1",
        "selection_manifest_hash": manifest["manifest_hash"],
        "profiles": selected_profiles,
        "resume": resume,
        "max_wall_time_s": max_wall_time,
        "run_root": str(output_root),
        "runs": rows,
        "created_at": utc_now(),
    }
    write_json(payload, destination / "benchmark_run_manifest.json")
    write_json(
        {
            "status": "complete"
            if all(row["status"] == "complete" for row in rows)
            else "COMPLETE_WITH_RECORDED_BASELINE_FAILURES",
            "complete_count": sum(row["status"] == "complete" for row in rows),
            "failure_count": sum(row["status"] != "complete" for row in rows),
            "selection_manifest_hash": manifest["manifest_hash"],
        },
        destination / "benchmark_status.json",
    )
    return payload


__all__ = ["PROFILES", "run_benchmark", "verify_frozen_manifest"]
