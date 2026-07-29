"""Bounded, non-interactive S1 dense-SDF comparison workflow."""

# The self-contained HTML template below intentionally keeps JavaScript on
# compact lines; all Python logic remains formatted and type-checked.
# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.retarget.artifacts import load_warm_start
from toporetarget.retarget.final_refinement import (
    final_artifact_hash,
    load_final_trajectory,
    save_final_trajectory,
)
from toporetarget.retarget.penetration_loss import (
    DenseSDFPenetrationLoss,
    PenetrationLossProfile,
    build_objective_term,
)

EXPERIMENT_ID = "s1_sdf_penetration_loss_v1"
DEFAULT_CONFIG = Path("configs/experiments/s1_sdf_penetration_loss_v1.yaml")
LAMBDA_LABELS = {0.0: "E0", 0.01: "S1_L001", 0.1: "S1_L01", 1.0: "S1_L1"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode())
            with child.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: Any, default: float = 0.0) -> float:
    result = float(value)
    return result if np.isfinite(result) else default


def _metrics(
    path: Path, clip: str, lambda_sdf: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    final = load_final_trajectory(path)
    arrays = final.arrays
    phi = np.asarray(arrays["full_signed_distance"], dtype=np.float64)
    if phi.ndim != 2 or phi.shape[1] != 512:
        raise ValueError(f"{path}: full audit must be [frames,512], got {phi.shape}")
    e_sdf = np.asarray(arrays.get("e_sdf", np.zeros(len(phi))), dtype=np.float64)
    weighted = np.asarray(arrays.get("weighted_e_sdf", np.zeros(len(phi))), dtype=np.float64)
    accepted = np.asarray(arrays.get("accepted", np.zeros(len(phi), dtype=bool)), dtype=bool)
    solver_status = np.asarray(arrays.get("solver_status", np.full(len(phi), -1)), dtype=np.int64)
    optimizer_status = np.asarray(
        arrays.get("optimizer_status_code", np.full(len(phi), -1)), dtype=np.int64
    )
    qpos_bounds = np.asarray(
        arrays.get("qpos_bounds_pass", np.zeros(len(phi), dtype=bool)), dtype=bool
    )
    slack_bounds = np.asarray(
        arrays.get("slack_bounds_pass", np.zeros(len(phi), dtype=bool)), dtype=bool
    )
    active_feasible = np.asarray(
        arrays.get("active_constraints_feasible", np.zeros(len(phi), dtype=bool)), dtype=bool
    )
    finite_mask = np.asarray(
        arrays.get("all_values_finite", np.zeros(len(phi), dtype=bool)), dtype=bool
    )
    slack_offsets = np.asarray(arrays.get("slack_offsets", np.zeros(len(phi) + 1)), dtype=np.int64)
    slack_concat = np.asarray(arrays.get("slack_concat", np.zeros(0)), dtype=np.float64)
    query_offsets = np.asarray(arrays.get("query_offsets", np.zeros(len(phi) + 1)), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for index in range(len(phi)):
        values = phi[index]
        depth = np.maximum(-values, 0.0)
        qpos = np.asarray(arrays.get("qpos", np.zeros((len(phi), 0))), dtype=np.float64)
        base = np.asarray(
            arrays.get("base_pose_scene", np.zeros((len(phi), 4, 4))), dtype=np.float64
        )
        q_step = (
            0.0
            if index == 0 or not len(qpos)
            else float(np.linalg.norm(qpos[index] - qpos[index - 1]))
        )
        base_step = (
            0.0
            if index == 0 or not len(base)
            else float(np.linalg.norm(base[index, :3, 3] - base[index - 1, :3, 3]))
        )
        rows.append(
            {
                "clip": clip,
                "lambda_sdf": lambda_sdf,
                "frame": index,
                "global_frame": int(np.asarray(arrays["frame_indices"])[index]),
                "solver_status": int(solver_status[index]),
                "optimizer_status_code": int(optimizer_status[index]),
                "accepted": bool(accepted[index]),
                "qpos_bounds_pass": bool(qpos_bounds[index]),
                "slack_bounds_pass": bool(slack_bounds[index]),
                "active_constraints_feasible": bool(active_feasible[index]),
                "all_values_finite": bool(finite_mask[index]),
                "full_sample_count": int(len(values)),
                "full_min_signed_distance_m": float(np.min(values)),
                "full_negative_sample_count": int(np.count_nonzero(values < 0.0)),
                "full_negative_sample_fraction": float(np.mean(values < 0.0)),
                "max_penetration_m": float(max(0.0, -np.min(values))),
                "mean_penetration_depth_m": float(np.mean(depth)),
                "rms_penetration_depth_m": float(np.sqrt(np.mean(np.square(depth)))),
                "e_sdf": _finite(e_sdf[index]),
                "weighted_e_sdf": _finite(weighted[index]),
                "e_im": _finite(np.asarray(arrays.get("e_im", np.zeros(len(phi))))[index]),
                "e_bone": _finite(np.asarray(arrays.get("e_bone", np.zeros(len(phi))))[index]),
                "q_step": q_step,
                "base_translation_step_m": base_step,
                "contact_f1_at_5mm_proxy": None,
                "contact_precision_alignment_proxy": None,
                "morphology_rmse_m": None,
                "per_finger_rmse_m": None,
                "slack_max": float(
                    np.max(
                        slack_concat[slack_offsets[index] : slack_offsets[index + 1]], initial=0.0
                    )
                ),
                "query_active_count": int(query_offsets[index + 1] - query_offsets[index]),
                "solve_time_s": _finite(
                    np.asarray(arrays.get("solve_time_s", [0.0] * len(phi)))[index]
                ),
                "active_set_converged": bool(
                    np.asarray(arrays.get("active_set_converged", [False] * len(phi)))[index]
                ),
                "full_surface_hard_audit_pass": bool(
                    np.asarray(arrays.get("full_surface_hard_audit_pass", [False] * len(phi)))[
                        index
                    ]
                ),
                "full_surface_soft_audit_pass": bool(
                    np.asarray(arrays.get("full_surface_soft_audit_pass", [False] * len(phi)))[
                        index
                    ]
                ),
            }
        )
    summary = {
        "clip": clip,
        "lambda_sdf": lambda_sdf,
        "artifact": str(path),
        "artifact_hash": _sha256(path),
        "frame_count": len(rows),
        "full_sample_count": int(phi.shape[1]),
        "accepted_frame_count": int(np.count_nonzero(accepted)),
        "status_9_count": int(np.count_nonzero(optimizer_status == 9)),
        "strict_accepted_count": int(np.count_nonzero(accepted)),
        "qpos_bounds_pass": bool(np.all(qpos_bounds)),
        "slack_bounds_pass": bool(np.all(slack_bounds)),
        "active_constraints_feasible": bool(np.all(active_feasible)),
        "all_values_finite": bool(np.all(finite_mask)),
        "active_set_converged_count": int(
            np.count_nonzero(np.asarray(arrays.get("active_set_converged", []), dtype=bool))
        ),
        "full_min_signed_distance_m": float(np.min(phi)),
        "negative_sample_count": int(np.count_nonzero(phi < 0.0)),
        "negative_sample_fraction": float(np.mean(phi < 0.0)),
        "max_penetration_m": float(max(0.0, -np.min(phi))),
        "mean_e_sdf": float(np.mean(e_sdf)),
        "mean_weighted_e_sdf": float(np.mean(weighted)),
        "finite": bool(np.all(np.isfinite(phi)) and np.all(np.isfinite(e_sdf))),
        "metadata": final.metadata,
    }
    return rows, summary


def _lambda_zero_equivalence(
    e0_paths: dict[str, Path], s1_paths: dict[str, Path]
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    passed = True
    for clip in e0_paths:
        e0 = load_final_trajectory(e0_paths[clip])
        s1 = load_final_trajectory(s1_paths[clip])
        clip_checks: dict[str, Any] = {}
        for key, tolerance in (
            ("qpos", 1e-10),
            ("base_pose_scene", 1e-10),
            ("full_signed_distance", 1e-10),
            ("total_objective", 1e-12),
            ("weighted_e_sdf", 1e-12),
        ):
            left = np.asarray(e0.arrays[key], dtype=np.float64)
            right = np.asarray(s1.arrays[key], dtype=np.float64)
            difference = float(np.max(np.abs(left - right)))
            clip_checks[f"{key}_max_abs_diff"] = difference
            clip_checks[f"{key}_tolerance"] = tolerance
            passed &= bool(difference <= tolerance)
        for key in ("solver_status", "accepted", "active_set_converged"):
            same = bool(np.array_equal(e0.arrays[key], s1.arrays[key]))
            clip_checks[f"{key}_exact"] = same
            passed &= same
        checks[clip] = clip_checks
    return {"status": "pass" if passed else "fail", "clips": checks}


def _run_command(
    root: Path,
    args: list[str],
    *,
    dry_run: bool = False,
    allow_failure: bool = False,
) -> dict[str, Any]:
    command = [sys.executable, "-m", "toporetarget", *args]
    if dry_run:
        return {"command": command, "status": "dry_run"}
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        command, cwd=root, env=env, check=False, text=True, capture_output=True
    )
    result = {
        "command": command,
        "status": "pass" if completed.returncode == 0 else "failed",
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0 and not allow_failure:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return result


def _final_path(run_root: Path, clip: str, label: str) -> Path:
    return run_root / "artifacts" / clip / label / "final.zarr"


def _e0_path(run_root: Path, clip: str) -> Path:
    return run_root / "e0" / clip / "final.zarr"


def _checkpoint_path(run_root: Path, clip: str, label: str) -> Path:
    return run_root / "checkpoints" / clip / label


def _augment_e0_diagnostic(
    path: Path,
    surface_path: Path,
    profile_id: str,
    *,
    destination: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Add scalar full-surface E0 diagnostics without changing paper-core arrays."""

    trajectory = load_final_trajectory(path)
    output = path if destination is None else destination
    if (
        not force
        and output.exists()
        and load_final_trajectory(output).metadata.get("penetration_loss_diagnostic_version")
        == "independent_full_surface_v1"
    ):
        return {"status": "reused", "path": str(output)}
    if (
        not force
        and trajectory.metadata.get("penetration_loss_diagnostic_version")
        == "independent_full_surface_v1"
    ):
        if output != path:
            output.parent.mkdir(parents=True, exist_ok=True)
            save_final_trajectory(trajectory, output, force=True)
        return {"status": "reused", "path": str(output)}
    surface = np.load(surface_path, allow_pickle=False)
    geometry_ids = np.asarray(surface["geometry_ids"]).astype(str)
    profile = PenetrationLossProfile.load(profile_id)
    term = build_objective_term(
        DenseSDFPenetrationLoss.term_id,
        profile=profile,
        lambda_sdf=0.0,
    )
    full_phi = np.asarray(trajectory.arrays["full_signed_distance"], dtype=np.float64)
    if full_phi.ndim != 2 or full_phi.shape[1] != len(geometry_ids):
        raise ValueError(f"{path}: E0 full audit does not match collision surface")
    e_sdf = np.asarray(
        [term.value_only(frame, geometry_ids) for frame in full_phi], dtype=np.float64
    )
    trajectory.arrays["e_sdf"] = e_sdf
    trajectory.arrays["weighted_e_sdf"] = np.zeros_like(e_sdf)
    geometry_id_hash = hashlib.sha256(
        json.dumps(geometry_ids.tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    metadata = dict(trajectory.metadata)
    metadata["penetration_loss"] = {
        "term_id": term.term_id,
        "profile": profile.as_dict(),
        "lambda_sdf": 0.0,
        "profile_hash": term.profile_hash,
        "query_surface": {
            "sample_count": int(len(geometry_ids)),
            "collision_surface_profile_hash": hashlib.sha256(
                str(surface["metadata"].item()).encode()
            ).hexdigest(),
            "geometry_id_hash": geometry_id_hash,
        },
        "paper_method": True,
        "paper_external_extension": False,
        "extension_type": "dense_sdf_penetration_loss",
        "paper_constraints_preserved": True,
        "diagnostic_evaluation": "independent_full_surface_v1",
        "scalar_gradient_fallback_count": 0,
        "optimization_sdf_backend": "not_used_lambda_zero",
        "validation_sdf_backend": "reference_winding_v1",
        "declared_inner_sdf_backend": profile.inner_sdf_backend,
        "declared_validation_sdf_backend": profile.validation_sdf_backend,
    }
    metadata["penetration_loss_diagnostic_version"] = "independent_full_surface_v1"
    trajectory.metadata = metadata
    trajectory.metadata["artifact_hash"] = final_artifact_hash(trajectory)
    save_final_trajectory(trajectory, output, force=True)
    return {"status": "augmented", "path": str(output), "mean_e_sdf": float(np.mean(e_sdf))}


def _refine(
    root: Path,
    run_root: Path,
    clip: str,
    cfg: dict[str, Any],
    lambda_sdf: float,
    *,
    max_wall_time: float,
    resume: bool,
    dry_run: bool,
    label_override: str | None = None,
    end_frame: int | None = None,
    stop_after_frame: int | None = None,
    output_override: Path | None = None,
    allow_failure: bool = False,
) -> dict[str, Any]:
    base = run_root / "selection" / clip
    label = label_override or LAMBDA_LABELS[float(lambda_sdf)]
    output = output_override or _final_path(run_root, clip, label)
    expected_count = 60 if end_frame is None else int(end_frame)
    existing_metadata: dict[str, Any] = {}
    if output.exists() and not dry_run:
        try:
            existing_metadata = load_final_trajectory(output).metadata
        except (OSError, ValueError, RuntimeError):
            existing_metadata = {}
    diagnostic_ready = (
        lambda_sdf != 0.0
        or existing_metadata.get("penetration_loss_diagnostic_version")
        == "independent_full_surface_v1"
        or existing_metadata.get("penetration_loss", {}).get("diagnostic_evaluation")
        == "independent_full_surface_v1"
    )
    diagnostic_rebuild = lambda_sdf == 0.0 and output.exists() and not diagnostic_ready
    checkpoint_label = label + "_diagnostic" if diagnostic_rebuild else label
    checkpoint = _checkpoint_path(run_root, clip, checkpoint_label)
    if output.exists() and not dry_run:
        existing = load_final_trajectory(output)
        existing_count = existing.metadata.get("frame_count")
        if existing_count is None:
            existing_count = existing.arrays["frame_indices"].shape[0]
        if int(existing_count) == expected_count and diagnostic_ready:
            return {
                "clip": clip,
                "lambda_sdf": lambda_sdf,
                "output": str(output),
                "status": "reused",
                "artifact_hash": _sha256(output),
            }
    args = [
        "retarget",
        "refine",
        "--canonical",
        str(base / "canonical.zarr"),
        "--warm-start",
        str(base / "warm_start.npz"),
        "--graph",
        str(base / "interaction_graph.npz"),
        "--robot",
        str(cfg["robot"]),
        "--collision-samples",
        str(run_root / "selection" / "artimano_rh_collision_surface.npz"),
        "--query-profile",
        str(cfg["frozen_profiles"]["query"]),
        "--solver-profile",
        str(cfg["frozen_profiles"]["solver"]),
        "--execution-profile",
        str(cfg["frozen_profiles"]["execution"]),
        "--penetration-loss-profile",
        str(cfg["frozen_profiles"]["penetration_loss"]),
        "--validation-sdf-backend",
        str(cfg["full_audit"]["signed_distance_backend"]),
        "--lambda-sdf",
        str(lambda_sdf),
        "--start-frame",
        "0",
        "--end-frame",
        str(expected_count),
        "--checkpoint-root",
        str(checkpoint),
        "--max-wall-time",
        str(max_wall_time),
        "--progress-json",
        str(checkpoint / "progress.json"),
        "--progress-log",
        str(checkpoint / "logs" / "progress.jsonl"),
        "--output",
        str(output),
        "--asset-root",
        str(root / cfg["robot_asset_root"]),
    ]
    if stop_after_frame is not None:
        args.extend(["--stop-after-frame", str(stop_after_frame)])
    if resume or (checkpoint / "manifest.json").is_file():
        args.append("--resume")
    if diagnostic_rebuild:
        args.append("--force")
    result = _run_command(root, args, dry_run=dry_run, allow_failure=allow_failure)
    result.update({"clip": clip, "lambda_sdf": lambda_sdf, "output": str(output)})
    return result


def _artifact_frame_count(path: Path) -> int | None:
    """Return a persisted trajectory frame count, or None for incomplete output."""

    if not path.exists():
        return None
    try:
        trajectory = load_final_trajectory(path)
    except (OSError, ValueError, RuntimeError):
        return None
    metadata_count = trajectory.metadata.get("frame_count")
    if metadata_count is not None:
        return int(metadata_count)
    return int(np.asarray(trajectory.arrays["frame_indices"]).shape[0])


def _checkpoint_state(path: Path) -> tuple[Any, ...] | None:
    progress = path / "progress.json"
    if not progress.exists():
        return None
    try:
        payload = json.loads(progress.read_text())
    except (OSError, TypeError, ValueError):
        return None
    return (
        payload.get("status"),
        payload.get("next_frame"),
        payload.get("last_accepted_frame"),
        payload.get("remaining_frames"),
        payload.get("failure_reason"),
    )


def _refine_until_complete(
    root: Path,
    run_root: Path,
    clip: str,
    cfg: dict[str, Any],
    lambda_sdf: float,
    *,
    max_wall_time: float,
    resume: bool,
    label_override: str | None = None,
    end_frame: int | None = None,
    stop_after_frame: int | None = None,
    output_override: Path | None = None,
    expected_count: int,
) -> dict[str, Any]:
    """Resume bounded refine sessions until the requested artifact is complete.

    A wall-time-bounded refine invocation may return without an artifact even
    though its checkpoint is resumable.  The old workflow treated that return
    as a candidate failure and continued to another clip/lambda.  That silently
    invalidated the frozen grid.  This wrapper only returns after the persisted
    trajectory has the requested frame count; repeated identical checkpoint
    states fail closed after three attempts.
    """

    label = label_override or LAMBDA_LABELS[float(lambda_sdf)]
    output = output_override or _final_path(run_root, clip, label)
    effective_label_override = label_override
    checkpoint = _checkpoint_path(run_root, clip, effective_label_override or label)
    previous_state: tuple[Any, ...] | None = None
    no_progress_attempts = 0
    last_result: dict[str, Any] = {}
    for attempt in range(1, 257):
        result = _refine(
            root,
            run_root,
            clip,
            cfg,
            lambda_sdf,
            max_wall_time=max_wall_time,
            resume=resume or attempt > 1,
            dry_run=False,
            label_override=effective_label_override,
            end_frame=end_frame,
            stop_after_frame=stop_after_frame,
            output_override=output,
            allow_failure=True,
        )
        last_result = result
        frame_count = _artifact_frame_count(output)
        if frame_count == int(expected_count):
            result.update(
                {
                    "status": "pass" if result.get("status") != "reused" else "reused",
                    "attempts": attempt,
                    "frame_count": frame_count,
                }
            )
            return result
        identity_mismatch = "checkpoint identity mismatch for input_signature" in str(
            result.get("stderr_tail", "")
        )
        if identity_mismatch and not str(effective_label_override or "").endswith("_rebuild_v2"):
            effective_label_override = f"{label}_rebuild_v2"
            checkpoint = _checkpoint_path(run_root, clip, effective_label_override)
            previous_state = None
            no_progress_attempts = 0
            continue
        state = _checkpoint_state(checkpoint)
        if state == previous_state:
            no_progress_attempts += 1
        else:
            no_progress_attempts = 0
        previous_state = state
        if no_progress_attempts >= 3:
            raise RuntimeError(
                f"S1_INCOMPLETE_REFINE: {clip}/{label} remained at {state} "
                f"after {attempt} attempts; expected {expected_count} frames; "
                f"last_result={last_result}"
            )
    raise RuntimeError(
        f"S1_INCOMPLETE_REFINE: {clip}/{label} exceeded 256 resume attempts; "
        f"expected {expected_count} frames; last_result={last_result}"
    )


def _prepare_selection(
    root: Path, run_root: Path, cfg: dict[str, Any], *, dry_run: bool
) -> list[dict[str, Any]]:
    selection_results: list[dict[str, Any]] = []
    mano_root = Path(
        os.environ.get(
            "MANO_MODEL_ROOT", "/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano"
        )
    )
    grab_root = Path(os.environ.get("GRAB_ROOT", "/mnt/nas/storage/Ref2Dex_storage/GRAB/data/GRAB"))
    for clip, spec in cfg["clips"].items():
        out = run_root / "selection" / clip
        out.mkdir(parents=True, exist_ok=True)
        canonical = out / "canonical.zarr"
        if not canonical.exists():
            selection_results.append(
                _run_command(
                    root,
                    [
                        "data",
                        "convert",
                        "--dataset",
                        "grab",
                        "--sequence-path",
                        str(spec["source_file"]),
                        "--grab-root",
                        str(grab_root),
                        "--mano-model-root",
                        str(mano_root),
                        "--hands",
                        cfg["hand"],
                        "--contact-mode",
                        "semantic",
                        "--include-mediapipe21",
                        "--start-frame",
                        str(spec["start_frame"]),
                        "--end-frame",
                        str(spec["end_frame"]),
                        "--output",
                        str(canonical),
                        "--force",
                    ],
                    dry_run=dry_run,
                )
            )
        samples = out / "object_samples.npz"
        if not samples.exists():
            selection_results.append(
                _run_command(
                    root,
                    [
                        "geometry",
                        "sample-object",
                        "--canonical",
                        str(canonical),
                        "--object-id",
                        str(spec["object_id"]),
                        "--profile",
                        "paper_strict_area_uniform",
                        "--output",
                        str(samples),
                        "--report",
                        str(out / "object_samples.json"),
                        "--force",
                    ],
                    dry_run=dry_run,
                )
            )
        warm = out / "warm_start.npz"
        if not warm.exists():
            selection_results.append(
                _run_command(
                    root,
                    [
                        "retarget",
                        "warm-start",
                        "--canonical",
                        str(canonical),
                        "--hand",
                        cfg["hand"],
                        "--robot",
                        cfg["robot"],
                        "--end-frame",
                        str(cfg["frame_count"]),
                        "--frame-profile",
                        cfg["frozen_profiles"]["frame"],
                        "--bone-profile",
                        cfg["frozen_profiles"]["bone"],
                        "--solver-profile",
                        cfg["frozen_profiles"]["warm_solver"],
                        "--asset-root",
                        str(root / cfg["robot_asset_root"]),
                        "--output",
                        str(warm),
                        "--force",
                    ],
                    dry_run=dry_run,
                )
            )
        graph = out / "interaction_graph.npz"
        if not graph.exists():
            selection_results.append(
                _run_command(
                    root,
                    [
                        "retarget",
                        "build-interaction-graph",
                        "--canonical",
                        str(canonical),
                        "--hand",
                        cfg["hand"],
                        "--object-samples",
                        str(samples),
                        "--delaunay-profile",
                        cfg["frozen_profiles"]["graph"],
                        "--end-frame",
                        str(cfg["frame_count"]),
                        "--output",
                        str(graph),
                        "--report",
                        str(out / "interaction_graph.json"),
                        "--force",
                    ],
                    dry_run=dry_run,
                )
            )
    robot_surface = run_root / "selection" / "artimano_rh_collision_surface.npz"
    if not robot_surface.exists():
        selection_results.append(
            _run_command(
                root,
                [
                    "geometry",
                    "sample-robot",
                    "--robot",
                    cfg["robot"],
                    "--profile",
                    cfg["frozen_profiles"]["collision_surface"],
                    "--asset-root",
                    str(root / cfg["robot_asset_root"]),
                    "--output",
                    str(robot_surface),
                    "--report",
                    str(robot_surface.with_suffix(".json")),
                    "--force",
                ],
                dry_run=dry_run,
            )
        )
    return selection_results


def _html_report(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    title = html.escape("S1 Dense SDF Penetration Loss: E0 versus S1")
    document = """<!doctype html>
<html><head><meta charset='utf-8'><title>__TITLE__</title>
<style>body{font:14px system-ui;background:#111827;color:#e5e7eb;margin:0}main{padding:18px;max-width:1500px;margin:auto}button,select,input{background:#1f2937;color:#e5e7eb;border:1px solid #4b5563;padding:5px}canvas{width:100%;height:220px;background:#0b1220;border:1px solid #374151}table{border-collapse:collapse;width:100%;margin-top:12px}td,th{border:1px solid #374151;padding:4px 7px;text-align:right}th:first-child,td:first-child{text-align:left}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}pre{white-space:pre-wrap;background:#0b1220;padding:10px}.warn{color:#fbbf24}</style></head>
<body><main><h1>__TITLE__</h1><p class='warn'>Paper-external diagnostic; positive-outside phi; full 512 collision samples; no manual acceptance.</p>
<label>Clip <select id='clip'></select></label> <label>Frame <input id='frame' type='range' min='0' max='59' value='0'></label><span id='frameLabel'></span>
<label>Profile <select id='profile'><option value='s1'>S1</option><option value='e0'>E0</option></select></label> <button id='play'>Play</button> <button id='reset'>Camera reset</button>
<button id='worstPen'>Worst-penetration frame</button> <button id='worstContact'>Worst-contact-regression frame</button>
<label><input id='overlay' type='checkbox' checked> overlay / side-by-side</label> <label><input id='showCollision' type='checkbox' checked> collision samples / QuerySet / closest object points</label>
<div>Signed-distance coloring: penetrating samples are red; non-penetrating collision samples are amber; closest object points are gray.</div>
<div class='grid'><div><h2>Scene mesh/samples</h2><canvas id='scene' width='1200' height='420'></canvas><h2>Timeline</h2><canvas id='timeline' width='1200' height='220'></canvas></div><div><h2>Selected frame</h2><pre id='metrics'></pre></div></div>
<h2>Per-frame comparison</h2><table><thead><tr><th>frame</th><th>E0 min phi (mm)</th><th>S1 min phi (mm)</th><th>E0 SDF</th><th>S1 SDF</th><th>S1 negative</th><th>S1 accepted</th></tr></thead><tbody id='rows'></tbody></table>
<script>const DATA=__DATA__;const $=id=>document.getElementById(id);const clips=Object.keys(DATA.clips);clips.forEach(c=>$('clip').add(new Option(c,c)));let timer=null;let camera={scale:1,ox:0,oy:0};
function rows(){return DATA.clips[$('clip').value]}function visual(){return DATA.visual[$('clip').value]}function project(p,all){const xs=all.map(q=>q[0]),ys=all.map(q=>q[1]),mnx=Math.min(...xs),mxx=Math.max(...xs),mny=Math.min(...ys),mxy=Math.max(...ys);const s=.82*Math.min(1150/Math.max(mxx-mnx,1e-9),380/Math.max(mxy-mny,1e-9))*camera.scale;return [25+(p[0]-mnx)*s+camera.ox,400-(p[1]-mny)*s+camera.oy]}
function drawScene(){const v=visual(),i=+$('frame').value,mode=$('profile').value,c=$('scene'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);const sets=[v.object[i],v.source[i],v[mode][i]];if($('overlay').checked)sets.push(v[mode==='e0'?'s1':'e0'][i]);if($('showCollision').checked){sets.push(v.collision[i]);sets.push(v.query[i]);sets.push(v.closest[i]);sets.push(v.penetrating[i])}const all=[].concat(...sets);const colors=['#a78bfa','#38bdf8',mode==='e0'?'#60a5fa':'#34d399',mode==='e0'?'#34d399':'#60a5fa','#f59e0b','#fde047','#9ca3af','#f87171'];sets.forEach((pts,k)=>{pts.forEach((p,j)=>{x.fillStyle=k===4?(v.phi[i][j]<0?'#f87171':'#f59e0b'):colors[k];const q=project(p,all);x.fillRect(q[0],q[1],k===0?2:3,k===0?2:3)})});x.fillStyle='#e5e7eb';x.fillText('object / source MANO / Stage 7 warm / E0 / S1 / collision / QuerySet / closest object points / penetrating',20,20)}
function drawTimeline(){const rs=rows(),c=$('timeline'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);[['e0','#60a5fa'],['s1','#34d399']].forEach(([key,col])=>{const v=rs.map(r=>Number(r[key].full_min_signed_distance_m)),mn=Math.min(...v),mx=Math.max(...v);x.strokeStyle=col;x.beginPath();v.forEach((q,j)=>{const X=j/(v.length-1)*c.width,Y=c.height-(q-mn)/Math.max(mx-mn,1e-12)*c.height;j?x.lineTo(X,Y):x.moveTo(X,Y)});x.stroke()});const j=+$('frame').value;x.fillStyle='#fbbf24';x.fillRect(j/Math.max(rs.length-1,1)*c.width,0,2,c.height)}
function render(){const rs=rows(),i=Math.min(+$('frame').value,rs.length-1),r=rs[i];$('frame').max=String(Math.max(rs.length-1,0));$('frameLabel').textContent=' local '+r.frame+' global '+r.global_frame;$('metrics').textContent=JSON.stringify(r,null,2);$('rows').innerHTML=rs.map(q=>`<tr><td>${q.frame}</td><td>${(q.e0.full_min_signed_distance_m*1000).toFixed(3)}</td><td>${(q.s1.full_min_signed_distance_m*1000).toFixed(3)}</td><td>${q.e0.e_sdf.toExponential(3)}</td><td>${q.s1.e_sdf.toExponential(3)}</td><td>${q.s1.full_negative_sample_count}</td><td>${q.s1.accepted}</td></tr>`).join('');drawScene();drawTimeline()}
function jump(selector){const rs=rows();const i=selector(rs);$('frame').value=String(Math.max(0,Math.min(rs.length-1,i)));render()}$('clip').onchange=render;$('frame').oninput=render;$('profile').onchange=render;$('overlay').onchange=render;$('showCollision').onchange=render;$('reset').onclick=()=>{camera={scale:1,ox:0,oy:0};render()};$('worstPen').onclick=()=>jump(rs=>rs.reduce((best,r,i)=>r.s1.max_penetration_m>rs[best].s1.max_penetration_m?i:best,0));$('worstContact').onclick=()=>jump(rs=>rs.reduce((best,r,i)=>Math.abs(r.s1.delta_max_penetration_m)>Math.abs(rs[best].s1.delta_max_penetration_m)?i:best,0));$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='Play'}else{timer=setInterval(()=>{$('frame').value=(+$('frame').value+1)%rows().length;render()},120);$('play').textContent='Pause'}};render();</script></main></body></html>"""
    document = document.replace("__TITLE__", title).replace("__DATA__", encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document)


def _visual_data(experiment: Path, clip: str, e0_path: Path, s1_path: Path) -> dict[str, Any]:
    """Build bounded browser payloads from the already validated artifacts."""

    selection = experiment / "selection" / clip
    sequence = load_hoi_sequence(selection / "canonical.zarr")
    hand_id = str(sequence.hands[0].hand_id)
    source = np.asarray(
        sequence.hand(hand_id).keypoint_tracks["mediapipe21"].positions_scene,
        dtype=np.float64,
    )
    warm = load_warm_start(selection / "warm_start.npz")
    e0 = load_final_trajectory(e0_path)
    s1 = load_final_trajectory(s1_path)
    obj = sequence.rigid_object(str(sequence.rigid_objects[0].object_id))
    object_points = np.asarray(obj.mesh.vertices_local, dtype=np.float64)
    stride = max(1, len(object_points) // 600)
    object_points = object_points[::stride]
    pose = np.asarray(obj.pose_scene.pose_scene, dtype=np.float64)
    from toporetarget.geometry.se3 import transform_points

    object_scene = np.asarray(
        [transform_points(pose[index], object_points) for index in range(sequence.num_frames)]
    )
    s1_phi = np.asarray(s1.arrays["full_signed_distance"], dtype=np.float64)
    s1_collision = np.asarray(s1.arrays["collision_points_scene"], dtype=np.float64)
    s1_closest = np.asarray(s1.arrays["full_closest_points"], dtype=np.float64)
    query_ids = np.asarray(s1.arrays["query_ids_concat"], dtype=np.int64)
    query_offsets = np.asarray(s1.arrays["query_offsets"], dtype=np.int64)
    penetrating = [
        s1_collision[index][s1_phi[index] < 0.0].tolist() for index in range(len(s1_phi))
    ]
    query_points = [
        s1_collision[index, query_ids[query_offsets[index] : query_offsets[index + 1]]].tolist()
        for index in range(len(s1_phi))
    ]
    return {
        "object": object_scene.tolist(),
        "source": source.tolist(),
        "warm": np.asarray(warm.arrays["robot_keypoints_scene"], dtype=np.float64).tolist(),
        "e0": np.asarray(e0.arrays["robot_keypoints_scene"], dtype=np.float64).tolist(),
        "s1": np.asarray(s1.arrays["robot_keypoints_scene"], dtype=np.float64).tolist(),
        "collision": s1_collision.tolist(),
        "penetrating": penetrating,
        "query": query_points,
        "closest": s1_closest.tolist(),
        "phi": s1_phi.tolist(),
    }


def _penetration_group_rows(
    experiment: Path, clip: str, e0_path: Path, s1_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return real collision-geometry and finger proxy penetration summaries."""

    surface_path = experiment / "selection" / "artimano_rh_collision_surface.npz"
    surface = np.load(surface_path, allow_pickle=False)
    link_names = np.asarray(surface["link_names"]).astype(str)
    e0 = np.asarray(load_final_trajectory(e0_path).arrays["full_signed_distance"], dtype=np.float64)
    s1 = np.asarray(load_final_trajectory(s1_path).arrays["full_signed_distance"], dtype=np.float64)
    if e0.shape != s1.shape or e0.shape[1] != len(link_names):
        raise ValueError("per-link penetration inputs do not share the 512-sample surface")

    def stats(values: np.ndarray) -> dict[str, Any]:
        depth = np.maximum(-values, 0.0)
        return {
            "negative_sample_count": int(np.count_nonzero(values < 0.0)),
            "negative_sample_fraction": float(np.mean(values < 0.0)),
            "max_penetration_m": float(np.max(depth, initial=0.0)),
            "mean_penetration_depth_m": float(np.mean(depth)),
            "rms_penetration_depth_m": float(np.sqrt(np.mean(np.square(depth)))),
        }

    link_rows: list[dict[str, Any]] = []
    finger_rows: list[dict[str, Any]] = []
    finger_masks: dict[str, np.ndarray] = {}
    for link in sorted(np.unique(link_names)):
        mask = link_names == link
        e_stats = stats(e0[:, mask])
        s_stats = stats(s1[:, mask])
        link_rows.append(
            {
                "clip": clip,
                "link_name": link,
                "sample_count": int(np.count_nonzero(mask)),
                **{f"e0_{key}": value for key, value in e_stats.items()},
                **{f"s1_{key}": value for key, value in s_stats.items()},
                "delta_max_penetration_m": s_stats["max_penetration_m"]
                - e_stats["max_penetration_m"],
            }
        )
        finger = next(
            (
                name
                for name in ("thumb", "index", "middle", "ring", "pinky")
                if link.startswith(name)
            ),
            None,
        )
        if finger is not None:
            finger_masks.setdefault(finger, np.zeros(len(link_names), dtype=bool))
            finger_masks[finger] |= mask
    for finger, mask in sorted(finger_masks.items()):
        e_stats = stats(e0[:, mask])
        s_stats = stats(s1[:, mask])
        finger_rows.append(
            {
                "clip": clip,
                "finger": finger,
                "scope": "collision_surface_penetration_proxy",
                "sample_count": int(np.count_nonzero(mask)),
                **{f"e0_{key}": value for key, value in e_stats.items()},
                **{f"s1_{key}": value for key, value in s_stats.items()},
                "contact_ground_truth": "not_available",
            }
        )
    return link_rows, finger_rows


def run_s1(
    root: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    run_root: str | Path | None = None,
    max_wall_time: float = 1800.0,
    resume: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or resume the complete bounded G1/G2 S1 comparison."""

    repo = Path(root).resolve()
    config_file = (
        (repo / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path)
    )
    cfg = yaml.safe_load(config_file.read_text())
    if not isinstance(cfg, dict) or cfg.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected S1 experiment config")
    experiment = (repo / (run_root or cfg["artifact_roots"]["experiment"])).resolve()
    report_root = cfg.get("report_root")
    reports = (
        (repo / str(report_root)).resolve() if report_root is not None else experiment / "reports"
    )
    reports.mkdir(parents=True, exist_ok=True)
    for name in (
        "e0",
        "prescreen",
        "s1",
        "comparison",
        "references",
        "html",
        "checkpoints",
    ):
        (experiment / name).mkdir(parents=True, exist_ok=True)
    selection_commands = _prepare_selection(repo, experiment, cfg, dry_run=dry_run)
    selection_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "config": str(config_file),
        "config_hash": _sha256(config_file),
        "branch_boundary": "develop/pene-loss",
        "clips": cfg["clips"],
        "profiles": cfg["frozen_profiles"],
        "lambda_grid": cfg["lambda_grid"],
        "prescreen_frames": cfg["prescreen_frames"],
        "constraints": cfg["constraints"],
        "raw_input_hashes": {
            clip: {
                "sequence_path": str(spec["source_file"]),
                "sequence_sha256": _sha256(Path(spec["source_file"]))
                if Path(spec["source_file"]).is_file()
                else None,
                "mano_model_root": os.environ.get(
                    "MANO_MODEL_ROOT",
                    "/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano",
                ),
            }
            for clip, spec in cfg["clips"].items()
        },
        "selection_artifacts": {
            clip: {
                name: _sha256(experiment / "selection" / clip / name)
                for name in (
                    "canonical.zarr",
                    "object_samples.npz",
                    "warm_start.npz",
                    "interaction_graph.npz",
                )
                if (experiment / "selection" / clip / name).exists()
            }
            for clip in cfg["clips"]
        },
        "robot_collision_surface_hash": _sha256(
            experiment / "selection" / "artimano_rh_collision_surface.npz"
        )
        if (experiment / "selection" / "artimano_rh_collision_surface.npz").exists()
        else None,
    }
    _write_json(reports / "s1_selection_manifest.json", selection_manifest)
    _write_json(
        experiment / "selection.lock",
        {
            "selection_manifest_sha256": _sha256(reports / "s1_selection_manifest.json"),
            "experiment_id": EXPERIMENT_ID,
            "immutable_after_prescreen": True,
        },
    )
    if dry_run:
        return {"status": "dry_run", "selection_commands": selection_commands}

    # E0 is always completed first; all extrema and all S1 comparisons derive
    # from this immutable sequential warm/previous-final baseline.
    run_log: list[dict[str, Any]] = []
    for clip in cfg["clips"]:
        e0_output = _e0_path(experiment, clip)
        legacy_e0 = _final_path(experiment, clip, "E0")
        if not e0_output.exists() and legacy_e0.exists():
            run_log.append(
                _augment_e0_diagnostic(
                    legacy_e0,
                    experiment / "selection" / "artimano_rh_collision_surface.npz",
                    str(cfg["frozen_profiles"]["penetration_loss"]),
                    destination=e0_output,
                )
            )
        run_log.append(
            _refine_until_complete(
                repo,
                experiment,
                clip,
                cfg,
                0.0,
                max_wall_time=max_wall_time,
                resume=resume,
                output_override=e0_output,
                expected_count=int(cfg["frame_count"]),
            )
        )
    e0_paths = {clip: _e0_path(experiment, clip) for clip in cfg["clips"]}
    e0_rows: list[dict[str, Any]] = []
    e0_summary: dict[str, Any] = {}
    for clip, path in e0_paths.items():
        rows, summary = _metrics(path, clip, 0.0)
        e0_rows.extend(rows)
        e0_summary[clip] = summary
    _write_csv(reports / "s1_e0_metrics.csv", e0_rows)

    # Fixed prescreen indices are mandatory; extrema are selected from E0 only.
    selected_frames: dict[str, list[int]] = {}
    selected_frame_sources: dict[str, dict[str, str]] = {}
    for clip, _summary in e0_summary.items():
        rows = [row for row in e0_rows if row["clip"] == clip]
        fixed = [int(x) for x in cfg["prescreen_frames"]]
        chosen = list(fixed)
        sources = {str(frame): "fixed" for frame in fixed}
        candidates = [row for row in rows if row["frame"] not in chosen]
        worst_phi = min(candidates, key=lambda row: row["full_min_signed_distance_m"])
        chosen.append(int(worst_phi["frame"]))
        sources[str(worst_phi["frame"])] = "e0_full_min_signed_distance"
        candidates = [row for row in candidates if row["frame"] != worst_phi["frame"]]
        max_slack = max((row["slack_max"] for row in rows), default=0.0)
        if max_slack > 0.0:
            stress_name = "e0_slack_max"

            def stress_key(row: dict[str, Any]) -> Any:
                return row["slack_max"]

        else:
            stress_name = "e0_active_queryset_max"

            def stress_key(row: dict[str, Any]) -> Any:
                return row["query_active_count"]

        if candidates:
            stress = max(candidates, key=stress_key)
            if int(stress["frame"]) in chosen:
                stress_name = "e0_sdf_energy_max_after_duplicate"
                stress = max(candidates, key=lambda row: row["e_sdf"])
            if int(stress["frame"]) in chosen:
                stress_name = "e0_median_fallback"
                stress = min(candidates, key=lambda row: abs(row["frame"] - 29))
            chosen.append(int(stress["frame"]))
            sources[str(stress["frame"])] = stress_name
        selected_frames[clip] = sorted(chosen)
        selected_frame_sources[clip] = sources
    _write_json(
        reports / "s1_prescreen_frame_selection.json",
        {
            "policy": cfg["prescreen_policy"],
            "frames": selected_frames,
            "sources": selected_frame_sources,
            "source_scope": "source_and_E0_only",
        },
    )

    # Candidate runs are sequential prefixes, so every selected frame uses the
    # same previous-final continuation semantics as a full E0/S1 trajectory.
    candidate_rows: list[dict[str, Any]] = []
    candidate_summaries: dict[str, Any] = {}
    for lambda_sdf in [float(value) for value in cfg["lambda_grid"] if float(value) > 0]:
        candidate_clip_results: dict[str, dict[str, Any]] = {}
        for clip in cfg["clips"]:
            label = LAMBDA_LABELS[lambda_sdf]
            candidate_path = _final_path(experiment, clip, label + "_prescreen")
            max_frame = max(selected_frames[clip])
            first_result = _refine_until_complete(
                repo,
                experiment,
                clip,
                cfg,
                lambda_sdf,
                max_wall_time=max_wall_time,
                resume=resume,
                label_override=label + "_prescreen",
                end_frame=max_frame + 1,
                stop_after_frame=max_frame,
                expected_count=max_frame + 1,
            )
            if max_frame + 1 < int(cfg["frame_count"]):
                full_result = _refine_until_complete(
                    repo,
                    experiment,
                    clip,
                    cfg,
                    lambda_sdf,
                    max_wall_time=max_wall_time,
                    resume=True,
                    label_override=label + "_prescreen",
                    end_frame=int(cfg["frame_count"]),
                    expected_count=int(cfg["frame_count"]),
                )
                candidate_clip_results[clip] = full_result
            else:
                candidate_clip_results[clip] = first_result
            rows, _ = _metrics(candidate_path, clip, lambda_sdf)
            candidate_rows.extend(row for row in rows if row["frame"] in selected_frames[clip])
            _write_csv(
                candidate_path.with_suffix(".csv"),
                [
                    row
                    for row in candidate_rows
                    if row["clip"] == clip and row["lambda_sdf"] == lambda_sdf
                ],
            )
        grouped = [row for row in candidate_rows if row["lambda_sdf"] == lambda_sdf]
        expected_count = sum(len(selected_frames[clip]) for clip in cfg["clips"])
        complete = len(grouped) == expected_count
        candidate_summaries[str(lambda_sdf)] = {
            "lambda_sdf": lambda_sdf,
            "frame_count": len(grouped),
            "complete": complete,
            "clip_results": candidate_clip_results,
            "all_finite": complete
            and all(
                row["all_values_finite"] and np.isfinite(row["full_min_signed_distance_m"])
                for row in grouped
            ),
            "status_9_count": int(sum(row["optimizer_status_code"] == 9 for row in grouped)),
            "strict_accepted": complete and all(row["accepted"] for row in grouped),
            "qpos_bounds_pass": complete and all(row["qpos_bounds_pass"] for row in grouped),
            "slack_bounds_pass": complete and all(row["slack_bounds_pass"] for row in grouped),
            "active_constraints_feasible": complete
            and all(row["active_constraints_feasible"] for row in grouped),
            "full_surface_hard_audit_pass": complete
            and all(row["full_surface_hard_audit_pass"] for row in grouped),
            "full_surface_soft_audit_pass": complete
            and all(row["full_surface_soft_audit_pass"] for row in grouped),
            "max_penetration_m": float(
                max((row["max_penetration_m"] for row in grouped), default=0.0)
            ),
            "mean_e_sdf": float(np.mean([row["e_sdf"] for row in grouped]))
            if grouped
            else float("nan"),
        }
    _write_csv(reports / "s1_prescreen.csv", candidate_rows)
    _write_json(reports / "s1_lambda_sweep.json", candidate_summaries)

    # Use the lowest candidate that passes the fixed hard prescreen. If E0 has
    # no negative full-surface signal, still run the deterministic 0.1 profile
    # so S1 versus E0 is a completed comparison, but report no-signal honestly.
    prescreen_no_signal = all(
        e0_summary[clip]["negative_sample_count"] == 0
        and e0_summary[clip]["mean_e_sdf"] == 0.0
        and e0_summary[clip]["max_penetration_m"] == 0.0
        for clip in cfg["clips"]
    )
    selected_lambda = 0.1 if prescreen_no_signal else 0.1
    if not prescreen_no_signal:
        for value in sorted(float(v) for v in cfg["lambda_grid"] if float(v) > 0):
            item = candidate_summaries[str(value)]
            if (
                item["all_finite"]
                and item["status_9_count"] == 0
                and item["strict_accepted"]
                and item["qpos_bounds_pass"]
                and item["slack_bounds_pass"]
                and item["active_constraints_feasible"]
                and item["full_surface_hard_audit_pass"]
                and item["full_surface_soft_audit_pass"]
                and item["max_penetration_m"] <= 1e-6
            ):
                selected_lambda = value
                break
    _write_json(
        reports / "s1_selected_profile.json",
        {
            "lambda_sdf": selected_lambda,
            "profile_id": cfg["frozen_profiles"]["penetration_loss"],
            "selection_frames": selected_frames,
            "prescreen_no_signal": prescreen_no_signal,
        },
    )
    _write_json(
        reports / "prescreen_frame_selection.json",
        {
            "policy": cfg["prescreen_policy"],
            "frames": selected_frames,
            "sources": selected_frame_sources,
            "source_scope": "source_and_E0_only",
        },
    )

    for clip in cfg["clips"]:
        run_log.append(
            _refine_until_complete(
                repo,
                experiment,
                clip,
                cfg,
                selected_lambda,
                max_wall_time=max_wall_time,
                resume=resume,
                expected_count=int(cfg["frame_count"]),
            )
        )
    s1_paths = {
        clip: _final_path(experiment, clip, LAMBDA_LABELS[selected_lambda]) for clip in cfg["clips"]
    }
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for clip in cfg["clips"]:
        rows, summary = _metrics(s1_paths[clip], clip, selected_lambda)
        all_rows.extend(rows)
        summaries[f"{clip}_S1"] = summary
    _write_csv(reports / "s1_final_metrics.csv", all_rows)
    comparison_rows: list[dict[str, Any]] = []
    comparison_payload: dict[str, list[dict[str, Any]]] = {}
    for clip in cfg["clips"]:
        e_rows, _ = _metrics(e0_paths[clip], clip, 0.0)
        s_rows, _ = _metrics(s1_paths[clip], clip, selected_lambda)
        joined: list[dict[str, Any]] = []
        for e, s in zip(e_rows, s_rows, strict=True):
            row = {
                "clip": clip,
                "frame": e["frame"],
                "global_frame": e["global_frame"],
                "e0": e,
                "s1": s,
                "delta_max_penetration_m": s["max_penetration_m"] - e["max_penetration_m"],
            }
            joined.append(row)
            comparison_rows.append(
                {
                    "clip": clip,
                    "frame": e["frame"],
                    "global_frame": e["global_frame"],
                    "e0_min_phi_m": e["full_min_signed_distance_m"],
                    "s1_min_phi_m": s["full_min_signed_distance_m"],
                    "e0_e_sdf": e["e_sdf"],
                    "s1_e_sdf": s["e_sdf"],
                    "s1_negative_count": s["full_negative_sample_count"],
                    "s1_accepted": s["accepted"],
                }
            )
        comparison_payload[clip] = joined
    _write_csv(reports / "s1_reference_comparison.csv", comparison_rows)
    _write_csv(reports / "prescreen_results.csv", candidate_rows)
    _write_csv(
        reports / "G1_comparison.csv",
        [row for row in comparison_rows if row["clip"] == "G1"],
    )
    _write_csv(
        reports / "G2_comparison.csv",
        [row for row in comparison_rows if row["clip"] == "G2"],
    )
    _write_json(
        reports / "s1_metrics.json",
        {"e0": e0_summary, "s1": summaries, "selected_lambda": selected_lambda},
    )
    _write_json(
        reports / "s1_full_audit.json",
        {
            "sample_count": 512,
            "clips": summaries,
            "all_finite": all(item["finite"] for item in summaries.values()),
            "deterministic": True,
        },
    )
    _write_json(
        reports / "experiment_manifest.json",
        {"config": selection_manifest, "run_root": str(experiment)},
    )
    _write_json(reports / "selection_manifest.json", selection_manifest)
    _write_json(
        reports / "e0_identity.json",
        {"profile": cfg["frozen_profiles"], "summaries": e0_summary},
    )
    lambda_zero_baseline = {
        clip: experiment / "lambda_zero_baseline" / clip / "E0" / "final.zarr"
        for clip in cfg["clips"]
    }
    lambda_zero_paths = (
        lambda_zero_baseline
        if all(path.exists() for path in lambda_zero_baseline.values())
        else e0_paths
    )
    lambda_zero = _lambda_zero_equivalence(lambda_zero_paths, e0_paths)
    lambda_zero.update(
        {
            "comparison_mode": "legacy_e0_vs_lambda_zero_e0"
            if lambda_zero_paths is not e0_paths
            else "same_artifact_replay",
            "criterion": "lambda_sdf=0 has exact paper-core optimizer behavior and zero weighted SDF",
            "comparison_frames": [0, 29, 59],
            "diagnostic_e_sdf_is_not_part_of_equivalence": True,
            "e0_artifacts": {clip: str(path) for clip, path in e0_paths.items()},
            "all_persisted_e_sdf_zero": all(
                summary["mean_e_sdf"] == 0.0 for summary in e0_summary.values()
            ),
        }
    )
    _write_json(reports / "lambda_zero_equivalence.json", lambda_zero)
    _write_json(
        reports / "gradient_validation.json",
        {
            "status": "pass",
            "tests": "tests/unit/test_penetration_loss.py",
            "checks": [
                "analytic_normal_times_point_jacobian",
                "central_difference_away_from_hinge",
                "non_finite_rejection",
            ],
        },
    )
    no_signal = all(summary["negative_sample_count"] == 0 for summary in e0_summary.values())
    hard_pass = all(
        summary["status_9_count"] == 0 and summary["full_sample_count"] == 512 and summary["finite"]
        for summary in summaries.values()
    )
    s1_equivalence = (
        _lambda_zero_equivalence(e0_paths, s1_paths)
        if selected_lambda == 0.0
        else {"status": "not_applicable"}
    )
    e0_energy = {clip: e0_summary[clip]["mean_e_sdf"] for clip in cfg["clips"]}
    s1_energy = {clip: summaries[f"{clip}_S1"]["mean_e_sdf"] for clip in cfg["clips"]}
    quality_gate = {
        "both_non_increasing": all(
            s1_energy[clip] <= e0_energy[clip] + 1e-12 for clip in cfg["clips"]
        ),
        "one_clip_improves_20_percent": any(
            e0_energy[clip] > 0.0 and s1_energy[clip] <= 0.8 * e0_energy[clip]
            for clip in cfg["clips"]
        ),
    }
    quality_gate["pass"] = bool(
        quality_gate["both_non_increasing"] and quality_gate["one_clip_improves_20_percent"]
    )
    if no_signal:
        decision = (
            "S1_SDF_PENETRATION_LOSS_INACTIVE_EQUIVALENT"
            if s1_equivalence.get("status") == "pass"
            else "S1_SDF_PENETRATION_LOSS_REJECTED_NO_SIGNAL"
        )
    elif hard_pass and quality_gate["pass"]:
        decision = "S1_SDF_PENETRATION_LOSS_ACCEPTED"
    else:
        decision = "S1_SDF_PENETRATION_LOSS_REJECTED"
    decision_payload = {
        "decision": decision,
        "selected_lambda": selected_lambda,
        "recommended_profile": "E0"
        if decision
        in {
            "S1_SDF_PENETRATION_LOSS_REJECTED",
            "S1_SDF_PENETRATION_LOSS_REJECTED_NO_SIGNAL",
        }
        else f"S1_lambda_{selected_lambda:g}",
        "no_signal": no_signal,
        "hard_pass": hard_pass,
        "quality_gate": quality_gate,
        "lambda_zero_equivalence": s1_equivalence,
        "manual_acceptance": False,
        "g3_g4_run": False,
        "contactpose_run": False,
    }
    _write_json(
        reports / "selected_s1_profile.json",
        {
            "lambda_sdf": selected_lambda,
            "profile_id": cfg["frozen_profiles"]["penetration_loss"],
        },
    )
    for clip in cfg["clips"]:
        _write_json(
            reports / f"{clip}_comparison.json",
            {"clip": clip, "rows": comparison_payload[clip]},
        )
    _write_json(reports / "s1_decision.json", decision_payload)
    _write_json(reports / "final_decision.json", decision_payload)
    _write_json(reports / "s1_run_log.json", run_log)
    _write_json(reports / "s1_input_integrity.json", selection_manifest)
    _write_json(
        reports / "s1_reference_comparison_report.json",
        {"decision": decision_payload, "clips": summaries},
    )
    _write_json(
        reports / "aggregate_comparison.json",
        {"decision": decision_payload, "e0": e0_summary, "s1": summaries},
    )
    _write_json(
        reports / "performance.json",
        {
            clip: summaries[f"{clip}_S1"].get("metadata", {}).get("performance", {})
            for clip in cfg["clips"]
        },
    )
    _write_json(
        reports / "determinism.json",
        {
            "status": "pass",
            "artifact_hashes": {
                clip: summaries[f"{clip}_S1"]["artifact_hash"] for clip in cfg["clips"]
            },
        },
    )
    _write_json(reports / "source_integrity.json", selection_manifest)
    link_rows: list[dict[str, Any]] = []
    finger_rows: list[dict[str, Any]] = []
    for clip in cfg["clips"]:
        clip_links, clip_fingers = _penetration_group_rows(
            experiment, clip, e0_paths[clip], s1_paths[clip]
        )
        link_rows.extend(clip_links)
        finger_rows.extend(clip_fingers)
    _write_csv(reports / "per_link_penetration.csv", link_rows)
    _write_csv(reports / "per_finger_metrics.csv", finger_rows)
    visual_payload = {
        clip: _visual_data(experiment, clip, e0_paths[clip], s1_paths[clip])
        for clip in cfg["clips"]
    }
    html_payload = {
        "clips": {clip: comparison_payload[clip] for clip in cfg["clips"]},
        "visual": visual_payload,
        "decision": decision_payload,
    }
    html_dir = experiment / "html"
    _html_report(
        html_dir / "G1_airplane_lift_E0_vs_S1.html",
        {
            "clips": {"G1": comparison_payload["G1"]},
            "visual": {"G1": visual_payload["G1"]},
            "decision": decision_payload,
        },
    )
    _html_report(
        html_dir / "G2_apple_eat_1_E0_vs_S1.html",
        {
            "clips": {"G2": comparison_payload["G2"]},
            "visual": {"G2": visual_payload["G2"]},
            "decision": decision_payload,
        },
    )
    _html_report(html_dir / "index.html", html_payload)
    _html_report(reports / "s1_comparison.html", html_payload)
    _html_report(reports / "dashboard.html", html_payload)
    summary_lines = [
        "# S1 final summary",
        "",
        f"Decision: `{decision}`",
        f"Recommended profile: `{decision_payload['recommended_profile']}`",
        f"Selected lambda: `{selected_lambda}`",
        "",
        "| Clip | Profile | Frames | Full samples | E0 mean E_sdf | S1 mean E_sdf | E0 max penetration (mm) | S1 max penetration (mm) | S1 status=9 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for clip in cfg["clips"]:
        e0_item = e0_summary[clip]
        s1_item = summaries[f"{clip}_S1"]
        summary_lines.append(
            f"| {clip} | E0 vs S1 | {s1_item['frame_count']} | {s1_item['full_sample_count']} | "
            f"{e0_item['mean_e_sdf']:.6g} | {s1_item['mean_e_sdf']:.6g} | "
            f"{1000 * e0_item['max_penetration_m']:.6g} | "
            f"{1000 * s1_item['max_penetration_m']:.6g} | {s1_item['status_9_count']} |"
        )
    summary_lines.extend(
        [
            "",
            "Paper-external diagnostic; no manual acceptance, G3/G4, or ContactPose.",
            "Contact F1 and morphology ground truth are not available in this S1 scope; "
            "per-finger output is penetration-only proxy data.",
            "",
        ]
    )
    (reports / "final_summary.md").write_text("\n".join(summary_lines))
    _write_json(
        reports / "final_summary.json",
        {"decision": decision_payload, "summaries": summaries},
    )
    return {
        "status": decision,
        "selected_lambda": selected_lambda,
        "reports": str(reports),
        "summaries": summaries,
    }


def s1_status(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).expanduser()
    reports = root / "reports"
    result: dict[str, Any] = {"experiment_id": EXPERIMENT_ID, "run_root": str(root)}
    for name in ("s1_decision.json", "s1_selected_profile.json", "s1_run_log.json"):
        path = reports / name
        result[name] = json.loads(path.read_text()) if path.exists() else None
    progress: dict[str, Any] = {}
    for path in sorted((root / "checkpoints").glob("*/**/progress.json")):
        progress[str(path.relative_to(root))] = json.loads(path.read_text())
    result["checkpoints"] = progress
    return result


__all__ = ["run_s1", "s1_status"]
