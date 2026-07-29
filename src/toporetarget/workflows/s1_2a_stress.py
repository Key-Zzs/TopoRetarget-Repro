"""S1.2A E0 penetration-active stress discovery and controlled evaluation.

This module is a paper-external experiment lane.  It never changes the S1
loss, solver, query set, collision surface, or historical S1 artifacts.  The
only selection input before the E0 probe is the source GRAB record; after the
probe, ranking uses E0 robot collision penetration only.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.data.storage import load_hoi_sequence, save_hoi_sequence
from toporetarget.geometry.signed_distance.reference import ReferenceSignedDistanceBackend
from toporetarget.quality.html import render_clip_html, smoke_html
from toporetarget.quality.schema import ClipSpec
from toporetarget.retarget.artifacts import load_warm_start
from toporetarget.retarget.final_refinement import (
    ConvexHullSignedDistanceBackend,
    load_final_trajectory,
)
from toporetarget.workflows import s1_penetration as s1
from toporetarget.workflows.s1_signal_rich import (
    _backend_metrics,
    _scan_one,
    _write_csv,
    _write_json,
)
from toporetarget.workflows.s1_signal_rich import _stable_hash as stable_hash

EXPERIMENT_ID = "s1_2a_e0_penetration_stress_v1"
DEFAULT_CONFIG = Path("configs/experiments/s1_2a_e0_penetration_stress_v1.yaml")
EXCLUDED = {
    "s1/airplane_lift",
    "s1/apple_eat_1",
    "s1/banana_lift",
    "s1/alarmclock_lift",
}
PROFILE = "dense_squared_hinge_deadzone1mm_v2"
LAMBDA_SDF = 0.1
DEAD_ZONE_M = 0.001
PROBE_FRAMES = (0, 30, 59)
FINGER_KEYPOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FAILURE_CLASSES = {
    "graph": "GRAPH_FAILURE",
    "solver": "SOLVER_FAILURE",
    "sdf": "SDF_FAILURE",
    "timeout": "TIMEOUT",
    "artifact": "INVALID_ARTIFACT",
}


def _sha256(path: str | Path) -> str | None:
    source = Path(path)
    if not source.exists():
        return None
    digest = hashlib.sha256()
    if source.is_file():
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    else:
        for child in sorted(item for item in source.rglob("*") if item.is_file()):
            digest.update(str(child.relative_to(source)).encode())
            digest.update(b"\0")
            digest.update((_sha256(child) or "").encode())
    return digest.hexdigest()


def _cfg(repo: Path, path: str | Path) -> tuple[dict[str, Any], Path]:
    config = Path(path)
    config = config if config.is_absolute() else repo / config
    value = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"unexpected S1.2A configuration: {config}")
    return value, config.resolve()


def _candidate_config(
    cfg: dict[str, Any], row: dict[str, Any], *, frame_count: int
) -> dict[str, Any]:
    return {
        "experiment_id": "s1_sdf_penetration_loss_v1",
        "robot": cfg["robot_id"],
        "robot_asset_root": cfg["robot_asset_root"],
        "hand": "right",
        "frame_count": frame_count,
        "native_fps": float(cfg["native_fps"]),
        "clips": {
            "candidate": {
                "sequence_id": row["sequence"],
                "source_file": row["source_file"],
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "object_id": "primary",
                "object_mesh": row["object"],
            }
        },
        "frozen_profiles": dict(cfg["frozen_profiles"]),
        "full_audit": dict(cfg["full_audit"]),
    }


def _run_frozen_command(
    repo: Path,
    args: list[str],
    cfg: dict[str, Any],
    *,
    timeout_seconds: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    limits = cfg.get("resource_limits", {})
    env_map = {
        "omp_threads": "OMP_NUM_THREADS",
        "mkl_threads": "MKL_NUM_THREADS",
        "openblas_threads": "OPENBLAS_NUM_THREADS",
        "numexpr_threads": "NUMEXPR_NUM_THREADS",
    }
    previous: dict[str, str | None] = {}
    for config_key, env_key in env_map.items():
        if config_key in limits:
            previous[env_key] = os.environ.get(env_key)
            os.environ[env_key] = str(limits[config_key])
    try:
        if timeout_seconds is not None:
            command = [sys.executable, "-m", "toporetarget", *args]
            env = os.environ.copy()
            env["PYTHONNOUSERSITE"] = "1"
            env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
            try:
                completed = subprocess.run(
                    command,
                    cwd=repo,
                    env=env,
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                return {
                    "command": command,
                    "status": "failed",
                    "returncode": -9,
                    "stdout_tail": str(exc.stdout or "")[-2000:],
                    "stderr_tail": "TIMEOUT: external wall-time guard",
                }
            return {
                "command": command,
                "status": "pass" if completed.returncode == 0 else "failed",
                "returncode": int(completed.returncode),
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        return s1._run_command(repo, args, **kwargs)
    finally:
        for env_key, value in previous.items():
            if value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = value


def _take_frames(sequence: Any, indices: tuple[int, ...]) -> Any:
    """Create a lossless canonical sub-cache for fixed non-contiguous frames."""

    result = copy.deepcopy(sequence)
    old_count = sequence.num_frames
    selected = np.asarray(indices, dtype=np.int64)
    result.metadata.timestamps = result.metadata.timestamps[selected]
    result.metadata.num_frames = len(selected)
    result.metadata.metadata = dict(result.metadata.metadata)
    result.metadata.metadata["selected_source_local_frames"] = list(indices)

    def take(value: Any) -> Any:
        array = np.asarray(value)
        if array.ndim and array.shape[0] == old_count:
            return array[selected]
        return value

    for hand in result.hands:
        hand.wrist_pose_scene.pose_scene = take(hand.wrist_pose_scene.pose_scene)
        if hand.wrist_pose_scene.valid is not None:
            hand.wrist_pose_scene.valid = take(hand.wrist_pose_scene.valid)
        if hand.valid is not None:
            hand.valid = take(hand.valid)
        if hand.vertices_scene is not None:
            hand.vertices_scene = take(hand.vertices_scene)
        if hand.mano_parameters is not None:
            for name in ("global_orient_aa", "hand_pose_aa", "transl", "betas"):
                value = getattr(hand.mano_parameters, name)
                if value is not None:
                    setattr(hand.mano_parameters, name, take(value))
        for track in hand.keypoint_tracks.values():
            track.positions_scene = take(track.positions_scene)
            if track.valid is not None:
                track.valid = take(track.valid)
            if track.confidence is not None:
                track.confidence = take(track.confidence)
    for obj in result.rigid_objects:
        obj.pose_scene.pose_scene = take(obj.pose_scene.pose_scene)
        if obj.pose_scene.valid is not None:
            obj.pose_scene.valid = take(obj.pose_scene.valid)
        if obj.valid is not None:
            obj.valid = take(obj.valid)
    for obj in result.articulated_objects:
        for part in obj.parts:
            part.pose_scene.pose_scene = take(part.pose_scene.pose_scene)
            if part.pose_scene.valid is not None:
                part.pose_scene.valid = take(part.pose_scene.valid)
            if part.valid is not None:
                part.valid = take(part.valid)
    for contact in result.contacts:
        contact.valid = take(contact.valid)
        for name in ("labels", "vertex_associations", "binary", "semantic_labels"):
            value = getattr(contact, name)
            if value is not None:
                setattr(contact, name, take(value))
    result.validate()
    return result


def _failure_class(text: str) -> str:
    lowered = text.lower()
    if "timeout" in lowered or "timed out" in lowered or "wall time" in lowered:
        return FAILURE_CLASSES["timeout"]
    if "graph" in lowered or "delaunay" in lowered or "qhull" in lowered:
        return FAILURE_CLASSES["graph"]
    if "sdf" in lowered or "signed distance" in lowered or "winding" in lowered:
        return FAILURE_CLASSES["sdf"]
    if "solver" in lowered or "slsqp" in lowered or "status" in lowered:
        return FAILURE_CLASSES["solver"]
    return FAILURE_CLASSES["artifact"]


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metrics = row.get("e0_metrics", {})
    return (
        -int(metrics.get("frames_gt_1mm", 0)),
        -float(metrics.get("mean_excess_penetration_m", 0.0)),
        -float(metrics.get("max_penetration_m", 0.0)),
        -int(metrics.get("active_link_count", 0)),
        str(row.get("sequence", "")),
    )


def _probe_metrics(path: Path, surface_path: Path) -> dict[str, Any]:
    final = load_final_trajectory(path)
    phi = np.asarray(final.arrays["full_signed_distance"], dtype=np.float64)
    if phi.ndim != 2 or phi.shape[1] != 512:
        raise ValueError(f"invalid E0 probe full surface shape: {phi.shape}")
    depth = np.maximum(-phi, 0.0)
    per_frame = np.max(depth, axis=1)
    surface = np.load(surface_path, allow_pickle=False)
    links = np.asarray(surface["link_names"]).astype(str)
    per_link, per_finger = _group_penetration_metrics(depth, links)
    return {
        "frame_count": int(phi.shape[0]),
        "full_sample_count": int(phi.shape[1]),
        "frames_gt_1mm": int(np.count_nonzero(per_frame > DEAD_ZONE_M)),
        "frames_gt_2mm": int(np.count_nonzero(per_frame > 0.002)),
        "mean_excess_penetration_m": float(np.mean(np.maximum(per_frame - DEAD_ZONE_M, 0.0))),
        "max_penetration_m": float(np.max(depth)),
        "rms_penetration_m": float(np.sqrt(np.mean(np.square(depth)))),
        "penetration_energy": float(np.mean(np.square(np.maximum(depth - DEAD_ZONE_M, 0.0)))),
        "negative_sample_fraction": float(np.mean(phi < 0.0)),
        "active_link_count": int(
            sum(np.any(phi[:, links == link] < -DEAD_ZONE_M) for link in np.unique(links))
        ),
        "per_link_penetration": per_link,
        "per_finger_penetration": per_finger,
        "strict_accepted_count": int(np.count_nonzero(final.arrays["accepted"])),
        "status_9_count": int(
            np.count_nonzero(np.asarray(final.arrays.get("optimizer_status_code", [])) == 9)
        ),
        "status_0_count": int(
            np.count_nonzero(np.asarray(final.arrays.get("optimizer_status_code", [])) == 0)
        ),
        "checkpoint_complete": bool(
            json.loads((path.parent / "checkpoint" / "progress.json").read_text()).get("status")
            == "complete"
        )
        if (path.parent / "checkpoint" / "progress.json").is_file()
        else False,
        "finite": bool(np.all(np.isfinite(phi))),
        "artifact_hash": _sha256(path),
    }


def _strict_summary(metrics: dict[str, Any], expected: int) -> bool:
    return bool(
        metrics["frame_count"] == expected
        and metrics["full_sample_count"] == 512
        and metrics["strict_accepted_count"] == expected
        and metrics["status_9_count"] == 0
        and metrics["finite"]
    )


def _group_penetration_metrics(
    depth: np.ndarray, links: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    def metrics_for(mask: np.ndarray) -> dict[str, Any]:
        values = depth[:, mask]
        frame_max = np.max(values, axis=1)
        return {
            "mean_excess_penetration_m": float(np.mean(np.maximum(frame_max - DEAD_ZONE_M, 0.0))),
            "max_penetration_m": float(np.max(values)),
            "frames_gt_1mm": int(np.count_nonzero(frame_max > DEAD_ZONE_M)),
        }

    per_link = {link: metrics_for(links == link) for link in sorted(np.unique(links).tolist())}
    finger_names = ("thumb", "index", "middle", "ring", "pinky")
    per_finger = {
        finger: metrics_for(np.char.startswith(links, finger))
        for finger in finger_names
        if np.any(np.char.startswith(links, finger))
    }
    per_finger["palm"] = metrics_for(links == "palm")
    return per_link, per_finger


def _prepare_robot_surface(repo: Path, experiment: Path, cfg: dict[str, Any]) -> Path:
    output = experiment / "selection" / "artimano_rh_collision_surface.npz"
    if not output.exists():
        _run_frozen_command(
            repo,
            [
                "geometry",
                "sample-robot",
                "--robot",
                cfg["robot_id"],
                "--profile",
                cfg["frozen_profiles"]["collision_surface"],
                "--asset-root",
                str(repo / cfg["robot_asset_root"]),
                "--output",
                str(output),
                "--report",
                str(output.with_suffix(".json")),
                "--force",
            ],
            cfg,
        )
    return output


def _prepare_inputs(
    repo: Path,
    experiment: Path,
    cfg: dict[str, Any],
    row: dict[str, Any],
    candidate_id: str,
    *,
    probe: bool,
) -> dict[str, Path]:
    root = experiment / "e0_probe" / candidate_id if probe else experiment / "stress" / candidate_id
    root.mkdir(parents=True, exist_ok=True)
    full_canonical = root / "source_canonical.zarr" if probe else root / "canonical.zarr"
    canonical = root / "canonical.zarr"
    if not full_canonical.exists():
        _run_frozen_command(
            repo,
            [
                "data",
                "convert",
                "--dataset",
                "grab",
                "--sequence-path",
                str(row["source_file"]),
                "--grab-root",
                str(Path(cfg["grab_root"]).resolve()),
                "--mano-model-root",
                str(Path(cfg["mano_root"]).resolve()),
                "--hands",
                "right",
                "--contact-mode",
                "semantic",
                "--include-mediapipe21",
                "--start-frame",
                str(row["start_frame"]),
                "--end-frame",
                str(row["end_frame"]),
                "--output",
                str(full_canonical),
                "--force",
            ],
            cfg,
        )
    if probe and not canonical.exists():
        sequence = load_hoi_sequence(full_canonical)
        save_hoi_sequence(_take_frames(sequence, PROBE_FRAMES), canonical)
    samples = root / "object_samples.npz"
    if not samples.exists():
        _run_frozen_command(
            repo,
            [
                "geometry",
                "sample-object",
                "--canonical",
                str(canonical),
                "--object-id",
                "primary",
                "--profile",
                "paper_strict_area_uniform",
                "--output",
                str(samples),
                "--report",
                str(root / "object_samples.json"),
                "--force",
            ],
            cfg,
        )
    warm = root / "warm_start.npz"
    if not warm.exists():
        _run_frozen_command(
            repo,
            [
                "retarget",
                "warm-start",
                "--canonical",
                str(canonical),
                "--hand",
                "right",
                "--robot",
                cfg["robot_id"],
                "--end-frame",
                str(3 if probe else 60),
                "--frame-profile",
                cfg["frozen_profiles"]["frame"],
                "--bone-profile",
                cfg["frozen_profiles"]["bone"],
                "--solver-profile",
                cfg["frozen_profiles"]["warm_solver"],
                "--asset-root",
                str(repo / cfg["robot_asset_root"]),
                "--output",
                str(warm),
                "--force",
            ],
            cfg,
        )
    graph = root / "interaction_graph.npz"
    if not graph.exists():
        _run_frozen_command(
            repo,
            [
                "retarget",
                "build-interaction-graph",
                "--canonical",
                str(canonical),
                "--hand",
                "right",
                "--object-samples",
                str(samples),
                "--delaunay-profile",
                cfg["frozen_profiles"]["graph"],
                "--end-frame",
                str(3 if probe else 60),
                "--output",
                str(graph),
                "--report",
                str(root / "interaction_graph.json"),
                "--force",
            ],
            cfg,
        )
    return {"root": root, "canonical": canonical, "warm": warm, "graph": graph, "samples": samples}


def _refine(
    repo: Path,
    experiment_root: Path,
    paths: dict[str, Path],
    cfg: dict[str, Any],
    *,
    label: str,
    lambda_sdf: float,
    expected: int,
    resume: bool,
) -> dict[str, Any]:
    output = paths["root"] / label / "final.zarr"
    checkpoint = paths["root"] / label / "checkpoint"
    if output.exists():
        try:
            metrics = _probe_metrics(
                output, experiment_root / "selection" / "artimano_rh_collision_surface.npz"
            )
            if _strict_summary(metrics, expected):
                return {"status": "reused", "output": str(output), "metrics": metrics}
        except (OSError, ValueError, RuntimeError, KeyError):
            pass
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "retarget",
        "refine",
        "--canonical",
        str(paths["canonical"]),
        "--warm-start",
        str(paths["warm"]),
        "--graph",
        str(paths["graph"]),
        "--robot",
        cfg["robot_id"],
        "--collision-samples",
        str(experiment_root / "selection" / "artimano_rh_collision_surface.npz"),
        "--query-profile",
        cfg["frozen_profiles"]["query"],
        "--solver-profile",
        cfg["frozen_profiles"]["solver"],
        "--execution-profile",
        cfg["frozen_profiles"]["execution"],
        "--penetration-loss-profile",
        PROFILE,
        "--validation-sdf-backend",
        cfg["full_audit"]["signed_distance_backend"],
        "--lambda-sdf",
        str(lambda_sdf),
        "--start-frame",
        "0",
        "--end-frame",
        str(expected),
        "--checkpoint-root",
        str(checkpoint),
        "--max-wall-time",
        str(cfg["resource_limits"]["max_wall_time_per_clip"]),
        "--progress-json",
        str(checkpoint / "progress.json"),
        "--progress-log",
        str(checkpoint / "logs" / "progress.jsonl"),
        "--output",
        str(output),
        "--asset-root",
        str(repo / cfg["robot_asset_root"]),
    ]
    if resume:
        args.append("--resume")
    # A full 60-frame run may legitimately stop at the configured wall-time
    # boundary with a complete checkpoint but without the final zarr.  Resume
    # the same immutable checkpoint until it materializes or progress is
    # demonstrably stagnant; this is execution resilience, not per-clip tuning.
    max_sessions = (
        1 if expected <= 3 else int(cfg.get("resource_limits", {}).get("max_resume_sessions", 64))
    )
    last_marker: tuple[Any, ...] | None = None
    stagnant_sessions = 0
    result: dict[str, Any] = {"status": "not_run"}
    for _ in range(max_sessions):
        result = _run_frozen_command(
            repo,
            args,
            cfg,
            allow_failure=True,
            timeout_seconds=float(cfg["resource_limits"]["max_wall_time_per_clip"]),
        )
        result["output"] = str(output)
        if output.exists():
            try:
                result["metrics"] = _probe_metrics(
                    output,
                    experiment_root / "selection" / "artimano_rh_collision_surface.npz",
                )
                result["strict_accepted"] = _strict_summary(result["metrics"], expected)
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                result["artifact_error"] = f"{type(exc).__name__}:{exc}"
        if result.get("strict_accepted", False):
            return result
        progress_path = checkpoint / "progress.json"
        progress: dict[str, Any] = {}
        if progress_path.is_file():
            try:
                loaded = json.loads(progress_path.read_text())
                if isinstance(loaded, dict):
                    progress = loaded
            except (OSError, json.JSONDecodeError):
                pass
        marker = (
            progress.get("status"),
            progress.get("next_frame"),
            progress.get("last_accepted_frame"),
            progress.get("remaining_frames"),
        )
        if marker == last_marker:
            stagnant_sessions += 1
        else:
            stagnant_sessions = 0
            last_marker = marker
        if stagnant_sessions >= 2:
            break
        if expected <= 3:
            break
    result["strict_accepted"] = False
    result["failure_class"] = _failure_class(
        json.dumps(result, sort_keys=True, default=str)
        + json.dumps(progress if "progress" in locals() else {}, sort_keys=True)
    )
    return result


def scan_source_candidates(
    repo: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    experiment_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo).resolve()
    cfg, config_file = _cfg(root, config_path)
    experiment = (root / str(experiment_root or cfg["output_root"])).resolve()
    reports, selection = experiment / "reports", experiment / "selection"
    reports.mkdir(parents=True, exist_ok=True)
    selection.mkdir(parents=True, exist_ok=True)
    output = selection / "source_candidates.json"
    if output.is_file() and json.loads(output.read_text()).get("config_hash") == _sha256(
        config_file
    ):
        return json.loads(output.read_text())
    grab_root = Path(cfg["grab_root"]).resolve()
    legacy = root / ".local/experiments/s1_1_signal_rich_grab_v1/reports/source_candidate_scan.json"
    legacy_payload = json.loads(legacy.read_text()) if legacy.is_file() else None
    if legacy_payload and legacy_payload.get("grab_root") == str(grab_root):
        # The previous S1.1 run already audited this immutable raw pool.  Reuse
        # that source-only evidence rather than recomputing expensive NFS mesh
        # topology for every object; the fallback below is complete on a fresh
        # checkout without the legacy artifact.
        rows = [dict(row) for row in legacy_payload.get("rows", [])]
        reused_from = _sha256(legacy)
    else:
        files = sorted((grab_root / "grab").glob("*/*.npz"))
        from toporetarget.data.contacts.grab import load_grab_contact_mapping

        mapping = load_grab_contact_mapping()
        rows = []
        for path in files:
            row = _scan_one(path, grab_root, cfg, mapping)
            rows.append(row)
        reused_from = None
    for row in rows:
        if row.get("sequence") in EXCLUDED:
            row["eligible"] = False
            row["exclusion_reason"] = "frozen G1/G2/G3/G4 exclusion"
        row["source_only_selection"] = True
        row["s1_or_e0_used_for_selection"] = False
    eligible = [row for row in rows if row.get("eligible")]
    eligible.sort(
        key=lambda row: (-float(row.get("source_score", -math.inf)), str(row["sequence"]))
    )
    payload = {
        "schema_version": "toporetarget.s1_2a.source_candidates.v1",
        "experiment_id": EXPERIMENT_ID,
        "config": str(config_file),
        "config_hash": _sha256(config_file),
        "grab_root": str(grab_root),
        "candidate_count": len(rows),
        "source_valid_count": len(eligible),
        "excluded_sequences": sorted(EXCLUDED),
        "rows": rows,
        "selection_scope": "source_only",
        "s1_or_e0_used": False,
        "scan_order": "lexicographic native GRAB path",
        "reused_source_only_scan_hash": reused_from,
    }
    _write_json(output, payload)
    _write_csv(selection / "source_candidates.csv", rows)
    _write_json(reports / "discovery_summary.json", {"source_scan": payload})
    return payload


def _warm_probe_one(
    repo: Path, experiment: Path, cfg: dict[str, Any], row: dict[str, Any], candidate_id: str
) -> dict[str, Any]:
    path = experiment / "e0_probe" / candidate_id / "warm_result.json"
    if path.is_file():
        return json.loads(path.read_text())
    try:
        paths = _prepare_inputs(repo, experiment, cfg, row, candidate_id, probe=True)
        warm = load_warm_start(paths["warm"])
        arrays = warm.arrays
        success = np.asarray(arrays.get("solver_success", np.ones(3, dtype=bool)), dtype=bool)
        finite = all(np.all(np.isfinite(np.asarray(value))) for value in arrays.values())
        result = {
            "candidate_id": candidate_id,
            "sequence": row["sequence"],
            "subject": row.get("subject"),
            "object": row.get("object"),
            "frame_range": [row["start_frame"], row["end_frame"]],
            "selected_source_local_frames": list(PROBE_FRAMES),
            "selected_source_global_frames": [
                int(row["start_frame"]) + item for item in PROBE_FRAMES
            ],
            "source_row": row,
            "paths": {key: str(value) for key, value in paths.items()},
            "warm_pass": bool(len(success) == 3 and np.all(success) and finite),
            "solver_success_count": int(np.count_nonzero(success)),
            "finite": finite,
            "artifact_hash": _sha256(paths["warm"]),
            "status": "pass" if len(success) == 3 and np.all(success) and finite else "WARM_FAILED",
        }
    except Exception as exc:  # every candidate is isolated
        result = {
            "candidate_id": candidate_id,
            "sequence": row.get("sequence"),
            "subject": row.get("subject"),
            "object": row.get("object"),
            "source_row": row,
            "warm_pass": False,
            "status": "WARM_FAILED",
            "failure": f"{type(exc).__name__}:{exc}",
        }
    _write_json(path, result)
    return result


def run_warm_probes(
    repo: str | Path, config_path: str | Path, experiment_root: str | Path
) -> dict[str, Any]:
    root = Path(repo).resolve()
    cfg, _ = _cfg(root, config_path)
    experiment = Path(experiment_root).resolve()
    scan = json.loads((experiment / "selection" / "source_candidates.json").read_text())
    rows = sorted(
        [row for row in scan["rows"] if row.get("eligible")],
        key=lambda row: (str(row.get("source_file")), int(row.get("start_frame", 0))),
    )
    results = []
    for index, row in enumerate(rows, 1):
        results.append(_warm_probe_one(root, experiment, cfg, row, f"W{index:04d}"))
        if sum(bool(item.get("warm_pass")) for item in results) >= int(
            cfg["max_warm_pass_candidates"]
        ):
            break
    payload = {
        "schema_version": "toporetarget.s1_2a.warm_candidates.v1",
        "total_source_valid": len(rows),
        "warm_pass": sum(bool(item.get("warm_pass")) for item in results),
        "warm_failed": sum(not bool(item.get("warm_pass")) for item in results),
        "warm_not_attempted": len(rows) - len(results),
        "warm_lane_bounded": True,
        "warm_pass_limit": int(cfg["max_warm_pass_candidates"]),
        "rows": results,
    }
    _write_json(experiment / "selection" / "warm_candidates.json", payload)
    _write_csv(experiment / "selection" / "warm_candidates.csv", results)
    return payload


def run_e0_probes(
    repo: str | Path, config_path: str | Path, experiment_root: str | Path, *, resume: bool = True
) -> dict[str, Any]:
    root = Path(repo).resolve()
    cfg, _ = _cfg(root, config_path)
    experiment = Path(experiment_root).resolve()
    warm = json.loads((experiment / "selection" / "warm_candidates.json").read_text())
    candidates = [item for item in warm["rows"] if item.get("warm_pass")][
        : int(cfg["max_e0_probe_candidates"])
    ]
    candidates.sort(
        key=lambda item: (
            int(item.get("source_row", {}).get("object_mesh_audit", {}).get("face_count", 10**12)),
            str(item.get("object", "")),
            str(item.get("sequence", "")),
            str(item.get("candidate_id", "")),
        )
    )
    _prepare_robot_surface(root, experiment, cfg)
    rows: list[dict[str, Any]] = []
    for item in candidates:
        row = item["source_row"]
        candidate_id = str(item["candidate_id"])
        result_path = experiment / "e0_probe" / candidate_id / "e0_result.json"
        if resume and result_path.is_file():
            rows.append(json.loads(result_path.read_text()))
            continue
        try:
            paths = _prepare_inputs(root, experiment, cfg, row, candidate_id, probe=True)
            result = _refine(
                root,
                experiment,
                paths,
                cfg,
                label="E0",
                lambda_sdf=0.0,
                expected=3,
                resume=resume,
            )
            if result.get("metrics") is None:
                raise RuntimeError(result.get("artifact_error", "E0 probe artifact missing"))
            metrics = result["metrics"]
            result = {
                "candidate_id": candidate_id,
                "sequence": row["sequence"],
                "subject": row.get("subject"),
                "object": row.get("object"),
                "frame_range": [row["start_frame"], row["end_frame"]],
                "selected_source_local_frames": list(PROBE_FRAMES),
                "selected_source_global_frames": [
                    int(row["start_frame"]) + item for item in PROBE_FRAMES
                ],
                "source_row": row,
                "paths": {key: str(value) for key, value in paths.items()},
                "status": "pass" if result.get("strict_accepted") else "failed",
                "strict_accepted": bool(result.get("strict_accepted")),
                "e0_metrics": metrics,
                "failure_class": result.get("failure_class"),
                "solver_result": result,
            }
        except Exception as exc:
            result = {
                "candidate_id": candidate_id,
                "sequence": row.get("sequence"),
                "subject": row.get("subject"),
                "object": row.get("object"),
                "frame_range": [row.get("start_frame"), row.get("end_frame")],
                "source_row": row,
                "status": "failed",
                "strict_accepted": False,
                "failure_class": _failure_class(f"{type(exc).__name__}:{exc}"),
                "failure": f"{type(exc).__name__}:{exc}",
            }
        _write_json(result_path, result)
        rows.append(result)
    rows.sort(key=lambda item: (str(item.get("sequence")), str(item.get("candidate_id"))))
    payload = {
        "schema_version": "toporetarget.s1_2a.e0_probe.v1",
        "probe_frame_policy": {"local_frames": list(PROBE_FRAMES), "count": 3},
        "max_candidates": int(cfg["max_e0_probe_candidates"]),
        "count": len(rows),
        "probe_pass": sum(bool(item.get("strict_accepted")) for item in rows),
        "failure_counts": {
            name: sum(item.get("failure_class") == name for item in rows)
            for name in sorted(set(FAILURE_CLASSES.values()))
        },
        "rows": rows,
        "selection_scope": "source_only_then_warm_then_E0",
        "s1_results_used": False,
    }
    _write_json(experiment / "selection" / "e0_probe_candidates.json", payload)
    _write_csv(experiment / "selection" / "e0_probe_candidates.csv", rows)
    return payload


def freeze_stress_clips(
    repo: str | Path, config_path: str | Path, experiment_root: str | Path
) -> dict[str, Any]:
    root = Path(repo).resolve()
    cfg, config_file = _cfg(root, config_path)
    experiment = Path(experiment_root).resolve()
    probe = json.loads((experiment / "selection" / "e0_probe_candidates.json").read_text())
    candidates = [item for item in probe["rows"] if item.get("strict_accepted")]
    candidates.sort(key=_rank_key)
    chosen = candidates[: int(cfg["stress_count"])]
    rows = []
    for index, item in enumerate(chosen, 1):
        row = item["source_row"]
        rows.append(
            {
                "clip_id": f"Stress{index}",
                "candidate_id": item["candidate_id"],
                "subject": item["subject"],
                "sequence": item["sequence"],
                "object": item["object"],
                "frames": [int(row["start_frame"]), int(row["start_frame"]) + 60],
                "hand": "right",
                "robot": cfg["robot_id"],
                "e0_metrics": item["e0_metrics"],
                "source_hash": row.get("source_hash"),
                "vtemp_hash": row.get("vtemp_hash"),
                "object_mesh_hash": row.get("object_mesh_hash"),
                "e0_profile_hash": _sha256(
                    root / "configs" / "experiments" / "s1_2a_e0_penetration_stress_v1.yaml"
                ),
                "selection_scope": "E0_probe_only",
                "s1_results_used": False,
            }
        )
    valid = len(rows) == int(cfg["stress_count"])
    payload = {
        "schema_version": "toporetarget.s1_2a.stress_selection.v1",
        "status": "FROZEN" if valid else "S1_NO_VALID_STRESS_CASE_FOUND",
        "config": str(config_file),
        "config_hash": _sha256(config_file),
        "selected_units": rows,
        "candidate_active_count": len(candidates),
        "within_subject_stress_set": len({item["subject"] for item in rows}) == 1
        if rows
        else False,
        "selection_algorithm": (
            "lexicographic E0 robot penetration: frames_gt_1mm, mean_excess, "
            "max, active_links, sequence_id"
        ),
        "selection_algorithm_hash": stable_hash(
            {"name": "e0_penetration_lexicographic", "version": 1, "excluded": sorted(EXCLUDED)}
        ),
        "source_candidates_hash": _sha256(experiment / "selection" / "source_candidates.json"),
        "e0_probe_candidates_hash": _sha256(experiment / "selection" / "e0_probe_candidates.json"),
        "s1_results_used": False,
        "replacement_after_freeze_forbidden": True,
    }
    selection = experiment / "selection"
    _write_json(selection / "stress_selection_manifest.json", payload)
    _write_json(experiment / "reports" / "stress_selection_manifest.json", payload)
    _write_csv(selection / "stress_selection.csv", rows)
    (selection / "stress_selection.lock").write_text(stable_hash(payload) + "\n", encoding="utf-8")
    return payload


def _backend_audit_for_clip(
    experiment: Path, clip: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    sequence = load_hoi_sequence(paths["canonical"])
    object_track = sequence.rigid_objects[0]
    final = load_final_trajectory(paths["root"] / "E0" / "final.zarr")
    reference = ReferenceSignedDistanceBackend(
        object_track.mesh.vertices_local, object_track.mesh.faces, sign_mode="strict"
    )
    fast = ConvexHullSignedDistanceBackend(
        object_track.mesh.vertices_local, object_track.mesh.faces, reference.mesh_hash
    )
    fast_values, ref_values, fast_normals, ref_normals = [], [], [], []
    for frame, points in enumerate(np.asarray(final.arrays["collision_points_scene"])):
        pose = object_track.pose_scene.pose_scene[frame]
        fast_result = fast.query_scene(points, pose)
        ref_result = reference.query_scene(points, pose)
        fast_values.append(np.asarray(fast_result.signed_distance))
        ref_values.append(np.asarray(ref_result.signed_distance))
        fast_normals.append(np.asarray(fast_result.surface_normals))
        ref_normals.append(np.asarray(ref_result.surface_normals))
    fast_array = np.concatenate(fast_values)
    ref_array = np.concatenate(ref_values)
    active = (fast_array < -DEAD_ZONE_M) | (ref_array < -DEAD_ZONE_M)
    if not np.any(active):
        return {
            "clip_id": clip["clip_id"],
            "status": "FAST_BACKEND_MISS_PENETRATION_SIGNAL",
            "pass": False,
            "active_sample_count": 0,
            "reference_backend": reference.describe(),
            "fast_backend": "convex_hull_exact_solver_only",
        }
    metrics = _backend_metrics(
        fast_array[active],
        ref_array[active],
        np.concatenate(fast_normals)[active],
        np.concatenate(ref_normals)[active],
    )
    return {
        "clip_id": clip["clip_id"],
        "status": "pass" if metrics["gate_pass"] else "FAST_BACKEND_MISS_PENETRATION_SIGNAL",
        "pass": bool(metrics["gate_pass"]),
        "active_sample_count": int(np.count_nonzero(active)),
        "active_region_only": True,
        **metrics,
        "reference_backend": reference.describe(),
        "fast_backend": "convex_hull_exact_solver_only",
    }


def audit_backends(
    repo: str | Path, config_path: str | Path, experiment_root: str | Path
) -> dict[str, Any]:
    root = Path(repo).resolve()
    cfg, _ = _cfg(root, config_path)
    experiment = Path(experiment_root).resolve()
    manifest = json.loads((experiment / "selection" / "stress_selection_manifest.json").read_text())
    rows = []
    for clip in manifest.get("selected_units", []):
        paths = {
            "root": experiment / "stress" / clip["clip_id"],
            "canonical": experiment / "stress" / clip["clip_id"] / "canonical.zarr",
        }
        try:
            row = _backend_audit_for_clip(experiment, clip, paths)
        except Exception as exc:
            row = {
                "clip_id": clip["clip_id"],
                "status": "FAILED",
                "pass": False,
                "error": f"{type(exc).__name__}:{exc}",
            }
        rows.append(row)
        _write_json(experiment / "stress" / clip["clip_id"] / "backend_audit.json", row)
    payload = {
        "status": "pass" if rows and all(row.get("pass") for row in rows) else "mismatch",
        "clips": rows,
        "gate": cfg["backend_consistency_gate"],
    }
    _write_json(experiment / "reports" / "backend_audit.json", payload)
    _write_json(experiment / "reports" / "fast_reference_consistency.json", payload)
    return payload


def _contact_f1(
    reference: ReferenceSignedDistanceBackend,
    object_pose: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    a = np.asarray(reference.query_scene(left, object_pose).signed_distance) <= 0.005
    b = np.asarray(reference.query_scene(right, object_pose).signed_distance) <= 0.005
    tp = int(np.count_nonzero(a & b))
    fp = int(np.count_nonzero(~a & b))
    fn = int(np.count_nonzero(a & ~b))
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def _trajectory_metrics(
    sequence: Any,
    path: Path,
    object_reference: ReferenceSignedDistanceBackend,
    collision_links: np.ndarray | None = None,
) -> dict[str, Any]:
    final = load_final_trajectory(path)
    arrays = final.arrays
    phi = np.asarray(arrays["full_signed_distance"], dtype=np.float64)
    depth = np.maximum(-phi, 0.0)
    frame_max = np.max(depth, axis=1)
    source_kp = np.asarray(sequence.hands[0].keypoint_tracks["mediapipe21"].positions_scene)
    robot_kp = np.asarray(arrays["robot_keypoints_scene"], dtype=np.float64)
    object_track = sequence.rigid_objects[0]
    f1 = []
    morphology = []
    morphology_by_finger: dict[str, list[float]] = {name: [] for name in FINGER_KEYPOINTS}
    for index in range(len(phi)):
        pose = object_track.pose_scene.pose_scene[index]
        f1.append(_contact_f1(object_reference, pose, source_kp[index], robot_kp[index]))
        morphology.append(float(np.sqrt(np.mean(np.square(source_kp[index] - robot_kp[index])))))
        for name, keypoints in FINGER_KEYPOINTS.items():
            morphology_by_finger[name].append(
                float(
                    np.sqrt(
                        np.mean(np.square(source_kp[index, keypoints] - robot_kp[index, keypoints]))
                    )
                )
            )
    qpos = np.asarray(arrays["qpos"], dtype=np.float64)
    base = np.asarray(arrays["base_pose_scene"], dtype=np.float64)[:, :3, 3]
    nfev = int(np.sum(np.asarray(arrays.get("nfev", [0]), dtype=np.int64)))
    per_link: dict[str, Any] = {}
    per_finger: dict[str, Any] = {}
    if collision_links is not None:
        per_link, per_finger = _group_penetration_metrics(depth, collision_links)

    def max_diff(value: np.ndarray, order: int) -> float:
        return (
            0.0
            if len(value) <= order
            else float(np.max(np.linalg.norm(np.diff(value, n=order, axis=0), axis=-1)))
        )

    return {
        "frame_count": int(len(phi)),
        "full_sample_count": int(phi.shape[1]),
        "mean_excess_penetration_m": float(np.mean(np.maximum(frame_max - DEAD_ZONE_M, 0.0))),
        "max_penetration_m": float(np.max(depth)),
        "rms_penetration_m": float(np.sqrt(np.mean(np.square(depth)))),
        "penetration_energy": float(np.mean(np.square(np.maximum(depth - DEAD_ZONE_M, 0.0)))),
        "frames_gt_1mm": int(np.count_nonzero(frame_max > DEAD_ZONE_M)),
        "frames_gt_2mm": int(np.count_nonzero(frame_max > 0.002)),
        "frame_ratio_gt_1mm": float(np.mean(frame_max > DEAD_ZONE_M)),
        "frame_ratio_gt_2mm": float(np.mean(frame_max > 0.002)),
        "negative_sample_fraction": float(np.mean(phi < 0.0)),
        "per_link_penetration": per_link,
        "per_finger_penetration": per_finger,
        "e_im": float(np.mean(np.asarray(arrays.get("e_im", [0.0])))),
        "contact_f1_at_5mm_proxy": float(np.mean(f1)),
        "morphology_rmse_m": float(np.mean(morphology)),
        "morphology_rmse_by_finger_m": {
            name: float(np.mean(values)) for name, values in morphology_by_finger.items()
        },
        "q_velocity": max_diff(qpos, 1),
        "q_jerk": max_diff(qpos, 3),
        "base_jerk": max_diff(base, 3),
        "solver_success": bool(np.all(np.asarray(arrays.get("accepted", []), dtype=bool))),
        "strict_accepted_count": int(
            np.count_nonzero(np.asarray(arrays.get("accepted", []), dtype=bool))
        ),
        "status_9_count": int(
            np.count_nonzero(np.asarray(arrays.get("optimizer_status_code", [])) == 9)
        ),
        "status_0_count": int(
            np.count_nonzero(np.asarray(arrays.get("optimizer_status_code", [])) == 0)
        ),
        "finite": bool(np.all(np.isfinite(phi))),
        "runtime_s": float(np.sum(np.asarray(arrays.get("solve_time_s", [0.0]), dtype=np.float64))),
        "nfev": nfev,
        "sdf_callback_cost_proxy": {
            "function_evaluations": nfev,
            "full_surface_queries": nfev * int(phi.shape[1]),
            "measured_directly": False,
        },
        "artifact_hash": _sha256(path),
    }


def _comparison_delta(e0: dict[str, Any], s1_metrics: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in e0.items():
        if (
            isinstance(value, (int, float))
            and key in s1_metrics
            and isinstance(s1_metrics[key], (int, float))
        ):
            result[key] = {"e0": value, "s1": s1_metrics[key], "delta": s1_metrics[key] - value}
    return result


def run_full_evaluation(
    repo: str | Path, config_path: str | Path, experiment_root: str | Path, *, resume: bool = True
) -> dict[str, Any]:
    root = Path(repo).resolve()
    cfg, _ = _cfg(root, config_path)
    experiment = Path(experiment_root).resolve()
    manifest = json.loads((experiment / "selection" / "stress_selection_manifest.json").read_text())
    records = []
    _prepare_robot_surface(root, experiment, cfg)
    for clip in manifest.get("selected_units", []):
        clip_id = clip["clip_id"]
        row = next(
            item["source_row"]
            for item in json.loads(
                (experiment / "selection" / "e0_probe_candidates.json").read_text()
            )["rows"]
            if item["candidate_id"] == clip["candidate_id"]
        )
        paths: dict[str, Path] = {}
        try:
            paths = _prepare_inputs(root, experiment, cfg, row, clip_id, probe=False)
            reference = ReferenceSignedDistanceBackend(
                load_hoi_sequence(paths["canonical"]).rigid_objects[0].mesh.vertices_local,
                load_hoi_sequence(paths["canonical"]).rigid_objects[0].mesh.faces,
                sign_mode="strict",
            )
            e0 = _refine(
                root, experiment, paths, cfg, label="E0", lambda_sdf=0.0, expected=60, resume=resume
            )
            s1_result = _refine(
                root,
                experiment,
                paths,
                cfg,
                label="S1",
                lambda_sdf=LAMBDA_SDF,
                expected=60,
                resume=resume,
            )
            if e0.get("metrics") is None or s1_result.get("metrics") is None:
                raise RuntimeError("missing E0/S1 full artifact")
            sequence = load_hoi_sequence(paths["canonical"])
            collision_links = np.asarray(
                np.load(
                    experiment / "selection" / "artimano_rh_collision_surface.npz",
                    allow_pickle=False,
                )["link_names"]
            ).astype(str)
            e0_metrics = _trajectory_metrics(
                object_reference=reference,
                sequence=sequence,
                path=Path(e0["output"]),
                collision_links=collision_links,
            )
            s1_metrics = _trajectory_metrics(
                object_reference=reference,
                sequence=sequence,
                path=Path(s1_result["output"]),
                collision_links=collision_links,
            )
            record = {
                "clip_id": clip_id,
                "status": "complete"
                if _strict_summary(e0_metrics, 60) and _strict_summary(s1_metrics, 60)
                else "strict_failed",
                "source": {
                    "sequence": row["sequence"],
                    "frame_range": [row["start_frame"], row["end_frame"]],
                },
                "paths": {key: str(value) for key, value in paths.items()},
                "E0": e0_metrics,
                "S1": s1_metrics,
                "delta": _comparison_delta(e0_metrics, s1_metrics),
            }
        except Exception as exc:
            record = {
                "clip_id": clip_id,
                "status": "failed",
                "failure": f"{type(exc).__name__}:{exc}",
                "failure_class": _failure_class(f"{type(exc).__name__}:{exc}"),
                "paths": {key: str(value) for key, value in paths.items()},
            }
        _write_json(experiment / "stress" / clip_id / "comparison.json", record)
        records.append(record)
    payload = {
        "status": "complete"
        if records and all(item["status"] == "complete" for item in records)
        else "partial",
        "profile": PROFILE,
        "lambda_sdf": LAMBDA_SDF,
        "dead_zone_m": DEAD_ZONE_M,
        "records": records,
    }
    _write_json(experiment / "reports" / "final_comparison.json", payload)
    rows = []
    for record in records:
        for metric, values in record.get("delta", {}).items():
            rows.append({"clip_id": record["clip_id"], "metric": metric, **values})
    _write_csv(experiment / "reports" / "final_comparison.csv", rows)
    _write_json(experiment / "reports" / "full_run_status.json", payload)
    return payload


def decide(experiment_root: str | Path) -> dict[str, Any]:
    experiment = Path(experiment_root)
    selection = json.loads(
        (experiment / "selection" / "stress_selection_manifest.json").read_text()
    )
    if selection.get("status") != "FROZEN":
        return {
            "final_status": "S1_NO_VALID_STRESS_CASE_FOUND",
            "decision": "NO_VALID_STRESS_CASE_FOUND",
            "reason": selection.get("status"),
        }
    backend = json.loads((experiment / "reports" / "backend_audit.json").read_text())
    comparison = json.loads((experiment / "reports" / "final_comparison.json").read_text())
    records = comparison.get("records", [])
    gates = []
    for record in records:
        e0, s1_metrics = record.get("E0", {}), record.get("S1", {})
        if not e0 or not s1_metrics:
            continue
        e0_energy = float(e0.get("penetration_energy", 0.0))
        improvement = (
            0.0
            if e0_energy <= 0
            else (e0_energy - float(s1_metrics.get("penetration_energy", 0.0))) / e0_energy
        )
        contact_drop = float(s1_metrics.get("contact_f1_at_5mm_proxy", 0.0)) - float(
            e0.get("contact_f1_at_5mm_proxy", 0.0)
        )
        morphology_delta = float(s1_metrics.get("morphology_rmse_m", 0.0)) - float(
            e0.get("morphology_rmse_m", 0.0)
        )
        jerk_ratio = (
            0.0
            if float(e0.get("q_jerk", 0.0)) == 0
            else (float(s1_metrics.get("q_jerk", 0.0)) - float(e0.get("q_jerk", 0.0)))
            / float(e0["q_jerk"])
        )
        gates.append(
            {
                "clip_id": record["clip_id"],
                "energy_improvement": improvement,
                "max_not_increased": float(s1_metrics.get("max_penetration_m", math.inf))
                <= float(e0.get("max_penetration_m", -math.inf)),
                "contact_f1_delta": contact_drop,
                "morphology_delta_m": morphology_delta,
                "jerk_ratio": jerk_ratio,
                "pass": improvement >= 0.20
                and float(s1_metrics.get("max_penetration_m", math.inf))
                <= float(e0.get("max_penetration_m", -math.inf))
                and contact_drop > -0.05
                and morphology_delta < 0.0005
                and jerk_ratio < 0.20,
            }
        )
    accepted = bool(
        len(gates) == 3
        and sum(item["pass"] for item in gates) >= 2
        and backend.get("status") == "pass"
        and comparison.get("status") == "complete"
    )
    macro = float(np.mean([item["energy_improvement"] for item in gates])) if gates else 0.0
    if accepted:
        decision = "S1_CONDITIONALLY_ACCEPTED_ON_STRESS_SET"
    elif gates and macro < 0.05:
        decision = "S1_REJECTED_ON_STRESS_SET"
    elif gates:
        decision = "S1_REJECTED_DUE_TO_REGRESSION"
    else:
        decision = "S1_EVALUATION_INCOMPLETE"
    return {
        "final_status": "S1_STRESS_DISCOVERY_COMPLETE",
        "decision": decision,
        "accepted_clip_count": sum(item["pass"] for item in gates),
        "macro_energy_improvement": macro,
        "gates": gates,
        "backend_status": backend.get("status"),
        "comparison_status": comparison.get("status"),
        "lambda_study_allowed": False,
        "global_default_changed": False,
    }


def _write_markdown_reports(
    experiment: Path,
    source: dict[str, Any],
    warm: dict[str, Any],
    probe: dict[str, Any],
    selection: dict[str, Any],
    backend: dict[str, Any],
    comparison: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    selected = selection.get("selected_units", [])
    discovery_lines = [
        "# S1.2A discovery summary",
        "",
        f"- Source candidates: {source.get('candidate_count', 0)}",
        f"- Source-valid candidates: {source.get('source_valid_count', 0)}",
        f"- Warm passes / failures / not attempted: {warm.get('warm_pass', 0)} / "
        f"{warm.get('warm_failed', 0)} / {warm.get('warm_not_attempted', 0)}",
        f"- E0 probes: {probe.get('count', 0)}; strict passes: {probe.get('probe_pass', 0)}",
        f"- E0 failure counts: `{json.dumps(probe.get('failure_counts', {}), sort_keys=True)}`",
        f"- Frozen selection: `{selection.get('status')}` ({len(selected)} clips)",
        f"- Selection used S1 results: `{selection.get('s1_results_used', True)}`",
        "",
        "## Frozen clips",
        "",
        "| clip | sequence | frames >1 mm | mean excess (m) | max (m) | active links |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in selected:
        metrics = item.get("e0_metrics", {})
        discovery_lines.append(
            f"| {item.get('clip_id')} | {item.get('sequence')} | "
            f"{metrics.get('frames_gt_1mm', 0)} | "
            f"{metrics.get('mean_excess_penetration_m', 0.0):.6g} | "
            f"{metrics.get('max_penetration_m', 0.0):.6g} | {metrics.get('active_link_count', 0)} |"
        )
    discovery_lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Final decision: `{decision.get('decision')}`",
            f"- Backend audit: `{backend.get('status')}`",
            f"- Full comparison: `{comparison.get('status')}`",
            "- The frozen S1 formulation and global default are unchanged.",
        ]
    )
    (experiment / "reports" / "discovery_summary.md").write_text(
        "\n".join(discovery_lines) + "\n", encoding="utf-8"
    )

    comparison_lines = [
        "# S1.2A E0 versus S1 comparison",
        "",
        f"- Profile: `{PROFILE}`",
        f"- Lambda: `{LAMBDA_SDF}`; dead zone: `{DEAD_ZONE_M * 1000:g} mm`",
        f"- Overall status: `{comparison.get('status')}`",
        "",
        "| clip | metric | E0 | S1 | delta (S1-E0) |",
        "|---|---|---:|---:|---:|",
    ]
    for record in comparison.get("records", []):
        for metric, values in record.get("delta", {}).items():
            comparison_lines.append(
                f"| {record.get('clip_id')} | {metric} | {values.get('e0', 0.0):.8g} | "
                f"{values.get('s1', 0.0):.8g} | {values.get('delta', 0.0):.8g} |"
            )
    comparison_lines.extend(
        [
            "",
            "## Automatic gate",
            "",
            f"- Decision: `{decision.get('decision')}`",
            f"- Backend consistency: `{backend.get('status')}`",
            "- This result is stress-set scoped and does not change the global default.",
        ]
    )
    (experiment / "reports" / "final_comparison.md").write_text(
        "\n".join(comparison_lines) + "\n", encoding="utf-8"
    )


def generate_html(
    repo: str | Path, config_path: str | Path, experiment_root: str | Path
) -> dict[str, Any]:
    root = Path(repo).resolve()
    cfg, _ = _cfg(root, config_path)
    experiment = Path(experiment_root).resolve()
    manifest = json.loads((experiment / "selection" / "stress_selection_manifest.json").read_text())
    html_root = experiment / "html"
    html_root.mkdir(parents=True, exist_ok=True)
    links: list[str] = []
    comparison: dict[str, Any] = json.loads(
        (experiment / "reports" / "final_comparison.json").read_text()
    )
    for clip in manifest.get("selected_units", []):
        clip_id = clip["clip_id"]
        root_path = experiment / "stress" / clip_id
        output = html_root / f"{clip_id}_E0_vs_S1.html"
        spec = ClipSpec(clip_id, clip["sequence"], clip["subject"], clip["object"], *clip["frames"])
        record: dict[str, Any] = next(
            (item for item in comparison.get("records", []) if item.get("clip_id") == clip_id),
            {},
        )
        e0_path = root_path / "E0" / "final.zarr"
        s1_path = root_path / "S1" / "final.zarr"
        if record.get("status") == "complete" and e0_path.exists() and s1_path.exists():
            render_clip_html(
                clip=spec,
                canonical_path=root_path / "canonical.zarr",
                source_path=root_path / "canonical.zarr",
                profile_paths={
                    "warm": (root_path / "warm_start.npz", True, "Stage 7 warm start"),
                    "E0": (e0_path, False, "E0 baseline"),
                    "S1": (s1_path, False, "E0 + frozen SDF loss"),
                },
                output=output,
                asset_root=root / cfg["robot_asset_root"],
                recommended_profile="S1",
                diagnostic={"experiment": EXPERIMENT_ID, "comparison": record},
            )
        else:
            failure = html.escape(
                json.dumps(
                    {
                        "clip_id": clip_id,
                        "status": record.get("status", "missing"),
                        "failure": record.get("failure", "full artifacts unavailable"),
                        "failure_class": record.get("failure_class"),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            output.write_text(
                "<!doctype html><meta charset='utf-8'>"
                f"<title>{html.escape(clip_id)} E0 vs S1 unavailable</title>"
                f"<h1>{html.escape(clip_id)} E0 vs S1 unavailable</h1>"
                "<p>This frozen stress case did not produce complete E0/S1 artifacts. "
                "It was retained in the report and was not replaced.</p>"
                f"<pre>{failure}</pre>",
                encoding="utf-8",
            )
        links.append(str(output))
    index = (
        "<!doctype html><meta charset='utf-8'><title>S1.2A E0 vs S1</title>"
        "<h1>S1.2A E0 Penetration Stress Discovery</h1><ul>"
        + "".join(
            f"<li><a href='{html.escape(Path(path).name)}'>{html.escape(Path(path).name)}</a></li>"
            for path in links
        )
        + "</ul>"
    )
    (html_root / "index.html").write_text(index, encoding="utf-8")
    smoke: dict[str, dict[str, Any]] = {}
    for path in links:
        clip_id = Path(path).stem.split("_E0_vs_S1", 1)[0]
        smoke_record: dict[str, Any] = next(
            (item for item in comparison.get("records", []) if item.get("clip_id") == clip_id),
            {},
        )
        smoke[path] = (
            smoke_html(path, expected_frames=60, profiles=3)
            if smoke_record.get("status") == "complete"
            else {
                "schema_version": "failure-page",
                "path": str(Path(path).resolve()),
                "status": "blocked",
                "reason": smoke_record.get("failure", "full artifacts unavailable"),
            }
        )
    smoke_status = (
        "pass"
        if all(item["status"] == "pass" for item in smoke.values())
        else "partial"
        if any(item["status"] == "blocked" for item in smoke.values())
        else "fail"
    )
    _write_json(
        experiment / "reports" / "html_smoke.json",
        {
            "status": smoke_status,
            "files": links,
            "clips": smoke,
        },
    )
    return {"index": str(html_root / "index.html"), "clips": links}


def run_s1_2a(
    repo: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    experiment_root: str | Path | None = None,
    resume: bool = True,
    generate: bool = True,
) -> dict[str, Any]:
    root = Path(repo).resolve()
    cfg, config_file = _cfg(root, config_path)
    experiment = (root / str(experiment_root or cfg["output_root"])).resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    _write_json(
        experiment / "reports" / "experiment_manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config": str(config_file),
            "config_hash": _sha256(config_file),
            "branch": "develop/pene-loss",
            "raw_data_unchanged": True,
            "s1_formulation_unchanged": True,
            "historical_s1_root": str(root / ".local/experiments/s1_sdf_penetration_loss_v1"),
        },
    )
    scan_source_candidates(root, config_file, experiment)
    run_warm_probes(root, config_file, experiment)
    run_e0_probes(root, config_file, experiment, resume=resume)
    freeze_stress_clips(root, config_file, experiment)
    selection = json.loads(
        (experiment / "selection" / "stress_selection_manifest.json").read_text()
    )
    if selection.get("status") == "FROZEN":
        # Full preparation must happen before backend audit and comparison.
        run_full_evaluation(root, config_file, experiment, resume=resume)
        audit_backends(root, config_file, experiment)
    else:
        _write_json(
            experiment / "reports" / "backend_audit.json",
            {"status": "not_run", "reason": selection.get("status")},
        )
        _write_json(
            experiment / "reports" / "final_comparison.json", {"status": "not_run", "records": []}
        )
    decision = decide(experiment)
    if generate and selection.get("status") == "FROZEN":
        decision["html"] = generate_html(root, config_file, experiment)
    _write_json(experiment / "reports" / "final_decision.json", decision)
    _write_json(
        experiment / "reports" / "discovery_summary.json",
        {
            "source": json.loads((experiment / "selection" / "source_candidates.json").read_text()),
            "warm": json.loads((experiment / "selection" / "warm_candidates.json").read_text()),
            "e0_probe": json.loads(
                (experiment / "selection" / "e0_probe_candidates.json").read_text()
            ),
            "selection": selection,
            "decision": decision,
        },
    )
    source = json.loads((experiment / "selection" / "source_candidates.json").read_text())
    warm = json.loads((experiment / "selection" / "warm_candidates.json").read_text())
    probe = json.loads((experiment / "selection" / "e0_probe_candidates.json").read_text())
    backend = json.loads((experiment / "reports" / "backend_audit.json").read_text())
    comparison = json.loads((experiment / "reports" / "final_comparison.json").read_text())
    _write_markdown_reports(
        experiment, source, warm, probe, selection, backend, comparison, decision
    )
    return decision


def status(experiment_root: str | Path) -> dict[str, Any]:
    root = Path(experiment_root)
    result: dict[str, Any] = {"experiment_id": EXPERIMENT_ID, "experiment_root": str(root)}
    for name in (
        "discovery_summary.json",
        "backend_audit.json",
        "final_comparison.json",
        "final_decision.json",
        "html_smoke.json",
    ):
        path = root / "reports" / name
        result[name] = json.loads(path.read_text()) if path.is_file() else None
    return result


__all__ = [
    "DEFAULT_CONFIG",
    "audit_backends",
    "decide",
    "freeze_stress_clips",
    "generate_html",
    "run_e0_probes",
    "run_full_evaluation",
    "run_s1_2a",
    "run_warm_probes",
    "scan_source_candidates",
    "status",
]
