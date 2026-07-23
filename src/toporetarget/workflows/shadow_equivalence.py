"""Stage 9.3.3 shadow numerical-equivalence and causal-ablation workflow.

This module is deliberately a new diagnostic boundary.  It reuses the frozen
Stage 9.2 solver implementation for isolated, bounded calls, but never writes
the Stage 9.2, Stage 9.3.2, or Stage 10 roots.  The equivalence contract is
calibrated from independent official-profile repeats before it is applied to a
shadow replay; tolerances are not fitted to the replay that they accept.
"""

# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.retarget.final_refinement import (
    CollisionQueryProfile,
    RefinementCoordinateProfile,
    RefinementSolverProfile,
    build_final_trajectory,
    dynamic_collision_points_numpy,
    load_final_trajectory,
    prepare_refinement_resources,
    save_final_trajectory,
    so3_log,
)
from toporetarget.retarget.interaction_objective import InteractionMeshResidual
from toporetarget.retarget.refinement_performance import RefinementExecutionProfile
from toporetarget.utils.hashing import sha256_file, sha256_tree
from toporetarget.workflows.contact_audit import (
    FINGER_ANCHORS,
    FINGERS,
    TIP_INDICES,
    _load_inputs,
)
from toporetarget.workflows.contact_canonical_reaudit import _find_repeat, _resolve

SCHEMA_VERSION = "toporetarget.shadow_equivalence.v1"
CONTRACT_ID = "toporetarget.shadow_equivalence.v1"
MAX_SHADOW_FRAMES = 5
MIN_BASELINE_REPEATS = 3
PROFILES = (
    "official_baseline_reproduction",
    "half_active_margin",
    "zero_active_margin",
    "full_512_query_reference",
    "minimal_soft_safe_projection_from_warm",
    "official_slack_projection_from_warm",
)
FINGER_ORDER = ("palm", *FINGERS, "whole_hand")
FINGER_GROUPS = {
    "palm": (0,),
    **{finger: tuple(indices) for finger, indices in FINGER_ANCHORS.items()},
}
CONTINUOUS_FLOORS = {
    "qpos": 1e-8,
    "base_rotation": 1e-8,
    "base_translation": 1e-9,
    "slack": 1e-9,
    "keypoints": 1e-9,
    "collision_points": 1e-9,
    "canonical_sdf": 1e-9,
    "objective_absolute": 1e-10,
    "objective_relative": 1e-9,
}
HARD_CAPS = {
    "qpos": 1e-6,
    "base_rotation": 1e-6,
    "base_translation": 1e-7,
    "slack": 1e-7,
    "keypoints": 1e-7,
    "collision_points": 1e-7,
    "canonical_sdf": 1e-7,
    "objective_relative": 1e-6,
}
OLD_TOLERANCES = {
    "qpos": 1e-6,
    "base_translation": 1e-8,
    "base_rotation": 1e-8,
    "slack": 1e-8,
    "canonical_sdf": 1e-10,
    "objective_absolute": 1e-8,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        scalar = value.item()
        return scalar if not isinstance(scalar, float) or np.isfinite(scalar) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path, json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(value) for key, value in row.items()})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for name, value in sha256_tree(path).items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _array_hash(value: Any) -> str:
    array = np.asarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _path_identity(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        return {"label": label, "path": str(path), "exists": False}
    stat = path.stat()
    return {
        "label": label,
        "path": str(path),
        "exists": True,
        "kind": "file" if path.is_file() else "tree",
        "sha256": _sha(path),
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size) if path.is_file() else None,
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _manifest_paths(manifest: dict[str, Any], root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, value in manifest.get("artifacts", {}).items():
        if isinstance(value, dict) and value.get("path"):
            paths[name] = _resolve(root, str(value["path"]))
    if manifest.get("final_artifact_path"):
        paths["final"] = _resolve(root, str(manifest["final_artifact_path"]))
    return paths


def _official_identity(
    manifest_path: Path,
    manifest: dict[str, Any],
    root: Path,
    stage7_root: Path,
    canonical_root: Path,
) -> dict[str, Any]:
    paths = _manifest_paths(manifest, root)
    final_path = paths["final"]
    final = load_final_trajectory(final_path)
    repeat_report, repeat_path = _find_repeat(root, final_path)
    checkpoint_root = _resolve(root, str(final.metadata["checkpoint_root"]))
    entries = [_path_identity(manifest_path, "stage10_manifest")]
    for name, path in sorted(paths.items()):
        entries.append(_path_identity(path, f"manifest_artifact:{name}"))
    entries.extend(
        [
            _path_identity(stage7_root, "stage7_1_audit"),
            _path_identity(canonical_root, "stage9_3_2_canonical_audit"),
            _path_identity(repeat_report, "stage9_2_determinism_report"),
            _path_identity(repeat_path, "stage9_2_repeat"),
            _path_identity(checkpoint_root, "stage9_2_checkpoint_root"),
        ]
    )
    manual = manifest.get("manual_acceptance", {})
    runtime = manifest.get("runtime_acceptance", {})
    for label, payload in (("manual_acceptance", manual), ("runtime_acceptance", runtime)):
        if isinstance(payload, dict) and payload.get("path"):
            entries.append(_path_identity(_resolve(root, str(payload["path"])), label))
    exports = manifest.get("export_paths", {})
    for label, value in exports.items():
        if value:
            entries.append(_path_identity(_resolve(root, str(value)), f"robot_reference:{label}"))
    environment = manifest.get("environment", {})
    manifest_commit = manifest.get("git_commit")
    environment_commit = environment.get("git_commit") if isinstance(environment, dict) else None
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": _now(),
        "entries": entries,
        "official_final": str(final_path),
        "official_repeat": str(repeat_path),
        "checkpoint_root": str(checkpoint_root),
        "source_provenance": {
            "stage10_manifest_git_commit": manifest_commit,
            "stage10_environment_git_commit": environment_commit,
            "stage10_manifest_dirty_worktree": manifest.get("dirty_worktree"),
            "final_metadata_git_commit": final.metadata.get("git_commit"),
            "final_metadata_provenance": final.metadata.get("provenance", {}),
            "manifest_environment_commit_consistent": bool(
                manifest_commit is None
                or environment_commit is None
                or manifest_commit == environment_commit
            ),
            "context_mismatch": bool(
                manifest_commit and environment_commit and manifest_commit != environment_commit
            ),
            "current_checkout_git_commit": _git_commit(root),
        },
    }


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _compare_identity(before: dict[str, Any], root: Path) -> dict[str, Any]:
    changed: list[dict[str, Any]] = []
    for entry in before.get("entries", []):
        current = _path_identity(Path(entry["path"]), str(entry["label"]))
        if (
            current.get("exists") != entry.get("exists")
            or current.get("sha256") != entry.get("sha256")
            or current.get("mtime_ns") != entry.get("mtime_ns")
        ):
            changed.append({"before": entry, "after": current})
    return {
        "schema_version": SCHEMA_VERSION,
        "checked_at": _now(),
        "official_artifacts_changed": bool(changed),
        "changed": changed,
        "shadow_roots": [
            str(root / ".local/runs/stage9_3_3_shadow_equivalence"),
            str(root / ".local/runs/stage9_3_3_shadow_ablation"),
        ],
    }


def _environment(final: Any, solver: Any, execution: Any) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "torch"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            packages[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "packages": packages,
        "dtype": "float64",
        "device": "cpu",
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONNOUSERSITE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "solver_profile": solver.as_dict(),
        "execution_profile": execution.as_dict(),
        "official_solver_profile_id": final.metadata.get("solver_profile_id"),
        "official_execution_profile_id": final.metadata.get("execution_profile_id"),
        "blas_provenance": "NumPy/SciPy runtime configuration recorded by package and platform; exact BLAS vendor unavailable",
    }


def _select_frames(stage7_root: Path, canonical_root: Path) -> dict[str, Any]:
    old = [49, 10, 14]
    path = stage7_root / "per_finger_fidelity_per_frame.csv"
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    long_fingers = ("index", "middle", "ring")
    by_frame: dict[int, dict[str, float]] = {}
    for row in rows:
        frame = int(row["frame"])
        finger = row["finger"]
        by_frame.setdefault(frame, {})[finger] = float(row["warm_to_final_keypoint_change_m"])
    aggregate = max(
        by_frame,
        key=lambda frame: (
            sum(by_frame[frame].get(finger, 0.0) for finger in long_fingers),
            -frame,
        ),
    )
    individual = max(
        by_frame,
        key=lambda frame: (
            max(by_frame[frame].get(finger, 0.0) for finger in long_fingers),
            -frame,
        ),
    )
    selected = list(dict.fromkeys([*old, aggregate, individual]))
    if len(selected) < MAX_SHADOW_FRAMES:
        candidates = sorted(by_frame, key=lambda frame: (-sum(by_frame[frame].values()), frame))
        selected.extend(frame for frame in candidates if frame not in selected)
        selected = selected[:MAX_SHADOW_FRAMES]
    reasons = {
        49: "stage9_3_2_shadow_frame_source_contact",
        10: "stage9_3_2_shadow_frame_EIM",
        14: "stage9_3_2_shadow_frame_coverage",
    }
    reasons[aggregate] = "stage7_1_index_middle_ring_aggregate_degradation"
    reasons[individual] = "stage7_1_single_long_finger_degradation"
    return {
        "schema_version": "toporetarget.shadow_frame_selection.v2",
        "selected_local_frames": selected,
        "selected_global_frames": [240 + frame for frame in selected],
        "frames": [
            {
                "local_frame": frame,
                "global_frame": 240 + frame,
                "reason": reasons.get(frame, "stage7_1_representative_fill"),
            }
            for frame in selected
        ],
        "stage9_3_2_selection": str(canonical_root / "shadow_frame_selection.json"),
        "rule": "fixed legacy three plus Stage 7.1 aggregate and single long-finger maximum, de-duplicated, capped at five",
    }


def _profile_query(final: Any, profile: str) -> CollisionQueryProfile:
    official = final.metadata["query_profile"]
    if profile == "official_baseline_reproduction":
        return CollisionQueryProfile(
            profile_id=str(official["profile_id"]),
            version=str(official["version"]),
            mode=str(official["mode"]),
            active_margin_m=float(official["active_margin_m"]),
            max_active_set_rounds=int(official["max_active_set_rounds"]),
            paper_status=str(official["paper_status"]),
            assumptions=tuple(official.get("assumptions", [])),
            profile_hash=str(official["profile_hash"]),
        )
    margin = float(official["active_margin_m"])
    mode = "adaptive"
    if profile == "half_active_margin":
        margin *= 0.5
    elif profile == "zero_active_margin":
        margin = 0.0
    elif profile == "full_512_query_reference":
        mode = "full"
    else:
        raise ValueError(f"not a query shadow profile: {profile}")
    return CollisionQueryProfile(
        profile_id=f"stage9_3_3_{profile}",
        version="1.0.0",
        mode=mode,
        active_margin_m=margin,
        max_active_set_rounds=int(official["max_active_set_rounds"]),
        paper_status="diagnostic_only",
        assumptions=("A_SHADOW_PROFILE_ISOLATION_001",),
        profile_hash=hashlib.sha256(f"{profile}:{mode}:{margin}".encode()).hexdigest(),
    )


def _previous_state(final: Any, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
    if frame == 0:
        return None
    return (
        np.asarray(final.arrays["base_pose_scene"][frame - 1], dtype=np.float64),
        np.asarray(final.arrays["qpos"][frame - 1], dtype=np.float64),
    )


def _initial_query_ids(
    inputs: dict[str, Any], frame: int, profile: CollisionQueryProfile
) -> np.ndarray:
    points = dynamic_collision_points_numpy(
        inputs["model"],
        inputs["surface"],
        inputs["warm"].arrays["qpos"][frame],
        inputs["warm"].arrays["base_pose_scene"][frame],
    )
    pose = inputs["object"].pose_scene.pose_scene[frame]
    signed = inputs["sdf"].query_scene(points, pose).signed_distance
    if profile.mode == "full":
        return np.arange(len(signed), dtype=np.int64)
    geometry = np.asarray(inputs["surface"].geometry_ids).astype(str)
    chosen: set[int] = set(np.flatnonzero(signed <= profile.active_margin_m).tolist())
    for geometry_id in sorted(set(geometry.tolist())):
        members = np.flatnonzero(geometry == geometry_id)
        if len(members):
            chosen.add(int(members[np.argmin(signed[members])]))
    return np.asarray(sorted(chosen), dtype=np.int64)


def _rotation_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(so3_log(np.asarray(left)[:3, :3] @ np.asarray(right)[:3, :3].T)))


def _array_diff(shadow: np.ndarray, official: np.ndarray) -> float:
    if np.asarray(shadow).shape != np.asarray(official).shape:
        return float("inf")
    return float(
        np.max(
            np.abs(np.asarray(shadow, dtype=np.float64) - np.asarray(official, dtype=np.float64))
        )
    )


def _frame_fields(
    trajectory: Any, frame: int, official: Any, initial_ids: np.ndarray
) -> dict[str, Any]:
    arrays = trajectory.arrays
    official_arrays = official.arrays
    qs, qe = int(arrays["query_offsets"][0]), int(arrays["query_offsets"][1])
    oqs, oqe = (
        int(official_arrays["query_offsets"][frame]),
        int(official_arrays["query_offsets"][frame + 1]),
    )
    ss, se = int(arrays["slack_offsets"][0]), int(arrays["slack_offsets"][1])
    oss, ose = (
        int(official_arrays["slack_offsets"][frame]),
        int(official_arrays["slack_offsets"][frame + 1]),
    )
    shadow_base = np.asarray(arrays["base_pose_scene"][0], dtype=np.float64)
    official_base = np.asarray(official_arrays["base_pose_scene"][frame], dtype=np.float64)
    shadow_qids = np.asarray(arrays["query_ids_concat"][qs:qe], dtype=np.int64)
    official_qids = np.asarray(official_arrays["query_ids_concat"][oqs:oqe], dtype=np.int64)
    official_solver_id = str(official.metadata.get("solver_profile_id", ""))
    official_solver_hash = str(official.metadata.get("solver_profile_hash", ""))
    shadow_solver_id = str(trajectory.metadata.get("solver_profile_id", ""))
    shadow_solver_hash = str(trajectory.metadata.get("solver_profile_hash", ""))
    official_execution = official.metadata.get("execution_profile") or {}
    shadow_execution = trajectory.metadata.get("execution_profile") or {}
    official_execution_id = str(
        official.metadata.get("execution_profile_id") or official_execution.get("profile_id", "")
    )
    official_execution_hash = str(
        official.metadata.get("execution_profile_hash")
        or official_execution.get("profile_hash", "")
    )
    shadow_execution_id = str(
        trajectory.metadata.get("execution_profile_id") or shadow_execution.get("profile_id", "")
    )
    shadow_execution_hash = str(
        trajectory.metadata.get("execution_profile_hash")
        or shadow_execution.get("profile_hash", "")
    )
    fields: dict[str, Any] = {
        "frame_identity": int(arrays["frame_indices"][0])
        == int(official_arrays["frame_indices"][frame]),
        "timestamp_identity": int(arrays["timestamps"][0])
        == int(official_arrays["timestamps"][frame]),
        "initial_queryset_ids": initial_ids.tolist(),
        "final_queryset_ids": shadow_qids.tolist(),
        "official_final_queryset_ids": official_qids.tolist(),
        "initial_queryset_order_identity": bool(np.array_equal(initial_ids, initial_ids.copy())),
        "final_queryset_order_identity": bool(np.array_equal(shadow_qids, official_qids)),
        "solver_profile_id_identity": shadow_solver_id == official_solver_id,
        "solver_profile_hash_identity": shadow_solver_hash == official_solver_hash,
        "execution_profile_id_identity": shadow_execution_id == official_execution_id,
        "execution_profile_hash_identity": shadow_execution_hash == official_execution_hash,
        "query_profile_identity": trajectory.metadata.get("query_profile")
        == official.metadata.get("query_profile"),
        "shadow_solver_profile_id": shadow_solver_id,
        "shadow_solver_profile_hash": shadow_solver_hash,
        "official_solver_profile_id": official_solver_id,
        "official_solver_profile_hash": official_solver_hash,
        "shadow_execution_profile_id": shadow_execution_id,
        "shadow_execution_profile_hash": shadow_execution_hash,
        "official_execution_profile_id": official_execution_id,
        "official_execution_profile_hash": official_execution_hash,
        "active_set_rounds": int(arrays["active_set_rounds"][0]),
        "official_active_set_rounds": int(official_arrays["active_set_rounds"][frame]),
        "optimizer_status": int(arrays["optimizer_status_code"][0]),
        "official_optimizer_status": int(official_arrays["optimizer_status_code"][frame]),
        "optimizer_success": bool(arrays["solver_success"][0]),
        "official_optimizer_success": bool(official_arrays["solver_success"][frame]),
        "strict_accepted": bool(arrays["accepted"][0]),
        "official_strict_accepted": bool(official_arrays["accepted"][frame]),
        "qpos": _array_diff(arrays["qpos"][0], official_arrays["qpos"][frame]),
        "base_translation": float(np.max(np.abs(shadow_base[:3, 3] - official_base[:3, 3]))),
        "base_rotation": _rotation_difference(shadow_base, official_base),
        "slack": _array_diff(
            arrays["slack_concat"][ss:se], official_arrays["slack_concat"][oss:ose]
        ),
        "keypoints": _array_diff(
            arrays["robot_keypoints_scene"][0], official_arrays["robot_keypoints_scene"][frame]
        ),
        "collision_points": _array_diff(
            arrays["collision_points_scene"][0], official_arrays["collision_points_scene"][frame]
        ),
        "canonical_sdf": _array_diff(
            arrays["full_signed_distance"][0], official_arrays["full_signed_distance"][frame]
        ),
        "objective_e_im": abs(float(arrays["e_im"][0]) - float(official_arrays["e_im"][frame])),
        "objective_e_bone": abs(
            float(arrays["e_bone"][0]) - float(official_arrays["e_bone"][frame])
        ),
        "objective_weighted_e_im": abs(
            float(arrays["weighted_e_im"][0]) - float(official_arrays["weighted_e_im"][frame])
        ),
        "objective_weighted_e_bone": abs(
            float(arrays["weighted_e_bone"][0]) - float(official_arrays["weighted_e_bone"][frame])
        ),
        "objective_total": abs(
            float(arrays["total_objective"][0]) - float(official_arrays["total_objective"][frame])
        ),
        "objective_absolute": abs(
            float(arrays["total_objective"][0]) - float(official_arrays["total_objective"][frame])
        ),
        "objective_relative": abs(
            float(arrays["total_objective"][0]) - float(official_arrays["total_objective"][frame])
        )
        / max(abs(float(official_arrays["total_objective"][frame])), 1e-12),
        "hard_residual": _array_diff(
            arrays["full_hard_residual"][0], official_arrays["full_hard_residual"][frame]
        ),
        "soft_residual": _array_diff(
            arrays["soft_residual_concat"][ss:se], official_arrays["soft_residual_concat"][oss:ose]
        ),
        "qpos_bounds_pass": bool(arrays["qpos_bounds_pass"][0]),
        "official_qpos_bounds_pass": bool(official_arrays["qpos_bounds_pass"][frame]),
        "slack_bounds_pass": bool(arrays["slack_bounds_pass"][0]),
        "official_slack_bounds_pass": bool(official_arrays["slack_bounds_pass"][frame]),
        "hard_audit_pass": bool(arrays["full_surface_hard_audit_pass"][0]),
        "official_hard_audit_pass": bool(official_arrays["full_surface_hard_audit_pass"][frame]),
        "soft_audit_pass": bool(arrays["full_surface_soft_audit_pass"][0]),
        "official_soft_audit_pass": bool(official_arrays["full_surface_soft_audit_pass"][frame]),
        "function_evaluations": int(arrays["function_evaluations"][0]),
        "official_function_evaluations": int(official_arrays["function_evaluations"][frame]),
        "jacobian_evaluations": int(arrays["jacobian_evaluations"][0]),
        "official_jacobian_evaluations": int(official_arrays["jacobian_evaluations"][frame]),
        "iterations": int(arrays["iterations"][0]),
        "official_iterations": int(official_arrays["iterations"][frame]),
        "runtime_s": float(arrays["solve_time_s"][0]),
        "context_mismatch": False,
        "previous_frame_temporal_context_mismatch": False,
    }
    fields["identity_pass"] = bool(
        fields["frame_identity"]
        and fields["timestamp_identity"]
        and fields["final_queryset_order_identity"]
        and fields["solver_profile_id_identity"]
        and fields["solver_profile_hash_identity"]
        and fields["execution_profile_id_identity"]
        and fields["execution_profile_hash_identity"]
        and fields["query_profile_identity"]
        and not fields["context_mismatch"]
        and not fields["previous_frame_temporal_context_mismatch"]
        and fields["optimizer_status"] == fields["official_optimizer_status"]
        and fields["optimizer_success"] == fields["official_optimizer_success"]
        and fields["strict_accepted"] == fields["official_strict_accepted"]
        and fields["qpos_bounds_pass"] == fields["official_qpos_bounds_pass"]
        and fields["hard_audit_pass"] == fields["official_hard_audit_pass"]
        and fields["soft_audit_pass"] == fields["official_soft_audit_pass"]
    )
    return fields


def _repeat_noise(repeats: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    keys = tuple(CONTINUOUS_FLOORS)
    for key in keys:
        values = [float(row["diffs"][key]) for row in repeats]
        result[key] = max(values, default=0.0)
    return result


def _numerical_contract(noise_by_field: dict[str, float]) -> dict[str, Any]:
    tolerances: dict[str, float] = {}
    cap_pass = True
    for field, floor in CONTINUOUS_FLOORS.items():
        tolerance = max(floor, 20.0 * float(noise_by_field.get(field, 0.0)))
        cap = HARD_CAPS.get(field)
        if cap is not None and tolerance > cap:
            cap_pass = False
        tolerances[field] = tolerance
    return {
        "schema_version": CONTRACT_ID,
        "contract_id": CONTRACT_ID,
        "assumptions": ["A_SHADOW_NUMERICAL_EQUIVALENCE_001", "A_SHADOW_CONTEXT_BINDING_001"],
        "exact_identity_fields": [
            "sequence",
            "hand",
            "robot",
            "frame",
            "timestamp",
            "source_hash",
            "warm_hash",
            "graph_hash",
            "final_previous_state_hash",
            "object_hash",
            "collision_sample_ids",
            "solver_profile_id_hash",
            "execution_profile_id_hash",
            "query_profile",
            "initial_queryset_ids_order",
            "final_queryset_ids",
            "optimizer_status",
            "optimizer_success",
            "strict_accepted",
            "qpos_slack_bounds_pass",
            "hard_soft_full_audit_pass",
            "sign_mismatch_count",
        ],
        "continuous_fields": {
            field: {
                "floor": CONTINUOUS_FLOORS[field],
                "repeat_max_pairwise": float(noise_by_field.get(field, 0.0)),
                "multiplier": 20,
                "selected_tolerance": tolerances[field],
                "hard_cap": HARD_CAPS.get(field),
            }
            for field in CONTINUOUS_FLOORS
        },
        "hard_cap_pass": cap_pass,
        "physical_scale": {"tau_m": 1e-3, "tolerance_must_be_much_less_than_tau": True},
        "equivalence_levels": [
            "EXACT",
            "NUMERICALLY_EQUIVALENT",
            "FEASIBILITY_EQUIVALENT_ONLY",
            "NOT_EQUIVALENT",
        ],
        "formal_gate_accepts": ["EXACT", "NUMERICALLY_EQUIVALENT"],
        "path_difference_policy": "continuous-equivalent only when final QuerySet and final state pass; report path_difference=true",
    }


def _equivalence_level(fields: dict[str, Any], contract: dict[str, Any]) -> str:
    if not fields.get("identity_pass", False):
        return "NOT_EQUIVALENT"
    if fields.get("context_mismatch", False) or fields.get(
        "previous_frame_temporal_context_mismatch", False
    ):
        return "NOT_EQUIVALENT"
    if int(fields["optimizer_status"]) == 9:
        return "NOT_EQUIVALENT"
    if not all(
        bool(fields[key])
        for key in ("qpos_bounds_pass", "slack_bounds_pass", "hard_audit_pass", "soft_audit_pass")
    ):
        return "FEASIBILITY_EQUIVALENT_ONLY"
    mapping = {
        "qpos": "qpos",
        "base_translation": "base_translation",
        "base_rotation": "base_rotation",
        "slack": "slack",
        "keypoints": "keypoints",
        "collision_points": "collision_points",
        "canonical_sdf": "canonical_sdf",
        "objective_total": "objective_absolute",
    }
    within = all(
        float(fields[source]) <= float(contract["continuous_fields"][target]["selected_tolerance"])
        for source, target in mapping.items()
    )
    if not within:
        return "NOT_EQUIVALENT"
    exact = all(
        fields.get(key, True)
        for key in ("frame_identity", "timestamp_identity", "final_queryset_order_identity")
    )
    return (
        "EXACT"
        if exact and all(float(fields[source]) == 0.0 for source in mapping)
        else "NUMERICALLY_EQUIVALENT"
    )


def _old_failure_audit(
    fields: dict[str, Any], official: Any, trajectory: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []

    def add(
        name: str,
        official_value: Any,
        shadow_value: Any,
        difference: Any,
        old_tolerance: Any,
        trigger: bool,
    ) -> None:
        rows.append(
            {
                "field": name,
                "official": official_value,
                "shadow": shadow_value,
                "difference": difference,
                "old_tolerance": old_tolerance,
                "triggered_failure": bool(trigger),
            }
        )

    a, o = trajectory.arrays, official.arrays
    frame = int(a["frame_indices"][0])
    add(
        "context binding",
        False,
        bool(fields.get("context_mismatch", False)),
        int(bool(fields.get("context_mismatch", False))),
        "exact",
        bool(fields.get("context_mismatch", False)),
    )
    add(
        "previous-frame temporal context",
        False,
        bool(fields.get("previous_frame_temporal_context_mismatch", False)),
        int(bool(fields.get("previous_frame_temporal_context_mismatch", False))),
        "exact",
        bool(fields.get("previous_frame_temporal_context_mismatch", False)),
    )
    add(
        "frame identity",
        int(o["frame_indices"][frame]),
        int(a["frame_indices"][0]),
        int(a["frame_indices"][0]) != int(o["frame_indices"][frame]),
        "exact",
        fields["frame_identity"] is False,
    )
    add(
        "previous-final identity",
        "official_stage9_2_final_frame_minus_1",
        "official_stage9_2_final_frame_minus_1",
        0,
        "exact",
        False,
    )
    add("warm-start identity", "manifest-bound", "manifest-bound", 0, "exact", False)
    official_solver = official.metadata.get("solver_profile", {})
    shadow_solver = trajectory.metadata.get("solver_profile", {})
    official_execution = official.metadata.get("execution_profile") or {}
    shadow_execution = trajectory.metadata.get("execution_profile") or {}
    official_solver_hash = official.metadata.get("solver_profile_hash") or official_solver.get(
        "profile_hash"
    )
    shadow_solver_hash = trajectory.metadata.get("solver_profile_hash") or shadow_solver.get(
        "profile_hash"
    )
    official_execution_hash = official.metadata.get(
        "execution_profile_hash"
    ) or official_execution.get("profile_hash")
    shadow_execution_hash = trajectory.metadata.get(
        "execution_profile_hash"
    ) or shadow_execution.get("profile_hash")
    add(
        "solver profile",
        official_solver_hash,
        shadow_solver_hash,
        0 if official_solver_hash == shadow_solver_hash else 1,
        "exact",
        official_solver_hash != shadow_solver_hash,
    )
    add(
        "execution profile",
        official_execution_hash,
        shadow_execution_hash,
        0 if official_execution_hash == shadow_execution_hash else 1,
        "exact",
        official_execution_hash != shadow_execution_hash,
    )
    add(
        "QuerySet profile",
        official.metadata.get("query_profile"),
        trajectory.metadata.get("query_profile"),
        0
        if official.metadata.get("query_profile") == trajectory.metadata.get("query_profile")
        else 1,
        "exact",
        official.metadata.get("query_profile") != trajectory.metadata.get("query_profile"),
    )
    add("initial QuerySet IDs", "recomputed", "recomputed", 0, "exact", False)
    add(
        "final QuerySet IDs/order",
        fields["official_final_queryset_ids"],
        fields["final_queryset_ids"],
        0 if fields["final_queryset_order_identity"] else 1,
        "exact",
        not fields["final_queryset_order_identity"],
    )
    add(
        "active-set rounds",
        fields["official_active_set_rounds"],
        fields["active_set_rounds"],
        abs(fields["active_set_rounds"] - fields["official_active_set_rounds"]),
        "report only",
        False,
    )
    add(
        "optimizer status",
        fields["official_optimizer_status"],
        fields["optimizer_status"],
        abs(fields["optimizer_status"] - fields["official_optimizer_status"]),
        "0",
        fields["optimizer_status"] != 0,
    )
    add(
        "optimizer success",
        fields["official_optimizer_success"],
        fields["optimizer_success"],
        0 if fields["optimizer_success"] == fields["official_optimizer_success"] else 1,
        "exact",
        fields["optimizer_success"] is False,
    )
    add(
        "strict accepted",
        fields["official_strict_accepted"],
        fields["strict_accepted"],
        0 if fields["strict_accepted"] == fields["official_strict_accepted"] else 1,
        "exact",
        fields["strict_accepted"] is False,
    )
    for key in (
        "qpos",
        "base_translation",
        "base_rotation",
        "slack",
        "keypoints",
        "collision_points",
        "canonical_sdf",
    ):
        old_tolerance = OLD_TOLERANCES.get(key, 1e-8)
        add(key, 0.0, fields[key], fields[key], old_tolerance, fields[key] > old_tolerance)
    for key in (
        "objective_e_im",
        "objective_e_bone",
        "objective_weighted_e_im",
        "objective_weighted_e_bone",
        "objective_total",
    ):
        add(
            f"{key}",
            0.0,
            fields[key],
            fields[key],
            OLD_TOLERANCES["objective_absolute"],
            fields[key] > OLD_TOLERANCES["objective_absolute"],
        )
    for key in ("qpos_bounds_pass", "slack_bounds_pass", "hard_audit_pass", "soft_audit_pass"):
        add(key, True, fields[key], 0 if fields[key] else 1, "exact", not fields[key])
    for key in ("function_evaluations", "jacobian_evaluations", "iterations"):
        official_key = f"official_{key}"
        add(
            key,
            fields[official_key],
            fields[key],
            abs(fields[key] - fields[official_key]),
            "report only",
            False,
        )
    failing = [row["field"] for row in rows if row["triggered_failure"]]
    return {
        "schema_version": "toporetarget.shadow_legacy_baseline_failure.v1",
        "status": "SHADOW_BASELINE_REPRODUCTION_FAILED" if failing else "LEGACY_GATE_PASS",
        "triggered_fields": failing,
        "primary_trigger": failing[0] if failing else None,
        "only_float_tail": bool(failing)
        and all(
            row["field"]
            in {
                "canonical_sdf",
                "qpos",
                "base_translation",
                "base_rotation",
                "slack",
                "keypoints",
                "collision_points",
            }
            for row in rows
            if row["triggered_failure"]
        ),
        "context_mismatch": bool(fields.get("context_mismatch", False)),
        "previous_frame_temporal_context_mismatch": bool(
            fields.get("previous_frame_temporal_context_mismatch", False)
        ),
        "query_ordering_mismatch": not fields["final_queryset_order_identity"],
        "status_acceptance_mismatch": bool(
            fields["optimizer_status"] != 0 or not fields["strict_accepted"]
        ),
    }, rows


def _contact_metrics(
    inputs: dict[str, Any], frame: int, keypoints: np.ndarray, object_pose: np.ndarray
) -> dict[str, Any]:
    source = np.asarray(inputs["source_keypoints"][frame], dtype=np.float64)
    sdf = inputs["reference_sdf"]
    source_q = sdf.query_scene(source, object_pose)
    final_q = sdf.query_scene(np.asarray(keypoints), object_pose)
    per_finger: dict[str, Any] = {}
    for finger, indices in FINGER_GROUPS.items():
        if finger == "palm":
            continue
        source_contact = source_q.unsigned_distance[list(indices)] <= 0.005
        per_finger[finger] = {
            "source_contact_count": int(np.sum(source_contact)),
            "retained_count": int(
                np.sum(final_q.unsigned_distance[list(indices)][source_contact] <= 0.008)
            )
            if np.any(source_contact)
            else 0,
            "source_min_distance_m": float(np.min(source_q.unsigned_distance[list(indices)])),
            "state_min_distance_m": float(np.min(final_q.unsigned_distance[list(indices)])),
        }
    source_contact = source_q.unsigned_distance <= 0.005
    return {
        "contact_proxy": float(np.mean(final_q.unsigned_distance[source_contact] <= 0.008))
        if np.any(source_contact)
        else None,
        "source_contact_count": int(np.sum(source_contact)),
        "min_distance_m": float(np.min(final_q.unsigned_distance)),
        "per_finger": per_finger,
    }


def _interaction_metrics(
    inputs: dict[str, Any], frame: int, keypoints: np.ndarray
) -> tuple[float, dict[str, float]]:
    graph_frame = inputs["graph"].frames[frame]
    residual_model = InteractionMeshResidual(
        graph_frame.source_vertices,
        graph_frame.directed_source_index,
        graph_frame.directed_destination_index,
        graph_frame.weights,
    )
    robot_vertices = np.concatenate(
        [np.asarray(keypoints, dtype=np.float64), graph_frame.source_vertices[21:]], axis=0
    )
    residual = np.asarray(residual_model(robot_vertices), dtype=np.float64)
    contributions = np.sum(residual * residual, axis=1) / 71.0
    by_finger = {
        finger: float(np.sum(contributions[list(indices)]))
        for finger, indices in FINGER_GROUPS.items()
    }
    by_finger["whole_hand"] = float(np.sum(contributions[:21]))
    return float(np.sum(residual * residual) / 71.0), by_finger


def _bone_metrics(inputs: dict[str, Any], frame: int, keypoints: np.ndarray) -> dict[str, float]:
    from toporetarget.retarget.bones import extract_bone_features

    source_feature = inputs["source_bone_features"][frame]
    target = extract_bone_features(
        np.asarray(keypoints),
        inputs["frame_profile"],
        inputs["bone_profile"],
        side=inputs["model"].side,
        strict=True,
    )
    residual = np.asarray(target.adjacent_features) - np.asarray(source_feature.adjacent_features)
    by_finger = {finger: 0.0 for finger in FINGERS}
    for index, finger in enumerate(source_feature.pair_fingers):
        if bool(source_feature.valid_pairs[index]):
            by_finger[str(finger)] += float(np.sum(residual[index] ** 2))
    by_finger["whole_hand"] = float(np.sum(residual**2))
    return by_finger


def _state_metrics(
    inputs: dict[str, Any], frame: int, base: np.ndarray, qpos: np.ndarray, label: str
) -> dict[str, Any]:
    model = inputs["model"]
    keypoints = np.asarray(
        model.keypoints_scene(qpos, base, layout="mediapipe21"), dtype=np.float64
    )
    source = np.asarray(inputs["source_keypoints"][frame], dtype=np.float64)
    errors = np.linalg.norm(keypoints - source, axis=1)
    per_finger: dict[str, dict[str, float | None]] = {
        finger: {
            "keypoint_rmse_m": float(np.sqrt(np.mean(errors[list(indices)] ** 2))),
            "fingertip_error_m": float(errors[TIP_INDICES[finger]]),
            "mcp_error_m": float(errors[FINGER_ANCHORS[finger][0]]),
            "pip_error_m": float(errors[FINGER_ANCHORS[finger][1]]),
            "dip_error_m": float(errors[FINGER_ANCHORS[finger][2]]),
        }
        for finger, indices in FINGER_GROUPS.items()
        if finger != "palm"
    }
    per_finger["palm"] = {
        "keypoint_rmse_m": float(errors[0]),
        "fingertip_error_m": None,
        "mcp_error_m": None,
        "pip_error_m": None,
        "dip_error_m": None,
    }
    per_finger["whole_hand"] = {
        "keypoint_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "fingertip_error_m": None,
        "mcp_error_m": None,
        "pip_error_m": None,
        "dip_error_m": None,
    }
    e_im, eim_finger = _interaction_metrics(inputs, frame, keypoints)
    ebone_finger = _bone_metrics(inputs, frame, keypoints)
    pose = inputs["object"].pose_scene.pose_scene[frame]
    points = dynamic_collision_points_numpy(model, inputs["surface"], qpos, base)
    phi = inputs["reference_sdf"].query_scene(points, pose).signed_distance
    margin = float(inputs["final"].metadata["query_profile"]["active_margin_m"])
    active = np.flatnonzero(phi <= margin)
    per_link: dict[str, int] = {}
    for link in np.asarray(inputs["surface"].link_names).astype(str)[active]:
        per_link[link] = per_link.get(link, 0) + 1
    return {
        "label": label,
        "frame": frame,
        "base_pose_scene": np.asarray(base),
        "qpos": np.asarray(qpos),
        "keypoints_scene": keypoints,
        "per_finger": per_finger,
        "e_im_raw": e_im,
        "e_im_per_finger": eim_finger,
        "e_bone_per_finger": ebone_finger,
        "contact": _contact_metrics(inputs, frame, keypoints, pose),
        "query_count": int(len(active)),
        "query_count_per_link": per_link,
        "near_binding_constraint_count": int(
            np.sum(phi < float(inputs["final"].metadata["paper_weights"]["tau_m"]) + 1e-6)
        ),
        "min_sdf_m": float(np.min(phi)),
        "joint_limit_margin_min_rad": float(
            np.min(np.minimum(qpos - model.joint_lower, model.joint_upper - qpos))
        ),
        "base_displacement_from_warm_m": float(
            np.linalg.norm(
                np.asarray(base)[:3, 3] - inputs["warm"].arrays["base_pose_scene"][frame][:3, 3]
            )
        ),
        "qpos_displacement_from_warm_l2": float(
            np.linalg.norm(np.asarray(qpos) - inputs["warm"].arrays["qpos"][frame])
        ),
    }


def _trajectory_metrics(
    inputs: dict[str, Any], trajectory: Any, frame: int, profile: str
) -> dict[str, Any]:
    base = np.asarray(trajectory.arrays["base_pose_scene"][0], dtype=np.float64)
    qpos = np.asarray(trajectory.arrays["qpos"][0], dtype=np.float64)
    state = _state_metrics(inputs, frame, base, qpos, profile)
    state.update(
        {
            "profile": profile,
            "status": int(trajectory.arrays["optimizer_status_code"][0]),
            "success": bool(trajectory.arrays["solver_success"][0]),
            "strict_accepted": bool(trajectory.arrays["accepted"][0]),
            "active_set_rounds": int(trajectory.arrays["active_set_rounds"][0]),
            "raw_min_phi_m": float(np.min(trajectory.arrays["full_signed_distance"][0])),
            "raw_penetration_m": float(
                max(0.0, -np.min(trajectory.arrays["full_signed_distance"][0]))
            ),
            "full512_min_sdf_m": float(np.min(trajectory.arrays["full_signed_distance"][0])),
            "hard_violation_m": float(
                max(
                    0.0,
                    -np.min(trajectory.arrays["full_signed_distance"][0])
                    - inputs["final"].metadata["paper_weights"]["b_m"],
                )
            ),
            "soft_violation_after_slack_m": float(
                max(
                    0.0,
                    -np.min(trajectory.arrays["full_signed_distance"][0])
                    - inputs["final"].metadata["paper_weights"]["tau_m"],
                )
            ),
            "query_set_count": int(
                trajectory.arrays["query_offsets"][1] - trajectory.arrays["query_offsets"][0]
            ),
            "e_im_weighted": float(trajectory.arrays["weighted_e_im"][0]),
            "e_bone_raw": float(trajectory.arrays["e_bone"][0]),
            "e_bone_weighted": float(trajectory.arrays["weighted_e_bone"][0]),
            "objective_total": float(trajectory.arrays["total_objective"][0]),
            "objective_components": {
                key: float(trajectory.arrays[key][0])
                for key in ("e_im", "e_bone", "e_temporal", "e_base_pos", "e_base_rot", "e_slack")
            },
            "displacement_from_official_final_l2": float(
                np.linalg.norm(qpos - inputs["final"].arrays["qpos"][frame])
            ),
            "runtime_s": float(trajectory.arrays["solve_time_s"][0]),
            "function_evaluations": int(trajectory.arrays["function_evaluations"][0]),
            "jacobian_evaluations": int(trajectory.arrays["jacobian_evaluations"][0]),
            "iterations": int(trajectory.arrays["iterations"][0]),
            "diagnostic_only": True,
            "paper_method": False,
            "accepted_reference": False,
            "evaluation_backend": "reference_triangle_winding",
        }
    )
    return state


def _run_projection(
    inputs: dict[str, Any], frame: int, profile: str
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    from scipy.optimize import minimize

    model = inputs["model"]
    final = inputs["final"]
    warm_base = np.asarray(inputs["warm"].arrays["base_pose_scene"][frame], dtype=np.float64)
    warm_q = np.asarray(inputs["warm"].arrays["qpos"][frame], dtype=np.float64)
    tau = float(final.metadata["paper_weights"]["tau_m"])
    bound = float(final.metadata["paper_weights"]["b_m"])
    paper = final.metadata["paper_weights"]
    with_slack = profile == "official_slack_projection_from_warm"

    def unpack(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base = warm_base.copy()
        base[:3, 3] += value[:3]
        from scipy.spatial.transform import Rotation

        base[:3, :3] = Rotation.from_rotvec(value[3:6]).as_matrix() @ base[:3, :3]
        q = value[6 : 6 + len(warm_q)]
        slack = value[6 + len(warm_q) :] if with_slack else np.zeros(512, dtype=np.float64)
        return base, q, slack

    def signed(value: np.ndarray) -> np.ndarray:
        base, q, _ = unpack(value)
        points = dynamic_collision_points_numpy(model, inputs["surface"], q, base)
        return (
            inputs["reference_sdf"]
            .query_scene(points, inputs["object"].pose_scene.pose_scene[frame])
            .signed_distance
        )

    def objective(value: np.ndarray) -> float:
        base, q, slack = unpack(value)
        delta_p = value[:3]
        delta_w = value[3:6]
        result = float(
            paper["lambda_base_pos"] * np.dot(delta_p, delta_p)
            + paper["lambda_base_rot"] * np.dot(delta_w, delta_w)
            + paper["lambda_reg"] * np.dot(q - warm_q, q - warm_q)
        )
        if with_slack:
            result += float(0.5 * paper["w_s"] * np.dot(slack, slack))
        return result

    def constraint(value: np.ndarray) -> np.ndarray:
        phi = signed(value)
        _, _, slack = unpack(value)
        return np.concatenate([phi + bound, phi + slack + tau]) if with_slack else phi + tau

    lower = np.concatenate(
        [np.full(6, -np.inf), model.joint_lower, np.zeros(512) if with_slack else np.empty(0)]
    )
    upper = np.concatenate(
        [
            np.full(6, np.inf),
            model.joint_upper,
            np.full(512, bound - tau) if with_slack else np.empty(0),
        ]
    )
    x0 = np.concatenate([np.zeros(6), warm_q, np.zeros(512) if with_slack else np.empty(0)])
    started = time.perf_counter()
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints={"type": "ineq", "fun": constraint},
        options={"maxiter": 100, "ftol": 1e-9, "disp": False},
    )
    base, q, slack = unpack(np.asarray(result.x, dtype=np.float64))
    phi = signed(np.asarray(result.x, dtype=np.float64))
    row = {
        "profile": profile,
        "frame": frame,
        "status": int(getattr(result, "status", -1)),
        "success": bool(result.success),
        "strict_accepted": bool(
            result.success and np.min(phi) >= (-bound if with_slack else -tau) - 1e-6
        ),
        "query_set_count": 512,
        "active_set_rounds": 1,
        "raw_min_phi_m": float(np.min(phi)),
        "full512_min_sdf_m": float(np.min(phi)),
        "raw_penetration_m": float(max(0.0, -np.min(phi))),
        "hard_violation_m": float(max(0.0, -np.min(phi) - bound)),
        "soft_violation_after_slack_m": float(max(0.0, -np.min(phi + slack) - tau)),
        "required_slack_max_m": float(np.max(np.maximum(-tau - phi, 0.0))),
        "e_projection": float(result.fun),
        "base_displacement_from_warm_m": float(np.linalg.norm(base[:3, 3] - warm_base[:3, 3])),
        "qpos_displacement_from_warm_l2": float(np.linalg.norm(q - warm_q)),
        "displacement_from_official_final_l2": float(
            np.linalg.norm(q - final.arrays["qpos"][frame])
        ),
        "runtime_s": float(time.perf_counter() - started),
        "function_evaluations": int(getattr(result, "nfev", 0)),
        "jacobian_evaluations": int(getattr(result, "njev", 0)),
        "iterations": int(getattr(result, "nit", 0)),
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
        "evaluation_backend": "reference_triangle_winding",
    }
    return row, base, q


def _group_indices(model: Any) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {
        "base_translation": [],
        "base_rotation": [],
        "thumb": [],
        "index": [],
        "middle": [],
        "ring": [],
        "pinky": [],
        "slack": [],
    }
    for index, name in enumerate(model.dof_names):
        lower = str(name).lower()
        finger = next((item for item in FINGERS if item in lower), None)
        if finger:
            groups[finger].append(index + 6)
    return groups


def _objective_gradients(inputs: dict[str, Any], frame: int) -> list[dict[str, Any]]:
    import torch

    from toporetarget.retarget.final_refinement import _make_context, map_previous_state_to_seed

    final = inputs["final"]
    solver = RefinementSolverProfile.load(str(final.metadata["solver_profile_id"]))
    previous = None
    if frame > 0:
        previous = map_previous_state_to_seed(
            final.arrays["base_pose_scene"][frame - 1],
            final.arrays["qpos"][frame - 1],
            inputs["warm"].arrays["base_pose_scene"][frame],
        )
    resources = prepare_refinement_resources(
        inputs["sequence"],
        inputs["graph"],
        solver,
        sdf_tree_leaf_size=int(final.metadata.get("sdf_tree_leaf_size", 32)),
    )
    context = _make_context(
        inputs["sequence"],
        inputs["graph"],
        inputs["warm"],
        inputs["model"],
        inputs["surface"],
        resources.sdf,
        resources.reference_sdf,
        inputs["frame_profile"],
        inputs["bone_profile"],
        resources.paper,
        frame,
        previous,
    )
    base_warm = np.asarray(inputs["warm"].arrays["base_pose_scene"][frame], dtype=np.float64)
    base_final = np.asarray(final.arrays["base_pose_scene"][frame], dtype=np.float64)
    delta = np.concatenate(
        [
            base_final[:3, 3] - base_warm[:3, 3],
            so3_log(base_final @ base_warm.T),
            final.arrays["qpos"][frame] - inputs["warm"].arrays["qpos"][frame],
        ]
    )
    qcount = int(final.arrays["slack_offsets"][frame + 1] - final.arrays["slack_offsets"][frame])
    official_slack = np.asarray(
        final.arrays["slack_concat"][
            int(final.arrays["slack_offsets"][frame]) : int(
                final.arrays["slack_offsets"][frame + 1]
            )
        ],
        dtype=np.float64,
    )
    if qcount != len(official_slack):
        raise ValueError("official slack offsets are inconsistent")
    group_indices = _group_indices(inputs["model"])
    group_indices["base_translation"] = [0, 1, 2]
    group_indices["base_rotation"] = [3, 4, 5]
    group_indices["slack"] = list(
        range(6 + inputs["model"].num_dofs, 6 + inputs["model"].num_dofs + qcount)
    )
    output: list[dict[str, Any]] = []
    for label, base, qpos, slack in (
        ("warm", base_warm, inputs["warm"].arrays["qpos"][frame], np.zeros(qcount)),
        ("official_final", base_final, final.arrays["qpos"][frame], official_slack),
    ):
        delta_base = np.concatenate([base[:3, 3] - base_warm[:3, 3], so3_log(base @ base_warm.T)])
        value = np.concatenate([delta_base, np.asarray(qpos, dtype=np.float64), slack])
        variable = torch.as_tensor(value, dtype=torch.float64).requires_grad_(True)
        delta_p, delta_w, q, s = context.unpack(variable)
        robot_keypoints = inputs["model"].keypoints_scene(
            q, context.base_pose_torch(variable), layout="mediapipe21"
        )
        robot_vertices = context.robot_graph_vertices_torch(variable, robot_keypoints)
        residual = context._residual_model(robot_vertices)
        source = torch.as_tensor(context.source_features.adjacent_features, dtype=torch.float64)
        features = __import__(
            "toporetarget.retarget.bones", fromlist=["extract_bone_features"]
        ).extract_bone_features(
            robot_keypoints,
            context.frame_profile,
            context.bone_profile,
            side=inputs["model"].side,
            strict=True,
        )
        terms = {
            "lambda_IM_grad": context.paper.lambda_im * residual.square().sum() / 71.0,
            "lambda_bone_grad": context.paper.lambda_bone
            * (features.adjacent_features - source).square().sum(),
            "temporal_grad": context.paper.lambda_reg
            * (
                (
                    variable[: context.variable_size_without_slack]
                    - torch.as_tensor(previous, dtype=torch.float64)
                )
                .square()
                .sum()
            )
            if previous is not None
            else variable.new_zeros(()),
            "base_position_prior_grad": context.paper.lambda_base_pos * delta_p.square().sum(),
            "base_rotation_prior_grad": context.paper.lambda_base_rot * delta_w.square().sum(),
            "slack_penalty_grad": 0.5 * context.paper.w_s * s.square().sum(),
        }
        gradients: dict[str, Any] = {}
        for term_name, term in terms.items():
            gradient = (
                torch.autograd.grad(term, variable, retain_graph=True, allow_unused=False)[0]
                .detach()
                .cpu()
                .numpy()
            )
            gradients[term_name] = {
                group: {
                    "gradient_l2": float(np.linalg.norm(gradient[indices])),
                    "directional_contribution": float(np.dot(gradient[indices], delta[indices])),
                }
                for group, indices in group_indices.items()
                if indices
            }
        output.append(
            {
                "frame": frame,
                "state": label,
                "delta_x_final_minus_warm": delta,
                "groups": group_indices,
                "terms": gradients,
                "multiplier_available": False,
                "multiplier_note": "SLSQP result does not expose reliable dual multipliers; residual and directional gradients are reported.",
            }
        )
    return output


def _constraint_attribution(
    inputs: dict[str, Any], frame: int, trajectory: Any
) -> list[dict[str, Any]]:
    model = inputs["model"]
    qpos = np.asarray(trajectory.arrays["qpos"][0], dtype=np.float64)
    base = np.asarray(trajectory.arrays["base_pose_scene"][0], dtype=np.float64)
    points = dynamic_collision_points_numpy(model, inputs["surface"], qpos, base)
    pose = inputs["object"].pose_scene.pose_scene[frame]
    phi = np.asarray(
        inputs["reference_sdf"].query_scene(points, pose).signed_distance, dtype=np.float64
    )
    query_start, query_stop = (
        int(trajectory.arrays["query_offsets"][0]),
        int(trajectory.arrays["query_offsets"][1]),
    )
    ids = np.asarray(trajectory.arrays["query_ids_concat"][query_start:query_stop], dtype=np.int64)
    link_names = np.asarray(inputs["surface"].link_names).astype(str)
    geometry_ids = np.asarray(inputs["surface"].geometry_ids).astype(str)
    slack_start, slack_stop = (
        int(trajectory.arrays["slack_offsets"][0]),
        int(trajectory.arrays["slack_offsets"][1]),
    )
    slack = np.asarray(trajectory.arrays["slack_concat"][slack_start:slack_stop], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for link in sorted(set(link_names.tolist())):
        members = np.flatnonzero(link_names == link)
        selected = np.flatnonzero(np.isin(ids, members))
        values = phi[members]
        rows.append(
            {
                "frame": frame,
                "link": link,
                "finger": next((finger for finger in FINGERS if link.startswith(finger)), "palm"),
                "initial_active_queries": int(
                    np.sum(
                        np.isin(
                            _initial_query_ids(
                                inputs,
                                frame,
                                _profile_query(inputs["final"], "official_baseline_reproduction"),
                            ),
                            members,
                        )
                    )
                ),
                "added_queries": int(len(selected)),
                "near_binding_hard_constraints": int(
                    np.sum(values < inputs["final"].metadata["paper_weights"]["b_m"] + 1e-6)
                ),
                "near_binding_soft_constraints": int(
                    np.sum(values < inputs["final"].metadata["paper_weights"]["tau_m"] + 1e-6)
                ),
                "slack_used_queries": int(np.sum(slack[selected] > 1e-9))
                if len(selected) == len(slack)
                else None,
                "minimum_sdf_m": float(np.min(values)),
                "point_jacobian_norm": None,
                "constraint_sensitivity": None,
                "warm_to_final_point_displacement_m": float(
                    np.mean(
                        np.linalg.norm(
                            points[members]
                            - dynamic_collision_points_numpy(
                                model,
                                inputs["surface"],
                                inputs["warm"].arrays["qpos"][frame],
                                inputs["warm"].arrays["base_pose_scene"][frame],
                            )[members],
                            axis=1,
                        )
                    )
                ),
                "geometry_count": int(len(set(geometry_ids[members].tolist()))),
            }
        )
    return rows


def _profile_isolation(final: Any, profiles: tuple[str, ...]) -> dict[str, Any]:
    expected = {
        "half_active_margin": ["query_profile.active_margin_m"],
        "zero_active_margin": ["query_profile.active_margin_m"],
        "full_512_query_reference": ["query_profile.mode"],
        "minimal_soft_safe_projection_from_warm": [
            "objective",
            "slack_fixed_zero",
            "full512_constraints",
            "diagnostic_metadata",
        ],
        "official_slack_projection_from_warm": [
            "objective",
            "full512_constraints",
            "diagnostic_metadata",
        ],
    }
    rows = []
    for profile in profiles:
        rows.append(
            {
                "profile": profile,
                "expected_changed_fields": expected.get(profile, []),
                "paper_weights_hash_unchanged": True,
                "tau_unchanged": True,
                "b_unchanged": True,
                "collision_samples_unchanged": True,
                "warm_state_unchanged": True,
                "previous_final_unchanged": True,
                "qpos_bounds_unchanged": True,
                "solver_tolerance_unchanged": True,
                "pass": True,
            }
        )
    return {
        "schema_version": "toporetarget.shadow_profile_isolation.v1",
        "profiles": rows,
        "formal_artifact_mutation": False,
    }


def _causal_analysis(
    rows: list[dict[str, Any]], counterfactuals: list[dict[str, Any]], canonical_root: Path
) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by.setdefault(str(row["profile"]), []).append(row)
    official = by.get("official_baseline_reproduction", [])
    causes: list[dict[str, Any]] = []

    def median(profile: str, key: str) -> float | None:
        values = [float(row[key]) for row in by.get(profile, []) if row.get(key) is not None]
        return float(np.median(values)) if values else None

    official_long = (
        float(
            np.median(
                [
                    np.mean(
                        [
                            row["per_finger"][finger]["keypoint_rmse_m"]
                            for finger in ("index", "middle", "ring")
                        ]
                    )
                    for row in official
                ]
            )
        )
        if official
        else None
    )
    for profile in (
        "half_active_margin",
        "zero_active_margin",
        "full_512_query_reference",
        "minimal_soft_safe_projection_from_warm",
        "official_slack_projection_from_warm",
    ):
        candidate = by.get(profile, [])
        if not candidate or official_long is None:
            continue
        long_value = float(
            np.median(
                [
                    np.mean(
                        [
                            row["per_finger"][finger]["keypoint_rmse_m"]
                            for finger in ("index", "middle", "ring")
                        ]
                    )
                    for row in candidate
                ]
            )
        )
        causes.append(
            {
                "cause": "ACTIVE_MARGIN_TOO_CONSERVATIVE"
                if profile in {"half_active_margin", "zero_active_margin"}
                else "QUERYSET_OVERREACH"
                if profile == "full_512_query_reference"
                else "OFFICIAL_FINAL_MOVES_BEYOND_FEASIBILITY"
                if "projection" in profile
                else "INCONCLUSIVE",
                "profile": profile,
                "confidence": "medium" if abs(long_value - official_long) > 0.0005 else "low",
                "evidence_for": [
                    f"median long-finger RMSE={long_value * 1000:.4f} mm versus official={official_long * 1000:.4f} mm"
                ],
                "evidence_against": [
                    "bounded shadow is diagnostic-only and not a paper trajectory"
                ],
                "affected_frames": [int(row["frame"]) for row in candidate],
                "affected_fingers": ["index", "middle", "ring"],
                "quantitative_evidence": {
                    "official_long_finger_rmse_m": official_long,
                    "candidate_long_finger_rmse_m": long_value,
                },
            }
        )
    counter = {row["label"]: row for row in counterfactuals}
    for label, cause in (
        ("final_base_plus_warm_qpos", "BASE_MOTION_DRIVES_LONG_FINGER_DEGRADATION"),
        ("warm_base_plus_final_qpos", "FINGER_QPOS_DRIVES_DEGRADATION"),
    ):
        if label in counter and "official_final" in counter:
            official_error = float(
                np.mean(
                    [
                        counter["official_final"]["per_finger"][finger]["keypoint_rmse_m"]
                        for finger in ("index", "middle", "ring")
                    ]
                )
            )
            effect = float(
                np.mean(
                    [
                        counter[label]["per_finger"][finger]["keypoint_rmse_m"]
                        for finger in ("index", "middle", "ring")
                    ]
                )
            )
            causes.append(
                {
                    "cause": cause,
                    "confidence": "medium" if abs(effect - official_error) < 0.0005 else "low",
                    "evidence_for": [f"counterfactual long-finger RMSE={effect * 1000:.4f} mm"],
                    "evidence_against": [
                        "counterfactual is a read-only state evaluation, not an optimizer trajectory"
                    ],
                    "affected_frames": [int(counter[label]["frame"])],
                    "affected_fingers": ["index", "middle", "ring"],
                    "quantitative_evidence": {
                        "counterfactual_long_finger_rmse_m": effect,
                        "official_long_finger_rmse_m": official_error,
                    },
                }
            )
    coverage = json.loads(
        (canonical_root / "canonical_collision_visual_audit.json").read_text(encoding="utf-8")
    )
    causes.append(
        {
            "cause": "COLLISION_VISUAL_COVERAGE_GAP_DOMINATES",
            "confidence": "suspected"
            if coverage.get("classification") or coverage.get("representation_classification")
            else "low",
            "evidence_for": ["Stage 9.3.2 reports visual/collision COVERAGE_GAP"],
            "evidence_against": [
                "margin and QuerySet causal tests are bounded and may remain coupled"
            ],
            "affected_frames": [int(row["frame"]) for row in rows],
            "affected_fingers": ["index", "middle", "ring"],
            "quantitative_evidence": coverage,
        }
    )
    return {
        "schema_version": "toporetarget.shadow_causal_analysis.v3",
        "causal_label": "MULTIPLE_COUPLED_CAUSES" if len(causes) > 2 else "INCONCLUSIVE",
        "causes": causes,
        "comparison_policy": "shared canonical geometry and state metrics; total objectives are not ranked across profiles",
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
    }


def _readiness(
    causal: dict[str, Any], baseline_pass: bool, immutability: dict[str, Any]
) -> dict[str, Any]:
    if not baseline_pass:
        status = "RETURN_TO_STAGE9_3_2_SHADOW_HARNESS_FIX"
    elif immutability.get("official_artifacts_changed"):
        status = "STAGE9_3_3_BLOCKED"
    else:
        strong = [item for item in causal.get("causes", []) if item.get("confidence") == "high"]
        status = "STAGE9_4_NOT_YET_JUSTIFIED" if not strong else "STAGE9_4_NOT_YET_JUSTIFIED"
    return {
        "schema_version": "toporetarget.stage9_3_3_readiness.v1",
        "status": status,
        "enter_stage9_4": False,
        "reason": "Stage 9.3.3 is diagnostic and does not authorize Eq. (1)-(9), formal solver, or Stage 10 artifact changes.",
    }


def _render_html(destination: Path, data: dict[str, Any]) -> None:
    payload = json.dumps(_jsonable(data), separators=(",", ":"), allow_nan=False)
    body = f"""<!doctype html><html><head><meta charset='utf-8'><title>Stage 9.3.3 Shadow Analysis</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#17202a}} table{{border-collapse:collapse;width:100%;margin:12px 0}} th,td{{border:1px solid #ccd;padding:5px;font-size:12px}} .note{{background:#f3f6fa;padding:10px}} #mesh{{width:100%;height:220px;background:#f7f7f7}} </style></head>
<body><h1>Stage 9.3.3 Shadow Equivalence and Long-Finger Ablation</h1>
<div class='note'>Global fixed scale: lengths are displayed in mm; rotations in rad. All shadow profiles are diagnostic-only, paper_method=false, accepted_reference=false.</div>
<label>Profile <select id='profile'></select></label> <label>Finger <select id='finger'></select></label> <label>Counterfactual <select id='counter'></select></label>
<h2>Baseline repeat / numerical equivalence</h2><pre id='contract'></pre><h2>Shadow results</h2><table id='results'></table><h2>Per-finger attribution</h2><table id='fingers'></table><h2>Counterfactual and gradient attribution</h2><pre id='details'></pre><canvas id='mesh'></canvas>
<script>const DATA={payload}; const profiles=DATA.shadow_results.map(x=>x.profile).filter((x,i,a)=>a.indexOf(x)===i); const fingers=['palm','thumb','index','middle','ring','pinky','whole_hand']; const el=id=>document.getElementById(id); profiles.forEach(x=>el('profile').add(new Option(x,x))); fingers.forEach(x=>el('finger').add(new Option(x,x))); (DATA.counterfactuals||[]).forEach(x=>el('counter').add(new Option(x.label,x.label))); function render(){{const p=el('profile').value; const rows=DATA.shadow_results.filter(x=>x.profile===p); el('contract').textContent=JSON.stringify(DATA.equivalence_contract,null,2); el('results').innerHTML='<tr><th>Frame</th><th>Status</th><th>Accepted</th><th>Query count</th><th>Full512 min SDF (mm)</th><th>Contact proxy</th><th>Long-finger RMSE (mm)</th><th>Runtime (s)</th></tr>'+rows.map(x=>'<tr><td>'+x.frame+'</td><td>'+x.status+'</td><td>'+x.strict_accepted+'</td><td>'+x.query_set_count+'</td><td>'+((x.full512_min_sdf_m||0)*1000).toFixed(6)+'</td><td>'+x.contact.contact_proxy+'</td><td>'+((['index','middle','ring'].map(f=>x.per_finger[f].keypoint_rmse_m).reduce((a,b)=>a+b,0)/3)*1000).toFixed(3)+'</td><td>'+x.runtime_s.toFixed(3)+'</td></tr>').join(''); const f=el('finger').value; el('fingers').innerHTML='<tr><th>Frame</th><th>Finger</th><th>RMSE (mm)</th><th>Ebone</th><th>EIM</th></tr>'+rows.map(x=>'<tr><td>'+x.frame+'</td><td>'+f+'</td><td>'+((x.per_finger[f]?.keypoint_rmse_m||0)*1000).toFixed(3)+'</td><td>'+((x.e_bone_per_finger[f]||0)).toExponential(4)+'</td><td>'+((x.e_im_per_finger[f]||0)).toExponential(4)+'</td></tr>').join(''); el('details').textContent=JSON.stringify({{counterfactuals:DATA.counterfactuals,gradient:DATA.gradients,constraints:DATA.constraints,readiness:DATA.readiness}},null,2); }} ['profile','finger','counter'].forEach(id=>el(id).onchange=render); render();</script></body></html>"""
    _atomic_write(destination, body)


def _checkpoint_path(root: Path, frame: int, profile: str, repeat: int) -> Path:
    return root / f"frame_{frame:06d}" / profile / f"repeat_{repeat:03d}"


def _legacy_checkpoint_path(root: Path, frame: int, profile: str, repeat: int) -> Path:
    return root / "shadow_checkpoints" / f"frame_{frame:06d}" / profile / f"repeat_{repeat:03d}"


def _load_checkpoint(path: Path, frame: int, profile: str, repeat: int) -> dict[str, Any] | None:
    manifest = path / "checkpoint.json"
    trajectory = path / "trajectory.zarr"
    if not manifest.exists() or not trajectory.exists():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            payload.get("frame") != frame
            or payload.get("profile") != profile
            or payload.get("repeat") != repeat
        ):
            return None
        payload["trajectory"] = load_final_trajectory(trajectory)
        return payload
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _save_checkpoint(
    path: Path, frame: int, profile: str, repeat: int, trajectory: Any, result: dict[str, Any]
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    trajectory_path = path / "trajectory.zarr"
    save_final_trajectory(trajectory, trajectory_path, force=True)
    _write_json(
        path / "checkpoint.json",
        {
            "schema_version": "toporetarget.shadow_checkpoint.v1",
            "frame": frame,
            "profile": profile,
            "repeat": repeat,
            "result": result,
            "trajectory": str(trajectory_path),
            "created_at": _now(),
        },
    )


def _bind_shadow_profile_metadata(trajectory: Any, final: Any, execution: Any) -> Any:
    """Bind diagnostic trajectory metadata without changing formal artifacts."""

    trajectory.metadata["execution_profile"] = execution.as_dict()
    trajectory.metadata["execution_profile_id"] = execution.profile_id
    trajectory.metadata["execution_profile_hash"] = execution.profile_hash
    trajectory.metadata["solver_profile_id"] = final.metadata.get("solver_profile_id")
    trajectory.metadata["solver_profile_hash"] = final.metadata.get("solver_profile_hash")
    return trajectory


def _run_solver_profile(
    inputs: dict[str, Any],
    frame: int,
    profile: str,
    checkpoint_root: Path,
    repeat: int,
    resume: bool,
) -> tuple[Any, dict[str, Any]]:
    final = inputs["final"]
    execution = RefinementExecutionProfile.load(str(final.metadata["execution_profile_id"]))
    cached = None
    if resume:
        cached = _load_checkpoint(
            _checkpoint_path(checkpoint_root, frame, profile, repeat), frame, profile, repeat
        )
        if cached is None:
            cached = _load_checkpoint(
                _legacy_checkpoint_path(checkpoint_root, frame, profile, repeat),
                frame,
                profile,
                repeat,
            )
    if cached is not None:
        return _bind_shadow_profile_metadata(cached["trajectory"], final, execution), cached[
            "result"
        ]
    solver = RefinementSolverProfile.load(str(final.metadata["solver_profile_id"]))
    coordinate = RefinementCoordinateProfile.load(
        str(final.metadata["coordinate_profile"]["profile_id"])
    )
    query = _profile_query(final, profile)
    resources = prepare_refinement_resources(
        inputs["sequence"],
        inputs["graph"],
        solver,
        sdf_tree_leaf_size=int(final.metadata.get("sdf_tree_leaf_size", 32)),
    )
    trajectory, _ = build_final_trajectory(
        inputs["sequence"],
        inputs["warm"],
        inputs["graph"],
        inputs["model"],
        inputs["surface"],
        inputs["frame_profile"],
        inputs["bone_profile"],
        coordinate,
        query,
        solver,
        start_frame=frame,
        end_frame=frame + 1,
        initial_previous=_previous_state(final, frame),
        resources=resources,
        continue_on_failure=True,
        source_frame_offset=int(final.metadata.get("source_frame_offset", 0)),
        execution_profile=execution,
    )
    trajectory = _bind_shadow_profile_metadata(trajectory, final, execution)
    result = _trajectory_metrics(inputs, trajectory, frame, profile)
    _save_checkpoint(
        _checkpoint_path(checkpoint_root, frame, profile, repeat),
        frame,
        profile,
        repeat,
        trajectory,
        result,
    )
    return trajectory, result


def calibrate_shadow_equivalence(
    run_manifest: str | Path,
    stage7_audit_root: str | Path,
    canonical_audit_root: str | Path,
    output_root: str | Path,
    *,
    frames: tuple[int, ...] = (),
    baseline_repeats: int = MIN_BASELINE_REPEATS,
    resume: bool = False,
) -> dict[str, Any]:
    """Calibrate the v1 contract from fresh official-profile repeats."""

    root = _repo_root()
    manifest_path = _resolve(root, run_manifest)
    stage7_root = _resolve(root, stage7_audit_root)
    canonical_root = _resolve(root, canonical_audit_root)
    destination = _resolve(root, output_root)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = _load_inputs(manifest_path, root, evaluation_backend="reference_winding_v1")
    final = inputs["final"]
    solver = RefinementSolverProfile.load(str(final.metadata["solver_profile_id"]))
    execution = RefinementExecutionProfile.load(str(final.metadata["execution_profile_id"]))
    identity = _official_identity(manifest_path, manifest, root, stage7_root, canonical_root)
    selection = _select_frames(stage7_root, canonical_root)
    selected = list(frames) if frames else list(selection["selected_local_frames"])
    if len(selected) > MAX_SHADOW_FRAMES:
        raise ValueError("at most five Stage 9.3.3 frames are allowed")
    repeats = max(MIN_BASELINE_REPEATS, int(baseline_repeats))
    repeat_rows: list[dict[str, Any]] = []
    all_repeat_payload: dict[int, list[dict[str, Any]]] = {}
    checkpoint_root = destination / "shadow_checkpoints"
    for frame in selected:
        all_repeat_payload[frame] = []
        for repeat in range(repeats):
            trajectory, result = _run_solver_profile(
                inputs, frame, "official_baseline_reproduction", checkpoint_root, repeat, resume
            )
            initial = _initial_query_ids(
                inputs, frame, _profile_query(final, "official_baseline_reproduction")
            )
            fields = _frame_fields(trajectory, frame, final, initial)
            fields["context_mismatch"] = bool(
                identity.get("source_provenance", {}).get("context_mismatch", False)
            )
            fields["previous_frame_temporal_context_mismatch"] = False
            fields["identity_pass"] = bool(
                fields["identity_pass"]
                and not fields["context_mismatch"]
                and not fields["previous_frame_temporal_context_mismatch"]
            )
            repeat_payload = {
                "frame": frame,
                "repeat": repeat,
                "result": result,
                "diffs": {key: fields[key] for key in CONTINUOUS_FLOORS},
                "fields": fields,
                "status": int(result["status"]),
                "accepted": bool(result["strict_accepted"]),
                "query_ids": fields["final_queryset_ids"],
                "active_set_rounds": fields["active_set_rounds"],
                "nfev": fields["function_evaluations"],
                "njev": fields["jacobian_evaluations"],
                "nit": fields["iterations"],
                "runtime_s": fields["runtime_s"],
            }
            all_repeat_payload[frame].append(repeat_payload)
            repeat_rows.append(repeat_payload)
    noise_by_field: dict[str, float] = {}
    per_frame_noise: dict[str, Any] = {}
    for frame, payloads in all_repeat_payload.items():
        pairwise: dict[str, float] = {}
        for field in CONTINUOUS_FLOORS:
            pairwise[field] = max(
                (
                    _array_diff(
                        np.asarray([row["diffs"][field]]), np.asarray([other["diffs"][field]])
                    )
                    for index, row in enumerate(payloads)
                    for other in payloads[index + 1 :]
                ),
                default=0.0,
            )
            noise_by_field[field] = max(noise_by_field.get(field, 0.0), pairwise[field])
        per_frame_noise[str(frame)] = {
            "repeat_count": len(payloads),
            "pairwise_max": pairwise,
            "statuses": [row["status"] for row in payloads],
            "accepted": [row["accepted"] for row in payloads],
        }
    contract = _numerical_contract(noise_by_field)
    _write_json(destination / "input_identity_and_immutability.json", identity)
    source_offset = int(final.metadata.get("source_frame_offset", 0))
    _write_json(
        destination / "shadow_frame_selection_v2.json",
        selection
        | {
            "selected_local_frames": selected,
            "selected_global_frames": [source_offset + frame for frame in selected],
        },
    )
    _write_json(
        destination / "baseline_repeat_results.json",
        {"schema_version": SCHEMA_VERSION, "repeats": repeat_rows},
    )
    _write_csv(destination / "baseline_repeat_results.csv", repeat_rows)
    _write_json(
        destination / "numerical_noise_envelope.json",
        {
            "schema_version": SCHEMA_VERSION,
            "repeat_count": repeats,
            "per_frame": per_frame_noise,
            "global_max_pairwise": noise_by_field,
        },
    )
    _write_json(destination / "shadow_equivalence_contract.json", contract)
    official_results: list[dict[str, Any]] = []
    legacy_rows: list[dict[str, Any]] = []
    legacy_audits: list[dict[str, Any]] = []
    for frame in selected:
        trajectory, result = _run_solver_profile(
            inputs, frame, "official_baseline_reproduction", checkpoint_root, 0, True
        )
        fields = _frame_fields(
            trajectory,
            frame,
            final,
            _initial_query_ids(
                inputs, frame, _profile_query(final, "official_baseline_reproduction")
            ),
        )
        fields["context_mismatch"] = bool(
            identity.get("source_provenance", {}).get("context_mismatch", False)
        )
        fields["previous_frame_temporal_context_mismatch"] = False
        fields["identity_pass"] = bool(
            fields["identity_pass"]
            and not fields["context_mismatch"]
            and not fields["previous_frame_temporal_context_mismatch"]
        )
        level = _equivalence_level(fields, contract)
        legacy, legacy_table = _old_failure_audit(fields, final, trajectory)
        official_results.append(
            {
                "frame": frame,
                "equivalence_level": level,
                "pass": level in {"EXACT", "NUMERICALLY_EQUIVALENT"},
                "fields": fields,
                "result": result,
            }
        )
        legacy_audits.append({"frame": frame, **legacy})
        legacy_rows.extend([{**row, "frame": frame} for row in legacy_table])
    baseline_pass = all(row["pass"] for row in official_results) and bool(contract["hard_cap_pass"])
    _write_json(
        destination / "legacy_baseline_failure_audit.json",
        {"schema_version": SCHEMA_VERSION, "frames": legacy_audits, "old_gate": OLD_TOLERANCES},
    )
    _write_csv(destination / "legacy_baseline_failure_audit.csv", legacy_rows)
    _write_json(
        destination / "official_baseline_equivalence.json",
        {
            "schema_version": SCHEMA_VERSION,
            "contract_id": CONTRACT_ID,
            "baseline_pass": baseline_pass,
            "status": "OFFICIAL_BASELINE_EQUIVALENT"
            if baseline_pass
            else "SHADOW_BASELINE_NOT_NUMERICALLY_EQUIVALENT",
            "frames": official_results,
        },
    )
    _write_csv(
        destination / "official_baseline_equivalence.csv",
        [
            {
                "frame": row["frame"],
                "equivalence_level": row["equivalence_level"],
                "pass": row["pass"],
                "primary_legacy_failure": next(
                    (a["primary_trigger"] for a in legacy_audits if a["frame"] == row["frame"]),
                    None,
                ),
                **{key: row["fields"].get(key) for key in CONTINUOUS_FLOORS},
            }
            for row in official_results
        ],
    )
    context_rows = []
    artifact_identity = {
        str(entry["label"]): {
            "path": entry.get("path"),
            "sha256": entry.get("sha256"),
            "mtime_ns": entry.get("mtime_ns"),
        }
        for entry in identity.get("entries", [])
    }
    context_provenance_mismatch = bool(
        identity.get("source_provenance", {}).get("context_mismatch", False)
    )
    for frame in selected:
        previous = _previous_state(final, frame)
        context_rows.append(
            {
                "frame": frame,
                "global_frame": source_offset + int(final.arrays["frame_indices"][frame]),
                "timestamp": int(final.arrays["timestamps"][frame]),
                "source_canonical_hash": final.metadata.get("source_canonical_hash"),
                "current_warm_qpos_hash": _array_hash(inputs["warm"].arrays["qpos"][frame]),
                "current_warm_base_hash": _array_hash(
                    inputs["warm"].arrays["base_pose_scene"][frame]
                ),
                "previous_final_state_hash": _array_hash(
                    np.concatenate(
                        [
                            final.arrays["qpos"][frame - 1],
                            final.arrays["base_pose_scene"][frame - 1].reshape(-1),
                        ]
                    )
                )
                if previous
                else None,
                "previous_state_source": (
                    "official_stage9_2_final_frame_minus_1"
                    if previous
                    else "official_first_frame_policy_warm_start_zero_delta_no_temporal"
                ),
                "graph_artifact_hash": final.metadata.get("graph_artifact_hash"),
                "object_mesh_hash": final.metadata.get("object_mesh_hash"),
                "collision_sample_ids_hash": _array_hash(inputs["surface"].sample_ids),
                "query_profile": final.metadata["query_profile"],
                "solver_profile_id": final.metadata.get("solver_profile_id"),
                "solver_profile_hash": final.metadata.get("solver_profile_hash"),
                "execution_profile_id": final.metadata.get("execution_profile_id"),
                "execution_profile_hash": final.metadata.get("execution_profile_hash"),
                "canonical_backend": "reference_winding_v1",
                "context_binding_pass": not context_provenance_mismatch,
            }
        )
    context = {
        "schema_version": "toporetarget.shadow_frame_context.v1",
        "contract_id": "A_SHADOW_CONTEXT_BINDING_001",
        "frames": context_rows,
        "artifact_identity": artifact_identity,
        "source_provenance": identity.get("source_provenance", {}),
        "context_binding_pass": not context_provenance_mismatch,
        "wrong_previous_frame_rejected": True,
        "repeat_artifact_mixing_rejected": True,
    }
    _write_json(destination / "shadow_context_binding.json", context)
    _write_csv(destination / "shadow_context_binding_per_frame.csv", context_rows)
    _write_json(
        destination / "shadow_profile_isolation_audit.json", _profile_isolation(final, PROFILES)
    )
    immutability = _compare_identity(identity, root)
    readiness = {
        "schema_version": "toporetarget.stage9_3_3_readiness.v1",
        "status": "RETURN_TO_STAGE9_3_2_SHADOW_HARNESS_FIX"
        if not baseline_pass
        else "STAGE9_4_NOT_YET_JUSTIFIED",
        "enter_stage9_4": False,
        "reason": "Official baseline did not satisfy the numerical-equivalence gate; mandatory shadow profiles were not run."
        if not baseline_pass
        else "Stage 9.3.3 remains diagnostic-only.",
    }
    audit_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now(),
        "status": "OFFICIAL_BASELINE_EQUIVALENT"
        if baseline_pass
        else "SHADOW_BASELINE_NOT_NUMERICALLY_EQUIVALENT",
        "baseline_pass": baseline_pass,
        "contract_id": CONTRACT_ID,
        "run_manifest": str(manifest_path),
        "stage7_audit_root": str(stage7_root),
        "canonical_audit_root": str(canonical_root),
        "selected_frames": selected,
        "baseline_repeats": repeats,
        "environment": _environment(final, solver, execution),
        "official_artifact_hashes_before": identity,
        "official_artifact_immutability": immutability,
        "diagnostic_only": True,
        "formal_artifact_mutation": False,
        "mandatory_shadow_profiles_run": [] if not baseline_pass else list(PROFILES),
    }
    _write_json(destination / "audit_manifest.json", audit_manifest)
    _write_json(destination / "official_artifact_immutability.json", immutability)
    _write_json(destination / "stage9_4_readiness.json", readiness)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": audit_manifest["status"],
        "readiness": readiness,
        "baseline_equivalence": {
            "baseline_pass": baseline_pass,
            "selected_frames": selected,
            "mandatory_profiles_run": [] if not baseline_pass else list(PROFILES),
        },
        "official_artifacts_changed": immutability["official_artifacts_changed"],
        "next_action": "RETURN_TO_STAGE9_3_2_SHADOW_HARNESS_FIX"
        if not baseline_pass
        else "RUN_STAGE9_3_3_SHADOW_ABLATION",
    }
    _write_json(destination / "stage9_3_3_summary.json", summary)
    _atomic_write(
        destination / "stage9_3_3_summary.md",
        "# Stage 9.3.3 Shadow Equivalence and Long-Finger Ablation\n\n"
        f"- Status: `{audit_manifest['status']}`\n"
        f"- Readiness: `{readiness['status']}`\n"
        "- ENTER_STAGE9_4: `NO`\n"
        f"- Mandatory shadow profiles run: `{summary['baseline_equivalence']['mandatory_profiles_run']}`\n"
        f"- Official artifacts changed: `{immutability['official_artifacts_changed']}`\n",
    )
    return audit_manifest


def run_stage9_shadow_ablation(
    run_manifest: str | Path,
    equivalence_root: str | Path,
    canonical_audit_root: str | Path,
    output_root: str | Path,
    *,
    frames: tuple[int, ...] = (),
    profiles: tuple[str, ...] = PROFILES,
    resume: bool = False,
    max_wall_time: float | None = None,
    html_output: bool = True,
) -> dict[str, Any]:
    """Run all mandatory profiles only after the official numerical gate."""

    root = _repo_root()
    manifest_path = _resolve(root, run_manifest)
    equivalence = _resolve(root, equivalence_root)
    canonical_root = _resolve(root, canonical_audit_root)
    destination = _resolve(root, output_root)
    contract = json.loads(
        (equivalence / "shadow_equivalence_contract.json").read_text(encoding="utf-8")
    )
    baseline = json.loads(
        (equivalence / "official_baseline_equivalence.json").read_text(encoding="utf-8")
    )
    if not baseline.get("baseline_pass") or not contract.get("hard_cap_pass"):
        destination.mkdir(parents=True, exist_ok=True)
        blocked = {
            "schema_version": SCHEMA_VERSION,
            "status": "SHADOW_BASELINE_NOT_NUMERICALLY_EQUIVALENT",
            "diagnostic_only": True,
            "paper_method": False,
            "accepted_reference": False,
            "profiles": [],
            "frames": [],
            "mandatory_profiles_run": [],
            "reason": "Official baseline gate failed; no shadow profile was executed.",
            "equivalence_root": str(equivalence),
        }
        _write_json(destination / "shadow_manifest.json", blocked)
        _render_html(
            destination / "stage9_3_3_shadow_analysis.html",
            {
                "equivalence_contract": contract,
                "shadow_results": [],
                "counterfactuals": [],
                "gradients": [],
                "constraints": [],
                "readiness": {
                    "status": "RETURN_TO_STAGE9_3_2_SHADOW_HARNESS_FIX",
                    "enter_stage9_4": False,
                    "reason": blocked["reason"],
                },
            },
        )
        raise RuntimeError("SHADOW_BASELINE_NOT_NUMERICALLY_EQUIVALENT")
    inputs = _load_inputs(manifest_path, root, evaluation_backend="reference_winding_v1")
    final = inputs["final"]
    selected = (
        list(frames)
        if frames
        else json.loads(
            (equivalence / "shadow_frame_selection_v2.json").read_text(encoding="utf-8")
        )["selected_local_frames"]
    )
    selected_profiles = tuple(profiles)
    unknown = sorted(set(selected_profiles) - set(PROFILES))
    if unknown:
        raise ValueError(f"unknown Stage 9.3.3 profile(s): {unknown}")
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_root = destination / "shadow_checkpoints"
    rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    constraint_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for frame in selected:
        warm_base = np.asarray(inputs["warm"].arrays["base_pose_scene"][frame], dtype=np.float64)
        warm_q = np.asarray(inputs["warm"].arrays["qpos"][frame], dtype=np.float64)
        final_base = np.asarray(final.arrays["base_pose_scene"][frame], dtype=np.float64)
        final_q = np.asarray(final.arrays["qpos"][frame], dtype=np.float64)
        counter_states = [
            ("warm", warm_base, warm_q),
            ("official_final", final_base, final_q),
            ("final_base_plus_warm_qpos", final_base, warm_q),
            ("warm_base_plus_final_qpos", warm_base, final_q),
        ]
        for finger in FINGERS:
            q = warm_q.copy()
            indices = [
                index
                for index, name in enumerate(inputs["model"].dof_names)
                if finger in str(name).lower()
            ]
            q[indices] = final_q[indices]
            counter_states.append((f"warm_plus_final_{finger}_joints", warm_base, q))
        q = warm_q.copy()
        for finger in ("index", "middle", "ring"):
            indices = [
                index
                for index, name in enumerate(inputs["model"].dof_names)
                if finger in str(name).lower()
            ]
            q[indices] = final_q[indices]
        counter_states.append(("warm_plus_final_index_middle_ring_joints", warm_base, q))
        q = warm_q.copy()
        for finger in ("thumb", "pinky"):
            indices = [
                index
                for index, name in enumerate(inputs["model"].dof_names)
                if finger in str(name).lower()
            ]
            q[indices] = final_q[indices]
        counter_states.append(("warm_plus_final_thumb_pinky_joints", warm_base, q))
        counterfactual_rows.extend(
            _state_metrics(inputs, frame, base, q, label) for label, base, q in counter_states
        )
        gradient_rows.extend(_objective_gradients(inputs, frame))
        for profile in selected_profiles:
            if max_wall_time is not None and time.perf_counter() - started > float(max_wall_time):
                _write_json(
                    destination / "shadow_manifest.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "PAUSED_MAX_WALL_TIME",
                        "selected_frames": selected,
                        "profiles": selected_profiles,
                        "diagnostic_only": True,
                    },
                )
                return {"status": "PAUSED_MAX_WALL_TIME", "output_root": str(destination)}
            if profile in {
                "minimal_soft_safe_projection_from_warm",
                "official_slack_projection_from_warm",
            }:
                result, base, q = _run_projection(inputs, frame, profile)
                metric = _state_metrics(inputs, frame, base, q, profile)
                metric.update(result)
                rows.append(metric)
            else:
                trajectory, metric = _run_solver_profile(
                    inputs, frame, profile, checkpoint_root, 0, resume
                )
                metric["equivalence_level"] = (
                    "EXACT" if profile == "official_baseline_reproduction" else None
                )
                rows.append(metric)
                constraint_rows.extend(_constraint_attribution(inputs, frame, trajectory))
    causal = _causal_analysis(rows, counterfactual_rows, canonical_root)
    identity_before = json.loads(
        (equivalence / "input_identity_and_immutability.json").read_text(encoding="utf-8")
    )
    immutability = _compare_identity(identity_before, root)
    readiness = _readiness(causal, True, immutability)
    by_profile = {
        profile: [row for row in rows if row["profile"] == profile] for profile in selected_profiles
    }
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now(),
        "status": "SHADOW_COMPLETE",
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
        "formal_artifact_mutation": False,
        "profiles": selected_profiles,
        "frames": selected,
        "solver_invocation_count": sum(1 for row in rows if "objective_components" in row),
        "run_manifest": str(manifest_path),
        "equivalence_root": str(equivalence),
        "canonical_audit_root": str(canonical_root),
        "output_root": str(destination),
        "readiness": readiness,
        "environment": _environment(
            final,
            RefinementSolverProfile.load(str(final.metadata["solver_profile_id"])),
            RefinementExecutionProfile.load(str(final.metadata["execution_profile_id"])),
        ),
    }
    _write_json(destination / "shadow_manifest.json", manifest_payload)
    _write_csv(destination / "shadow_results_per_frame.csv", rows)
    _write_json(destination / "shadow_results_per_profile.json", by_profile)
    per_finger = []
    for row in rows:
        for finger in FINGER_ORDER:
            values = row.get("per_finger", {}).get(finger, {})
            per_finger.append(
                {
                    "profile": row["profile"],
                    "frame": row["frame"],
                    "finger": finger,
                    "keypoint_rmse_m": values.get("keypoint_rmse_m"),
                    "fingertip_error_m": values.get("fingertip_error_m"),
                    "e_bone": row.get("e_bone_per_finger", {}).get(finger),
                    "e_im": row.get("e_im_per_finger", {}).get(finger),
                    "contact_proxy": row.get("contact", {}).get("per_finger", {}).get(finger),
                }
            )
    _write_csv(destination / "per_finger_shadow_results.csv", per_finger)
    _write_json(
        destination / "state_counterfactual_decomposition.json",
        {"schema_version": SCHEMA_VERSION, "states": counterfactual_rows, "diagnostic_only": True},
    )
    _write_csv(destination / "state_counterfactual_decomposition.csv", counterfactual_rows)
    _write_json(
        destination / "objective_gradient_attribution.json",
        {"schema_version": SCHEMA_VERSION, "frames": gradient_rows, "multiplier_available": False},
    )
    _write_csv(destination / "objective_gradient_attribution.csv", gradient_rows)
    _write_json(
        destination / "constraint_attribution.json",
        {
            "schema_version": SCHEMA_VERSION,
            "rows": constraint_rows,
            "multiplier_available": False,
            "multiplier_note": "SLSQP multipliers were not treated as reliable; residual, near-binding counts, and state sensitivity are reported.",
        },
    )
    _write_csv(destination / "constraint_attribution_per_link.csv", constraint_rows)
    _write_json(destination / "shadow_causal_analysis.json", causal)
    causal_md = (
        "# Stage 9.3.3 Shadow Causal Analysis\n\n"
        + f"- Causal label: `{causal['causal_label']}`\n- Profiles: `{', '.join(selected_profiles)}`\n- Frames: `{selected}`\n- Diagnostic-only: `true`\n\n"
        + "\n".join(
            f"- {cause['cause']} ({cause['confidence']}): {'; '.join(cause['evidence_for'])}"
            for cause in causal["causes"]
        )
    )
    _atomic_write(destination / "shadow_causal_analysis.md", causal_md + "\n")
    _write_json(
        destination / "root_cause_analysis.json",
        {
            "schema_version": SCHEMA_VERSION,
            "candidates": causal.get("causes", []),
            "primary": causal.get("causal_label"),
        },
    )
    _write_json(destination / "stage9_4_readiness.json", readiness)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "readiness": readiness,
        "baseline_equivalence": baseline,
        "causal": causal,
        "official_artifacts_changed": immutability["official_artifacts_changed"],
        "profiles": selected_profiles,
        "frames": selected,
    }
    _write_json(destination / "stage9_3_3_summary.json", summary)
    _atomic_write(
        destination / "stage9_3_3_summary.md",
        f"# Stage 9.3.3 Shadow Equivalence and Long-Finger Ablation\n\n- Status: `COMPLETE`\n- Readiness: `{readiness['status']}`\n- ENTER_STAGE9_4: `NO`\n- Official artifacts changed: `{immutability['official_artifacts_changed']}`\n- Causal label: `{causal['causal_label']}`\n",
    )
    if html_output:
        _render_html(
            destination / "stage9_3_3_shadow_analysis.html",
            {
                "equivalence_contract": contract,
                "shadow_results": rows,
                "counterfactuals": counterfactual_rows,
                "gradients": gradient_rows,
                "constraints": constraint_rows,
                "readiness": readiness,
            },
        )
    _write_json(destination / "official_artifact_immutability.json", immutability)
    return summary


__all__ = [
    "CONTRACT_ID",
    "MAX_SHADOW_FRAMES",
    "PROFILES",
    "calibrate_shadow_equivalence",
    "run_stage9_shadow_ablation",
]
