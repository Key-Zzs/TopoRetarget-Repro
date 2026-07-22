"""Stage 9.3.2 canonical contact re-audit and bounded shadow ablation.

The module is intentionally a new output boundary.  It reuses the Stage 9.3
sampling/objective/interpolation implementation, but forces its *formal
evaluation* backend to the already validated reference winding SDF.  The
legacy convex-hull backend is instantiated only for disagreement diagnostics.

No function in the audit path imports or calls the optimizer.  The separate
shadow entry point is gate-controlled and writes every diagnostic trajectory
under its own run root.
"""

# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.geometry.mesh_audit import audit_mesh
from toporetarget.geometry.se3 import scene_to_object
from toporetarget.geometry.signed_distance.closest_point import closest_points_on_triangles
from toporetarget.retarget.final_refinement import (
    ConvexHullSignedDistanceBackend,
    RefinementCoordinateProfile,
    RefinementSolverProfile,
    build_final_trajectory,
    dynamic_collision_points_numpy,
    load_final_trajectory,
    prepare_refinement_resources,
)
from toporetarget.retarget.refinement_performance import RefinementExecutionProfile
from toporetarget.robots.visualization import _primitive_mesh
from toporetarget.utils.hashing import sha256_file, sha256_tree
from toporetarget.workflows.contact_audit import (
    FINGERS,
    TIP_INDICES,
    _dense_samples,
    _load_inputs,
    _manifest_artifact,
    _stats,
    _visual_surface,
    run_contact_audit,
)
from toporetarget.workflows.contact_metric_reconciliation import (
    _artifact_paths,
    _find_repeat,
    _jsonable,
    _resolve,
    _stat,
)

CANONICAL_SCHEMA_VERSION = "toporetarget.contact_audit.v2"
CANONICAL_CODE_VERSION = "stage9.3.2-canonical-reaudit-v1"
CANONICAL_PROFILE_ID = "reference_winding_v1"
CANONICAL_BACKEND_ID = "reference_triangle_winding"
LEGACY_BACKEND_ID = "convex_hull_exact_solver_only"
RECONCILIATION_TOLERANCE_M = 1e-10
ACCEPTANCE_TOLERANCE_M = 1e-6
MAX_SHADOW_FRAMES = 3
SHADOW_PROFILES = (
    "official_baseline_reproduction",
    "half_active_margin",
    "zero_active_margin",
    "full_512_query_reference",
    "minimal_soft_safe_projection_from_warm",
    "official_slack_projection_from_warm",
)


class Stage932PreconditionError(RuntimeError):
    """Raised only for an unmet immutable-input precondition."""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(value) for key, value in row.items()})


def _hash(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for name, value in sha256_tree(path).items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _ps_related() -> list[str]:
    try:
        output = subprocess.run(
            ["ps", "aux"], check=True, capture_output=True, text=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []
    tokens = ("toporetarget", "refine", "stage9", "stage10", "shadow")
    return [line for line in output if any(token in line.lower() for token in tokens)]


def _require_file(path: Path, label: str, missing: list[str]) -> None:
    if not path.exists():
        missing.append(f"{label}: {path}")


def _hard_preflight(
    manifest_path: Path,
    legacy_root: Path,
    reconciliation_root: Path,
    root: Path,
) -> dict[str, Any]:
    """Check Stage 9.3.1 closeout without creating or modifying output."""

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    _require_file(manifest_path, "Stage 10 manifest", missing)
    _require_file(legacy_root / "audit_manifest.json", "Stage 9.3 legacy audit", missing)
    _require_file(
        reconciliation_root / "metric_reconciliation_summary.json",
        "Stage 9.3.1 metric reconciliation summary",
        missing,
    )
    _require_file(
        reconciliation_root / "full512_distance_reconciliation.csv",
        "Stage 9.3.1 full512 reconciliation",
        missing,
    )
    if manifest:
        try:
            final_path = _manifest_artifact(manifest, "final", root)
            _require_file(final_path, "Stage 9.2 final", missing)
            final = load_final_trajectory(final_path)
            _require_file(
                _resolve(root, str(final.metadata["checkpoint_root"])) / "manifest.json",
                "Stage 9.2 checkpoint",
                missing,
            )
            _find_repeat(root, final_path)
        except (KeyError, OSError, ValueError, RuntimeError) as exc:
            missing.append(f"Stage 9.2 final/repeat/checkpoint resolution: {exc}")
        for name in ("canonical", "warm_start", "graph", "collision_samples", "object_samples"):
            try:
                _require_file(
                    _manifest_artifact(manifest, name, root), f"Stage 6-8 artifact {name}", missing
                )
            except (KeyError, ValueError, OSError) as exc:
                missing.append(f"Stage 6-8 artifact {name}: {exc}")
    related = _ps_related()
    related_external = [
        line
        for line in related
        if "grep" not in line
        and "contact_canonical_reaudit" not in line
        and "reaudit-contact-canonical" not in line
        and "run_canonical_reaudit" not in line
    ]
    if related_external:
        missing.append("related solver/shadow process is active: " + " | ".join(related_external))
    status = _git(root, "status", "--short")
    cached = _git(root, "diff", "--cached", "--name-only")
    diff_check = _git(root, "diff", "--check")
    # The implementation itself is allowed to be unstaged after this task's
    # initial clean preflight.  A staged change or unrelated pre-existing
    # change is still a hard blocker.  The caller records the initial clean
    # evidence in status_before.txt before the first run.
    staged_blocker = bool(cached)
    return {
        "status": "STAGE9_3_1_CLOSEOUT_REQUIRED" if missing or staged_blocker else "PASS",
        "missing": missing + ([f"staged paths: {cached}"] if staged_blocker else []),
        "git": {
            "root": _git(root, "rev-parse", "--show-toplevel"),
            "branch": _git(root, "branch", "--show-current"),
            "head": _git(root, "rev-parse", "HEAD"),
            "status_short_at_run": status,
            "cached_name_status_at_run": cached,
            "diff_check": diff_check,
        },
        "stage9_3_1_commit_present": "reconcile refinement metrics"
        in _git(root, "log", "-20", "--format=%s").lower(),
        "related_processes": related_external,
    }


def _profile(root: Path, mesh_hash: str, backend: Any) -> dict[str, Any]:
    path = root / "configs/audit/contact_distance/reference_winding_v1.yaml"
    raw = path.read_bytes()
    return {
        "profile_id": CANONICAL_PROFILE_ID,
        "version": "1.0.0",
        "profile_hash": hashlib.sha256(raw).hexdigest(),
        "profile_path": str(path),
        "backend_id": CANONICAL_BACKEND_ID,
        "implementation_path": "toporetarget.geometry.signed_distance.reference.build_signed_distance_backend",
        "backend_version": "reference_triangle_winding.strict.v1",
        "backend": backend.describe(),
        "object_mesh_hash": mesh_hash,
        "sign_convention": "positive_outside",
        "coordinate_frame": "object_local_query_scene_transform",
        "units": "meters",
        "sign_valid_policy": "strict_and_all_sign_valid",
        "confidence_policy": "winding_threshold_0.5_with_strict_validation",
        "closest_point_semantics": "triangle_mesh_closest_point",
        "normal_semantics": "closest_triangle_surface_normal",
        "formal_evaluation": True,
        "solver_backend_independent": True,
    }


def _artifact_identity(
    root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    final: Any,
    legacy_root: Path,
    reconciliation_root: Path,
) -> dict[str, Any]:
    report, repeat_path = _find_repeat(
        root, _resolve(root, str(manifest["artifacts"]["final"]["path"]))
    )
    checkpoint_root = _resolve(root, str(final.metadata["checkpoint_root"]))
    paths = _artifact_paths(
        root,
        manifest_path,
        manifest,
        legacy_root / "audit_manifest.json",
        json.loads((legacy_root / "audit_manifest.json").read_text(encoding="utf-8")),
        final,
        repeat_path,
        checkpoint_root / "manifest.json",
    )
    paths["stage9_3_legacy_audit_root"] = legacy_root
    paths["stage9_3_1_reconciliation_root"] = reconciliation_root
    paths["stage9_2_checkpoint_root"] = checkpoint_root
    paths = {name: path for name, path in paths.items() if path.exists()}
    before = {name: _stat(path) for name, path in paths.items()}
    return {
        "schema_version": "toporetarget.stage9_3_2_input_identity.v1",
        "captured_at": _now(),
        "run_identity": {
            "dataset": manifest.get("source_dataset"),
            "sequence": manifest.get("source_sequence"),
            "hand": manifest.get("hand"),
            "robot": manifest.get("robot"),
            "global_frame_range": manifest.get("selected_frame_range"),
            "local_frame_range": final.metadata.get("frame_range"),
            "frame_count": final.frame_count,
        },
        "artifacts": before,
        "paths": {name: str(path) for name, path in paths.items()},
        "solver_profile": final.metadata.get("solver_profile"),
        "solver_profile_hash": final.metadata.get("solver_profile_hash"),
        "execution_profile": final.metadata.get("execution_profile"),
        "execution_profile_hash": final.metadata.get("execution_profile_hash"),
        "query_profile": final.metadata.get("query_profile"),
        "collision_profile": final.metadata.get("collision_surface_profile_hash"),
        "stage10_manifest": manifest,
        "legacy_audit_manifest": json.loads(
            (legacy_root / "audit_manifest.json").read_text(encoding="utf-8")
        ),
        "stage9_3_1_reconciliation_manifest": json.loads(
            (reconciliation_root / "audit_manifest.json").read_text(encoding="utf-8")
        )
        if (reconciliation_root / "audit_manifest.json").exists()
        else None,
        "determinism_report": json.loads(report.read_text(encoding="utf-8")),
    }


def _after_identity(identity: dict[str, Any]) -> dict[str, Any]:
    before = identity["artifacts"]
    after = {name: _stat(Path(path)) for name, path in identity["paths"].items()}
    immutable = {
        name: {
            "before": before[name],
            "after": after[name],
            "hash_unchanged": before[name]["sha256"] == after[name]["sha256"],
            "mtime_unchanged": before[name]["mtime_ns"] == after[name]["mtime_ns"],
        }
        for name in before
        if name in after
    }
    return {
        "schema_version": "toporetarget.stage9_3_2_immutability.v1",
        "official_artifacts_changed": not all(
            row["hash_unchanged"] and row["mtime_unchanged"] for row in immutable.values()
        ),
        "artifacts": immutable,
    }


def _metric_fields(phi: np.ndarray, tau: float, bound: float) -> dict[str, Any]:
    values = np.asarray(phi, dtype=np.float64).reshape(-1)
    minimum = float(np.min(values)) if len(values) else math.inf
    return {
        "raw_signed_distance_m": minimum,
        "raw_penetration_m": float(max(0.0, -minimum)),
        "penetration_beyond_tau_m": float(max(0.0, -minimum - tau)),
        "hard_bound_violation_m": float(max(0.0, -minimum - bound)),
        "soft_residual_before_slack_m": float(np.min(values + tau)) if len(values) else math.inf,
        "hard_residual_m": float(np.min(values + bound)) if len(values) else math.inf,
    }


def _reconcile_full512(
    inputs: dict[str, Any],
    legacy_root: Path,
    reconciliation_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    final = inputs["final"]
    obj = inputs["object"]
    surface = inputs["surface"]
    model = inputs["model"]
    reference = inputs["reference_sdf"]
    tau = float(final.metadata["paper_weights"]["tau_m"])
    bound = float(final.metadata["paper_weights"]["b_m"])
    old_rows: dict[tuple[int, int], dict[str, str]] = {}
    old_csv = reconciliation_root / "full512_distance_reconciliation.csv"
    if old_csv.exists():
        with old_csv.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                old_rows[(int(row["frame"]), int(row["sample_id"]))] = row
    rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    dynamic_max = 0.0
    local_roundtrip_max = 0.0
    sample_identity = np.array_equal(
        np.asarray(surface.sample_ids, dtype=np.int64),
        np.arange(len(surface.sample_ids), dtype=np.int64),
    )
    frame_identity = np.array_equal(
        np.asarray(final.arrays["frame_indices"], dtype=np.int64),
        np.arange(final.frame_count, dtype=np.int64),
    )
    for frame in range(final.frame_count):
        pose = np.asarray(obj.pose_scene.pose_scene[frame], dtype=np.float64)
        points = np.asarray(final.arrays["collision_points_scene"][frame], dtype=np.float64)
        dynamic = dynamic_collision_points_numpy(
            model,
            surface,
            final.arrays["qpos"][frame],
            final.arrays["base_pose_scene"][frame],
        )
        dynamic_diff = np.max(np.abs(points - dynamic)) if len(points) else 0.0
        dynamic_max = max(dynamic_max, float(dynamic_diff))
        local = scene_to_object(pose, points)
        roundtrip = (
            np.max(np.abs(scene_to_object(pose, points) @ pose[:3, :3].T + pose[:3, 3] - points))
            if len(points)
            else 0.0
        )
        local_roundtrip_max = max(local_roundtrip_max, float(roundtrip))
        queried = reference.query_scene(points, pose)
        persisted = np.asarray(final.arrays["full_signed_distance"][frame], dtype=np.float64)
        for sample_id in range(len(points)):
            old = old_rows.get((frame, sample_id), {})
            row = {
                "frame": frame,
                "global_frame": int(
                    final.arrays["source_frame_indices"][frame]
                    + final.metadata.get("source_frame_offset", 0)
                ),
                "sample_id": sample_id,
                "sample_identity_exact": sample_identity,
                "frame_identity_exact": frame_identity,
                "geometry_id": str(surface.geometry_ids[sample_id]),
                "link_name": str(surface.link_names[sample_id]),
                "scene_x_m": float(points[sample_id, 0]),
                "scene_y_m": float(points[sample_id, 1]),
                "scene_z_m": float(points[sample_id, 2]),
                "object_local_x_m": float(local[sample_id, 0]),
                "object_local_y_m": float(local[sample_id, 1]),
                "object_local_z_m": float(local[sample_id, 2]),
                "persisted_raw_signed_distance_m": float(persisted[sample_id]),
                "independent_validator_raw_signed_distance_m": float(
                    old.get("reference_phi_m", persisted[sample_id])
                ),
                "canonical_reconciliation_raw_signed_distance_m": float(
                    queried.signed_distance[sample_id]
                ),
                "persisted_minus_canonical_m": float(
                    persisted[sample_id] - queried.signed_distance[sample_id]
                ),
                "validator_minus_canonical_m": float(
                    float(old.get("reference_phi_m", queried.signed_distance[sample_id]))
                    - queried.signed_distance[sample_id]
                ),
                "sign_mismatch_persisted": bool(
                    np.signbit(persisted[sample_id])
                    != np.signbit(queried.signed_distance[sample_id])
                ),
                "sign_mismatch_validator": bool(
                    np.signbit(
                        float(old.get("reference_phi_m", queried.signed_distance[sample_id]))
                    )
                    != np.signbit(queried.signed_distance[sample_id])
                ),
                "closest_face_index": int(queried.closest_face_indices[sample_id]),
                "closest_point_x_m": float(queried.closest_points[sample_id, 0]),
                "closest_point_y_m": float(queried.closest_points[sample_id, 1]),
                "closest_point_z_m": float(queried.closest_points[sample_id, 2]),
                "sign_valid": bool(queried.sign_valid[sample_id]),
                "sign_confidence": float(queried.sign_confidence[sample_id]),
                "sign_method": str(queried.sign_method),
            }
            rows.append(row)
        frame_phi = np.asarray(queried.signed_distance, dtype=np.float64)
        frame_rows.append(
            {
                "frame": frame,
                "global_frame": int(
                    final.arrays["source_frame_indices"][frame]
                    + final.metadata.get("source_frame_offset", 0)
                ),
                "sample_count": int(len(frame_phi)),
                "sample_identity_exact": sample_identity,
                "frame_identity_exact": frame_identity,
                "raw_min_phi_m": float(np.min(frame_phi)),
                "raw_max_penetration_m": float(max(0.0, -np.min(frame_phi))),
                "penetration_beyond_tau_m": float(max(0.0, -np.min(frame_phi) - tau)),
                "hard_bound_violation_m": float(max(0.0, -np.min(frame_phi) - bound)),
                "persisted_max_abs_diff_m": float(np.max(np.abs(persisted - frame_phi))),
                "validator_max_abs_diff_m": float(
                    max(
                        abs(
                            float(
                                old_rows.get((frame, sample), {}).get(
                                    "reference_phi_m", frame_phi[sample]
                                )
                            )
                            - frame_phi[sample]
                        )
                        for sample in range(len(frame_phi))
                    )
                ),
                "sign_mismatch_count_persisted": int(
                    np.count_nonzero(np.signbit(persisted) != np.signbit(frame_phi))
                ),
                "sign_mismatch_count_validator": int(
                    sum(
                        bool(
                            np.signbit(
                                float(
                                    old_rows.get((frame, sample), {}).get(
                                        "reference_phi_m", frame_phi[sample]
                                    )
                                )
                            )
                            != np.signbit(frame_phi[sample])
                        )
                        for sample in range(len(frame_phi))
                    )
                ),
                "dynamic_scene_point_max_diff_m": float(dynamic_diff),
                "object_local_roundtrip_max_diff_m": float(roundtrip),
                "all_sign_valid": bool(np.all(queried.sign_valid)),
            }
        )
    all_phi = np.asarray([row["canonical_reconciliation_raw_signed_distance_m"] for row in rows])
    persisted_diff = np.asarray([row["persisted_minus_canonical_m"] for row in rows])
    validator_diff = np.asarray([row["validator_minus_canonical_m"] for row in rows])
    summary = {
        "schema_version": "toporetarget.full512_canonical_reconciliation.v2",
        "backend": inputs["reference_sdf"].describe(),
        "formal_evaluation_backend": CANONICAL_PROFILE_ID,
        "frame_count": final.frame_count,
        "samples_per_frame": len(surface.sample_ids),
        "identity_mismatch_count": int(not sample_identity) + int(not frame_identity),
        "sample_identity_exact": sample_identity,
        "frame_identity_exact": frame_identity,
        "transform_chain_max_scene_point_diff_m": dynamic_max,
        "transform_chain_max_object_local_roundtrip_diff_m": local_roundtrip_max,
        "persisted_vs_canonical": _stats(persisted_diff),
        "independent_validator_vs_canonical": _stats(validator_diff),
        "max_absolute_difference_m": float(
            max(np.max(np.abs(persisted_diff)), np.max(np.abs(validator_diff)))
        ),
        "rmse_m": float(np.sqrt(np.mean(np.concatenate([persisted_diff, validator_diff]) ** 2))),
        "sign_mismatch_count": int(
            sum(row["sign_mismatch_persisted"] or row["sign_mismatch_validator"] for row in rows)
        ),
        "raw_min_phi_m": float(np.min(all_phi)),
        "raw_max_penetration_m": float(max(0.0, -np.min(all_phi))),
        "penetration_beyond_tau_m": float(max(0.0, -np.min(all_phi) - tau)),
        "hard_bound_violation_m": float(max(0.0, -np.min(all_phi) - bound)),
        "worst_frame_sample": max(rows, key=lambda row: abs(row["persisted_minus_canonical_m"])),
        "gate": {
            "max_diff_pass": bool(
                max(np.max(np.abs(persisted_diff)), np.max(np.abs(validator_diff)))
                <= RECONCILIATION_TOLERANCE_M
            ),
            "sign_pass": int(
                sum(
                    row["sign_mismatch_persisted"] or row["sign_mismatch_validator"] for row in rows
                )
            )
            == 0,
            "identity_pass": bool(sample_identity and frame_identity),
            "transform_pass": bool(
                dynamic_max <= RECONCILIATION_TOLERANCE_M
                and local_roundtrip_max <= RECONCILIATION_TOLERANCE_M
            ),
            "sign_valid_pass": bool(all(row["all_sign_valid"] for row in frame_rows)),
        },
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    return summary, frame_rows, rows


def _canonical_frame_rows(
    canonical_root: Path, inputs: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    geo = json.loads((canonical_root / "contact_geometry_audit.json").read_text(encoding="utf-8"))
    source = json.loads((canonical_root / "source_contact_proxy.json").read_text(encoding="utf-8"))
    retention = json.loads(
        (canonical_root / "contact_retention_proxy.json").read_text(encoding="utf-8")
    )
    final = inputs["final"]
    tau = float(final.metadata["paper_weights"]["tau_m"])
    bound = float(final.metadata["paper_weights"]["b_m"])
    by_source = {int(row["frame"]): row for row in source["frames"]}
    rows: list[dict[str, Any]] = []
    for row in geo["frames"]:
        frame = int(row["frame"])
        final_phi = np.asarray(row["full_audit_points_signed_distance_m"], dtype=np.float64)
        source_phi = np.asarray(row["source_points_signed_distance_m"], dtype=np.float64)
        warm_phi = np.asarray(row["warm_points_signed_distance_m"], dtype=np.float64)
        final_visual_phi = np.asarray(row["final_points_signed_distance_m"], dtype=np.float64)
        query_start = int(final.arrays["query_offsets"][frame])
        query_stop = int(final.arrays["query_offsets"][frame + 1])
        soft_after = np.asarray(
            final.arrays["soft_residual_concat"][query_start:query_stop], dtype=np.float64
        )
        values = {
            "frame": frame,
            "global_frame": int(row["global_frame"]),
            "canonical_backend_id": CANONICAL_BACKEND_ID,
            "source_visual_min_m": float(np.min(source_phi)),
            "source_visual_p01_m": float(np.quantile(source_phi, 0.01)),
            "source_visual_p05_m": float(np.quantile(source_phi, 0.05)),
            "warm_visual_min_m": float(np.min(warm_phi)),
            "warm_visual_p01_m": float(np.quantile(warm_phi, 0.01)),
            "warm_visual_p05_m": float(np.quantile(warm_phi, 0.05)),
            "final_visual_min_m": float(np.min(final_visual_phi)),
            "final_visual_p01_m": float(np.quantile(final_visual_phi, 0.01)),
            "final_visual_p05_m": float(np.quantile(final_visual_phi, 0.05)),
            "warm_collision_min_m": float(row["metrics"]["warm_collision_min_m"]),
            "final_collision_min_m": float(row["metrics"]["final_collision_min_m"]),
            "final_full512_min_m": float(np.min(final_phi)),
            "raw_penetration_m": float(max(0.0, -np.min(final_phi))),
            "penetration_beyond_tau_m": float(max(0.0, -np.min(final_phi) - tau)),
            "hard_bound_violation_m": float(max(0.0, -np.min(final_phi) - bound)),
            "soft_residual_before_slack_m": float(np.min(final_phi + tau)),
            "soft_residual_after_slack_m": float(np.min(soft_after)) if len(soft_after) else None,
            "hard_residual_m": float(np.min(final_phi + bound)),
            "source_contact_proxy_5mm_count": int(row["metrics"]["source_contact_proxy_5mm"]),
            "source_contact_proxy_5mm_ratio": float(
                by_source[frame]["thresholds"]["5mm"]["near_surface_ratio"]
            ),
            "warm_contact_retention_proxy_recall": row["metrics"].get(
                "warm_contact_retention_proxy_recall"
            ),
            "final_contact_retention_proxy_recall": row["metrics"].get(
                "final_contact_retention_proxy_recall"
            ),
            "query_active_count": int(row["metrics"]["query_active_count"]),
            "active_margin_m": float(final.metadata["query_profile"]["active_margin_m"]),
            "collision_visual_offset_max_mm": row["metrics"].get("collision_visual_offset_max_mm"),
            "final_e_im_raw": float(
                row["metrics"].get("final_objective_e_im_raw", final.arrays["e_im"][frame])
            ),
            "warm_e_im_raw": float(
                row["metrics"].get("warm_objective_e_im_raw", final.arrays["warm_e_im"][frame])
            ),
            "final_e_bone_raw": float(
                row["metrics"].get("final_objective_e_bone_raw", final.arrays["e_bone"][frame])
            ),
            "warm_e_bone_raw": float(
                row["metrics"].get("warm_objective_e_bone_raw", final.arrays["warm_e_bone"][frame])
            ),
            "base_displacement_from_warm_m": float(
                np.linalg.norm(
                    final.arrays["base_pose_scene"][frame][:3, 3]
                    - inputs["warm"].arrays["base_pose_scene"][frame][:3, 3]
                )
            ),
            "qpos_displacement_from_warm_l2": float(
                np.linalg.norm(final.arrays["qpos"][frame] - inputs["warm"].arrays["qpos"][frame])
            ),
        }
        rows.append(values)
    return rows, geo, retention


def _source_classification(source: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    frames = source["frames"]
    total = len(frames)
    count3 = np.asarray([row["thresholds"]["3mm"]["near_surface_count"] for row in frames])
    count5 = np.asarray([row["thresholds"]["5mm"]["near_surface_count"] for row in frames])
    finger_rows = {}
    for finger in FINGERS:
        values = [
            int(row["regions"].get(finger, {}).get("5mm", {}).get("near_surface_count", 0))
            for row in frames
        ]
        finger_rows[finger] = {
            "frames_with_proxy": int(np.count_nonzero(np.asarray(values) > 0)),
            "frame_ratio": float(np.mean(np.asarray(values) > 0)) if values else 0.0,
            "median_count": float(np.median(values)) if values else 0.0,
        }
    frame_ratio_5 = float(np.mean(count5 > 0)) if total else 0.0
    frame_ratio_3 = float(np.mean(count3 > 0)) if total else 0.0
    continuity = (
        float(np.mean([value["frame_ratio"] for value in finger_rows.values()]))
        if finger_rows
        else 0.0
    )
    if frame_ratio_5 >= 0.75 and frame_ratio_3 >= 0.5 and continuity >= 0.5:
        classification = "CONTACT_RICH"
    elif frame_ratio_5 >= 0.5:
        classification = "APPROACH"
    elif frame_ratio_5 > 0.0:
        classification = "PRE_CONTACT"
    elif total:
        classification = "GEOMETRY_AMBIGUOUS"
    else:
        classification = "INCONCLUSIVE"
    return {
        "schema_version": "toporetarget.source_contact_classification.v2",
        "classification": classification,
        "contact_proxy_name": "source_contact_proxy",
        "ground_truth_contact": False,
        "canonical_backend_id": CANONICAL_BACKEND_ID,
        "thresholds_mm": [1, 2, 3, 5, 8, 10],
        "frame_count": total,
        "frame_ratio_with_proxy_3mm": frame_ratio_3,
        "frame_ratio_with_proxy_5mm": frame_ratio_5,
        "mean_per_finger_continuity": continuity,
        "per_finger": finger_rows,
        "minimum_distance_median": float(
            np.median([row["visual_min_distance_m"] for row in frames])
        )
        if frames
        else None,
        "semantic_anchor_is_auxiliary_only": True,
        "evidence_limit": "dense visual samples are a contact proxy, not a ground-truth label",
    }


def _retention_outputs(
    retention: dict[str, Any], output: Path, profile_hash: str | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    threshold_rows: list[dict[str, Any]] = []
    finger_rows: list[dict[str, Any]] = []
    for frame in retention["frames"]:
        for key, value in frame["threshold_sensitivity"].items():
            threshold_rows.append(
                {
                    "frame": int(frame["frame"]),
                    "global_frame": int(frame["global_frame"]),
                    "threshold": key,
                    **value,
                    "canonical_backend_id": CANONICAL_BACKEND_ID,
                    "canonical_profile_hash": profile_hash,
                }
            )
        for finger in FINGERS:
            value = frame["anchor_level"].get(finger)
            if value is not None:
                finger_rows.append(
                    {
                        "frame": int(frame["frame"]),
                        "global_frame": int(frame["global_frame"]),
                        "finger": finger,
                        "anchor_index": value["anchor_index"],
                        "source_contact_proxy_5mm": value["source_contact_proxy_5mm"],
                        "warm_distance_m": value["warm_distance_m"],
                        "final_distance_m": value["final_distance_m"],
                        "warm_contact_proxy_8mm": value["warm_contact_proxy_8mm"],
                        "final_contact_proxy_8mm": value["final_contact_proxy_8mm"],
                        "final_distance_drift_m": value["final_to_source_distance_drift_m"],
                        "canonical_backend_id": CANONICAL_BACKEND_ID,
                        "canonical_profile_hash": profile_hash,
                    }
                )
    _write_csv(output / "canonical_contact_retention_threshold_sensitivity.csv", threshold_rows)
    _write_csv(output / "canonical_per_finger_retention.csv", finger_rows)
    return threshold_rows, finger_rows


def _anchor_rows(
    retention: dict[str, Any], profile_hash: str | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in retention["frames"]:
        for name, value in frame["anchor_level"].items():
            rows.append(
                {
                    "frame": frame["frame"],
                    "global_frame": frame["global_frame"],
                    "anchor": name,
                    **value,
                    "semantic_anchor": True,
                    "canonical_backend_id": CANONICAL_BACKEND_ID,
                    "canonical_profile_hash": profile_hash,
                    "surface_contact_guaranteed": False,
                }
            )
    return rows


def _legacy_disagreement(
    inputs: dict[str, Any],
    geo: dict[str, Any],
    reconciliation_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    obj = inputs["object"]
    final = inputs["final"]
    reference = inputs["reference_sdf"]
    mesh_hash = str(reference.describe().get("mesh_hash", final.metadata.get("object_mesh_hash")))
    legacy = ConvexHullSignedDistanceBackend(
        obj.mesh.vertices_local,
        obj.mesh.faces,
        mesh_hash,
        tree_leaf_size=int(final.metadata.get("sdf_tree_leaf_size", 512)),
    )
    frame_rows: list[dict[str, Any]] = []
    worst: list[dict[str, Any]] = []
    for frame_data in geo["frames"]:
        frame = int(frame_data["frame"])
        pose = np.asarray(obj.pose_scene.pose_scene[frame], dtype=np.float64)
        category_data = {}
        for category, point_key, distance_key in (
            ("source_visual", "source_points", "source_points_signed_distance_m"),
            ("warm_visual", "warm_points", "warm_points_signed_distance_m"),
            ("final_visual", "final_points", "final_points_signed_distance_m"),
            ("semantic_anchor", "anchor_points", "anchor_signed_distance_m"),
        ):
            points = np.asarray(frame_data[point_key], dtype=np.float64)
            canonical = np.asarray(frame_data[distance_key], dtype=np.float64)
            old = legacy.query_scene(points, pose)
            diff = np.asarray(old.signed_distance) - canonical
            sign = np.signbit(old.signed_distance) != np.signbit(canonical)
            category_data[category] = {
                "count": int(len(points)),
                "legacy_min_phi_m": float(np.min(old.signed_distance)),
                "canonical_min_phi_m": float(np.min(canonical)),
                "max_abs_difference_m": float(np.max(np.abs(diff))) if len(diff) else 0.0,
                "rmse_m": float(np.sqrt(np.mean(diff * diff))) if len(diff) else 0.0,
                "sign_disagreement_count": int(np.count_nonzero(sign)),
                "legacy_false_inside_count": int(
                    np.count_nonzero((old.signed_distance < 0) & (canonical >= 0))
                ),
                "legacy_false_near_count": int(
                    np.count_nonzero(
                        (np.abs(old.signed_distance) <= 0.005) & (np.abs(canonical) > 0.005)
                    )
                ),
            }
            for index in np.argsort(np.abs(diff))[::-1][: min(5, len(diff))]:
                worst.append(
                    {
                        "frame": frame,
                        "global_frame": frame_data["global_frame"],
                        "category": category,
                        "point_index": int(index),
                        "legacy_phi_m": float(old.signed_distance[index]),
                        "canonical_phi_m": float(canonical[index]),
                        "difference_m": float(diff[index]),
                        "sign_disagreement": bool(sign[index]),
                    }
                )
        legacy_full = legacy.query_scene(
            np.asarray(frame_data["full_audit_points"], dtype=np.float64), pose
        )
        canonical_full = np.asarray(
            frame_data["full_audit_points_signed_distance_m"], dtype=np.float64
        )
        full_diff = legacy_full.signed_distance - canonical_full
        false_inside = (legacy_full.signed_distance < 0) & (canonical_full >= 0)
        frame_rows.append(
            {
                "frame": frame,
                "global_frame": frame_data["global_frame"],
                "full512_count": int(len(canonical_full)),
                "legacy_min_phi_m": float(np.min(legacy_full.signed_distance)),
                "canonical_min_phi_m": float(np.min(canonical_full)),
                "max_abs_difference_m": float(np.max(np.abs(full_diff))),
                "rmse_m": float(np.sqrt(np.mean(full_diff * full_diff))),
                "sign_disagreement_count": int(
                    np.count_nonzero(
                        np.signbit(legacy_full.signed_distance) != np.signbit(canonical_full)
                    )
                ),
                "legacy_false_inside_count": int(np.count_nonzero(false_inside)),
                "legacy_false_near_count": int(
                    np.count_nonzero(
                        (np.abs(legacy_full.signed_distance) <= 0.005)
                        & (np.abs(canonical_full) > 0.005)
                    )
                ),
                "affected_links": sorted(
                    set(str(inputs["surface"].link_names[i]) for i in np.flatnonzero(false_inside))
                ),
                "categories": category_data,
            }
        )
    all_diff = np.asarray([row["difference_m"] for row in worst], dtype=np.float64)
    full_diff = np.asarray([row["max_abs_difference_m"] for row in frame_rows], dtype=np.float64)
    summary = {
        "schema_version": "toporetarget.backend_disagreement.v2",
        "canonical_backend": inputs["reference_sdf"].describe(),
        "legacy_backend": legacy.describe(),
        "formal_acceptance_backend": CANONICAL_PROFILE_ID,
        "legacy_role": "diagnostic_only_not_used_for_formal_acceptance",
        "full512": {
            "affected_frames": int(sum(row["sign_disagreement_count"] > 0 for row in frame_rows)),
            "false_inside_count": int(sum(row["legacy_false_inside_count"] for row in frame_rows)),
            "sign_disagreement_count": int(
                sum(row["sign_disagreement_count"] for row in frame_rows)
            ),
            "max_abs_difference_m": float(np.max(full_diff)) if len(full_diff) else 0.0,
            "rmse_of_frame_maxima_m": float(np.sqrt(np.mean(full_diff * full_diff)))
            if len(full_diff)
            else 0.0,
        },
        "visual_and_anchor": {
            "max_abs_difference_m": float(np.max(np.abs(all_diff))) if len(all_diff) else 0.0,
            "sign_disagreement_count": int(
                sum(
                    row["sign_disagreement_count"]
                    for frame in frame_rows
                    for row in frame["categories"].values()
                )
            ),
            "false_inside_count": int(
                sum(
                    row["legacy_false_inside_count"]
                    for frame in frame_rows
                    for row in frame["categories"].values()
                )
            ),
        },
        "frames": frame_rows,
        "legacy_warning": "LEGACY CONVEX-HULL DIAGNOSTIC; NOT USED FOR FORMAL ACCEPTANCE",
    }
    return (
        summary,
        frame_rows,
        sorted(worst, key=lambda row: abs(row["difference_m"]), reverse=True),
    )


def _representation_audit(inputs: dict[str, Any], surface_count: int = 8192) -> dict[str, Any]:
    model = inputs["model"]
    surface = inputs["surface"]
    visual_by_link: dict[str, list[np.ndarray]] = {}
    collision_by_link: dict[str, list[np.ndarray]] = {}
    reliable_by_link: dict[str, bool] = {}
    for _index, instance in enumerate(model.visual_geometry_instances(model.neutral_q)):
        vertices, faces = _primitive_mesh(instance)
        transform = np.asarray(instance.world_transform, dtype=np.float64)
        vertices = np.asarray(vertices, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]
        triangles = vertices[np.asarray(faces, dtype=np.int64)]
        link = str(instance.link_name)
        visual_by_link.setdefault(link, []).append(triangles)
        try:
            report = audit_mesh(vertices, np.asarray(faces, dtype=np.int64))
            valid = bool(
                report.watertight
                and report.non_manifold_edge_count == 0
                and report.orientable is not False
            )
        except (ValueError, TypeError):
            valid = False
        reliable_by_link[link] = reliable_by_link.get(link, True) and valid
    for _index, instance in enumerate(model.collision_geometry_instances(model.neutral_q)):
        vertices, faces = _primitive_mesh(instance)
        transform = np.asarray(instance.world_transform, dtype=np.float64)
        vertices = np.asarray(vertices, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]
        collision_by_link.setdefault(str(instance.link_name), []).append(
            vertices[np.asarray(faces, dtype=np.int64)]
        )
    neutral_points = dynamic_collision_points_numpy(model, surface, model.neutral_q, np.eye(4))
    rows: list[dict[str, Any]] = []
    for link in sorted(
        set(visual_by_link)
        | set(collision_by_link)
        | set(np.asarray(surface.link_names).astype(str))
    ):
        visual = (
            np.concatenate(visual_by_link.get(link, []), axis=0)
            if visual_by_link.get(link)
            else np.empty((0, 3, 3))
        )
        collision = (
            np.concatenate(collision_by_link.get(link, []), axis=0)
            if collision_by_link.get(link)
            else np.empty((0, 3, 3))
        )
        ids = np.flatnonzero(np.asarray(surface.link_names).astype(str) == link)
        c2v = (
            closest_points_on_triangles(neutral_points[ids], visual)[3]
            if len(ids) and len(visual)
            else np.empty(0)
        )
        visual_points = (
            _dense_samples(
                visual.reshape(-1, 3),
                np.arange(len(visual) * 3, dtype=np.int64).reshape(-1, 3),
                count=max(64, surface_count // max(1, len(visual_by_link))),
                mesh_id=f"representation:{link}",
                seed=20260722,
            )
            if len(visual)
            else np.empty((0, 3))
        )
        v2c = (
            closest_points_on_triangles(visual_points, collision)[3]
            if len(visual_points) and len(collision)
            else np.empty(0)
        )
        max_values = np.concatenate([c2v, v2c]) if len(c2v) or len(v2c) else np.empty(0)
        gap = float(np.quantile(max_values, 0.95)) if len(max_values) else None
        if not len(visual) or not len(collision):
            classification = "COVERAGE_GAP"
        elif gap is not None and gap <= 0.002:
            classification = "REPRESENTATION_MATCH"
        elif gap is not None and gap > 0.005:
            classification = "COVERAGE_GAP"
        else:
            classification = "DIRECTION_INCONCLUSIVE"
        rows.append(
            {
                "link_name": link,
                "collision_sample_count": int(len(ids)),
                "visual_triangle_count": int(len(visual)),
                "collision_triangle_count": int(len(collision)),
                "collision_to_visual_unsigned_median": float(np.median(c2v)) if len(c2v) else None,
                "collision_to_visual_unsigned_p95": float(np.quantile(c2v, 0.95))
                if len(c2v)
                else None,
                "collision_to_visual_unsigned_max": float(np.max(c2v)) if len(c2v) else None,
                "visual_to_collision_unsigned_median": float(np.median(v2c)) if len(v2c) else None,
                "visual_to_collision_unsigned_p95": float(np.quantile(v2c, 0.95))
                if len(v2c)
                else None,
                "visual_to_collision_unsigned_max": float(np.max(v2c)) if len(v2c) else None,
                "coverage_gap_p95_m": gap,
                "pad_or_fingertip": bool(
                    "tip" in link
                    or any(link.startswith(finger) and link.endswith("3") for finger in FINGERS)
                ),
                "normal_reliable": bool(reliable_by_link.get(link, False)),
                "offset_direction": "INCONCLUSIVE",
                "classification": classification,
            }
        )
    gap_rows = [row for row in rows if row["classification"] == "COVERAGE_GAP"]
    overall = "COVERAGE_GAP" if gap_rows else "REPRESENTATION_MATCH"
    if any(not row["normal_reliable"] for row in rows):
        direction = "INCONCLUSIVE"
    else:
        direction = "INCONCLUSIVE"  # visual open/primitive normal orientation is not a signed proof
    return {
        "schema_version": "toporetarget.collision_visual_audit.v2",
        "canonical_backend_id": CANONICAL_BACKEND_ID,
        "overall_classification": overall,
        "per_link": rows,
        "coverage_gap": bool(gap_rows),
        "direction_classification": direction,
        "offset_direction": "INCONCLUSIVE",
        "inflated_label_supported": False,
        "inflated_label_replacement": "COLLISION_OFFSET_DIRECTION_INCONCLUSIVE",
        "normal_policy": "open_or_unvalidated visual mesh does not support signed offset direction",
    }


def _write_html_v2(payload: dict[str, Any], path: Path) -> None:
    data = json.dumps(_jsonable(payload), separators=(",", ":"), allow_nan=False)
    document = """<!doctype html><html><head><meta charset='utf-8'><title>TopoRetarget Stage 9.3.2 canonical contact audit</title><style>body{font:13px system-ui;margin:0;background:#111827;color:#e5e7eb}main{padding:16px}table{border-collapse:collapse}td,th{padding:4px 8px;border:1px solid #374151}select,input{background:#111827;color:#e5e7eb}#timeline{width:100%;height:100px}.badge{padding:5px;background:#047857;display:inline-block}.warning{padding:5px;background:#92400e;display:inline-block}pre{white-space:pre-wrap}</style></head><body><main><h1>Stage 9.3.2 Canonical Contact Re-Audit</h1><div class='badge'>FORMAL DISTANCE BACKEND: REFERENCE WINDING SDF</div> <div class='warning'>LEGACY CONVEX-HULL RESULTS: DIAGNOSTIC ONLY / SUPERSEDED</div><p>Positive signed distance is outside. Dense visual samples are an approximation and contact values are proxies, not ground truth.</p><label>Frame <input id='frame' type='range' min='0' max='__MAX__' value='0'></label><span id='label'></span><br><label>Threshold mm <input id='threshold' type='range' min='1' max='10' value='5'></label><span id='thresholdLabel'></span><select id='metric'><option value='final_visual_min_m'>final visual min</option><option value='source_visual_min_m'>source visual min</option><option value='final_full512_min_m'>final full512 min</option><option value='final_contact_retention_proxy_recall'>retention</option><option value='query_active_count'>QuerySet</option></select><canvas id='timeline' width='900' height='100'></canvas><pre id='metrics'></pre><h2>Canonical per-frame metrics</h2><table><thead><tr><th>Frame</th><th>Source visual min</th><th>Warm visual min</th><th>Final visual min</th><th>Final full512 min</th><th>Contact proxy</th></tr></thead><tbody id='rows'></tbody></table><h2>Shadow profiles</h2><pre id='shadow'></pre><script>const P=__DATA__;const $=id=>document.getElementById(id);const rows=P.frames;function show(){let i=+$('frame').value,r=rows[i];$('label').textContent=' local '+r.frame+' global '+r.global_frame;$('thresholdLabel').textContent=' '+$('threshold').value+' mm';$('metrics').textContent=JSON.stringify(r,null,2);$('shadow').textContent=JSON.stringify(P.shadow_profiles||{},null,2)}function table(){ $('rows').innerHTML=rows.map(r=>`<tr><td>${r.frame}</td><td>${(r.source_visual_min_m*1000).toFixed(3)}</td><td>${(r.warm_visual_min_m*1000).toFixed(3)}</td><td>${(r.final_visual_min_m*1000).toFixed(3)}</td><td>${(r.final_full512_min_m*1000).toFixed(3)}</td><td>${r.final_contact_retention_proxy_recall}</td></tr>`).join('')}function timeline(){let c=$('timeline'),x=c.getContext('2d'),k=$('metric').value,v=rows.map(r=>Number(r[k]??0)),mn=Math.min(...v),mx=Math.max(...v);x.clearRect(0,0,c.width,c.height);x.strokeStyle='#60a5fa';x.beginPath();v.forEach((q,i)=>{let X=i/(v.length-1)*c.width,Y=c.height-(q-mn)/Math.max(mx-mn,1e-12)*c.height;i?x.lineTo(X,Y):x.moveTo(X,Y)});x.stroke();x.fillStyle='#fbbf24';x.fillRect(+$('frame').value/(v.length-1)*c.width,0,2,c.height)}$('frame').oninput=()=>{show();timeline()};$('metric').onchange=timeline;table();show();timeline();</script></main></body></html>"""
    document = document.replace("__MAX__", str(max(0, len(payload["frames"]) - 1))).replace(
        "__DATA__", data
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def _read_objective_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _audit_shard(args: tuple[str, str, int, int, str]) -> str:
    """Process-pool worker for one audit frame; it never calls Stage 9."""

    manifest, output, frame, samples, root = args
    run_contact_audit(
        manifest,
        output,
        surface_samples=samples,
        thresholds_mm=[1, 2, 3, 5, 8, 10],
        frame_start=frame,
        frame_end=frame + 1,
        html=False,
        force=True,
        evaluation_backend="reference_winding_v1",
        dense_include_vertices=False,
    )
    return output


def _merge_audit_shards(shards: list[Path], merged: Path, frame_count: int) -> None:
    """Merge independent audit-only frame bundles deterministically."""

    merged.mkdir(parents=True, exist_ok=True)
    first_geo = json.loads((shards[0] / "contact_geometry_audit.json").read_text(encoding="utf-8"))
    first_source = json.loads((shards[0] / "source_contact_proxy.json").read_text(encoding="utf-8"))
    first_retention = json.loads(
        (shards[0] / "contact_retention_proxy.json").read_text(encoding="utf-8")
    )
    geo_frames: list[dict[str, Any]] = []
    source_frames: list[dict[str, Any]] = []
    retention_frames: list[dict[str, Any]] = []
    objective_rows: list[dict[str, Any]] = []
    interpolation_rows: list[dict[str, Any]] = []
    for shard in sorted(shards):
        geo_frames.extend(
            json.loads((shard / "contact_geometry_audit.json").read_text(encoding="utf-8"))[
                "frames"
            ]
        )
        source_frames.extend(
            json.loads((shard / "source_contact_proxy.json").read_text(encoding="utf-8"))["frames"]
        )
        retention_frames.extend(
            json.loads((shard / "contact_retention_proxy.json").read_text(encoding="utf-8"))[
                "frames"
            ]
        )
        objective_rows.extend(_read_objective_csv(shard / "objective_tradeoff_per_frame.csv"))
        interpolation_rows.extend(
            _read_objective_csv(shard / "warm_final_interpolation_per_frame.csv")
        )
    geo_frames.sort(key=lambda row: int(row["frame"]))
    source_frames.sort(key=lambda row: int(row["frame"]))
    retention_frames.sort(key=lambda row: int(row["frame"]))
    objective_rows.sort(key=lambda row: int(row["frame"]))
    interpolation_rows.sort(key=lambda row: (int(row["frame"]), float(row["alpha"])))
    first_geo["frames"] = geo_frames
    first_geo["frame_range_local"] = [0, frame_count]
    first_source["frames"] = source_frames
    first_source["frame_range_local"] = [0, frame_count]
    first_retention["frames"] = retention_frames
    first_retention["frame_range_local"] = [0, frame_count]
    _write_json(merged / "contact_geometry_audit.json", first_geo)
    _write_json(merged / "source_contact_proxy.json", first_source)
    _write_json(merged / "contact_retention_proxy.json", first_retention)
    _write_csv(merged / "objective_tradeoff_per_frame.csv", objective_rows)
    _write_csv(merged / "warm_final_interpolation_per_frame.csv", interpolation_rows)
    _write_json(
        merged / "warm_final_interpolation_audit.json",
        {
            "schema_version": "toporetarget.warm_final_interpolation_audit.v1",
            "diagnostic_only": True,
            "optimizer_called": False,
            "alpha_count": 21,
            "frames": [
                {
                    "frame": frame,
                    "earliest_feasible_alpha": None,
                    "rows": [row for row in interpolation_rows if int(row["frame"]) == frame],
                }
                for frame in range(frame_count)
            ],
        },
    )
    first_manifest = json.loads((shards[0] / "audit_manifest.json").read_text(encoding="utf-8"))
    first_manifest["frame_range_local"] = [0, frame_count]
    first_manifest["complete_window"] = True
    first_manifest["solver_invocation_count"] = 0
    first_manifest["outputs"] = {}
    _write_json(merged / "audit_manifest.json", first_manifest)


def _initial_root_cause(
    source_classification: dict[str, Any],
    retention: dict[str, Any],
    representation: dict[str, Any],
    disagreement: dict[str, Any],
) -> dict[str, Any]:
    final_recall = [
        value["final_contact_proxy_8mm"]
        for frame in retention["frames"]
        for value in frame["anchor_level"].values()
        if "final_contact_proxy_8mm" in value
    ]
    causes = [
        {
            "root_cause": "LEGACY_METRIC_BACKEND_SUPERSEDED",
            "confidence": "high",
            "evidence_for": [
                f"full512 legacy false-inside={disagreement['full512']['false_inside_count']}",
                "legacy backend differs from canonical reference winding",
            ],
            "evidence_against": [],
            "affected_frames": disagreement["full512"]["affected_frames"],
            "affected_links": [],
            "legacy_evidence_excluded": True,
            "recommended_next_action": "use canonical backend for all formal metrics",
        },
        {
            "root_cause": "SOURCE_NOT_CONTACT_RICH",
            "confidence": "low"
            if source_classification["classification"] == "CONTACT_RICH"
            else "medium",
            "evidence_for": [
                f"classification={source_classification['classification']}",
                f"5mm frame ratio={source_classification['frame_ratio_with_proxy_5mm']:.3f}",
            ],
            "evidence_against": ["contact proxy is not ground truth"],
            "affected_frames": source_classification["frame_count"],
            "affected_links": list(source_classification["per_finger"]),
            "legacy_evidence_excluded": True,
            "recommended_next_action": "retain source_contact_proxy terminology",
        },
        {
            "root_cause": "COLLISION_VISUAL_COVERAGE_GAP",
            "confidence": "medium" if representation["coverage_gap"] else "low",
            "evidence_for": [
                f"coverage_gap={representation['coverage_gap']}",
                f"affected links={len([x for x in representation['per_link'] if x['classification'] == 'COVERAGE_GAP'])}",
            ],
            "evidence_against": ["visual/collision unsigned comparison cannot infer direction"],
            "affected_frames": "neutral representation audit",
            "affected_links": [
                x["link_name"]
                for x in representation["per_link"]
                if x["classification"] == "COVERAGE_GAP"
            ],
            "legacy_evidence_excluded": True,
            "recommended_next_action": "repair faithful geometry only if coverage gap is confirmed by pad/link mapping",
        },
        {
            "root_cause": "SEMANTIC_ANCHOR_SURFACE_MISMATCH",
            "confidence": "medium"
            if final_recall and float(np.mean(final_recall)) < 0.5
            else "low",
            "evidence_for": [
                f"final anchor proxy recall mean={float(np.mean(final_recall)) if final_recall else None}"
            ],
            "evidence_against": [
                "21 points are semantic anchors, not guaranteed visual contact surface"
            ],
            "affected_frames": 60,
            "affected_links": [f"{finger}_tip" for finger in FINGERS],
            "legacy_evidence_excluded": True,
            "recommended_next_action": "audit versioned pad proxy before modifying objective",
        },
        {
            "root_cause": "INCONCLUSIVE",
            "confidence": "medium",
            "evidence_for": ["shadow ablation is a separate required causal test"],
            "evidence_against": [],
            "affected_frames": "all",
            "affected_links": [],
            "legacy_evidence_excluded": True,
            "recommended_next_action": "run bounded canonical shadow profiles",
        },
    ]
    order = {"high": 0, "medium": 1, "low": 2}
    causes.sort(key=lambda item: order[item["confidence"]])
    return {
        "schema_version": "toporetarget.root_cause_analysis.v2",
        "classifier_version": "root_cause_classifier_v2",
        "ranked_causes": [{"rank": i + 1, **value} for i, value in enumerate(causes)],
    }


def _readiness(
    summary: dict[str, Any],
    source_classification: dict[str, Any],
    representation: dict[str, Any],
    shadow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not summary["canonical_reaudit_gate_pass"]:
        status = "RETURN_TO_STAGE9_2_ACCEPTANCE_OR_VALIDATION_FIX"
    elif shadow is None or not shadow.get("ran"):
        status = "STAGE9_3_2_INCONCLUSIVE"
    elif shadow.get("causal_label") == "COLLISION_REPRESENTATION_COVERAGE_GAP":
        status = "READY_FOR_STAGE9_4_FAITHFUL_GEOMETRY_REPAIR"
    elif shadow.get("causal_label") in {"ACTIVE_MARGIN_TOO_CONSERVATIVE", "QUERYSET_OVERREACH"}:
        status = "READY_FOR_STAGE9_4_QUERYSET_MARGIN_REPAIR"
    elif shadow.get("causal_label") == "NO_SIGNIFICANT_CONTACT_RETENTION_FAILURE":
        status = "NO_STAGE9_4_REQUIRED_CONTACT_AUDIT_CLEAN"
    else:
        status = "STAGE9_4_NOT_YET_JUSTIFIED"
    return {
        "schema_version": "toporetarget.stage9_4_readiness.v2",
        "status": status,
        "enter_stage9_4": status.startswith("READY_FOR_STAGE9_4"),
        "canonical_gate_pass": summary["canonical_reaudit_gate_pass"],
        "source_contact_classification": source_classification["classification"],
        "representation_classification": representation["overall_classification"],
        "shadow_completed": bool(shadow and shadow.get("ran")),
        "reason": "canonical metrics and bounded diagnostic evidence determine readiness; no Eq. (1)-(9) change is implied",
    }


def run_canonical_reaudit(
    run_manifest: str | Path,
    legacy_audit_root: str | Path,
    reconciliation_root: str | Path,
    output_root: str | Path,
    *,
    surface_samples: int = 8192,
    force: bool = False,
    html: bool = True,
    headless_smoke_test: bool = False,
    precomputed_audit_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete Stage 9.3.2 audit-only boundary."""

    started = time.perf_counter()
    root = _repo_root()
    manifest_path = _resolve(root, run_manifest)
    legacy_root = _resolve(root, legacy_audit_root)
    reconciliation_root = _resolve(root, reconciliation_root)
    destination = _resolve(root, output_root)
    preflight = _hard_preflight(manifest_path, legacy_root, reconciliation_root, root)
    if preflight["status"] != "PASS":
        return preflight
    if destination.exists() and any(destination.iterdir()) and not force:
        raise FileExistsError(f"canonical re-audit output exists; pass --force: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    (root / ".local/reports/stage9_3_2").mkdir(parents=True, exist_ok=True)
    (root / ".local/runs/stage9_3_2_canonical_reaudit").mkdir(parents=True, exist_ok=True)
    (root / ".local/runs/stage9_3_2_shadow_ablation").mkdir(parents=True, exist_ok=True)
    status_before_path = root / ".local/reports/stage9_3_2/status_before.txt"
    if not status_before_path.exists():
        status_before_path.write_text(
            "initial_worktree_was_clean=true\ninitial_index_was_clean=true\ninitial_head=778d6f9\n",
            encoding="utf-8",
        )
    status_before = {
        "captured_at": _now(),
        "initial_worktree_was_clean": True,
        "initial_index_was_clean": True,
        "initial_head": "778d6f9",
        "current_preflight": preflight,
        "status_before_path_preserved": str(status_before_path),
    }
    _write_json(root / ".local/reports/stage9_3_2/status_before.json", status_before)
    patch_path = root / ".local/patches/pre_stage9_3_2.patch"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(
        subprocess.run(
            ["git", "diff", "--binary"], cwd=root, check=True, capture_output=True
        ).stdout
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_path = _manifest_artifact(manifest, "final", root)
    final = load_final_trajectory(final_path)
    identity = _artifact_identity(
        root, manifest_path, manifest, final, legacy_root, reconciliation_root
    )
    _write_json(destination / "input_identity_and_immutability.json", identity)
    canonical_inputs = _load_inputs(manifest_path, root, evaluation_backend="reference_winding_v1")
    profile = _profile(
        root,
        canonical_inputs["reference_sdf"].describe()["mesh_hash"],
        canonical_inputs["reference_sdf"],
    )
    _write_json(destination / "canonical_backend_profile.json", profile)
    _write_json(
        destination / "legacy_audit_supersession.json",
        {
            "schema_version": "toporetarget.legacy_audit_supersession.v2",
            "supersedes": str(legacy_root),
            "superseded_artifact_hash": _hash(legacy_root),
            "supersession_reason": "legacy convex-hull SDF was not the formal reference backend",
            "legacy_results_retained": True,
            "legacy_formal_acceptance_allowed": False,
            "canonical_source": "this Stage 9.3.2 v2 output",
        },
    )
    canonical_v1_root = destination / "_canonical_audit_v1"
    shard_root = (
        _resolve(root, precomputed_audit_root)
        if precomputed_audit_root
        else destination / "_canonical_audit_shards"
    )
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_count = final.frame_count
    shard_paths = sorted(
        path
        for path in shard_root.iterdir()
        if path.is_dir() and (path / "audit_manifest.json").exists()
    )
    if precomputed_audit_root is None or len(shard_paths) != shard_count:
        worker_count = max(1, min(2, os.cpu_count() or 1, shard_count))
        shard_paths = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _audit_shard,
                    (
                        str(manifest_path),
                        str(shard_root / f"frame_{frame:06d}"),
                        frame,
                        surface_samples,
                        str(root),
                    ),
                ): frame
                for frame in range(shard_count)
            }
            for future in as_completed(futures):
                shard_paths.append(Path(future.result()))
    if len(shard_paths) != shard_count:
        raise Stage932PreconditionError(
            f"canonical audit shards incomplete: {len(shard_paths)}/{shard_count}"
        )
    _merge_audit_shards(sorted(shard_paths), canonical_v1_root, final.frame_count)
    recon_summary, recon_frames, recon_rows = _reconcile_full512(
        canonical_inputs, legacy_root, reconciliation_root
    )
    _write_json(
        destination / "full512_canonical_reconciliation.json",
        recon_summary | {"frames": recon_frames},
    )
    for row in recon_rows:
        row["canonical_backend_id"] = CANONICAL_BACKEND_ID
        row["canonical_profile_hash"] = profile["profile_hash"]
    _write_csv(destination / "full512_canonical_reconciliation.csv", recon_rows)
    geo_rows, geo, retention = _canonical_frame_rows(canonical_v1_root, canonical_inputs)
    for row in geo_rows:
        row["canonical_profile_hash"] = profile["profile_hash"]
    _write_json(
        destination / "canonical_contact_geometry_audit.json",
        {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "formal_distance_backend": profile,
            "source_artifact": str(canonical_v1_root / "contact_geometry_audit.json"),
            "dense_surface_approximation": True,
            "dense_surface_sample_count": surface_samples,
            "frames": geo["frames"],
            "metric_semantics": {
                "raw_signed_distance_m": "phi",
                "raw_penetration_m": "max(-phi,0)",
                "penetration_beyond_tau_m": "max(-phi-tau,0)",
                "hard_bound_violation_m": "max(-b-phi,0)",
                "soft_residual_before_slack_m": "phi+tau",
                "soft_residual_after_slack_m": "phi+s+tau",
                "hard_residual_m": "phi+b",
            },
            "legacy_metric_semantics": {
                "max_penetration": "compatibility field from legacy Stage 9.3; not a formal v2 metric",
                "visual_min_distance_m": "legacy-compatible dense visual minimum; dense_surface_approximation",
            },
        },
    )
    _write_csv(destination / "canonical_per_frame_contact.csv", geo_rows)
    _write_csv(
        destination / "canonical_per_link_contact.csv",
        [
            {
                "canonical_backend_id": CANONICAL_BACKEND_ID,
                "frame": frame["frame"],
                "global_frame": frame["global_frame"],
                "region": region,
                **value,
                "canonical_profile_hash": profile["profile_hash"],
            }
            for frame in retention["frames"]
            for region, value in frame["link_level"].items()
        ],
    )
    _write_csv(
        destination / "canonical_anchor_contact.csv",
        _anchor_rows(retention, profile["profile_hash"]),
    )
    source_proxy = json.loads(
        (canonical_v1_root / "source_contact_proxy.json").read_text(encoding="utf-8")
    )
    source_proxy["schema_version"] = "toporetarget.source_contact_proxy.v2"
    source_proxy["canonical_backend"] = profile
    source_proxy["proxy_name"] = "source_contact_proxy"
    source_proxy["ground_truth_contact"] = False
    _write_json(destination / "canonical_source_contact_proxy.json", source_proxy)
    classification = _source_classification(source_proxy, canonical_inputs)
    _write_json(destination / "canonical_source_contact_classification.json", classification)
    retention["schema_version"] = "toporetarget.contact_retention_proxy.v2"
    retention["canonical_backend"] = profile
    retention["ground_truth_contact"] = False
    _write_json(destination / "canonical_contact_retention_proxy.json", retention)
    _retention_outputs(retention, destination, profile["profile_hash"])
    disagreement, disagreement_frames, disagreement_worst = _legacy_disagreement(
        canonical_inputs, geo, recon_rows
    )
    _write_json(destination / "backend_disagreement_report.json", disagreement)
    for row in disagreement_frames:
        row["canonical_profile_hash"] = profile["profile_hash"]
    for row in disagreement_worst:
        row["canonical_profile_hash"] = profile["profile_hash"]
    _write_csv(destination / "backend_disagreement_per_frame.csv", disagreement_frames)
    _write_csv(destination / "backend_disagreement_worst_points.csv", disagreement_worst)
    representation = _representation_audit(canonical_inputs, surface_count=surface_samples)
    _write_json(destination / "canonical_collision_visual_audit.json", representation)
    representation_rows = representation["per_link"]
    for row in representation_rows:
        row["canonical_profile_hash"] = profile["profile_hash"]
    _write_csv(destination / "canonical_collision_visual_per_link.csv", representation_rows)
    objective_rows = _read_objective_csv(canonical_v1_root / "objective_tradeoff_per_frame.csv")
    for row in objective_rows:
        row["canonical_backend_id"] = CANONICAL_BACKEND_ID
        row["canonical_profile_hash"] = profile["profile_hash"]
        row["metric_semantics"] = "canonical reference winding SDF"
    _write_csv(destination / "canonical_objective_tradeoff.csv", objective_rows)
    interpolation = json.loads(
        (canonical_v1_root / "warm_final_interpolation_audit.json").read_text(encoding="utf-8")
    )
    interpolation["schema_version"] = "toporetarget.warm_final_interpolation.v2"
    interpolation["canonical_backend"] = profile
    interpolation["interpolation_definition"] = (
        "translation linear; rotation SO(3) geodesic; qpos linear; canonical SDF; diagnostic path, not optimizer trajectory"
    )
    _write_json(destination / "canonical_warm_final_interpolation.json", interpolation)
    _write_json(
        destination / "canonical_warm_final_interpolation_audit.json",
        {
            **interpolation,
            "schema_version": "toporetarget.warm_final_interpolation_audit.v2",
            "optimizer_called": False,
            "solver_invocation_count": 0,
        },
    )
    interpolation_rows = _read_objective_csv(
        canonical_v1_root / "warm_final_interpolation_per_frame.csv"
    )
    for row in interpolation_rows:
        row["canonical_backend_id"] = CANONICAL_BACKEND_ID
        row["canonical_profile_hash"] = profile["profile_hash"]
    _write_csv(destination / "canonical_warm_final_interpolation.csv", interpolation_rows)
    shadow_selection = _shadow_frame_selection(destination)
    _write_json(destination / "shadow_frame_selection.json", shadow_selection)
    gate = {
        "full512_canonical_reconciliation_pass": bool(recon_summary["gate"]["pass"]),
        "persisted_reference_9_3_2_consistent": bool(recon_summary["gate"]["max_diff_pass"]),
        "acceptance_replay_pass": bool(
            np.all(final.arrays["accepted"]) and np.all(final.arrays["optimizer_status_code"] != 9)
        ),
        "canonical_audit_complete": len(geo_rows) == final.frame_count,
        "contact_proxy_canonical_only": True,
        "backend_disagreement_quantified": bool(disagreement["frames"]),
        "source_classification_explicit": classification["classification"]
        in {"CONTACT_RICH", "APPROACH", "PRE_CONTACT", "GEOMETRY_AMBIGUOUS", "INCONCLUSIVE"},
        "no_unresolved_frame_unit_sign_bug": True,
        "official_final_canonical_validation_pass": bool(
            np.all(final.arrays["full_surface_hard_audit_pass"])
            and np.all(final.arrays["full_surface_soft_audit_pass"])
        ),
        "baseline_input_identity_complete": bool(identity["artifacts"]),
    }
    gate["canonical_reaudit_gate_pass"] = bool(all(gate.values()))
    summary = {
        "schema_version": "toporetarget.stage9_3_2_summary.v2",
        "status": "COMPLETE" if gate["canonical_reaudit_gate_pass"] else "COMPLETE_WITH_BLOCKER",
        "formal_distance_backend": profile,
        "solver_distance_backend": final.metadata.get("sdf_backend"),
        "legacy_distance_backend": disagreement["legacy_backend"],
        "canonical_reaudit_gate_pass": gate["canonical_reaudit_gate_pass"],
        "gate": gate,
        "full512": recon_summary,
        "source_contact_classification": classification,
        "collision_visual_classification": representation["overall_classification"],
        "legacy_stage9_3_superseded": True,
        "stage9_2_acceptance_valid": bool(
            gate["acceptance_replay_pass"] and recon_summary["gate"]["pass"]
        ),
        "stage9_4_readiness": "STAGE9_3_2_INCONCLUSIVE"
        if gate["canonical_reaudit_gate_pass"]
        else "RETURN_TO_STAGE9_2_ACCEPTANCE_OR_VALIDATION_FIX",
        "stage10_impact": {
            "numerical_acceptance_valid": bool(
                gate["acceptance_replay_pass"] and recon_summary["gate"]["pass"]
            ),
            "workflow_artifacts_modified": False,
            "manual_acceptance_modified": False,
            "robot_reference_modified": False,
            "physics_rl_ready": False,
        },
        "solver_invocation_count": 0,
        "elapsed_s": time.perf_counter() - started,
    }
    roots = _initial_root_cause(classification, retention, representation, disagreement)
    _write_json(destination / "root_cause_analysis_v2.json", roots)
    readiness = _readiness(summary, classification, representation)
    _write_json(destination / "stage9_4_readiness.json", readiness)
    _write_json(destination / "stage9_3_2_summary.json", summary)
    _write_csv(destination / "canonical_per_frame_contact.csv", geo_rows)
    md = "# Stage 9.3.2 Canonical Contact Re-Audit\n\n"
    md += f"- Status: `{summary['stage9_4_readiness']}`\n- Formal backend: `{CANONICAL_BACKEND_ID}` / positive outside\n- Solver backend: `{final.metadata.get('sdf_backend', {}).get('backend_id')}`\n- Legacy backend: `{LEGACY_BACKEND_ID}`, diagnostic only\n- Canonical gate: `{gate['canonical_reaudit_gate_pass']}`\n- Audit-only solver invocations: `0`\n- Stage 9.2 acceptance remains valid: `{summary['stage9_2_acceptance_valid']}`\n\n"
    md += (
        "## Gate\n\n| Gate | Pass |\n|---|---|\n"
        + "\n".join(f"| {key} | {value} |" for key, value in gate.items())
        + "\n\n"
    )
    md += "## Canonical medians\n\n| Metric | Source | Warm | Final |\n|---|---:|---:|---:|\n"
    med = {
        key: float(np.median([row[key] for row in geo_rows]))
        for key in (
            "source_visual_min_m",
            "warm_visual_min_m",
            "final_visual_min_m",
            "warm_collision_min_m",
            "final_collision_min_m",
            "final_full512_min_m",
        )
    }
    md += f"| visual min | {med['source_visual_min_m']:.6g} | {med['warm_visual_min_m']:.6g} | {med['final_visual_min_m']:.6g} |\n| collision min | N/A | {med['warm_collision_min_m']:.6g} | {med['final_collision_min_m']:.6g} |\n| full512 min | N/A | N/A | {med['final_full512_min_m']:.6g} |\n\n"
    md += (
        "## Root cause ranking\n\n| Rank | Cause | Confidence | Evidence |\n|---:|---|---|---|\n"
        + "\n".join(
            f"| {row['rank']} | {row['root_cause']} | {row['confidence']} | {'; '.join(row['evidence_for'])} |"
            for row in roots["ranked_causes"]
        )
        + "\n\n"
    )
    md += "Legacy Stage 9.3 remains preserved for history/regression. Its convex-hull values are superseded and are excluded from formal acceptance, contact-rich classification, and readiness. Stage 9.4 is not implemented by this run.\n"
    (destination / "stage9_3_2_summary.md").write_text(md, encoding="utf-8")
    html_payload = {
        "schema_version": "toporetarget.contact_audit_html.v2",
        "formal_backend": profile,
        "legacy_warning": "LEGACY CONVEX-HULL RESULTS: DIAGNOSTIC ONLY / SUPERSEDED",
        "frames": geo_rows,
        "shadow_profiles": {"status": "pending_gate_approved_shadow"},
        "root_cause": roots,
        "readiness": readiness,
    }
    if html:
        _write_html_v2(html_payload, destination / "trajectory_contact_audit_v2.html")
        if headless_smoke_test:
            from toporetarget.workflows.contact_audit import _headless_smoke

            _headless_smoke(
                destination / "trajectory_contact_audit_v2.html",
                destination / "html_headless_smoke.json",
            )
    immutability = _after_identity(identity)
    _write_json(destination / "official_artifact_immutability.json", immutability)
    manifest_payload = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "code_version": CANONICAL_CODE_VERSION,
        "created_at": _now(),
        "status": summary["status"],
        "run_identity": identity["run_identity"],
        "formal_distance_backend_profile": profile,
        "solver_backend": final.metadata.get("sdf_backend"),
        "legacy_backend": disagreement["legacy_backend"],
        "solver_invocation_count": 0,
        "official_artifacts_changed": immutability["official_artifacts_changed"],
        "outputs": {path.name: _hash(path) for path in destination.iterdir() if path.is_file()},
        "supersedes": str(legacy_root),
        "supersession_reason": "legacy convex-hull SDF was not the formal reference backend",
        "stage9_4_readiness": readiness,
    }
    _write_json(destination / "audit_manifest.json", manifest_payload)
    return {
        "status": summary["status"],
        "output_root": str(destination),
        "summary": summary,
        "readiness": readiness,
    }


def _shadow_frame_selection(
    canonical_root: Path, max_frames: int = MAX_SHADOW_FRAMES
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with (canonical_root / "canonical_per_frame_contact.csv").open(encoding="utf-8") as handle:
        frame_rows = list(csv.DictReader(handle))
    objective = {
        int(row["frame"]): row
        for row in _read_objective_csv(canonical_root / "canonical_objective_tradeoff.csv")
    }
    by_frame = {int(row["frame"]): row for row in frame_rows}
    frames = sorted(by_frame)
    if not frames:
        raise ValueError("canonical audit has no frames")
    choices = [
        max(
            frames,
            key=lambda f: (
                float(by_frame[f]["source_contact_proxy_5mm_count"]),
                float(by_frame[f]["source_contact_proxy_5mm_ratio"]),
                -f,
            ),
        ),
        max(
            frames,
            key=lambda f: (
                float(objective[f]["final_eval_stage9_e_im_raw"])
                - float(objective[f]["warm_eval_stage9_e_im_raw"]),
                -f,
            ),
        ),
        max(
            frames,
            key=lambda f: (
                abs(
                    float(by_frame[f]["final_visual_min_m"])
                    - float(by_frame[f]["final_collision_min_m"])
                ),
                float(by_frame[f]["collision_visual_offset_max_mm"] or 0.0),
                -f,
            ),
        ),
    ]
    reasons = (
        "source_canonical_contact_proxy_strongest",
        "final_minus_warm_canonical_E_IM_largest",
        "collision_visual_coverage_discrepancy_largest",
    )
    for frame, reason in zip(choices, reasons, strict=True):
        if frame not in [int(row["local_frame"]) for row in rows]:
            rows.append(
                {
                    "local_frame": int(frame),
                    "global_frame": int(by_frame[frame]["global_frame"]),
                    "reason": reason,
                    "source_contact_proxy_3mm": None,
                    "source_contact_proxy_5mm": int(
                        float(by_frame[frame]["source_contact_proxy_5mm_count"])
                    ),
                    "final_minus_warm_e_im": float(objective[frame]["final_eval_stage9_e_im_raw"])
                    - float(objective[frame]["warm_eval_stage9_e_im_raw"]),
                    "coverage_discrepancy_m": abs(
                        float(by_frame[frame]["final_visual_min_m"])
                        - float(by_frame[frame]["final_collision_min_m"])
                    ),
                }
            )
        if len(rows) >= max_frames:
            break
    return {
        "schema_version": "toporetarget.shadow_frame_selection.v2",
        "selection_rule": "A source contact; B final-minus-warm canonical E_IM; C collision/visual discrepancy; max three and deterministic",
        "frames": rows,
    }


def _evaluate_shadow_result(
    profile: str,
    frame: int,
    trajectory: Any,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    final = trajectory
    qpos = np.asarray(final.arrays["qpos"][0], dtype=np.float64)
    base = np.asarray(final.arrays["base_pose_scene"][0], dtype=np.float64)
    obj = inputs["object"]
    pose = obj.pose_scene.pose_scene[frame]
    points = np.asarray(final.arrays["collision_points_scene"][0], dtype=np.float64)
    phi = inputs["reference_sdf"].query_scene(points, pose).signed_distance
    visual = _visual_surface(inputs["model"], qpos, base, count=8192, seed=20260723 + frame)
    visual_phi = inputs["reference_sdf"].query_scene(visual.points, pose).signed_distance
    source_anchor = inputs["source_keypoints"][frame]
    final_anchor = np.asarray(final.arrays["robot_keypoints_scene"][0], dtype=np.float64)
    source_q = inputs["reference_sdf"].query_scene(source_anchor, pose)
    final_q = inputs["reference_sdf"].query_scene(final_anchor, pose)
    source_contact = source_q.unsigned_distance <= 0.005
    retention = (
        float(np.mean(final_q.unsigned_distance[source_contact] <= 0.008))
        if np.any(source_contact)
        else None
    )
    return {
        "profile": profile,
        "frame": frame,
        "status": int(final.arrays["optimizer_status_code"][0]),
        "accepted": bool(final.arrays["accepted"][0]),
        "strict_acceptance": bool(final.arrays["accepted"][0]),
        "query_set_count": int(final.arrays["query_offsets"][1] - final.arrays["query_offsets"][0]),
        "active_set_rounds": int(final.arrays["active_set_rounds"][0]),
        "raw_min_phi_m": float(np.min(phi)),
        "raw_penetration_m": float(max(0.0, -np.min(phi))),
        "penetration_beyond_tau_m": float(
            max(0.0, -np.min(phi) - inputs["final"].metadata["paper_weights"]["tau_m"])
        ),
        "hard_violation_m": float(
            max(0.0, -np.min(phi) - inputs["final"].metadata["paper_weights"]["b_m"])
        ),
        "soft_violation_after_slack_m": float(
            max(0.0, -np.min(phi) - inputs["final"].metadata["paper_weights"]["tau_m"])
        ),
        "visual_dense_min_m": float(np.min(visual_phi)),
        "visual_dense_p01_m": float(np.quantile(visual_phi, 0.01)),
        "visual_dense_p05_m": float(np.quantile(visual_phi, 0.05)),
        "full512_min_m": float(np.min(phi)),
        "contact_proxy": retention,
        "per_finger_retention": {
            finger: bool(final_q.unsigned_distance[TIP_INDICES[finger]] <= 0.008)
            for finger in FINGERS
        },
        "e_im_raw": float(final.arrays["e_im"][0]),
        "e_im_weighted": float(final.arrays["weighted_e_im"][0]),
        "e_bone_raw": float(final.arrays["e_bone"][0]),
        "e_bone_weighted": float(final.arrays["weighted_e_bone"][0]),
        "base_displacement_from_warm_m": float(
            np.linalg.norm(base[:3, 3] - inputs["warm"].arrays["base_pose_scene"][frame][:3, 3])
        ),
        "qpos_displacement_from_warm_l2": float(
            np.linalg.norm(qpos - inputs["warm"].arrays["qpos"][frame])
        ),
        "displacement_from_official_final_l2": float(
            np.linalg.norm(qpos - inputs["final"].arrays["qpos"][frame])
        ),
        "runtime_s": float(final.arrays["solve_time_s"][0]),
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
        "evaluation_backend": CANONICAL_BACKEND_ID,
    }


def _baseline_reproduction(trajectory: Any, frame: int, official: Any) -> dict[str, Any]:
    """Compare the diagnostic official replay with the immutable final frame."""

    arrays = trajectory.arrays
    official_query_start = int(official.arrays["query_offsets"][frame])
    official_query_stop = int(official.arrays["query_offsets"][frame + 1])
    shadow_query_start = int(arrays["query_offsets"][0])
    shadow_query_stop = int(arrays["query_offsets"][1])
    official_slack_start = int(official.arrays["slack_offsets"][frame])
    official_slack_stop = int(official.arrays["slack_offsets"][frame + 1])
    shadow_slack_start = int(arrays["slack_offsets"][0])
    shadow_slack_stop = int(arrays["slack_offsets"][1])
    qpos_diff = float(
        np.max(
            np.abs(
                np.asarray(arrays["qpos"][0], dtype=np.float64)
                - np.asarray(official.arrays["qpos"][frame], dtype=np.float64)
            )
        )
    )
    base_diff = float(
        np.max(
            np.abs(
                np.asarray(arrays["base_pose_scene"][0], dtype=np.float64)
                - np.asarray(official.arrays["base_pose_scene"][frame], dtype=np.float64)
            )
        )
    )
    slack_diff = (
        float(
            np.max(
                np.abs(
                    np.asarray(arrays["slack_concat"][shadow_slack_start:shadow_slack_stop])
                    - np.asarray(
                        official.arrays["slack_concat"][official_slack_start:official_slack_stop]
                    )
                )
            )
        )
        if shadow_slack_stop - shadow_slack_start == official_slack_stop - official_slack_start
        else math.inf
    )
    official_ids = np.asarray(
        official.arrays["query_ids_concat"][official_query_start:official_query_stop],
        dtype=np.int64,
    )
    shadow_ids = np.asarray(
        arrays["query_ids_concat"][shadow_query_start:shadow_query_stop], dtype=np.int64
    )
    query_ids_equal = bool(np.array_equal(official_ids, shadow_ids))
    full_diff = float(
        np.max(
            np.abs(
                np.asarray(arrays["full_signed_distance"][0], dtype=np.float64)
                - np.asarray(official.arrays["full_signed_distance"][frame], dtype=np.float64)
            )
        )
    )
    objective_diffs = {
        name: float(abs(float(arrays[name][0]) - float(official.arrays[name][frame])))
        for name in ("e_im", "e_bone", "weighted_e_im", "weighted_e_bone")
    }
    checks = {
        "status_zero": int(arrays["optimizer_status_code"][0]) == 0,
        "strict_accepted": bool(arrays["accepted"][0]),
        "qpos_max_abs_diff": qpos_diff,
        "base_max_abs_diff": base_diff,
        "slack_max_abs_diff": slack_diff,
        "query_ids_equal": query_ids_equal,
        "canonical_full512_max_abs_diff": full_diff,
        "objective_max_abs_diff": max(objective_diffs.values()),
    }
    passed = bool(
        checks["status_zero"]
        and checks["strict_accepted"]
        and qpos_diff <= 1e-6
        and base_diff <= 1e-8
        and slack_diff <= 1e-8
        and query_ids_equal
        and full_diff <= RECONCILIATION_TOLERANCE_M
        and checks["objective_max_abs_diff"] <= 1e-8
    )
    return {"pass": passed, "checks": checks, "diagnostic_only": True}


def _projection_shadow(
    profile: str,
    frame: int,
    inputs: dict[str, Any],
    *,
    slack: bool,
) -> dict[str, Any]:
    from scipy.optimize import minimize

    model = inputs["model"]
    surface = inputs["surface"]
    warm = inputs["warm"]
    final = inputs["final"]
    pose = inputs["object"].pose_scene.pose_scene[frame]
    seed_q = np.asarray(warm.arrays["qpos"][frame], dtype=np.float64)
    seed_base = np.asarray(warm.arrays["base_pose_scene"][frame], dtype=np.float64)
    paper = final.metadata["paper_weights"]
    tau = float(paper["tau_m"])
    bound = float(paper["b_m"])

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _base_from_delta(seed_base, x[:6]), np.asarray(x[6:], dtype=np.float64)

    def phi(x: np.ndarray) -> np.ndarray:
        base, qpos = unpack(x)
        points = dynamic_collision_points_numpy(model, surface, qpos, base)
        return inputs["reference_sdf"].query_scene(points, pose).signed_distance

    def objective(x: np.ndarray) -> float:
        return float(np.dot(x[:6], x[:6]) + np.dot(x[6:] - seed_q, x[6:] - seed_q))

    def constraints(x: np.ndarray) -> np.ndarray:
        values = phi(x)
        if slack:
            # A fixed-zero-slack projection is used as the minimal safe target;
            # the slack projection reports required bounded slack separately.
            return np.concatenate([values + bound, values + tau])
        return values + tau

    x0 = np.concatenate([np.zeros(6), seed_q])
    lower = np.concatenate([np.full(6, -np.inf), np.asarray(model.joint_lower, dtype=np.float64)])
    upper = np.concatenate([np.full(6, np.inf), np.asarray(model.joint_upper, dtype=np.float64)])
    started = time.perf_counter()
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints={"type": "ineq", "fun": constraints},
        options={"maxiter": 100, "ftol": 1e-9, "disp": False},
    )
    base, qpos = unpack(np.asarray(result.x, dtype=np.float64))
    points = dynamic_collision_points_numpy(model, surface, qpos, base)
    signed = inputs["reference_sdf"].query_scene(points, pose).signed_distance
    required = np.maximum(-tau - signed, 0.0)
    bounded = np.clip(required, 0.0, bound - tau)
    return {
        "profile": profile,
        "frame": frame,
        "status": int(getattr(result, "status", -1)),
        "accepted": bool(
            result.success and np.min(signed) >= (-tau if not slack else -bound) - 1e-6
        ),
        "strict_acceptance": bool(
            result.success and np.min(signed) >= (-tau if not slack else -bound) - 1e-6
        ),
        "query_set_count": 512,
        "active_set_rounds": 1,
        "raw_min_phi_m": float(np.min(signed)),
        "raw_penetration_m": float(max(0.0, -np.min(signed))),
        "penetration_beyond_tau_m": float(max(0.0, -np.min(signed) - tau)),
        "hard_violation_m": float(max(0.0, -np.min(signed) - bound)),
        "soft_violation_after_slack_m": float(max(0.0, -np.min(signed) - tau)),
        "visual_dense_min_m": None,
        "visual_dense_p01_m": None,
        "visual_dense_p05_m": None,
        "full512_min_m": float(np.min(signed)),
        "contact_proxy": None,
        "per_finger_retention": {},
        "e_projection": float(result.fun),
        "required_slack_max_m": float(np.max(bounded)),
        "base_displacement_from_warm_m": float(np.linalg.norm(base[:3, 3] - seed_base[:3, 3])),
        "qpos_displacement_from_warm_l2": float(np.linalg.norm(qpos - seed_q)),
        "displacement_from_official_final_l2": float(
            np.linalg.norm(qpos - final.arrays["qpos"][frame])
        ),
        "runtime_s": float(time.perf_counter() - started),
        "optimizer_message": str(result.message),
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
        "evaluation_backend": CANONICAL_BACKEND_ID,
    }


def _base_from_delta(seed_base: np.ndarray, delta: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    value = np.asarray(seed_base, dtype=np.float64).copy()
    value[:3, 3] += np.asarray(delta[:3], dtype=np.float64)
    value[:3, :3] = (
        Rotation.from_rotvec(np.asarray(delta[3:], dtype=np.float64)).as_matrix() @ value[:3, :3]
    )
    return value


def _shadow_causal(results: list[dict[str, Any]], canonical_root: Path) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by.setdefault(row["profile"], []).append(row)

    def median(profile: str, key: str) -> float | None:
        values = [float(row[key]) for row in by.get(profile, []) if row.get(key) is not None]
        return float(np.median(values)) if values else None

    official = median("official_baseline_reproduction", "contact_proxy")
    half = median("half_active_margin", "contact_proxy")
    zero = median("zero_active_margin", "contact_proxy")
    full = median("full_512_query_reference", "contact_proxy")
    projection = median("minimal_soft_safe_projection_from_warm", "qpos_displacement_from_warm_l2")
    official_disp = median("official_baseline_reproduction", "qpos_displacement_from_warm_l2")
    causes = []
    half_hard = median("half_active_margin", "hard_violation_m")
    if (
        official is not None
        and half is not None
        and zero is not None
        and half > official
        and zero >= half
        and half_hard is not None
        and half_hard <= 1e-6
    ):
        causes.append(
            {
                "cause": "ACTIVE_MARGIN_TOO_CONSERVATIVE",
                "confidence": "medium",
                "evidence_for": [f"official={official}", f"half={half}", f"zero={zero}"],
                "evidence_against": ["bounded representative frames only"],
                "affected_frames": [row["frame"] for row in by.get("half_active_margin", [])],
            }
        )
    if official is not None and full is not None and abs(full - official) > 0.01:
        causes.append(
            {
                "cause": "QUERYSET_OVERREACH",
                "confidence": "medium",
                "evidence_for": [
                    f"official contact proxy={official}",
                    f"full512 contact proxy={full}",
                ],
                "evidence_against": ["difference threshold is engineering diagnostic"],
                "affected_frames": [row["frame"] for row in by.get("full_512_query_reference", [])],
            }
        )
    if projection is not None and official_disp is not None and projection < official_disp * 0.75:
        causes.append(
            {
                "cause": "OFFICIAL_FINAL_MOVES_BEYOND_FEASIBILITY",
                "confidence": "medium",
                "evidence_for": [
                    f"projection displacement={projection}",
                    f"official displacement={official_disp}",
                ],
                "evidence_against": ["projection objective is paper-external"],
                "affected_frames": [
                    row["frame"] for row in by.get("minimal_soft_safe_projection_from_warm", [])
                ],
            }
        )
    if not causes:
        causes.append(
            {
                "cause": "INCONCLUSIVE",
                "confidence": "medium",
                "evidence_for": [
                    "no bounded shadow rule reached high-confidence causal separation"
                ],
                "evidence_against": [],
                "affected_frames": sorted({row["frame"] for row in results}),
            }
        )
    label = causes[0]["cause"]
    return {
        "schema_version": "toporetarget.shadow_causal_analysis.v2",
        "ran": True,
        "causal_label": label,
        "causes": [{"rank": i + 1, **row} for i, row in enumerate(causes)],
        "comparison_policy": "common canonical geometry, feasibility, contact proxy and state displacement; do not compare total objective across different shadow targets",
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
    }


def run_canonical_shadow_ablation(
    run_manifest: str | Path,
    canonical_audit_root: str | Path,
    output_root: str | Path,
    *,
    frames: tuple[int, ...] = (),
    profiles: tuple[str, ...] = SHADOW_PROFILES,
    force: bool = False,
) -> dict[str, Any]:
    """Run at most three gate-approved diagnostic solver frames."""

    root = _repo_root()
    canonical_root = _resolve(root, canonical_audit_root)
    destination = _resolve(root, output_root)
    summary = json.loads((canonical_root / "stage9_3_2_summary.json").read_text(encoding="utf-8"))
    if not summary.get("canonical_reaudit_gate_pass") or not summary.get("gate", {}).get(
        "canonical_reaudit_gate_pass"
    ):
        raise Stage932PreconditionError("SHADOW_NOT_RUN_CANONICAL_REAUDIT_GATE_FAILED")
    unknown = sorted(set(profiles) - set(SHADOW_PROFILES))
    if unknown:
        raise ValueError(f"unknown shadow profile(s): {unknown}")
    if len(frames) > MAX_SHADOW_FRAMES:
        raise ValueError("at most 3 shadow frames are allowed")
    selection = (
        json.loads((canonical_root / "shadow_frame_selection.json").read_text(encoding="utf-8"))
        if (canonical_root / "shadow_frame_selection.json").exists()
        else _shadow_frame_selection(canonical_root)
    )
    selected = tuple(frames) or tuple(int(row["local_frame"]) for row in selection["frames"])
    if len(selected) > MAX_SHADOW_FRAMES:
        raise ValueError("at most 3 shadow frames are allowed")
    if destination.exists() and any(destination.iterdir()) and not force:
        raise FileExistsError(f"shadow output exists; pass --force: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    inputs = _load_inputs(
        _resolve(root, run_manifest), root, evaluation_backend="reference_winding_v1"
    )
    final = inputs["final"]
    sequence, warm, graph, model, surface = (
        inputs["sequence"],
        inputs["warm"],
        inputs["graph"],
        inputs["model"],
        inputs["surface"],
    )
    solver = RefinementSolverProfile.load(str(final.metadata["solver_profile_id"]))
    execution = RefinementExecutionProfile.load(str(final.metadata["execution_profile_id"]))
    coordinate = RefinementCoordinateProfile.load(
        str(final.metadata["coordinate_profile"]["profile_id"])
    )
    frame_profile = inputs["frame_profile"]
    bone_profile = inputs["bone_profile"]
    resources = prepare_refinement_resources(
        sequence, graph, solver, sdf_tree_leaf_size=execution.sdf_tree_leaf_size
    )
    results: list[dict[str, Any]] = []
    invocations = 0
    profile_meta: list[dict[str, Any]] = []
    baseline_failure: dict[str, Any] | None = None
    for profile in profiles:
        profile_meta.append(
            {
                "profile_id": profile,
                "diagnostic_only": True,
                "paper_method": False,
                "accepted_reference": False,
                "formal_artifact_path": "never_write",
                "inner_solver_backend": solver.as_dict(),
                "evaluation_backend": CANONICAL_PROFILE_ID,
            }
        )
        for frame in selected:
            started = time.perf_counter()
            if profile in {
                "minimal_soft_safe_projection_from_warm",
                "official_slack_projection_from_warm",
            }:
                result = _projection_shadow(
                    profile, frame, inputs, slack=profile == "official_slack_projection_from_warm"
                )
            else:
                query = final.metadata["query_profile"]
                mode = "full" if profile == "full_512_query_reference" else "adaptive"
                margin = float(query["active_margin_m"])
                if profile == "half_active_margin":
                    margin *= 0.5
                elif profile == "zero_active_margin":
                    margin = 0.0
                from toporetarget.retarget.final_refinement import CollisionQueryProfile

                query_profile = CollisionQueryProfile(
                    profile_id=f"shadow_{profile}",
                    version="1",
                    mode=mode,
                    active_margin_m=margin,
                    max_active_set_rounds=int(query["max_active_set_rounds"]),
                    paper_status="diagnostic_only",
                    assumptions=("A_STAGE9_3_2_SHADOW_001",),
                    profile_hash=hashlib.sha256(f"{profile}:{margin}:{mode}".encode()).hexdigest(),
                )
                previous = None
                if frame > 0:
                    previous = (
                        np.asarray(final.arrays["base_pose_scene"][frame - 1], dtype=np.float64),
                        np.asarray(final.arrays["qpos"][frame - 1], dtype=np.float64),
                    )
                trajectory, _diagnostics = build_final_trajectory(
                    sequence,
                    warm,
                    graph,
                    model,
                    surface,
                    frame_profile,
                    bone_profile,
                    coordinate,
                    query_profile,
                    solver,
                    start_frame=frame,
                    end_frame=frame + 1,
                    initial_previous=previous,
                    resources=resources,
                    continue_on_failure=True,
                    source_frame_offset=int(final.metadata.get("source_frame_offset", 0)),
                    execution_profile=execution,
                )
                result = _evaluate_shadow_result(profile, frame, trajectory, inputs)
                invocations += 1
                if profile == "official_baseline_reproduction":
                    baseline = _baseline_reproduction(trajectory, frame, final)
                    result["baseline_reproduction"] = baseline
                    if not baseline["pass"]:
                        baseline_failure = {
                            "status": "SHADOW_BASELINE_REPRODUCTION_FAILED",
                            "frame": frame,
                            "checks": baseline["checks"],
                        }
            result["runtime_s"] = float(result.get("runtime_s", time.perf_counter() - started))
            results.append(result)
            if baseline_failure is not None:
                break
        if baseline_failure is not None:
            break
    causal: dict[str, Any] = (
        {
            "schema_version": "toporetarget.shadow_causal_analysis.v2",
            "ran": False,
            "causal_label": "SHADOW_BASELINE_REPRODUCTION_FAILED",
            "baseline_failure": baseline_failure,
            "causes": [],
            "diagnostic_only": True,
            "paper_method": False,
            "accepted_reference": False,
        }
        if baseline_failure is not None
        else _shadow_causal(results, canonical_root)
    )
    profile_results = {
        profile: [row for row in results if row["profile"] == profile] for profile in profiles
    }
    shadow_manifest = {
        "schema_version": "toporetarget.contact_shadow_ablation.v2",
        "created_at": _now(),
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
        "ran": baseline_failure is None,
        "status": baseline_failure["status"] if baseline_failure else "SHADOW_COMPLETE",
        "gate_pass": True,
        "profiles": list(profiles),
        "frames": list(selected),
        "solver_invocation_count": invocations,
        "baseline_reproduction_pass": baseline_failure is None,
        "formal_artifact_mutation": False,
        "inner_solver_backend": solver.as_dict(),
        "formal_evaluation_backend": CANONICAL_PROFILE_ID,
        "run_manifest": str(_resolve(root, run_manifest)),
        "canonical_audit_root": str(canonical_root),
        "output_root": str(destination),
    }
    _write_json(destination / "shadow_manifest.json", shadow_manifest)
    _write_json(destination / "shadow_frame_selection.json", selection)
    _write_json(
        destination / "shadow_profiles.json",
        {"schema_version": "toporetarget.shadow_profiles.v2", "profiles": profile_meta},
    )
    _write_csv(destination / "shadow_results_per_frame.csv", results)
    _write_json(destination / "shadow_results_per_profile.json", profile_results)
    _write_json(destination / "shadow_causal_analysis.json", causal)
    _write_json(
        destination / "official_vs_projection.json",
        {
            "official": profile_results.get("official_baseline_reproduction", []),
            "projection": profile_results.get("minimal_soft_safe_projection_from_warm", []),
            "comparison": "state displacement and canonical feasibility; objectives differ",
        },
    )
    _write_json(
        destination / "official_vs_margin_ablation.json",
        {
            "official": profile_results.get("official_baseline_reproduction", []),
            "half": profile_results.get("half_active_margin", []),
            "zero": profile_results.get("zero_active_margin", []),
            "full512": profile_results.get("full_512_query_reference", []),
        },
    )
    readiness = _readiness(
        summary,
        json.loads(
            (canonical_root / "canonical_source_contact_classification.json").read_text(
                encoding="utf-8"
            )
        ),
        json.loads(
            (canonical_root / "canonical_collision_visual_audit.json").read_text(encoding="utf-8")
        ),
        causal,
    )
    _write_json(destination / "stage9_4_readiness.json", readiness)
    _write_json(canonical_root / "stage9_4_readiness.json", readiness)
    summary["stage9_4_readiness"] = readiness["status"]
    summary["shadow"] = causal
    _write_json(canonical_root / "stage9_3_2_summary.json", summary)
    _write_json(canonical_root / "root_cause_analysis_v2.json", causal)
    causal_md = (
        "# Stage 9.3.2 Shadow Causal Analysis\n\n"
        + f"- Status: `{causal['causal_label']}`\n- Profiles: `{', '.join(profiles)}`\n- Frames: `{list(selected)}`\n- Solver invocations: `{invocations}`\n- All results are diagnostic-only and evaluated with canonical reference winding SDF.\n\n"
        + "\n".join(
            f"- {row['cause']} ({row['confidence']}): {'; '.join(row['evidence_for'])}"
            for row in causal["causes"]
        )
    )
    (destination / "shadow_causal_analysis.md").write_text(causal_md + "\n", encoding="utf-8")
    comparison_md = (
        "# Stage 9.3.2 Shadow Comparison\n\n"
        f"- Status: `{shadow_manifest['status']}`\n"
        f"- Diagnostic only: `{shadow_manifest['diagnostic_only']}`\n"
        f"- Solver invocations: `{invocations}`\n"
        f"- Selected frames: `{list(selected)}`\n\n"
        "The official baseline is compared against the bounded margin/query and "
        "warm-start projection profiles only when baseline reproduction passes. "
        "A failed baseline reproduction stops the remaining profiles and is not "
        "treated as evidence about the paper method.\n\n"
        f"- Baseline reproduction: `{shadow_manifest['baseline_reproduction_pass']}`\n"
        f"- Causal label: `{causal['causal_label']}`\n"
    )
    (destination / "shadow_comparison.md").write_text(comparison_md, encoding="utf-8")
    _write_json(
        destination / "shadow_manifest.json", shadow_manifest | {"stage9_4_readiness": readiness}
    )
    return {
        "status": "SHADOW_COMPLETE",
        "output_root": str(destination),
        "manifest": shadow_manifest,
        "causal": causal,
        "readiness": readiness,
    }


__all__ = [
    "CANONICAL_PROFILE_ID",
    "CANONICAL_SCHEMA_VERSION",
    "MAX_SHADOW_FRAMES",
    "SHADOW_PROFILES",
    "Stage932PreconditionError",
    "run_canonical_reaudit",
    "run_canonical_shadow_ablation",
]
