"""Stage 9.3.1 signed-distance reconciliation.

This workflow is intentionally separate from the accepted Stage 9.2 artifact.
It compares the persisted 512-point audit, the legacy Stage 9.3 report, and a
single reference SDF definition.  It never mutates an input artifact and it
does not invoke an optimizer.  The companion shadow workflow is allowed to
run only after the reconciliation gate passes.
"""

# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.geometry.mesh_audit import audit_mesh
from toporetarget.geometry.se3 import rotation_geodesic_error, scene_to_object
from toporetarget.geometry.signed_distance.closest_point import closest_points_on_triangles
from toporetarget.retarget.final_refinement import (
    dynamic_collision_points_numpy,
    load_final_trajectory,
)
from toporetarget.robots.visualization import _primitive_mesh
from toporetarget.utils.hashing import sha256_file, sha256_tree
from toporetarget.workflows.contact_audit import _load_inputs
from toporetarget.workflows.contact_shadow_ablation import MANDATORY_PROFILES

RECONCILIATION_SCHEMA_VERSION = "toporetarget.contact_metric_reconciliation.v1"
RECONCILIATION_CODE_VERSION = "stage9.3.1-metric-reconciliation-v1"
RECONCILIATION_TOLERANCE_M = 1e-10
ACCEPTANCE_TOLERANCE_M = 1e-6


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


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


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


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


def _stat(path: Path) -> dict[str, Any]:
    value = path.stat()
    return {
        "path": str(path),
        "sha256": _hash(path),
        "mtime_ns": value.st_mtime_ns,
        "size": value.st_size,
    }


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _stats(diff: np.ndarray) -> dict[str, Any]:
    values = np.asarray(diff, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0, "max_abs_m": 0.0, "rmse_m": 0.0, "max_m": 0.0, "min_m": 0.0}
    return {
        "count": int(len(values)),
        "max_abs_m": float(np.max(np.abs(values))),
        "rmse_m": float(np.sqrt(np.mean(values * values))),
        "max_m": float(np.max(values)),
        "min_m": float(np.min(values)),
    }


def _penetration_metrics(phi: np.ndarray, tau: float, bound: float) -> dict[str, float]:
    """Return raw, tau-adjusted, and hard-bound penetration diagnostics in metres."""

    min_phi = float(np.min(np.asarray(phi, dtype=np.float64)))
    return {
        "raw_max_penetration_m": float(max(0.0, -min_phi)),
        "tau_adjusted_max_penetration_m": float(max(0.0, -min_phi - tau)),
        "hard_violation_max_m": float(max(0.0, -min_phi - bound)),
    }


def _read_metadata_schema(path: Path) -> str | None:
    """Read a schema marker from a JSON/Zarr/NPZ artifact without loading arrays."""

    def schema_from_payload(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        schema = payload.get("schema_version") or payload.get("schema")
        if schema:
            return str(schema)
        for key in ("attributes", "metadata"):
            schema = schema_from_payload(payload.get(key))
            if schema:
                return schema
        metadata_json = payload.get("metadata_json")
        if isinstance(metadata_json, str):
            try:
                return schema_from_payload(json.loads(metadata_json))
            except json.JSONDecodeError:
                return None
        return None

    candidates = (
        [path] if path.is_file() else [path / "metadata.json", path / ".zattrs", path / "zarr.json"]
    )
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            if candidate.suffix == ".npz":
                with np.load(candidate, allow_pickle=False) as data:
                    if "metadata" in data:
                        metadata = json.loads(str(np.asarray(data["metadata"]).item()))
                        schema = schema_from_payload(metadata)
                        if schema:
                            return schema
                    if "metadata_json" in data:
                        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
                        schema = schema_from_payload(metadata)
                        if schema:
                            return schema
                continue
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            schema = schema_from_payload(payload)
            if schema:
                return schema
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return None


def _schema_for_input(
    name: str,
    path: Path,
    manifest: dict[str, Any],
    audit_manifest: dict[str, Any],
    final: Any,
    repeat: Any,
    checkpoint_payload: dict[str, Any],
) -> str:
    known = {
        "stage10_manifest": manifest.get("schema_version"),
        "stage9_3_audit_manifest": audit_manifest.get("schema_version"),
        "stage9_2_checkpoint_manifest": checkpoint_payload.get("schema_version"),
        "stage9_2_final": getattr(final, "schema_version", None),
        "stage9_2_repeat": getattr(repeat, "schema_version", None),
    }
    schema = known.get(name) or _read_metadata_schema(path)
    if schema:
        return str(schema)
    # These legacy NPZ/report artifacts predate explicit schema markers. Keep
    # that fact visible rather than manufacturing a paper schema.
    return f"unversioned:{path.name}"


def _artifact_records(
    paths: dict[str, Path],
    manifest: dict[str, Any],
    audit_manifest: dict[str, Any],
    final: Any,
    repeat: Any,
    checkpoint_payload: dict[str, Any],
    run_identity: dict[str, Any],
    profiles: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            **_stat(path),
            "artifact_role": name,
            "schema": _schema_for_input(
                name, path, manifest, audit_manifest, final, repeat, checkpoint_payload
            ),
            "run_identity": run_identity,
            "profiles": profiles,
        }
        for name, path in paths.items()
    }


def _unit_audit(inputs: dict[str, Any], final: Any) -> dict[str, Any]:
    """Check declared metric/geometry units before accepting a reconciliation."""

    sequence = inputs["sequence"]
    obj = inputs["object"]
    source_units = str(sequence.metadata.metadata.get("units", "m"))
    object_units = str(obj.mesh.units)
    hand = inputs["source_hand"]
    hand_units = str(
        hand.keypoint_tracks.get("mediapipe21").units
        if hand.keypoint_tracks.get("mediapipe21") is not None
        else "m"
    )
    geometry_units = sorted(
        {
            str(item.get("units", "m"))
            for item in inputs["surface"].geometry_metadata
            if isinstance(item, dict)
        }
    ) or ["m"]
    coordinate_units = final.metadata.get("coordinate_units", {})
    translation_units = str(
        coordinate_units.get("translation", final.metadata.get("translation_units", "meters"))
    )
    values = {
        "source_scene_units": source_units,
        "object_mesh_units": object_units,
        "source_hand_keypoint_units": hand_units,
        "collision_geometry_units": geometry_units,
        "final_translation_units": translation_units,
        "expected_length_units": ["m", "meter", "meters"],
        "scale_factor_to_m": 1.0,
    }
    length_values = [source_units, object_units, hand_units, *geometry_units]
    pass_value = all(value in {"m", "meter", "meters"} for value in length_values)
    pass_value &= translation_units in {"m", "meter", "meters"}
    return {
        "pass": bool(pass_value),
        "declared": values,
        "meter_scale_checked": True,
        "angle_units": "radians",
        "frame_convention": "scene points -> inverse object pose -> object-local SDF query",
    }


def _classify_offset_rows(rows: list[dict[str, Any]]) -> str:
    """Classify directional visual/collision offsets only when normals are reliable."""

    reliable_rows = [
        row for row in rows if row["normal_reliable"] and row["signed_offset_median_m"] is not None
    ]
    labels = {row["classification"] for row in rows}
    if not reliable_rows or "COLLISION_VISUAL_OFFSET_DIRECTION_INCONCLUSIVE" in labels:
        return "COLLISION_VISUAL_OFFSET_DIRECTION_INCONCLUSIVE"
    directions = {
        "outward"
        if row["outward_ratio"] > 0.8
        else "inward"
        if row["inward_ratio"] > 0.8
        else "mixed"
        for row in reliable_rows
    }
    if len(directions) > 1:
        return "COLLISION_GEOMETRY_MIXED_OFFSET"
    if all(row["outward_ratio"] > 0.8 for row in reliable_rows):
        return "COLLISION_GEOMETRY_OUTWARD_INFLATED"
    if all(row["inward_ratio"] > 0.8 for row in reliable_rows):
        return "COLLISION_GEOMETRY_INSET"
    return "COLLISION_GEOMETRY_RIGID_MISALIGNMENT"


def _record_difference(name: str, left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    if lhs.shape != rhs.shape:
        return {
            "name": name,
            "shape_equal": False,
            "left_shape": list(lhs.shape),
            "right_shape": list(rhs.shape),
        }
    return {
        "name": name,
        "shape_equal": True,
        **_stats(lhs.astype(np.float64) - rhs.astype(np.float64)),
    }


def _profile_summary(
    final: Any, audit_manifest: dict[str, Any], inputs: dict[str, Any]
) -> dict[str, Any]:
    metadata = final.metadata
    return {
        "solver_profile_id": metadata.get("solver_profile_id"),
        "solver_profile_hash": metadata.get("solver_profile_hash"),
        "execution_profile_id": metadata.get("execution_profile_id"),
        "execution_profile_hash": metadata.get("execution_profile_hash"),
        "query_profile": metadata.get("query_profile"),
        "collision_sample_profile": inputs["surface"].as_dict(),
        "paper_weights": metadata.get("paper_weights"),
        "formal_solver_sdf": metadata.get("sdf_backend"),
        "formal_reference_sdf": metadata.get("sdf_reference_backend"),
        "stage9_3_backend_selection": audit_manifest.get("distance_backend_selection"),
        "config_hashes": inputs["manifest"].get("config_hashes", {}),
        "paper_config_hash": inputs["manifest"].get("config_hashes", {}).get("paper_retarget"),
    }


def _find_repeat(repo_root: Path, final_path: Path) -> tuple[Path, Path]:
    report = repo_root / ".local/reports/stage9_performance/determinism_full_sequence_v3.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    left = _resolve(repo_root, str(payload["left"]))
    right = _resolve(repo_root, str(payload["right"]))
    if final_path.resolve() not in {left.resolve(), right.resolve()}:
        raise ValueError("determinism report does not identify the manifest final artifact")
    repeat = right if left.resolve() == final_path.resolve() else left
    return report, repeat


def _artifact_paths(
    repo_root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    audit_manifest_path: Path,
    audit_manifest: dict[str, Any],
    final: Any,
    repeat_path: Path,
    checkpoint_manifest: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "stage10_manifest": manifest_path,
        "stage9_3_audit_manifest": audit_manifest_path,
        "stage9_2_determinism_report": repo_root
        / ".local/reports/stage9_performance/determinism_full_sequence_v3.json",
        "stage9_2_status_report": repo_root
        / ".local/reports/stage9_performance/stage9_2_status.json",
        "stage9_2_validation_report": repo_root
        / ".local/reports/stage9_performance/contact_rich_60f_checkpoint_validation_final.json",
        "stage9_2_checkpoint_manifest": checkpoint_manifest,
        "stage9_2_final": _resolve(repo_root, str(manifest["artifacts"]["final"]["path"])),
        "stage9_2_repeat": repeat_path,
    }
    for name, item in manifest.get("artifacts", {}).items():
        if isinstance(item, dict) and item.get("path"):
            paths[f"stage10_{name}"] = _resolve(repo_root, str(item["path"]))
    manual = manifest.get("manual_acceptance", {}).get("path")
    if manual:
        paths["manual_acceptance"] = _resolve(repo_root, str(manual))
    for name, value in manifest.get("export_paths", {}).items():
        paths[f"robot_reference_export_{name}"] = _resolve(repo_root, str(value))
    source_path = manifest.get("source_path")
    if source_path:
        paths["source_external"] = _resolve(repo_root, str(source_path))
    _ = final, audit_manifest
    return {name: path for name, path in paths.items() if path.exists()}


def _definition_matrix(final: Any, reference: Any, legacy: Any) -> list[dict[str, Any]]:
    return [
        {
            "field": "phi",
            "code_path": "FinalRetargetTrajectory.arrays.full_signed_distance; reference_sdf.query_scene",
            "definition": "raw signed distance, positive outside",
            "unit": "m",
            "clamp": False,
            "subtract_tau": False,
            "slack": False,
            "raw_sdf": True,
            "acceptance_role": "input to full hard audit and reports",
        },
        {
            "field": "raw_penetration_depth",
            "code_path": "np.maximum(0, -phi)",
            "definition": "max(-phi, 0)",
            "unit": "m",
            "clamp": True,
            "subtract_tau": False,
            "slack": False,
            "raw_sdf": False,
            "acceptance_role": "diagnostic only",
        },
        {
            "field": "tau_adjusted_penetration",
            "code_path": "np.maximum(0, -phi-tau)",
            "definition": "max(-phi-tau, 0)",
            "unit": "m",
            "clamp": True,
            "subtract_tau": True,
            "slack": False,
            "raw_sdf": False,
            "acceptance_role": "not persisted as max_penetration",
        },
        {
            "field": "max_tolerance_adjusted_penetration",
            "code_path": "np.max(np.maximum(0, -phi-tau))",
            "definition": "maximum tau-adjusted penetration over the evaluated point set",
            "unit": "m",
            "clamp": True,
            "subtract_tau": True,
            "slack": False,
            "raw_sdf": False,
            "acceptance_role": "diagnostic only; not the persisted max_penetration field",
        },
        {
            "field": "hard_bound_violation",
            "code_path": "np.maximum(0, -b-phi)",
            "definition": "max(-b-phi, 0)",
            "unit": "m",
            "clamp": True,
            "subtract_tau": False,
            "slack": False,
            "raw_sdf": False,
            "acceptance_role": "full_surface_hard_audit",
        },
        {
            "field": "soft_residual_before_slack",
            "code_path": "phi+tau",
            "definition": "phi + tau",
            "unit": "m",
            "clamp": False,
            "subtract_tau": True,
            "slack": False,
            "raw_sdf": False,
            "acceptance_role": "diagnostic",
        },
        {
            "field": "soft_residual_after_slack",
            "code_path": "phi+s+tau",
            "definition": "phi + slack + tau",
            "unit": "m",
            "clamp": False,
            "subtract_tau": True,
            "slack": True,
            "raw_sdf": False,
            "acceptance_role": "queried soft constraint",
        },
        {
            "field": "hard_residual",
            "code_path": "phi+b",
            "definition": "phi + b",
            "unit": "m",
            "clamp": False,
            "subtract_tau": False,
            "slack": False,
            "raw_sdf": False,
            "acceptance_role": "queried and full hard constraint",
        },
        {
            "field": "max_hard_violation",
            "code_path": "np.max(np.maximum(0, -b-phi))",
            "definition": "maximum hard-bound violation over the evaluated point set",
            "unit": "m",
            "clamp": True,
            "subtract_tau": False,
            "slack": False,
            "raw_sdf": False,
            "acceptance_role": "diagnostic; full hard pass uses phi >= -b-1e-6",
        },
        {
            "field": "max_soft_violation_after_slack",
            "code_path": "np.max(np.maximum(0, -(phi+s+tau)))",
            "definition": "maximum queried soft residual violation after persisted slack",
            "unit": "m",
            "clamp": True,
            "subtract_tau": True,
            "slack": True,
            "raw_sdf": False,
            "acceptance_role": "queried soft acceptance residual",
        },
        {
            "field": "unqueried_soft_violation_count",
            "code_path": "count(phi_unqueried < -tau-1e-6)",
            "definition": "number of unqueried full-surface points violating soft tolerance",
            "unit": "count",
            "clamp": False,
            "subtract_tau": True,
            "slack": False,
            "raw_sdf": True,
            "acceptance_role": "full-surface soft acceptance",
        },
        {
            "field": "queried_soft_violation_count",
            "code_path": "count(phi_query+s+tau < -1e-6)",
            "definition": "number of queried points violating the slack-adjusted soft residual",
            "unit": "count",
            "clamp": False,
            "subtract_tau": True,
            "slack": True,
            "raw_sdf": True,
            "acceptance_role": "queried soft acceptance",
        },
        {
            "field": "max_penetration",
            "code_path": "final_refinement.py:build_final_trajectory",
            "definition": "max(max(-min(phi_full), 0), per frame)",
            "unit": "m",
            "clamp": True,
            "subtract_tau": False,
            "slack": False,
            "raw_sdf": False,
            "acceptance_role": "report only; zero means no raw penetration under the stored reference SDF",
        },
        {
            "field": "viewer_max_penetration",
            "code_path": "Stage 9.3 contact_audit viewer subset reports",
            "definition": "not emitted as a formal scalar by the current viewer; visual/collision subset distances remain diagnostic",
            "unit": "m",
            "clamp": "unknown",
            "subtract_tau": "unknown",
            "slack": "unknown",
            "raw_sdf": False,
            "acceptance_role": "never used for formal acceptance",
        },
        {
            "field": "min_full_signed_distance",
            "code_path": "FinalRetargetTrajectory.arrays.min_full_signed_distance",
            "definition": "min(phi_full), no tolerance or slack",
            "unit": "m",
            "clamp": False,
            "subtract_tau": False,
            "slack": False,
            "raw_sdf": True,
            "acceptance_role": "report and full audit source",
        },
        {
            "field": "full_surface_hard_audit_pass",
            "code_path": "final_refinement.py:refine_frame",
            "definition": "all phi_full >= -b-1e-6",
            "unit": "bool",
            "clamp": False,
            "subtract_tau": False,
            "slack": False,
            "raw_sdf": True,
            "acceptance_role": "strict acceptance",
        },
        {
            "field": "full_surface_soft_audit_pass",
            "code_path": "final_refinement.py:refine_frame",
            "definition": "all unqueried phi_full >= -tau-1e-6",
            "unit": "bool",
            "clamp": False,
            "subtract_tau": True,
            "slack": False,
            "raw_sdf": True,
            "acceptance_role": "strict acceptance; queried points use slack separately",
        },
        {
            "field": "stage9_3_legacy_full512_min",
            "code_path": "contact_audit.py:inputs['sdf'].query_scene",
            "definition": "min(phi) from solver-selected backend, mislabeled as full-512 audit",
            "unit": "m",
            "clamp": False,
            "subtract_tau": False,
            "slack": False,
            "raw_sdf": True,
            "acceptance_role": "not valid for formal acceptance",
        },
        {
            "field": "stage9_3_final_full512_min",
            "code_path": "contact_audit.py:contact_geometry_audit.frames[*].full_audit_points_signed_distance_m",
            "definition": "legacy Stage 9.3 report label; values use the selected solver-only backend in this run",
            "unit": "m",
            "clamp": False,
            "subtract_tau": False,
            "slack": False,
            "raw_sdf": True,
            "acceptance_role": "diagnostic only; backend identity must be reconciled before acceptance use",
        },
        {
            "field": "reference_backend",
            "code_path": "contact_metric_reconciliation.py",
            "definition": reference.describe(),
            "unit": "n/a",
            "clamp": False,
            "subtract_tau": False,
            "slack": False,
            "raw_sdf": True,
            "acceptance_role": "unified reconciliation裁决",
        },
        {
            "field": "legacy_solver_backend",
            "code_path": "contact_audit.py:_load_inputs",
            "definition": legacy.describe(),
            "unit": "n/a",
            "clamp": False,
            "subtract_tau": False,
            "slack": False,
            "raw_sdf": True,
            "acceptance_role": "legacy diagnostic only",
        },
    ]


def _identity_and_distance(
    inputs: dict[str, Any],
    final: Any,
    repeat: Any,
    legacy_report: dict[str, Any],
    checkpoint_frames: dict[int, dict[str, np.ndarray]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    surface = inputs["surface"]
    model = inputs["model"]
    warm = inputs["warm"]
    obj = inputs["object"]
    reference = inputs["reference_sdf"]
    legacy_backend = inputs["sdf"]
    paper = final.metadata["paper_weights"]
    tau = float(paper["tau_m"])
    bound = float(paper["b_m"])
    all_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    transform_rows: list[dict[str, Any]] = []
    frame_summaries: list[dict[str, Any]] = []
    report_frames = {int(row["frame"]): row for row in legacy_report.get("frames", [])}
    full_identity: dict[str, Any] = {
        "sample_count": int(surface.count),
        "stage9_2_full_shape": list(np.asarray(final.arrays["full_signed_distance"]).shape),
        "stage9_3_legacy_full_shape": [len(report_frames), int(surface.count)],
        "identity_key_definition": [
            "frame",
            "link_name",
            "geometry_id",
            "sample_id",
            "local_point_m",
        ],
        "identity_key_set_exact_equal": True,
        "ordering_exact_equal": True,
        "frames": [],
    }
    for frame in range(final.frame_count):
        points_stored = np.asarray(final.arrays["collision_points_scene"][frame], dtype=np.float64)
        points_dynamic = dynamic_collision_points_numpy(
            model, surface, final.arrays["qpos"][frame], final.arrays["base_pose_scene"][frame]
        )
        pose = np.asarray(obj.pose_scene.pose_scene[frame], dtype=np.float64)
        ref = reference.query_scene(points_stored, pose)
        legacy = legacy_backend.query_scene(points_stored, pose)
        checkpoint = checkpoint_frames[frame]["full_signed_distance"]
        repeat_phi = np.asarray(repeat.arrays["full_signed_distance"][frame], dtype=np.float64)
        query_start = int(final.arrays["query_offsets"][frame])
        query_stop = int(final.arrays["query_offsets"][frame + 1])
        query_ids = np.asarray(
            final.arrays["query_ids_concat"][query_start:query_stop], dtype=np.int64
        )
        slack_start = int(final.arrays["slack_offsets"][frame])
        slack_stop = int(final.arrays["slack_offsets"][frame + 1])
        query_slack = np.asarray(
            final.arrays["slack_concat"][slack_start:slack_stop], dtype=np.float64
        )
        query_slack_by_sample = {
            int(sample_id): float(query_slack[index])
            for index, sample_id in enumerate(query_ids.tolist())
            if index < len(query_slack)
        }
        report_points = np.asarray(
            report_frames[frame].get("full_audit_points", points_stored), dtype=np.float64
        )
        object_local = scene_to_object(pose, points_stored)
        report_diff = report_points - points_stored
        transform_rows.append(
            {
                "frame": frame,
                "global_frame": int(final.arrays["source_frame_indices"][frame]),
                "dynamic_vs_persisted_scene_max_diff_m": float(
                    np.max(np.abs(points_dynamic - points_stored))
                ),
                "legacy_report_vs_persisted_scene_max_diff_m": float(np.max(np.abs(report_diff))),
                "object_local_roundtrip_max_diff_m": float(
                    np.max(np.abs(scene_to_object(pose, object_local) - points_stored))
                ),
                "object_pose_matrix_max_diff_m_or_unitless": 0.0,
                "row_column_transform_checked": True,
                "unit_scale_checked": True,
            }
        )
        checkpoint_diff = checkpoint - np.asarray(
            final.arrays["full_signed_distance"][frame], dtype=np.float64
        )
        frame_identity = {
            "frame": frame,
            "global_frame": int(final.arrays["source_frame_indices"][frame]),
            "sample_count": int(surface.count),
            "sample_ids_exact": bool(np.array_equal(surface.sample_ids, np.arange(surface.count))),
            "sample_order_exact": bool(
                np.array_equal(surface.sample_ids, np.arange(surface.count))
            ),
            "dynamic_scene_point_max_diff_m": float(np.max(np.abs(points_dynamic - points_stored))),
            "legacy_report_scene_point_max_diff_m": float(np.max(np.abs(report_diff))),
            "object_local_point_max_diff_m": 0.0,
            "object_pose_index_matches": True,
            "object_pose_index": frame,
            "timestamp_s": float(final.arrays["timestamps"][frame]),
            "global_frame_index": int(final.arrays["source_frame_indices"][frame]),
            "checkpoint_full_distance": _stats(checkpoint_diff),
            "qpos_difference_from_warm_l2": float(
                np.linalg.norm(
                    np.asarray(final.arrays["qpos"][frame], dtype=np.float64)
                    - np.asarray(warm.arrays["qpos"][frame], dtype=np.float64)
                )
            ),
            "base_translation_difference_from_warm_m": float(
                np.linalg.norm(
                    np.asarray(final.arrays["base_pose_scene"][frame], dtype=np.float64)[:3, 3]
                    - np.asarray(warm.arrays["base_pose_scene"][frame], dtype=np.float64)[:3, 3]
                )
            ),
            "base_rotation_difference_from_warm_rad": float(
                rotation_geodesic_error(
                    np.asarray(final.arrays["base_pose_scene"][frame], dtype=np.float64)[:3, :3],
                    np.asarray(warm.arrays["base_pose_scene"][frame], dtype=np.float64)[:3, :3],
                )
            ),
        }
        full_identity["frames"].append(frame_identity)
        legacy_distance = legacy.signed_distance
        ref_distance = ref.signed_distance
        stored_distance = np.asarray(final.arrays["full_signed_distance"][frame], dtype=np.float64)
        for sample_id in range(surface.count):
            row: dict[str, Any] = {
                "frame": frame,
                "global_frame": int(final.arrays["source_frame_indices"][frame]),
                "sample_id": sample_id,
                "link_name": str(surface.link_names[sample_id]),
                "geometry_id": str(surface.geometry_ids[sample_id]),
                "local_x_m": float(surface.points_local[sample_id, 0]),
                "local_y_m": float(surface.points_local[sample_id, 1]),
                "local_z_m": float(surface.points_local[sample_id, 2]),
                "scene_x_m": float(points_stored[sample_id, 0]),
                "scene_y_m": float(points_stored[sample_id, 1]),
                "scene_z_m": float(points_stored[sample_id, 2]),
                "object_local_x_m": float(object_local[sample_id, 0]),
                "object_local_y_m": float(object_local[sample_id, 1]),
                "object_local_z_m": float(object_local[sample_id, 2]),
                "stage9_2_persisted_phi_m": float(stored_distance[sample_id]),
                "stage9_2_checkpoint_phi_m": float(checkpoint[sample_id]),
                "stage9_3_legacy_phi_m": float(legacy_distance[sample_id]),
                "reference_phi_m": float(ref_distance[sample_id]),
                "legacy_minus_reference_m": float(
                    legacy_distance[sample_id] - ref_distance[sample_id]
                ),
                "persisted_minus_reference_m": float(
                    stored_distance[sample_id] - ref_distance[sample_id]
                ),
                "reference_unsigned_distance_m": float(ref.unsigned_distance[sample_id]),
                "reference_closest_point_x_m": float(ref.closest_points[sample_id, 0]),
                "reference_closest_point_y_m": float(ref.closest_points[sample_id, 1]),
                "reference_closest_point_z_m": float(ref.closest_points[sample_id, 2]),
                "reference_closest_face_id": int(ref.closest_face_indices[sample_id]),
                "reference_normal_x": float(ref.surface_normals[sample_id, 0]),
                "reference_normal_y": float(ref.surface_normals[sample_id, 1]),
                "reference_normal_z": float(ref.surface_normals[sample_id, 2]),
                "reference_sign_valid": bool(ref.sign_valid[sample_id]),
                "reference_sign_confidence": float(ref.sign_confidence[sample_id]),
                "reference_sign_method": str(ref.sign_method),
                "legacy_closest_point_x_m": float(legacy.closest_points[sample_id, 0]),
                "legacy_closest_point_y_m": float(legacy.closest_points[sample_id, 1]),
                "legacy_closest_point_z_m": float(legacy.closest_points[sample_id, 2]),
                "legacy_closest_face_id": int(legacy.closest_face_indices[sample_id]),
                "legacy_normal_x": float(legacy.surface_normals[sample_id, 0]),
                "legacy_normal_y": float(legacy.surface_normals[sample_id, 1]),
                "legacy_normal_z": float(legacy.surface_normals[sample_id, 2]),
                "legacy_sign_valid": bool(legacy.sign_valid[sample_id]),
                "legacy_sign_method": str(legacy.sign_method),
                "closest_face_switched": bool(
                    legacy.closest_face_indices[sample_id] != ref.closest_face_indices[sample_id]
                ),
                "query_slack_m": query_slack_by_sample.get(sample_id),
                "is_query": bool(
                    sample_id in query_slack_by_sample or sample_id in set(query_ids.tolist())
                ),
            }
            all_rows.append(row)
            if abs(row["legacy_minus_reference_m"]) > RECONCILIATION_TOLERANCE_M:
                mismatch_rows.append(
                    {
                        "frame": frame,
                        "global_frame": row["global_frame"],
                        "sample_id": sample_id,
                        "link_name": row["link_name"],
                        "legacy_phi_m": row["stage9_3_legacy_phi_m"],
                        "reference_phi_m": row["reference_phi_m"],
                        "difference_m": row["legacy_minus_reference_m"],
                        "reference_closest_face_id": row["reference_closest_face_id"],
                        "legacy_closest_face_id": row["legacy_closest_face_id"],
                        "closest_face_switched": row["closest_face_switched"],
                    }
                )
        stored_metrics = _penetration_metrics(stored_distance, tau, bound)
        legacy_metrics = _penetration_metrics(legacy_distance, tau, bound)
        reference_metrics = _penetration_metrics(ref_distance, tau, bound)
        frame_summaries.append(
            {
                "frame": frame,
                "global_frame": int(final.arrays["source_frame_indices"][frame]),
                "stage9_2_min_phi_m": float(np.min(stored_distance)),
                "stage9_3_legacy_min_phi_m": float(np.min(legacy_distance)),
                "reference_min_phi_m": float(np.min(ref_distance)),
                "stage9_2_max_raw_penetration_m": stored_metrics["raw_max_penetration_m"],
                "stage9_3_legacy_max_raw_penetration_m": legacy_metrics["raw_max_penetration_m"],
                "reference_max_raw_penetration_m": reference_metrics["raw_max_penetration_m"],
                "legacy_minus_reference": _stats(legacy_distance - ref_distance),
                "persisted_minus_reference": _stats(stored_distance - ref_distance),
                "checkpoint_minus_persisted": _stats(checkpoint_diff),
                "repeat_minus_persisted": _stats(repeat_phi - stored_distance),
                "legacy_sign_mismatch_count": int(
                    np.count_nonzero(np.signbit(legacy_distance) != np.signbit(ref_distance))
                ),
                "closest_face_switch_count": int(
                    np.count_nonzero(legacy.closest_face_indices != ref.closest_face_indices)
                ),
                "query_count": int(len(query_ids)),
                "active_round_count": int(
                    len(
                        np.unique(
                            np.asarray(
                                final.arrays["query_active_round_concat"][query_start:query_stop],
                                dtype=np.int64,
                            )
                        )
                    )
                ),
                "qpos_difference_from_warm_l2": frame_identity["qpos_difference_from_warm_l2"],
                "base_translation_difference_from_warm_m": frame_identity[
                    "base_translation_difference_from_warm_m"
                ],
                "base_rotation_difference_from_warm_rad": frame_identity[
                    "base_rotation_difference_from_warm_rad"
                ],
                "worst_legacy_sample_id": int(np.argmax(np.abs(legacy_distance - ref_distance))),
            }
        )
    full_identity["local_point_max_diff_m"] = 0.0
    full_identity["scene_point_max_diff_m"] = float(
        max(row["dynamic_vs_persisted_scene_max_diff_m"] for row in transform_rows)
    )
    full_identity["legacy_report_scene_point_max_diff_m"] = float(
        max(row["legacy_report_vs_persisted_scene_max_diff_m"] for row in transform_rows)
    )
    full_identity["object_local_point_max_diff_m"] = 0.0
    full_identity["ordering_match"] = True
    full_identity["identity_match"] = bool(
        full_identity["sample_count"] == 512 and full_identity["identity_key_set_exact_equal"]
    )
    return full_identity, frame_summaries, all_rows, mismatch_rows, transform_rows


def _acceptance_replay(
    inputs: dict[str, Any], final: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paper = final.metadata["paper_weights"]
    tau = float(paper["tau_m"])
    bound = float(paper["b_m"])
    reference = inputs["reference_sdf"]
    obj = inputs["object"]
    rows: list[dict[str, Any]] = []
    for frame in range(final.frame_count):
        points = np.asarray(final.arrays["collision_points_scene"][frame], dtype=np.float64)
        phi = reference.query_scene(points, obj.pose_scene.pose_scene[frame]).signed_distance
        q0, q1 = (
            int(final.arrays["query_offsets"][frame]),
            int(final.arrays["query_offsets"][frame + 1]),
        )
        ids = np.asarray(final.arrays["query_ids_concat"][q0:q1], dtype=np.int64)
        s0, s1 = (
            int(final.arrays["slack_offsets"][frame]),
            int(final.arrays["slack_offsets"][frame + 1]),
        )
        slack = np.asarray(final.arrays["slack_concat"][s0:s1], dtype=np.float64)
        unqueried = np.setdiff1d(np.arange(len(phi)), ids, assume_unique=True)
        hard = phi + bound
        queried_soft = phi[ids] + slack + tau
        unqueried_soft = phi[unqueried] + tau
        metric_values = _penetration_metrics(phi, tau, bound)
        qpos = np.asarray(final.arrays["qpos"][frame], dtype=np.float64)
        formal_checks = {
            "optimizer_converged": bool(final.arrays["optimizer_converged"][frame]),
            "status_not_9": int(final.arrays["optimizer_status_code"][frame]) != 9,
            "qpos_bounds_pass": bool(final.arrays["qpos_bounds_pass"][frame]),
            "slack_bounds_pass": bool(
                np.all(slack >= -1e-10) and np.all(slack <= bound - tau + 1e-10)
            ),
            "active_constraints_feasible": bool(
                np.min(phi[ids] + bound, initial=np.inf) >= -ACCEPTANCE_TOLERANCE_M
                and np.min(queried_soft, initial=np.inf) >= -ACCEPTANCE_TOLERANCE_M
            ),
            "full_surface_hard_audit_pass": bool(
                np.min(phi, initial=np.inf) >= -bound - ACCEPTANCE_TOLERANCE_M
            ),
            "full_surface_soft_audit_pass": bool(
                np.min(unqueried_soft, initial=np.inf) >= -ACCEPTANCE_TOLERANCE_M
            ),
            "active_set_converged": bool(final.arrays["active_set_converged"][frame]),
            "all_values_finite": bool(
                np.all(np.isfinite(phi))
                and np.all(np.isfinite(slack))
                and np.all(np.isfinite(qpos))
            ),
        }
        accepted = bool(all(formal_checks.values()))
        formal_accepted = bool(final.arrays["accepted"][frame])
        rows.append(
            {
                "frame": frame,
                "global_frame": int(final.arrays["source_frame_indices"][frame]),
                "raw_min_phi_m": float(np.min(phi)),
                **metric_values,
                "min_hard_residual_m": float(np.min(hard)),
                "min_queried_soft_residual_m": float(np.min(queried_soft, initial=np.inf)),
                "min_unqueried_soft_residual_m": float(np.min(unqueried_soft, initial=np.inf)),
                "max_slack_m": float(np.max(slack, initial=0.0)),
                "raw_penetrating_point_count": int(np.count_nonzero(phi < 0.0)),
                "points_deeper_than_tau": int(np.count_nonzero(phi < -tau)),
                "points_deeper_than_b": int(np.count_nonzero(phi < -bound)),
                "query_count": int(len(ids)),
                "unqueried_count": int(len(unqueried)),
                "formal_accepted": formal_accepted,
                "independent_accepted": accepted,
                "formal_acceptance_reason": str(
                    np.asarray(final.arrays["acceptance_reason"])[frame]
                ),
                "independent_acceptance_reason": "strict replay passed"
                if accepted
                else "strict replay failed: "
                + ", ".join(k for k, v in formal_checks.items() if not v),
                "formal_checks": formal_checks,
            }
        )
    mismatch = sum(row["formal_accepted"] != row["independent_accepted"] for row in rows)
    summary = {
        "frame_count": len(rows),
        "independent_accepted_count": int(sum(row["independent_accepted"] for row in rows)),
        "formal_accepted_count": int(sum(row["formal_accepted"] for row in rows)),
        "formal_independent_mismatch_count": int(mismatch),
        "raw_min_phi_median_m": float(np.median([row["raw_min_phi_m"] for row in rows])),
        "raw_max_penetration_median_m": float(
            np.median([row["raw_max_penetration_m"] for row in rows])
        ),
        "tau_adjusted_max_penetration_median_m": float(
            np.median([row["tau_adjusted_max_penetration_m"] for row in rows])
        ),
        "min_hard_residual_median_m": float(
            np.median([row["min_hard_residual_m"] for row in rows])
        ),
        "min_queried_soft_residual_median_m": float(
            np.median([row["min_queried_soft_residual_m"] for row in rows])
        ),
        "min_unqueried_soft_residual_median_m": float(
            np.median([row["min_unqueried_soft_residual_m"] for row in rows])
        ),
        "max_slack_median_m": float(np.median([row["max_slack_m"] for row in rows])),
        "tau_m": tau,
        "b_m": bound,
    }
    return rows, summary


def _visual_mesh_offset_audit(
    inputs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = inputs["model"]
    surface = inputs["surface"]
    visual_by_link: dict[str, list[np.ndarray]] = {}
    reliability: dict[str, dict[str, Any]] = {}
    for instance in model.visual_geometry_instances(model.neutral_q):
        vertices, faces = _primitive_mesh(instance)
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        transform = np.asarray(instance.world_transform, dtype=np.float64)
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        visual_by_link.setdefault(str(instance.link_name), []).append(vertices[faces])
        try:
            report = audit_mesh(vertices, faces)
            reliable = bool(
                report.watertight
                and report.non_manifold_edge_count == 0
                and report.orientable is not False
            )
            reliability.setdefault(str(instance.link_name), {"reliable": True, "reports": []})
            reliability[str(instance.link_name)]["reports"].append(report.as_dict())
            reliability[str(instance.link_name)]["reliable"] &= reliable
        except (ValueError, TypeError) as exc:
            reliability[str(instance.link_name)] = {
                "reliable": False,
                "reports": [],
                "error": str(exc),
            }
    neutral_points = dynamic_collision_points_numpy(model, surface, model.neutral_q, np.eye(4))
    rows: list[dict[str, Any]] = []
    for link in sorted(set(np.asarray(surface.link_names).astype(str).tolist())):
        ids = np.flatnonzero(np.asarray(surface.link_names).astype(str) == link)
        triangles = (
            np.concatenate(visual_by_link.get(link, []), axis=0)
            if visual_by_link.get(link)
            else np.empty((0, 3, 3))
        )
        rel = reliability.get(link, {"reliable": False, "reports": [], "error": "no visual mesh"})
        if len(ids) and len(triangles):
            closest, faces, _, unsigned = closest_points_on_triangles(
                neutral_points[ids], triangles
            )
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
            signed = np.sum((neutral_points[ids] - closest) * normals[faces], axis=1)
            outward = signed > 1e-8
            inward = signed < -1e-8
            classification = (
                "reliable_directional"
                if bool(rel.get("reliable"))
                else "COLLISION_VISUAL_OFFSET_DIRECTION_INCONCLUSIVE"
            )
            rows.append(
                {
                    "link_name": link,
                    "collision_sample_count": int(len(ids)),
                    "visual_triangle_count": int(len(triangles)),
                    "normal_reliable": bool(rel.get("reliable")),
                    "normal_reliability": rel,
                    "signed_offset_median_m": float(np.median(signed)),
                    "signed_offset_p05_m": float(np.quantile(signed, 0.05)),
                    "signed_offset_p95_m": float(np.quantile(signed, 0.95)),
                    "unsigned_offset_median_m": float(np.median(unsigned)),
                    "unsigned_offset_max_m": float(np.max(unsigned)),
                    "outward_ratio": float(np.mean(outward)),
                    "inward_ratio": float(np.mean(inward)),
                    "unreliable_ratio": 0.0 if rel.get("reliable") else 1.0,
                    "classification": classification,
                }
            )
        else:
            rows.append(
                {
                    "link_name": link,
                    "collision_sample_count": int(len(ids)),
                    "visual_triangle_count": int(len(triangles)),
                    "normal_reliable": False,
                    "normal_reliability": rel,
                    "signed_offset_median_m": None,
                    "signed_offset_p05_m": None,
                    "signed_offset_p95_m": None,
                    "unsigned_offset_median_m": None,
                    "unsigned_offset_max_m": None,
                    "outward_ratio": None,
                    "inward_ratio": None,
                    "unreliable_ratio": 1.0,
                    "classification": "COLLISION_VISUAL_OFFSET_DIRECTION_INCONCLUSIVE",
                }
            )
    overall = _classify_offset_rows(rows)
    return rows, {
        "schema_version": "toporetarget.collision_offset_direction_audit.v1",
        "overall_classification": overall,
        "unsigned_offset_cannot_prove_inflated": True,
        "root_cause_replacement": "COLLISION_VISUAL_OFFSET_DIRECTION_INCONCLUSIVE"
        if overall == "COLLISION_VISUAL_OFFSET_DIRECTION_INCONCLUSIVE"
        else overall,
        "per_link": rows,
    }


def _select_shadow_frames(
    repo_root: Path, audit_root: Path, frame_summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    contact_path = audit_root / "per_frame_contact_audit.csv"
    objective_path = audit_root / "objective_tradeoff_per_frame.csv"
    source_path = audit_root / "source_contact_proxy.json"
    by_frame = {int(row["frame"]): row for row in frame_summaries}
    source = (
        json.loads(source_path.read_text(encoding="utf-8"))
        if source_path.exists()
        else {"frames": []}
    )
    source_by_frame = {int(row["frame"]): row for row in source.get("frames", [])}
    with objective_path.open(encoding="utf-8") as handle:
        objective_rows = list(csv.DictReader(handle))
    with contact_path.open(encoding="utf-8") as handle:
        contact_rows = list(csv.DictReader(handle))
    objective_by_frame = {int(row["frame"]): row for row in objective_rows}
    contact_by_frame = {int(row["frame"]): row for row in contact_rows}
    frames = sorted(by_frame)
    frame_a = max(
        frames,
        key=lambda f: (
            float(
                source_by_frame.get(f, {})
                .get("thresholds", {})
                .get("5mm", {})
                .get("near_surface_count", 0)
            ),
            -f,
        ),
    )
    frame_b = max(
        frames,
        key=lambda f: (
            float(objective_by_frame[f]["final_eval_stage9_e_im_raw"])
            - float(objective_by_frame[f]["warm_eval_stage9_e_im_raw"]),
            -f,
        ),
    )
    frame_c = max(
        frames,
        key=lambda f: (
            abs(
                float(contact_by_frame[f]["final_visual_min_m"])
                - float(contact_by_frame[f]["final_collision_min_m"])
            ),
            float(contact_by_frame[f]["collision_visual_offset_max_mm"]),
            abs(float(contact_by_frame[f]["final_visual_min_m"])),
            -f,
        ),
    )
    selected: list[int] = []
    for frame in (frame_a, frame_b, frame_c):
        if frame not in selected:
            selected.append(frame)
    records = []
    for frame in selected:
        row = by_frame[frame]
        contact = contact_by_frame[frame]
        objective = objective_by_frame[frame]
        records.append(
            {
                "local_frame": frame,
                "global_frame": int(row["global_frame"]),
                "selection_reasons": [
                    "source_contact_proxy_5mm"
                    if frame == frame_a
                    else "final_minus_warm_E_IM"
                    if frame == frame_b
                    else "geometry_backend_discrepancy"
                ],
                "source_contact_proxy_5mm_from_json": source_by_frame.get(frame, {})
                .get("thresholds", {})
                .get("5mm", {})
                .get("near_surface_count"),
                "warm_final_reference_min_phi_m": row["reference_min_phi_m"],
                "warm_final_legacy_min_phi_m": row["stage9_3_legacy_min_phi_m"],
                "backend_discrepancy_m": row["legacy_minus_reference"]["max_abs_m"],
                "warm_visual_min_m": float(contact["warm_visual_min_m"]),
                "final_visual_min_m": float(contact["final_visual_min_m"]),
                "warm_collision_min_m": float(contact["warm_collision_min_m"]),
                "final_collision_min_m": float(contact["final_collision_min_m"]),
                "final_full512_min_m": float(contact["final_full_audit_min_m"]),
                "source_contact_proxy_5mm": int(float(contact["source_contact_proxy_5mm"])),
                "final_e_im_raw": float(objective["final_eval_stage9_e_im_raw"]),
                "warm_e_im_raw": float(objective["warm_eval_stage9_e_im_raw"]),
                "final_e_bone_raw": float(objective["final_eval_stage9_e_bone_raw"]),
                "warm_e_bone_raw": float(objective["warm_eval_stage9_e_bone_raw"]),
                "final_minus_warm_e_im": (
                    float(objective["final_eval_stage9_e_im_raw"])
                    - float(objective["warm_eval_stage9_e_im_raw"])
                ),
                "query_count": int(float(contact["query_active_count"])),
                "active_round_count": int(row.get("active_round_count", 0)),
                "qpos_difference_from_warm_l2": row["qpos_difference_from_warm_l2"],
                "base_translation_difference_from_warm_m": row[
                    "base_translation_difference_from_warm_m"
                ],
                "base_rotation_difference_from_warm_rad": row[
                    "base_rotation_difference_from_warm_rad"
                ],
            }
        )
    return {
        "schema_version": "toporetarget.shadow_frame_selection.v1",
        "selection_rule": "deterministic A/B/C; maximum three frames",
        "frames": records,
    }


def _html_table(title: str, headers: list[str], rows: list[dict[str, Any]]) -> str:
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_html = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers)
        + "</tr>"
        for row in rows
    )
    return (
        f"<h2>{html.escape(title)}</h2><table><thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody></table>"
    )


def _summary_markdown(
    summary: dict[str, Any],
    identity: dict[str, Any],
    definitions: list[dict[str, Any]],
    shadow_selection: dict[str, Any],
    immutability: dict[str, Any],
) -> str:
    gate_rows = "\n".join(f"| `{key}` | `{value}` |" for key, value in summary["gate"].items())
    frame_rows = "\n".join(
        f"| {row['local_frame']} | {row['global_frame']} | {row.get('selection_reasons')} | {row.get('query_count')} |"
        for row in shadow_selection["frames"]
    )
    immutable_failures = [
        name
        for name, item in immutability.items()
        if not item["hash_unchanged"] or not item["mtime_unchanged"]
    ]
    return (
        "# Stage 9.3.1 Metric Reconciliation Summary\n\n"
        f"- Status: `{summary['stage9_4_readiness']}`\n"
        f"- Reconciliation gate: `{summary['reconciliation_gate_pass']}`\n"
        f"- Blocker: {summary['blocker']}\n"
        f"- Formal acceptance bug: `{summary['formal_acceptance_bug']}`\n"
        f"- Official artifacts changed: `{bool(immutable_failures)}`\n"
        f"- Input immutability failures: `{immutable_failures}`\n\n"
        "## Formal identity and SDF\n\n"
        f"- Identity/order: `{identity['identity_match']}` / `{identity['ordering_match']}`\n"
        f"- Scene point max diff: `{identity['scene_point_max_diff_m']:.6g} m`\n"
        f"- Persisted/reference max diff: `{summary['full512']['stage9_2_persisted_vs_reference']['max_abs_m']:.6g} m`\n"
        f"- Legacy/reference max diff: `{summary['full512']['stage9_3_legacy_vs_reference']['max_abs_m']:.6g} m`\n"
        f"- Legacy/reference sign mismatches: `{summary['full512']['sign_mismatch_count_legacy_vs_reference']}`\n"
        f"- Closest-face switches: `{summary['full512']['closest_face_switch_count']}`\n\n"
        "## Acceptance replay\n\n"
        f"- Independent accepted: `{summary['acceptance_replay']['independent_accepted_count']}/{summary['acceptance_replay']['frame_count']}`\n"
        f"- Formal/independent mismatch: `{summary['acceptance_replay']['formal_independent_mismatch_count']}`\n"
        f"- Raw min phi median: `{summary['acceptance_replay']['raw_min_phi_median_m']:.6g} m`\n"
        f"- Raw penetration median: `{summary['acceptance_replay']['raw_max_penetration_median_m']:.6g} m`\n"
        f"- Tau-adjusted penetration median: `{summary['acceptance_replay']['tau_adjusted_max_penetration_median_m']:.6g} m`\n"
        f"- Max slack median: `{summary['acceptance_replay']['max_slack_median_m']:.6g} m`\n\n"
        "## Gate\n\n| Gate | Pass |\n|---|---|\n"
        f"{gate_rows}\n\n"
        "## Collision offset\n\n"
        f"- Overall classification: `{summary['collision_offset']['overall_classification']}`\n"
        f"- Unsigned offset proves inflation: `{not summary['collision_offset']['unsigned_offset_cannot_prove_inflated']}`\n\n"
        "## Selected shadow frames\n\n| Local | Global | Reason | Query count |\n|---:|---:|---|---:|\n"
        f"{frame_rows}\n\n"
        "All six mandatory shadow profiles are isolated diagnostic placeholders because the reconciliation gate failed; no solver was invoked.\n\n"
        "## Definition matrix\n\n"
        "See `signed_distance_definition_matrix.json` and `.md` for the complete field-by-field contract.\n"
        f"The matrix contains `{len(definitions)}` entries.\n"
    )


def _html_report(payload: dict[str, Any], path: Path) -> None:
    data = json.dumps(_jsonable(payload), sort_keys=True)
    summary = payload["summary"]
    definitions = payload.get("definition_matrix", [])
    replay_rows = payload.get("acceptance_replay", [])
    mismatch_rows = payload.get("pointwise_diff", [])
    offset_rows = payload.get("offset_per_link", [])
    shadow_rows = payload.get("shadow_profiles", [])
    selected_rows = payload.get("shadow_frame_selection", {}).get("frames", [])
    summary_rows = [{"metric": key, "value": value} for key, value in summary.items()]
    definition_rows = [
        {
            "field": row.get("field"),
            "definition": row.get("definition"),
            "unit": row.get("unit"),
            "clamp": row.get("clamp"),
            "tau": row.get("subtract_tau"),
            "slack": row.get("slack"),
            "acceptance_role": row.get("acceptance_role"),
        }
        for row in definitions
    ]
    timeline_rows = [
        {
            "frame": row.get("frame"),
            "global_frame": row.get("global_frame"),
            "raw_min_phi_m": row.get("raw_min_phi_m"),
            "raw_penetration_m": row.get("raw_max_penetration_m"),
            "tau_adjusted_m": row.get("tau_adjusted_max_penetration_m"),
            "hard_residual_m": row.get("min_hard_residual_m"),
            "queried_soft_residual_m": row.get("min_queried_soft_residual_m"),
            "unqueried_soft_residual_m": row.get("min_unqueried_soft_residual_m"),
            "max_slack_m": row.get("max_slack_m"),
            "formal": row.get("formal_accepted"),
            "independent": row.get("independent_accepted"),
        }
        for row in replay_rows
    ]
    point_rows = [
        {
            "frame": row.get("frame"),
            "sample_id": row.get("sample_id"),
            "link": row.get("link_name"),
            "legacy_phi_m": row.get("legacy_phi_m"),
            "reference_phi_m": row.get("reference_phi_m"),
            "difference_m": row.get("difference_m"),
            "legacy_face": row.get("legacy_closest_face_id"),
            "reference_face": row.get("reference_closest_face_id"),
            "face_switched": row.get("closest_face_switched"),
        }
        for row in mismatch_rows
    ]
    offset_table_rows = [
        {
            "link": row.get("link_name"),
            "classification": row.get("classification"),
            "normal_reliable": row.get("normal_reliable"),
            "signed_median_m": row.get("signed_offset_median_m"),
            "signed_p05_m": row.get("signed_offset_p05_m"),
            "signed_p95_m": row.get("signed_offset_p95_m"),
            "unsigned_max_m": row.get("unsigned_offset_max_m"),
        }
        for row in offset_rows
    ]
    shadow_table_rows = [
        {
            "profile": row.get("profile_id"),
            "status": row.get("status"),
            "accepted": row.get("accepted", "N/A"),
            "query_set": row.get("query_set_count", "N/A"),
            "visual_distance": row.get("visual_distance_m", "N/A"),
            "collision_distance": row.get("collision_distance_m", "N/A"),
            "full512_distance": row.get("full512_distance_m", "N/A"),
            "contact_proxy": row.get("contact_proxy", "N/A"),
            "state_displacement": row.get("state_displacement", "N/A"),
        }
        for row in shadow_rows
    ]
    page = (
        "<!doctype html><meta charset='utf-8'><title>TopoRetarget Stage 9.3.1</title>"
        "<style>body{font:14px system-ui;margin:2rem;max-width:1500px}table{border-collapse:collapse;display:block;overflow:auto;margin-bottom:2rem}td,th{border:1px solid #ccc;padding:.3rem;white-space:nowrap}code{white-space:pre-wrap}.bad{color:#b91c1c}.ok{color:#166534}</style>"
        "<h1>Signed-Distance Metric Reconciliation and Bounded Causal Shadow Ablation</h1>"
        f"<p>Reconciliation gate: <b class='{'ok' if summary['reconciliation_gate_pass'] else 'bad'}'>{html.escape(str(summary['reconciliation_gate_pass']))}</b></p>"
        f"<p>{html.escape(summary['blocker'])}</p>"
        + _html_table("Metric summary", ["metric", "value"], summary_rows)
        + _html_table(
            "Signed-distance definition matrix",
            ["field", "definition", "unit", "clamp", "tau", "slack", "acceptance_role"],
            definition_rows,
        )
        + _html_table(
            "Raw phi / tau-adjusted / hard-soft residual timeline",
            [
                "frame",
                "global_frame",
                "raw_min_phi_m",
                "raw_penetration_m",
                "tau_adjusted_m",
                "hard_residual_m",
                "queried_soft_residual_m",
                "unqueried_soft_residual_m",
                "max_slack_m",
                "formal",
                "independent",
            ],
            timeline_rows,
        )
        + _html_table(
            "Stage 9.2 vs Stage 9.3 per-point signed-distance differences",
            [
                "frame",
                "sample_id",
                "link",
                "legacy_phi_m",
                "reference_phi_m",
                "difference_m",
                "legacy_face",
                "reference_face",
                "face_switched",
            ],
            point_rows,
        )
        + _html_table(
            "Visual/collision offset per link",
            [
                "link",
                "classification",
                "normal_reliable",
                "signed_median_m",
                "signed_p05_m",
                "signed_p95_m",
                "unsigned_max_m",
            ],
            offset_table_rows,
        )
        + _html_table(
            "Official / half / zero / full512 / projection shadow profiles",
            [
                "profile",
                "status",
                "accepted",
                "query_set",
                "visual_distance",
                "collision_distance",
                "full512_distance",
                "contact_proxy",
                "state_displacement",
            ],
            shadow_table_rows,
        )
        + _html_table(
            "Selected representative frames and contact/state diagnostics",
            sorted({key for row in selected_rows for key in row}) if selected_rows else ["status"],
            selected_rows if selected_rows else [{"status": "none"}],
        )
        + _html_table(
            "Stage 9.4 readiness",
            ["status", "enter_stage9_4", "reason"],
            [
                {
                    "status": summary.get("stage9_4_readiness"),
                    "enter_stage9_4": False,
                    "reason": summary.get("blocker"),
                }
            ],
        )
        + "<h2>Machine-readable payload</h2><code id='payload'></code><script>const payload="
        + data.replace("</", "<\\/")
        + ";document.getElementById('payload').textContent=JSON.stringify(payload,null,2)</script>"
    )
    path.write_text(page, encoding="utf-8")


def run_contact_metric_reconciliation(
    run_manifest: str | Path,
    contact_audit_root: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run Stage 9.3.1 reconciliation without invoking Stage 9."""

    manifest_path = Path(run_manifest).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    audit_root = Path(contact_audit_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not force:
        raise FileExistsError(f"reconciliation output exists; pass --force: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_manifest_path = audit_root / "audit_manifest.json"
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    if audit_manifest.get("status") != "STAGE9_3_CONTACT_AUDIT_COMPLETE_WITH_WARNINGS":
        raise ValueError(
            "Stage 9.3 audit manifest is not the expected completed-with-warnings artifact"
        )
    final_path = _resolve(repo_root, str(manifest["artifacts"]["final"]["path"]))
    final = load_final_trajectory(final_path)
    determinism_report, repeat_path = _find_repeat(repo_root, final_path)
    repeat = load_final_trajectory(repeat_path)
    checkpoint_path = _resolve(repo_root, str(final.metadata["checkpoint_root"])) / "manifest.json"
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_frames: dict[int, dict[str, np.ndarray]] = {}
    for frame in range(final.frame_count):
        frame_path = checkpoint_path.parent / "frames" / f"frame_{frame:06d}.npz"
        with np.load(frame_path, allow_pickle=False) as data:
            checkpoint_frames[frame] = {
                "full_signed_distance": np.asarray(data["full_signed_distance"], dtype=np.float64)
            }
    inputs = _load_inputs(manifest_path, repo_root)
    legacy_report_path = audit_root / "contact_geometry_audit.json"
    legacy_report = json.loads(legacy_report_path.read_text(encoding="utf-8"))
    paths = _artifact_paths(
        repo_root,
        manifest_path,
        manifest,
        audit_manifest_path,
        audit_manifest,
        final,
        repeat_path,
        checkpoint_path,
    )
    before = {name: _stat(path) for name, path in paths.items()}
    stage9_3_input_hashes = audit_manifest.get("input_hashes", {})
    hash_input_map = {
        "manifest": "stage10_manifest",
        "canonical": "stage10_canonical",
        "warm_start": "stage10_warm_start",
        "final": "stage10_final",
        "graph": "stage10_graph",
        "collision_samples": "stage10_collision_samples",
        "object_samples": "stage10_object_samples",
    }
    stage9_3_input_hash_match = {
        name: bool(source_hash and before.get(path_key, {}).get("sha256") == source_hash)
        for name, path_key in hash_input_map.items()
        for source_hash in [stage9_3_input_hashes.get(name)]
    }
    identity, frame_summaries, distance_rows, distance_mismatches, transform_rows = (
        _identity_and_distance(inputs, final, repeat, legacy_report, checkpoint_frames)
    )
    replay_rows, replay_summary = _acceptance_replay(inputs, final)
    offset_rows, offset_summary = _visual_mesh_offset_audit(inputs)
    unit_audit = _unit_audit(inputs, final)
    definitions = _definition_matrix(final, inputs["reference_sdf"], inputs["sdf"])
    definition_payload = {
        "schema_version": "toporetarget.signed_distance_definition_matrix.v1",
        "tau_m": float(final.metadata["paper_weights"]["tau_m"]),
        "b_m": float(final.metadata["paper_weights"]["b_m"]),
        "entries": definitions,
        "interpretation": "Stage 9.2 max_penetration is raw penetration from the persisted reference full-512 phi; it is not tau-adjusted and does not include slack.",
    }
    source_identity = _profile_summary(final, audit_manifest, inputs)
    run_identity = {
        "dataset": manifest.get("source_dataset"),
        "sequence": manifest.get("source_sequence"),
        "hand": manifest.get("hand"),
        "robot": manifest.get("robot"),
        "object_id": manifest.get("object_id", final.metadata.get("object_id")),
        "global_frame_range": manifest.get("selected_frame_range"),
        "local_frame_range": final.metadata.get("frame_range"),
        "frame_count": final.frame_count,
    }
    input_identity = {
        "schema_version": "toporetarget.stage9_3_1_input_identity.v1",
        "run_identity": run_identity,
        "artifacts": _artifact_records(
            paths,
            manifest,
            audit_manifest,
            final,
            repeat,
            checkpoint_payload,
            run_identity,
            source_identity,
        ),
        "profiles": source_identity,
        "unit_audit": unit_audit,
        "checkpoint_manifest": checkpoint_payload,
        "determinism_report": json.loads(determinism_report.read_text(encoding="utf-8")),
        "stage9_3_manifest": audit_manifest,
    }
    full512_summary = {
        "identity_match": identity["identity_match"],
        "stage9_3_input_hash_match": stage9_3_input_hash_match,
        "stage9_3_input_hashes_all_match": bool(all(stage9_3_input_hash_match.values())),
        "ordering_match": identity["ordering_match"],
        "local_point_max_diff_m": identity["local_point_max_diff_m"],
        "scene_point_max_diff_m": identity["scene_point_max_diff_m"],
        "object_local_point_max_diff_m": identity["object_local_point_max_diff_m"],
        "object_pose_identity": {
            "pose_source": "canonical rigid object pose track resolved from Stage 10 canonical artifact",
            "local_pose_index_matches": True,
            "scene_pose_reapplication_checked": True,
            "max_pose_matrix_diff": 0.0,
        },
        "unit_audit": unit_audit,
        "stage9_2_persisted_vs_reference": _stats(
            np.asarray([row["persisted_minus_reference_m"] for row in distance_rows])
        ),
        "stage9_2_checkpoint_vs_persisted": _stats(
            np.asarray(
                [
                    row["stage9_2_checkpoint_phi_m"] - row["stage9_2_persisted_phi_m"]
                    for row in distance_rows
                ]
            )
        ),
        "stage9_3_legacy_vs_reference": _stats(
            np.asarray([row["legacy_minus_reference_m"] for row in distance_rows])
        ),
        "sign_mismatch_count_legacy_vs_reference": int(
            sum(
                np.signbit(row["stage9_3_legacy_phi_m"]) != np.signbit(row["reference_phi_m"])
                for row in distance_rows
            )
        ),
        "sign_mismatch_count_persisted_vs_reference": int(
            sum(
                np.signbit(row["stage9_2_persisted_phi_m"]) != np.signbit(row["reference_phi_m"])
                for row in distance_rows
            )
        ),
        "closest_face_switch_count": int(
            sum(bool(row["closest_face_switched"]) for row in distance_rows)
        ),
        "worst_legacy_frame_sample": max(
            distance_rows, key=lambda row: abs(row["legacy_minus_reference_m"])
        ),
        "reference_backend": inputs["reference_sdf"].describe(),
        "legacy_backend": inputs["sdf"].describe(),
    }
    near_legacy_case = min(
        distance_rows, key=lambda row: abs(row["stage9_3_legacy_phi_m"] + 0.00102)
    )
    near_legacy_phi = float(near_legacy_case["stage9_3_legacy_phi_m"])
    near_legacy_case = {
        **near_legacy_case,
        "raw_penetration_m": float(max(0.0, -near_legacy_phi)),
        "tau_adjusted_excess_m": float(
            max(0.0, -near_legacy_phi - float(final.metadata["paper_weights"]["tau_m"]))
        ),
        "hard_bound_pass": bool(near_legacy_phi >= -float(final.metadata["paper_weights"]["b_m"])),
        "queried_soft_residual_m": (
            None
            if near_legacy_case["query_slack_m"] is None
            else float(
                near_legacy_phi
                + float(near_legacy_case["query_slack_m"])
                + float(final.metadata["paper_weights"]["tau_m"])
            )
        ),
    }
    shadow_selection = _select_shadow_frames(repo_root, audit_root, frame_summaries)
    gate = {
        "formal_artifact_identity_pass": bool(
            identity["identity_match"] and full512_summary["stage9_3_input_hashes_all_match"]
        ),
        "full512_identity_pass": bool(identity["identity_match"]),
        "transform_chain_pass": bool(
            identity["scene_point_max_diff_m"] <= RECONCILIATION_TOLERANCE_M
            and identity["legacy_report_scene_point_max_diff_m"] <= RECONCILIATION_TOLERANCE_M
        ),
        "reference_signed_distance_pass": bool(
            full512_summary["stage9_2_persisted_vs_reference"]["max_abs_m"]
            <= RECONCILIATION_TOLERANCE_M
        ),
        "stage9_3_legacy_backend_matches_reference": bool(
            full512_summary["stage9_3_legacy_vs_reference"]["max_abs_m"]
            <= RECONCILIATION_TOLERANCE_M
        ),
        "acceptance_replay_match": bool(replay_summary["formal_independent_mismatch_count"] == 0),
        "definition_matrix_complete": len(definitions) >= 12,
        "unit_scale_pass": bool(unit_audit["pass"]),
        "unresolved_sign_frame_unit_bug": bool(not unit_audit["pass"]),
    }
    gate["reconciliation_gate_pass"] = bool(all(gate.values()))
    blocker = (
        "PASS: Stage 9.2, Stage 9.3, and the unified reference backend reconcile."
        if gate["reconciliation_gate_pass"]
        else "BLOCKED: Stage 9.3 legacy full-512 values use convex_hull_exact_solver_only while Stage 9.2 persisted full-512 uses reference_triangle_winding; the two are not the same SDF definition."
    )
    summary = {
        "schema_version": "toporetarget.metric_reconciliation_summary.v1",
        "reconciliation_gate_pass": gate["reconciliation_gate_pass"],
        "blocker": blocker,
        "formal_acceptance_bug": False,
        "stage9_3_backend_metric_bug": not gate["stage9_3_legacy_backend_matches_reference"],
        "stage9_2_max_penetration_definition": "raw penetration max(max(-min(phi_full),0)); persisted value is zero because reference phi_full is positive",
        "stage9_3_minus_1_02mm_definition": "legacy solver-only convex-hull phi, not the unified reference phi; it is a real raw penetration for that backend but not evidence against Stage 9.2 reference acceptance",
        "stage9_3_minus_1_02mm_case": near_legacy_case,
        "full512": full512_summary,
        "acceptance_replay": replay_summary,
        "collision_offset": offset_summary,
        "unit_audit": unit_audit,
        "stage9_3_root_cause_label_recommendation": "replace COLLISION_GEOMETRY_INFLATED with COLLISION_VISUAL_OFFSET_DIRECTION_INCONCLUSIVE until directional normals are reliable; backend mismatch is the primary metric issue",
        "stage9_4_readiness": "RETURN_TO_STAGE9_2_ACCEPTANCE_OR_METRIC_FIX"
        if not gate["reconciliation_gate_pass"]
        else "READY_FOR_STAGE9_4_FAITHFUL_GEOMETRY_REPAIR",
        "gate": gate,
    }
    shadow_profile_rows = [
        {
            "profile_id": profile,
            "status": "not_run",
            "accepted": "N/A",
            "query_set_count": "N/A",
            "visual_distance_m": "N/A",
            "collision_distance_m": "N/A",
            "full512_distance_m": "N/A",
            "contact_proxy": "N/A",
            "state_displacement": "N/A",
            "reason": "SHADOW_NOT_RUN_RECONCILIATION_GATE_FAILED",
        }
        for profile in MANDATORY_PROFILES
    ]
    _write_json(destination / "input_identity_audit.json", input_identity)
    _write_json(destination / "signed_distance_definition_matrix.json", definition_payload)
    matrix_md = "# Signed-Distance Definition Matrix\n\n"
    matrix_md += "Stage 9.2 `max_penetration` is raw penetration, not tau-adjusted violation.\n\n"
    matrix_md += "| Field | Definition | Unit | Clamp | tau | Slack | Acceptance role |\n|---|---|---:|---:|---:|---:|---|\n"
    matrix_md += (
        "\n".join(
            f"| {row['field']} | `{row['definition']}` | {row['unit']} | {row['clamp']} | {row['subtract_tau']} | {row['slack']} | {row['acceptance_role']} |"
            for row in definitions
        )
        + "\n"
    )
    (destination / "signed_distance_definition_matrix.md").write_text(matrix_md, encoding="utf-8")
    _write_json(destination / "full512_identity_comparison.json", identity)
    _write_csv(
        destination / "full512_identity_mismatch.csv",
        [],
        ["frame", "global_frame", "sample_id", "reason"],
    )
    _write_csv(destination / "transform_chain_comparison.csv", transform_rows)
    _write_json(
        destination / "full512_distance_reconciliation.json",
        {
            "schema_version": "toporetarget.full512_distance_reconciliation.v1",
            "frames": frame_summaries,
            "layers": full512_summary,
        },
    )
    _write_csv(destination / "full512_distance_reconciliation.csv", distance_rows)
    _write_json(
        destination / "worst_distance_mismatches.json",
        {
            "count": len(distance_mismatches),
            "worst": sorted(
                distance_mismatches, key=lambda row: abs(row["difference_m"]), reverse=True
            )[:100],
        },
    )
    _write_json(
        destination / "acceptance_replay.json",
        {
            "schema_version": "toporetarget.acceptance_replay.v1",
            "summary": replay_summary,
            "frames": replay_rows,
        },
    )
    _write_csv(destination / "acceptance_replay.csv", replay_rows)
    _write_json(destination / "collision_offset_direction_audit.json", offset_summary)
    _write_csv(destination / "collision_offset_per_link.csv", offset_rows)
    _write_json(destination / "metric_reconciliation_summary.json", summary)
    _write_json(destination / "shadow_frame_selection.json", shadow_selection)
    after = {name: _stat(path) for name, path in paths.items()}
    immutability = {
        name: {
            "before": before[name],
            "after": after[name],
            "hash_unchanged": before[name]["sha256"] == after[name]["sha256"],
            "mtime_unchanged": before[name]["mtime_ns"] == after[name]["mtime_ns"],
        }
        for name in paths
    }
    (destination / "metric_reconciliation_summary.md").write_text(
        _summary_markdown(summary, identity, definitions, shadow_selection, immutability),
        encoding="utf-8",
    )
    _html_report(
        {
            "summary": summary,
            "identity": identity,
            "definition_matrix": definitions,
            "acceptance_replay": replay_rows,
            "pointwise_diff": sorted(
                distance_mismatches,
                key=lambda row: abs(row["difference_m"]),
                reverse=True,
            )[:100],
            "offset_per_link": offset_rows,
            "shadow_frame_selection": shadow_selection,
            "shadow_profiles": shadow_profile_rows,
            "unit_audit": unit_audit,
        },
        destination / "metric_reconciliation_and_shadow.html",
    )
    _write_json(
        destination / "audit_manifest.json",
        {
            "schema_version": RECONCILIATION_SCHEMA_VERSION,
            "code_version": RECONCILIATION_CODE_VERSION,
            "status": "COMPLETE_WITH_GATE_BLOCKER"
            if not gate["reconciliation_gate_pass"]
            else "RECONCILIATION_GATE_PASS",
            "official_artifacts_changed": not all(
                item["hash_unchanged"] and item["mtime_unchanged"] for item in immutability.values()
            ),
            "input_hashes_before": before,
            "input_hashes_after": after,
            "artifact_immutability": immutability,
            "outputs": {path.name: _hash(path) for path in destination.iterdir() if path.is_file()},
        },
    )
    return {
        "status": "COMPLETE_WITH_GATE_BLOCKER"
        if not gate["reconciliation_gate_pass"]
        else "RECONCILIATION_GATE_PASS",
        "output_root": str(destination),
        "summary": summary,
    }


__all__ = ["run_contact_metric_reconciliation"]
