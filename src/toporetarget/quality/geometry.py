"""Geometry provenance and source-contact audits for the fixed quality set."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.adapters.base import FrameRange
from toporetarget.data.contacts.grab import load_grab_contact_mapping
from toporetarget.data.readers.grab import load_grab_auxiliary, load_ply_mesh
from toporetarget.data.storage import load_hoi_sequence
from toporetarget.geometry.mesh_audit import audit_mesh
from toporetarget.geometry.signed_distance.derived_proxy import (
    ObjectSDFGeometryPolicy,
    build_hybrid_signed_distance_backend,
    write_json,
)

from .schema import QUALITY_SCHEMA_VERSION, ClipSpec, file_hash

FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


def _finger_name(label_name: str) -> str | None:
    lowered = label_name.lower()
    for finger in FINGER_TIPS:
        if finger in lowered:
            return finger
    return None


def _mapping_names() -> dict[int, str]:
    return {
        int(item["id"]): str(item["name"]) for item in load_grab_contact_mapping().table().values()
    }


def _geometry_root(experiment_root: str | Path) -> Path:
    root = Path(experiment_root).expanduser().resolve() / "geometry"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_proxy_manifest(path: Path) -> dict[str, Any] | None:
    manifest = path / "proxy_manifest.json"
    if not manifest.is_file():
        return None
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def build_geometry_artifacts(
    selection: dict[str, Any],
    experiment_root: str | Path,
    *,
    policy_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build or reuse one auditable proxy cache entry per frozen object."""

    root = _geometry_root(experiment_root)
    policy = ObjectSDFGeometryPolicy.load(policy_path)
    rows: list[dict[str, Any]] = []
    for selected in selection["selected_units"]:
        unit_id = str(selected["unit_id"])
        object_path = Path(str(selected["object_mesh_path"])).expanduser().resolve()
        vertices, faces = load_ply_mesh(object_path)
        source_audit = audit_mesh(
            vertices,
            faces,
            source_path=object_path,
            degenerate_area_threshold=policy.degenerate_area_threshold_m2,
        )
        artifact_root = root / source_audit.mesh_hash
        existing = _read_proxy_manifest(artifact_root)
        reused = bool(
            existing
            and existing.get("source_mesh_hash") == source_audit.mesh_hash
            and existing.get("policy", {}).get("policy_hash") == policy.policy_hash
            and existing.get("schema_version") == policy.schema_version
        )
        if not reused:
            build_hybrid_signed_distance_backend(
                vertices,
                faces,
                policy=policy,
                source_path=object_path,
                artifact_root=artifact_root,
            )
        manifest = _read_proxy_manifest(artifact_root)
        if manifest is None:
            raise RuntimeError(f"geometry artifact missing after build: {artifact_root}")
        source_stat = object_path.stat()
        row = {
            "unit_id": unit_id,
            "object_name": str(selected["object_name"]),
            "object_mesh_path": str(object_path),
            "source_file_hash": file_hash(object_path),
            "source_file_size": int(source_stat.st_size),
            "source_file_mtime_ns": int(source_stat.st_mtime_ns),
            "source_mesh_hash": source_audit.mesh_hash,
            "source_audit": source_audit.as_dict(),
            "proxy_mesh_hash": manifest.get("proxy_mesh_hash"),
            "proxy_candidate_id": manifest.get("candidate_id"),
            "proxy_candidate_method": manifest.get("candidate_method"),
            "proxy_audit": manifest.get("proxy_audit"),
            "surface_deviation": manifest.get("surface_deviation"),
            "patch_area_m2": manifest.get("patch_area_m2"),
            "patch_area_ratio": manifest.get("patch_area_ratio"),
            "cache_signature": manifest.get("cache_signature"),
            "policy_hash": policy.policy_hash,
            "profile_id": policy.profile_id,
            "artifact_root": str(artifact_root),
            "reused_existing_geometry": reused,
            "raw_asset_modified": False,
            "proxy_used_for_sign_only": True,
        }
        write_json(row, artifact_root / "geometry_index_row.json")
        rows.append(row)
    write_json(
        {
            "schema_version": QUALITY_SCHEMA_VERSION,
            "geometry_schema_version": policy.schema_version,
            "profile_id": policy.profile_id,
            "policy_hash": policy.policy_hash,
            "raw_asset_modified": False,
            "proxy_used_for_sign_only": True,
            "rows": rows,
        },
        root / "geometry_manifest.json",
    )
    write_json(rows, root / "source_mesh_audit.json")
    fields = (
        "unit_id",
        "object_name",
        "object_mesh_path",
        "source_file_hash",
        "source_file_size",
        "source_file_mtime_ns",
        "source_mesh_hash",
        "proxy_mesh_hash",
        "proxy_candidate_id",
        "proxy_candidate_method",
        "cache_signature",
        "policy_hash",
        "artifact_root",
        "raw_asset_modified",
        "proxy_used_for_sign_only",
    )
    with (root / "source_mesh_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "geometry_schema_version": policy.schema_version,
        "profile_id": policy.profile_id,
        "policy_hash": policy.policy_hash,
        "rows": rows,
        "root": str(root),
    }


def audit_source_contact_boundary(
    *,
    geometry_manifest: dict[str, Any],
    canonical_path: str | Path,
    source_path: str | Path,
    clip: ClipSpec,
) -> dict[str, Any]:
    """Audit semantic source contact tips against the original boundary."""

    row = next(item for item in geometry_manifest["rows"] if item["unit_id"] == clip.unit_id)
    artifact_root = Path(str(row["artifact_root"]))
    manifest = _read_proxy_manifest(artifact_root)
    if manifest is None:
        raise RuntimeError(f"proxy manifest missing: {artifact_root}")
    sequence = load_hoi_sequence(canonical_path)
    try:
        object_track = sequence.primary_rigid_object()
    except (KeyError, ValueError):
        raise RuntimeError("canonical sequence has no explicit primary object") from None
    backend, geometry = build_hybrid_signed_distance_backend(
        object_track.mesh.vertices_local,
        object_track.mesh.faces,
        source_path=row["object_mesh_path"],
        artifact_root=None,
    )
    hand = next(item for item in sequence.hands if item.side == clip.hand)
    keypoints = np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64)
    poses = np.asarray(object_track.pose_scene.pose_scene, dtype=np.float64)
    raw = load_grab_auxiliary(
        source_path,
        frame_range=FrameRange(clip.start_frame, clip.end_frame),
        include_table=False,
        contact_mode="semantic",
    )
    labels = np.asarray(raw["contact"]["object"], dtype=np.int64)
    names = _mapping_names()
    frames: list[dict[str, Any]] = []
    near_count = 0
    patch_count = 0
    active_sample_count = 0
    for index in range(min(len(labels), len(keypoints), len(poses))):
        active: dict[str, bool] = {finger: False for finger in FINGER_TIPS}
        for label in np.unique(labels[index]):
            finger = _finger_name(names.get(int(label), ""))
            if finger is not None and int(label) != 0:
                active[finger] = True
        sample_ids = [FINGER_TIPS[finger] for finger, value in active.items() if value]
        query = backend.query_scene(keypoints[index, sample_ids], poses[index])
        frame_near = int(
            np.count_nonzero(query.near_original_boundary)
            if query.near_original_boundary is not None
            else 0
        )
        frame_patch = int(
            np.count_nonzero(query.proxy_closest_is_synthetic_patch)
            if query.proxy_closest_is_synthetic_patch is not None
            else 0
        )
        near_count += frame_near
        patch_count += frame_patch
        active_sample_count += len(sample_ids)
        frames.append(
            {
                "local_frame": index,
                "global_frame": clip.start_frame + index,
                "active_source_contact_regions": [
                    finger for finger, value in active.items() if value
                ],
                "active_sample_count": len(sample_ids),
                "near_original_boundary_count": frame_near,
                "proxy_patch_count": frame_patch,
                "boundary_distance_m": None
                if query.original_boundary_distance is None
                else np.asarray(query.original_boundary_distance).tolist(),
            }
        )
    report = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "geometry_schema_version": manifest.get("schema_version"),
        "unit_id": clip.unit_id,
        "source_mesh_hash": geometry.source_mesh_hash,
        "proxy_mesh_hash": geometry.proxy_mesh_hash,
        "boundary_exclusion_radius_m": backend.boundary_exclusion_radius_m,
        "active_source_contact_sample_count": active_sample_count,
        "source_contact_near_boundary_count": near_count,
        "source_contact_proxy_patch_count": patch_count,
        "source_active_contact_near_boundary_count": near_count,
        "conflict": near_count > 0 or patch_count > 0,
        "status": "SIGN_PROXY_CONTACT_REGION_CONFLICT"
        if near_count > 0 or patch_count > 0
        else "pass",
        "frames": frames,
    }
    write_json(report, artifact_root / "source_contact_boundary_report.json")
    return report


__all__ = ["audit_source_contact_boundary", "build_geometry_artifacts"]
