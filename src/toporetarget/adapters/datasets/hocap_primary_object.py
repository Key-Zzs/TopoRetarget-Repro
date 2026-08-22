"""Fail-closed HOCap primary-object authority resolution.

HOCap sequences declare every object pose in ``meta.yaml`` but do not declare
which object a downstream single-object retargeting clip should use.  This
module resolves that missing semantic only from the selected raw clip: MANO
MediaPipe21 keypoints, posed object triangle meshes, and a frozen shared
profile.  It never inspects retargeting, policy, or physical outcomes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.schema import HOISequence


class HOCapPrimaryObjectError(RuntimeError):
    """Raised when a primary-object claim is missing, ambiguous, or drifts."""


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HOCapPrimaryObjectResolverProfileV1:
    """One outcome-independent resolver profile shared by every held-out clip."""

    profile_id: str = "hocap_primary_object_raw_surface_proximity_v1"
    distance_backend: str = "exact_point_to_triangle_bvh"
    keypoint_layout: str = "mediapipe21"
    ranking_quantile: float = 0.05
    maximum_winner_quantile_m: float = 0.040
    minimum_runner_up_margin_m: float = 0.010
    near_surface_threshold_m: float = 0.050
    minimum_near_frames: int = 5
    minimum_consecutive_near_frames: int = 5

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {**value, "profile_sha256": _stable_hash(value)}


def _exact_unsigned_distance(
    vertices: np.ndarray, faces: np.ndarray, points_local: np.ndarray
) -> tuple[np.ndarray, str]:
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    points_local = np.ascontiguousarray(points_local, dtype=np.float64)
    try:
        from toporetarget.geometry.signed_distance.compiled_sdf_cpu import (
            CompiledBVHHandle,
            compiled_available,
        )

        if compiled_available():
            handle = CompiledBVHHandle(vertices, faces)
            return np.asarray(handle.query(points_local)[3]), handle.backend_id
    except (ImportError, OSError, RuntimeError):
        pass
    from toporetarget.geometry.signed_distance.closest_point import ObjectLocalBVH

    tree = ObjectLocalBVH(vertices[faces])
    return np.asarray(tree.query(points_local)[3]), tree.backend_id


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def resolve_hocap_primary_object(
    sequence: HOISequence,
    *,
    profile: HOCapPrimaryObjectResolverProfileV1 | None = None,
) -> dict[str, Any]:
    """Return a raw-only authority receipt; unresolved evidence stays unresolved."""

    selected_profile = profile or HOCapPrimaryObjectResolverProfileV1()
    if sequence.metadata.dataset_name != "hocap":
        raise HOCapPrimaryObjectError("PRIMARY_OBJECT_RESOLVER_DATASET_NOT_HOCAP")
    if (
        len(sequence.hands) != 1
        or selected_profile.keypoint_layout not in sequence.hands[0].keypoint_tracks
    ):
        raise HOCapPrimaryObjectError("PRIMARY_OBJECT_RESOLVER_KEYPOINT_TRACK_INVALID")
    points = np.asarray(
        sequence.hands[0].keypoint_tracks[selected_profile.keypoint_layout].positions_scene,
        dtype=np.float64,
    )
    if points.ndim != 3 or points.shape[2] != 3 or not np.isfinite(points).all():
        raise HOCapPrimaryObjectError("PRIMARY_OBJECT_RESOLVER_KEYPOINT_VALUES_INVALID")
    if len(points) < selected_profile.minimum_near_frames:
        raise HOCapPrimaryObjectError("PRIMARY_OBJECT_RESOLVER_CLIP_TOO_SHORT")
    if not sequence.rigid_objects:
        raise HOCapPrimaryObjectError("PRIMARY_OBJECT_RESOLVER_NO_OBJECTS")

    candidates: list[dict[str, Any]] = []
    backend_ids: set[str] = set()
    for obj in sequence.rigid_objects:
        poses = np.asarray(obj.pose_scene.pose_scene, dtype=np.float64)
        if poses.shape != (len(points), 4, 4) or not np.isfinite(poses).all():
            raise HOCapPrimaryObjectError(f"PRIMARY_OBJECT_RESOLVER_POSE_INVALID:{obj.object_id}")
        rotation = poses[:, :3, :3]
        translation = poses[:, :3, 3]
        # Row-vector form of p_local = R^T (p_scene - t).
        local = np.einsum("fkj,fji->fki", points - translation[:, None, :], rotation).reshape(-1, 3)
        distances, backend_id = _exact_unsigned_distance(
            obj.mesh.vertices_local, obj.mesh.faces, local
        )
        backend_ids.add(backend_id)
        per_keypoint = distances.reshape(len(points), points.shape[1])
        per_frame = np.min(per_keypoint, axis=1)
        near = per_frame < selected_profile.near_surface_threshold_m
        candidates.append(
            {
                "object_id": obj.object_id,
                "mesh_hash": obj.mesh.mesh_hash,
                "vertex_count": int(len(obj.mesh.vertices_local)),
                "face_count": int(len(obj.mesh.faces)),
                "minimum_surface_distance_m": float(np.min(per_frame)),
                "ranking_quantile_surface_distance_m": float(
                    np.quantile(per_frame, selected_profile.ranking_quantile)
                ),
                "median_frame_surface_distance_m": float(np.median(per_frame)),
                "near_surface_frame_count": int(np.count_nonzero(near)),
                "near_surface_frame_fraction": float(np.mean(near)),
                "longest_consecutive_near_surface_frames": _longest_true_run(near),
                "keypoint_near_surface_fraction": float(
                    np.mean(per_keypoint < selected_profile.near_surface_threshold_m)
                ),
            }
        )

    ranked = sorted(
        candidates,
        key=lambda item: (item["ranking_quantile_surface_distance_m"], item["object_id"]),
    )
    winner = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    margin = (
        float(runner["ranking_quantile_surface_distance_m"])
        - float(winner["ranking_quantile_surface_distance_m"])
        if runner is not None
        else float("inf")
    )
    failures: list[str] = []
    if winner["ranking_quantile_surface_distance_m"] > selected_profile.maximum_winner_quantile_m:
        failures.append("PRIMARY_OBJECT_NO_CANDIDATE_CLOSE_ENOUGH")
    if margin < selected_profile.minimum_runner_up_margin_m:
        failures.append("PRIMARY_OBJECT_CANDIDATE_MARGIN_TOO_SMALL")
    if winner["near_surface_frame_count"] < selected_profile.minimum_near_frames:
        failures.append("PRIMARY_OBJECT_NEAR_FRAME_SUPPORT_TOO_SMALL")
    if (
        winner["longest_consecutive_near_surface_frames"]
        < selected_profile.minimum_consecutive_near_frames
    ):
        failures.append("PRIMARY_OBJECT_PERSISTENCE_TOO_SMALL")
    status = "RESOLVED" if not failures else "UNRESOLVED"
    return {
        "schema_version": "HOCapPrimaryObjectResolutionV1",
        "status": status,
        "primary_object_id": winner["object_id"] if status == "RESOLVED" else None,
        "sequence": sequence.metadata.sequence_id,
        "frame_count": int(len(points)),
        "authority_kind": "confirmed_raw_interaction_v1" if status == "RESOLVED" else None,
        "official_primary_annotation_available": False,
        "resolver_profile": selected_profile.as_dict(),
        "distance_backend_ids": sorted(backend_ids),
        "ranking_field": "ranking_quantile_surface_distance_m",
        "winner_runner_up_margin_m": margin,
        "candidate_metrics": ranked,
        "failure_reasons": failures,
        "outcome_inputs_used": False,
    }


def validate_primary_object_authority(authority: Mapping[str, Any]) -> None:
    value = dict(authority)
    expected = value.pop("authority_sha256", None)
    if value.get("schema_version") != "HOCapPrimaryObjectAuthorityV1":
        raise HOCapPrimaryObjectError("PRIMARY_OBJECT_AUTHORITY_SCHEMA_INVALID")
    if not isinstance(expected, str) or _stable_hash(value) != expected:
        raise HOCapPrimaryObjectError("PRIMARY_OBJECT_AUTHORITY_HASH_DRIFT")
    mappings = value.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        raise HOCapPrimaryObjectError("PRIMARY_OBJECT_AUTHORITY_MAPPINGS_MISSING")
    for row in mappings:
        if row.get("status") != "RESOLVED" or not row.get("primary_object_id"):
            raise HOCapPrimaryObjectError("PRIMARY_OBJECT_AUTHORITY_CONTAINS_UNRESOLVED")
        if row["primary_object_id"] not in row.get("available_object_ids", []):
            raise HOCapPrimaryObjectError("PRIMARY_OBJECT_AUTHORITY_OBJECT_NOT_DECLARED")


def load_primary_object_authority(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_primary_object_authority(value)
    return value


def primary_object_from_authority(
    authority: Mapping[str, Any],
    *,
    sequence: str,
    available_object_ids: list[str] | tuple[str, ...] | None = None,
) -> str:
    validate_primary_object_authority(authority)
    matches = [row for row in authority["mappings"] if row.get("sequence") == sequence]
    if len(matches) != 1:
        raise HOCapPrimaryObjectError(
            f"PRIMARY_OBJECT_AUTHORITY_SEQUENCE_CARDINALITY:{sequence}:{len(matches)}"
        )
    row = matches[0]
    primary = str(row["primary_object_id"])
    if available_object_ids is not None:
        available = [str(value) for value in available_object_ids]
        if list(row.get("available_object_ids", [])) != available:
            raise HOCapPrimaryObjectError("PRIMARY_OBJECT_AUTHORITY_OBJECT_SET_DRIFT")
        if primary not in available:
            raise HOCapPrimaryObjectError("PRIMARY_OBJECT_AUTHORITY_OBJECT_NOT_AVAILABLE")
    return primary


__all__ = [
    "HOCapPrimaryObjectError",
    "HOCapPrimaryObjectResolverProfileV1",
    "load_primary_object_authority",
    "primary_object_from_authority",
    "resolve_hocap_primary_object",
    "validate_primary_object_authority",
]
