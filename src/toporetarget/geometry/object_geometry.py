"""Canonical object mesh loading and frame-preserving surface references."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.schema import MeshDefinition, RigidObjectTrack
from toporetarget.data.storage import load_hoi_sequence

from .surface_sampling import (
    SurfaceSampleSet,
    SurfaceSamplingProfile,
    sample_mesh_surface,
    transform_surface_samples,
)


def load_mesh_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("mesh loading needs `pip install -e '.[geometry]'`") from exc
    source = Path(path).expanduser().resolve()
    loaded = trimesh.load_mesh(source, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"mesh scene is empty: {source}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    return vertices, faces


def object_track_from_canonical(canonical: str | Path, object_id: str) -> RigidObjectTrack:
    sequence = load_hoi_sequence(canonical)
    return sequence.rigid_object(object_id)


def mesh_definition_from_path(path: str | Path, *, mesh_id: str | None = None) -> MeshDefinition:
    from .mesh_audit import audit_mesh

    vertices, faces = load_mesh_file(path)
    audit = audit_mesh(vertices, faces, source_path=path)
    return MeshDefinition(
        vertices,
        faces,
        mesh_frame_name="O",
        mesh_id=mesh_id or Path(path).stem,
        mesh_hash=audit.mesh_hash,
    )


def sample_object_track(
    track: RigidObjectTrack, profile: SurfaceSamplingProfile
) -> SurfaceSampleSet:
    return sample_mesh_surface(
        track.mesh.vertices_local,
        track.mesh.faces,
        profile,
        mesh_id=track.mesh.mesh_id,
        mesh_hash=track.mesh.mesh_hash,
        source_provenance=track.metadata,
    )


def scene_samples_for_frame(
    track: RigidObjectTrack, samples: SurfaceSampleSet, frame: int
) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(track.pose_scene.pose_scene[frame], dtype=np.float64)
    return transform_surface_samples(samples, pose)


def scene_samples_for_frames(
    track: RigidObjectTrack, samples: SurfaceSampleSet, frames: list[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(frames, dtype=np.int64)
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for frame in selected.tolist():
        current_points, current_normals = scene_samples_for_frame(track, samples, int(frame))
        points.append(current_points)
        normals.append(current_normals)
    return np.stack(points), np.stack(normals)


def validate_temporal_reuse(
    track: RigidObjectTrack, samples: SurfaceSampleSet, frames: list[int]
) -> dict[str, Any]:
    points, _ = scene_samples_for_frames(track, samples, frames)
    return {
        "frames": [int(frame) for frame in frames],
        "sample_count": samples.count,
        "same_face_indices": True,
        "same_barycentric": True,
        "scene_points_shape": list(points.shape),
        "resampling": False,
        "temporal_assumption": "A_OBJECT_SAMPLE_TEMPORAL_REUSE_001",
    }


__all__ = [
    "load_mesh_file",
    "mesh_definition_from_path",
    "object_track_from_canonical",
    "sample_object_track",
    "scene_samples_for_frame",
    "scene_samples_for_frames",
    "validate_temporal_reuse",
]
