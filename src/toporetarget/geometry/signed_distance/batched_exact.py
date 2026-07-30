"""Persistent, original-mesh exact signed-distance contexts.

This module deliberately reuses the strict triangle/winding mathematics of
``ReferenceSignedDistanceBackend``.  It changes construction and query
scheduling only: a mesh-local proximity context is built once, retained for a
run, and used for whole point batches in the original object coordinate frame.
It is therefore suitable for a solver inner loop without turning a grid,
proxy, repaired mesh, or convexification into geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from toporetarget.geometry.mesh_audit import audit_mesh
from toporetarget.geometry.se3 import invert_transform, transform_points, transform_vectors

from .base import SignedDistanceBackend, SignedDistanceQueryResult
from .reference import ReferenceSignedDistanceBackend


@dataclass
class ObjectLocalProximityContext:
    """One immutable original-mesh BVH/winding context with query counters."""

    vertices: np.ndarray
    faces: np.ndarray
    query_chunk_size: int = 1024
    face_chunk_size: int = 4096
    winding_device: str | None = "cpu"
    _reference: ReferenceSignedDistanceBackend = field(init=False, repr=False)
    mesh_hash: str = field(init=False)
    mesh_load_count: int = field(default=1, init=False)
    bvh_build_count: int = field(default=1, init=False)
    proximity_context_build_count: int = field(default=1, init=False)
    query_batch_count: int = field(default=0, init=False)
    query_point_count: int = field(default=0, init=False)
    closest_query_time: float = field(default=0.0, init=False)
    sign_query_time: float = field(default=0.0, init=False)
    transform_time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        self.mesh_hash = str(audit_mesh(self.vertices, self.faces).mesh_hash)
        # The reference backend owns an exact triangle AABB tree and the
        # generalized-winding precomputed triangle representation.  Keeping
        # this instance alive is the persistence contract.
        self._reference = ReferenceSignedDistanceBackend(
            self.vertices,
            self.faces,
            mesh_hash=self.mesh_hash,
            sign_mode="strict",
            query_chunk_size=self.query_chunk_size,
            face_chunk_size=self.face_chunk_size,
            closest_acceleration="tree",
            winding_device=self.winding_device,
        )

    def query_local(self, points_local: np.ndarray) -> SignedDistanceQueryResult:
        points = np.asarray(points_local, dtype=np.float64)
        started = perf_counter()
        result = self._reference.query_local(points)
        elapsed = perf_counter() - started
        # The current reference implementation performs exact closest-point
        # and strict winding as a single vectorized query.  Attribute the full
        # observed duration conservatively to both logical stages rather than
        # inventing a split that is not observable at this boundary.
        self.closest_query_time += elapsed
        self.sign_query_time += elapsed
        self.query_batch_count += 1
        self.query_point_count += int(points.reshape(-1, 3).shape[0])
        return result

    def counters(self) -> dict[str, int | float]:
        return {
            "mesh_load_count": self.mesh_load_count,
            "bvh_build_count": self.bvh_build_count,
            "proximity_context_build_count": self.proximity_context_build_count,
            "query_batch_count": self.query_batch_count,
            "query_point_count": self.query_point_count,
            "closest_query_time": self.closest_query_time,
            "sign_query_time": self.sign_query_time,
            "transform_time": self.transform_time,
        }


class BatchedOriginalMeshProximityBackend(SignedDistanceBackend):
    """Reference-faithful original-mesh backend backed by one persistent context."""

    backend_id = "original_mesh_batched_exact_bvh_v1"

    def __init__(self, context: ObjectLocalProximityContext) -> None:
        self.context = context
        self.mesh_hash = context.mesh_hash

    @classmethod
    def build(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        query_chunk_size: int = 1024,
        face_chunk_size: int = 4096,
        winding_device: str | None = "cpu",
    ) -> BatchedOriginalMeshProximityBackend:
        return cls(
            ObjectLocalProximityContext(
                vertices,
                faces,
                query_chunk_size=query_chunk_size,
                face_chunk_size=face_chunk_size,
                winding_device=winding_device,
            )
        )

    def query_local(self, points_local: np.ndarray) -> SignedDistanceQueryResult:
        result = self.context.query_local(points_local)
        result.backend_id = self.backend_id
        return result

    def query_scene(
        self, points_scene: np.ndarray, object_pose_scene: np.ndarray
    ) -> SignedDistanceQueryResult:
        started = perf_counter()
        local = transform_points(invert_transform(object_pose_scene), np.asarray(points_scene))
        self.context.transform_time += perf_counter() - started
        result = self.query_local(local)
        started = perf_counter()
        result.closest_points = transform_points(object_pose_scene, result.closest_points)
        result.surface_normals = transform_vectors(object_pose_scene, result.surface_normals)
        result.surface_normals /= np.maximum(
            np.linalg.norm(result.surface_normals, axis=-1, keepdims=True), 1e-15
        )
        self.context.transform_time += perf_counter() - started
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "profile_id": self.backend_id,
            "mesh_hash": self.mesh_hash,
            "sign_convention": "positive_outside",
            "source_geometry": "original_strict_watertight_mesh",
            "distance_semantics": "exact_original_triangle_closest_point",
            "sign_semantics": "strict_reference_generalized_winding",
            "object_local": True,
            "persistent_bvh": True,
            "counters": self.context.counters(),
        }

    def audit(self) -> dict[str, Any]:
        return self.context._reference.audit()


# The long name is intentional: reports use it to distinguish the selected
# solver context from the independent reference audit instance.
ReferenceFaithfulSignedDistanceBackend = BatchedOriginalMeshProximityBackend


__all__ = [
    "BatchedOriginalMeshProximityBackend",
    "ObjectLocalProximityContext",
    "ReferenceFaithfulSignedDistanceBackend",
]
