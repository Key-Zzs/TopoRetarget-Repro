"""Reference signed-distance backend with strict and fallback sign modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.geometry.mesh_audit import MeshAuditReport, audit_mesh
from toporetarget.geometry.se3 import invert_transform, transform_points, transform_vectors

from .base import SignedDistanceBackend, SignedDistanceQueryResult
from .closest_point import (
    TriangleAABBTree,
    TriangleCentroidBoundTree,
    closest_points_on_triangles,
)
from .winding import generalized_winding_number, winding_sign


class SignedDistanceError(RuntimeError):
    """Raised when a requested sign mode is not valid for a mesh."""


class ReferenceSignedDistanceBackend(SignedDistanceBackend):
    backend_id = "reference_triangle_winding"

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        mesh_hash: str | None = None,
        sign_mode: str = "strict",
        surface_tolerance: float = 1e-8,
        winding_threshold: float = 0.5,
        confidence_threshold: float = 0.05,
        query_chunk_size: int = 256,
        face_chunk_size: int = 4096,
        closest_acceleration: str = "tree",
        winding_device: str | None = None,
        closest_device: str | None = None,
        source_path: str | Path | None = None,
    ) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.audit_report: MeshAuditReport = audit_mesh(
            self.vertices, self.faces, source_path=source_path
        )
        if not self.audit_report.valid_face_indices or not self.audit_report.triangular_faces:
            raise ValueError(
                "signed-distance backend requires finite triangular faces with valid indices"
            )
        if self.audit_report.derived_valid_face_count == 0:
            raise ValueError("cannot build signed-distance backend for a mesh without valid faces")
        if sign_mode not in {"strict", "winding", "unsigned_only"}:
            raise ValueError("sign_mode must be strict, winding, or unsigned_only")
        if sign_mode == "strict" and not self._strict_eligible:
            raise SignedDistanceError(
                "strict signed distance requires watertight, orientable, consistently wound, "
                f"non-degenerate mesh; audit={self.audit_report.as_dict()}"
            )
        self.sign_mode = sign_mode
        self.mesh_hash = mesh_hash or self.audit_report.mesh_hash
        self.surface_tolerance = float(surface_tolerance)
        self.winding_threshold = float(winding_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.query_chunk_size = int(query_chunk_size)
        self.face_chunk_size = int(face_chunk_size)
        if closest_acceleration not in {"tree", "vectorized", "centroid_bound"}:
            raise ValueError("closest_acceleration must be tree, vectorized, or centroid_bound")
        requested_winding_device = winding_device
        requested_closest_device = closest_device
        if winding_device is not None and str(winding_device).startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    winding_device = "cpu"
            except ImportError:
                winding_device = None
        if closest_device is not None and str(closest_device).startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    closest_device = None
                    if closest_acceleration == "vectorized":
                        closest_acceleration = "tree"
            except ImportError:
                closest_device = None
                if closest_acceleration == "vectorized":
                    closest_acceleration = "tree"
        if (
            requested_winding_device is not None
            and str(requested_winding_device).startswith("cuda")
            and str(winding_device) == "cpu"
        ):
            query_chunk_size = min(int(query_chunk_size), 256)
            face_chunk_size = min(int(face_chunk_size), 4096)
        self.query_chunk_size = int(query_chunk_size)
        self.face_chunk_size = int(face_chunk_size)
        self.closest_acceleration = closest_acceleration
        self.winding_device = winding_device
        self.closest_device = closest_device
        self.requested_winding_device = requested_winding_device
        self.requested_closest_device = requested_closest_device
        self._face_indices = np.flatnonzero(self._valid_face_mask)
        self._triangles = self.vertices[self.faces[self._valid_face_mask]]
        self._face_normals = np.cross(
            self._triangles[:, 1] - self._triangles[:, 0],
            self._triangles[:, 2] - self._triangles[:, 0],
        )
        self._face_normals /= np.maximum(
            np.linalg.norm(self._face_normals, axis=1, keepdims=True), 1e-15
        )
        # Keep the reference mesh and its exact closest-point acceleration
        # structure alive for the whole run.  This changes only query
        # scheduling; the leaf computation and winding sign remain the
        # reference implementation.
        self._closest_tree: TriangleAABBTree | TriangleCentroidBoundTree | None
        if closest_acceleration == "tree":
            self._closest_tree = TriangleAABBTree(self._triangles)
        elif closest_acceleration == "centroid_bound":
            self._closest_tree = TriangleCentroidBoundTree(self._triangles)
        else:
            self._closest_tree = None
        if self.audit_report.signed_volume is not None and self.audit_report.signed_volume < 0:
            self._face_normals *= -1.0

    @property
    def _valid_face_mask(self) -> np.ndarray:
        tri = self.vertices[self.faces]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        return area > self.audit_report.degenerate_area_threshold

    @property
    def _strict_eligible(self) -> bool:
        return bool(
            self.audit_report.watertight
            and self.audit_report.non_manifold_edge_count == 0
            and self.audit_report.winding_consistent is not False
            and self.audit_report.orientable is not False
            and self.audit_report.near_zero_area_faces == 0
        )

    def audit(self) -> dict[str, Any]:
        return self.audit_report.as_dict()

    def describe(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "sign_mode": self.sign_mode,
            "mesh_hash": self.mesh_hash,
            "sign_convention": "positive_outside",
            "surface_tolerance": self.surface_tolerance,
            "winding_threshold": self.winding_threshold,
            "confidence_threshold": self.confidence_threshold,
            "query_chunk_size": self.query_chunk_size,
            "face_chunk_size": self.face_chunk_size,
            "acceleration": {"rtree": False, "pyembree": False, "reference_fallback": True},
            "triangle_aabb": True,
            "closest_acceleration": self.closest_acceleration,
            "winding_device": self.winding_device or "cpu_numpy",
            "closest_device": self.closest_device or "cpu_numpy",
            "requested_winding_device": self.requested_winding_device or "cpu_numpy",
            "requested_closest_device": self.requested_closest_device or "cpu_numpy",
            "convex_halfspace_sign": False,
            "audit_sign_reliability": self.audit_report.sign_reliability,
        }

    def query_local(self, points_local: np.ndarray) -> SignedDistanceQueryResult:
        points = np.asarray(points_local, dtype=np.float64)
        original_shape = points.shape[:-1]
        points = points.reshape(-1, 3)
        closest, local_faces, barycentric, unsigned = closest_points_on_triangles(
            points,
            self._triangles,
            query_chunk_size=self.query_chunk_size,
            face_chunk_size=self.face_chunk_size,
            tree=None if self.closest_device is not None else self._closest_tree,
            device=self.closest_device,
        )
        original_faces = self._face_indices[local_faces]
        normals = self._face_normals[local_faces].copy()
        non_smooth = np.any(barycentric < 1e-7, axis=1)
        winding_value: np.ndarray | None = None
        if self.sign_mode == "unsigned_only":
            inside = None
            sign_valid = np.zeros(len(points), dtype=bool)
            confidence = np.zeros(len(points), dtype=np.float64)
            signed = np.full(len(points), np.nan, dtype=np.float64)
            method = "unsigned_only"
        else:
            winding_value = generalized_winding_number(
                points,
                self._triangles,
                query_chunk_size=self.query_chunk_size,
                face_chunk_size=self.face_chunk_size,
                device=self.winding_device,
            )
            inside, confidence, ambiguous, _ = winding_sign(
                winding_value,
                threshold=self.winding_threshold,
                confidence_threshold=self.confidence_threshold,
            )
            sign_valid = ~ambiguous
            if self.sign_mode == "strict":
                sign_valid[:] = True
            elif not self.audit_report.watertight or self.audit_report.non_manifold_edge_count:
                # Generalized winding remains useful as a diagnostic on open
                # surfaces, but it is never promoted to a valid penetration
                # sign without a closed mesh audit.
                confidence = np.minimum(confidence, 0.25)
                sign_valid[:] = False
            signed = np.where(inside, -unsigned, unsigned)
            method = (
                "strict_winding_number"
                if self.sign_mode == "strict"
                else "generalized_winding_number"
            )
            # The face winding can be reversed in the source mesh.  Distance
            # sign is geometric; the derived normal is explicitly outward.
            direction = np.where(inside[:, None], closest - points, points - closest)
            direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
            fallback = direction / np.maximum(direction_norm, 1e-15)
            use_fallback = np.linalg.norm(normals, axis=1) <= 1e-15
            normals[use_fallback] = fallback[use_fallback]
            if np.any(inside):
                normals[inside] = np.where(
                    (np.sum(normals[inside] * (closest[inside] - points[inside]), axis=1) < 0)[
                        :, None
                    ],
                    -normals[inside],
                    normals[inside],
                )
        on_surface = unsigned <= self.surface_tolerance
        if inside is not None:
            inside = np.where(on_surface, False, inside)
        result = SignedDistanceQueryResult(
            signed_distance=signed.reshape(original_shape),
            unsigned_distance=unsigned.reshape(original_shape),
            closest_points=closest.reshape((*original_shape, 3)),
            closest_face_indices=original_faces.reshape(original_shape),
            closest_barycentric=barycentric.reshape((*original_shape, 3)),
            surface_normals=normals.reshape((*original_shape, 3)),
            inside=None if inside is None else inside.reshape(original_shape),
            on_surface=on_surface.reshape(original_shape),
            valid=np.ones(original_shape, dtype=bool),
            sign_valid=sign_valid.reshape(original_shape),
            sign_confidence=confidence.reshape(original_shape),
            sign_method=method,
            backend_id=self.backend_id,
            mesh_hash=self.mesh_hash,
            winding_value=None if winding_value is None else winding_value.reshape(original_shape),
            non_smooth=non_smooth.reshape(original_shape),
            gradient_valid=(sign_valid & ~non_smooth).reshape(original_shape),
        )
        return result

    def query_scene(
        self, points_scene: np.ndarray, object_pose_scene: np.ndarray
    ) -> SignedDistanceQueryResult:
        points = np.asarray(points_scene, dtype=np.float64)
        local = transform_points(invert_transform(object_pose_scene), points)
        result = self.query_local(local)
        result.closest_points = transform_points(object_pose_scene, result.closest_points)
        result.surface_normals = transform_vectors(object_pose_scene, result.surface_normals)
        result.surface_normals /= np.maximum(
            np.linalg.norm(result.surface_normals, axis=-1, keepdims=True), 1e-15
        )
        return result


def build_signed_distance_backend(
    vertices: np.ndarray, faces: np.ndarray, **kwargs: Any
) -> ReferenceSignedDistanceBackend:
    return ReferenceSignedDistanceBackend(vertices, faces, **kwargs)


__all__ = ["ReferenceSignedDistanceBackend", "SignedDistanceError", "build_signed_distance_backend"]
