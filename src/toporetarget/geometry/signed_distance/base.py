"""Common signed-distance query contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SignedDistanceQueryResult:
    signed_distance: np.ndarray
    unsigned_distance: np.ndarray
    closest_points: np.ndarray
    closest_face_indices: np.ndarray
    closest_barycentric: np.ndarray
    surface_normals: np.ndarray
    inside: np.ndarray | None
    on_surface: np.ndarray
    valid: np.ndarray
    sign_valid: np.ndarray
    sign_confidence: np.ndarray
    sign_method: str
    backend_id: str
    mesh_hash: str
    winding_value: np.ndarray | None = None
    non_smooth: np.ndarray | None = None
    gradient_valid: np.ndarray | None = None
    proxy_closest_face_indices: np.ndarray | None = None
    proxy_closest_is_synthetic_patch: np.ndarray | None = None
    original_boundary_distance: np.ndarray | None = None
    near_original_boundary: np.ndarray | None = None
    geometry_metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "signed_distance": self.signed_distance.tolist(),
            "unsigned_distance": self.unsigned_distance.tolist(),
            "closest_points": self.closest_points.tolist(),
            "closest_face_indices": self.closest_face_indices.tolist(),
            "closest_barycentric": self.closest_barycentric.tolist(),
            "surface_normals": self.surface_normals.tolist(),
            "inside": None if self.inside is None else self.inside.tolist(),
            "on_surface": self.on_surface.tolist(),
            "valid": self.valid.tolist(),
            "sign_valid": self.sign_valid.tolist(),
            "sign_confidence": self.sign_confidence.tolist(),
            "sign_method": self.sign_method,
            "backend_id": self.backend_id,
            "mesh_hash": self.mesh_hash,
        }
        if self.winding_value is not None:
            result["winding_value"] = self.winding_value.tolist()
        if self.non_smooth is not None:
            result["non_smooth"] = self.non_smooth.tolist()
        if self.gradient_valid is not None:
            result["gradient_valid"] = self.gradient_valid.tolist()
        if self.proxy_closest_face_indices is not None:
            result["proxy_closest_face_indices"] = self.proxy_closest_face_indices.tolist()
        if self.proxy_closest_is_synthetic_patch is not None:
            result["proxy_closest_is_synthetic_patch"] = (
                self.proxy_closest_is_synthetic_patch.tolist()
            )
        if self.original_boundary_distance is not None:
            result["original_boundary_distance"] = self.original_boundary_distance.tolist()
        if self.near_original_boundary is not None:
            result["near_original_boundary"] = self.near_original_boundary.tolist()
        if self.geometry_metadata is not None:
            result["geometry_metadata"] = self.geometry_metadata
        return result


class SignedDistanceBackend:
    """Interface shared by strict, winding, and unsigned-only query modes."""

    backend_id = "signed-distance"

    def query_local(self, points_local: np.ndarray) -> SignedDistanceQueryResult:
        raise NotImplementedError

    def query_scene(
        self, points_scene: np.ndarray, object_pose_scene: np.ndarray
    ) -> SignedDistanceQueryResult:
        from toporetarget.geometry.se3 import invert_transform, transform_points, transform_vectors

        points = np.asarray(points_scene, dtype=np.float64)
        local_result = self.query_local(
            transform_points(invert_transform(object_pose_scene), points)
        )
        rotation = np.asarray(object_pose_scene, dtype=np.float64)[..., :3, :3]
        local_result.closest_points = transform_points(
            object_pose_scene, local_result.closest_points
        )
        local_result.surface_normals = transform_vectors(
            object_pose_scene, local_result.surface_normals
        )
        local_result.surface_normals = local_result.surface_normals / np.maximum(
            np.linalg.norm(local_result.surface_normals, axis=-1, keepdims=True), 1e-15
        )
        _ = rotation
        return local_result

    def audit(self) -> dict[str, Any]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        raise NotImplementedError


def local_linearization(result: SignedDistanceQueryResult) -> dict[str, np.ndarray]:
    """Return the local first-order signed-distance data, without q-space Jacobians."""

    gradient_valid = result.sign_valid.copy()
    if result.gradient_valid is not None:
        gradient_valid &= result.gradient_valid
    return {
        "phi": result.signed_distance.copy(),
        "closest_point": result.closest_points.copy(),
        "normal": result.surface_normals.copy(),
        "gradient_valid": gradient_valid,
        "non_smooth": np.zeros_like(gradient_valid, dtype=bool)
        if result.non_smooth is None
        else result.non_smooth.copy(),
    }


__all__ = ["SignedDistanceBackend", "SignedDistanceQueryResult", "local_linearization"]
