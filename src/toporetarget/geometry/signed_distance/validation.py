"""Small analytic validation helpers for Stage 6 reports and tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import SignedDistanceQueryResult
from .reference import ReferenceSignedDistanceBackend, SignedDistanceError


def make_synthetic_mesh(shape: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("synthetic geometry needs trimesh") from exc
    if shape == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    elif shape == "cube":
        mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    elif shape == "open_cube":
        mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        mesh.update_faces(np.arange(len(mesh.faces) - 2))
        mesh.remove_unreferenced_vertices()
    else:
        raise ValueError(f"unknown synthetic shape: {shape}")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def validate_synthetic_shape(shape: str, *, sign_mode: str = "strict") -> dict[str, Any]:
    vertices, faces = make_synthetic_mesh(shape)
    try:
        backend = ReferenceSignedDistanceBackend(vertices, faces, sign_mode=sign_mode)
    except SignedDistanceError as exc:
        return {
            "status": "expected_failure",
            "shape": shape,
            "sign_mode": sign_mode,
            "error": str(exc),
        }
    points = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    result = backend.query_local(points)
    point_to_triangle_residual = float(
        np.max(
            np.abs(
                np.linalg.norm(points - result.closest_points, axis=1) - result.unsigned_distance
            )
        )
    )
    return {
        "status": "pass",
        "shape": shape,
        "sign_mode": sign_mode,
        "backend": backend.describe(),
        "point_to_triangle_residual_max_error": point_to_triangle_residual,
        "queries": result.as_dict(),
    }


class AnalyticShapeBackend:
    """Analytic oracle used to validate the mesh backend independently."""

    def __init__(self, shape: str, *, radius: float = 1.0, half_extent: float = 1.0) -> None:
        if shape not in {"sphere", "cube"}:
            raise ValueError("analytic shape must be sphere or cube")
        self.shape = shape
        self.radius = float(radius)
        self.half_extent = float(half_extent)
        self.sign_mode = "analytic"
        self.backend_id = f"analytic_{shape}"

    def query_local(self, points: np.ndarray) -> SignedDistanceQueryResult:
        value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self.shape == "sphere":
            norm = np.linalg.norm(value, axis=1)
            safe = np.maximum(norm, 1e-15)
            normal = value / safe[:, None]
            normal[norm <= 1e-15] = np.array([1.0, 0.0, 0.0])
            closest = normal * self.radius
            signed = norm - self.radius
        else:
            half = self.half_extent
            q = np.abs(value) - half
            outside = np.maximum(q, 0.0)
            outside_distance = np.linalg.norm(outside, axis=1)
            inside_distance = np.minimum(np.max(q, axis=1), 0.0)
            signed = outside_distance + inside_distance
            closest = np.clip(value, -half, half)
            inside = np.all(q < 0.0, axis=1)
            if np.any(inside):
                rows = np.flatnonzero(inside)
                axis = np.argmax(q[rows], axis=1)
                closest[rows, axis] = np.where(value[rows, axis] >= 0.0, half, -half)
            normal = value - closest
            normal_length = np.linalg.norm(normal, axis=1)
            on_or_inside = normal_length <= 1e-15
            if np.any(on_or_inside):
                rows = np.flatnonzero(on_or_inside)
                axis = np.argmax(q[rows], axis=1)
                normal[rows] = 0.0
                normal[rows, axis] = np.where(value[rows, axis] >= 0.0, 1.0, -1.0)
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-15)
        inside = signed < 0.0
        return SignedDistanceQueryResult(
            signed_distance=signed,
            unsigned_distance=np.abs(signed),
            closest_points=closest,
            closest_face_indices=np.full(len(value), -1, dtype=np.int64),
            closest_barycentric=np.full((len(value), 3), np.nan),
            surface_normals=normal,
            inside=inside,
            on_surface=np.isclose(signed, 0.0, atol=1e-10),
            valid=np.ones(len(value), dtype=bool),
            sign_valid=np.ones(len(value), dtype=bool),
            sign_confidence=np.ones(len(value), dtype=np.float64),
            sign_method=f"analytic_{self.shape}",
            backend_id=f"analytic_{self.shape}",
            mesh_hash=f"analytic_{self.shape}",
            winding_value=None,
            non_smooth=np.zeros(len(value), dtype=bool),
            gradient_valid=np.ones(len(value), dtype=bool),
        )


def validate_analytic_shape(shape: str) -> dict[str, Any]:
    backend = AnalyticShapeBackend(shape)
    points = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.3, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    result = backend.query_local(points)
    if shape == "sphere":
        expected = np.linalg.norm(points, axis=1) - 1.0
        safe = np.maximum(np.linalg.norm(points, axis=1), 1e-15)
        expected_normals = points / safe[:, None]
        expected_normals[0] = [1.0, 0.0, 0.0]
        expected_closest = expected_normals
    else:
        offset = np.abs(points) - 1.0
        expected = np.linalg.norm(np.maximum(offset, 0.0), axis=1) + np.minimum(
            np.max(offset, axis=1), 0.0
        )
        expected_closest = np.clip(points, -1.0, 1.0)
        inside = np.all(offset < 0.0, axis=1)
        inside_rows = np.flatnonzero(inside)
        inside_axes = np.argmax(offset[inside_rows], axis=1)
        expected_closest[inside_rows, inside_axes] = np.where(
            points[inside_rows, inside_axes] >= 0.0, 1.0, -1.0
        )
        expected_normals = points - expected_closest
        zero_rows = np.flatnonzero(np.linalg.norm(expected_normals, axis=1) <= 1e-15)
        zero_axes = np.argmax(offset[zero_rows], axis=1)
        expected_normals[zero_rows] = 0.0
        expected_normals[zero_rows, zero_axes] = np.where(
            points[zero_rows, zero_axes] >= 0.0, 1.0, -1.0
        )
        expected_normals /= np.maximum(
            np.linalg.norm(expected_normals, axis=1, keepdims=True), 1e-15
        )
    error = float(np.max(np.abs(result.signed_distance - expected)))
    closest_error = float(np.max(np.linalg.norm(result.closest_points - expected_closest, axis=1)))
    normal_error = float(np.max(np.linalg.norm(result.surface_normals - expected_normals, axis=1)))

    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.asarray([0.2, -0.3, 0.5])
    scene_points = points @ rotation.T + translation
    recovered = (scene_points - translation) @ rotation
    scene_result = backend.query_local(recovered)
    equivariance_error = max(
        float(np.max(np.abs(scene_result.signed_distance - result.signed_distance))),
        float(
            np.max(
                np.linalg.norm(
                    scene_result.closest_points @ rotation.T
                    + translation
                    - (result.closest_points @ rotation.T + translation),
                    axis=1,
                )
            )
        ),
    )
    gradient_point = np.asarray([[1.4, 0.2, 0.1]], dtype=np.float64)
    epsilon = 1e-6
    gradient = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        delta = np.zeros((1, 3), dtype=np.float64)
        delta[0, axis] = epsilon
        gradient[axis] = float(
            (
                backend.query_local(gradient_point + delta).signed_distance[0]
                - backend.query_local(gradient_point - delta).signed_distance[0]
            )
            / (2.0 * epsilon)
        )
    gradient_error = float(
        np.linalg.norm(gradient - backend.query_local(gradient_point).surface_normals[0])
    )
    reference_vertices, reference_faces = make_synthetic_mesh(shape)
    reference = ReferenceSignedDistanceBackend(
        reference_vertices, reference_faces, sign_mode="strict"
    )
    classification = reference.query_local(
        np.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64)
    )
    return {
        "status": "pass"
        if max(error, closest_error, normal_error, equivariance_error, gradient_error) <= 1e-9
        else "fail",
        "shape": shape,
        "backend": backend.backend_id,
        "signed_distance_max_error": error,
        "closest_point_max_error": closest_error,
        "normal_max_error": normal_error,
        "rigid_transform_equivariance_max_error": equivariance_error,
        "gradient_finite_difference_max_error": gradient_error,
        "winding_classification": {
            "inside": classification.inside,
            "sign_valid": classification.sign_valid,
            "signed_distance": classification.signed_distance,
            "sign_confidence": classification.sign_confidence,
        },
        "query": result.as_dict(),
    }


__all__ = [
    "AnalyticShapeBackend",
    "make_synthetic_mesh",
    "validate_analytic_shape",
    "validate_synthetic_shape",
]
