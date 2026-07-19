"""Small, explicit SE(3) helpers using column vectors and right-handed frames."""

from __future__ import annotations

import numpy as np


def _as_transform(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape[-2:] != (4, 4):
        raise ValueError(f"transform must end in (4,4), got {value.shape}")
    return value


def validate_transform(transform: np.ndarray, *, atol: float = 1e-6) -> bool:
    value = _as_transform(transform)
    if not np.all(np.isfinite(value)):
        raise ValueError("transform contains NaN or Inf")
    if not np.allclose(value[..., 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol):
        raise ValueError("transform last row must be [0,0,0,1]")
    rotation = value[..., :3, :3]
    if not np.allclose(np.matmul(np.swapaxes(rotation, -1, -2), rotation), np.eye(3), atol=atol):
        raise ValueError("rotation is not orthonormal")
    if not np.allclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise ValueError("rotation determinant must be +1")
    return True


def invert_transform(transform: np.ndarray) -> np.ndarray:
    value = _as_transform(transform)
    rotation = value[..., :3, :3]
    translation = value[..., :3, 3]
    result = np.zeros_like(value)
    result[..., :3, :3] = np.swapaxes(rotation, -1, -2)
    result[..., :3, 3] = -np.einsum("...ij,...j->...i", result[..., :3, :3], translation)
    result[..., 3, 3] = 1.0
    return result


def compose_transform(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.matmul(_as_transform(first), _as_transform(second))


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    value = _as_transform(transform)
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.shape[-1] != 3:
        raise ValueError(f"points must end in 3, got {point_array.shape}")
    rotation = value[..., :3, :3]
    translation = value[..., :3, 3]
    return np.einsum("...ij,...nj->...ni", rotation, point_array) + translation[..., None, :]


def transform_vectors(transform: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    value = _as_transform(transform)
    vector_array = np.asarray(vectors, dtype=np.float64)
    if vector_array.shape[-1] != 3:
        raise ValueError(f"vectors must end in 3, got {vector_array.shape}")
    return np.einsum("...ij,...nj->...ni", value[..., :3, :3], vector_array)


def relative_transform(parent_to_scene: np.ndarray, child_to_scene: np.ndarray) -> np.ndarray:
    """Return ``T^parent_child = (T^scene_parent)^-1 T^scene_child``."""

    return compose_transform(invert_transform(parent_to_scene), child_to_scene)


def scene_to_wrist(wrist_pose_scene: np.ndarray, points_scene: np.ndarray) -> np.ndarray:
    return transform_points(invert_transform(wrist_pose_scene), points_scene)


def wrist_to_scene(wrist_pose_scene: np.ndarray, points_wrist: np.ndarray) -> np.ndarray:
    return transform_points(wrist_pose_scene, points_wrist)


def scene_to_object(object_pose_scene: np.ndarray, points_scene: np.ndarray) -> np.ndarray:
    return transform_points(invert_transform(object_pose_scene), points_scene)


def object_to_scene(object_pose_scene: np.ndarray, points_object: np.ndarray) -> np.ndarray:
    return transform_points(object_pose_scene, points_object)


def rotation_geodesic_error(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the angle in radians between two rotation matrices."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    relative = np.matmul(np.swapaxes(a[..., :3, :3], -1, -2), b[..., :3, :3])
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def pose_translation_error(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(first)[..., :3, 3] - np.asarray(second)[..., :3, 3], axis=-1)


def pose_rotation_error(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return rotation_geodesic_error(first, second)


__all__ = [
    "compose_transform",
    "invert_transform",
    "object_to_scene",
    "pose_rotation_error",
    "pose_translation_error",
    "relative_transform",
    "rotation_geodesic_error",
    "scene_to_object",
    "scene_to_wrist",
    "transform_points",
    "transform_vectors",
    "validate_transform",
    "wrist_to_scene",
]
