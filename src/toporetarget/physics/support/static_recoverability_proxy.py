"""Outcome-independent support proxy for static recoverability studies.

This module deliberately does not call :mod:`resolver`.  Source support stays
unresolved; the result here is a separately labelled controlled study input.
Only canonical collision geometry, its frozen world pose, gravity, and shared
generic support constants may influence the proxy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .types import (
    FinitePlanarSupportProxy,
    NominalSupportMaterialV1,
    StablePreContactDetectionContractV1,
    SupportExtentContractV1,
    SupportType,
)


@dataclass(frozen=True)
class StaticRecoverabilitySupportProxyParameterV1:
    """Frozen, dataset-independent construction constants."""

    schema_version: str = "StaticRecoverabilitySupportProxyParameterV1"
    bottom_patch_tolerance_m: float = (
        StablePreContactDetectionContractV1().support_patch_tolerance_m
    )
    support_extent_margin_m: float = SupportExtentContractV1().support_extent_margin_m
    table_thickness_m: float = SupportExtentContractV1().table_thickness_m
    material: NominalSupportMaterialV1 = NominalSupportMaterialV1()

    def __post_init__(self) -> None:
        if self.bottom_patch_tolerance_m <= 0.0:
            raise ValueError("STATIC_PROXY_BOTTOM_PATCH_TOLERANCE_INVALID")
        if self.support_extent_margin_m <= 0.0 or self.table_thickness_m <= 0.0:
            raise ValueError("STATIC_PROXY_EXTENT_CONSTANT_INVALID")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["material"] = self.material.as_dict()
        return value


def _array(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"STATIC_PROXY_{name.upper()}_INVALID")
    return result


def _support_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic tangent axes and a local-to-world rotation."""

    # Choose the world axis least aligned with the normal.  The tie-break is
    # NumPy's first argmin, so serialization is deterministic.
    basis = np.eye(3, dtype=np.float64)
    seed = basis[int(np.argmin(np.abs(basis @ normal)))]
    tangent_u = seed - float(seed @ normal) * normal
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)
    tangent_v /= np.linalg.norm(tangent_v)
    rotation = np.column_stack((tangent_u, tangent_v, normal))
    return np.column_stack((tangent_u, tangent_v)), rotation


def _quaternion_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper rotation matrix to deterministic-sign WXYZ."""

    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(rotation).as_quat()
    value = np.asarray([xyzw[3], *xyzw[:3]], dtype=np.float64)
    if value[0] < 0.0:
        value *= -1.0
    value /= np.linalg.norm(value)
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def build_static_recoverability_support_proxy(
    collision_vertices_object: Any,
    object_pose_world: Any,
    *,
    gravity_world_mps2: Any = (0.0, 0.0, -9.81),
    parameters: StaticRecoverabilitySupportProxyParameterV1 | None = None,
) -> tuple[FinitePlanarSupportProxy, dict[str, Any]]:
    """Construct one gravity-aligned finite support below a frozen object pose.

    The tangent plane is the supporting plane of the canonical convex collision
    proxy along gravity.  A tolerance-expanded bottom patch is audited so the
    construction does not silently depend on an arbitrary raw-mesh outlier.
    The plane itself remains tangent to the lower envelope: the object pose is
    never moved or pre-relaxed.
    """

    frozen = parameters or StaticRecoverabilitySupportProxyParameterV1()
    vertices = np.asarray(collision_vertices_object, dtype=np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) < 4
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("STATIC_PROXY_COLLISION_VERTICES_INVALID")
    pose = _array(object_pose_world, shape=(4, 4), name="object_pose")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9):
        raise ValueError("STATIC_PROXY_OBJECT_POSE_HOMOGENEOUS_ROW_INVALID")
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2.0e-5):
        raise ValueError("STATIC_PROXY_OBJECT_POSE_ROTATION_INVALID")
    gravity = _array(gravity_world_mps2, shape=(3,), name="gravity")
    magnitude = float(np.linalg.norm(gravity))
    if magnitude <= 0.0:
        raise ValueError("STATIC_PROXY_GRAVITY_ZERO")
    normal = -gravity / magnitude
    tangents, rotation = _support_frame(normal)

    world = vertices @ pose[:3, :3].T + pose[:3, 3]
    heights = world @ normal
    bottom_height = float(np.min(heights))
    patch_mask = heights <= bottom_height + frozen.bottom_patch_tolerance_m
    patch = world[patch_mask]
    if len(patch) == 0:
        raise AssertionError("STATIC_PROXY_BOTTOM_PATCH_EMPTY")

    projected_all = world @ tangents
    low = projected_all.min(axis=0)
    high = projected_all.max(axis=0)
    extent = high - low + 2.0 * frozen.support_extent_margin_m
    if not np.isfinite(extent).all() or np.any(extent <= 0.0):
        raise ValueError("STATIC_PROXY_PROJECTED_EXTENT_INVALID")
    tangent_center = 0.5 * (low + high)
    top_center = tangents @ tangent_center + normal * bottom_height
    quaternion = _quaternion_wxyz(rotation)
    proxy = FinitePlanarSupportProxy(
        table_pose=tuple(float(item) for item in (*top_center, *quaternion)),
        table_extent=(float(extent[0]), float(extent[1])),
        table_thickness=frozen.table_thickness_m,
        plane_normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        plane_offset=bottom_height,
        material=frozen.material,
        representation="static_kinematic_rigid_box",
    )

    projected_patch = patch @ tangents
    patch_extent = projected_patch.max(axis=0) - projected_patch.min(axis=0)
    audit = {
        "schema_version": "ObjectBottomSupportEnvelopeV1",
        "support_type": SupportType.STATIC_RECOVERABILITY_PLANAR_PROXY.value,
        "authority_layer": "STATIC_RECOVERABILITY_STUDY_SUPPORT_PROXY",
        "source_scene_fidelity_claim": False,
        "study_proxy": True,
        "input_vertex_count": int(len(vertices)),
        "bottom_patch_vertex_count": int(np.count_nonzero(patch_mask)),
        "bottom_patch_tolerance_m": frozen.bottom_patch_tolerance_m,
        "bottom_patch_projected_extent_m": patch_extent.tolist(),
        "bottom_height_along_support_normal_m": bottom_height,
        "object_projection_extent_m": (high - low).tolist(),
        "support_center_world_m": top_center.tolist(),
        "support_normal_world": normal.tolist(),
        "support_extent_m": extent.tolist(),
        "gravity_world_mps2": gravity.tolist(),
        "object_pose_changed": False,
        "hand_geometry_used": False,
        "outcome_fields_used": [],
        "parameters": frozen.as_dict(),
    }
    return proxy, audit


__all__ = [
    "StaticRecoverabilitySupportProxyParameterV1",
    "build_static_recoverability_support_proxy",
]
