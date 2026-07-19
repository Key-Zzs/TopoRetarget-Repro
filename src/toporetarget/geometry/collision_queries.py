"""Low-level robot-surface versus object-SDF probes; no optimization is performed."""

from __future__ import annotations

from typing import Any

import numpy as np

from .robot_surface import RobotSurfaceSampleSet
from .signed_distance.base import SignedDistanceBackend


def query_robot_surface_against_object(
    robot_surface_points_scene: np.ndarray,
    object_sdf: SignedDistanceBackend,
    object_pose_scene: np.ndarray,
    *,
    link_names: np.ndarray | None = None,
    geometry_ids: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    result = object_sdf.query_scene(
        np.asarray(robot_surface_points_scene, dtype=np.float64), object_pose_scene
    )
    penetration = np.where(result.sign_valid, np.maximum(-result.signed_distance, 0.0), np.nan)
    payload: dict[str, Any] = {
        "signed_distance": result.signed_distance,
        "unsigned_distance": result.unsigned_distance,
        "closest_points_scene": result.closest_points,
        "surface_normals_scene": result.surface_normals,
        "inside": result.inside,
        "sign_confidence": result.sign_confidence,
        "sign_valid": result.sign_valid,
        "valid": result.valid,
        "penetration_depth": penetration,
        "backend_id": result.backend_id,
        "sign_method": result.sign_method,
        "mesh_hash": result.mesh_hash,
        "sample_ids": np.arange(len(result.signed_distance), dtype=np.int64)
        if sample_ids is None
        else sample_ids,
        "link_names": np.asarray(["unknown"] * len(result.signed_distance))
        if link_names is None
        else link_names,
        "geometry_ids": np.asarray(["unknown"] * len(result.signed_distance))
        if geometry_ids is None
        else geometry_ids,
        "final_query_set": False,
        "optimization": False,
    }
    return payload


def probe_robot_surface(
    samples: RobotSurfaceSampleSet,
    object_sdf: SignedDistanceBackend,
    object_pose_scene: np.ndarray,
) -> dict[str, Any]:
    return query_robot_surface_against_object(
        samples.points_scene,
        object_sdf,
        object_pose_scene,
        link_names=samples.link_names,
        geometry_ids=samples.geometry_ids,
        sample_ids=samples.sample_ids,
    )


def json_ready_probe(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in payload.items()
    }


__all__ = ["json_ready_probe", "probe_robot_surface", "query_robot_surface_against_object"]
