"""Geometry instance expansion; visual and collision remain separate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .kinematics import forward_kinematics_numpy
from .model import UrdfModel


@dataclass(frozen=True)
class RobotGeometryInstance:
    link_name: str
    geometry_type: str
    kind: str
    geometry: dict[str, Any]
    local_transform: np.ndarray
    link_transform_base: np.ndarray
    world_transform: np.ndarray
    source_file: str | None
    resolved_path: str | None
    source_hash: str | None

    @property
    def transform_base(self) -> np.ndarray:
        return self.link_transform_base @ self.local_transform

    def as_dict(self) -> dict[str, Any]:
        return {
            "link_name": self.link_name,
            "geometry_type": self.geometry_type,
            "kind": self.kind,
            "geometry": self.geometry,
            "local_transform": self.local_transform.tolist(),
            "link_transform_base": self.link_transform_base.tolist(),
            "world_transform": self.world_transform.tolist(),
            "source_file": self.source_file,
            "resolved_path": self.resolved_path,
            "source_hash": self.source_hash,
        }


def _instances(
    model: UrdfModel,
    qpos: Any,
    *,
    base_pose_scene: np.ndarray | None,
    kind: str,
) -> list[RobotGeometryInstance]:
    q = np.asarray(qpos)
    if q.ndim != 1:
        raise ValueError("geometry instances currently require one qpos with shape [N]")
    transforms = forward_kinematics_numpy(model, q)
    base = (
        np.eye(4, dtype=np.float64)
        if base_pose_scene is None
        else np.asarray(base_pose_scene, dtype=np.float64)
    )
    if base.shape != (4, 4):
        raise ValueError(f"base_pose_scene must have shape [4,4], got {base.shape}")
    result: list[RobotGeometryInstance] = []
    for link_name in model.link_names:
        link = model.links[link_name]
        geometries = link.visuals if kind == "visual" else link.collisions
        for geometry in geometries:
            result.append(
                RobotGeometryInstance(
                    link_name=link_name,
                    geometry_type=geometry.geometry_type,
                    kind=kind,
                    geometry=dict(geometry.parameters),
                    local_transform=geometry.origin,
                    link_transform_base=transforms[link_name],
                    world_transform=base @ transforms[link_name] @ geometry.origin,
                    source_file=geometry.source_file,
                    resolved_path=None
                    if geometry.resolved_path is None
                    else str(geometry.resolved_path),
                    source_hash=geometry.source_hash,
                )
            )
    return result


def visual_geometry_instances(
    model: UrdfModel, qpos: Any, base_pose_scene: np.ndarray | None = None
) -> list[RobotGeometryInstance]:
    return _instances(model, qpos, base_pose_scene=base_pose_scene, kind="visual")


def collision_geometry_instances(
    model: UrdfModel, qpos: Any, base_pose_scene: np.ndarray | None = None
) -> list[RobotGeometryInstance]:
    return _instances(model, qpos, base_pose_scene=base_pose_scene, kind="collision")


def geometry_summary(model: UrdfModel) -> dict[str, Any]:
    visual_links = {link.name for link in model.links.values() if link.visuals}
    collision_links = {link.name for link in model.links.values() if link.collisions}
    unresolved = []
    for link in model.links.values():
        for geometry in (*link.visuals, *link.collisions):
            if geometry.geometry_type == "mesh" and geometry.resolved_path is None:
                unresolved.append(geometry.source_file)
    return {
        "visual_geometry_count": sum(len(link.visuals) for link in model.links.values()),
        "collision_geometry_count": sum(len(link.collisions) for link in model.links.values()),
        "visual_links": sorted(visual_links),
        "collision_links": sorted(collision_links),
        "missing_visual_links": sorted(set(model.link_names) - visual_links),
        "missing_collision_links": sorted(set(model.link_names) - collision_links),
        "unresolved_mesh_references": sorted(item for item in unresolved if item is not None),
        "fixed_tip_links_with_collision": sorted(
            joint.child for joint in model.fixed_joints if model.links[joint.child].collisions
        ),
    }


__all__ = [
    "RobotGeometryInstance",
    "collision_geometry_instances",
    "geometry_summary",
    "visual_geometry_instances",
]
