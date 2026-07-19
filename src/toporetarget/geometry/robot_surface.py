"""Collision-only robot surface sampling and FK transforms."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.robots.urdf.geometry import RobotGeometryInstance

from .object_geometry import load_mesh_file
from .surface_sampling import SurfaceSamplingProfile, sample_mesh_surface


@dataclass(frozen=True)
class RobotSurfaceSamplingProfile:
    profile_id: str
    version: str
    count_per_geometry: int
    method: str
    seed: int
    source: str
    paper_status: str
    visual_fallback: bool
    tip_visual_fallback: bool
    assumptions: tuple[str, ...] = ()

    @property
    def profile_hash(self) -> str:
        value = {
            "profile_id": self.profile_id,
            "version": self.version,
            "count_per_geometry": self.count_per_geometry,
            "method": self.method,
            "seed": self.seed,
            "source": self.source,
            "paper_status": self.paper_status,
            "visual_fallback": self.visual_fallback,
            "tip_visual_fallback": self.tip_visual_fallback,
            "assumptions": list(self.assumptions),
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass
class RobotSurfaceSampleSet:
    robot_name: str
    side: str
    profile: RobotSurfaceSamplingProfile
    geometry_ids: np.ndarray
    link_names: np.ndarray
    geometry_types: np.ndarray
    sample_ids: np.ndarray
    points_local: np.ndarray
    normals_local: np.ndarray
    points_base: np.ndarray
    normals_base: np.ndarray
    points_scene: np.ndarray
    normals_scene: np.ndarray
    geometry_metadata: list[dict[str, Any]] = field(default_factory=list)
    source_provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.points_local)

    def as_dict(self) -> dict[str, Any]:
        return {
            "robot_name": self.robot_name,
            "side": self.side,
            "profile_id": self.profile.profile_id,
            "profile_hash": self.profile.profile_hash,
            "profile": self.profile.__dict__,
            "sample_count": self.count,
            "geometry_count": len(self.geometry_metadata),
            "geometry_metadata": self.geometry_metadata,
            "link_coverage": sorted(set(str(value) for value in self.link_names.tolist())),
            "source_provenance": self.source_provenance,
        }

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        temporary = destination.with_name(f".{destination.name}.tmp")
        metadata = {
            "robot_name": self.robot_name,
            "side": self.side,
            "profile": {
                **self.profile.__dict__,
                "assumptions": list(self.profile.assumptions),
            },
            "geometry_metadata": self.geometry_metadata,
            "source_provenance": self.source_provenance,
        }
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                geometry_ids=self.geometry_ids,
                link_names=self.link_names,
                geometry_types=self.geometry_types,
                sample_ids=self.sample_ids,
                points_local=self.points_local,
                normals_local=self.normals_local,
                points_base=self.points_base,
                normals_base=self.normals_base,
                points_scene=self.points_scene,
                normals_scene=self.normals_scene,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        temporary.replace(destination)
        return destination


def load_robot_surface_profile(
    profile_id: str, *, repo_root: str | Path | None = None
) -> RobotSurfaceSamplingProfile:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    path = root / "configs" / "geometry" / "robot_collision_sampling.yaml"
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    values = (loaded.get("profiles") or {}).get(profile_id)
    if not isinstance(values, dict):
        raise KeyError(f"unknown robot surface profile: {profile_id}")
    return RobotSurfaceSamplingProfile(
        profile_id=profile_id,
        version=str(values.get("version", "1")),
        count_per_geometry=int(values["count_per_geometry"]),
        method=str(values.get("method", "area_uniform_triangles")),
        seed=int(values.get("seed", 0)),
        source=str(values.get("source", "engineering")),
        paper_status=str(values.get("paper_status", "not_paper_specified")),
        visual_fallback=bool(values.get("visual_fallback", False)),
        tip_visual_fallback=bool(values.get("tip_visual_fallback", False)),
        assumptions=tuple(str(item) for item in values.get("assumptions", [])),
    )


def _geometry_mesh(instance: RobotGeometryInstance) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    if instance.geometry_type == "mesh":
        if instance.resolved_path is None:
            raise FileNotFoundError(f"unresolved collision mesh: {instance.source_file}")
        vertices, faces = load_mesh_file(instance.resolved_path)
        vertices *= np.asarray(instance.geometry.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
        return vertices, faces
    if instance.geometry_type == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(instance.geometry["radius"]))
    elif instance.geometry_type == "box":
        mesh = trimesh.creation.box(extents=np.asarray(instance.geometry["size"], dtype=np.float64))
    elif instance.geometry_type == "cylinder":
        mesh = trimesh.creation.cylinder(
            radius=float(instance.geometry["radius"]), height=float(instance.geometry["length"])
        )
    else:
        raise ValueError(f"unsupported collision geometry type: {instance.geometry_type}")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def _transform(
    points: np.ndarray, normals: np.ndarray, transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rotation = transform[:3, :3]
    return points @ rotation.T + transform[:3, 3], normals @ rotation.T


def sample_robot_collision_surface(
    model: Any,
    qpos: np.ndarray,
    profile: RobotSurfaceSamplingProfile,
    *,
    base_pose_scene: np.ndarray | None = None,
) -> RobotSurfaceSampleSet:
    if profile.visual_fallback or profile.tip_visual_fallback:
        raise ValueError("visual fallback is forbidden for collision sampling")
    instances = model.collision_geometry_instances(
        np.asarray(qpos, dtype=np.float64), base_pose_scene
    )
    if not instances:
        raise ValueError(f"{model.name} has no collision geometry")
    base_pose = (
        np.eye(4, dtype=np.float64)
        if base_pose_scene is None
        else np.asarray(base_pose_scene, dtype=np.float64)
    )
    geometry_ids: list[str] = []
    links: list[str] = []
    types: list[str] = []
    sample_ids: list[int] = []
    local_points: list[np.ndarray] = []
    local_normals: list[np.ndarray] = []
    base_points: list[np.ndarray] = []
    base_normals: list[np.ndarray] = []
    scene_points: list[np.ndarray] = []
    scene_normals: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    sample_offset = 0
    for geometry_index, instance in enumerate(instances):
        vertices, faces = _geometry_mesh(instance)
        local_profile = SurfaceSamplingProfile(
            profile_id=f"{profile.profile_id}:{geometry_index}",
            version=profile.version,
            method="area_uniform_triangles",
            count=profile.count_per_geometry,
            seed=profile.seed + geometry_index,
            source=profile.source,
            assumptions=profile.assumptions,
        )
        samples = sample_mesh_surface(
            vertices, faces, local_profile, mesh_id=f"{model.name}:{geometry_index}"
        )
        geometry_id = f"collision:{geometry_index}:{instance.link_name}:{instance.geometry_type}"
        transform_base = instance.transform_base
        points_b, normals_b = _transform(
            samples.points_local, samples.normals_local, transform_base
        )
        points_s, normals_s = _transform(points_b, normals_b, base_pose)
        geometry_ids.extend([geometry_id] * samples.count)
        links.extend([instance.link_name] * samples.count)
        types.extend([instance.geometry_type] * samples.count)
        sample_ids.extend(range(sample_offset, sample_offset + samples.count))
        sample_offset += samples.count
        local_points.append(samples.points_local)
        local_normals.append(samples.normals_local)
        base_points.append(points_b)
        base_normals.append(normals_b)
        scene_points.append(points_s)
        scene_normals.append(normals_s)
        metadata.append(
            {
                "geometry_id": geometry_id,
                "link_name": instance.link_name,
                "geometry_type": instance.geometry_type,
                "source_file": instance.source_file,
                "resolved_path": instance.resolved_path,
                "source_hash": instance.source_hash,
                "sample_count": samples.count,
            }
        )
    return RobotSurfaceSampleSet(
        robot_name=model.name,
        side=model.side,
        profile=profile,
        geometry_ids=np.asarray(geometry_ids, dtype="U"),
        link_names=np.asarray(links, dtype="U"),
        geometry_types=np.asarray(types, dtype="U"),
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        points_local=np.concatenate(local_points),
        normals_local=np.concatenate(local_normals),
        points_base=np.concatenate(base_points),
        normals_base=np.concatenate(base_normals),
        points_scene=np.concatenate(scene_points),
        normals_scene=np.concatenate(scene_normals),
        geometry_metadata=metadata,
        source_provenance={
            "robot_name": model.name,
            "side": model.side,
            "urdf_hash": model.urdf_hash,
            "asset_manifest_hash": model.asset_manifest_hash,
            "collision_only": True,
            "visual_fallback": False,
            "tip_visual_fallback": False,
        },
    )


def transform_robot_surface_to_scene(
    samples: RobotSurfaceSampleSet, base_pose_scene: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return _transform(
        samples.points_base, samples.normals_base, np.asarray(base_pose_scene, dtype=np.float64)
    )


__all__ = [
    "RobotSurfaceSampleSet",
    "RobotSurfaceSamplingProfile",
    "load_robot_surface_profile",
    "sample_robot_collision_surface",
    "transform_robot_surface_to_scene",
]
