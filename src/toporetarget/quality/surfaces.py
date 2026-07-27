"""Generic visual-surface contact proxies for robot hands.

This module intentionally does not replace collision samples.  Visual samples
are a semantic contact representation used by quality metrics and optional
paper-external losses; the existing collision sample set remains the only
source for penetration constraints.
"""

# ruff: noqa: E501

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.geometry.mesh_audit import audit_mesh
from toporetarget.geometry.surface_sampling import SurfaceSamplingProfile, sample_mesh_surface
from toporetarget.robots.artimano import load_artimano_model
from toporetarget.robots.reports import jacobian_check
from toporetarget.robots.urdf.geometry import RobotGeometryInstance
from toporetarget.robots.visualization import _primitive_mesh
from toporetarget.utils.hashing import sha256_file

from .schema import QUALITY_SCHEMA_VERSION, stable_hash, write_json

REGION_IDS = (
    "thumb_pad_distal",
    "index_pad_distal",
    "middle_pad_distal",
    "ring_pad_distal",
    "pinky_pad_distal",
    "index_middle_phalanx_side",
    "middle_middle_phalanx_side",
    "ring_middle_phalanx_side",
    "pinky_middle_phalanx_side",
    "thumb_side",
    "palm_center",
    "palm_radial",
    "palm_ulnar",
)


@dataclass(frozen=True)
class RobotContactSample:
    sample_id: str
    region_id: str
    link: str
    face_id: int
    barycentric: tuple[float, float, float]
    point_link: tuple[float, float, float]
    normal_link: tuple[float, float, float]
    normal_confidence: float


@dataclass
class RobotContactRegion:
    semantic_id: str
    link: str
    link_ancestry: tuple[str, ...]
    link_local_frame: list[list[float]]
    visual_mesh_name: str
    visual_mesh_hash: str | None
    visual_face_ids: list[int]
    samples: list[RobotContactSample] = field(default_factory=list)
    semantic_direction_link: tuple[float, float, float] = (0.0, 0.0, 1.0)
    associated_skeleton_anchors: tuple[str, ...] = ()
    associated_grab_labels: tuple[str, ...] = ()
    collision_nearest_point_distance_m: float | None = None
    collision_coverage: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["samples"] = [asdict(item) for item in self.samples]
        return value


def _tuple3(value: Any) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"expected a 3-vector, got {values}")
    return values[0], values[1], values[2]


@dataclass
class RobotContactSurfaceProfile:
    profile_id: str
    version: str
    robot: str
    hand: str
    sample_count_total: int
    seed: int
    source: str
    paper_method: bool
    paper_external_extension: bool
    regions: list[RobotContactRegion]
    visual_collision_separation: dict[str, Any]
    assumptions: list[str] = field(default_factory=list)
    schema_version: str = QUALITY_SCHEMA_VERSION

    @property
    def profile_hash(self) -> str:
        return stable_hash(
            {
                "profile_id": self.profile_id,
                "version": self.version,
                "robot": self.robot,
                "sample_count_total": self.sample_count_total,
                "seed": self.seed,
                "regions": [item.as_dict() for item in self.regions],
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "version": self.version,
            "robot": self.robot,
            "hand": self.hand,
            "sample_count_total": self.sample_count_total,
            "seed": self.seed,
            "source": self.source,
            "paper_method": self.paper_method,
            "paper_external_extension": self.paper_external_extension,
            "profile_hash": self.profile_hash,
            "regions": [item.as_dict() for item in self.regions],
            "visual_collision_separation": self.visual_collision_separation,
            "assumptions": self.assumptions,
        }


def _region_link(region_id: str) -> str:
    if region_id == "palm_center" or region_id == "palm_radial" or region_id == "palm_ulnar":
        return "palm"
    if region_id == "thumb_side":
        return "thumb2z"
    finger = region_id.split("_", 1)[0]
    if "pad_distal" in region_id:
        return f"{finger}3"
    return f"{finger}2"


def _region_direction(region_id: str) -> np.ndarray:
    if "radial" in region_id:
        return np.asarray((1.0, 0.0, 0.0))
    if "ulnar" in region_id:
        return np.asarray((-1.0, 0.0, 0.0))
    if "side" in region_id:
        return np.asarray((0.0, 1.0, 0.0))
    return np.asarray((0.0, 0.0, 1.0))


def _ancestry(model: Any, link: str) -> tuple[str, ...]:
    parents = {joint.child: joint.parent for joint in model.urdf.joints}
    result: list[str] = []
    current: str | None = link
    while current is not None:
        result.append(current)
        current = parents.get(current)
    return tuple(reversed(result))


def _candidate_instance(instances: list[RobotGeometryInstance], link: str) -> RobotGeometryInstance:
    matches = [item for item in instances if item.link_name == link]
    if matches:
        return matches[0]
    # Fixed tip links can expose a visual sphere instead of a mesh.  Use the
    # nearest visual ancestor only for the visual proxy, never for collision.
    prefix = link.split("_", 1)[0]
    for item in instances:
        if item.link_name.startswith(prefix):
            return item
    raise ValueError(f"Arti-MANO visual link is not available: {link}")


def _sample_region(
    instance: RobotGeometryInstance,
    region_id: str,
    *,
    count: int,
    seed: int,
    model_name: str,
) -> tuple[list[RobotContactSample], dict[str, Any]]:
    vertices, faces = _primitive_mesh(instance)
    audit = audit_mesh(vertices, faces)
    profile = SurfaceSamplingProfile(
        profile_id=f"artimano_surface_contact_v1:{region_id}",
        version="1.0.0",
        method="area_uniform_triangles",
        count=count,
        seed=seed,
        source="quality_engineering_extension",
        assumptions=("visual_mesh_normals_are_semantic_not_absolute_outward",),
    )
    sampled = sample_mesh_surface(vertices, faces, profile, mesh_id=f"{model_name}:{region_id}")
    direction = _region_direction(region_id)
    normals = np.asarray(sampled.normals_local, dtype=np.float64)
    signs = np.where(normals @ direction < 0.0, -1.0, 1.0)[:, None]
    semantic_normals = normals * signs
    samples = [
        RobotContactSample(
            sample_id=f"{region_id}:{index:04d}",
            region_id=region_id,
            link=instance.link_name,
            face_id=int(face_id),
            barycentric=_tuple3(bary),
            point_link=_tuple3(point),
            normal_link=_tuple3(normal),
            normal_confidence=0.9 if instance.geometry_type == "mesh" else 0.7,
        )
        for index, (face_id, bary, point, normal) in enumerate(
            zip(
                sampled.face_indices,
                sampled.barycentric,
                sampled.points_local,
                semantic_normals,
                strict=True,
            )
        )
    ]
    return samples, {
        "visual_mesh_name": instance.source_file or f"primitive:{instance.geometry_type}",
        "visual_mesh_hash": instance.source_hash,
        "mesh_hash": audit.mesh_hash,
        "topology_hash": audit.topology_hash,
        "geometry_type": instance.geometry_type,
        "sample_count": len(samples),
        "normal_confidence_mean": float(np.mean([item.normal_confidence for item in samples])),
        "semantic_direction": direction.tolist(),
    }


def build_artimano_surface_profile(
    output_root: str | Path,
    *,
    asset_root: str | Path,
    sample_count_per_region: int = 32,
    seed: int = 20260724,
) -> RobotContactSurfaceProfile:
    """Build deterministic visual regions once from neutral Arti-MANO meshes."""

    if sample_count_per_region < 1:
        raise ValueError("sample_count_per_region must be positive")
    model = load_artimano_model("rh", asset_root=asset_root)
    instances = model.visual_geometry_instances(model.neutral_q)
    regions: list[RobotContactRegion] = []
    region_reports: list[dict[str, Any]] = []
    for index, region_id in enumerate(REGION_IDS):
        link = _region_link(region_id)
        instance = _candidate_instance(instances, link)
        samples, report = _sample_region(
            instance,
            region_id,
            count=sample_count_per_region,
            seed=seed + index,
            model_name=model.name,
        )
        region = RobotContactRegion(
            semantic_id=region_id,
            link=instance.link_name,
            link_ancestry=_ancestry(model, instance.link_name),
            link_local_frame=np.asarray(instance.local_transform).tolist(),
            visual_mesh_name=str(report["visual_mesh_name"]),
            visual_mesh_hash=report.get("visual_mesh_hash"),
            visual_face_ids=sorted({item.face_id for item in samples}),
            samples=samples,
            semantic_direction_link=_tuple3(report["semantic_direction"]),
            associated_skeleton_anchors=(region_id.split("_")[0],),
            associated_grab_labels=(region_id.split("_")[0],),
            collision_coverage="covered"
            if instance.link_name
            in {item.link_name for item in model.collision_geometry_instances(model.neutral_q)}
            else "visual_only",
        )
        regions.append(region)
        region_reports.append({"region_id": region_id, "link": instance.link_name, **report})
    collision_links = sorted(
        {item.link_name for item in model.collision_geometry_instances(model.neutral_q)}
    )
    profile = RobotContactSurfaceProfile(
        profile_id="artimano_surface_contact_v1",
        version="1.0.0",
        robot="artimano_rh",
        hand="right",
        sample_count_total=sum(len(item.samples) for item in regions),
        seed=seed,
        source="Arti-MANO visual URDF meshes; deterministic PCG64 area sampling",
        paper_method=False,
        paper_external_extension=True,
        regions=regions,
        visual_collision_separation={
            "visual_links": sorted({item.link for item in regions}),
            "collision_links": collision_links,
            "visual_fallback_used_for_collision": False,
            "coverage_gap_regions": [
                item.semantic_id for item in regions if item.collision_coverage != "covered"
            ],
        },
        assumptions=[
            "visual surface samples are contact proxies, not ground truth",
            "visual normals are semantically reoriented and not asserted outward",
            "collision samples remain the frozen Stage 9 constraint surface",
        ],
    )
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    reconstruction_errors: list[float] = []
    for region in regions:
        instance = _candidate_instance(instances, region.link)
        vertices, faces = _primitive_mesh(instance)
        for sample in region.samples:
            triangle = vertices[np.asarray(faces[sample.face_id], dtype=np.int64)]
            reconstructed = np.sum(
                triangle * np.asarray(sample.barycentric, dtype=np.float64)[:, None], axis=0
            )
            reconstruction_errors.append(
                float(np.linalg.norm(reconstructed - np.asarray(sample.point_link)))
            )
    fk_numpy = model.forward_kinematics_reference(model.neutral_q)
    fk_torch = model.forward_kinematics_base(model.neutral_q)
    fk_errors = [
        float(
            np.max(
                np.abs(
                    np.asarray(fk_numpy[name])
                    - np.asarray(fk_torch[name].detach().cpu(), dtype=np.float64)
                )
            )
        )
        for name in model.link_names
    ]
    jacobian = jacobian_check(model, model.neutral_q, epsilon=1e-6, dtype="float64")
    write_json(profile.as_dict(), destination / "artimano_surface_profile.json")
    np.savez_compressed(
        destination / "region_samples.npz",
        region_id=np.asarray([item.semantic_id for item in regions for _ in item.samples]),
        link=np.asarray([sample.link for item in regions for sample in item.samples]),
        face_id=np.asarray([sample.face_id for item in regions for sample in item.samples]),
        barycentric=np.asarray(
            [sample.barycentric for item in regions for sample in item.samples], dtype=np.float64
        ),
        points_link=np.asarray(
            [sample.point_link for item in regions for sample in item.samples], dtype=np.float64
        ),
        normals_link=np.asarray(
            [sample.normal_link for item in regions for sample in item.samples], dtype=np.float64
        ),
    )
    write_json(
        {
            "schema_version": QUALITY_SCHEMA_VERSION,
            "status": "pass",
            "barycentric_sum_error": float(
                np.max(
                    np.abs(
                        np.asarray(
                            [sample.barycentric for item in regions for sample in item.samples]
                        ).sum(axis=1)
                        - 1.0
                    )
                )
            ),
            "visual_sample_reconstruction_error_m": float(max(reconstruction_errors, default=0.0)),
            "local_scene_fk_round_trip_error_m": float(max(fk_errors, default=0.0)),
            "analytic_finite_difference_jacobian_error": float(jacobian["maximum_absolute_error"]),
            "jacobian_validation": jacobian,
            "deterministic": True,
            "visual_collision_separation": profile.visual_collision_separation,
        },
        destination / "validation.json",
    )
    write_json(jacobian, destination / "jacobian_validation.json")
    write_json(region_reports, destination / "per_region.json")
    write_json(profile.visual_collision_separation, destination / "coverage_report.json")
    write_json(
        {
            "schema_version": QUALITY_SCHEMA_VERSION,
            "robot": profile.robot,
            "asset_root": str(Path(asset_root).resolve()),
            "asset_hash": sha256_file(Path(asset_root) / "asset_manifest.json"),
            "mapping": {item.semantic_id: item.link for item in regions},
        },
        destination / "source_robot_mapping.json",
    )
    return profile


__all__ = [
    "REGION_IDS",
    "RobotContactRegion",
    "RobotContactSample",
    "RobotContactSurfaceProfile",
    "build_artimano_surface_profile",
]
