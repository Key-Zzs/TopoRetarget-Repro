"""Deterministic object-local and triangle-surface sampling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.geometry.mesh_audit import audit_mesh


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _hash_mapping(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_canonical(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SurfaceSamplingProfile:
    profile_id: str
    version: str
    method: str
    count: int
    seed: int
    rng_algorithm: str = "numpy_pcg64"
    exclude_degenerate_faces: bool = True
    degenerate_area_threshold: float = 1e-12
    normal_mode: str = "face_normal"
    source: str = "engineering"
    assumptions: tuple[str, ...] = ()
    paper_locked_fields: tuple[str, ...] = ()
    paper_status: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method != "area_uniform_triangles":
            raise ValueError(f"unsupported surface sampling method: {self.method}")
        if self.count <= 0:
            raise ValueError("surface sample count must be positive")
        if self.normal_mode not in {"face_normal"}:
            raise ValueError("only face_normal is currently supported")

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "method": self.method,
            "count": self.count,
            "seed": self.seed,
            "rng_algorithm": self.rng_algorithm,
            "exclude_degenerate_faces": self.exclude_degenerate_faces,
            "degenerate_area_threshold": self.degenerate_area_threshold,
            "normal_mode": self.normal_mode,
            "source": self.source,
            "assumptions": list(self.assumptions),
            "paper_locked_fields": list(self.paper_locked_fields),
            "paper_status": dict(self.paper_status),
        }

    @property
    def profile_hash(self) -> str:
        return _hash_mapping(self.as_dict())


@dataclass
class SurfaceSampleSet:
    mesh_id: str
    mesh_hash: str
    topology_hash: str
    profile_id: str
    profile_hash: str
    face_indices: np.ndarray
    barycentric: np.ndarray
    points_local: np.ndarray
    normals_local: np.ndarray
    valid: np.ndarray
    degenerate_faces_excluded: int
    profile: dict[str, Any]
    mesh_array_hash: str | None = None
    source_provenance: dict[str, Any] = field(default_factory=dict)
    scale: np.ndarray | float = 1.0

    @property
    def count(self) -> int:
        return int(self.points_local.shape[0])

    def validate(
        self, vertices: np.ndarray, faces: np.ndarray, *, atol: float = 1e-12
    ) -> dict[str, Any]:
        values = np.asarray(vertices, dtype=np.float64) * np.asarray(self.scale, dtype=np.float64)
        triangles = np.asarray(faces, dtype=np.int64)
        reconstructed = np.einsum(
            "ni,nij->nj", self.barycentric, values[triangles[self.face_indices]]
        )
        audit = audit_mesh(values, triangles)
        return {
            "count": self.count,
            "count_exact": self.count == int(self.profile["count"]),
            "face_indices_valid": bool(
                np.all((self.face_indices >= 0) & (self.face_indices < len(triangles)))
            ),
            "barycentric_nonnegative": bool(np.all(self.barycentric >= -atol)),
            "barycentric_sum_max_error": float(np.max(np.abs(self.barycentric.sum(axis=1) - 1.0)))
            if self.count
            else 0.0,
            "point_reconstruction_max_error": float(
                np.max(np.linalg.norm(reconstructed - self.points_local, axis=1))
            )
            if self.count
            else 0.0,
            "mesh_hash_match": audit.mesh_hash in {self.mesh_hash, self.mesh_array_hash},
            "topology_hash_match": audit.topology_hash == self.topology_hash,
            "valid_count": int(np.count_nonzero(self.valid)),
            "mesh_hash": self.mesh_hash,
            "mesh_array_hash": self.mesh_array_hash,
            "profile_hash": self.profile_hash,
            "seed": self.profile.get("seed"),
            "method": self.profile.get("method"),
            "count_source": self.profile.get("count_source", self.profile.get("source")),
        }

    def as_metadata(self) -> dict[str, Any]:
        return {
            "mesh_id": self.mesh_id,
            "mesh_hash": self.mesh_hash,
            "mesh_array_hash": self.mesh_array_hash,
            "topology_hash": self.topology_hash,
            "profile_id": self.profile_id,
            "profile_hash": self.profile_hash,
            "count": self.count,
            "seed": self.profile.get("seed"),
            "method": self.profile.get("method"),
            "count_source": self.profile.get("count_source", self.profile.get("source")),
            "degenerate_faces_excluded": self.degenerate_faces_excluded,
            "profile": self.profile,
            "source_provenance": self.source_provenance,
            "scale": np.asarray(self.scale).tolist(),
        }

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"sample artifact exists; pass overwrite=True: {destination}")
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                face_indices=self.face_indices,
                barycentric=self.barycentric,
                points_local=self.points_local,
                normals_local=self.normals_local,
                valid=self.valid,
                metadata=np.asarray(json.dumps(self.as_metadata(), sort_keys=True)),
            )
        temporary.replace(destination)
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        vertices: np.ndarray | None = None,
        faces: np.ndarray | None = None,
    ) -> SurfaceSampleSet:
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            result = cls(
                mesh_id=str(metadata["mesh_id"]),
                mesh_hash=str(metadata["mesh_hash"]),
                mesh_array_hash=metadata.get("mesh_array_hash"),
                topology_hash=str(metadata["topology_hash"]),
                profile_id=str(metadata["profile_id"]),
                profile_hash=str(metadata["profile_hash"]),
                face_indices=np.asarray(data["face_indices"], dtype=np.int64),
                barycentric=np.asarray(data["barycentric"], dtype=np.float64),
                points_local=np.asarray(data["points_local"], dtype=np.float64),
                normals_local=np.asarray(data["normals_local"], dtype=np.float64),
                valid=np.asarray(data["valid"], dtype=bool),
                degenerate_faces_excluded=int(metadata["degenerate_faces_excluded"]),
                profile=dict(metadata["profile"]),
                source_provenance=dict(metadata.get("source_provenance", {})),
                scale=np.asarray(metadata.get("scale", 1.0)),
            )
        profile_for_hash = dict(result.profile)
        profile_for_hash.pop("count_source", None)
        if _hash_mapping(profile_for_hash) != result.profile_hash:
            raise ValueError("surface artifact profile hash mismatch")
        if vertices is not None and faces is not None:
            scaled_vertices = np.asarray(vertices, dtype=np.float64) * np.asarray(
                result.scale, dtype=np.float64
            )
            audit = audit_mesh(scaled_vertices, faces)
            if (
                audit.mesh_hash not in {result.mesh_hash, result.mesh_array_hash}
                or audit.topology_hash != result.topology_hash
            ):
                raise ValueError("surface artifact mesh/topology hash mismatch")
        return result


def load_surface_profile(
    profile_id: str, *, repo_root: str | Path | None = None
) -> SurfaceSamplingProfile:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    path = root / "configs" / "geometry" / "object_surface_sampling.yaml"
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    values = (loaded.get("profiles") or {}).get(profile_id)
    if not isinstance(values, dict):
        raise KeyError(f"unknown object surface profile: {profile_id}")
    count_value = values.get("count")
    if isinstance(count_value, dict):
        source = root / str(count_value.get("source", "configs/paper/retarget.yaml"))
        paper = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        count = int(paper[str(count_value.get("key"))])
        count_source = f"{source.relative_to(root)}:{count_value.get('key')}"
    else:
        if count_value is None:
            raise ValueError(f"profile {profile_id} does not define a sample count")
        count = int(count_value)
        count_source = "profile"
    return SurfaceSamplingProfile(
        profile_id=profile_id,
        version=str(values.get("version", "1")),
        method=str(values.get("method", "area_uniform_triangles")),
        count=count,
        seed=int(values["seed"]),
        rng_algorithm=str(values.get("rng", "numpy_pcg64")),
        exclude_degenerate_faces=bool(values.get("exclude_degenerate_faces", True)),
        degenerate_area_threshold=float(values.get("degenerate_area_threshold", 1e-12)),
        normal_mode=str(values.get("normal_mode", "face_normal")),
        source=count_source,
        assumptions=tuple(str(item) for item in values.get("assumptions", [])),
        paper_locked_fields=tuple(str(item) for item in values.get("paper_locked_fields", [])),
        paper_status=dict(values.get("paper_status", {})),
    )


def sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    profile: SurfaceSamplingProfile,
    *,
    mesh_id: str = "mesh",
    mesh_hash: str | None = None,
    source_path: str | Path | None = None,
    source_provenance: dict[str, Any] | None = None,
    scale: float | np.ndarray = 1.0,
) -> SurfaceSampleSet:
    """Sample fixed local anchors with an explicit PCG64 generator."""

    original_vertices = np.asarray(vertices, dtype=np.float64)
    original_faces = np.asarray(faces, dtype=np.int64)
    scaled_vertices = original_vertices * np.asarray(scale, dtype=np.float64)
    audit = audit_mesh(scaled_vertices, original_faces, source_path=source_path)
    valid = np.ones(len(original_faces), dtype=bool)
    if profile.exclude_degenerate_faces:
        tri = scaled_vertices[original_faces]
        areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        valid &= areas > profile.degenerate_area_threshold
    if not np.any(valid):
        raise ValueError("mesh has no non-degenerate faces available for surface sampling")
    face_pool = np.flatnonzero(valid)
    triangles = scaled_vertices[original_faces[face_pool]]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    probabilities = areas / areas.sum()
    rng = np.random.Generator(np.random.PCG64(profile.seed))
    selected_pool = rng.choice(len(face_pool), size=profile.count, replace=True, p=probabilities)
    face_indices = face_pool[selected_pool].astype(np.int64)
    u = rng.random(profile.count)
    v = rng.random(profile.count)
    root = np.sqrt(u)
    barycentric = np.column_stack((1.0 - root, root * (1.0 - v), root * v))
    chosen = scaled_vertices[original_faces[face_indices]]
    points = np.einsum("ni,nij->nj", barycentric, chosen)
    normals = np.cross(chosen[:, 1] - chosen[:, 0], chosen[:, 2] - chosen[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1)
    normals = normals / normal_lengths[:, None]
    metadata = {
        "source_path": None if source_path is None else str(source_path),
        "seed": profile.seed,
        "rng_algorithm": profile.rng_algorithm,
        "method": profile.method,
        "count_source": profile.source,
        "assumptions": list(profile.assumptions),
    }
    return SurfaceSampleSet(
        mesh_id=mesh_id,
        mesh_hash=mesh_hash or audit.mesh_hash,
        mesh_array_hash=audit.mesh_hash,
        topology_hash=audit.topology_hash,
        profile_id=profile.profile_id,
        profile_hash=profile.profile_hash,
        face_indices=face_indices,
        barycentric=barycentric,
        points_local=points,
        normals_local=normals,
        valid=np.ones(profile.count, dtype=bool),
        degenerate_faces_excluded=int(len(original_faces) - len(face_pool)),
        profile=profile.as_dict(),
        source_provenance={**metadata, **dict(source_provenance or {})},
        scale=np.asarray(scale),
    )


def transform_surface_samples(
    samples: SurfaceSampleSet, transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    from toporetarget.geometry.se3 import transform_points, transform_vectors

    return transform_points(transform, samples.points_local[None, ...])[0], transform_vectors(
        transform, samples.normals_local[None, ...]
    )[0]


__all__ = [
    "SurfaceSampleSet",
    "SurfaceSamplingProfile",
    "load_surface_profile",
    "sample_mesh_surface",
    "transform_surface_samples",
]
