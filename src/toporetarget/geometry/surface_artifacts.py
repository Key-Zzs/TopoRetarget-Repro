"""Hash-keyed, disposable surface-sample artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .surface_sampling import SurfaceSampleSet, SurfaceSamplingProfile


def surface_artifact_key(
    mesh_hash: str,
    topology_hash: str,
    profile: SurfaceSamplingProfile,
    *,
    scale: object = 1.0,
    code_version: str = "stage6-v1",
) -> str:
    value = f"{mesh_hash}:{topology_hash}:{profile.profile_hash}:{scale!r}:{code_version}"
    return hashlib.sha256(value.encode()).hexdigest()


def default_surface_cache_path(
    root: str | Path, samples: SurfaceSampleSet, *, key: str | None = None
) -> Path:
    destination = Path(root) / "object_surface" / samples.mesh_hash
    name = f"{key or samples.profile_hash}.npz"
    return destination / name


def save_surface_artifact(
    samples: SurfaceSampleSet, path: str | Path, *, overwrite: bool = False
) -> Path:
    return samples.save(path, overwrite=overwrite)


def load_surface_artifact(path: str | Path, *, vertices=None, faces=None) -> SurfaceSampleSet:
    return SurfaceSampleSet.load(path, vertices=vertices, faces=faces)


__all__ = [
    "default_surface_cache_path",
    "load_surface_artifact",
    "save_surface_artifact",
    "surface_artifact_key",
]
