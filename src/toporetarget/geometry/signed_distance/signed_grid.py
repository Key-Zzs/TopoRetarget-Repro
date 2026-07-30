"""Deterministic original-mesh signed-distance grids for Stage 9 inner loops.

The grid is deliberately an acceleration structure, not a geometry proxy:
every vertex is evaluated by :class:`ReferenceSignedDistanceBackend` on the
original strict/watertight mesh.  Final audits must continue to use that
triangle-winding backend directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.geometry.mesh_audit import audit_mesh

from .base import SignedDistanceBackend, SignedDistanceQueryResult
from .reference import ReferenceSignedDistanceBackend

GRID_PROFILE_PREFIX = "original_mesh_signed_grid_"
GRID_SCHEMA_VERSION = "toporetarget.original_mesh_signed_grid.v1"


def grid_resolution_from_profile(profile_id: str) -> int | None:
    """Return the declared isotropic-longest-axis resolution, if applicable."""

    if not profile_id.startswith(GRID_PROFILE_PREFIX) or not profile_id.endswith("_v1"):
        return None
    value = profile_id[len(GRID_PROFILE_PREFIX) : -len("_v1")]
    try:
        resolution = int(value)
    except ValueError:
        return None
    return resolution if resolution >= 8 else None


def _json_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _trilinear(values: np.ndarray, indices: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Interpolate a scalar or trailing-vector grid at already-valid cells."""

    i, j, k = indices.T
    fx, fy, fz = fractions.T
    trailing = (None,) * (values.ndim - 3)
    c000 = values[i, j, k]
    c100 = values[i + 1, j, k]
    c010 = values[i, j + 1, k]
    c110 = values[i + 1, j + 1, k]
    c001 = values[i, j, k + 1]
    c101 = values[i + 1, j, k + 1]
    c011 = values[i, j + 1, k + 1]
    c111 = values[i + 1, j + 1, k + 1]
    fx = fx[(...,) + trailing]
    fy = fy[(...,) + trailing]
    fz = fz[(...,) + trailing]
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


class OriginalMeshSignedGridSDFBackend(SignedDistanceBackend):
    """Trilinear signed grid generated from an unmodified watertight mesh.

    ``resolution`` is the number of grid nodes on the longest padded axis;
    other axes preserve the same voxel spacing.  Its only mutable diagnostic
    is the out-of-grid counter, which does not affect query results.
    """

    backend_id = "original_mesh_signed_grid"

    def __init__(
        self,
        *,
        signed_distance_grid: np.ndarray,
        gradient_grid: np.ndarray,
        origin: np.ndarray,
        voxel_size: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        vertices: np.ndarray,
        faces: np.ndarray,
        mesh_hash: str,
        profile_id: str,
        metadata: dict[str, Any],
    ) -> None:
        self.signed_distance_grid = np.asarray(signed_distance_grid, dtype=np.float64)
        self.gradient_grid = np.asarray(gradient_grid, dtype=np.float64)
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.voxel_size = np.asarray(voxel_size, dtype=np.float64).reshape(3)
        self.bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(3)
        self.bbox_max = np.asarray(bbox_max, dtype=np.float64).reshape(3)
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.mesh_hash = str(mesh_hash)
        self.profile_id = str(profile_id)
        self.metadata = dict(metadata)
        if self.signed_distance_grid.ndim != 3 or min(self.signed_distance_grid.shape) < 2:
            raise ValueError(
                "signed grid must be three-dimensional with at least two nodes per axis"
            )
        if self.gradient_grid.shape != (*self.signed_distance_grid.shape, 3):
            raise ValueError("gradient grid shape must be [nx,ny,nz,3]")
        if not np.all(np.isfinite(self.signed_distance_grid)) or not np.all(
            np.isfinite(self.gradient_grid)
        ):
            raise ValueError("signed grid values and gradients must be finite")
        if not np.all(np.isfinite(self.voxel_size)) or np.any(self.voxel_size <= 0.0):
            raise ValueError("voxel sizes must be finite and positive")
        # Outside the padded grid the object-bbox lower bound is safe but can
        # be very loose for distant hand samples.  Keep an exact *unsigned*
        # triangle-distance accelerator for that provably exterior branch:
        # it cannot affect sign, does not run winding, and preserves the
        # required positive-outside/fail-closed policy.
        self._exterior_distance = ReferenceSignedDistanceBackend(
            self.vertices,
            self.faces,
            mesh_hash=self.mesh_hash,
            sign_mode="unsigned_only",
            query_chunk_size=4096,
            face_chunk_size=4096,
            closest_acceleration="tree",
        )
        self.out_of_grid_query_count = 0

    @property
    def grid_max(self) -> np.ndarray:
        return self.origin + self.voxel_size * (np.asarray(self.signed_distance_grid.shape) - 1)

    @classmethod
    def build(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        resolution: int,
        profile_id: str | None = None,
        cache_root: str | Path | None = None,
        query_batch_size: int = 4096,
    ) -> OriginalMeshSignedGridSDFBackend:
        """Load or build the grid from exact triangle/winding values.

        The cache key includes the raw original mesh hash and every geometric
        construction parameter.  No repair, hull, or object-specific setting
        participates in generation.
        """

        if resolution < 8:
            raise ValueError("grid resolution must be at least 8")
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        mesh_audit = audit_mesh(vertices, faces)
        if not (
            mesh_audit.watertight
            and mesh_audit.non_manifold_edge_count == 0
            and mesh_audit.winding_consistent is not False
            and mesh_audit.orientable is not False
            and mesh_audit.near_zero_area_faces == 0
        ):
            raise ValueError("original-mesh signed grid requires a strict watertight source mesh")
        mesh_hash = str(mesh_audit.mesh_hash)
        profile_id = profile_id or f"{GRID_PROFILE_PREFIX}{resolution}_v1"
        if grid_resolution_from_profile(profile_id) != resolution:
            raise ValueError("grid profile and resolution disagree")
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        extent = bbox_max - bbox_min
        diagonal = float(np.linalg.norm(extent))
        if not np.isfinite(diagonal) or diagonal <= 0.0:
            raise ValueError("original mesh bounding box must have positive diagonal")
        padding = max(0.02, 0.25 * diagonal)
        origin = bbox_min - padding
        grid_max = bbox_max + padding
        grid_extent = grid_max - origin
        longest = float(np.max(grid_extent))
        shape = np.maximum(
            2, np.rint(grid_extent / longest * (resolution - 1)).astype(np.int64) + 1
        )
        voxel_size = grid_extent / (shape - 1)
        cache_descriptor = {
            "schema_version": GRID_SCHEMA_VERSION,
            "mesh_hash": mesh_hash,
            "profile_id": profile_id,
            "resolution": int(resolution),
            "padding_m": padding,
            "origin": origin.tolist(),
            "shape": shape.tolist(),
            "voxel_size": voxel_size.tolist(),
        }
        cache_key = _json_hash(cache_descriptor)
        cache_path: Path | None = None
        if cache_root is not None:
            cache_path = Path(cache_root).expanduser().resolve() / f"{cache_key}.npz"
            if cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as loaded:
                    stored = json.loads(str(loaded["metadata_json"].item()))
                    if stored.get("cache_key") != cache_key:
                        raise ValueError(f"signed-grid cache key mismatch: {cache_path}")
                    return cls(
                        signed_distance_grid=loaded["signed_distance_grid"],
                        gradient_grid=loaded["gradient_grid"],
                        origin=loaded["origin"],
                        voxel_size=loaded["voxel_size"],
                        bbox_min=loaded["bbox_min"],
                        bbox_max=loaded["bbox_max"],
                        vertices=vertices,
                        faces=faces,
                        mesh_hash=mesh_hash,
                        profile_id=profile_id,
                        metadata=stored,
                    )
        started = time.perf_counter()
        partial_path: Path | None = None
        progress_path: Path | None = None
        start_z = 0
        winding_device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                winding_device = "cuda"
        except ImportError:  # pragma: no cover - CPU-only minimal environments
            pass
        reference = ReferenceSignedDistanceBackend(
            vertices,
            faces,
            mesh_hash=mesh_hash,
            sign_mode="strict",
            query_chunk_size=min(max(int(query_batch_size), 1), 4096),
            face_chunk_size=4096,
            closest_acceleration="tree",
            # CUDA is an exact scheduling accelerator for the same winding
            # computation, not an alternate sign implementation.
            winding_device=winding_device,
            closest_device=None,
        )
        axes = tuple(
            origin[axis] + voxel_size[axis] * np.arange(shape[axis], dtype=np.float64)
            for axis in range(3)
        )
        grid_shape = tuple(int(value) for value in shape)
        if cache_path is None:
            grid: np.ndarray = np.empty(grid_shape, dtype=np.float64)
        else:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path = cache_path.with_suffix(".partial.npy")
            progress_path = cache_path.with_suffix(".partial.json")
            if partial_path.exists() and progress_path.exists():
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                if (
                    progress.get("cache_key") != cache_key
                    or tuple(progress.get("shape", [])) != grid_shape
                ):
                    raise ValueError(f"incompatible resumable signed-grid cache: {partial_path}")
                start_z = int(progress.get("completed_z", 0))
                if not 0 <= start_z <= grid_shape[2]:
                    raise ValueError(f"invalid resumable signed-grid progress: {progress_path}")
                grid = np.lib.format.open_memmap(partial_path, mode="r+", dtype=np.float64)
            else:
                grid = np.lib.format.open_memmap(
                    partial_path, mode="w+", dtype=np.float64, shape=grid_shape
                )

        def write_progress(completed_z: int) -> None:
            if progress_path is None:
                return
            payload = {
                "schema_version": GRID_SCHEMA_VERSION,
                "cache_key": cache_key,
                "shape": list(grid_shape),
                "completed_z": int(completed_z),
            }
            temporary = progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, progress_path)

        # Process z slabs to avoid materialising a second full 3-D point array.
        xy = np.stack(np.meshgrid(axes[0], axes[1], indexing="ij"), axis=-1).reshape(-1, 2)
        for z_index, z_value in enumerate(axes[2][start_z:], start=start_z):
            points = np.empty((len(xy), 3), dtype=np.float64)
            points[:, :2] = xy
            points[:, 2] = z_value
            values: list[np.ndarray] = []
            for start in range(0, len(points), query_batch_size):
                query_result = reference.query_local(points[start : start + query_batch_size])
                if not np.all(query_result.sign_valid) or not np.all(query_result.valid):
                    raise ValueError("strict reference winding produced an invalid grid sample")
                values.append(
                    np.asarray(query_result.signed_distance, dtype=np.float64).reshape(-1)
                )
            grid[:, :, z_index] = np.concatenate(values).reshape(grid.shape[:2])
            if partial_path is not None:
                assert isinstance(grid, np.memmap)
                grid.flush()
                write_progress(z_index + 1)
        gradients = np.stack(np.gradient(grid, *voxel_size, edge_order=2), axis=-1)
        metadata = {
            **cache_descriptor,
            "cache_key": cache_key,
            "backend_id": f"{profile_id}",
            "source_geometry": "original_strict_watertight_mesh",
            "sign_source": "reference_triangle_winding",
            "sign_convention": "positive_outside",
            "generation_time_s": time.perf_counter() - started,
            "memory_bytes": int(grid.nbytes + gradients.nbytes),
            "grid_node_count": int(grid.size),
            "cache_path": None if cache_path is None else str(cache_path),
        }
        backend = cls(
            signed_distance_grid=grid,
            gradient_grid=gradients,
            origin=origin,
            voxel_size=voxel_size,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            vertices=vertices,
            faces=faces,
            mesh_hash=mesh_hash,
            profile_id=profile_id,
            metadata=metadata,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = json.dumps(metadata, sort_keys=True)
            with tempfile.NamedTemporaryFile(
                prefix=f".{cache_path.stem}.", suffix=".npz", dir=cache_path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
            try:
                np.savez(
                    temporary,
                    signed_distance_grid=grid,
                    gradient_grid=gradients,
                    origin=origin,
                    voxel_size=voxel_size,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    metadata_json=np.asarray(descriptor),
                )
                os.replace(temporary, cache_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            if partial_path is not None and partial_path.exists():
                partial_path.unlink()
            if progress_path is not None and progress_path.exists():
                progress_path.unlink()
        return backend

    def audit(self) -> dict[str, Any]:
        return {
            "strict_original_mesh": True,
            "grid_shape": list(self.signed_distance_grid.shape),
            "out_of_grid_query_count": int(self.out_of_grid_query_count),
            "cache_key": self.metadata.get("cache_key"),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "backend_id": self.profile_id,
            "backend_kind": self.backend_id,
            "mesh_hash": self.mesh_hash,
            "profile_id": self.profile_id,
            "sign_convention": "positive_outside",
            "source_geometry": "original_strict_watertight_mesh",
            "sign_source": "reference_triangle_winding",
            "interpolation": "trilinear",
            "grid_shape": list(self.signed_distance_grid.shape),
            "voxel_size_m": self.voxel_size.tolist(),
            "padding_m": float(self.metadata["padding_m"]),
            "memory_bytes": int(self.metadata["memory_bytes"]),
            "generation_time_s": float(self.metadata["generation_time_s"]),
            "out_of_grid_query_count": int(self.out_of_grid_query_count),
            "cache_key": self.metadata["cache_key"],
            "cache_path": self.metadata.get("cache_path"),
        }

    def query_local(self, points_local: np.ndarray) -> SignedDistanceQueryResult:
        points = np.asarray(points_local, dtype=np.float64)
        shape = points.shape[:-1]
        if points.shape[-1:] != (3,):
            raise ValueError("signed-grid queries must have trailing dimension 3")
        flat = points.reshape(-1, 3)
        coordinates = (flat - self.origin) / self.voxel_size
        maximum_coordinate = np.asarray(self.signed_distance_grid.shape, dtype=np.float64) - 1.0
        in_grid = np.all((coordinates >= 0.0) & (coordinates <= maximum_coordinate), axis=1)
        signed = np.empty(len(flat), dtype=np.float64)
        normals = np.empty((len(flat), 3), dtype=np.float64)
        exterior_closest: np.ndarray | None = None
        if np.any(in_grid):
            local_coordinates = coordinates[in_grid]
            lower = np.floor(local_coordinates).astype(np.int64)
            # Queries on an upper grid face use the final valid cell with t=1.
            lower = np.minimum(lower, np.asarray(self.signed_distance_grid.shape) - 2)
            fractions = local_coordinates - lower
            signed[in_grid] = _trilinear(self.signed_distance_grid, lower, fractions)
            normals[in_grid] = _trilinear(self.gradient_grid, lower, fractions)
        if np.any(~in_grid):
            outside = flat[~in_grid]
            lower_delta = np.maximum(self.bbox_min - outside, 0.0)
            upper_delta = np.maximum(outside - self.bbox_max, 0.0)
            lower_bound = np.linalg.norm(lower_delta + upper_delta, axis=1)
            # A grid includes the padded bounding box; any out-of-grid point
            # must therefore be provably outside the original mesh bbox.
            if np.any(lower_bound <= 0.0):
                raise ValueError("out-of-grid query is not provably exterior; refusing to clip")
            self.out_of_grid_query_count += int(np.count_nonzero(~in_grid))
            exact = self._exterior_distance.query_local(outside)
            unsigned = np.asarray(exact.unsigned_distance, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(unsigned)) or np.any(unsigned <= 0.0):
                raise ValueError("provably exterior exact-distance query is invalid")
            # Exact exterior distance is itself a positive lower bound and
            # avoids falsely rejecting a grid solely because its finite
            # padding leaves far-away collision samples out of range.
            signed[~in_grid] = unsigned
            exterior_closest = np.asarray(exact.closest_points, dtype=np.float64).reshape(-1, 3)
            normals[~in_grid] = outside - exterior_closest
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        zero_gradient = norm[:, 0] <= 1e-12
        if np.any(zero_gradient):
            # A zero interpolated gradient is non-smooth; retain a finite,
            # deterministic normal and mark it invalid for analytic Jacobians.
            fallback = flat[zero_gradient] - 0.5 * (self.bbox_min + self.bbox_max)
            fallback_norm = np.linalg.norm(fallback, axis=1, keepdims=True)
            fallback[fallback_norm[:, 0] <= 1e-15] = np.array([1.0, 0.0, 0.0])
            normals[zero_gradient] = fallback / np.maximum(
                np.linalg.norm(fallback, axis=1, keepdims=True), 1e-15
            )
            norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(norm, 1e-15)
        closest = flat - signed[:, None] * normals
        if exterior_closest is not None:
            closest[~in_grid] = exterior_closest
        inside = signed < 0.0
        on_surface = np.abs(signed) <= 1e-8
        gradient_valid = (~zero_gradient) & in_grid
        return SignedDistanceQueryResult(
            signed_distance=signed.reshape(shape),
            unsigned_distance=np.abs(signed).reshape(shape),
            closest_points=closest.reshape((*shape, 3)),
            closest_face_indices=np.full(shape, -1, dtype=np.int64),
            closest_barycentric=np.full((*shape, 3), np.nan, dtype=np.float64),
            surface_normals=normals.reshape((*shape, 3)),
            inside=inside.reshape(shape),
            on_surface=on_surface.reshape(shape),
            valid=np.ones(shape, dtype=bool),
            sign_valid=np.ones(shape, dtype=bool),
            sign_confidence=np.ones(shape, dtype=np.float64),
            sign_method="precomputed_reference_winding_trilinear_grid",
            backend_id=self.profile_id,
            mesh_hash=self.mesh_hash,
            non_smooth=(zero_gradient | ~in_grid).reshape(shape),
            gradient_valid=gradient_valid.reshape(shape),
            geometry_metadata={
                "cache_key": self.metadata["cache_key"],
                "out_of_grid_query_count": int(self.out_of_grid_query_count),
            },
        )


__all__ = [
    "GRID_PROFILE_PREFIX",
    "GRID_SCHEMA_VERSION",
    "OriginalMeshSignedGridSDFBackend",
    "grid_resolution_from_profile",
]
