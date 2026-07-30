"""Portable compiled exact closest-point queries for ambiguous spatial FD.

This module is deliberately a narrow adapter: it leaves the established
hybrid sign backend, cache certificate, and final audit untouched.  The C++
handle owns only the immutable source-mesh BVH; generalized-winding signs
remain the already-qualified reference implementation.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any

import numpy as np

from toporetarget.geometry.se3 import invert_transform, transform_points, transform_vectors

from .base import SignedDistanceQueryResult
from .derived_proxy import HybridSignedDistanceBackend, _point_segment_distance
from .winding import winding_sign

COMPILED_SDF_CPU_BACKEND_ID = "compiled_sdf_cpu_v1"
COMPILED_SPATIAL_FD_BACKEND_ID = "compiled_spatial_central_fd_v1"
COMPILED_EXACT_SIGN_BACKEND_ID = "compiled_batched_generalized_winding_v1"


class CompiledSDFUnavailable(RuntimeError):
    """Raised when the optional local extension was not built."""


def _extension_path() -> Path:
    configured = os.environ.get("TOPORETARGET_COMPILED_SDF_CPU_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(__file__).resolve().parents[4]
    for build_id in ("compiled_exact_sign_v1", "compiled_sdf_cpu_v1"):
        for suffix in (".cpython-312-x86_64-linux-gnu.so", ".so"):
            candidate = root / ".local" / "build" / build_id / f"_compiled_sdf_cpu{suffix}"
            if candidate.is_file():
                return candidate
    return root / ".local" / "build" / "compiled_exact_sign_v1" / "_compiled_sdf_cpu.so"


def _load_extension() -> ModuleType:
    existing = sys.modules.get("_compiled_sdf_cpu")
    if existing is not None:
        return existing
    path = _extension_path()
    if not path.is_file():
        raise CompiledSDFUnavailable(
            "compiled SDF CPU extension is unavailable: "
            f"{path}; run scripts/build_compiled_sdf_cpu.py"
        )
    spec = importlib.util.spec_from_file_location("_compiled_sdf_cpu", path)
    if spec is None or spec.loader is None:
        raise CompiledSDFUnavailable(f"cannot load compiled SDF CPU extension: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_compiled_sdf_cpu"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("_compiled_sdf_cpu", None)
        raise
    return module


def compiled_available() -> bool:
    try:
        _load_extension()
    except (CompiledSDFUnavailable, ImportError, OSError):
        return False
    return True


def _require_points(value: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(value)
    if points.dtype != np.float64:
        raise TypeError(f"{name} must have dtype float64")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not points.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite")
    return points


class CompiledBVHHandle:
    """Persistent object-local exact BVH, backed by the compiled extension."""

    backend_id = COMPILED_SDF_CPU_BACKEND_ID

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, *, leaf_size: int = 32) -> None:
        vertex_array = _require_points(np.asarray(vertices), name="vertices")
        face_array = np.asarray(faces)
        if face_array.dtype != np.int64:
            raise TypeError("faces must have dtype int64")
        if face_array.ndim != 2 or face_array.shape[1] != 3 or not face_array.flags.c_contiguous:
            raise ValueError("faces must be a C-contiguous array with shape (F, 3)")
        if len(vertex_array) == 0 or len(face_array) == 0:
            raise ValueError("compiled BVH requires a non-empty mesh")
        if np.any(face_array < 0) or np.any(face_array >= len(vertex_array)):
            raise ValueError("faces contain an invalid vertex index")
        self.vertices = vertex_array
        self.faces = face_array
        self.leaf_size = int(leaf_size)
        self._native = _load_extension().CompiledBVHHandle(vertex_array, face_array, self.leaf_size)

    def query(
        self, points_object_float64: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._native.query(
            _require_points(points_object_float64, name="points_object_float64")
        )

    def stats(self) -> dict[str, int]:
        return {key: int(value) for key, value in self._native.stats().items()}


class CompiledGeneralizedWindingHandle:
    """Persistent float64, deterministic generalized-winding triangle handle."""

    backend_id = COMPILED_EXACT_SIGN_BACKEND_ID

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        vertex_array = _require_points(np.asarray(vertices), name="vertices")
        face_array = np.asarray(faces)
        if face_array.dtype != np.int64:
            raise TypeError("faces must have dtype int64")
        if face_array.ndim != 2 or face_array.shape[1] != 3 or not face_array.flags.c_contiguous:
            raise ValueError("faces must be a C-contiguous array with shape (F, 3)")
        if len(vertex_array) == 0 or len(face_array) == 0:
            raise ValueError("compiled winding requires a non-empty mesh")
        if np.any(face_array < 0) or np.any(face_array >= len(vertex_array)):
            raise ValueError("faces contain an invalid vertex index")
        extension = _load_extension()
        if not hasattr(extension, "CompiledGeneralizedWindingHandle"):
            raise CompiledSDFUnavailable(
                "compiled extension lacks generalized winding; rebuild with "
                "scripts/build_compiled_sdf_cpu.py"
            )
        self.vertices = vertex_array
        self.faces = face_array
        self._native = extension.CompiledGeneralizedWindingHandle(vertex_array, face_array)

    def query(self, points_object_float64: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._native.query(
                _require_points(points_object_float64, name="points_object_float64")
            ),
            dtype=np.float64,
        )

    def stats(self) -> dict[str, int]:
        return {key: int(value) for key, value in self._native.stats().items()}


def compiled_exact_query(
    handle: CompiledBVHHandle,
    points_object_float64: np.ndarray,
    *,
    return_face_ids: bool = True,
    return_barycentric: bool = True,
    return_closest_points: bool = True,
    return_unsigned_distance: bool = True,
) -> dict[str, np.ndarray]:
    """Run one exact batched query with explicit, validated float64 inputs."""

    closest, faces, barycentric, unsigned = handle.query(points_object_float64)
    output: dict[str, np.ndarray] = {}
    if return_face_ids:
        output["face_id"] = faces
    if return_barycentric:
        output["barycentric"] = barycentric
    if return_closest_points:
        output["closest_point"] = closest
    if return_unsigned_distance:
        output["unsigned_distance"] = unsigned
    return output


@dataclass
class CompiledSpatialFDResult:
    gradient_scene: np.ndarray
    probe_result: SignedDistanceQueryResult
    elapsed_s: float
    probe_count: int


class CompiledSpatialFDBackend:
    """Compiled source closest-point path paired with the reference sign path."""

    backend_id = COMPILED_SPATIAL_FD_BACKEND_ID

    def __init__(
        self,
        reference: HybridSignedDistanceBackend,
        *,
        leaf_size: int = 32,
        compiled_winding: bool = False,
        fd_probe_safety_margin: float = 1e-12,
    ) -> None:
        self.reference = reference
        self.handle = CompiledBVHHandle(
            np.ascontiguousarray(reference.original.vertices, dtype=np.float64),
            np.ascontiguousarray(reference.original.faces, dtype=np.int64),
            leaf_size=leaf_size,
        )
        self._source_normals = np.asarray(reference.original._face_normals, dtype=np.float64)
        self.compiled_winding = bool(compiled_winding)
        self.fd_probe_safety_margin = float(fd_probe_safety_margin)
        if not np.isfinite(self.fd_probe_safety_margin) or self.fd_probe_safety_margin < 0.0:
            raise ValueError("fd_probe_safety_margin must be finite and non-negative")
        self.winding_handle = (
            CompiledGeneralizedWindingHandle(
                np.ascontiguousarray(reference.proxy.vertices, dtype=np.float64),
                np.ascontiguousarray(reference.proxy.faces, dtype=np.int64),
            )
            if self.compiled_winding
            else None
        )
        self.probe_sign_stats: dict[str, int] = {
            "total_fd_probes": 0,
            "certified_probe_reuse": 0,
            "exact_probe_sign_calls": 0,
            "false_reuse": 0,
            "surface_crossing_count": 0,
            "invalidations": 0,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "exact_closest_point_backend": COMPILED_SDF_CPU_BACKEND_ID,
            "exact_sign_backend": (
                COMPILED_EXACT_SIGN_BACKEND_ID
                if self.compiled_winding
                else "reference_exact_generalized_winding_v2"
            ),
            "certified_fd_probe_sign_reuse": self.compiled_winding,
            "fallback_backend": "fast_exact_v2_python",
            "threads": 1,
            "stats": self.handle.stats(),
        }

    def _closest_local(
        self, points_local: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points = _require_points(np.asarray(points_local), name="points_object_float64")
        closest, local_faces, barycentric, unsigned = self.handle.query(points)
        source_faces = self.reference.geometry.source_distance_face_ids[local_faces]
        normals = self._source_normals[local_faces].copy()
        boundary_distance = _point_segment_distance(
            closest, self.reference.original_boundary_segments
        )
        return closest, source_faces, barycentric, unsigned, normals, boundary_distance

    def _sign_local(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return sign state, using reference only inside a strict threshold band."""
        if not self.compiled_winding:
            proxy = self.reference.proxy.query_local(points)
            assert proxy.inside is not None
            return (
                np.asarray(proxy.inside, dtype=bool).reshape(-1),
                np.asarray(proxy.sign_valid, dtype=bool).reshape(-1),
                np.asarray(proxy.sign_confidence, dtype=np.float64).reshape(-1),
                np.asarray(proxy.winding_value, dtype=np.float64).reshape(-1),
                np.asarray(proxy.closest_face_indices, dtype=np.int64).reshape(-1),
                np.full(len(points), "EXACT_GENERALIZED_WINDING", dtype="<U40"),
            )
        assert self.winding_handle is not None
        winding = self.winding_handle.query(points)
        inside, confidence, ambiguous, _magnitude = winding_sign(
            winding,
            threshold=self.reference.proxy.winding_threshold,
            confidence_threshold=self.reference.proxy.confidence_threshold,
        )
        fallback = ambiguous | ~np.isfinite(winding)
        valid = ~fallback
        faces = np.full(len(points), -1, dtype=np.int64)
        source = np.full(len(points), "COMPILED_GENERALIZED_WINDING", dtype="<U40")
        if np.any(fallback):
            reference = self.reference.proxy.query_local(points[fallback])
            assert reference.inside is not None
            inside[fallback] = np.asarray(reference.inside, dtype=bool).reshape(-1)
            valid[fallback] = np.asarray(reference.sign_valid, dtype=bool).reshape(-1)
            confidence[fallback] = np.asarray(reference.sign_confidence, dtype=np.float64).reshape(
                -1
            )
            if reference.winding_value is not None:
                winding[fallback] = np.asarray(reference.winding_value, dtype=np.float64).reshape(
                    -1
                )
            faces[fallback] = np.asarray(reference.closest_face_indices, dtype=np.int64).reshape(-1)
            source[fallback] = "REFERENCE_NEAR_THRESHOLD_FALLBACK"
        return inside, valid, confidence, winding, faces, source

    def _result_from_parts(
        self,
        *,
        closest: np.ndarray,
        source_faces: np.ndarray,
        barycentric: np.ndarray,
        unsigned: np.ndarray,
        normals: np.ndarray,
        boundary_distance: np.ndarray,
        inside: np.ndarray,
        sign_valid: np.ndarray,
        confidence: np.ndarray,
        winding: np.ndarray,
        proxy_faces: np.ndarray,
        sign_source: np.ndarray,
    ) -> SignedDistanceQueryResult:
        signed = np.where(inside, -unsigned, unsigned)
        near_boundary = boundary_distance <= self.reference.boundary_exclusion_radius_m
        synthetic = np.isin(
            proxy_faces, np.asarray(sorted(self.reference.synthetic_face_set), dtype=np.int64)
        )
        non_smooth = np.any(barycentric < 1e-7, axis=1)
        valid = sign_valid & np.isfinite(signed)
        return SignedDistanceQueryResult(
            signed_distance=signed,
            unsigned_distance=unsigned,
            closest_points=closest,
            closest_face_indices=source_faces,
            closest_barycentric=barycentric,
            surface_normals=normals,
            inside=inside,
            on_surface=unsigned <= self.reference.original.surface_tolerance,
            valid=valid,
            sign_valid=valid,
            sign_confidence=confidence,
            sign_method="hybrid_original_distance_proxy_sign_reference_exact",
            backend_id=self.backend_id,
            mesh_hash=self.reference.mesh_hash,
            winding_value=winding,
            non_smooth=non_smooth,
            gradient_valid=valid & ~non_smooth,
            proxy_closest_face_indices=proxy_faces,
            proxy_closest_is_synthetic_patch=synthetic,
            original_boundary_distance=boundary_distance,
            near_original_boundary=near_boundary,
            geometry_metadata=self.reference.geometry.compact_dict(),
            sign=np.where(signed >= 0.0, 1, -1).astype(np.int8),
            sign_source=sign_source,
            sign_reliable=valid.copy(),
        )

    def _query_local(self, points_local: np.ndarray) -> SignedDistanceQueryResult:
        points = _require_points(np.asarray(points_local), name="points_object_float64")
        closest, source_faces, barycentric, unsigned, normals, boundary_distance = (
            self._closest_local(points)
        )
        inside, sign_valid, confidence, winding, proxy_faces, sign_source = self._sign_local(points)
        return self._result_from_parts(
            closest=closest,
            source_faces=source_faces,
            barycentric=barycentric,
            unsigned=unsigned,
            normals=normals,
            boundary_distance=boundary_distance,
            inside=inside,
            sign_valid=sign_valid,
            confidence=confidence,
            winding=winding,
            proxy_faces=proxy_faces,
            sign_source=sign_source,
        )

    def spatial_fd_gradient_scene(
        self, base_points_scene: np.ndarray, object_pose_scene: np.ndarray, fd_step: float
    ) -> CompiledSpatialFDResult:
        points = _require_points(np.asarray(base_points_scene), name="base_points_scene")
        step = float(fd_step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("fd_step must be finite and positive")
        started = perf_counter()
        local = np.ascontiguousarray(transform_points(invert_transform(object_pose_scene), points))
        offsets = np.concatenate((np.eye(3), -np.eye(3)), axis=0)
        probes = np.ascontiguousarray(
            (local[:, None, :] + step * offsets[None, :, :]).reshape(-1, 3)
        )
        if not self.compiled_winding:
            result = self._query_local(probes)
        else:
            base = self._query_local(local)
            reuse_rows = np.asarray(base.sign_reliable, dtype=bool) & (
                np.abs(np.asarray(base.signed_distance, dtype=np.float64))
                > step + self.fd_probe_safety_margin
            )
            closest, source_faces, barycentric, unsigned, normals, boundary_distance = (
                self._closest_local(probes)
            )
            row_ids = np.repeat(np.arange(len(points), dtype=np.int64), 6)
            reuse = reuse_rows[row_ids]
            inside = np.empty(len(probes), dtype=bool)
            sign_valid = np.empty(len(probes), dtype=bool)
            confidence = np.empty(len(probes), dtype=np.float64)
            winding = np.full(len(probes), np.nan, dtype=np.float64)
            proxy_faces = np.full(len(probes), -1, dtype=np.int64)
            sign_source = np.full(len(probes), "CERTIFIED_FD_PROBE_REUSE", dtype="<U40")
            base_inside = np.asarray(base.inside, dtype=bool).reshape(-1)
            inside[reuse] = base_inside[row_ids[reuse]]
            sign_valid[reuse] = True
            confidence[reuse] = 1.0
            unresolved = ~reuse
            if np.any(unresolved):
                fields = self._sign_local(probes[unresolved])
                (
                    inside[unresolved],
                    sign_valid[unresolved],
                    confidence[unresolved],
                    winding[unresolved],
                    proxy_faces[unresolved],
                    sign_source[unresolved],
                ) = fields
            self.probe_sign_stats["total_fd_probes"] += int(len(probes))
            self.probe_sign_stats["certified_probe_reuse"] += int(np.count_nonzero(reuse))
            self.probe_sign_stats["exact_probe_sign_calls"] += int(np.count_nonzero(unresolved))
            self.probe_sign_stats["invalidations"] += int(np.count_nonzero(~reuse_rows))
            result = self._result_from_parts(
                closest=closest,
                source_faces=source_faces,
                barycentric=barycentric,
                unsigned=unsigned,
                normals=normals,
                boundary_distance=boundary_distance,
                inside=inside,
                sign_valid=sign_valid,
                confidence=confidence,
                winding=winding,
                proxy_faces=proxy_faces,
                sign_source=sign_source,
            )
        phi = np.asarray(result.signed_distance, dtype=np.float64).reshape(len(points), 6)
        gradient_local = (phi[:, :3] - phi[:, 3:]) / (2.0 * step)
        gradient_scene = transform_vectors(object_pose_scene, gradient_local)
        return CompiledSpatialFDResult(
            gradient_scene=gradient_scene,
            probe_result=result,
            elapsed_s=perf_counter() - started,
            probe_count=int(len(probes)),
        )


__all__ = [
    "COMPILED_SDF_CPU_BACKEND_ID",
    "COMPILED_SPATIAL_FD_BACKEND_ID",
    "CompiledBVHHandle",
    "CompiledGeneralizedWindingHandle",
    "COMPILED_EXACT_SIGN_BACKEND_ID",
    "CompiledSDFUnavailable",
    "CompiledSpatialFDBackend",
    "CompiledSpatialFDResult",
    "compiled_available",
    "compiled_exact_query",
]
