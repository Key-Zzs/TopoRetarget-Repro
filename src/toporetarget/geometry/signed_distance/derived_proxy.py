"""Auditable sign proxies and hybrid signed-distance queries for open meshes.

The original mesh remains the only source for closest points and distance
magnitudes.  A derived watertight mesh is used only to classify the sign.  The
policy is intentionally object-agnostic: a valid source uses identity, while
an open source first receives deterministic local boundary repair and only
then the fixed voxel fallback.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.geometry.mesh_audit import MeshAuditReport, audit_mesh
from toporetarget.geometry.se3 import invert_transform, transform_points, transform_vectors

from .base import SignedDistanceBackend, SignedDistanceQueryResult
from .closest_point import TriangleAABBTree, closest_points_on_triangles
from .reference import ReferenceSignedDistanceBackend

DERIVED_SDF_PROXY_SCHEMA_VERSION = "toporetarget.derived_sdf_proxy.v1"
HYBRID_SIGNED_DISTANCE_PROFILE_ID = "hybrid_original_distance_proxy_sign_v1"
DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[4]
    / "configs"
    / "geometry"
    / "sdf_backends"
    / f"{HYBRID_SIGNED_DISTANCE_PROFILE_ID}.yaml"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    )


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ObjectSDFGeometryPolicy:
    """Versioned, object-independent geometry policy."""

    profile_id: str
    schema_version: str
    degenerate_area_threshold_m2: float
    surface_sample_count: int
    surface_sample_seed: int
    boundary_exclusion_min_m: float
    boundary_exclusion_bbox_scale: float
    patch_offset_bbox_scale: float
    patch_offset_min_m: float
    voxel_longest_axis_resolution: int
    voxel_closing_voxels: int
    surface_p95_floor_m: float
    surface_p95_bbox_scale: float
    surface_max_floor_m: float
    surface_max_bbox_scale: float
    bbox_extent_relative_tolerance: float
    max_patch_area_ratio: float
    policy_hash: str
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> ObjectSDFGeometryPolicy:
        source = Path(path or DEFAULT_POLICY_PATH).expanduser().resolve()
        raw = source.read_bytes()
        values = yaml.safe_load(raw) or {}
        if not isinstance(values, dict):
            raise ValueError(f"geometry policy must be a mapping: {source}")
        profile_id = str(values.get("profile_id", HYBRID_SIGNED_DISTANCE_PROFILE_ID))
        schema_version = str(values.get("schema_version", DERIVED_SDF_PROXY_SCHEMA_VERSION))
        policy = cls(
            profile_id=profile_id,
            schema_version=schema_version,
            degenerate_area_threshold_m2=float(values.get("degenerate_area_threshold_m2", 1e-12)),
            surface_sample_count=int(values.get("surface_sample_count", 20000)),
            surface_sample_seed=int(values.get("surface_sample_seed", 20260724)),
            boundary_exclusion_min_m=float(values.get("boundary_exclusion_min_m", 0.002)),
            boundary_exclusion_bbox_scale=float(values.get("boundary_exclusion_bbox_scale", 0.01)),
            patch_offset_bbox_scale=float(values.get("patch_offset_bbox_scale", 0.001)),
            patch_offset_min_m=float(values.get("patch_offset_min_m", 1e-5)),
            voxel_longest_axis_resolution=int(values.get("voxel_longest_axis_resolution", 256)),
            voxel_closing_voxels=int(values.get("voxel_closing_voxels", 1)),
            surface_p95_floor_m=float(values.get("surface_p95_floor_m", 0.001)),
            surface_p95_bbox_scale=float(values.get("surface_p95_bbox_scale", 0.005)),
            surface_max_floor_m=float(values.get("surface_max_floor_m", 0.003)),
            surface_max_bbox_scale=float(values.get("surface_max_bbox_scale", 0.015)),
            bbox_extent_relative_tolerance=float(
                values.get("bbox_extent_relative_tolerance", 0.01)
            ),
            max_patch_area_ratio=float(values.get("max_patch_area_ratio", 0.05)),
            policy_hash=_sha256_bytes(raw),
            source_path=source,
        )
        if policy.profile_id != HYBRID_SIGNED_DISTANCE_PROFILE_ID:
            raise ValueError(f"unsupported geometry policy: {policy.profile_id}")
        if policy.surface_sample_count < 20000:
            raise ValueError("surface sample count must be at least 20000")
        if policy.voxel_longest_axis_resolution <= 0 or policy.voxel_closing_voxels < 0:
            raise ValueError("invalid voxel fallback policy")
        return policy

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "schema_version": self.schema_version,
            "degenerate_area_threshold_m2": self.degenerate_area_threshold_m2,
            "surface_sample_count": self.surface_sample_count,
            "surface_sample_seed": self.surface_sample_seed,
            "boundary_exclusion_min_m": self.boundary_exclusion_min_m,
            "boundary_exclusion_bbox_scale": self.boundary_exclusion_bbox_scale,
            "patch_offset_bbox_scale": self.patch_offset_bbox_scale,
            "patch_offset_min_m": self.patch_offset_min_m,
            "voxel_longest_axis_resolution": self.voxel_longest_axis_resolution,
            "voxel_closing_voxels": self.voxel_closing_voxels,
            "surface_p95_floor_m": self.surface_p95_floor_m,
            "surface_p95_bbox_scale": self.surface_p95_bbox_scale,
            "surface_max_floor_m": self.surface_max_floor_m,
            "surface_max_bbox_scale": self.surface_max_bbox_scale,
            "bbox_extent_relative_tolerance": self.bbox_extent_relative_tolerance,
            "max_patch_area_ratio": self.max_patch_area_ratio,
            "policy_hash": self.policy_hash,
        }


def _valid_face_mask(
    vertices: np.ndarray, faces: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(vertices, dtype=np.float64)
    face_value = np.asarray(faces, dtype=np.int64)
    valid = np.all(face_value >= 0, axis=1) & np.all(face_value < len(value), axis=1)
    areas = np.zeros(len(face_value), dtype=np.float64)
    if np.any(valid):
        tri = value[face_value[valid]]
        areas[valid] = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        )
    return valid & (areas > threshold), areas


def _deduplicate_faces(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seen: set[tuple[int, int, int]] = set()
    keep: list[int] = []
    for index, face in enumerate(np.asarray(faces, dtype=np.int64)):
        sorted_face = sorted((int(face[0]), int(face[1]), int(face[2])))
        key = (sorted_face[0], sorted_face[1], sorted_face[2])
        if key in seen:
            continue
        seen.add(key)
        keep.append(index)
    return np.asarray(faces, dtype=np.int64)[keep], np.asarray(keep, dtype=np.int64)


def _deduplicate_vertices(
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(vertices, dtype=np.float64)
    unique, first_ids, inverse = np.unique(value, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first_ids, kind="stable")
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order), dtype=np.int64)
    return unique[order], remap[inverse], order


def _orient_consistently(faces: np.ndarray) -> np.ndarray:
    """Orient each connected triangle component deterministically."""

    value = np.asarray(faces, dtype=np.int64).copy()
    edge_map: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for index, face in enumerate(value):
        for start, end in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            key = (min(start, end), max(start, end))
            sign = 1 if (start, end) == key else -1
            edge_map[key].append((index, sign))
    graph: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    for entries in edge_map.values():
        if len(entries) != 2:
            continue
        (first, first_sign), (second, second_sign) = entries
        same_direction = first_sign == second_sign
        graph[first].append((second, same_direction))
        graph[second].append((first, same_direction))
    flipped: dict[int, bool] = {}
    for root in range(len(value)):
        if root in flipped:
            continue
        flipped[root] = False
        queue: deque[int] = deque([root])
        while queue:
            current = queue.popleft()
            for other, same_direction in graph[current]:
                expected = flipped[current] ^ same_direction
                if other in flipped:
                    if flipped[other] != expected:
                        raise ValueError("mesh orientation is not orientable")
                else:
                    flipped[other] = expected
                    queue.append(other)
    for index, should_flip in flipped.items():
        if should_flip:
            value[index] = value[index][[0, 2, 1]]
    return value


def _boundary_loops(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[
    list[list[int]],
    list[tuple[int, int]],
    dict[tuple[int, int], list[tuple[int, int, int]]],
]:
    edge_records: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for face_index, face in enumerate(np.asarray(faces, dtype=np.int64)):
        for start, end in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge_records[(min(start, end), max(start, end))].append((start, end, face_index))
    boundary = sorted(key for key, values in edge_records.items() if len(values) == 1)
    adjacency: dict[int, list[int]] = defaultdict(list)
    for start, end in boundary:
        adjacency[start].append(end)
        adjacency[end].append(start)
    if any(len(neighbours) != 2 for neighbours in adjacency.values()):
        raise ValueError("boundary does not decompose into degree-two loops")
    loops: list[list[int]] = []
    seen: set[int] = set()
    for root in sorted(adjacency):
        if root in seen:
            continue
        loop: list[int] = []
        previous = -1
        current = root
        while True:
            loop.append(current)
            seen.add(current)
            candidates = sorted(
                item for item in adjacency[current] if item != previous and item not in loop
            )
            next_vertex = candidates[0] if candidates else root
            if next_vertex == root:
                break
            previous, current = current, next_vertex
        loops.append(loop)
    return loops, boundary, edge_records


def _clean_source_mesh(
    vertices: np.ndarray, faces: np.ndarray, policy: ObjectSDFGeometryPolicy
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(vertices, dtype=np.float64)
    raw_faces = np.asarray(faces, dtype=np.int64)
    valid, _ = _valid_face_mask(value, raw_faces, policy.degenerate_area_threshold_m2)
    face_ids = np.flatnonzero(valid)
    proxy_vertices, vertex_inverse, _ = _deduplicate_vertices(value)
    remapped_faces = vertex_inverse[raw_faces[valid]]
    cleaned, first_ids = _deduplicate_faces(remapped_faces)
    source_face_ids = face_ids[first_ids]
    cleaned = _orient_consistently(cleaned)
    return proxy_vertices.copy(), cleaned, source_face_ids, face_ids


def _polygon_normal(points: np.ndarray) -> np.ndarray:
    normal = np.zeros(3, dtype=np.float64)
    for index in range(len(points)):
        normal += np.cross(points[index], points[(index + 1) % len(points)])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-15:
        return np.zeros(3, dtype=np.float64)
    return normal / norm


def _local_repair(
    vertices: np.ndarray, faces: np.ndarray, policy: ObjectSDFGeometryPolicy
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[list[int]],
    dict[str, Any],
]:
    proxy_vertices, cleaned_faces, source_face_ids, _ = _clean_source_mesh(vertices, faces, policy)
    loops, boundary_edges, edge_records = _boundary_loops(proxy_vertices, cleaned_faces)
    if not loops:
        return (
            proxy_vertices,
            cleaned_faces,
            source_face_ids,
            np.empty(0, dtype=np.int64),
            [],
            {
                "boundary_edges": [],
                "patch_offset_m": 0.0,
                "patch_area_m2": 0.0,
                "patch_candidates": [],
            },
        )
    patch_offset = max(
        policy.patch_offset_min_m,
        policy.patch_offset_bbox_scale * float(np.linalg.norm(np.ptp(proxy_vertices, axis=0))),
    )
    output_vertices = list(proxy_vertices)
    output_faces = list(cleaned_faces.tolist())
    output_source_ids = list(source_face_ids.tolist())
    synthetic_ids: list[int] = []
    patch_area = 0.0
    patch_candidates: list[dict[str, Any]] = []
    for loop in loops:
        points = proxy_vertices[loop]
        normal = np.zeros(3, dtype=np.float64)
        adjacent_face_ids: set[int] = set()
        for index in range(len(loop)):
            start = loop[index]
            end = loop[(index + 1) % len(loop)]
            key = (min(start, end), max(start, end))
            for _start, _end, face_id in edge_records[key]:
                adjacent_face_ids.add(face_id)
        for face_id in sorted(adjacent_face_ids):
            triangle = proxy_vertices[cleaned_faces[face_id]]
            normal += np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if np.linalg.norm(normal) <= 1e-15:
            normal = _polygon_normal(points)
        normal /= max(float(np.linalg.norm(normal)), 1e-15)
        center_index = len(output_vertices)
        output_vertices.append(points.mean(axis=0) + normal * patch_offset)
        patch_candidates.append(
            {
                "boundary_loop_index": len(patch_candidates),
                "boundary_vertex_ids": [int(item) for item in loop],
                "candidate_center": np.asarray(output_vertices[-1]).tolist(),
                "normal": np.asarray(normal).tolist(),
                "offset_m": patch_offset,
            }
        )
        same_direction_count = 0
        for index in range(len(loop)):
            start = loop[index]
            end = loop[(index + 1) % len(loop)]
            key = (min(start, end), max(start, end))
            source_direction = edge_records[key][0][:2]
            same_direction_count += int(
                source_direction == (loop[index], loop[(index + 1) % len(loop)])
            )
        if same_direction_count == len(loop):
            loop = list(reversed(loop))
        elif same_direction_count not in {0, len(loop)}:
            raise ValueError("boundary loop orientation is ambiguous")
        for index in range(len(loop)):
            face = (loop[index], loop[(index + 1) % len(loop)], center_index)
            output_faces.append(face)
            output_source_ids.append(-1)
            synthetic_ids.append(len(output_faces) - 1)
            triangle = np.asarray(output_vertices, dtype=np.float64)[list(face)]
            patch_area += 0.5 * float(
                np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
            )
    output_faces_array = np.asarray(output_faces, dtype=np.int64)
    output_vertices_array = np.asarray(output_vertices, dtype=np.float64)
    output_faces_array = _orient_consistently(output_faces_array)
    audit = audit_mesh(output_vertices_array, output_faces_array)
    if audit.signed_volume is not None and audit.signed_volume < 0:
        output_faces_array = output_faces_array[:, [0, 2, 1]]
    return (
        output_vertices_array,
        output_faces_array,
        np.asarray(output_source_ids, dtype=np.int64),
        np.asarray(synthetic_ids, dtype=np.int64),
        loops,
        {
            "boundary_edges": [list(edge) for edge in boundary_edges],
            "patch_offset_m": patch_offset,
            "patch_area_m2": patch_area,
            "patch_candidates": patch_candidates,
        },
    )


def _surface_samples(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    triangle = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    area = 0.5 * np.linalg.norm(
        np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1
    )
    probabilities = area / np.sum(area)
    rng = np.random.default_rng(seed)
    face_ids = rng.choice(len(triangle), size=count, p=probabilities)
    first = np.sqrt(rng.random(count))
    second = rng.random(count)
    return (
        (1.0 - first)[:, None] * triangle[face_ids, 0]
        + first[:, None] * (1.0 - second)[:, None] * triangle[face_ids, 1]
        + first[:, None] * second[:, None] * triangle[face_ids, 2]
    )


def _surface_deviation(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    proxy_vertices: np.ndarray,
    proxy_faces: np.ndarray,
    policy: ObjectSDFGeometryPolicy,
) -> dict[str, Any]:
    source_audit = audit_mesh(source_vertices, source_faces)
    source_samples = _surface_samples(
        source_vertices,
        source_faces,
        policy.surface_sample_count,
        policy.surface_sample_seed,
    )
    proxy_samples = _surface_samples(
        proxy_vertices,
        proxy_faces,
        policy.surface_sample_count,
        policy.surface_sample_seed + 1,
    )
    source_tree = TriangleAABBTree(source_vertices[source_faces])
    proxy_tree = TriangleAABBTree(proxy_vertices[proxy_faces])
    source_to_proxy = closest_points_on_triangles(
        source_samples, proxy_vertices[proxy_faces], tree=proxy_tree, query_chunk_size=256
    )[3]
    proxy_to_source = closest_points_on_triangles(
        proxy_samples, source_vertices[source_faces], tree=source_tree, query_chunk_size=256
    )[3]
    source_extent = np.ptp(source_vertices, axis=0)
    proxy_extent = np.ptp(proxy_vertices, axis=0)
    bbox_relative_error = float(
        np.max(np.abs(proxy_extent - source_extent) / np.maximum(source_extent, 1e-15))
    )
    symmetric = np.concatenate((source_to_proxy, proxy_to_source))
    diagonal = float(source_audit.bounding_box_diagonal or 0.0)
    p95_gate = max(policy.surface_p95_floor_m, policy.surface_p95_bbox_scale * diagonal)
    max_gate = max(policy.surface_max_floor_m, policy.surface_max_bbox_scale * diagonal)
    return {
        "sample_count_per_direction": policy.surface_sample_count,
        "seed_original": policy.surface_sample_seed,
        "seed_proxy": policy.surface_sample_seed + 1,
        "original_to_proxy": {
            "median_m": float(np.median(source_to_proxy)),
            "p95_m": float(np.quantile(source_to_proxy, 0.95)),
            "max_m": float(np.max(source_to_proxy)),
        },
        "proxy_to_original": {
            "median_m": float(np.median(proxy_to_source)),
            "p95_m": float(np.quantile(proxy_to_source, 0.95)),
            "max_m": float(np.max(proxy_to_source)),
        },
        "symmetric_chamfer_mean_m": float(np.mean(symmetric)),
        "bbox_extent_relative_error": bbox_relative_error,
        "bbox_diagonal_m": diagonal,
        "gates": {
            "p95_limit_m": p95_gate,
            "max_limit_m": max_gate,
            "bbox_extent_relative_limit": policy.bbox_extent_relative_tolerance,
            "p95_pass": float(np.quantile(symmetric, 0.95)) <= p95_gate,
            "max_pass": float(np.max(symmetric)) <= max_gate,
            "bbox_pass": bbox_relative_error <= policy.bbox_extent_relative_tolerance,
        },
    }


@dataclass
class DerivedWatertightSignProxy:
    source_audit: MeshAuditReport
    proxy_audit: MeshAuditReport
    candidate_id: str
    candidate_method: str
    source_vertices: np.ndarray
    source_faces: np.ndarray
    source_distance_vertices: np.ndarray
    proxy_vertices: np.ndarray
    proxy_faces: np.ndarray
    proxy_source_face_ids: np.ndarray
    source_distance_faces: np.ndarray
    source_distance_face_ids: np.ndarray
    synthetic_face_ids: np.ndarray
    removed_face_ids: np.ndarray
    near_zero_face_ids: np.ndarray
    boundary_loops: list[list[int]]
    boundary_edges: list[tuple[int, int]]
    patch_candidates: list[dict[str, Any]]
    patch_area_m2: float
    surface_deviation: dict[str, Any]
    policy: ObjectSDFGeometryPolicy
    source_path: str | None = None

    @property
    def source_mesh_hash(self) -> str:
        return self.source_audit.mesh_hash

    @property
    def proxy_mesh_hash(self) -> str:
        return self.proxy_audit.mesh_hash

    @property
    def cache_signature(self) -> str:
        return _json_hash(
            {
                "source_mesh_hash": self.source_mesh_hash,
                "proxy_mesh_hash": self.proxy_mesh_hash,
                "profile_id": self.policy.profile_id,
                "policy_hash": self.policy.policy_hash,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DERIVED_SDF_PROXY_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "candidate_method": self.candidate_method,
            "source_path": self.source_path,
            "source_mesh_hash": self.source_mesh_hash,
            "proxy_mesh_hash": self.proxy_mesh_hash,
            "proxy_used_for_sign_only": True,
            "paper_unspecified_geometry_engineering": True,
            "raw_asset_modified": False,
            "source_audit": self.source_audit.as_dict(),
            "proxy_audit": self.proxy_audit.as_dict(),
            "proxy_source_face_ids": self.proxy_source_face_ids.tolist(),
            "source_distance_face_ids": self.source_distance_face_ids.tolist(),
            "synthetic_face_ids": self.synthetic_face_ids.tolist(),
            "removed_face_ids": self.removed_face_ids.tolist(),
            "near_zero_face_ids": self.near_zero_face_ids.tolist(),
            "boundary_loops": self.boundary_loops,
            "boundary_edges": [list(edge) for edge in self.boundary_edges],
            "boundary_loop_positions": [
                self.source_distance_vertices[np.asarray(loop, dtype=np.int64)].tolist()
                for loop in self.boundary_loops
            ],
            "patch_candidates": self.patch_candidates,
            "patch_area_m2": self.patch_area_m2,
            "patch_area_ratio": self.patch_area_m2
            / max(float(self.source_audit.surface_area or 0.0), 1e-15),
            "surface_deviation": self.surface_deviation,
            "policy": self.policy.as_dict(),
            "cache_signature": self.cache_signature,
        }

    def compact_dict(self, *, artifact_root: str | None = None) -> dict[str, Any]:
        """Return checkpoint-safe provenance without embedding face-ID arrays."""

        return {
            "schema_version": DERIVED_SDF_PROXY_SCHEMA_VERSION,
            "profile_id": self.policy.profile_id,
            "policy_hash": self.policy.policy_hash,
            "candidate_id": self.candidate_id,
            "candidate_method": self.candidate_method,
            "source_mesh_hash": self.source_mesh_hash,
            "proxy_mesh_hash": self.proxy_mesh_hash,
            "cache_signature": self.cache_signature,
            "proxy_used_for_sign_only": True,
            "paper_unspecified_geometry_engineering": True,
            "raw_asset_modified": False,
            "artifact_root": artifact_root,
            "source_audit": self.source_audit.as_dict(),
            "proxy_audit": self.proxy_audit.as_dict(),
            "surface_deviation": self.surface_deviation,
            "patch_area_m2": self.patch_area_m2,
            "patch_area_ratio": self.patch_area_m2
            / max(float(self.source_audit.surface_area or 0.0), 1e-15),
            "boundary_loop_count": len(self.boundary_loops),
            "synthetic_face_count": int(len(self.synthetic_face_ids)),
            "near_zero_face_count": int(len(self.near_zero_face_ids)),
        }


def _candidate_is_accepted(
    proxy: DerivedWatertightSignProxy, policy: ObjectSDFGeometryPolicy
) -> bool:
    audit = proxy.proxy_audit
    deviation = proxy.surface_deviation
    gates = deviation.get("gates", {})
    patch_ratio = proxy.patch_area_m2 / max(float(proxy.source_audit.surface_area or 0.0), 1e-15)
    return bool(
        audit.watertight
        and audit.winding_consistent is True
        and audit.orientable is True
        and audit.boundary_edge_count == 0
        and audit.non_manifold_edge_count == 0
        and audit.near_zero_area_faces == 0
        and audit.signed_volume is not None
        and audit.signed_volume > 0
        and bool(gates.get("p95_pass", False))
        and bool(gates.get("max_pass", False))
        and bool(gates.get("bbox_pass", False))
        and patch_ratio <= policy.max_patch_area_ratio
    )


def _voxel_fallback(
    vertices: np.ndarray, faces: np.ndarray, policy: ObjectSDFGeometryPolicy
) -> tuple[np.ndarray, np.ndarray]:
    """Build the fixed-resolution fallback without using a convex hull."""

    try:
        import trimesh
        from skimage.measure import marching_cubes
    except ImportError as exc:  # pragma: no cover - optional fallback dependency
        raise RuntimeError("voxel sign proxy requires trimesh and scikit-image") from exc
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    diagonal = float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))
    pitch = diagonal / max(policy.voxel_longest_axis_resolution - 1, 1)
    if pitch <= 0:
        raise ValueError("cannot voxelize a zero-size mesh")
    voxels = mesh.voxelized(pitch)
    if policy.voxel_closing_voxels:
        voxels = voxels.fill()
    matrix = np.asarray(voxels.matrix, dtype=np.uint8)
    if matrix.ndim != 3 or not np.any(matrix):
        raise ValueError("voxel fallback produced an empty occupancy grid")
    volume = matrix
    if policy.voxel_closing_voxels:
        from scipy import ndimage

        structure = ndimage.generate_binary_structure(3, 1)
        for _ in range(policy.voxel_closing_voxels):
            volume = ndimage.binary_closing(volume, structure=structure)
    padded = np.pad(volume.astype(np.float32), 1, mode="constant")
    proxy_vertices, proxy_faces, _, _ = marching_cubes(padded, level=0.5, spacing=(pitch,) * 3)
    voxel_origin = getattr(voxels, "origin", None)
    if voxel_origin is None:
        voxel_origin = np.asarray(voxels.transform[:3, 3], dtype=np.float64)
    origin = np.asarray(voxel_origin, dtype=np.float64) - pitch
    return (
        np.asarray(proxy_vertices, dtype=np.float64) + origin,
        np.asarray(proxy_faces, dtype=np.int64),
    )


def build_derived_sign_proxy(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    policy: ObjectSDFGeometryPolicy | None = None,
    source_path: str | Path | None = None,
) -> DerivedWatertightSignProxy:
    """Select identity, local repair, or fixed voxel fallback deterministically."""

    selected_policy = policy or ObjectSDFGeometryPolicy.load()
    source_vertices = np.asarray(vertices, dtype=np.float64).copy()
    source_faces = np.asarray(faces, dtype=np.int64).copy()
    source_audit = audit_mesh(
        source_vertices,
        source_faces,
        source_path=source_path,
        degenerate_area_threshold=selected_policy.degenerate_area_threshold_m2,
    )
    source_valid_faces = source_faces[
        _valid_face_mask(
            source_vertices,
            source_faces,
            selected_policy.degenerate_area_threshold_m2,
        )[0]
    ]
    valid_mask, _ = _valid_face_mask(
        source_vertices,
        source_faces,
        selected_policy.degenerate_area_threshold_m2,
    )
    valid_indices = np.all(source_faces >= 0, axis=1) & np.all(
        source_faces < len(source_vertices), axis=1
    )
    near_zero_face_ids = np.flatnonzero(valid_indices & ~valid_mask).astype(np.int64)
    source_distance_vertices = source_vertices.copy()
    if (
        source_audit.watertight
        and source_audit.winding_consistent is True
        and source_audit.orientable is True
        and source_audit.non_manifold_edge_count == 0
        and source_audit.near_zero_area_faces == 0
        and source_audit.signed_volume is not None
        and source_audit.signed_volume > 0
    ):
        proxy_vertices = source_vertices.copy()
        proxy_faces = source_faces.copy()
        proxy_source_ids = np.arange(len(proxy_faces), dtype=np.int64)
        synthetic_ids = np.empty(0, dtype=np.int64)
        removed_ids = np.empty(0, dtype=np.int64)
        source_distance_faces = source_faces.copy()
        source_distance_face_ids = np.arange(len(source_faces), dtype=np.int64)
        boundary_edges: list[tuple[int, int]] = []
        patch_candidates: list[dict[str, Any]] = []
        loops: list[list[int]] = []
        candidate_id = "candidate_0_identity"
        candidate_method = "identity"
        patch_area = 0.0
    else:
        (
            proxy_vertices,
            proxy_faces,
            proxy_source_ids,
            synthetic_ids,
            loops,
            repair_meta,
        ) = _local_repair(source_vertices, source_faces, selected_policy)
        source_distance_vertices, _, _ = _deduplicate_vertices(source_vertices)
        boundary_edges = [(int(edge[0]), int(edge[1])) for edge in repair_meta["boundary_edges"]]
        patch_candidates = list(repair_meta.get("patch_candidates", []))
        source_face_mask = proxy_source_ids >= 0
        source_distance_faces = proxy_faces[source_face_mask].copy()
        source_distance_face_ids = proxy_source_ids[source_face_mask].copy()
        removed_ids = np.setdiff1d(
            np.arange(len(source_faces), dtype=np.int64),
            np.asarray(proxy_source_ids[proxy_source_ids >= 0], dtype=np.int64),
            assume_unique=False,
        )
        proxy_audit = audit_mesh(proxy_vertices, proxy_faces)
        temporary = DerivedWatertightSignProxy(
            source_audit,
            proxy_audit,
            "candidate_1_local_repair",
            "topological_local_repair",
            source_vertices,
            source_faces,
            source_distance_vertices,
            proxy_vertices,
            proxy_faces,
            proxy_source_ids,
            source_distance_faces,
            source_distance_face_ids,
            synthetic_ids,
            removed_ids,
            near_zero_face_ids,
            loops,
            boundary_edges,
            patch_candidates,
            float(repair_meta["patch_area_m2"]),
            _surface_deviation(
                source_vertices,
                source_valid_faces,
                proxy_vertices,
                proxy_faces,
                selected_policy,
            ),
            selected_policy,
            None if source_path is None else str(Path(source_path).expanduser()),
        )
        if _candidate_is_accepted(temporary, selected_policy):
            return temporary
        proxy_vertices, proxy_faces = _voxel_fallback(
            source_vertices, source_faces, selected_policy
        )
        proxy_source_ids = np.full(len(proxy_faces), -1, dtype=np.int64)
        synthetic_ids = np.arange(len(proxy_faces), dtype=np.int64)
        candidate_id = "candidate_2_voxel_fallback"
        candidate_method = "fixed_resolution_voxel_marching_cubes"
        patch_area = 0.0
        removed_ids = np.empty(0, dtype=np.int64)
        source_distance_faces = proxy_faces[:0].copy()
        source_distance_face_ids = np.empty(0, dtype=np.int64)
    proxy_audit = audit_mesh(proxy_vertices, proxy_faces)
    result = DerivedWatertightSignProxy(
        source_audit,
        proxy_audit,
        candidate_id,
        candidate_method,
        source_vertices,
        source_faces,
        source_distance_vertices,
        proxy_vertices,
        proxy_faces,
        proxy_source_ids,
        source_distance_faces,
        source_distance_face_ids,
        synthetic_ids,
        removed_ids,
        near_zero_face_ids,
        loops,
        boundary_edges,
        patch_candidates,
        patch_area,
        _surface_deviation(
            source_vertices,
            source_valid_faces,
            proxy_vertices,
            proxy_faces,
            selected_policy,
        ),
        selected_policy,
        None if source_path is None else str(Path(source_path).expanduser()),
    )
    if not _candidate_is_accepted(result, selected_policy):
        raise RuntimeError(
            "DERIVED_SDF_PROXY_FAILED: " + json.dumps(result.as_dict(), sort_keys=True, default=str)
        )
    return result


def _boundary_segments(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    _, boundary, _ = _boundary_loops(vertices, faces)
    if not boundary:
        return np.empty((0, 2, 3), dtype=np.float64)
    return np.asarray([[vertices[a], vertices[b]] for a, b in boundary], dtype=np.float64)


def _point_segment_distance(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    segment_value = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 3)
    if len(segment_value) == 0:
        return np.full(len(value), np.inf, dtype=np.float64)
    start = segment_value[:, 0][None, :, :]
    delta = (segment_value[:, 1] - segment_value[:, 0])[None, :, :]
    offset = value[:, None, :] - start
    factor = np.sum(offset * delta, axis=-1) / np.maximum(np.sum(delta * delta, axis=-1), 1e-30)
    factor = np.clip(factor, 0.0, 1.0)
    closest = start + factor[..., None] * delta
    return np.sqrt(np.min(np.sum((value[:, None, :] - closest) ** 2, axis=-1), axis=1))


class HybridSignedDistanceBackend(SignedDistanceBackend):
    """Original closest-point magnitude plus proxy-only sign."""

    backend_id = HYBRID_SIGNED_DISTANCE_PROFILE_ID

    def __init__(
        self,
        geometry: DerivedWatertightSignProxy,
        *,
        query_chunk_size: int = 256,
        face_chunk_size: int = 4096,
    ) -> None:
        self.geometry = geometry
        self.policy = geometry.policy
        self.mesh_hash = geometry.source_mesh_hash
        self.proxy_mesh_hash = geometry.proxy_mesh_hash
        self.original = ReferenceSignedDistanceBackend(
            geometry.source_distance_vertices,
            geometry.source_distance_faces,
            mesh_hash=geometry.source_mesh_hash,
            sign_mode="unsigned_only",
            query_chunk_size=query_chunk_size,
            face_chunk_size=face_chunk_size,
            closest_acceleration="tree",
            winding_device="cpu",
        )
        self.proxy = ReferenceSignedDistanceBackend(
            geometry.proxy_vertices,
            geometry.proxy_faces,
            mesh_hash=geometry.proxy_mesh_hash,
            sign_mode="strict",
            query_chunk_size=query_chunk_size,
            face_chunk_size=face_chunk_size,
            closest_acceleration="tree",
            winding_device="cpu",
        )
        self.original_boundary_segments = _boundary_segments(
            geometry.source_distance_vertices, geometry.source_distance_faces
        )
        self.boundary_exclusion_radius_m = max(
            self.policy.boundary_exclusion_min_m,
            self.policy.boundary_exclusion_bbox_scale
            * float(np.linalg.norm(np.ptp(geometry.source_vertices, axis=0))),
        )
        self.synthetic_face_set = set(int(item) for item in geometry.synthetic_face_ids.tolist())

    def audit(self) -> dict[str, Any]:
        return {
            "schema_version": DERIVED_SDF_PROXY_SCHEMA_VERSION,
            "backend_id": self.backend_id,
            "source": self.geometry.source_audit.as_dict(),
            "proxy": self.geometry.proxy_audit.as_dict(),
            "proxy_used_for_sign_only": True,
            "boundary_exclusion_radius_m": self.boundary_exclusion_radius_m,
            "cache_signature": self.geometry.cache_signature,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "sign_convention": "positive_outside",
            "mesh_hash": self.mesh_hash,
            "source_mesh_hash": self.geometry.source_mesh_hash,
            "proxy_mesh_hash": self.geometry.proxy_mesh_hash,
            "proxy_candidate_id": self.geometry.candidate_id,
            "proxy_used_for_sign_only": True,
            "paper_unspecified_geometry_engineering": True,
            "policy_hash": self.policy.policy_hash,
            "cache_signature": self.geometry.cache_signature,
            "boundary_exclusion_radius_m": self.boundary_exclusion_radius_m,
        }

    def query_local(self, points_local: np.ndarray) -> SignedDistanceQueryResult:
        points = np.asarray(points_local, dtype=np.float64)
        original = self.original.query_local(points)
        proxy = self.proxy.query_local(points)
        assert proxy.inside is not None
        signed = np.where(proxy.inside, -original.unsigned_distance, original.unsigned_distance)
        closest_face_indices = self.geometry.source_distance_face_ids[
            np.asarray(original.closest_face_indices, dtype=np.int64)
        ]
        boundary_distance = _point_segment_distance(
            original.closest_points.reshape(-1, 3), self.original_boundary_segments
        ).reshape(original.unsigned_distance.shape)
        near_boundary = boundary_distance <= self.boundary_exclusion_radius_m
        proxy_faces = np.asarray(proxy.closest_face_indices, dtype=np.int64)
        synthetic = np.isin(
            proxy_faces,
            np.asarray(sorted(self.synthetic_face_set), dtype=np.int64),
        )
        valid = np.asarray(proxy.sign_valid, dtype=bool) & np.isfinite(signed)
        non_smooth = (
            np.zeros_like(valid, dtype=bool)
            if original.non_smooth is None
            else np.asarray(original.non_smooth, dtype=bool)
        )
        return SignedDistanceQueryResult(
            signed_distance=signed,
            unsigned_distance=original.unsigned_distance,
            closest_points=original.closest_points,
            closest_face_indices=closest_face_indices,
            closest_barycentric=original.closest_barycentric,
            surface_normals=original.surface_normals,
            inside=proxy.inside,
            on_surface=original.on_surface,
            valid=valid,
            sign_valid=valid,
            sign_confidence=proxy.sign_confidence,
            sign_method="hybrid_original_distance_proxy_sign",
            backend_id=self.backend_id,
            mesh_hash=self.mesh_hash,
            winding_value=proxy.winding_value,
            non_smooth=non_smooth,
            gradient_valid=valid & ~non_smooth,
            proxy_closest_face_indices=proxy_faces,
            proxy_closest_is_synthetic_patch=synthetic,
            original_boundary_distance=boundary_distance,
            near_original_boundary=near_boundary,
            geometry_metadata={
                "source_mesh_hash": self.geometry.source_mesh_hash,
                "proxy_mesh_hash": self.geometry.proxy_mesh_hash,
                "proxy_candidate_id": self.geometry.candidate_id,
                "proxy_used_for_sign_only": True,
                "boundary_exclusion_radius_m": self.boundary_exclusion_radius_m,
            },
        )

    def query_scene(
        self, points_scene: np.ndarray, object_pose_scene: np.ndarray
    ) -> SignedDistanceQueryResult:
        local = transform_points(invert_transform(object_pose_scene), np.asarray(points_scene))
        result = self.query_local(local)
        result.closest_points = transform_points(object_pose_scene, result.closest_points)
        result.surface_normals = transform_vectors(object_pose_scene, result.surface_normals)
        result.surface_normals /= np.maximum(
            np.linalg.norm(result.surface_normals, axis=-1, keepdims=True), 1e-15
        )
        return result


def build_hybrid_signed_distance_backend(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    policy: ObjectSDFGeometryPolicy | None = None,
    source_path: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> tuple[HybridSignedDistanceBackend, DerivedWatertightSignProxy]:
    selected_policy = policy or ObjectSDFGeometryPolicy.load()
    geometry = build_derived_sign_proxy(
        vertices, faces, policy=selected_policy, source_path=source_path
    )
    if artifact_root is not None:
        write_geometry_artifact(geometry, artifact_root)
    return HybridSignedDistanceBackend(geometry), geometry


def write_geometry_artifact(geometry: DerivedWatertightSignProxy, output_root: str | Path) -> Path:
    output = Path(output_root).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    write_json(geometry.as_dict(), output / "proxy_manifest.json")
    np.savez_compressed(
        output / "proxy_mesh.npz",
        vertices=geometry.proxy_vertices,
        faces=geometry.proxy_faces,
        proxy_source_face_ids=geometry.proxy_source_face_ids,
        synthetic_face_ids=geometry.synthetic_face_ids,
    )
    np.savez_compressed(
        output / "source_mesh.npz",
        vertices=geometry.source_vertices,
        faces=geometry.source_faces,
    )
    np.savez_compressed(
        output / "source_distance_mesh.npz",
        vertices=geometry.source_distance_vertices,
        faces=geometry.source_distance_faces,
        source_face_ids=geometry.source_distance_face_ids,
    )
    geometry.source_audit.write_json(output / "source_mesh_audit.json")
    geometry.source_audit.write_csv(output / "source_mesh_audit.csv")
    geometry.proxy_audit.write_json(output / "proxy_mesh_audit.json")
    (output / "boundary_loops.json").write_text(
        json.dumps(
            {
                "boundary_loops": geometry.boundary_loops,
                "synthetic_face_ids": geometry.synthetic_face_ids.tolist(),
                "removed_face_ids": geometry.removed_face_ids.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DERIVED_SDF_PROXY_SCHEMA_VERSION",
    "DEFAULT_POLICY_PATH",
    "HYBRID_SIGNED_DISTANCE_PROFILE_ID",
    "DerivedWatertightSignProxy",
    "HybridSignedDistanceBackend",
    "ObjectSDFGeometryPolicy",
    "build_derived_sign_proxy",
    "build_hybrid_signed_distance_backend",
    "write_geometry_artifact",
]
