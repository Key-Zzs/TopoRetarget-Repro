"""Reference point-to-triangle closest-point implementation.

The implementation is deliberately independent of trimesh's optional rtree
dependency.  It evaluates the Ericson analytic regions in chunks, so the
returned point is a triangle distance rather than a nearest-vertex proxy.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import numpy as np


def _closest_point_pairs(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact closest points for every point/triangle pair in a block."""

    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    tri = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if len(tri) == 0:
        raise ValueError("closest-point query requires at least one triangle")
    a = tri[None, :, 0, :]
    b = tri[None, :, 1, :]
    c = tri[None, :, 2, :]
    point = p[:, None, :]
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = np.sum(ab * ap, axis=-1)
    d2 = np.sum(ac * ap, axis=-1)
    bp = point - b
    d3 = np.sum(ab * bp, axis=-1)
    d4 = np.sum(ac * bp, axis=-1)
    cp = point - c
    d5 = np.sum(ab * cp, axis=-1)
    d6 = np.sum(ac * cp, axis=-1)
    bary = np.zeros((*p.shape[:1], len(tri), 3), dtype=np.float64)
    # Vertex A
    mask = (d1 <= 0) & (d2 <= 0)
    bary[..., 0] = np.where(mask, 1.0, bary[..., 0])
    # Vertex B
    mask_b = (d3 >= 0) & (d4 <= d3)
    bary[..., 1] = np.where(mask_b, 1.0, bary[..., 1])
    # Edge AB
    vc = d1 * d4 - d3 * d2
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v_ab = d1 / np.maximum(d1 - d3, 1e-30)
    bary[..., 0] = np.where(mask_ab, 1.0 - v_ab, bary[..., 0])
    bary[..., 1] = np.where(mask_ab, v_ab, bary[..., 1])
    # Vertex C
    mask_c = (d6 >= 0) & (d5 <= d6)
    bary[..., 2] = np.where(mask_c, 1.0, bary[..., 2])
    # Edge AC
    vb = d5 * d2 - d1 * d6
    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w_ac = d2 / np.maximum(d2 - d6, 1e-30)
    bary[..., 0] = np.where(mask_ac, 1.0 - w_ac, bary[..., 0])
    bary[..., 2] = np.where(mask_ac, w_ac, bary[..., 2])
    # Edge BC
    va = d3 * d6 - d5 * d4
    mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w_bc = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-30)
    bary[..., 1] = np.where(mask_bc, 1.0 - w_bc, bary[..., 1])
    bary[..., 2] = np.where(mask_bc, w_bc, bary[..., 2])
    # Face interior
    interior = ~(mask | mask_b | mask_ab | mask_c | mask_ac | mask_bc)
    denominator = np.maximum(va + vb + vc, 1e-30)
    bary[..., 0] = np.where(interior, 1.0 - (vb + vc) / denominator, bary[..., 0])
    bary[..., 1] = np.where(interior, vb / denominator, bary[..., 1])
    bary[..., 2] = np.where(interior, vc / denominator, bary[..., 2])
    # Degenerate triangles are not normally passed by the backend. Keep a
    # finite vertex fallback for direct callers and BVH leaves.
    lengths = np.linalg.norm(np.cross(ab, ac), axis=-1)
    degenerate = lengths <= 1e-15
    if np.any(degenerate):
        candidates = np.stack((a, b, c), axis=-2)
        distances = np.sum((point[..., None, :] - candidates) ** 2, axis=-1)
        nearest = np.argmin(distances, axis=-1)
        fallback = np.eye(3, dtype=np.float64)[nearest]
        bary = np.where(degenerate[..., None], fallback, bary)
    closest = bary[..., 0, None] * a + bary[..., 1, None] * b + bary[..., 2, None] * c
    distance2 = np.sum((point - closest) ** 2, axis=-1)
    return closest, bary, distance2


@dataclass(frozen=True)
class _AABBNode:
    minimum: np.ndarray
    maximum: np.ndarray
    start: int = 0
    stop: int = 0
    left: int | None = None
    right: int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class TriangleAABBTree:
    """Exact branch-and-bound closest-triangle queries for a static mesh.

    The lower bound is the squared distance to a node AABB.  A node is pruned
    only when that bound is strictly larger than the best exact triangle
    distance already found, so the returned triangle remains exact and
    deterministic.  The tree is built once by the persistent SDF backend.
    """

    def __init__(self, triangles: np.ndarray, *, leaf_size: int = 32) -> None:
        started = time.perf_counter()
        self.triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
        if len(self.triangles) == 0:
            raise ValueError("AABB tree requires at least one triangle")
        if leaf_size <= 0:
            raise ValueError("AABB leaf size must be positive")
        self.leaf_size = int(leaf_size)
        self._triangle_minimum = np.min(self.triangles, axis=1)
        self._triangle_maximum = np.max(self.triangles, axis=1)
        self._centroids = 0.5 * (self._triangle_minimum + self._triangle_maximum)
        self._order = np.arange(len(self.triangles), dtype=np.int64)
        self._nodes: list[_AABBNode] = []
        self.root = self._build(0, len(self._order))
        self.build_time_s = time.perf_counter() - started
        self._stats = {
            "query_count": 0,
            "queried_point_count": 0,
            "candidate_triangle_evaluations": 0,
            "node_visits": 0,
        }

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def leaf_count(self) -> int:
        return sum(node.is_leaf for node in self._nodes)

    def stats(self) -> dict[str, float | int]:
        points = int(self._stats["queried_point_count"])
        candidates = int(self._stats["candidate_triangle_evaluations"])
        return {
            "triangle_count": int(len(self.triangles)),
            "node_count": self.node_count,
            "leaf_count": self.leaf_count,
            "build_time_s": float(self.build_time_s),
            **{key: int(value) for key, value in self._stats.items()},
            "mean_candidates_per_query": float(candidates / points) if points else 0.0,
            "max_candidates_per_query": int(self.leaf_size),
        }

    def _build(self, start: int, stop: int) -> int:
        node_index = len(self._nodes)
        self._nodes.append(_AABBNode(np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)))
        indices = self._order[start:stop]
        minimum = np.min(self._triangle_minimum[indices], axis=0)
        maximum = np.max(self._triangle_maximum[indices], axis=0)
        count = stop - start
        if count <= self.leaf_size:
            self._nodes[node_index] = _AABBNode(minimum, maximum, start, stop)
            return node_index
        axis = int(np.argmax(maximum - minimum))
        sorted_indices = indices[np.argsort(self._centroids[indices, axis], kind="mergesort")]
        self._order[start:stop] = sorted_indices
        middle = start + count // 2
        left = self._build(start, middle)
        right = self._build(middle, stop)
        self._nodes[node_index] = _AABBNode(minimum, maximum, left=left, right=right)
        return node_index

    @staticmethod
    def _lower_bound2(point: np.ndarray, node: _AABBNode) -> float:
        delta = np.maximum(np.maximum(node.minimum - point, 0.0), point - node.maximum)
        return float(np.dot(delta, delta))

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self._stats["query_count"] += 1
        self._stats["queried_point_count"] += len(values)
        closest = np.empty_like(values)
        faces = np.empty(len(values), dtype=np.int64)
        barycentric = np.empty((len(values), 3), dtype=np.float64)
        distance2 = np.empty(len(values), dtype=np.float64)
        for index, point in enumerate(values):
            queue: list[tuple[float, int]] = [(0.0, self.root)]
            best_distance2 = float("inf")
            best_face = -1
            best_point = np.zeros(3, dtype=np.float64)
            best_bary = np.zeros(3, dtype=np.float64)
            while queue:
                lower_bound2, node_index = heapq.heappop(queue)
                if lower_bound2 > best_distance2:
                    break
                node = self._nodes[node_index]
                self._stats["node_visits"] += 1
                if node.is_leaf:
                    leaf_indices = self._order[node.start : node.stop]
                    self._stats["candidate_triangle_evaluations"] += len(leaf_indices)
                    leaf_points, leaf_bary, leaf_distance2 = _closest_point_pairs(
                        point[None, :], self.triangles[leaf_indices]
                    )
                    local = int(np.argmin(leaf_distance2[0]))
                    candidate_distance2 = float(leaf_distance2[0, local])
                    candidate_face = int(leaf_indices[local])
                    if candidate_distance2 < best_distance2 or (
                        candidate_distance2 == best_distance2
                        and (best_face < 0 or candidate_face < best_face)
                    ):
                        best_distance2 = candidate_distance2
                        best_face = candidate_face
                        best_point = leaf_points[0, local]
                        best_bary = leaf_bary[0, local]
                    continue
                assert node.left is not None and node.right is not None
                left_node = self._nodes[node.left]
                right_node = self._nodes[node.right]
                heapq.heappush(queue, (self._lower_bound2(point, left_node), node.left))
                heapq.heappush(queue, (self._lower_bound2(point, right_node), node.right))
            if best_face < 0:  # pragma: no cover - root always contains a leaf
                raise RuntimeError("AABB query did not visit a triangle leaf")
            closest[index] = best_point
            faces[index] = best_face
            barycentric[index] = best_bary
            distance2[index] = best_distance2
        return closest, faces, barycentric, np.sqrt(np.maximum(distance2, 0.0))


class ObjectLocalBVH(TriangleAABBTree):
    """Versioned exact object-local BVH used by persistent SDF backends."""

    backend_id = "exact_object_local_bvh_v1"


class TriangleCentroidBoundTree:
    """Exact closest-triangle queries using centroid lower-bound pruning.

    The centroid index is only an accelerator.  Each candidate is evaluated
    by ``_closest_point_pairs`` and every omitted triangle is proven farther
    away by its bounding-sphere lower bound, so this returns the same exact
    triangle distance and deterministic face tie-break as the AABB backend.
    """

    def __init__(self, triangles: np.ndarray) -> None:
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:  # pragma: no cover - optional accelerator
            raise RuntimeError("centroid-bound accelerator requires scipy") from exc
        self.triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
        if len(self.triangles) == 0:
            raise ValueError("centroid-bound tree requires at least one triangle")
        self.centroids = np.mean(self.triangles, axis=1)
        self.radii = np.max(
            np.linalg.norm(self.triangles - self.centroids[:, None, :], axis=-1), axis=1
        )
        self.max_radius = float(np.max(self.radii))
        self._tree = cKDTree(self.centroids)

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        _, nearest = self._tree.query(values, k=1, workers=-1)
        nearest = np.asarray(nearest, dtype=np.int64)
        seed_triangles = self.triangles[nearest]
        seed_points, _, seed_distance2 = _closest_point_pairs(values, seed_triangles)
        seed_distance = np.sqrt(np.maximum(seed_distance2[:, 0], 0.0))
        radii = seed_distance + self.max_radius
        candidate_lists = self._tree.query_ball_point(values, radii, workers=-1, return_sorted=True)
        closest = np.empty_like(values)
        faces = np.empty(len(values), dtype=np.int64)
        barycentric = np.empty((len(values), 3), dtype=np.float64)
        distance2 = np.empty(len(values), dtype=np.float64)
        for index, candidates in enumerate(candidate_lists):
            if not candidates:  # pragma: no cover - nearest centroid is included
                candidates = [int(nearest[index])]
            candidate_indices = np.asarray(candidates, dtype=np.int64)
            candidate_points, candidate_bary, candidate_distance2 = _closest_point_pairs(
                values[index : index + 1], self.triangles[candidate_indices]
            )
            local = int(np.argmin(candidate_distance2[0]))
            best_distance2 = float(candidate_distance2[0, local])
            tied = np.flatnonzero(
                np.isclose(candidate_distance2[0], best_distance2, rtol=0.0, atol=0.0)
            )
            if len(tied) > 1:
                local = int(tied[np.argmin(candidate_indices[tied])])
            closest[index] = candidate_points[0, local]
            faces[index] = candidate_indices[local]
            barycentric[index] = candidate_bary[0, local]
            distance2[index] = best_distance2
        return closest, faces, barycentric, np.sqrt(np.maximum(distance2, 0.0))


def closest_points_on_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    query_chunk_size: int = 256,
    face_chunk_size: int = 4096,
    tree: TriangleAABBTree | TriangleCentroidBoundTree | None = None,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    queries = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mesh = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if len(mesh) == 0:
        raise ValueError("closest-point query requires at least one triangle")
    if device is not None:
        try:
            import torch

            torch_device = torch.device(device)
            query_tensor = torch.as_tensor(queries, dtype=torch.float64, device=torch_device)
            mesh_tensor = torch.as_tensor(mesh, dtype=torch.float64, device=torch_device)
            result_points = np.empty_like(queries)
            result_bary = np.empty((len(queries), 3), dtype=np.float64)
            result_faces = np.empty(len(queries), dtype=np.int64)
            result_dist2 = np.full(len(queries), np.inf, dtype=np.float64)
            for q_start in range(0, len(queries), query_chunk_size):
                q_end = min(len(queries), q_start + query_chunk_size)
                p = query_tensor[q_start:q_end]
                best_dist = torch.full(
                    (len(p),), float("inf"), dtype=torch.float64, device=torch_device
                )
                best_p = torch.zeros((len(p), 3), dtype=torch.float64, device=torch_device)
                best_bary = torch.zeros((len(p), 3), dtype=torch.float64, device=torch_device)
                best_face = torch.zeros((len(p),), dtype=torch.int64, device=torch_device)
                for f_start in range(0, len(mesh), face_chunk_size):
                    tri = mesh_tensor[f_start : min(len(mesh), f_start + face_chunk_size)]
                    a, b, c = tri[None, :, 0, :], tri[None, :, 1, :], tri[None, :, 2, :]
                    point = p[:, None, :]
                    ab, ac = b - a, c - a
                    ap, bp, cp = point - a, point - b, point - c
                    d1, d2 = torch.sum(ab * ap, dim=-1), torch.sum(ac * ap, dim=-1)
                    d3, d4 = torch.sum(ab * bp, dim=-1), torch.sum(ac * bp, dim=-1)
                    d5, d6 = torch.sum(ab * cp, dim=-1), torch.sum(ac * cp, dim=-1)
                    bary = torch.zeros(
                        (len(p), len(tri), 3), dtype=torch.float64, device=torch_device
                    )
                    mask = (d1 <= 0) & (d2 <= 0)
                    bary[..., 0] = torch.where(mask, 1.0, bary[..., 0])
                    mask_b = (d3 >= 0) & (d4 <= d3)
                    bary[..., 1] = torch.where(mask_b, 1.0, bary[..., 1])
                    vc = d1 * d4 - d3 * d2
                    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
                    v_ab = d1 / torch.clamp(d1 - d3, min=1e-30)
                    bary[..., 0] = torch.where(mask_ab, 1.0 - v_ab, bary[..., 0])
                    bary[..., 1] = torch.where(mask_ab, v_ab, bary[..., 1])
                    mask_c = (d6 >= 0) & (d5 <= d6)
                    bary[..., 2] = torch.where(mask_c, 1.0, bary[..., 2])
                    vb = d5 * d2 - d1 * d6
                    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
                    w_ac = d2 / torch.clamp(d2 - d6, min=1e-30)
                    bary[..., 0] = torch.where(mask_ac, 1.0 - w_ac, bary[..., 0])
                    bary[..., 2] = torch.where(mask_ac, w_ac, bary[..., 2])
                    va = d3 * d6 - d5 * d4
                    mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
                    w_bc = (d4 - d3) / torch.clamp((d4 - d3) + (d5 - d6), min=1e-30)
                    bary[..., 1] = torch.where(mask_bc, 1.0 - w_bc, bary[..., 1])
                    bary[..., 2] = torch.where(mask_bc, w_bc, bary[..., 2])
                    interior = ~(mask | mask_b | mask_ab | mask_c | mask_ac | mask_bc)
                    denominator = torch.clamp(va + vb + vc, min=1e-30)
                    bary[..., 0] = torch.where(
                        interior, 1.0 - (vb + vc) / denominator, bary[..., 0]
                    )
                    bary[..., 1] = torch.where(interior, vb / denominator, bary[..., 1])
                    bary[..., 2] = torch.where(interior, vc / denominator, bary[..., 2])
                    closest = (
                        bary[..., 0, None] * a + bary[..., 1, None] * b + bary[..., 2, None] * c
                    )
                    distance = torch.sum((point - closest) ** 2, dim=-1)
                    local_face = torch.argmin(distance, dim=1)
                    rows = torch.arange(len(p), device=torch_device)
                    local_distance = distance[rows, local_face]
                    update = local_distance < best_dist
                    best_dist = torch.where(update, local_distance, best_dist)
                    best_p = torch.where(update[:, None], closest[rows, local_face], best_p)
                    best_bary = torch.where(update[:, None], bary[rows, local_face], best_bary)
                    best_face = torch.where(update, local_face + f_start, best_face)
                result_points[q_start:q_end] = best_p.detach().cpu().numpy()
                result_bary[q_start:q_end] = best_bary.detach().cpu().numpy()
                result_faces[q_start:q_end] = best_face.detach().cpu().numpy()
                result_dist2[q_start:q_end] = best_dist.detach().cpu().numpy()
            return result_points, result_faces, result_bary, np.sqrt(np.maximum(result_dist2, 0.0))
        except (ImportError, RuntimeError, ValueError) as exc:
            if str(device).startswith("cuda"):
                raise RuntimeError(
                    f"exact closest-point accelerator unavailable: {device}"
                ) from exc
    if tree is not None:
        if not np.shares_memory(tree.triangles, mesh) and not np.array_equal(tree.triangles, mesh):
            raise ValueError("AABB tree triangles do not match the closest-point query mesh")
        return tree.query(queries)
    result_points = np.empty_like(queries)
    result_bary = np.empty((len(queries), 3), dtype=np.float64)
    result_faces = np.empty(len(queries), dtype=np.int64)
    result_dist2 = np.full(len(queries), np.inf, dtype=np.float64)
    for q_start in range(0, len(queries), query_chunk_size):
        q_end = min(len(queries), q_start + query_chunk_size)
        np_p = queries[q_start:q_end]
        np_best_p = np.zeros_like(np_p)
        np_best_bary = np.zeros((len(np_p), 3), dtype=np.float64)
        np_best_face = np.zeros(len(np_p), dtype=np.int64)
        np_best_dist = np.full(len(np_p), np.inf, dtype=np.float64)
        for f_start in range(0, len(mesh), face_chunk_size):
            f_end = min(len(mesh), f_start + face_chunk_size)
            np_tri = mesh[f_start:f_end]
            np_closest, np_bary, np_distances = _closest_point_pairs(np_p, np_tri)
            np_local_face = np.argmin(np_distances, axis=1)
            np_local_distance = np_distances[np.arange(len(np_p)), np_local_face]
            np_update = np_local_distance < np_best_dist
            if np.any(np_update):
                np_rows = np.arange(len(np_p))[np_update]
                np_selected = np_local_face[np_update]
                np_best_dist[np_update] = np_local_distance[np_update]
                np_best_p[np_update] = np_closest[np_rows, np_selected]
                np_best_bary[np_update] = np_bary[np_rows, np_selected]
                np_best_face[np_update] = f_start + np_selected
        result_points[q_start:q_end] = np_best_p
        result_bary[q_start:q_end] = np_best_bary
        result_faces[q_start:q_end] = np_best_face
        result_dist2[q_start:q_end] = np_best_dist
    return result_points, result_faces, result_bary, np.sqrt(np.maximum(result_dist2, 0.0))


__all__ = [
    "ObjectLocalBVH",
    "TriangleCentroidBoundTree",
    "TriangleAABBTree",
    "closest_points_on_triangles",
]
