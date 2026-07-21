"""Reference point-to-triangle closest-point implementation.

The implementation is deliberately independent of trimesh's optional rtree
dependency.  It evaluates the Ericson analytic regions in chunks, so the
returned point is a triangle distance rather than a nearest-vertex proxy.
"""

from __future__ import annotations

import heapq
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

    def _build(self, start: int, stop: int) -> int:
        node_index = len(self._nodes)
        self._nodes.append(
            _AABBNode(np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
        )
        indices = self._order[start:stop]
        minimum = np.min(self._triangle_minimum[indices], axis=0)
        maximum = np.max(self._triangle_maximum[indices], axis=0)
        count = stop - start
        if count <= self.leaf_size:
            self._nodes[node_index] = _AABBNode(minimum, maximum, start, stop)
            return node_index
        axis = int(np.argmax(maximum - minimum))
        sorted_indices = indices[
            np.argsort(self._centroids[indices, axis], kind="mergesort")
        ]
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
                if node.is_leaf:
                    leaf_indices = self._order[node.start : node.stop]
                    leaf_points, leaf_bary, leaf_distance2 = _closest_point_pairs(
                        point[None, :], self.triangles[leaf_indices]
                    )
                    local = int(np.argmin(leaf_distance2[0]))
                    candidate_distance2 = float(leaf_distance2[0, local])
                    candidate_face = int(leaf_indices[local])
                    if (
                        candidate_distance2 < best_distance2
                        or (
                            candidate_distance2 == best_distance2
                            and (best_face < 0 or candidate_face < best_face)
                        )
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


def closest_points_on_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    query_chunk_size: int = 256,
    face_chunk_size: int = 4096,
    tree: TriangleAABBTree | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    queries = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mesh = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if len(mesh) == 0:
        raise ValueError("closest-point query requires at least one triangle")
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
        p = queries[q_start:q_end]
        best_p = np.zeros_like(p)
        best_bary = np.zeros((len(p), 3), dtype=np.float64)
        best_face = np.zeros(len(p), dtype=np.int64)
        best_dist = np.full(len(p), np.inf, dtype=np.float64)
        for f_start in range(0, len(mesh), face_chunk_size):
            f_end = min(len(mesh), f_start + face_chunk_size)
            tri = mesh[f_start:f_end]
            closest, bary, distances = _closest_point_pairs(p, tri)
            local_face = np.argmin(distances, axis=1)
            local_distance = distances[np.arange(len(p)), local_face]
            update = local_distance < best_dist
            if np.any(update):
                rows = np.arange(len(p))[update]
                selected = local_face[update]
                best_dist[update] = local_distance[update]
                best_p[update] = closest[rows, selected]
                best_bary[update] = bary[rows, selected]
                best_face[update] = f_start + selected
        result_points[q_start:q_end] = best_p
        result_bary[q_start:q_end] = best_bary
        result_faces[q_start:q_end] = best_face
        result_dist2[q_start:q_end] = best_dist
    return result_points, result_faces, result_bary, np.sqrt(np.maximum(result_dist2, 0.0))


__all__ = ["closest_points_on_triangles"]
