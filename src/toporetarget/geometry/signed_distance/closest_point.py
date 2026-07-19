"""Reference point-to-triangle closest-point implementation.

The implementation is deliberately independent of trimesh's optional rtree
dependency.  It evaluates the Ericson analytic regions in chunks, so the
returned point is a triangle distance rather than a nearest-vertex proxy.
"""

from __future__ import annotations

import numpy as np


def closest_points_on_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    query_chunk_size: int = 256,
    face_chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    queries = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mesh = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if len(mesh) == 0:
        raise ValueError("closest-point query requires at least one triangle")
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
            # Degenerate triangles are not normally passed by the backend. Keep
            # a finite vertex fallback for direct callers.
            lengths = np.linalg.norm(np.cross(ab, ac), axis=-1)
            degenerate = lengths <= 1e-15
            if np.any(degenerate):
                candidates = np.stack((a, b, c), axis=-2)
                distances = np.sum((point[..., None, :] - candidates) ** 2, axis=-1)
                nearest = np.argmin(distances, axis=-1)
                fallback = np.eye(3, dtype=np.float64)[nearest]
                bary = np.where(degenerate[..., None], fallback, bary)
            closest = (
                np.einsum("qfi,qfi->qf", bary[..., 0:1], a)
                if False
                else (bary[..., 0, None] * a + bary[..., 1, None] * b + bary[..., 2, None] * c)
            )
            distances = np.sum((point - closest) ** 2, axis=-1)
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
