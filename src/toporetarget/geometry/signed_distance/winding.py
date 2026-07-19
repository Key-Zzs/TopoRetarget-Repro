"""Generalized winding-number sign evaluation by triangle solid angles."""

from __future__ import annotations

import numpy as np


def generalized_winding_number(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    query_chunk_size: int = 256,
    face_chunk_size: int = 4096,
) -> np.ndarray:
    queries = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mesh = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if len(mesh) == 0:
        raise ValueError("winding query requires at least one triangle")
    output = np.zeros(len(queries), dtype=np.float64)
    four_pi = 4.0 * np.pi
    for q_start in range(0, len(queries), query_chunk_size):
        q_end = min(len(queries), q_start + query_chunk_size)
        p = queries[q_start:q_end, None, None, :]
        total = np.zeros(q_end - q_start, dtype=np.float64)
        for f_start in range(0, len(mesh), face_chunk_size):
            tri = mesh[f_start : min(len(mesh), f_start + face_chunk_size)]
            a = tri[:, 0][None, :, :] - p[..., 0, :]
            b = tri[:, 1][None, :, :] - p[..., 0, :]
            c = tri[:, 2][None, :, :] - p[..., 0, :]
            la = np.linalg.norm(a, axis=-1)
            lb = np.linalg.norm(b, axis=-1)
            lc = np.linalg.norm(c, axis=-1)
            numerator = np.einsum("qfi,qfi->qf", a, np.cross(b, c))
            denominator = la * lb * lc + np.einsum("qfi,qfi->qf", a, b) * lc
            denominator += np.einsum("qfi,qfi->qf", b, c) * la
            denominator += np.einsum("qfi,qfi->qf", c, a) * lb
            total += np.sum(2.0 * np.arctan2(numerator, denominator), axis=1)
        output[q_start:q_end] = total / four_pi
    return output


def winding_sign(
    winding: np.ndarray, *, threshold: float = 0.5, confidence_threshold: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(winding, dtype=np.float64)
    magnitude = np.abs(value)
    inside = magnitude >= threshold
    confidence = np.clip(2.0 * np.abs(magnitude - threshold), 0.0, 1.0)
    ambiguous = confidence < confidence_threshold
    return inside, confidence, ambiguous, magnitude


__all__ = ["generalized_winding_number", "winding_sign"]
