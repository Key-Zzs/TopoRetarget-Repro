"""Generalized winding-number sign evaluation by triangle solid angles."""

from __future__ import annotations

import numpy as np


def generalized_winding_number(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    query_chunk_size: int = 256,
    face_chunk_size: int = 4096,
    device: str | None = None,
) -> np.ndarray:
    queries = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mesh = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if len(mesh) == 0:
        raise ValueError("winding query requires at least one triangle")
    output = np.zeros(len(queries), dtype=np.float64)
    if device is not None:
        try:
            import torch

            torch_device = torch.device(device)
            query_tensor = torch.as_tensor(queries, dtype=torch.float64, device=torch_device)
            mesh_tensor = torch.as_tensor(mesh, dtype=torch.float64, device=torch_device)
            four_pi = 4.0 * np.pi
            for q_start in range(0, len(queries), query_chunk_size):
                q_end = min(len(queries), q_start + query_chunk_size)
                p = query_tensor[q_start:q_end, None, None, :]
                total = torch.zeros(q_end - q_start, dtype=torch.float64, device=torch_device)
                for f_start in range(0, len(mesh), face_chunk_size):
                    tri = mesh_tensor[f_start : min(len(mesh), f_start + face_chunk_size)]
                    origin = p[..., 0, :]
                    a = tri[:, 0][None, :, :] - origin
                    b = tri[:, 1][None, :, :] - origin
                    c = tri[:, 2][None, :, :] - origin
                    la = torch.linalg.vector_norm(a, dim=-1)
                    lb = torch.linalg.vector_norm(b, dim=-1)
                    lc = torch.linalg.vector_norm(c, dim=-1)
                    numerator = torch.einsum("qfi,qfi->qf", a, torch.cross(b, c, dim=-1))
                    denominator = la * lb * lc
                    denominator = denominator + torch.einsum("qfi,qfi->qf", a, b) * lc
                    denominator = denominator + torch.einsum("qfi,qfi->qf", b, c) * la
                    denominator = denominator + torch.einsum("qfi,qfi->qf", c, a) * lb
                    total = total + torch.sum(2.0 * torch.atan2(numerator, denominator), dim=1)
                output[q_start:q_end] = total.detach().cpu().numpy() / four_pi
            return output
        except (ImportError, RuntimeError, ValueError) as exc:
            if str(device).startswith("cuda"):
                raise RuntimeError(f"exact winding accelerator unavailable: {device}") from exc
    four_pi = 4.0 * np.pi
    for q_start in range(0, len(queries), query_chunk_size):
        q_end = min(len(queries), q_start + query_chunk_size)
        np_p = queries[q_start:q_end, None, None, :]
        np_total = np.zeros(q_end - q_start, dtype=np.float64)
        for f_start in range(0, len(mesh), face_chunk_size):
            np_tri = mesh[f_start : min(len(mesh), f_start + face_chunk_size)]
            np_a = np_tri[:, 0][None, :, :] - np_p[..., 0, :]
            np_b = np_tri[:, 1][None, :, :] - np_p[..., 0, :]
            np_c = np_tri[:, 2][None, :, :] - np_p[..., 0, :]
            np_la = np.linalg.norm(np_a, axis=-1)
            np_lb = np.linalg.norm(np_b, axis=-1)
            np_lc = np.linalg.norm(np_c, axis=-1)
            numerator = np.einsum("qfi,qfi->qf", np_a, np.cross(np_b, np_c))
            denominator = np_la * np_lb * np_lc + np.einsum("qfi,qfi->qf", np_a, np_b) * np_lc
            denominator += np.einsum("qfi,qfi->qf", np_b, np_c) * np_la
            denominator += np.einsum("qfi,qfi->qf", np_c, np_a) * np_lb
            np_total += np.sum(2.0 * np.arctan2(numerator, denominator), axis=1)
        output[q_start:q_end] = np_total / four_pi
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
