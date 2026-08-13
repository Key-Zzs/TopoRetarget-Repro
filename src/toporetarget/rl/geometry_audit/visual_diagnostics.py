"""Unsigned visual/proxy surface diagnostics that never authorize the formal gate."""

from __future__ import annotations

from typing import Any

import numpy as np


def unsigned_surface_diagnostics(
    *,
    visual_vertices: np.ndarray,
    visual_faces: np.ndarray,
    proxy_vertices: np.ndarray,
    proxy_faces: np.ndarray,
) -> dict[str, Any]:
    import trimesh

    visual = trimesh.Trimesh(
        vertices=np.asarray(visual_vertices, dtype=np.float64),
        faces=np.asarray(visual_faces, dtype=np.int64),
        process=False,
    )
    proxy = trimesh.Trimesh(
        vertices=np.asarray(proxy_vertices, dtype=np.float64),
        faces=np.asarray(proxy_faces, dtype=np.int64),
        process=False,
    )
    _, visual_to_proxy, _ = trimesh.proximity.closest_point_naive(proxy, visual.vertices)
    _, proxy_to_visual, _ = trimesh.proximity.closest_point_naive(visual, proxy.vertices)
    visual_bbox_extent = np.ptp(visual.vertices, axis=0)
    proxy_bbox_extent = np.ptp(proxy.vertices, axis=0)
    visual_bbox_volume = float(np.prod(visual_bbox_extent))
    proxy_bbox_volume = float(np.prod(proxy_bbox_extent))

    def summary(values: np.ndarray) -> dict[str, float]:
        distance = np.asarray(values, dtype=np.float64)
        return {
            "mean_m": float(distance.mean()),
            "p95_m": float(np.quantile(distance, 0.95)),
            "max_m": float(distance.max(initial=0.0)),
        }

    return {
        "schema_version": "Stage16DVisualProxyUnsignedDiagnosticsV1",
        "formal_gate_authority": False,
        "signed_penetration_available": False,
        "visual_mesh_repaired": False,
        "distance_method": "trimesh triangle-surface closest point, unsigned",
        "visual_to_proxy": summary(visual_to_proxy),
        "proxy_to_visual": summary(proxy_to_visual),
        "symmetric_hausdorff_upper_diagnostic_m": float(
            max(visual_to_proxy.max(initial=0.0), proxy_to_visual.max(initial=0.0))
        ),
        "bbox": {
            "visual_extent_m": visual_bbox_extent.tolist(),
            "proxy_extent_m": proxy_bbox_extent.tolist(),
            "proxy_to_visual_volume_ratio": (
                proxy_bbox_volume / visual_bbox_volume if visual_bbox_volume > 0.0 else None
            ),
        },
        "possible_missed_visual_protrusion_m": float(visual_to_proxy.max(initial=0.0)),
        "possible_overconservative_hull_region_m": float(proxy_to_visual.max(initial=0.0)),
        "limitation": "collision proxy is formal runtime geometry but is not visual truth",
    }


__all__ = ["unsigned_surface_diagnostics"]
