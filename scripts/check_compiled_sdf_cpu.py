#!/usr/bin/env python3
"""Run focused exactness and API-safety checks for the local CPU kernel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from toporetarget.geometry.signed_distance.closest_point import closest_points_on_triangles
from toporetarget.geometry.signed_distance.compiled_sdf_cpu import CompiledBVHHandle
from toporetarget.geometry.signed_distance.validation import make_synthetic_mesh


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / ".local/reports/compiled_sdf_cpu_v1/kernel_check.json"
    vertices, faces = make_synthetic_mesh("cube")
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    rng = np.random.default_rng(20260730)
    points = np.ascontiguousarray(
        np.vstack((rng.normal(size=(512, 3)), [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])),
        dtype=np.float64,
    )
    handle = CompiledBVHHandle(vertices, faces)
    closest, face, bary, unsigned = handle.query(points)
    expected = closest_points_on_triangles(points, vertices[faces], tree=None)
    distance_error = float(np.max(np.abs(unsigned - expected[3])))
    closest_error = float(np.max(np.abs(closest - expected[0])))
    result = {
        "status": "pass" if distance_error <= 1e-10 and closest_error <= 1e-10 else "failed",
        "distance_error_m": distance_error,
        "closest_error_m": closest_error,
        "all_finite": bool(np.isfinite(closest).all() and np.isfinite(bary).all()),
        "face_id_geometric_ties_allowed": True,
        "face_id_mismatch_count": int(np.count_nonzero(face != expected[1])),
        "stats": handle.stats(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
