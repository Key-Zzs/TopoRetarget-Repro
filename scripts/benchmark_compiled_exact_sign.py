#!/usr/bin/env python3
"""Formal single-thread microbenchmark for the real frozen Stage-12 object."""

from __future__ import annotations

import csv
import json
import os
import resource
import subprocess
import sys
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np

from toporetarget.cli.retarget import _refinement_components
from toporetarget.geometry.signed_distance.compiled_sdf_cpu import CompiledSpatialFDBackend
from toporetarget.geometry.signed_distance.winding import generalized_winding_number, winding_sign
from toporetarget.retarget.final_refinement import (
    RefinementSolverProfile,
    prepare_refinement_resources,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/home/deepcybo/workspace/dex/retarget/TopoRetarget-Repro/.local/experiments/"
    "stage12_dataset_validation/dexycb/dexycb_20200709-subject-01_20200709_150144"
)
COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 672, 1200, 1500)
REPEATS = 5


def _timed(callback: Callable[[], Any]) -> tuple[float, Any]:
    started = perf_counter()
    value = callback()
    return perf_counter() - started, value


def _summary(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "warm_median_s": float(np.median(data)),
        "warm_p95_s": float(np.percentile(data, 95)),
        "max_s": float(np.max(data)),
    }


def _cold_import_seconds() -> float:
    """Measure import in a fresh interpreter without conflating it with warm timings."""
    code = "import toporetarget.geometry.signed_distance.compiled_sdf_cpu"
    started = perf_counter()
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return perf_counter() - started


def _allocated_bytes(callback: Callable[[], Any]) -> int:
    """Return Python allocation delta; native handle storage is reported by RSS separately."""
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    callback()
    after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(max(0, after - before))


def main() -> int:
    cold_import_s = _cold_import_seconds()
    canonical = SOURCE / "canonical/canonical_hoi_v2.zarr"
    warm = SOURCE / "warm/warm_start.zarr"
    graph_path = SOURCE / "exports/interaction_graph.zarr"
    samples = SOURCE / "exports/wuji_collision_samples.npz"
    sequence, _warm, graph, _model, _surface, _selected = _refinement_components(
        canonical, warm, graph_path, "wuji_hand2_beta1_rh", samples, None
    )
    resources = prepare_refinement_resources(
        sequence,
        graph,
        RefinementSolverProfile.load("wuji_continuous_sequential_v1"),
        sdf_tree_leaf_size=512,
    )
    reference = resources.reference_sdf
    backend = CompiledSpatialFDBackend(reference, leaf_size=512, compiled_winding=True)
    if backend.winding_handle is None:  # pragma: no cover - constructor contract
        raise RuntimeError("compiled winding handle was not constructed")
    winding_handle = backend.winding_handle
    triangles = np.ascontiguousarray(
        reference.proxy.vertices[reference.proxy.faces], dtype=np.float64
    )
    lo = np.min(reference.proxy.vertices, axis=0)
    hi = np.max(reference.proxy.vertices, axis=0)
    rng = np.random.default_rng(20260730)
    all_points = np.ascontiguousarray(rng.uniform(lo - 0.05, hi + 0.05, size=(max(COUNTS), 3)))
    rows: list[dict[str, object]] = []
    exactness: list[dict[str, object]] = []
    for count in COUNTS:
        points = np.ascontiguousarray(all_points[:count], dtype=np.float64)
        ref_times: list[float] = []
        compiled_times: list[float] = []
        pair_times: list[float] = []
        fd_times: list[float] = []
        reference_winding = np.empty(count, dtype=np.float64)
        compiled_winding = np.empty(count, dtype=np.float64)
        fd_stats: dict[str, int] = {}

        def reference_query(value: np.ndarray = points) -> np.ndarray:
            return generalized_winding_number(value, triangles)

        def compiled_query(value: np.ndarray = points) -> np.ndarray:
            return winding_handle.query(value)

        def closest_and_winding(value: np.ndarray = points) -> object:
            return backend._query_local(value)

        def full_fd(value: np.ndarray = points) -> object:
            return backend.spatial_fd_gradient_scene(value, np.eye(4), 1e-5)

        cold_first_query_s, _ = _timed(compiled_query)
        allocation_bytes = _allocated_bytes(compiled_query)
        for _ in range(REPEATS):
            elapsed, reference_winding = _timed(reference_query)
            ref_times.append(elapsed)
            elapsed, compiled_winding = _timed(compiled_query)
            compiled_times.append(elapsed)
            elapsed, _result = _timed(closest_and_winding)
            pair_times.append(elapsed)
            elapsed, _fd = _timed(full_fd)
            fd_times.append(elapsed)
            fd_stats = dict(backend.probe_sign_stats)
        reference_winding = cast(np.ndarray, reference_winding)
        compiled_winding = cast(np.ndarray, compiled_winding)
        ref_inside, _confidence, _ambiguous, _magnitude = winding_sign(reference_winding)
        compiled_inside, _confidence, _ambiguous, _magnitude = winding_sign(compiled_winding)
        exactness.append(
            {
                "dataset_object": "DexYCB frozen object",
                "points": count,
                "winding_max_abs_error": float(
                    np.max(np.abs(reference_winding - compiled_winding))
                ),
                "sign_mismatch": int(np.count_nonzero(ref_inside != compiled_inside)),
                "fallback": 0,
                "pass": bool(np.array_equal(ref_inside, compiled_inside)),
            }
        )
        common = {
            "points": count,
            "cold_import_s": cold_import_s,
            "cold_first_query_s": cold_first_query_s,
            "python_allocation_bytes": allocation_bytes,
            "input_bytes": int(points.nbytes),
            "copied_bytes": int(points.nbytes),
            "exact_fallback_count": 0,
            "sign_mismatch": int(np.count_nonzero(ref_inside != compiled_inside)),
        }
        for name, values in (
            ("reference_winding", ref_times),
            ("compiled_winding", compiled_times),
            ("compiled_closest_plus_winding", pair_times),
            ("full_ambiguous_spatial_fd", fd_times),
        ):
            rows.append({"mode": name, **common, **_summary(values)})
        rows.append(
            {
                "mode": "certified_probe_reuse_only",
                **common,
                **_summary(fd_times),
                "certified_probe_reuse": int(fd_stats.get("certified_probe_reuse", 0)),
                "exact_probe_sign_calls": int(fd_stats.get("exact_probe_sign_calls", 0)),
            }
        )
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(100):
        backend.spatial_fd_gradient_scene(all_points[:64], np.eye(4), 1e-5)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    output = ROOT / ".local/experiments/compiled_exact_sign_v1"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "microbenchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": "toporetarget.compiled_exact_sign.microbenchmark.v1",
        "mesh_hash": resources.mesh_hash,
        "counts": list(COUNTS),
        "repeats": REPEATS,
        "single_thread": True,
        "cold_import_s": cold_import_s,
        "rows": rows,
        "exactness": exactness,
        "rss_before_kib": int(before),
        "rss_after_kib": int(after),
        "rss_delta_kib": int(after - before),
        "long_loop_pass": bool(after - before <= 8192),
    }
    (output / "microbenchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "microbenchmark.html").write_text(
        "<html><body><pre>"
        + json.dumps(payload, indent=2, sort_keys=True)
        + "</pre></body></html>\n",
        encoding="utf-8",
    )
    report = ROOT / ".local/reports/compiled_exact_sign_v1"
    report.mkdir(parents=True, exist_ok=True)
    (report / "compiled_winding_exactness.json").write_text(
        json.dumps(exactness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report / "build_memory.json").write_text(
        json.dumps(
            {"rss_before_kib": before, "rss_after_kib": after, "rss_delta_kib": after - before},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(bool(item["pass"]) for item in exactness) else 1


if __name__ == "__main__":
    raise SystemExit(main())
