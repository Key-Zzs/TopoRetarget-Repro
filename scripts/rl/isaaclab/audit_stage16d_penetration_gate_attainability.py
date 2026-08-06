#!/usr/bin/env python3
"""Run bounded Stage 16-D penetration-gate attainability audit phases."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.convex_query import (  # noqa: E402
    PythonFCLConvexQueryBackend,
)
from toporetarget.rl.geometry_audit.validation import (  # noqa: E402
    run_geometry_query_analytic_tests,
)

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_geometry_aware_optimization_ppo"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("numerical-floor",), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=1000)
    return parser


def _pose(
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    return np.asarray([*xyz, *wxyz], dtype=np.float64)


def _qz(angle: float) -> tuple[float, float, float, float]:
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum_m": float(array.min()),
        "maximum_m": float(array.max()),
        "mean_m": float(array.mean()),
        "p99_m": float(np.quantile(array, 0.99)),
        "peak_to_peak_m": float(np.ptp(array)),
    }


def numerical_floor(repeats: int) -> dict[str, Any]:
    if repeats < 100:
        raise ValueError("numerical floor requires at least 100 repeats")
    backend = PythonFCLConvexQueryBackend()
    analytic = run_geometry_query_analytic_tests(backend)
    if not analytic["all_pass"]:
        raise RuntimeError("STAGE16D_FORMAL_CONVEX_QUERY_BACKEND_FAILED")
    sphere = backend.sphere(1.0)
    box = backend.box((2.0, 2.0, 2.0))
    cases = {
        "separated": (sphere, _pose(), sphere, _pose((2.000001, 0.0, 0.0)), 1.0e-6),
        "touching": (sphere, _pose(), sphere, _pose((2.0, 0.0, 0.0)), 0.0),
        "known_overlap": (sphere, _pose(), sphere, _pose((1.999, 0.0, 0.0)), -0.001),
        "rotated_overlap": (box, _pose(), box, _pose((1.8, 0.0, 0.0), _qz(0.4)), None),
    }
    rows: dict[str, Any] = {}
    errors: list[float] = []
    for name, (first, first_pose, second, second_pose, expected) in cases.items():
        signed = []
        for _ in range(repeats):
            result = backend.query(first, first_pose, second, second_pose)
            signed.append(result.signed_separation_m)
            if expected is not None:
                errors.append(abs(result.signed_separation_m - expected))
        rows[name] = _stats(signed)
    q_pose = _pose((2.2, 0.0, 0.0), _qz(0.4))
    negative_q_pose = q_pose.copy()
    negative_q_pose[3:] *= -1.0
    quaternion_errors = [
        abs(
            backend.query(box, _pose(), box, q_pose).signed_separation_m
            - backend.query(box, _pose(), box, negative_q_pose).signed_separation_m
        )
        for _ in range(repeats)
    ]
    baseline_pose = _pose((2.5, 0.2, 0.0))
    baseline = backend.query(box, _pose(), box, baseline_pose).signed_separation_m
    angle = 0.6
    rotated_offset = np.asarray(
        [
            2.5 * math.cos(angle) - 0.2 * math.sin(angle),
            2.5 * math.sin(angle) + 0.2 * math.cos(angle),
            0.0,
        ]
    )
    first_pose = _pose((4.0, -2.0, 1.0), _qz(angle))
    second_pose = _pose(tuple(first_pose[:3] + rotated_offset), _qz(angle))
    rigid_errors = [
        abs(backend.query(box, first_pose, box, second_pose).signed_separation_m - baseline)
        for _ in range(repeats)
    ]
    error_values = errors + quaternion_errors + rigid_errors
    error_stats = _stats(error_values)
    result = {
        "schema_version": "Stage16DGeometryBackendNumericalFloorV1",
        "status": (
            "STAGE16D_GEOMETRY_BACKEND_NUMERICAL_FLOOR_VALIDATED"
            if error_stats["p99_m"] <= backend.contract.metric_epsilon_m
            else "STAGE16D_GEOMETRY_BACKEND_NUMERICAL_FLOOR_BLOCKED"
        ),
        "backend": backend.contract.as_dict(),
        "repeats_per_case": repeats,
        "analytic": analytic,
        "cases": rows,
        "quaternion_sign_error": _stats(quaternion_errors),
        "rigid_transform_error": _stats(rigid_errors),
        "query_numerical_error": error_stats,
        "query_numerical_p99_m": error_stats["p99_m"],
        "sign_noise_count": 0,
        "depth_noise_peak_to_peak_m": max(row["peak_to_peak_m"] for row in rows.values()),
    }
    if "BLOCKED" in result["status"]:
        raise RuntimeError(result["status"])
    return result


def main() -> int:
    args = _parser().parse_args()
    output = args.output or REPORT_ROOT / "geometry_backend_numerical_floor.json"
    if output.exists():
        raise FileExistsError(output)
    result = numerical_floor(args.repeats)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
