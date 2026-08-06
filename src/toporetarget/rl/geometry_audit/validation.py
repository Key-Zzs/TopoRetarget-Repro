"""Analytic qualification suite for the formal python-fcl query backend."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .convex_query import ConvexQueryResult, PythonFCLConvexQueryBackend


def _pose(
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    return np.asarray([*xyz, *wxyz], dtype=np.float64)


def _qz(angle: float) -> tuple[float, float, float, float]:
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def _result(
    name: str,
    query: ConvexQueryResult,
    passed: bool,
    **details: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "signed_separation_m": query.signed_separation_m,
        "penetration_depth_m": query.penetration_depth_m,
        "depenetration_direction_for_second": list(query.depenetration_direction_for_second),
        "converged": query.converged,
        **details,
    }


def _moved_second_pose(
    pose: np.ndarray, query: ConvexQueryResult, extra_distance: float
) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float64).copy()
    result[:3] += (query.penetration_depth_m + extra_distance) * np.asarray(
        query.depenetration_direction_for_second
    )
    return result


def run_geometry_query_analytic_tests(
    backend: PythonFCLConvexQueryBackend,
) -> dict[str, Any]:
    tolerance = backend.contract.metric_epsilon_m
    rows: list[dict[str, Any]] = []

    sphere = backend.sphere(1.0)
    for name, center, expected in (
        ("sphere_sphere_separated", 3.0, 1.0),
        ("sphere_sphere_touching", 2.0, 0.0),
        ("sphere_sphere_overlap", 1.5, -0.5),
    ):
        query = backend.query(sphere, _pose(), sphere, _pose((center, 0.0, 0.0)))
        rows.append(
            _result(
                name,
                query,
                abs(query.signed_separation_m - expected) <= tolerance,
                expected_signed_separation_m=expected,
            )
        )

    box = backend.box((2.0, 2.0, 2.0))
    box_overlap_pose = _pose((1.5, 0.0, 0.0))
    box_overlap = backend.query(box, _pose(), box, box_overlap_pose)
    moved = backend.query(
        box,
        _pose(),
        box,
        _moved_second_pose(box_overlap_pose, box_overlap, 10.0 * tolerance),
    )
    rows.append(
        _result(
            "box_box_axis_aligned_and_depenetration",
            box_overlap,
            abs(box_overlap.signed_separation_m + 0.5) <= tolerance
            and moved.signed_separation_m >= 0.0,
            expected_signed_separation_m=-0.5,
            moved_signed_separation_m=moved.signed_separation_m,
        )
    )

    rotated_pose = _pose((1.8, 0.0, 0.0), _qz(math.pi / 4.0))
    rotated = backend.query(box, _pose(), box, rotated_pose)
    rotated_moved = backend.query(
        box,
        _pose(),
        box,
        _moved_second_pose(rotated_pose, rotated, 10.0 * tolerance),
    )
    rows.append(
        _result(
            "rotated_box_box_and_depenetration",
            rotated,
            rotated.signed_separation_m < 0.0 and rotated_moved.signed_separation_m >= 0.0,
            moved_signed_separation_m=rotated_moved.signed_separation_m,
        )
    )

    capsule = backend.capsule(0.5, 2.0)
    capsule_query = backend.query(box, _pose(), capsule, _pose((0.75, 0.0, 0.0)))
    capsule_moved = backend.query(
        box,
        _pose(),
        capsule,
        _moved_second_pose(_pose((0.75, 0.0, 0.0)), capsule_query, 10.0 * tolerance),
    )
    rows.append(
        _result(
            "capsule_box_and_depenetration",
            capsule_query,
            capsule_query.signed_separation_m < 0.0 and capsule_moved.signed_separation_m >= 0.0,
            moved_signed_separation_m=capsule_moved.signed_separation_m,
        )
    )

    vertices = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    convex = backend.convex(vertices, faces)
    convex_query = backend.query(convex, _pose(), convex, _pose((1.25, 0.0, 0.0)))
    rows.append(
        _result(
            "convex_hull_known_translation",
            convex_query,
            abs(convex_query.signed_separation_m + 0.75) <= tolerance,
            expected_signed_separation_m=-0.75,
        )
    )

    baseline = backend.query(box, _pose(), box, _pose((2.5, 0.2, 0.0)))
    rigid_first = _pose((4.0, -2.0, 1.0), _qz(0.6))
    offset_world = (
        np.asarray([2.5, 0.2, 0.0])
        @ np.asarray(
            [
                [math.cos(0.6), -math.sin(0.6), 0.0],
                [math.sin(0.6), math.cos(0.6), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ).T
    )
    rigid_second = _pose(tuple(rigid_first[:3] + offset_world), _qz(0.6))
    rigid = backend.query(box, rigid_first, box, rigid_second)
    rows.append(
        _result(
            "rigid_transform_invariance",
            rigid,
            abs(rigid.signed_separation_m - baseline.signed_separation_m) <= tolerance,
            baseline_signed_separation_m=baseline.signed_separation_m,
        )
    )

    order_forward = backend.query(box, _pose(), sphere, _pose((1.2, 0.0, 0.0)))
    order_reverse = backend.query(sphere, _pose((1.2, 0.0, 0.0)), box, _pose())
    directions_opposite = (
        np.linalg.norm(
            np.asarray(order_forward.depenetration_direction_for_second)
            + np.asarray(order_reverse.depenetration_direction_for_second)
        )
        <= tolerance
    )
    rows.append(
        _result(
            "shape_order_symmetry",
            order_forward,
            abs(order_forward.signed_separation_m - order_reverse.signed_separation_m) <= tolerance
            and directions_opposite,
            reverse_signed_separation_m=order_reverse.signed_separation_m,
            directions_opposite=bool(directions_opposite),
        )
    )

    quaternion_pose = _pose((2.2, 0.0, 0.0), _qz(0.4))
    q_result = backend.query(box, _pose(), box, quaternion_pose)
    quaternion_pose[3:] *= -1.0
    negative_q_result = backend.query(box, _pose(), box, quaternion_pose)
    rows.append(
        _result(
            "quaternion_sign_invariance",
            q_result,
            abs(q_result.signed_separation_m - negative_q_result.signed_separation_m) <= tolerance,
            negative_q_signed_separation_m=negative_q_result.signed_separation_m,
        )
    )

    small = backend.sphere(0.5)
    small_query = backend.query(small, _pose(), small, _pose((1.5, 0.0, 0.0)))
    rows.append(
        _result(
            "scale_consistency",
            small_query,
            abs(small_query.signed_separation_m - 0.5) <= tolerance,
            expected_signed_separation_m=0.5,
        )
    )

    near_gap = backend.query(sphere, _pose(), sphere, _pose((2.0 + 1.0e-6, 0.0, 0.0)))
    near_overlap = backend.query(sphere, _pose(), sphere, _pose((2.0 - 1.0e-6, 0.0, 0.0)))
    rows.append(
        _result(
            "near_touch_numerical_stability",
            near_gap,
            near_gap.signed_separation_m > 0.0 and near_overlap.signed_separation_m < 0.0,
            overlap_signed_separation_m=near_overlap.signed_separation_m,
        )
    )

    deterministic_first = backend.query(box, _pose(), box, rotated_pose)
    deterministic_second = backend.query(box, _pose(), box, rotated_pose)
    rows.append(
        _result(
            "deterministic_repeat",
            deterministic_first,
            deterministic_first == deterministic_second,
        )
    )

    passed = all(row["pass"] and row["converged"] for row in rows)
    return {
        "schema_version": "Stage16DGeometryQueryAnalyticTestsV1",
        "backend": backend.contract.as_dict(),
        "tests": rows,
        "test_count": len(rows),
        "passed_count": sum(bool(row["pass"]) for row in rows),
        "all_pass": passed,
        "status": (
            "STAGE16D_FORMAL_CONVEX_QUERY_BACKEND_VALIDATED"
            if passed
            else "STAGE16D_FORMAL_CONVEX_QUERY_BACKEND_BLOCKED"
        ),
    }


def require_qualified_backend(factory: Callable[[], PythonFCLConvexQueryBackend]) -> dict[str, Any]:
    result = run_geometry_query_analytic_tests(factory())
    if not result["all_pass"]:
        failures = [row["name"] for row in result["tests"] if not row["pass"]]
        raise RuntimeError(f"STAGE16D_FORMAL_CONVEX_QUERY_BACKEND_FAILED:{failures}")
    return result


__all__ = ["require_qualified_backend", "run_geometry_query_analytic_tests"]
