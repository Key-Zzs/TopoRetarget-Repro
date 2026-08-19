"""Qualified python-fcl convex separation and penetration queries.

The dependency is imported lazily so the base ``toporetarget-rl`` package keeps
working without Isaac or python-fcl installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import GEOMETRY_QUERY_CONTRACT, GeometryQueryContractV1
from .runtime_geometry import ConvexProxyGeometry
from .transforms import quaternion_matrix_wxyz


def _tuple3(value: np.ndarray, *, label: str) -> tuple[float, float, float]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise RuntimeError(f"STAGE16D_CONVEX_QUERY_INVALID_VECTOR3:{label}")
    return (float(vector[0]), float(vector[1]), float(vector[2]))


@dataclass(frozen=True)
class ConvexQueryResult:
    signed_separation_m: float
    penetration_depth_m: float
    depenetration_direction_for_second: tuple[float, float, float]
    nearest_point_first: tuple[float, float, float] | None
    nearest_point_second: tuple[float, float, float] | None
    converged: bool
    colliding: bool


class PythonFCLConvexQueryBackend:
    """Single qualified collision backend for formal Stage 16-D metrics."""

    def __init__(self, contract: GeometryQueryContractV1 = GEOMETRY_QUERY_CONTRACT) -> None:
        try:
            import fcl
        except ImportError as exc:  # pragma: no cover - exercised in the Isaac environment
            raise RuntimeError("STAGE16D_FORMAL_CONVEX_QUERY_BACKEND_UNAVAILABLE") from exc
        version = str(getattr(fcl, "__version__", ""))
        if version != contract.backend_version:
            raise RuntimeError(
                f"STAGE16D_CONVEX_BACKEND_VERSION_DRIFT:{version}!={contract.backend_version}"
            )
        self.fcl = fcl
        self.contract = contract

    def convex(self, vertices: np.ndarray, faces: np.ndarray) -> Any:
        points = np.asarray(vertices, dtype=np.float64)
        triangles = np.asarray(faces, dtype=np.int32)
        encoded_faces = np.concatenate(
            [np.concatenate((np.asarray([3], dtype=np.int32), row)) for row in triangles]
        )
        return self.fcl.Convex(points, len(triangles), encoded_faces)

    def proxy_shape(self, proxy: ConvexProxyGeometry) -> Any:
        return self.convex(proxy.scaled_vertices, proxy.faces)

    def sphere(self, radius: float) -> Any:
        return self.fcl.Sphere(float(radius))

    def box(self, side_xyz: tuple[float, float, float]) -> Any:
        return self.fcl.Box(*side_xyz)

    def capsule(self, radius: float, length: float) -> Any:
        return self.fcl.Capsule(float(radius), float(length))

    def transform(self, pose_xyz_wxyz: np.ndarray) -> Any:
        pose = np.asarray(pose_xyz_wxyz, dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("python-fcl transform must be finite xyz+wxyz")
        return self.fcl.Transform(quaternion_matrix_wxyz(pose[3:]), pose[:3])

    def query(
        self,
        first_shape: Any,
        first_pose: np.ndarray,
        second_shape: Any,
        second_pose: np.ndarray,
        *,
        collision_mtd_only: bool = False,
    ) -> ConvexQueryResult:
        first_object = self.fcl.CollisionObject(first_shape, self.transform(first_pose))
        second_object = self.fcl.CollisionObject(second_shape, self.transform(second_pose))
        collision_request = self.fcl.CollisionRequest(
            num_max_contacts=1,
            enable_contact=True,
            gjk_solver_type=self.fcl.GJKSolverType.GST_LIBCCD,
        )
        collision_result = self.fcl.CollisionResult()
        self.fcl.collide(first_object, second_object, collision_request, collision_result)
        if collision_result.is_collision and collision_mtd_only:
            if not collision_result.contacts:
                raise RuntimeError("STAGE16D_CONVEX_QUERY_MISSING_CONTACT_MTD")
            contact = collision_result.contacts[0]
            depth = max(0.0, float(contact.penetration_depth))
            direction = np.asarray(contact.normal, dtype=np.float64)
            if not np.isfinite(depth) or not np.isfinite(direction).all():
                raise RuntimeError("STAGE16D_CONVEX_QUERY_NONFINITE_OVERLAP")
            norm = float(np.linalg.norm(direction))
            if depth > self.contract.numerical_tolerance_m and norm <= 0.0:
                raise RuntimeError("STAGE16D_CONVEX_QUERY_INVALID_MTD_DIRECTION")
            if norm > 0.0:
                direction /= norm
            return ConvexQueryResult(
                signed_separation_m=-depth,
                penetration_depth_m=depth,
                depenetration_direction_for_second=_tuple3(direction, label="overlap_normal"),
                nearest_point_first=None,
                nearest_point_second=None,
                converged=True,
                colliding=depth > self.contract.numerical_tolerance_m,
            )
        distance_request = self.fcl.DistanceRequest(
            enable_nearest_points=True,
            enable_signed_distance=True,
            gjk_solver_type=self.fcl.GJKSolverType.GST_LIBCCD,
        )
        distance_result = self.fcl.DistanceResult()
        distance = float(
            self.fcl.distance(first_object, second_object, distance_request, distance_result)
        )
        if collision_result.is_collision:
            if not collision_result.contacts:
                raise RuntimeError("STAGE16D_CONVEX_QUERY_MISSING_CONTACT_MTD")
            contact = collision_result.contacts[0]
            depth = max(0.0, float(contact.penetration_depth))
            direction = np.asarray(contact.normal, dtype=np.float64)
            if not np.isfinite(depth) or not np.isfinite(direction).all():
                raise RuntimeError("STAGE16D_CONVEX_QUERY_NONFINITE_OVERLAP")
            norm = float(np.linalg.norm(direction))
            if depth > self.contract.numerical_tolerance_m and norm <= 0.0:
                raise RuntimeError("STAGE16D_CONVEX_QUERY_INVALID_MTD_DIRECTION")
            if norm > 0.0:
                direction /= norm
            if not np.isfinite(distance) or distance > self.contract.metric_epsilon_m:
                raise RuntimeError(
                    "STAGE16D_CONVEX_QUERY_SIGNED_DISTANCE_MTD_SIGN_MISMATCH:"
                    f"distance={distance}:depth={depth}"
                )
            return ConvexQueryResult(
                signed_separation_m=-depth,
                penetration_depth_m=depth,
                depenetration_direction_for_second=_tuple3(direction, label="overlap_normal"),
                nearest_point_first=None,
                nearest_point_second=None,
                converged=True,
                colliding=depth > self.contract.numerical_tolerance_m,
            )
        point_first = np.asarray(distance_result.nearest_points[0], dtype=np.float64)
        point_second = np.asarray(distance_result.nearest_points[1], dtype=np.float64)
        if (
            not np.isfinite(distance)
            or not np.isfinite(point_first).all()
            or not np.isfinite(point_second).all()
        ):
            raise RuntimeError("STAGE16D_CONVEX_QUERY_NONFINITE_DISTANCE")
        direction = point_second - point_first
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 0.0 else np.zeros(3, dtype=np.float64)
        signed = max(0.0, distance)
        if signed <= self.contract.numerical_tolerance_m:
            signed = 0.0
        return ConvexQueryResult(
            signed_separation_m=signed,
            penetration_depth_m=0.0,
            depenetration_direction_for_second=_tuple3(direction, label="separation_direction"),
            nearest_point_first=_tuple3(point_first, label="nearest_point_first"),
            nearest_point_second=_tuple3(point_second, label="nearest_point_second"),
            converged=True,
            colliding=False,
        )


__all__ = [
    "ConvexQueryResult",
    "PythonFCLConvexQueryBackend",
]
