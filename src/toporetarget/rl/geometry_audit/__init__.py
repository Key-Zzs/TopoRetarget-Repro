"""Metric-compatible runtime collision geometry qualification for Stage 16-D."""

from .contracts import (
    GEOMETRY_METRIC_CONTRACT,
    GEOMETRY_QUERY_CONTRACT,
    RuntimeCollisionProxyPenetrationV1,
)
from .metrics import aggregate_penetration, qualify_source_corrected
from .transforms import compose_poses, quaternion_matrix_wxyz, transform_points

__all__ = [
    "GEOMETRY_METRIC_CONTRACT",
    "GEOMETRY_QUERY_CONTRACT",
    "RuntimeCollisionProxyPenetrationV1",
    "aggregate_penetration",
    "compose_poses",
    "qualify_source_corrected",
    "quaternion_matrix_wxyz",
    "transform_points",
]
