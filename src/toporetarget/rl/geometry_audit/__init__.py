"""Metric-compatible runtime collision geometry qualification for Stage 16-D."""

from .contracts import (
    GEOMETRY_METRIC_CONTRACT,
    GEOMETRY_QUERY_CONTRACT,
    RuntimeCollisionProxyPenetrationV1,
)
from .dynamic_contact_reference import (
    BOOTSTRAP_UCB_CONTRACT,
    SelectedStableCalibrationV1,
    decide_geometry_v1_v2,
)
from .metrics import aggregate_penetration, qualify_source_corrected
from .transforms import compose_poses, quaternion_matrix_wxyz, transform_points

__all__ = [
    "GEOMETRY_METRIC_CONTRACT",
    "GEOMETRY_QUERY_CONTRACT",
    "BOOTSTRAP_UCB_CONTRACT",
    "RuntimeCollisionProxyPenetrationV1",
    "SelectedStableCalibrationV1",
    "aggregate_penetration",
    "compose_poses",
    "decide_geometry_v1_v2",
    "qualify_source_corrected",
    "quaternion_matrix_wxyz",
    "transform_points",
]
