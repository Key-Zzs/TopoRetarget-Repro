"""Versioned, additive trajectory evaluation for TopoRetarget."""

from .aggregate import aggregate_rollouts, timeline_rows
from .contracts import EvaluationSuiteV2, PhysicsEpisodeEvidence
from .hand_metrics import EvaluationFingertipSetV1, EvaluationJointSetV1, hand_metric_series
from .object_metrics import object_metric_series, quaternion_geodesic_deg
from .stage16_pf_v2_causal_lift import (
    Stage16PhysicalFunctionalityV2Contract,
    evaluate_stage16_physical_functionality_v2,
)
from .success_metrics import trajectory_success

__all__ = [
    "EvaluationFingertipSetV1",
    "EvaluationJointSetV1",
    "EvaluationSuiteV2",
    "PhysicsEpisodeEvidence",
    "aggregate_rollouts",
    "hand_metric_series",
    "object_metric_series",
    "quaternion_geodesic_deg",
    "Stage16PhysicalFunctionalityV2Contract",
    "timeline_rows",
    "trajectory_success",
    "evaluate_stage16_physical_functionality_v2",
]
