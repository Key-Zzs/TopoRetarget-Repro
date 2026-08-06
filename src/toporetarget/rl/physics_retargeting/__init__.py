"""Physics-consistent retargeting contracts and bounded optimization tools."""

from .contracts import (
    PHYSICS_CONSISTENT_RETARGETING_PROTOCOL,
    PersistentContactTopologyV1,
    PhysicsConsistentTaskGateV1,
    TaskSemanticContractV1,
)
from .recovery import Stage16DRecoveryStateMachine
from .spline_actions import PiecewiseSplineResidualV1

__all__ = [
    "PHYSICS_CONSISTENT_RETARGETING_PROTOCOL",
    "PhysicsConsistentTaskGateV1",
    "PersistentContactTopologyV1",
    "PiecewiseSplineResidualV1",
    "Stage16DRecoveryStateMachine",
    "TaskSemanticContractV1",
]
