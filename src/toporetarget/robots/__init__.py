"""Generic robot-hand kinematics and Arti-MANO target adapters."""

from .anchors import RobotKeypointSet
from .artimano import load_artimano_model
from .base import RobotHandModel
from .contracts import (
    RobotCollisionProfile,
    RobotHandAssetBundle,
    RobotKinematicSpec,
    RobotSemanticAnchorProfile,
    RobotSimulationSpec,
    RobotSurfaceProfile,
)
from .registry import RobotHandRegistry, get_robot_registry
from .spec import RobotHandSpec

__all__ = [
    "RobotHandModel",
    "RobotHandAssetBundle",
    "RobotKinematicSpec",
    "RobotSemanticAnchorProfile",
    "RobotSurfaceProfile",
    "RobotCollisionProfile",
    "RobotSimulationSpec",
    "RobotKeypointSet",
    "RobotHandRegistry",
    "RobotHandSpec",
    "get_robot_registry",
    "load_artimano_model",
]
