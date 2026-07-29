"""RobotHandPlugin v1 instances backed by the existing generic FK registry."""

from .registry import RobotHandPluginRegistry, get_robot_plugin_registry

__all__ = ["RobotHandPluginRegistry", "get_robot_plugin_registry"]
