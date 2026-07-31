"""Paper Table-5 ranges with reproducible independent switches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

PAPER_DOMAIN_RANDOMIZATION_ID = "paper_table5_v1"


@dataclass(frozen=True)
class RandomizationSwitches:
    """Independent Table-5 switches; a disabled item has a nominal identity value."""

    observation_noise: bool = True
    observation_delay: bool = True
    reference_reset: bool = True
    object_com: bool = True
    robot_friction_and_geometry: bool = True
    object_mass_and_inertia: bool = True
    pd: bool = True
    joint_dynamics: bool = True
    encoder_bias: bool = True
    robot_link_mass_and_inertia: bool = True
    external_disturbance: bool = True


@dataclass(frozen=True)
class DomainRandomizationConfig:
    enabled: bool = True
    switches: RandomizationSwitches = field(default_factory=RandomizationSwitches)
    joint_position_noise_std_rad: float = 0.02
    joint_velocity_noise_std_radps: float = 0.05
    axis_position_noise_std_m: float = 0.002
    axis_orientation_noise_std_rad: float = 0.01
    observation_delay_steps: tuple[int, int] = (0, 2)
    reset_joint_range_rad: tuple[float, float] = (-0.02, 0.02)
    reset_object_position_range_m: tuple[float, float] = (-0.005, 0.005)
    reset_object_orientation_angle_rad: tuple[float, float] = (-0.03, 0.03)
    object_com_offset_m: tuple[float, float] = (-0.003, 0.003)
    robot_friction_scale: tuple[float, float] = (0.7, 1.3)
    robot_collision_geometry_scale: tuple[float, float] = (0.9, 1.1)
    object_mass_inertia_scale: tuple[float, float] = (0.4, 1.6)
    pd_stiffness_scale: tuple[float, float] = (0.75, 1.5)
    pd_damping_scale: tuple[float, float] = (0.5, 2.0)
    joint_damping_scale: tuple[float, float] = (0.3, 3.0)
    joint_armature_scale: tuple[float, float] = (0.75, 1.3)
    joint_friction_loss_scale: tuple[float, float] = (0.5, 2.0)
    encoder_bias_rad: tuple[float, float] = (-0.01, 0.01)
    robot_link_inertia_scale: tuple[float, float] = (0.4, 1.5)
    robot_link_mass_scale: tuple[float, float] = (0.4, 1.5)
    external_object_force_n: tuple[float, float] = (-0.25, 0.25)
    external_object_torque_nm: tuple[float, float] = (-0.00375, 0.00375)
    disturbance_interval_s: tuple[float, float] = (0.6, 1.8)


def _log_uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return float(np.exp(rng.uniform(np.log(lower), np.log(upper))))


def sample_randomization(
    rng: np.random.Generator, config: DomainRandomizationConfig = DomainRandomizationConfig()
) -> dict[str, Any]:
    """Sample all non-observed Table-5 runtime parameters deterministically."""

    active = asdict(config.switches)
    if not config.enabled:
        active = {key: False for key in active}

    def uniform(bounds: tuple[float, float]) -> float:
        return float(rng.uniform(*bounds))

    def vector(bounds: tuple[float, float]) -> list[float]:
        return rng.uniform(*bounds, size=3).tolist()

    orientation_axis = rng.normal(size=3)
    orientation_axis /= np.linalg.norm(orientation_axis)
    return {
        "enabled": config.enabled,
        "active_switches": active,
        "config": asdict(config),
        "observation_delay_steps": (
            int(
                rng.integers(
                    config.observation_delay_steps[0], config.observation_delay_steps[1] + 1
                )
            )
            if active["observation_delay"]
            else 0
        ),
        "reset_joint_noise_rad": (
            uniform(config.reset_joint_range_rad) if active["reference_reset"] else 0.0
        ),
        "reset_object_position_noise_m": (
            vector(config.reset_object_position_range_m)
            if active["reference_reset"]
            else [0.0, 0.0, 0.0]
        ),
        "reset_object_orientation_axis": orientation_axis.tolist(),
        "reset_object_orientation_angle_rad": (
            uniform(config.reset_object_orientation_angle_rad) if active["reference_reset"] else 0.0
        ),
        "object_com_offset_m": (
            vector(config.object_com_offset_m) if active["object_com"] else [0.0, 0.0, 0.0]
        ),
        "robot_friction_scale": (
            uniform(config.robot_friction_scale) if active["robot_friction_and_geometry"] else 1.0
        ),
        "robot_collision_geometry_scale": (
            uniform(config.robot_collision_geometry_scale)
            if active["robot_friction_and_geometry"]
            else 1.0
        ),
        "object_mass_inertia_scale": (
            uniform(config.object_mass_inertia_scale) if active["object_mass_and_inertia"] else 1.0
        ),
        "pd_stiffness_scale": (
            _log_uniform(rng, config.pd_stiffness_scale) if active["pd"] else 1.0
        ),
        "pd_damping_scale": (_log_uniform(rng, config.pd_damping_scale) if active["pd"] else 1.0),
        "joint_damping_scale": (
            _log_uniform(rng, config.joint_damping_scale) if active["joint_dynamics"] else 1.0
        ),
        "joint_armature_scale": (
            uniform(config.joint_armature_scale) if active["joint_dynamics"] else 1.0
        ),
        "joint_friction_loss_scale": (
            _log_uniform(rng, config.joint_friction_loss_scale) if active["joint_dynamics"] else 1.0
        ),
        "encoder_bias_rad": (uniform(config.encoder_bias_rad) if active["encoder_bias"] else 0.0),
        "robot_link_inertia_scale": (
            uniform(config.robot_link_inertia_scale)
            if active["robot_link_mass_and_inertia"]
            else 1.0
        ),
        "robot_link_mass_scale": (
            uniform(config.robot_link_mass_scale) if active["robot_link_mass_and_inertia"] else 1.0
        ),
        "external_object_force_n": (
            vector(config.external_object_force_n)
            if active["external_disturbance"]
            else [0.0, 0.0, 0.0]
        ),
        "external_object_torque_nm": (
            vector(config.external_object_torque_nm)
            if active["external_disturbance"]
            else [0.0, 0.0, 0.0]
        ),
        "next_disturbance_s": (
            uniform(config.disturbance_interval_s)
            if active["external_disturbance"]
            else float("inf")
        ),
    }


def apply_observation_noise(
    *,
    q: np.ndarray,
    qdot: np.ndarray,
    axis_points: np.ndarray,
    rng: np.random.Generator,
    config: DomainRandomizationConfig = DomainRandomizationConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not config.enabled or not config.switches.observation_noise:
        return np.asarray(q).copy(), np.asarray(qdot).copy(), np.asarray(axis_points).copy()
    axes = np.asarray(axis_points, dtype=np.float64)
    if axes.shape != (6, 3):
        raise ValueError("axis_points must have shape [6,3]")
    rotation_axis = rng.normal(size=3)
    rotation_axis /= np.linalg.norm(rotation_axis)
    angle = float(rng.normal(0.0, config.axis_orientation_noise_std_rad))
    cross = np.asarray(
        [
            [0.0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0.0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0.0],
        ]
    )
    rotation = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
    center = axes.mean(axis=0)
    rotated = (axes - center) @ rotation.T + center
    return (
        np.asarray(q)
        + rng.normal(0.0, config.joint_position_noise_std_rad, size=np.asarray(q).shape),
        np.asarray(qdot)
        + rng.normal(0.0, config.joint_velocity_noise_std_radps, size=np.asarray(qdot).shape),
        rotated + rng.normal(0.0, config.axis_position_noise_std_m, size=axes.shape),
    )


__all__ = [
    "DomainRandomizationConfig",
    "PAPER_DOMAIN_RANDOMIZATION_ID",
    "RandomizationSwitches",
    "apply_observation_noise",
    "sample_randomization",
]
