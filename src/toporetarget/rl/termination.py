"""Mutually exclusive literal Table-4 reference-tracking termination."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class TerminationType(StrEnum):
    SUCCESS_REFERENCE_COMPLETE = "SUCCESS_REFERENCE_COMPLETE"
    FAILURE_TIMEOUT = "FAILURE_TIMEOUT"
    FAILURE_OBJECT_HEIGHT = "FAILURE_OBJECT_HEIGHT"
    FAILURE_OBJECT_LINEAR_VELOCITY = "FAILURE_OBJECT_LINEAR_VELOCITY"
    FAILURE_OBJECT_ANGULAR_VELOCITY = "FAILURE_OBJECT_ANGULAR_VELOCITY"
    FAILURE_OBJECT_POSITION = "FAILURE_OBJECT_POSITION"
    FAILURE_OBJECT_ORIENTATION = "FAILURE_OBJECT_ORIENTATION"
    FAILURE_OBJECT_AXIS_POINT = "FAILURE_OBJECT_AXIS_POINT"
    FAILURE_NUMERICAL = "FAILURE_NUMERICAL"
    FAILURE_SIMULATOR = "FAILURE_SIMULATOR"


@dataclass(frozen=True)
class TerminationProfile:
    episode_control_steps: int = 400
    object_height_min_m: float = 0.06
    object_linear_velocity_max_mps: float = 10.0
    object_angular_velocity_max_radps: float = 500.0
    object_position_error_max_m: float = 0.05
    object_orientation_error_max_rad: float = float(np.deg2rad(45.0))
    object_axis_point_error_max_m: float = 0.05


PAPER_TERMINATION = TerminationProfile()


@dataclass(frozen=True)
class TerminationInput:
    step: int
    reference_index: int
    reference_frame_count: int
    object_height_m: float
    object_linear_velocity_mps: float
    object_angular_velocity_radps: float
    object_position_error_m: float
    object_orientation_error_rad: float
    max_axis_point_error_m: float
    simulator_error: bool = False


def classify_termination(
    value: TerminationInput, *, profile: TerminationProfile = PAPER_TERMINATION
) -> TerminationType | None:
    """Return exactly one terminal class, with deterministic fail-closed priority."""

    numbers = np.asarray(
        [
            value.object_height_m,
            value.object_linear_velocity_mps,
            value.object_angular_velocity_radps,
            value.object_position_error_m,
            value.object_orientation_error_rad,
            value.max_axis_point_error_m,
        ]
    )
    if value.simulator_error:
        return TerminationType.FAILURE_SIMULATOR
    if not np.isfinite(numbers).all():
        return TerminationType.FAILURE_NUMERICAL
    if value.object_height_m < profile.object_height_min_m:
        return TerminationType.FAILURE_OBJECT_HEIGHT
    if value.object_linear_velocity_mps > profile.object_linear_velocity_max_mps:
        return TerminationType.FAILURE_OBJECT_LINEAR_VELOCITY
    if value.object_angular_velocity_radps > profile.object_angular_velocity_max_radps:
        return TerminationType.FAILURE_OBJECT_ANGULAR_VELOCITY
    if value.object_position_error_m > profile.object_position_error_max_m:
        return TerminationType.FAILURE_OBJECT_POSITION
    if value.object_orientation_error_rad > profile.object_orientation_error_max_rad:
        return TerminationType.FAILURE_OBJECT_ORIENTATION
    if value.max_axis_point_error_m > profile.object_axis_point_error_max_m:
        return TerminationType.FAILURE_OBJECT_AXIS_POINT
    if value.reference_index >= value.reference_frame_count - 1:
        return TerminationType.SUCCESS_REFERENCE_COMPLETE
    if value.step >= profile.episode_control_steps:
        return TerminationType.FAILURE_TIMEOUT
    return None


__all__ = [
    "PAPER_TERMINATION",
    "TerminationInput",
    "TerminationProfile",
    "TerminationType",
    "classify_termination",
]
