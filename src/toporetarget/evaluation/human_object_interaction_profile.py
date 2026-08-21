"""Offline human-object contact and coupling profiles.

The profile in this module is descriptive.  It is not a reward, a success
gate, or a functional-grasp label.  All angular quantities are derived from
poses with the Reference Kinematics V2 estimator family.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
from scipy.spatial.transform import Rotation

from toporetarget.rl.reference_tracking.reference_kinematics import (
    derive_angular_velocity_world_wxyz,
    quaternion_to_matrix_wxyz,
    so3_log,
)

HUMAN_OBJECT_COUPLING_CONTACT_PROFILE_V1: Final = "HumanObjectCouplingContactProfileV1"


@dataclass(frozen=True)
class HumanObjectCouplingContactProfileContractV1:
    """Outcome-independent schema contract for source and rollout profiles."""

    schema_version: str = HUMAN_OBJECT_COUPLING_CONTACT_PROFILE_V1
    hand_frame: str = "authoritative wrist/root frame"
    relative_pose: str = "T_hand_to_object = inverse(T_world_hand) @ T_world_object"
    angular_estimator: str = "ReferenceKinematicsV2.SO3_log_centered_world_with_one_sided_endpoints"
    linear_estimator: str = "centered_pose_difference_with_one_sided_endpoints"
    coupling_ratio_epsilon: float = 1.0e-8
    variation_window_steps: int = 5
    raw_functional_grasp_binary_required: bool = False
    hard_coupling_threshold_defined: bool = False
    outcome_tuned: bool = False
    force_closure_claimed: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != HUMAN_OBJECT_COUPLING_CONTACT_PROFILE_V1:
            raise ValueError("HOC_PROFILE_SCHEMA_DRIFT")
        if self.coupling_ratio_epsilon <= 0.0 or self.variation_window_steps < 1:
            raise ValueError("HOC_PROFILE_NUMERICAL_CONTRACT_INVALID")
        if (
            self.raw_functional_grasp_binary_required
            or self.hard_coupling_threshold_defined
            or self.outcome_tuned
            or self.force_closure_claimed
        ):
            raise ValueError("HOC_PROFILE_UNSUPPORTED_BINARY_OR_OUTCOME_CLAIM")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_pose(name: str, value: np.ndarray, frame_count: int) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (frame_count, 7) or not np.isfinite(pose).all():
        raise ValueError(f"HOC_PROFILE_{name}_POSE_INVALID")
    quaternion_norm = np.linalg.norm(pose[:, 3:], axis=1)
    if np.any(quaternion_norm <= 1.0e-8):
        raise ValueError(f"HOC_PROFILE_{name}_QUATERNION_INVALID")
    pose = pose.copy()
    pose[:, 3:] /= quaternion_norm[:, None]
    return pose


def _pose_from_translation_rotation(
    translation: np.ndarray, rotation_matrix: np.ndarray
) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return np.concatenate((translation, quaternion_xyzw[:, 3:4], quaternion_xyzw[:, :3]), axis=1)


def _derivative(values: np.ndarray, timestamps_s: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    result[0] = (array[1] - array[0]) / (timestamps_s[1] - timestamps_s[0])
    result[-1] = (array[-1] - array[-2]) / (timestamps_s[-1] - timestamps_s[-2])
    if len(array) > 2:
        duration = timestamps_s[2:] - timestamps_s[:-2]
        reshape = (len(duration),) + (1,) * (array.ndim - 1)
        result[1:-1] = (array[2:] - array[:-2]) / duration.reshape(reshape)
    return result


def _window_rms_variation(values: np.ndarray, window_steps: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty(len(array), dtype=np.float64)
    radius = window_steps // 2
    for frame in range(len(array)):
        start = max(0, frame - radius)
        stop = min(len(array), frame + radius + 1)
        selected = array[start:stop]
        center = selected.mean(axis=0)
        result[frame] = float(np.sqrt(np.mean(np.sum((selected - center) ** 2, axis=-1))))
    return result


def _optional_vector(
    value: np.ndarray | None,
    *,
    name: str,
    frame_count: int,
    dtype: type[np.floating] | type[np.integer] | type[np.bool_],
    default: float | int | bool,
) -> np.ndarray:
    if value is None:
        return np.full(frame_count, default, dtype=dtype)
    array = np.asarray(value, dtype=dtype)
    if array.shape != (frame_count,):
        raise ValueError(f"HOC_PROFILE_{name}_SHAPE_INVALID")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"HOC_PROFILE_{name}_NONFINITE")
    return array


def build_human_object_interaction_profile(
    *,
    hand_pose_world_wxyz: np.ndarray,
    object_pose_world_wxyz: np.ndarray,
    timestamps_s: np.ndarray,
    minimum_surface_distance_m: np.ndarray | None = None,
    near_contact_vertex_count: np.ndarray | None = None,
    near_contact_vertex_fraction: np.ndarray | None = None,
    contact_component_count: np.ndarray | None = None,
    region_contact: np.ndarray | None = None,
    topology_normal_opposition_score: np.ndarray | None = None,
    topology_contact_spread_m: np.ndarray | None = None,
    strict_v4_reward_target: np.ndarray | None = None,
    any_hand_surface_contact: np.ndarray | None = None,
    multi_region_contact: np.ndarray | None = None,
    opposing_contact_topology: np.ndarray | None = None,
    contract: HumanObjectCouplingContactProfileContractV1 | None = None,
) -> dict[str, np.ndarray]:
    """Build a continuous contact/coupling profile in shared time semantics.

    ``region_contact`` may describe MANO regions, robot fingers, or another
    declared morphology-level region set.  The core intentionally assigns no
    semantic names and defines no coupling/grasp threshold.
    """

    frozen = contract or HumanObjectCouplingContactProfileContractV1()
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if (
        timestamps.ndim != 1
        or len(timestamps) < 2
        or not np.isfinite(timestamps).all()
        or not np.all(np.diff(timestamps) > 0.0)
    ):
        raise ValueError("HOC_PROFILE_TIMESTAMPS_INVALID")
    frame_count = len(timestamps)
    hand_pose = _validate_pose("HAND", hand_pose_world_wxyz, frame_count)
    object_pose = _validate_pose("OBJECT", object_pose_world_wxyz, frame_count)

    hand_rotation = quaternion_to_matrix_wxyz(hand_pose[:, 3:])
    object_rotation = quaternion_to_matrix_wxyz(object_pose[:, 3:])
    relative_rotation = np.swapaxes(hand_rotation, -1, -2) @ object_rotation
    relative_translation = np.einsum(
        "tij,tj->ti",
        np.swapaxes(hand_rotation, -1, -2),
        object_pose[:, :3] - hand_pose[:, :3],
    )
    relative_pose = _pose_from_translation_rotation(relative_translation, relative_rotation)

    hand_linear_velocity = _derivative(hand_pose[:, :3], timestamps)
    object_linear_velocity = _derivative(object_pose[:, :3], timestamps)
    relative_linear_velocity_hand = _derivative(relative_translation, timestamps)
    hand_angular_velocity = derive_angular_velocity_world_wxyz(hand_pose[:, 3:], timestamps)
    object_angular_velocity = derive_angular_velocity_world_wxyz(object_pose[:, 3:], timestamps)
    relative_angular_velocity_hand = derive_angular_velocity_world_wxyz(
        relative_pose[:, 3:], timestamps
    )

    hand_linear_speed = np.linalg.norm(hand_linear_velocity, axis=1)
    object_linear_speed = np.linalg.norm(object_linear_velocity, axis=1)
    relative_linear_speed = np.linalg.norm(relative_linear_velocity_hand, axis=1)
    hand_angular_speed = np.linalg.norm(hand_angular_velocity, axis=1)
    object_angular_speed = np.linalg.norm(object_angular_velocity, axis=1)
    relative_angular_speed = np.linalg.norm(relative_angular_velocity_hand, axis=1)
    epsilon = frozen.coupling_ratio_epsilon

    if region_contact is None:
        region = np.zeros((frame_count, 0), dtype=bool)
    else:
        region = np.asarray(region_contact, dtype=bool)
        if region.ndim != 2 or region.shape[0] != frame_count:
            raise ValueError("HOC_PROFILE_REGION_CONTACT_SHAPE_INVALID")

    minimum_distance = _optional_vector(
        minimum_surface_distance_m,
        name="MINIMUM_SURFACE_DISTANCE",
        frame_count=frame_count,
        dtype=np.float64,
        default=np.nan,
    )
    # NaN is the explicit unavailable marker for geometry quantities.  The
    # generic optional helper rejects it, so restore this one deliberately.
    if minimum_surface_distance_m is None:
        minimum_distance[:] = np.nan

    return {
        "timestamps_s": timestamps,
        "relative_pose_hand_object_wxyz": relative_pose,
        "relative_translation_hand_m": relative_translation,
        "relative_rotation_vector_hand_rad": so3_log(relative_rotation),
        "hand_linear_velocity_world_mps": hand_linear_velocity,
        "object_linear_velocity_world_mps": object_linear_velocity,
        "relative_linear_velocity_hand_mps": relative_linear_velocity_hand,
        "hand_angular_velocity_world_radps": hand_angular_velocity,
        "object_angular_velocity_world_radps": object_angular_velocity,
        "relative_angular_velocity_hand_radps": relative_angular_velocity_hand,
        "hand_linear_speed_mps": hand_linear_speed,
        "object_linear_speed_mps": object_linear_speed,
        "relative_linear_speed_mps": relative_linear_speed,
        "hand_angular_speed_radps": hand_angular_speed,
        "object_angular_speed_radps": object_angular_speed,
        "relative_angular_speed_radps": relative_angular_speed,
        "linear_coupling_ratio": relative_linear_speed
        / (hand_linear_speed + object_linear_speed + epsilon),
        "angular_coupling_ratio": relative_angular_speed
        / (hand_angular_speed + object_angular_speed + epsilon),
        "relative_translation_window_rms_m": _window_rms_variation(
            relative_translation, frozen.variation_window_steps
        ),
        "relative_rotation_window_rms_rad": _window_rms_variation(
            so3_log(relative_rotation), frozen.variation_window_steps
        ),
        "minimum_surface_distance_m": minimum_distance,
        "near_contact_vertex_count": _optional_vector(
            near_contact_vertex_count,
            name="NEAR_CONTACT_VERTEX_COUNT",
            frame_count=frame_count,
            dtype=np.int64,
            default=0,
        ),
        "near_contact_vertex_fraction": _optional_vector(
            near_contact_vertex_fraction,
            name="NEAR_CONTACT_VERTEX_FRACTION",
            frame_count=frame_count,
            dtype=np.float64,
            default=0.0,
        ),
        "contact_component_count": _optional_vector(
            contact_component_count,
            name="CONTACT_COMPONENT_COUNT",
            frame_count=frame_count,
            dtype=np.int64,
            default=0,
        ),
        "region_contact": region,
        "number_of_active_regions": region.sum(axis=1, dtype=np.int64),
        "normal_opposition_score": _optional_vector(
            topology_normal_opposition_score,
            name="NORMAL_OPPOSITION_SCORE",
            frame_count=frame_count,
            dtype=np.float64,
            default=0.0,
        ),
        "contact_spread_m": _optional_vector(
            topology_contact_spread_m,
            name="CONTACT_SPREAD",
            frame_count=frame_count,
            dtype=np.float64,
            default=0.0,
        ),
        "strict_v4_reward_target": _optional_vector(
            strict_v4_reward_target,
            name="STRICT_V4_REWARD_TARGET",
            frame_count=frame_count,
            dtype=np.bool_,
            default=False,
        ),
        "any_hand_surface_contact": _optional_vector(
            any_hand_surface_contact,
            name="ANY_HAND_SURFACE_CONTACT",
            frame_count=frame_count,
            dtype=np.bool_,
            default=False,
        ),
        "multi_region_contact": _optional_vector(
            multi_region_contact,
            name="MULTI_REGION_CONTACT",
            frame_count=frame_count,
            dtype=np.bool_,
            default=False,
        ),
        "opposing_contact_topology": _optional_vector(
            opposing_contact_topology,
            name="OPPOSING_CONTACT_TOPOLOGY",
            frame_count=frame_count,
            dtype=np.bool_,
            default=False,
        ),
    }


__all__ = [
    "HUMAN_OBJECT_COUPLING_CONTACT_PROFILE_V1",
    "HumanObjectCouplingContactProfileContractV1",
    "build_human_object_interaction_profile",
]
