"""Export Stage16ReferenceClip v1 from the repository RobotReference v2 contract."""

from __future__ import annotations

from typing import Any

import numpy as np

from toporetarget.contracts.reference import RobotReferenceV2

from .axis_points import OBJECT_AXIS_PROFILE_ID, object_axis_points_from_poses
from .contracts import Stage16ReferenceClip
from .resampling import REFERENCE_RESAMPLER_ID, resample_reference_20hz
from .tracked_links import TRACKED_LINK_PROFILE_ID, TRACKED_LINKS_WUJI_RH, select_tracked_links


def central_difference(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Central differences with one-sided endpoints as specified for Stage 16."""

    value = np.asarray(values, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if value.shape[0] < 2 or times.shape != (value.shape[0],):
        raise ValueError("central difference requires at least two aligned frames")
    result = np.empty_like(value)
    result[0] = (value[1] - value[0]) / (times[1] - times[0])
    result[-1] = (value[-1] - value[-2]) / (times[-1] - times[-2])
    if value.shape[0] > 2:
        denominator = (times[2:] - times[:-2]).reshape((-1,) + (1,) * (value.ndim - 1))
        result[1:-1] = (value[2:] - value[:-2]) / denominator
    return result


def export_stage16_reference(
    reference: RobotReferenceV2,
    *,
    tracked_link_profile: tuple[str, ...] = TRACKED_LINKS_WUJI_RH,
    axis_length_m: float = 0.05,
    resample_to_hz: float = 20.0,
    extra_provenance: dict[str, Any] | None = None,
) -> Stage16ReferenceClip:
    """Convert a valid robot reference without using its floating base as an action."""

    reference.validate()
    links = select_tracked_links(
        reference.tracked_link_positions, reference.tracked_link_names, profile=tracked_link_profile
    )
    source = Stage16ReferenceClip(
        timestamps=reference.timestamps,
        q_finger_ref=reference.qpos_reference,
        object_pose_base_ref=reference.object_pose_base,
        object_axis_points_base_ref=object_axis_points_from_poses(
            reference.object_pose_base, axis_length_m=axis_length_m
        ),
        tracked_link_positions_base_ref=links,
        joint_order=reference.joint_order,
        tracked_link_names=tracked_link_profile,
        provenance={
            "source_schema": reference.schema_version,
            "source_robot_hash": reference.robot_hash,
            "dataset_provenance": reference.dataset_provenance,
            "coordinates": "robot_base_metres_radians",
            "axis_profile": OBJECT_AXIS_PROFILE_ID,
            "tracked_link_profile": TRACKED_LINK_PROFILE_ID,
            "floating_base_policy": "reference_generation_only_not_policy_action",
            **(extra_provenance or {}),
        },
        qdot_ref=central_difference(reference.qpos_reference, reference.timestamps),
        object_velocity_ref=np.concatenate(
            [
                central_difference(reference.object_pose_base[:, :3, 3], reference.timestamps),
                np.zeros((reference.num_frames, 3)),
            ],
            axis=1,
        ),
        reference_indices=np.arange(reference.num_frames, dtype=np.int64),
        metadata={"source_fps": reference.fps, "resampler": REFERENCE_RESAMPLER_ID},
    )
    source.validate()
    return resample_reference_20hz(source, target_hz=resample_to_hz)


__all__ = ["central_difference", "export_stage16_reference"]
