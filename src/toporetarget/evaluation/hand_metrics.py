"""Shared hand-keypoint and fingertip sets for Evaluation Suite V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

_JOINT_NAMES = (
    "r_wrist",
    "r_thumb_proximal",
    "r_thumb_middle",
    "r_thumb_distal",
    "r_index_finger_proximal",
    "r_index_finger_middle",
    "r_index_finger_distal",
    "r_middle_finger_proximal",
    "r_middle_finger_middle",
    "r_middle_finger_distal",
    "r_ring_finger_proximal",
    "r_ring_finger_middle",
    "r_ring_finger_distal",
    "r_pinky_proximal",
    "r_pinky_middle",
    "r_pinky_distal",
)

_FINGERTIP_NAMES = (
    "r_thumb_distal",
    "r_index_finger_distal",
    "r_middle_finger_distal",
    "r_ring_finger_distal",
    "r_pinky_distal",
)


@dataclass(frozen=True)
class EvaluationJointSetV1:
    identifier: str = "EvaluationJointSetV1"
    joint_names: tuple[str, ...] = _JOINT_NAMES
    actual_source: str = "offline_fk_collision_body_root_positions"
    reference_source: str = "tracked_link_positions_world_ref"
    coordinate_convention: str = "common_world_frame_after_env_origin_removal"
    mapping_rule: str = "name_equal_collision_body_to_tracked_link"
    engineering_approximation: bool = True

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationFingertipSetV1:
    identifier: str = "EvaluationFingertipSetV1"
    fingertip_names: tuple[str, ...] = _FINGERTIP_NAMES
    actual_source: str = "distal_collision_body_root"
    reference_source: str = "tracked_link_positions_world_ref"
    coordinate_convention: str = "common_world_frame_after_env_origin_removal"
    mapping_rule: str = "distal_link_root_as_tip_engineering_landmark"
    engineering_approximation: bool = True

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _indices(names: tuple[str, ...], source_names: list[str] | tuple[str, ...]) -> list[int]:
    available = {str(name): index for index, name in enumerate(source_names)}
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"evaluation mapping missing names: {missing}")
    return [available[name] for name in names]


def hand_metric_series(
    actual_collision_pose_world: np.ndarray,
    actual_collision_names: list[str] | tuple[str, ...],
    reference_link_positions_world: np.ndarray,
    reference_link_names: list[str] | tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Return mean-per-frame joint/fingertip errors in cm using shared mapping."""

    actual = np.asarray(actual_collision_pose_world, dtype=np.float64)
    reference = np.asarray(reference_link_positions_world, dtype=np.float64)
    if actual.ndim != 3 or actual.shape[2] != 7:
        raise ValueError("actual collision poses must have shape [T, B, 7]")
    if reference.ndim != 3 or reference.shape[2] != 3 or actual.shape[0] != reference.shape[0]:
        raise ValueError("reference link positions must have shape [T, L, 3] aligned with actual")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("hand metric inputs must be finite")
    joint = EvaluationJointSetV1()
    tips = EvaluationFingertipSetV1()
    joint_actual = _indices(joint.joint_names, actual_collision_names)
    joint_reference = _indices(joint.joint_names, reference_link_names)
    tip_actual = _indices(tips.fingertip_names, actual_collision_names)
    tip_reference = _indices(tips.fingertip_names, reference_link_names)
    return {
        "e_j_cm": np.linalg.norm(
            actual[:, joint_actual, :3] - reference[:, joint_reference], axis=-1
        ).mean(axis=-1)
        * 100.0,
        "e_ft_cm": np.linalg.norm(
            actual[:, tip_actual, :3] - reference[:, tip_reference], axis=-1
        ).mean(axis=-1)
        * 100.0,
    }
