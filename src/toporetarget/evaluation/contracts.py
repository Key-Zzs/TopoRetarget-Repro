"""Frozen contracts for the additive Evaluation Suite V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvaluationSuiteV2:
    """Common single- and future-bi-manual trajectory success contract."""

    identifier: str = "TopoRetargetEvaluationSuiteV2"
    object_rotation_threshold_deg: float = 30.0
    object_translation_threshold_cm: float = 3.0
    hand_joint_threshold_cm: float = 8.0
    fingertip_threshold_cm: float = 6.0
    coordinate_convention: str = "common_world_frame_after_env_origin_removal"
    primary_aggregation: str = "mean_over_observed_valid_trajectory_frames"
    incomplete_episode_policy: str = "diagnostic_metrics_only__kinematic_success_false"
    object_rotation_metric: str = "raw_object_orientation_SO3_geodesic_deg"
    object_translation_metric: str = "object_root_origin_euclidean_cm"
    hand_joint_metric: str = "shared_EvaluationJointSetV1_euclidean_cm"
    fingertip_metric: str = "shared_EvaluationFingertipSetV1_euclidean_cm"
    future_bimanual_policy: str = "object_and_left_hand_and_right_hand"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicsEpisodeEvidence:
    """The V2 physical requirements kept separate from kinematic tracking."""

    terminal_contact_pass: bool
    terminal_stability_pass: bool
    contact_causality_pass: bool
    inter_finger_penetration_pass: bool
    absolute_hand_object_penetration_pass: bool
    action_bounds_pass: bool
    no_hidden_force: bool
    no_object_rollout_state_write: bool
    no_wrist_root_teleport: bool

    @property
    def success(self) -> bool:
        return all(asdict(self).values())

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)
