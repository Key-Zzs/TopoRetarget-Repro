"""Frozen public contracts for the Stage 16-D.5 PPO-26D lane."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

ACTION_DIMENSION = 26
WRIST_ACTION_DIMENSION = 6
FINGER_ACTION_DIMENSION = 20
LOOKAHEAD_OFFSETS = (0, 1, 3, 5)


@dataclass(frozen=True)
class Stage16DReferenceResidualAction26DV1:
    """Policy-space residual action; it is deliberately not a joint action."""

    identifier: str = "Stage16DReferenceResidualAction26DV1"
    action_dimension: int = ACTION_DIMENSION
    wrist_slice: tuple[int, int] = (0, 6)
    wrist_translation_slice: tuple[int, int] = (0, 3)
    wrist_rotation_slice: tuple[int, int] = (3, 6)
    finger_slice: tuple[int, int] = (6, 26)
    wrist_translation_scale_m: float = 0.01
    wrist_rotation_scale_rad: float = 0.08726646259971647
    finger_joint_range_fraction: float = 0.10
    wrist_residual_frame: str = "reference_wrist_local"
    wrist_composition: str = "T_reference_compose_deltaT"
    wrist_target_adapter: str = "existing_se3_to_explicit_serial_3p3r"
    direct_articulation_action: bool = False
    direct_object_action: bool = False
    direct_object_state_write: bool = False
    hidden_force_or_attachment: bool = False

    def __post_init__(self) -> None:
        if self.action_dimension != ACTION_DIMENSION:
            raise ValueError("PPO26D action dimension must be 26")
        if self.wrist_slice != (0, WRIST_ACTION_DIMENSION):
            raise ValueError("PPO26D wrist semantic must occupy action[0:6]")
        if self.finger_slice != (WRIST_ACTION_DIMENSION, ACTION_DIMENSION):
            raise ValueError("PPO26D fingers must occupy action[6:26]")
        if self.direct_articulation_action or self.direct_object_action:
            raise ValueError("policy action may not bypass the residual/SE3 contract")

    def validate_shape(self, shape: tuple[int, ...]) -> None:
        if len(shape) != 2 or shape[-1] != self.action_dimension:
            raise ValueError("PPO26D actions must have shape [num_envs, 26]")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage16DPPO26DObservationV2:
    """One frozen, non-privileged 764-D actor-observation semantic map."""

    identifier: str = "Stage16DPPO26DObservationV2"
    dimension: int = 764
    lookahead_offsets: tuple[int, ...] = LOOKAHEAD_OFFSETS
    representation_frame: str = "robot_base_or_current_wrist_relative"
    contains_future_actual_state: bool = False
    contains_future_object_outcome: bool = False
    contains_cem_candidate: bool = False
    contains_penetration_future: bool = False
    contains_success_label: bool = False

    def field_dimensions(self) -> dict[str, int]:
        """Return the serialized order used by the IsaacLab PPO environment."""

        return {
            "current_wrist_pose": 6,
            "current_wrist_pose_error": 6,
            "current_wrist_twist": 6,
            "current_finger_q": 20,
            "current_finger_qdot": 20,
            "previous_action_26d": 26,
            "current_object_axis_points": 18,
            "current_object_wrist_relative_pose": 6,
            "current_object_twist": 6,
            "current_object_base_pose": 6,
            "current_object_reference_pose": 6,
            "reference_k_wrist_pose": 6,
            "reference_k_wrist_twist": 6,
            "reference_k_finger_q": 20,
            "reference_k_object_axis_points": 18,
            "reference_k_tracked_links_base": 48,
            "reference_k_tracked_links_wrist": 48,
            "reference_k1_k3_k5": 492,
        }

    def __post_init__(self) -> None:
        if self.lookahead_offsets != LOOKAHEAD_OFFSETS:
            raise ValueError("PPO26D lookahead must be current/+1/+3/+5 exactly once")
        if any(
            (
                self.contains_future_actual_state,
                self.contains_future_object_outcome,
                self.contains_cem_candidate,
                self.contains_penetration_future,
                self.contains_success_label,
            )
        ):
            raise ValueError("PPO26D actor observation contains privileged information")
        if sum(self.field_dimensions().values()) != self.dimension:
            raise AssertionError("PPO26D observation semantic map no longer totals 764")

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "fields": self.field_dimensions()}


@dataclass(frozen=True)
class Stage16DPPO26DTrainingConfigV1:
    """Paper PPO settings with the declared Stage 16-D timing adaptation."""

    identifier: str = "Stage16DPPO26DTrainingConfigV1"
    rollout_length: int = 40
    learning_rate: float = 1.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    epochs: int = 4
    minibatches: int = 32
    entropy_coefficient: float = 0.001
    max_grad_norm: float = 1.0
    target_kl: float = 0.03
    normalized_observation_abs_limit: float = 100.0
    action_saturation_absolute_threshold: float = 0.98
    action_saturation_fraction_limit: float = 0.25
    observation_normalization: bool = True
    advantage_normalization: bool = True
    actor_hidden: tuple[int, ...] = (512, 256, 128)
    critic_hidden: tuple[int, ...] = (512, 512, 256, 128)
    activation: str = "ELU"
    control_hz: float = 20.0
    physics_hz: float = 120.0
    decimation: int = 6
    runtime_reference_samples: int = 321

    def __post_init__(self) -> None:
        if self.rollout_length != 40:
            raise ValueError("PPO26D rollout length is frozen at 40")
        if (self.epochs, self.minibatches) != (4, 32):
            raise ValueError("PPO26D uses the paper 4-epoch/32-minibatch update")
        if self.runtime_reference_samples != 321 or self.decimation != 6:
            raise ValueError("PPO26D must preserve the Stage16-D factor-8 timing contract")
        if not 0.0 < self.target_kl < 1.0:
            raise ValueError("PPO26D target KL must be in (0, 1)")
        if not 0.0 < self.action_saturation_fraction_limit < 1.0:
            raise ValueError("PPO26D action saturation fraction limit must be in (0, 1)")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "ACTION_DIMENSION",
    "FINGER_ACTION_DIMENSION",
    "LOOKAHEAD_OFFSETS",
    "Stage16DPPO26DObservationV2",
    "Stage16DPPO26DTrainingConfigV1",
    "Stage16DReferenceResidualAction26DV1",
    "WRIST_ACTION_DIMENSION",
]
