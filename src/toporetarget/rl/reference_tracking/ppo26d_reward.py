"""Paper tracking reward plus the required controllable-wrist adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from toporetarget.rl.environments.isaaclab_backend.reward_terms import (
    Stage16WorldWristRewardProfileV1,
    world_wrist_reward_terms,
)
from toporetarget.rl.reference_tracking.reference_gated_contact import (
    ReferenceGatedContactRewardContractV1,
    reference_gated_contact_reward,
)
from toporetarget.rl.reference_tracking.strict_per_finger_contact import (
    StrictPerFingerContactRewardV4,
    strict_per_finger_contact_reward,
)


@dataclass(frozen=True)
class TopoRetargetReferenceTrackingReward26DV1:
    identifier: str = "TopoRetargetReferenceTrackingReward26DV1"
    object_weight: float = 8.0
    object_sigma_m: float = 0.04
    link_weight: float = 1.0
    link_sigma_m: float = 0.025
    finger_weight: float = 1.0
    finger_sigma_normalized: float = 0.10
    smoothness_weight: float = -0.01
    wrist_position_weight: float = 2.0
    wrist_position_sigma_m: float = 0.02
    wrist_rotation_weight: float = 1.0
    wrist_rotation_sigma_rad: float = 0.17453292519943295
    terminal_contact_bonus: float = 0.0
    penetration_reward: float = 0.0
    inter_finger_penalty: float = 0.0
    clip_specific_bonus: float = 0.0
    semantic_task_bonus: float = 0.0

    def __post_init__(self) -> None:
        if any(
            value != 0.0
            for value in (
                self.terminal_contact_bonus,
                self.penetration_reward,
                self.inter_finger_penalty,
                self.clip_specific_bonus,
                self.semantic_task_bonus,
            )
        ):
            raise ValueError("PPO26D reward must not receive post-PPO qualification terms")

    def profile(self) -> Stage16WorldWristRewardProfileV1:
        return Stage16WorldWristRewardProfileV1(
            object_weight=self.object_weight,
            object_sigma_m=self.object_sigma_m,
            link_weight=self.link_weight,
            link_sigma_m=self.link_sigma_m,
            finger_weight=self.finger_weight,
            finger_sigma_normalized=self.finger_sigma_normalized,
            wrist_position_weight=self.wrist_position_weight,
            wrist_position_sigma_m=self.wrist_position_sigma_m,
            wrist_rotation_weight=self.wrist_rotation_weight,
            wrist_rotation_sigma_rad=self.wrist_rotation_sigma_rad,
            smoothness_weight=self.smoothness_weight,
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def ppo26d_reward_terms(
    *, profile: Stage16WorldWristRewardProfileV1 | None = None, **kwargs: torch.Tensor
) -> dict[str, torch.Tensor]:
    terms = world_wrist_reward_terms(
        profile=profile or TopoRetargetReferenceTrackingReward26DV1().profile(), **kwargs
    )
    return {
        **terms,
        "r_object": terms["object"],
        "r_link": terms["tracked_links"],
        "r_finger": terms["finger_joints"],
        "r_wrist_translation": terms["wrist_position"],
        "r_wrist_rotation": terms["wrist_rotation"],
        "r_wrist": terms["wrist_position"] + terms["wrist_rotation"],
    }


@dataclass(frozen=True)
class TopoRetargetReferenceTrackingReward26DV2:
    """V1 tracking reward plus V2-reference object-twist tracking only.

    The V1 pose, link, finger, wrist, and smoothness terms are deliberately
    retained verbatim.  The two added terms compare *world-frame signed
    twists* to the qualified V2 reference; they never penalize speed alone.
    """

    identifier: str = "TopoRetargetReferenceTrackingReward26DV2"
    reference_kinematics_version: int = 2
    object_weight: float = 8.0
    object_sigma_m: float = 0.04
    link_weight: float = 1.0
    link_sigma_m: float = 0.025
    finger_weight: float = 1.0
    finger_sigma_normalized: float = 0.10
    smoothness_weight: float = -0.01
    wrist_position_weight: float = 2.0
    wrist_position_sigma_m: float = 0.02
    wrist_rotation_weight: float = 1.0
    wrist_rotation_sigma_rad: float = 0.17453292519943295
    object_velocity_weight: float = 0.5
    object_velocity_sigma_mps: float = 0.075
    object_angular_velocity_weight: float = 0.5
    object_angular_velocity_sigma_radps: float = 0.125
    terminal_contact_bonus: float = 0.0
    penetration_reward: float = 0.0
    inter_finger_penalty: float = 0.0
    clip_specific_bonus: float = 0.0
    semantic_task_bonus: float = 0.0

    def __post_init__(self) -> None:
        if self.reference_kinematics_version != 2:
            raise ValueError("Reward V2 requires reference_kinematics_version == 2")
        if self.object_velocity_sigma_mps <= 0.0 or self.object_angular_velocity_sigma_radps <= 0.0:
            raise ValueError("Reward V2 twist sigmas must be positive")
        if self.object_velocity_weight < 0.0 or self.object_angular_velocity_weight < 0.0:
            raise ValueError("Reward V2 twist weights must be non-negative")
        if self.object_velocity_weight + self.object_angular_velocity_weight > 2.0:
            raise ValueError("Reward V2 combined twist contribution exceeds the frozen 2.0 bound")
        if any(
            value != 0.0
            for value in (
                self.terminal_contact_bonus,
                self.penetration_reward,
                self.inter_finger_penalty,
                self.clip_specific_bonus,
                self.semantic_task_bonus,
            )
        ):
            raise ValueError("PPO26D reward must not receive post-PPO qualification terms")

    def profile(self) -> Stage16WorldWristRewardProfileV1:
        return Stage16WorldWristRewardProfileV1(
            object_weight=self.object_weight,
            object_sigma_m=self.object_sigma_m,
            link_weight=self.link_weight,
            link_sigma_m=self.link_sigma_m,
            finger_weight=self.finger_weight,
            finger_sigma_normalized=self.finger_sigma_normalized,
            wrist_position_weight=self.wrist_position_weight,
            wrist_position_sigma_m=self.wrist_position_sigma_m,
            wrist_rotation_weight=self.wrist_rotation_weight,
            wrist_rotation_sigma_rad=self.wrist_rotation_sigma_rad,
            smoothness_weight=self.smoothness_weight,
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def ppo26d_reward_v2_object_twist_terms(
    *,
    object_twist_world: torch.Tensor,
    object_twist_world_ref: torch.Tensor,
    profile: TopoRetargetReferenceTrackingReward26DV2,
    **kwargs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the frozen V1 terms plus the two approved signed-twist terms."""

    if (
        object_twist_world.shape != object_twist_world_ref.shape
        or object_twist_world.shape[-1] != 6
    ):
        raise ValueError("Reward V2 requires matching [N, 6] world object twists")
    base = ppo26d_reward_terms(profile=profile.profile(), **kwargs)
    linear_error = torch.linalg.vector_norm(
        object_twist_world[:, :3] - object_twist_world_ref[:, :3], dim=-1
    )
    angular_error = torch.linalg.vector_norm(
        object_twist_world[:, 3:] - object_twist_world_ref[:, 3:], dim=-1
    )
    linear_term = torch.exp(-(linear_error / profile.object_velocity_sigma_mps).square())
    angular_term = torch.exp(
        -(angular_error / profile.object_angular_velocity_sigma_radps).square()
    )
    total = (
        base["total"]
        + profile.object_velocity_weight * linear_term
        + profile.object_angular_velocity_weight * angular_term
    )
    terms = {
        **base,
        "e_obj_vel": linear_error,
        "e_obj_ang_vel": angular_error,
        "r_obj_vel": linear_term,
        "r_obj_ang_vel": angular_term,
        "r_obj_vel_weighted": profile.object_velocity_weight * linear_term,
        "r_obj_ang_vel_weighted": profile.object_angular_velocity_weight * angular_term,
        "total": total,
    }
    if not all(bool(torch.isfinite(value).all()) for value in terms.values()):
        raise FloatingPointError("Stage 16-D Reward V2 became non-finite")
    return terms


@dataclass(frozen=True)
class TopoRetargetReferenceTrackingReward26DV3(TopoRetargetReferenceTrackingReward26DV2):
    """Reward V2 plus exactly one reference-gated fingertip-contact term."""

    identifier: str = "TopoRetargetReferenceTrackingReward26DV3"
    contact_weight: float = 1.0
    contact_distance_threshold_m: float = 0.03
    contact_epsilon_n: float = 1.0e-5
    contact_force_scale_lambda_n: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.contact_weight != 1.0:
            raise ValueError("Reward V3 contact weight is frozen at 1.0")
        if self.contact_distance_threshold_m != 0.03:
            raise ValueError("Reward V3 contact threshold is frozen at 0.03 m")
        if self.contact_epsilon_n != 1.0e-5:
            raise ValueError("Reward V3 contact epsilon is frozen at 1e-5")
        if self.contact_force_scale_lambda_n <= self.contact_epsilon_n:
            raise ValueError("Reward V3 requires a frozen positive contact-force scale")

    def contact_contract(self) -> ReferenceGatedContactRewardContractV1:
        return ReferenceGatedContactRewardContractV1(
            xi_c_m=self.contact_distance_threshold_m,
            contact_weight=self.contact_weight,
            epsilon_n=self.contact_epsilon_n,
        )


def ppo26d_reward_v3_reference_gated_contact_terms(
    *,
    object_twist_world: torch.Tensor,
    object_twist_world_ref: torch.Tensor,
    reference_expected_contact_mask: torch.Tensor,
    fingertip_object_pair_force_world: torch.Tensor,
    profile: TopoRetargetReferenceTrackingReward26DV3,
    **kwargs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the untouched V2 terms and the sole V3 additive term."""

    v2 = ppo26d_reward_v2_object_twist_terms(
        object_twist_world=object_twist_world,
        object_twist_world_ref=object_twist_world_ref,
        profile=profile,
        **kwargs,
    )
    contact = reference_gated_contact_reward(
        reference_expected_mask=reference_expected_contact_mask,
        fingertip_object_pair_force_world=fingertip_object_pair_force_world,
        lambda_c_n=profile.contact_force_scale_lambda_n,
        contract=profile.contact_contract(),
    )
    total = v2["total"] + contact["r_contact"]
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("Stage 16-D Reward V3 became non-finite")
    return {**v2, **contact, "total": total}


@dataclass(frozen=True)
class TopoRetargetReferenceTrackingReward26DV4(TopoRetargetReferenceTrackingReward26DV2):
    """Reward V2 plus one source-confirmed strict per-finger contact term.

    V4 deliberately inherits V2, rather than V3: the aggregate proximity-mask
    term is absent by construction and cannot be accidentally double-counted.
    """

    identifier: str = "TopoRetargetReferenceTrackingReward26DV4"
    contact_weight: float = 1.0
    contact_epsilon_n: float = 1.0e-5
    contact_numerical_floor_n: float = 1.0e-4
    contact_force_scale_lambda_tip_n: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.contact_weight != 1.0:
            raise ValueError("Reward V4 contact weight is frozen at 1.0")
        if self.contact_epsilon_n != 1.0e-5:
            raise ValueError("Reward V4 contact epsilon is frozen at 1e-5")
        if self.contact_numerical_floor_n <= 0.0:
            raise ValueError("Reward V4 contact numerical floor must be positive")
        if self.contact_force_scale_lambda_tip_n <= self.contact_epsilon_n:
            raise ValueError("Reward V4 requires a frozen positive single-tip force scale")

    def contact_contract(self) -> StrictPerFingerContactRewardV4:
        return StrictPerFingerContactRewardV4(
            contact_weight=self.contact_weight,
            epsilon_n=self.contact_epsilon_n,
            numerical_floor_n=self.contact_numerical_floor_n,
        )


def ppo26d_reward_v4_strict_per_finger_contact_terms(
    *,
    object_twist_world: torch.Tensor,
    object_twist_world_ref: torch.Tensor,
    source_contact_mask: torch.Tensor,
    fingertip_object_pair_force_world: torch.Tensor,
    fingertip_object_pair_presence: torch.Tensor,
    profile: TopoRetargetReferenceTrackingReward26DV4,
    **kwargs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return frozen V2 terms plus exactly V4's independent-finger term."""

    v2 = ppo26d_reward_v2_object_twist_terms(
        object_twist_world=object_twist_world,
        object_twist_world_ref=object_twist_world_ref,
        profile=profile,
        **kwargs,
    )
    contact = strict_per_finger_contact_reward(
        source_contact_mask=source_contact_mask,
        fingertip_object_pair_force_world=fingertip_object_pair_force_world,
        pair_presence=fingertip_object_pair_presence,
        lambda_tip_n=profile.contact_force_scale_lambda_tip_n,
        contract=profile.contact_contract(),
    )
    total = v2["total"] + contact["r_contact_v4"]
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("Stage 16-D Reward V4 became non-finite")
    return {**v2, **contact, "total": total}


__all__ = [
    "TopoRetargetReferenceTrackingReward26DV1",
    "TopoRetargetReferenceTrackingReward26DV2",
    "TopoRetargetReferenceTrackingReward26DV3",
    "TopoRetargetReferenceTrackingReward26DV4",
    "ppo26d_reward_terms",
    "ppo26d_reward_v2_object_twist_terms",
    "ppo26d_reward_v3_reference_gated_contact_terms",
    "ppo26d_reward_v4_strict_per_finger_contact_terms",
]
