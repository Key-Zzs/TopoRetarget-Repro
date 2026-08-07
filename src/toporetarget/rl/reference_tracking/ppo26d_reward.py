"""Paper tracking reward plus the required controllable-wrist adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from toporetarget.rl.environments.isaaclab_backend.reward_terms import (
    Stage16WorldWristRewardProfileV1,
    world_wrist_reward_terms,
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


def ppo26d_reward_terms(**kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
    terms = world_wrist_reward_terms(
        profile=TopoRetargetReferenceTrackingReward26DV1().profile(), **kwargs
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


__all__ = ["TopoRetargetReferenceTrackingReward26DV1", "ppo26d_reward_terms"]
