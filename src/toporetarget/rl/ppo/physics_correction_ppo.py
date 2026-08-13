"""Frozen Stage 16-D PPO and curriculum contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .trainer import PPOConfig

SAMPLE_LADDER = (1_048_576, 4_194_304, 16_777_216, 67_108_864)
CURRICULUM = ("P0", "P1", "P2", "P3")


@dataclass(frozen=True)
class PhysicsCorrectionPPOV1:
    observation_dimension: int = 764
    action_dimension: int = 26
    rollout_length: int = 16
    maximum_seeds_per_clip: int = 2
    checkpoint_evaluation_episodes: int = 20
    frame_zero_final_qualification: bool = True
    clip_identity_in_observation: bool = False
    observation_normalization: bool = True
    advantage_normalization: bool = True

    def __post_init__(self) -> None:
        if self.observation_dimension != 764 or self.action_dimension != 26:
            raise ValueError("Stage16D PPO dimensions are frozen at 764/26")
        if self.rollout_length != 16 or self.maximum_seeds_per_clip != 2:
            raise ValueError("Stage16D PPO rollout and seed budgets are frozen")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "PhysicsCorrectionPPOV1",
            **asdict(self),
            "sample_ladder": list(SAMPLE_LADDER),
            "curriculum": list(CURRICULUM),
            "ppo": PPOConfig().as_dict(),
            "actor_hidden": [512, 256, 128],
            "critic_hidden": [512, 512, 256, 128],
            "activation": "ELU",
        }


def next_sample_target(cumulative_samples: int) -> int | None:
    if cumulative_samples < 0:
        raise ValueError("sample count cannot be negative")
    return next((value for value in SAMPLE_LADDER if value > cumulative_samples), None)


def qualification_status(
    *,
    success_rate: float,
    semantic_reach_rate: float,
    contact_pass_rate: float,
    penetration_pass_rate: float,
    hard_contract_pass: bool,
) -> str:
    values = (success_rate, semantic_reach_rate, contact_pass_rate, penetration_pass_rate)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("PPO qualification rates must be in [0,1]")
    passed = (
        success_rate >= 0.90
        and semantic_reach_rate >= 0.90
        and contact_pass_rate >= 0.90
        and penetration_pass_rate >= 0.90
        and hard_contract_pass
    )
    return "STAGE16D_SINGLE_CLIP_PPO_VALIDATED" if passed else "STAGE16D_SINGLE_CLIP_PPO_PARTIAL"


__all__ = [
    "CURRICULUM",
    "SAMPLE_LADDER",
    "PhysicsCorrectionPPOV1",
    "next_sample_target",
    "qualification_status",
]
