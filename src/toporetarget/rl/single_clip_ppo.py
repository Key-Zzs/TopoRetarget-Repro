"""Frozen Stage-16B per-clip PPO qualification helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SINGLE_CLIP_SAMPLE_LADDER = (32_768, 131_072, 524_288, 2_097_152, 8_388_608)
ADAPTIVE_ORACLE_VALIDATED = "STAGE16B_ADAPTIVE_MULTI_HORIZON_ORACLE_VALIDATED"
PPO_ENTRY_AUTHORIZED = "STAGE16B_SINGLE_CLIP_PPO_ENTRY_AUTHORIZED"


@dataclass(frozen=True)
class WorldWristPrefixCurriculumV1:
    stage_fractions: tuple[tuple[str, float], ...] = (
        ("C0", 0.25),
        ("C1", 0.50),
        ("C2", 0.75),
        ("C3", 1.00),
    )

    def maximum_start_index(self, stage: str, transition_count: int) -> int:
        if transition_count < 1:
            raise ValueError("transition_count must be positive")
        fractions = dict(self.stage_fractions)
        if stage not in fractions:
            raise ValueError(f"unknown curriculum stage: {stage}")
        return min(int(transition_count * fractions[stage]) - 1, transition_count - 1)


@dataclass(frozen=True)
class AdaptiveOracleBehaviorCloningV1:
    max_epochs: int = 50
    validation_fraction: float = 0.20
    early_stopping_patience: int = 5
    actor_only: bool = True
    critic_value_targets_from_oracle: bool = False
    label: str = "ENGINEERING_ORACLE_WARMSTART"

    def validate(self) -> None:
        if not 1 <= self.max_epochs <= 50:
            raise ValueError("BC max_epochs must be in [1, 50]")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("BC validation_fraction must be in (0, 1)")
        if self.early_stopping_patience < 1:
            raise ValueError("BC early_stopping_patience must be positive")
        if not self.actor_only or self.critic_value_targets_from_oracle:
            raise ValueError("oracle BC may warm-start only the actor")


class FrameZeroSingleClipCheckpointSelector:
    """Use retained deterministic frame-0 episodes in the frozen formal order."""

    @staticmethod
    def rank(record: dict[str, Any]) -> tuple[float, ...]:
        return (
            float(record["success_rate"]),
            float(record["final_frame_reach_rate"]),
            float(record["progress_ratio"]),
            -float(record["object_position_error_m"]),
            -float(record["max_axis_point_error_m"]),
            -float(record["object_rotation_error_deg"]),
            -float(record["wrist_tracking_error"]),
            -float(record["action_saturation_fraction"]),
            -float(record["action_smoothness"]),
        )

    def select(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            raise ValueError("checkpoint selection requires at least one candidate")
        return max(records, key=self.rank)


def deterministic_rollout_seed(base_seed: int, environment_index: int) -> int:
    if environment_index < 0:
        raise ValueError("environment_index cannot be negative")
    return int(base_seed + 1_000_003 * environment_index)


def oracle_authorizes_single_clip_ppo(report: dict[str, Any]) -> bool:
    return (
        report.get("status") == ADAPTIVE_ORACLE_VALIDATED
        and report.get("ppo_entry") == PPO_ENTRY_AUTHORIZED
        and len(report.get("clips", ())) == 2
        and all(bool(clip.get("passes_gate")) for clip in report["clips"])
    )


def frozen_single_clip_ppo_contract() -> dict[str, Any]:
    curriculum = WorldWristPrefixCurriculumV1()
    cloning = AdaptiveOracleBehaviorCloningV1()
    cloning.validate()
    return {
        "id": "world_wrist_single_clip_ppo_v1",
        "physics": "world_wrist_freebody_nominal_v1",
        "action_dim": 26,
        "sample_ladder": list(SINGLE_CLIP_SAMPLE_LADDER),
        "checkpoint_selector": "FrameZeroSingleClipCheckpointSelector",
        "prefix_curriculum": asdict(curriculum),
        "oracle_behavior_cloning": asdict(cloning),
        "clip_identity_in_observation": False,
        "domain_randomization": False,
        "observation_noise": False,
        "formal_evaluation": "20_episode_frame0_deterministic",
    }


__all__ = [
    "SINGLE_CLIP_SAMPLE_LADDER",
    "AdaptiveOracleBehaviorCloningV1",
    "FrameZeroSingleClipCheckpointSelector",
    "WorldWristPrefixCurriculumV1",
    "deterministic_rollout_seed",
    "frozen_single_clip_ppo_contract",
    "oracle_authorizes_single_clip_ppo",
]
