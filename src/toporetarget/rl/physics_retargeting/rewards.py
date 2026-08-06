"""Frozen Stage 16-D physics-consistent reward profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PhysicsConsistentRewardProfileV1:
    identifier: str = "semantic_balanced_v1"
    wrist_fidelity: float = 1.5
    finger_fidelity: float = 1.0
    link_fidelity: float = 1.0
    contact_coverage: float = 3.0
    contact_persistence: float = 2.0
    contact_onset: float = 0.5
    final_topology: float = 2.0
    forbidden_contact: float = -2.0
    penetration: float = -4.0
    impulse_outlier: float = -1.0
    object_stability: float = -0.25
    action_effort: float = -0.01
    action_first_difference: float = -0.02
    action_second_difference: float = -0.01
    semantic_progress: float = 4.0
    relative_pose_progress: float = 1.0
    source_object_soft_prior: float = 0.10
    terminal_success: float = 10.0
    catastrophic_failure: float = -10.0

    def __post_init__(self) -> None:
        if self.identifier not in {
            "semantic_balanced_v1",
            "contact_priority_v1",
            "source_fidelity_priority_v1",
        }:
            raise ValueError("unknown Stage16D reward profile")

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": "physics_consistent_retargeting_reward_v1", **asdict(self)}


def physics_consistent_reward_terms(
    metrics: dict[str, torch.Tensor],
    profile: PhysicsConsistentRewardProfileV1,
) -> dict[str, torch.Tensor]:
    """Score semantic and contact metrics without strict object-pose termination."""

    required = {
        "wrist_fidelity",
        "finger_fidelity",
        "link_fidelity",
        "contact_coverage",
        "contact_persistence",
        "contact_onset_alignment",
        "final_topology",
        "forbidden_contact",
        "penetration_m",
        "impulse_outlier",
        "object_instability",
        "action_effort",
        "action_first_difference",
        "action_second_difference",
        "semantic_progress",
        "relative_pose_progress",
        "source_object_soft_prior",
        "terminal_success",
        "catastrophic_failure",
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"Stage16D reward misses metrics: {sorted(missing)}")
    if any(not bool(torch.isfinite(metrics[name]).all()) for name in required):
        raise ValueError("Stage16D reward metrics must be finite")
    weighted = {
        "wrist_fidelity": profile.wrist_fidelity * metrics["wrist_fidelity"],
        "finger_fidelity": profile.finger_fidelity * metrics["finger_fidelity"],
        "link_fidelity": profile.link_fidelity * metrics["link_fidelity"],
        "contact_coverage": profile.contact_coverage * metrics["contact_coverage"],
        "contact_persistence": profile.contact_persistence * metrics["contact_persistence"],
        "contact_onset": profile.contact_onset * metrics["contact_onset_alignment"],
        "final_topology": profile.final_topology * metrics["final_topology"],
        "forbidden_contact": profile.forbidden_contact * metrics["forbidden_contact"],
        "penetration": profile.penetration * (metrics["penetration_m"] / 0.003).square(),
        "impulse_outlier": profile.impulse_outlier * metrics["impulse_outlier"],
        "object_stability": profile.object_stability * metrics["object_instability"],
        "action_effort": profile.action_effort * metrics["action_effort"],
        "action_first_difference": (
            profile.action_first_difference * metrics["action_first_difference"]
        ),
        "action_second_difference": (
            profile.action_second_difference * metrics["action_second_difference"]
        ),
        "semantic_progress": profile.semantic_progress * metrics["semantic_progress"],
        "relative_pose_progress": (
            profile.relative_pose_progress * metrics["relative_pose_progress"]
        ),
        "source_object_soft_prior": (
            profile.source_object_soft_prior * metrics["source_object_soft_prior"]
        ),
        "terminal_success": profile.terminal_success * metrics["terminal_success"],
        "catastrophic_failure": (profile.catastrophic_failure * metrics["catastrophic_failure"]),
    }
    weighted["total"] = torch.stack(tuple(weighted.values()), dim=0).sum(dim=0)
    return weighted


__all__ = ["PhysicsConsistentRewardProfileV1", "physics_consistent_reward_terms"]
