"""All-rollout nominal and robust evaluation accounting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    termination: str
    success: bool
    final_frame_reached: bool
    object_position_error_m: float
    object_rotation_error_deg: float
    max_axis_point_error_m: float
    link_rmse_m: float
    normalized_joint_error: float
    progress_ratio: float
    return_value: float
    action_magnitude: float
    action_first_difference: float
    action_second_difference: float


def summarize_episodes(episodes: Iterable[EpisodeMetrics]) -> dict[str, object]:
    rows = list(episodes)
    if not rows:
        raise ValueError(
            "evaluation requires at least one episode; success-only filtering is forbidden"
        )
    counter = Counter(item.termination for item in rows)

    def mean(field: str) -> float:
        return float(np.mean([getattr(item, field) for item in rows]))

    successes = [item for item in rows if item.success]
    return {
        "episode_count": len(rows),
        "success_rate": float(np.mean([item.success for item in rows])),
        "final_frame_reach_rate": float(np.mean([item.final_frame_reached for item in rows])),
        "object_position_error_cm_all": mean("object_position_error_m") * 100.0,
        "object_rotation_error_deg_all": mean("object_rotation_error_deg"),
        "max_axis_point_error_m_all": mean("max_axis_point_error_m"),
        "link_rmse_m_all": mean("link_rmse_m"),
        "normalized_joint_error_all": mean("normalized_joint_error"),
        "progress_ratio_all": mean("progress_ratio"),
        "return_all": mean("return_value"),
        "action_magnitude_all": mean("action_magnitude"),
        "action_first_difference_all": mean("action_first_difference"),
        "action_second_difference_all": mean("action_second_difference"),
        "termination_distribution": dict(sorted(counter.items())),
        "successful_rollouts_only_count": len(successes),
        "successful_rollouts_only_position_cm": (
            float(np.mean([item.object_position_error_m for item in successes]) * 100.0)
            if successes
            else None
        ),
    }


__all__ = ["EpisodeMetrics", "summarize_episodes"]
