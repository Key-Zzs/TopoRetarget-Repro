"""All-rollout nominal and robust evaluation accounting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class ClipEvaluation:
    """All-episode metrics for one fixed reference clip."""

    clip_id: str
    episodes: tuple[EpisodeMetrics, ...]
    summary: dict[str, object]


class FrameZeroEvaluator:
    """Deterministic frame-0 evaluator that retains failed episodes."""

    def __init__(self, *, episodes_per_clip: int = 20, seed: int = 0) -> None:
        if episodes_per_clip < 1:
            raise ValueError("episodes_per_clip must be positive")
        self.episodes_per_clip = episodes_per_clip
        self.seed = seed

    def evaluate(
        self,
        clips: Iterable[tuple[str, Any]],
        run_episode: Callable[[Any, int, int], EpisodeMetrics],
    ) -> tuple[ClipEvaluation, ...]:
        """Run every clip from reference index zero with no success filtering.

        ``run_episode`` owns the simulator and must call its deterministic actor
        with ``reset(reference_index=0)``.  The callback receives clip, episode
        index, and a deterministic per-episode seed.
        """

        results: list[ClipEvaluation] = []
        for clip_id, clip in clips:
            episodes = tuple(
                run_episode(clip, episode_index, self.seed + episode_index)
                for episode_index in range(self.episodes_per_clip)
            )
            results.append(
                ClipEvaluation(
                    clip_id=clip_id,
                    episodes=episodes,
                    summary=summarize_episodes(episodes),
                )
            )
        return tuple(results)


class WorstClipCheckpointSelector:
    """Select a checkpoint by worst-clip frame-0 performance first."""

    @staticmethod
    def rank(record: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        clips = list(record["clips"])
        success = [float(clip["success_rate"]) for clip in clips]
        reach = [float(clip["final_frame_reach_rate"]) for clip in clips]
        return (
            min(success),
            float(record["overall_success_rate"]),
            min(reach),
            -float(record["overall_object_position_error_m"]),
            -float(record["max_axis_point_error_m"]),
            -float(record["rotation_error_deg"]),
        )

    def select(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        candidates = list(records)
        if not candidates:
            raise ValueError("checkpoint selection requires at least one candidate")
        return max(candidates, key=self.rank)


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
