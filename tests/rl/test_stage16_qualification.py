from __future__ import annotations

import numpy as np

from toporetarget.rl.evaluation import (
    EpisodeMetrics,
    FrameZeroEvaluator,
    WorstClipCheckpointSelector,
)
from toporetarget.rl.oracle import OracleResidualController
from toporetarget.rl.ppo.trainer import PPOConfig
from toporetarget.rl.state_machine import (
    Stage161RecoveryStateMachine,
    Stage162RecoveryStateMachine,
    Stage163RecoveryStateMachine,
)


def _metric(success: bool, progress: float) -> EpisodeMetrics:
    return EpisodeMetrics(
        termination="SUCCESS_REFERENCE_COMPLETE" if success else "FAILURE_OBJECT_POSITION",
        success=success,
        final_frame_reached=success,
        object_position_error_m=0.01,
        object_rotation_error_deg=1.0,
        max_axis_point_error_m=0.01,
        link_rmse_m=0.001,
        normalized_joint_error=0.01,
        progress_ratio=progress,
        return_value=1.0,
        action_magnitude=0.1,
        action_first_difference=0.1,
        action_second_difference=0.1,
    )


def test_frame_zero_evaluator_retains_failures_and_uses_deterministic_seeds() -> None:
    calls: list[tuple[str, int, int]] = []

    def run(clip: str, episode: int, seed: int) -> EpisodeMetrics:
        calls.append((clip, episode, seed))
        return _metric(success=episode == 1, progress=episode / 2)

    result = FrameZeroEvaluator(episodes_per_clip=3, seed=11).evaluate(
        [("a", "clip-a"), ("b", "clip-b")], run
    )
    assert [item.clip_id for item in result] == ["a", "b"]
    assert all(len(item.episodes) == 3 for item in result)
    assert calls == [
        ("clip-a", 0, 11),
        ("clip-a", 1, 12),
        ("clip-a", 2, 13),
        ("clip-b", 0, 11),
        ("clip-b", 1, 12),
        ("clip-b", 2, 13),
    ]


def test_worst_clip_selector_is_not_overall_only() -> None:
    selector = WorstClipCheckpointSelector()
    high_overall = {
        "id": "overall",
        "clips": [
            {"success_rate": 1.0, "final_frame_reach_rate": 1.0},
            {"success_rate": 0.0, "final_frame_reach_rate": 0.0},
        ],
        "overall_success_rate": 0.5,
        "overall_object_position_error_m": 0.01,
        "max_axis_point_error_m": 0.01,
        "rotation_error_deg": 1.0,
    }
    balanced = {
        "id": "balanced",
        "clips": [
            {"success_rate": 0.8, "final_frame_reach_rate": 0.8},
            {"success_rate": 0.8, "final_frame_reach_rate": 0.8},
        ],
        "overall_success_rate": 0.8,
        "overall_object_position_error_m": 0.03,
        "max_axis_point_error_m": 0.03,
        "rotation_error_deg": 3.0,
    }
    assert selector.select([high_overall, balanced])["id"] == "balanced"


def test_oracle_has_same_normalized_action_shape_and_no_object_input() -> None:
    controller = OracleResidualController(joint_gain=0.5, action_scale_fraction=0.05)
    q = np.zeros(3)
    action_a = controller.action(
        q=q,
        q_ref=np.zeros(3),
        q_ref_next=np.full(3, 0.01),
        joint_lower=-np.ones(3),
        joint_upper=np.ones(3),
    )
    action_b = controller.action(
        q=q,
        q_ref=np.zeros(3),
        q_ref_next=np.full(3, 0.01),
        joint_lower=-np.ones(3),
        joint_upper=np.ones(3),
    )
    assert action_a.shape == (3,)
    assert np.array_equal(action_a, action_b)
    assert np.all(np.abs(action_a) <= 1.0)


def test_qualification_state_machine_budgets_are_versioned() -> None:
    assert Stage161RecoveryStateMachine().budget.major_repairs == 12
    assert Stage162RecoveryStateMachine().budget.major_repairs == 16
    assert Stage163RecoveryStateMachine().budget.major_repairs == 16
    config = PPOConfig()
    assert config.epochs == 4
    assert config.minibatches == 32
