from __future__ import annotations

import numpy as np
import pytest
import torch

from toporetarget.rl.ppo.behavior_cloning import (
    BehaviorCloningConfigV1,
    train_actor_behavior_cloning,
)
from toporetarget.rl.ppo.checkpoint_selection import select_physics_correction_checkpoint
from toporetarget.rl.ppo.demonstrations import (
    PhysicsCorrectionTrajectoryV1,
    split_demonstrations_by_trajectory,
)
from toporetarget.rl.ppo.networks import ActorCritic
from toporetarget.rl.ppo.physics_correction_ppo import (
    SAMPLE_LADDER,
    PhysicsCorrectionPPOV1,
    next_sample_target,
)


def _trajectory(index: int) -> PhysicsCorrectionTrajectoryV1:
    frames = 4
    return PhysicsCorrectionTrajectoryV1(
        trajectory_id=f"trajectory-{index}",
        clip="synthetic",
        observations=np.zeros((frames, 764), dtype=np.float32),
        actions=np.zeros((frames, 26), dtype=np.float32),
        next_observations=np.zeros((frames, 764), dtype=np.float32),
        rewards=np.zeros(frames, dtype=np.float32),
        semantic_progress=np.linspace(0.0, 1.0, frames, dtype=np.float32),
        contact_topology=np.zeros((frames, 6), dtype=np.float32),
        done=np.array([False, False, False, True]),
        success=True,
        source_hash="source",
        corrected_hash=f"corrected-{index}",
    )


def test_demonstration_split_is_by_complete_trajectory() -> None:
    train, validation = split_demonstrations_by_trajectory([_trajectory(i) for i in range(5)])
    assert len(train) == 4
    assert len(validation) == 1
    assert {row.trajectory_id for row in train}.isdisjoint(
        {row.trajectory_id for row in validation}
    )


def test_actor_only_bc_preserves_critic(tmp_path) -> None:
    model = ActorCritic(764, 26)
    critic = {name: value.detach().clone() for name, value in model.critic.state_dict().items()}
    result = train_actor_behavior_cloning(
        model=model,
        train_observations=np.zeros((8, 764), dtype=np.float32),
        train_actions=np.zeros((8, 26), dtype=np.float32),
        validation_observations=np.zeros((4, 764), dtype=np.float32),
        validation_actions=np.zeros((4, 26), dtype=np.float32),
        output_dir=tmp_path,
        config=BehaviorCloningConfigV1(max_epochs=2, patience=1, batch_size=4),
    )
    assert result["actor_only"]
    assert all(
        torch.equal(value, critic[name]) for name, value in model.critic.state_dict().items()
    )
    assert (tmp_path / "bc_best.pt").is_file()
    assert (tmp_path / "bc_last.pt").is_file()


def test_sample_ladder_and_curriculum_are_frozen() -> None:
    contract = PhysicsCorrectionPPOV1().as_dict()
    assert tuple(contract["sample_ladder"]) == SAMPLE_LADDER
    assert contract["curriculum"] == ["P0", "P1", "P2", "P3"]
    assert next_sample_target(0) == SAMPLE_LADDER[0]
    assert next_sample_target(SAMPLE_LADDER[-1]) is None


def test_checkpoint_selector_requires_frame_zero() -> None:
    record = {
        "frame_zero_full_episode": True,
        "success_rate": 0.9,
        "semantic_reach_rate": 0.9,
        "contact_pass_rate": 0.9,
        "penetration_pass_rate": 1.0,
        "robot_deviation": 0.1,
        "action_smoothness": 0.1,
        "checkpoint": "a.pt",
    }
    assert select_physics_correction_checkpoint([record]) == record
    with pytest.raises(ValueError, match="frame-zero"):
        select_physics_correction_checkpoint([{**record, "frame_zero_full_episode": False}])
