from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from toporetarget.rl.contact_skill_collapse import (
    command_tracking_metrics,
    detect_contact_milestones,
    lift_timing,
)
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer


def test_c0_cli_defaults_to_contact_preserving_uniform_rsi() -> None:
    from scripts.rl.isaaclab.train_stage16_full_trajectory_p3 import _resolve_training_reset

    assert _resolve_training_reset("C0", None) == "uniform_rsi"
    assert _resolve_training_reset("C0", "frame0") == "frame0"


def test_later_physical_stage_cli_keeps_frame0_default() -> None:
    from scripts.rl.isaaclab.train_stage16_full_trajectory_p3 import _resolve_training_reset

    assert _resolve_training_reset("C1", None) == "frame0"


def test_contact_milestones_follow_preregistered_boundaries() -> None:
    rows = [
        {"update": 1, "samples": 40, "episodes": 10, "contact_episodes": 10},
        {"update": 2, "samples": 80, "episodes": 10, "contact_episodes": 9},
        {"update": 3, "samples": 120, "episodes": 10, "contact_episodes": 5},
        {"update": 4, "samples": 160, "episodes": 10, "contact_episodes": 0},
        {"update": 5, "samples": 200, "episodes": 10, "contact_episodes": 0},
        {"update": 6, "samples": 240, "episodes": 10, "contact_episodes": 0},
    ]

    result = detect_contact_milestones(rows, baseline_contact_episodes=10)

    assert result["U_FIRST_DEGRADATION"] == {"update": 2, "samples": 80}
    assert result["U_MAJOR_COLLAPSE"] == {"update": 3, "samples": 120}
    assert result["U_ZERO_CONTACT"] == {"update": 4, "samples": 160}
    assert result["U_PERSISTENT_ZERO"] == {
        "update": 6,
        "samples": 240,
        "run_start_update": 4,
        "run_start_samples": 160,
    }


def test_command_decomposition_and_lift_timing_are_frame_consistent() -> None:
    count = 12
    identity = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (count, 1))
    wrist_reference = np.concatenate((np.zeros((count, 3)), identity), axis=-1)
    wrist_target = wrist_reference.copy()
    wrist_target[:, 0] = 0.01
    wrist_pose = wrist_target.copy()
    wrist_pose[:, 0] += 0.001
    finger_reference = np.zeros((count, 20))
    finger_target = np.full((count, 20), 0.2)
    finger_q = finger_target + 0.01
    phase = np.asarray(["PRE_CONTACT"] * 2 + ["CONTACT"] * 3 + ["GRASP"] * 2 + ["LIFT"] * 5)
    presence = np.zeros((count, 21), dtype=bool)
    presence[5:9, 0] = True
    wrist_twist = np.zeros((count, 6))
    wrist_twist[2:7, 2] = 0.03
    object_pose = np.zeros((count, 7))
    object_pose[:, 3] = 1.0
    object_pose[8:, 2] = 0.006
    trace = {
        "wrist_reference": wrist_reference,
        "wrist_target": wrist_target,
        "wrist_pose": wrist_pose,
        "finger_reference": finger_reference,
        "finger_target": finger_target,
        "finger_q": finger_q,
        "phase": phase,
        "hand_object_pair_presence": presence,
        "wrist_twist_world": wrist_twist,
        "object_pose": object_pose,
    }

    command = command_tracking_metrics(trace)
    timing = lift_timing(trace)

    assert command["wrist_position_ref_to_command_m"]["mean"] == pytest.approx(0.01)
    assert command["wrist_position_command_to_actual_m"]["mean"] == pytest.approx(0.001)
    assert command["finger_ref_to_command_rad"]["mean"] == pytest.approx(0.2)
    assert command["finger_command_to_actual_rad"]["mean"] == pytest.approx(0.01)
    assert timing["persistent_contact"] == 5
    assert timing["reference_lift_onset"] == 7
    assert timing["actual_wrist_up_onset"] == 2
    assert timing["object_lift_onset"] == 8
    assert timing["premature_lift"]


def test_exact_batch_persistence_contains_update_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ReferenceBank:
        frame_count = 321

    class FakeEnv:
        num_envs = 2
        reference_bank = ReferenceBank()

        def __init__(self) -> None:
            self._reference_index = torch.zeros(2, dtype=torch.long)
            self._last_reward_terms = {"total": torch.ones(2)}

        def reset(self):
            return {"policy": torch.zeros(2, 764)}, {}

        def step(self, action):
            del action
            self._reference_index += 1
            done = torch.zeros(2, dtype=torch.bool)
            return {"policy": torch.zeros(2, 764)}, torch.ones(2), done, done, {}

        def rsi_report(self):
            return {"reset_reference_index": "frame0"}

    trainer = PPO26DTrainer(observation_dim=764, device="cpu")
    monkeypatch.setattr(
        trainer.trainer,
        "update",
        lambda storage, last_value: {
            "actor_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "ratio": 1.0,
            "action_std": 1.0,
            "sample_count": float(storage.sample_count),
            "updates": 1.0,
        },
    )
    batch = tmp_path / "exact" / "update_0001.pt"

    result = trainer.collect_and_update(FakeEnv(), rollout_length=3, exact_batch_path=batch)
    payload = torch.load(batch, map_location="cpu", weights_only=False)

    assert payload["schema_version"] == "Stage16ContactCollapseExactPPOBatchV1"
    assert payload["observations"].shape == (3, 2, 764)
    assert payload["actions"].shape == (3, 2, 26)
    assert payload["reference_indices"].shape == (3, 2)
    assert payload["advantages"].shape == (3, 2)
    assert payload["returns"].shape == (3, 2)
    assert "rng_before_optimizer_update" in payload
    assert result["actor_parameter_update_norm"] == pytest.approx(0.0)
