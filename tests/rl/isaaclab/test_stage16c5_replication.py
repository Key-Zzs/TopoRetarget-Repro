"""Replication error and tolerance contracts that do not need Isaac imports."""

from __future__ import annotations

import torch

from toporetarget.rl.isaaclab_oracle.metrics import state_differences
from toporetarget.rl.isaaclab_oracle.tolerance import freeze_tolerances


def _view() -> dict[str, torch.Tensor]:
    root = torch.tensor([[0.0, 0.0, 0.0, 1.0] + [0.0] * 9])
    return {
        "robot_joint_pos": torch.zeros((1, 26)),
        "robot_joint_vel": torch.zeros((1, 26)),
        "robot_root_state": root,
        "active_object_root_state": root.clone(),
    }


def test_replication_metric_and_noise_floor_have_hard_caps() -> None:
    first = _view()
    second = _view()
    assert all(value == 0.0 for value in state_differences(first, second).values())
    report = freeze_tolerances(
        {
            "wrist_position_m": [0.0] * 20,
            "object_position_m": [0.0] * 20,
            "quaternion_geodesic_rad": [0.0] * 20,
            "joint_position_rad": [0.0] * 20,
            "linear_velocity_si": [0.0] * 20,
            "angular_velocity_si": [0.0] * 20,
            "reward": [0.0] * 20,
        }
    )
    assert report["status"] == "REPLICATION_TOLERANCES_FROZEN"
    assert report["metrics"]["object_position_m"]["frozen_tolerance"] <= 1.0e-3


def test_noise_floor_fails_closed_above_hard_cap() -> None:
    samples = {
        "wrist_position_m": [0.0] * 19 + [1.0],
        "object_position_m": [0.0] * 20,
        "quaternion_geodesic_rad": [0.0] * 20,
        "joint_position_rad": [0.0] * 20,
        "linear_velocity_si": [0.0] * 20,
        "angular_velocity_si": [0.0] * 20,
        "reward": [0.0] * 20,
    }
    assert freeze_tolerances(samples)["status"] == "PHYSX_REPLICATION_BASELINE_NONDETERMINISM"
