# ruff: noqa: E402 -- optional backend gate must precede the MuJoCo CLI import.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="requires the optional toporetarget[rl] extra")

from scripts.rl.visualize_hocap_world_wrist_policy_mujoco import (
    AdaptiveOracleVisualizationPolicy,
    _policy,
)
from toporetarget.rl.ppo.checkpoint import save_checkpoint
from toporetarget.rl.ppo.trainer import PPOConfig, PPOTrainer


def test_adaptive_oracle_hud_metadata_includes_all_candidate_gates() -> None:
    row = {
        "remaining": 12,
        "selected_requested_horizon": 5,
        "selected_effective_horizon": 5,
        "reason": "lexicographic_gate_first",
        "candidates": [
            {
                "requested_horizon": horizon,
                "diagnostics": {
                    "predicted_gate_violation": gate,
                    "minimum_gate_margin": 1.0 - gate,
                    "predicted_termination": None,
                },
            }
            for horizon, gate in ((1, 0.8), (5, 0.2), (10, 0.5))
        ],
    }
    metadata = AdaptiveOracleVisualizationPolicy._metadata_from_row(row)
    assert metadata["selected_horizon"] == 5
    assert metadata["remaining"] == 12
    assert metadata["candidate_gate_values"] == {"H1": 0.8, "H5": 0.2, "H10": 0.5}


def test_visualization_cli_exposes_adaptive_horizon_and_gate_overlays() -> None:
    script = Path("scripts/rl/visualize_hocap_world_wrist_policy_mujoco.py")
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "adaptive-oracle" in completed.stdout
    assert "--show-selected-horizon" in completed.stdout
    assert "--show-gate-margins" in completed.stdout
    assert "--mode {interactive,headless}" in completed.stdout


def test_visualization_loads_a_26d_ppo_checkpoint(tmp_path: Path) -> None:
    trainer = PPOTrainer(8, 26, config=PPOConfig(), device="cpu")
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint.pt",
        {
            "model": trainer.model.state_dict(),
            "normalizer": trainer.normalizer.state_dict(),
            "action_dim": 26,
        },
    )
    backend = SimpleNamespace(
        observation=lambda _state=None: np.zeros(8, dtype=np.float32),
    )
    args = SimpleNamespace(
        policy="ppo",
        checkpoint=checkpoint,
        device="cpu",
        deterministic=True,
    )
    policy = _policy(args, backend)
    action = policy({})
    assert action.shape == (26,)
    assert np.max(np.abs(action)) <= 1.0
