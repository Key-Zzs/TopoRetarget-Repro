from __future__ import annotations

import json

import numpy as np
import pytest

from toporetarget.rl.physics_retargeting.contracts import PhysicsConsistentTaskGateV1
from toporetarget.rl.physics_retargeting.export import export_physics_consistent_trajectory
from toporetarget.rl.physics_retargeting.qualification import (
    qualify_physics_consistent_replicas,
)
from toporetarget.rl.physics_retargeting.trajectory_recorder import (
    PhysicsConsistentTrajectoryRecorderV1,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "actions": np.zeros((321, 26), dtype=np.float32),
        "object_pose": np.zeros((321, 7), dtype=np.float32),
        "finger_q": np.zeros((321, 20), dtype=np.float32),
        "wrist_pose": np.zeros((321, 7), dtype=np.float32),
    }


def test_export_reload_and_provenance(tmp_path) -> None:
    paths = export_physics_consistent_trajectory(
        output_dir=tmp_path / "trajectory",
        arrays=_arrays(),
        manifest={"clip": "synthetic", "source_hash": "abc"},
        quality={"status": "PARTIAL"},
    )
    with np.load(paths["trajectory"], allow_pickle=False) as source:
        assert source["actions"].shape == (321, 26)
    manifest = json.loads((tmp_path / "trajectory/manifest.json").read_text())
    assert manifest["object_trajectory_role"] == "free_physx_rollout_output_not_decision_variable"
    assert not manifest["source_overwritten"]
    with pytest.raises(FileExistsError):
        export_physics_consistent_trajectory(
            output_dir=tmp_path / "trajectory",
            arrays=_arrays(),
            manifest={},
            quality={},
        )


def test_recorder_requires_all_321_samples() -> None:
    recorder = PhysicsConsistentTrajectoryRecorderV1("synthetic")
    with pytest.raises(RuntimeError, match="incomplete trajectory"):
        recorder.finalize()


def test_qualification_rejects_degenerate_no_contact() -> None:
    gate = PhysicsConsistentTaskGateV1(
        clip="synthetic",
        object_bbox_diagonal_m=0.1,
        minimum_contact_recall=0.5,
        minimum_semantic_progress=0.3,
        minimum_object_motion_m=0.01,
        minimum_object_rotation_deg=0.0,
        terminal_window_control_steps=5,
        workspace_radius_m=0.5,
    )
    row = {
        "clip": "synthetic",
        "success": False,
        "semantic_progress": 0.0,
        "contact_recall": 0.0,
        "max_penetration_m": 0.0,
        "p95_penetration_m": 0.0,
        "max_inter_finger_penetration_m": 0.0,
        "no_hidden_control": True,
        "formal_object_state_writes": 0,
        "formal_wrist_state_writes": 0,
        "action_bounds_pass": True,
        "contact_causality_pass": False,
        "complete_trajectory": True,
        "terminal_stability_pass": False,
        "numerical_pass": True,
        "termination": "TIMEOUT",
    }
    result = qualify_physics_consistent_replicas([dict(row) for _ in range(20)], gate)
    assert result["classification"] == "DEGENERATE_SEED"
    assert result["ppo_entry"] == "PPO_NOT_AUTHORIZED_FOR_CLIP"
