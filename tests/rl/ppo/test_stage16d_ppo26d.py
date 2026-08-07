from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from toporetarget.rl.ppo.gpu_capacity import (
    GpuCapacityMeasurement,
    select_ppo26d_environment_capacity,
)
from toporetarget.rl.ppo.ppo26d_contract import (
    Stage16DPPO26DObservationV2,
    Stage16DReferenceResidualAction26DV1,
)
from toporetarget.rl.reference_tracking.ppo26d_reference import (
    export_factor8_reference,
    inspect_source_reference,
)
from toporetarget.rl.reference_tracking.ppo26d_reward import (
    TopoRetargetReferenceTrackingReward26DV1,
)
from toporetarget.rl.reference_tracking.ppo26d_rsi import (
    rsi_histogram,
    sample_uniform_reference_indices,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _source_reference(path: Path) -> None:
    frames = 41
    metadata = {
        "joint_order": [f"joint_{index}" for index in range(20)],
        "tracked_link_names": [f"link_{index}" for index in range(16)],
    }
    quaternion = np.zeros((frames, 4), dtype=np.float32)
    quaternion[:, 0] = 1.0
    np.savez_compressed(
        path,
        timestamps=np.arange(frames, dtype=np.float32) / 20.0,
        wrist_pose_translation_world_ref=np.zeros((frames, 3), dtype=np.float32),
        wrist_pose_quaternion_world_ref_wxyz=quaternion,
        wrist_twist_world_ref=np.zeros((frames, 6), dtype=np.float32),
        q_finger_ref=np.zeros((frames, 20), dtype=np.float32),
        qdot_finger_ref=np.zeros((frames, 20), dtype=np.float32),
        object_pose_translation_world_ref=np.zeros((frames, 3), dtype=np.float32),
        object_pose_quaternion_world_ref_wxyz=quaternion,
        object_twist_world_ref=np.zeros((frames, 6), dtype=np.float32),
        object_axis_points_world_ref=np.zeros((frames, 6, 3), dtype=np.float32),
        tracked_link_positions_world_ref=np.zeros((frames, 16, 3), dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata)),
    )


def test_action_and_observation_contracts_are_frozen() -> None:
    action = Stage16DReferenceResidualAction26DV1()
    observation = Stage16DPPO26DObservationV2()
    assert action.action_dimension == 26
    assert action.wrist_slice == (0, 6)
    assert action.finger_slice == (6, 26)
    assert not action.direct_articulation_action
    assert observation.dimension == 764
    assert observation.lookahead_offsets == (0, 1, 3, 5)
    assert sum(observation.field_dimensions().values()) == 764


def test_factor8_reference_contract_exports_321_samples(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    destination = tmp_path / "derived.npz"
    _source_reference(source)
    result = export_factor8_reference(source, destination)
    assert inspect_source_reference(source)["source_frames"] == 41
    assert result["contract"]["runtime_samples"] == 321
    with np.load(destination, allow_pickle=False) as archive:
        assert archive["timestamps"].shape == (321,)
        assert archive["q_finger_ref"].shape == (321, 20)
        assert archive["tracked_link_positions_world_ref"].shape == (321, 16, 3)


def test_rsi_samples_valid_full_reference_range() -> None:
    values = sample_uniform_reference_indices(np.random.default_rng(8), count=1024, frame_count=321)
    report = rsi_histogram(values, frame_count=321)
    assert values.min() >= 0 and values.max() < 321
    assert report["sample_count"] == 1024
    assert set(report["phase_counts"]) == {
        "approach",
        "first_contact",
        "persistent_contact",
        "terminal",
    }


def test_reward_excludes_post_ppo_bonus_leakage() -> None:
    profile = TopoRetargetReferenceTrackingReward26DV1()
    assert profile.terminal_contact_bonus == 0.0
    assert profile.penetration_reward == 0.0
    assert profile.inter_finger_penalty == 0.0
    with pytest.raises(ValueError, match="post-PPO"):
        TopoRetargetReferenceTrackingReward26DV1(terminal_contact_bonus=1.0)


def test_gpu_capacity_requires_update_headroom_and_95_percent_rule() -> None:
    rows = [
        GpuCapacityMeasurement(512, 800.0, 16000.0, 6000.0, 7000.0, True, True),
        GpuCapacityMeasurement(1024, 960.0, 16000.0, 9000.0, 4000.0, True, True),
        GpuCapacityMeasurement(1536, 1000.0, 16000.0, 13000.0, 1800.0, True, True),
        GpuCapacityMeasurement(2048, 980.0, 16000.0, 11000.0, 3200.0, True, True),
    ]
    result = select_ppo26d_environment_capacity(rows)
    assert result["selected_num_envs"] == 1024
    assert "95 percent" in result["selection_reason"]


def test_ppo26d_runtime_has_no_rollout_state_write_in_action_method() -> None:
    path = (
        REPO_ROOT
        / "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "def _apply_action" not in source
    assert "PPO26D_ROLLOUT_STATE_WRITE_FORBIDDEN" in source
    assert "def rollout_state_write_report" in source
    assert (
        "direct_articulation_action"
        in (REPO_ROOT / "src/toporetarget/rl/ppo/ppo26d_contract.py").read_text()
    )


def test_torch_is_available_for_ppo26d_pure_contracts() -> None:
    assert torch.isfinite(torch.tensor([0.0])).all()
