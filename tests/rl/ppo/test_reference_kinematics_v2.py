from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from toporetarget.rl.environments.isaaclab_backend.reference_bank import WorldWristReferenceBank
from toporetarget.rl.reference_tracking.phase3_state import (
    ReferenceKinematicsPhase3Transitions,
    Stage16DReferenceKinematicsPhase3StateMachine,
)
from toporetarget.rl.reference_tracking.ppo26d_reference import export_factor8_reference
from toporetarget.rl.reference_tracking.ppo26d_reward import (
    TopoRetargetReferenceTrackingReward26DV2,
    ppo26d_reward_v2_object_twist_terms,
)
from toporetarget.rl.reference_tracking.reference_kinematics import (
    angular_velocity_body_from_world,
    derive_angular_velocity_world_wxyz,
    derive_linear_velocity,
    materialize_reference_kinematics_v2,
    qualify_reference_kinematics_v2,
)


def _source(path: Path) -> None:
    frames = 41
    times = np.arange(frames, dtype=np.float64) * 0.05
    angle = times * 0.4
    quaternion = np.stack(
        (np.cos(angle / 2.0), np.zeros(frames), np.zeros(frames), np.sin(angle / 2.0)), axis=-1
    )
    position = np.stack((0.1 * times, np.square(times), np.zeros(frames)), axis=-1)
    twist = np.concatenate(
        (
            derive_linear_velocity(position, times),
            derive_angular_velocity_world_wxyz(quaternion, times),
        ),
        axis=-1,
    )
    # This emulates the concrete defect: V1 used an endpoint tangent which is
    # not the second-order pose/time derivative of the terminal source pose.
    twist[-1, :3] += np.array((0.2, -0.1, 0.05))
    zero_axis = np.zeros((frames, 6, 3), dtype=np.float64)
    zero_links = np.zeros((frames, 16, 3), dtype=np.float64)
    np.savez_compressed(
        path,
        timestamps=times,
        wrist_pose_translation_world_ref=position,
        wrist_pose_quaternion_world_ref_wxyz=quaternion,
        wrist_twist_world_ref=twist,
        q_finger_ref=np.zeros((frames, 20), dtype=np.float64),
        qdot_finger_ref=np.zeros((frames, 20), dtype=np.float64),
        object_pose_translation_world_ref=position,
        object_pose_quaternion_world_ref_wxyz=quaternion,
        object_twist_world_ref=twist,
        object_axis_points_world_ref=zero_axis,
        tracked_link_positions_world_ref=zero_links,
        object_axis_points_wrist_ref=zero_axis,
        tracked_link_positions_wrist_ref=zero_links,
        metadata=np.asarray(
            json.dumps(
                {
                    "joint_order": [f"joint_{index}" for index in range(20)],
                    "tracked_link_names": [f"link_{index}" for index in range(16)],
                    "quaternion_convention": "wxyz_active_right_handed_shortest_rotation",
                }
            )
        ),
    )


def test_linear_derivative_uses_second_order_center_and_endpoints() -> None:
    times = np.arange(9, dtype=np.float64) * 0.1
    values = np.stack((np.square(times), times**3), axis=-1)
    expected = np.empty_like(values)
    expected[:, 0] = 2.0 * times
    expected[1:-1, 1] = 3.0 * np.square(times[1:-1]) + 0.1**2
    expected[0, 1] = -2.0 * 0.1**2
    expected[-1, 1] = 3.0 * times[-1] ** 2 - 2.0 * 0.1**2
    np.testing.assert_allclose(derive_linear_velocity(values, times), expected, atol=1.0e-12)


def test_so3_world_derivative_and_explicit_body_conversion() -> None:
    times = np.arange(9, dtype=np.float64) * 0.1
    angle = 0.8 * times
    quaternion = np.stack(
        (np.cos(angle / 2.0), np.zeros(9), np.zeros(9), np.sin(angle / 2.0)), axis=-1
    )
    world = derive_angular_velocity_world_wxyz(quaternion, times)
    np.testing.assert_allclose(world, np.tile((0.0, 0.0, 0.8), (9, 1)), atol=1.0e-12)
    np.testing.assert_allclose(
        angular_velocity_body_from_world(quaternion, world), world, atol=1.0e-12
    )


def test_v2_rebuilds_bad_terminal_tangent_preserves_keys_and_loads_bank(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    v1 = tmp_path / "v1.npz"
    v2 = tmp_path / "v2.npz"
    _source(source)
    export_factor8_reference(source, v1)
    materialize_reference_kinematics_v2(source, v1, v2)
    report = qualify_reference_kinematics_v2(source, v1, v2)
    assert report["status"] == "STAGE16D_REFERENCE_KINEMATICS_V2_VALIDATED"
    assert report["v1_audit"]["pose_rebuild_required"] is True
    assert report["v1_to_v2_pose_change"]["pose_changed"] is True
    assert report["checks"]["source_key_preservation"] is True
    with np.load(source, allow_pickle=False) as native, np.load(v2, allow_pickle=False) as repaired:
        np.testing.assert_allclose(
            repaired["object_pose_translation_world_ref"][::8],
            native["object_pose_translation_world_ref"],
            atol=1.0e-7,
        )
        assert np.array_equal(repaired["timestamps"], np.arange(321, dtype=np.float64) * 0.05)
    bank = WorldWristReferenceBank({"hocap_170105": v2, "hocap_170650": v2}, device="cpu")
    bank.apply_uniform_time_scale(8)
    assert bank.frame_count == 321
    assert bank.manifest.identifier == "world_wrist_reference_bank_kinematics_v2"


def test_reward_v2_tracks_signed_world_twist_and_retains_v1_terms() -> None:
    count = 2
    profile = TopoRetargetReferenceTrackingReward26DV2(
        object_velocity_sigma_mps=0.5,
        object_angular_velocity_sigma_radps=1.0,
    )
    zeros = torch.zeros
    terms = ppo26d_reward_v2_object_twist_terms(
        object_axis_points=zeros((count, 6, 3)),
        object_axis_points_ref=zeros((count, 6, 3)),
        tracked_links=zeros((count, 16, 3)),
        tracked_links_ref=zeros((count, 16, 3)),
        finger_q=zeros((count, 20)),
        finger_q_ref=zeros((count, 20)),
        joint_lower=-torch.ones(20),
        joint_upper=torch.ones(20),
        wrist_position=zeros((count, 3)),
        wrist_quaternion_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        wrist_position_ref=zeros((count, 3)),
        wrist_quaternion_ref_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        action=zeros((count, 26)),
        previous_action=zeros((count, 26)),
        second_previous_action=zeros((count, 26)),
        object_twist_world=torch.tensor([[0.5, 0.0, 0.0, 0.0, 1.0, 0.0]]).repeat(count, 1),
        object_twist_world_ref=zeros((count, 6)),
        profile=profile,
    )
    assert torch.allclose(terms["e_obj_vel"], torch.full((count,), 0.5))
    assert torch.allclose(terms["e_obj_ang_vel"], torch.ones(count))
    assert torch.allclose(terms["r_obj_vel"], torch.full((count,), np.exp(-1.0)))
    assert torch.allclose(terms["r_obj_ang_vel"], torch.full((count,), np.exp(-1.0)))
    assert torch.allclose(
        terms["total"],
        terms["r_object"] * profile.object_weight
        + terms["r_link"] * profile.link_weight
        + terms["r_finger"] * profile.finger_weight
        + terms["r_wrist_translation"] * profile.wrist_position_weight
        + terms["r_wrist_rotation"] * profile.wrist_rotation_weight
        + terms["r_obj_vel_weighted"]
        + terms["r_obj_ang_vel_weighted"],
    )
    with pytest.raises(ValueError, match="combined twist contribution"):
        TopoRetargetReferenceTrackingReward26DV2(
            object_velocity_weight=1.5, object_angular_velocity_weight=0.6
        )


def test_reward_v2_rejects_equal_speed_in_opposite_world_direction() -> None:
    """The added objective is a signed reference match, never speed matching."""

    profile = TopoRetargetReferenceTrackingReward26DV2(
        object_velocity_sigma_mps=0.1,
        object_angular_velocity_sigma_radps=0.1,
    )
    zeros = torch.zeros
    terms = ppo26d_reward_v2_object_twist_terms(
        object_axis_points=zeros((1, 6, 3)),
        object_axis_points_ref=zeros((1, 6, 3)),
        tracked_links=zeros((1, 16, 3)),
        tracked_links_ref=zeros((1, 16, 3)),
        finger_q=zeros((1, 20)),
        finger_q_ref=zeros((1, 20)),
        joint_lower=-torch.ones(20),
        joint_upper=torch.ones(20),
        wrist_position=zeros((1, 3)),
        wrist_quaternion_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        wrist_position_ref=zeros((1, 3)),
        wrist_quaternion_ref_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        action=zeros((1, 26)),
        previous_action=zeros((1, 26)),
        second_previous_action=zeros((1, 26)),
        object_twist_world=torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.1, 0.0]]),
        object_twist_world_ref=torch.tensor([[-0.1, 0.0, 0.0, 0.0, -0.1, 0.0]]),
        profile=profile,
    )
    assert torch.allclose(terms["e_obj_vel"], torch.tensor([0.2]))
    assert torch.allclose(terms["e_obj_ang_vel"], torch.tensor([0.2]))
    assert torch.all(terms["r_obj_vel"] < 0.02)
    assert torch.all(terms["r_obj_ang_vel"] < 0.02)


def test_phase3_recovery_ledger_is_strictly_forward_and_evidence_backed() -> None:
    ledger = ReferenceKinematicsPhase3Transitions()
    ledger.transition(
        Stage16DReferenceKinematicsPhase3StateMachine.KINEMATICS_QUALIFICATION,
        reason="STAGE16D_REFERENCE_KINEMATICS_V2_VALIDATED",
    )
    ledger.transition(
        Stage16DReferenceKinematicsPhase3StateMachine.PHASE3_ENTRY,
        reason="PHASE3_OBJECT_TWIST_REWARD_RECOMMENDED",
    )
    assert ledger.transitions[-1]["to"] == "PHASE3_ENTRY"
    with pytest.raises(ValueError, match="advance monotonically"):
        ledger.transition(
            Stage16DReferenceKinematicsPhase3StateMachine.PHASE1_RERUN,
            reason="rewind is forbidden",
        )
    with pytest.raises(ValueError, match="evidence reason"):
        ledger.transition(
            Stage16DReferenceKinematicsPhase3StateMachine.PHASE3_P1,
            reason="",
        )
