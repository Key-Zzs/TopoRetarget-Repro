from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from toporetarget.rl.axis_points import object_axis_points_from_poses
from toporetarget.rl.contracts import Stage16ReferenceClip
from toporetarget.rl.dynamic_coupling import (
    ObjectAwareResidualOracle,
    ResetVelocityProfile,
    finite_difference,
    reference_velocities,
)
from toporetarget.rl.environments.mujoco_backend import (
    MujocoReferenceTrackingBackend,
    materialize_free_object_scene,
)
from toporetarget.rl.failure_classifier import FailureClass
from toporetarget.rl.state_machine import (
    DynamicCouplingPhase,
    Stage161DynamicCouplingStateMachine,
)


def _backend(tmp_path: Path) -> MujocoReferenceTrackingBackend:
    source = Path("third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml")
    model = mujoco.MjModel.from_xml_path(str(source))
    bounds = model.jnt_range[: model.njnt].copy()
    joint_order = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)
    )
    timestamps = np.asarray([0.0, 0.05, 0.10])
    poses = np.broadcast_to(np.eye(4), (timestamps.size, 4, 4)).copy()
    poses[:, 2, 3] = 0.15
    poses[:, 0, 3] = np.asarray([0.0, 0.001, 0.002])
    q_ref = np.broadcast_to(bounds.mean(axis=1), (timestamps.size, model.njnt)).copy()
    q_ref[:, 0] += np.asarray([0.0, 0.01, 0.02])
    clip = Stage16ReferenceClip(
        timestamps=timestamps,
        q_finger_ref=q_ref,
        object_pose_base_ref=poses,
        object_axis_points_base_ref=object_axis_points_from_poses(poses),
        tracked_link_positions_base_ref=np.zeros((timestamps.size, 2, 3)),
        joint_order=joint_order,
        tracked_link_names=("r_wrist", "r_thumb_distal"),
        provenance={"dataset": "synthetic_dynamic_coupling"},
    )
    scene = materialize_free_object_scene(source, tmp_path, include_ground=False)
    backend = MujocoReferenceTrackingBackend(
        scene_path=scene,
        reference=clip,
        joint_lower=bounds[:, 0],
        joint_upper=bounds[:, 1],
        seed=5,
    )
    qdot, object_velocity = reference_velocities(clip)
    backend.set_reference_velocities(qdot=qdot, object_velocity=object_velocity)
    return backend


def test_finite_difference_uses_central_interior_and_one_sided_endpoints() -> None:
    timestamps = np.asarray([0.0, 1.0, 2.0, 3.0])
    signal = np.square(timestamps)[:, None]
    derivative = finite_difference(signal, timestamps).reshape(-1)
    assert derivative.tolist() == pytest.approx([1.0, 2.0, 4.0, 5.0])


def test_velocity_reset_profiles_snapshot_and_contact_trace(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    zero = backend.reset(reference_index=0, velocity_profile=ResetVelocityProfile.ZERO.value)
    assert np.allclose(zero["qdot"], 0.0)
    full = backend.reset(
        reference_index=0, velocity_profile=ResetVelocityProfile.FULL_REFERENCE.value
    )
    assert full["qdot"][0] == pytest.approx(0.2)
    assert full["object_velocity"][0] == pytest.approx(0.02)
    snapshot = backend.snapshot()
    backend.step(np.zeros(20))
    backend.restore(snapshot)
    assert backend.reference_index == 0
    assert backend.contact_report()["object_geom"]["contype"] == 1
    assert len(backend.collision_configuration()) >= 2


def test_object_aware_oracle_clones_state_and_outputs_bounded_20d_action(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.reset(reference_index=0, velocity_profile=ResetVelocityProfile.ZERO.value)
    snapshot = backend.snapshot()
    oracle = ObjectAwareResidualOracle()
    action = oracle.action(backend)
    assert action.shape == (20,)
    assert np.all(np.abs(action) <= 1.0)
    assert backend.reference_index == snapshot.reference_index
    assert np.array_equal(backend.data.qpos, snapshot.qpos)
    assert oracle.last_diagnostics is not None
    assert oracle.last_diagnostics.rank >= 0


def test_dynamic_state_machine_is_ordered_and_bounded() -> None:
    machine = Stage161DynamicCouplingStateMachine()
    first = machine.record_dynamic(
        phase=DynamicCouplingPhase.STEP_A_PD,
        failure_class=FailureClass.ACTUATOR_OR_PD_FAILURE,
        evidence={},
        repair="none",
        rerun_scope="both",
    )
    assert first.phase == "STEP_A_PD"
    with pytest.raises(ValueError, match="cannot skip"):
        machine.record_dynamic(
            phase=DynamicCouplingPhase.STEP_C_VELOCITY_RESET,
            failure_class=FailureClass.OBJECT_DYNAMICS_FAILURE,
            evidence={},
            repair="none",
            rerun_scope="both",
        )
