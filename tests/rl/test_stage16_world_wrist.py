from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="requires the optional toporetarget[rl] extra")

from toporetarget.contracts.reference import RobotReferenceV2
from toporetarget.geometry.se3 import invert_transform, relative_transform, transform_points
from toporetarget.rl.axis_points import object_axis_points_from_poses
from toporetarget.rl.environments.world_wrist_backend import (
    CartesianWristImpedanceController,
    WorldWristFingerBackend,
    WorldWristObservationContractV1,
    WristFingerActionScaleV1,
    WristImpedanceProfileV1,
    materialize_world_wrist_free_object_scene,
    mujoco_freejoint_velocity_from_world_twist,
    world_twist_from_mujoco_freejoint_velocity,
)
from toporetarget.rl.object_dynamics_audit import (
    impulse_sensitivity_candidates,
    inertial_wrench_demand,
    reference_accelerations,
    support_model_audit,
)
from toporetarget.rl.tracked_links import TRACKED_LINKS_WUJI_RH
from toporetarget.rl.world_wrist import (
    WorldWristFingerReferenceV1,
    export_world_wrist_reference,
    matrix_from_quaternion_wxyz,
    quaternion_wxyz_from_matrix,
)
from toporetarget.rl.world_wrist_oracle import (
    ContactAwareMPCConfig,
    WorldWristFingerObjectAwareOracle,
)


def _reference(model: mujoco.MjModel) -> WorldWristFingerReferenceV1:
    timestamps = np.asarray([0.0, 0.05, 0.10])
    wrist = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    wrist[:, 0, 3] = np.asarray([0.0, 0.002, 0.004])
    object_pose = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    object_pose[:, 2, 3] = 0.15
    object_pose[:, 0, 3] = np.asarray([0.0, 0.002, 0.004])
    axes = object_axis_points_from_poses(object_pose)
    links = np.zeros((3, len(TRACKED_LINKS_WUJI_RH), 3))
    joint_order = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)
    )
    assert all(name is not None for name in joint_order)
    q = np.broadcast_to(model.jnt_range[: model.njnt].mean(axis=1), (3, model.njnt)).copy()
    return WorldWristFingerReferenceV1(
        timestamps=timestamps,
        source_frame_indices=np.asarray([0, 1, 2]),
        wrist_pose_world_ref=wrist,
        wrist_twist_world_ref=np.zeros((3, 6)),
        q_finger_ref=q,
        qdot_finger_ref=np.zeros_like(q),
        object_pose_world_ref=object_pose,
        object_twist_world_ref=np.zeros((3, 6)),
        object_axis_points_world_ref=axes,
        tracked_link_positions_world_ref=links,
        object_pose_wrist_ref=relative_transform(wrist, object_pose),
        object_axis_points_wrist_ref=transform_points(invert_transform(wrist), axes),
        tracked_link_positions_wrist_ref=transform_points(invert_transform(wrist), links),
        joint_order=tuple(name for name in joint_order if name is not None),
        tracked_link_names=TRACKED_LINKS_WUJI_RH,
        provenance={"test": "world_wrist"},
    )


def test_world_reference_round_trip_and_relative_reconstruction(tmp_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path("third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml")
    reference = _reference(model)
    report = reference.validate(
        joint_lower=model.jnt_range[: model.njnt, 0],
        joint_upper=model.jnt_range[: model.njnt, 1],
    )
    assert report["control_hz"] == pytest.approx(20.0)
    assert report["relative_reconstruction"]["translation_max_error_m"] <= 1e-12
    path = reference.to_npz(tmp_path / "reference.npz")
    restored = WorldWristFingerReferenceV1.from_npz(path)
    assert restored.content_hash() == reference.content_hash()
    assert np.allclose(restored.wrist_quaternion_world_ref_wxyz[:, 0], 1.0)


def test_stage12_robot_reference_exports_world_pose_and_resamples_to_20hz() -> None:
    model = mujoco.MjModel.from_xml_path("third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml")
    timestamps = np.asarray([0.0, 0.1, 0.2])
    wrist = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    wrist[:, 0, 3] = np.asarray([0.0, 0.01, 0.02])
    object_base = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    object_base[:, 2, 3] = 0.15
    joint_order = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index))
        for index in range(model.njnt)
    )
    source = RobotReferenceV2(
        qpos_reference=np.broadcast_to(model.jnt_range[: model.njnt].mean(axis=1), (3, model.njnt)),
        base_pose=wrist,
        object_pose_base=object_base,
        tracked_link_positions=np.zeros((3, len(TRACKED_LINKS_WUJI_RH), 3)),
        timestamps=timestamps,
        fps=10.0,
        joint_order=joint_order,
        robot_hash="test-wuji-hash",
        dataset_provenance={"source_final_artifact": "stage12-final", "clip": "synthetic"},
        frame_indices=np.asarray([10, 11, 12]),
        tracked_link_names=TRACKED_LINKS_WUJI_RH,
    )
    exported = export_world_wrist_reference(
        source, source_hashes={"stage12_final": "abc123"}, resample_to_hz=20.0
    )
    validation = exported.validate()
    assert exported.frame_count == 5
    assert validation["control_hz"] == pytest.approx(20.0)
    assert np.allclose(exported.wrist_translation_world_ref[:, 0], np.linspace(0.0, 0.02, 5))
    assert np.allclose(
        exported.object_pose_world_ref[:, 0, 3], exported.wrist_translation_world_ref[:, 0]
    )
    assert exported.provenance["source_hashes"] == {"stage12_final": "abc123"}


def test_quaternion_shortest_sign_and_se3_residual_controller() -> None:
    rotation = np.diag([-1.0, -1.0, 1.0])
    quaternion = quaternion_wxyz_from_matrix(rotation)
    assert quaternion[0] >= 0.0
    assert np.allclose(matrix_from_quaternion_wxyz(quaternion), rotation)
    controller = CartesianWristImpedanceController(
        WristImpedanceProfileV1(), wrist_mass_kg=1.0, wrist_inertia_kgm2=np.ones(3)
    )
    target = controller.target_pose(
        np.eye(4), np.asarray([1, 0, 0, 0, 0, 1.0]), WristFingerActionScaleV1()
    )
    assert target[:3, 3] == pytest.approx([0.01, 0.0, 0.0])
    assert np.linalg.norm(target[:3, :3] - np.eye(3)) > 0.0
    result = controller.compute(
        target_pose_world=target,
        target_twist_world=np.zeros(6),
        current_pose_world=np.eye(4),
        current_freejoint_velocity=np.zeros(6),
    )
    assert np.linalg.norm(result.applied_wrench_world[:3]) <= 25.0
    assert np.linalg.norm(result.applied_wrench_world[3:]) <= 1.5


def test_default_wrist_profile_stays_below_discrete_time_stability_ceiling() -> None:
    profile = WristImpedanceProfileV1()
    minimum_effective_inertia = 7.9e-4
    physics_timestep_s = 0.01
    discrete_natural_frequency = (
        np.sqrt(profile.rotation_stiffness_nmprad / minimum_effective_inertia) * physics_timestep_s
    )
    assert discrete_natural_frequency < 0.6


def test_world_twist_and_mujoco_freejoint_velocity_frame_contract() -> None:
    pose = np.eye(4)
    pose[:3, :3] = matrix_from_quaternion_wxyz(np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]))
    world_twist = np.asarray([1.0, 2.0, 3.0, 0.2, -0.4, 0.6])
    freejoint_velocity = mujoco_freejoint_velocity_from_world_twist(pose, world_twist)
    assert freejoint_velocity[:3] == pytest.approx(world_twist[:3])
    assert freejoint_velocity[3:] == pytest.approx(pose[:3, :3].T @ world_twist[3:])
    assert world_twist_from_mujoco_freejoint_velocity(pose, freejoint_velocity) == pytest.approx(
        world_twist
    )

    controller = CartesianWristImpedanceController(
        WristImpedanceProfileV1(), wrist_mass_kg=1.0, wrist_inertia_kgm2=np.ones(3)
    )
    result = controller.compute(
        target_pose_world=pose,
        target_twist_world=world_twist,
        current_pose_world=pose,
        current_freejoint_velocity=freejoint_velocity,
    )
    assert result.linear_velocity_error_world_mps == pytest.approx(np.zeros(3))
    assert result.angular_velocity_error_local_radps == pytest.approx(np.zeros(3))


def test_world_backend_has_two_freejoints_26d_actions_and_clone_only_oracle(tmp_path: Path) -> None:
    source = Path("third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml")
    model = mujoco.MjModel.from_xml_path(str(source))
    mesh = tmp_path / "object.obj"
    mesh.write_text(
        "v 0 0 0\nv 0.02 0 0\nv 0 0.02 0\nv 0 0 0.02\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n",
        encoding="utf-8",
    )
    scene = materialize_world_wrist_free_object_scene(source, tmp_path / "scene", object_mesh=mesh)
    backend = WorldWristFingerBackend(
        scene_path=scene,
        reference=_reference(model),
        joint_lower=model.jnt_range[: model.njnt, 0],
        joint_upper=model.jnt_range[: model.njnt, 1],
        seed=3,
    )
    state = backend.reset(reference_index=0)
    assert backend.action_dim == 26
    assert backend.wrist_qpos_address != backend.object_qpos_address
    assert backend.model_report()["formal_rollout_object_pose_write"] is False
    expected_wrist_qvel = mujoco_freejoint_velocity_from_world_twist(
        backend.reference.wrist_pose_world_ref[0], backend.reference.wrist_twist_world_ref[0]
    )
    assert backend.data.qvel[
        backend.wrist_dof_address : backend.wrist_dof_address + 6
    ] == pytest.approx(expected_wrist_qvel)
    assert state["wrist_twist"] == pytest.approx(backend.reference.wrist_twist_world_ref[0])
    assert backend.observation(state).shape == (WorldWristObservationContractV1(20, 16).dimension,)
    snapshot = backend.snapshot()
    warmstart = backend.data.qacc_warmstart.copy()
    oracle = WorldWristFingerObjectAwareOracle()
    action = oracle.action(backend, horizon=5)
    assert action.shape == (26,)
    assert np.all(np.abs(action) <= 1.0)
    assert oracle.last_action_sequence is not None
    assert oracle.last_action_sequence.shape == (2, 26)
    assert oracle.last_diagnostics is not None
    assert oracle.last_diagnostics.sequence_shape == (2, 26)
    assert oracle.last_diagnostics.evaluated_sequences == 100
    assert oracle.last_diagnostics.direct_object_control is False
    assert np.array_equal(snapshot.qpos, backend.data.qpos)
    assert np.array_equal(warmstart, backend.data.qacc_warmstart)
    replay_backend = WorldWristFingerBackend(
        scene_path=scene,
        reference=_reference(model),
        joint_lower=model.jnt_range[: model.njnt, 0],
        joint_upper=model.jnt_range[: model.njnt, 1],
        seed=3,
    )
    replay_backend.reset(reference_index=0)
    backend.transition(action)
    replay_backend.transition(action)
    second_action = oracle.action(backend, horizon=5)
    _, reward, _ = backend.transition(second_action)
    replay_backend.transition(second_action)
    assert np.array_equal(backend.data.qpos, replay_backend.data.qpos)
    assert np.array_equal(backend.data.qvel, replay_backend.data.qvel)
    assert np.array_equal(backend.data.qacc_warmstart, replay_backend.data.qacc_warmstart)
    assert np.isfinite(list(reward.values())).all()
    assert backend.last_control is not None
    assert len(backend.last_physics_trace) == backend.decimation
    assert {
        "hand_object_contact_count",
        "hand_object_normal_force_n",
        "hand_object_normal_impulse_ns",
        "hand_object_max_penetration_m",
    }.issubset(backend.last_physics_trace[0])
    assert backend.model_report()["support_constraint"] == "none_freejoint"


def test_contact_aware_mpc_budget_and_object_dynamics_audit_are_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="population"):
        ContactAwareMPCConfig(population=49).validate()
    acceleration = reference_accelerations(
        np.asarray([0.0, 0.05, 0.1]),
        np.asarray([[0.0] * 6, [0.1] + [0.0] * 5, [0.2] + [0.0] * 5]),
    )
    assert acceleration[:, 0] == pytest.approx(2.0)

    source = Path("third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml")
    model = mujoco.MjModel.from_xml_path(str(source))
    mesh = tmp_path / "object.obj"
    mesh.write_text(
        "v 0 0 0\nv 0.02 0 0\nv 0 0.02 0\nv 0 0 0.02\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n",
        encoding="utf-8",
    )
    scene = materialize_world_wrist_free_object_scene(source, tmp_path / "audit", object_mesh=mesh)
    backend = WorldWristFingerBackend(
        scene_path=scene,
        reference=_reference(model),
        joint_lower=model.jnt_range[: model.njnt, 0],
        joint_upper=model.jnt_range[: model.njnt, 1],
    )
    support = support_model_audit(backend)
    assert support["classification"] == "UNSUPPORTED_FREE_BODY_ZERO_GRAVITY_NO_DAMPING"
    assert support["dataset_support_provenance"] == "unresolved"
    assert inertial_wrench_demand(backend)["frames_with_nontrivial_demand"] == 0
    candidates = impulse_sensitivity_candidates(
        mass_kg=0.05,
        principal_inertia_kgm2=np.asarray([1e-4, 2e-4, 3e-4]),
        impulse_ns=0.01,
    )
    assert [row["shared_mass_inertia_scale"] for row in candidates] == [0.5, 1.0, 2.0, 5.0]
    assert not any(row["physical_provenance_eligible"] for row in candidates)
