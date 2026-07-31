from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from toporetarget.rl.axis_points import object_axis_points_from_poses
from toporetarget.rl.contracts import Stage16ReferenceClip
from toporetarget.rl.environments.mujoco_backend import (
    MujocoReferenceTrackingBackend,
    materialize_free_object_scene,
)
from toporetarget.rl.observations import ObservationContract
from toporetarget.rl.randomization import DomainRandomizationConfig
from toporetarget.rl.termination import TerminationType


def test_mujoco_backend_compiles_resets_steps_and_builds_observation(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    source = Path("third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml")
    model = mujoco.MjModel.from_xml_path(str(source))
    joint_order = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)
    )
    joint_bounds = model.jnt_range[: model.njnt].copy()
    timestamps = np.asarray([0.0, 0.05, 0.10])
    poses = np.broadcast_to(np.eye(4), (timestamps.size, 4, 4)).copy()
    poses[:, 2, 3] = 0.15
    q_ref = np.broadcast_to(joint_bounds.mean(axis=1), (timestamps.size, model.njnt)).copy()
    clip = Stage16ReferenceClip(
        timestamps=timestamps,
        q_finger_ref=q_ref,
        object_pose_base_ref=poses,
        object_axis_points_base_ref=object_axis_points_from_poses(poses),
        tracked_link_positions_base_ref=np.zeros((timestamps.size, 2, 3)),
        joint_order=joint_order,
        tracked_link_names=("r_wrist", "r_thumb_distal"),
        provenance={"dataset": "synthetic_backend_qualification"},
    )
    scene = materialize_free_object_scene(source, tmp_path)
    backend = MujocoReferenceTrackingBackend(
        scene_path=scene,
        reference=clip,
        joint_lower=joint_bounds[:, 0],
        joint_upper=joint_bounds[:, 1],
        seed=3,
    )
    state = backend.reset()
    observation = backend.observation(state)
    next_state, reward, terminal = backend.transition(np.zeros(model.njnt))
    assert scene.is_file()
    assert state["q"].shape == (model.njnt,)
    assert observation.shape == (ObservationContract(model.njnt, 2).dimension,)
    assert next_state["object_pose"].shape == (4, 4)
    assert np.isfinite(list(reward.values())).all()
    assert terminal in {None, TerminationType.SUCCESS_REFERENCE_COMPLETE.value}
    assert "pd" in backend.capabilities.supported_randomizations
    backend.apply_randomization(DomainRandomizationConfig(enabled=False))
    first_state = backend.reset(reference_index=0)
    assert np.allclose(first_state["object_pose"], clip.object_pose_base_ref[0])


def test_mujoco_backend_references_a_per_object_mesh_without_copying(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    source = Path("third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml")
    object_mesh = tmp_path / "object.obj"
    object_mesh.write_text(
        "v 0 0 0\nv 0.02 0 0\nv 0 0.02 0\nv 0 0 0.02\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n",
        encoding="utf-8",
    )
    scene = materialize_free_object_scene(
        source, tmp_path / "scene", object_mesh=object_mesh, include_ground=False
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    assert model.nmesh >= 1
    assert str(object_mesh.resolve()) in scene.read_text(encoding="utf-8")
    assert "stage16_ground" not in scene.read_text(encoding="utf-8")
