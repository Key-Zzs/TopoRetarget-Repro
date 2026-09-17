from __future__ import annotations

import json

import numpy as np

from toporetarget.rl.static_retarget_reference import (
    STATIC_REFERENCE_FRAMES,
    build_static_retarget_reference_arrays,
)
from toporetarget.rl.tracked_links import TRACKED_LINKS_WUJI_RH


class _Robot:
    dof_names = tuple(f"joint_{index}" for index in range(20))

    @staticmethod
    def forward_kinematics_reference(qpos: np.ndarray) -> dict[str, np.ndarray]:
        assert qpos.shape == (20,)
        result = {}
        for index, name in enumerate(TRACKED_LINKS_WUJI_RH):
            pose = np.eye(4)
            pose[:3, 3] = (index * 0.001, 0.0, 0.0)
            result[name] = pose
        return result


def test_static_reference_repeats_exact_frozen_state_with_zero_velocity() -> None:
    base = np.eye(4)
    base[:3, 3] = (0.1, 0.2, 0.3)
    obj = np.eye(4)
    obj[:3, 3] = (0.4, 0.5, 0.6)
    q = np.linspace(-0.2, 0.2, 20)
    arrays = build_static_retarget_reference_arrays(
        qpos=q,
        base_pose_world=base,
        object_pose_world=obj,
        robot_model=_Robot(),
        source_frame_id=3567,
        anchor_id="A01",
    )
    assert arrays["q_finger_ref"].shape == (STATIC_REFERENCE_FRAMES, 20)
    assert np.allclose(arrays["q_finger_ref"], q)
    assert np.all(arrays["qdot_finger_ref"] == 0.0)
    assert np.all(arrays["wrist_twist_world_ref"] == 0.0)
    assert np.all(arrays["object_twist_world_ref"] == 0.0)
    assert np.allclose(arrays["T_world_wrist_ref"], base)
    assert np.allclose(arrays["T_world_object_ref"], obj)
    metadata = json.loads(str(arrays["metadata"].item()))
    assert metadata["q_old_used"] is False
    assert metadata["static_reference_index"].startswith("single_constant_state")
    assert metadata["reference_kinematics_version"] == 2
    assert metadata["time_scale"] == 8
