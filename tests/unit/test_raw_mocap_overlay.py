from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from toporetarget.rl.geometry_audit.raw_mocap_overlay import (
    interpolate_mano_pca_pose,
    interpolate_object_pose,
    pose_wxyz_to_matrix,
)


def test_object_interpolation_uses_shortest_arc_slerp() -> None:
    timestamps = np.array([0.0, 1.0])
    poses = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 2.0, 0.0, 0.0],
        ]
    )
    result = interpolate_object_pose(timestamps, poses, np.array([0.5]))
    assert np.allclose(result[0, :3, 3], [1.0, 0.0, 0.0])
    assert np.allclose(result[0, :3, :3], np.eye(3), atol=1.0e-12)


def test_mano_global_orientation_uses_so3_interpolation() -> None:
    timestamps = np.array([0.0, 1.0])
    pose = np.zeros((2, 51))
    pose[1, :3] = [0.0, 0.0, np.pi]
    pose[:, 3] = [2.0, 4.0]
    result = interpolate_mano_pca_pose(timestamps, pose, np.array([0.5]))
    assert np.allclose(
        Rotation.from_rotvec(result[0, :3]).as_matrix(),
        Rotation.from_euler("z", 90, degrees=True).as_matrix(),
    )
    assert result[0, 3] == 3.0


def test_wxyz_pose_round_trip_preserves_pose() -> None:
    pose = np.array(
        [
            [
                0.3,
                -0.2,
                0.7,
                *Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_quat()[[3, 0, 1, 2]],
            ]
        ]
    )
    matrix = pose_wxyz_to_matrix(pose)
    assert np.allclose(matrix[0, :3, 3], pose[0, :3])
    assert np.allclose(
        matrix[0, :3, :3], Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    )
