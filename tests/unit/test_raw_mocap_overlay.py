from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from toporetarget.rl.geometry_audit.raw_mocap_overlay import (
    decimate_visual_mesh,
    interpolate_mano_pca_pose,
    interpolate_object_pose,
    pose_wxyz_to_matrix,
)


def test_visual_mesh_decimation_is_bounded_and_preserves_source_arrays() -> None:
    vertices = np.array([[x, y, 0.0] for y in range(5) for x in range(5)], dtype=np.float64)
    faces = np.array(
        [[y * 5 + x, y * 5 + x + 1, (y + 1) * 5 + x] for y in range(4) for x in range(4)]
        + [
            [y * 5 + x + 1, (y + 1) * 5 + x + 1, (y + 1) * 5 + x]
            for y in range(4)
            for x in range(4)
        ],
        dtype=np.int64,
    )
    vertices_before = vertices.copy()
    faces_before = faces.copy()

    display_vertices, display_faces = decimate_visual_mesh(vertices, faces, max_faces=8)

    assert len(display_faces) <= 8
    assert len(display_faces) > 0
    assert np.isfinite(display_vertices).all()
    assert (display_faces >= 0).all()
    assert display_faces.max() < len(display_vertices)
    assert np.all(display_faces[:, 0] != display_faces[:, 1])
    assert np.all(display_faces[:, 1] != display_faces[:, 2])
    assert np.array_equal(vertices, vertices_before)
    assert np.array_equal(faces, faces_before)


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
