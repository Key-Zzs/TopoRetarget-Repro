import numpy as np
import pytest

from toporetarget.geometry.se3 import (
    compose_transform,
    invert_transform,
    object_to_scene,
    pose_rotation_error,
    relative_transform,
    scene_to_object,
    scene_to_wrist,
    transform_points,
    validate_transform,
    wrist_to_scene,
)


def _z_rotation(angle: float) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    result[:3, 3] = [0.1, -0.2, 0.3]
    return result


def test_se3_inverse_compose_and_scene_wrist_object_round_trips() -> None:
    wrist = _z_rotation(0.3)
    obj = _z_rotation(-0.2)
    validate_transform(wrist)
    validate_transform(obj)
    assert np.allclose(compose_transform(wrist, invert_transform(wrist)), np.eye(4))
    points = np.array([[0.1, 0.2, 0.3], [-0.2, 0.4, 0.0]])
    np.testing.assert_allclose(wrist_to_scene(wrist, scene_to_wrist(wrist, points)), points)
    np.testing.assert_allclose(object_to_scene(obj, scene_to_object(obj, points)), points)
    np.testing.assert_allclose(relative_transform(wrist, obj), invert_transform(wrist) @ obj)


def test_transform_points_uses_column_vector_convention() -> None:
    transform = np.eye(4)
    transform[0, 3] = 1.0
    np.testing.assert_allclose(transform_points(transform, np.zeros((1, 3))), [[1.0, 0.0, 0.0]])


def test_rotation_error_is_radians_and_det_is_positive() -> None:
    first = _z_rotation(0.0)
    second = _z_rotation(np.deg2rad(1.0))
    assert float(pose_rotation_error(first, second)) == pytest.approx(np.deg2rad(1.0), abs=1e-8)
    assert np.linalg.det(first[:3, :3]) == pytest.approx(1.0)


def test_reflection_and_bad_last_row_fail() -> None:
    reflected = np.eye(4)
    reflected[2, 2] = -1.0
    with pytest.raises(ValueError, match="determinant"):
        validate_transform(reflected)
    bad = np.eye(4)
    bad[3, 3] = 2.0
    with pytest.raises(ValueError, match="last row"):
        validate_transform(bad)
