from __future__ import annotations

import numpy as np

from toporetarget.rl.physics_retargeting.geometry_audit import (
    convex_proxy_point_metrics,
    inverse_transform_points,
    transform_points,
)


def test_pose_transform_roundtrip() -> None:
    points = np.asarray([[0.0, 1.0, 2.0], [0.2, -0.1, 0.3]])
    pose = np.asarray([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        inverse_transform_points(transform_points(points, pose), pose), points
    )


def test_convex_proxy_inside_and_outside() -> None:
    cube = np.asarray([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
    penetration, gap = convex_proxy_point_metrics(cube, np.asarray([[0.0, 0.0, 0.0]]))
    assert penetration == 1.0
    assert gap == 0.0
    penetration, gap = convex_proxy_point_metrics(cube, np.asarray([[2.0, 0.0, 0.0]]))
    assert penetration == 0.0
    assert gap == 1.0
