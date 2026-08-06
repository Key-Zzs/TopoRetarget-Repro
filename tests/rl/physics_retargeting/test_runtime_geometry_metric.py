from __future__ import annotations

import numpy as np

from toporetarget.rl.geometry_audit.contracts import (
    GEOMETRY_METRIC_CONTRACT,
    GEOMETRY_QUERY_CONTRACT,
)
from toporetarget.rl.geometry_audit.metrics import (
    aggregate_penetration,
    qualify_source_corrected,
)
from toporetarget.rl.geometry_audit.runtime_geometry import ConvexProxyGeometry
from toporetarget.rl.geometry_audit.transforms import (
    compose_poses,
    quaternion_matrix_wxyz,
    transform_points,
)


def test_transform_and_quaternion_sign_invariance() -> None:
    angle = 0.7
    quaternion = np.asarray([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
    np.testing.assert_allclose(
        quaternion_matrix_wxyz(quaternion), quaternion_matrix_wxyz(-quaternion), atol=1.0e-14
    )
    first = np.asarray([1.0, 2.0, 3.0, *quaternion])
    second = np.asarray([0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    composed = compose_poses(first, second)
    np.testing.assert_allclose(
        transform_points(np.asarray([[0.0, 0.0, 0.0]]), composed)[0],
        transform_points(np.asarray([[0.5, 0.0, 0.0]]), first)[0],
    )


def test_per_frame_worst_prevents_pair_dilution() -> None:
    values = np.zeros((20, 2), dtype=np.float64)
    values[-2:, :] = np.asarray([[0.004, 0.002], [0.012, 0.005]])
    pair = np.zeros_like(values, dtype=np.int64)
    aggregate = aggregate_penetration(values, pair, ("hand<->object",))
    assert aggregate["max_penetration_m"] == 0.012
    assert aggregate["p95_penetration_m"] > 0.010
    assert aggregate["active_p95_penetration_m"] > 0.010
    assert aggregate["all_frame_p95_penetration_m"] < 0.005
    assert aggregate["over_3mm_frame_replica_count"] == 3
    assert aggregate["over_10mm_frame_replica_count"] == 1
    assert aggregate["worst_frame"] == 19
    assert aggregate["worst_replica"] == 0


def test_relative_gate_has_frozen_zero_handling() -> None:
    source = {"max_penetration_m": 0.0, "p95_penetration_m": 0.0}
    within = {
        "max_penetration_m": GEOMETRY_QUERY_CONTRACT.metric_epsilon_m,
        "p95_penetration_m": GEOMETRY_QUERY_CONTRACT.metric_epsilon_m,
    }
    outside = {
        "max_penetration_m": 2.0 * GEOMETRY_QUERY_CONTRACT.metric_epsilon_m,
        "p95_penetration_m": 0.0,
    }
    assert qualify_source_corrected(source, within)["formal_pass"]
    assert not qualify_source_corrected(source, outside)["formal_pass"]


def test_visual_geometry_is_never_formal_authority() -> None:
    assert GEOMETRY_METRIC_CONTRACT.visual_geometry_role == "unsigned diagnostic only"
    assert GEOMETRY_METRIC_CONTRACT.geometry_authority == "C.1 authored runtime collision proxies"


def test_runtime_proxy_manifest_row_fails_closed_on_geometry_hash_drift() -> None:
    proxy = ConvexProxyGeometry(
        shape_id="tetrahedron",
        body_name="test",
        geometry_type="convex_hull",
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        faces=np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
        local_pose_xyz_wxyz=np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        scale_xyz=np.ones(3),
        source_asset_path="source.usd",
        source_asset_sha256="source-hash",
        generated_asset_path="runtime.usd",
        generated_asset_sha256="runtime-hash",
    )
    row = proxy.as_dict()
    assert ConvexProxyGeometry.from_dict(row).shape_id == "tetrahedron"
    row["convex_vertices_m"][0][0] = 0.25
    with np.testing.assert_raises_regex(ValueError, "geometry hash mismatch"):
        ConvexProxyGeometry.from_dict(row)
