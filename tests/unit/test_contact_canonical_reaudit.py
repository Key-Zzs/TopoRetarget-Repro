from __future__ import annotations

import numpy as np

from toporetarget.geometry.signed_distance.closest_point import (
    TriangleAABBTree,
    TriangleCentroidBoundTree,
    closest_points_on_triangles,
)
from toporetarget.workflows.contact_canonical_reaudit import (
    CANONICAL_BACKEND_ID,
    CANONICAL_PROFILE_ID,
    LEGACY_BACKEND_ID,
    SHADOW_PROFILES,
    _base_from_delta,
    _metric_fields,
    _readiness,
    _source_classification,
)


def _source_frames(count: int = 4) -> list[dict[str, object]]:
    regions = {
        finger: {"5mm": {"near_surface_count": 2}}
        for finger in ("thumb", "index", "middle", "ring", "pinky")
    }
    return [
        {
            "frame": frame,
            "visual_min_distance_m": -0.001,
            "thresholds": {
                "3mm": {"near_surface_count": 2, "near_surface_ratio": 0.2},
                "5mm": {"near_surface_count": 4, "near_surface_ratio": 0.4},
            },
            "regions": regions,
        }
        for frame in range(count)
    ]


def test_metric_contract_keeps_raw_tau_and_hard_definitions_separate() -> None:
    values = _metric_fields(np.asarray([-0.004, 0.002]), tau=0.001, bound=0.003)
    assert values["raw_signed_distance_m"] == -0.004
    assert values["raw_penetration_m"] == 0.004
    assert values["penetration_beyond_tau_m"] == 0.003
    assert values["hard_bound_violation_m"] == 0.001
    assert values["soft_residual_before_slack_m"] == -0.003
    assert values["hard_residual_m"] == -0.001


def test_source_classification_is_explicitly_proxy_only() -> None:
    report = _source_classification({"frames": _source_frames()}, {})
    assert report["classification"] == "CONTACT_RICH"
    assert report["contact_proxy_name"] == "source_contact_proxy"
    assert report["ground_truth_contact"] is False
    assert report["canonical_backend_id"] == CANONICAL_BACKEND_ID


def test_readiness_fails_closed_before_shadow() -> None:
    summary = {"canonical_reaudit_gate_pass": False}
    source = {"classification": "CONTACT_RICH"}
    representation = {"overall_classification": "REPRESENTATION_MATCH"}
    report = _readiness(summary, source, representation)
    assert report["status"] == "RETURN_TO_STAGE9_2_ACCEPTANCE_OR_VALIDATION_FIX"
    assert report["enter_stage9_4"] is False


def test_shadow_contract_is_diagnostic_and_backend_separate() -> None:
    assert CANONICAL_PROFILE_ID != LEGACY_BACKEND_ID
    assert len(SHADOW_PROFILES) == 6


def test_exact_centroid_bound_closest_point_matches_aabb() -> None:
    triangles = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        ]
    )
    points = np.asarray([[0.2, 0.2, 0.1], [0.8, 0.1, 0.9], [3.0, 2.0, 1.0]])
    aabb = closest_points_on_triangles(points, triangles, tree=TriangleAABBTree(triangles))
    centroid = closest_points_on_triangles(
        points, triangles, tree=TriangleCentroidBoundTree(triangles)
    )
    np.testing.assert_allclose(aabb[0], centroid[0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(aabb[3], centroid[3], atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(aabb[1], centroid[1])


def test_shadow_rotation_delta_uses_so3_exp_equivalent() -> None:
    seed = np.eye(4)
    result = _base_from_delta(seed, np.asarray([0.1, -0.2, 0.3, 0.0, 0.0, np.pi / 2]))
    np.testing.assert_allclose(result[:3, 3], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(result[:3, :3] @ result[:3, :3].T, np.eye(3), atol=1e-12)
