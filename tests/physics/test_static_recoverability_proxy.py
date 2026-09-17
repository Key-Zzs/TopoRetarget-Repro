from __future__ import annotations

import numpy as np

from toporetarget.physics.support import (
    SupportType,
    build_static_recoverability_support_proxy,
    support_collision_policy,
)


def _box() -> np.ndarray:
    return np.asarray(
        [[x, y, z] for x in (-0.02, 0.02) for y in (-0.03, 0.03) for z in (-0.04, 0.04)],
        dtype=np.float64,
    )


def test_static_proxy_is_separate_from_source_support_and_deterministic() -> None:
    pose = np.eye(4)
    pose[:3, 3] = (0.1, -0.2, 0.25)
    first, first_audit = build_static_recoverability_support_proxy(_box(), pose)
    second, second_audit = build_static_recoverability_support_proxy(_box(), pose)
    assert first.as_dict() == second.as_dict()
    assert first_audit == second_audit
    assert first.plane_offset == 0.21
    assert np.allclose(first.table_extent, (0.08, 0.10))
    assert first_audit["support_type"] == "STATIC_RECOVERABILITY_PLANAR_PROXY"
    assert SupportType.STATIC_RECOVERABILITY_PLANAR_PROXY not in {
        SupportType.SOURCE_EXPLICIT_SUPPORT,
        SupportType.SOURCE_RECONSTRUCTED_SUPPORT,
        SupportType.INFERRED_PLANAR_SUPPORT,
    }


def test_static_proxy_uses_existing_pairwise_collision_policy() -> None:
    policy = support_collision_policy(SupportType.STATIC_RECOVERABILITY_PLANAR_PROXY)
    assert policy["object_support_collision"] is True
    assert policy["hand_support_collision"] is False
    assert policy["global_support_collision_disabled"] is False


def test_static_proxy_follows_frozen_object_pose_without_outcome_inputs() -> None:
    pose = np.eye(4)
    pose[:3, 3] = (0.0, 0.0, 1.0)
    proxy, audit = build_static_recoverability_support_proxy(_box(), pose)
    assert np.isclose(proxy.plane_offset, 0.96)
    assert audit["object_pose_changed"] is False
    assert audit["hand_geometry_used"] is False
    assert audit["outcome_fields_used"] == []
