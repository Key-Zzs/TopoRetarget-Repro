from __future__ import annotations

import numpy as np
import pytest

from toporetarget.physics.support import (
    SupportResolutionMode,
    SupportResolutionStatus,
    SupportType,
    compare_support_counterfactuals,
    detect_stable_pre_contact_interval,
    infer_planar_support,
    resolve_support,
    summarize_static_support_test,
    validate_and_finalize_resolution,
    write_finite_planar_support_usda,
)
from toporetarget.physics.support.types import FinitePlanarSupportProxy, SupportInterval


def _cube_trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [[x, y, z] for x in (-0.05, 0.05) for y in (-0.04, 0.04) for z in (-0.01, 0.01)],
        dtype=np.float64,
    )
    timestamps = np.arange(12, dtype=np.float64) * 0.05
    translation = np.zeros((12, 3), dtype=np.float64)
    translation[:, :2] = [0.1, -0.2]
    translation[8:, 2] = np.arange(4, dtype=np.float64) * 0.02
    quaternion = np.zeros((12, 4), dtype=np.float64)
    quaternion[:, 0] = 1.0
    return vertices, timestamps, translation, quaternion


def test_explicit_source_support_wins_over_inference() -> None:
    vertices, timestamps, translation, quaternion = _cube_trajectory()
    result = resolve_support(
        dataset="fake",
        sequence="explicit",
        object_visual_vertices_local=vertices,
        object_collision_vertices_local=vertices,
        object_pose_translation_world=translation,
        object_pose_quaternion_world_wxyz=quaternion,
        timestamps=timestamps,
        gravity_world_mps2=(0.0, 0.0, -9.81),
        source_support={
            "explicit": True,
            "explicit_validated": True,
            "validation": {"explicit": True},
            "explicit_metadata": {"table_id": "source-table-7"},
        },
    )
    assert result.support_type is SupportType.SOURCE_EXPLICIT_SUPPORT
    assert result.status == SupportResolutionStatus.SOURCE_SUPPORT_VALIDATED.value
    assert result.support_inferred is False


def test_recovered_source_support_is_distinct() -> None:
    vertices, timestamps, translation, quaternion = _cube_trajectory()
    result = resolve_support(
        dataset="fake",
        sequence="recovered",
        object_visual_vertices_local=vertices,
        object_pose_translation_world=translation,
        object_pose_quaternion_world_wxyz=quaternion,
        timestamps=timestamps,
        gravity_world_mps2=(0.0, 0.0, -9.81),
        source_support={
            "recovered": True,
            "recovered_validated": True,
            "recovered_assets": [{"path": "/data/scene/table.usd"}],
            "validation": {"recovered": True},
        },
    )
    assert result.support_type is SupportType.SOURCE_RECOVERED_SUPPORT
    assert result.source_recovered is True
    assert result.source_explicit is False


def test_source_rejection_does_not_silently_fallback_in_source_only_mode() -> None:
    vertices, timestamps, translation, quaternion = _cube_trajectory()
    result = resolve_support(
        dataset="fake",
        sequence="rejected",
        object_visual_vertices_local=vertices,
        object_pose_translation_world=translation,
        object_pose_quaternion_world_wxyz=quaternion,
        timestamps=timestamps,
        gravity_world_mps2=(0.0, 0.0, -9.81),
        source_support={"explicit": True, "validation": {"explicit": False}},
        mode=SupportResolutionMode.SOURCE_ONLY,
    )
    assert result.support_type is SupportType.UNKNOWN
    assert result.support_inferred is False
    assert "SOURCE_ONLY" in result.diagnostics["reason"]


def test_auto_mode_falls_back_to_inferred_planar_support_with_provenance() -> None:
    vertices, timestamps, translation, quaternion = _cube_trajectory()
    result = resolve_support(
        dataset="fake",
        sequence="inferred",
        object_visual_vertices_local=vertices,
        object_collision_vertices_local=vertices[::2],
        object_pose_translation_world=translation,
        object_pose_quaternion_world_wxyz=quaternion,
        timestamps=timestamps,
        gravity_world_mps2=(0.0, 0.0, -9.81),
        source_support={"recovered": True, "validation": {"recovered": False}},
    )
    assert result.support_type is SupportType.INFERRED_PLANAR_SUPPORT
    assert result.support_inferred is True
    assert result.source_explicit is False
    assert result.source_recovered is False
    assert result.plane_normal == pytest.approx((0.0, 0.0, 1.0))
    assert result.table_proxy is not None
    assert result.status == SupportResolutionStatus.SUPPORT_RECONSTRUCTION_BLOCKED.value


def test_no_stable_interval_blocks_inference() -> None:
    vertices, timestamps, translation, quaternion = _cube_trajectory()
    translation[:, 0] = np.arange(len(timestamps), dtype=np.float64) * 0.02
    result = detect_stable_pre_contact_interval(
        timestamps=timestamps,
        object_translation_world=translation,
        object_quaternion_world_wxyz=quaternion,
        gravity=(0.0, 0.0, -9.81),
    )
    assert result.interval is None
    assert result.status == "PLANAR_SUPPORT_INFERENCE_NOT_AUTHORIZED"


def test_gravity_axis_independence_uses_upward_normal() -> None:
    vertices = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=float)
    translation = np.zeros((8, 3), dtype=float)
    quaternion = np.zeros((8, 4), dtype=float)
    quaternion[:, 0] = 1.0
    interval = SupportInterval(0, 8, "test")
    fit, _, _ = infer_planar_support(
        visual_vertices_local=vertices,
        collision_vertices_local=vertices,
        object_translation_world=translation,
        object_quaternion_world_wxyz=quaternion,
        gravity=(0.0, -9.81, 0.0),
        stable_interval=interval,
    )
    assert fit.plane_normal == pytest.approx((0.0, 1.0, 0.0))


def test_visual_collision_discrepancy_is_reported() -> None:
    vertices, timestamps, translation, quaternion = _cube_trajectory()
    collision = vertices.copy()
    collision[:, 2] += 0.003
    fit, _, _ = infer_planar_support(
        visual_vertices_local=vertices,
        collision_vertices_local=collision,
        object_translation_world=translation,
        object_quaternion_world_wxyz=quaternion,
        gravity=(0.0, 0.0, -9.81),
        stable_interval=SupportInterval(0, 8, "test"),
    )
    assert fit.delta_support_geometry == pytest.approx(-0.003)


def test_static_support_summary_and_causal_ab() -> None:
    records = [
        {
            "position_world_m": [0.0, 0.0, 0.1],
            "linear_velocity_world_mps": [0.0, 0.0, 0.0],
            "angular_velocity_world_radps": [0.0, 0.0, 0.0],
            "orientation_world_wxyz": [1.0, 0.0, 0.0, 0.0],
            "support_contact": True,
            "support_force_world_n": [0.0, 0.0, 0.4905],
        }
    ] * 4
    with_support = summarize_static_support_test(
        records,
        support_active=True,
        mass_kg=0.05,
        gravity_world_mps2=(0.0, 0.0, -9.81),
        support_normal=(0.0, 0.0, 1.0),
    )
    without_support = dict(with_support)
    without_support.update({"status": "FAIL", "position_drift_max_m": 0.2})
    comparison = compare_support_counterfactuals(with_support, without_support)
    assert with_support["status"] == "PASS"
    assert with_support["rotation_drift_source"] == "quaternion_pose_drift_from_first_record"
    assert comparison["causal_support_effect"] is True


def test_finalization_requires_geometry_and_physics() -> None:
    vertices, timestamps, translation, quaternion = _cube_trajectory()
    result = resolve_support(
        dataset="fake",
        sequence="pending",
        object_visual_vertices_local=vertices,
        object_pose_translation_world=translation,
        object_pose_quaternion_world_wxyz=quaternion,
        timestamps=timestamps,
        gravity_world_mps2=(0.0, 0.0, -9.81),
    )
    finalized = validate_and_finalize_resolution(
        result,
        geometry={"status": "PASS"},
        physics={"status": "FAIL"},
    )
    assert finalized.status == SupportResolutionStatus.SUPPORT_RECONSTRUCTION_BLOCKED.value


def test_runtime_proxy_is_local_frame_and_has_no_force_path(tmp_path) -> None:
    proxy = FinitePlanarSupportProxy(
        table_pose=(0.3, -0.2, 0.1, 1.0, 0.0, 0.0, 0.0),
        table_extent=(0.4, 0.5),
        table_thickness=0.02,
        plane_normal=(0.0, 0.0, 1.0),
        plane_offset=0.1,
    )
    asset = write_finite_planar_support_usda(proxy, tmp_path / "support.usda")
    text = asset.read_text(encoding="utf-8")
    assert "double3 xformOp:translate = (0.0, 0.0, 0.0)" in text
    assert "quatf xformOp:orient = (1.0, 0.0, 0.0, 0.0)" in text
    assert "support_force_injection" not in text
    assert "kinematicEnabled = 1" in text
