from __future__ import annotations

import numpy as np

from toporetarget.adapters.datasets import GrabDatasetAdapterV1, get_dataset_adapter_registry
from toporetarget.contracts.canonical import (
    CanonicalHOIv2,
    load_canonical_hoi,
    migrate_v1_to_v2,
    save_canonical_hoi,
)
from toporetarget.contracts.metrics import MetricType, get_metric_registry
from toporetarget.contracts.reference import (
    load_robot_reference,
    migrate_reference_v1_to_v2,
    save_robot_reference,
    validate_reference,
)
from toporetarget.contracts.robot import get_robot_plugin_registry
from toporetarget.data.storage import load_hoi_sequence, save_hoi_sequence
from toporetarget.data.synthetic import make_synthetic_sequence


def test_canonical_v1_migration_preserves_values_and_old_cache(tmp_path) -> None:
    source = tmp_path / "canonical_v1.zarr"
    original = make_synthetic_sequence(num_frames=4)
    save_hoi_sequence(original, source)
    migrated = migrate_v1_to_v2(source)
    assert isinstance(migrated, CanonicalHOIv2)
    assert migrated.metadata.schema_version == "toporetarget.hoi.v2"
    np.testing.assert_array_equal(migrated.timestamps, original.timestamps)
    assert load_canonical_hoi(source).metadata.schema_version == "toporetarget.hoi.v2"
    assert (source / "canonical_v2.json").exists() is False

    destination = tmp_path / "canonical_v2.zarr"
    save_canonical_hoi(migrated, destination)
    assert load_canonical_hoi(destination).metadata.schema_version == "toporetarget.hoi.v2"
    # The compatibility writer must not turn a v1 cache into an in-place v2 cache.
    assert load_hoi_sequence(source).metadata.schema_version == "toporetarget.hoi.v1"


def test_dataset_registry_exposes_required_grab_surface() -> None:
    registry = get_dataset_adapter_registry()
    assert registry.names() == ("contactpose", "dexycb", "grab", "hocap", "oakink")
    adapter = registry.create("grab")
    assert isinstance(adapter, GrabDatasetAdapterV1)
    assert adapter.descriptor.capabilities.canonical_hoi
    assert adapter.descriptor.capabilities.bimanual
    for method in (
        "discover",
        "index",
        "describe",
        "load_sequence",
        "convert_to_canonical",
        "validate",
        "visualize",
    ):
        assert callable(getattr(adapter, method))


def test_robot_plugin_registry_has_artimano_and_wuji_without_special_case() -> None:
    registry = get_robot_plugin_registry()
    names = registry.names()
    assert "artimano_rh" in names
    assert "wuji_hand2_beta1_rh" in names
    plugin = registry.get("wuji_hand2_beta1_rh")
    assert plugin.kinematics.actuated_joint_order == plugin.spec.dof_order
    assert plugin.reference_export_profile.schema_version == "toporetarget.robot_reference.v2"
    assert plugin.capabilities.rl_ready is False


def _v1_reference(tmp_path):
    t, d, links = 3, 2, 2
    base = np.repeat(np.eye(4)[None], t, axis=0)
    base[:, 0, 3] = np.arange(t) * 0.01
    object_scene = base.copy()
    object_scene[:, 1, 3] = 0.2
    link_poses = np.repeat(np.eye(4)[None, None], t * links, axis=0).reshape(t, links, 4, 4)
    link_poses[:, 0, 0, 3] = 0.1
    link_poses[:, 1, 1, 3] = 0.2
    path = tmp_path / "robot_reference_v1.npz"
    np.savez_compressed(
        path,
        timestamps=np.arange(t, dtype=np.float64) / 10.0,
        frame_indices=np.arange(t),
        qpos=np.arange(t * d, dtype=np.float64).reshape(t, d),
        base_pose_scene=base,
        object_pose_scene=object_scene,
        robot_link_poses_scene=link_poses,
        metadata=np.asarray(
            '{"schema_version":"toporetarget.robot_reference.v1",'
            '"native_fps":10.0,"robot":"synthetic",'
            '"source_sequence":"s1/demo","source_hash":"source"}'
        ),
    )
    return path, base, object_scene, link_poses


def test_robot_reference_v1_migration_and_npz_roundtrip(tmp_path) -> None:
    source, base, object_scene, link_poses = _v1_reference(tmp_path)
    destination = tmp_path / "robot_reference_v2.npz"
    migrated = migrate_reference_v1_to_v2(
        source,
        destination,
        robot_hash="synthetic-hash",
        joint_order=("j0", "j1"),
    )
    assert validate_reference(migrated)["valid"]
    assert migrated.schema_version == "toporetarget.robot_reference.v2"
    np.testing.assert_array_equal(migrated.qpos_reference, np.arange(6).reshape(3, 2))
    np.testing.assert_array_equal(migrated.base_pose, base)
    np.testing.assert_allclose(
        migrated.object_pose_base,
        np.matmul(np.linalg.inv(base), object_scene),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        migrated.tracked_link_positions,
        link_poses[..., :3, 3] - base[:, None, :3, 3],
        atol=1e-12,
    )
    loaded = load_robot_reference(destination)
    np.testing.assert_array_equal(loaded.qpos_reference, migrated.qpos_reference)

    zarr_destination = tmp_path / "robot_reference_v2.zarr"
    save_robot_reference(migrated, zarr_destination)
    zarr_loaded = load_robot_reference(zarr_destination)
    np.testing.assert_array_equal(zarr_loaded.object_pose_base, migrated.object_pose_base)


def test_metric_registry_declares_types_and_rejects_proxy_ground_truth() -> None:
    registry = get_metric_registry()
    report = registry.validate()
    assert report["valid"]
    assert registry.get("grab_contact_precision_proxy").metric_type == MetricType.DATASET_PROXY
    assert registry.get("contact_precision_eq10").metric_type == MetricType.PAPER_EXACT
    assert registry.get("solve_time_ms_per_unit").metric_type == MetricType.ENGINEERING_DIAGNOSTIC
