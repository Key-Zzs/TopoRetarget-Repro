from __future__ import annotations

import numpy as np
import pytest

from toporetarget.rl.stage12_reference import (
    Stage12ReferenceError,
    build_robot_reference_from_stage12_final,
)
from toporetarget.robots import get_robot_registry


def _accepted_final_arrays() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    frames = 3
    dof_count = 20
    links = 2
    timestamps = np.asarray([0.0, 0.05, 0.10])
    base_pose = np.broadcast_to(np.eye(4), (frames, 4, 4)).copy()
    base_pose[:, 0, 3] = 1.0
    link_poses = np.broadcast_to(np.eye(4), (frames, links, 4, 4)).copy()
    link_poses[:, 0, 0, 3] = 1.1
    link_poses[:, 1, 1, 3] = 0.2
    arrays = {
        "qpos": np.zeros((frames, dof_count)),
        "base_pose_scene": base_pose,
        "robot_link_poses": link_poses,
        "timestamps": timestamps,
        "source_frame_indices": np.asarray([0, 1, 2]),
        "final_accepted": np.ones(frames, dtype=bool),
        "trajectory_continuous": np.ones(frames, dtype=bool),
        "valid_mask": np.ones(frames, dtype=bool),
    }
    metadata: dict[str, object] = {
        "robot_name": "wuji_hand2_beta1_rh",
        "robot_dof_count": dof_count,
        "robot_spec_hash": "synthetic-wuji-spec-hash",
        "robot_link_names": ["r_wrist", "r_thumb_distal"],
        "source_sequence_id": "synthetic_hocap",
        "acceptance_policy_id": "strict_optimizer_converged_audits_and_continuity_v1",
    }
    return arrays, metadata


def _object_poses() -> np.ndarray:
    poses = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    poses[:, 0, 3] = np.asarray([1.2, 1.3, 1.4])
    return poses


def test_build_stage12_reference_converts_scene_poses_to_wuji_base_frame() -> None:
    arrays, metadata = _accepted_final_arrays()
    timestamps = np.asarray([0.0, 0.05, 0.10])

    reference = build_robot_reference_from_stage12_final(
        final_arrays=arrays,
        object_pose_scene=_object_poses(),
        canonical_timestamps=timestamps,
        final_metadata=metadata,
        final_artifact="/local/final.zarr",
        canonical_artifact="/local/canonical.zarr",
        manifest_artifact="/local/manifest.json",
        manifest_sha256="f" * 64,
    )

    assert reference.validate()["valid"]
    assert reference.joint_order == get_robot_registry().get_spec("wuji_hand2_beta1_rh").dof_order
    assert np.allclose(reference.object_pose_base[:, 0, 3], [0.2, 0.3, 0.4])
    assert np.allclose(reference.tracked_link_positions[:, 0, 0], 0.1)
    assert reference.dataset_provenance["kind"] == "accepted_stage12_hocap_final"


def test_build_stage12_reference_rejects_an_unaccepted_final_frame() -> None:
    arrays, metadata = _accepted_final_arrays()
    arrays["final_accepted"][-1] = False

    with pytest.raises(Stage12ReferenceError, match="unaccepted frames"):
        build_robot_reference_from_stage12_final(
            final_arrays=arrays,
            object_pose_scene=_object_poses(),
            canonical_timestamps=np.asarray([0.0, 0.05, 0.10]),
            final_metadata=metadata,
            final_artifact="/local/final.zarr",
            canonical_artifact="/local/canonical.zarr",
            manifest_artifact="/local/manifest.json",
            manifest_sha256="f" * 64,
        )
