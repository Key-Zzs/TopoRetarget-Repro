from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from toporetarget.rl.environments.isaaclab_backend.reference_bank import WorldWristReferenceBank


def _reference(path: Path, *, frames: int = 41) -> None:
    quaternion = np.zeros((frames, 4), dtype=np.float32)
    quaternion[:, 0] = 1.0
    np.savez_compressed(
        path,
        timestamps=np.arange(frames, dtype=np.float32) / 20.0,
        wrist_pose_translation_world_ref=np.zeros((frames, 3), dtype=np.float32),
        wrist_pose_quaternion_world_ref_wxyz=quaternion,
        wrist_twist_world_ref=np.zeros((frames, 6), dtype=np.float32),
        q_finger_ref=np.zeros((frames, 20), dtype=np.float32),
        qdot_finger_ref=np.zeros((frames, 20), dtype=np.float32),
        object_pose_translation_world_ref=np.zeros((frames, 3), dtype=np.float32),
        object_pose_quaternion_world_ref_wxyz=quaternion,
        object_twist_world_ref=np.zeros((frames, 6), dtype=np.float32),
        object_axis_points_world_ref=np.zeros((frames, 6, 3), dtype=np.float32),
        tracked_link_positions_world_ref=np.zeros((frames, 16, 3), dtype=np.float32),
        object_axis_points_wrist_ref=np.zeros((frames, 6, 3), dtype=np.float32),
        tracked_link_positions_wrist_ref=np.zeros((frames, 16, 3), dtype=np.float32),
        metadata=np.asarray(
            json.dumps(
                {
                    "joint_order": [f"joint_{index}" for index in range(20)],
                    "tracked_link_names": [f"link_{index}" for index in range(16)],
                }
            )
        ),
    )


def test_single_clip_bank_preserves_independent_runtime_domain(tmp_path: Path) -> None:
    source = tmp_path / "hocap_111118.npz"
    _reference(source)

    bank = WorldWristReferenceBank({"hocap_111118": source}, device="cpu")

    assert bank.clip_ids == ("hocap_111118",)
    assert bank.q_finger_ref.shape == (1, 41, 20)
    assert bank.valid_mask.shape == (1, 41)
    assert bank.assignment(3, balanced=True).tolist() == [0, 0, 0]
    assert bank.assignment(2, balanced=False, fixed_clip="hocap_111118").tolist() == [0, 0]

    bank.apply_uniform_time_scale(8)
    assert bank.frame_count == 321
    assert bank.valid_mask.shape == (1, 321)


def test_empty_bank_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one clip"):
        WorldWristReferenceBank({}, device="cpu")


def test_single_clip_bank_accepts_manifest_bound_variable_length(tmp_path: Path) -> None:
    source = tmp_path / "hocap_111118.npz"
    _reference(source, frames=57)

    bank = WorldWristReferenceBank({"hocap_111118": source}, device="cpu")

    assert bank.frame_count == 57
    assert bank.manifest.source_frame_count == 57
    assert bank.valid_mask.shape == (1, 57)
    bank.apply_uniform_time_scale(8)
    assert bank.frame_count == 449
    assert bank.valid_mask.shape == (1, 449)
