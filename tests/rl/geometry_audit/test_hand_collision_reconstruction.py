from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (
    HAND_COLLISION_BODY_NAMES,
    reconstruct_hand_collision_body_pose,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_reconstruction_uses_captured_wrist_transform() -> None:
    wrist_pose = np.array(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    finger_q = np.zeros((2, 20), dtype=np.float64)

    poses = reconstruct_hand_collision_body_pose(wrist_pose, finger_q, repo_root=REPO_ROOT)

    assert poses.shape == (2, len(HAND_COLLISION_BODY_NAMES), 7)
    assert np.isfinite(poses).all()
    assert np.all(np.linalg.norm(poses[..., 3:7], axis=-1) > 1.0e-8)
    np.testing.assert_allclose(
        poses[1, :, :3] - poses[0, :, :3],
        np.broadcast_to([0.1, -0.2, 0.3], (len(HAND_COLLISION_BODY_NAMES), 3)),
    )


@pytest.mark.parametrize(
    ("wrist_pose", "finger_q", "message"),
    [
        (np.zeros((2, 6)), np.zeros((2, 20)), "wrist_pose"),
        (np.zeros((2, 7)), np.zeros((2, 19)), "finger_q"),
        (
            np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            np.zeros((1, 20)),
            "zero quaternion",
        ),
    ],
)
def test_reconstruction_rejects_invalid_captured_state(
    wrist_pose: np.ndarray, finger_q: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        reconstruct_hand_collision_body_pose(wrist_pose, finger_q, repo_root=REPO_ROOT)
