"""Pure reconstruction of Wuji collision-body poses from physical joint state."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from toporetarget.robots.registry import get_robot_registry

from .transforms import pose_from_matrix, pose_matrix

HAND_COLLISION_BODY_NAMES = (
    "r_wrist",
    "r_index_finger_proximal",
    "r_index_finger_proximal_abd",
    "r_index_finger_middle",
    "r_index_finger_distal",
    "r_middle_finger_proximal",
    "r_middle_finger_proximal_abd",
    "r_middle_finger_middle",
    "r_middle_finger_distal",
    "r_pinky_proximal",
    "r_pinky_proximal_abd",
    "r_pinky_middle",
    "r_pinky_distal",
    "r_ring_finger_proximal",
    "r_ring_finger_proximal_abd",
    "r_ring_finger_middle",
    "r_ring_finger_distal",
    "r_thumb_proximal",
    "r_thumb_proximal_abd",
    "r_thumb_middle",
    "r_thumb_distal",
)


def reconstruct_hand_collision_body_pose(
    wrist_pose: np.ndarray,
    finger_q: np.ndarray,
    *,
    repo_root: Path,
) -> np.ndarray:
    """Rebuild `[frames, 21, xyz+wxyz]` poses from captured physical state.

    The finger configuration and wrist pose come from the live PhysX rollout.
    Forward kinematics is intentionally run after the simulator process has
    ended, because reading all collision-body articulation tensors during the
    Isaac Sim callback can terminate the process without a Python exception.
    """

    wrists = np.asarray(wrist_pose, dtype=np.float64)
    fingers = np.asarray(finger_q, dtype=np.float64)
    if wrists.ndim != 2 or wrists.shape[1] != 7:
        raise ValueError(f"wrist_pose must be [frames, 7], got {wrists.shape}")
    if fingers.shape != (wrists.shape[0], 20):
        raise ValueError(f"finger_q must be [{wrists.shape[0]}, 20], got {fingers.shape}")
    if not np.isfinite(wrists).all() or not np.isfinite(fingers).all():
        raise ValueError("captured wrist/finger state must be finite")
    if np.any(np.linalg.norm(wrists[:, 3:7], axis=1) < 1.0e-8):
        raise ValueError("captured wrist_pose contains a zero quaternion")
    model = get_robot_registry(repo_root=repo_root).load("wuji_hand2_beta1_rh")
    result = np.empty((wrists.shape[0], len(HAND_COLLISION_BODY_NAMES), 7), dtype=np.float64)
    for frame, (wrist, qpos) in enumerate(zip(wrists, fingers, strict=True)):
        wrist_matrix = pose_matrix(wrist)
        transforms = model.forward_kinematics_reference(qpos)
        for body_index, body_name in enumerate(HAND_COLLISION_BODY_NAMES):
            result[frame, body_index] = pose_from_matrix(wrist_matrix @ transforms[body_name])
    if not np.isfinite(result).all() or np.any(np.linalg.norm(result[..., 3:7], axis=-1) < 1.0e-8):
        raise RuntimeError("reconstructed hand collision-body pose is invalid")
    return result


__all__ = ["HAND_COLLISION_BODY_NAMES", "reconstruct_hand_collision_body_pose"]
