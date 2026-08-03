"""Pure-Python semantic checks shared by Stage 16-C Isaac Lab qualifiers.

The frozen world-wrist references preserve both resampled joint trajectories and
resampled tracked-link trajectories.  The latter is an observation/reward
feature and must remain byte-for-byte unchanged.  Since the two fields are
interpolated independently, however, the stored link points are not guaranteed
to equal a fresh forward-kinematics evaluation at every 20 Hz frame.  C3-0 is
an actuator-independent identity check, so it must compare the imported asset
against the deterministic FK target derived from the immutable wrist pose and
joint arrays, rather than mistake interpolation residual for a frame error.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toporetarget.robots.registry import get_robot_registry


@dataclass(frozen=True)
class FullyKinematicLinkTargets:
    """C3-0 FK targets derived without changing a frozen reference file."""

    positions_world: np.ndarray
    manifest: dict[str, object]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_fully_kinematic_link_targets(
    paths: Mapping[str, str | Path], *, repo_root: Path
) -> dict[str, FullyKinematicLinkTargets]:
    """Derive link-origin targets from frozen wrist pose and joint arrays.

    This is deliberately a read-only qualification aid.  It neither rewrites
    the frozen NPZ nor substitutes its stored link feature in the runtime
    observation or reward contract.
    """

    expected_clips = {"hocap_170105", "hocap_170650"}
    if set(paths) != expected_clips:
        raise ValueError(f"C3-0 requires exactly {sorted(expected_clips)}")
    model = get_robot_registry(repo_root=repo_root).load("wuji_hand2_beta1_rh")
    result: dict[str, FullyKinematicLinkTargets] = {}
    for clip_id in sorted(paths):
        path = Path(paths[clip_id]).resolve()
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            joint_order = tuple(str(name) for name in metadata["joint_order"])
            tracked_names = tuple(str(name) for name in metadata["tracked_link_names"])
            if joint_order != model.dof_names:
                raise ValueError(f"C3-0 joint order differs from frozen Wuji model: {path}")
            q_finger = np.asarray(payload["q_finger_ref"], dtype=np.float64)
            wrist_poses = np.asarray(payload["T_world_wrist_ref"], dtype=np.float64)
            stored_links = np.asarray(payload["tracked_link_positions_world_ref"], dtype=np.float64)
        if q_finger.shape != (41, len(joint_order)) or wrist_poses.shape != (41, 4, 4):
            raise ValueError(f"C3-0 requires 41 frozen wrist/joint frames: {path}")
        if stored_links.shape != (41, len(tracked_names), 3):
            raise ValueError(f"C3-0 tracked-link shape mismatch: {path}")
        positions = np.empty_like(stored_links)
        for frame, (q_frame, wrist_pose) in enumerate(zip(q_finger, wrist_poses, strict=True)):
            transforms = model.forward_kinematics_reference(q_frame)
            positions[frame] = np.stack(
                [(wrist_pose @ transforms[name])[:3, 3] for name in tracked_names], axis=0
            )
        stored_residual = np.linalg.norm(positions - stored_links, axis=-1)
        result[clip_id] = FullyKinematicLinkTargets(
            positions_world=positions.astype(np.float32),
            manifest={
                "contract": "c3_0_kinematic_link_targets_v1",
                "clip": clip_id,
                "reference_path": str(path),
                "reference_sha256": _sha256(path),
                "source": "frozen T_world_wrist_ref + q_finger_ref + canonical Wuji URDF FK",
                "source_urdf": str(model.urdf.urdf_path),
                "source_urdf_sha256": model.urdf_hash,
                "robot_spec_sha256": model.spec_hash,
                "tracked_link_names": list(tracked_names),
                "stored_link_field": "tracked_link_positions_world_ref",
                "stored_link_field_preserved": True,
                "stored_link_interpolation_residual_max_m": float(stored_residual.max()),
                "stored_link_interpolation_residual_rms_m": float(
                    np.sqrt(np.mean(np.square(stored_residual)))
                ),
            },
        )
    return result


__all__ = ["FullyKinematicLinkTargets", "derive_fully_kinematic_link_targets"]
