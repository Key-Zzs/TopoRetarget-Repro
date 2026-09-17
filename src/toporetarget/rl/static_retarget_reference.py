"""Adapter from one frozen retarget state to a constant physical reference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.geometry.se3 import invert_transform, transform_points
from toporetarget.rl.axis_points import object_axis_points_from_poses
from toporetarget.rl.tracked_links import TRACKED_LINKS_WUJI_RH
from toporetarget.rl.world_wrist import quaternion_wxyz_from_matrix

STATIC_REFERENCE_FRAMES = 321
STATIC_REFERENCE_SOURCE_FRAMES = 41
STATIC_REFERENCE_TIME_SCALE = 8


def _pose(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"STATIC_REFERENCE_{name.upper()}_INVALID")
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9):
        raise ValueError(f"STATIC_REFERENCE_{name.upper()}_HOMOGENEOUS_ROW_INVALID")
    return result


def build_static_retarget_reference_arrays(
    *,
    qpos: Any,
    base_pose_world: Any,
    object_pose_world: Any,
    robot_model: Any,
    source_frame_id: int,
    anchor_id: str,
) -> dict[str, np.ndarray]:
    """Return the exact 321-sample constant Stage16 reference representation."""

    q = np.asarray(qpos, dtype=np.float64)
    if q.shape != (20,) or not np.isfinite(q).all():
        raise ValueError("STATIC_REFERENCE_QPOS_INVALID")
    base = _pose(base_pose_world, "base_pose")
    obj = _pose(object_pose_world, "object_pose")
    if not anchor_id or any(token in anchor_id for token in ("/", "\\", "..")):
        raise ValueError("STATIC_REFERENCE_ANCHOR_ID_INVALID")

    fk = robot_model.forward_kinematics_reference(q)
    missing = [name for name in TRACKED_LINKS_WUJI_RH if name not in fk]
    if missing:
        raise ValueError(f"STATIC_REFERENCE_TRACKED_LINK_MISSING:{missing}")
    links_wrist = np.stack([np.asarray(fk[name])[:3, 3] for name in TRACKED_LINKS_WUJI_RH])
    links_world = transform_points(base[None], links_wrist[None])[0]
    object_axes = object_axis_points_from_poses(obj[None])[0]
    base_inverse = invert_transform(base[None])[0]
    object_pose_wrist = base_inverse @ obj
    object_axes_wrist = transform_points(base_inverse[None], object_axes[None])[0]

    count = STATIC_REFERENCE_FRAMES

    def repeat(value: np.ndarray) -> np.ndarray:
        return np.repeat(np.asarray(value)[None], count, axis=0)

    timestamps = np.arange(count, dtype=np.float64) / 20.0
    zeros6 = np.zeros((count, 6), dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "timestamps": timestamps,
        "source_frame_indices": np.full(count, int(source_frame_id), dtype=np.int64),
        "T_world_wrist_ref": repeat(base).astype(np.float32),
        "wrist_pose_translation_world_ref": repeat(base[:3, 3]).astype(np.float32),
        "wrist_pose_quaternion_world_ref_wxyz": repeat(
            quaternion_wxyz_from_matrix(base[:3, :3])
        ).astype(np.float32),
        "wrist_twist_world_ref": zeros6.copy(),
        "q_finger_ref": repeat(q).astype(np.float32),
        "qdot_finger_ref": np.zeros((count, 20), dtype=np.float32),
        "T_world_object_ref": repeat(obj).astype(np.float32),
        "object_pose_translation_world_ref": repeat(obj[:3, 3]).astype(np.float32),
        "object_pose_quaternion_world_ref_wxyz": repeat(
            quaternion_wxyz_from_matrix(obj[:3, :3])
        ).astype(np.float32),
        "object_twist_world_ref": zeros6.copy(),
        "object_axis_points_world_ref": repeat(object_axes).astype(np.float32),
        "tracked_link_positions_world_ref": repeat(links_world).astype(np.float32),
        "T_wrist_object_ref": repeat(object_pose_wrist).astype(np.float32),
        "object_axis_points_wrist_ref": repeat(object_axes_wrist).astype(np.float32),
        "tracked_link_positions_wrist_ref": repeat(links_wrist).astype(np.float32),
    }
    metadata = {
        "schema_version": "StaticRetargetReferenceAdapterV1",
        "identifier": "StaticRetargetReferenceAdapterV1",
        "reference_profile": "world_wrist_finger_residual_v1",
        "reference_frame": "world_scene",
        "joint_order": list(robot_model.dof_names),
        "tracked_link_names": list(TRACKED_LINKS_WUJI_RH),
        "quaternion_convention": "wxyz_active_right_handed_shortest_rotation",
        "units": {"translation": "m", "angles": "rad", "time": "s"},
        "reference_kinematics_version": 2,
        "angular_velocity_convention": "world: [omega]_x = R_dot @ R_T",
        "time_scale": STATIC_REFERENCE_TIME_SCALE,
        "source_frames": STATIC_REFERENCE_SOURCE_FRAMES,
        "runtime_samples": count,
        "control_hz": 20.0,
        "static_reference_index": "single_constant_state_repeated_over_runtime_domain",
        "anchor_id": anchor_id,
        "source_frame_id": int(source_frame_id),
        "q_old_used": False,
        "velocity_authority": "exact_zero_for_constant_reference",
        "hand_target_constant": True,
        "object_target_constant": True,
    }
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)
    return arrays


def write_static_retarget_reference(
    destination: Path,
    *,
    qpos: Any,
    base_pose_world: Any,
    object_pose_world: Any,
    robot_model: Any,
    source_frame_id: int,
    anchor_id: str,
) -> Path:
    arrays = build_static_retarget_reference_arrays(
        qpos=qpos,
        base_pose_world=base_pose_world,
        object_pose_world=object_pose_world,
        robot_model=robot_model,
        source_frame_id=source_frame_id,
        anchor_id=anchor_id,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # NumPy's current stubs treat arbitrary keyword values as the optional
    # ``allow_pickle`` flag even though savez_compressed accepts array kwargs.
    np.savez_compressed(destination, **arrays)  # type: ignore[arg-type]
    return destination


__all__ = [
    "STATIC_REFERENCE_FRAMES",
    "STATIC_REFERENCE_SOURCE_FRAMES",
    "STATIC_REFERENCE_TIME_SCALE",
    "build_static_retarget_reference_arrays",
    "write_static_retarget_reference",
]
