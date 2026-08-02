"""GPU-resident, immutable Stage 16-C world-wrist reference bank."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class WorldWristReferenceBankManifest:
    identifier: str
    frame_count: int
    control_hz: float
    clip_ids: tuple[str, ...]
    joint_order: tuple[str, ...]
    tracked_link_names: tuple[str, ...]
    hashes: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "frame_count": self.frame_count,
            "control_hz": self.control_hz,
            "clip_ids": list(self.clip_ids),
            "joint_order": list(self.joint_order),
            "tracked_link_names": list(self.tracked_link_names),
            "hashes": self.hashes,
        }


class WorldWristReferenceBank:
    """Load both immutable references exactly once and copy them to CUDA once."""

    wrist_pose_translation_world_ref: torch.Tensor
    wrist_pose_quaternion_world_ref_wxyz: torch.Tensor
    wrist_twist_world_ref: torch.Tensor
    q_finger_ref: torch.Tensor
    qdot_finger_ref: torch.Tensor
    object_pose_translation_world_ref: torch.Tensor
    object_pose_quaternion_world_ref_wxyz: torch.Tensor
    object_twist_world_ref: torch.Tensor
    object_axis_points_world_ref: torch.Tensor
    tracked_link_positions_world_ref: torch.Tensor
    object_axis_points_wrist_ref: torch.Tensor
    tracked_link_positions_wrist_ref: torch.Tensor

    REQUIRED_FIELDS = (
        "timestamps",
        "wrist_pose_translation_world_ref",
        "wrist_pose_quaternion_world_ref_wxyz",
        "wrist_twist_world_ref",
        "q_finger_ref",
        "qdot_finger_ref",
        "object_pose_translation_world_ref",
        "object_pose_quaternion_world_ref_wxyz",
        "object_twist_world_ref",
        "object_axis_points_world_ref",
        "tracked_link_positions_world_ref",
        "object_axis_points_wrist_ref",
        "tracked_link_positions_wrist_ref",
    )

    def __init__(self, paths: Mapping[str, str | Path], *, device: torch.device | str) -> None:
        if set(paths) != {"hocap_170105", "hocap_170650"}:
            raise ValueError("the frozen bank contains exactly hocap_170105 and hocap_170650")
        arrays: dict[str, list[np.ndarray]] = {field: [] for field in self.REQUIRED_FIELDS}
        self.hashes: dict[str, str] = {}
        joint_order: tuple[str, ...] | None = None
        link_order: tuple[str, ...] | None = None
        frame_count: int | None = None
        timestamps: np.ndarray | None = None
        for clip_id in sorted(paths):
            path = Path(paths[clip_id])
            if not path.is_file():
                raise FileNotFoundError(path)
            self.hashes[clip_id] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as source:
                missing = set(self.REQUIRED_FIELDS) - set(source.files)
                if missing:
                    raise ValueError(f"{path} misses fields: {sorted(missing)}")
                metadata = json.loads(str(source["metadata"].item()))
                current_joint_order = tuple(metadata["joint_order"])
                current_link_order = tuple(metadata["tracked_link_names"])
                source_timestamps = np.asarray(source["timestamps"], dtype=np.float64)
                current_timestamps = source_timestamps.astype(np.float32)
                if (
                    current_timestamps.shape != (41,)
                    or not np.all(np.diff(source_timestamps) > 0.0)
                    or not np.isclose(np.median(np.diff(source_timestamps)), 0.05, atol=1.0e-8)
                ):
                    raise ValueError(f"{path} is not the frozen 41-frame, 20 Hz reference")
                if joint_order is None:
                    joint_order, link_order, frame_count, timestamps = (
                        current_joint_order,
                        current_link_order,
                        current_timestamps.size,
                        current_timestamps,
                    )
                else:
                    assert link_order is not None and timestamps is not None
                    if (
                        current_joint_order != joint_order
                        or current_link_order != link_order
                        or current_timestamps.shape != timestamps.shape
                        or not np.allclose(current_timestamps, timestamps)
                    ):
                        raise ValueError(
                            "reference clips do not share the frozen Stage 16-C contract"
                        )
                for field in self.REQUIRED_FIELDS:
                    value = np.asarray(source[field], dtype=np.float32)
                    if not np.isfinite(value).all():
                        raise ValueError(f"{path} has non-finite {field}")
                    arrays[field].append(value)
        assert joint_order is not None and link_order is not None and frame_count is not None
        if len(joint_order) != 20 or len(link_order) != 16:
            raise ValueError("reference bank requires 20 joints and 16 tracked links")
        self.device = torch.device(device)
        self.clip_ids = tuple(sorted(paths))
        self.joint_order = joint_order
        self.tracked_link_names = link_order
        self.frame_count = frame_count
        for field, values in arrays.items():
            setattr(self, field, torch.as_tensor(np.stack(values), device=self.device))
        self.valid_mask = torch.ones((2, self.frame_count), dtype=torch.bool, device=self.device)
        self.manifest = WorldWristReferenceBankManifest(
            identifier="world_wrist_reference_bank_v1",
            frame_count=self.frame_count,
            control_hz=20.0,
            clip_ids=self.clip_ids,
            joint_order=self.joint_order,
            tracked_link_names=self.tracked_link_names,
            hashes=dict(self.hashes),
        )
        self.object_axis_points_local = self._local_axis_points()

    def _local_axis_points(self) -> torch.Tensor:
        quat = self.object_pose_quaternion_world_ref_wxyz[:, 0]
        rotation = quaternion_to_matrix_wxyz(quat)
        points = self.object_axis_points_world_ref[:, 0]
        position = self.object_pose_translation_world_ref[:, 0, None, :]
        return torch.matmul(
            rotation.transpose(-1, -2), (points - position).transpose(-1, -2)
        ).transpose(-1, -2)

    def indices(
        self, clip_index: torch.Tensor, reference_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if clip_index.ndim != 1 or reference_index.shape != clip_index.shape:
            raise ValueError("clip_index and reference_index must both have shape [num_envs]")
        if bool(torch.any(clip_index < 0)) or bool(torch.any(clip_index >= len(self.clip_ids))):
            raise ValueError("clip index outside bank")
        return clip_index.long(), reference_index.long().clamp(0, self.frame_count - 1)

    def gather(
        self, field: str, clip_index: torch.Tensor, reference_index: torch.Tensor
    ) -> torch.Tensor:
        clips, frames = self.indices(clip_index, reference_index)
        return getattr(self, field)[clips, frames]

    def assignment(self, num_envs: int, *, balanced: bool) -> torch.Tensor:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if balanced:
            return torch.arange(num_envs, device=self.device) % len(self.clip_ids)
        return torch.zeros(num_envs, dtype=torch.long, device=self.device)


def quaternion_to_matrix_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    q = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-12)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


__all__ = [
    "WorldWristReferenceBank",
    "WorldWristReferenceBankManifest",
    "quaternion_to_matrix_wxyz",
]
