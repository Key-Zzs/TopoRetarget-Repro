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
    source_frame_count: int = 41
    source_control_hz: float = 20.0
    reference_time_scale: int = 1

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "frame_count": self.frame_count,
            "control_hz": self.control_hz,
            "clip_ids": list(self.clip_ids),
            "joint_order": list(self.joint_order),
            "tracked_link_names": list(self.tracked_link_names),
            "hashes": self.hashes,
            "source_frame_count": self.source_frame_count,
            "source_control_hz": self.source_control_hz,
            "reference_time_scale": self.reference_time_scale,
        }


class WorldWristReferenceBank:
    """Load immutable world-wrist references once and copy them to the target device.

    The original C2/C4 experiment happened to use two development clips.  That
    is a property of its input manifest, not of the reference representation:
    a production per-clip lineage must also be able to load exactly one
    independently materialized reference.  All entries in a bank still have
    to share the same kinematic schema and runtime time domain; callers that
    need different durations create separate banks (which is precisely what
    independent held-out lineages do).
    """

    timestamps: torch.Tensor
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
        if not paths:
            raise ValueError("reference bank requires at least one clip")
        if len({str(clip_id) for clip_id in paths}) != len(paths):
            raise ValueError("reference bank clip IDs must be unique")
        arrays: dict[str, list[np.ndarray]] = {field: [] for field in self.REQUIRED_FIELDS}
        self.hashes: dict[str, str] = {}
        joint_order: tuple[str, ...] | None = None
        link_order: tuple[str, ...] | None = None
        frame_count: int | None = None
        timestamps: np.ndarray | None = None
        source_frame_count: int | None = None
        reference_time_scale: int | None = None
        identifier: str | None = None
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
                kinematics_version = int(metadata.get("reference_kinematics_version", 1))
                current_scale = int(metadata.get("time_scale", 1))
                current_source_frames = int(metadata.get("source_frames", source_timestamps.size))
                expected_frames = 321 if kinematics_version == 2 else 41
                if (
                    current_timestamps.shape != (expected_frames,)
                    or not np.all(np.diff(source_timestamps) > 0.0)
                    or not np.isclose(np.median(np.diff(source_timestamps)), 0.05, atol=1.0e-8)
                ):
                    raise ValueError(f"{path} is not a valid 20 Hz Stage 16 reference")
                if kinematics_version == 2 and (
                    current_scale != 8
                    or current_source_frames != 41
                    or metadata.get("angular_velocity_convention")
                    != "world: [omega]_x = R_dot @ R_T"
                ):
                    raise ValueError(f"{path} is not a valid Stage16DReferenceKinematicsV2")
                if joint_order is None:
                    (
                        joint_order,
                        link_order,
                        frame_count,
                        timestamps,
                        source_frame_count,
                        reference_time_scale,
                        identifier,
                    ) = (
                        current_joint_order,
                        current_link_order,
                        current_timestamps.size,
                        current_timestamps,
                        current_source_frames,
                        current_scale,
                        (
                            "world_wrist_reference_bank_kinematics_v2"
                            if kinematics_version == 2
                            else "world_wrist_reference_bank_v1"
                        ),
                    )
                else:
                    assert (
                        link_order is not None
                        and timestamps is not None
                        and source_frame_count is not None
                        and reference_time_scale is not None
                    )
                    if (
                        current_joint_order != joint_order
                        or current_link_order != link_order
                        or current_timestamps.shape != timestamps.shape
                        or not np.allclose(current_timestamps, timestamps)
                        or current_source_frames != source_frame_count
                        or current_scale != reference_time_scale
                    ):
                        raise ValueError(
                            "reference clips do not share the frozen Stage 16-C contract"
                        )
                for field in self.REQUIRED_FIELDS:
                    value = np.asarray(source[field], dtype=np.float32)
                    if not np.isfinite(value).all():
                        raise ValueError(f"{path} has non-finite {field}")
                    arrays[field].append(value)
        assert (
            joint_order is not None
            and link_order is not None
            and frame_count is not None
            and source_frame_count is not None
            and reference_time_scale is not None
            and identifier is not None
        )
        if len(joint_order) != 20 or len(link_order) != 16:
            raise ValueError("reference bank requires 20 joints and 16 tracked links")
        self.device = torch.device(device)
        self.clip_ids = tuple(sorted(paths))
        self.joint_order = joint_order
        self.tracked_link_names = link_order
        self.frame_count = frame_count
        for field, values in arrays.items():
            setattr(self, field, torch.as_tensor(np.stack(values), device=self.device))
        self.valid_mask = torch.ones(
            (len(self.clip_ids), self.frame_count), dtype=torch.bool, device=self.device
        )
        self.manifest = WorldWristReferenceBankManifest(
            identifier=identifier,
            frame_count=self.frame_count,
            control_hz=20.0,
            clip_ids=self.clip_ids,
            joint_order=self.joint_order,
            tracked_link_names=self.tracked_link_names,
            hashes=dict(self.hashes),
            source_frame_count=source_frame_count,
            source_control_hz=20.0,
            reference_time_scale=reference_time_scale,
        )
        self.object_axis_points_local = self._local_axis_points()

    @staticmethod
    def _linear_retime(
        values: torch.Tensor, interval: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        start = values[:, interval]
        end = values[:, interval + 1]
        weight = alpha.reshape((1, -1) + (1,) * (values.ndim - 2))
        return start + weight * (end - start)

    @staticmethod
    def _quaternion_retime(
        values: torch.Tensor, interval: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """Shortest-arc normalized interpolation without importing tensor_math."""

        start = values[:, interval]
        end = values[:, interval + 1]
        dot = (start * end).sum(dim=-1, keepdim=True)
        end = torch.where(dot < 0.0, -end, end)
        weight = alpha[None, :, None]
        result = (1.0 - weight) * start + weight * end
        return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(1.0e-12)

    @staticmethod
    def _hermite_retime(
        values: torch.Tensor,
        derivatives: torch.Tensor,
        interval: torch.Tensor,
        alpha: torch.Tensor,
        *,
        source_dt_s: float,
    ) -> torch.Tensor:
        start = values[:, interval]
        end = values[:, interval + 1]
        velocity_start = derivatives[:, interval]
        velocity_end = derivatives[:, interval + 1]
        weight = alpha.reshape((1, -1) + (1,) * (values.ndim - 2))
        weight2 = weight.square()
        weight3 = weight2 * weight
        return (
            (2.0 * weight3 - 3.0 * weight2 + 1.0) * start
            + (weight3 - 2.0 * weight2 + weight) * source_dt_s * velocity_start
            + (-2.0 * weight3 + 3.0 * weight2) * end
            + (weight3 - weight2) * source_dt_s * velocity_end
        )

    def apply_uniform_time_scale(self, time_scale: int) -> None:
        """Materialize a 20 Hz, uniformly retimed view while preserving source keys.

        The source NPZs and their hashes remain unchanged.  Every original key is
        present at ``retimed_index = source_index * time_scale``.  The runtime
        still advances at 20 Hz, so policy action and observation cadence stay
        unchanged while reference velocity and acceleration demand decrease.
        """

        if isinstance(time_scale, bool) or not isinstance(time_scale, int) or time_scale < 1:
            raise ValueError("reference_time_scale must be a positive integer")
        if time_scale == self.manifest.reference_time_scale:
            return
        if self.manifest.reference_time_scale != 1:
            raise RuntimeError("reference bank has already been retimed")

        source_frames = self.frame_count
        source_dt_s = 1.0 / self.manifest.source_control_hz
        retimed_frames = (source_frames - 1) * time_scale + 1
        retimed_index = torch.arange(retimed_frames, device=self.device)
        source_coordinate = retimed_index.to(torch.float32) / float(time_scale)
        interval = torch.floor(source_coordinate).to(torch.long).clamp_max(source_frames - 2)
        alpha = (source_coordinate - interval.to(torch.float32)).clamp(0.0, 1.0)
        alpha[-1] = 1.0

        wrist_position = self._hermite_retime(
            self.wrist_pose_translation_world_ref,
            self.wrist_twist_world_ref[..., :3],
            interval,
            alpha,
            source_dt_s=source_dt_s,
        )
        object_position = self._hermite_retime(
            self.object_pose_translation_world_ref,
            self.object_twist_world_ref[..., :3],
            interval,
            alpha,
            source_dt_s=source_dt_s,
        )
        finger_position = self._hermite_retime(
            self.q_finger_ref,
            self.qdot_finger_ref,
            interval,
            alpha,
            source_dt_s=source_dt_s,
        )
        self.wrist_pose_translation_world_ref = wrist_position
        self.wrist_pose_quaternion_world_ref_wxyz = self._quaternion_retime(
            self.wrist_pose_quaternion_world_ref_wxyz, interval, alpha
        )
        self.wrist_twist_world_ref = self._linear_retime(
            self.wrist_twist_world_ref, interval, alpha
        ) / float(time_scale)
        self.q_finger_ref = finger_position
        self.qdot_finger_ref = self._linear_retime(self.qdot_finger_ref, interval, alpha) / float(
            time_scale
        )
        self.object_pose_translation_world_ref = object_position
        self.object_pose_quaternion_world_ref_wxyz = self._quaternion_retime(
            self.object_pose_quaternion_world_ref_wxyz, interval, alpha
        )
        self.object_twist_world_ref = self._linear_retime(
            self.object_twist_world_ref, interval, alpha
        ) / float(time_scale)
        for field in (
            "object_axis_points_world_ref",
            "tracked_link_positions_world_ref",
            "object_axis_points_wrist_ref",
            "tracked_link_positions_wrist_ref",
        ):
            setattr(self, field, self._linear_retime(getattr(self, field), interval, alpha))
        source_timestamps = self.timestamps
        start_time = source_timestamps[:, :1]
        self.timestamps = (
            start_time
            + torch.arange(retimed_frames, dtype=source_timestamps.dtype, device=self.device)[None]
            / self.manifest.control_hz
        )
        self.frame_count = retimed_frames
        self.valid_mask = torch.ones(
            (len(self.clip_ids), retimed_frames), dtype=torch.bool, device=self.device
        )
        self.manifest = WorldWristReferenceBankManifest(
            identifier="world_wrist_reference_bank_uniform_retimed_v1",
            frame_count=retimed_frames,
            control_hz=self.manifest.control_hz,
            clip_ids=self.clip_ids,
            joint_order=self.joint_order,
            tracked_link_names=self.tracked_link_names,
            hashes=dict(self.hashes),
            source_frame_count=source_frames,
            source_control_hz=self.manifest.source_control_hz,
            reference_time_scale=time_scale,
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

    def clip_index(self, clip_id: str) -> int:
        """Resolve one clip identifier without silently falling back to index zero."""

        try:
            return self.clip_ids.index(clip_id)
        except ValueError as error:
            raise ValueError(f"unknown fixed reference clip: {clip_id}") from error

    def assignment(
        self, num_envs: int, *, balanced: bool, fixed_clip: str | None = None
    ) -> torch.Tensor:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if fixed_clip is not None:
            return torch.full(
                (num_envs,), self.clip_index(fixed_clip), dtype=torch.long, device=self.device
            )
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
