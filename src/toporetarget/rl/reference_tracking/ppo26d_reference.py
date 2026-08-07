"""Reference provenance and factor-8 export for Stage 16-D.5 PPO."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_FRAMES = 41
RUNTIME_FRAMES = 321
CONTROL_HZ = 20.0
REFERENCE_TIME_SCALE = 8

REFERENCE_FIELDS = (
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
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Stage16DPPO26DReferenceV1:
    identifier: str = "Stage16DPPO26DReferenceV1"
    source: str = "formal_factor8_stage16d_reference"
    source_frames: int = SOURCE_FRAMES
    runtime_samples: int = RUNTIME_FRAMES
    control_hz: float = CONTROL_HZ
    time_scale: int = REFERENCE_TIME_SCALE
    wrist_pose: str = "base_frame_translation_and_wxyz"
    wrist_twist: str = "world_linear_angular_when_available"
    finger_q: str = "canonical_20d"
    finger_qdot: str = "canonical_20d"
    object_pose: str = "base_frame_translation_and_wxyz"
    object_axis_points: str = "six_base_frame_points"
    tracked_links: str = "16_base_frame_points"
    forbidden_sources: tuple[str, ...] = (
        "corrected_yellow_object_trajectory",
        "cem_output_trajectory",
        "ppo_rollout_trajectory",
    )

    def __post_init__(self) -> None:
        if (self.source_frames, self.runtime_samples, self.time_scale) != (41, 321, 8):
            raise ValueError("PPO26D reference must be the Stage16-D 41-to-321 factor-8 view")
        if self.control_hz != 20.0:
            raise ValueError("PPO26D control reference is fixed at 20 Hz")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_source_reference(path: Path) -> dict[str, Any]:
    """Validate the frozen source NPZ without treating derived trajectories as references."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(REFERENCE_FIELDS) - set(archive.files))
        if missing:
            raise ValueError(f"PPO26D source reference is missing fields: {missing}")
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
        if timestamps.shape != (SOURCE_FRAMES,):
            raise ValueError("PPO26D source reference must have 41 source samples")
        if not np.all(np.diff(timestamps) > 0.0) or not np.isclose(
            np.median(np.diff(timestamps)), 1.0 / CONTROL_HZ, atol=1.0e-8
        ):
            raise ValueError("PPO26D source reference must be a 20 Hz sequence")
        shapes = {name: list(np.asarray(archive[name]).shape) for name in REFERENCE_FIELDS}
        for name in REFERENCE_FIELDS:
            value = np.asarray(archive[name])
            if value.shape[0] != SOURCE_FRAMES or not np.isfinite(value).all():
                raise ValueError(f"PPO26D source reference has invalid {name}")
        metadata = json.loads(str(archive["metadata"].item()))
    if len(metadata.get("joint_order", ())) != 20:
        raise ValueError("PPO26D reference needs exactly 20 canonical finger joints")
    if len(metadata.get("tracked_link_names", ())) != 16:
        raise ValueError("PPO26D reference needs exactly 16 tracked links")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_frames": SOURCE_FRAMES,
        "control_hz": CONTROL_HZ,
        "fields": shapes,
        "joint_order": list(metadata["joint_order"]),
        "tracked_link_names": list(metadata["tracked_link_names"]),
    }


def _interpolate_linear(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    left = np.floor(coordinates).astype(np.int64).clip(0, SOURCE_FRAMES - 2)
    alpha = (coordinates - left).reshape((RUNTIME_FRAMES,) + (1,) * (values.ndim - 1))
    alpha[-1] = 1.0
    return (1.0 - alpha) * values[left] + alpha * values[left + 1]


def _interpolate_quaternion(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    left = np.floor(coordinates).astype(np.int64).clip(0, SOURCE_FRAMES - 2)
    alpha = (coordinates - left)[:, None]
    alpha[-1] = 1.0
    start, end = values[left], values[left + 1].copy()
    end[np.sum(start * end, axis=-1) < 0.0] *= -1.0
    result = (1.0 - alpha) * start + alpha * end
    return result / np.linalg.norm(result, axis=-1, keepdims=True).clip(1.0e-12)


def export_factor8_reference(source: Path, destination: Path) -> dict[str, Any]:
    """Materialize a standalone 321-sample PPO reference from the official NPZ."""

    inspection = inspect_source_reference(source)
    coordinates = np.arange(RUNTIME_FRAMES, dtype=np.float64) / REFERENCE_TIME_SCALE
    with np.load(source, allow_pickle=False) as archive:
        arrays: dict[str, np.ndarray] = {}
        for name in REFERENCE_FIELDS:
            values = np.asarray(archive[name], dtype=np.float32)
            arrays[name] = (
                _interpolate_quaternion(values, coordinates).astype(np.float32)
                if "quaternion" in name
                else _interpolate_linear(values, coordinates).astype(np.float32)
            )
        metadata = json.loads(str(archive["metadata"].item()))
    payload = {
        **arrays,
        "timestamps": np.arange(RUNTIME_FRAMES, dtype=np.float32) / CONTROL_HZ,
        "metadata": np.asarray(
            json.dumps(
                {
                    **metadata,
                    "schema_version": "Stage16DPPO26DReferenceV1",
                    "source_sha256": inspection["sha256"],
                    "source_frames": SOURCE_FRAMES,
                    "runtime_samples": RUNTIME_FRAMES,
                    "control_hz": CONTROL_HZ,
                    "time_scale": REFERENCE_TIME_SCALE,
                },
                sort_keys=True,
            )
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)  # type: ignore[arg-type]
    return {
        "contract": Stage16DPPO26DReferenceV1().as_dict(),
        "source": inspection,
        "destination": str(destination.resolve()),
        "destination_sha256": sha256_file(destination),
    }


__all__ = [
    "CONTROL_HZ",
    "REFERENCE_FIELDS",
    "REFERENCE_TIME_SCALE",
    "RUNTIME_FRAMES",
    "SOURCE_FRAMES",
    "Stage16DPPO26DReferenceV1",
    "export_factor8_reference",
    "inspect_source_reference",
    "sha256_file",
]
