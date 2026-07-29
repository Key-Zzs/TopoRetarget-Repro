"""Contracts for continuity-aware, full-state Wuji retargeting.

The historical Stage 9 solver uses a local seed-delta chart.  This module
keeps the chart convention explicit and provides the small, pure functions
used by the engineering extension.  It intentionally contains no filtering
or post-hoc trajectory editing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.geometry.se3 import invert_transform

CONTINUOUS_PROFILE_ID = "wuji_continuous_full_state_v1"
SEQUENTIAL_PROFILE_ID = "wuji_continuous_sequential_v1"
CONTINUOUS_PROFILE_IDS = frozenset({CONTINUOUS_PROFILE_ID, SEQUENTIAL_PROFILE_ID})
CONTINUITY_SCHEMA_VERSION = "toporetarget.trajectory_continuity.v1"
BASE_CORRECTION_CONVENTION = "scene_local_seed_delta_exp_left"
S_POS_M = 0.010
S_ROT_RAD = float(np.deg2rad(5.0))
S_Q_RAD = 0.050
LAMBDA_CORR = 0.25


def is_continuous_profile(profile_id: str) -> bool:
    """Return whether ``profile_id`` uses the Wuji continuity contract."""

    return str(profile_id) in CONTINUOUS_PROFILE_IDS


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def so3_log_np(rotation: np.ndarray) -> np.ndarray:
    """Stable SO(3) logarithm for one float64 rotation matrix."""

    value = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    skew = np.asarray(
        [value[2, 1] - value[1, 2], value[0, 2] - value[2, 0], value[1, 0] - value[0, 1]],
        dtype=np.float64,
    )
    if theta < 1.0e-10:
        return 0.5 * skew
    if np.pi - theta < 1.0e-6:
        diagonal = np.maximum(np.diag(value) + 1.0, 0.0)
        axis = np.sqrt(diagonal / 2.0)
        pivot = int(np.argmax(axis))
        if axis[pivot] < 1.0e-12:
            return np.zeros(3, dtype=np.float64)
        for index in range(3):
            if index != pivot:
                axis[index] = (value[pivot, index] + value[index, pivot]) / (4.0 * axis[pivot])
        return theta * axis / max(float(np.linalg.norm(axis)), 1.0e-15)
    return theta / (2.0 * np.sin(theta)) * skew


def _validate_pose(value: np.ndarray, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite (4,4) transform")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-10):
        raise ValueError(f"{name} has an invalid homogeneous row")
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1.0e-8):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1.0e-8):
        raise ValueError(f"{name} rotation determinant is not +1")
    return pose


def encode_base_correction(
    warm_base_scene: np.ndarray,
    final_base_scene: np.ndarray,
    convention: str = BASE_CORRECTION_CONVENTION,
) -> np.ndarray:
    """Encode ``final`` in the existing local seed-delta chart.

    The rotational delta is left-multiplied and expressed in scene axes.  The
    translation delta is additive in scene coordinates; this is the exact
    convention used by ``_FrameContext.base_pose_torch``.
    """

    if convention != BASE_CORRECTION_CONVENTION:
        raise ValueError(f"unsupported base correction convention: {convention}")
    warm = _validate_pose(warm_base_scene, "warm_base_scene")
    final = _validate_pose(final_base_scene, "final_base_scene")
    rotation_delta = final[:3, :3] @ warm[:3, :3].T
    return np.concatenate(
        [final[:3, 3] - warm[:3, 3], so3_log_np(rotation_delta)], dtype=np.float64
    )


def decode_base_correction(
    warm_base_scene: np.ndarray,
    correction: np.ndarray,
    convention: str = BASE_CORRECTION_CONVENTION,
) -> np.ndarray:
    """Decode a six-vector in the existing seed-delta chart."""

    if convention != BASE_CORRECTION_CONVENTION:
        raise ValueError(f"unsupported base correction convention: {convention}")
    warm = _validate_pose(warm_base_scene, "warm_base_scene")
    value = np.asarray(correction, dtype=np.float64).reshape(-1)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("base correction must be six finite values")
    theta = float(np.linalg.norm(value[3:]))
    if theta < 1.0e-12:
        rotation_delta = np.eye(3)
    else:
        axis = value[3:] / theta
        skew = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        rotation_delta = np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_delta @ warm[:3, :3]
    result[:3, 3] = warm[:3, 3] + value[:3]
    return result


@dataclass(frozen=True)
class PropagatedRetargetState:
    predicted_base_scene: np.ndarray
    predicted_qpos: np.ndarray
    base_correction: np.ndarray
    q_correction: np.ndarray
    q_clamp_count: int
    previous_frame: int | None
    current_frame: int
    initialization_source: str


def transport_previous_final_to_current_warm(
    previous_warm_base_scene: np.ndarray,
    previous_final_base_scene: np.ndarray,
    current_warm_base_scene: np.ndarray,
    previous_warm_qpos: np.ndarray,
    previous_final_qpos: np.ndarray,
    current_warm_qpos: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    *,
    previous_frame: int | None = None,
    current_frame: int = 0,
) -> PropagatedRetargetState:
    """Transport the previous accepted correction through the current warm chart."""

    correction = encode_base_correction(previous_warm_base_scene, previous_final_base_scene)
    predicted_base = decode_base_correction(current_warm_base_scene, correction)
    previous_q = np.asarray(previous_final_qpos, dtype=np.float64) - np.asarray(
        previous_warm_qpos, dtype=np.float64
    )
    raw_q = np.asarray(current_warm_qpos, dtype=np.float64) + previous_q
    lower = np.asarray(q_lower, dtype=np.float64)
    upper = np.asarray(q_upper, dtype=np.float64)
    if raw_q.shape != lower.shape or raw_q.shape != upper.shape:
        raise ValueError("q bounds and qpos shapes differ")
    predicted_q = np.clip(raw_q, lower, upper)
    return PropagatedRetargetState(
        predicted_base_scene=predicted_base,
        predicted_qpos=predicted_q,
        base_correction=encode_base_correction(current_warm_base_scene, predicted_base),
        q_correction=predicted_q - np.asarray(current_warm_qpos, dtype=np.float64),
        q_clamp_count=int(np.count_nonzero(predicted_q != raw_q)),
        previous_frame=previous_frame,
        current_frame=int(current_frame),
        initialization_source=(
            "propagated_previous_final" if previous_frame is not None else "warm_first_frame"
        ),
    )


def correction_temporal_residual(
    predicted_base_scene: np.ndarray,
    final_base_scene: np.ndarray,
    predicted_qpos: np.ndarray,
    final_qpos: np.ndarray,
) -> np.ndarray:
    """Return normalized chart-consistent base/finger correction residuals."""

    predicted = _validate_pose(predicted_base_scene, "predicted_base_scene")
    final = _validate_pose(final_base_scene, "final_base_scene")
    base_error = invert_transform(predicted) @ final
    return np.concatenate(
        [
            base_error[:3, 3] / S_POS_M,
            so3_log_np(base_error[:3, :3]) / S_ROT_RAD,
            (np.asarray(final_qpos, dtype=np.float64) - np.asarray(predicted_qpos)) / S_Q_RAD,
        ]
    )


def correction_temporal_energy(
    predicted_base_scene: np.ndarray,
    final_base_scene: np.ndarray,
    predicted_qpos: np.ndarray,
    final_qpos: np.ndarray,
    *,
    lambda_corr: float = LAMBDA_CORR,
) -> float:
    residual = correction_temporal_residual(
        predicted_base_scene, final_base_scene, predicted_qpos, final_qpos
    )
    base = residual[:6]
    finger = residual[6:]
    return float(lambda_corr) * (
        float(np.mean(base[:3] ** 2)) + float(np.mean(base[3:] ** 2)) + float(np.mean(finger**2))
    )


def continuity_metrics(
    predicted_base_scene: np.ndarray,
    final_base_scene: np.ndarray,
    predicted_qpos: np.ndarray,
    final_qpos: np.ndarray,
    *,
    predicted_keypoints_scene: np.ndarray | None = None,
    final_keypoints_scene: np.ndarray | None = None,
    frame: int = 0,
) -> dict[str, Any]:
    predicted = _validate_pose(predicted_base_scene, "predicted_base_scene")
    final = _validate_pose(final_base_scene, "final_base_scene")
    error = invert_transform(predicted) @ final
    translation = float(np.linalg.norm(error[:3, 3]))
    rotation = float(np.linalg.norm(so3_log_np(error[:3, :3])))
    q_inf = float(
        np.max(np.abs(np.asarray(final_qpos, dtype=np.float64) - np.asarray(predicted_qpos)))
    )
    if predicted_keypoints_scene is None or final_keypoints_scene is None:
        excess = float("nan")
    else:
        excess = float(
            np.max(
                np.linalg.norm(
                    np.asarray(final_keypoints_scene, dtype=np.float64)
                    - np.asarray(predicted_keypoints_scene, dtype=np.float64),
                    axis=-1,
                )
            )
        )
    reasons: list[str] = []
    if translation > S_POS_M:
        reasons.append("base_translation")
    if rotation > S_ROT_RAD:
        reasons.append("base_rotation")
    if q_inf > S_Q_RAD:
        reasons.append("finger_correction")
    if not np.isfinite(excess) or excess > 0.020:
        reasons.append("excess_keypoint_displacement")
    return {
        "schema_version": CONTINUITY_SCHEMA_VERSION,
        "frame": int(frame),
        "delta_base_translation_m": translation,
        "delta_base_rotation_rad": rotation,
        "delta_base_rotation_deg": float(np.rad2deg(rotation)),
        "delta_finger_inf_rad": q_inf,
        "excess_keypoint_max_m": excess,
        "trajectory_continuous": not reasons,
        "continuity_failure_reasons": reasons,
        "thresholds": {
            "base_translation_m": S_POS_M,
            "base_rotation_rad": S_ROT_RAD,
            "base_rotation_deg": 5.0,
            "finger_correction_inf_rad": S_Q_RAD,
            "excess_keypoint_max_m": 0.020,
        },
    }


@dataclass(frozen=True)
class RecedingHorizonWindow:
    target_frame: int
    left_anchor_frame: int | None
    variable_frames: tuple[int, ...]
    window_size: int = 5

    @classmethod
    def for_target(
        cls, target_frame: int, frame_count: int, window_size: int = 5
    ) -> RecedingHorizonWindow:
        if window_size != 5:
            raise ValueError("Wuji continuous profile requires a five-frame window")
        if not 0 <= target_frame < frame_count:
            raise ValueError("target frame outside trajectory")
        return cls(
            target_frame=int(target_frame),
            left_anchor_frame=None if target_frame == 0 else int(target_frame - 1),
            variable_frames=tuple(range(target_frame, min(frame_count, target_frame + 4))),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "target_frame": self.target_frame,
            "left_anchor_frame": self.left_anchor_frame,
            "variable_frames": list(self.variable_frames),
            "center_only_commit": True,
            "future_hint_only": True,
        }


@dataclass(frozen=True)
class ContinuousRetargetProfile:
    values: dict[str, Any]
    profile_hash: str
    source_path: Path

    @classmethod
    def load(
        cls, root: Path | None = None, profile_id: str = CONTINUOUS_PROFILE_ID
    ) -> ContinuousRetargetProfile:
        repo = root or Path(__file__).resolve().parents[3]
        path = repo / "configs" / "retarget" / "refinement_solvers" / f"{profile_id}.yaml"
        raw = path.read_bytes()
        values = yaml.safe_load(raw) or {}
        if not isinstance(values, dict) or values.get("profile_id") != profile_id:
            raise ValueError("invalid Wuji continuous profile")
        return cls(dict(values), _sha256(raw), path)

    def as_dict(self) -> dict[str, Any]:
        return {**self.values, "profile_hash": self.profile_hash}


def transport_round_trip_report() -> dict[str, Any]:
    """Deterministic numerical audit used by tests and the experiment report."""

    warm = np.eye(4, dtype=np.float64)
    warm[:3, 3] = [0.1, -0.2, 0.3]
    correction = np.array([0.01, -0.02, 0.03, 0.1, -0.05, 0.02], dtype=np.float64)
    final = decode_base_correction(warm, correction)
    encoded = encode_base_correction(warm, final)
    recovered = decode_base_correction(warm, encoded)
    return {
        "schema_version": "toporetarget.base_correction_convention_audit.v1",
        "convention": BASE_CORRECTION_CONVENTION,
        "rotation_application": "left_multiply_scene_frame",
        "translation_application": "additive_scene_frame",
        "root_frame": "r_wrist / scene base pose",
        "encode_decode_max_abs": float(np.max(np.abs(recovered - final))),
        "correction_round_trip_max_abs": float(np.max(np.abs(encoded - correction))),
        "passed": bool(
            np.max(np.abs(recovered - final)) <= 1e-10
            and np.max(np.abs(encoded - correction)) <= 1e-10
        ),
    }


__all__ = [
    "BASE_CORRECTION_CONVENTION",
    "CONTINUITY_SCHEMA_VERSION",
    "CONTINUOUS_PROFILE_ID",
    "CONTINUOUS_PROFILE_IDS",
    "ContinuousRetargetProfile",
    "LAMBDA_CORR",
    "PropagatedRetargetState",
    "RecedingHorizonWindow",
    "S_POS_M",
    "S_Q_RAD",
    "S_ROT_RAD",
    "SEQUENTIAL_PROFILE_ID",
    "continuity_metrics",
    "correction_temporal_energy",
    "correction_temporal_residual",
    "decode_base_correction",
    "encode_base_correction",
    "is_continuous_profile",
    "so3_log_np",
    "transport_previous_final_to_current_warm",
    "transport_round_trip_report",
]
