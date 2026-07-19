"""Comparison metric calculation and machine-readable error reports."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.schema import HOISequence
from toporetarget.geometry.se3 import (
    object_to_scene,
    pose_rotation_error,
    pose_translation_error,
)


def _summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"available": False, "reason": "no finite overlapping values"}
    return {
        "available": True,
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _metric(result: dict[str, Any], name: str, values: np.ndarray | None) -> None:
    if values is None:
        result["metrics"][name] = {"available": False, "reason": "unavailable"}
        return
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    result["per_frame"][name] = [float(item) for item in values]
    result["metrics"][name] = _summary(values)


def _vertex_errors(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if first.shape != second.shape or first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(
            f"vertex tracks must have matching [T,V,3] shape, got {first.shape}/{second.shape}"
        )
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    rmse = np.sqrt(np.mean(np.sum(delta**2, axis=-1), axis=1))
    maximum = np.max(np.linalg.norm(delta, axis=-1), axis=1)
    return rmse, maximum


def _keypoint_errors(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if first.shape != second.shape or first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(
            f"keypoint tracks must have matching [T,K,3] shape, got {first.shape}/{second.shape}"
        )
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    distances = np.linalg.norm(delta, axis=-1)
    return np.sqrt(np.mean(np.sum(delta**2, axis=-1), axis=1)), np.max(distances, axis=1)


@dataclass
class ComparisonResult:
    """JSON/CSV-friendly comparison result."""

    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return self.payload

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination

    def write_csv(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "available", "mean", "median", "p95", "max", "reason"])
            for name, value in self.payload["metrics"].items():
                writer.writerow(
                    [
                        name,
                        value.get("available", False),
                        value.get("mean", ""),
                        value.get("median", ""),
                        value.get("p95", ""),
                        value.get("max", ""),
                        value.get("reason", ""),
                    ]
                )
        return destination


class ComparisonMetrics:
    """Compute raw/canonical errors without mutating either sequence."""

    @classmethod
    def compute(cls, raw: HOISequence, canonical: HOISequence) -> ComparisonResult:
        raw.validate()
        canonical.validate()
        frame_count = raw.num_frames == canonical.num_frames
        timestamp_count = raw.timestamps.size == canonical.timestamps.size
        shared = min(raw.num_frames, canonical.num_frames)
        payload: dict[str, Any] = {
            "schema_version": raw.metadata.schema_version,
            "frame_count_match": frame_count,
            "timestamp_count_match": timestamp_count,
            "num_frames_raw": raw.num_frames,
            "num_frames_canonical": canonical.num_frames,
            "frame_indices": list(range(shared)),
            "per_frame": {},
            "metrics": {},
        }
        if shared:
            _metric(
                payload,
                "timestamp_abs_error_s",
                np.abs(raw.timestamps[:shared] - canonical.timestamps[:shared]),
            )
            payload["metrics"]["timestamp_max_abs_error_s"] = _summary(
                np.abs(raw.timestamps[:shared] - canonical.timestamps[:shared])
            )
        else:
            payload["metrics"]["timestamp_max_abs_error_s"] = {
                "available": False,
                "reason": "empty sequence",
            }
        if not timestamp_count:
            payload["metrics"]["timestamp_count_match"] = {"available": True, "match": False}
        else:
            payload["metrics"]["timestamp_count_match"] = {"available": True, "match": True}
        payload["metrics"]["frame_count_match"] = {"available": True, "match": frame_count}

        raw_hands = {item.hand_id: item for item in raw.hands}
        canonical_hands = {item.hand_id: item for item in canonical.hands}
        common_hands = sorted(raw_hands.keys() & canonical_hands.keys())
        if common_hands:
            hand_first = raw_hands[common_hands[0]]
            hand_second = canonical_hands[common_hands[0]]
            if hand_first.vertices_scene is not None and hand_second.vertices_scene is not None:
                rmse: np.ndarray | None
                maximum: np.ndarray | None
                try:
                    rmse, maximum = _vertex_errors(
                        hand_first.vertices_scene[:shared], hand_second.vertices_scene[:shared]
                    )
                except ValueError:
                    rmse = maximum = None
                _metric(payload, "hand_vertex_rmse_m", rmse)
                _metric(payload, "hand_vertex_max_error_m", maximum)
            else:
                _metric(payload, "hand_vertex_rmse_m", None)
                _metric(payload, "hand_vertex_max_error_m", None)
            wrist_a = hand_first.wrist_pose_scene.pose_scene[:shared]
            wrist_b = hand_second.wrist_pose_scene.pose_scene[:shared]
            if wrist_a.shape == wrist_b.shape:
                _metric(
                    payload, "wrist_translation_error_m", pose_translation_error(wrist_a, wrist_b)
                )
                _metric(
                    payload,
                    "wrist_rotation_geodesic_deg",
                    np.rad2deg(pose_rotation_error(wrist_a, wrist_b)),
                )
            else:
                _metric(payload, "wrist_translation_error_m", None)
                _metric(payload, "wrist_rotation_geodesic_deg", None)
            common_layouts = sorted(
                set(hand_first.keypoint_tracks) & set(hand_second.keypoint_tracks)
            )
            if common_layouts:
                key_a = hand_first.keypoint_tracks[common_layouts[0]].positions_scene[:shared]
                key_b = hand_second.keypoint_tracks[common_layouts[0]].positions_scene[:shared]
                rmse = None
                maximum = None
                try:
                    rmse, maximum = _keypoint_errors(key_a, key_b)
                except ValueError:
                    rmse = maximum = None
                _metric(payload, "hand_keypoint_rmse_m", rmse)
                _metric(payload, "hand_keypoint_max_error_m", maximum)
            else:
                _metric(payload, "hand_keypoint_rmse_m", None)
                _metric(payload, "hand_keypoint_max_error_m", None)
        else:
            for name in (
                "hand_vertex_rmse_m",
                "hand_vertex_max_error_m",
                "hand_keypoint_rmse_m",
                "hand_keypoint_max_error_m",
                "wrist_translation_error_m",
                "wrist_rotation_geodesic_deg",
            ):
                _metric(payload, name, None)

        raw_objects = {item.object_id: item for item in raw.rigid_objects}
        canonical_objects = {item.object_id: item for item in canonical.rigid_objects}
        common_objects = sorted(raw_objects.keys() & canonical_objects.keys())
        if common_objects:
            object_first = raw_objects[common_objects[0]]
            object_second = canonical_objects[common_objects[0]]
            pose_a = object_first.pose_scene.pose_scene[:shared]
            pose_b = object_second.pose_scene.pose_scene[:shared]
            if pose_a.shape == pose_b.shape:
                _metric(
                    payload,
                    "object_pose_translation_error_m",
                    pose_translation_error(pose_a, pose_b),
                )
                _metric(
                    payload,
                    "object_pose_rotation_geodesic_deg",
                    np.rad2deg(pose_rotation_error(pose_a, pose_b)),
                )
            else:
                _metric(payload, "object_pose_translation_error_m", None)
                _metric(payload, "object_pose_rotation_geodesic_deg", None)
            try:
                local_a = np.broadcast_to(
                    object_first.mesh.vertices_local,
                    (shared, object_first.mesh.vertices_local.shape[0], 3),
                )
                local_b = np.broadcast_to(
                    object_second.mesh.vertices_local,
                    (shared, object_second.mesh.vertices_local.shape[0], 3),
                )
                world_a = object_to_scene(pose_a, local_a)
                world_b = object_to_scene(pose_b, local_b)
                rmse, maximum = _vertex_errors(world_a, world_b)
            except (ValueError, IndexError):
                rmse = maximum = None
            _metric(payload, "object_world_vertex_rmse_m", rmse)
            _metric(payload, "object_world_vertex_max_error_m", maximum)
        else:
            for name in (
                "object_pose_translation_error_m",
                "object_pose_rotation_geodesic_deg",
                "object_world_vertex_rmse_m",
                "object_world_vertex_max_error_m",
            ):
                _metric(payload, name, None)
        return ComparisonResult(payload)


def compare_sequences(raw: HOISequence, canonical: HOISequence) -> ComparisonResult:
    return ComparisonMetrics.compute(raw, canonical)


__all__ = ["ComparisonMetrics", "ComparisonResult", "compare_sequences"]
